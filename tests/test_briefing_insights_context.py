"""
Test suite for ContextAssembler._get_insights_context() and its integration
into assemble_briefing_context().

The InsightEngine runs 14 correlators (relationship, cadence, mood, temporal,
spatial, routine, decision, topic, spending, communication-style, etc.) and
stores discoveries in the ``insights`` table.  Until this change those
discoveries never appeared in the morning briefing context — the LLM had no
access to them when generating the user's daily narrative.

This test suite validates:
- _get_insights_context() correctly filters by TTL, negative feedback, and
  confidence ordering.
- assemble_briefing_context() includes the insights section when insights exist
  and omits it (no noise) when none are available.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_insight(
    conn,
    insight_type: str = "behavioral_pattern",
    summary: str = "Test insight summary",
    confidence: float = 0.75,
    category: str = "behavioral_pattern",
    entity: str = None,
    staleness_ttl_hours: int = 168,
    feedback: str = None,
    created_at: str = None,
) -> str:
    """Insert an insight row directly into the insights table.

    Returns the generated insight ID for reference in assertions.
    """
    insight_id = str(uuid.uuid4())
    dedup_key = str(uuid.uuid4())

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO insights
           (id, type, summary, confidence, category, entity,
            staleness_ttl_hours, dedup_key, feedback, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            insight_id,
            insight_type,
            summary,
            confidence,
            category,
            entity,
            staleness_ttl_hours,
            dedup_key,
            feedback,
            created_at,
        ),
    )
    return insight_id


# ---------------------------------------------------------------------------
# Tests for _get_insights_context()
# ---------------------------------------------------------------------------


class TestGetInsightsContext:
    """Tests for the _get_insights_context() private method."""

    def test_returns_empty_string_when_no_insights(self, db, user_model_store):
        """Empty insights table should return empty string (no noise in briefing)."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_insights_context()
        assert result == ""

    def test_includes_active_insight(self, db, user_model_store):
        """A fresh, un-dismissed insight should appear in the context."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                insight_type="relationship_intelligence",
                summary="You haven't contacted Alice in 14 days — usual gap is 3 days",
                confidence=0.87,
                category="relationship_intelligence",
            )

        result = assembler._get_insights_context()
        assert "Behavioral patterns and insights:" in result
        assert "You haven't contacted Alice" in result
        assert "0.87" in result

    def test_uses_category_as_label_when_set(self, db, user_model_store):
        """When category is non-empty it should be used as the bracket label."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                insight_type="pattern",
                summary="Sends emails on Tuesday mornings",
                confidence=0.80,
                category="behavioral_pattern",
            )

        result = assembler._get_insights_context()
        assert "[behavioral_pattern]" in result
        # The raw type "pattern" should not appear as the label
        assert "[pattern]" not in result

    def test_falls_back_to_type_when_category_empty(self, db, user_model_store):
        """When category is empty string the type field should be used as label."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                insight_type="communication_style",
                summary="Your formality score is above average",
                confidence=0.73,
                category="",  # explicitly empty
            )

        result = assembler._get_insights_context()
        assert "[communication_style]" in result

    def test_excludes_negative_feedback_insights(self, db, user_model_store):
        """Insights marked with negative feedback should be excluded.

        These are observations the user has explicitly dismissed ("thumbs down"),
        so repeating them in the briefing would be annoying.
        """
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Dismissed insight — should not appear",
                confidence=0.85,
                feedback="negative",
            )

        result = assembler._get_insights_context()
        assert result == ""

    def test_includes_positive_feedback_insights(self, db, user_model_store):
        """Insights with positive feedback should still be included."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Insight with positive feedback",
                confidence=0.80,
                feedback="positive",
            )

        result = assembler._get_insights_context()
        assert "Insight with positive feedback" in result

    def test_includes_null_feedback_insights(self, db, user_model_store):
        """Insights with no feedback (NULL) should be included."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Insight with no feedback yet",
                confidence=0.75,
                feedback=None,
            )

        result = assembler._get_insights_context()
        assert "Insight with no feedback yet" in result

    def test_excludes_stale_insights_beyond_ttl(self, db, user_model_store):
        """Insights older than their staleness_ttl_hours should be excluded.

        Each insight stores its own TTL so correlators can tune freshness
        independently (e.g., spatial insights may have a shorter TTL than
        mood trends).
        """
        assembler = ContextAssembler(db, user_model_store)
        # Created 10 hours ago with a 5-hour TTL → stale
        old_time = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Stale insight — should not appear",
                confidence=0.90,
                staleness_ttl_hours=5,  # TTL is only 5 hours
                created_at=old_time,
            )

        result = assembler._get_insights_context()
        assert result == ""

    def test_includes_insight_within_ttl(self, db, user_model_store):
        """Insights created within their staleness_ttl_hours should be included."""
        assembler = ContextAssembler(db, user_model_store)
        # Created 3 hours ago with a 24-hour TTL → still fresh
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Fresh insight within TTL",
                confidence=0.78,
                staleness_ttl_hours=24,
                created_at=recent_time,
            )

        result = assembler._get_insights_context()
        assert "Fresh insight within TTL" in result

    def test_stale_excluded_but_fresh_included(self, db, user_model_store):
        """Mixed freshness: only insights within their TTL window appear."""
        assembler = ContextAssembler(db, user_model_store)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Old insight — stale",
                confidence=0.88,
                staleness_ttl_hours=168,
                created_at=old_time,
            )
            _insert_insight(
                conn,
                summary="Recent insight — fresh",
                confidence=0.72,
                staleness_ttl_hours=168,
                created_at=recent_time,
            )

        result = assembler._get_insights_context()
        assert "Recent insight — fresh" in result
        assert "Old insight — stale" not in result

    def test_sorted_by_confidence_descending(self, db, user_model_store):
        """Multiple insights should appear in order of confidence, highest first."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(conn, summary="Low confidence insight", confidence=0.45)
            _insert_insight(conn, summary="High confidence insight", confidence=0.92)
            _insert_insight(conn, summary="Medium confidence insight", confidence=0.68)

        result = assembler._get_insights_context()
        lines = result.split("\n")
        idx_high = next(i for i, l in enumerate(lines) if "High confidence" in l)
        idx_med = next(i for i, l in enumerate(lines) if "Medium confidence" in l)
        idx_low = next(i for i, l in enumerate(lines) if "Low confidence" in l)
        assert idx_high < idx_med < idx_low

    def test_caps_at_10_insights(self, db, user_model_store):
        """The method should return at most 10 insights to respect token budget."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            for i in range(15):
                _insert_insight(
                    conn,
                    summary=f"Insight number {i}",
                    confidence=0.5 + i * 0.01,
                )

        result = assembler._get_insights_context()
        bullet_lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(bullet_lines) == 10

    def test_confidence_formatted_to_two_decimal_places(self, db, user_model_store):
        """Confidence should appear formatted to two decimal places."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(conn, summary="Precision test", confidence=0.8)

        result = assembler._get_insights_context()
        assert "0.80" in result

    def test_multiple_insight_types_all_included(self, db, user_model_store):
        """Various insight types (relationship, behavioral, communication) all appear."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            for cat in [
                "relationship_intelligence",
                "behavioral_pattern",
                "communication_style",
            ]:
                _insert_insight(
                    conn,
                    summary=f"A {cat} insight",
                    category=cat,
                    confidence=0.75,
                )

        result = assembler._get_insights_context()
        assert "[relationship_intelligence]" in result
        assert "[behavioral_pattern]" in result
        assert "[communication_style]" in result

    def test_negative_feedback_excluded_but_positive_included(self, db, user_model_store):
        """When both dismissed and accepted insights exist, only accepted appear."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Dismissed insight",
                confidence=0.90,
                feedback="negative",
            )
            _insert_insight(
                conn,
                summary="Accepted insight",
                confidence=0.85,
                feedback="positive",
            )

        result = assembler._get_insights_context()
        assert "Accepted insight" in result
        assert "Dismissed insight" not in result


# ---------------------------------------------------------------------------
# Integration tests: assemble_briefing_context() includes insights
# ---------------------------------------------------------------------------


class TestBriefingContextIncludesInsights:
    """Integration tests confirming insights appear in the full briefing context."""

    def test_briefing_includes_insights_section(self, db, user_model_store):
        """assemble_briefing_context() should include insights when they exist."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                insight_type="behavioral_pattern",
                summary="You typically reply to emails within 2 hours on weekdays",
                confidence=0.83,
                category="behavioral_pattern",
            )

        context = assembler.assemble_briefing_context()
        assert "Behavioral patterns and insights:" in context
        assert "typically reply to emails" in context

    def test_briefing_omits_insights_section_when_empty(self, db, user_model_store):
        """When no active insights exist, the briefing should not contain the header."""
        assembler = ContextAssembler(db, user_model_store)

        context = assembler.assemble_briefing_context()
        assert "Behavioral patterns and insights:" not in context

    def test_briefing_insights_appear_after_predictions(self, db, user_model_store):
        """Insights section should appear after predictions in the briefing.

        The ordering is intentional: predictions are actionable (do X soon),
        insights are contextual (you always do Y). Actionable first, then
        narrative context.
        """
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            # Insert a prediction
            pred_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence,
                    confidence_gate, was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id,
                    "reminder",
                    "Review Q1 metrics before standup",
                    0.85,
                    "DEFAULT",
                    1,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            # Insert an insight
            _insert_insight(
                conn,
                summary="You are most productive on Tuesday mornings",
                confidence=0.78,
                category="behavioral_pattern",
            )

        context = assembler.assemble_briefing_context()
        pred_pos = context.find("Active predictions from the system:")
        insight_pos = context.find("Behavioral patterns and insights:")

        assert pred_pos != -1, "Predictions section missing"
        assert insight_pos != -1, "Insights section missing"
        assert pred_pos < insight_pos, "Insights should come after predictions"

    def test_briefing_does_not_break_existing_sections(self, db, user_model_store):
        """Adding insights context should not remove any other briefing sections."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Test insight for structural validation",
                confidence=0.75,
            )

        context = assembler.assemble_briefing_context()
        # Verify all original sections are still present
        assert "Current time:" in context
        assert "Upcoming calendar events" in context
        assert "Pending tasks:" in context or "Pending tasks: none" in context
        assert "Messages in last 12 hours:" in context

    def test_briefing_insights_excluded_when_negative_feedback(self, db, user_model_store):
        """Dismissed insights should not appear in the briefing."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_insight(
                conn,
                summary="Insight user dismissed last week",
                confidence=0.95,
                feedback="negative",
            )

        context = assembler.assemble_briefing_context()
        assert "Behavioral patterns and insights:" not in context
        assert "Insight user dismissed last week" not in context
