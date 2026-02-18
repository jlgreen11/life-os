"""
Test suite for ContextAssembler._get_predictions_context() and its integration
into assemble_briefing_context().

The prediction engine generates high-accuracy predictions (opportunity: 97.6%,
reminder: 100%, routine_deviation: 100%) but they were never included in the
morning briefing context. This test suite validates the new
_get_predictions_context() method that surfaces active, surfaced, unresolved
predictions to the LLM briefing generator.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_prediction(
    conn,
    prediction_type: str = "reminder",
    description: str = "Test prediction",
    confidence: float = 0.75,
    confidence_gate: str = "SUGGEST",
    was_surfaced: int = 1,
    resolved_at: str = None,
    filter_reason: str = None,
    suggested_action: str = None,
    created_at: str = None,
) -> str:
    """Insert a prediction row directly into the predictions table.

    Returns the generated prediction ID for reference in assertions.
    """
    pred_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT INTO predictions
           (id, prediction_type, description, confidence, confidence_gate,
            was_surfaced, resolved_at, filter_reason, suggested_action, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pred_id,
            prediction_type,
            description,
            confidence,
            confidence_gate,
            was_surfaced,
            resolved_at,
            filter_reason,
            suggested_action,
            created_at,
        ),
    )
    return pred_id


# ---------------------------------------------------------------------------
# Tests for _get_predictions_context()
# ---------------------------------------------------------------------------

class TestGetPredictionsContext:
    """Tests for the _get_predictions_context() private method."""

    def test_returns_empty_string_when_no_predictions(self, db, user_model_store):
        """Empty predictions table should return empty string (no noise in briefing)."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_predictions_context()
        assert result == ""

    def test_returns_empty_for_unsurfaced_predictions(self, db, user_model_store):
        """Predictions that were not surfaced (was_surfaced=0) should be excluded.

        These are filtered predictions that never made it to the user,
        so they should not appear in the briefing.
        """
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(conn, was_surfaced=0)

        result = assembler._get_predictions_context()
        assert result == ""

    def test_returns_empty_for_resolved_predictions(self, db, user_model_store):
        """Resolved predictions (resolved_at IS NOT NULL) should be excluded.

        Once the predicted condition has been confirmed or dismissed,
        it should no longer appear in the briefing.
        """
        assembler = ContextAssembler(db, user_model_store)
        resolved_time = datetime.now(timezone.utc).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_prediction(conn, was_surfaced=1, resolved_at=resolved_time)

        result = assembler._get_predictions_context()
        assert result == ""

    def test_returns_empty_for_filtered_predictions(self, db, user_model_store):
        """Predictions with a filter_reason should be excluded.

        filter_reason indicates the prediction failed a quality gate
        (e.g., automated sender, low confidence) and should not appear.
        """
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                was_surfaced=1,
                filter_reason="automated_sender_fast_path",
            )

        result = assembler._get_predictions_context()
        assert result == ""

    def test_returns_empty_for_stale_predictions(self, db, user_model_store):
        """Predictions older than 7 days should be excluded from the briefing.

        Stale predictions are no longer actionable and would add noise
        to the briefing context window.
        """
        assembler = ContextAssembler(db, user_model_store)
        old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_prediction(conn, was_surfaced=1, created_at=old_time)

        result = assembler._get_predictions_context()
        assert result == ""

    def test_includes_active_surfaced_prediction(self, db, user_model_store):
        """Active, surfaced, unresolved predictions within 7 days should appear."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                prediction_type="reminder",
                description="Follow up with alice@example.com about project",
                confidence=0.82,
                was_surfaced=1,
            )

        result = assembler._get_predictions_context()
        assert "Active predictions from the system:" in result
        assert "[reminder]" in result
        assert "Follow up with alice@example.com about project" in result
        assert "0.82" in result

    def test_formats_prediction_with_suggested_action(self, db, user_model_store):
        """Predictions with a suggested_action should include it after ' — suggested:'."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                prediction_type="opportunity",
                description="Haven't replied to Bob in 12 days",
                confidence=0.78,
                suggested_action="Send a brief check-in message",
                was_surfaced=1,
            )

        result = assembler._get_predictions_context()
        assert "— suggested: Send a brief check-in message" in result

    def test_omits_suggested_action_when_none(self, db, user_model_store):
        """When suggested_action is NULL, no ' — suggested:' fragment should appear."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                description="Calendar conflict detected",
                confidence=0.91,
                was_surfaced=1,
                suggested_action=None,
            )

        result = assembler._get_predictions_context()
        assert "— suggested:" not in result

    def test_multiple_predictions_sorted_by_confidence(self, db, user_model_store):
        """When multiple predictions exist, they should be sorted by confidence DESC."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                description="Low confidence prediction",
                confidence=0.45,
                was_surfaced=1,
            )
            _insert_prediction(
                conn,
                description="High confidence prediction",
                confidence=0.91,
                was_surfaced=1,
            )
            _insert_prediction(
                conn,
                description="Medium confidence prediction",
                confidence=0.72,
                was_surfaced=1,
            )

        result = assembler._get_predictions_context()
        lines = result.split("\n")
        # Find line indices for each prediction
        indices = {
            "high": next(i for i, l in enumerate(lines) if "High confidence" in l),
            "medium": next(i for i, l in enumerate(lines) if "Medium confidence" in l),
            "low": next(i for i, l in enumerate(lines) if "Low confidence" in l),
        }
        assert indices["high"] < indices["medium"] < indices["low"]

    def test_caps_at_10_predictions(self, db, user_model_store):
        """The method should return at most 10 predictions to respect token budget."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            for i in range(15):
                _insert_prediction(
                    conn,
                    description=f"Prediction number {i}",
                    confidence=0.5 + i * 0.01,
                    was_surfaced=1,
                )

        result = assembler._get_predictions_context()
        # Count the bullet lines (lines starting with "- [")
        bullet_lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(bullet_lines) == 10

    def test_multiple_prediction_types_all_included(self, db, user_model_store):
        """All prediction types (reminder, opportunity, conflict, etc.) should appear."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            for ptype in ["reminder", "opportunity", "conflict", "routine_deviation"]:
                _insert_prediction(
                    conn,
                    prediction_type=ptype,
                    description=f"A {ptype} prediction",
                    confidence=0.70,
                    was_surfaced=1,
                )

        result = assembler._get_predictions_context()
        assert "[reminder]" in result
        assert "[opportunity]" in result
        assert "[conflict]" in result
        assert "[routine_deviation]" in result

    def test_excludes_stale_but_includes_recent(self, db, user_model_store):
        """Predictions older than 7 days should be excluded; newer ones included."""
        assembler = ContextAssembler(db, user_model_store)
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                description="Stale prediction from 10 days ago",
                confidence=0.85,
                was_surfaced=1,
                created_at=old_time,
            )
            _insert_prediction(
                conn,
                description="Recent prediction from 2 hours ago",
                confidence=0.75,
                was_surfaced=1,
                created_at=recent_time,
            )

        result = assembler._get_predictions_context()
        assert "Recent prediction from 2 hours ago" in result
        assert "Stale prediction from 10 days ago" not in result


# ---------------------------------------------------------------------------
# Integration tests: assemble_briefing_context() includes predictions
# ---------------------------------------------------------------------------

class TestBriefingContextIncludesPredictions:
    """Integration tests confirming predictions appear in the full briefing context."""

    def test_briefing_includes_active_predictions_section(self, db, user_model_store):
        """assemble_briefing_context() should include predictions when they exist."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                prediction_type="reminder",
                description="Prepare for Q1 Planning meeting",
                confidence=0.88,
                was_surfaced=1,
                suggested_action="Review agenda and gather metrics",
            )

        context = assembler.assemble_briefing_context()
        assert "Active predictions from the system:" in context
        assert "[reminder]" in context
        assert "Prepare for Q1 Planning meeting" in context

    def test_briefing_omits_predictions_section_when_empty(self, db, user_model_store):
        """When no active predictions exist, the briefing should not contain the header."""
        assembler = ContextAssembler(db, user_model_store)

        context = assembler.assemble_briefing_context()
        assert "Active predictions from the system:" not in context

    def test_briefing_predictions_do_not_break_existing_sections(self, db, user_model_store):
        """Adding predictions context should not remove any other briefing sections."""
        assembler = ContextAssembler(db, user_model_store)

        with db.get_connection("user_model") as conn:
            _insert_prediction(
                conn,
                description="Test prediction",
                confidence=0.75,
                was_surfaced=1,
            )

        context = assembler.assemble_briefing_context()
        # Verify all original sections are still present in the output
        assert "Current time:" in context
        assert "Upcoming calendar events" in context
        assert "Pending tasks:" in context or "Pending tasks: none" in context
        assert "Messages in last 12 hours:" in context
