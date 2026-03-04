"""Tests for feedback-aware insight deduplication.

Verifies that _deduplicate() in InsightEngine respects user feedback
when computing effective staleness TTL windows:
- dismissed/not_relevant: TTL * 4 (suppress longer)
- useful: TTL * 0.5 (resurface sooner)
- None/unknown: original TTL unchanged
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.models import Insight


def _insert_insight_row(conn, *, dedup_key, feedback=None, age_hours=0, staleness_ttl_hours=168):
    """Insert a row into the insights table with a specific age and feedback value."""
    created_at = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    conn.execute(
        """INSERT INTO insights (id, type, summary, confidence, evidence, category,
           entity, staleness_ttl_hours, dedup_key, feedback, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            "behavioral_pattern",
            "test insight",
            0.8,
            "[]",
            "test",
            None,
            staleness_ttl_hours,
            dedup_key,
            feedback,
            created_at,
        ),
    )
    conn.commit()


def _make_insight(dedup_key="test_dedup", staleness_ttl_hours=168):
    """Create an Insight model instance with a preset dedup_key."""
    return Insight(
        type="behavioral_pattern",
        summary="test insight",
        confidence=0.8,
        category="test",
        staleness_ttl_hours=staleness_ttl_hours,
        dedup_key=dedup_key,
    )


def _make_engine(db):
    """Construct a minimal InsightEngine with required dependencies."""
    from unittest.mock import MagicMock

    from services.insight_engine.engine import InsightEngine

    # InsightEngine requires db and ums; ums is not used by _deduplicate
    mock_ums = MagicMock()
    return InsightEngine(db=db, ums=mock_ums)


class TestDismissedInsightSuppressedLonger:
    """Dismissed insights use TTL * 4 — suppressed at 200h but allowed at 700h (168*4=672)."""

    def test_suppressed_within_extended_window(self, db):
        """Insight at 200h age with feedback='dismissed' should be suppressed (200 < 672)."""
        engine = _make_engine(db)
        dedup_key = "dismissed_test_1"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="dismissed", age_hours=200)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 0, "Dismissed insight should be suppressed within 4x TTL window"

    def test_allowed_past_extended_window(self, db):
        """Insight at 700h age with feedback='dismissed' should be allowed (700 > 672)."""
        engine = _make_engine(db)
        dedup_key = "dismissed_test_2"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="dismissed", age_hours=700)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "Dismissed insight should resurface past 4x TTL window"


class TestNotRelevantInsightSuppressedLonger:
    """not_relevant insights use TTL * 4 — same scaling as dismissed."""

    def test_suppressed_within_extended_window(self, db):
        """Insight at 200h age with feedback='not_relevant' should be suppressed (200 < 672)."""
        engine = _make_engine(db)
        dedup_key = "not_relevant_test_1"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="not_relevant", age_hours=200)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 0, "not_relevant insight should be suppressed within 4x TTL window"

    def test_allowed_past_extended_window(self, db):
        """Insight at 700h age with feedback='not_relevant' should be allowed (700 > 672)."""
        engine = _make_engine(db)
        dedup_key = "not_relevant_test_2"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="not_relevant", age_hours=700)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "not_relevant insight should resurface past 4x TTL window"


class TestUsefulInsightResurfacesSooner:
    """Useful insights use TTL * 0.5 — allowed at 100h (past 84h = 168*0.5)."""

    def test_allowed_past_half_ttl(self, db):
        """Insight at 100h age with feedback='useful' should be allowed (100 > 84)."""
        engine = _make_engine(db)
        dedup_key = "useful_test_1"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="useful", age_hours=100)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "Useful insight should resurface past 0.5x TTL window"

    def test_suppressed_within_half_ttl(self, db):
        """Insight at 40h age with feedback='useful' should be suppressed (40 < 84)."""
        engine = _make_engine(db)
        dedup_key = "useful_test_2"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="useful", age_hours=40)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 0, "Useful insight should be suppressed within 0.5x TTL window"


class TestNoFeedbackUsesOriginalTTL:
    """No feedback (None) uses the original staleness_ttl_hours unchanged."""

    def test_suppressed_within_original_ttl(self, db):
        """Insight at 100h age with no feedback should be suppressed (100 < 168)."""
        engine = _make_engine(db)
        dedup_key = "no_feedback_test_1"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback=None, age_hours=100)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 0, "No-feedback insight should be suppressed within original TTL"

    def test_allowed_past_original_ttl(self, db):
        """Insight at 200h age with no feedback should be allowed (200 > 168)."""
        engine = _make_engine(db)
        dedup_key = "no_feedback_test_2"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback=None, age_hours=200)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "No-feedback insight should resurface past original TTL"


class TestUnknownFeedbackUsesOriginalTTL:
    """Unknown feedback values are treated as None (fail-open)."""

    def test_suppressed_within_original_ttl(self, db):
        """Insight with unknown feedback='bogus' at 100h should be suppressed (100 < 168)."""
        engine = _make_engine(db)
        dedup_key = "unknown_feedback_test_1"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="bogus", age_hours=100)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 0, "Unknown feedback should use original TTL (suppressed)"

    def test_allowed_past_original_ttl(self, db):
        """Insight with unknown feedback='bogus' at 200h should be allowed (200 > 168)."""
        engine = _make_engine(db)
        dedup_key = "unknown_feedback_test_2"

        with db.get_connection("user_model") as conn:
            _insert_insight_row(conn, dedup_key=dedup_key, feedback="bogus", age_hours=200)

        insight = _make_insight(dedup_key=dedup_key, staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "Unknown feedback should use original TTL (allowed)"


class TestDedupFreshInsights:
    """Insights with no prior row in the DB are always fresh."""

    def test_no_prior_row_is_fresh(self, db):
        """An insight with a dedup_key not in the DB should pass through."""
        engine = _make_engine(db)
        insight = _make_insight(dedup_key="brand_new_key", staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "Insight with no prior DB row should be treated as fresh"


class TestDBErrorTreatsAsFresh:
    """If the DB query fails, the insight passes through (fail-open)."""

    def test_db_error_passes_through(self, db):
        """Simulate a DB error by dropping the insights table."""
        engine = _make_engine(db)

        # Drop the insights table to force a DB error
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS insights")
            conn.commit()

        insight = _make_insight(dedup_key="error_test", staleness_ttl_hours=168)
        result = engine._deduplicate([insight])
        assert len(result) == 1, "DB error should fail-open and treat insight as fresh"
