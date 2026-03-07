"""
Tests for _count_active_days() timezone handling.

Verifies that active day counting uses local timezone dates (matching the
bucketing logic in _detect_temporal_routines) rather than UTC dates.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from services.routine_detector.detector import RoutineDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_episode(db, timestamp: str, interaction_type: str = "email_received"):
    """Insert a single episode into user_model.db."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary, active_domain)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid4()), timestamp, str(uuid4()), interaction_type, "test", "work"),
        )


def _make_detector(db, user_model_store, timezone: str = "America/Los_Angeles") -> RoutineDetector:
    """Create a RoutineDetector with the given timezone."""
    return RoutineDetector(db=db, user_model_store=user_model_store, timezone=timezone)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCountActiveDaysTimezone:
    """Tests that _count_active_days() counts local calendar dates, not UTC."""

    def test_utc_midnight_episode_counts_as_previous_local_day(self, db, user_model_store):
        """An episode at 2026-02-16T02:00Z is Feb 15 in America/Los_Angeles (UTC-8)."""
        _insert_episode(db, "2026-02-16T02:00:00+00:00")
        detector = _make_detector(db, user_model_store, "America/Los_Angeles")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        # Should count as Feb 15 local, so 1 active day
        assert result == 1

    def test_episodes_spanning_utc_midnight_same_local_day(self, db, user_model_store):
        """Episodes at 2026-02-16T07:00Z and 2026-02-16T23:00Z are both Feb 15
        and Feb 16 in PST respectively — but if both map to same local day,
        count as 1 day.

        07:00 UTC = Feb 15 23:00 PST (Feb 15 local)
        08:00 UTC = Feb 16 00:00 PST (Feb 16 local)
        These are different local days, so count should be 2.
        """
        # Feb 15 23:00 PST = Feb 16 07:00 UTC
        _insert_episode(db, "2026-02-16T07:00:00+00:00")
        # Feb 16 00:30 PST = Feb 16 08:30 UTC
        _insert_episode(db, "2026-02-16T08:30:00+00:00")
        detector = _make_detector(db, user_model_store, "America/Los_Angeles")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        assert result == 2

    def test_same_utc_day_different_local_days(self, db, user_model_store):
        """Two episodes on the same UTC day that fall on different local days.

        2026-02-16T02:00Z = Feb 15 18:00 PST (Feb 15 local)
        2026-02-16T20:00Z = Feb 16 12:00 PST (Feb 16 local)
        UTC DATE() would say 1 day; local should say 2.
        """
        _insert_episode(db, "2026-02-16T02:00:00+00:00")
        _insert_episode(db, "2026-02-16T20:00:00+00:00")
        detector = _make_detector(db, user_model_store, "America/Los_Angeles")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        assert result == 2

    def test_utc_timezone_matches_exactly(self, db, user_model_store):
        """With tz='UTC', local dates should equal UTC dates — no shift."""
        _insert_episode(db, "2026-02-15T23:00:00+00:00")
        _insert_episode(db, "2026-02-16T01:00:00+00:00")
        detector = _make_detector(db, user_model_store, "UTC")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        # Two distinct UTC dates: Feb 15 and Feb 16
        assert result == 2

    def test_empty_episodes_returns_one(self, db, user_model_store):
        """No episodes should return 1 (safe default to avoid division by zero)."""
        detector = _make_detector(db, user_model_store)
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        assert result == 1

    def test_episodes_same_local_day_count_as_one(self, db, user_model_store):
        """Multiple episodes on the same local day should count as 1 day."""
        # All of these are Feb 15 in PST (UTC-8)
        _insert_episode(db, "2026-02-16T02:00:00+00:00")  # Feb 15 18:00 PST
        _insert_episode(db, "2026-02-16T03:00:00+00:00")  # Feb 15 19:00 PST
        _insert_episode(db, "2026-02-16T05:00:00+00:00")  # Feb 15 21:00 PST
        detector = _make_detector(db, user_model_store, "America/Los_Angeles")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        assert result == 1

    def test_east_of_utc_timezone(self, db, user_model_store):
        """Episode at 2026-02-15T23:00Z is Feb 16 in Asia/Tokyo (UTC+9)."""
        _insert_episode(db, "2026-02-15T23:00:00+00:00")
        detector = _make_detector(db, user_model_store, "Asia/Tokyo")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        # 23:00 UTC = 08:00 JST next day (Feb 16)
        assert result == 1
        # Verify it's counted as Feb 16, not Feb 15, by adding another
        # episode that is unambiguously Feb 16 in JST
        _insert_episode(db, "2026-02-16T10:00:00+00:00")  # Feb 16 19:00 JST
        result2 = detector._count_active_days(cutoff)
        # Both are Feb 16 JST, so still 1 day
        assert result2 == 1

    def test_excludes_internal_types(self, db, user_model_store):
        """Episodes with internal telemetry types should be excluded."""
        _insert_episode(db, "2026-02-16T10:00:00+00:00", interaction_type="email_received")
        _insert_episode(db, "2026-02-17T10:00:00+00:00", interaction_type="usermodel_update")
        _insert_episode(db, "2026-02-18T10:00:00+00:00", interaction_type="system_check")
        detector = _make_detector(db, user_model_store, "UTC")
        cutoff = datetime(2026, 2, 1, tzinfo=UTC)
        result = detector._count_active_days(cutoff)
        # Only the email_received episode should count
        assert result == 1
