"""
Integration tests for routine detector temporal detection with realistic episode data.

Verifies the full end-to-end flow: episodes in the database → detect_routines() → stored routines.
Uses real DatabaseManager and UserModelStore instances (no mocks) with temporary SQLite databases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from services.routine_detector.detector import RoutineDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# PDT offset: UTC-7.  A 7 AM Pacific timestamp is 14:00 UTC.
_PDT_OFFSET_HOURS = 7


def _utc_for_pacific_hour(day_offset: int, local_hour: int, base_date: datetime | None = None) -> str:
    """Return an ISO 8601 UTC timestamp corresponding to *local_hour* in US/Pacific.

    Args:
        day_offset: Number of days before 'now' (0 = today, 1 = yesterday, ...).
        local_hour: Desired hour in Pacific time (0-23).
        base_date: Optional fixed base date; defaults to ``datetime.now(UTC)``.

    Returns:
        ISO 8601 timestamp string in UTC.
    """
    base = base_date or datetime.now(UTC)
    day = base - timedelta(days=day_offset)
    # Replace time to the desired UTC hour that maps to local_hour Pacific
    utc_hour = local_hour + _PDT_OFFSET_HOURS
    dt = day.replace(hour=utc_hour % 24, minute=0, second=0, microsecond=0)
    return dt.isoformat()


def _insert_episode(
    db,
    timestamp: str,
    interaction_type: str = "email_received",
    event_id: str | None = None,
    content_summary: str = "Test episode",
    active_domain: str = "work",
):
    """Insert a single episode into user_model.db."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary, active_domain)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid4()),
                timestamp,
                event_id or str(uuid4()),
                interaction_type,
                content_summary,
                active_domain,
            ),
        )


def _insert_event(db, event_id: str, event_type: str, timestamp: str):
    """Insert a single event into events.db."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
               (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_id, event_type, "google", timestamp, "normal", "{}", "{}"),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMorningEmailRoutineDetected:
    """Insert 10 morning email episodes across 10 days; expect a 'Morning routine'."""

    def test_morning_email_routine_detected(self, db, user_model_store):
        """Episodes at 7-9 AM Pacific for 10 consecutive days should produce a morning routine."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        # Lower min_episodes_for_detection so we don't need 50+ episodes
        detector.min_episodes_for_detection = 5

        base = datetime.now(UTC)
        for day in range(10):
            # 7 AM Pacific = 14:00 UTC (during PDT)
            ts = _utc_for_pacific_hour(day, local_hour=7, base_date=base)
            _insert_episode(db, ts, interaction_type="email_received")

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1, f"Expected at least 1 routine, got {len(routines)}"
        morning_routines = [r for r in routines if "morning" in r["trigger"].lower()]
        assert len(morning_routines) >= 1, (
            f"Expected a morning routine, got triggers: {[r['trigger'] for r in routines]}"
        )
        # Verify steps contain the email action
        steps = morning_routines[0]["steps"]
        step_actions = [s["action"] for s in steps]
        assert "email_received" in step_actions, (
            f"Expected 'email_received' in steps, got {step_actions}"
        )


class TestEveningRoutineMultipleTypes:
    """Insert episodes for 8 evenings with two different types; expect multi-step routine."""

    def test_evening_routine_multiple_types(self, db, user_model_store):
        """Two event types at 6-8 PM Pacific for 8 evenings should produce an evening routine with 2+ steps."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        detector.min_episodes_for_detection = 5

        base = datetime.now(UTC)
        for day in range(8):
            # 6 PM Pacific = 01:00 UTC next day (during PDT) — evening bucket (17-22)
            ts1 = _utc_for_pacific_hour(day, local_hour=18, base_date=base)
            _insert_episode(db, ts1, interaction_type="email_sent")

            ts2 = _utc_for_pacific_hour(day, local_hour=19, base_date=base)
            _insert_episode(db, ts2, interaction_type="calendar_event_attended")

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1, f"Expected at least 1 routine, got {len(routines)}"
        evening_routines = [r for r in routines if "evening" in r["trigger"].lower()]
        assert len(evening_routines) >= 1, (
            f"Expected an evening routine, got triggers: {[r['trigger'] for r in routines]}"
        )
        steps = evening_routines[0]["steps"]
        assert len(steps) >= 2, f"Expected 2+ steps, got {len(steps)}: {steps}"


class TestInsufficientDaysNoRoutine:
    """Only 2 mornings of data — below min_occurrences=3.  Expect 0 routines."""

    def test_insufficient_days_no_routine(self, db, user_model_store):
        """With only 2 days of episodes, no routine should be detected."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        detector.min_episodes_for_detection = 1  # Don't skip to fallback

        base = datetime.now(UTC)
        for day in range(2):
            ts = _utc_for_pacific_hour(day, local_hour=8, base_date=base)
            _insert_episode(db, ts, interaction_type="email_received")

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) == 0, f"Expected 0 routines for only 2 days, got {len(routines)}"


class TestLowConsistencyNoRoutine:
    """Morning episodes on 3 of 15 active days (20%) — below 60% threshold.  Expect 0 routines."""

    def test_low_consistency_no_routine(self, db, user_model_store):
        """A pattern occurring on 20% of active days should not be detected as a routine."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        detector.min_episodes_for_detection = 5

        base = datetime.now(UTC)

        # Insert morning episodes on only 3 days
        for day in [0, 5, 10]:
            ts = _utc_for_pacific_hour(day, local_hour=7, base_date=base)
            _insert_episode(db, ts, interaction_type="email_received")

        # Insert midday episodes on all 15 days to establish 15 active days
        # Use a different type to avoid accidentally creating a midday routine
        # that passes consistency. Use a unique-per-day type so no single type
        # meets min_occurrences.
        for day in range(15):
            ts = _utc_for_pacific_hour(day, local_hour=12, base_date=base)
            _insert_episode(db, ts, interaction_type=f"activity_day_{day}")

        routines = detector.detect_routines(lookback_days=30)

        # Filter to morning routines only — other buckets shouldn't form routines either
        morning_routines = [r for r in routines if "morning" in r["trigger"].lower()]
        assert len(morning_routines) == 0, (
            f"Expected 0 morning routines with 20% consistency, got {len(morning_routines)}"
        )


class TestFallbackHandlesUnknownInteractionTypes:
    """Episodes with interaction_type='unknown' but valid event_ids should be recovered via fallback."""

    def test_fallback_handles_unknown_interaction_types(self, db, user_model_store):
        """The fallback should derive interaction types from linked events when episodes have 'unknown' type."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        # Force fallback by setting a high threshold — no 'unknown' episodes pass
        # the primary query filter.
        detector.min_episodes_for_detection = 5

        base = datetime.now(UTC)
        for day in range(10):
            event_id = str(uuid4())
            ts = _utc_for_pacific_hour(day, local_hour=8, base_date=base)

            # Insert the event with a real type in events.db
            _insert_event(db, event_id, "email.received", ts)

            # Insert the episode with 'unknown' interaction_type (simulates stale data)
            _insert_episode(
                db, ts,
                interaction_type="unknown",
                event_id=event_id,
            )

        routines = detector.detect_routines(lookback_days=30)

        assert len(routines) >= 1, (
            f"Expected fallback to recover routines from 'unknown' episodes, got {len(routines)}"
        )
        # The backfill should convert 'email.received' → 'email_received'
        all_step_actions = []
        for r in routines:
            all_step_actions.extend(s["action"] for s in r["steps"])
        assert "email_received" in all_step_actions, (
            f"Expected 'email_received' in routine steps after fallback, got {all_step_actions}"
        )


class TestRoutineStoredInDatabase:
    """After detect_routines(), calling store_routines() should persist to the routines table."""

    def test_routine_stored_in_database(self, db, user_model_store):
        """Detected routines should be retrievable from UserModelStore.get_routines() after storage."""
        detector = RoutineDetector(db, user_model_store, timezone="America/Los_Angeles")
        detector.min_episodes_for_detection = 5

        base = datetime.now(UTC)
        for day in range(10):
            ts = _utc_for_pacific_hour(day, local_hour=7, base_date=base)
            _insert_episode(db, ts, interaction_type="email_received")

        routines = detector.detect_routines(lookback_days=30)
        assert len(routines) >= 1, "Pre-condition: need at least 1 detected routine"

        # Store them via the detector's store_routines method
        stored_count = detector.store_routines(routines)
        assert stored_count >= 1, f"Expected at least 1 routine stored, got {stored_count}"

        # Verify persistence via UserModelStore
        stored = user_model_store.get_routines()
        assert len(stored) >= 1, f"Expected routines in DB, got {len(stored)}"

        # Verify the stored routine has expected fields
        routine = stored[0]
        assert routine["name"], "Stored routine should have a name"
        assert routine["trigger"], "Stored routine should have a trigger"
        assert len(routine["steps"]) >= 1, "Stored routine should have at least 1 step"
        assert routine["consistency_score"] > 0, "Stored routine should have positive consistency"
        assert routine["times_observed"] > 0, "Stored routine should have positive observation count"
