"""
Tests for RoutineDetector timezone-aware time-of-day bucketing.

Validates that episode timestamps stored in UTC are correctly converted to the
user's local timezone before bucketing into time-of-day slots (morning, midday,
afternoon, evening, night).  Without this conversion, a user in UTC-5 would
have their 7 AM morning activity bucketed as 'midday' (12:00 UTC).

Uses the same fixture patterns as tests/test_routine_detector.py.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector


class TestHourToBucket:
    """Unit tests for the static _hour_to_bucket helper."""

    @pytest.mark.parametrize(
        "hour, expected",
        [
            (0, "night"),
            (1, "night"),
            (4, "night"),
            (5, "morning"),
            (7, "morning"),
            (10, "morning"),
            (11, "midday"),
            (13, "midday"),
            (14, "afternoon"),
            (16, "afternoon"),
            (17, "evening"),
            (22, "evening"),
            (23, "night"),
        ],
    )
    def test_bucket_boundaries(self, hour, expected):
        """Each hour should map to the correct bucket at the boundary."""
        assert RoutineDetector._hour_to_bucket(hour) == expected


class TestTemporalRoutineTimezone:
    """Test that temporal routine detection uses the configured timezone."""

    def test_morning_utc_minus5_bucketed_correctly(self, db, user_model_store):
        """Episodes at 7 AM local (America/New_York, UTC-5) are stored as
        12:00 UTC.  Without timezone conversion, 12:00 UTC is bucketed as
        'midday'.  With conversion, it should be bucketed as 'morning'.
        """
        detector = RoutineDetector(db, user_model_store, timezone="America/New_York")

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        # Insert 5 episodes at 12:00 UTC = 7:00 AM Eastern (UTC-5 in winter)
        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=12)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "check_email",
                "content_summary": "Morning email check",
            })
            # Second action shortly after for the routine to have multiple steps
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=15)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "review_calendar",
                "content_summary": "Review calendar",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        # Should detect a MORNING routine, not midday
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        midday_routines = [r for r in routines if r["trigger"] == "midday"]

        assert len(morning_routines) >= 1, (
            "Expected 'morning' routine for 7 AM Eastern activities; "
            f"got triggers: {[r['trigger'] for r in routines]}"
        )
        # Midday should NOT contain these actions
        midday_actions = set()
        for r in midday_routines:
            midday_actions.update(s["action"] for s in r["steps"])
        assert "check_email" not in midday_actions, (
            "check_email at 7 AM local should NOT appear in midday bucket"
        )

    def test_utc_midnight_is_evening_in_utc_minus5(self, db, user_model_store):
        """Episodes at 00:00 UTC = 7:00 PM Eastern (UTC-5) should bucket as
        'evening' when timezone is America/New_York.
        """
        detector = RoutineDetector(db, user_model_store, timezone="America/New_York")

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=0)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "evening_reading",
                "content_summary": "Evening reading",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=20)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "evening_journaling",
                "content_summary": "Evening journaling",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        evening_routines = [r for r in routines if r["trigger"] == "evening"]
        assert len(evening_routines) >= 1, (
            "Expected 'evening' routine for 7 PM Eastern activities at 00:00 UTC; "
            f"got triggers: {[r['trigger'] for r in routines]}"
        )

    def test_default_utc_behavior(self, db, user_model_store):
        """When no timezone is configured (default UTC), bucketing should use
        raw UTC hours — same behavior as before the fix.
        """
        detector = RoutineDetector(db, user_model_store)  # Default timezone="UTC"

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        # Episodes at 08:00 UTC → should be 'morning' in UTC
        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=8)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "default_tz_action",
                "content_summary": "Default TZ action",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "default_tz_followup",
                "content_summary": "Default TZ followup",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected 'morning' routine for 8 AM UTC with default timezone; "
            f"got triggers: {[r['trigger'] for r in routines]}"
        )

    def test_positive_offset_timezone_asia_tokyo(self, db, user_model_store):
        """Asia/Tokyo is UTC+9.  An episode at 22:00 UTC = 07:00 JST (next day).
        This should bucket as 'morning' in Tokyo timezone.
        """
        detector = RoutineDetector(db, user_model_store, timezone="Asia/Tokyo")

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        # 22:00 UTC = 07:00 JST next day → morning bucket
        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=22)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "tokyo_morning_email",
                "content_summary": "Tokyo morning email",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=15)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "tokyo_morning_calendar",
                "content_summary": "Tokyo morning calendar",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected 'morning' routine for 7 AM Tokyo (22:00 UTC); "
            f"got triggers: {[r['trigger'] for r in routines]}"
        )
        # Verify the actions are NOT in evening (22 UTC would be evening without tz conversion)
        evening_actions = set()
        for r in routines:
            if r["trigger"] == "evening":
                evening_actions.update(s["action"] for s in r["steps"])
        assert "tokyo_morning_email" not in evening_actions, (
            "tokyo_morning_email at 7 AM JST should NOT appear in evening bucket"
        )

    def test_afternoon_utc_is_morning_in_utc_minus5(self, db, user_model_store):
        """14:00 UTC = 9:00 AM Eastern (UTC-5).  Without timezone conversion,
        this would be bucketed as 'afternoon'.  With 'America/New_York', it
        should be 'morning'.
        """
        detector = RoutineDetector(db, user_model_store, timezone="America/New_York")

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=14)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "morning_standup",
                "content_summary": "Morning standup meeting",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "morning_notes",
                "content_summary": "Morning notes",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected 'morning' routine for 9 AM Eastern (14:00 UTC); "
            f"got triggers: {[r['trigger'] for r in routines]}"
        )

    def test_full_detect_routines_with_timezone(self, db, user_model_store):
        """Integration test: detect_routines() end-to-end with a non-UTC timezone."""
        detector = RoutineDetector(db, user_model_store, timezone="America/New_York")

        base_date = datetime(2026, 2, 15, tzinfo=timezone.utc)

        # Create 7 AM local (12:00 UTC) morning routine over 5 days
        for day_offset in range(5):
            utc_time = base_date + timedelta(days=day_offset, hours=12)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": utc_time.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "full_test_email",
                "content_summary": "Full test email",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (utc_time + timedelta(minutes=15)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "full_test_calendar",
                "content_summary": "Full test calendar",
            })

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "End-to-end detect_routines() should find morning routine for "
            "7 AM Eastern activities"
        )

    def test_timestamps_without_tzinfo_treated_as_utc(self, db, user_model_store):
        """Episode timestamps without timezone info (naive) should be treated as UTC."""
        detector = RoutineDetector(db, user_model_store, timezone="America/New_York")

        base_date = datetime(2026, 2, 15)

        # Insert naive timestamps (no +00:00 suffix) at 12:00 = 7 AM Eastern
        for day_offset in range(5):
            naive_time = base_date + timedelta(days=day_offset, hours=12)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": naive_time.isoformat(),  # No timezone suffix
                "event_id": str(uuid.uuid4()),
                "interaction_type": "naive_ts_action",
                "content_summary": "Naive timestamp action",
            })
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": (naive_time + timedelta(minutes=10)).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "naive_ts_followup",
                "content_summary": "Naive timestamp followup",
            })

        routines = detector._detect_temporal_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Naive timestamps should be treated as UTC and converted; "
            "12:00 naive → 7 AM Eastern → morning bucket"
        )
