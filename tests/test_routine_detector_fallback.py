"""
Tests for the episode-based fallback routine detection in RoutineDetector.

Validates that when signal profiles are unavailable (< 25 samples), the
routine detector falls back to classifying episodes directly from their
linked event types and still detects recurring patterns.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.routine_detector.detector import EVENT_TYPE_TO_ACTIVITY, RoutineDetector
from storage.event_store import EventStore


def _create_episode(user_model_store, *, timestamp, event_id, interaction_type="unknown", **kwargs):
    """Helper to create a test episode with sensible defaults."""
    episode = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        "event_id": event_id,
        "interaction_type": interaction_type,
        "content_summary": kwargs.get("content_summary", interaction_type),
        **{k: v for k, v in kwargs.items() if k != "content_summary"},
    }
    user_model_store.store_episode(episode)
    return episode


def _create_event(event_store, *, event_type, timestamp=None):
    """Helper to create a test event in the events database and return its ID."""
    event_id = str(uuid.uuid4())
    ts = timestamp or datetime.now(timezone.utc)
    event_store.store_event({
        "id": event_id,
        "type": event_type,
        "source": "test",
        "timestamp": ts.isoformat(),
        "priority": "normal",
        "payload": json.dumps({}),
        "metadata": json.dumps({}),
    })
    return event_id


class TestEpisodeFallbackDetection:
    """Tests for _detect_routines_from_episodes_fallback()."""

    def test_fallback_detects_daily_email_routine(self, db, user_model_store):
        """Fallback should detect a daily email routine from episode data.

        Creates 20 episodes at 9am on 15 different days, all linked to
        'email.received' events. The fallback should classify them as
        'email_check' and detect a morning routine.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=15)

        # Create email.received events and episodes on 15 different days at 9am
        for day_offset in range(15):
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",  # No usable interaction_type
                content_summary="Email received",
            )

        routines = detector._detect_routines_from_episodes_fallback(lookback_days=30)

        # Should detect at least one morning routine with email_check activity
        assert len(routines) >= 1
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1

        # Check that email_check is among the detected steps
        routine = morning_routines[0]
        email_steps = [s for s in routine["steps"] if s["action"] == "email_check"]
        assert len(email_steps) >= 1
        assert routine["consistency_score"] >= 0.6

    def test_fallback_uses_existing_interaction_type_when_available(self, db, user_model_store):
        """Fallback should prefer existing interaction_type over event lookup.

        Episodes with a meaningful interaction_type should use it directly
        instead of falling back to EVENT_TYPE_TO_ACTIVITY mapping.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=str(uuid.uuid4()),
                interaction_type="morning_workout",
                content_summary="Morning workout",
            )

        routines = detector._detect_routines_from_episodes_fallback(lookback_days=30)

        assert len(routines) >= 1
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1
        workout_steps = [s for s in morning_routines[0]["steps"] if s["action"] == "morning_workout"]
        assert len(workout_steps) >= 1

    def test_fallback_consistency_threshold_scaling(self, db, user_model_store):
        """Fallback should use the same _effective_consistency_threshold() scaling.

        With < 7 active days, the threshold should be 0.3 (lenient), allowing
        sparser patterns to be detected.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        # Create episodes on 5 days, with email at 9am on 3 of them
        for day_offset in range(5):
            ts_filler = base_date.replace(hour=14, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            _create_episode(
                user_model_store,
                timestamp=ts_filler,
                event_id=str(uuid.uuid4()),
                interaction_type=f"filler_{day_offset}",
                content_summary=f"Filler {day_offset}",
            )

        # Morning email on 3 of 5 days → consistency 0.6, passes 0.3 threshold
        for day_offset in [0, 2, 4]:
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
                content_summary="Email check",
            )

        routines = detector._detect_routines_from_episodes_fallback(lookback_days=30)

        # With 5 active days, threshold is 0.3. 3/5 = 0.6 should pass.
        assert len(routines) >= 1
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1

    def test_fallback_rejects_low_consistency(self, db, user_model_store):
        """Fallback should not detect routines below the effective threshold.

        With 35 active days, threshold is 0.6. A pattern appearing on 5 of 35
        days (consistency ~0.14) should not be detected.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=35)

        # Create filler on all 35 days
        for day_offset in range(35):
            ts = base_date.replace(hour=14, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=str(uuid.uuid4()),
                interaction_type=f"filler_rej_{day_offset}",
                content_summary=f"Filler {day_offset}",
            )

        # Email on only 5 of 35 days → consistency ~0.14
        for day_offset in [0, 7, 14, 21, 28]:
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
                content_summary="Sparse email",
            )

        routines = detector._detect_routines_from_episodes_fallback(lookback_days=40)

        # 5/35 ≈ 0.14 < 0.6 threshold → should not detect
        email_routines = [
            r for r in routines
            if any(s["action"] == "email_check" for s in r.get("steps", []))
        ]
        assert len(email_routines) == 0

    def test_fallback_marks_detection_method(self, db, user_model_store):
        """Routines from the fallback should include detection_method='episode_fallback'."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        base_date = datetime.now(timezone.utc) - timedelta(days=10)
        for day_offset in range(10):
            ts = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=str(uuid.uuid4()),
                interaction_type="daily_standup",
                content_summary="Daily standup",
            )

        routines = detector._detect_routines_from_episodes_fallback(lookback_days=30)

        assert len(routines) >= 1
        assert all(r.get("detection_method") == "episode_fallback" for r in routines)


class TestFallbackIntegration:
    """Tests that detect_routines() correctly invokes the fallback path."""

    def test_fallback_not_called_when_temporal_has_sufficient_samples(self, db, user_model_store):
        """Fallback should NOT be called when temporal profile has >= 25 samples.

        Even if temporal detection returns 0 routines, the fallback should be
        skipped when the profile has enough data — the lack of routines is a
        genuine result, not a data availability issue.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        # Store a temporal profile with 30 samples
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                ("temporal", json.dumps({"test": True}), 30, datetime.now(timezone.utc).isoformat()),
            )

        with patch.object(detector, "_detect_routines_from_episodes_fallback") as mock_fallback:
            detector.detect_routines(lookback_days=30)
            mock_fallback.assert_not_called()

    def test_fallback_called_when_temporal_has_insufficient_samples(self, db, user_model_store):
        """Fallback should be called when temporal profile has < 25 samples
        AND primary temporal detection returns 0 routines.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        # Store a temporal profile with only 10 samples
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                ("temporal", json.dumps({"test": True}), 10, datetime.now(timezone.utc).isoformat()),
            )

        with patch.object(
            detector, "_detect_routines_from_episodes_fallback", return_value=[]
        ) as mock_fallback:
            detector.detect_routines(lookback_days=30)
            mock_fallback.assert_called_once_with(30)

    def test_fallback_called_when_no_temporal_profile_exists(self, db, user_model_store):
        """Fallback should be called when temporal profile doesn't exist at all.

        Missing profile means 0 samples, which is < 25 threshold.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        with patch.object(
            detector, "_detect_routines_from_episodes_fallback", return_value=[]
        ) as mock_fallback:
            detector.detect_routines(lookback_days=30)
            mock_fallback.assert_called_once_with(30)

    def test_fallback_not_called_when_temporal_finds_routines(self, db, user_model_store):
        """Fallback should NOT be called when temporal detection returns routines.

        If the primary path succeeds, there's no need for the fallback
        regardless of profile sample count.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        # Create a real temporal pattern that primary detection will find
        base_date = datetime.now(timezone.utc) - timedelta(days=10)
        for day_offset in range(10):
            ts = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=str(uuid.uuid4()),
                interaction_type="real_morning_email",
                content_summary="Morning email",
            )
            _create_episode(
                user_model_store,
                timestamp=ts + timedelta(minutes=15),
                event_id=str(uuid.uuid4()),
                interaction_type="real_morning_calendar",
                content_summary="Calendar review",
            )

        with patch.object(detector, "_detect_routines_from_episodes_fallback") as mock_fallback:
            routines = detector.detect_routines(lookback_days=30)
            # Primary should find routines, so fallback is not called
            mock_fallback.assert_not_called()
            assert len(routines) >= 1

    def test_end_to_end_fallback_produces_routines(self, db, user_model_store):
        """End-to-end test: when primary temporal detection fails completely
        (e.g., its internal fallback also returns nothing) and no temporal
        profile exists, detect_routines() should use the episode fallback.

        We simulate this by patching _detect_temporal_routines to return []
        while having real episode data that the fallback can classify.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=10)

        for day_offset in range(10):
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
                content_summary="Email received",
            )

        # Simulate primary temporal detection failing (returns empty)
        # No temporal profile exists → 0 samples → fallback should trigger
        with patch.object(detector, "_detect_temporal_routines", return_value=[]):
            routines = detector.detect_routines(lookback_days=30)

        # The fallback should detect a morning email_check routine
        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1
        email_steps = [s for s in morning_routines[0]["steps"] if s["action"] == "email_check"]
        assert len(email_steps) >= 1


class TestEventTypeToActivityMapping:
    """Tests for the EVENT_TYPE_TO_ACTIVITY mapping and classification."""

    def test_mapping_contains_expected_entries(self):
        """The mapping should contain the core event type classifications."""
        assert EVENT_TYPE_TO_ACTIVITY["email.received"] == "email_check"
        assert EVENT_TYPE_TO_ACTIVITY["email.sent"] == "email_compose"
        assert EVENT_TYPE_TO_ACTIVITY["calendar.event.created"] == "calendar_review"
        assert EVENT_TYPE_TO_ACTIVITY["notification.created"] == "notification_check"
        assert EVENT_TYPE_TO_ACTIVITY["system.connector.sync_complete"] == "system_maintenance"

    def test_classify_event_type_exact_match(self, db, user_model_store):
        """_classify_event_type_to_activity should match event types exactly."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        event_id = _create_event(event_store, event_type="email.received")
        result = detector._classify_event_type_to_activity(event_id)
        assert result == "email_check"

    def test_classify_event_type_prefix_match(self, db, user_model_store):
        """_classify_event_type_to_activity should match by prefix fallback."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        # "email.received.important" should match "email.received" prefix
        event_id = _create_event(event_store, event_type="email.received.important")
        result = detector._classify_event_type_to_activity(event_id)
        assert result == "email_check"

    def test_classify_event_type_unknown_returns_none(self, db, user_model_store):
        """_classify_event_type_to_activity should return None for unknown types."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        event_id = _create_event(event_store, event_type="some.unknown.event.type")
        result = detector._classify_event_type_to_activity(event_id)
        assert result is None

    def test_classify_event_type_missing_event(self, db, user_model_store):
        """_classify_event_type_to_activity should return None for missing events."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        result = detector._classify_event_type_to_activity("nonexistent-event-id")
        assert result is None


class TestGetTemporalProfileSampleCount:
    """Tests for _get_temporal_profile_sample_count()."""

    def test_returns_zero_when_no_profile_exists(self, db, user_model_store):
        """Should return 0 when no temporal profile has been stored."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        assert detector._get_temporal_profile_sample_count() == 0

    def test_returns_sample_count_from_profile(self, db, user_model_store):
        """Should return the samples_count from the stored temporal profile."""
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, ?)""",
                ("temporal", json.dumps({"test": True}), 42, datetime.now(timezone.utc).isoformat()),
            )

        assert detector._get_temporal_profile_sample_count() == 42
