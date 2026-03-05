"""
Tests that the episode-based fallback runs unconditionally when primary
temporal detection returns 0 routines, regardless of profile sample count.

Previously the fallback was gated by ``profile_samples < 25``, meaning it
would stop running once enough signal profile samples accumulated — even if
primary detection still returned 0 routines.  The fix removes that guard so
the fallback always fires when primary detection finds nothing.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.routine_detector.detector import RoutineDetector
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


def _set_temporal_profile_samples(db, count):
    """Insert a temporal signal profile with the given samples_count."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, ?)""",
            ("temporal", json.dumps({"test": True}), count, datetime.now(timezone.utc).isoformat()),
        )


class TestFallbackRunsUnconditionally:
    """The episode fallback must run whenever primary detection returns 0
    routines, regardless of how many profile samples exist."""

    def test_fallback_runs_with_high_profile_samples(self, db, user_model_store):
        """Fallback should run when primary detection returns 0 routines
        AND profile_samples >= 25.

        This was the broken case: the old ``profile_samples < 25`` guard
        prevented the fallback from running once the temporal signal profile
        had accumulated enough samples.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        # Set profile samples well above the old threshold
        _set_temporal_profile_samples(db, 100)

        with patch.object(
            detector, "_detect_routines_from_episodes_fallback", return_value=[]
        ) as mock_fallback:
            detector.detect_routines(lookback_days=30)
            mock_fallback.assert_called_once_with(30)

    def test_fallback_runs_with_low_profile_samples(self, db, user_model_store):
        """Fallback should run when primary detection returns 0 routines
        AND profile_samples < 25 (the previously-working case).
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        _set_temporal_profile_samples(db, 10)

        with patch.object(
            detector, "_detect_routines_from_episodes_fallback", return_value=[]
        ) as mock_fallback:
            detector.detect_routines(lookback_days=30)
            mock_fallback.assert_called_once_with(30)

    def test_fallback_skipped_when_primary_finds_routines(self, db, user_model_store):
        """Fallback should NOT run when primary temporal detection returns
        routines, regardless of profile sample count.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")

        _set_temporal_profile_samples(db, 5)

        # Create enough real episode data for primary detection to find routines
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
            mock_fallback.assert_not_called()
            assert len(routines) >= 1

    def test_end_to_end_fallback_with_high_samples(self, db, user_model_store):
        """End-to-end: episode fallback produces routines even when profile
        has many samples, as long as primary detection returns empty.

        Patches _detect_temporal_routines to return [] and verifies the
        fallback still runs and detects routines from raw episode data.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        # High profile samples — old code would skip fallback here
        _set_temporal_profile_samples(db, 200)

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

        # Force primary detection to return empty
        with patch.object(detector, "_detect_temporal_routines", return_value=[]):
            routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1
        email_steps = [s for s in morning_routines[0]["steps"] if s["action"] == "email_check"]
        assert len(email_steps) >= 1
