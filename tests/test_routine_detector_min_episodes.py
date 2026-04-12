"""
Tests for the adaptive min_episodes threshold in RoutineDetector.

Validates that _effective_min_episodes() scales the fallback trigger threshold
based on data age, allowing cold-start email-dominated installations to detect
routines long before 50 typed episodes have accumulated.

Test scenarios:
- Threshold tiers: correct value for each data_age_days tier
- Cold-start (5-day system): 15 typed episodes trigger routine detection via fallback
- Mature system (30+ days): threshold stays at 50 (hardcoded base)
- Fallback path triggered when primary episodes < adaptive threshold
- 0 episodes returns early without error
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.routine_detector.detector import RoutineDetector
from storage.event_store import EventStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_event(event_store: EventStore, *, event_type: str, timestamp: datetime) -> str:
    """Insert a test event into the events database and return its ID."""
    event_id = str(uuid.uuid4())
    event_store.store_event(
        {
            "id": event_id,
            "type": event_type,
            "source": "test",
            "timestamp": timestamp.isoformat(),
            "priority": "normal",
            "payload": json.dumps({}),
            "metadata": json.dumps({}),
        }
    )
    return event_id


def _create_episode(
    user_model_store,
    *,
    timestamp: datetime,
    event_id: str,
    interaction_type: str = "unknown",
) -> dict:
    """Insert a test episode into the user_model database."""
    episode = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "event_id": event_id,
        "interaction_type": interaction_type,
        "content_summary": interaction_type,
    }
    user_model_store.store_episode(episode)
    return episode


# ---------------------------------------------------------------------------
# Unit tests for _effective_min_episodes()
# ---------------------------------------------------------------------------

class TestEffectiveMinEpisodes:
    """Unit tests for the _effective_min_episodes() method threshold tiers."""

    def test_first_week_returns_10(self, db, user_model_store):
        """data_age_days < 7 should return threshold of 10 (most lenient)."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=5, data_age_days=3)
        assert result == 10

    def test_boundary_day_6_returns_10(self, db, user_model_store):
        """data_age_days == 6 is still within the < 7 tier, should return 10."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=0, data_age_days=6)
        assert result == 10

    def test_second_week_returns_20(self, db, user_model_store):
        """data_age_days in [7, 14) should return threshold of 20."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=15, data_age_days=10)
        assert result == 20

    def test_boundary_day_7_returns_20(self, db, user_model_store):
        """data_age_days == 7 moves into the second tier, should return 20."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=8, data_age_days=7)
        assert result == 20

    def test_first_month_returns_35(self, db, user_model_store):
        """data_age_days in [14, 30) should return threshold of 35."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=25, data_age_days=20)
        assert result == 35

    def test_boundary_day_14_returns_35(self, db, user_model_store):
        """data_age_days == 14 moves into the first-month tier, should return 35."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=0, data_age_days=14)
        assert result == 35

    def test_mature_system_returns_base_50(self, db, user_model_store):
        """data_age_days >= 30 should return the full base threshold (50)."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=48, data_age_days=30)
        assert result == detector.min_episodes_for_detection == 50

    def test_very_old_system_returns_base_50(self, db, user_model_store):
        """A system with 200 days of data should still cap at 50."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=200, data_age_days=200)
        assert result == 50

    def test_zero_data_age_returns_10(self, db, user_model_store):
        """data_age_days == 0 (cold-start with no typed episodes) should return 10."""
        detector = RoutineDetector(db, user_model_store)
        result = detector._effective_min_episodes(episode_count=0, data_age_days=0)
        assert result == 10

    def test_custom_base_threshold_respected(self, db, user_model_store):
        """If min_episodes_for_detection is overridden, the mature tier should use it."""
        detector = RoutineDetector(db, user_model_store)
        detector.min_episodes_for_detection = 100
        result = detector._effective_min_episodes(episode_count=80, data_age_days=60)
        # Mature tier should use the overridden value (100), not the literal 50
        assert result == 100


# ---------------------------------------------------------------------------
# Integration tests — 5-day-old system with 15 typed episodes
# ---------------------------------------------------------------------------

class TestColdStartDetection:
    """Verify that a 5-day-old system with only 15 typed episodes can detect routines."""

    def test_detection_works_with_15_episodes_on_5_day_system(self, db, user_model_store):
        """15 typed episodes on a 5-day system should be enough to detect routines.

        The adaptive threshold for data_age_days=5 is 10, so 15 episodes >= 10
        and the fallback should NOT be needed (primary path runs).  A morning
        email routine across 5 days should be detected.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        # 5-day-old system: episodes from 5 days ago up to today
        now = datetime.now(UTC)
        base = now - timedelta(days=5)

        # Place 15 email_check episodes at 9am across 5 days (3 per day)
        for day in range(5):
            for minute in (0, 20, 40):
                ts = base.replace(hour=9, minute=minute, second=0, microsecond=0) + timedelta(days=day)
                event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
                _create_episode(
                    user_model_store,
                    timestamp=ts,
                    event_id=event_id,
                    interaction_type="email_check",
                )

        # 15 typed episodes, data_age_days=5 → effective threshold = 10
        # 15 >= 10, so the fallback should not be triggered
        import unittest.mock as mock

        with mock.patch.object(detector, "_fallback_temporal_episodes") as mock_fallback:
            routines = detector._detect_temporal_routines(lookback_days=10)
            # Fallback is NOT triggered because 15 >= 10
            mock_fallback.assert_not_called()

        # Should still detect a morning routine
        assert len(routines) >= 1, (
            f"Expected at least 1 routine on 5-day system with 15 episodes, got {routines}"
        )
        triggers = {r["trigger"] for r in routines}
        assert "morning" in triggers, f"Expected 'morning' routine, got triggers: {triggers}"

    def test_fallback_triggered_when_primary_below_adaptive_threshold(self, db, user_model_store):
        """When primary episodes < adaptive threshold, fallback path activates.

        On a 5-day system (threshold=10): if only 5 typed episodes exist (5<10),
        the fallback fires.  The fallback-derived episodes (from email.received
        events with unknown interaction_type) should allow routine detection.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        now = datetime.now(UTC)
        base = now - timedelta(days=5)

        # Only 5 typed episodes (below the adaptive threshold of 10 for 5-day system)
        for day in range(5):
            ts = base.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        # 30 more episodes with 'unknown' type that will be recovered by fallback
        for day in range(5):
            for minute in range(1, 7):
                ts = base.replace(hour=9, minute=minute * 5, second=0, microsecond=0) + timedelta(days=day)
                event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
                _create_episode(
                    user_model_store,
                    timestamp=ts,
                    event_id=event_id,
                    interaction_type="unknown",
                )

        # 5 typed < 10 threshold → fallback fires
        import unittest.mock as mock

        with mock.patch.object(
            detector, "_fallback_temporal_episodes", wraps=detector._fallback_temporal_episodes
        ) as mock_fallback:
            routines = detector._detect_temporal_routines(lookback_days=10)
            mock_fallback.assert_called_once()

        # Should detect a routine from the fallback-recovered data
        assert len(routines) >= 1, "Expected at least 1 routine after fallback on 5-day system"


# ---------------------------------------------------------------------------
# Integration tests — 30+ day mature system keeps threshold at 50
# ---------------------------------------------------------------------------

class TestMatureSystemThreshold:
    """Verify that the threshold stays at 50 for a 30+ day system."""

    def test_threshold_is_50_for_30_day_system(self, db, user_model_store):
        """On a 30-day system, the effective threshold should be 50 (unchanged).

        With 100 typed episodes (well above 50), the fallback should NOT fire.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        now = datetime.now(UTC)
        base = now - timedelta(days=35)  # 35 days of data → data_age_days=35 → threshold=50

        # 100 typed episodes spread across 30 days at 9am
        for i in range(100):
            day_offset = i % 30
            ts = base.replace(hour=9, minute=i % 60, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        import unittest.mock as mock

        with mock.patch.object(detector, "_fallback_temporal_episodes") as mock_fallback:
            routines = detector._detect_temporal_routines(lookback_days=60)
            # 100 >= 50 → fallback is NOT triggered
            mock_fallback.assert_not_called()

        # Morning routine should still be detected
        assert len(routines) >= 1

    def test_fallback_triggers_at_49_episodes_on_mature_system(self, db, user_model_store):
        """On a 30+ day system with exactly 49 typed episodes, fallback fires.

        49 < 50 (the mature threshold) → fallback activates.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        now = datetime.now(UTC)
        base = now - timedelta(days=35)

        # 49 typed episodes spread across 30 days at 9am
        for i in range(49):
            day_offset = i % 30
            ts = base.replace(hour=9, minute=i % 60, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        import unittest.mock as mock

        with mock.patch.object(
            detector, "_fallback_temporal_episodes", return_value=[]
        ) as mock_fallback:
            detector._detect_temporal_routines(lookback_days=60)
            # 49 < 50 → fallback is called
            mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Edge case — 0 episodes returns early without error
# ---------------------------------------------------------------------------

class TestZeroEpisodesEdgeCase:
    """Verify that 0 episodes results in early return without error."""

    def test_zero_episodes_returns_empty_list(self, db, user_model_store):
        """With no episodes at all, detect_routines() should return an empty list."""
        detector = RoutineDetector(db, user_model_store)
        routines = detector.detect_routines(lookback_days=30)
        assert routines == [], f"Expected [] but got {routines}"

    def test_zero_episodes_no_exception(self, db, user_model_store):
        """_detect_temporal_routines() with 0 episodes must not raise."""
        detector = RoutineDetector(db, user_model_store)
        try:
            result = detector._detect_temporal_routines(lookback_days=30)
        except Exception as exc:
            pytest.fail(f"_detect_temporal_routines raised on 0 episodes: {exc}")
        assert isinstance(result, list)

    def test_zero_typed_episodes_uses_lenient_threshold(self, db, user_model_store):
        """With 0 typed episodes but many unknown-typed ones, the fallback fires.

        data_age_days defaults to 0 when no typed episodes exist, giving an
        effective threshold of 10.  0 < 10, so the fallback runs.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        now = datetime.now(UTC)
        base = now - timedelta(days=3)

        # No typed episodes, only 'unknown'-typed ones linked to email.received
        for day in range(3):
            for minute in range(5):
                ts = base.replace(hour=9, minute=minute * 10, second=0, microsecond=0) + timedelta(days=day)
                event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
                _create_episode(
                    user_model_store,
                    timestamp=ts,
                    event_id=event_id,
                    interaction_type="unknown",
                )

        import unittest.mock as mock

        with mock.patch.object(
            detector, "_fallback_temporal_episodes", wraps=detector._fallback_temporal_episodes
        ) as mock_fallback:
            detector._detect_temporal_routines(lookback_days=10)
            # 0 typed < 10 (lenient cold-start threshold) → fallback fires
            mock_fallback.assert_called_once()
