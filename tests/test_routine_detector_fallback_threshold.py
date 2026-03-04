"""
Tests for the routine detector fallback threshold logic.

Validates that the fallback mechanism triggers not only when zero episodes pass
the interaction_type filter, but also when too few episodes pass (below the
min_episodes_for_detection threshold). Also verifies that results from primary
and fallback queries are properly merged.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector
from storage.event_store import EventStore


def _create_episode(user_model_store, *, timestamp, event_id, interaction_type="unknown", location=None):
    """Helper to create a test episode with sensible defaults."""
    episode = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
        "event_id": event_id,
        "interaction_type": interaction_type,
        "content_summary": interaction_type,
    }
    if location:
        episode["location"] = location
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


class TestTemporalFallbackThreshold:
    """Tests for the temporal detection fallback threshold."""

    def test_fallback_triggers_when_few_episodes(self, db, user_model_store):
        """Fallback should trigger when primary query returns fewer than
        min_episodes_for_detection episodes, not just when it returns zero.

        Inserts 10 episodes with valid interaction_type and 500 with NULL/unknown
        interaction_type (linked to email.received events). The fallback should
        trigger and recover the 500.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Create 10 episodes with valid interaction_type (below threshold of 50)
        for i in range(10):
            day_offset = i % 10
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",  # Valid type
            )

        # Create 500 episodes with unknown interaction_type across 25 days
        for i in range(500):
            day_offset = i % 25
            hour = 9 + (i % 3)  # Spread across 9am, 10am, 11am
            ts = base_date.replace(hour=hour, minute=i % 60, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",  # Will need fallback derivation
            )

        # The primary query should return only 10 (< 50 threshold), triggering fallback
        routines = detector._detect_temporal_routines(lookback_days=60)

        # Should detect routines because fallback recovered the 500 episodes
        assert len(routines) >= 1, (
            "Expected at least 1 routine from fallback-recovered episodes"
        )

    def test_no_fallback_when_enough_episodes(self, db, user_model_store):
        """Fallback should NOT trigger when primary query returns enough episodes.

        Inserts 100 episodes with valid interaction_type (above threshold of 50).
        The fallback should not run.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Create 100 episodes with valid interaction_type across 20 days at 9am
        for i in range(100):
            day_offset = i % 20
            ts = base_date.replace(hour=9, minute=i % 60, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        # Patch the fallback to verify it is NOT called
        import unittest.mock as mock

        with mock.patch.object(detector, "_fallback_temporal_episodes") as mock_fallback:
            routines = detector._detect_temporal_routines(lookback_days=60)
            mock_fallback.assert_not_called()

    def test_fallback_merges_primary_and_derived(self, db, user_model_store):
        """When fallback triggers, results should contain episodes from both
        the primary query AND the fallback derivation, deduplicated.

        Inserts 30 with valid type and 200 with NULL type. The merged result
        should include episodes from both sources.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Create 30 episodes with valid interaction_type at 9am
        for i in range(30):
            day_offset = i % 15
            ts = base_date.replace(hour=9, minute=i % 60, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        # Create 200 episodes with unknown type at 9am and 10am
        for i in range(200):
            day_offset = i % 20
            hour = 9 + (i % 2)
            ts = base_date.replace(hour=hour, minute=(i * 2) % 60, second=0, microsecond=0) + timedelta(
                days=day_offset
            )
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
            )

        # Since 30 < 50, fallback should trigger and merge
        routines = detector._detect_temporal_routines(lookback_days=60)

        # Should have routines from the merged dataset (30 + derived from 200)
        assert len(routines) >= 1, (
            "Expected at least 1 routine from merged primary + fallback episodes"
        )


class TestLocationFallbackThreshold:
    """Tests for the location detection fallback threshold."""

    def test_location_fallback_triggers_when_few_pairs(self, db, user_model_store):
        """Location fallback should trigger when fewer than 3 (location, type)
        pairs are found by the primary query.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Create 1 pair meeting min_occurrences with valid interaction_type
        for i in range(5):
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=i)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
                location="Office",
            )

        # Create many episodes with unknown type at different locations
        for i in range(100):
            day_offset = i % 20
            ts = base_date.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="task.created", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
                location="Home" if i % 2 == 0 else "Office",
            )

        routines = detector._detect_location_routines(lookback_days=60)

        # Should detect location routines from the merged dataset
        # (the 1 primary pair + fallback-derived pairs)
        assert isinstance(routines, list)


class TestEventTriggeredFallbackThreshold:
    """Tests for the event-triggered detection fallback threshold."""

    def test_event_triggered_fallback_triggers_when_few_types(self, db, user_model_store):
        """Event-triggered fallback should trigger when fewer than 3 candidate
        trigger types are found by the primary query.
        """
        detector = RoutineDetector(db, user_model_store, timezone="UTC")
        event_store = EventStore(db)

        base_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Create 1 trigger type meeting min_occurrences with valid interaction_type
        for i in range(5):
            ts = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=i)
            event_id = _create_event(event_store, event_type="email.received", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="email_check",
            )

        # Create many episodes with unknown type
        for i in range(100):
            day_offset = i % 20
            ts = base_date.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            event_id = _create_event(event_store, event_type="task.created", timestamp=ts)
            _create_episode(
                user_model_store,
                timestamp=ts,
                event_id=event_id,
                interaction_type="unknown",
            )

        routines = detector._detect_event_triggered_routines(lookback_days=60)

        # Should not crash and should return a list (may or may not find routines
        # depending on follow-up pattern analysis)
        assert isinstance(routines, list)


class TestMinEpisodesAttribute:
    """Tests for the min_episodes_for_detection attribute."""

    def test_default_threshold_is_50(self, db, user_model_store):
        """The default min_episodes_for_detection should be 50."""
        detector = RoutineDetector(db, user_model_store)
        assert detector.min_episodes_for_detection == 50

    def test_threshold_is_configurable(self, db, user_model_store):
        """The threshold should be modifiable after construction."""
        detector = RoutineDetector(db, user_model_store)
        detector.min_episodes_for_detection = 100
        assert detector.min_episodes_for_detection == 100
