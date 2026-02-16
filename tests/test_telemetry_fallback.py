"""Test telemetry fallback when event bus is unavailable.

CRITICAL BUG (iteration 143):
    UserModelStore telemetry had 100% failure rate when NATS wasn't running.
    This broke ALL observability: 340K+ predictions generated but 0 telemetry
    events published.

    Root cause: Telemetry required event bus is_connected=True. When NATS isn't
    running (local dev, tests, or connection failures), is_connected=False and
    all telemetry was silently skipped.

    Fix: Dual-path telemetry with automatic fallback to event store when event
    bus is unavailable. This test verifies the fallback path works correctly.
"""

import asyncio
from datetime import datetime, timezone

from models.core import ConfidenceGate
from models.user_model import Prediction
from storage.event_store import EventStore
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def test_telemetry_fallback_no_event_bus(db: DatabaseManager, event_store: EventStore):
    """Telemetry should fall back to event store when event bus is unavailable."""

    # Create UserModelStore with event_store but NO event bus
    ums = UserModelStore(db, event_bus=None, event_store=event_store)

    # Store a prediction (this will trigger telemetry)
    prediction = Prediction(
        prediction_type="test",
        description="Test prediction for telemetry fallback",
        confidence=0.75,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
    )

    ums.store_prediction(prediction.model_dump())

    # Verify the prediction was stored
    with db.get_connection("user_model") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_type = 'test'"
        ).fetchone()[0]
        assert count == 1, "Prediction should be stored"

    # Verify telemetry event was written to event store (fallback path)
    with db.get_connection("events") as conn:
        telemetry_events = conn.execute(
            """SELECT * FROM events
               WHERE type = 'usermodel.prediction.generated'
               AND json_extract(payload, '$.prediction_type') = 'test'"""
        ).fetchall()

        assert len(telemetry_events) == 1, "Telemetry event should be written via fallback"
        event = telemetry_events[0]
        assert event["source"] == "user_model_store"
        assert event["type"] == "usermodel.prediction.generated"


def test_telemetry_fallback_disconnected_event_bus(db: DatabaseManager, event_store: EventStore):
    """Telemetry should fall back to event store when event bus is disconnected."""

    # Create a mock event bus that reports as disconnected
    class DisconnectedEventBus:
        @property
        def is_connected(self):
            return False

    disconnected_bus = DisconnectedEventBus()
    ums = UserModelStore(db, event_bus=disconnected_bus, event_store=event_store)

    # Store a prediction
    prediction = Prediction(
        prediction_type="test_disconnected",
        description="Test with disconnected bus",
        confidence=0.6,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="48_hours",
    )

    ums.store_prediction(prediction.model_dump())

    # Verify telemetry event was written via fallback
    with db.get_connection("events") as conn:
        telemetry_events = conn.execute(
            """SELECT * FROM events
               WHERE type = 'usermodel.prediction.generated'
               AND json_extract(payload, '$.prediction_type') = 'test_disconnected'"""
        ).fetchall()

        assert len(telemetry_events) == 1, "Telemetry should fall back when bus is disconnected"


async def test_telemetry_via_event_bus_when_connected(db: DatabaseManager, event_store: EventStore):
    """Telemetry should use event bus when available and connected."""

    # Create a mock event bus that reports as connected and tracks publishes
    class ConnectedEventBus:
        def __init__(self):
            self.published_events = []

        @property
        def is_connected(self):
            return True

        async def publish(self, event_type, payload, source):
            self.published_events.append((event_type, payload, source))

    connected_bus = ConnectedEventBus()
    ums = UserModelStore(db, event_bus=connected_bus, event_store=event_store)

    # Store a prediction in an async context (so event loop is available)
    prediction = Prediction(
        prediction_type="test_connected",
        description="Test with connected bus",
        confidence=0.8,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="6_hours",
    )

    ums.store_prediction(prediction.model_dump())

    # Give async task time to complete
    await asyncio.sleep(0.1)

    # Verify event was published via event bus (primary path)
    assert len(connected_bus.published_events) == 1, "Event should be published via bus"
    event_type, payload, source = connected_bus.published_events[0]
    assert event_type == "usermodel.prediction.generated"
    assert payload["prediction_type"] == "test_connected"
    assert source == "user_model_store"


def test_telemetry_fallback_all_paths_fail(db: DatabaseManager):
    """If both event bus and event store fail, telemetry should log warning but not crash."""

    # Create UserModelStore with NO event bus and NO event store
    ums = UserModelStore(db, event_bus=None, event_store=None)

    # This should not crash even though telemetry can't be published anywhere
    prediction = Prediction(
        prediction_type="test_no_telemetry",
        description="Test with no telemetry paths available",
        confidence=0.5,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="12_hours",
    )

    # Should complete without raising an exception
    ums.store_prediction(prediction.model_dump())

    # Verify prediction was still stored (telemetry failure doesn't break storage)
    with db.get_connection("user_model") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE prediction_type = 'test_no_telemetry'"
        ).fetchone()[0]
        assert count == 1, "Prediction storage should succeed even if telemetry fails"


def test_episode_telemetry_fallback(db: DatabaseManager, event_store: EventStore):
    """Episode storage telemetry should also use fallback path."""

    ums = UserModelStore(db, event_bus=None, event_store=event_store)

    # Store an episode
    episode = {
        "id": "test-episode-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-1",
        "interaction_type": "email",
        "content_summary": "Test episode",
        "contacts_involved": ["test@example.com"],
        "topics": ["testing"],
        "entities": [],
    }

    ums.store_episode(episode)

    # Verify telemetry event was written via fallback
    with db.get_connection("events") as conn:
        telemetry_events = conn.execute(
            """SELECT * FROM events
               WHERE type = 'usermodel.episode.stored'
               AND json_extract(payload, '$.episode_id') = 'test-episode-1'"""
        ).fetchall()

        assert len(telemetry_events) == 1, "Episode telemetry should use fallback"


def test_signal_profile_telemetry_fallback(db: DatabaseManager, event_store: EventStore):
    """Signal profile update telemetry should use fallback path."""

    ums = UserModelStore(db, event_bus=None, event_store=event_store)

    # Update a signal profile
    profile_data = {
        "samples_count": 5,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "metrics": {"test_metric": 0.75},
    }

    ums.update_signal_profile("test_profile", profile_data)

    # Verify telemetry event was written via fallback
    with db.get_connection("events") as conn:
        telemetry_events = conn.execute(
            """SELECT * FROM events
               WHERE type = 'usermodel.signal_profile.updated'
               AND json_extract(payload, '$.profile_type') = 'test_profile'"""
        ).fetchall()

        assert len(telemetry_events) == 1, "Signal profile telemetry should use fallback"
