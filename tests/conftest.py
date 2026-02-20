"""
Life OS — Shared test fixtures.

Provides a temporary data directory and real DatabaseManager / store
instances backed by throwaway SQLite databases so that every test
starts with a clean, fully-initialized schema.
"""

import tempfile

import pytest

from storage.manager import DatabaseManager
from storage.event_store import EventStore
from storage.user_model_store import UserModelStore


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Return a temporary directory path string for database files."""
    return str(tmp_path)


@pytest.fixture()
def db(tmp_data_dir):
    """A fully-initialized DatabaseManager using a temporary data directory.

    All five SQLite databases (events, entities, state, user_model,
    preferences) are created with their complete schemas so that
    downstream stores and services can operate without mocks.
    """
    manager = DatabaseManager(data_dir=tmp_data_dir)
    manager.initialize_all()
    return manager


@pytest.fixture()
def event_store(db):
    """An EventStore wired to the temporary DatabaseManager."""
    return EventStore(db)


@pytest.fixture()
def event_bus():
    """A mock EventBus for testing (no real NATS connection required)."""
    from unittest.mock import AsyncMock

    class PublishWrapper:
        """Wrapper that makes publish() both callable and have AsyncMock attributes."""
        def __init__(self, bus):
            self._bus = bus
            self._mock = AsyncMock()

        async def __call__(self, event_type: str, payload: dict, **kwargs):
            """Forward to the real publish implementation."""
            # Call mock for assertions
            await self._mock(event_type, payload, **kwargs)
            # Call real implementation
            return await self._bus._publish_impl(event_type, payload, **kwargs)

        # Proxy all AsyncMock attributes
        @property
        def called(self):
            return self._mock.called

        @property
        def call_count(self):
            return self._mock.call_count

        @property
        def call_args(self):
            return self._mock.call_args

        @property
        def call_args_list(self):
            return self._mock.call_args_list

        def assert_called(self):
            return self._mock.assert_called()

        def assert_called_once(self):
            return self._mock.assert_called_once()

        def assert_not_called(self):
            return self._mock.assert_not_called()

        def reset_mock(self):
            return self._mock.reset_mock()

    class MockEventBus:
        """Functional mock EventBus with subscriber routing."""
        def __init__(self):
            self.is_connected = True
            self._subscribers = {}
            self._published_events = []
            self.publish = PublishWrapper(self)
            self.subscribe = AsyncMock(side_effect=self._subscribe_impl)
            self.subscribe_all = AsyncMock()

        async def _publish_impl(self, event_type: str, payload: dict, **kwargs):
            """Actual publish implementation."""
            # Store event for verification
            event = {
                "type": event_type,
                "payload": payload,
                "source": kwargs.get("source", ""),
                "priority": kwargs.get("priority", "normal"),
                "metadata": kwargs.get("metadata", {}),
            }
            self._published_events.append(event)

            # Call registered subscribers
            for pattern, callback in self._subscribers.items():
                # Simple pattern matching (exact or wildcard)
                if self._pattern_matches(pattern, event_type):
                    try:
                        await callback(event)
                    except Exception:
                        pass  # Ignore subscriber errors in tests

            return "mock-event-id"

        async def _subscribe_impl(self, pattern: str, callback, **kwargs):
            """Subscribe implementation that stores callback for publish to invoke."""
            self._subscribers[pattern] = callback

        def _pattern_matches(self, pattern: str, event_type: str) -> bool:
            """Check if pattern matches event_type."""
            if pattern == "*":
                return True
            if pattern == event_type:
                return True
            if pattern.endswith(".*") and event_type.startswith(pattern[:-2] + "."):
                return True
            return False

    return MockEventBus()


@pytest.fixture()
def user_model_store(db, event_bus):
    """A UserModelStore wired to the temporary DatabaseManager and mock event bus."""
    return UserModelStore(db, event_bus=event_bus)


@pytest.fixture()
def prediction_engine(db, user_model_store):
    """A PredictionEngine wired to the temporary DatabaseManager and UserModelStore."""
    from services.prediction_engine.engine import PredictionEngine
    return PredictionEngine(db, user_model_store, timezone="UTC")


@pytest.fixture()
def notification_manager(db, event_bus):
    """A NotificationManager wired to the temporary DatabaseManager and mock event bus."""
    from services.notification_manager.manager import NotificationManager
    return NotificationManager(db, event_bus, config={}, timezone="UTC")
