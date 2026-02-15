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
    class MockEventBus:
        """Minimal mock that satisfies UserModelStore telemetry calls."""
        def __init__(self):
            self.is_connected = True

        async def publish(self, subject: str, data: dict, source: str = None, priority: str = None):
            """No-op publish for tests. Accepts optional priority parameter."""
            pass

        async def subscribe_all(self, handler):
            """No-op subscribe for tests."""
            pass

    return MockEventBus()


@pytest.fixture()
def user_model_store(db, event_bus):
    """A UserModelStore wired to the temporary DatabaseManager and mock event bus."""
    return UserModelStore(db, event_bus=event_bus)


@pytest.fixture()
def notification_manager(db, event_bus):
    """A NotificationManager wired to the temporary DatabaseManager and mock event bus."""
    from services.notification_manager.manager import NotificationManager
    return NotificationManager(db, event_bus, config={})
