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
def user_model_store(db):
    """A UserModelStore wired to the temporary DatabaseManager (no event bus)."""
    return UserModelStore(db)
