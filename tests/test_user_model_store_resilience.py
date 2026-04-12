"""
Tests for UserModelStore resilience when user_model.db is corrupt.

Verifies that the four critical methods (store_episode, update_signal_profile,
get_signal_profile, store_mood) fail-open when the database raises errors.
Telemetry is only emitted on successful writes to prevent phantom events.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from storage.user_model_store import UserModelStore


@pytest.fixture()
def corrupt_store(db, event_store):
    """A UserModelStore whose get_connection raises DatabaseError for user_model.

    Uses a real DatabaseManager but patches get_connection to simulate
    a corrupt user_model.db. The event_store fallback is wired so that
    telemetry events can be verified even when the DB is broken.
    """
    store = UserModelStore(db, event_bus=None, event_store=event_store)

    original_get_connection = db.get_connection

    from contextlib import contextmanager

    @contextmanager
    def broken_connection(db_name):
        """Raise DatabaseError for user_model, pass through for others."""
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        with original_get_connection(db_name) as conn:
            yield conn

    db.get_connection = broken_connection
    return store


def _make_episode():
    """Create a minimal valid episode dict for testing."""
    return {
        "id": "ep-test-001",
        "timestamp": "2026-01-01T00:00:00Z",
        "event_id": "evt-test-001",
        "interaction_type": "test",
        "content_summary": "test episode",
    }


class TestStoreEpisodeResilience:
    """store_episode should not raise when user_model.db is corrupt."""

    def test_does_not_raise_on_db_error(self, corrupt_store):
        """store_episode silently handles DatabaseError without propagating."""
        episode = _make_episode()
        # Should not raise
        corrupt_store.store_episode(episode)

    def test_logs_warning_on_db_error(self, corrupt_store, caplog):
        """store_episode logs a warning with the error details."""
        import logging

        episode = _make_episode()
        with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
            corrupt_store.store_episode(episode)

        assert any("store_episode" in record.message for record in caplog.records)
        assert any("malformed" in record.message for record in caplog.records)

    def test_no_telemetry_on_db_error(self, corrupt_store, event_store):
        """Telemetry must NOT fire when the DB write fails (phantom telemetry fix).

        Previously, ``_emit_telemetry()`` was called OUTSIDE the try/except block
        in ``store_episode()``, so it fired even when the INSERT failed — producing
        1,865 phantom ``usermodel.episode.stored`` events while 0 actual episode rows
        existed in user_model.db.

        After the fix (mirroring ``update_signal_profile`` and
        ``update_semantic_fact``), telemetry is only emitted after a successful
        write.  This test guards against a regression to the old phantom-telemetry
        behavior.
        """
        episode = _make_episode()
        corrupt_store.store_episode(episode)

        # Telemetry falls back to event_store when event_bus is None.
        # Since the DB write failed, NO telemetry event should be written.
        with event_store.db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE type = 'usermodel.episode.stored'"
            ).fetchall()
        assert len(rows) == 0, (
            "Telemetry was emitted despite DB write failure — phantom telemetry regression"
        )


class TestUpdateSignalProfileResilience:
    """update_signal_profile should not raise when user_model.db is corrupt."""

    def test_does_not_raise_on_db_error(self, corrupt_store):
        """update_signal_profile silently handles DatabaseError."""
        corrupt_store.update_signal_profile("linguistic", {"formality": 0.7})

    def test_logs_warning_on_db_error(self, corrupt_store, caplog):
        """update_signal_profile logs a warning with the error details."""
        import logging

        with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
            corrupt_store.update_signal_profile("cadence", {"avg_interval": 300})

        assert any("update_signal_profile" in record.message for record in caplog.records)
        assert any("malformed" in record.message for record in caplog.records)

    def test_no_telemetry_on_db_error(self, corrupt_store, event_store):
        """Telemetry must NOT fire when the DB write fails (phantom telemetry fix).

        Previously, telemetry was emitted outside the try/except block so it
        fired even on failed writes — inflating event counts with phantom
        events.  After the fix, telemetry only fires on successful writes.
        """
        corrupt_store.update_signal_profile("mood", {"energy": 0.6})

        with event_store.db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE type = 'usermodel.signal_profile.updated'"
            ).fetchall()
        assert len(rows) == 0, (
            "Telemetry was emitted despite DB write failure — phantom telemetry bug"
        )


class TestGetSignalProfileResilience:
    """get_signal_profile should return None when user_model.db is corrupt."""

    def test_returns_none_on_db_error(self, corrupt_store):
        """get_signal_profile returns None (same as 'not found') on error."""
        result = corrupt_store.get_signal_profile("linguistic")
        assert result is None

    def test_logs_warning_on_db_error(self, corrupt_store, caplog):
        """get_signal_profile logs a warning with the error details."""
        import logging

        with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
            corrupt_store.get_signal_profile("cadence")

        assert any("get_signal_profile" in record.message for record in caplog.records)
        assert any("malformed" in record.message for record in caplog.records)


class TestStoreMoodResilience:
    """store_mood should not raise when user_model.db is corrupt."""

    def test_does_not_raise_on_db_error(self, corrupt_store):
        """store_mood silently handles DatabaseError."""
        corrupt_store.store_mood({"energy_level": 0.8, "stress_level": 0.2})

    def test_logs_warning_on_db_error(self, corrupt_store, caplog):
        """store_mood logs a warning with the error details."""
        import logging

        with caplog.at_level(logging.WARNING, logger="storage.user_model_store"):
            corrupt_store.store_mood({"energy_level": 0.5})

        assert any("store_mood" in record.message for record in caplog.records)
        assert any("malformed" in record.message for record in caplog.records)

    def test_telemetry_still_emitted_on_db_error(self, corrupt_store, event_store):
        """Telemetry event is emitted even when DB write fails."""
        corrupt_store.store_mood({"energy_level": 0.5, "trend": "declining"})

        with event_store.db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE type = 'usermodel.mood.recorded'"
            ).fetchall()
        assert len(rows) >= 1


class TestNormalOperationUnchanged:
    """Verify that error handling does not affect normal (healthy DB) operation."""

    def test_store_episode_works_normally(self, user_model_store, db):
        """store_episode still stores data when DB is healthy."""
        episode = _make_episode()
        user_model_store.store_episode(episode)

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM episodes WHERE id = ?", ("ep-test-001",)
            ).fetchone()
        assert row is not None

    def test_update_and_get_signal_profile_works_normally(self, user_model_store):
        """update/get signal profile round-trips data when DB is healthy."""
        user_model_store.update_signal_profile("linguistic", {"formality": 0.9})
        result = user_model_store.get_signal_profile("linguistic")
        assert result is not None
        assert result["data"]["formality"] == 0.9

    def test_store_mood_works_normally(self, user_model_store, db):
        """store_mood still stores data when DB is healthy."""
        user_model_store.store_mood({"energy_level": 0.7, "stress_level": 0.4})

        with db.get_connection("user_model") as conn:
            row = conn.execute("SELECT * FROM mood_history LIMIT 1").fetchone()
        assert row is not None
        assert row["energy_level"] == 0.7
