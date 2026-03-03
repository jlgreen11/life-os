"""
Tests for LifeOS._fresh_start_user_model_db() — the last-resort recovery
path that archives an unrecoverably corrupt user_model.db and creates a
completely fresh one.

Verifies that:
- A corrupt DB is archived with a timestamped .unrecoverable suffix
- WAL/SHM sidecar files are cleaned up before the fresh start
- A missing DB file is handled gracefully (just creates a new one)
- The freshly created DB passes all 7 integrity probe queries
- The _db_health_loop calls fresh start after the rebuild cap is exceeded
"""

import asyncio
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.manager import DatabaseManager


@pytest.fixture()
def lifeos_instance(db):
    """Create a minimal LifeOS instance with injected test DB.

    Uses dependency injection to provide a real DatabaseManager backed
    by temporary SQLite databases, avoiding any NATS or Ollama dependency.
    Includes mocked notification_manager and event_bus for notification tests.
    """
    from main import LifeOS

    instance = LifeOS(
        config={"data_dir": "./data", "ai": {}, "connectors": {}},
        db=db,
    )
    instance.shutdown_event = asyncio.Event()
    instance.notification_manager = AsyncMock()
    instance.event_bus = AsyncMock()
    instance.event_bus.is_connected = True
    return instance


def _corrupt_database(db_path: Path, offset: int = 4096, num_bytes: int = 1024) -> None:
    """Write zero bytes into the middle of a database file to simulate corruption.

    Seeks past the SQLite header to corrupt B-tree or overflow pages.
    """
    with open(db_path, "r+b") as f:
        f.seek(offset)
        f.write(bytes(num_bytes))


class TestFreshStartCreatesCleanDB:
    """Verify _fresh_start_user_model_db() creates a working DB from a corrupt one."""

    def test_fresh_start_creates_clean_db(self, lifeos_instance):
        """Create a corrupt user_model.db, call fresh start, verify it works."""
        db_path = Path(lifeos_instance.db._databases["user_model"])

        # Insert data so the DB has pages to corrupt
        with lifeos_instance.db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        # Corrupt the database
        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        # Fresh start should succeed
        result = lifeos_instance._fresh_start_user_model_db()
        assert result is True

        # The new DB should be fully functional
        with lifeos_instance.db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ep-fresh", "2026-01-01T00:00:00Z", "evt-1", "email", "Test after fresh start"),
            )
            row = conn.execute("SELECT id FROM episodes WHERE id = ?", ("ep-fresh",)).fetchone()
            assert row is not None
            assert row["id"] == "ep-fresh"


class TestFreshStartArchivesBehavior:
    """Verify the corrupt DB is archived with the correct naming convention."""

    def test_fresh_start_archives_with_timestamp(self, lifeos_instance):
        """Verify the archived file has .unrecoverable.{timestamp} suffix."""
        db_path = Path(lifeos_instance.db._databases["user_model"])

        # Insert data and corrupt
        with lifeos_instance.db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)

        lifeos_instance._fresh_start_user_model_db()

        # Check that an archive file with the correct pattern exists
        data_dir = db_path.parent
        archives = list(data_dir.glob("user_model.db.unrecoverable.*"))
        assert len(archives) >= 1, f"Expected archive file but found: {list(data_dir.iterdir())}"

        # Verify the timestamp format (YYYYMMDD_HHMMSS)
        archive_name = archives[0].name
        assert re.search(r"\.unrecoverable\.\d{8}_\d{6}$", archive_name), (
            f"Archive name should end with .unrecoverable.YYYYMMDD_HHMMSS, got: {archive_name}"
        )


class TestFreshStartCleansWalShm:
    """Verify WAL and SHM sidecar files are removed during fresh start."""

    def test_fresh_start_cleans_wal_shm(self, lifeos_instance):
        """Create WAL/SHM files alongside DB, verify they are cleaned up."""
        db_path = Path(lifeos_instance.db._databases["user_model"])
        wal_path = db_path.with_name(db_path.name + "-wal")
        shm_path = db_path.with_name(db_path.name + "-shm")

        # Insert data to populate DB
        with lifeos_instance.db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                ("test", json.dumps({"data": "x" * 500})),
            )

        # Ensure WAL and SHM exist (create if they don't)
        if not wal_path.exists():
            wal_path.write_bytes(b"fake WAL data")
        if not shm_path.exists():
            shm_path.write_bytes(b"fake SHM data")

        assert wal_path.exists(), "WAL should exist before fresh start"
        assert shm_path.exists(), "SHM should exist before fresh start"

        _corrupt_database(db_path, offset=4096, num_bytes=1024)

        result = lifeos_instance._fresh_start_user_model_db()
        assert result is True

        # WAL/SHM should have been removed
        assert not wal_path.exists(), "WAL should have been removed during fresh start"
        assert not shm_path.exists(), "SHM should have been removed during fresh start"


class TestFreshStartOnMissingDB:
    """Verify fresh start handles a missing DB file gracefully."""

    def test_fresh_start_on_missing_db(self, lifeos_instance):
        """Call when no DB file exists, verify it creates one."""
        db_path = Path(lifeos_instance.db._databases["user_model"])

        # Delete the database file
        db_path.unlink(missing_ok=True)
        # Also delete WAL/SHM if present
        for suffix in ["-wal", "-shm"]:
            sidecar = db_path.with_name(db_path.name + suffix)
            sidecar.unlink(missing_ok=True)

        assert not db_path.exists()

        result = lifeos_instance._fresh_start_user_model_db()
        assert result is True

        # The new DB should be functional
        with lifeos_instance.db.get_connection("user_model") as conn:
            row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            assert row[0] == 0  # Empty but functional


class TestVerifyIntegrityAfterFreshStart:
    """Verify all 7 probe queries pass on the fresh DB."""

    def test_verify_integrity_after_fresh_start(self, lifeos_instance):
        """After fresh start, all 7 probe queries should succeed."""
        db_path = Path(lifeos_instance.db._databases["user_model"])

        # Corrupt and fresh-start
        with lifeos_instance.db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)

        lifeos_instance._fresh_start_user_model_db()

        # Run the same 7 probe queries used by _verify_user_model_integrity
        with lifeos_instance.db.get_connection("user_model") as conn:
            conn.execute("SELECT content_full FROM episodes LIMIT 1").fetchone()
            conn.execute("SELECT SUM(LENGTH(data)) FROM signal_profiles").fetchone()
            conn.execute(
                "SELECT SUM(LENGTH(value)) + SUM(LENGTH(source_episodes)) FROM semantic_facts"
            ).fetchone()
            conn.execute(
                "SELECT SUM(LENGTH(steps)) + SUM(LENGTH(variations)) FROM routines"
            ).fetchone()
            conn.execute("SELECT SUM(LENGTH(contributing_signals)) FROM mood_history").fetchone()
            conn.execute("SELECT SUM(LENGTH(supporting_signals)) FROM predictions").fetchone()
            conn.execute("SELECT SUM(LENGTH(evidence)) FROM insights").fetchone()

        # Also verify via the method itself
        assert lifeos_instance._verify_user_model_integrity() is True


class TestHealthLoopFreshStartIntegration:
    """Verify _db_health_loop calls fresh start after rebuild cap is exceeded."""

    async def test_fourth_corruption_triggers_fresh_start(self, lifeos_instance):
        """On the 4th corruption, fresh start should be called instead of giving up."""
        lifeos_instance._runtime_db_rebuilds = 3

        original_get_connection = lifeos_instance.db.get_connection

        @contextmanager
        def corrupted_connection(db_name):
            if db_name == "user_model":
                mock_conn = MagicMock()
                mock_conn.execute.side_effect = sqlite3.DatabaseError(
                    "database disk image is malformed"
                )
                yield mock_conn
            else:
                with original_get_connection(db_name) as conn:
                    yield conn

        original_sleep = asyncio.sleep

        async def sleep_then_shutdown(seconds):
            lifeos_instance.shutdown_event.set()
            await original_sleep(0)

        with (
            patch.object(lifeos_instance.db, "get_connection", side_effect=corrupted_connection),
            patch.object(
                lifeos_instance, "_fresh_start_user_model_db", return_value=True
            ) as mock_fresh_start,
            patch.object(
                lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
            ) as mock_backfill,
            patch("main.ws_manager"),
            patch("asyncio.sleep", side_effect=sleep_then_shutdown),
        ):
            await lifeos_instance._db_health_loop()

        # Fresh start should have been called
        mock_fresh_start.assert_called_once()
        # Backfills should run after successful fresh start
        mock_backfill.assert_called_once()
        # Counter should be reset to 0 after success
        assert lifeos_instance._runtime_db_rebuilds == 0

        # Verify "reset and recovering" notification was created
        lifeos_instance.notification_manager.create_notification.assert_called()
        call_kwargs = lifeos_instance.notification_manager.create_notification.call_args[1]
        assert "reset" in call_kwargs["title"].lower()
        assert call_kwargs["priority"] == "critical"

    async def test_fresh_start_failure_sends_manual_intervention_notification(self, lifeos_instance):
        """When fresh start fails, a critical notification for manual intervention is sent."""
        lifeos_instance._runtime_db_rebuilds = 3

        original_get_connection = lifeos_instance.db.get_connection

        @contextmanager
        def corrupted_connection(db_name):
            if db_name == "user_model":
                mock_conn = MagicMock()
                mock_conn.execute.side_effect = sqlite3.DatabaseError(
                    "database disk image is malformed"
                )
                yield mock_conn
            else:
                with original_get_connection(db_name) as conn:
                    yield conn

        original_sleep = asyncio.sleep

        async def sleep_then_shutdown(seconds):
            lifeos_instance.shutdown_event.set()
            await original_sleep(0)

        with (
            patch.object(lifeos_instance.db, "get_connection", side_effect=corrupted_connection),
            patch.object(
                lifeos_instance, "_fresh_start_user_model_db", return_value=False
            ) as mock_fresh_start,
            patch.object(
                lifeos_instance, "_verify_and_retry_backfills", new_callable=AsyncMock
            ) as mock_backfill,
            patch("main.ws_manager"),
            patch("asyncio.sleep", side_effect=sleep_then_shutdown),
        ):
            await lifeos_instance._db_health_loop()

        mock_fresh_start.assert_called_once()
        # Backfills should NOT run after failed fresh start
        mock_backfill.assert_not_called()

        # Verify manual intervention notification was sent
        lifeos_instance.notification_manager.create_notification.assert_called()
        call_kwargs = lifeos_instance.notification_manager.create_notification.call_args[1]
        assert "manual intervention" in call_kwargs["title"].lower()
        assert call_kwargs["priority"] == "critical"

    async def test_health_loop_still_probes_after_cap_exceeded(self, lifeos_instance):
        """After exceeding 3 rebuilds, the loop should still probe (not bail out early)."""
        lifeos_instance._runtime_db_rebuilds = 4

        probe_called = False
        original_get_connection = lifeos_instance.db.get_connection

        @contextmanager
        def tracking_connection(db_name):
            nonlocal probe_called
            if db_name == "user_model":
                probe_called = True
            with original_get_connection(db_name) as conn:
                yield conn

        original_sleep = asyncio.sleep

        async def sleep_then_shutdown(seconds):
            lifeos_instance.shutdown_event.set()
            await original_sleep(0)

        with (
            patch.object(lifeos_instance.db, "get_connection", side_effect=tracking_connection),
            patch("asyncio.sleep", side_effect=sleep_then_shutdown),
        ):
            await lifeos_instance._db_health_loop()

        # The probe should still run (the early bail-out was removed)
        assert probe_called is True
