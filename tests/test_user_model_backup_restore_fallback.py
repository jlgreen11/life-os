"""Tests for user_model.db backup-restore fallback during corruption recovery.

Verifies that _rebuild_user_model_db_if_corrupted falls back to
restore_from_backup when the dump/rebuild path fails, and that it
returns accurate boolean status indicating whether recovery succeeded.
"""

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from storage.manager import DatabaseManager


@pytest.fixture()
def healthy_db(tmp_path):
    """Create a fully initialized DatabaseManager with healthy user_model.db."""
    db = DatabaseManager(data_dir=str(tmp_path))
    db.initialize_all()
    return db


@pytest.fixture()
def life_os_with_db(healthy_db, tmp_path):
    """Create a minimal LifeOS-like object with the methods under test.

    Rather than instantiating the full LifeOS class (which requires NATS,
    Ollama, etc.), we import and bind just the methods we need to a simple
    namespace that has a `db` attribute.
    """
    from main import LifeOS

    # Build a minimal object with the required attributes
    obj = MagicMock(spec=LifeOS)
    obj.db = healthy_db
    obj.notification_manager = MagicMock()
    obj.notification_manager.create_notification = AsyncMock()
    obj.event_bus = None
    obj.shutdown_event = asyncio.Event()
    obj._runtime_db_rebuilds = 0
    obj._last_backup_time = None

    # Bind the real methods to our mock object
    obj._rebuild_user_model_db_if_corrupted = LifeOS._rebuild_user_model_db_if_corrupted.__get__(obj)
    obj._verify_user_model_integrity = LifeOS._verify_user_model_integrity.__get__(obj)
    obj._try_restore_user_model_from_backup = LifeOS._try_restore_user_model_from_backup.__get__(obj)

    return obj


def _corrupt_user_model_db(db: DatabaseManager):
    """Overwrite user_model.db with garbage bytes to simulate corruption."""
    db_path = db.data_dir / "user_model.db"
    # Remove WAL/SHM first so SQLite doesn't recover from them
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.parent / (db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    # Write garbage bytes — enough to be a valid file but not valid SQLite
    with open(db_path, "wb") as f:
        f.write(b"CORRUPT" * 1000)


def _create_backup_of_healthy_db(db: DatabaseManager) -> str:
    """Create a backup of user_model.db while it's still healthy.

    Returns the backup path.
    """
    backup_path = db.backup_database("user_model")
    assert backup_path is not None, "Failed to create backup"
    return backup_path


class TestRebuildReturnsBoolean:
    """Verify _rebuild_user_model_db_if_corrupted returns True/False."""

    async def test_returns_true_when_db_is_healthy(self, life_os_with_db):
        """A healthy DB should return True without making any changes."""
        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        assert result is True

    async def test_returns_false_when_corrupted_and_no_backups(self, life_os_with_db):
        """A corrupted DB with no backups should return False."""
        _corrupt_user_model_db(life_os_with_db.db)
        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        assert result is False


class TestBackupRestoreFallback:
    """Verify backup-restore fallback when rebuild fails."""

    async def test_falls_back_to_backup_when_connection_fails(self, life_os_with_db):
        """When the DB can't even be opened, should try backup restore."""
        db = life_os_with_db.db

        # Create a backup while healthy
        backup_path = _create_backup_of_healthy_db(db)
        assert Path(backup_path).exists()

        # Corrupt the DB so severely it can't be connected to
        _corrupt_user_model_db(db)

        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        assert result is True

        # Verify the DB is actually healthy after restore
        assert life_os_with_db._verify_user_model_integrity() is True

    async def test_falls_back_to_backup_when_dump_fails(self, life_os_with_db):
        """When data dump fails, should try backup restore."""
        db = life_os_with_db.db

        # Insert some data first so the DB is non-trivial
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT OR IGNORE INTO episodes (id, timestamp, event_id, content_summary) "
                "VALUES (?, ?, ?, ?)",
                ("ep1", "2026-01-01T00:00:00Z", "evt1", "test episode"),
            )

        # Create a backup while healthy
        _create_backup_of_healthy_db(db)

        # Corrupt the DB
        _corrupt_user_model_db(db)

        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        assert result is True

        # Verify the DB is healthy after restore
        assert life_os_with_db._verify_user_model_integrity() is True

    async def test_returns_false_when_no_backups_exist(self, life_os_with_db):
        """When corruption is total and no backups exist, returns False."""
        _corrupt_user_model_db(life_os_with_db.db)

        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        assert result is False

    async def test_tries_multiple_backups(self, life_os_with_db):
        """If the newest backup is also corrupt, tries the next one."""
        db = life_os_with_db.db

        # Create a good backup first
        good_backup_path = _create_backup_of_healthy_db(db)

        # Create another "backup" that is actually corrupt
        backup_dir = db.data_dir / "backups"
        # The newest backup filename sorts last alphabetically, so give it
        # a later timestamp
        bad_backup = backup_dir / "user_model_29990101T000000.db"
        with open(bad_backup, "wb") as f:
            f.write(b"CORRUPT" * 500)

        # Corrupt the main DB
        _corrupt_user_model_db(db)

        result = await life_os_with_db._rebuild_user_model_db_if_corrupted()
        # Should succeed by falling back to the good backup after the bad one fails
        assert result is True
        assert life_os_with_db._verify_user_model_integrity() is True


class TestVerifyUserModelIntegrity:
    """Verify the _verify_user_model_integrity helper."""

    def test_returns_true_for_healthy_db(self, life_os_with_db):
        """A healthy DB passes all 7 probe queries."""
        assert life_os_with_db._verify_user_model_integrity() is True

    def test_returns_false_for_corrupted_db(self, life_os_with_db):
        """A corrupted DB fails probe queries."""
        _corrupt_user_model_db(life_os_with_db.db)
        assert life_os_with_db._verify_user_model_integrity() is False


class TestTryRestoreUserModelFromBackup:
    """Verify the _try_restore_user_model_from_backup helper."""

    def test_returns_false_when_no_backups(self, life_os_with_db):
        """No backups available → returns False."""
        _corrupt_user_model_db(life_os_with_db.db)
        result = life_os_with_db._try_restore_user_model_from_backup("test")
        assert result is False

    def test_returns_true_with_valid_backup(self, life_os_with_db):
        """Valid backup available → restores and returns True."""
        db = life_os_with_db.db
        _create_backup_of_healthy_db(db)
        _corrupt_user_model_db(db)

        result = life_os_with_db._try_restore_user_model_from_backup("test")
        assert result is True
        # DB should be queryable again
        assert life_os_with_db._verify_user_model_integrity() is True

    def test_skips_corrupt_backup_tries_next(self, life_os_with_db):
        """When a backup file fails integrity check, tries the next one."""
        db = life_os_with_db.db

        # Create a good backup
        _create_backup_of_healthy_db(db)

        # Create a corrupt "backup" with a later timestamp (sorted first)
        backup_dir = db.data_dir / "backups"
        bad_backup = backup_dir / "user_model_29990101T000000.db"
        with open(bad_backup, "wb") as f:
            f.write(b"NOT_SQLITE" * 100)

        _corrupt_user_model_db(db)

        result = life_os_with_db._try_restore_user_model_from_backup("test")
        assert result is True
        assert life_os_with_db._verify_user_model_integrity() is True
