"""Tests for DatabaseManager.backup_database() — automatic SQLite backup."""

import sqlite3
import time
from pathlib import Path

from storage.manager import DatabaseManager


class TestBackupDatabase:
    """Verify that backup_database() creates, prunes, and checkpoints correctly."""

    def test_backup_creates_file(self, db, tmp_data_dir):
        """A successful backup should produce a .db file in the backups/ directory."""
        result = db.backup_database("user_model")

        assert result is not None
        backup_path = Path(result)
        assert backup_path.exists()
        assert backup_path.parent == Path(tmp_data_dir) / "backups"
        assert backup_path.name.startswith("user_model_")
        assert backup_path.suffix == ".db"

    def test_backup_returns_path(self, db):
        """Return value should be the absolute path string of the backup file."""
        result = db.backup_database("user_model")

        assert result is not None
        assert isinstance(result, str)
        assert Path(result).is_absolute()

    def test_backup_creates_directory(self, db, tmp_data_dir):
        """The backups/ subdirectory should be created automatically."""
        backup_dir = Path(tmp_data_dir) / "backups"
        assert not backup_dir.exists()

        db.backup_database("user_model")

        assert backup_dir.is_dir()

    def test_backup_prunes_old_backups(self, db, tmp_data_dir):
        """Only the 3 most recent backups should be kept after pruning."""
        backup_dir = Path(tmp_data_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create 4 pre-existing backups with distinct timestamps.
        for i in range(4):
            fake = backup_dir / f"user_model_2026010{i}T000000.db"
            fake.write_text("fake")

        # The 5th backup is the real one from backup_database().
        db.backup_database("user_model")

        remaining = sorted(backup_dir.glob("user_model_*.db"))
        assert len(remaining) == 3, f"Expected 3 backups, got {len(remaining)}: {remaining}"

    def test_backup_handles_missing_db(self, tmp_data_dir):
        """Calling backup for a db that has no file on disk should return None."""
        manager = DatabaseManager(data_dir=tmp_data_dir)
        # Don't initialize — the .db file doesn't exist yet.

        result = manager.backup_database("user_model")

        assert result is None

    def test_backup_handles_unknown_db_name(self, db):
        """An unrecognised db_name should return None without raising."""
        result = db.backup_database("nonexistent_db")

        assert result is None

    def test_backup_checkpoints_wal(self, db, tmp_data_dir):
        """Data written before the backup should be queryable from the backup file."""
        # Write a row into user_model.db (episodes table exists after initialize_all).
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO episodes (id, event_id, interaction_type, timestamp, content_summary, content_full) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("ep-backup-test", "evt-1", "test", "2026-01-01T00:00:00Z", "summary", "full content"),
            )

        backup_path = db.backup_database("user_model")
        assert backup_path is not None

        # Open the backup independently and verify the row exists.
        conn2 = sqlite3.connect(backup_path)
        row = conn2.execute("SELECT content_full FROM episodes WHERE id = ?", ("ep-backup-test",)).fetchone()
        conn2.close()

        assert row is not None
        assert row[0] == "full content"

    def test_backup_multiple_databases(self, db):
        """backup_database() should work for any of the 5 databases, not just user_model."""
        for db_name in ("events", "entities", "state", "user_model", "preferences"):
            result = db.backup_database(db_name)
            assert result is not None, f"Backup failed for {db_name}"
            assert Path(result).exists()
