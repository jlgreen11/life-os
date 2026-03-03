"""
Test suite for DatabaseManager backup listing and restore-from-backup methods.

Tests cover:
    1. list_backups() returns empty list when no backups exist
    2. list_backups() returns backups sorted newest-first
    3. list_backups() includes all expected metadata keys
    4. restore_from_backup() restores a valid backup successfully
    5. restore_from_backup() rejects corrupt backup files
    6. restore_from_backup() rejects path traversal attempts
    7. restore_from_backup() archives the original database before overwriting
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


class TestListBackups:
    """Tests for DatabaseManager.list_backups()."""

    def test_list_backups_empty_when_no_backups(self):
        """list_backups() should return an empty list when the backup directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            result = db_manager.list_backups("user_model")

            assert result == []

    def test_list_backups_empty_when_dir_exists_but_no_matching_files(self):
        """list_backups() should return an empty list when backup dir exists but has no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create the backup dir with an unrelated file
            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()
            (backup_dir / "other_file.txt").write_text("not a backup")

            result = db_manager.list_backups("user_model")

            assert result == []

    def test_list_backups_returns_sorted_by_newest(self):
        """list_backups() should return backups sorted by timestamp, newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create fake backup files with different timestamps
            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()

            timestamps = ["20260301T100000", "20260303T120000", "20260302T080000"]
            for ts in timestamps:
                backup_file = backup_dir / f"user_model_{ts}.db"
                # Create a minimal valid SQLite database
                conn = sqlite3.connect(str(backup_file))
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.commit()
                conn.close()

            result = db_manager.list_backups("user_model")

            assert len(result) == 3
            # Newest first
            assert "2026-03-03" in result[0]["created_at"]
            assert "2026-03-02" in result[1]["created_at"]
            assert "2026-03-01" in result[2]["created_at"]

    def test_list_backups_includes_metadata(self):
        """Each backup dict should have all expected metadata keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()

            # Create a backup file
            backup_file = backup_dir / "user_model_20260303T120000.db"
            conn = sqlite3.connect(str(backup_file))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            result = db_manager.list_backups("user_model")

            assert len(result) == 1
            backup = result[0]

            # Verify all expected keys are present
            assert "path" in backup
            assert "filename" in backup
            assert "db_name" in backup
            assert "created_at" in backup
            assert "size_bytes" in backup
            assert "age_hours" in backup

            # Verify values
            assert backup["filename"] == "user_model_20260303T120000.db"
            assert backup["db_name"] == "user_model"
            assert backup["size_bytes"] > 0
            assert backup["age_hours"] >= 0
            assert "2026-03-03T12:00:00" in backup["created_at"]

    def test_list_backups_filters_by_db_name(self):
        """list_backups() should only return backups for the specified db_name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()

            # Create backups for different databases
            for name in ["user_model", "events", "state"]:
                backup_file = backup_dir / f"{name}_20260303T120000.db"
                conn = sqlite3.connect(str(backup_file))
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.commit()
                conn.close()

            result = db_manager.list_backups("user_model")
            assert len(result) == 1
            assert result[0]["db_name"] == "user_model"

    def test_list_backups_skips_unparseable_filenames(self):
        """list_backups() should skip files whose timestamps can't be parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()

            # Create a file with a valid timestamp and one with garbage
            good = backup_dir / "user_model_20260303T120000.db"
            bad = backup_dir / "user_model_not-a-timestamp.db"
            for f in [good, bad]:
                conn = sqlite3.connect(str(f))
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.commit()
                conn.close()

            result = db_manager.list_backups("user_model")
            assert len(result) == 1
            assert result[0]["filename"] == "user_model_20260303T120000.db"


class TestRestoreFromBackup:
    """Tests for DatabaseManager.restore_from_backup()."""

    def test_restore_from_valid_backup(self):
        """restore_from_backup() should restore data from a valid backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Insert data into user_model.db
            with db_manager.get_connection("user_model") as conn:
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                    ("test_fact", "preference", "coffee"),
                )

            # Create a backup
            backup_path = db_manager.backup_database("user_model")
            assert backup_path is not None

            # Now modify the original to simulate different state
            with db_manager.get_connection("user_model") as conn:
                conn.execute("DELETE FROM semantic_facts WHERE key = 'test_fact'")
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                    ("new_fact", "preference", "tea"),
                )

            # Verify pre-restore state
            with db_manager.get_connection("user_model") as conn:
                row = conn.execute("SELECT value FROM semantic_facts WHERE key = 'test_fact'").fetchone()
                assert row is None  # original data gone
                row = conn.execute("SELECT value FROM semantic_facts WHERE key = 'new_fact'").fetchone()
                assert row["value"] == "tea"

            # Restore from backup
            result = db_manager.restore_from_backup(backup_path, "user_model")
            assert result is True

            # Verify data is restored
            with db_manager.get_connection("user_model") as conn:
                row = conn.execute("SELECT value FROM semantic_facts WHERE key = 'test_fact'").fetchone()
                assert row is not None
                assert row["value"] == "coffee"
                # The "new" data should be gone since we restored the old backup
                row = conn.execute("SELECT value FROM semantic_facts WHERE key = 'new_fact'").fetchone()
                assert row is None

    def test_restore_rejects_corrupt_backup(self):
        """restore_from_backup() should return False for a corrupt backup file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create a corrupt "backup" file
            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()
            corrupt_backup = backup_dir / "user_model_20260303T120000.db"
            corrupt_backup.write_bytes(b"\xff\xfe\xfd\xfc" * 256)

            result = db_manager.restore_from_backup(str(corrupt_backup), "user_model")
            assert result is False

    def test_restore_rejects_path_traversal(self):
        """restore_from_backup() should reject paths outside data/backups/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create a valid SQLite file outside the backup directory
            outside_file = Path(tmpdir) / "evil.db"
            conn = sqlite3.connect(str(outside_file))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            result = db_manager.restore_from_backup(str(outside_file), "user_model")
            assert result is False

    def test_restore_rejects_path_traversal_with_dotdot(self):
        """restore_from_backup() should reject paths using .. to escape backups dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create a valid file and try to reference it via path traversal
            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()
            outside_file = Path(tmpdir) / "evil.db"
            conn = sqlite3.connect(str(outside_file))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            traversal_path = str(backup_dir / ".." / "evil.db")
            result = db_manager.restore_from_backup(traversal_path, "user_model")
            assert result is False

    def test_restore_archives_original(self):
        """restore_from_backup() should archive the current database as .pre_restore.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Insert identifiable data so we can verify the archive
            with db_manager.get_connection("user_model") as conn:
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                    ("original_data", "preference", "original"),
                )

            # Create a backup (this will capture the "original" data)
            backup_path = db_manager.backup_database("user_model")
            assert backup_path is not None

            # Modify the database to create a different state pre-restore
            with db_manager.get_connection("user_model") as conn:
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                    ("modified_data", "preference", "modified"),
                )

            # Restore from backup
            result = db_manager.restore_from_backup(backup_path, "user_model")
            assert result is True

            # Verify the archive file was created
            archive_path = Path(tmpdir) / "user_model.pre_restore.db"
            assert archive_path.exists()

            # Verify the archive contains the pre-restore data (including "modified_data")
            archive_conn = sqlite3.connect(str(archive_path))
            archive_conn.row_factory = sqlite3.Row
            row = archive_conn.execute(
                "SELECT value FROM semantic_facts WHERE key = 'modified_data'"
            ).fetchone()
            assert row is not None
            assert row["value"] == "modified"
            archive_conn.close()

    def test_restore_removes_stale_wal_shm(self):
        """restore_from_backup() should remove -wal and -shm files from the old database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create fake WAL/SHM files
            db_path = Path(tmpdir) / "user_model.db"
            wal_path = Path(tmpdir) / "user_model.db-wal"
            shm_path = Path(tmpdir) / "user_model.db-shm"
            wal_path.write_bytes(b"fake wal data")
            shm_path.write_bytes(b"fake shm data")

            # Create a valid backup
            backup_path = db_manager.backup_database("user_model")
            assert backup_path is not None

            # Restore — should clean up WAL/SHM
            result = db_manager.restore_from_backup(backup_path, "user_model")
            assert result is True

            assert not wal_path.exists()
            assert not shm_path.exists()

    def test_restore_nonexistent_backup_returns_false(self):
        """restore_from_backup() should return False when the backup file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            result = db_manager.restore_from_backup("/nonexistent/path/backup.db", "user_model")
            assert result is False

    def test_restore_unknown_db_name_returns_false(self):
        """restore_from_backup() should return False for an unknown database name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Create a valid backup file to pass the file check
            backup_dir = Path(tmpdir) / "backups"
            backup_dir.mkdir()
            backup_file = backup_dir / "fake_20260303T120000.db"
            conn = sqlite3.connect(str(backup_file))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            result = db_manager.restore_from_backup(str(backup_file), "nonexistent_db")
            assert result is False
