"""
Tests for automatic database corruption detection and recovery.

Verifies that DatabaseManager can detect a corrupted user_model.db at
startup, back it up, and recreate a functional database from scratch.
"""

from pathlib import Path

from storage.manager import DatabaseManager


class TestCorruptedUserModelDbRecovery:
    """Verify that a corrupted user_model.db is automatically recovered."""

    def test_corrupted_user_model_db_is_recovered(self, tmp_path):
        """A corrupted user_model.db should be backed up and recreated.

        Writes garbage bytes to user_model.db, then calls initialize_all().
        After recovery the database should be fully functional and a
        ``.corrupt.`` backup file should exist in the data directory.
        """
        data_dir = str(tmp_path)

        # Write garbage bytes to create a corrupt database file.
        corrupt_db = tmp_path / "user_model.db"
        corrupt_db.write_bytes(b"THIS IS NOT A VALID SQLITE DATABASE FILE " * 100)

        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # The recovered database should be fully functional.
        with manager.get_connection("user_model") as conn:
            # Insert a test episode row to prove the schema is intact.
            conn.execute(
                "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test-1", "2026-01-01T00:00:00Z", "evt-1", "test", "hello"),
            )
            row = conn.execute("SELECT id FROM episodes WHERE id = ?", ("test-1",)).fetchone()
            assert row is not None
            assert row["id"] == "test-1"

        # A .corrupt. backup file should have been created.
        backup_files = list(tmp_path.glob("user_model.db.corrupt.*"))
        assert len(backup_files) >= 1, "Expected at least one .corrupt backup file"

    def test_corrupted_wal_and_shm_are_also_backed_up(self, tmp_path):
        """WAL and SHM sidecar files should be renamed alongside the main db."""
        data_dir = str(tmp_path)

        # Create a corrupt db file plus fake WAL/SHM sidecars.
        (tmp_path / "user_model.db").write_bytes(b"CORRUPT" * 50)
        (tmp_path / "user_model.db-wal").write_bytes(b"wal data")
        (tmp_path / "user_model.db-shm").write_bytes(b"shm data")

        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # All three sidecar backups should exist.
        assert len(list(tmp_path.glob("user_model.db.corrupt.*"))) >= 1
        assert len(list(tmp_path.glob("user_model.db-wal.corrupt.*"))) >= 1
        assert len(list(tmp_path.glob("user_model.db-shm.corrupt.*"))) >= 1

        # The recovered main db should be functional.
        with manager.get_connection("user_model") as conn:
            tables = [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            assert "episodes" in tables

    def test_healthy_db_not_touched(self, tmp_path):
        """A healthy user_model.db should NOT be backed up or recreated."""
        data_dir = str(tmp_path)

        # First initialisation creates a healthy database.
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # Second initialisation should leave it untouched.
        manager2 = DatabaseManager(data_dir=data_dir)
        manager2.initialize_all()

        # No backup files should exist.
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 0, f"Unexpected backup files: {backup_files}"

    def test_missing_db_created_normally(self, tmp_path):
        """initialize_all() on an empty data dir should create all dbs cleanly."""
        data_dir = str(tmp_path)

        # No pre-existing files — everything is created from scratch.
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # user_model.db should exist and be functional.
        db_file = tmp_path / "user_model.db"
        assert db_file.exists()

        with manager.get_connection("user_model") as conn:
            tables = [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            assert "episodes" in tables
            assert "semantic_facts" in tables
            assert "routines" in tables
            assert "signal_profiles" in tables

        # No backup files should exist.
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 0

    def test_recovery_only_applies_to_user_model(self, tmp_path):
        """_check_and_recover_db is only called for user_model, not events or entities."""
        data_dir = str(tmp_path)

        # Create a healthy events.db first, then corrupt user_model.db.
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # Verify events.db is healthy after normal init.
        with manager.get_connection("events") as conn:
            tables = [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            assert "events" in tables

        # Now corrupt user_model.db and re-initialise.
        corrupt_db = tmp_path / "user_model.db"
        corrupt_db.write_bytes(b"CORRUPT GARBAGE DATA" * 100)

        manager2 = DatabaseManager(data_dir=data_dir)
        manager2.initialize_all()

        # user_model.db should be recovered.
        backup_files = list(tmp_path.glob("user_model.db.corrupt.*"))
        assert len(backup_files) >= 1

        # events.db should NOT have been touched — no backup for it.
        events_backups = list(tmp_path.glob("events.db.corrupt.*"))
        assert len(events_backups) == 0
