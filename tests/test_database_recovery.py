"""
Tests for database corruption detection and recovery mechanisms.

Verifies that DatabaseManager can:
1. Detect corrupted databases via get_database_health() and PRAGMA quick_check
2. Run deep blob probes that catch overflow-page corruption in user_model.db
3. Recover from corruption via _check_and_recover_db() by backing up corrupt
   files and allowing schema re-initialization
4. Handle WAL/SHM sidecar files during recovery

These tests exercise the critical self-healing path that is actively used in
production when user_model.db becomes corrupted.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


def _corrupt_database(db_path: Path, offset: int = 4096, num_bytes: int = 1024) -> None:
    """Write zero bytes into the middle of a database file to simulate corruption.

    Seeks past the SQLite header to corrupt B-tree or overflow pages that
    PRAGMA quick_check will detect.

    Args:
        db_path: Path to the database file.
        offset: Byte offset to start writing corruption.
        num_bytes: Number of zero bytes to write.
    """
    with open(db_path, "r+b") as f:
        f.seek(offset)
        f.write(bytes(num_bytes))


class TestCorruptionDetection:
    """Group 1: Verify get_database_health() detects corrupted databases."""

    def test_detects_corrupted_database(self, db):
        """Corrupting user_model.db should cause get_database_health() to report 'corrupted'."""
        db_path = db.data_dir / "user_model.db"

        # Insert some data first so the DB has enough pages to corrupt
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )
            for i in range(50):
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value, source_episodes) VALUES (?, ?, ?, ?)",
                    (f"fact_{i}", "preference", "value_" * 100, json.dumps(["ep1", "ep2"])),
                )

        # Corrupt the database at multiple offsets to ensure detection
        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        health = db.get_database_health()
        assert health["user_model"]["status"] == "corrupted"
        assert len(health["user_model"]["errors"]) > 0

    def test_healthy_dbs_unaffected_by_one_corrupt(self, db):
        """Corrupting only user_model.db should leave the other 4 databases healthy."""
        db_path = db.data_dir / "user_model.db"

        # Insert data then corrupt
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        health = db.get_database_health()

        # The other four databases should all still be healthy
        for db_name in ("events", "entities", "state", "preferences"):
            assert health[db_name]["status"] == "ok", (
                f"{db_name} should be healthy but got errors: {health[db_name]['errors']}"
            )

    def test_missing_database_file(self, db):
        """Deleting a database file should not crash get_database_health()."""
        db_path = db.data_dir / "user_model.db"
        db_path.unlink()

        # Should handle gracefully — either report an error status or handle missing file
        health = db.get_database_health()
        assert "user_model" in health
        # The file is gone, so size_bytes should be 0
        assert health["user_model"]["size_bytes"] == 0


class TestDeepBlobProbes:
    """Group 2: Verify deep blob probes for user_model.db."""

    def test_blob_probes_pass_on_healthy_db(self, db):
        """Blob probes should pass when user_model.db contains valid data."""
        with db.get_connection("user_model") as conn:
            # Insert data into tables that blob probes scan
            conn.execute(
                "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary, content_full) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("ep-1", "2026-01-01T00:00:00Z", "evt-1", "email", "Summary", "Full content " * 100),
            )
            conn.execute(
                "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                ("linguistic", json.dumps({"patterns": "data_" + "x" * 500})),
            )
            conn.execute(
                "INSERT INTO semantic_facts (key, category, value, source_episodes) VALUES (?, ?, ?, ?)",
                ("test_key", "preference", "large_value_" * 100, json.dumps(["ep1", "ep2"])),
            )

        health = db.get_database_health()
        assert health["user_model"]["status"] == "ok"
        assert health["user_model"]["errors"] == []

    def test_health_check_includes_size_bytes(self, db):
        """After inserting data, size_bytes should be positive and path should end with user_model.db."""
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                ("linguistic", json.dumps({"patterns": "data_" + "x" * 500})),
            )

        health = db.get_database_health()
        assert health["user_model"]["size_bytes"] > 0
        assert health["user_model"]["path"].endswith("user_model.db")


class TestRecoveryMechanism:
    """Group 3: Verify _check_and_recover_db() recovery mechanism."""

    def test_recovery_on_corrupt_db(self, db):
        """_check_and_recover_db should return True when database is corrupted."""
        db_path = db.data_dir / "user_model.db"

        # Insert data then corrupt
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        result = db._check_and_recover_db("user_model")
        assert result is True

    def test_recovery_creates_backup(self, db):
        """After recovery, a .corrupt.* backup file should exist in the data directory."""
        db_path = db.data_dir / "user_model.db"

        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        db._check_and_recover_db("user_model")

        # Check that a backup file was created matching user_model.db.corrupt.*
        backups = list(db.data_dir.glob("user_model.db.corrupt.*"))
        assert len(backups) >= 1, f"Expected backup file but found: {list(db.data_dir.iterdir())}"

    def test_recovery_removes_original(self, db):
        """After recovery, the original user_model.db should no longer exist (renamed to .corrupt.*)."""
        db_path = db.data_dir / "user_model.db"

        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        db._check_and_recover_db("user_model")

        assert not db_path.exists(), "Original user_model.db should have been renamed"

    def test_recovery_on_healthy_db_returns_false(self, db):
        """_check_and_recover_db should return False when database is healthy."""
        result = db._check_and_recover_db("user_model")
        assert result is False

    def test_recovery_on_missing_db_returns_false(self, db):
        """_check_and_recover_db should return False when database file doesn't exist."""
        db_path = db.data_dir / "user_model.db"
        db_path.unlink()

        result = db._check_and_recover_db("user_model")
        assert result is False

    def test_schema_reinit_after_recovery(self, db):
        """After recovery backs up the corrupt file, _init_user_model_db should create a working DB."""
        db_path = db.data_dir / "user_model.db"

        # Insert data and corrupt
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        # Recovery renames the corrupt file
        recovered = db._check_and_recover_db("user_model")
        assert recovered is True
        assert not db_path.exists()

        # Re-initialize the user_model database
        db._init_user_model_db()

        # The fresh database should be fully functional
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ep-new", "2026-01-01T00:00:00Z", "evt-new", "email", "Test summary"),
            )

        with db.get_connection("user_model") as conn:
            row = conn.execute("SELECT id FROM episodes WHERE id = ?", ("ep-new",)).fetchone()
            assert row is not None
            assert row["id"] == "ep-new"


class TestWalShmSidecarHandling:
    """Group 4: Verify WAL/SHM sidecar files are handled during recovery."""

    def test_recovery_backs_up_wal_sidecar(self, db):
        """Recovery should also rename the -wal sidecar file alongside the main DB."""
        db_path = db.data_dir / "user_model.db"
        wal_path = db.data_dir / "user_model.db-wal"

        # Insert data to generate WAL
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        # Create a WAL sidecar file (may already exist from WAL mode writes)
        if not wal_path.exists():
            wal_path.write_bytes(b"fake wal data for testing")

        assert wal_path.exists(), "WAL file should exist before recovery"

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        db._check_and_recover_db("user_model")

        # WAL file should have been renamed
        assert not wal_path.exists(), "WAL sidecar should have been renamed during recovery"

        # A backup of the WAL file should exist
        wal_backups = list(db.data_dir.glob("user_model.db-wal.corrupt.*"))
        assert len(wal_backups) >= 1, f"Expected WAL backup but found: {list(db.data_dir.iterdir())}"

    def test_recovery_backs_up_shm_sidecar(self, db):
        """Recovery should also rename the -shm sidecar file alongside the main DB."""
        db_path = db.data_dir / "user_model.db"
        shm_path = db.data_dir / "user_model.db-shm"

        # Insert data
        with db.get_connection("user_model") as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO signal_profiles (profile_type, data) VALUES (?, ?)",
                    (f"type_{i}", json.dumps({"pattern": "x" * 500})),
                )

        # Create an SHM sidecar file (may already exist from WAL mode)
        if not shm_path.exists():
            shm_path.write_bytes(b"fake shm data for testing")

        assert shm_path.exists(), "SHM file should exist before recovery"

        _corrupt_database(db_path, offset=4096, num_bytes=1024)
        _corrupt_database(db_path, offset=8192, num_bytes=1024)

        db._check_and_recover_db("user_model")

        # SHM file should have been renamed
        assert not shm_path.exists(), "SHM sidecar should have been renamed during recovery"

        # A backup of the SHM file should exist
        shm_backups = list(db.data_dir.glob("user_model.db-shm.corrupt.*"))
        assert len(shm_backups) >= 1, f"Expected SHM backup but found: {list(db.data_dir.iterdir())}"
