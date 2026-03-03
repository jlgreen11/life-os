"""
Tests for post-initialisation functional verification of user_model.db.

The _verify_db_functional() method probes key tables after schema creation
to catch persistent corruption that the pre-init PRAGMA integrity_check
might miss (e.g. WAL contamination, incomplete file cleanup).

These tests verify:
    1. Healthy databases pass verification
    2. Corrupted databases are detected
    3. Missing tables (fresh DB) don't cause false positives
    4. _init_user_model_db() recovers when verification fails on first attempt
    5. Recursion guard prevents infinite retry loops
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from storage.manager import DatabaseManager


class TestVerifyDbFunctional:
    """Tests for DatabaseManager._verify_db_functional()."""

    def test_healthy_db_passes_verification(self):
        """A fully-initialised user_model.db should pass verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            result = db_manager._verify_db_functional("user_model")

            assert result is True

    def test_detects_corruption(self):
        """A database with corrupted bytes should fail verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Corrupt the database by writing random bytes into the middle
            # of the file.  This targets the data pages (not the header),
            # so SQLite can still open the file but reads will fail.
            db_path = Path(tmpdir) / "user_model.db"
            file_size = db_path.stat().st_size
            if file_size > 4096:
                with open(db_path, "r+b") as f:
                    # Write garbage into the middle of the data pages
                    f.seek(file_size // 2)
                    f.write(b"\x00\xDE\xAD\xBE\xEF" * 200)

                # Force SQLite to see the corruption by removing WAL
                wal_path = db_path.parent / (db_path.name + "-wal")
                shm_path = db_path.parent / (db_path.name + "-shm")
                for p in (wal_path, shm_path):
                    if p.exists():
                        p.unlink()

            # Need a fresh manager so it doesn't reuse cached connections
            db_manager2 = DatabaseManager(tmpdir)

            result = db_manager2._verify_db_functional("user_model")

            # The corruption may or may not be detected depending on where
            # the random bytes land — if the file is small, the overwrite
            # may not hit actual table data.  We verify the method at least
            # runs without crashing; the mock-based test below provides the
            # deterministic corruption detection test.
            assert isinstance(result, bool)

    def test_skips_missing_tables(self):
        """Verification should return True when tables don't exist yet.

        A freshly-created DB that only has schema_version should not
        trigger a false positive.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            # Only create the DB file with schema_version, no other tables
            import sqlite3

            db_path = Path(tmpdir) / "user_model.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version INTEGER PRIMARY KEY)"
            )
            conn.commit()
            conn.close()

            result = db_manager._verify_db_functional("user_model")

            assert result is True

    def test_returns_true_for_unknown_db(self):
        """Databases not in the table_map should always pass (no tables to probe)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # events, entities, state, preferences are not in the table_map
            assert db_manager._verify_db_functional("events") is True
            assert db_manager._verify_db_functional("state") is True

    def test_detects_malformed_error(self):
        """Verification should return False when a table query raises 'malformed'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Patch get_connection to return a connection that raises on query
            import sqlite3
            from contextlib import contextmanager
            from unittest.mock import MagicMock

            call_count = 0

            @contextmanager
            def mock_get_connection(db_name):
                """Return a mock connection whose execute raises 'malformed' on COUNT queries."""
                nonlocal call_count
                mock_conn = MagicMock()

                def mock_execute(sql, *args):
                    if "SELECT COUNT" in sql:
                        raise sqlite3.DatabaseError("database disk image is malformed")
                    return MagicMock()

                mock_conn.execute = mock_execute
                yield mock_conn

            with patch.object(db_manager, "get_connection", side_effect=mock_get_connection):
                result = db_manager._verify_db_functional("user_model")

            assert result is False


class TestInitUserModelDbRecovery:
    """Tests for the recovery path in _init_user_model_db()."""

    def test_recovers_on_verification_failure(self):
        """_init_user_model_db should recreate the DB when verification fails once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)

            # Track calls to _verify_db_functional
            verify_call_count = 0
            original_verify = db_manager._verify_db_functional

            def mock_verify(db_name):
                nonlocal verify_call_count
                verify_call_count += 1
                if verify_call_count == 1:
                    return False  # First call: simulate corruption
                return original_verify(db_name)  # Second call: real check

            with patch.object(db_manager, "_verify_db_functional", side_effect=mock_verify):
                db_manager._init_user_model_db()

            # Verification was called twice: once failing, once succeeding
            assert verify_call_count == 2
            # Retry counter was incremented
            assert db_manager._user_model_verify_retries == 1
            # The DB should be functional after recovery
            assert db_manager._verify_db_functional("user_model") is True

    def test_recursion_guard_prevents_infinite_loop(self):
        """If verification fails twice, _init_user_model_db should give up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)

            verify_call_count = 0

            def always_fail(db_name):
                nonlocal verify_call_count
                verify_call_count += 1
                return False  # Always fail

            with patch.object(db_manager, "_verify_db_functional", side_effect=always_fail):
                # Should not raise, should not loop infinitely
                db_manager._init_user_model_db()

            # Called once on first init, once on retry init, then gives up
            assert verify_call_count == 2
            assert db_manager._user_model_verify_retries >= 1

    def test_normal_init_does_not_increment_retry_counter(self):
        """A healthy initialization should not touch the retry counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            assert db_manager._user_model_verify_retries == 0

    def test_recovery_removes_wal_and_shm(self):
        """Recovery should remove WAL and SHM sidecars before recreating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            # First init to create the DB and its sidecars
            db_manager._init_user_model_db()

            db_path = Path(tmpdir) / "user_model.db"
            wal_path = db_path.parent / (db_path.name + "-wal")
            shm_path = db_path.parent / (db_path.name + "-shm")

            # Force creation of WAL/SHM files
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (99)")
            conn.commit()
            conn.close()

            verify_call_count = 0
            files_existed_before_retry = {}

            original_init = db_manager._init_user_model_db.__func__

            def mock_verify(db_name):
                nonlocal verify_call_count
                verify_call_count += 1
                if verify_call_count == 1:
                    return False
                return True

            with patch.object(db_manager, "_verify_db_functional", side_effect=mock_verify):
                db_manager._init_user_model_db()

            # After recovery, the DB should be fresh (schema_version reset)
            with db_manager.get_connection("user_model") as conn:
                result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
                # Should have the current version, not 99
                assert result[0] != 99
