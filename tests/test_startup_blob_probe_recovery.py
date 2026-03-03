"""
Tests for blob overflow page probe during startup DB recovery.

Verifies that ``_check_and_recover_db('user_model')`` runs blob probes
in addition to ``PRAGMA quick_check``, catching overflow page corruption
that quick_check misses in large JSON TEXT columns.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

from storage.manager import DatabaseManager


class _CorruptBlobConnection:
    """Wraps a real sqlite3.Connection but raises on specific blob probe queries.

    Simulates the "database disk image is malformed" error that SQLite raises
    when trying to read corrupt overflow pages in large TEXT/BLOB columns.
    """

    def __init__(self, real_conn, corrupt_query_fragment="SUM(LENGTH(data)) FROM signal_profiles"):
        self._real = real_conn
        self._corrupt_fragment = corrupt_query_fragment

    def execute(self, sql, *args, **kwargs):
        """Raise DatabaseError if the query matches the corrupt fragment."""
        if self._corrupt_fragment in sql:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return self._real.execute(sql, *args, **kwargs)

    def close(self):
        """Forward close to the real connection."""
        return self._real.close()

    def __getattr__(self, name):
        """Proxy all other attributes to the real connection."""
        return getattr(self._real, name)


def _make_connect_wrapper(corrupt_query_fragment, trigger_count=None):
    """Create a sqlite3.connect wrapper that returns corrupt connections for user_model.db.

    Args:
        corrupt_query_fragment: SQL fragment that triggers the simulated corruption error.
        trigger_count: If set, only the Nth connect call to user_model gets the wrapper.
                       If None, all user_model connections are wrapped.
    """
    real_connect = sqlite3.connect
    call_counter = {"n": 0}

    def wrapper(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if "user_model" in str(database):
            call_counter["n"] += 1
            # The blob probes use the second connection opened during
            # _check_and_recover_db (first is for PRAGMA quick_check).
            if trigger_count is None or call_counter["n"] == trigger_count:
                return _CorruptBlobConnection(conn, corrupt_query_fragment)
        return conn

    return wrapper


class TestBlobProbeRecoveryOnStartup:
    """Verify blob overflow probes during user_model.db recovery."""

    def test_healthy_user_model_passes_blob_probes(self, tmp_path):
        """A fully healthy user_model.db should pass both quick_check and blob probes."""
        data_dir = str(tmp_path)
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # Running recovery again on a healthy DB should return False (no recovery needed).
        result = manager._check_and_recover_db("user_model")
        assert result is False

        # No backup files should exist.
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 0

    def test_blob_probe_failure_triggers_recovery(self, tmp_path):
        """When a blob probe raises a non-missing-table error, recovery should trigger.

        Simulates blob overflow corruption by wrapping sqlite3.connect to return
        a connection that raises "database disk image is malformed" on the
        signal_profiles probe query.
        """
        data_dir = str(tmp_path)

        # Step 1: Create a healthy, fully-initialized user_model.db.
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # Step 2: Wrap sqlite3.connect so the blob probe connection raises.
        # The second connect call on user_model.db during _check_and_recover_db
        # is the blob probe connection (first is for PRAGMA quick_check).
        wrapper = _make_connect_wrapper("SUM(LENGTH(data)) FROM signal_profiles", trigger_count=2)

        with patch("storage.manager.sqlite3.connect", side_effect=wrapper):
            result = manager._check_and_recover_db("user_model")

        assert result is True, "Expected recovery to trigger on blob probe failure"

        # A .corrupt. backup file should have been created.
        backup_files = list(tmp_path.glob("user_model.db.corrupt.*"))
        assert len(backup_files) >= 1, "Expected a .corrupt backup file"

    def test_missing_tables_do_not_trigger_false_positive(self, tmp_path):
        """An empty/new DB with no tables should NOT trigger blob probe corruption.

        When tables don't exist yet (e.g. fresh install), the probes should
        get 'no such table' errors which are intentionally ignored.
        """
        data_dir = str(tmp_path)

        # Create a valid but empty SQLite database (no tables at all).
        db_path = tmp_path / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        manager = DatabaseManager(data_dir=data_dir)

        # _check_and_recover_db should pass — missing tables are not corruption.
        result = manager._check_and_recover_db("user_model")
        assert result is False, "Missing tables should not trigger corruption detection"

        # No backup files should exist.
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 0

    def test_non_user_model_databases_skip_blob_probes(self, tmp_path):
        """Blob probes should only run for user_model, not for other databases.

        Verifies that _check_and_recover_db for a non-user_model database
        does not attempt any blob probe queries — even if those queries would
        fail on that database.
        """
        data_dir = str(tmp_path)
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # The state database should pass recovery without any blob probes.
        result = manager._check_and_recover_db("state")
        assert result is False

        # No backup files should exist for state.db.
        backup_files = list(tmp_path.glob("state.db.corrupt.*"))
        assert len(backup_files) == 0

    def test_blob_probe_logs_distinct_warning(self, tmp_path, caplog):
        """Blob overflow corruption should log a distinct warning message."""
        import logging

        data_dir = str(tmp_path)
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        wrapper = _make_connect_wrapper("SUM(LENGTH(data)) FROM signal_profiles", trigger_count=2)

        with patch("storage.manager.sqlite3.connect", side_effect=wrapper):
            with caplog.at_level(logging.WARNING, logger="storage.manager"):
                manager._check_and_recover_db("user_model")

        # The distinct blob overflow warning should be present.
        assert any(
            "Blob overflow corruption detected" in record.message for record in caplog.records
        ), "Expected 'Blob overflow corruption detected' warning in logs"

    def test_recovery_produces_functional_db_after_blob_corruption(self, tmp_path):
        """After blob-corruption recovery, re-initialisation should produce a working DB.

        This is the end-to-end scenario: user_model.db has blob corruption,
        startup detects it, backs it up, and re-init creates a fresh schema.
        """
        data_dir = str(tmp_path)

        # Step 1: Create a healthy database.
        manager = DatabaseManager(data_dir=data_dir)
        manager.initialize_all()

        # Step 2: Trigger blob corruption recovery.
        wrapper = _make_connect_wrapper("SUM(LENGTH(data)) FROM signal_profiles", trigger_count=2)

        with patch("storage.manager.sqlite3.connect", side_effect=wrapper):
            recovered = manager._check_and_recover_db("user_model")

        assert recovered is True

        # Step 3: Re-initialise — this simulates what _init_user_model_db does
        # after _check_and_recover_db backs up the corrupt file.
        manager2 = DatabaseManager(data_dir=data_dir)
        manager2.initialize_all()

        # Step 4: Verify the new database is fully functional.
        with manager2.get_connection("user_model") as conn:
            tables = [
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            assert "episodes" in tables
            assert "semantic_facts" in tables
            assert "routines" in tables
            assert "signal_profiles" in tables
            assert "predictions" in tables

            # Verify we can write and read data.
            conn.execute(
                "INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test-1", "2026-01-01T00:00:00Z", "evt-1", "test", "recovery works"),
            )
            row = conn.execute("SELECT content_summary FROM episodes WHERE id = ?", ("test-1",)).fetchone()
            assert row is not None
            assert row["content_summary"] == "recovery works"

    def test_partial_schema_does_not_trigger_false_positive(self, tmp_path):
        """A DB with only some tables should not trigger false corruption.

        Some tables might exist while others don't (e.g. mid-migration).
        Only actual read errors should trigger recovery, not missing tables.
        """
        data_dir = str(tmp_path)

        # Create a DB with only the episodes table.
        db_path = tmp_path / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                content_full TEXT
            )
        """)
        conn.commit()
        conn.close()

        manager = DatabaseManager(data_dir=data_dir)

        # Missing tables (signal_profiles, semantic_facts, etc.) should be
        # skipped gracefully — only the episodes probe should run.
        result = manager._check_and_recover_db("user_model")
        assert result is False, "Partial schema should not trigger false-positive corruption"
