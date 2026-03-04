"""
Tests for SQLite busy_timeout PRAGMA and WAL checkpoint support.

Verifies that DatabaseManager sets ``PRAGMA busy_timeout=5000`` on every
connection so that concurrent async writers wait instead of raising
``sqlite3.OperationalError: database is locked``.  Also verifies the
``checkpoint_wal()`` maintenance method.
"""

import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


class TestBusyTimeout:
    """Verify that busy_timeout is set on every database connection."""

    def test_busy_timeout_is_set(self, db):
        """get_connection() should set PRAGMA busy_timeout=5000."""
        with db.get_connection("events") as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == 5000

    def test_busy_timeout_on_all_databases(self, db):
        """busy_timeout should be set for all five databases."""
        for db_name in ["events", "entities", "state", "user_model", "preferences"]:
            with db.get_connection(db_name) as conn:
                timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                assert timeout == 5000, f"busy_timeout not set for {db_name}"

    def test_concurrent_writers_do_not_get_locked_error(self, db):
        """Two threads writing to the same DB should not raise 'database is locked'.

        With busy_timeout=5000, the second writer waits up to 5 seconds for the
        lock instead of failing immediately.
        """
        errors: list[Exception] = []
        barrier = threading.Barrier(2, timeout=5)

        def writer(value: str):
            """Write a row from a separate thread."""
            try:
                barrier.wait()
                with db.get_connection("state") as conn:
                    conn.execute(
                        "INSERT INTO kv_store (key, value) VALUES (?, ?)",
                        (f"busy_test_{value}", value),
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer, args=("thread_1",))
        t2 = threading.Thread(target=writer, args=("thread_2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Neither thread should have hit a "database is locked" error
        for err in errors:
            assert "database is locked" not in str(err), (
                f"Concurrent write failed with: {err}"
            )

        # Both rows should be present
        with db.get_connection("state") as conn:
            cursor = conn.execute(
                "SELECT key FROM kv_store WHERE key LIKE 'busy_test_%' ORDER BY key"
            )
            rows = [r["key"] for r in cursor.fetchall()]
            assert "busy_test_thread_1" in rows
            assert "busy_test_thread_2" in rows


class TestCheckpointWal:
    """Verify the checkpoint_wal() maintenance method."""

    def test_checkpoint_succeeds_on_valid_database(self, db):
        """checkpoint_wal() should run without error on an initialized database."""
        # Write some data to create WAL frames
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                ("ckpt_test", "preference", "coffee"),
            )

        # Should not raise
        db.checkpoint_wal("user_model")

    def test_checkpoint_works_on_all_databases(self, db):
        """checkpoint_wal() should succeed for every database name."""
        for db_name in ["events", "entities", "state", "user_model", "preferences"]:
            db.checkpoint_wal(db_name)  # should not raise

    def test_checkpoint_raises_for_unknown_database(self, db):
        """checkpoint_wal() should raise KeyError for an unrecognised database name."""
        with pytest.raises(KeyError):
            db.checkpoint_wal("nonexistent_db")

    def test_checkpoint_called_during_initialize_all(self):
        """initialize_all() should call checkpoint_wal('user_model') at the end.

        Verified by checking the WAL file is zero-length or absent after
        initialize_all() completes — the TRUNCATE checkpoint should clear it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatabaseManager(tmpdir)
            manager.initialize_all()

            wal_path = Path(tmpdir) / "user_model.db-wal"
            # After checkpoint(TRUNCATE), the WAL file should either not exist
            # or be empty (zero length).
            if wal_path.exists():
                assert wal_path.stat().st_size == 0, (
                    "WAL file should be truncated after initialize_all()"
                )
