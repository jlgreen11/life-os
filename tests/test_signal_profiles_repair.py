"""
Tests for signal_profiles corruption detection and auto-repair logic.

The _repair_signal_profiles_if_corrupted() method in main.py detects when the
signal_profiles table's data column is unreadable (SQLite B-tree corruption)
and rebuilds the table so that subsequent backfill methods can repopulate it.
"""

import sqlite3

import pytest


@pytest.mark.asyncio
async def test_repair_healthy_table_is_noop(db, user_model_store):
    """Healthy signal_profiles table: repair method returns without modifying anything.

    Confirms that the corruption check is a cheap, non-destructive read that
    leaves an intact table untouched.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from main import LifeOS

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db

    # Insert a legitimate profile row so the read test can find it.
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count) VALUES (?, ?, ?)",
            ("test_profile", '{"key": "value"}', 5),
        )

    # Should complete without raising and without touching the row count.
    await life_os._repair_signal_profiles_if_corrupted()

    with db.get_connection("user_model") as conn:
        count = conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()[0]
    assert count == 1, "Repair should not drop healthy rows"


@pytest.mark.asyncio
async def test_repair_drops_and_recreates_on_read_error(db):
    """Simulated read failure: method drops and re-creates signal_profiles.

    When reading the data column raises an exception (simulating B-tree
    corruption), the repair method must:
      1. Drop the table.
      2. Re-create it with the same schema.
      3. Leave the table empty so backfill guards can repopulate it.
    """
    from unittest.mock import patch

    from main import LifeOS

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db

    # Seed a row that will be read during the health check.
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count) VALUES (?, ?, ?)",
            ("relationships", '{"contacts": {}}', 10),
        )

    # Patch get_connection to raise a corruption error on the first call
    # (the health-check SELECT), then allow subsequent calls to proceed normally.
    original_get_connection = db.get_connection
    call_count = [0]

    from contextlib import contextmanager

    @contextmanager
    def patched_get_connection(db_name):
        call_count[0] += 1
        if call_count[0] == 1 and db_name == "user_model":
            # Simulate "database disk image is malformed" on first read
            raise sqlite3.DatabaseError("database disk image is malformed")
        # All other calls proceed normally (for DROP + CREATE)
        with original_get_connection(db_name) as conn:
            yield conn

    db.get_connection = patched_get_connection

    try:
        await life_os._repair_signal_profiles_if_corrupted()
    finally:
        db.get_connection = original_get_connection

    # After repair: table must exist and be empty (ready for backfill).
    with db.get_connection("user_model") as conn:
        count = conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()[0]
        # Verify the schema is intact — all expected columns must exist
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(signal_profiles)").fetchall()
        ]

    assert count == 0, "Repaired table should be empty so backfills can repopulate"
    assert "profile_type" in cols
    assert "data" in cols
    assert "samples_count" in cols
    assert "updated_at" in cols


@pytest.mark.asyncio
async def test_repair_detects_partial_corruption_missed_by_limit1(db):
    """Partial table corruption: SUM(LENGTH(data)) catches what LIMIT 1 misses.

    This test validates the core bug fix: before this fix, the health check used
    ``SELECT profile_type, data FROM signal_profiles LIMIT 1``.  In a partially
    corrupted table, the first row is stored in early B-tree pages (usually
    uncorrupted) while later rows occupy overflow pages that are damaged.  The
    LIMIT 1 check fetches only the first row and returns success — a false negative
    that leaves the system silently unable to read any profile beyond the first.

    The fix changes the check to ``SELECT SUM(LENGTH(data)) FROM signal_profiles``,
    which forces SQLite to read every row's overflow pages.  A single corrupted
    page causes the aggregate to fail, correctly triggering the repair.

    Simulation strategy:
      We intercept get_connection for the first user_model call and yield a mock
      connection whose execute() raises a DatabaseError specifically when the SQL
      contains "SUM(LENGTH(data))" — the exact signature of the fixed health check.
      All subsequent calls (DROP + CREATE) use the real connection so the repair
      can complete.

      sqlite3.Connection.execute is a C-level built-in and cannot be patched
      directly.  Instead, we build a thin wrapper object whose execute() delegates
      to the real connection but raises for the aggregate query.
    """
    import sqlite3
    from contextlib import contextmanager

    from main import LifeOS

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db

    # Seed two rows — in the real bug, only the first row's data is readable
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count) VALUES (?, ?, ?)",
            ("linguistic", '{"formality": 0.3}', 10),
        )
        conn.execute(
            "INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count) VALUES (?, ?, ?)",
            ("relationships", '{"contacts": {}}', 20),
        )

    original_get_connection = db.get_connection
    health_check_attempted = [False]
    sum_query_attempted = [False]

    class _PartiallyCorruptedConnection:
        """Wraps a real sqlite3.Connection and raises on the SUM aggregate query.

        This simulates partial B-tree corruption: the first row is readable
        (LIMIT 1 would pass) but the aggregate over all rows hits a corrupted
        overflow page and raises DatabaseError.
        """

        def __init__(self, real_conn):
            self._conn = real_conn

        def execute(self, sql, *args, **kwargs):
            if "SUM(LENGTH(data))" in sql:
                sum_query_attempted[0] = True
                raise sqlite3.DatabaseError("database disk image is malformed")
            return self._conn.execute(sql, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    @contextmanager
    def patched_connection(db_name):
        with original_get_connection(db_name) as real_conn:
            if db_name == "user_model" and not health_check_attempted[0]:
                health_check_attempted[0] = True
                # Yield the wrapper; subsequent calls use the real connection
                # so DROP and CREATE execute normally.
                yield _PartiallyCorruptedConnection(real_conn)
            else:
                yield real_conn

    db.get_connection = patched_connection
    try:
        await life_os._repair_signal_profiles_if_corrupted()
    finally:
        db.get_connection = original_get_connection

    # The SUM query must have been attempted — otherwise we haven't exercised the fix.
    assert sum_query_attempted[0], (
        "Repair must use SUM(LENGTH(data)) to detect corruption, not LIMIT 1"
    )

    # After repair: table exists and is empty (ready for backfill).
    with db.get_connection("user_model") as conn:
        count = conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()[0]
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(signal_profiles)").fetchall()
        ]

    assert count == 0, "Repair must clear the corrupted table for backfill to repopulate"
    assert "profile_type" in cols
    assert "data" in cols
    assert "samples_count" in cols
    assert "updated_at" in cols


@pytest.mark.asyncio
async def test_repair_tolerates_repair_failure(db):
    """If the repair itself fails, the method logs and returns without raising.

    This ensures startup cannot be blocked by a repair attempt that fails
    (e.g., disk full, permissions error).
    """
    from unittest.mock import patch

    from main import LifeOS

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db

    original_get_connection = db.get_connection
    call_count = [0]

    from contextlib import contextmanager

    @contextmanager
    def always_fail(db_name):
        call_count[0] += 1
        raise sqlite3.DatabaseError("database disk image is malformed")

    db.get_connection = always_fail

    try:
        # Must not raise — fail-open behaviour
        await life_os._repair_signal_profiles_if_corrupted()
    finally:
        db.get_connection = original_get_connection

    # No assertion needed beyond "did not raise"
