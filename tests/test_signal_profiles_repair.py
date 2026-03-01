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
