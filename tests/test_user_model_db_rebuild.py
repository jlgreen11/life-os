"""
Tests for _rebuild_user_model_db_if_corrupted in LifeOS startup.

Verifies that when user_model.db has deep B-tree corruption (episodes.content_full
unreadable), the system:
  1. Detects the corruption via a probe query
  2. Recovers all readable episode columns (id, timestamp, interaction_type, etc.)
  3. Preserves semantic_facts, routines, predictions, insights
  4. Creates an empty signal_profiles table (to be repopulated by backfills)
  5. Replaces the corrupted DB with the rebuilt one
  6. Does nothing when the DB is already healthy
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_user_model(path: str) -> None:
    """Create a schema-correct user_model.db for rebuild tests.

    Uses ``DatabaseManager.initialize_all()`` to produce an identical schema to
    what production creates, then inserts minimal test rows.  This approach ensures
    the test schema is always in sync with migrations — no need to manually mirror
    ``CREATE TABLE`` statements that can drift as the schema evolves.

    Args:
        path: Absolute path for the user_model.db file to create.  The parent
              directory is used as the data_dir for the temporary DatabaseManager.
    """
    import os
    from storage.manager import DatabaseManager

    data_dir = os.path.dirname(path)
    # initialize_all() creates all 5 databases using the current production schema
    # and runs migrations up to CURRENT_VERSION.  This is the authoritative source.
    tmp_db = DatabaseManager(data_dir)
    tmp_db.initialize_all()

    # Insert minimal test rows with named columns to survive future schema additions.
    conn = sqlite3.connect(path)
    conn.executescript("""
        INSERT INTO episodes
            (id, timestamp, event_id, interaction_type, content_summary, content_full)
            VALUES
            ('ep1', '2026-01-01T08:00:00Z', 'ev1', 'email_received', 'Test email', 'full body'),
            ('ep2', '2026-01-02T08:00:00Z', 'ev2', 'email_sent', 'Sent reply', 'reply body');

        INSERT INTO semantic_facts (key, category, value, confidence)
            VALUES ('morning_email', 'implicit_preference', 'true', 0.7);

        INSERT INTO insights (id, type, summary, confidence)
            VALUES ('ins1', 'behavioral_pattern', 'Reads email in morning', 0.8);

        INSERT INTO signal_profiles (profile_type, data, samples_count)
            VALUES ('temporal', '{"morning": 0.8}', 10);
    """)
    conn.commit()
    conn.close()


def _make_life_os(data_dir: Path):
    """Build a minimal LifeOS-like object with a real DatabaseManager."""
    from storage.manager import DatabaseManager
    from main import LifeOS

    db = DatabaseManager(str(data_dir))
    obj = MagicMock()
    obj.db = db
    obj._rebuild_user_model_db_if_corrupted = (
        LifeOS._rebuild_user_model_db_if_corrupted.__get__(obj)
    )
    return obj


def _make_corrupting_life_os(data_dir: Path):
    """Build a LifeOS-like object whose first get_connection raises on content_full reads.

    This simulates the production B-tree corruption scenario where:
    - content_full, contacts_involved, topics, entities overflow pages are corrupted
    - All other columns (id, timestamp, interaction_type, etc.) remain readable
    - COUNT(*) still works (uses B-tree header, not leaf data)

    Implementation: replaces db.get_connection with a patched context manager that
    yields a ``MagicMock`` connection on the first ``user_model`` call.  The mock's
    ``execute()`` raises ``sqlite3.DatabaseError`` for queries that touch
    ``content_full`` or ``SUM(LENGTH(data))``, matching the real corruption signature.
    After the first call, all connections use the real ``get_connection`` so the
    rebuild's data-dump phase can read the actual test database.

    Note: ``sqlite3.Connection.execute`` is a read-only C-extension attribute and
    cannot be monkey-patched on a live connection instance.  Using a ``MagicMock``
    for the probe call avoids this limitation while still triggering the exact
    ``sqlite3.DatabaseError`` that production corruption produces.
    """
    from unittest.mock import MagicMock as _MagicMock

    from storage.manager import DatabaseManager
    from main import LifeOS

    db = DatabaseManager(str(data_dir))
    obj = MagicMock()
    obj.db = db

    probe_calls = {"done": False}  # Mutable container so nested function can update it

    original_get_connection = db.get_connection

    @contextlib.contextmanager
    def patched_get_connection(db_name: str):
        """Yield a mock connection on the first user_model call to simulate corruption.

        Subsequent calls fall through to the real ``get_connection`` so that the
        rebuild's data-dump phase reads the actual test database rows.
        """
        if db_name == "user_model" and not probe_calls["done"]:
            probe_calls["done"] = True

            # MagicMock(spec=sqlite3.Connection) restricts attributes to those that
            # exist on a real Connection, preventing accidental typo-driven passes.
            mock_conn = _MagicMock(spec=sqlite3.Connection)

            def raising_execute(sql: str, *args, **kwargs):
                """Raise DatabaseError for queries that touch corrupted overflow pages."""
                if "content_full" in sql or "SUM(LENGTH(data))" in sql:
                    raise sqlite3.DatabaseError("database disk image is malformed")
                # PRAGMA statements and other benign queries return a no-op mock result
                result = _MagicMock()
                result.fetchone.return_value = None
                result.fetchall.return_value = []
                return result

            mock_conn.execute.side_effect = raising_execute
            yield mock_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    # Replace the instance-level get_connection so self.db.get_connection() routes
    # through our patch.  Instance attributes shadow class methods in Python's MRO,
    # so this assignment takes precedence without touching the class.
    db.get_connection = patched_get_connection

    obj._rebuild_user_model_db_if_corrupted = (
        LifeOS._rebuild_user_model_db_if_corrupted.__get__(obj)
    )
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def healthy_data_dir(tmp_path):
    """Temporary data dir with a valid (uncorrupted) user_model.db."""
    _create_test_user_model(str(tmp_path / "user_model.db"))
    return tmp_path


@pytest.fixture()
def simulated_corrupt_data_dir(tmp_path):
    """Temporary data dir with a user_model.db whose content_full reads are patched to fail."""
    _create_test_user_model(str(tmp_path / "user_model.db"))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: healthy DB → no-op
# ---------------------------------------------------------------------------

class TestHealthyDatabase:
    @pytest.mark.asyncio
    async def test_no_rebuild_when_healthy(self, healthy_data_dir):
        """_rebuild_user_model_db_if_corrupted is a no-op when DB is healthy."""
        life_os = _make_life_os(healthy_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        # No archive file should exist — confirms no rebuild happened
        assert not (healthy_data_dir / "user_model.db.corrupted").exists(), \
            "Archive should not exist when DB is healthy"

    @pytest.mark.asyncio
    async def test_data_intact_after_healthy_probe(self, healthy_data_dir):
        """No data is lost when DB is healthy."""
        life_os = _make_life_os(healthy_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        conn = sqlite3.connect(str(healthy_data_dir / "user_model.db"))
        count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        assert count == 2, f"Expected 2 episodes, got {count}"


# ---------------------------------------------------------------------------
# Tests: simulated corruption → rebuild
# ---------------------------------------------------------------------------

class TestCorruptedDatabase:
    @pytest.mark.asyncio
    async def test_rebuild_archives_corrupted_db(self, simulated_corrupt_data_dir):
        """When probe fails, the old DB is archived as user_model.db.corrupted."""
        life_os = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        archive = simulated_corrupt_data_dir / "user_model.db.corrupted"
        assert archive.exists(), "Corrupted DB should be archived after rebuild"

    @pytest.mark.asyncio
    async def test_fresh_db_is_fully_readable(self, simulated_corrupt_data_dir):
        """After rebuild, user_model.db is a fresh, readable database."""
        life_os = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = simulated_corrupt_data_dir / "user_model.db"
        assert db_path.exists(), "New user_model.db should exist after rebuild"

        conn = sqlite3.connect(str(db_path))
        try:
            # All core tables should be accessible without errors
            conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()
            conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_episodes_preserved_with_safe_columns(self, simulated_corrupt_data_dir):
        """Episode rows survive rebuild; safe columns (id, timestamp, etc.) are intact."""
        life_os = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        conn = sqlite3.connect(str(simulated_corrupt_data_dir / "user_model.db"))
        rows = conn.execute(
            "SELECT id, interaction_type, content_summary FROM episodes ORDER BY id"
        ).fetchall()
        conn.close()

        assert len(rows) == 2, f"Expected 2 episodes after rebuild, got {len(rows)}"
        ids = [r[0] for r in rows]
        assert "ep1" in ids
        assert "ep2" in ids
        types = {r[0]: r[1] for r in rows}
        assert types["ep1"] == "email_received"
        assert types["ep2"] == "email_sent"

    @pytest.mark.asyncio
    async def test_semantic_facts_preserved(self, simulated_corrupt_data_dir):
        """semantic_facts table survives the rebuild intact."""
        life_os = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        conn = sqlite3.connect(str(simulated_corrupt_data_dir / "user_model.db"))
        facts = conn.execute(
            "SELECT category, key, value, confidence FROM semantic_facts"
        ).fetchall()
        conn.close()

        assert len(facts) == 1, f"Expected 1 fact, got {len(facts)}"
        assert facts[0][0] == "implicit_preference"
        assert facts[0][1] == "morning_email"
        assert abs(facts[0][3] - 0.7) < 0.001

    @pytest.mark.asyncio
    async def test_signal_profiles_empty_after_rebuild(self, simulated_corrupt_data_dir):
        """signal_profiles is empty after rebuild — backfills will repopulate it."""
        life_os = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os._rebuild_user_model_db_if_corrupted()

        conn = sqlite3.connect(str(simulated_corrupt_data_dir / "user_model.db"))
        count = conn.execute("SELECT COUNT(*) FROM signal_profiles").fetchone()[0]
        conn.close()

        assert count == 0, (
            f"signal_profiles should be empty after rebuild (got {count}); "
            "backfills will repopulate it on next startup"
        )

    @pytest.mark.asyncio
    async def test_idempotent_on_second_call(self, simulated_corrupt_data_dir):
        """After a successful rebuild, a second call to a fresh object is a no-op."""
        # First call: triggers rebuild (corruption simulated)
        life_os1 = _make_corrupting_life_os(simulated_corrupt_data_dir)
        await life_os1._rebuild_user_model_db_if_corrupted()

        archive = simulated_corrupt_data_dir / "user_model.db.corrupted"
        assert archive.exists(), "Archive should exist after first rebuild"
        archive_mtime = archive.stat().st_mtime

        # Second call: DB is now healthy, should be a no-op
        life_os2 = _make_life_os(simulated_corrupt_data_dir)
        await life_os2._rebuild_user_model_db_if_corrupted()

        # Archive should be unchanged (no second rebuild happened)
        assert archive.stat().st_mtime == archive_mtime, \
            "Archive mtime changed — second rebuild should not have fired"
