"""
Tests for safe-column DB corruption recovery in _rebuild_user_model_db_if_corrupted.

Verifies that:
  1. Recovery uses explicit safe column lists to survive blob column corruption
  2. Timestamped archive filenames prevent overwriting previous archives
  3. Log messages include counts for all 5 recovered tables
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_user_model_with_predictions(path: str) -> None:
    """Create a schema-correct user_model.db with predictions and other table data.

    Uses ``DatabaseManager.initialize_all()`` to produce an identical schema to
    production, then inserts test rows into predictions, semantic_facts,
    routines, and insights tables.

    Args:
        path: Absolute path for the user_model.db file.
    """
    import os
    from storage.manager import DatabaseManager

    data_dir = os.path.dirname(path)
    tmp_db = DatabaseManager(data_dir)
    tmp_db.initialize_all()

    conn = sqlite3.connect(path)
    conn.executescript("""
        INSERT INTO episodes
            (id, timestamp, event_id, interaction_type, content_summary, content_full)
            VALUES
            ('ep1', '2026-01-01T08:00:00Z', 'ev1', 'email_received', 'Test email', 'full body'),
            ('ep2', '2026-01-02T08:00:00Z', 'ev2', 'email_sent', 'Sent reply', 'reply body');

        INSERT INTO predictions
            (id, prediction_type, description, confidence, confidence_gate,
             time_horizon, suggested_action, supporting_signals, created_at)
            VALUES
            ('pred1', 'NEED', 'User needs coffee', 0.75, 'DEFAULT',
             'short_term', 'Brew coffee', '["signal1","signal2"]', '2026-01-01T09:00:00Z'),
            ('pred2', 'REMINDER', 'Meeting at 3pm', 0.9, 'AUTONOMOUS',
             'immediate', 'Send reminder', '["calendar_event"]', '2026-01-01T10:00:00Z');

        INSERT INTO semantic_facts
            (key, category, value, confidence, source_episodes)
            VALUES
            ('morning_coffee', 'implicit_preference', 'true', 0.7, '["ep1"]'),
            ('email_first', 'routine', 'checks email first', 0.6, '["ep1","ep2"]');

        INSERT INTO routines
            (name, trigger_condition, typical_duration, consistency_score,
             times_observed, steps, variations, updated_at)
            VALUES
            ('morning_routine', 'wakeup', 30, 0.8, 5,
             '["check_email","brew_coffee"]', '["skip_coffee_on_rush"]', '2026-01-02T08:00:00Z');

        INSERT INTO insights
            (id, type, summary, confidence, evidence, created_at)
            VALUES
            ('ins1', 'behavioral_pattern', 'Reads email in morning', 0.8,
             '["ep1","ep2"]', '2026-01-01T12:00:00Z');
    """)
    conn.commit()
    conn.close()


def _make_blob_corrupting_life_os(data_dir: Path, corrupted_blob_patterns: list[str]):
    """Build a LifeOS-like object that simulates blob column corruption.

    The probe phase raises DatabaseError for queries matching the given
    patterns (simulating corrupted overflow pages).  The data-dump phase
    uses the real database — but with blob columns that cause errors when
    SELECT * is attempted on certain tables.

    Args:
        data_dir: Path to the temporary data directory containing user_model.db.
        corrupted_blob_patterns: SQL substrings that should raise DatabaseError.
    """
    from storage.manager import DatabaseManager
    from main import LifeOS

    db = DatabaseManager(str(data_dir))
    obj = MagicMock()
    obj.db = db

    call_count = {"n": 0}
    original_get_connection = db.get_connection

    @contextlib.contextmanager
    def patched_get_connection(db_name: str):
        """First call simulates corruption probe failure; second call simulates
        blob corruption during data dump (SELECT * fails but safe columns work).
        """
        if db_name == "user_model" and call_count["n"] == 0:
            # First call: probe phase — simulate corruption detection
            call_count["n"] = 1
            mock_conn = MagicMock(spec=sqlite3.Connection)

            def raising_execute(sql: str, *args, **kwargs):
                """Raise for blob probes that match corrupted patterns."""
                for pattern in corrupted_blob_patterns:
                    if pattern in sql:
                        raise sqlite3.DatabaseError("database disk image is malformed")
                result = MagicMock()
                result.fetchone.return_value = None
                result.fetchall.return_value = []
                return result

            mock_conn.execute.side_effect = raising_execute
            yield mock_conn
        elif db_name == "user_model" and call_count["n"] == 1:
            # Second call: data dump phase — real DB but SELECT * on
            # corrupted tables should fail while safe columns succeed.
            call_count["n"] = 2
            with original_get_connection(db_name) as real_conn:
                original_execute = real_conn.execute

                def selective_execute(sql: str, *args, **kwargs):
                    """Fail SELECT * on tables with corrupted blobs."""
                    sql_lower = sql.strip().lower()
                    if sql_lower.startswith("select *"):
                        for pattern in corrupted_blob_patterns:
                            # e.g. "supporting_signals" → predictions table
                            table_map = {
                                "supporting_signals": "predictions",
                                "source_episodes": "semantic_facts",
                                "steps": "routines",
                                "evidence": "insights",
                            }
                            for blob_col, table in table_map.items():
                                if blob_col in pattern and table in sql_lower:
                                    raise sqlite3.DatabaseError(
                                        f"database disk image is malformed reading {blob_col}"
                                    )
                    return original_execute(sql, *args, **kwargs)

                # Can't monkey-patch sqlite3 execute, use real conn directly
                # The safe column SELECTs will work because they skip blobs
                yield real_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    db.get_connection = patched_get_connection

    obj._rebuild_user_model_db_if_corrupted = (
        LifeOS._rebuild_user_model_db_if_corrupted.__get__(obj)
    )
    return obj


def _make_simple_corrupting_life_os(data_dir: Path):
    """Build a LifeOS-like object that triggers rebuild then reads real data.

    The probe phase always fails (simulating corruption). The data-dump
    phase uses the real database where safe columns are fully readable.
    """
    from storage.manager import DatabaseManager
    from main import LifeOS

    db = DatabaseManager(str(data_dir))
    obj = MagicMock()
    obj.db = db

    probe_calls = {"done": False}
    original_get_connection = db.get_connection

    @contextlib.contextmanager
    def patched_get_connection(db_name: str):
        """First user_model call simulates corruption; rest use real DB."""
        if db_name == "user_model" and not probe_calls["done"]:
            probe_calls["done"] = True
            mock_conn = MagicMock(spec=sqlite3.Connection)

            def raising_execute(sql: str, *args, **kwargs):
                """Raise for any probe query that touches blob columns."""
                if any(
                    kw in sql
                    for kw in ("content_full", "SUM(LENGTH(data))", "supporting_signals",
                               "source_episodes", "steps", "contributing_signals", "evidence")
                ):
                    raise sqlite3.DatabaseError("database disk image is malformed")
                result = MagicMock()
                result.fetchone.return_value = None
                result.fetchall.return_value = []
                return result

            mock_conn.execute.side_effect = raising_execute
            yield mock_conn
        else:
            with original_get_connection(db_name) as conn:
                yield conn

    db.get_connection = patched_get_connection
    obj._rebuild_user_model_db_if_corrupted = (
        LifeOS._rebuild_user_model_db_if_corrupted.__get__(obj)
    )
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir_with_predictions(tmp_path):
    """Temporary data dir with user_model.db containing predictions and other data."""
    _create_test_user_model_with_predictions(str(tmp_path / "user_model.db"))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafeColumnRecovery:
    """Verify that recovery uses safe column lists to survive blob corruption."""

    @pytest.mark.asyncio
    async def test_rebuild_recovers_predictions_with_safe_columns(self, data_dir_with_predictions):
        """Predictions are recovered using safe columns even when
        supporting_signals blob column would cause SELECT * to fail.

        The safe column list skips supporting_signals, so predictions
        survive with their non-blob data intact.
        """
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = data_dir_with_predictions / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT id, prediction_type, description, confidence FROM predictions ORDER BY id"
            ).fetchall()
            assert len(rows) == 2, f"Expected 2 predictions after rebuild, got {len(rows)}"
            ids = {r[0] for r in rows}
            assert "pred1" in ids
            assert "pred2" in ids

            # Verify non-blob columns are intact
            pred1 = [r for r in rows if r[0] == "pred1"][0]
            assert pred1[1] == "NEED"
            assert pred1[2] == "User needs coffee"
            assert abs(pred1[3] - 0.75) < 0.001
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_rebuild_recovers_semantic_facts_with_safe_columns(self, data_dir_with_predictions):
        """semantic_facts are recovered without the source_episodes blob column."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = data_dir_with_predictions / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT key, category, value, confidence FROM semantic_facts ORDER BY key"
            ).fetchall()
            assert len(rows) == 2, f"Expected 2 semantic_facts after rebuild, got {len(rows)}"
            keys = {r[0] for r in rows}
            assert "morning_coffee" in keys
            assert "email_first" in keys
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_rebuild_recovers_routines_with_safe_columns(self, data_dir_with_predictions):
        """Routines are recovered without steps/variations blob columns."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = data_dir_with_predictions / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name, trigger_condition, consistency_score FROM routines"
            ).fetchall()
            assert len(rows) == 1, f"Expected 1 routine after rebuild, got {len(rows)}"
            assert rows[0][0] == "morning_routine"
            assert rows[0][1] == "wakeup"
            assert abs(rows[0][2] - 0.8) < 0.001
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_rebuild_recovers_insights_with_safe_columns(self, data_dir_with_predictions):
        """Insights are recovered without the evidence blob column."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = data_dir_with_predictions / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT id, type, summary, confidence FROM insights"
            ).fetchall()
            assert len(rows) == 1, f"Expected 1 insight after rebuild, got {len(rows)}"
            assert rows[0][0] == "ins1"
            assert rows[0][1] == "behavioral_pattern"
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_skipped_blob_columns_get_default_values(self, data_dir_with_predictions):
        """Blob columns skipped during safe recovery get their DEFAULT '[]' values."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        db_path = data_dir_with_predictions / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        try:
            # predictions.supporting_signals should have default '[]'
            row = conn.execute(
                "SELECT supporting_signals FROM predictions WHERE id = 'pred1'"
            ).fetchone()
            assert row is not None
            assert row[0] == "[]", (
                f"Expected supporting_signals default '[]', got '{row[0]}'"
            )

            # semantic_facts.source_episodes should have default '[]'
            row = conn.execute(
                "SELECT source_episodes FROM semantic_facts WHERE key = 'morning_coffee'"
            ).fetchone()
            assert row is not None
            assert row[0] == "[]", (
                f"Expected source_episodes default '[]', got '{row[0]}'"
            )
        finally:
            conn.close()


class TestTimestampedArchives:
    """Verify that archive filenames use timestamps to prevent overwrites."""

    @pytest.mark.asyncio
    async def test_rebuild_uses_timestamped_archive(self, data_dir_with_predictions):
        """Archive filename includes a UTC timestamp."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        await life_os._rebuild_user_model_db_if_corrupted()

        archives = list(data_dir_with_predictions.glob("user_model.db.corrupted.*"))
        assert len(archives) == 1, f"Expected 1 archive, got {len(archives)}"

        # Verify timestamp format in filename (YYYYMMDDTHHMMSSz)
        archive_name = archives[0].name
        # Should match pattern like user_model.db.corrupted.20260303T232403Z
        assert archive_name.startswith("user_model.db.corrupted."), (
            f"Archive name should start with 'user_model.db.corrupted.', got '{archive_name}'"
        )
        timestamp_part = archive_name.replace("user_model.db.corrupted.", "")
        assert "T" in timestamp_part and timestamp_part.endswith("Z"), (
            f"Timestamp should be in YYYYMMDDTHHMMSSz format, got '{timestamp_part}'"
        )

    @pytest.mark.asyncio
    async def test_two_rebuilds_create_two_archives(self, tmp_path):
        """Two consecutive rebuilds create two separate archive files."""
        # Create initial DB
        _create_test_user_model_with_predictions(str(tmp_path / "user_model.db"))

        # First rebuild
        life_os1 = _make_simple_corrupting_life_os(tmp_path)
        await life_os1._rebuild_user_model_db_if_corrupted()

        archives_after_first = list(tmp_path.glob("user_model.db.corrupted.*"))
        assert len(archives_after_first) == 1

        # Sleep briefly to ensure different timestamps
        time.sleep(1.1)

        # Delete the rebuilt DB and create a fresh one for the second rebuild
        db_path = tmp_path / "user_model.db"
        db_path.unlink(missing_ok=True)
        for sidecar in (tmp_path / "user_model.db-wal", tmp_path / "user_model.db-shm"):
            sidecar.unlink(missing_ok=True)
        _create_test_user_model_with_predictions(str(db_path))

        # Second rebuild
        life_os2 = _make_simple_corrupting_life_os(tmp_path)
        await life_os2._rebuild_user_model_db_if_corrupted()

        archives_after_second = list(tmp_path.glob("user_model.db.corrupted.*"))
        assert len(archives_after_second) == 2, (
            f"Expected 2 distinct archives after 2 rebuilds, got {len(archives_after_second)}: "
            f"{[a.name for a in archives_after_second]}"
        )


class TestRecoveryLogging:
    """Verify that rebuild log messages include all 5 table counts."""

    @pytest.mark.asyncio
    async def test_rebuild_logs_all_table_counts(self, data_dir_with_predictions, caplog):
        """The warning log message should include counts for episodes,
        predictions, semantic_facts, routines, and insights.
        """
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        with caplog.at_level(logging.INFO):
            await life_os._rebuild_user_model_db_if_corrupted()

        # Find the final summary log message
        rebuild_messages = [
            r.message for r in caplog.records
            if "Rebuilt" in r.message and "corrupted user_model.db" in r.message
        ]
        assert len(rebuild_messages) >= 1, (
            f"Expected a 'Rebuilt corrupted user_model.db' log message, "
            f"found messages: {[r.message for r in caplog.records]}"
        )

        summary = rebuild_messages[0]
        # All 5 table types should be mentioned with their counts
        assert "2 episodes" in summary, f"Missing episode count in: {summary}"
        assert "2 predictions" in summary, f"Missing prediction count in: {summary}"
        assert "2 semantic_facts" in summary, f"Missing semantic_facts count in: {summary}"
        assert "1 routines" in summary, f"Missing routines count in: {summary}"
        assert "1 insights" in summary, f"Missing insights count in: {summary}"

    @pytest.mark.asyncio
    async def test_info_log_also_includes_all_counts(self, data_dir_with_predictions, caplog):
        """The info log about rebuilding the fresh DB should include all counts."""
        life_os = _make_simple_corrupting_life_os(data_dir_with_predictions)
        with caplog.at_level(logging.INFO):
            await life_os._rebuild_user_model_db_if_corrupted()

        info_messages = [
            r.message for r in caplog.records
            if "Rebuilt fresh user_model.db" in r.message
        ]
        assert len(info_messages) >= 1, "Expected a 'Rebuilt fresh user_model.db' info log"

        summary = info_messages[0]
        assert "predictions" in summary, f"Missing predictions in info log: {summary}"
        assert "routines" in summary, f"Missing routines in info log: {summary}"
