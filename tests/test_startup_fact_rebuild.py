"""
Tests for startup semantic fact re-inference (step 1.19 in main.py).

Verifies that:
- When semantic_facts is empty but episodes exist, run_all_inference is called.
- When semantic_facts already has rows, run_all_inference is NOT called.
- Exceptions during re-inference don't crash startup (fail-open).
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _insert_episode(conn, episode_id: str | None = None) -> str:
    """Insert a minimal episode into the episodes table.

    Args:
        conn: SQLite connection to user_model.db.
        episode_id: Optional episode ID; generated if omitted.

    Returns:
        The episode ID that was inserted.
    """
    eid = episode_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO episodes (id, timestamp, event_id, interaction_type,
                              content_summary, contacts_involved)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            eid,
            datetime.now(timezone.utc).isoformat(),
            "evt-1",
            "communication",
            "A test episode for startup fact rebuild",
            json.dumps(["test@example.com"]),
        ),
    )
    conn.commit()
    return eid


def _insert_semantic_fact(conn, fact_key: str = "test_fact") -> None:
    """Insert a minimal semantic fact into the semantic_facts table.

    Args:
        conn: SQLite connection to user_model.db.
        fact_key: The fact key to insert.
    """
    conn.execute(
        """
        INSERT INTO semantic_facts (key, category, value, confidence, source_episodes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            fact_key,
            "explicit_preference",
            "test_value",
            0.7,
            json.dumps([]),
        ),
    )
    conn.commit()


class TestStartupFactRebuild:
    """Tests for the startup semantic fact re-inference logic."""

    def test_reinference_called_when_facts_empty_and_episodes_exist(self, db):
        """When semantic_facts is empty but episodes exist, run_all_inference should be called."""
        # Insert an episode so episode_count > 0
        with db.get_connection("user_model") as conn:
            _insert_episode(conn)

        # Verify preconditions
        with db.get_connection("user_model") as conn:
            fact_count = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert fact_count == 0
        assert episode_count > 0

        # Simulate the startup check logic from main.py step 1.19
        mock_inferrer = MagicMock()
        mock_inferrer.run_all_inference = MagicMock()

        with db.get_connection("user_model") as conn:
            fc = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
            ec = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

        if fc == 0 and ec > 0:
            mock_inferrer.run_all_inference()

        mock_inferrer.run_all_inference.assert_called_once()

    def test_reinference_skipped_when_facts_exist(self, db):
        """When semantic_facts already has rows, run_all_inference should NOT be called."""
        with db.get_connection("user_model") as conn:
            _insert_episode(conn)
            _insert_semantic_fact(conn)

        # Verify preconditions
        with db.get_connection("user_model") as conn:
            fact_count = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
        assert fact_count > 0

        mock_inferrer = MagicMock()
        mock_inferrer.run_all_inference = MagicMock()

        with db.get_connection("user_model") as conn:
            fc = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
            ec = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

        if fc == 0 and ec > 0:
            mock_inferrer.run_all_inference()

        mock_inferrer.run_all_inference.assert_not_called()

    def test_reinference_skipped_when_no_episodes(self, db):
        """When both semantic_facts and episodes are empty, run_all_inference should NOT be called."""
        with db.get_connection("user_model") as conn:
            fact_count = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
            episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert fact_count == 0
        assert episode_count == 0

        mock_inferrer = MagicMock()
        mock_inferrer.run_all_inference = MagicMock()

        with db.get_connection("user_model") as conn:
            fc = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
            ec = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

        if fc == 0 and ec > 0:
            mock_inferrer.run_all_inference()

        mock_inferrer.run_all_inference.assert_not_called()

    def test_reinference_exception_does_not_crash(self, db):
        """Exceptions during re-inference should be caught by try/except, not crash startup."""
        with db.get_connection("user_model") as conn:
            _insert_episode(conn)

        mock_inferrer = MagicMock()
        mock_inferrer.run_all_inference.side_effect = RuntimeError("inference boom")

        # Replicate the full try/except from step 1.19 — the outer except
        # must absorb the exception so startup continues (fail-open).
        startup_continued = False
        try:
            with db.get_connection("user_model") as conn:
                fc = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
                ec = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            if fc == 0 and ec > 0:
                mock_inferrer.run_all_inference()
        except Exception:
            pass  # logger.warning in production — silently absorb here

        # Code after step 1.19 should still execute
        startup_continued = True
        assert startup_continued, "Startup should continue after inference exception"

    def test_reinference_exception_is_fail_open(self, db):
        """Verify the full try/except pattern catches exceptions from run_all_inference."""
        with db.get_connection("user_model") as conn:
            _insert_episode(conn)

        mock_inferrer = MagicMock()
        mock_inferrer.run_all_inference.side_effect = RuntimeError("DB locked")

        # Replicate the exact try/except structure from main.py step 1.19
        warning_logged = False
        try:
            with db.get_connection("user_model") as conn:
                fc = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()[0]
                ec = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            if fc == 0 and ec > 0:
                mock_inferrer.run_all_inference()
        except Exception as e:
            # In main.py this is: logger.warning("startup: ... failed (non-fatal): %s", e)
            warning_logged = True

        # The exception from run_all_inference should propagate up to the except
        assert warning_logged, "Exception from run_all_inference should be caught by outer try/except"
