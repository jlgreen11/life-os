"""
Tests for BehavioralAccuracyTracker resilience under user_model.db corruption.

Verifies that the BehavioralAccuracyTracker handles database corruption
gracefully using fail-open error handling, matching the resilience pattern
used in RoutineDetector, InsightEngine, and NotificationManager.

Coverage:
1. __init__ survives corrupted user_model.db (critical: prevents startup crash)
2. _ensure_resolution_reason_column() returns None on corruption
3. _backfill_automated_sender_tags() returns None on corruption
4. run_inference_cycle() returns empty stats dict on corruption

Uses the same corruption simulation pattern as
test_notification_manager_db_corruption.py: patches db.get_connection to raise
sqlite3.DatabaseError for the 'user_model' database while allowing other
databases to work normally.
"""

import logging
import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corrupt_user_model(db):
    """Context manager factory that makes user_model DB raise DatabaseError.

    Patches db.get_connection so that calls with 'user_model' raise
    sqlite3.DatabaseError('database disk image is malformed'),
    while calls with any other database name work normally.
    """
    original_get_connection = db.get_connection

    @contextmanager
    def corrupted_get_connection(db_name):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        with original_get_connection(db_name) as conn:
            yield conn

    return corrupted_get_connection


@pytest.fixture
def corrupt_user_model_operational(db):
    """Same as corrupt_user_model but raises OperationalError instead.

    OperationalError is a subclass of DatabaseError and is the specific
    exception SQLite raises for malformed database files.
    """
    original_get_connection = db.get_connection

    @contextmanager
    def corrupted_get_connection(db_name):
        if db_name == "user_model":
            raise sqlite3.OperationalError("database disk image is malformed")
        with original_get_connection(db_name) as conn:
            yield conn

    return corrupted_get_connection


# ---------------------------------------------------------------------------
# Test: __init__ survives corrupted user_model.db
# ---------------------------------------------------------------------------


class TestInitSurvivesCorruptedDb:
    """Verify BehavioralAccuracyTracker.__init__() does not crash when
    user_model.db is corrupted."""

    def test_init_survives_corrupted_user_model_db(self, db, corrupt_user_model, caplog):
        """Constructor should catch DatabaseError from schema migration and
        backfill methods and still produce a usable tracker instance."""
        db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING):
            # Must NOT raise — this is the critical startup-crash fix
            tracker = BehavioralAccuracyTracker(db)

        assert tracker is not None
        assert tracker.db is db

        # Should have logged warnings about the unavailable DB
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("unavailable" in msg or "user_model.db" in msg.lower() for msg in warning_messages), (
            f"Expected a warning about user_model.db being unavailable, got: {warning_messages}"
        )

    def test_init_survives_operational_error(self, db, corrupt_user_model_operational, caplog):
        """Constructor should also handle OperationalError (subclass of DatabaseError)."""
        db.get_connection = corrupt_user_model_operational

        with caplog.at_level(logging.WARNING):
            tracker = BehavioralAccuracyTracker(db)

        assert tracker is not None


# ---------------------------------------------------------------------------
# Test: _ensure_resolution_reason_column() handles corruption
# ---------------------------------------------------------------------------


class TestEnsureResolutionReasonColumnCorruption:
    """Verify _ensure_resolution_reason_column() handles DB corruption."""

    def test_ensure_resolution_reason_column_handles_corruption(self, db, corrupt_user_model, caplog):
        """Method should return None (not raise) when user_model.db is corrupted."""
        # First create the tracker normally (needs working DB for __init__)
        tracker = BehavioralAccuracyTracker(db)

        # Now corrupt the DB and call the method directly
        tracker.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING):
            result = tracker._ensure_resolution_reason_column()

        # Should return None, not raise
        assert result is None

        # Should have logged a warning
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("_ensure_resolution_reason_column" in msg for msg in warning_messages)


# ---------------------------------------------------------------------------
# Test: _backfill_automated_sender_tags() handles corruption
# ---------------------------------------------------------------------------


class TestBackfillAutomatedSenderTagsCorruption:
    """Verify _backfill_automated_sender_tags() handles DB corruption."""

    def test_backfill_automated_sender_tags_handles_corruption(self, db, corrupt_user_model, caplog):
        """Method should return None (not raise) when user_model.db is corrupted."""
        # First create the tracker normally
        tracker = BehavioralAccuracyTracker(db)

        # Now corrupt the DB and call the method directly
        tracker.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING):
            result = tracker._backfill_automated_sender_tags()

        # Should return None, not raise
        assert result is None

        # Should have logged a warning about the query phase failure
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("_backfill_automated_sender_tags" in msg for msg in warning_messages)

    def test_backfill_handles_corruption_during_update_phase(self, db, caplog):
        """Method should handle corruption that occurs during the update phase
        (second with-block) after the query phase succeeds."""
        from datetime import datetime, timezone

        tracker = BehavioralAccuracyTracker(db)

        # Insert a prediction that will be found by the query phase
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, resolution_reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "pred-backfill-corrupt-1",
                    "opportunity",
                    "It's been 45 days since you last contacted noreply@example.com",
                    0.5,
                    "SUGGEST",
                    1,
                    0,
                    datetime.now(timezone.utc).isoformat(),
                    None,  # resolution_reason is NULL — will be found by backfill
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        # Corrupt only after the first get_connection call succeeds (query phase)
        original_get_connection = db.get_connection
        call_count = 0

        @contextmanager
        def corrupt_on_second_call(db_name):
            nonlocal call_count
            if db_name == "user_model":
                call_count += 1
                if call_count > 1:
                    raise sqlite3.DatabaseError("database disk image is malformed")
            with original_get_connection(db_name) as conn:
                yield conn

        tracker.db.get_connection = corrupt_on_second_call

        with caplog.at_level(logging.WARNING):
            # Must NOT raise
            result = tracker._backfill_automated_sender_tags()

        assert result is None


# ---------------------------------------------------------------------------
# Test: run_inference_cycle() survives corrupted DB
# ---------------------------------------------------------------------------


class TestRunInferenceCycleCorruption:
    """Verify run_inference_cycle() handles DB corruption gracefully."""

    async def test_run_inference_cycle_survives_corrupted_db(self, db, corrupt_user_model, caplog):
        """run_inference_cycle() should return a stats dict (not raise) when
        user_model.db is corrupted during the surfaced predictions query."""
        # Create tracker with working DB first
        tracker = BehavioralAccuracyTracker(db)

        # Now corrupt the DB
        tracker.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING):
            stats = await tracker.run_inference_cycle()

        # Should return the default empty stats dict
        assert isinstance(stats, dict)
        assert stats["marked_accurate"] == 0
        assert stats["marked_inaccurate"] == 0
        assert stats["surfaced"] == 0
        assert stats["filtered"] == 0

        # Should have logged a warning
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("surfaced predictions" in msg for msg in warning_messages)

    async def test_run_inference_cycle_handles_operational_error(self, db, corrupt_user_model_operational, caplog):
        """run_inference_cycle() should handle OperationalError gracefully."""
        tracker = BehavioralAccuracyTracker(db)
        tracker.db.get_connection = corrupt_user_model_operational

        with caplog.at_level(logging.WARNING):
            stats = await tracker.run_inference_cycle()

        assert isinstance(stats, dict)
        assert stats["marked_accurate"] == 0

    async def test_run_inference_cycle_corruption_after_surfaced_query(self, db, caplog):
        """If corruption occurs after the surfaced SELECT succeeds (during the
        filtered SELECT), cycle should still return partial stats and not raise."""
        from datetime import datetime, timedelta, timezone

        tracker = BehavioralAccuracyTracker(db)

        # The surfaced query will find no predictions (empty table), so it
        # moves on to the filtered query. We corrupt on the second call.
        original_get_connection = db.get_connection
        call_count = 0

        @contextmanager
        def corrupt_on_second_user_model_call(db_name):
            nonlocal call_count
            if db_name == "user_model":
                call_count += 1
                if call_count > 1:
                    raise sqlite3.DatabaseError("database disk image is malformed")
            with original_get_connection(db_name) as conn:
                yield conn

        tracker.db.get_connection = corrupt_on_second_user_model_call

        with caplog.at_level(logging.WARNING):
            stats = await tracker.run_inference_cycle()

        assert isinstance(stats, dict)
        # Stats from the surfaced phase should still be present (all zeros since no data)
        assert "marked_accurate" in stats
        assert "filtered" in stats

    async def test_run_inference_cycle_update_corruption_skips_prediction(self, db, caplog):
        """If corruption occurs during a prediction UPDATE, the cycle should
        skip that prediction and continue, logging a warning."""
        from datetime import datetime, timedelta, timezone

        tracker = BehavioralAccuracyTracker(db)

        # Insert a surfaced prediction that will be found
        created_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, resolved_at, created_at, suggested_action, supporting_signals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "pred-update-corrupt-1",
                    "reminder",
                    "Reply to test@example.com",
                    0.5,
                    "SUGGEST",
                    1,
                    None,  # Not yet resolved
                    created_at,
                    "Send a reply",
                    '{"contact_email": "test@example.com"}',
                ),
            )

        # Insert a matching event so _infer_accuracy returns a result
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, payload)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "evt-reply-1",
                    "message.sent",
                    "test",
                    datetime.now(timezone.utc).isoformat(),
                    '{"to": "test@example.com", "body": "reply"}',
                ),
            )

        # Corrupt only on the UPDATE (second user_model call for surfaced loop)
        original_get_connection = db.get_connection
        call_count = 0

        @contextmanager
        def corrupt_on_update(db_name):
            nonlocal call_count
            if db_name == "user_model":
                call_count += 1
                if call_count > 1:
                    raise sqlite3.DatabaseError("database disk image is malformed")
            with original_get_connection(db_name) as conn:
                yield conn

        tracker.db.get_connection = corrupt_on_update

        with caplog.at_level(logging.WARNING):
            stats = await tracker.run_inference_cycle()

        # The cycle should complete (not raise) even though the UPDATE failed
        assert isinstance(stats, dict)
