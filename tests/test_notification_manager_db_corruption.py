"""
Tests for NotificationManager resilience under user_model.db corruption.

Verifies that the 4 user_model.db access points in NotificationManager
handle database corruption gracefully using fail-open error handling:

1. _mark_prediction_surfaced() — logs warning, notification delivery proceeds
2. _update_linked_prediction() — logs warning, mark_acted_on/dismiss still succeed
3. auto_resolve_stale_predictions() — continues processing remaining items
4. auto_resolve_filtered_predictions() — returns 0 and logs warning

Uses the same corruption simulation pattern as test_semantic_inferrer_db_resilience.py:
patches db.get_connection to raise sqlite3.OperationalError for the 'user_model'
database while allowing other databases to work normally.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.notification_manager.manager import NotificationManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_event_bus():
    """Mock event bus with is_connected and publish."""
    bus = MagicMock()
    bus.is_connected = True
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def notification_manager(db, mock_event_bus):
    """Create a NotificationManager with test database."""
    return NotificationManager(db, mock_event_bus, config={}, timezone="UTC")


@pytest.fixture
def create_prediction(db):
    """Helper to create a prediction in the user_model database."""
    def _create(prediction_id: str, was_surfaced: int = 0):
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    "reminder",
                    "Test prediction",
                    0.75,
                    "SUGGEST",
                    was_surfaced,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    return _create


@pytest.fixture
def create_prediction_notification(db):
    """Helper to create a prediction-linked notification in the state database."""
    def _create(notif_id: str, prediction_id: str, status: str = "delivered",
                delivered_at: str | None = None):
        now = delivered_at or datetime.now(timezone.utc).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Test notification", "body", "normal",
                 prediction_id, "prediction", status, now),
            )
    return _create


@pytest.fixture
def corrupt_user_model(db):
    """Context manager that makes user_model DB raise OperationalError.

    Patches db.get_connection so that calls with 'user_model' raise
    sqlite3.OperationalError('database disk image is malformed'),
    while calls with any other database name work normally.
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
# Test: _mark_prediction_surfaced() fail-open
# ---------------------------------------------------------------------------

class TestMarkPredictionSurfacedCorruption:
    """Verify _mark_prediction_surfaced() handles DB corruption gracefully."""

    def test_logs_warning_but_does_not_raise(self, notification_manager, db, corrupt_user_model, caplog):
        """_mark_prediction_surfaced() should log a warning but not raise
        when user_model.db is corrupt."""
        # Seed a prediction so there's a valid target
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("pred-surf-1", "reminder", "test", 0.75, "SUGGEST", 0,
                 datetime.now(timezone.utc).isoformat()),
            )

        # Now corrupt the user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            # Must NOT raise
            notification_manager._mark_prediction_surfaced("pred-surf-1")

        # Should have logged a warning
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1
        assert "Failed to mark prediction" in warning_records[0].message
        assert "pred-surf-1" in warning_records[0].message

    async def test_notification_delivery_still_succeeds_with_corrupt_user_model(
        self, notification_manager, db, corrupt_user_model
    ):
        """When _mark_prediction_surfaced fails due to corruption, the
        notification should still be delivered successfully."""
        # Set notification mode to frequent so normal priority is immediate
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
                ("notification_mode", "frequent"),
            )

        # Create a prediction first (before corruption)
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("pred-deliver-1", "reminder", "test", 0.75, "SUGGEST", 0,
                 datetime.now(timezone.utc).isoformat()),
            )

        # Now corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        # Notification creation should still succeed
        notif_id = await notification_manager.create_notification(
            title="Prediction alert",
            priority="high",
            source_event_id="pred-deliver-1",
            domain="prediction",
        )

        assert notif_id is not None


# ---------------------------------------------------------------------------
# Test: _update_linked_prediction() fail-open
# ---------------------------------------------------------------------------

class TestUpdateLinkedPredictionCorruption:
    """Verify _update_linked_prediction() handles DB corruption gracefully."""

    def test_logs_warning_but_does_not_raise(
        self, notification_manager, db, create_prediction,
        create_prediction_notification, corrupt_user_model, caplog
    ):
        """_update_linked_prediction() should log a warning but not raise
        when user_model.db is corrupt."""
        create_prediction("pred-link-1")
        create_prediction_notification("notif-link-1", "pred-link-1")

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            # Must NOT raise
            notification_manager._update_linked_prediction("notif-link-1", was_accurate=True)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1
        assert "Failed to update linked prediction" in warning_records[0].message
        assert "notif-link-1" in warning_records[0].message

    async def test_mark_acted_on_succeeds_with_corrupt_user_model(
        self, notification_manager, db, create_prediction,
        create_prediction_notification, corrupt_user_model
    ):
        """mark_acted_on() should complete successfully even when
        _update_linked_prediction fails due to user_model.db corruption.

        The notification status should still be updated to 'acted_on' in the
        state DB, and feedback should still be logged in preferences DB.
        """
        create_prediction("pred-act-1")
        create_prediction_notification("notif-act-1", "pred-act-1")

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        # mark_acted_on should NOT raise
        await notification_manager.mark_acted_on("notif-act-1")

        # Verify notification status was updated in state DB (which is not corrupt)
        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status FROM notifications WHERE id = ?", ("notif-act-1",)
            ).fetchone()
        assert row["status"] == "acted_on"

    async def test_dismiss_succeeds_with_corrupt_user_model(
        self, notification_manager, db, create_prediction,
        create_prediction_notification, corrupt_user_model
    ):
        """dismiss() should complete successfully even when
        _update_linked_prediction fails due to user_model.db corruption.

        The notification status should still be updated to 'dismissed' in the
        state DB, and feedback should still be logged in preferences DB.
        """
        create_prediction("pred-dis-1")
        create_prediction_notification("notif-dis-1", "pred-dis-1")

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        # dismiss should NOT raise
        await notification_manager.dismiss("notif-dis-1")

        # Verify notification status was updated in state DB
        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status FROM notifications WHERE id = ?", ("notif-dis-1",)
            ).fetchone()
        assert row["status"] == "dismissed"


# ---------------------------------------------------------------------------
# Test: auto_resolve_stale_predictions() fail-open
# ---------------------------------------------------------------------------

class TestAutoResolveStaleCorruption:
    """Verify auto_resolve_stale_predictions() handles DB corruption gracefully."""

    async def test_continues_processing_after_corruption(
        self, notification_manager, db, create_prediction, corrupt_user_model, caplog
    ):
        """When user_model.db is corrupt, auto_resolve_stale_predictions() should
        skip the failed prediction update but continue processing remaining items.

        The _mark_status and _log_automatic_feedback calls use state/preferences
        DBs and should still work for each item.
        """
        # Create 3 stale prediction notifications
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        for i in range(3):
            pred_id = f"pred-stale-{i}"
            notif_id = f"notif-stale-{i}"
            create_prediction(pred_id, was_surfaced=1)
            with db.get_connection("state") as conn:
                conn.execute(
                    """INSERT INTO notifications
                       (id, title, body, priority, source_event_id, domain, status, delivered_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (notif_id, "Stale prediction", "body", "normal",
                     pred_id, "prediction", "delivered", old_time),
                )

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            # Must NOT raise
            resolved = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        # No predictions should have been resolved (all user_model writes failed)
        assert resolved == 0

        # Should have logged warnings for each failed prediction
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "Failed to auto-resolve stale prediction" in r.message
        ]
        assert len(warning_records) == 3

        # But the notification statuses should still be updated (state DB works)
        for i in range(3):
            with db.get_connection("state") as conn:
                row = conn.execute(
                    "SELECT status FROM notifications WHERE id = ?",
                    (f"notif-stale-{i}",),
                ).fetchone()
            assert row["status"] == "expired", (
                f"Notification notif-stale-{i} should be expired even when user_model.db is corrupt"
            )

    async def test_feedback_still_logged_when_prediction_update_fails(
        self, notification_manager, db, create_prediction, corrupt_user_model
    ):
        """Even when the prediction update in user_model.db fails,
        _log_automatic_feedback should still write to the preferences DB."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        create_prediction("pred-fb-1", was_surfaced=1)
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("notif-fb-1", "Stale", "body", "normal",
                 "pred-fb-1", "prediction", "delivered", old_time),
            )

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        # Feedback should still be logged in the preferences DB
        with db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_log WHERE action_id = ?",
                ("notif-fb-1",),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["feedback_type"] == "dismissed"


# ---------------------------------------------------------------------------
# Test: auto_resolve_filtered_predictions() fail-open
# ---------------------------------------------------------------------------

class TestAutoResolveFilteredCorruption:
    """Verify auto_resolve_filtered_predictions() handles DB corruption gracefully."""

    def test_returns_zero_and_logs_warning(
        self, notification_manager, db, create_prediction, corrupt_user_model, caplog
    ):
        """auto_resolve_filtered_predictions() should return 0 and log a warning
        when user_model.db is corrupt."""
        # Create a filtered prediction (was_surfaced=0)
        create_prediction("pred-filt-1", was_surfaced=0)

        # Corrupt user_model DB
        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            result = notification_manager.auto_resolve_filtered_predictions(timeout_hours=0)

        assert result == 0

        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "Failed to auto-resolve filtered predictions" in r.message
        ]
        assert len(warning_records) == 1

    def test_does_not_raise_with_database_error(
        self, notification_manager, db, create_prediction
    ):
        """auto_resolve_filtered_predictions() should handle sqlite3.DatabaseError
        (parent class of OperationalError) without raising."""
        create_prediction("pred-filt-2", was_surfaced=0)

        original_get_connection = db.get_connection

        @contextmanager
        def raise_database_error(db_name):
            if db_name == "user_model":
                raise sqlite3.DatabaseError("database corruption")
            with original_get_connection(db_name) as conn:
                yield conn

        notification_manager.db.get_connection = raise_database_error

        # Must NOT raise
        result = notification_manager.auto_resolve_filtered_predictions(timeout_hours=0)
        assert result == 0


# ---------------------------------------------------------------------------
# Test: error logging includes traceback info
# ---------------------------------------------------------------------------

class TestCorruptionErrorLogging:
    """Verify that corruption errors are logged with exc_info for debugging."""

    def test_mark_prediction_surfaced_logs_with_exc_info(
        self, notification_manager, corrupt_user_model, caplog
    ):
        """_mark_prediction_surfaced should log with exc_info=True for traceback."""
        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            notification_manager._mark_prediction_surfaced("pred-trace-1")

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1
        # logger.warning with exc_info=True sets exc_info on the record
        records_with_traceback = [r for r in warning_records if r.exc_info]
        assert len(records_with_traceback) >= 1, (
            "Expected warning record with exc_info for post-incident diagnosis"
        )

    def test_update_linked_prediction_logs_with_exc_info(
        self, notification_manager, db, create_prediction,
        create_prediction_notification, corrupt_user_model, caplog
    ):
        """_update_linked_prediction should log with exc_info=True for traceback."""
        create_prediction("pred-trace-2")
        create_prediction_notification("notif-trace-2", "pred-trace-2")

        notification_manager.db.get_connection = corrupt_user_model

        with caplog.at_level(logging.WARNING, logger="services.notification_manager.manager"):
            notification_manager._update_linked_prediction("notif-trace-2", was_accurate=True)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        records_with_traceback = [r for r in warning_records if r.exc_info]
        assert len(records_with_traceback) >= 1
