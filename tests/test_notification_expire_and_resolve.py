"""
Tests for NotificationManager timestamp format and auto-resolve race condition fixes.

Covers:
- expire_stale_notifications uses correct timestamp format matching SQLite's
  strftime('%Y-%m-%dT%H:%M:%fZ', 'now') where %f = SS.SSS
- auto_resolve_stale_predictions does NOT log spurious feedback when the
  prediction was already resolved by user action
- auto_resolve_stale_predictions DOES log feedback for genuinely stale predictions
- auto_resolve_stale_predictions always marks notifications as expired regardless
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.notification_manager.manager import NotificationManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_event_bus():
    """Mock event bus with is_connected and publish."""
    bus = MagicMock()
    bus.is_connected = True
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def notification_manager(db, mock_event_bus):
    """Create a NotificationManager instance with test database."""
    return NotificationManager(db, mock_event_bus, config={}, timezone="UTC")


@pytest.fixture
def create_prediction(db):
    """Helper to create a prediction in the user_model database."""
    def _create(prediction_id: str, was_surfaced: int = 0, resolved_at=None,
                was_accurate=None, user_response=None):
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, created_at, resolved_at, was_accurate, user_response)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    "reminder",
                    "Test prediction",
                    0.75,
                    "SUGGEST",
                    was_surfaced,
                    datetime.now(timezone.utc).isoformat(),
                    resolved_at,
                    was_accurate,
                    user_response,
                ),
            )
    return _create


@pytest.fixture
def create_delivered_prediction_notification(db, mock_event_bus):
    """Helper to create a delivered prediction notification with a backdated timestamp.

    Inserts directly into the DB to avoid side effects from create_notification
    (dedup checks, delivery routing, etc.).
    """
    def _create(notif_id: str, prediction_id: str, hours_ago: int = 25):
        delivered_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'delivered', ?)""",
                (notif_id, "Test Prediction", "body", "normal", prediction_id,
                 "prediction", delivered_at),
            )
    return _create


# ============================================================================
# Bug 1: Timestamp format in expire_stale_notifications
# ============================================================================


class TestExpireTimestampFormat:
    """Tests that expire_stale_notifications uses the correct timestamp format.

    The DB uses SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now') where %f = SS.SSS,
    producing timestamps like '2026-03-01T05:46:28.123Z'. The Python cutoff must
    use the same format for string comparison to work correctly.
    """

    def test_expire_correctly_expires_old_notification(self, notification_manager, db):
        """A notification older than max_age_hours should be expired."""
        notif_id = "notif-old"
        # Insert a notification with a created_at 3 days ago in DB format (SS.SSS)
        old_time = (datetime.now(timezone.utc) - timedelta(days=3))
        created_at = old_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{old_time.microsecond // 1000:03d}Z"

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, priority, status, created_at)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (notif_id, "Old Notification", "normal", created_at),
            )

        expired = notification_manager.expire_stale_notifications(max_age_hours=48)
        assert expired == 1

        with db.get_connection("state") as conn:
            row = conn.execute("SELECT status FROM notifications WHERE id = ?", (notif_id,)).fetchone()
            assert row["status"] == "expired"

    def test_expire_does_not_expire_recent_notification(self, notification_manager, db):
        """A notification newer than max_age_hours should NOT be expired."""
        notif_id = "notif-recent"
        # Insert a notification created 1 hour ago in DB format (SS.SSS)
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1))
        created_at = recent_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{recent_time.microsecond // 1000:03d}Z"

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, priority, status, created_at)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (notif_id, "Recent Notification", "normal", created_at),
            )

        expired = notification_manager.expire_stale_notifications(max_age_hours=48)
        assert expired == 0

        with db.get_connection("state") as conn:
            row = conn.execute("SELECT status FROM notifications WHERE id = ?", (notif_id,)).fetchone()
            assert row["status"] == "pending"

    def test_expire_boundary_with_zero_seconds(self, notification_manager, db):
        """Regression: notifications at second=0 must still be expired correctly.

        The old format '%Y-%m-%dT%H:%M:%fZ' (Python) produced cutoff strings like
        '2026-03-01T05:46:000000Z' (no seconds, 6-digit microseconds). A notification
        at '2026-03-01T05:46:00.500Z' would compare '00.500Z' < '000000Z' incorrectly
        (because '.' < '0' in ASCII), preventing it from being expired.
        """
        notif_id = "notif-zero-sec"
        # Create a notification 3 days ago where seconds = 0
        old_time = datetime.now(timezone.utc) - timedelta(days=3)
        # Force seconds=0, microseconds=500000 to trigger the old bug
        old_time = old_time.replace(second=0, microsecond=500000)
        created_at = old_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{old_time.microsecond // 1000:03d}Z"
        # e.g., "2026-02-28T12:30:00.500Z"

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, priority, status, created_at)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (notif_id, "Zero Second Notification", "normal", created_at),
            )

        expired = notification_manager.expire_stale_notifications(max_age_hours=48)
        assert expired == 1, "Notification at second=0 boundary should be expired"

    def test_expire_only_affects_pending_status(self, notification_manager, db):
        """Only pending notifications should be expired, not delivered/read ones."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=3))
        created_at = old_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, priority, status, created_at)
                   VALUES (?, ?, ?, 'delivered', ?)""",
                ("notif-delivered", "Delivered", "normal", created_at),
            )
            conn.execute(
                """INSERT INTO notifications
                   (id, title, priority, status, created_at)
                   VALUES (?, ?, ?, 'pending', ?)""",
                ("notif-pending", "Pending", "normal", created_at),
            )

        expired = notification_manager.expire_stale_notifications(max_age_hours=48)
        assert expired == 1  # Only the pending one


# ============================================================================
# Bug 2: auto_resolve_stale_predictions race condition
# ============================================================================


class TestAutoResolveRaceCondition:
    """Tests that auto_resolve_stale_predictions handles already-resolved predictions.

    When a prediction is already resolved (e.g., user acted on it), the UPDATE
    statement matches zero rows. In that case, _log_automatic_feedback must NOT
    be called, because logging a spurious 'dismissed' entry would falsely depress
    the prediction accuracy multiplier.
    """

    @pytest.mark.asyncio
    async def test_no_feedback_logged_when_prediction_already_resolved(
        self, notification_manager, db, create_prediction,
        create_delivered_prediction_notification,
    ):
        """Feedback should NOT be logged when the prediction was already resolved by user."""
        prediction_id = "pred-already-resolved"
        notif_id = "notif-already-resolved"

        # Create prediction that is already resolved (user acted on it)
        create_prediction(
            prediction_id,
            was_surfaced=1,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            was_accurate=1,
            user_response="acted_on",
        )

        # Create a stale delivered notification for it
        create_delivered_prediction_notification(notif_id, prediction_id, hours_ago=25)

        # Spy on _log_automatic_feedback
        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        # The prediction was already resolved, so resolved_count should be 0
        assert resolved_count == 0

        # CRITICAL: feedback should NOT have been logged
        mock_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_logged_for_genuinely_stale_prediction(
        self, notification_manager, db, create_prediction,
        create_delivered_prediction_notification,
    ):
        """Feedback SHOULD be logged when the prediction is genuinely stale (unresolved)."""
        prediction_id = "pred-genuinely-stale"
        notif_id = "notif-genuinely-stale"

        # Create an unresolved prediction
        create_prediction(prediction_id, was_surfaced=1)

        # Create a stale delivered notification for it
        create_delivered_prediction_notification(notif_id, prediction_id, hours_ago=25)

        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        assert resolved_count == 1

        # Feedback SHOULD have been logged for the genuinely stale prediction
        mock_feedback.assert_called_once()
        call_kwargs = mock_feedback.call_args
        assert call_kwargs[1]["feedback_type"] == "dismissed"
        assert call_kwargs[1]["context"]["auto_resolved"] is True

    @pytest.mark.asyncio
    async def test_notification_always_marked_expired(
        self, notification_manager, db, create_prediction,
        create_delivered_prediction_notification,
    ):
        """Notification should be marked expired even when prediction was already resolved."""
        prediction_id = "pred-pre-resolved"
        notif_id = "notif-pre-resolved"

        # Create already-resolved prediction
        create_prediction(
            prediction_id,
            was_surfaced=1,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            was_accurate=1,
            user_response="acted_on",
        )

        # Create stale delivered notification
        create_delivered_prediction_notification(notif_id, prediction_id, hours_ago=25)

        await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        # Notification should still be expired even though prediction was already resolved
        with db.get_connection("state") as conn:
            row = conn.execute("SELECT status FROM notifications WHERE id = ?", (notif_id,)).fetchone()
            assert row["status"] == "expired"

    @pytest.mark.asyncio
    async def test_resolved_count_correct_with_mixed_predictions(
        self, notification_manager, db, create_prediction,
        create_delivered_prediction_notification,
    ):
        """resolved_count should only count genuinely newly-resolved predictions."""
        # One already-resolved prediction
        create_prediction(
            "pred-resolved",
            was_surfaced=1,
            resolved_at=datetime.now(timezone.utc).isoformat(),
            was_accurate=1,
            user_response="acted_on",
        )
        create_delivered_prediction_notification("notif-resolved", "pred-resolved", hours_ago=25)

        # Two genuinely stale predictions
        create_prediction("pred-stale-1", was_surfaced=1)
        create_delivered_prediction_notification("notif-stale-1", "pred-stale-1", hours_ago=25)

        create_prediction("pred-stale-2", was_surfaced=1)
        create_delivered_prediction_notification("notif-stale-2", "pred-stale-2", hours_ago=25)

        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

        # Only the 2 genuinely stale predictions should be counted
        assert resolved_count == 2

        # Feedback should only be logged twice (not 3 times)
        assert mock_feedback.call_count == 2

        # All 3 notifications should be expired
        with db.get_connection("state") as conn:
            for nid in ["notif-resolved", "notif-stale-1", "notif-stale-2"]:
                row = conn.execute("SELECT status FROM notifications WHERE id = ?", (nid,)).fetchone()
                assert row["status"] == "expired", f"Notification {nid} should be expired"
