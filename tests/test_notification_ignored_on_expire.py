"""
Tests for notification.ignored feedback when stale notifications expire.

Covers:
- expire_stale_notifications returns (count, list_of_ids) tuple
- _log_automatic_feedback is called for each expired notification with feedback_type='ignored'
- _publish_notification_ignored_events publishes notification.ignored events on the bus
- get_digest() calls _publish_notification_ignored_events after expiry
- Non-pending notifications (delivered, dismissed) are NOT included in ignored feedback
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


def _insert_notification(db, notif_id, status="pending", hours_ago=72):
    """Helper to insert a notification with a backdated created_at timestamp."""
    created_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, priority, status, created_at)
               VALUES (?, ?, 'normal', ?, ?)""",
            (notif_id, f"Notification {notif_id}", status, created_at),
        )


# ============================================================================
# Return type: (count, list_of_ids)
# ============================================================================


class TestExpireReturnType:
    """Tests that expire_stale_notifications returns a (count, ids) tuple."""

    def test_returns_tuple_with_count_and_ids(self, notification_manager, db):
        """Return value should be a tuple of (int, list[str])."""
        _insert_notification(db, "notif-a", hours_ago=72)
        _insert_notification(db, "notif-b", hours_ago=73)

        result = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert isinstance(result, tuple)
        assert len(result) == 2
        count, ids = result
        assert count == 2
        assert set(ids) == {"notif-a", "notif-b"}

    def test_returns_zero_count_and_empty_list_when_nothing_expired(self, notification_manager, db):
        """When no notifications are expired, return (0, [])."""
        _insert_notification(db, "notif-recent", hours_ago=1)

        count, ids = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert count == 0
        assert ids == []

    def test_returns_zero_tuple_on_error(self, notification_manager, db):
        """On exception, return (0, []) to fail-open."""
        # Force an error by closing the DB connection pool
        with patch.object(notification_manager.db, "get_connection", side_effect=Exception("DB error")):
            count, ids = notification_manager.expire_stale_notifications()

        assert count == 0
        assert ids == []


# ============================================================================
# Feedback logging for expired notifications
# ============================================================================


class TestIgnoredFeedbackLogging:
    """Tests that _log_automatic_feedback is called for each expired notification."""

    def test_feedback_logged_for_each_expired_notification(self, notification_manager, db):
        """Each expired notification should get an 'ignored' feedback entry."""
        _insert_notification(db, "notif-x", hours_ago=72)
        _insert_notification(db, "notif-y", hours_ago=96)

        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            notification_manager.expire_stale_notifications(max_age_hours=48)

        assert mock_feedback.call_count == 2

        # Verify call arguments for each notification
        call_args_list = [call.kwargs for call in mock_feedback.call_args_list]
        notif_ids_called = {args["action_id"] for args in call_args_list}
        assert notif_ids_called == {"notif-x", "notif-y"}

        for args in call_args_list:
            assert args["action_type"] == "notification"
            assert args["feedback_type"] == "ignored"
            assert args["context"]["explicit_user_action"] is False
            assert args["context"]["action"] == "expired_ignored"

    def test_feedback_written_to_preferences_db(self, notification_manager, db):
        """Feedback entries should actually be written to the feedback_log table."""
        _insert_notification(db, "notif-z", hours_ago=72)

        notification_manager.expire_stale_notifications(max_age_hours=48)

        with db.get_connection("preferences") as conn:
            rows = conn.execute(
                """SELECT action_id, action_type, feedback_type, context
                   FROM feedback_log
                   WHERE action_id = 'notif-z'"""
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row["action_type"] == "notification"
        assert row["feedback_type"] == "ignored"
        context = json.loads(row["context"])
        assert context["explicit_user_action"] is False
        assert context["action"] == "expired_ignored"

    def test_no_feedback_when_nothing_expires(self, notification_manager, db):
        """No feedback should be logged when no notifications are expired."""
        _insert_notification(db, "notif-fresh", hours_ago=1)

        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            notification_manager.expire_stale_notifications(max_age_hours=48)

        mock_feedback.assert_not_called()

    def test_non_pending_notifications_excluded_from_feedback(self, notification_manager, db):
        """Only pending notifications should trigger ignored feedback, not delivered/dismissed."""
        _insert_notification(db, "notif-delivered", status="delivered", hours_ago=72)
        _insert_notification(db, "notif-dismissed", status="dismissed", hours_ago=72)
        _insert_notification(db, "notif-pending", status="pending", hours_ago=72)

        with patch.object(notification_manager, "_log_automatic_feedback") as mock_feedback:
            count, ids = notification_manager.expire_stale_notifications(max_age_hours=48)

        assert count == 1
        assert ids == ["notif-pending"]
        mock_feedback.assert_called_once()
        assert mock_feedback.call_args.kwargs["action_id"] == "notif-pending"


# ============================================================================
# Bus event publishing for expired notifications
# ============================================================================


class TestPublishNotificationIgnoredEvents:
    """Tests that _publish_notification_ignored_events publishes bus events."""

    @pytest.mark.asyncio
    async def test_publishes_event_for_each_notification(self, notification_manager, mock_event_bus):
        """Should publish a notification.ignored event for each notification ID."""
        await notification_manager._publish_notification_ignored_events(["id-1", "id-2", "id-3"])

        assert mock_event_bus.publish.call_count == 3

        # Verify each call used the correct subject and payload
        for i, call in enumerate(mock_event_bus.publish.call_args_list):
            args, kwargs = call
            assert args[0] == "notification.ignored"
            assert args[1]["notification_id"] in ["id-1", "id-2", "id-3"]
            assert kwargs["source"] == "notification_manager"

    @pytest.mark.asyncio
    async def test_skips_when_bus_not_connected(self, notification_manager, mock_event_bus):
        """Should not publish when bus is not connected."""
        mock_event_bus.is_connected = False

        await notification_manager._publish_notification_ignored_events(["id-1"])

        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_bus_is_none(self, db):
        """Should not raise when bus is None."""
        nm = NotificationManager(db, None, config={}, timezone="UTC")

        # Should not raise
        await nm._publish_notification_ignored_events(["id-1"])

    @pytest.mark.asyncio
    async def test_continues_on_publish_error(self, notification_manager, mock_event_bus):
        """Should continue publishing remaining events even if one fails."""
        # Make the second call fail
        mock_event_bus.publish.side_effect = [None, Exception("Bus error"), None]

        await notification_manager._publish_notification_ignored_events(["id-1", "id-2", "id-3"])

        # All 3 should have been attempted
        assert mock_event_bus.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_list_does_nothing(self, notification_manager, mock_event_bus):
        """Should not publish anything when given an empty list."""
        await notification_manager._publish_notification_ignored_events([])

        mock_event_bus.publish.assert_not_called()


# ============================================================================
# get_digest() integration with notification.ignored events
# ============================================================================


class TestGetDigestPublishesIgnoredEvents:
    """Tests that get_digest() publishes notification.ignored events after expiry."""

    @pytest.mark.asyncio
    async def test_get_digest_publishes_ignored_events_for_expired(self, notification_manager, db, mock_event_bus):
        """get_digest should publish notification.ignored events for expired notifications."""
        _insert_notification(db, "notif-stale", hours_ago=72)

        await notification_manager.get_digest()

        # Find the notification.ignored publish call among all bus publishes
        ignored_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if call.args[0] == "notification.ignored"
        ]
        assert len(ignored_calls) == 1
        assert ignored_calls[0].args[1]["notification_id"] == "notif-stale"

    @pytest.mark.asyncio
    async def test_get_digest_no_ignored_events_when_nothing_expired(self, notification_manager, db, mock_event_bus):
        """get_digest should not publish ignored events when nothing was expired."""
        _insert_notification(db, "notif-fresh", hours_ago=1)

        await notification_manager.get_digest()

        ignored_calls = [
            call for call in mock_event_bus.publish.call_args_list
            if call.args[0] == "notification.ignored"
        ]
        assert len(ignored_calls) == 0
