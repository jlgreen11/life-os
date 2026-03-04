"""Tests for notification expiry — verifies that expire_stale_notifications()
correctly marks old pending notifications as expired while preserving recent
and non-pending notifications.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from services.notification_manager.manager import NotificationManager


def _insert_notification(db, *, notification_id=None, status="pending", hours_ago=72):
    """Insert a notification with a specific age and status for testing.

    Args:
        db: DatabaseManager instance.
        notification_id: Custom ID, or auto-generated UUID.
        status: Notification status ('pending', 'delivered', 'read', etc.).
        hours_ago: How many hours in the past to set created_at.
    """
    nid = notification_id or str(uuid.uuid4())
    created_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (nid, f"Test notification {nid[:8]}", "Test body", "normal", status, created_at),
        )
    return nid


def _make_notification_manager(db):
    """Create a NotificationManager with minimal dependencies for testing."""
    bus = AsyncMock()
    bus.is_connected = False
    config = {}
    return NotificationManager(db=db, event_bus=bus, config=config)


def test_expire_stale_notifications_clears_old_pending(db):
    """Pending notifications older than 48h should be marked 'expired'."""
    nm = _make_notification_manager(db)

    # Insert 3 old pending notifications (72 hours old)
    old_ids = [_insert_notification(db, hours_ago=72) for _ in range(3)]

    count, expired_ids = nm.expire_stale_notifications()

    assert count == 3
    assert set(expired_ids) == set(old_ids)

    # Verify status changed in the database
    with db.get_connection("state") as conn:
        for nid in old_ids:
            row = conn.execute(
                "SELECT status FROM notifications WHERE id = ?", (nid,)
            ).fetchone()
            assert row["status"] == "expired"


def test_expire_preserves_recent_pending(db):
    """Pending notifications less than 48h old should NOT be expired."""
    nm = _make_notification_manager(db)

    recent_id = _insert_notification(db, hours_ago=1)

    count, expired_ids = nm.expire_stale_notifications()

    assert count == 0
    assert expired_ids == []

    # Verify status is still 'pending'
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (recent_id,)
        ).fetchone()
        assert row["status"] == "pending"


def test_expire_ignores_delivered_and_read(db):
    """Only 'pending' notifications should be expired — delivered/read are untouched."""
    nm = _make_notification_manager(db)

    delivered_id = _insert_notification(db, status="delivered", hours_ago=72)
    read_id = _insert_notification(db, status="read", hours_ago=72)

    count, expired_ids = nm.expire_stale_notifications()

    assert count == 0
    assert expired_ids == []

    # Verify statuses unchanged
    with db.get_connection("state") as conn:
        row_d = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (delivered_id,)
        ).fetchone()
        assert row_d["status"] == "delivered"

        row_r = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (read_id,)
        ).fetchone()
        assert row_r["status"] == "read"


def test_expire_logs_feedback(db):
    """Expired notifications should generate 'ignored' feedback entries."""
    nm = _make_notification_manager(db)

    old_id = _insert_notification(db, hours_ago=72)

    count, expired_ids = nm.expire_stale_notifications()
    assert count == 1

    # Check that feedback was logged in the preferences database
    with db.get_connection("preferences") as conn:
        rows = conn.execute(
            "SELECT action_id, action_type, feedback_type FROM feedback_log WHERE action_id = ?",
            (old_id,),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["action_type"] == "notification"
    assert rows[0]["feedback_type"] == "ignored"
