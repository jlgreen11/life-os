"""
Tests for auto-delivery of stale batched notifications.

Validates that NotificationManager.auto_deliver_stale_batch() proactively
delivers pending notifications that have been waiting too long, preventing
them from silently expiring at the 48-hour mark.
"""

import uuid
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
    config = {}
    return NotificationManager(db, mock_event_bus, config, timezone="UTC")


def _insert_notification(db, *, notif_id=None, title="Test", body="Body",
                         priority="normal", status="pending", domain=None,
                         source_event_id=None, hours_ago=0):
    """Insert a notification with a specific age into the state database."""
    notif_id = notif_id or str(uuid.uuid4())
    created_at = (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO notifications (id, title, body, priority, status, domain, "
            "source_event_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (notif_id, title, body, priority, status, domain, source_event_id, created_at),
        )
    return notif_id


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_delivers_old_pending(db, notification_manager):
    """Notifications older than max_pending_hours are delivered; newer ones stay pending."""
    recent_id = _insert_notification(db, title="Recent", hours_ago=1)
    old_id = _insert_notification(db, title="Old", hours_ago=7)
    very_old_id = _insert_notification(db, title="Very old", hours_ago=25)

    count = await notification_manager.auto_deliver_stale_batch(max_pending_hours=6)

    assert count == 2

    with db.get_connection("state") as conn:
        recent = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (recent_id,)
        ).fetchone()
        old = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (old_id,)
        ).fetchone()
        very_old = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (very_old_id,)
        ).fetchone()

    assert recent["status"] == "pending"
    assert old["status"] == "delivered"
    assert very_old["status"] == "delivered"


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_handles_empty(db, notification_manager):
    """Returns 0 with no errors when there are no pending notifications."""
    count = await notification_manager.auto_deliver_stale_batch(max_pending_hours=6)
    assert count == 0


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_marks_prediction_surfaced(db, notification_manager):
    """Prediction-domain notifications have their prediction marked as surfaced."""
    pred_id = str(uuid.uuid4())

    # Create a predictions table entry in user_model.db for the surfacing check.
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO predictions "
            "(id, prediction_type, description, confidence, confidence_gate, was_surfaced) "
            "VALUES (?, 'NEED', 'test prediction', 0.5, 'SUGGEST', 0)",
            (pred_id,),
        )

    _insert_notification(
        db, title="Prediction alert", hours_ago=8,
        domain="prediction", source_event_id=pred_id,
    )

    count = await notification_manager.auto_deliver_stale_batch(max_pending_hours=6)
    assert count == 1

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()

    assert row is not None
    assert row["was_surfaced"] == 1


@pytest.mark.asyncio
async def test_auto_deliver_does_not_touch_delivered_or_expired(db, notification_manager):
    """Already-delivered and expired notifications are not re-delivered."""
    delivered_id = _insert_notification(
        db, title="Already delivered", hours_ago=10, status="delivered"
    )
    expired_id = _insert_notification(
        db, title="Already expired", hours_ago=10, status="expired"
    )

    count = await notification_manager.auto_deliver_stale_batch(max_pending_hours=6)
    assert count == 0

    with db.get_connection("state") as conn:
        delivered = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (delivered_id,)
        ).fetchone()
        expired = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (expired_id,)
        ).fetchone()

    assert delivered["status"] == "delivered"
    assert expired["status"] == "expired"


@pytest.mark.asyncio
async def test_auto_deliver_clears_in_memory_batch(db, notification_manager):
    """Auto-delivered notifications are removed from the in-memory _pending_batch."""
    old_id = _insert_notification(db, title="Old batched", hours_ago=8)

    # Simulate the notification being in the in-memory batch.
    notification_manager._pending_batch = [
        {"id": old_id, "title": "Old batched", "body": "Body", "priority": "normal"},
        {"id": "keep-me", "title": "Fresh", "body": "Body", "priority": "normal"},
    ]

    await notification_manager.auto_deliver_stale_batch(max_pending_hours=6)

    remaining_ids = [item["id"] for item in notification_manager._pending_batch]
    assert old_id not in remaining_ids
    assert "keep-me" in remaining_ids
