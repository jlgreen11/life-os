"""
Tests for NotificationManager.auto_deliver_stale_batch().

This method is the safety net that auto-delivers batched notifications
before they expire. Without it, notifications that sit in 'pending' status
(because the user hasn't checked the dashboard) would silently expire
after 48 hours, meaning the system's intelligence never reaches the user.

Tests cover:
- Delivering old pending notifications past the threshold
- Skipping recent pending notifications within the threshold
- Marking linked predictions as surfaced on auto-delivery
- Ignoring non-pending notifications (delivered, expired)
- Respecting custom threshold values
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.notification_manager.manager import NotificationManager


# ============================================================================
# Helpers
# ============================================================================


def _insert_notification(db, *, notif_id=None, title="Test", body="Body",
                         priority="normal", domain=None, source_event_id=None,
                         status="pending", hours_ago=0):
    """Insert a notification with a created_at timestamp offset by hours_ago.

    Args:
        db: DatabaseManager instance.
        notif_id: Optional notification ID (auto-generated if omitted).
        title: Notification title.
        body: Notification body text.
        priority: Priority level (critical/high/normal/low).
        domain: Notification domain (e.g., 'prediction').
        source_event_id: ID linking back to the source event/prediction.
        status: Notification status (pending/delivered/expired/etc.).
        hours_ago: How many hours in the past to set created_at.

    Returns:
        The notification ID.
    """
    notif_id = notif_id or str(uuid.uuid4())
    created_at = (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, source_event_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, title, body, priority, domain, source_event_id, status, created_at),
        )
    return notif_id


def _get_notification_status(db, notif_id):
    """Read the current status of a notification from the DB."""
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()
    return row["status"] if row else None


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_delivers_old_pending(db, event_bus):
    """Pending notifications older than the threshold are auto-delivered."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Insert 3 notifications created 8 hours ago (beyond default 6-hour threshold)
    ids = [
        _insert_notification(db, title=f"Old notification {i}", hours_ago=8)
        for i in range(3)
    ]

    delivered_count = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered_count == 3, "All 3 old pending notifications should be delivered"
    for nid in ids:
        assert _get_notification_status(db, nid) == "delivered", (
            f"Notification {nid} should have status 'delivered'"
        )


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_skips_recent_pending(db, event_bus):
    """Pending notifications within the threshold are not delivered."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Insert 2 notifications created 2 hours ago (within 6-hour threshold)
    ids = [
        _insert_notification(db, title=f"Recent notification {i}", hours_ago=2)
        for i in range(2)
    ]

    delivered_count = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered_count == 0, "No recent notifications should be delivered"
    for nid in ids:
        assert _get_notification_status(db, nid) == "pending", (
            f"Notification {nid} should remain 'pending'"
        )


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_marks_prediction_surfaced(db, event_bus):
    """Auto-delivering a prediction notification marks the prediction as surfaced.

    This tests the critical integration between notification delivery and
    prediction accuracy tracking. If a prediction notification is auto-delivered,
    the linked prediction must be marked was_surfaced=1 so the accuracy
    feedback loop can include it in calculations.
    """
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    pred_id = f"pred-{uuid.uuid4()}"

    # Insert a prediction row with was_surfaced=0
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, 'need', 'test prediction', 0.5, 'suggest', ?)""",
            (pred_id, datetime.now(timezone.utc).isoformat()),
        )

    # Insert a prediction notification created 8 hours ago
    notif_id = _insert_notification(
        db, title="Prediction alert", domain="prediction",
        source_event_id=pred_id, hours_ago=8,
    )

    delivered_count = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered_count == 1
    assert _get_notification_status(db, notif_id) == "delivered"

    # Verify prediction was marked as surfaced
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row is not None, "Prediction row should exist"
    assert row["was_surfaced"] == 1, "Prediction should be marked as surfaced after auto-delivery"


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_skips_non_pending(db, event_bus):
    """Only notifications with status='pending' are auto-delivered."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Insert old notifications with non-pending statuses
    delivered_id = _insert_notification(
        db, title="Already delivered", status="delivered", hours_ago=8,
    )
    expired_id = _insert_notification(
        db, title="Already expired", status="expired", hours_ago=8,
    )

    delivered_count = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered_count == 0, "Non-pending notifications should not be processed"
    assert _get_notification_status(db, delivered_id) == "delivered"
    assert _get_notification_status(db, expired_id) == "expired"


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_custom_threshold(db, event_bus):
    """The max_pending_hours threshold is respected with custom values."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Insert a notification created 3 hours ago
    notif_id = _insert_notification(db, title="3h old", hours_ago=3)

    # With a 4-hour threshold, the notification should NOT be delivered
    count_skip = await mgr.auto_deliver_stale_batch(max_pending_hours=4)
    assert count_skip == 0, "3h-old notification should not be delivered with 4h threshold"
    assert _get_notification_status(db, notif_id) == "pending"

    # With a 2-hour threshold, the notification SHOULD be delivered
    count_deliver = await mgr.auto_deliver_stale_batch(max_pending_hours=2)
    assert count_deliver == 1, "3h-old notification should be delivered with 2h threshold"
    assert _get_notification_status(db, notif_id) == "delivered"
