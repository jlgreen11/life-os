"""
Tests for the graduated auto-delivery timing in NotificationManager.

Covers:
- High-priority notifications delivered after 30 min (high_priority_hours=0.5)
- Normal-priority notifications delivered after 60 min (max_pending_hours=1)
- Fresh notifications (< 30 min) are NOT auto-delivered regardless of priority
- delivery_health() returns accurate pipeline counts and delivery_rate
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.notification_manager.manager import NotificationManager


# ============================================================================
# Helpers
# ============================================================================


def _insert_notification(
    db,
    *,
    notif_id=None,
    title="Test",
    body="Body",
    priority="normal",
    domain=None,
    source_event_id=None,
    status="pending",
    minutes_ago=0,
):
    """Insert a notification with a created_at timestamp offset by minutes_ago.

    Args:
        db: DatabaseManager instance.
        notif_id: Optional notification ID (auto-generated if omitted).
        title: Notification title.
        body: Notification body text.
        priority: Priority level (critical/high/normal/low).
        domain: Notification domain (e.g., 'prediction').
        source_event_id: ID linking back to the source event/prediction.
        status: Notification status (pending/delivered/expired/etc.).
        minutes_ago: How many minutes in the past to set created_at.

    Returns:
        The notification ID string.
    """
    notif_id = notif_id or str(uuid.uuid4())
    created_at = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, source_event_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, title, body, priority, domain, source_event_id, status, created_at),
        )
    return notif_id


def _get_status(db, notif_id):
    """Return the current status of a notification from the DB."""
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()
    return row["status"] if row else None


def _make_mgr(db, event_bus):
    """Construct a NotificationManager with test dependencies."""
    return NotificationManager(db, event_bus, config={}, timezone="UTC")


# ============================================================================
# Graduated delivery: high/critical threshold (0.5 h = 30 min)
# ============================================================================


@pytest.mark.asyncio
async def test_high_priority_delivered_after_35_minutes(db, event_bus):
    """A high-priority notification pending for 35 minutes gets auto-delivered.

    35 minutes > high_priority_hours (0.5 h = 30 min), so it should be
    picked up and marked 'delivered'.
    """
    mgr = _make_mgr(db, event_bus)

    notif_id = _insert_notification(
        db, title="Urgent alert", priority="high", minutes_ago=35
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 1, "High-priority 35min-old notification should be auto-delivered"
    assert _get_status(db, notif_id) == "delivered"


@pytest.mark.asyncio
async def test_critical_priority_delivered_after_35_minutes(db, event_bus):
    """A critical-priority notification pending for 35 minutes gets auto-delivered.

    Critical notifications use the same high_priority_hours threshold as high.
    """
    mgr = _make_mgr(db, event_bus)

    notif_id = _insert_notification(
        db, title="Critical system alert", priority="critical", minutes_ago=35
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 1, "Critical-priority 35min-old notification should be auto-delivered"
    assert _get_status(db, notif_id) == "delivered"


# ============================================================================
# Graduated delivery: normal threshold (1 h = 60 min)
# ============================================================================


@pytest.mark.asyncio
async def test_normal_priority_delivered_after_65_minutes(db, event_bus):
    """A normal-priority notification pending for 65 minutes gets auto-delivered.

    65 minutes > max_pending_hours (1 h = 60 min), so it should be delivered.
    """
    mgr = _make_mgr(db, event_bus)

    notif_id = _insert_notification(
        db, title="Normal notification", priority="normal", minutes_ago=65
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 1, "Normal-priority 65min-old notification should be auto-delivered"
    assert _get_status(db, notif_id) == "delivered"


@pytest.mark.asyncio
async def test_low_priority_delivered_after_65_minutes(db, event_bus):
    """A low-priority notification pending for 65 minutes gets auto-delivered.

    Low uses the same normal threshold (max_pending_hours).
    """
    mgr = _make_mgr(db, event_bus)

    notif_id = _insert_notification(
        db, title="Low priority note", priority="low", minutes_ago=65
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 1, "Low-priority 65min-old notification should be auto-delivered"
    assert _get_status(db, notif_id) == "delivered"


# ============================================================================
# Too-fresh notifications: NOT auto-delivered
# ============================================================================


@pytest.mark.asyncio
async def test_fresh_notification_not_delivered_after_20_minutes(db, event_bus):
    """A notification pending for only 20 minutes is NOT auto-delivered.

    20 min < high_priority_hours (30 min), so even a high-priority notification
    should be left alone — it's too fresh.
    """
    mgr = _make_mgr(db, event_bus)

    # Even a high-priority notification that's only 20 min old should NOT be
    # auto-delivered — it hasn't reached either threshold yet.
    high_id = _insert_notification(
        db, title="Very recent high", priority="high", minutes_ago=20
    )
    normal_id = _insert_notification(
        db, title="Very recent normal", priority="normal", minutes_ago=20
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 0, "20-min-old notifications should NOT be auto-delivered"
    assert _get_status(db, high_id) == "pending", "High-priority 20min notification should stay pending"
    assert _get_status(db, normal_id) == "pending", "Normal 20min notification should stay pending"


@pytest.mark.asyncio
async def test_high_priority_not_delivered_if_within_high_threshold(db, event_bus):
    """A high-priority notification at 35 min is delivered, but a 25-min-old one is not.

    This verifies the threshold boundary precisely.
    """
    mgr = _make_mgr(db, event_bus)

    old_id = _insert_notification(
        db, title="Old high", priority="high", minutes_ago=35
    )
    fresh_id = _insert_notification(
        db, title="Fresh high", priority="high", minutes_ago=25
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 1, "Only the 35-min-old notification should be delivered"
    assert _get_status(db, old_id) == "delivered"
    assert _get_status(db, fresh_id) == "pending"


@pytest.mark.asyncio
async def test_normal_not_delivered_between_30_and_60_minutes(db, event_bus):
    """A normal notification at 45 min is NOT auto-delivered (too fresh for normal threshold).

    45 min > high_priority_hours (30 min) but < max_pending_hours (60 min).
    Normal/low notifications should only be delivered after max_pending_hours.
    """
    mgr = _make_mgr(db, event_bus)

    notif_id = _insert_notification(
        db, title="Normal in window", priority="normal", minutes_ago=45
    )

    delivered = await mgr.auto_deliver_stale_batch(
        max_pending_hours=1, high_priority_hours=0.5
    )

    assert delivered == 0, "Normal 45min notification should NOT be delivered (needs 60min)"
    assert _get_status(db, notif_id) == "pending"


# ============================================================================
# delivery_health() accuracy
# ============================================================================


def test_delivery_health_empty_database(db, event_bus):
    """delivery_health() returns all-zero counts on a fresh database."""
    mgr = _make_mgr(db, event_bus)

    health = mgr.delivery_health()

    assert health["total_created"] == 0
    assert health["delivered"] == 0
    assert health["expired"] == 0
    assert health["pending"] == 0
    assert health["other"] == 0
    assert health["delivery_rate"] == 0.0


def test_delivery_health_counts_all_statuses(db, event_bus):
    """delivery_health() counts each status bucket correctly.

    Insert 3 delivered, 2 expired, 1 pending, 1 batched, and 1 acted_on
    notification, then verify each field in the result.
    """
    mgr = _make_mgr(db, event_bus)

    # Insert notifications with varied statuses.
    for i in range(3):
        _insert_notification(db, title=f"Delivered {i}", status="delivered")
    for i in range(2):
        _insert_notification(db, title=f"Expired {i}", status="expired")
    _insert_notification(db, title="Pending", status="pending")
    _insert_notification(db, title="Batched", status="batched")
    _insert_notification(db, title="Acted on", status="acted_on")

    health = mgr.delivery_health()

    # Total = 3 + 2 + 1 + 1 + 1 = 8
    assert health["total_created"] == 8
    assert health["delivered"] == 3
    assert health["expired"] == 2
    # pending + batched
    assert health["pending"] == 2
    # acted_on
    assert health["other"] == 1
    # delivery_rate = 3/8 = 0.375
    assert abs(health["delivery_rate"] - 0.375) < 0.0001


def test_delivery_health_rate_is_zero_with_no_deliveries(db, event_bus):
    """delivery_health() returns delivery_rate=0.0 when nothing is delivered."""
    mgr = _make_mgr(db, event_bus)

    for i in range(5):
        _insert_notification(db, title=f"Expired {i}", status="expired")

    health = mgr.delivery_health()

    assert health["total_created"] == 5
    assert health["delivered"] == 0
    assert health["expired"] == 5
    assert health["delivery_rate"] == 0.0


def test_delivery_health_rate_is_one_when_all_delivered(db, event_bus):
    """delivery_health() returns delivery_rate=1.0 when everything is delivered."""
    mgr = _make_mgr(db, event_bus)

    for i in range(4):
        _insert_notification(db, title=f"Delivered {i}", status="delivered")

    health = mgr.delivery_health()

    assert health["total_created"] == 4
    assert health["delivered"] == 4
    assert health["expired"] == 0
    assert health["delivery_rate"] == 1.0
