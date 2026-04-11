"""
Tests for notification batch durability via DB-backed 'batched' status.

Previously batch-routed notifications were held in an in-memory list
(_pending_batch) that was lost on every server restart, causing them to
silently expire.  The fix stores batch-routed notifications with
status='batched' in the SQLite state DB so they survive restarts.

Tests verify:
- Batch-routed notifications get status='batched' in the DB (not 'pending').
- get_digest() returns batched notifications read from the DB.
- Batched notifications survive a simulated restart (new NotificationManager).
- auto_deliver_stale_batch() delivers old batched notifications.
- expire_stale_notifications() expires old batched notifications.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.notification_manager.manager import NotificationManager

# ============================================================================
# Helpers
# ============================================================================


def _make_mgr(db, event_bus, *, notification_mode: str = "batched") -> NotificationManager:
    """Return a NotificationManager with the given notification mode pre-set.

    'batched' mode causes normal/low notifications to be batch-routed;
    other modes can be passed for specific test scenarios.
    """
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    # Set the user preference so _decide_delivery() routes to 'batch'.
    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value) VALUES (?, ?)",
            ("notification_mode", notification_mode),
        )
    return mgr


def _insert_notification(db, *, notif_id=None, title="Test", body="Body",
                          priority="normal", domain=None, source_event_id=None,
                          status="batched", hours_ago=0):
    """Insert a notification directly into the DB with full timestamp control.

    Args:
        db: DatabaseManager instance.
        notif_id: Optional notification ID (auto-generated if omitted).
        title: Notification title.
        body: Notification body text.
        priority: Priority level (critical/high/normal/low).
        domain: Notification domain (e.g., 'prediction').
        source_event_id: ID linking back to the source event/prediction.
        status: Notification status (batched/pending/delivered/expired/etc.).
        hours_ago: How many hours in the past to set created_at.

    Returns:
        The notification ID string.
    """
    notif_id = notif_id or str(uuid.uuid4())
    created_at = (
        datetime.now(UTC) - timedelta(hours=hours_ago)
    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, source_event_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, title, body, priority, domain, source_event_id, status, created_at),
        )
    return notif_id


def _get_status(db, notif_id: str) -> str | None:
    """Return the current DB status of a notification, or None if not found."""
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (notif_id,)
        ).fetchone()
    return row["status"] if row else None


# ============================================================================
# Tests: create_notification() stores 'batched' status
# ============================================================================


@pytest.mark.asyncio
async def test_batch_routed_notification_stored_as_batched(db, event_bus):
    """A notification routed to batch delivery gets status='batched' in the DB.

    This is the core invariant of the fix: batch-routed notifications must be
    durable — stored in the DB rather than an in-memory list — so they survive
    server restarts.
    """
    mgr = _make_mgr(db, event_bus, notification_mode="batched")

    notif_id = await mgr.create_notification(
        title="Weekly summary",
        body="Here's what happened this week.",
        priority="normal",
    )

    assert notif_id is not None, "create_notification should return a notification ID"
    assert _get_status(db, notif_id) == "batched", (
        "Batch-routed notification must be stored with status='batched', not 'pending'"
    )


@pytest.mark.asyncio
async def test_batch_routed_low_priority_stored_as_batched(db, event_bus):
    """Low-priority notifications also get status='batched' under 'frequent' mode."""
    mgr = _make_mgr(db, event_bus, notification_mode="frequent")

    notif_id = await mgr.create_notification(
        title="Low-priority update",
        body="Something minor happened.",
        priority="low",
    )

    assert notif_id is not None
    assert _get_status(db, notif_id) == "batched", (
        "Low-priority notification in 'frequent' mode should be stored as 'batched'"
    )


@pytest.mark.asyncio
async def test_immediate_notification_not_stored_as_batched(db, event_bus):
    """Critical/high notifications are delivered immediately, never 'batched'."""
    mgr = _make_mgr(db, event_bus, notification_mode="batched")

    notif_id = await mgr.create_notification(
        title="URGENT: System alert",
        body="Something critical happened.",
        priority="critical",
    )

    assert notif_id is not None
    status = _get_status(db, notif_id)
    assert status != "batched", (
        f"Critical notification should never be stored as 'batched' (got '{status}')"
    )


# ============================================================================
# Tests: get_digest() reads from DB
# ============================================================================


@pytest.mark.asyncio
async def test_get_digest_returns_batched_from_db(db, event_bus):
    """get_digest() delivers notifications stored with status='batched' in the DB."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    id1 = _insert_notification(db, title="Batch item 1", status="batched")
    id2 = _insert_notification(db, title="Batch item 2", status="batched")

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert id1 in digest_ids, "Batched notification 1 should appear in digest"
    assert id2 in digest_ids, "Batched notification 2 should appear in digest"


@pytest.mark.asyncio
async def test_get_digest_marks_batched_as_delivered(db, event_bus):
    """Notifications returned by get_digest() are marked 'delivered' in the DB."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    notif_id = _insert_notification(db, title="Will be delivered", status="batched")

    await mgr.get_digest()

    assert _get_status(db, notif_id) == "delivered", (
        "Notification should be marked 'delivered' after appearing in a digest"
    )


@pytest.mark.asyncio
async def test_get_digest_excludes_pending_notifications(db, event_bus):
    """get_digest() only returns 'batched' items, not 'pending' (immediate-queue)."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    batched_id = _insert_notification(db, title="Batch", status="batched")
    pending_id = _insert_notification(db, title="Pending (immediate)", status="pending")

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert batched_id in digest_ids, "Batched notification should appear in digest"
    assert pending_id not in digest_ids, (
        "Pending (immediate-queue) notification should not appear in digest"
    )


@pytest.mark.asyncio
async def test_get_digest_empty_when_no_batched(db, event_bus):
    """get_digest() returns an empty list when there are no batched notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()

    assert digest == [], "Digest should be empty when no batched notifications exist"


@pytest.mark.asyncio
async def test_get_digest_prediction_surfacing(db, event_bus):
    """get_digest() marks linked predictions as surfaced (was_surfaced=1)."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    pred_id = f"pred-{uuid.uuid4()}"
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, 'need', 'test prediction', 0.5, 'suggest', ?)""",
            (pred_id, datetime.now(UTC).isoformat()),
        )

    _insert_notification(
        db, title="Prediction notice", domain="prediction",
        source_event_id=pred_id, status="batched",
    )

    await mgr.get_digest()

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row is not None
    assert row["was_surfaced"] == 1, "Prediction should be marked as surfaced after digest delivery"


# ============================================================================
# Tests: restart durability
# ============================================================================


@pytest.mark.asyncio
async def test_batched_notifications_survive_restart(db, event_bus):
    """Batched notifications stored in the DB survive a simulated server restart.

    A new NotificationManager instance (simulating a fresh process after
    restart) must still deliver notifications that were batched before the
    restart, because they are in the DB with status='batched'.
    """
    # First manager: create a batch-routed notification
    mgr1 = _make_mgr(db, event_bus, notification_mode="batched")
    notif_id = await mgr1.create_notification(
        title="Pre-restart batch item",
        body="Created before the restart.",
        priority="normal",
    )
    assert notif_id is not None
    assert _get_status(db, notif_id) == "batched", "Should be stored as 'batched'"

    # Simulate restart: create a brand-new manager instance (no in-memory state)
    mgr2 = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr2.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert notif_id in digest_ids, (
        "Batched notification should survive restart and appear in the next digest"
    )
    assert _get_status(db, notif_id) == "delivered", (
        "Notification should be marked 'delivered' after post-restart digest"
    )


# ============================================================================
# Tests: auto_deliver_stale_batch()
# ============================================================================


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_delivers_old_batched(db, event_bus):
    """auto_deliver_stale_batch() delivers old batched notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    notif_id = _insert_notification(
        db, title="Old batched item", status="batched", hours_ago=8,
    )

    delivered = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered == 1, "One stale batched notification should have been delivered"
    assert _get_status(db, notif_id) == "delivered"


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_skips_recent_batched(db, event_bus):
    """auto_deliver_stale_batch() does not deliver recently-batched notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    notif_id = _insert_notification(
        db, title="Recent batched item", status="batched", hours_ago=2,
    )

    delivered = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered == 0, "Recent batched notification should not be auto-delivered"
    assert _get_status(db, notif_id) == "batched"


@pytest.mark.asyncio
async def test_auto_deliver_stale_batch_handles_both_statuses(db, event_bus):
    """auto_deliver_stale_batch() processes both 'pending' and 'batched' notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    pending_id = _insert_notification(
        db, title="Old pending", status="pending", hours_ago=8,
    )
    batched_id = _insert_notification(
        db, title="Old batched", status="batched", hours_ago=8,
    )

    delivered = await mgr.auto_deliver_stale_batch(max_pending_hours=6)

    assert delivered == 2, "Both pending and batched stale notifications should be delivered"
    assert _get_status(db, pending_id) == "delivered"
    assert _get_status(db, batched_id) == "delivered"


# ============================================================================
# Tests: expire_stale_notifications()
# ============================================================================


def test_expire_stale_notifications_expires_old_batched(db, event_bus):
    """expire_stale_notifications() expires old 'batched' notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    notif_id = _insert_notification(
        db, title="Old batched", status="batched", hours_ago=50,
    )

    expired_count, expired_ids = mgr.expire_stale_notifications(max_age_hours=48)

    assert expired_count >= 1, "At least one batched notification should be expired"
    assert notif_id in expired_ids, "Old batched notification should be in expired IDs"
    assert _get_status(db, notif_id) == "expired"


def test_expire_stale_notifications_skips_recent_batched(db, event_bus):
    """expire_stale_notifications() does not expire recently-batched notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    notif_id = _insert_notification(
        db, title="Recent batched", status="batched", hours_ago=2,
    )

    mgr.expire_stale_notifications(max_age_hours=48)

    assert _get_status(db, notif_id) == "batched", (
        "Recent batched notification should not be expired"
    )


def test_expire_stale_notifications_handles_both_statuses(db, event_bus):
    """expire_stale_notifications() expires both old 'pending' and 'batched' notifications."""
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    old_pending_id = _insert_notification(
        db, title="Old pending", status="pending", hours_ago=50,
    )
    old_batched_id = _insert_notification(
        db, title="Old batched", status="batched", hours_ago=50,
    )

    expired_count, expired_ids = mgr.expire_stale_notifications(max_age_hours=48)

    assert expired_count >= 2
    assert old_pending_id in expired_ids
    assert old_batched_id in expired_ids
    assert _get_status(db, old_pending_id) == "expired"
    assert _get_status(db, old_batched_id) == "expired"
