"""
Tests for notification batch durability across service restarts.

With the DB-backed 'batched' status, get_digest() reads directly from the
database rather than an in-memory list, so there is nothing to recover after
a restart — batched notifications simply remain in the DB with
status='batched' until the next digest window.

These tests verify the DB-backed approach is correct end-to-end.
"""

import uuid
from datetime import datetime, timezone

import pytest

from services.notification_manager.manager import NotificationManager


def _insert_batched_notification(db, notif_id=None, title="Test", body="Body",
                                  priority="normal", domain=None,
                                  source_event_id=None, action_url=None):
    """Insert a batched notification directly into the DB (bypassing the manager).

    Uses status='batched' to match how create_notification() stores batch-routed
    notifications after the in-memory list was replaced with a DB-backed status.
    """
    notif_id = notif_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, source_event_id, action_url, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'batched', ?)""",
            (notif_id, title, body, priority, domain, source_event_id, action_url, now),
        )
    return notif_id


@pytest.mark.asyncio
async def test_get_digest_returns_db_batched_notifications(db, event_bus):
    """get_digest() returns DB-persisted batched notifications.

    The new DB-backed design reads status='batched' directly from the DB,
    so there is no in-memory list to lose or recover.
    """
    id1 = _insert_batched_notification(db, title="Meeting reminder")
    id2 = _insert_batched_notification(db, title="Email summary")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert id1 in digest_ids, "DB-persisted batched notification should appear in digest"
    assert id2 in digest_ids, "DB-persisted batched notification should appear in digest"


@pytest.mark.asyncio
async def test_batched_notifications_marked_delivered(db, event_bus):
    """Notifications delivered via get_digest() are marked as 'delivered'."""
    id1 = _insert_batched_notification(db, title="Recovered item")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    await mgr.get_digest()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (id1,)
        ).fetchone()
    assert row["status"] == "delivered"


@pytest.mark.asyncio
async def test_no_duplicate_batched_notifications_in_digest(db, event_bus):
    """Each batched notification appears exactly once in the digest."""
    notif_id = _insert_batched_notification(db, title="Single notification")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()
    ids = [item["id"] for item in digest]

    assert ids.count(notif_id) == 1, "Notification should appear exactly once in digest"


@pytest.mark.asyncio
async def test_prediction_surfacing_for_batched_items(db, event_bus):
    """Prediction notifications delivered via get_digest() trigger _mark_prediction_surfaced."""
    pred_id = "pred-" + str(uuid.uuid4())
    notif_id = _insert_batched_notification(
        db, title="Prediction alert", domain="prediction",
        source_event_id=pred_id,
    )

    # Insert a prediction row so _mark_prediction_surfaced has something to update
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR IGNORE INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, 'need', 'test prediction', 0.5, 'suggest', ?)""",
            (pred_id, datetime.now(timezone.utc).isoformat()),
        )

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()

    # Verify the notification was delivered
    assert any(item["id"] == notif_id for item in digest)

    # Verify prediction was marked as surfaced
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row is not None, "Prediction row should exist"
    assert row["was_surfaced"] == 1, "Prediction should be marked as surfaced"


@pytest.mark.asyncio
async def test_new_manager_instance_delivers_pre_existing_batched(db, event_bus):
    """A new manager instance delivers notifications batched before it was created.

    With the DB-backed design, batched notifications persist across manager
    restarts automatically — no recovery step needed.
    """
    id1 = _insert_batched_notification(db, title="Pre-existing", priority="normal")
    id2 = _insert_batched_notification(db, title="Also pre-existing", priority="low")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert id1 in digest_ids, "Pre-existing batched notification should appear in digest"
    assert id2 in digest_ids, "Pre-existing batched notification should appear in digest"


@pytest.mark.asyncio
async def test_delivered_notifications_not_included_in_digest(db, event_bus):
    """Notifications already delivered should not appear in digest."""
    notif_id = _insert_batched_notification(db, title="Already delivered")
    # Mark it as delivered (simulating a previous digest call)
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE notifications SET status = 'delivered' WHERE id = ?",
            (notif_id,),
        )

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert notif_id not in digest_ids, "Already-delivered notification should not reappear"
