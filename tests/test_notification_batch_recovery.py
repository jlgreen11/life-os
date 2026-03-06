"""
Tests for notification batch recovery across service restarts.

Verifies that get_digest() picks up DB-persisted pending notifications
even when the in-memory _pending_batch is empty (simulating a restart
where _recover_pending_batch failed or was incomplete).
"""

import uuid
from datetime import datetime, timezone

import pytest

from services.notification_manager.manager import NotificationManager


def _insert_pending_notification(db, notif_id=None, title="Test", body="Body",
                                  priority="normal", domain=None,
                                  source_event_id=None, action_url=None):
    """Insert a pending notification directly into the DB (bypassing the manager)."""
    notif_id = notif_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, domain, source_event_id, action_url, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (notif_id, title, body, priority, domain, source_event_id, action_url, now),
        )
    return notif_id


@pytest.mark.asyncio
async def test_get_digest_recovers_db_pending_when_batch_empty(db, event_bus):
    """get_digest() returns DB-persisted pending notifications even when
    _pending_batch is empty (simulating a restart where recovery was skipped)."""
    # Insert pending notifications directly into DB
    id1 = _insert_pending_notification(db, title="Meeting reminder")
    id2 = _insert_pending_notification(db, title="Email summary")

    # Create manager with empty _pending_batch (simulating failed recovery)
    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    # The init recovery may have loaded these — clear it to simulate a fresh state
    mgr._pending_batch = []

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert id1 in digest_ids, "DB-persisted notification should appear in digest"
    assert id2 in digest_ids, "DB-persisted notification should appear in digest"


@pytest.mark.asyncio
async def test_recovered_notifications_marked_delivered(db, event_bus):
    """Notifications recovered from DB are marked as 'delivered' after get_digest()."""
    id1 = _insert_pending_notification(db, title="Recovered item")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    mgr._pending_batch = []

    await mgr.get_digest()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?", (id1,)
        ).fetchone()
    assert row["status"] == "delivered"


@pytest.mark.asyncio
async def test_no_duplicates_between_memory_and_db(db, event_bus):
    """Items in _pending_batch are not duplicated by the DB recovery query."""
    notif_id = _insert_pending_notification(db, title="In both places")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    # Simulate the notification being in both _pending_batch and DB
    mgr._pending_batch = [{
        "id": notif_id,
        "title": "In both places",
        "body": "Body",
        "priority": "normal",
        "domain": None,
        "source_event_id": None,
        "action_url": None,
    }]

    digest = await mgr.get_digest()
    ids = [item["id"] for item in digest]

    assert ids.count(notif_id) == 1, "Notification should appear exactly once in digest"


@pytest.mark.asyncio
async def test_prediction_surfacing_for_recovered_items(db, event_bus):
    """Prediction notifications recovered from DB trigger _mark_prediction_surfaced."""
    pred_id = "pred-" + str(uuid.uuid4())
    notif_id = _insert_pending_notification(
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
    mgr._pending_batch = []

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
async def test_init_recovery_loads_pending_from_db(db, event_bus):
    """_recover_pending_batch on init loads DB-persisted pending notifications."""
    id1 = _insert_pending_notification(db, title="Pre-existing", priority="normal")
    id2 = _insert_pending_notification(db, title="Also pre-existing", priority="low")

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")

    batch_ids = {item["id"] for item in mgr._pending_batch}
    assert id1 in batch_ids
    assert id2 in batch_ids


@pytest.mark.asyncio
async def test_delivered_notifications_not_recovered(db, event_bus):
    """Notifications already delivered should not appear in digest recovery."""
    notif_id = _insert_pending_notification(db, title="Already delivered")
    # Mark it as delivered
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE notifications SET status = 'delivered' WHERE id = ?",
            (notif_id,),
        )

    mgr = NotificationManager(db, event_bus, config={}, timezone="UTC")
    mgr._pending_batch = []

    digest = await mgr.get_digest()
    digest_ids = {item["id"] for item in digest}

    assert notif_id not in digest_ids, "Already-delivered notification should not reappear"
