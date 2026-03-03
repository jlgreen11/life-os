"""
Tests for notification deduplication in NotificationManager.

Verifies that create_notification() suppresses duplicate notifications
based on source_event_id and title+domain within a 10-minute window,
returning the existing notification ID instead of creating a new row.
"""

import pytest

from services.notification_manager.manager import NotificationManager


@pytest.fixture()
def notif_mgr(db, event_bus):
    """A NotificationManager wired to the temporary DatabaseManager and mock event bus."""
    return NotificationManager(db, event_bus, config={}, timezone="UTC")


async def test_duplicate_title_domain_suppressed(notif_mgr, db):
    """Creating two notifications with the same title and domain should only insert one row."""
    await notif_mgr.create_notification(title="Meeting in 5 min", domain="calendar")
    await notif_mgr.create_notification(title="Meeting in 5 min", domain="calendar")

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 1


async def test_different_title_not_suppressed(notif_mgr, db):
    """Notifications with different titles should not be deduplicated."""
    await notif_mgr.create_notification(title="Meeting in 5 min", domain="calendar")
    await notif_mgr.create_notification(title="Lunch reminder", domain="calendar")

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 2


async def test_same_source_event_id_suppressed(notif_mgr, db):
    """Two notifications with the same source_event_id should be deduplicated."""
    await notif_mgr.create_notification(
        title="Prediction: rain", source_event_id="evt-1", domain="prediction"
    )
    await notif_mgr.create_notification(
        title="Prediction: rain", source_event_id="evt-1", domain="prediction"
    )

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 1


async def test_dedup_returns_existing_id(notif_mgr):
    """The deduplicated call should return the ID of the first notification."""
    first_id = await notif_mgr.create_notification(
        title="Duplicate test", domain="test"
    )
    second_id = await notif_mgr.create_notification(
        title="Duplicate test", domain="test"
    )

    assert first_id is not None
    assert second_id is not None
    assert first_id == second_id


async def test_old_duplicate_not_suppressed(notif_mgr, db):
    """Notifications older than 10 minutes should not trigger deduplication."""
    # Create the first notification
    first_id = await notif_mgr.create_notification(
        title="Stale alert", domain="system"
    )

    # Backdate the first notification's created_at by 15 minutes so it falls
    # outside the 10-minute dedup window.
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE notifications SET created_at = datetime('now', '-15 minutes') WHERE id = ?",
            (first_id,),
        )

    # Create a second notification with the same title+domain — should NOT be suppressed
    second_id = await notif_mgr.create_notification(
        title="Stale alert", domain="system"
    )

    assert second_id != first_id

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 2


async def test_source_event_dedup_only_pending(notif_mgr, db):
    """source_event_id dedup only applies to pending notifications.

    If the existing notification has been delivered (no longer pending),
    a new notification with the same source_event_id should be created
    (assuming the title+domain dedup window has also passed).
    """
    first_id = await notif_mgr.create_notification(
        title="Prediction: rain", source_event_id="evt-2", domain="prediction"
    )

    # Mark the first notification as delivered (no longer pending)
    notif_mgr._mark_status(first_id, "delivered")

    # Backdate so the title+domain 10-minute window also expires
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE notifications SET created_at = datetime('now', '-15 minutes') WHERE id = ?",
            (first_id,),
        )

    # A second notification with the same source_event_id should now be created
    second_id = await notif_mgr.create_notification(
        title="Prediction: rain", source_event_id="evt-2", domain="prediction"
    )

    assert second_id != first_id

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 2


async def test_different_domain_not_suppressed(notif_mgr, db):
    """Same title but different domain should not be deduplicated."""
    await notif_mgr.create_notification(title="Alert", domain="email")
    await notif_mgr.create_notification(title="Alert", domain="calendar")

    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM notifications").fetchone()["cnt"]

    assert count == 2
