"""
Tests for automatic resolution of stale prediction notifications.

This feature closes the feedback loop by marking ignored predictions as
inaccurate after a timeout period, allowing the prediction engine to learn
which prediction types are unhelpful and suppress them via confidence decay.
"""

import pytest
from datetime import datetime, timedelta, timezone
from services.notification_manager.manager import NotificationManager


@pytest.fixture
def notification_manager(db, event_bus):
    """Create a NotificationManager instance for testing."""
    return NotificationManager(db, event_bus, config={})


@pytest.mark.asyncio
async def test_auto_resolve_stale_predictions_basic(db, notification_manager):
    """
    Test basic auto-resolution: stale prediction notifications (>24h old,
    still "delivered") should be marked as inaccurate and expired.
    """
    # Create a prediction in the user_model DB
    prediction_id = "pred-123"
    now = datetime.now(timezone.utc)
    stale_time = (now - timedelta(hours=25)).isoformat()  # 25 hours ago

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Test reminder", 0.7, "DEFAULT",
             "24_hours", "Do something", 1),
        )

    # Create a notification linked to this prediction, delivered 25h ago
    notif_id = "notif-456"
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, "Reminder", "Test body", "normal", prediction_id, "prediction", "delivered", stale_time),
        )

    # Run auto-resolution with 24h timeout
    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    # Should resolve exactly 1 prediction
    assert resolved_count == 1, "Should auto-resolve 1 stale prediction"

    # Check notification status changed to "expired"
    with db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT status FROM notifications WHERE id = ?",
            (notif_id,),
        ).fetchone()
        assert notif["status"] == "expired", "Notification should be marked expired"

    # Check prediction was marked inaccurate with user_response='ignored'
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0, "Prediction should be marked inaccurate"
        assert pred["resolved_at"] is not None, "Resolved timestamp should be set"
        assert pred["user_response"] == "ignored", "User response should be 'ignored'"


@pytest.mark.asyncio
async def test_auto_resolve_respects_timeout(db, notification_manager):
    """
    Test that auto-resolution respects the timeout parameter: notifications
    delivered within the timeout window should NOT be resolved.
    """
    prediction_id = "pred-fresh"
    now = datetime.now(timezone.utc)
    recent_time = (now - timedelta(hours=12)).isoformat()  # Only 12h ago

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Fresh reminder", 0.7, "DEFAULT",
             "24_hours", "Do something", 1),
        )

    notif_id = "notif-fresh"
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, "Reminder", "Test body", "normal", prediction_id, "prediction", "delivered", recent_time),
        )

    # Run with 24h timeout — this 12h-old notification should NOT be resolved
    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    assert resolved_count == 0, "Should not resolve notifications within timeout window"

    # Verify prediction is still unresolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] is None, "Prediction should still be unresolved"
        assert pred["resolved_at"] is None, "No resolved timestamp yet"


@pytest.mark.asyncio
async def test_auto_resolve_skips_non_delivered_status(db, notification_manager):
    """
    Test that auto-resolution ONLY affects notifications in "delivered" status.
    Notifications that are "read", "acted_on", or "dismissed" should be skipped
    because they already have explicit user feedback.
    """
    now = datetime.now(timezone.utc)
    stale_time = (now - timedelta(hours=30)).isoformat()

    # Create 3 stale notifications with different statuses
    for idx, status in enumerate(["read", "acted_on", "dismissed"]):
        pred_id = f"pred-{status}"
        notif_id = f"notif-{status}"

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, was_surfaced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pred_id, "reminder", f"{status} reminder", 0.7, "DEFAULT",
                 "24_hours", "Do something", 1),
            )

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, "Reminder", "Test", "normal", pred_id, "prediction", status, stale_time),
            )

    # Run auto-resolution — should resolve 0 (all have non-delivered status)
    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    assert resolved_count == 0, "Should not resolve notifications with explicit user actions"


@pytest.mark.asyncio
async def test_auto_resolve_skips_non_prediction_domain(db, notification_manager):
    """
    Test that auto-resolution ONLY affects prediction-domain notifications.
    Other domains (email, calendar, etc.) should be skipped.
    """
    now = datetime.now(timezone.utc)
    stale_time = (now - timedelta(hours=30)).isoformat()

    # Create a stale notification with domain="email" (not "prediction")
    notif_id = "notif-email"
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, "Email alert", "Test", "normal", "event-123", "email", "delivered", stale_time),
        )

    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    assert resolved_count == 0, "Should not resolve non-prediction notifications"


@pytest.mark.asyncio
async def test_auto_resolve_skips_already_resolved(db, notification_manager):
    """
    Test that auto-resolution skips predictions that are already resolved.
    The WHERE clause includes `resolved_at IS NULL` to prevent double-resolving.
    """
    prediction_id = "pred-already-resolved"
    now = datetime.now(timezone.utc)
    stale_time = (now - timedelta(hours=30)).isoformat()

    # Create a prediction that's already resolved
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, was_surfaced, was_accurate, resolved_at, user_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Already resolved", 0.7, "DEFAULT",
             "24_hours", "Do something", 1, 1, now.isoformat(), "acted_on"),
        )

    notif_id = "notif-already-resolved"
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, "Reminder", "Test", "normal", prediction_id, "prediction", "delivered", stale_time),
        )

    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    # Should resolve 0 (prediction already has resolved_at timestamp)
    assert resolved_count == 0, "Should not re-resolve already-resolved predictions"


@pytest.mark.asyncio
async def test_auto_resolve_multiple_stale_predictions(db, notification_manager):
    """
    Test that auto-resolution correctly handles multiple stale predictions
    in a single pass.
    """
    now = datetime.now(timezone.utc)
    stale_time = (now - timedelta(hours=30)).isoformat()

    # Create 5 stale prediction notifications
    for i in range(5):
        pred_id = f"pred-multi-{i}"
        notif_id = f"notif-multi-{i}"

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    time_horizon, suggested_action, was_surfaced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (pred_id, "reminder", f"Reminder {i}", 0.7, "DEFAULT",
                 "24_hours", "Do something", 1),
            )

        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, delivered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (notif_id, f"Reminder {i}", "Test", "normal", pred_id, "prediction", "delivered", stale_time),
            )

    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    assert resolved_count == 5, "Should resolve all 5 stale predictions"

    # Verify all predictions marked inaccurate
    with db.get_connection("user_model") as conn:
        resolved = conn.execute(
            """SELECT COUNT(*) as cnt FROM predictions
               WHERE id LIKE 'pred-multi-%'
                 AND was_accurate = 0
                 AND resolved_at IS NOT NULL
                 AND user_response = 'ignored'"""
        ).fetchone()
        assert resolved["cnt"] == 5, "All 5 predictions should be marked inaccurate"


@pytest.mark.asyncio
async def test_auto_resolve_custom_timeout(db, notification_manager):
    """
    Test that the timeout_hours parameter works correctly with custom values.
    A notification delivered 13 hours ago should be resolved with a 12h timeout
    but NOT resolved with a 24h timeout.
    """
    prediction_id = "pred-custom-timeout"
    now = datetime.now(timezone.utc)
    delivered_13h_ago = (now - timedelta(hours=13)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Custom timeout", 0.7, "DEFAULT",
             "24_hours", "Do something", 1),
        )

    notif_id = "notif-custom-timeout"
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (notif_id, "Reminder", "Test", "normal", prediction_id, "prediction", "delivered", delivered_13h_ago),
        )

    # Test with 12h timeout — should resolve (13h > 12h)
    resolved_12h = await notification_manager.auto_resolve_stale_predictions(timeout_hours=12)
    assert resolved_12h == 1, "Should resolve with 12h timeout"

    # Verify prediction was resolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["resolved_at"] is not None, "Prediction should be resolved"
