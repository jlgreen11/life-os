"""
Tests for prediction surfacing fix.

Critical bug fixed in this module:
    Predictions were marked as was_surfaced=1 even when their notifications
    were suppressed, causing them to never be auto-resolved by the
    auto_resolve_stale_predictions() method (which only looks for delivered
    notifications with status='delivered').

The fix ensures predictions are only marked as surfaced when notifications
are actually delivered (either immediately or via batch digest), not when
they are suppressed.

Test coverage:
    1. Immediate delivery → prediction marked as surfaced
    2. Batched delivery → prediction marked as surfaced when digest is retrieved
    3. Suppressed notification → prediction NOT marked as surfaced
    4. Auto-resolve works correctly with the fix
"""

import pytest
from datetime import datetime, timedelta, timezone
from services.notification_manager.manager import NotificationManager


@pytest.mark.asyncio
async def test_immediate_delivery_marks_prediction_surfaced(db, notification_manager):
    """
    When a prediction notification is delivered immediately, the prediction
    should be marked as surfaced (was_surfaced=1).
    """
    # Store a prediction
    pred_id = "pred-001"
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
               VALUES (?, 'reminder', 'Test prediction', 0.7, 'suggest', 0)""",
            (pred_id,),
        )

    # Set notification mode to "frequent" so normal priority gets delivered immediately
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('notification_mode', '"frequent"')"""
        )

    # Create a prediction notification (should be delivered immediately in frequent mode)
    await notification_manager.create_notification(
        title="Reminder: Test",
        body="Test prediction notification",
        priority="normal",
        source_event_id=pred_id,
        domain="prediction",
    )

    # Assert: Prediction is now marked as surfaced
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 1, "Prediction should be marked as surfaced when notification is delivered"


@pytest.mark.asyncio
async def test_batched_delivery_marks_prediction_surfaced_on_digest(db, notification_manager):
    """
    When a prediction notification is batched, the prediction should NOT be
    marked as surfaced immediately, but should be marked when the digest is
    retrieved (actual delivery).
    """
    # Store a prediction
    pred_id = "pred-002"
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
               VALUES (?, 'reminder', 'Test batched prediction', 0.7, 'suggest', 0)""",
            (pred_id,),
        )

    # Set notification mode to "batched" so low-priority notifications get batched
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('notification_mode', '"batched"')"""
        )

    # Create a low-priority prediction notification (should be batched)
    await notification_manager.create_notification(
        title="Reminder: Test",
        body="Test batched prediction notification",
        priority="low",
        source_event_id=pred_id,
        domain="prediction",
    )

    # Assert: Prediction is NOT yet marked as surfaced (notification not delivered yet)
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 0, "Prediction should not be surfaced until batch is delivered"

    # Retrieve the digest (simulates actual delivery)
    digest = await notification_manager.get_digest()
    assert len(digest) == 1, "Digest should contain the batched notification"

    # Assert: Prediction is NOW marked as surfaced (batch delivered)
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 1, "Prediction should be marked as surfaced when batch digest is retrieved"


@pytest.mark.asyncio
async def test_suppressed_notification_does_not_mark_prediction_surfaced(db, notification_manager):
    """
    CRITICAL BUG FIX: When a prediction notification is suppressed (e.g., during
    quiet hours or by user preference), the prediction should NOT be marked as
    surfaced since the user never saw it.

    Before this fix, predictions were marked as surfaced even when suppressed,
    causing them to never be auto-resolved (auto_resolve_stale_predictions only
    looks for delivered notifications).
    """
    # Store a prediction
    pred_id = "pred-003"
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
               VALUES (?, 'reminder', 'Test suppressed prediction', 0.7, 'suggest', 0)""",
            (pred_id,),
        )

    # Set notification mode to "minimal" so normal-priority notifications get suppressed
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('notification_mode', '"minimal"')"""
        )

    # Create a normal-priority prediction notification (should be suppressed in minimal mode)
    result = await notification_manager.create_notification(
        title="Reminder: Test",
        body="Test suppressed prediction notification",
        priority="normal",
        source_event_id=pred_id,
        domain="prediction",
    )

    # Assert: Notification was suppressed (returns None)
    assert result is None, "Notification should be suppressed in minimal mode"

    # Assert: Prediction is NOT marked as surfaced (critical fix)
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 0, "Prediction should NOT be surfaced when notification is suppressed"

    # Assert: Notification status is "suppressed" in the database
    with db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT status FROM notifications WHERE source_event_id = ?", (pred_id,)
        ).fetchone()
        assert notif["status"] == "suppressed", "Notification should be marked as suppressed"


@pytest.mark.asyncio
async def test_auto_resolve_filtered_predictions_works_with_fix(db, notification_manager):
    """
    With the fix in place, auto_resolve_filtered_predictions() should correctly
    resolve predictions that were never surfaced (suppressed notifications).

    This is the intended behavior: predictions that were filtered out before
    surfacing should be auto-resolved with was_accurate=NULL after a timeout.
    """
    # Store a prediction that will be suppressed
    pred_id = "pred-004"
    created_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES (?, 'reminder', 'Test filtered prediction', 0.7, 'suggest', 0, ?)""",
            (pred_id, created_at),
        )

    # Set notification mode to "minimal" so notification gets suppressed
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('notification_mode', '"minimal"')"""
        )

    # Create a suppressed notification
    await notification_manager.create_notification(
        title="Reminder: Test",
        body="Test filtered prediction notification",
        priority="normal",
        source_event_id=pred_id,
        domain="prediction",
    )

    # Assert: Prediction is not surfaced (suppressed)
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced, resolved_at FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 0, "Prediction should not be surfaced"
        assert pred["resolved_at"] is None, "Prediction should not be resolved yet"

    # Run auto-resolve for filtered predictions (1 hour timeout)
    resolved_count = notification_manager.auto_resolve_filtered_predictions(timeout_hours=1)

    # Assert: Prediction was auto-resolved
    assert resolved_count == 1, "Should resolve 1 filtered prediction"

    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()
        assert pred["was_accurate"] is None, "Filtered prediction should have NULL accuracy"
        assert pred["user_response"] == "filtered", "User response should be 'filtered'"
        assert pred["resolved_at"] is not None, "Prediction should be resolved"


@pytest.mark.asyncio
async def test_auto_resolve_stale_predictions_ignores_unsurfaced(db, notification_manager):
    """
    auto_resolve_stale_predictions() should only resolve predictions that were
    surfaced and delivered (have notifications with status='delivered').

    Predictions that were never surfaced (suppressed) should be ignored by this
    method and handled by auto_resolve_filtered_predictions() instead.
    """
    # Store two predictions: one surfaced, one not
    surfaced_pred_id = "pred-005-surfaced"
    unsurfaced_pred_id = "pred-005-unsurfaced"
    created_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, created_at)
               VALUES
                   (?, 'reminder', 'Surfaced prediction', 0.7, 'suggest', 1, ?),
                   (?, 'reminder', 'Unsurfaced prediction', 0.7, 'suggest', 0, ?)""",
            (surfaced_pred_id, created_at, unsurfaced_pred_id, created_at),
        )

    # Create a delivered notification for the surfaced prediction
    delivered_at = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications
               (id, title, body, priority, source_event_id, domain, status, delivered_at)
               VALUES (?, 'Test', 'Test', 'normal', ?, 'prediction', 'delivered', ?)""",
            ("notif-001", surfaced_pred_id, delivered_at),
        )

    # Run auto-resolve for stale predictions (24 hour timeout)
    resolved_count = await notification_manager.auto_resolve_stale_predictions(timeout_hours=24)

    # Assert: Only the surfaced prediction was resolved
    assert resolved_count == 1, "Should resolve only the surfaced prediction"

    with db.get_connection("user_model") as conn:
        surfaced_pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (surfaced_pred_id,),
        ).fetchone()
        assert surfaced_pred["was_accurate"] == 0, "Surfaced prediction should be marked inaccurate"
        assert surfaced_pred["user_response"] == "ignored", "User response should be 'ignored'"
        assert surfaced_pred["resolved_at"] is not None, "Surfaced prediction should be resolved"

        unsurfaced_pred = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (unsurfaced_pred_id,),
        ).fetchone()
        assert unsurfaced_pred["was_accurate"] is None, "Unsurfaced prediction should have NULL accuracy"
        assert unsurfaced_pred["user_response"] is None, "Unsurfaced prediction should have NULL response"
        assert unsurfaced_pred["resolved_at"] is None, "Unsurfaced prediction should not be resolved by this method"


@pytest.mark.asyncio
async def test_critical_priority_bypasses_quiet_hours_and_marks_surfaced(db, notification_manager):
    """
    Critical-priority prediction notifications should bypass quiet hours and
    be delivered immediately, marking the prediction as surfaced.
    """
    # Store a prediction
    pred_id = "pred-006"
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
               VALUES (?, 'risk', 'Critical prediction', 0.9, 'autonomous', 0)""",
            (pred_id,),
        )

    # Set quiet hours to cover current time
    current_hour = datetime.now(timezone.utc).hour
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('quiet_hours_start', ?), ('quiet_hours_end', ?)""",
            (f'"{current_hour:02d}:00"', f'"{(current_hour + 1) % 24:02d}:00"'),
        )

    # Create a critical-priority prediction notification (should bypass quiet hours)
    result = await notification_manager.create_notification(
        title="RISK: Critical Test",
        body="Critical prediction notification",
        priority="critical",
        source_event_id=pred_id,
        domain="prediction",
    )

    # Assert: Notification was delivered (not suppressed)
    assert result is not None, "Critical notification should bypass quiet hours"

    # Assert: Prediction is marked as surfaced
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
        assert pred["was_surfaced"] == 1, "Critical prediction should be marked as surfaced"

    # Assert: Notification status is "delivered"
    with db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT status FROM notifications WHERE source_event_id = ?", (pred_id,)
        ).fetchone()
        assert notif["status"] == "delivered", "Critical notification should be delivered"


@pytest.mark.asyncio
async def test_multiple_batched_predictions_all_marked_surfaced(db, notification_manager):
    """
    When multiple prediction notifications are batched and delivered together
    via get_digest(), all predictions should be marked as surfaced.
    """
    # Store 3 predictions
    pred_ids = [f"pred-007-{i}" for i in range(3)]
    with db.get_connection("user_model") as conn:
        for pred_id in pred_ids:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
                   VALUES (?, 'reminder', 'Batched prediction', 0.7, 'suggest', 0)""",
                (pred_id,),
            )

    # Set notification mode to "batched"
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value)
               VALUES ('notification_mode', '"batched"')"""
        )

    # Create 3 low-priority prediction notifications (all should be batched)
    for pred_id in pred_ids:
        await notification_manager.create_notification(
            title=f"Reminder: {pred_id}",
            body="Test batched prediction",
            priority="low",
            source_event_id=pred_id,
            domain="prediction",
        )

    # Assert: None are surfaced yet
    with db.get_connection("user_model") as conn:
        surfaced_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]
        assert surfaced_count == 0, "No predictions should be surfaced before digest delivery"

    # Retrieve the digest
    digest = await notification_manager.get_digest()
    assert len(digest) == 3, "Digest should contain all 3 batched notifications"

    # Assert: All predictions are now marked as surfaced
    with db.get_connection("user_model") as conn:
        surfaced_count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]
        assert surfaced_count == 3, "All 3 predictions should be marked as surfaced after digest delivery"
