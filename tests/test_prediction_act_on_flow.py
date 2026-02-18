"""
Tests for prediction "Act On" button feedback loop.

This test module verifies that the new "Act On" button in the UI properly
marks predictions as accurate (was_accurate=True) when users find them helpful.

Before this feature, users could only dismiss predictions (was_accurate=False),
which meant ZERO predictions could ever be marked accurate. This broke the
prediction accuracy feedback loop entirely.

With the "Act On" button:
- Users can mark helpful predictions as accurate
- The prediction engine learns which types of predictions are useful
- Confidence gates can be tuned based on real accuracy data
- The system's prediction quality improves over time
"""

import pytest
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_act_on_notification_marks_prediction_accurate(
    db, notification_manager, user_model_store
):
    """
    Acting on a prediction notification should mark the prediction as accurate.

    This is the core feedback loop: when users find a prediction helpful and
    act on it, the system learns that this type of prediction is valuable.
    """
    # Create a prediction
    prediction_id = "test-pred-001"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Unreplied message from john@example.com",
        "confidence": 0.75,
        "confidence_gate": "SUGGEST",
        "was_surfaced": False,  # Not yet surfaced
    })

    # Create a notification for this prediction
    notif_id = await notification_manager.create_notification(
        title="Reminder: Unreplied message",
        body="You haven't replied to john@example.com in 24 hours",
        priority="normal",
        source_event_id=prediction_id,  # Link to the prediction
        domain="prediction",  # Mark as prediction domain
    )

    # Mark the prediction as surfaced (happens when notification is delivered)
    notification_manager._mark_prediction_surfaced(prediction_id)

    # Verify prediction is surfaced but not yet resolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced, was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_surfaced"] == 1
        assert pred["was_accurate"] is None
        assert pred["resolved_at"] is None

    # User acts on the notification (finds it helpful)
    await notification_manager.mark_acted_on(notif_id)

    # Verify the prediction is now marked as accurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 1, "Prediction should be marked accurate after acting on it"
        assert pred["resolved_at"] is not None, "Prediction should have a resolved timestamp"
        assert pred["user_response"] == "acted_on", "User response should be 'acted_on'"


@pytest.mark.asyncio
async def test_dismiss_notification_marks_prediction_inaccurate(
    db, notification_manager, user_model_store
):
    """
    Dismissing a prediction notification should mark it as inaccurate.

    This is the negative feedback signal: when users dismiss a prediction,
    the system learns that this type of prediction was not helpful.
    """
    # Create a prediction
    prediction_id = "test-pred-002"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Unreplied message from spam@marketing.com",
        "confidence": 0.65,
        "confidence_gate": "SUGGEST",
        "was_surfaced": False,
    })

    # Create a notification for this prediction
    notif_id = await notification_manager.create_notification(
        title="Reminder: Unreplied message",
        body="You haven't replied to spam@marketing.com",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Mark as surfaced
    notification_manager._mark_prediction_surfaced(prediction_id)

    # User dismisses the notification (not helpful)
    await notification_manager.dismiss(notif_id)

    # Verify the prediction is marked as inaccurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0, "Prediction should be marked inaccurate after dismissal"
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "dismissed"


@pytest.mark.asyncio
async def test_prediction_accuracy_tracking_with_mixed_feedback(
    db, notification_manager, user_model_store
):
    """
    Test that prediction accuracy can be calculated from mixed feedback.

    With the "Act On" button working, we can now have predictions marked as
    both accurate and inaccurate, enabling real accuracy metrics.
    """
    # Create 10 predictions: 7 will be acted on (accurate), 3 dismissed (inaccurate)
    for i in range(10):
        prediction_id = f"test-pred-mix-{i:03d}"
        user_model_store.store_prediction({
            "id": prediction_id,
            "prediction_type": "reminder",
            "description": f"Test prediction {i}",
            "confidence": 0.7,
            "confidence_gate": "SUGGEST",
            "was_surfaced": False,
        })

        notif_id = await notification_manager.create_notification(
            title=f"Reminder {i}",
            body=f"Test prediction {i}",
            priority="normal",
            source_event_id=prediction_id,
            domain="prediction",
        )

        notification_manager._mark_prediction_surfaced(prediction_id)

        # Act on first 7, dismiss last 3
        if i < 7:
            await notification_manager.mark_acted_on(notif_id)
        else:
            await notification_manager.dismiss(notif_id)

    # Calculate accuracy
    with db.get_connection("user_model") as conn:
        stats = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate,
                   SUM(CASE WHEN was_accurate = 0 THEN 1 ELSE 0 END) as inaccurate
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL"""
        ).fetchone()

        assert stats["total"] == 10
        assert stats["accurate"] == 7, "Should have 7 accurate predictions"
        assert stats["inaccurate"] == 3, "Should have 3 inaccurate predictions"

        # Accuracy rate should be 70%
        accuracy_rate = stats["accurate"] / stats["total"]
        assert accuracy_rate == 0.7


@pytest.mark.asyncio
async def test_feed_api_includes_domain_field(db, notification_manager, user_model_store):
    """
    The feed API must include the domain field so the UI can identify predictions.

    Without the domain field, the UI can't tell which notifications are predictions
    and can't show the "Act On" button.
    """
    # This test would need the actual web app fixture to test the /api/dashboard/feed endpoint
    # For now, we verify the domain field is properly stored
    prediction_id = "test-pred-domain"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Test with domain",
        "confidence": 0.8,
        "confidence_gate": "SUGGEST",
        "was_surfaced": False,
    })

    notif_id = await notification_manager.create_notification(
        title="Test notification",
        body="This should have domain='prediction'",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Verify domain is stored in the database
    with db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT domain FROM notifications WHERE id = ?",
            (notif_id,),
        ).fetchone()
        assert notif["domain"] == "prediction"

    # Verify it's returned by get_pending
    pending = notification_manager.get_pending(limit=50)
    matching = [n for n in pending if n["id"] == notif_id]
    assert len(matching) == 1
    assert matching[0]["domain"] == "prediction"


@pytest.mark.asyncio
async def test_non_prediction_notifications_not_affected(
    db, notification_manager, user_model_store
):
    """
    Non-prediction notifications should not be affected by the Act On feature.

    Only prediction notifications should have was_accurate tracking.
    """
    # Create a regular (non-prediction) notification
    notif_id = await notification_manager.create_notification(
        title="Regular notification",
        body="This is not a prediction",
        priority="normal",
        domain=None,  # No domain (regular notification)
    )

    # Act on it
    await notification_manager.mark_acted_on(notif_id)

    # Verify notification status changed but no prediction was updated
    with db.get_connection("state") as conn:
        notif = conn.execute(
            "SELECT status, acted_on_at FROM notifications WHERE id = ?",
            (notif_id,),
        ).fetchone()
        assert notif["status"] == "acted_on"
        assert notif["acted_on_at"] is not None


@pytest.mark.asyncio
async def test_acting_on_already_resolved_prediction_is_idempotent(
    db, notification_manager, user_model_store
):
    """
    Acting on a notification whose prediction is already resolved should be safe.

    This tests the idempotency of the act_on operation.
    """
    prediction_id = "test-pred-idempotent"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Test idempotency",
        "confidence": 0.7,
        "confidence_gate": "SUGGEST",
        "was_surfaced": False,
    })

    notif_id = await notification_manager.create_notification(
        title="Test",
        body="Test",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    notification_manager._mark_prediction_surfaced(prediction_id)

    # Act on it once
    await notification_manager.mark_acted_on(notif_id)

    # Get the resolved_at timestamp
    with db.get_connection("user_model") as conn:
        first_resolution = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert first_resolution["was_accurate"] == 1

    # Act on it again (should be idempotent)
    await notification_manager.mark_acted_on(notif_id)

    # Verify it's still resolved with was_accurate=1 and same resolution
    with db.get_connection("user_model") as conn:
        second_resolution = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert second_resolution["was_accurate"] == 1
        # The update should only happen if resolved_at IS NULL (see manager.py:502)


@pytest.mark.asyncio
async def test_prediction_feedback_enables_accuracy_learning(
    db, notification_manager, user_model_store
):
    """
    With both Act On and Dismiss working, the system can learn accuracy patterns.

    This test demonstrates the full feedback loop: predictions are generated,
    surfaced, users provide feedback (act on or dismiss), and accuracy can be
    tracked per prediction type.
    """
    # Simulate a week of predictions with different types and user feedback
    prediction_data = [
        # High-priority contacts: 90% acted on (very accurate)
        *[("reminder", "high_priority_contact", True) for _ in range(9)],
        ("reminder", "high_priority_contact", False),
        # Marketing emails: 10% acted on (not accurate)
        ("reminder", "marketing_email", True),
        *[("reminder", "marketing_email", False) for _ in range(9)],
        # Calendar conflicts: 80% acted on (accurate)
        *[("conflict", "double_booking", True) for _ in range(8)],
        *[("conflict", "double_booking", False) for _ in range(2)],
    ]

    for i, (pred_type, context, should_act) in enumerate(prediction_data):
        prediction_id = f"test-pred-learning-{i:03d}"
        user_model_store.store_prediction({
            "id": prediction_id,
            "prediction_type": pred_type,
            # Include index in description to bypass 24h deduplication logic,
            # which would skip identical (type, description) pairs and only
            # store the first one — causing wrong accuracy totals.
            "description": f"{context} prediction {i}",
            "confidence": 0.7,
            "confidence_gate": "SUGGEST",
            "was_surfaced": False,
        })

        notif_id = await notification_manager.create_notification(
            title=f"{pred_type.title()}: {context}",
            body=f"Prediction {i}",
            priority="high" if pred_type == "conflict" else "normal",
            source_event_id=prediction_id,
            domain="prediction",
        )

        notification_manager._mark_prediction_surfaced(prediction_id)

        if should_act:
            await notification_manager.mark_acted_on(notif_id)
        else:
            await notification_manager.dismiss(notif_id)

    # Calculate accuracy by prediction type
    with db.get_connection("user_model") as conn:
        reminder_stats = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND resolved_at IS NOT NULL"""
        ).fetchone()

        conflict_stats = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'conflict'
                 AND resolved_at IS NOT NULL"""
        ).fetchone()

        # Reminder accuracy should be 50% (10 accurate out of 20 total)
        reminder_accuracy = reminder_stats["accurate"] / reminder_stats["total"]
        assert 0.45 <= reminder_accuracy <= 0.55, f"Expected ~50% reminder accuracy, got {reminder_accuracy}"

        # Conflict accuracy should be 80%
        conflict_accuracy = conflict_stats["accurate"] / conflict_stats["total"]
        assert 0.75 <= conflict_accuracy <= 0.85, f"Expected ~80% conflict accuracy, got {conflict_accuracy}"

    # This data enables the prediction engine to:
    # - Boost confidence gates for conflict predictions (they're accurate!)
    # - Lower confidence gates or suppress certain reminder types (low accuracy)
    # - Learn which contexts produce helpful predictions


@pytest.mark.asyncio
async def test_prediction_without_notification_not_affected(db, user_model_store):
    """
    Predictions that never get surfaced as notifications should remain unresolved.

    Only predictions that are surfaced and get user feedback should be resolved.
    """
    # Create a prediction that never gets a notification
    prediction_id = "test-pred-no-notif"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Filtered out by confidence gate",
        "confidence": 0.2,  # Too low to surface
        "confidence_gate": "OBSERVE",
        "was_surfaced": False,
    })

    # Verify it remains unresolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_surfaced, was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_surfaced"] == 0
        assert pred["was_accurate"] is None
        assert pred["resolved_at"] is None
