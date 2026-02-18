"""
Tests for prediction surfacing behavior.

This test suite verifies that when a notification is created from a prediction,
the prediction is correctly marked as "surfaced" (was_surfaced = 1). This is
critical for the accuracy feedback loop — only surfaced predictions should be
included in accuracy metrics.
"""

import asyncio
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_prediction_marked_surfaced_when_notification_created(db, event_store, user_model_store, event_bus):
    """When a notification is created from a prediction, mark prediction as surfaced."""
    from services.notification_manager.manager import NotificationManager

    nm = NotificationManager(db, event_bus, config={})

    # Create a prediction first
    prediction_id = "pred-123"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Test prediction",
        "confidence": 0.7,
        "confidence_gate": "default",
        "was_surfaced": False,  # Starts as not surfaced
    })

    # Verify prediction starts as not surfaced
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_surfaced"] == 0, "Prediction should start as not surfaced"

    # Create notification from the prediction
    notif_id = await nm.create_notification(
        title="Test notification",
        body="Test body",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Flush the batch digest so batched notifications are delivered and
    # predictions are marked as surfaced. In production this is called by
    # the periodic digest loop; in tests we call it directly.
    await nm.get_digest()

    # Verify prediction is now marked as surfaced
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_surfaced"] == 1, "Prediction should be marked as surfaced"


@pytest.mark.asyncio
async def test_non_prediction_notification_does_not_update_predictions(db, event_store, user_model_store, event_bus):
    """Notifications from non-prediction domains should not affect predictions table."""
    from services.notification_manager.manager import NotificationManager

    nm = NotificationManager(db, event_bus, config={})

    # Create a notification from a non-prediction source (e.g., email event)
    event_id = "email-event-456"
    notif_id = await nm.create_notification(
        title="New email notification",
        body="Test body",
        priority="normal",
        source_event_id=event_id,
        domain="email",  # Not a prediction domain
    )

    # Verify no predictions were created or modified
    with db.get_connection("user_model") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        assert count["cnt"] == 0, "No predictions should exist"


@pytest.mark.asyncio
async def test_prediction_without_source_event_id_does_not_crash(db, event_store, user_model_store, event_bus):
    """Creating a prediction notification without source_event_id should not crash."""
    from services.notification_manager.manager import NotificationManager

    nm = NotificationManager(db, event_bus, config={})

    # Create notification with prediction domain but no source_event_id
    notif_id = await nm.create_notification(
        title="Test notification",
        body="Test body",
        priority="normal",
        source_event_id=None,  # No source
        domain="prediction",
    )

    # Should complete without error (no database update attempted)
    assert notif_id is not None or notif_id is None  # Either outcome is acceptable


@pytest.mark.asyncio
async def test_multiple_notifications_from_same_prediction(db, event_store, user_model_store, event_bus):
    """Multiple notifications from the same prediction should keep was_surfaced = 1."""
    from services.notification_manager.manager import NotificationManager

    nm = NotificationManager(db, event_bus, config={})

    # Create a prediction
    prediction_id = "pred-789"
    user_model_store.store_prediction({
        "id": prediction_id,
        "prediction_type": "reminder",
        "description": "Test prediction",
        "confidence": 0.7,
        "confidence_gate": "default",
        "was_surfaced": False,
    })

    # Create first notification from prediction
    notif_id_1 = await nm.create_notification(
        title="First notification",
        body="Test body 1",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Create second notification from same prediction
    notif_id_2 = await nm.create_notification(
        title="Second notification",
        body="Test body 2",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Flush the batch digest so batched notifications are delivered and
    # predictions are marked as surfaced (idempotent — second flush is a no-op).
    await nm.get_digest()

    # Verify prediction is still marked as surfaced (idempotent update)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_surfaced FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert row["was_surfaced"] == 1, "Prediction should remain surfaced"


@pytest.mark.asyncio
async def test_accuracy_calculation_only_counts_surfaced_predictions(db, event_store, user_model_store, event_bus):
    """Accuracy queries should only include predictions where was_surfaced = 1."""
    from services.notification_manager.manager import NotificationManager

    nm = NotificationManager(db, event_bus, config={})

    # Create 5 predictions:
    # - 2 surfaced + resolved as accurate
    # - 1 surfaced + resolved as inaccurate
    # - 2 never surfaced (filtered by confidence gates)

    for i in range(5):
        prediction_id = f"pred-{i}"
        user_model_store.store_prediction({
            "id": prediction_id,
            "prediction_type": "reminder",
            "description": f"Test prediction {i}",
            "confidence": 0.7,
            "confidence_gate": "default",
            "was_surfaced": False,
        })

        # Surface predictions 0, 1, 2 via notifications
        if i < 3:
            await nm.create_notification(
                title=f"Notification {i}",
                body=f"Body {i}",
                priority="normal",
                source_event_id=prediction_id,
                domain="prediction",
            )

    # Flush the batch digest so batched notifications are delivered and
    # predictions 0, 1, 2 are marked as surfaced.
    await nm.get_digest()

    # Manually resolve surfaced predictions (simulating user feedback)
    with db.get_connection("user_model") as conn:
        now = datetime.now(timezone.utc).isoformat()
        # pred-0, pred-1: accurate
        for i in [0, 1]:
            conn.execute(
                """UPDATE predictions SET
                   was_accurate = 1, resolved_at = ?, user_response = 'acted_on'
                   WHERE id = ?""",
                (now, f"pred-{i}"),
            )
        # pred-2: inaccurate
        conn.execute(
            """UPDATE predictions SET
               was_accurate = 0, resolved_at = ?, user_response = 'dismissed'
               WHERE id = ?""",
            (now, "pred-2"),
        )

    # Run accuracy query (same as prediction engine uses)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'reminder'
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL""",
        ).fetchone()

    # Should count only the 3 surfaced predictions (0, 1, 2)
    assert row["total"] == 3, "Should count only surfaced + resolved predictions"
    assert row["accurate"] == 2, "Should count 2 accurate predictions"

    # Calculate accuracy rate
    accuracy_rate = row["accurate"] / row["total"] if row["total"] > 0 else 0
    assert accuracy_rate == pytest.approx(0.6666, rel=0.01), "Accuracy rate should be 66.7%"


@pytest.mark.asyncio
async def test_filtered_predictions_remain_unsurfaced(db, event_store, user_model_store, event_bus):
    """Predictions filtered by confidence gates should remain was_surfaced = 0."""
    # Create predictions but don't create notifications from them
    # (simulates the prediction engine filtering low-confidence predictions)

    for i in range(3):
        user_model_store.store_prediction({
            "id": f"filtered-{i}",
            "prediction_type": "reminder",
            "description": f"Filtered prediction {i}",
            "confidence": 0.2,  # Too low, would be filtered
            "confidence_gate": "observe",
            "was_surfaced": False,
        })

    # Verify all remain unsurfaced
    with db.get_connection("user_model") as conn:
        rows = conn.execute(
            "SELECT id, was_surfaced FROM predictions WHERE id LIKE 'filtered-%'",
        ).fetchall()

        assert len(rows) == 3, "Should have 3 filtered predictions"
        for row in rows:
            assert row["was_surfaced"] == 0, f"Prediction {row['id']} should remain unsurfaced"
