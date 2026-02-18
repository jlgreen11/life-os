"""
Test suite for notification action feedback and prediction accuracy tracking.

This test suite verifies the complete feedback loop:
1. Predictions are created and stored with was_surfaced flag
2. Notifications can be created from predictions
3. Acting on notifications marks predictions as accurate
4. Dismissing notifications marks predictions as inaccurate
5. Prediction accuracy data is correctly queried by the prediction engine
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.notification_manager.manager import NotificationManager


@pytest.mark.asyncio
async def test_act_on_notification_marks_prediction_accurate(db, event_store, user_model_store, event_bus):
    """Acting on a notification should mark the linked prediction as accurate."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})

    # Create a prediction first
    prediction = Prediction(
        id=str(uuid.uuid4()),
        prediction_type="reminder",
        description="You usually exercise on Mondays",
        confidence=0.7,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
        was_surfaced=True,
    )
    user_model_store.store_prediction(prediction.model_dump())

    # Create a notification linked to this prediction
    # (prediction_id is passed via source_event_id, domain must be "prediction")
    notif_id = await manager.create_notification(
        title="Exercise Reminder",
        body=prediction.description,
        priority="normal",
        source_event_id=prediction.id,
        domain="prediction",
    )

    # Act on the notification
    await manager.mark_acted_on(notif_id)

    # Verify prediction was marked accurate
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    assert row is not None, "Prediction should exist"
    assert row["was_accurate"] == 1, "Prediction should be marked accurate"
    assert row["user_response"] == "acted_on", "User response should be 'acted_on'"
    assert row["resolved_at"] is not None, "Resolved timestamp should be set"


@pytest.mark.asyncio
async def test_dismiss_notification_marks_prediction_inaccurate(db, event_store, user_model_store, event_bus):
    """Dismissing a notification should mark the linked prediction as inaccurate."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})

    # Create a prediction
    prediction = Prediction(
        id=str(uuid.uuid4()),
        prediction_type="need",
        description="You might need to pack for your trip",
        confidence=0.65,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="24_hours",
        was_surfaced=True,
    )
    user_model_store.store_prediction(prediction.model_dump())

    # Create a notification linked to this prediction
    # (prediction_id is passed via source_event_id, domain must be "prediction")
    notif_id = await manager.create_notification(
        title="Packing Reminder",
        body=prediction.description,
        priority="normal",
        source_event_id=prediction.id,
        domain="prediction",
    )

    # Dismiss the notification
    await manager.dismiss(notif_id)

    # Verify prediction was marked inaccurate
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    assert row is not None, "Prediction should exist"
    assert row["was_accurate"] == 0, "Prediction should be marked inaccurate"
    assert row["user_response"] == "dismissed", "User response should be 'dismissed'"
    assert row["resolved_at"] is not None, "Resolved timestamp should be set"


@pytest.mark.asyncio
async def test_notification_without_prediction_doesnt_crash(db, event_store, user_model_store, event_bus):
    """Notifications not linked to predictions should still work normally."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})

    # Create a notification without a prediction_id
    notif_id = await manager.create_notification(
        title="System Alert",
        body="Connector sync completed",
        priority="low",
    )

    # Acting on it should not crash (no prediction to update)
    await manager.mark_acted_on(notif_id)

    # Verify the notification status was updated
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM notifications WHERE id = ?",
            (notif_id,),
        ).fetchone()

    assert row["status"] == "acted_on"


@pytest.mark.asyncio
async def test_prediction_accuracy_query_counts_correctly(db, user_model_store):
    """The accuracy multiplier query should correctly count resolved predictions."""
    # Insert 10 resolved predictions: 7 accurate, 3 inaccurate
    for i in range(7):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="reminder",
            description=f"Test prediction {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            was_surfaced=True,
        )
        user_model_store.store_prediction(pred.model_dump())

        # Mark as accurate
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_accurate = 1, resolved_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pred.id),
            )

    for i in range(3):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="reminder",
            description=f"Test prediction inaccurate {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            was_surfaced=True,
        )
        user_model_store.store_prediction(pred.model_dump())

        # Mark as inaccurate
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_accurate = 0, resolved_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pred.id),
            )

    # Query the accuracy stats (same query the engine uses)
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = ?
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL""",
            ("reminder",),
        ).fetchone()

    assert row["total"] == 10, "Should count all 10 resolved predictions"
    assert row["accurate"] == 7, "Should count 7 accurate predictions"

    accuracy_rate = row["accurate"] / row["total"]
    assert accuracy_rate == 0.7, "Accuracy rate should be 70%"


@pytest.mark.asyncio
async def test_unsurfaced_predictions_excluded_from_accuracy_calculation(db, user_model_store):
    """Predictions that weren't surfaced shouldn't affect accuracy calculations."""
    # Create 5 unsurfaced predictions (confidence too low, filtered out)
    for i in range(5):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="need",
            description=f"Unsurfaced prediction {i}",
            confidence=0.4,  # Below threshold
            confidence_gate=ConfidenceGate.OBSERVE,
            time_horizon="24_hours",
            was_surfaced=False,  # Key: not surfaced
        )
        user_model_store.store_prediction(pred.model_dump())

    # Create 2 surfaced, resolved predictions
    for i in range(2):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="need",
            description=f"Surfaced prediction {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            was_surfaced=True,
        )
        user_model_store.store_prediction(pred.model_dump())

        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_accurate = 1, resolved_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pred.id),
            )

    # Accuracy query should only count surfaced predictions
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT COUNT(*) as total
               FROM predictions
               WHERE prediction_type = 'need'
                 AND was_surfaced = 1
                 AND resolved_at IS NOT NULL"""
        ).fetchone()

    assert row["total"] == 2, "Should only count surfaced predictions"


@pytest.mark.asyncio
async def test_multiple_prediction_types_tracked_separately(db, user_model_store):
    """Each prediction type should have independent accuracy tracking."""
    # Create accurate "reminder" predictions (unique descriptions to bypass deduplication)
    for i in range(3):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="reminder",
            description=f"Reminder prediction {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            was_surfaced=True,
        )
        user_model_store.store_prediction(pred.model_dump())
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_accurate = 1, resolved_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pred.id),
            )

    # Create inaccurate "conflict" predictions (unique descriptions to bypass deduplication)
    for i in range(2):
        pred = Prediction(
            id=str(uuid.uuid4()),
            prediction_type="conflict",
            description=f"Conflict prediction {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            was_surfaced=True,
        )
        user_model_store.store_prediction(pred.model_dump())
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_accurate = 0, resolved_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), pred.id),
            )

    # Query reminder accuracy
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT COUNT(*) as total, SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'reminder' AND was_surfaced = 1 AND resolved_at IS NOT NULL"""
        ).fetchone()
    assert row["total"] == 3
    assert row["accurate"] == 3  # 100% accuracy for reminders

    # Query conflict accuracy
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            """SELECT COUNT(*) as total, SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
               FROM predictions
               WHERE prediction_type = 'conflict' AND was_surfaced = 1 AND resolved_at IS NOT NULL"""
        ).fetchone()
    assert row["total"] == 2
    assert row["accurate"] == 0  # 0% accuracy for conflicts
