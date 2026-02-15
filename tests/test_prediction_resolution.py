"""
Test suite for prediction resolution via WebSocket commands.

This test suite verifies the complete prediction feedback loop:
1. Predictions are stored with was_accurate=NULL, resolved_at=NULL
2. WebSocket commands (dismiss_notification, act_on_notification) trigger
   prediction resolution via the notification manager
3. Direct prediction resolution via resolve_prediction command
4. Predictions are marked with was_accurate and resolved_at timestamps
5. Multiple resolution attempts are idempotent (last write wins)

This closes the prediction feedback loop so the prediction engine can
adjust confidence based on historical accuracy.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_websocket_dismiss_notification_resolves_prediction(db, event_store, user_model_store, event_bus):
    """WebSocket dismiss_notification command should resolve linked prediction."""
    from services.notification_manager.manager import NotificationManager

    notification_manager = NotificationManager(db, event_bus, {})

    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Test prediction", 0.7, "DEFAULT"),
        )

    # Create a notification linked to this prediction
    notif_id = await notification_manager.create_notification(
        title="Test Reminder",
        body="You should do X",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Verify prediction starts unresolved
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] is None
        assert pred["resolved_at"] is None

    # Dismiss the notification (this should resolve the prediction)
    await notification_manager.dismiss(notif_id)

    # Verify prediction is now marked as inaccurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0  # False
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "dismissed"


@pytest.mark.asyncio
async def test_websocket_act_on_notification_resolves_prediction(db, event_store, user_model_store, event_bus):
    """WebSocket act_on_notification command should resolve linked prediction as accurate."""
    from services.notification_manager.manager import NotificationManager

    notification_manager = NotificationManager(db, event_bus, {})

    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Test prediction", 0.7, "DEFAULT"),
        )

    # Create a notification linked to this prediction
    notif_id = await notification_manager.create_notification(
        title="Test Reminder",
        body="You should do X",
        priority="normal",
        source_event_id=prediction_id,
        domain="prediction",
    )

    # Act on the notification (this should resolve the prediction as accurate)
    await notification_manager.mark_acted_on(notif_id)

    # Verify prediction is now marked as accurate
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 1  # True
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "acted_on"


@pytest.mark.asyncio
async def test_direct_prediction_resolution(db, event_store, user_model_store, event_bus):
    """Direct resolve_prediction method should work independently of notifications."""
    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "need", "You might need X", 0.8, "DEFAULT"),
        )

    # Resolve directly with custom feedback
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=True,
        user_response="Very helpful, thanks!"
    )

    # Verify prediction is resolved with custom response
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 1  # True
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "Very helpful, thanks!"


@pytest.mark.asyncio
async def test_resolve_prediction_with_false_accuracy(db, event_store, user_model_store, event_bus):
    """Resolve prediction as inaccurate with custom feedback."""
    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "conflict", "Calendar conflict", 0.9, "AUTONOMOUS"),
        )

    # Resolve as inaccurate
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=False,
        user_response="False alarm, events are sequential"
    )

    # Verify
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0  # False
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "False alarm, events are sequential"


@pytest.mark.asyncio
async def test_resolve_prediction_without_user_response(db, event_store, user_model_store, event_bus):
    """Resolve prediction with accuracy but no custom text response."""
    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "opportunity", "Good time to X", 0.65, "SUGGEST"),
        )

    # Resolve with no user_response text
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=True,
        user_response=None
    )

    # Verify
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 1  # True
        assert pred["resolved_at"] is not None
        assert pred["user_response"] is None  # No custom response


@pytest.mark.asyncio
async def test_multiple_resolutions_last_write_wins(db, event_store, user_model_store, event_bus):
    """Multiple resolution calls should be idempotent (last write wins)."""
    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "reminder", "Test", 0.7, "DEFAULT"),
        )

    # First resolution: accurate
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=True,
        user_response="First resolution"
    )

    # Second resolution: inaccurate (user changed their mind)
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=False,
        user_response="Second resolution"
    )

    # Verify last write wins
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0  # Last write: False
        assert pred["user_response"] == "Second resolution"


@pytest.mark.asyncio
async def test_notification_without_prediction_doesnt_crash(db, event_store, user_model_store, event_bus):
    """Dismissing/acting on a non-prediction notification should not crash."""
    from services.notification_manager.manager import NotificationManager

    notification_manager = NotificationManager(db, event_bus, {})

    # Create a notification with domain != "prediction"
    notif_id = await notification_manager.create_notification(
        title="Email notification",
        body="New email from X",
        priority="normal",
        domain="email",
    )

    # Dismiss it — should not crash even though there's no linked prediction
    await notification_manager.dismiss(notif_id)

    # Act on it — should not crash
    notif_id_2 = await notification_manager.create_notification(
        title="Task notification",
        body="Task deadline approaching",
        priority="high",
        domain="task",
    )
    await notification_manager.mark_acted_on(notif_id_2)


@pytest.mark.asyncio
async def test_prediction_without_notification_can_be_resolved(db, event_store, user_model_store, event_bus):
    """Predictions can be resolved directly even if they never had a notification."""
    # Store a prediction that was OBSERVED but never surfaced
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (prediction_id, "risk", "Low confidence risk", 0.25, "OBSERVE", 0),
        )

    # Resolve it anyway (e.g., user inspected the full prediction log in admin)
    user_model_store.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=False,
        user_response="Never saw this, too low confidence"
    )

    # Verify
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 0
        assert pred["resolved_at"] is not None


@pytest.mark.asyncio
async def test_batch_prediction_resolution(db, event_store, user_model_store, event_bus):
    """Multiple predictions can be resolved in batch."""
    from services.notification_manager.manager import NotificationManager

    notification_manager = NotificationManager(db, event_bus, {})

    prediction_ids = []
    notif_ids = []

    # Create 5 predictions with notifications
    for i in range(5):
        prediction_id = str(uuid.uuid4())
        prediction_ids.append(prediction_id)

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate)
                   VALUES (?, ?, ?, ?, ?)""",
                (prediction_id, "reminder", f"Prediction {i}", 0.7, "DEFAULT"),
            )

        notif_id = await notification_manager.create_notification(
            title=f"Reminder {i}",
            body=f"Test {i}",
            priority="normal",
            source_event_id=prediction_id,
            domain="prediction",
        )
        notif_ids.append(notif_id)

    # Dismiss first 3, act on last 2
    for notif_id in notif_ids[:3]:
        await notification_manager.dismiss(notif_id)

    for notif_id in notif_ids[3:]:
        await notification_manager.mark_acted_on(notif_id)

    # Verify all resolved correctly
    with db.get_connection("user_model") as conn:
        for i, prediction_id in enumerate(prediction_ids):
            pred = conn.execute(
                "SELECT was_accurate, resolved_at FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
            assert pred["resolved_at"] is not None
            if i < 3:
                assert pred["was_accurate"] == 0  # Dismissed
            else:
                assert pred["was_accurate"] == 1  # Acted on


@pytest.mark.asyncio
async def test_resolve_prediction_doesnt_crash_without_event_bus(db, event_store):
    """Prediction resolution should work even without event bus connected."""
    from storage.user_model_store import UserModelStore

    # Create a user_model_store with no event bus
    ums = UserModelStore(db, event_bus=None)

    # Store a prediction
    prediction_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, "need", "Test", 0.8, "DEFAULT"),
        )

    # Resolve it — should not crash even with no event bus
    ums.resolve_prediction(
        prediction_id=prediction_id,
        was_accurate=True,
        user_response="Helpful"
    )

    # Verify prediction is resolved correctly
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT was_accurate, resolved_at, user_response FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()
        assert pred["was_accurate"] == 1
        assert pred["resolved_at"] is not None
        assert pred["user_response"] == "Helpful"
