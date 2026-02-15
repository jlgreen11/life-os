"""
Tests for notification feedback loop integration.

Verifies that user responses to notifications (act on / dismiss) properly
flow through the feedback collector to enable learning and model improvement.
"""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, timezone

from models.core import ConfidenceGate, FeedbackType
from models.user_model import Prediction
from services.notification_manager.manager import NotificationManager
from services.feedback_collector.collector import FeedbackCollector
from storage.user_model_store import UserModelStore


@pytest.mark.asyncio
async def test_acting_on_notification_records_feedback(db, event_bus):
    """Acting on a notification should record positive feedback in feedback_log."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    # Create a notification
    notif_id = await manager.create_notification(
        title="Email from boss",
        body="Need Q4 report",
        priority="high",
        source_event_id="email-123",
        domain="email",
    )

    # Process the notification response
    await collector.process_notification_response(
        notification_id=notif_id,
        response_type="engaged",
        response_time_seconds=10,
    )

    # Verify feedback was recorded
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,),
        ).fetchone()

    assert feedback is not None, "Feedback should be recorded"
    assert feedback["action_type"] == "notification"
    assert feedback["feedback_type"] == FeedbackType.ENGAGED.value
    assert feedback["response_latency_seconds"] == 10

    # Verify context includes domain and priority
    context = json.loads(feedback["context"])
    assert context["domain"] == "email"
    assert context["priority"] == "high"
    assert "hour_of_day" in context


@pytest.mark.asyncio
async def test_dismissing_notification_records_negative_feedback(db, event_bus):
    """Dismissing a notification should record negative feedback."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    notif_id = await manager.create_notification(
        title="Calendar reminder",
        body="Meeting in 15 min",
        priority="normal",
        source_event_id="cal-456",
        domain="calendar",
    )

    await collector.process_notification_response(
        notification_id=notif_id,
        response_type="dismissed",
        response_time_seconds=2,
    )

    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,),
        ).fetchone()

    assert feedback is not None
    assert feedback["feedback_type"] == FeedbackType.DISMISSED.value


@pytest.mark.asyncio
async def test_domain_inference_from_various_event_types():
    """Test the domain inference helper function."""
    from main import LifeOS

    # Create a minimal instance just to test the method
    test_cases = [
        ("email.received", "email"),
        ("email.sent", "email"),
        ("message.received", "message"),
        ("message.sent", "message"),
        ("calendar.event.created", "calendar"),
        ("calendar.conflict.detected", "calendar"),
        ("system.connector.sync_complete", "system"),
        ("usermodel.prediction.generated", "usermodel"),
    ]

    for event_type, expected_domain in test_cases:
        # Test the static helper directly
        inferred_domain = LifeOS._infer_domain_from_event_type(None, event_type)
        assert inferred_domain == expected_domain, \
            f"Event type '{event_type}' should infer domain '{expected_domain}', got '{inferred_domain}'"


@pytest.mark.asyncio
async def test_domain_inference_fallback_for_malformed_types():
    """Malformed event types should fall back to 'system' domain."""
    from main import LifeOS

    assert LifeOS._infer_domain_from_event_type(None, "") == "system"
    assert LifeOS._infer_domain_from_event_type(None, "nodots") == "system"
    assert LifeOS._infer_domain_from_event_type(None, None) == "system"


@pytest.mark.asyncio
async def test_multiple_feedback_entries_created_independently(db, event_bus):
    """Each notification interaction should create a separate feedback entry."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    # Create two notifications
    notif_1 = await manager.create_notification(
        title="Email 1",
        body="First email",
        priority="normal",
        source_event_id="email-1",
        domain="email",
    )

    notif_2 = await manager.create_notification(
        title="Email 2",
        body="Second email",
        priority="high",
        source_event_id="email-2",
        domain="email",
    )

    # Process different responses
    await collector.process_notification_response(notif_1, "engaged", 5)
    await collector.process_notification_response(notif_2, "dismissed", 1)

    # Verify two separate feedback entries
    with db.get_connection("preferences") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM feedback_log").fetchone()
        assert count["cnt"] == 2, "Should have two feedback entries"

        engaged = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ? AND feedback_type = ?",
            (notif_1, FeedbackType.ENGAGED.value),
        ).fetchone()
        assert engaged is not None

        dismissed = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ? AND feedback_type = ?",
            (notif_2, FeedbackType.DISMISSED.value),
        ).fetchone()
        assert dismissed is not None


@pytest.mark.asyncio
async def test_prediction_notifications_update_both_accuracy_and_feedback(db, event_bus, user_model_store):
    """Prediction notifications should update was_accurate AND record feedback."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    collector = FeedbackCollector(db, user_model_store, event_bus)

    # Create a prediction
    prediction = Prediction(
        id=str(uuid.uuid4()),
        prediction_type="reminder",
        description="Time to exercise",
        confidence=0.7,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
        was_surfaced=False,
    )

    user_model_store.store_prediction(prediction.model_dump())

    # Create a notification from the prediction
    notif_id = await manager.create_notification(
        title="Reminder",
        body=prediction.description,
        priority="normal",
        source_event_id=prediction.id,
        domain="prediction",
    )

    # Act on it
    await manager.mark_acted_on(notif_id)
    await asyncio.sleep(0.05)

    # Verify prediction was updated
    with db.get_connection("user_model") as conn:
        pred = conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    assert pred["was_accurate"] == 1, "Prediction should be marked accurate"
    assert pred["user_response"] == "acted_on"
    assert pred["resolved_at"] is not None

    # Also verify feedback was recorded
    await collector.process_notification_response(notif_id, "engaged", 0)

    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,),
        ).fetchone()

    assert feedback is not None, "Feedback should also be recorded for predictions"


@pytest.mark.asyncio
async def test_feedback_without_source_event_id_still_works(db, event_bus):
    """Notifications without source_event_id should still record feedback."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    # Create a notification without source_event_id
    notif_id = await manager.create_notification(
        title="System Alert",
        body="Connector sync failed",
        priority="low",
        source_event_id=None,
        domain="system",
    )

    await collector.process_notification_response(notif_id, "dismissed", 1)

    # Verify feedback was still recorded
    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,),
        ).fetchone()

    assert feedback is not None
    assert feedback["feedback_type"] == FeedbackType.DISMISSED.value


@pytest.mark.asyncio
async def test_feedback_enriched_with_all_context_fields(db, event_bus):
    """Feedback context should include priority, domain, and hour_of_day."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    notif_id = await manager.create_notification(
        title="High Priority Email",
        body="Urgent matter",
        priority="critical",
        source_event_id="email-999",
        domain="email",
    )

    await collector.process_notification_response(notif_id, "engaged", 15)

    with db.get_connection("preferences") as conn:
        feedback = conn.execute(
            "SELECT * FROM feedback_log WHERE action_id = ?",
            (notif_id,),
        ).fetchone()

    context = json.loads(feedback["context"])

    # Verify all expected context fields
    assert "priority" in context
    assert context["priority"] == "critical"
    assert "domain" in context
    assert context["domain"] == "email"
    assert "hour_of_day" in context
    assert 0 <= context["hour_of_day"] < 24


@pytest.mark.asyncio
async def test_fast_dismissal_creates_different_learning_signal(db, event_bus):
    """Fast dismissal (<2 sec) should indicate irrelevance."""
    manager = NotificationManager(db=db, event_bus=event_bus, config={})
    ums = UserModelStore(db, event_bus)
    collector = FeedbackCollector(db, ums, event_bus)

    notif_id = await manager.create_notification(
        title="Marketing email",
        body="Special offer",
        priority="low",
        source_event_id="email-spam",
        domain="email",
    )

    # Quick dismissal (< 2 seconds)
    await collector.process_notification_response(notif_id, "dismissed", 1.5)

    # The feedback collector should learn from this via semantic facts
    # Check if a semantic fact was created about dismissing this domain
    with db.get_connection("user_model") as conn:
        fact = conn.execute(
            "SELECT * FROM semantic_facts WHERE key LIKE ?",
            ("%notification_irrelevant_email%",),
        ).fetchone()

    # This should create a semantic fact about dismissing email notifications quickly
    assert fact is not None, "Quick dismissal should create a semantic fact"
    assert "quickly dismisses" in fact["value"].lower()
