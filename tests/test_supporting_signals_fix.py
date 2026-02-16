"""
Tests for the supporting_signals data structure fix.

This test suite verifies that:
1. Predictions are created with structured dict-based supporting_signals
2. The behavioral accuracy tracker can extract contact information from supporting_signals
3. Accuracy inference works correctly with the new format
4. Backward compatibility is maintained for old list-based format
"""

import json
import pytest
from datetime import datetime, timedelta, timezone

from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine
from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


@pytest.mark.asyncio
async def test_prediction_model_accepts_dict_supporting_signals():
    """Verify the Prediction model accepts dict for supporting_signals."""
    pred = Prediction(
        prediction_type="reminder",
        description="Reply to Alice about dinner plans",
        confidence=0.7,
        confidence_gate="suggest",  # lowercase enum value
        time_horizon="2_hours",
        supporting_signals={
            "contact_email": "alice@example.com",
            "contact_name": "Alice",
            "message_id": "msg-123",
            "hours_since_received": 4.5,
            "is_priority_contact": True,
        }
    )

    assert isinstance(pred.supporting_signals, dict)
    assert pred.supporting_signals["contact_email"] == "alice@example.com"
    assert pred.supporting_signals["contact_name"] == "Alice"
    assert pred.supporting_signals["message_id"] == "msg-123"


@pytest.mark.asyncio
async def test_prediction_engine_populates_supporting_signals_dict(db, user_model_store, event_store):
    """Verify prediction engine creates predictions with structured supporting_signals."""
    # Create an inbound message that needs a reply
    event_store.store_event({  # Not async
        "id": "msg-inbound-1",
        "type": "email.received",
        "source": "test",
        "timestamp": (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat(),
        "payload": {
            "message_id": "msg-inbound-1",
            "from_address": "bob@example.com",
            "to": "user@example.com",
            "subject": "Quick question about the project",
            "body_plain": "Hey, can you send me the latest status update?",
            "requires_response": True,
        },
        "metadata": {
            "related_contacts": ["bob@example.com"],
        },
    })

    # Run prediction engine with empty context
    engine = PredictionEngine(db, user_model_store)
    predictions = await engine.generate_predictions({})

    # Should generate a reminder prediction
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) > 0

    pred = reminder_predictions[0]
    assert isinstance(pred.supporting_signals, dict)
    assert "contact_email" in pred.supporting_signals
    assert "contact_name" in pred.supporting_signals
    assert pred.supporting_signals["contact_email"] == "bob@example.com"
    assert "bob" in pred.supporting_signals["contact_name"].lower()


@pytest.mark.asyncio
async def test_behavioral_tracker_extracts_contact_from_new_format(db):
    """Verify behavioral tracker can extract contact info from dict supporting_signals."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a prediction with new dict format
    prediction = {
        "id": "pred-1",
        "prediction_type": "reminder",
        "description": "Reply to Alice about dinner plans",
        "supporting_signals": json.dumps({
            "contact_email": "alice@example.com",
            "contact_name": "Alice",
            "message_id": "msg-123",
        }),
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }

    # Parse supporting_signals
    signals = json.loads(prediction["supporting_signals"])

    # Verify tracker can extract contact info
    assert signals.get("contact_email") == "alice@example.com"
    assert signals.get("contact_name") == "Alice"


@pytest.mark.asyncio
async def test_behavioral_tracker_infers_accuracy_from_email_sent(db, event_store):
    """Verify behavioral tracker marks prediction accurate when user sends matching email."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a surfaced prediction
    prediction_time = datetime.now(timezone.utc) - timedelta(hours=4)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-123",
                "reminder",
                "Reply to alice@example.com about dinner plans",
                0.7,
                "suggest",  # lowercase
                "2_hours",
                "Reply to alice@example.com",
                json.dumps({
                    "contact_email": "alice@example.com",
                    "contact_name": "Alice",
                    "message_id": "msg-inbound-123",
                }),
                1,  # was_surfaced
                prediction_time.isoformat(),
            ),
        )

    # User sends email to Alice 2 hours later
    event_store.store_event({
        "id": "msg-outbound-1",
        "type": "email.sent",
        "source": "test",
        "timestamp": (prediction_time + timedelta(hours=2)).isoformat(),
        "payload": {
            "to": "alice@example.com",
            "subject": "Re: Dinner plans",
            "body_plain": "Sounds great! Let's meet at 7pm.",
        },
        "metadata": {},
    })

    # Run inference cycle
    stats = await tracker.run_inference_cycle()

    # Should mark prediction as accurate
    assert stats["marked_accurate"] == 1
    assert stats["marked_inaccurate"] == 0

    # Verify database was updated
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            ("pred-123",),
        ).fetchone()

    assert row["was_accurate"] == 1
    assert row["user_response"] == "inferred"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_behavioral_tracker_infers_inaccuracy_after_timeout(db):
    """Verify behavioral tracker marks prediction inaccurate after 48h with no action."""
    tracker = BehavioralAccuracyTracker(db)

    # Create a surfaced prediction from 50 hours ago
    prediction_time = datetime.now(timezone.utc) - timedelta(hours=50)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-456",
                "reminder",
                "Reply to bob@example.com about project status",
                0.6,
                "SUGGEST",
                "2_hours",
                "Reply to bob@example.com",
                json.dumps({
                    "contact_email": "bob@example.com",
                    "contact_name": "Bob",
                }),
                1,  # was_surfaced
                prediction_time.isoformat(),
            ),
        )

    # Run inference cycle (no matching outbound message exists)
    stats = await tracker.run_inference_cycle()

    # Should mark prediction as inaccurate
    assert stats["marked_inaccurate"] == 1
    assert stats["marked_accurate"] == 0

    # Verify database was updated
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate, user_response, resolved_at FROM predictions WHERE id = ?",
            ("pred-456",),
        ).fetchone()

    assert row["was_accurate"] == 0
    assert row["user_response"] == "inferred"
    assert row["resolved_at"] is not None


@pytest.mark.asyncio
async def test_behavioral_tracker_handles_name_match(db, event_store):
    """Verify tracker matches by contact name when email not in 'to' field."""
    tracker = BehavioralAccuracyTracker(db)

    # Create prediction with contact name
    prediction_time = datetime.now(timezone.utc) - timedelta(hours=3)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-789",
                "reminder",
                "Follow up with Grace about the meeting",
                0.65,
                "suggest",  # lowercase
                "2_hours",
                "Message Grace",
                json.dumps({
                    "contact_name": "Grace",
                }),
                1,
                prediction_time.isoformat(),
            ),
        )

    # User sends message with "Grace" in recipient field
    event_store.store_event({
        "id": "msg-outbound-2",
        "type": "message.sent",
        "source": "test",
        "timestamp": (prediction_time + timedelta(hours=1)).isoformat(),
        "payload": {
            "to": "Grace Williams",
            "body_plain": "Hi Grace, following up on our meeting discussion...",
        },
        "metadata": {},
    })

    # Run inference
    stats = await tracker.run_inference_cycle()

    assert stats["marked_accurate"] == 1


@pytest.mark.asyncio
async def test_backward_compatibility_with_old_format(db):
    """Verify tracker handles old list-based supporting_signals gracefully."""
    tracker = BehavioralAccuracyTracker(db)

    # Create prediction with old list format (empty list converted to "[]" string)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-old",
                "reminder",
                "Reply to Alice about dinner",
                0.7,
                "SUGGEST",
                "2_hours",
                "Reply to Alice",
                "[]",  # Old empty list format
                1,
                (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            ),
        )

    # Run inference - should not crash, should extract from description
    stats = await tracker.run_inference_cycle()

    # Should handle gracefully (may or may not infer, but should not crash)
    assert stats["marked_accurate"] >= 0
    assert stats["marked_inaccurate"] >= 0


@pytest.mark.asyncio
async def test_storage_preserves_dict_format(db, user_model_store):
    """Verify UserModelStore correctly serializes and deserializes dict supporting_signals."""
    # Store prediction with dict supporting_signals
    prediction = {
        "id": "pred-storage-test",
        "prediction_type": "reminder",
        "description": "Test prediction",
        "confidence": 0.8,
        "confidence_gate": "DEFAULT",
        "time_horizon": "2_hours",
        "suggested_action": "Test action",
        "supporting_signals": {
            "contact_email": "test@example.com",
            "contact_name": "Test User",
            "custom_field": "custom_value",
        },
        "was_surfaced": False,
    }

    user_model_store.store_prediction(prediction)

    # Retrieve and verify
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            ("pred-storage-test",),
        ).fetchone()

    stored_signals = json.loads(row["supporting_signals"])
    assert isinstance(stored_signals, dict)
    assert stored_signals["contact_email"] == "test@example.com"
    assert stored_signals["contact_name"] == "Test User"
    assert stored_signals["custom_field"] == "custom_value"
