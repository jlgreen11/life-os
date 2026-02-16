"""
Test suite for supporting_signals storage in predictions.

This test verifies the critical bug fix that changes the default value for
supporting_signals from [] (list) to {} (dict) in UserModelStore.store_prediction().

Without this fix, all predictions are stored with empty supporting_signals,
which breaks the behavioral accuracy tracking system's ability to automatically
resolve predictions (311,722 unresolved reminders as of iteration 58).
"""

import json
import pytest
from datetime import datetime, timezone
from models.user_model import Prediction, ConfidenceGate


def test_supporting_signals_stored_as_dict(user_model_store):
    """Verify supporting_signals are stored as dict, not list."""
    prediction = Prediction(
        prediction_type="reminder",
        description="Test prediction",
        confidence=0.8,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="2_hours",
        suggested_action="Test action",
        supporting_signals={
            "contact_email": "test@example.com",
            "contact_name": "Test User",
            "message_id": "msg-123",
            "hours_since_received": 4.5,
            "is_priority_contact": True,
            "requires_response": False,
        },
    )

    # Store the prediction
    user_model_store.store_prediction(prediction.model_dump())

    # Retrieve from database
    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    # Parse the JSON
    stored_signals = json.loads(row["supporting_signals"])

    # CRITICAL: Must be a dict, not a list
    assert isinstance(stored_signals, dict), (
        f"supporting_signals must be stored as dict, got {type(stored_signals)}"
    )

    # Verify all fields are preserved
    assert stored_signals["contact_email"] == "test@example.com"
    assert stored_signals["contact_name"] == "Test User"
    assert stored_signals["message_id"] == "msg-123"
    assert stored_signals["hours_since_received"] == 4.5
    assert stored_signals["is_priority_contact"] is True
    assert stored_signals["requires_response"] is False


def test_supporting_signals_empty_dict_default(user_model_store):
    """Verify empty supporting_signals default to {}, not []."""
    prediction = Prediction(
        prediction_type="conflict",
        description="Calendar conflict",
        confidence=0.95,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
        suggested_action="Reschedule",
        # No supporting_signals provided
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    stored_signals = json.loads(row["supporting_signals"])

    # Must default to empty dict, not empty list
    assert isinstance(stored_signals, dict)
    assert stored_signals == {}


def test_supporting_signals_with_nested_structures(user_model_store):
    """Verify complex nested structures in supporting_signals are preserved."""
    prediction = Prediction(
        prediction_type="opportunity",
        description="Networking opportunity",
        confidence=0.7,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="this_week",
        suggested_action="Attend event",
        supporting_signals={
            "event_details": {
                "name": "Tech Conference",
                "location": "San Francisco",
                "attendees": ["Alice", "Bob", "Charlie"],
            },
            "relevance_scores": {
                "topic_match": 0.9,
                "contact_overlap": 0.7,
                "timing_score": 0.8,
            },
            "past_events": [
                {"name": "Event 1", "attended": True, "rating": 4},
                {"name": "Event 2", "attended": False, "rating": None},
            ],
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    stored_signals = json.loads(row["supporting_signals"])

    assert isinstance(stored_signals, dict)
    assert stored_signals["event_details"]["name"] == "Tech Conference"
    assert len(stored_signals["event_details"]["attendees"]) == 3
    assert stored_signals["relevance_scores"]["topic_match"] == 0.9
    assert len(stored_signals["past_events"]) == 2


def test_supporting_signals_with_special_characters(user_model_store):
    """Verify supporting_signals handle special characters correctly."""
    prediction = Prediction(
        prediction_type="reminder",
        description="Follow up with José",
        confidence=0.75,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="2_hours",
        suggested_action="Send email",
        supporting_signals={
            "contact_email": "josé@example.com",
            "contact_name": "José María García-López",
            "message_preview": 'Subject: "Re: Project Ñ" — Let\'s discuss!',
            "special_chars": "Symbols: €£¥ émojis: 🎉🚀",
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    stored_signals = json.loads(row["supporting_signals"])

    assert stored_signals["contact_email"] == "josé@example.com"
    assert stored_signals["contact_name"] == "José María García-López"
    assert "Ñ" in stored_signals["message_preview"]
    assert "🎉🚀" in stored_signals["special_chars"]


def test_supporting_signals_backward_compatibility(user_model_store):
    """Verify reading old predictions with list-type supporting_signals doesn't crash."""
    # Simulate old prediction with list-type supporting_signals (legacy bug)
    prediction_id = "legacy-prediction-123"

    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, was_surfaced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "reminder",
                "Legacy prediction",
                0.8,
                "default",
                "2_hours",
                "Test action",
                json.dumps([]),  # OLD BUG: stored as empty list
                False,
            ),
        )

    # Read it back
    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction_id,),
        ).fetchone()

    stored_signals = json.loads(row["supporting_signals"])

    # Should be a list (legacy format)
    assert isinstance(stored_signals, list)
    assert stored_signals == []


def test_reminder_supporting_signals_structure(user_model_store):
    """Verify reminder predictions have the correct supporting_signals structure for behavioral tracking."""
    prediction = Prediction(
        prediction_type="reminder",
        description='Unreplied message from alice@example.com: "Project update" (5 hours ago)',
        confidence=0.85,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="2_hours",
        suggested_action="Reply to alice@example.com",
        supporting_signals={
            "contact_email": "alice@example.com",
            "contact_name": "alice",
            "message_id": "msg-456",
            "hours_since_received": 5.0,
            "is_priority_contact": True,
            "requires_response": True,
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    signals = json.loads(row["supporting_signals"])

    # Verify all required fields for behavioral accuracy tracking
    assert "contact_email" in signals
    assert "contact_name" in signals
    assert "message_id" in signals
    assert "hours_since_received" in signals
    assert "is_priority_contact" in signals
    assert "requires_response" in signals

    # Verify types
    assert isinstance(signals["contact_email"], str)
    assert isinstance(signals["hours_since_received"], (int, float))
    assert isinstance(signals["is_priority_contact"], bool)


def test_conflict_supporting_signals_structure(user_model_store):
    """Verify conflict predictions have correct supporting_signals structure."""
    prediction = Prediction(
        prediction_type="conflict",
        description="Calendar overlap: 'Meeting A' and 'Meeting B' overlap by 30 minutes",
        confidence=0.95,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
        suggested_action="Reschedule one event",
        supporting_signals={
            "conflicting_event_ids": ["event-123", "event-456"],
            "overlap_minutes": 30,
            "event_titles": ["Meeting A", "Meeting B"],
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    signals = json.loads(row["supporting_signals"])

    assert "conflicting_event_ids" in signals
    assert len(signals["conflicting_event_ids"]) == 2
    assert signals["overlap_minutes"] == 30


def test_supporting_signals_with_null_values(user_model_store):
    """Verify supporting_signals handle None/null values correctly."""
    prediction = Prediction(
        prediction_type="need",
        description="User may need help",
        confidence=0.6,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="this_week",
        suggested_action="Check in",
        supporting_signals={
            "contact_email": "bob@example.com",
            "contact_name": None,  # Name not available
            "last_interaction": None,
            "priority_score": 0.5,
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    signals = json.loads(row["supporting_signals"])

    assert isinstance(signals, dict)
    assert signals["contact_email"] == "bob@example.com"
    assert signals["contact_name"] is None
    assert signals["last_interaction"] is None
    assert signals["priority_score"] == 0.5


def test_supporting_signals_numeric_types(user_model_store):
    """Verify supporting_signals preserve numeric types (int vs float)."""
    prediction = Prediction(
        prediction_type="risk",
        description="Risk detected",
        confidence=0.8,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="24_hours",
        suggested_action="Review",
        supporting_signals={
            "integer_count": 42,
            "float_score": 0.75,
            "large_number": 1234567890,
            "decimal_precision": 3.14159,
            "zero_int": 0,
            "zero_float": 0.0,
        },
    )

    user_model_store.store_prediction(prediction.model_dump())

    with user_model_store.db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?",
            (prediction.id,),
        ).fetchone()

    signals = json.loads(row["supporting_signals"])

    # JSON doesn't distinguish int/float, but values should be preserved
    assert signals["integer_count"] == 42
    assert signals["float_score"] == 0.75
    assert signals["large_number"] == 1234567890
    assert abs(signals["decimal_precision"] - 3.14159) < 0.00001
    assert signals["zero_int"] == 0
    assert signals["zero_float"] == 0.0


def test_batch_predictions_with_supporting_signals(user_model_store):
    """Verify multiple predictions with different supporting_signals structures."""
    predictions = [
        Prediction(
            prediction_type="reminder",
            description="Reminder 1",
            confidence=0.8,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="2_hours",
            supporting_signals={"contact_email": "user1@example.com"},
        ),
        Prediction(
            prediction_type="conflict",
            description="Conflict 1",
            confidence=0.95,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            supporting_signals={"conflicting_event_ids": ["e1", "e2"]},
        ),
        Prediction(
            prediction_type="opportunity",
            description="Opportunity 1",
            confidence=0.7,
            confidence_gate=ConfidenceGate.SUGGEST,
            time_horizon="this_week",
            supporting_signals={"event_name": "Conference", "relevance": 0.9},
        ),
    ]

    # Store all predictions
    for pred in predictions:
        user_model_store.store_prediction(pred.model_dump())

    # Verify all were stored with dict-type supporting_signals
    with user_model_store.db.get_connection("user_model") as conn:
        rows = conn.execute(
            "SELECT id, prediction_type, supporting_signals FROM predictions WHERE id IN (?, ?, ?)",
            (predictions[0].id, predictions[1].id, predictions[2].id),
        ).fetchall()

    assert len(rows) == 3

    for row in rows:
        signals = json.loads(row["supporting_signals"])
        assert isinstance(signals, dict), f"Prediction {row['id']} has non-dict supporting_signals"

        if row["prediction_type"] == "reminder":
            assert "contact_email" in signals
        elif row["prediction_type"] == "conflict":
            assert "conflicting_event_ids" in signals
        elif row["prediction_type"] == "opportunity":
            assert "event_name" in signals
