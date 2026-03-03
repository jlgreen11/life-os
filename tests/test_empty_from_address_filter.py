"""
Tests for empty/missing from_address filtering in reminder predictions.

Verifies that the prediction engine skips messages with missing or empty
from_address fields, preventing broken predictions with blank sender info
from being created.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from models.core import EventType
from services.prediction_engine.engine import PredictionEngine


async def test_skip_message_with_empty_from_address(db, user_model_store):
    """Test that messages with empty from_address are skipped."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event with EMPTY from_address
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-empty-from",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-empty-from-123",
                    "from_address": "",  # Empty string
                    "subject": "Test Message",
                    "snippet": "This is a test",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create a prediction for empty from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, "Should not create predictions for empty from_address"


async def test_skip_message_with_missing_from_address(db, user_model_store):
    """Test that messages without from_address field are skipped."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event with NO from_address field at all
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-missing-from",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-missing-from-456",
                    # from_address field is completely missing
                    "subject": "Test Message",
                    "snippet": "This is a test",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create a prediction for missing from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, "Should not create predictions for missing from_address"


async def test_skip_message_with_whitespace_only_from_address(db, user_model_store):
    """Test that messages with whitespace-only from_address are skipped."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event with whitespace-only from_address
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-whitespace-from",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-whitespace-from-789",
                    "from_address": "   ",  # Only whitespace
                    "subject": "Test Message",
                    "snippet": "This is a test",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create a prediction for whitespace-only from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, "Should not create predictions for whitespace-only from_address"


async def test_valid_from_address_still_creates_prediction(db, user_model_store):
    """Test that valid from_address still creates predictions normally."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event with valid from_address
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-valid-from",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-valid-from-abc",
                    "from_address": "friend@example.com",  # Valid email
                    "subject": "Let's catch up",
                    "snippet": "Hey, how have you been?",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should create a prediction for valid from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 1, "Should create prediction for valid from_address"

    pred = reminder_predictions[0]
    # Description now uses resolved contact name (email prefix as fallback)
    assert "friend" in pred.description
    assert pred.supporting_signals.get("contact_email") == "friend@example.com"


async def test_empty_from_address_before_marketing_filter(db, user_model_store):
    """Test that empty from_address is checked BEFORE the marketing filter.

    This ensures that malformed events are rejected early and don't trigger
    expensive marketing filter logic.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create an email that would normally be filtered as marketing,
    # but has an empty from_address
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-empty-marketing",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-empty-marketing-xyz",
                    "from_address": "",  # Empty (malformed)
                    "subject": "Amazing Deals Inside!",
                    "snippet": "Click here to unsubscribe",  # Would trigger marketing filter
                    "body_plain": "unsubscribe unsubscribe",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create a prediction (rejected by empty check, not marketing filter)
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, "Should reject empty from_address before marketing filter"


async def test_null_from_address_json(db, user_model_store):
    """Test that JSON null from_address is handled correctly."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event with JSON null from_address
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-null-from",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-null-from-def",
                    "from_address": None,  # JSON null
                    "subject": "Test Message",
                    "snippet": "This is a test",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create a prediction for null from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, "Should not create predictions for null from_address"


async def test_supporting_signals_populated_correctly(db, user_model_store):
    """Test that supporting_signals contains correct contact_email for valid messages."""
    engine = PredictionEngine(db, user_model_store)

    # Create an email.received event
    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    from_addr = "colleague@company.com"
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "evt-colleague",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": "msg-colleague-ghi",
                    "from_address": from_addr,
                    "subject": "Project update",
                    "snippet": "Here's the latest status",
                }),
                json.dumps({"related_contacts": []}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should create a prediction with populated supporting_signals
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 1

    pred = reminder_predictions[0]
    signals = pred.supporting_signals

    # Verify supporting_signals has valid data
    assert signals.get("contact_email") == from_addr, "Should populate contact_email"
    assert signals.get("contact_name") == "colleague", "Should extract contact_name"
    assert signals.get("message_id") == "msg-colleague-ghi", "Should populate message_id"
    assert isinstance(signals.get("hours_since_received"), (int, float)), "Should populate hours"
    assert signals.get("is_priority_contact") is False, "Should set is_priority_contact"


@pytest.mark.parametrize("invalid_addr", [
    "",           # Empty string
    "   ",        # Whitespace only
    "\t\n",       # Tabs and newlines
    None,         # Will be converted to "" by .get()
])
async def test_various_invalid_from_addresses(db, user_model_store, invalid_addr):
    """Test that various forms of invalid from_address are all rejected."""
    engine = PredictionEngine(db, user_model_store)

    event_time = datetime.now(timezone.utc) - timedelta(hours=6)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"evt-invalid-{hash(str(invalid_addr))}",
                EventType.EMAIL_RECEIVED.value,
                "proton_mail",
                event_time.isoformat(),
                "normal",
                json.dumps({
                    "message_id": f"msg-invalid-{hash(str(invalid_addr))}",
                    "from_address": invalid_addr,
                    "subject": "Test",
                    "snippet": "Test",
                }),
                json.dumps({}),
            ),
        )

    # Generate predictions
    predictions = await engine.generate_predictions({"location": "home"})

    # Should NOT create predictions for any invalid from_address
    reminder_predictions = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(reminder_predictions) == 0, f"Should reject from_address: {repr(invalid_addr)}"
