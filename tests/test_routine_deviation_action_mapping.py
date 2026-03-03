"""
Test suite for expanded action-to-event-type mapping in routine deviation detection.

Previously, the event_type_mapping dict in _check_routine_deviations() only covered
7 of 20+ interaction types produced by _classify_interaction_type(). Unmapped actions
fell through to action.replace("_", "."), generating invalid event types that never
matched anything in events.db. This caused false deviation predictions for any routine
involving calendar, finance, call, location, context, or command actions.

This test suite verifies:
1. meeting_scheduled maps to calendar.event.created (not meeting.scheduled)
2. spending maps to finance.transaction.new (not "spending")
3. Unmapped routines still correctly generate deviation predictions
4. time_horizon is a valid ISO datetime string, not the literal string "today"
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager
from storage.user_model_store import UserModelStore


def _insert_routine(db: DatabaseManager, name: str, actions: list[str], consistency: float = 0.8):
    """Helper to insert a routine with given actions into the user_model database."""
    steps = [{"order": i, "action": action} for i, action in enumerate(actions)]
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, "morning", json.dumps(steps), 30.0, consistency, 50, json.dumps([])),
        )


def _insert_event(db: DatabaseManager, event_type: str, payload: dict | None = None):
    """Helper to insert an event with the given type occurring 'now'."""
    now = datetime.now(timezone.utc)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                event_type,
                "test",
                now.isoformat(),
                "normal",
                json.dumps(payload or {}),
                json.dumps({}),
            ),
        )


@pytest.mark.asyncio
async def test_meeting_scheduled_maps_to_calendar_event_created(db: DatabaseManager, user_model_store: UserModelStore):
    """meeting_scheduled action should match calendar.event.created events, not meeting.scheduled."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Planning routine", ["meeting_scheduled"])
    _insert_event(db, "calendar.event.created", {"summary": "Team standup"})

    predictions = await engine._check_routine_deviations({})

    # Should find the calendar event and NOT generate a false deviation
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_spending_maps_to_finance_transaction_new(db: DatabaseManager, user_model_store: UserModelStore):
    """spending action should match finance.transaction.new events, not "spending"."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Budget review routine", ["spending"])
    _insert_event(db, "finance.transaction.new", {"amount": -45.99, "merchant": "Grocery Store"})

    predictions = await engine._check_routine_deviations({})

    # Should find the finance event and NOT generate a false deviation
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_income_maps_to_finance_transaction_new(db: DatabaseManager, user_model_store: UserModelStore):
    """income action should match finance.transaction.new events."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Income tracking routine", ["income"])
    _insert_event(db, "finance.transaction.new", {"amount": 3000.00, "description": "Paycheck"})

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_call_answered_maps_to_call_received(db: DatabaseManager, user_model_store: UserModelStore):
    """call_answered action should match call.received events, not call.answered."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Call routine", ["call_answered"])
    _insert_event(db, "call.received", {"caller": "Mom"})

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_calendar_blocked_maps_to_calendar_event_created(db: DatabaseManager, user_model_store: UserModelStore):
    """calendar_blocked action should match calendar.event.created events."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Time blocking routine", ["calendar_blocked"])
    _insert_event(db, "calendar.event.created", {"summary": "Focus time"})

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_calendar_reviewed_maps_to_calendar_event_updated(db: DatabaseManager, user_model_store: UserModelStore):
    """calendar_reviewed action should match calendar.event.updated events."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Calendar review routine", ["calendar_reviewed"])
    _insert_event(db, "calendar.event.updated", {"summary": "Updated meeting"})

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_user_command_maps_to_system_user_command(db: DatabaseManager, user_model_store: UserModelStore):
    """user_command action should match system.user.command events."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Morning check-in routine", ["user_command"])
    _insert_event(db, "system.user.command", {"command": "briefing"})

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_deviation_generated_when_no_matching_events(db: DatabaseManager, user_model_store: UserModelStore):
    """Deviation prediction should be generated when routine steps have no matching events today."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Expense review routine", ["spending", "calendar_reviewed"])
    # No events inserted — routine not completed

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "routine_deviation"
    assert "Expense review routine" in pred.description
    assert pred.supporting_signals["routine_name"] == "Expense review routine"


@pytest.mark.asyncio
async def test_deviation_with_wrong_event_type_still_fires(db: DatabaseManager, user_model_store: UserModelStore):
    """A routine expecting spending events should still fire if only email events exist."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Finance routine", ["spending"])
    # Insert an unrelated event type — should not satisfy the spending action
    _insert_event(db, "email.received", {"subject": "Bank statement"})

    predictions = await engine._check_routine_deviations({})

    # Should generate a deviation because email.received doesn't match finance.transaction.new
    assert len(predictions) == 1
    assert "Finance routine" in predictions[0].description


@pytest.mark.asyncio
async def test_time_horizon_is_iso_datetime(db: DatabaseManager, user_model_store: UserModelStore):
    """time_horizon on deviation predictions should be a valid ISO datetime string, not 'today'."""
    engine = PredictionEngine(db, user_model_store)

    _insert_routine(db, "Test routine", ["email_received"])
    # No events — will generate a prediction

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 1
    pred = predictions[0]

    # Verify it's NOT the literal string "today"
    assert pred.time_horizon != "today"

    # Verify it parses as a valid ISO datetime
    parsed = datetime.fromisoformat(pred.time_horizon)
    assert parsed.hour == 23
    assert parsed.minute == 59
    assert parsed.second == 59


@pytest.mark.asyncio
async def test_multiple_unmapped_actions_in_one_routine(db: DatabaseManager, user_model_store: UserModelStore):
    """A routine with multiple previously-unmapped actions should correctly resolve all of them."""
    engine = PredictionEngine(db, user_model_store)

    # Routine with 3 actions that were all previously unmapped
    _insert_routine(db, "Complex routine", ["meeting_scheduled", "spending", "user_command"])

    # Insert events matching all three mapped types
    _insert_event(db, "calendar.event.created", {"summary": "Meeting"})
    _insert_event(db, "finance.transaction.new", {"amount": -20.00})
    _insert_event(db, "system.user.command", {"command": "status"})

    predictions = await engine._check_routine_deviations({})

    # All actions satisfied — no deviation
    assert len(predictions) == 0
