"""
Smoke tests for the PredictionEngine.

Verifies that the engine can be instantiated with real database
fixtures and that its basic attributes are set correctly.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


def test_prediction_engine_initializes(db, user_model_store):
    """PredictionEngine can be created with a real DB and UserModelStore."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    assert engine.db is db
    assert engine.ums is user_model_store


@pytest.mark.asyncio
async def test_prediction_engine_skips_when_no_new_events(db, event_store, user_model_store):
    """Prediction engine should return empty list when no new events since last run."""
    engine = PredictionEngine(db, user_model_store)

    # Engine must track its event cursor
    assert hasattr(engine, "_last_event_cursor"), "Engine must have _last_event_cursor attribute"

    # First run with no events — should return empty and set cursor
    predictions = await engine.generate_predictions({})
    assert predictions == []

    # Second run with still no new events — should skip via _has_new_events gate
    assert hasattr(engine, "_has_new_events"), "Engine must have _has_new_events method"
    assert engine._has_new_events() is False, "Should report no new events on second check"
    predictions = await engine.generate_predictions({})
    assert predictions == []


@pytest.mark.asyncio
async def test_prediction_engine_runs_when_new_events_exist(db, event_store, user_model_store):
    """Prediction engine should run when new events exist since last cursor."""
    engine = PredictionEngine(db, user_model_store)

    # First run sets cursor
    await engine.generate_predictions({})

    # Add a new event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "boss@company.com", "subject": "Urgent", "message_id": "msg-1"},
        "metadata": {},
    })

    # Engine should detect new events and run (not skip)
    assert engine._has_new_events() is True, "Should detect the new event"
    predictions = await engine.generate_predictions({})
    # No assertion on length — just that it ran without error
