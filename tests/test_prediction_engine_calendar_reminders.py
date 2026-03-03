"""
Tests for PredictionEngine._check_calendar_event_reminders().

Validates that the prediction engine generates REMINDER predictions for
upcoming calendar events in the 2-24 hour window, using only events.db
(no signal profiles required).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


def _store_calendar_event(event_store, start_time, title="Test Event", location="", is_all_day=False):
    """Helper to store a calendar event in events.db with the given start_time."""
    event_id = str(uuid.uuid4())
    end_time = start_time + timedelta(hours=1) if not is_all_day else start_time + timedelta(days=1)

    # Format start_time: date-only for all-day, ISO for timed
    if is_all_day:
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
    else:
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()

    event_store.store_event({
        "id": event_id,
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": json.dumps({
            "title": title,
            "start_time": start_str,
            "end_time": end_str,
            "location": location,
            "is_all_day": is_all_day,
        }),
        "metadata": {},
    })
    return event_id


# -------------------------------------------------------------------------
# Core functionality: events in the 2-24h window generate reminders
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upcoming_event_generates_reminder(db, event_store, user_model_store):
    """An event starting in 6 hours should produce a REMINDER prediction."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=6), title="Team Sync")

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "reminder"
    assert "Team Sync" in pred.description
    assert pred.confidence == 0.85
    assert pred.time_horizon == "24_hours"
    assert pred.supporting_signals["event_title"] == "Team Sync"
    assert pred.supporting_signals["calendar_event_id"] is not None
    assert 5.5 <= pred.supporting_signals["hours_until"] <= 6.5


@pytest.mark.asyncio
async def test_multiple_upcoming_events(db, event_store, user_model_store):
    """Multiple events in the window should each get a reminder."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=3), title="Standup")
    _store_calendar_event(event_store, now + timedelta(hours=8), title="Design Review")
    _store_calendar_event(event_store, now + timedelta(hours=20), title="Dinner")

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 3
    titles = {p.supporting_signals["event_title"] for p in predictions}
    assert titles == {"Standup", "Design Review", "Dinner"}


@pytest.mark.asyncio
async def test_location_included_in_description(db, event_store, user_model_store):
    """Events with a location should include it in the description."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(
        event_store, now + timedelta(hours=5),
        title="Lunch Meeting", location="Cafe Milano",
    )

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 1
    assert "Cafe Milano" in predictions[0].description
    assert predictions[0].supporting_signals["location"] == "Cafe Milano"


# -------------------------------------------------------------------------
# Filtering: events outside the 2-24h window are excluded
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_imminent_event_skipped(db, event_store, user_model_store):
    """Events less than 2 hours away should NOT generate reminders (too imminent)."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=1), title="Starting Soon")

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_far_future_event_skipped(db, event_store, user_model_store):
    """Events more than 24 hours away should NOT generate reminders (too far)."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=30), title="Next Week")

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_all_day_event_skipped(db, event_store, user_model_store):
    """All-day events should NOT generate time-based reminders."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(
        event_store, now + timedelta(hours=6),
        title="Holiday", is_all_day=True,
    )

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_past_event_skipped(db, event_store, user_model_store):
    """Events that already happened should NOT generate reminders."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now - timedelta(hours=3), title="Already Done")

    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Deduplication: same event doesn't produce duplicate reminders
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduplication_prevents_duplicate_reminders(db, event_store, user_model_store):
    """An event that already has a reminder prediction should not get another one."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    event_id = _store_calendar_event(event_store, now + timedelta(hours=6), title="Dedup Test")

    # First call should generate a reminder
    predictions_first = await engine._check_calendar_event_reminders({})
    assert len(predictions_first) == 1

    # Store the prediction in the predictions table (simulating what the engine does)
    pred = predictions_first[0]
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred.id, pred.prediction_type, pred.description,
                pred.confidence, pred.confidence_gate.value,
                pred.time_horizon, pred.suggested_action,
                json.dumps(pred.supporting_signals),
                now.isoformat(),
            ),
        )

    # Second call should skip this event (already predicted)
    predictions_second = await engine._check_calendar_event_reminders({})
    assert len(predictions_second) == 0


# -------------------------------------------------------------------------
# Graceful degradation
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_calendar_events_returns_empty(db, event_store, user_model_store):
    """When events.db has no calendar events, method returns empty list gracefully."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    predictions = await engine._check_calendar_event_reminders({})

    assert predictions == []


@pytest.mark.asyncio
async def test_corrupted_dedup_query_still_generates_reminders(db, event_store, user_model_store):
    """If the dedup query against user_model.db fails, reminders are still generated."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=6), title="Resilience Test")

    # Drop the predictions table to simulate a corrupted user_model.db
    with db.get_connection("user_model") as conn:
        conn.execute("DROP TABLE IF EXISTS predictions")

    # Should still generate a reminder (dedup failure is non-fatal)
    predictions = await engine._check_calendar_event_reminders({})

    assert len(predictions) == 1
    assert predictions[0].supporting_signals["event_title"] == "Resilience Test"


# -------------------------------------------------------------------------
# Integration: wired into generate_predictions
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calendar_reminders_in_generate_predictions(db, event_store, user_model_store):
    """Calendar reminders should appear in the full generate_predictions output."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    _store_calendar_event(event_store, now + timedelta(hours=5), title="Integration Test Meeting")

    # generate_predictions should include calendar reminders
    predictions = await engine.generate_predictions({})

    reminder_preds = [p for p in predictions if p.prediction_type == "reminder"]
    # At least one reminder should be generated (may be filtered by reaction scoring,
    # but we check the raw output includes it)
    assert any("Integration Test Meeting" in p.description for p in reminder_preds) or \
        any("Integration Test Meeting" in p.description for p in predictions)
