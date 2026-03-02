"""
Tests for prediction engine zero-output diagnostic logging.

Verifies that the prediction engine emits INFO-level log messages when
it produces zero predictions, covering the most common silent-failure
scenarios:
  1. Both triggers (event-based and time-based) are inactive
  2. Signal profiles are missing (follow-up needs can't boost priority contacts)
  3. Successful generation logs a completion summary with method breakdown

These tests use the caplog fixture at INFO level — the key improvement is that
operators no longer need to enable DEBUG globally to see why predictions are
not being generated.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_both_triggers_false_logs_skip(db, user_model_store, caplog):
    """When both triggers are inactive, engine should log an INFO-level skip message.

    This is the most common cause of zero predictions in production: the event
    cursor is current and the time-based interval hasn't elapsed. Previously this
    returned [] with no logging at all.
    """
    engine = PredictionEngine(db, user_model_store)

    # First run sets cursor and time-based run timestamp
    await engine.generate_predictions({})
    caplog.clear()

    # Second run immediately — neither trigger should fire
    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        predictions = await engine.generate_predictions({})

    assert predictions == []

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "Prediction engine skipped" in log_text
    assert "no new events" in log_text
    assert "time-based not due" in log_text


@pytest.mark.asyncio
async def test_empty_signal_profiles_logs_warning(db, event_store, user_model_store, caplog):
    """When the relationships signal profile is empty/unavailable, engine should
    log an INFO-level message explaining that priority contact boost is disabled.

    This covers the scenario where the system has events but no signal extraction
    has run yet — the prediction engine generates follow-up predictions but cannot
    boost confidence for priority contacts.
    """
    engine = PredictionEngine(db, user_model_store)

    # Force the time-based trigger to fire
    engine._last_time_based_run = None

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        await engine.generate_predictions({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "relationships signal profile is" in log_text
    assert "priority contact boost disabled" in log_text


@pytest.mark.asyncio
async def test_successful_generation_logs_summary(db, event_store, user_model_store, caplog):
    """Successful generation should log an INFO-level completion summary
    with the total count, method breakdown, and signal profile availability.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Create two overlapping calendar events to trigger a conflict prediction
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Team meeting",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2, minutes=30)).isoformat(),
        "payload": {
            "title": "Client call",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    # Force time-based trigger
    engine._last_time_based_run = None

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        predictions = await engine.generate_predictions({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "Prediction engine completed" in log_text
    assert "predictions generated" in log_text
    assert "Methods:" in log_text
    assert "Signal profiles available:" in log_text


@pytest.mark.asyncio
async def test_generation_stats_logged_at_info(db, event_store, user_model_store, caplog):
    """The per-method generation stats should be logged at INFO level,
    not DEBUG, so operators can see what each check method produced
    without enabling DEBUG globally.
    """
    engine = PredictionEngine(db, user_model_store)
    # Force time-based trigger
    engine._last_time_based_run = None

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        await engine.generate_predictions({})

    # Find the generation stats record specifically
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    stats_records = [r for r in info_records if "Generated predictions by type" in r.getMessage()]
    assert len(stats_records) >= 1, (
        "Expected 'Generated predictions by type' at INFO level, "
        f"but found only these INFO records: {[r.getMessage() for r in info_records]}"
    )


@pytest.mark.asyncio
async def test_calendar_conflicts_empty_events_logs_info(db, user_model_store, caplog):
    """When no calendar events exist, _check_calendar_conflicts should log
    at INFO level so operators can see why zero conflict predictions were generated.
    """
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        predictions = await engine._check_calendar_conflicts({})

    assert predictions == []
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    log_text = " ".join(r.getMessage() for r in info_records)
    assert "calendar_conflicts" in log_text
    assert "skipping" in log_text


@pytest.mark.asyncio
async def test_routine_deviations_empty_logs_info(db, user_model_store, caplog):
    """When no routines exist, _check_routine_deviations should log at INFO."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        predictions = await engine._check_routine_deviations({})

    assert predictions == []
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    log_text = " ".join(r.getMessage() for r in info_records)
    assert "routine_deviations" in log_text
    assert "0 routines" in log_text


@pytest.mark.asyncio
async def test_relationship_maintenance_no_profile_logs_info(db, user_model_store, caplog):
    """When the relationships profile doesn't exist, _check_relationship_maintenance
    should log at INFO level.
    """
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
        predictions = await engine._check_relationship_maintenance({})

    assert predictions == []
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    log_text = " ".join(r.getMessage() for r in info_records)
    assert "relationship_maintenance" in log_text
    assert "unavailable" in log_text
