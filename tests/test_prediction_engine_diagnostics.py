"""
Tests for prediction engine diagnostic logging.

Verifies that all prediction methods emit structured log records with
observability metrics (candidates evaluated, filtering reasons, etc.)
so we can identify why prediction types aren't generating predictions.

Migration note (iteration 198): assertions updated from capsys→caplog after
print() calls were replaced with logger.debug() / logger.warning().  The
[prediction_engine.xxx] prefix in log messages was also dropped because the
logging module includes the logger name automatically in structured records.
"""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from models.core import Priority
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager, UserModelStore


@pytest.mark.asyncio
async def test_calendar_conflicts_diagnostic_no_events(db, user_model_store, caplog):
    """Verify diagnostic logging when no calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_calendar_conflicts({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "calendar_conflicts" in log_text
    assert "0 calendar events found" in log_text or "No conflicts possible" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_calendar_conflicts_diagnostic_only_all_day_events(db, user_model_store, caplog):
    """Verify diagnostic logging when only all-day events exist (no conflicts possible)."""
    engine = PredictionEngine(db, user_model_store)

    # Create 3 all-day events in the 48h window
    now = datetime.now(timezone.utc)
    for i in range(3):
        event_date = (now + timedelta(days=i)).date().isoformat()
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"evt-{i}",
                    "calendar.event.created",
                    "caldav",
                    now.isoformat(),
                    Priority.NORMAL,
                    json.dumps({
                        "event_id": f"cal-{i}",
                        "title": f"All-day event {i}",
                        "start_time": event_date,
                        "end_time": event_date,
                        "is_all_day": True,
                        "attendees": [],
                    }),
                ),
            )
            conn.commit()

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_calendar_conflicts({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "calendar_conflicts" in log_text
    assert "Analyzed 3 synced events" in log_text
    assert "all_day=3" in log_text
    assert "timed=0" in log_text
    assert "skipped" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_routine_deviations_diagnostic_no_routines(db, user_model_store, caplog):
    """Verify diagnostic logging when no routines exist."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_routine_deviations({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "routine_deviations" in log_text
    assert "0 routines with consistency_score > 0.6" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_routine_deviations_diagnostic_with_routine(db, user_model_store, caplog):
    """Verify diagnostic logging when routines exist."""
    engine = PredictionEngine(db, user_model_store)

    # Create a routine with high consistency
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            ("morning_coffee", "time:08:00", json.dumps([{"action": "email_received"}]), 0.8, 10),
        )
        conn.commit()

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_routine_deviations({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "routine_deviations" in log_text
    assert "Analyzed 1 routines" in log_text


@pytest.mark.asyncio
async def test_relationship_maintenance_diagnostic_no_profile(db, user_model_store, caplog):
    """Verify diagnostic logging when relationships profile doesn't exist."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_relationship_maintenance({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "relationship_maintenance" in log_text
    assert "relationships profile not found" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_relationship_maintenance_diagnostic_with_contacts(db, user_model_store, caplog):
    """Verify diagnostic logging when relationship profile has contacts."""
    engine = PredictionEngine(db, user_model_store)

    # Create a relationships profile with some contacts
    now = datetime.now(timezone.utc)
    profile_data = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 10,
                "outbound_count": 5,
                "last_interaction": (now - timedelta(days=5)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=i)).isoformat() for i in range(10)
                ],
            },
            "noreply@marketing.com": {
                "interaction_count": 2,
                "last_interaction": now.isoformat(),
            },
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_relationship_maintenance({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "relationship_maintenance" in log_text
    assert "Analyzed 2 contacts" in log_text
    assert "marketing_filtered=" in log_text


@pytest.mark.asyncio
async def test_preparation_needs_diagnostic_no_events(db, user_model_store, caplog):
    """Verify diagnostic logging when no calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_preparation_needs({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "preparation_needs" in log_text
    assert "0 calendar events found" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_preparation_needs_diagnostic_with_events(db, user_model_store, caplog):
    """Verify diagnostic logging when calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    # Create a flight event in the 12-48h window
    now = datetime.now(timezone.utc)
    event_time = (now + timedelta(hours=24)).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "evt-1",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({
                    "event_id": "cal-1",
                    "title": "Flight to SFO",
                    "start_time": event_time,
                    "end_time": (now + timedelta(hours=26)).isoformat(),
                    "is_all_day": False,
                    "attendees": [],
                }),
            ),
        )
        conn.commit()

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_preparation_needs({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "preparation_needs" in log_text
    assert "Analyzed 1 synced events" in log_text
    assert "travel=" in log_text


@pytest.mark.asyncio
async def test_spending_patterns_diagnostic_no_transactions(db, user_model_store, caplog):
    """Verify diagnostic logging when no finance transactions exist."""
    engine = PredictionEngine(db, user_model_store)

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_spending_patterns({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "spending_patterns" in log_text
    assert "0 transactions found" in log_text
    assert "need ≥5" in log_text
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_spending_patterns_diagnostic_with_transactions(db, user_model_store, caplog):
    """Verify diagnostic logging when finance transactions exist."""
    engine = PredictionEngine(db, user_model_store)

    # Create some finance transactions
    now = datetime.now(timezone.utc)
    for i in range(10):
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"txn-{i}",
                    "finance.transaction.new",
                    "finance",
                    (now - timedelta(days=i)).isoformat(),
                    Priority.NORMAL,
                    json.dumps({
                        "amount": 50.0,
                        "category": "groceries" if i < 5 else "utilities",
                    }),
                ),
            )
            conn.commit()

    with caplog.at_level(logging.DEBUG, logger="services.prediction_engine.engine"):
        predictions = await engine._check_spending_patterns({})

    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "spending_patterns" in log_text
    assert "Analyzed 10 transactions" in log_text
    assert "categories=" in log_text
