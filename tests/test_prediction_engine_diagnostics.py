"""
Tests for prediction engine diagnostic logging.

Verifies that all prediction methods emit diagnostic logs with
observability metrics (candidates evaluated, filtering reasons, etc.)
so we can identify why prediction types aren't generating predictions.
"""

import json
from datetime import datetime, timedelta, timezone
from io import StringIO
import sys

import pytest

from models.core import Priority
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager, UserModelStore


@pytest.mark.asyncio
async def test_calendar_conflicts_diagnostic_no_events(db, user_model_store, capsys):
    """Verify diagnostic logging when no calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_calendar_conflicts({})

    # Should log that no events were found
    captured = capsys.readouterr()
    assert "[prediction_engine.calendar_conflicts]" in captured.out
    assert "0 calendar events found" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_calendar_conflicts_diagnostic_only_all_day_events(db, user_model_store, capsys):
    """Verify diagnostic logging when only all-day events exist (no conflicts possible)."""
    engine = PredictionEngine(db, user_model_store)

    # Create 3 all-day events in the 48h window
    now = datetime.now(timezone.utc)
    for i in range(3):
        event_date = (now + timedelta(days=i)).date().isoformat()
        event = {
            "id": f"evt-{i}",
            "type": "calendar.event.created",
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": Priority.NORMAL,
            "payload": json.dumps({
                "event_id": f"cal-{i}",
                "title": f"All-day event {i}",
                "start_time": event_date,
                "end_time": event_date,
                "is_all_day": True,
                "attendees": [],
            }),
        }
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event["id"], event["type"], event["source"], event["timestamp"],
                 event["priority"], event["payload"]),
            )
            conn.commit()

    predictions = await engine._check_calendar_conflicts({})

    # Should log that all events were all-day and no comparisons were made
    captured = capsys.readouterr()
    assert "[prediction_engine.calendar_conflicts]" in captured.out
    assert "Analyzed 3 synced events" in captured.out
    assert "all_day=3" in captured.out
    assert "timed=0" in captured.out
    assert "skipped" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_routine_deviations_diagnostic_no_routines(db, user_model_store, capsys):
    """Verify diagnostic logging when no routines exist."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_routine_deviations({})

    # Should log that no routines were found
    captured = capsys.readouterr()
    assert "[prediction_engine.routine_deviations]" in captured.out
    assert "0 routines with consistency_score > 0.6" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_routine_deviations_diagnostic_with_routine(db, user_model_store, capsys):
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

    predictions = await engine._check_routine_deviations({})

    # Should log routine analysis even if no predictions generated
    captured = capsys.readouterr()
    assert "[prediction_engine.routine_deviations]" in captured.out
    assert "Analyzed 1 routines" in captured.out


@pytest.mark.asyncio
async def test_relationship_maintenance_diagnostic_no_profile(db, user_model_store, capsys):
    """Verify diagnostic logging when relationships profile doesn't exist."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_relationship_maintenance({})

    # Should log that profile wasn't found
    captured = capsys.readouterr()
    assert "[prediction_engine.relationship_maintenance]" in captured.out
    assert "relationships profile not found" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_relationship_maintenance_diagnostic_with_contacts(db, user_model_store, capsys):
    """Verify diagnostic logging when relationship profile has contacts."""
    engine = PredictionEngine(db, user_model_store)

    # Create a relationships profile with some contacts
    now = datetime.now(timezone.utc)
    profile_data = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 10,
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

    predictions = await engine._check_relationship_maintenance({})

    # Should log contact analysis
    captured = capsys.readouterr()
    assert "[prediction_engine.relationship_maintenance]" in captured.out
    assert "Analyzed 2 contacts" in captured.out
    assert "marketing_filtered=" in captured.out


@pytest.mark.asyncio
async def test_preparation_needs_diagnostic_no_events(db, user_model_store, capsys):
    """Verify diagnostic logging when no calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_preparation_needs({})

    # Should log that no events were found
    captured = capsys.readouterr()
    assert "[prediction_engine.preparation_needs]" in captured.out
    assert "0 calendar events found" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_preparation_needs_diagnostic_with_events(db, user_model_store, capsys):
    """Verify diagnostic logging when calendar events exist."""
    engine = PredictionEngine(db, user_model_store)

    # Create a flight event in the 12-48h window
    now = datetime.now(timezone.utc)
    event_time = (now + timedelta(hours=24)).isoformat()
    event = {
        "id": "evt-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "priority": Priority.NORMAL,
        "payload": json.dumps({
            "event_id": "cal-1",
            "title": "Flight to SFO",
            "start_time": event_time,
            "end_time": (now + timedelta(hours=26)).isoformat(),
            "is_all_day": False,
            "attendees": [],
        }),
    }
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event["id"], event["type"], event["source"], event["timestamp"],
             event["priority"], event["payload"]),
        )
        conn.commit()

    predictions = await engine._check_preparation_needs({})

    # Should log event analysis and travel detection
    captured = capsys.readouterr()
    assert "[prediction_engine.preparation_needs]" in captured.out
    assert "Analyzed 1 synced events" in captured.out
    assert "travel=" in captured.out


@pytest.mark.asyncio
async def test_spending_patterns_diagnostic_no_transactions(db, user_model_store, capsys):
    """Verify diagnostic logging when no finance transactions exist."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_spending_patterns({})

    # Should log that not enough transactions were found
    captured = capsys.readouterr()
    assert "[prediction_engine.spending_patterns]" in captured.out
    assert "0 transactions found" in captured.out
    assert "need ≥5" in captured.out
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_spending_patterns_diagnostic_with_transactions(db, user_model_store, capsys):
    """Verify diagnostic logging when finance transactions exist."""
    engine = PredictionEngine(db, user_model_store)

    # Create some finance transactions
    now = datetime.now(timezone.utc)
    for i in range(10):
        event = {
            "id": f"txn-{i}",
            "type": "finance.transaction.new",
            "source": "finance",
            "timestamp": (now - timedelta(days=i)).isoformat(),
            "priority": Priority.NORMAL,
            "payload": json.dumps({
                "amount": 50.0,
                "category": "groceries" if i < 5 else "utilities",
            }),
        }
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event["id"], event["type"], event["source"], event["timestamp"],
                 event["priority"], event["payload"]),
            )
            conn.commit()

    predictions = await engine._check_spending_patterns({})

    # Should log transaction analysis
    captured = capsys.readouterr()
    assert "[prediction_engine.spending_patterns]" in captured.out
    assert "Analyzed 10 transactions" in captured.out
    assert "categories=" in captured.out
