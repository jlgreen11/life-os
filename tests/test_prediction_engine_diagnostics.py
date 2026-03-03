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
    assert "calendar_conflicts" in log_text and "skipping" in log_text
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
    assert "relationships signal profile unavailable" in log_text
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
    assert "0 calendar events" in log_text and "skipping" in log_text
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


# --- Tests for _count_calendar_event_types helper ---


def test_count_calendar_event_types_empty_db(db, user_model_store):
    """Helper returns (0, 0) when no calendar events exist."""
    engine = PredictionEngine(db, user_model_store)
    all_day, timed = engine._count_calendar_event_types()
    assert all_day == 0
    assert timed == 0


def test_count_calendar_event_types_mixed(db, user_model_store):
    """Helper correctly categorizes a mix of all-day and timed events."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    with db.get_connection("events") as conn:
        # 2 all-day events
        for i in range(2):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"allday-{i}",
                    "calendar.event.created",
                    "caldav",
                    now.isoformat(),
                    Priority.NORMAL,
                    json.dumps({"is_all_day": True, "title": f"All-day {i}"}),
                ),
            )
        # 3 timed events
        for i in range(3):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"timed-{i}",
                    "calendar.event.created",
                    "caldav",
                    now.isoformat(),
                    Priority.NORMAL,
                    json.dumps({"is_all_day": False, "title": f"Timed {i}"}),
                ),
            )
        conn.commit()

    all_day, timed = engine._count_calendar_event_types()
    assert all_day == 2
    assert timed == 3


def test_count_calendar_event_types_missing_is_all_day(db, user_model_store):
    """Events without is_all_day field are counted as timed (the default)."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    with db.get_connection("events") as conn:
        # Valid timed event
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "good-1",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({"is_all_day": False, "title": "Good event"}),
            ),
        )
        # Payload without is_all_day field — should default to timed
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "no-field-1",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({"title": "Event without is_all_day"}),
            ),
        )
        # Valid all-day event
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "good-2",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({"is_all_day": True, "title": "All-day event"}),
            ),
        )
        conn.commit()

    all_day, timed = engine._count_calendar_event_types()
    # Missing is_all_day defaults to falsy, so counted as timed
    assert all_day == 1
    assert timed == 2


# --- Tests for get_diagnostics need/conflict independence ---


@pytest.mark.asyncio
async def test_diagnostics_need_section_timed_count_independent_of_conflict(db, user_model_store):
    """The 'need' diagnostics section should report correct timed_count
    independently of the 'conflict' section — both share the same
    pre-computed helper values from _count_calendar_event_types().

    This is a regression test for a bug where timed_count was computed
    only in the conflict section and reused (potentially stale) in the
    need section.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert 2 timed events and 1 all-day event
    with db.get_connection("events") as conn:
        for i in range(2):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"timed-{i}",
                    "calendar.event.created",
                    "caldav",
                    now.isoformat(),
                    Priority.NORMAL,
                    json.dumps({"is_all_day": False, "title": f"Meeting {i}"}),
                ),
            )
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "allday-0",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({"is_all_day": True, "title": "Holiday"}),
            ),
        )
        conn.commit()

    diagnostics = await engine.get_diagnostics()

    # Conflict section should see the timed events
    conflict_data = diagnostics["prediction_types"]["conflict"]["data_available"]
    assert conflict_data["timed_events"] == 2
    assert conflict_data["all_day_events"] == 1

    # Need section should also see the timed events (not stale 0)
    need_data = diagnostics["prediction_types"]["need"]["data_available"]
    assert need_data["timed_events"] == 2
    assert need_data["total_events"] == 3

    # The need section should NOT have the all-day blocker
    need_blockers = diagnostics["prediction_types"]["need"]["blockers"]
    assert "All events are all-day" not in " ".join(need_blockers)


@pytest.mark.asyncio
async def test_diagnostics_need_section_all_day_only(db, user_model_store):
    """When only all-day events exist, the need section should report the blocker."""
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "allday-0",
                "calendar.event.created",
                "caldav",
                now.isoformat(),
                Priority.NORMAL,
                json.dumps({"is_all_day": True, "title": "Birthday"}),
            ),
        )
        conn.commit()

    diagnostics = await engine.get_diagnostics()

    need_data = diagnostics["prediction_types"]["need"]["data_available"]
    assert need_data["timed_events"] == 0
    assert need_data["total_events"] == 1

    need_blockers = diagnostics["prediction_types"]["need"]["blockers"]
    assert any("all-day" in b.lower() for b in need_blockers)


@pytest.mark.asyncio
async def test_diagnostics_need_section_no_events(db, user_model_store):
    """When no calendar events exist, need section reports appropriate blocker."""
    engine = PredictionEngine(db, user_model_store)

    diagnostics = await engine.get_diagnostics()

    need_data = diagnostics["prediction_types"]["need"]["data_available"]
    assert need_data["timed_events"] == 0
    assert need_data["total_events"] == 0

    need_blockers = diagnostics["prediction_types"]["need"]["blockers"]
    assert any("no calendar events" in b.lower() for b in need_blockers)
