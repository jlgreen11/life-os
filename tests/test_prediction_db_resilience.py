"""
Prediction Engine DB Resilience Tests

Verifies that the prediction engine handles user_model.db corruption gracefully,
following the fail-open architecture described in CLAUDE.md. When user_model.db is
corrupted (a real production scenario), checks that:
- Individual check failures don't crash the entire pipeline
- Working checks still produce predictions when others fail
- Error diagnostics are logged with correct check names
- generation_stats correctly records errors
- events.db-only predictions (calendar, follow-up, preparation, spending) continue
  generating despite user_model.db corruption
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_calendar_event(event_store, hours_from_now: float, title: str = "Test Meeting") -> str:
    """Insert a calendar event starting `hours_from_now` in the future.

    Returns the event ID.
    """
    now = datetime.now(timezone.utc)
    start = now + timedelta(hours=hours_from_now)
    end = start + timedelta(hours=1)
    event_id = str(uuid.uuid4())
    event_store.store_event({
        "id": event_id,
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": title,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
        },
        "metadata": {},
    })
    return event_id


def _make_email_event(event_store, hours_ago: float, from_address: str, subject: str) -> str:
    """Insert an inbound email event from `hours_ago` hours in the past.

    Returns the event ID.
    """
    now = datetime.now(timezone.utc)
    event_id = str(uuid.uuid4())
    event_store.store_event({
        "id": event_id,
        "type": "email.received",
        "source": "google",
        "timestamp": (now - timedelta(hours=hours_ago)).isoformat(),
        "payload": {
            "from_address": from_address,
            "subject": subject,
            "snippet": f"Message about {subject}",
            "body_plain": f"Hi, regarding {subject}. Please respond.",
            "message_id": f"msg-{event_id[:8]}",
        },
        "metadata": {},
    })
    return event_id


def _corrupt_user_model_connection(db, original_get_connection):
    """Return a wrapper around db.get_connection that raises DatabaseError for user_model.

    All other database names (events, state, entities, preferences) pass through
    to the real connection.
    """
    def patched_get_connection(db_name):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_get_connection(db_name)
    return patched_get_connection


# -------------------------------------------------------------------------
# Test 1: Routine deviations survives DB error
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routine_deviations_survives_db_error(db, event_store, user_model_store):
    """When user_model.db raises DatabaseError, generate_predictions() should not crash.

    The routine_deviations check queries user_model.db directly for routines.
    When that DB is corrupted, this check should fail gracefully while other
    checks that only use events.db (calendar_conflicts, preparation_needs,
    spending_patterns, follow_up_needs) are still attempted.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Insert overlapping calendar events so calendar_conflicts has work to do
    now = datetime.now(timezone.utc)
    _make_calendar_event(event_store, hours_from_now=3, title="Team meeting")
    _make_calendar_event(event_store, hours_from_now=3.25, title="Client call")

    # Patch get_connection to raise DatabaseError only for user_model
    original_get_connection = db.get_connection
    with mock.patch.object(
        db, "get_connection",
        side_effect=_corrupt_user_model_connection(db, original_get_connection),
    ):
        # Should NOT raise
        predictions = await engine.generate_predictions({})

    # The method should return a list (possibly empty, possibly with calendar predictions)
    assert isinstance(predictions, list)


# -------------------------------------------------------------------------
# Test 2: All checks fail gracefully
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_checks_fail_gracefully(db, event_store, user_model_store):
    """When ALL db connections raise DatabaseError, generate_predictions() should
    return an empty list rather than crashing.

    This simulates total DB corruption where both events.db and user_model.db
    are inaccessible.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Patch get_connection to raise for ALL database names
    with mock.patch.object(
        db, "get_connection",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        # Should NOT raise — each check is wrapped in its own try/except
        predictions = await engine.generate_predictions({})

    assert isinstance(predictions, list)
    # With everything broken, no predictions should be generated
    assert len(predictions) == 0


# -------------------------------------------------------------------------
# Test 3: Events DB checks work despite user_model corruption
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_db_checks_work_despite_user_model_corruption(db, event_store, user_model_store):
    """Calendar-based predictions should still be generated when user_model.db is corrupted.

    The _check_calendar_conflicts() and _check_preparation_needs() methods only
    query events.db. They should continue producing predictions even when
    user_model.db is completely broken.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Insert two overlapping calendar events (conflict detection)
    now = datetime.now(timezone.utc)
    _make_calendar_event(event_store, hours_from_now=3, title="Team standup")
    # Overlap: starts 15 min into the first event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Design review",
            "start_time": (now + timedelta(hours=3, minutes=15)).isoformat(),
            "end_time": (now + timedelta(hours=4, minutes=15)).isoformat(),
        },
        "metadata": {},
    })

    # Insert a travel event in the preparation window (12-48h out)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Flight to NYC",
            "start_time": (now + timedelta(hours=24)).isoformat(),
            "end_time": (now + timedelta(hours=30)).isoformat(),
        },
        "metadata": {},
    })

    # Corrupt user_model.db — only user_model connections raise
    original_get_connection = db.get_connection

    def corrupted_get_connection(db_name):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_get_connection(db_name)

    with mock.patch.object(db, "get_connection", side_effect=corrupted_get_connection):
        # Also mock ums.get_signal_profile to simulate user_model being broken
        with mock.patch.object(
            user_model_store, "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            # Also mock ums.store_prediction to simulate storage failure
            with mock.patch.object(
                user_model_store, "store_prediction",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ):
                predictions = await engine.generate_predictions({})

    # Calendar-based predictions should still be generated because they
    # only query events.db, not user_model.db.
    # At minimum we expect the overlap conflict prediction.
    conflict_preds = [p for p in predictions if p.prediction_type == "conflict"]
    assert len(conflict_preds) >= 1, (
        f"Expected at least 1 calendar conflict prediction despite user_model corruption, "
        f"got {len(conflict_preds)}. All predictions: {[p.prediction_type for p in predictions]}"
    )


# -------------------------------------------------------------------------
# Test 4: Follow-up needs works despite user_model corruption
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_needs_works_despite_user_model_corruption(db, event_store, user_model_store):
    """Follow-up predictions should still work when user_model.db is corrupted.

    The _check_follow_up_needs() method queries events.db for unreplied emails.
    When user_model.db is corrupted, it should:
    - Skip the dedup query (wrapped in try/except) and risk duplicates
    - Skip the priority contact boost (wrapped in try/except)
    - Still produce follow-up predictions from events.db email data
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Insert a real email older than 3 hours that needs follow-up
    _make_email_event(event_store, hours_ago=6, from_address="boss@company.com", subject="Q1 Report needed")

    # Corrupt user_model.db connections
    original_get_connection = db.get_connection

    def corrupted_get_connection(db_name):
        if db_name == "user_model":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_get_connection(db_name)

    with mock.patch.object(db, "get_connection", side_effect=corrupted_get_connection):
        with mock.patch.object(
            user_model_store, "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            with mock.patch.object(
                user_model_store, "store_prediction",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ):
                predictions = await engine.generate_predictions({})

    # Follow-up predictions should still be generated because the core
    # email query uses events.db and the user_model failures are wrapped
    # in try/except blocks.
    followup_preds = [p for p in predictions if p.prediction_type == "reminder"]
    assert len(followup_preds) >= 1, (
        f"Expected at least 1 follow-up reminder despite user_model corruption, "
        f"got {len(followup_preds)}. All predictions: {[p.prediction_type for p in predictions]}"
    )
    # Verify the prediction references the right contact (description uses resolved name, email prefix as fallback)
    assert any("boss" in (p.description or "") for p in followup_preds)


# -------------------------------------------------------------------------
# Test 5: Generation stats record errors correctly
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_stats_records_errors(db, event_store, user_model_store, caplog):
    """generation_stats should show 'error' for checks that fail due to DB corruption
    while showing numeric counts for checks that succeed.

    We corrupt only the _check_routine_deviations method (which queries user_model.db)
    and verify that the generation_stats log line records its error while other
    checks report numeric results.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Insert an email event to trigger follow-up checks and some calendar events
    _make_email_event(event_store, hours_ago=5, from_address="colleague@work.com", subject="Budget review")

    # Make _check_routine_deviations raise a DB error specifically
    original_check = engine._check_routine_deviations

    async def failing_routine_check(ctx):
        raise sqlite3.DatabaseError("database disk image is malformed")

    with mock.patch.object(engine, "_check_routine_deviations", side_effect=failing_routine_check):
        import logging
        with caplog.at_level(logging.INFO, logger="services.prediction_engine.engine"):
            predictions = await engine.generate_predictions({})

    # Check that the generation_stats were logged with routine_deviations as error
    # The log message format is: "Generated predictions by type: {generation_stats}..."
    stats_log = [r for r in caplog.records if "Generated predictions by type" in r.message]
    assert len(stats_log) >= 1, "Expected generation stats log message"

    stats_message = stats_log[0].message
    # routine_deviations should show 'error'
    assert "routine_deviations" in stats_message
    assert "error" in stats_message.split("routine_deviations")[1][:50], (
        f"Expected 'error' in routine_deviations stats, got: {stats_message}"
    )


# -------------------------------------------------------------------------
# Test 6: Prediction store failure doesn't crash
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prediction_store_failure_doesnt_crash(db, event_store, user_model_store):
    """When user_model_store.store_prediction() raises DatabaseError,
    generate_predictions() should still complete without crashing.

    The store_prediction call at the end of the pipeline (line 426) is wrapped
    in a per-prediction try/except. Even if storage fails, the method should
    return the list of generated predictions.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Insert overlapping events to ensure at least some predictions are generated
    now = datetime.now(timezone.utc)
    _make_calendar_event(event_store, hours_from_now=3, title="Morning standup")
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Sprint planning",
            "start_time": (now + timedelta(hours=3, minutes=10)).isoformat(),
            "end_time": (now + timedelta(hours=4, minutes=10)).isoformat(),
        },
        "metadata": {},
    })

    # Also add an email to trigger follow-up predictions
    _make_email_event(event_store, hours_ago=5, from_address="pm@company.com", subject="Sprint blockers")

    # Mock store_prediction to always fail with DatabaseError
    with mock.patch.object(
        user_model_store, "store_prediction",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        # Should NOT raise
        predictions = await engine.generate_predictions({})

    # Predictions should still be returned even though storage failed
    assert isinstance(predictions, list)
