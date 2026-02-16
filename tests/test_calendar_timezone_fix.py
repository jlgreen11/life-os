"""
Test suite for calendar prediction timezone-naive datetime bug fix.

This test suite verifies that iteration 128's fix resolves the critical bug where
all-day calendar events (stored as date-only strings like "2026-02-14") were
being parsed into timezone-naive datetimes, causing comparison errors with
timezone-aware datetimes and breaking ALL calendar-based predictions.

The fix ensures that any timezone-naive datetime parsed from event payloads
is converted to UTC-aware datetimes before comparison.
"""

import uuid
import pytest
from datetime import datetime, timedelta, timezone
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_calendar_conflicts_with_allday_events(db, event_store, user_model_store):
    """
    Test that all-day events (date-only format) don't cause timezone comparison errors.

    Before the fix, date-only strings like "2026-02-14" would parse into timezone-naive
    datetimes, causing the comparison `start_dt >= now and start_dt <= lookahead` to
    raise: "can't compare offset-naive and offset-aware datetimes"

    After the fix, naive datetimes are converted to UTC-aware before comparison.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create two all-day events in the next 48 hours with overlapping dates
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date().isoformat()
    day_after = (now + timedelta(days=2)).date().isoformat()

    # All-day event 1
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "event1",
            "calendar_id": "primary",
            "title": "All-Day Event 1",
            "description": "",
            "location": "",
            "start_time": tomorrow,  # Date-only format
            "end_time": day_after,   # Date-only format
            "is_all_day": True,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    # All-day event 2 (overlapping)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "event2",
            "calendar_id": "primary",
            "title": "All-Day Event 2",
            "description": "",
            "location": "",
            "start_time": tomorrow,  # Same date — should be filtered out as all-day
            "end_time": day_after,
            "is_all_day": True,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    # Before the fix, this would raise a timezone comparison error.
    # After the fix, it should run without error (though all-day events are skipped).
    try:
        predictions = await engine._check_calendar_conflicts(context)
        # All-day events are intentionally skipped in conflict detection (line 261)
        assert len(predictions) == 0
    except TypeError as e:
        if "can't compare offset-naive and offset-aware" in str(e):
            pytest.fail(f"Timezone comparison bug not fixed: {e}")
        raise


@pytest.mark.asyncio
async def test_calendar_conflicts_with_timed_events(db, event_store, user_model_store):
    """
    Test that timed events (full ISO timestamps) work correctly.

    This verifies the fix doesn't break the existing functionality for
    properly-formatted ISO timestamps with timezone info.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event 1: starts in 2 hours, lasts 1 hour
    event1_start = (now + timedelta(hours=2)).isoformat()
    event1_end = (now + timedelta(hours=3)).isoformat()

    # Event 2: starts in 2.5 hours (30 min overlap with event 1)
    event2_start = (now + timedelta(hours=2.5)).isoformat()
    event2_end = (now + timedelta(hours=4)).isoformat()

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "event1",
            "calendar_id": "primary",
            "title": "Meeting 1",
            "description": "",
            "location": "Office",
            "start_time": event1_start,
            "end_time": event1_end,
            "is_all_day": False,
            "attendees": ["alice@example.com"],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "event2",
            "calendar_id": "primary",
            "title": "Meeting 2",
            "description": "",
            "location": "Office",
            "start_time": event2_start,
            "end_time": event2_end,
            "is_all_day": False,
            "attendees": ["bob@example.com"],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    predictions = await engine._check_calendar_conflicts(context)

    # Should detect the overlap
    assert len(predictions) == 1
    assert predictions[0].prediction_type == "conflict"
    assert "overlap" in predictions[0].description.lower()


@pytest.mark.asyncio
async def test_preparation_needs_with_allday_events(db, event_store, user_model_store):
    """
    Test that preparation needs predictions handle all-day events correctly.

    The same timezone-naive bug affected _check_preparation_needs on line 772.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date().isoformat()
    day_after = (now + timedelta(days=2)).date().isoformat()

    # All-day event with travel keyword
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "trip",
            "calendar_id": "primary",
            "title": "Flight to NYC",  # Travel keyword
            "description": "Prepare luggage",
            "location": "Airport",
            "start_time": tomorrow,  # Date-only format
            "end_time": day_after,
            "is_all_day": True,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    # Before the fix, this would raise a timezone comparison error.
    # After the fix, it should run without error.
    try:
        predictions = await engine._check_preparation_needs(context)
        # The event is all-day but that doesn't exclude it from prep needs.
        # However, it needs to be 12-48 hours out. If tomorrow is < 12 hours, no prediction.
        # The important thing is NO timezone error.
        assert isinstance(predictions, list)
    except TypeError as e:
        if "can't compare offset-naive and offset-aware" in str(e):
            pytest.fail(f"Timezone comparison bug not fixed in preparation_needs: {e}")
        raise


@pytest.mark.asyncio
async def test_preparation_needs_with_timed_flight(db, event_store, user_model_store):
    """
    Test that preparation needs correctly triggers for timed travel events.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Flight in 24 hours (within 12-48 hour prep window)
    flight_start = (now + timedelta(hours=24)).isoformat()
    flight_end = (now + timedelta(hours=27)).isoformat()

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "flight",
            "calendar_id": "primary",
            "title": "Flight to San Francisco",  # Travel keyword
            "description": "SFO departure",
            "location": "Airport",
            "start_time": flight_start,
            "end_time": flight_end,
            "is_all_day": False,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    predictions = await engine._check_preparation_needs(context)

    # Should generate a preparation prediction for the flight
    assert len(predictions) == 1
    assert predictions[0].prediction_type == "need"
    assert "flight" in predictions[0].description.lower()
    assert predictions[0].confidence == 0.75


@pytest.mark.asyncio
async def test_mixed_allday_and_timed_events(db, event_store, user_model_store):
    """
    Test that a mix of all-day (date-only) and timed (ISO) events works correctly.

    This is the most realistic scenario — a calendar with both types of events.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date().isoformat()
    day_after = (now + timedelta(days=2)).date().isoformat()

    # All-day event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "allday",
            "calendar_id": "primary",
            "title": "Birthday",
            "description": "",
            "location": "",
            "start_time": tomorrow,
            "end_time": day_after,
            "is_all_day": True,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    # Timed event on the same day
    timed_start = (now + timedelta(hours=26)).isoformat()
    timed_end = (now + timedelta(hours=27)).isoformat()

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "meeting",
            "calendar_id": "primary",
            "title": "Team Sync",
            "description": "",
            "location": "Zoom",
            "start_time": timed_start,
            "end_time": timed_end,
            "is_all_day": False,
            "attendees": ["team@example.com"],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    # Should handle both without timezone errors
    try:
        predictions = await engine._check_calendar_conflicts(context)
        # All-day events are skipped, so only the timed event is considered
        # (and one event can't conflict with itself)
        assert isinstance(predictions, list)
    except TypeError as e:
        if "can't compare offset-naive and offset-aware" in str(e):
            pytest.fail(f"Timezone comparison bug with mixed event types: {e}")
        raise


@pytest.mark.asyncio
async def test_unparseable_dates_are_skipped(db, event_store, user_model_store):
    """
    Test that events with completely unparseable dates are skipped gracefully.
    """
    engine = PredictionEngine(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Event with malformed date
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "event_id": "bad",
            "calendar_id": "primary",
            "title": "Bad Event",
            "description": "",
            "location": "",
            "start_time": "not-a-date",  # Invalid
            "end_time": "also-invalid",
            "is_all_day": False,
            "attendees": [],
            "organizer": "user@example.com"
        },
        "metadata": {}
    })

    context = {"timestamp": now.isoformat(), "location": None}

    # Should not crash — invalid events are skipped
    predictions = await engine._check_calendar_conflicts(context)
    assert isinstance(predictions, list)
    assert len(predictions) == 0
