"""
Test for calendar conflict detection timestamp fix (iteration 107).

CRITICAL BUG: The original _check_calendar_conflicts implementation queried
events by their sync timestamp (event.timestamp) instead of the actual event
start_time in the payload. This caused ALL calendar conflict predictions to
be missed because:

1. CalDAV syncs events in batches (e.g., every 60 seconds)
2. event.timestamp records WHEN the event was synced to the database
3. payload.start_time records WHEN the event actually occurs
4. For future events, timestamp is in the past but start_time is in the future

The broken query:
    WHERE type = 'calendar.event.created'
    AND timestamp > now AND timestamp < now + 48h

This never matches future events because they were synced in the past!

The fix:
    WHERE type = 'calendar.event.created'
    AND timestamp > now - 30 days  # Get all recently synced events

Then parse payload.start_time and filter in Python to find events starting
in the next 48 hours.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine
from storage.event_store import EventStore


@pytest.mark.asyncio
async def test_calendar_conflicts_with_past_sync_timestamps(db, user_model_store):
    """
    Regression test for iteration 107 bug fix.

    Simulates realistic CalDAV sync behavior:
    - Events are synced to the database NOW (timestamp = now)
    - But the actual events start in the FUTURE (start_time = now + hours)

    The original broken implementation would find ZERO conflicts because it
    queried by timestamp (all in the past relative to the 48h lookahead window).

    The fixed implementation correctly finds conflicts by parsing start_time
    from the event payload.
    """
    engine = PredictionEngine(db, user_model_store)
    event_store = EventStore(db)

    # Simulate a CalDAV sync that happened 2 hours ago
    sync_time = datetime.now(timezone.utc) - timedelta(hours=2)

    # Event 1: Starts tomorrow at 2:00 PM, ends at 3:00 PM
    tomorrow_2pm = datetime.now(timezone.utc) + timedelta(days=1, hours=2)
    tomorrow_3pm = tomorrow_2pm + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),  # PAST: when synced
        "payload": {
            "event_id": "evt_001",
            "title": "Team meeting",
            "start_time": tomorrow_2pm.isoformat(),  # FUTURE: actual event
            "end_time": tomorrow_3pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Event 2: Starts tomorrow at 2:30 PM, ends at 3:30 PM
    # This OVERLAPS with Event 1 by 30 minutes
    tomorrow_230pm = tomorrow_2pm + timedelta(minutes=30)
    tomorrow_330pm = tomorrow_230pm + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),  # PAST: when synced
        "payload": {
            "event_id": "evt_002",
            "title": "Client call",
            "start_time": tomorrow_230pm.isoformat(),  # FUTURE: actual event
            "end_time": tomorrow_330pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # BEFORE THE FIX: This would return [] because both events have
    # timestamp < now, so they don't match the WHERE clause
    #
    # AFTER THE FIX: This correctly detects the overlap by parsing
    # start_time from the payload
    predictions = await engine._check_calendar_conflicts({})

    # Assert: Conflict detected
    assert len(predictions) >= 1, \
        "Should detect conflict even when sync timestamps are in the past"

    conflict = predictions[0]
    assert conflict.prediction_type == "conflict", \
        f"Expected conflict, got {conflict.prediction_type}"
    assert conflict.confidence == 0.95
    assert "overlap" in conflict.description.lower()
    assert "30 minutes" in conflict.description
    assert "Team meeting" in conflict.description
    assert "Client call" in conflict.description


@pytest.mark.asyncio
async def test_calendar_conflicts_ignores_past_events(db, user_model_store):
    """
    Events that have already occurred should not trigger conflict predictions.

    Even if the events overlap, if they're in the past (start_time < now),
    they should be filtered out before conflict detection runs.
    """
    engine = PredictionEngine(db, user_model_store)
    event_store = EventStore(db)

    # Events synced recently but occurred yesterday
    sync_time = datetime.now(timezone.utc) - timedelta(hours=1)
    yesterday_2pm = datetime.now(timezone.utc) - timedelta(days=1, hours=-2)
    yesterday_3pm = yesterday_2pm + timedelta(hours=1)
    yesterday_230pm = yesterday_2pm + timedelta(minutes=30)
    yesterday_330pm = yesterday_230pm + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Past meeting 1",
            "start_time": yesterday_2pm.isoformat(),
            "end_time": yesterday_3pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Past meeting 2",
            "start_time": yesterday_230pm.isoformat(),
            "end_time": yesterday_330pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})

    # Assert: No predictions for past events
    assert len(predictions) == 0, \
        "Should not create conflict predictions for events in the past"


@pytest.mark.asyncio
async def test_calendar_conflicts_ignores_all_day_events(db, user_model_store):
    """
    All-day events should not trigger conflict predictions.

    You can have multiple all-day calendar markers (birthdays, holidays, etc.)
    on the same day without it being a scheduling conflict.
    """
    engine = PredictionEngine(db, user_model_store)
    event_store = EventStore(db)

    sync_time = datetime.now(timezone.utc)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    # All-day event 1
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Birthday",
            "start_time": str(tomorrow),  # Date-only format
            "end_time": str(tomorrow),
            "is_all_day": True,
        },
        "metadata": {},
    })

    # All-day event 2 (same day)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Holiday",
            "start_time": str(tomorrow),
            "end_time": str(tomorrow),
            "is_all_day": True,
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})

    # Assert: No conflict for all-day events
    assert len(predictions) == 0, \
        "Should not create conflict predictions for all-day events"


@pytest.mark.asyncio
async def test_calendar_conflicts_detects_tight_transitions_with_past_sync(
    db, user_model_store
):
    """
    Tight transitions (gap < 15 min) should be detected even when sync
    timestamps are in the past.

    This tests the 'risk' prediction type (not 'conflict').
    """
    engine = PredictionEngine(db, user_model_store)
    event_store = EventStore(db)

    sync_time = datetime.now(timezone.utc) - timedelta(hours=3)
    tomorrow_2pm = datetime.now(timezone.utc) + timedelta(days=1, hours=2)
    tomorrow_3pm = tomorrow_2pm + timedelta(hours=1)
    tomorrow_310pm = tomorrow_3pm + timedelta(minutes=10)
    tomorrow_4pm = tomorrow_310pm + timedelta(minutes=50)

    # Event 1: 2:00 PM - 3:00 PM
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Team standup",
            "start_time": tomorrow_2pm.isoformat(),
            "end_time": tomorrow_3pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Event 2: 3:10 PM - 4:00 PM (only 10 min gap)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": sync_time.isoformat(),
        "payload": {
            "title": "Design review",
            "start_time": tomorrow_310pm.isoformat(),
            "end_time": tomorrow_4pm.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    predictions = await engine._check_calendar_conflicts({})

    # Assert: Risk prediction for tight transition
    assert len(predictions) >= 1, \
        "Should detect tight transition even with past sync timestamp"

    risk = predictions[0]
    assert risk.prediction_type == "risk"
    assert risk.confidence == 0.7
    assert "10 minutes between" in risk.description
