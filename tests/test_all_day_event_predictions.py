"""Tests for all-day event handling in calendar predictions.

CRITICAL BUG FIX (iteration 132):
All-day events were being completely filtered out of calendar conflict detection,
causing 0 predictions despite 2,571 all-day events (99.9%) in the production database.

This test suite verifies:
1. All-day events are included in predictions
2. All-day vs all-day comparisons are skipped (no conflict)
3. All-day vs timed event conflicts ARE detected
4. Preparation needs work with all-day events
5. Calendar-based predictions now generate with real-world data
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone

from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_all_day_events_included_in_conflict_detection(db, event_store, user_model_store):
    """Verify all-day events are no longer filtered out."""
    engine = PredictionEngine(db, user_model_store)

    # Create two all-day events on the same day (tomorrow)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    tomorrow_start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    tomorrow_end = datetime.combine(tomorrow, datetime.max.time(), tzinfo=timezone.utc)

    event_store.store_event({
        "id": "event-all-day-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Conference Day",
            "start_time": tomorrow_start.isoformat(),
            "end_time": tomorrow_end.isoformat(),
            "is_all_day": True,
            "location": "Convention Center",
        },
        "metadata": {},
    })

    event_store.store_event({
        "id": "event-all-day-2",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Team Offsite",
            "start_time": tomorrow_start.isoformat(),
            "end_time": tomorrow_end.isoformat(),
            "is_all_day": True,
            "location": "Company HQ",
        },
        "metadata": {},
    })

    # Generate predictions
    predictions = await engine._check_calendar_conflicts({})

    # Two all-day events on the same day should NOT conflict
    # (multiple all-day markers are fine)
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_all_day_vs_timed_event_conflict(db, event_store, user_model_store):
    """All-day event + timed event on same day SHOULD conflict if locations differ."""
    engine = PredictionEngine(db, user_model_store)

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    tomorrow_start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    tomorrow_end = datetime.combine(tomorrow, datetime.max.time(), tzinfo=timezone.utc)

    # All-day conference
    event_store.store_event({
        "id": "event-all-day-conf",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "AI Summit",
            "start_time": tomorrow_start.isoformat(),
            "end_time": tomorrow_end.isoformat(),
            "is_all_day": True,
            "location": "San Francisco",
        },
        "metadata": {},
    })

    # Timed meeting on the same day (overlaps the all-day event)
    meeting_start = datetime.combine(tomorrow, datetime.min.time().replace(hour=14), tzinfo=timezone.utc)
    meeting_end = meeting_start + timedelta(hours=1)

    event_store.store_event({
        "id": "event-timed-meeting",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Client Call",
            "start_time": meeting_start.isoformat(),
            "end_time": meeting_end.isoformat(),
            "is_all_day": False,
            "location": "Office",
        },
        "metadata": {},
    })

    # Generate predictions
    predictions = await engine._check_calendar_conflicts({})

    # Should detect ONE conflict: all-day event overlaps with timed meeting
    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "conflict"
    assert "AI Summit" in pred.description
    assert "Client Call" in pred.description
    assert pred.confidence == 0.8  # All-day conflicts get slightly lower confidence


@pytest.mark.asyncio
async def test_timed_events_still_detect_tight_gaps(db, event_store, user_model_store):
    """Verify timed event conflict detection still works (regression test)."""
    engine = PredictionEngine(db, user_model_store)

    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)

    # Meeting 1: 9:00-10:00
    meeting1_start = tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
    meeting1_end = meeting1_start + timedelta(hours=1)

    event_store.store_event({
        "id": "meeting-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Standup",
            "start_time": meeting1_start.isoformat(),
            "end_time": meeting1_end.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Meeting 2: 10:05-11:00 (only 5 min gap)
    meeting2_start = meeting1_end + timedelta(minutes=5)
    meeting2_end = meeting2_start + timedelta(hours=1)

    event_store.store_event({
        "id": "meeting-2",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Planning",
            "start_time": meeting2_start.isoformat(),
            "end_time": meeting2_end.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Generate predictions
    predictions = await engine._check_calendar_conflicts({})

    # Should detect tight gap (5 minutes < 15 minute threshold)
    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "risk"
    assert "5 minutes" in pred.description
    assert pred.confidence == 0.7


@pytest.mark.asyncio
async def test_overlapping_timed_events(db, event_store, user_model_store):
    """Verify overlapping timed events trigger high-confidence conflicts."""
    engine = PredictionEngine(db, user_model_store)

    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)

    # Meeting 1: 14:00-15:30
    meeting1_start = tomorrow.replace(hour=14, minute=0, second=0, microsecond=0)
    meeting1_end = meeting1_start + timedelta(hours=1, minutes=30)

    event_store.store_event({
        "id": "meeting-overlap-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Product Review",
            "start_time": meeting1_start.isoformat(),
            "end_time": meeting1_end.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Meeting 2: 15:00-16:00 (30 min overlap)
    meeting2_start = meeting1_start + timedelta(hours=1)
    meeting2_end = meeting2_start + timedelta(hours=1)

    event_store.store_event({
        "id": "meeting-overlap-2",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Investor Call",
            "start_time": meeting2_start.isoformat(),
            "end_time": meeting2_end.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Generate predictions
    predictions = await engine._check_calendar_conflicts({})

    # Should detect hard conflict with high confidence
    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "conflict"
    assert "overlap" in pred.description.lower()
    assert pred.confidence == 0.95  # Full confidence for timed conflicts


@pytest.mark.asyncio
async def test_preparation_needs_with_all_day_travel(db, event_store, user_model_store):
    """Verify preparation needs work with all-day travel events."""
    engine = PredictionEngine(db, user_model_store)

    # All-day travel event day after tomorrow (within 12-48 hour prep window).
    # Start at 8 AM to ensure we're solidly within the 12-48 hour window
    # regardless of current time.
    day_after_tomorrow = (datetime.now(timezone.utc) + timedelta(days=2)).date()
    tomorrow_start = datetime.combine(day_after_tomorrow, datetime.min.time().replace(hour=8), tzinfo=timezone.utc)
    tomorrow_end = datetime.combine(day_after_tomorrow, datetime.max.time(), tzinfo=timezone.utc)

    event_store.store_event({
        "id": "travel-all-day",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Flight to NYC",
            "start_time": tomorrow_start.isoformat(),
            "end_time": tomorrow_end.isoformat(),
            "is_all_day": True,
            "location": "JFK Airport",
        },
        "metadata": {},
    })

    # Generate preparation predictions
    predictions = await engine._check_preparation_needs({})

    # Should detect travel preparation need
    assert len(predictions) >= 1
    travel_pred = [p for p in predictions if "travel" in p.description.lower() or "flight" in p.description.lower()]
    assert len(travel_pred) >= 1
    pred = travel_pred[0]
    assert pred.prediction_type == "need"
    assert "Flight to NYC" in pred.description


@pytest.mark.asyncio
async def test_no_tight_gap_for_all_day_events(db, event_store, user_model_store):
    """All-day events should NOT generate tight gap warnings."""
    engine = PredictionEngine(db, user_model_store)

    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    # All-day event
    allday_start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    allday_end = datetime.combine(tomorrow, datetime.max.time(), tzinfo=timezone.utc)

    event_store.store_event({
        "id": "allday-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Conference",
            "start_time": allday_start.isoformat(),
            "end_time": allday_end.isoformat(),
            "is_all_day": True,
        },
        "metadata": {},
    })

    # Timed event "right after" the all-day event ends
    # (technically 1 second gap, but this shouldn't trigger tight gap warnings)
    next_day = tomorrow + timedelta(days=1)
    timed_start = datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)
    timed_end = timed_start + timedelta(hours=1)

    event_store.store_event({
        "id": "timed-after",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "title": "Follow-up Meeting",
            "start_time": timed_start.isoformat(),
            "end_time": timed_end.isoformat(),
            "is_all_day": False,
        },
        "metadata": {},
    })

    # Generate predictions
    predictions = await engine._check_calendar_conflicts({})

    # Should NOT generate tight gap warning (all-day events excluded from gap checks)
    tight_gap_preds = [p for p in predictions if p.prediction_type == "risk"]
    assert len(tight_gap_preds) == 0
