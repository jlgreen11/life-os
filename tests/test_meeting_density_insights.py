"""
Tests for the _meeting_density_insights fix: uses payload start_time
instead of event sync timestamp.

Also covers get_data_sufficiency_report including events-DB correlators.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_calendar_event(
    event_store,
    event_type: str,
    sync_timestamp: datetime,
    start_time: str,
):
    """Insert a calendar event with a specific sync timestamp and payload start_time.

    This lets us decouple the sync time (when the event was written to the log)
    from the actual calendar event start time.
    """
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "test",
        "timestamp": sync_timestamp.isoformat(),
        "priority": 2,
        "payload": {"start_time": start_time, "title": "Test Meeting"},
        "metadata": {},
    })


def _insert_event_no_payload(event_store, event_type: str, timestamp: datetime):
    """Insert a minimal event with no start_time in payload."""
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "test",
        "timestamp": timestamp.isoformat(),
        "priority": 2,
        "payload": {},
        "metadata": {},
    })


# ===========================================================================
# Test 1: Meeting density uses payload start_time, not event timestamp
# ===========================================================================


@pytest.mark.asyncio
async def test_meeting_density_uses_start_time_not_sync_timestamp(db, user_model_store, event_store):
    """Insert events synced on Sunday but with start_times on weekdays.

    The insight should identify the correct peak weekday, NOT Sunday.
    """
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # All events are synced on a Sunday
    sunday_sync = now.replace(hour=3, minute=0, second=0, microsecond=0)
    # Walk back to most recent Sunday
    while sunday_sync.weekday() != 6:
        sunday_sync -= timedelta(days=1)

    # 8 meetings with start_time on Wednesday (spread across 4 weeks)
    for week in range(4):
        wed = sunday_sync - timedelta(weeks=week)
        # Move to the Wednesday before this Sunday
        wed_start = wed - timedelta(days=4)  # Sunday - 4 = Wednesday
        for hour in range(2):
            _insert_calendar_event(
                event_store,
                "calendar.event.created",
                sync_timestamp=sunday_sync - timedelta(weeks=week),
                start_time=wed_start.replace(hour=9 + hour).isoformat(),
            )

    # 2 meetings on Monday and 2 on Friday to reach threshold of 10+
    for week in range(2):
        mon = sunday_sync - timedelta(weeks=week) - timedelta(days=6)  # Monday
        _insert_calendar_event(
            event_store,
            "calendar.event.created",
            sync_timestamp=sunday_sync - timedelta(weeks=week),
            start_time=mon.replace(hour=10).isoformat(),
        )

    for week in range(2):
        fri = sunday_sync - timedelta(weeks=week) - timedelta(days=2)  # Friday
        _insert_calendar_event(
            event_store,
            "calendar.event.created",
            sync_timestamp=sunday_sync - timedelta(weeks=week),
            start_time=fri.replace(hour=14).isoformat(),
        )

    insights = engine._meeting_density_insights()

    # Should detect Wednesday as peak, NOT Sunday
    assert len(insights) == 1
    assert insights[0].entity == "Wednesday"
    assert "Wednesday" in insights[0].summary
    assert insights[0].category == "meeting_density"


# ===========================================================================
# Test 2: Date-only start_time strings (all-day events) are parsed correctly
# ===========================================================================


@pytest.mark.asyncio
async def test_meeting_density_handles_date_only_start_time(db, user_model_store, event_store):
    """All-day events use date-only strings like '2026-03-05'. Verify they parse."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # Find a recent Thursday
    thursday = now
    while thursday.weekday() != 3:
        thursday -= timedelta(days=1)

    # 12 all-day events on Thursdays (date-only format)
    for week in range(4):
        target_date = thursday - timedelta(weeks=week)
        for _ in range(3):
            _insert_calendar_event(
                event_store,
                "calendar.event.created",
                sync_timestamp=now - timedelta(weeks=week),
                start_time=target_date.strftime("%Y-%m-%d"),  # Date-only
            )

    # 2 events on Tuesday (full ISO timestamp) for contrast
    tuesday = now
    while tuesday.weekday() != 1:
        tuesday -= timedelta(days=1)
    for week in range(2):
        target = tuesday - timedelta(weeks=week)
        _insert_calendar_event(
            event_store,
            "calendar.event.created",
            sync_timestamp=now - timedelta(weeks=week),
            start_time=target.replace(hour=10).isoformat(),
        )

    insights = engine._meeting_density_insights()
    assert len(insights) == 1
    assert insights[0].entity == "Thursday"


# ===========================================================================
# Test 3: calendar.event.updated events are included in the count
# ===========================================================================


@pytest.mark.asyncio
async def test_meeting_density_includes_updated_events(db, user_model_store, event_store):
    """calendar.event.updated events should be counted alongside created events."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # Find a recent Tuesday
    tuesday = now
    while tuesday.weekday() != 1:
        tuesday -= timedelta(days=1)

    # 6 created events on Tuesday
    for week in range(3):
        for i in range(2):
            _insert_calendar_event(
                event_store,
                "calendar.event.created",
                sync_timestamp=now - timedelta(weeks=week),
                start_time=(tuesday - timedelta(weeks=week)).replace(hour=9 + i).isoformat(),
            )

    # 4 updated events on Tuesday (these should also count)
    for week in range(2):
        for i in range(2):
            _insert_calendar_event(
                event_store,
                "calendar.event.updated",
                sync_timestamp=now - timedelta(weeks=week),
                start_time=(tuesday - timedelta(weeks=week)).replace(hour=14 + i).isoformat(),
            )

    # 2 events on Thursday to provide contrast
    thursday = now
    while thursday.weekday() != 3:
        thursday -= timedelta(days=1)
    for week in range(2):
        _insert_calendar_event(
            event_store,
            "calendar.event.created",
            sync_timestamp=now - timedelta(weeks=week),
            start_time=(thursday - timedelta(weeks=week)).replace(hour=11).isoformat(),
        )

    insights = engine._meeting_density_insights()
    # 10 Tuesday events (6 created + 4 updated) vs 2 Thursday
    assert len(insights) == 1
    assert insights[0].entity == "Tuesday"


# ===========================================================================
# Test 4: get_data_sufficiency_report includes events-DB correlator keys
# ===========================================================================


@pytest.mark.asyncio
async def test_data_sufficiency_report_includes_events_db_correlators(db, user_model_store):
    """The report must include keys for email and meeting density correlators."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    report = await engine.get_data_sufficiency_report()

    assert "_email_volume_insights" in report
    assert "_email_peak_hour_insights" in report
    assert "_meeting_density_insights" in report

    # With no events, all should report 'no_data'
    for key in ("_email_volume_insights", "_email_peak_hour_insights", "_meeting_density_insights"):
        assert report[key]["status"] == "no_data"
        assert report[key]["count"] == 0
        assert "min_required" in report[key]
        assert "source" in report[key]


@pytest.mark.asyncio
async def test_data_sufficiency_report_events_db_ready(db, user_model_store, event_store):
    """When enough events exist, the correlator status should be 'ready'."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # Insert 50 email events (enough for both email correlators)
    for i in range(50):
        _insert_event_no_payload(event_store, "email.received", now - timedelta(hours=i))

    # Insert 10 calendar events (enough for meeting density)
    for i in range(10):
        _insert_event_no_payload(event_store, "calendar.event.created", now - timedelta(hours=i))

    report = await engine.get_data_sufficiency_report()
    assert report["_email_volume_insights"]["status"] == "ready"
    assert report["_email_peak_hour_insights"]["status"] == "ready"
    assert report["_meeting_density_insights"]["status"] == "ready"


# ===========================================================================
# Test 5: Min-row threshold — fewer than 10 events returns empty
# ===========================================================================


@pytest.mark.asyncio
async def test_meeting_density_below_threshold_returns_empty(db, user_model_store, event_store):
    """With only 5 calendar events (below 10 threshold), no insights are produced."""
    engine = InsightEngine(db, user_model_store, timezone="UTC")
    now = datetime.now(timezone.utc)

    # Find a recent Monday
    monday = now
    while monday.weekday() != 0:
        monday -= timedelta(days=1)

    # Only 5 events — all on Monday to ensure strong signal, but below threshold
    for week in range(5):
        _insert_calendar_event(
            event_store,
            "calendar.event.created",
            sync_timestamp=now - timedelta(weeks=week),
            start_time=(monday - timedelta(weeks=week)).replace(hour=10).isoformat(),
        )

    insights = engine._meeting_density_insights()
    assert insights == []
