"""
Tests for ContextAssembler._get_calendar_context()

Verifies that the calendar context method correctly queries upcoming
calendar events from the events database instead of returning the old
placeholder string.

Coverage areas:
1. No events → returns "none" message
2. Today's timed event appears with start_time and title
3. All-day event is formatted with [all-day] prefix (no time noise)
4. Location is appended for timed events when available
5. All-day events never include location noise
6. Events beyond 7 days are excluded
7. Duplicate (title, start_time) rows are deduplicated (recurring syncs)
8. Events are sorted by start_time ascending
9. Cap of 20 events is respected
10. Non-calendar events are not included
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from services.ai_engine.context import ContextAssembler
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_cal_event(
    db,
    title: str,
    start_time: str,
    end_time: str | None = None,
    is_all_day: bool = False,
    location: str | None = None,
) -> None:
    """Insert a calendar.event.created event into the events table."""
    payload = json.dumps(
        {
            "event_id": f"evt_{title.replace(' ', '_')}_{start_time}",
            "calendar_id": "primary",
            "title": title,
            "description": None,
            "location": location,
            "start_time": start_time,
            "end_time": end_time or start_time,
            "is_all_day": is_all_day,
            "attendees": [],
            "organizer": "test@example.com",
        }
    )
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (type, source, timestamp, priority, payload, metadata)
               VALUES ('calendar.event.created', 'google', datetime('now'), 'normal', ?, '{}')""",
            (payload,),
        )


def _make_assembler(db) -> ContextAssembler:
    """Construct a ContextAssembler with a minimal UserModelStore."""
    ums = UserModelStore(db)
    return ContextAssembler(db, ums)


def _today(offset: int = 0) -> str:
    """Return date string for today + offset days, formatted YYYY-MM-DD."""
    return (date.today() + timedelta(days=offset)).isoformat()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_events_returns_none_message(db):
    """When no calendar events exist, context should indicate 'none'."""
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "none" in result.lower()
    # Must NOT return the old placeholder
    assert "connect CalDAV" not in result


def test_timed_event_today_appears(db):
    """A timed event starting today should appear in the calendar context."""
    _insert_cal_event(db, "Team Standup", _today(), _today(), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Team Standup" in result
    assert _today() in result
    # Timed events should NOT have [all-day] prefix
    assert "[all-day]" not in result


def test_all_day_event_has_all_day_prefix(db):
    """All-day events should be formatted with [all-day] prefix."""
    _insert_cal_event(db, "Company Holiday", _today(1), _today(2), is_all_day=True)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Company Holiday" in result
    assert "[all-day]" in result


def test_location_included_for_timed_event(db):
    """Location field should be appended for timed events when present."""
    _insert_cal_event(
        db,
        "Doctor Appointment",
        _today(2),
        _today(2),
        is_all_day=False,
        location="City Medical Center",
    )
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Doctor Appointment" in result
    assert "City Medical Center" in result


def test_all_day_event_no_location_noise(db):
    """All-day events should not include location in their line."""
    _insert_cal_event(
        db,
        "Heather's Birthday",
        _today(3),
        _today(4),
        is_all_day=True,
        location="Some Location",
    )
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    # Location should not appear next to an all-day event
    assert "Heather's Birthday" in result
    # The @ location separator should only appear on timed-event lines
    line = [l for l in result.splitlines() if "Heather's Birthday" in l][0]
    assert "@" not in line


def test_events_beyond_7_days_excluded(db):
    """Events starting more than 7 days out should not appear."""
    _insert_cal_event(db, "Far Future Meeting", _today(8), _today(8), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Far Future Meeting" not in result


def test_events_on_day_7_included(db):
    """Events starting exactly 7 days from today should be included (boundary)."""
    _insert_cal_event(db, "Week Out Event", _today(7), _today(7), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Week Out Event" in result


def test_deduplication_of_repeat_syncs(db):
    """Duplicate (title, start_time) rows from repeated syncs should appear once."""
    # Insert the same event 5 times (as would happen in repeated syncs)
    for _ in range(5):
        _insert_cal_event(db, "Recurring Meeting", _today(1), _today(1), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    # Should only appear once
    assert result.count("Recurring Meeting") == 1


def test_events_sorted_by_start_time(db):
    """Events should be listed in ascending start_time order."""
    _insert_cal_event(db, "Later Event", _today(5), _today(5), is_all_day=False)
    _insert_cal_event(db, "Earlier Event", _today(1), _today(1), is_all_day=False)
    _insert_cal_event(db, "Middle Event", _today(3), _today(3), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    earlier_idx = result.index("Earlier Event")
    middle_idx = result.index("Middle Event")
    later_idx = result.index("Later Event")
    assert earlier_idx < middle_idx < later_idx


def test_cap_at_20_events(db):
    """No more than 20 events should appear even if more exist."""
    for i in range(25):
        _insert_cal_event(db, f"Event {i:02d}", _today(0), _today(0), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    # Count the bullet lines
    bullet_lines = [l for l in result.splitlines() if l.startswith("- ")]
    assert len(bullet_lines) <= 20


def test_non_calendar_events_excluded(db):
    """email.received and other event types should never appear."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (type, source, timestamp, priority, payload, metadata)
               VALUES ('email.received', 'proton', datetime('now'), 'normal',
                       '{"subject": "Hello", "from_address": "x@example.com"}', '{}')"""
        )
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "email.received" not in result
    assert "Hello" not in result


def test_old_placeholder_never_returned(db):
    """The old hardcoded placeholder must never be returned regardless of data."""
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()
    assert "connect CalDAV to populate" not in result


def test_context_header_present(db):
    """The context string should include the section header."""
    _insert_cal_event(db, "Morning Yoga", _today(0), _today(0), is_all_day=False)
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()

    assert "Upcoming calendar events" in result


def test_untitled_event_handled(db):
    """Events with null title should appear as (untitled) not crash."""
    payload = json.dumps(
        {
            "event_id": "no_title_evt",
            "calendar_id": "primary",
            "title": None,
            "start_time": _today(2),
            "end_time": _today(2),
            "is_all_day": False,
            "attendees": [],
            "organizer": "test@example.com",
        }
    )
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (type, source, timestamp, priority, payload, metadata)
               VALUES ('calendar.event.created', 'google', datetime('now'), 'normal', ?, '{}')""",
            (payload,),
        )
    assembler = _make_assembler(db)
    result = assembler._get_calendar_context()
    assert "(untitled)" in result
