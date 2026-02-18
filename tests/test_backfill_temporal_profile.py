"""
Tests for temporal profile backfill script.

Verifies that the backfill script correctly processes historical events
through the TemporalExtractor and populates the temporal profile with
activity patterns, scheduling preferences, and planning horizons.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta

from scripts.backfill_temporal_profile import backfill_temporal_profile
from models.core import EventType


def test_backfill_temporal_profile_processes_user_initiated_events(db, user_model_store, tmp_path):
    """Backfill should process all user-initiated event types that trigger temporal extraction."""
    # Create a variety of user-initiated events
    events = [
        # Email sent (communication activity)
        {
            "id": "evt-email-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:30:00Z",
            "priority": "normal",
            "payload": {"to_addresses": ["friend@example.com"], "subject": "Hey"},
            "metadata": {},
        },
        # Message sent (communication activity)
        {
            "id": "evt-msg-1",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T15:45:00Z",
            "priority": "normal",
            "payload": {"to": "555-0100", "body": "Running late"},
            "metadata": {},
        },
        # Calendar event created (planning activity)
        {
            "id": "evt-cal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T09:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Team meeting",
                "start_time": "2026-02-15T14:00:00Z",  # Scheduled 5 days in advance
            },
            "metadata": {},
        },
        # Task created (planning activity)
        {
            "id": "evt-task-1",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T10:30:00Z",
            "priority": "normal",
            "payload": {"title": "Review code", "due_date": "2026-02-12"},
            "metadata": {},
        },
        # Task completed (work activity)
        {
            "id": "evt-task-2",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T16:00:00Z",
            "priority": "normal",
            "payload": {"title": "Review code"},
            "metadata": {},
        },
        # User command (command activity)
        {
            "id": "evt-cmd-1",
            "type": EventType.USER_COMMAND.value,
            "source": "cli",
            "timestamp": "2026-02-10T11:00:00Z",
            "priority": "normal",
            "payload": {"command": "show briefing"},
            "metadata": {},
        },
    ]

    # Insert events into events.db
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir, batch_size=100)

    # Verify all events were processed
    assert result["events_processed"] == 6
    assert result["signals_extracted"] >= 6  # At least one signal per event
    assert result["errors"] == 0

    # Verify temporal profile was created
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None
    assert profile["samples_count"] > 0

    # Verify profile data structure
    data = profile["data"]
    assert "activity_by_hour" in data
    assert "activity_by_day" in data
    assert "activity_by_type" in data
    assert "scheduled_hours" in data
    assert "advance_planning_days" in data


def test_backfill_temporal_profile_builds_hourly_activity_patterns(db, user_model_store):
    """Backfill should aggregate activity by hour to detect energy patterns."""
    # Create events at specific hours to test aggregation
    events = [
        # Three events at 2pm (14:00)
        {
            "id": "evt-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:15:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-2",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:30:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-3",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T14:45:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Two events at 3pm (15:00)
        {
            "id": "evt-4",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T15:10:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-5",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T15:20:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # One event at 9am (09:00)
        {
            "id": "evt-6",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T09:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Verify hourly activity counts
    profile = user_model_store.get_signal_profile("temporal")
    activity_by_hour = profile["data"]["activity_by_hour"]

    assert activity_by_hour["14"] == 3  # 2pm had 3 events
    assert activity_by_hour["15"] == 2  # 3pm had 2 events
    assert activity_by_hour["9"] == 1   # 9am had 1 event


def test_backfill_temporal_profile_builds_daily_activity_patterns(db, user_model_store):
    """Backfill should aggregate activity by day of week to detect weekly rhythms."""
    # Create events across different days of the week
    events = [
        # Monday
        {
            "id": "evt-mon-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-16T10:00:00Z",  # Monday
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Wednesday (two events)
        {
            "id": "evt-wed-1",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-18T14:00:00Z",  # Wednesday
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-wed-2",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-18T15:00:00Z",  # Wednesday
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Friday
        {
            "id": "evt-fri-1",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-20T16:00:00Z",  # Friday
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Verify daily activity counts
    profile = user_model_store.get_signal_profile("temporal")
    activity_by_day = profile["data"]["activity_by_day"]

    assert activity_by_day["monday"] == 1
    assert activity_by_day["wednesday"] == 2
    assert activity_by_day["friday"] == 1


def test_backfill_temporal_profile_classifies_activity_types(db, user_model_store):
    """Backfill should classify events into activity types (communication, planning, work)."""
    events = [
        # Communication events
        {
            "id": "evt-comm-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-comm-2",
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": "2026-02-10T11:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Planning events
        {
            "id": "evt-plan-1",
            "type": EventType.TASK_CREATED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T12:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-plan-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T13:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Work events
        {
            "id": "evt-work-1",
            "type": EventType.TASK_COMPLETED.value,
            "source": "web_ui",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Command event
        {
            "id": "evt-cmd-1",
            "type": EventType.USER_COMMAND.value,
            "source": "cli",
            "timestamp": "2026-02-10T15:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Verify activity type classification
    profile = user_model_store.get_signal_profile("temporal")
    activity_by_type = profile["data"]["activity_by_type"]

    assert activity_by_type["communication"] == 2
    assert activity_by_type["planning"] == 2
    assert activity_by_type["work"] == 1
    assert activity_by_type["command"] == 1


def test_backfill_temporal_profile_tracks_advance_planning_horizon(db, user_model_store):
    """Backfill should track advance planning days from calendar event creation."""
    now = datetime.now(timezone.utc)
    events = [
        # Event created 1 day in advance
        {
            "id": "evt-cal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Tomorrow's meeting",
                "start_time": (now + timedelta(days=1)).isoformat(),
            },
            "metadata": {},
        },
        # Event created 7 days in advance
        {
            "id": "evt-cal-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Next week's meeting",
                "start_time": (now + timedelta(days=7)).isoformat(),
            },
            "metadata": {},
        },
        # Event created 30 days in advance
        {
            "id": "evt-cal-3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Next month's meeting",
                "start_time": (now + timedelta(days=30)).isoformat(),
            },
            "metadata": {},
        },
        # Same-day event (advance_days = 0, should be excluded).
        # Using the event timestamp itself as the start_time guarantees
        # advance_days = 0 regardless of the UTC hour.  Previously used
        # "now + 2h", which crosses midnight after 22:00 UTC making
        # advance_days = 1 and causing a spurious fourth entry.
        {
            "id": "evt-cal-4",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": now.isoformat(),
            "priority": "normal",
            "payload": {
                "summary": "Today's meeting",
                "start_time": now.isoformat(),  # Same moment → advance_days always 0
            },
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Verify advance planning days tracking
    profile = user_model_store.get_signal_profile("temporal")
    advance_days = profile["data"]["advance_planning_days"]

    # Should have 3 entries (excluding same-day event)
    assert len(advance_days) == 3
    assert 1 in advance_days
    assert 7 in advance_days
    assert 30 in advance_days
    assert 0 not in advance_days  # Same-day events excluded


def test_backfill_temporal_profile_handles_malformed_timestamps(db, user_model_store):
    """Backfill should gracefully skip events with malformed timestamps."""
    events = [
        # Valid event
        {
            "id": "evt-valid",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Malformed timestamp
        {
            "id": "evt-malformed",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "invalid-timestamp",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Should process both events (both pass can_process check)
    # but only extract signals from the valid one (malformed timestamp fails in extract())
    assert result["events_processed"] == 2
    assert result["signals_extracted"] == 1  # Only one signal from valid event
    # Malformed event fails silently in extract() due to fail-open error handling


def test_backfill_temporal_profile_respects_limit(db, user_model_store):
    """Backfill should respect the limit parameter."""
    # Create 10 events
    events = [
        {
            "id": f"evt-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": f"2026-02-10T{10+i:02d}:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        }
        for i in range(10)
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill with limit=5
    result = backfill_temporal_profile(data_dir=db.data_dir, limit=5)

    # Should only process 5 events
    assert result["events_processed"] == 5


def test_backfill_temporal_profile_dry_run_mode(db, user_model_store):
    """Dry run should report what would be done without writing to database."""
    events = [
        {
            "id": "evt-1",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run in dry-run mode
    result = backfill_temporal_profile(data_dir=db.data_dir, dry_run=True)

    # Should report processing but not create profile
    assert result["dry_run"] is True
    assert result["events_processed"] == 1

    # Profile should NOT be created
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is None


def test_backfill_temporal_profile_ignores_inbound_events(db, user_model_store):
    """Backfill should ignore inbound/passive events that don't represent user activity."""
    events = [
        # User-initiated (should process)
        {
            "id": "evt-sent",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        # Inbound/passive (should ignore)
        {
            "id": "evt-received",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:05:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-msg-received",
            "type": EventType.MESSAGE_RECEIVED.value,
            "source": "imessage",
            "timestamp": "2026-02-10T14:10:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
        {
            "id": "evt-sync",
            "type": EventType.CONNECTOR_SYNC_COMPLETE.value,
            "source": "gmail",
            "timestamp": "2026-02-10T14:15:00Z",
            "priority": "normal",
            "payload": {},
            "metadata": {},
        },
    ]

    # Insert events
    with db.get_connection("events") as conn:
        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    json.dumps(event["payload"]),
                    json.dumps(event["metadata"]),
                ),
            )

    # Run backfill
    result = backfill_temporal_profile(data_dir=db.data_dir)

    # Should only process the email.sent event (1 event)
    assert result["events_processed"] == 1
    assert result["signals_extracted"] >= 1
