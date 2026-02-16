"""
Tests for spatial profile backfill script.

Verifies that the backfill script correctly processes historical events
through the SpatialExtractor and populates the spatial profile with
location-based behavioral patterns, visit frequencies, and dominant activities.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta

from scripts.backfill_spatial_profile import backfill_spatial_profile
from models.core import EventType


def test_backfill_spatial_profile_processes_calendar_events_with_location(db, user_model_store, tmp_path):
    """Backfill should process all calendar events with location data."""
    # Create calendar events with location data
    events = [
        # Calendar event with office location
        {
            "id": "evt-cal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Team meeting",
                "location": "Conference Room A, 123 Main St",
                "start_time": "2026-02-15T14:00:00Z",
                "end_time": "2026-02-15T15:00:00Z",
                "attendees": ["alice@company.com", "bob@company.com"],
            },
            "metadata": {},
        },
        # Calendar event with different location
        {
            "id": "evt-cal-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T09:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Client meeting",
                "location": "Starbucks, 456 Oak Ave",
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T11:30:00Z",
            },
            "metadata": {},
        },
        # Calendar event without location (should be skipped)
        {
            "id": "evt-cal-3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Zoom call",
                "start_time": "2026-02-17T14:00:00Z",
                "end_time": "2026-02-17T15:00:00Z",
            },
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
    result = backfill_spatial_profile(data_dir=db.data_dir, batch_size=100)

    # Verify only events with location were processed
    assert result["events_processed"] == 2
    assert result["signals_extracted"] >= 2  # At least one signal per event
    assert result["errors"] == 0

    # Verify spatial profile was created
    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None
    assert profile["samples_count"] > 0

    # Verify profile data structure
    data = profile["data"]
    assert "place_behaviors" in data


def test_backfill_spatial_profile_normalizes_location_strings(db, user_model_store):
    """Backfill should normalize location strings to group similar places."""
    # Create events with similar location strings (different formats)
    events = [
        {
            "id": "evt-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 1",
                "location": "Conference Room A",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        {
            "id": "evt-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 2",
                "location": "conference room a",  # Same place, different case
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T11:00:00Z",
            },
            "metadata": {},
        },
        {
            "id": "evt-3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-12T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 3",
                "location": "CONFERENCE ROOM A",  # Same place, uppercase
                "start_time": "2026-02-17T10:00:00Z",
                "end_time": "2026-02-17T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify all events were normalized to the same location
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    # Should have only 1 unique place (all normalized to "conference room a")
    assert len(place_behaviors) == 1
    normalized_location = list(place_behaviors.keys())[0]
    assert normalized_location == "conference room a"
    assert place_behaviors[normalized_location]["visit_count"] == 3


def test_backfill_spatial_profile_tracks_visit_counts(db, user_model_store):
    """Backfill should track visit count for each location."""
    # Create multiple events at the same location
    events = [
        {
            "id": f"evt-{i}",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": f"2026-02-{10+i:02d}T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": f"Meeting {i}",
                "location": "Office Building A",
                "start_time": f"2026-02-{15+i:02d}T10:00:00Z",
                "end_time": f"2026-02-{15+i:02d}T11:00:00Z",
            },
            "metadata": {},
        }
        for i in range(5)
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify visit count
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    office_location = "office building a"
    assert office_location in place_behaviors
    assert place_behaviors[office_location]["visit_count"] == 5


def test_backfill_spatial_profile_calculates_average_duration(db, user_model_store):
    """Backfill should calculate average duration per visit from calendar events."""
    # Create events with different durations
    events = [
        # 60-minute meeting
        {
            "id": "evt-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 1",
                "location": "Conference Room B",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        # 30-minute meeting
        {
            "id": "evt-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 2",
                "location": "Conference Room B",
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T10:30:00Z",
            },
            "metadata": {},
        },
        # 90-minute meeting
        {
            "id": "evt-3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-12T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting 3",
                "location": "Conference Room B",
                "start_time": "2026-02-17T10:00:00Z",
                "end_time": "2026-02-17T11:30:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify average duration: (60 + 30 + 90) / 3 = 60 minutes
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    location = "conference room b"
    assert location in place_behaviors
    assert place_behaviors[location]["average_duration_minutes"] == 60.0
    assert place_behaviors[location]["total_duration_minutes"] == 180.0


def test_backfill_spatial_profile_detects_dominant_domain(db, user_model_store):
    """Backfill should detect dominant domain (work vs personal) for each location."""
    # Create events at a location with different domains
    events = [
        # Work event (has attendees)
        {
            "id": "evt-work-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "title": "Team meeting",
                "location": "Office",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
                "attendees": ["alice@company.com"],
            },
            "metadata": {},
        },
        # Work event (has "meeting" in title)
        {
            "id": "evt-work-2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T10:00:00Z",
            "priority": "normal",
            "payload": {
                "title": "Client meeting",
                "location": "Office",
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T11:00:00Z",
            },
            "metadata": {},
        },
        # Personal event (no attendees, no meeting keyword)
        {
            "id": "evt-personal-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-12T10:00:00Z",
            "priority": "normal",
            "payload": {
                "title": "Lunch",
                "location": "Office",
                "start_time": "2026-02-17T12:00:00Z",
                "end_time": "2026-02-17T13:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify dominant domain is "work" (2 work events vs 1 personal)
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    location = "office"
    assert location in place_behaviors
    assert place_behaviors[location]["dominant_domain"] == "work"
    assert place_behaviors[location]["domain_counts"]["work"] == 2
    assert place_behaviors[location]["domain_counts"]["personal"] == 1


def test_backfill_spatial_profile_tracks_typical_activities(db, user_model_store):
    """Backfill should track typical activities at each location."""
    # All calendar events have activity_type = "calendar_event"
    events = [
        {
            "id": "evt-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Board Room",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify typical activities
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    location = "board room"
    assert location in place_behaviors
    assert "typical_activities" in place_behaviors[location]
    assert "calendar_event" in place_behaviors[location]["typical_activities"]


def test_backfill_spatial_profile_tracks_first_and_last_visit(db, user_model_store):
    """Backfill should track first and last visit timestamps for each location."""
    # Create events spread over time at the same location
    events = [
        {
            "id": "evt-first",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "First visit",
                "location": "Cafe",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        {
            "id": "evt-last",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-20T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Last visit",
                "location": "Cafe",
                "start_time": "2026-02-25T10:00:00Z",
                "end_time": "2026-02-25T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify first/last visit timestamps
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    location = "cafe"
    assert location in place_behaviors
    assert place_behaviors[location]["first_visit"] == "2026-02-10T10:00:00+00:00"
    assert place_behaviors[location]["last_visit"] == "2026-02-20T10:00:00+00:00"


def test_backfill_spatial_profile_handles_multiple_unique_locations(db, user_model_store):
    """Backfill should track multiple unique locations separately."""
    # Create events at different locations
    events = [
        {
            "id": "evt-loc1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Building A",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        {
            "id": "evt-loc2",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Building B",
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T11:00:00Z",
            },
            "metadata": {},
        },
        {
            "id": "evt-loc3",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-12T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Coffee Shop",
                "start_time": "2026-02-17T10:00:00Z",
                "end_time": "2026-02-17T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Verify unique places count
    assert result["unique_places"] == 3

    # Verify each location is tracked separately
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile["data"]["place_behaviors"]
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    assert "building a" in place_behaviors
    assert "building b" in place_behaviors
    assert "coffee shop" in place_behaviors


def test_backfill_spatial_profile_respects_limit(db, user_model_store):
    """Backfill should respect the limit parameter."""
    # Create 10 events
    events = [
        {
            "id": f"evt-{i}",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": f"2026-02-{10+i:02d}T10:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": f"Meeting {i}",
                "location": "Office",
                "start_time": f"2026-02-{15+i:02d}T10:00:00Z",
                "end_time": f"2026-02-{15+i:02d}T11:00:00Z",
            },
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
    result = backfill_spatial_profile(data_dir=db.data_dir, limit=5)

    # Should only process 5 events
    assert result["events_processed"] == 5


def test_backfill_spatial_profile_dry_run_mode(db, user_model_store):
    """Dry run should report what would be done without writing to database."""
    events = [
        {
            "id": "evt-1",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Office",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
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

    # Run in dry-run mode
    result = backfill_spatial_profile(data_dir=db.data_dir, dry_run=True)

    # Should report processing but not create profile
    assert result["dry_run"] is True
    assert result["events_processed"] == 1

    # Profile should NOT be created
    profile = user_model_store.get_signal_profile("spatial")
    assert profile is None


def test_backfill_spatial_profile_ignores_events_without_location(db, user_model_store):
    """Backfill should ignore calendar events without location data."""
    events = [
        # With location (should process)
        {
            "id": "evt-with-loc",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Conference Room",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        # Without location (should ignore)
        {
            "id": "evt-no-loc",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Zoom call",
                "start_time": "2026-02-16T10:00:00Z",
                "end_time": "2026-02-16T11:00:00Z",
            },
            "metadata": {},
        },
        # With empty location (should ignore)
        {
            "id": "evt-empty-loc",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-12T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Phone call",
                "location": "",
                "start_time": "2026-02-17T10:00:00Z",
                "end_time": "2026-02-17T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Should only process the event with location (1 event)
    assert result["events_processed"] == 1
    assert result["signals_extracted"] >= 1


def test_backfill_spatial_profile_handles_malformed_timestamps(db, user_model_store):
    """Backfill should gracefully handle events with malformed start/end times."""
    events = [
        # Valid event
        {
            "id": "evt-valid",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-10T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Office",
                "start_time": "2026-02-15T10:00:00Z",
                "end_time": "2026-02-15T11:00:00Z",
            },
            "metadata": {},
        },
        # Malformed start_time
        {
            "id": "evt-malformed",
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "source": "caldav",
            "timestamp": "2026-02-11T14:00:00Z",
            "priority": "normal",
            "payload": {
                "summary": "Meeting",
                "location": "Office",
                "start_time": "invalid-timestamp",
                "end_time": "2026-02-16T11:00:00Z",
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
    result = backfill_spatial_profile(data_dir=db.data_dir)

    # Both events should be processed (malformed timestamp doesn't prevent location extraction)
    # Duration calculation will fail for malformed event, but location still extracted
    assert result["events_processed"] == 2
    assert result["signals_extracted"] == 2
