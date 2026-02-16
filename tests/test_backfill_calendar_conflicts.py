"""
Tests for calendar conflict detection backfill script.

The backfill runs the same sweep-line algorithm as CalDAVConnector._detect_conflicts()
but processes ALL historical calendar events, not just the last 24 hours.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

# Import the backfill script
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.backfill_calendar_conflicts import detect_conflicts_for_all_events
from models.core import EventType


@pytest.mark.asyncio
async def test_backfill_detects_simple_overlap(db, event_bus):
    """Verify backfill detects two overlapping calendar events."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Create two overlapping events:
    # Event 1: 10:00 AM - 11:00 AM
    # Event 2: 10:30 AM - 11:30 AM
    now = datetime.now(timezone.utc)
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = event1_start + timedelta(hours=1)
    event2_start = event1_start + timedelta(minutes=30)
    event2_end = event2_start + timedelta(hours=1)

    # Store first event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Team Standup",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
            "calendar_id": "Work",
        }),
        "metadata": json.dumps({}),
    })

    # Store second event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Client Call",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
            "calendar_id": "Work",
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Verify stats
    assert stats["events_analyzed"] == 2
    assert stats["conflicts_detected"] == 1
    assert stats["events_published"] == 1
    assert stats["parse_errors"] == 0

    # Verify conflict event was published via event_bus
    published_conflicts = [
        e for e in event_bus._published_events
        if e["type"] == "calendar.conflict.detected"
    ]
    assert len(published_conflicts) == 1

    # Verify conflict payload structure
    payload = published_conflicts[0]["payload"]

    assert payload["event1"]["title"] == "Team Standup"
    assert payload["event2"]["title"] == "Client Call"
    assert "overlap_start" in payload
    assert "overlap_end" in payload

    # The overlap should be from 10:30 AM to 11:00 AM (30 minutes)
    overlap_start = datetime.fromisoformat(payload["overlap_start"])
    overlap_end = datetime.fromisoformat(payload["overlap_end"])
    assert overlap_start == event2_start
    assert overlap_end == event1_end


@pytest.mark.asyncio
async def test_backfill_skips_non_overlapping_events(db, event_bus):
    """Verify backfill does not create conflicts for well-spaced events."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Event 1: 10:00 AM - 11:00 AM
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = event1_start + timedelta(hours=1)

    # Event 2: 2:00 PM - 3:00 PM (3 hours later, no overlap)
    event2_start = now.replace(hour=14, minute=0, second=0, microsecond=0)
    event2_end = event2_start + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Morning Meeting",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Afternoon Meeting",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Verify no conflicts detected
    assert stats["events_analyzed"] == 2
    assert stats["conflicts_detected"] == 0
    assert stats["events_published"] == 0

    # Verify no conflict events were published
    published_conflicts = [
        e for e in event_bus._published_events
        if e["type"] == "calendar.conflict.detected"
    ]
    assert len(published_conflicts) == 0


@pytest.mark.asyncio
async def test_backfill_skips_all_day_events(db, event_bus):
    """Verify backfill ignores all-day events when detecting conflicts."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Two all-day events on the same day should NOT conflict
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "all-day-1",
            "title": "Team Offsite",
            "start_time": now.date().isoformat(),
            "end_time": now.date().isoformat(),
            "is_all_day": True,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "all-day-2",
            "title": "Vacation Day",
            "start_time": now.date().isoformat(),
            "end_time": now.date().isoformat(),
            "is_all_day": True,
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # All-day events should be skipped, so no conflicts
    assert stats["events_analyzed"] == 2
    assert stats["conflicts_detected"] == 0
    assert stats["events_published"] == 0


@pytest.mark.asyncio
async def test_backfill_handles_edge_touching_events(db, event_bus):
    """Events that touch at boundaries (end1 == start2) should not conflict."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Event 1: 10:00 AM - 11:00 AM
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = now.replace(hour=11, minute=0, second=0, microsecond=0)

    # Event 2: 11:00 AM - 12:00 PM (starts exactly when Event 1 ends)
    event2_start = event1_end
    event2_end = event2_start + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Morning Session",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Lunch Meeting",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Edge-touching events do NOT overlap (start2 >= end1)
    assert stats["conflicts_detected"] == 0
    assert stats["events_published"] == 0


@pytest.mark.asyncio
async def test_backfill_detects_multiple_overlaps(db, event_bus):
    """Verify backfill detects all pairwise conflicts in a multi-event scenario."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Event 1: 10:00 AM - 12:00 PM
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = event1_start + timedelta(hours=2)

    # Event 2: 11:00 AM - 1:00 PM (overlaps Event 1)
    event2_start = event1_start + timedelta(hours=1)
    event2_end = event2_start + timedelta(hours=2)

    # Event 3: 11:30 AM - 12:30 PM (overlaps both Event 1 and Event 2)
    event3_start = event1_start + timedelta(hours=1, minutes=30)
    event3_end = event3_start + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Workshop Part 1",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Team Sync",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-3",
            "title": "Emergency Call",
            "start_time": event3_start.isoformat(),
            "end_time": event3_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Should detect 3 pairwise conflicts:
    # (Event 1, Event 2), (Event 1, Event 3), (Event 2, Event 3)
    assert stats["events_analyzed"] == 3
    assert stats["conflicts_detected"] == 3
    assert stats["events_published"] == 3


@pytest.mark.asyncio
async def test_backfill_handles_malformed_payloads(db, event_bus):
    """Verify backfill gracefully handles events with missing or invalid fields."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Valid event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "valid-event",
            "title": "Valid Meeting",
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(hours=1)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Malformed event: missing start_time
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "malformed-event",
            "title": "Broken Meeting",
            "end_time": (now + timedelta(hours=1)).isoformat(),
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Should analyze 2 events but skip the malformed one during parsing
    assert stats["events_analyzed"] == 2
    assert stats["conflicts_detected"] == 0
    # Note: parse_errors won't increment because missing fields cause silent skip, not exception


@pytest.mark.asyncio
async def test_backfill_dry_run_mode(db, event_bus):
    """Verify dry_run=True reports conflicts without publishing events."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Two overlapping events
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = event1_start + timedelta(hours=1)
    event2_start = event1_start + timedelta(minutes=30)
    event2_end = event2_start + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Meeting A",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Meeting B",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run in dry-run mode
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=True)

    # Should detect the conflict but NOT publish events
    assert stats["conflicts_detected"] == 1
    assert stats["events_published"] == 0

    # Verify no conflict events were published (dry-run mode)
    published_conflicts = [
        e for e in event_bus._published_events
        if e["type"] == "calendar.conflict.detected"
    ]
    assert len(published_conflicts) == 0


@pytest.mark.asyncio
async def test_backfill_deduplicates_conflicts(db, event_bus):
    """Verify backfill doesn't create duplicate conflicts for the same event pair."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    now = datetime.now(timezone.utc)

    # Two overlapping events
    event1_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
    event1_end = event1_start + timedelta(hours=1)
    event2_start = event1_start + timedelta(minutes=30)
    event2_end = event2_start + timedelta(hours=1)

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-1",
            "title": "Meeting A",
            "start_time": event1_start.isoformat(),
            "end_time": event1_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "event-2",
            "title": "Meeting B",
            "start_time": event2_start.isoformat(),
            "end_time": event2_end.isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run backfill
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Should detect exactly 1 conflict (not 2, despite checking both directions)
    assert stats["conflicts_detected"] == 1
    assert stats["events_published"] == 1

    # Verify only 1 conflict event was published
    published_conflicts = [
        e for e in event_bus._published_events
        if e["type"] == "calendar.conflict.detected"
    ]
    assert len(published_conflicts) == 1


@pytest.mark.asyncio
async def test_backfill_with_no_events(db, event_bus):
    """Verify backfill handles empty database gracefully."""
    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Run backfill on empty database
    stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

    # Should report zero events analyzed
    assert stats["events_analyzed"] == 0
    assert stats["conflicts_detected"] == 0
    assert stats["events_published"] == 0
