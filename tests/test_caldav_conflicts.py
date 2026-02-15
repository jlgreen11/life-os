"""
Tests for CalDAV conflict detection.

Verifies that the CalDAV connector correctly identifies overlapping
calendar events and publishes calendar.conflict.detected events.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from connectors.caldav.connector import CalDAVConnector


@pytest.mark.asyncio
async def test_detect_conflicts_with_overlapping_events(db, event_store):
    """Conflict detection should identify two overlapping events."""
    # Create a CalDAV connector instance with mocked dependencies
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
        "calendars": ["Personal"],
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    # Inject the publish_event method to track published events
    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    # Create two overlapping calendar events
    now = datetime.now(timezone.utc)

    # Event 1: 10:00 - 11:00
    event1_id = str(uuid.uuid4())
    event_store.store_event({
        "id": event1_id,
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-1",
            "calendar_id": "Personal",
            "title": "Team Meeting",
            "start_time": (now + timedelta(hours=1)).isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
            "is_all_day": False,
            "location": "Conference Room A",
        }),
        "metadata": json.dumps({}),
    })

    # Event 2: 10:30 - 11:30 (overlaps with Event 1)
    event2_id = str(uuid.uuid4())
    event_store.store_event({
        "id": event2_id,
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-2",
            "calendar_id": "Personal",
            "title": "Client Call",
            "start_time": (now + timedelta(hours=1, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "is_all_day": False,
            "location": "Phone",
        }),
        "metadata": json.dumps({}),
    })

    # Run conflict detection
    await connector._detect_conflicts()

    # Verify a conflict event was published
    assert len(published_events) == 1
    conflict_event = published_events[0]

    assert conflict_event["type"] == "calendar.conflict.detected"
    assert conflict_event["priority"] == "high"

    payload = conflict_event["payload"]
    assert "event1" in payload
    assert "event2" in payload
    assert payload["event1"]["title"] == "Team Meeting"
    assert payload["event2"]["title"] == "Client Call"
    assert "overlap_start" in payload
    assert "overlap_end" in payload


@pytest.mark.asyncio
async def test_detect_conflicts_no_overlap(db, event_store):
    """Conflict detection should not flag non-overlapping events."""
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    # Create two non-overlapping events
    now = datetime.now(timezone.utc)

    # Event 1: 10:00 - 11:00
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-1",
            "calendar_id": "Personal",
            "title": "Morning Meeting",
            "start_time": (now + timedelta(hours=1)).isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Event 2: 14:00 - 15:00 (no overlap)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-2",
            "calendar_id": "Personal",
            "title": "Afternoon Meeting",
            "start_time": (now + timedelta(hours=5)).isoformat(),
            "end_time": (now + timedelta(hours=6)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run conflict detection
    await connector._detect_conflicts()

    # Verify no conflict events were published
    assert len(published_events) == 0


@pytest.mark.asyncio
async def test_detect_conflicts_skips_all_day_events(db, event_store):
    """All-day events should not trigger conflict detection."""
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    now = datetime.now(timezone.utc)

    # Create two all-day events on the same day
    for i in range(2):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "calendar.event.created",
            "source": "caldav",
            "timestamp": now.isoformat(),
            "payload": json.dumps({
                "event_id": f"cal-event-{i}",
                "calendar_id": "Personal",
                "title": f"All Day Event {i}",
                "start_time": now.date().isoformat(),
                "end_time": now.date().isoformat(),
                "is_all_day": True,
            }),
            "metadata": json.dumps({}),
        })

    # Run conflict detection
    await connector._detect_conflicts()

    # All-day events should be skipped, so no conflicts
    assert len(published_events) == 0


@pytest.mark.asyncio
async def test_detect_conflicts_handles_edge_touching(db, event_store):
    """Events that touch at boundaries (end1 == start2) should not conflict."""
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    now = datetime.now(timezone.utc)

    # Event 1: 10:00 - 11:00
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-1",
            "calendar_id": "Personal",
            "title": "First Meeting",
            "start_time": (now + timedelta(hours=1)).isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Event 2: 11:00 - 12:00 (starts exactly when Event 1 ends)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-2",
            "calendar_id": "Personal",
            "title": "Second Meeting",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run conflict detection
    await connector._detect_conflicts()

    # Back-to-back events should not be flagged as conflicts
    assert len(published_events) == 0


@pytest.mark.asyncio
async def test_detect_conflicts_with_multiple_overlaps(db, event_store):
    """Should detect multiple conflicts when three events all overlap."""
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    now = datetime.now(timezone.utc)

    # Event 1: 10:00 - 12:00
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-1",
            "calendar_id": "Personal",
            "title": "Long Meeting",
            "start_time": (now + timedelta(hours=1)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Event 2: 10:30 - 11:00 (overlaps with Event 1)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-2",
            "calendar_id": "Personal",
            "title": "Quick Call",
            "start_time": (now + timedelta(hours=1, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Event 3: 11:15 - 11:45 (overlaps with both Event 1 and Event 2)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": json.dumps({
            "event_id": "cal-event-3",
            "calendar_id": "Personal",
            "title": "Another Call",
            "start_time": (now + timedelta(hours=2, minutes=15)).isoformat(),
            "end_time": (now + timedelta(hours=2, minutes=45)).isoformat(),
            "is_all_day": False,
        }),
        "metadata": json.dumps({}),
    })

    # Run conflict detection
    await connector._detect_conflicts()

    # Should detect 2 conflicts: (1,2) and (1,3)
    # Event 2 (10:30-11:00) and Event 3 (11:15-11:45) don't overlap
    assert len(published_events) == 2

    # All should be calendar.conflict.detected events
    for event in published_events:
        assert event["type"] == "calendar.conflict.detected"
        assert event["priority"] == "high"


@pytest.mark.asyncio
async def test_detect_conflicts_with_no_events(db, event_store):
    """Conflict detection should handle empty calendar gracefully."""
    mock_event_bus = MagicMock()
    mock_event_bus.publish = AsyncMock()

    config = {
        "url": "https://test.example.com",
        "username": "test@example.com",
        "password": "test",
    }

    connector = CalDAVConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config
    )

    published_events = []

    async def track_publish(event_type, payload, **kwargs):
        published_events.append({
            "type": event_type,
            "payload": payload,
            **kwargs
        })

    connector.publish_event = track_publish

    # Run conflict detection with no events
    await connector._detect_conflicts()

    # Should complete without error and publish nothing
    assert len(published_events) == 0
