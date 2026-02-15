"""
Comprehensive test coverage for CalDAVConnector.

Tests calendar sync, event creation, conflict detection, timezone handling,
attendee parsing, all-day event logic, and error recovery paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from connectors.caldav.connector import CalDAVConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def caldav_config():
    """Minimal CalDAV connector configuration."""
    return {
        "url": "https://calendar.example.com/dav",
        "username": "test@example.com",
        "password": "test-password",
        "sync_interval": 60,
        "calendars": ["Personal", "Work"],
    }


@pytest.fixture
def mock_vevent():
    """Create a mock VEVENT object with common properties."""
    vevent = Mock()
    vevent.uid.value = "test-event-123"
    vevent.summary.value = "Test Meeting"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)
    vevent.description.value = "Test description"
    vevent.location.value = "Conference Room A"

    # Mock attendees as a list
    attendee1 = Mock()
    attendee1.value = "mailto:alice@example.com"
    attendee2 = Mock()
    attendee2.value = "mailto:bob@example.com"
    vevent.attendee = [attendee1, attendee2]

    # Mock organizer
    organizer = Mock()
    organizer.value = "mailto:organizer@example.com"
    vevent.organizer = organizer

    return vevent


@pytest.fixture
def mock_calendar_event(mock_vevent):
    """Create a mock caldav event object."""
    event = Mock()
    vobj = Mock()
    vobj.vevent = mock_vevent
    event.vobject_instance = vobj
    return event


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


def test_initialization(event_bus, db, caldav_config):
    """Test CalDAVConnector initializes with correct configuration."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    assert connector.CONNECTOR_ID == "caldav"
    assert connector.DISPLAY_NAME == "Calendar (CalDAV)"
    assert connector.SYNC_INTERVAL_SECONDS == 60
    assert connector._client is None
    assert connector._calendars == []
    assert connector.config == caldav_config


# ---------------------------------------------------------------------------
# Test: Authentication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_success(event_bus, db, caldav_config):
    """Test successful CalDAV authentication and calendar discovery."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # Mock the caldav library
    mock_calendar1 = Mock()
    mock_calendar1.name = "Personal"
    mock_calendar2 = Mock()
    mock_calendar2.name = "Work"
    mock_calendar3 = Mock()
    mock_calendar3.name = "Shared"

    mock_principal = Mock()
    mock_principal.calendars.return_value = [
        mock_calendar1,
        mock_calendar2,
        mock_calendar3,
    ]

    mock_client = Mock()
    mock_client.principal.return_value = mock_principal

    with patch("caldav.DAVClient", return_value=mock_client):
        result = await connector.authenticate()

    assert result is True
    assert connector._client is not None
    # Should filter to only "Personal" and "Work" calendars
    assert len(connector._calendars) == 2
    assert mock_calendar1 in connector._calendars
    assert mock_calendar2 in connector._calendars
    assert mock_calendar3 not in connector._calendars


@pytest.mark.asyncio
async def test_authenticate_no_filter(event_bus, db):
    """Test authentication without calendar filtering returns all calendars."""
    config = {
        "url": "https://calendar.example.com/dav",
        "username": "test@example.com",
        "password": "test-password",
    }
    connector = CalDAVConnector(event_bus, db, config)

    mock_calendar1 = Mock()
    mock_calendar1.name = "Personal"
    mock_calendar2 = Mock()
    mock_calendar2.name = "Work"

    mock_principal = Mock()
    mock_principal.calendars.return_value = [mock_calendar1, mock_calendar2]

    mock_client = Mock()
    mock_client.principal.return_value = mock_principal

    with patch("caldav.DAVClient", return_value=mock_client):
        result = await connector.authenticate()

    assert result is True
    assert len(connector._calendars) == 2


@pytest.mark.asyncio
async def test_authenticate_failure(event_bus, db, caldav_config):
    """Test authentication failure is handled gracefully."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    with patch("caldav.DAVClient", side_effect=Exception("Connection refused")):
        result = await connector.authenticate()

    assert result is False
    assert connector._client is None


# ---------------------------------------------------------------------------
# Test: Sync - Event Parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_basic_event(event_bus, db, caldav_config, mock_calendar_event):
    """Test syncing a basic calendar event with all standard fields."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # Set up authenticated state
    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [mock_calendar_event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()  # Mark as authenticated

    # Mock the publish_event method
    connector.publish_event = AsyncMock()

    # Mock conflict detection
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1

    # Verify event was published with correct payload
    assert connector.publish_event.call_count == 1
    call_args = connector.publish_event.call_args

    assert call_args[0][0] == "calendar.event.created"
    payload = call_args[0][1]

    assert payload["event_id"] == "test-event-123"
    assert payload["calendar_id"] == "Personal"
    assert payload["title"] == "Test Meeting"
    assert payload["description"] == "Test description"
    assert payload["location"] == "Conference Room A"
    assert payload["is_all_day"] is False
    assert payload["attendees"] == ["alice@example.com", "bob@example.com"]
    assert payload["organizer"] == "organizer@example.com"

    # Verify conflict detection was called
    connector._detect_conflicts.assert_called_once()


@pytest.mark.asyncio
async def test_sync_all_day_event(event_bus, db, caldav_config):
    """Test syncing an all-day event (date object instead of datetime)."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # Create an all-day event (date object has no 'hour' attribute)
    from datetime import date
    vevent = Mock()
    vevent.uid.value = "allday-123"
    vevent.summary.value = "Birthday"
    vevent.dtstart.value = date(2026, 2, 20)  # date object, not datetime
    vevent.dtend.value = date(2026, 2, 21)

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["is_all_day"] is True
    assert payload["title"] == "Birthday"


@pytest.mark.asyncio
async def test_sync_event_without_optional_fields(event_bus, db, caldav_config):
    """Test syncing an event that lacks optional fields like description, location."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # Create minimal event
    vevent = Mock()
    vevent.uid.value = "minimal-123"
    vevent.summary.value = "Quick Call"
    vevent.dtstart.value = datetime(2026, 2, 15, 14, 0, tzinfo=timezone.utc)
    # No dtend — should default to dtstart + 1 hour
    # No description, location, attendees, organizer

    # Remove optional attributes
    delattr(vevent, "dtend")
    delattr(vevent, "description")
    delattr(vevent, "location")
    delattr(vevent, "attendee")
    delattr(vevent, "organizer")

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["title"] == "Quick Call"
    assert payload["description"] is None
    assert payload["location"] is None
    assert payload["attendees"] == []
    assert payload["organizer"] is None

    # Verify default end time (1 hour after start)
    start_time = datetime.fromisoformat(payload["start_time"])
    end_time = datetime.fromisoformat(payload["end_time"])
    assert end_time == start_time + timedelta(hours=1)


@pytest.mark.asyncio
async def test_sync_event_without_uid(event_bus, db, caldav_config):
    """Test syncing an event without a UID falls back to hash."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    vevent = Mock()
    # No UID attribute
    delattr(vevent, "uid")
    vevent.summary.value = "No UID Event"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    # Should have a hash-based event_id
    assert isinstance(payload["event_id"], str)
    assert payload["event_id"] != ""


@pytest.mark.asyncio
async def test_sync_naive_datetime_timezone_normalization(event_bus, db, caldav_config):
    """Test that naive datetimes are normalized to UTC."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # Create event with naive datetimes (no tzinfo)
    vevent = Mock()
    vevent.uid.value = "naive-tz-123"
    vevent.summary.value = "Naive Time Event"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0)  # No tzinfo
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0)    # No tzinfo

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1
    payload = connector.publish_event.call_args[0][1]

    # Should have timezone info in ISO format
    assert "+" in payload["start_time"] or "Z" in payload["start_time"]


@pytest.mark.asyncio
async def test_sync_single_attendee(event_bus, db, caldav_config):
    """Test parsing a single attendee (not a list)."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    vevent = Mock()
    vevent.uid.value = "single-attendee-123"
    vevent.summary.value = "One-on-One"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)

    # Single attendee (not a list)
    attendee = Mock()
    attendee.value = "mailto:alice@example.com"
    vevent.attendee = attendee  # Not a list

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [event]
    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["attendees"] == ["alice@example.com"]


# ---------------------------------------------------------------------------
# Test: Sync - Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_no_client(event_bus, db, caldav_config):
    """Test sync returns 0 when not authenticated."""
    connector = CalDAVConnector(event_bus, db, caldav_config)
    # Don't set _client (not authenticated)

    count = await connector.sync()
    assert count == 0


@pytest.mark.asyncio
async def test_sync_calendar_error_continues(event_bus, db, caldav_config):
    """Test that an error in one calendar doesn't stop sync of others."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # First calendar raises error
    mock_calendar1 = Mock()
    mock_calendar1.name = "Personal"
    mock_calendar1.date_search.side_effect = Exception("Network error")

    # Second calendar succeeds
    vevent = Mock()
    vevent.uid.value = "success-123"
    vevent.summary.value = "Success Event"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)

    vobj = Mock()
    vobj.vevent = vevent
    event = Mock()
    event.vobject_instance = vobj

    mock_calendar2 = Mock()
    mock_calendar2.name = "Work"
    mock_calendar2.date_search.return_value = [event]

    connector._calendars = [mock_calendar1, mock_calendar2]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    # Should still sync the second calendar
    assert count == 1


@pytest.mark.asyncio
async def test_sync_event_parse_error_continues(event_bus, db, caldav_config):
    """Test that an error parsing one event doesn't stop sync of others."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    # First event raises parse error
    bad_event = Mock()
    bad_event.vobject_instance.vevent = None  # Will cause error

    # Second event is valid
    vevent = Mock()
    vevent.uid.value = "good-123"
    vevent.summary.value = "Good Event"
    vevent.dtstart.value = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
    vevent.dtend.value = datetime(2026, 2, 15, 11, 0, tzinfo=timezone.utc)

    vobj = Mock()
    vobj.vevent = vevent
    good_event = Mock()
    good_event.vobject_instance = vobj

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = [bad_event, good_event]

    connector._calendars = [mock_calendar]
    connector._client = Mock()

    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    count = await connector.sync()

    # Should successfully sync the good event
    assert count == 1


# ---------------------------------------------------------------------------
# Test: Event Creation (Execute)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_create_event_success(event_bus, db, caldav_config):
    """Test creating a new calendar event via execute."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_calendar = Mock()
    mock_calendar.save_event = Mock()
    connector._calendars = [mock_calendar]

    params = {
        "title": "New Meeting",
        "start_time": "2026-02-20T10:00:00Z",
        "end_time": "2026-02-20T11:00:00Z",
        "description": "Discuss project",
        "location": "Office",
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "created"
    assert result["title"] == "New Meeting"

    # Verify save_event was called with VCALENDAR
    mock_calendar.save_event.assert_called_once()
    vcal_str = mock_calendar.save_event.call_args[0][0]
    assert "BEGIN:VCALENDAR" in vcal_str
    assert "SUMMARY:New Meeting" in vcal_str
    assert "DTSTART:2026-02-20T10:00:00Z" in vcal_str


@pytest.mark.asyncio
async def test_execute_create_event_minimal(event_bus, db, caldav_config):
    """Test creating event with minimal params (only title and start_time)."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_calendar = Mock()
    mock_calendar.save_event = Mock()
    connector._calendars = [mock_calendar]

    params = {
        "title": "Quick Event",
        "start_time": "2026-02-20T10:00:00Z",
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "created"
    vcal_str = mock_calendar.save_event.call_args[0][0]
    assert "SUMMARY:Quick Event" in vcal_str
    # Should default DTEND to DTSTART
    assert "DTEND:2026-02-20T10:00:00Z" in vcal_str


@pytest.mark.asyncio
async def test_execute_create_event_no_calendars(event_bus, db, caldav_config):
    """Test creating event fails when no calendars are available."""
    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector._calendars = []  # No calendars

    params = {
        "title": "Event",
        "start_time": "2026-02-20T10:00:00Z",
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "error"
    assert "No calendars available" in result["details"]


@pytest.mark.asyncio
async def test_execute_create_event_server_error(event_bus, db, caldav_config):
    """Test creating event handles server errors gracefully."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_calendar = Mock()
    mock_calendar.save_event.side_effect = Exception("Server error")
    connector._calendars = [mock_calendar]

    params = {
        "title": "Event",
        "start_time": "2026-02-20T10:00:00Z",
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "error"
    assert "Server error" in result["details"]


@pytest.mark.asyncio
async def test_execute_unknown_action(event_bus, db, caldav_config):
    """Test execute raises ValueError for unknown actions."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("unknown_action", {})


# ---------------------------------------------------------------------------
# Test: Conflict Detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_conflicts_basic(event_bus, db, caldav_config):
    """Test basic conflict detection between overlapping events."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    # Create two overlapping events in the database
    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Event 1: 10:00-11:00
    payload1 = {
        "event_id": "event-1",
        "title": "Meeting A",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
        "calendar_id": "Personal",
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload1,
    })

    # Event 2: 10:30-11:30 (overlaps with Event 1)
    payload2 = {
        "event_id": "event-2",
        "title": "Meeting B",
        "start_time": "2026-02-15T10:30:00+00:00",
        "end_time": "2026-02-15T11:30:00+00:00",
        "is_all_day": False,
        "calendar_id": "Work",
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload2,
    })

    await connector._detect_conflicts()

    # Should publish one conflict event
    assert connector.publish_event.call_count == 1

    call_args = connector.publish_event.call_args
    assert call_args[0][0] == "calendar.conflict.detected"

    conflict_payload = call_args[0][1]
    assert conflict_payload["event1"]["title"] == "Meeting A"
    assert conflict_payload["event2"]["title"] == "Meeting B"
    assert conflict_payload["overlap_start"] == "2026-02-15T10:30:00+00:00"
    assert conflict_payload["overlap_end"] == "2026-02-15T11:00:00+00:00"


@pytest.mark.asyncio
async def test_detect_conflicts_no_overlap(event_bus, db, caldav_config):
    """Test conflict detection with non-overlapping events."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Event 1: 10:00-11:00
    payload1 = {
        "event_id": "event-1",
        "title": "Meeting A",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload1,
    })

    # Event 2: 11:00-12:00 (no overlap)
    payload2 = {
        "event_id": "event-2",
        "title": "Meeting B",
        "start_time": "2026-02-15T11:00:00+00:00",
        "end_time": "2026-02-15T12:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload2,
    })

    await connector._detect_conflicts()

    # Should not publish any conflict events
    assert connector.publish_event.call_count == 0


@pytest.mark.asyncio
async def test_detect_conflicts_all_day_ignored(event_bus, db, caldav_config):
    """Test that all-day events are excluded from conflict detection."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # All-day event
    payload1 = {
        "event_id": "allday-1",
        "title": "Birthday",
        "start_time": "2026-02-15",
        "end_time": "2026-02-16",
        "is_all_day": True,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload1,
    })

    # Timed event on same day
    payload2 = {
        "event_id": "timed-1",
        "title": "Meeting",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload2,
    })

    await connector._detect_conflicts()

    # Should not detect conflict (all-day events are excluded)
    assert connector.publish_event.call_count == 0


@pytest.mark.asyncio
async def test_detect_conflicts_multiple_overlaps(event_bus, db, caldav_config):
    """Test detecting multiple conflicts with three overlapping events."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Event 1: 10:00-12:00
    payload1 = {
        "event_id": "event-1",
        "title": "Long Meeting",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T12:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload1,
    })

    # Event 2: 10:30-11:00 (overlaps with Event 1)
    payload2 = {
        "event_id": "event-2",
        "title": "Short Meeting",
        "start_time": "2026-02-15T10:30:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload2,
    })

    # Event 3: 11:30-12:30 (overlaps with Event 1)
    payload3 = {
        "event_id": "event-3",
        "title": "Another Meeting",
        "start_time": "2026-02-15T11:30:00+00:00",
        "end_time": "2026-02-15T12:30:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload3,
    })

    await connector._detect_conflicts()

    # Should detect 2 conflicts (1-2 and 1-3)
    assert connector.publish_event.call_count == 2


@pytest.mark.asyncio
async def test_detect_conflicts_no_duplicate_pairs(event_bus, db, caldav_config):
    """Test that the same conflict pair isn't reported twice."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Create same events twice to ensure deduplication works
    for _ in range(2):
        payload1 = {
            "event_id": "event-1",
            "title": "Meeting A",
            "start_time": "2026-02-15T10:00:00+00:00",
            "end_time": "2026-02-15T11:00:00+00:00",
            "is_all_day": False,
        }
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "calendar.event.created",
            "source": "caldav",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload1,
        })

    await connector._detect_conflicts()

    # Should only report one conflict even though events are stored twice
    assert connector.publish_event.call_count == 1


@pytest.mark.asyncio
async def test_detect_conflicts_less_than_two_events(event_bus, db, caldav_config):
    """Test conflict detection with fewer than 2 events (no conflicts possible)."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Only one event
    payload = {
        "event_id": "event-1",
        "title": "Solo Meeting",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    })

    await connector._detect_conflicts()

    # Should not detect any conflicts
    assert connector.publish_event.call_count == 0


@pytest.mark.asyncio
async def test_detect_conflicts_malformed_event_skipped(event_bus, db, caldav_config):
    """Test that malformed events are skipped in conflict detection."""
    import uuid
    from datetime import datetime, timezone

    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    from storage.event_store import EventStore
    event_store = EventStore(db)

    # Malformed event (missing start_time)
    payload1 = {
        "event_id": "bad-event",
        "title": "Bad Event",
        "end_time": "2026-02-15T11:00:00+00:00",
        # Missing start_time
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload1,
    })

    # Valid event
    payload2 = {
        "event_id": "good-event",
        "title": "Good Event",
        "start_time": "2026-02-15T10:00:00+00:00",
        "end_time": "2026-02-15T11:00:00+00:00",
        "is_all_day": False,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload2,
    })

    # Should not crash, should skip malformed event
    await connector._detect_conflicts()

    # No conflicts should be detected (only one valid event)
    assert connector.publish_event.call_count == 0


@pytest.mark.asyncio
async def test_detect_conflicts_error_handled(event_bus, db, caldav_config):
    """Test that errors in conflict detection are handled gracefully."""
    connector = CalDAVConnector(event_bus, db, caldav_config)
    connector.publish_event = AsyncMock()

    # Force an error by mocking EventStore at the module level where it's imported
    with patch("storage.event_store.EventStore") as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_store.get_events.side_effect = Exception("Database error")

        # Should not raise, should handle error gracefully
        await connector._detect_conflicts()

    # No conflicts should be published due to error
    assert connector.publish_event.call_count == 0


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_connected(event_bus, db, caldav_config):
    """Test health check when connected."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_principal = Mock()
    mock_client = Mock()
    mock_client.principal.return_value = mock_principal

    mock_calendar = Mock()
    connector._calendars = [mock_calendar]
    connector._client = mock_client

    result = await connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "caldav"
    assert result["calendars"] == 1


@pytest.mark.asyncio
async def test_health_check_not_connected(event_bus, db, caldav_config):
    """Test health check when not connected."""
    connector = CalDAVConnector(event_bus, db, caldav_config)
    # No _client set

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Not connected" in result["details"]


@pytest.mark.asyncio
async def test_health_check_connection_error(event_bus, db, caldav_config):
    """Test health check when connection fails."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_client = Mock()
    mock_client.principal.side_effect = Exception("Connection timeout")
    connector._client = mock_client
    connector._calendars = []

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Connection timeout" in result["details"]


# ---------------------------------------------------------------------------
# Test: Sync Time Window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_time_window(event_bus, db, caldav_config):
    """Test that sync queries the correct time window (past 1 day, future 14 days)."""
    connector = CalDAVConnector(event_bus, db, caldav_config)

    mock_calendar = Mock()
    mock_calendar.name = "Personal"
    mock_calendar.date_search.return_value = []

    connector._calendars = [mock_calendar]
    connector._client = Mock()
    connector.publish_event = AsyncMock()
    connector._detect_conflicts = AsyncMock()

    await connector.sync()

    # Verify date_search was called with correct time window
    mock_calendar.date_search.assert_called_once()
    call_kwargs = mock_calendar.date_search.call_args[1]

    assert "start" in call_kwargs
    assert "end" in call_kwargs
    assert call_kwargs["expand"] is True

    # Check time window is approximately correct (allow some tolerance)
    now = datetime.now(timezone.utc)
    start = call_kwargs["start"]
    end = call_kwargs["end"]

    # Start should be ~1 day ago
    expected_start = now - timedelta(days=1)
    assert abs((start - expected_start).total_seconds()) < 60

    # End should be ~14 days ahead
    expected_end = now + timedelta(days=14)
    assert abs((end - expected_end).total_seconds()) < 60
