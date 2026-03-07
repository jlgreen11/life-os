"""
Tests for expanded SpatialExtractor event support.

Covers:
  1. calendar.event.updated — location changes on existing events
  2. email.received — timezone and location hints (low-confidence signals)
  3. Inferred locations stored separately from high-confidence known places
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.signal_extractor.spatial import SpatialExtractor


# ---------------------------------------------------------------------------
# calendar.event.updated
# ---------------------------------------------------------------------------


def test_can_process_calendar_updated_with_location(db, user_model_store):
    """Verify can_process returns True for calendar.event.updated with a location."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.updated",
        "payload": {
            "title": "Team Sync",
            "location": "Conference Room B",
        },
    }
    assert extractor.can_process(event) is True


def test_can_process_calendar_updated_without_location(db, user_model_store):
    """Verify can_process returns False for calendar.event.updated without a location."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.updated",
        "payload": {
            "title": "Team Sync",
        },
    }
    assert extractor.can_process(event) is False


def test_can_process_calendar_updated_whitespace_location(db, user_model_store):
    """Verify can_process returns False for calendar.event.updated with whitespace-only location."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.updated",
        "payload": {
            "title": "Team Sync",
            "location": "   ",
        },
    }
    assert extractor.can_process(event) is False


def test_extract_calendar_updated_produces_signal(db, user_model_store):
    """Verify extract() produces spatial signals for calendar.event.updated events."""
    extractor = SpatialExtractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    event = {
        "id": "evt-upd-001",
        "type": "calendar.event.updated",
        "timestamp": now.isoformat(),
        "source": "google_calendar",
        "payload": {
            "title": "Team Standup",
            "location": "Huddle Room 3, Floor 2",
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(minutes=30)).isoformat(),
            "attendees": ["alice@example.com"],
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    signal = signals[0]
    assert signal["signal_type"] == "spatial"
    assert signal["location"] == "huddle room 3 floor 2"
    assert abs(signal["duration_minutes"] - 30) < 0.1
    assert signal["domain"] == "work"  # attendees present
    assert signal["activity_type"] == "calendar_event"
    assert signal["source"] == "google_calendar"


def test_calendar_updated_updates_profile(db, user_model_store):
    """Verify calendar.event.updated events accumulate in the spatial profile."""
    extractor = SpatialExtractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    # First: create event at location
    event_created = {
        "type": "calendar.event.created",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Meeting",
            "location": "Room 101",
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(hours=1)).isoformat(),
        },
    }
    extractor.extract(event_created)

    # Then: update adds location data
    event_updated = {
        "type": "calendar.event.updated",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Meeting (moved)",
            "location": "Room 101",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
    }
    extractor.extract(event_updated)

    profile = user_model_store.get_signal_profile("spatial")
    pb_raw = profile["data"]["place_behaviors"]
    pb = json.loads(pb_raw) if isinstance(pb_raw, str) else pb_raw

    assert "room 101" in pb
    assert pb["room 101"]["visit_count"] == 2


def test_calendar_updated_personal_domain(db, user_model_store):
    """Verify calendar.event.updated without work indicators defaults to personal domain."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.updated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Yoga Class",
            "location": "Downtown Gym",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    }

    signals = extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["domain"] == "personal"


# ---------------------------------------------------------------------------
# email.received — timezone/location hints
# ---------------------------------------------------------------------------


def test_can_process_email_with_timezone(db, user_model_store):
    """Verify can_process returns True for email.received with timezone field."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "payload": {
            "subject": "Hello",
            "timezone": "America/Los_Angeles",
        },
    }
    assert extractor.can_process(event) is True


def test_can_process_email_with_sender_timezone(db, user_model_store):
    """Verify can_process returns True for email.received with sender_timezone field."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "payload": {
            "subject": "Update",
            "sender_timezone": "Europe/London",
        },
    }
    assert extractor.can_process(event) is True


def test_can_process_email_with_location(db, user_model_store):
    """Verify can_process returns True for email.received with location field."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "payload": {
            "subject": "Office visit",
            "location": "San Francisco, CA",
        },
    }
    assert extractor.can_process(event) is True


def test_can_process_email_without_timezone_or_location(db, user_model_store):
    """Verify can_process returns False for email.received without any location metadata."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "payload": {
            "subject": "Hello",
            "body": "How are you?",
            "from": "alice@example.com",
        },
    }
    assert extractor.can_process(event) is False


def test_extract_email_timezone_produces_location_hint(db, user_model_store):
    """Verify extract() produces location_hint signals for email events with timezone."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "gmail",
        "payload": {
            "subject": "Project Update",
            "timezone": "America/Los_Angeles",
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    signal = signals[0]
    assert signal["signal_type"] == "spatial"
    assert signal["type"] == "location_hint"
    assert signal["location"] == "west coast us"
    assert signal["confidence"] == 0.3
    assert signal["activity_type"] == "email_hint"
    assert signal["source"] == "gmail"


def test_extract_email_sender_timezone(db, user_model_store):
    """Verify sender_timezone field is used when timezone field is absent."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "sender_timezone": "Europe/London",
        },
    }

    signals = extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["location"] == "london uk"


def test_extract_email_explicit_location_over_timezone(db, user_model_store):
    """Verify explicit location field takes priority over timezone."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "location": "San Francisco Office",
            "timezone": "America/New_York",  # should be ignored
        },
    }

    signals = extractor.extract(event)
    assert len(signals) == 1
    assert signals[0]["location"] == "san francisco office"


def test_extract_email_unknown_timezone_returns_empty(db, user_model_store):
    """Verify unknown timezone that isn't in the mapping returns no signals."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "timezone": "Antarctica/McMurdo",
        },
    }

    # can_process is True (timezone field exists), but extract returns empty
    assert extractor.can_process(event) is True
    signals = extractor.extract(event)
    assert signals == []


# ---------------------------------------------------------------------------
# Inferred locations stored separately from known places
# ---------------------------------------------------------------------------


def test_inferred_locations_stored_separately(db, user_model_store):
    """Verify email-derived locations are stored in inferred_locations, not place_behaviors."""
    extractor = SpatialExtractor(db, user_model_store)

    # First add a high-confidence calendar signal
    cal_event = {
        "type": "calendar.event.created",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Meeting",
            "location": "Office Building",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    }
    extractor.extract(cal_event)

    # Then add a low-confidence email timezone signal
    email_event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "timezone": "Asia/Tokyo",
        },
    }
    extractor.extract(email_event)

    profile = user_model_store.get_signal_profile("spatial")
    data = profile["data"]

    # place_behaviors should only have the calendar location
    pb_raw = data.get("place_behaviors", "{}")
    pb = json.loads(pb_raw) if isinstance(pb_raw, str) else pb_raw
    assert "office building" in pb
    assert "japan" not in pb  # email hint should NOT be in place_behaviors

    # inferred_locations should have the email-derived region
    inferred_raw = data.get("inferred_locations", "{}")
    inferred = json.loads(inferred_raw) if isinstance(inferred_raw, str) else inferred_raw
    assert "japan" in inferred
    assert inferred["japan"]["observation_count"] == 1
    assert "email_timezone" in inferred["japan"]["sources"]


def test_inferred_locations_accumulate(db, user_model_store):
    """Verify multiple email signals for the same timezone accumulate."""
    extractor = SpatialExtractor(db, user_model_store)
    base_time = datetime.now(timezone.utc)

    for i in range(5):
        event = {
            "type": "email.received",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "payload": {
                "timezone": "America/New_York",
            },
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    inferred_raw = profile["data"].get("inferred_locations", "{}")
    inferred = json.loads(inferred_raw) if isinstance(inferred_raw, str) else inferred_raw

    assert "east coast us" in inferred
    assert inferred["east coast us"]["observation_count"] == 5
    assert inferred["east coast us"]["sources"]["email_timezone"] == 5


def test_inferred_locations_preserve_place_behaviors(db, user_model_store):
    """Verify adding inferred locations does not destroy existing place_behaviors."""
    extractor = SpatialExtractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Add calendar event first
    cal_event = {
        "type": "calendar.event.created",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Standup",
            "location": "HQ Floor 3",
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(minutes=15)).isoformat(),
        },
    }
    extractor.extract(cal_event)

    # Now add email hint — should NOT wipe place_behaviors
    email_event = {
        "type": "email.received",
        "timestamp": now.isoformat(),
        "payload": {"timezone": "Europe/Paris"},
    }
    extractor.extract(email_event)

    profile = user_model_store.get_signal_profile("spatial")
    data = profile["data"]

    # place_behaviors still has the calendar location
    pb_raw = data.get("place_behaviors", "{}")
    pb = json.loads(pb_raw) if isinstance(pb_raw, str) else pb_raw
    assert "hq floor 3" in pb
    assert pb["hq floor 3"]["visit_count"] == 1


def test_samples_count_increments_for_email_hints(db, user_model_store):
    """Verify samples_count increments for email-derived location hints."""
    extractor = SpatialExtractor(db, user_model_store)

    for i in range(3):
        event = {
            "type": "email.received",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"timezone": "America/Chicago"},
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    assert profile["samples_count"] == 3
