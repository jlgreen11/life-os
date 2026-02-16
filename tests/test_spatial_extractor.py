"""
Tests for SpatialExtractor — location-based behavioral pattern extraction.

Tests verify:
  1. Calendar events with location data are processed
  2. Location strings are normalized correctly
  3. Spatial profile accumulates place behaviors (visit count, duration, domain)
  4. Dominant domain and typical activities are computed correctly
  5. Current location can be inferred from recent observations
  6. Multiple visits to the same place aggregate correctly
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.signal_extractor.spatial import SpatialExtractor


def test_spatial_extractor_processes_calendar_with_location(db, user_model_store):
    """Verify extractor handles calendar events containing location data."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "id": "evt-001",
        "type": "calendar.event.created",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "caldav",
        "payload": {
            "title": "Team Meeting",
            "location": "Conference Room A, Building 3",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "attendees": ["alice@example.com", "bob@example.com"],
        },
    }

    # Assert: Extractor recognizes this event
    assert extractor.can_process(event) is True

    # Extract signals
    signals = extractor.extract(event)

    # Assert: One signal generated
    assert len(signals) == 1
    signal = signals[0]

    # Assert: Signal contains spatial data
    assert signal["signal_type"] == "spatial"
    assert signal["location"] == "conference room a building 3"  # normalized
    assert abs(signal["duration_minutes"] - 60) < 0.1  # floating point tolerance
    assert signal["domain"] == "work"  # inferred from attendees
    assert signal["activity_type"] == "calendar_event"


def test_spatial_extractor_ignores_events_without_location(db, user_model_store):
    """Verify extractor skips events that lack location data."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.created",
        "payload": {
            "title": "All-hands call",
            # No location field
        },
    }

    # Assert: Extractor skips this event
    assert extractor.can_process(event) is False


def test_location_normalization(db, user_model_store):
    """Verify location strings are normalized consistently."""
    extractor = SpatialExtractor(db, user_model_store)

    # Test various location formats
    test_cases = [
        ("Conference Room A, Building 3", "conference room a building 3"),
        ("123 Main St., Austin, TX", "123 main st austin tx"),
        ("  Home  ", "home"),
        ("Residence Inn by Marriott St. Louis Clayton, Clayton",
         "residence inn by marriott st louis clayton clayton"),
    ]

    for raw, expected in test_cases:
        normalized = extractor._normalize_location(raw)
        assert normalized == expected, f"Failed to normalize: {raw}"


def test_spatial_profile_accumulates_visits(db, user_model_store):
    """Verify spatial profile tracks multiple visits to the same location."""
    extractor = SpatialExtractor(db, user_model_store)

    base_time = datetime.now(timezone.utc)

    # Create 3 calendar events at "Office Building 5"
    for i in range(3):
        event = {
            "type": "calendar.event.created",
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "payload": {
                "title": f"Meeting {i+1}",
                "location": "Office Building 5",
                "start_time": (base_time + timedelta(days=i)).isoformat(),
                "end_time": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "attendees": ["colleague@work.com"],
            },
        }
        extractor.extract(event)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None

    # Deserialize place_behaviors
    place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
    if isinstance(place_behaviors_raw, str):
        place_behaviors = json.loads(place_behaviors_raw)
    else:
        place_behaviors = place_behaviors_raw

    # Assert: Office Building 5 has 3 visits
    office_key = "office building 5"
    assert office_key in place_behaviors
    office_place = place_behaviors[office_key]

    assert office_place["visit_count"] == 3
    assert office_place["dominant_domain"] == "work"
    assert office_place["total_duration_minutes"] == 180  # 3 hours total
    assert office_place["average_duration_minutes"] == 60  # 1 hour avg


def test_spatial_profile_tracks_dominant_domain(db, user_model_store):
    """Verify dominant domain is correctly computed from visit distribution."""
    extractor = SpatialExtractor(db, user_model_store)

    base_time = datetime.now(timezone.utc)

    # Create 5 work events and 2 personal events at "Coffee Shop"
    for i in range(5):
        event = {
            "type": "calendar.event.created",
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "payload": {
                "title": "Client Meeting",
                "location": "Coffee Shop Downtown",
                "start_time": (base_time + timedelta(hours=i)).isoformat(),
                "end_time": (base_time + timedelta(hours=i, minutes=30)).isoformat(),
                "attendees": ["client@example.com"],  # Work indicator
            },
        }
        extractor.extract(event)

    for i in range(2):
        event = {
            "type": "calendar.event.created",
            "timestamp": (base_time + timedelta(hours=i + 5)).isoformat(),
            "payload": {
                "title": "Casual Meetup",
                "location": "Coffee Shop Downtown",
                "start_time": (base_time + timedelta(hours=i + 5)).isoformat(),
                "end_time": (base_time + timedelta(hours=i + 5, minutes=30)).isoformat(),
                # No attendees = personal
            },
        }
        extractor.extract(event)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
    place_behaviors = json.loads(place_behaviors_raw) if isinstance(place_behaviors_raw, str) else place_behaviors_raw

    # Assert: Coffee Shop has "work" as dominant domain (5 work > 2 personal)
    coffee_key = "coffee shop downtown"
    assert coffee_key in place_behaviors
    coffee_place = place_behaviors[coffee_key]

    assert coffee_place["dominant_domain"] == "work"
    assert coffee_place["visit_count"] == 7


def test_spatial_profile_tracks_typical_activities(db, user_model_store):
    """Verify typical activities are extracted and ranked by frequency."""
    extractor = SpatialExtractor(db, user_model_store)

    base_time = datetime.now(timezone.utc)

    # Create multiple calendar events at "Conference Center"
    for i in range(10):
        event = {
            "type": "calendar.event.created",
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "payload": {
                "title": "Event",
                "location": "Conference Center",
                "start_time": (base_time + timedelta(days=i)).isoformat(),
                "end_time": (base_time + timedelta(days=i, hours=2)).isoformat(),
            },
        }
        extractor.extract(event)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
    place_behaviors = json.loads(place_behaviors_raw) if isinstance(place_behaviors_raw, str) else place_behaviors_raw

    # Assert: Conference Center has "calendar_event" as typical activity
    conf_key = "conference center"
    assert conf_key in place_behaviors
    conf_place = place_behaviors[conf_key]

    assert "calendar_event" in conf_place["typical_activities"]


def test_get_dominant_location_now(db, user_model_store):
    """Verify current location inference from recent observations."""
    extractor = SpatialExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Create an event at "Home" 30 minutes ago
    event_home = {
        "type": "calendar.event.created",
        "timestamp": (now - timedelta(minutes=30)).isoformat(),
        "payload": {
            "title": "Personal Time",
            "location": "Home",
            "start_time": (now - timedelta(minutes=30)).isoformat(),
            "end_time": (now - timedelta(minutes=15)).isoformat(),
        },
    }
    extractor.extract(event_home)

    # Create an older event at "Office" 5 hours ago
    event_office = {
        "type": "calendar.event.created",
        "timestamp": (now - timedelta(hours=5)).isoformat(),
        "payload": {
            "title": "Work Meeting",
            "location": "Office Building",
            "start_time": (now - timedelta(hours=5)).isoformat(),
            "end_time": (now - timedelta(hours=4)).isoformat(),
            "attendees": ["colleague@work.com"],
        },
    }
    extractor.extract(event_office)

    # Assert: Most recent location is "Home"
    current_location = extractor.get_dominant_location_now()
    assert current_location == "home"


def test_get_dominant_location_returns_none_if_no_recent_data(db, user_model_store):
    """Verify get_dominant_location_now returns None if no recent observations."""
    extractor = SpatialExtractor(db, user_model_store)

    # No events extracted yet
    current_location = extractor.get_dominant_location_now()
    assert current_location is None


def test_spatial_profile_samples_count_increments(db, user_model_store):
    """Verify samples_count increments with each observation."""
    extractor = SpatialExtractor(db, user_model_store)

    # Extract 3 events
    for i in range(3):
        event = {
            "type": "calendar.event.created",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "title": f"Event {i}",
                "location": f"Location {i}",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            },
        }
        extractor.extract(event)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")

    # Assert: samples_count is 3
    assert profile["samples_count"] == 3


def test_spatial_extractor_handles_ios_context_updates(db, user_model_store):
    """Verify iOS device proximity events are processed as spatial signals."""
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "ios.context.update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "device_proximity": "mac_mini_server",
        },
    }

    # Assert: Extractor recognizes this event
    assert extractor.can_process(event) is True

    # Extract signals
    signals = extractor.extract(event)

    # Assert: One signal generated
    assert len(signals) == 1
    signal = signals[0]

    # Assert: Location is device-based
    assert signal["location"] == "device:mac_mini_server"
    assert signal["domain"] == "personal"
    assert signal["activity_type"] == "device_proximity"


def test_spatial_profile_first_and_last_visit_timestamps(db, user_model_store):
    """Verify first_visit and last_visit timestamps are tracked correctly."""
    extractor = SpatialExtractor(db, user_model_store)

    base_time = datetime.now(timezone.utc)

    # First visit
    event1 = {
        "type": "calendar.event.created",
        "timestamp": base_time.isoformat(),
        "payload": {
            "title": "First Visit",
            "location": "Library",
            "start_time": base_time.isoformat(),
            "end_time": (base_time + timedelta(hours=1)).isoformat(),
        },
    }
    extractor.extract(event1)

    # Second visit 3 days later
    event2 = {
        "type": "calendar.event.created",
        "timestamp": (base_time + timedelta(days=3)).isoformat(),
        "payload": {
            "title": "Second Visit",
            "location": "Library",
            "start_time": (base_time + timedelta(days=3)).isoformat(),
            "end_time": (base_time + timedelta(days=3, hours=1)).isoformat(),
        },
    }
    extractor.extract(event2)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
    place_behaviors = json.loads(place_behaviors_raw) if isinstance(place_behaviors_raw, str) else place_behaviors_raw

    library_place = place_behaviors["library"]

    # Assert: first_visit is base_time, last_visit is 3 days later
    assert library_place["first_visit"] == base_time.isoformat()
    assert library_place["last_visit"] == (base_time + timedelta(days=3)).isoformat()


def test_spatial_profile_survives_pipeline_restarts(db, user_model_store):
    """Verify spatial profile persists across extractor instantiations."""
    # First extractor instance
    extractor1 = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.created",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Meeting",
            "location": "Building 7",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    }
    extractor1.extract(event)

    # Simulate restart: new extractor instance
    extractor2 = SpatialExtractor(db, user_model_store)

    # Extract another event at the same location
    event2 = {
        "type": "calendar.event.created",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Follow-up Meeting",
            "location": "Building 7",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    }
    extractor2.extract(event2)

    # Retrieve spatial profile
    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors_raw = profile.get("data", {}).get("place_behaviors", {})
    place_behaviors = json.loads(place_behaviors_raw) if isinstance(place_behaviors_raw, str) else place_behaviors_raw

    # Assert: Building 7 has 2 visits (accumulated across instances)
    building_place = place_behaviors["building 7"]
    assert building_place["visit_count"] == 2
