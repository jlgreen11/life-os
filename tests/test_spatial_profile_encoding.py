"""
Tests for spatial profile encoding correctness.

Regression tests for the double JSON-encoding bug where `json.dumps()` was
called on `place_behaviors` and `inferred_locations` before passing them to
`update_signal_profile()`, which itself calls `json.dumps()` again.  This
caused stored values to be nested JSON strings instead of dicts, breaking any
downstream code that accessed `profile['data']['place_behaviors']['some_key']`.

These tests verify that:
  1. `place_behaviors` is stored and retrieved as a plain dict, not a string.
  2. `inferred_locations` is stored and retrieved as a plain dict, not a string.
  3. `get_dominant_location_now()` works correctly (it reads `place_behaviors`).
  4. Consecutive profile updates accumulate data without encoding drift.
"""

from datetime import datetime, timedelta, timezone

import pytest

from services.signal_extractor.spatial import SpatialExtractor


# ---------------------------------------------------------------------------
# place_behaviors encoding
# ---------------------------------------------------------------------------


def test_place_behaviors_is_dict_not_string(db, user_model_store):
    """Verify place_behaviors is stored as a dict, not a JSON-encoded string.

    The original bug caused update_signal_profile() to receive a JSON string
    instead of a dict, which was then JSON-encoded *again* — resulting in a
    doubly-encoded value that deserialized to a string on read.
    """
    extractor = SpatialExtractor(db, user_model_store)

    event = {
        "type": "calendar.event.created",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "title": "Morning Standup",
            "location": "HQ Office",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "attendees": ["team@example.com"],
        },
    }
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None, "Spatial profile should exist after processing an event"

    place_behaviors = profile["data"]["place_behaviors"]

    # The value MUST be a dict — never a string.
    assert isinstance(place_behaviors, dict), (
        f"place_behaviors should be a dict but got {type(place_behaviors).__name__}: "
        f"{place_behaviors!r}"
    )

    # Sanity check: the location entry is actually there and accessible.
    hq_key = "hq office"
    assert hq_key in place_behaviors, f"Expected '{hq_key}' in place_behaviors"
    assert place_behaviors[hq_key]["visit_count"] == 1


def test_place_behaviors_dict_keys_accessible(db, user_model_store):
    """Verify nested dict keys inside place_behaviors are directly accessible.

    With double-encoding, `place_behaviors["hq office"]["visit_count"]` would
    raise a TypeError because indexing a string with a string key is invalid.
    This test catches that regression.
    """
    extractor = SpatialExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    for i in range(3):
        event = {
            "type": "calendar.event.created",
            "timestamp": (now + timedelta(days=i)).isoformat(),
            "payload": {
                "title": f"Event {i}",
                "location": "Downtown Studio",
                "start_time": (now + timedelta(days=i)).isoformat(),
                "end_time": (now + timedelta(days=i, hours=2)).isoformat(),
            },
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    place_behaviors = profile["data"]["place_behaviors"]

    # These dict accesses would raise TypeError if value were a string.
    studio_place = place_behaviors["downtown studio"]
    assert studio_place["visit_count"] == 3
    assert studio_place["dominant_domain"] == "personal"
    assert studio_place["total_duration_minutes"] == 360.0
    assert studio_place["average_duration_minutes"] == 120.0


# ---------------------------------------------------------------------------
# inferred_locations encoding
# ---------------------------------------------------------------------------


def test_inferred_locations_is_dict_not_string(db, user_model_store):
    """Verify inferred_locations is stored as a dict, not a JSON-encoded string.

    Email timezone hints are stored in `inferred_locations`.  The original bug
    pre-encoded this with `json.dumps()` before `update_signal_profile()`, so
    the stored value was a doubly-encoded string.
    """
    extractor = SpatialExtractor(db, user_model_store)

    # email.received with timezone metadata produces an inferred location.
    event = {
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "proton_mail",
        "payload": {
            "subject": "Hello from Tokyo",
            "timezone": "Asia/Tokyo",
        },
    }
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None, "Spatial profile should exist after email event"

    inferred_locations = profile["data"]["inferred_locations"]

    # The value MUST be a dict — never a string.
    assert isinstance(inferred_locations, dict), (
        f"inferred_locations should be a dict but got "
        f"{type(inferred_locations).__name__}: {inferred_locations!r}"
    )

    # The mapped region should be present.
    japan_key = "japan"
    assert japan_key in inferred_locations, (
        f"Expected 'japan' in inferred_locations, got keys: {list(inferred_locations)}"
    )
    assert inferred_locations[japan_key]["observation_count"] == 1


def test_inferred_locations_accumulate_across_events(db, user_model_store):
    """Verify inferred location counts increment correctly across multiple emails."""
    extractor = SpatialExtractor(db, user_model_store)

    base_time = datetime.now(timezone.utc)
    for i in range(4):
        event = {
            "type": "email.received",
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "source": "proton_mail",
            "payload": {
                "subject": f"Newsletter {i}",
                "timezone": "America/Los_Angeles",
            },
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    inferred = profile["data"]["inferred_locations"]

    assert isinstance(inferred, dict)
    west_key = "west coast us"
    assert west_key in inferred
    assert inferred[west_key]["observation_count"] == 4


# ---------------------------------------------------------------------------
# get_dominant_location_now
# ---------------------------------------------------------------------------


def test_get_dominant_location_now_returns_string(db, user_model_store):
    """Verify get_dominant_location_now() returns a non-empty string.

    This method reads `place_behaviors` internally.  If that value were a
    double-encoded string the iteration over its .items() would iterate over
    characters rather than location entries, and the method would return None.
    """
    extractor = SpatialExtractor(db, user_model_store)

    # Record a visit very recently so it falls within the 24-hour window.
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    event = {
        "type": "calendar.event.created",
        "timestamp": recent_time.isoformat(),
        "payload": {
            "title": "Gym Session",
            "location": "Local Gym",
            "start_time": recent_time.isoformat(),
            "end_time": (recent_time + timedelta(hours=1)).isoformat(),
        },
    }
    extractor.extract(event)

    location = extractor.get_dominant_location_now()

    assert location is not None, "Should return a location after a recent visit"
    assert isinstance(location, str), f"Expected str, got {type(location).__name__}"
    assert location == "local gym"


def test_get_dominant_location_now_picks_most_recent(db, user_model_store):
    """Verify get_dominant_location_now() selects the most recently visited place."""
    extractor = SpatialExtractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Older visit to "Office" (10 hours ago)
    office_event = {
        "type": "calendar.event.created",
        "timestamp": (now - timedelta(hours=10)).isoformat(),
        "payload": {
            "title": "Work Meeting",
            "location": "Office Tower",
            "start_time": (now - timedelta(hours=10)).isoformat(),
            "end_time": (now - timedelta(hours=9)).isoformat(),
            "attendees": ["boss@work.com"],
        },
    }
    extractor.extract(office_event)

    # More recent visit to "Home" (2 hours ago)
    home_event = {
        "type": "calendar.event.created",
        "timestamp": (now - timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Remote Work",
            "location": "Home Office",
            "start_time": (now - timedelta(hours=2)).isoformat(),
            "end_time": (now - timedelta(hours=1)).isoformat(),
        },
    }
    extractor.extract(home_event)

    location = extractor.get_dominant_location_now()
    assert location == "home office", (
        f"Expected 'home office' (most recent) but got '{location}'"
    )


# ---------------------------------------------------------------------------
# Profile survives mixed updates without encoding drift
# ---------------------------------------------------------------------------


def test_mixed_updates_preserve_both_fields(db, user_model_store):
    """Verify that updating inferred_locations doesn't corrupt place_behaviors.

    The `_update_inferred_location` method reads `place_behaviors` from the
    existing profile and carries it forward.  This test ensures neither field
    becomes double-encoded after interleaved writes.
    """
    extractor = SpatialExtractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    # First: calendar event writes place_behaviors
    cal_event = {
        "type": "calendar.event.created",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Design Review",
            "location": "Studio B",
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
        },
    }
    extractor.extract(cal_event)

    # Second: email event writes inferred_locations (and carries forward place_behaviors)
    email_event = {
        "type": "email.received",
        "timestamp": (now + timedelta(minutes=30)).isoformat(),
        "source": "proton_mail",
        "payload": {
            "subject": "Greetings from London",
            "timezone": "Europe/London",
        },
    }
    extractor.extract(email_event)

    profile = user_model_store.get_signal_profile("spatial")
    data = profile["data"]

    # Both fields must be dicts after interleaved writes.
    assert isinstance(data.get("place_behaviors"), dict), (
        "place_behaviors must be a dict after mixed updates"
    )
    assert isinstance(data.get("inferred_locations"), dict), (
        "inferred_locations must be a dict after mixed updates"
    )

    # place_behaviors should still hold the calendar event data.
    assert "studio b" in data["place_behaviors"], (
        "place_behaviors should still contain 'studio b' after email update"
    )

    # inferred_locations should hold the email hint.
    assert "london uk" in data["inferred_locations"], (
        "inferred_locations should contain 'london uk' after email update"
    )
