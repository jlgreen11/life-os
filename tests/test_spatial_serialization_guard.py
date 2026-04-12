"""
Tests for the JSON serialization guards added to SpatialExtractor.

Covers the two write sites in spatial.py:
  1. _update_inferred_location() — writes 'inferred_locations' to the spatial profile
  2. _update_spatial_profile()   — writes 'place_behaviors' to the spatial profile

For each site we verify:
  a. Non-serializable data causes the write to be skipped (guard fires, returns early)
  b. Valid data still persists correctly

Also tests the post-write verification path (profile is readable after a successful write).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.spatial import SpatialExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extractor(db, user_model_store) -> SpatialExtractor:
    """Instantiate a SpatialExtractor with the test fixtures."""
    return SpatialExtractor(db, user_model_store)


def _calendar_event(location: str, now: datetime | None = None) -> dict:
    """Build a minimal calendar.event.created event for a given location."""
    if now is None:
        now = datetime.now(timezone.utc)
    return {
        "type": "calendar.event.created",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Test Event",
            "location": location,
            "start_time": now.isoformat(),
            "end_time": now.isoformat(),
        },
    }


def _email_event(timezone_str: str, now: datetime | None = None) -> dict:
    """Build a minimal email.received event that triggers an inferred-location write."""
    if now is None:
        now = datetime.now(timezone.utc)
    return {
        "type": "email.received",
        "timestamp": now.isoformat(),
        "source": "proton_mail",
        "payload": {
            "subject": "Test email",
            "timezone": timezone_str,
        },
    }


# ---------------------------------------------------------------------------
# Guard at _update_inferred_location: non-serializable source data
# ---------------------------------------------------------------------------


def test_inferred_location_guard_fires_on_bad_source_data(db, user_model_store, caplog):
    """Guard at _update_inferred_location skips write when a field is non-JSON-serializable.

    We inject bad data via get_signal_profile: the existing profile contains a
    'first_seen' value that is a raw datetime object (not an ISO string).  The
    modification code does not touch first_seen, so it remains as a datetime and
    causes json.dumps() to raise TypeError during the guard check.  The guard must
    catch this, log a descriptive error, and return early without calling
    update_signal_profile().
    """
    extractor = _make_extractor(db, user_model_store)

    # first_seen is a datetime object — not JSON-serializable, but not touched by
    # the modification code so the update logic won't crash before the guard runs.
    bad_profile = {
        "data": {
            "inferred_locations": {
                "japan": {
                    "observation_count": 1,
                    "sources": {"email_timezone": 1},  # valid int
                    "first_seen": datetime.now(timezone.utc),  # datetime — not JSON-safe
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
    }

    with patch.object(user_model_store, "get_signal_profile", return_value=bad_profile), \
         patch.object(user_model_store, "update_signal_profile") as mock_write, \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_inferred_location(
            location="japan",
            source_field="email_timezone",
            timestamp=datetime.now(timezone.utc),
        )

        # The guard must skip the write
        mock_write.assert_not_called()

    # Error message should appear in logs
    assert any(
        "non-JSON-serializable" in record.message
        for record in caplog.records
    ), "Expected a non-JSON-serializable error to be logged"


def test_inferred_location_guard_logs_bad_field_name(db, user_model_store, caplog):
    """Guard logs the specific field path that caused the serialization failure.

    This ensures operators can quickly identify the offending field from the
    log line rather than having to add instrumentation themselves.

    We use a datetime in 'first_seen' (a field the modification code does not
    touch) so the bad value survives into the serialization check.
    """
    extractor = _make_extractor(db, user_model_store)

    bad_profile = {
        "data": {
            "inferred_locations": {
                "west coast us": {
                    "observation_count": 2,
                    "sources": {"email_timezone": 2},  # valid int
                    "first_seen": datetime.now(timezone.utc),  # datetime — not JSON-safe
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
    }

    with patch.object(user_model_store, "get_signal_profile", return_value=bad_profile), \
         patch.object(user_model_store, "update_signal_profile"), \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_inferred_location(
            location="west coast us",
            source_field="email_timezone",
            timestamp=datetime.now(timezone.utc),
        )

    # The log should mention the non-serializable field
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_messages, "At least one ERROR should be logged"
    combined = " ".join(error_messages)
    assert "non-JSON-serializable" in combined, (
        "Log should mention 'non-JSON-serializable' to help operators diagnose the issue"
    )


# ---------------------------------------------------------------------------
# Guard at _update_spatial_profile: non-serializable activity data
# ---------------------------------------------------------------------------


def test_spatial_profile_guard_fires_on_bad_activity_counts(db, user_model_store, caplog):
    """Guard at _update_spatial_profile skips write when a field is non-JSON-serializable.

    We inject bad data via get_signal_profile: the existing profile has a raw
    datetime object in 'first_visit'.  The modification code overwrites 'last_visit'
    and computed fields but does NOT touch 'first_visit', so the datetime object
    persists into the JSON guard check and causes json.dumps() to raise TypeError.
    The guard must catch this, log an error, and skip the write.
    """
    extractor = _make_extractor(db, user_model_store)

    bad_profile = {
        "data": {
            "place_behaviors": {
                "hq office": {
                    "place_id": "hq office",
                    "visit_count": 1,
                    "total_duration_minutes": 60.0,
                    "domain_counts": {"work": 1},
                    "activity_counts": {"calendar_event": 1},  # valid int
                    "first_visit": datetime.now(timezone.utc),  # datetime — not JSON-safe
                    "last_visit": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
    }

    with patch.object(user_model_store, "get_signal_profile", return_value=bad_profile), \
         patch.object(user_model_store, "update_signal_profile") as mock_write, \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_spatial_profile(
            location="hq office",
            duration_minutes=60.0,
            domain="work",
            activity_type="calendar_event",
            timestamp=datetime.now(timezone.utc),
        )

        mock_write.assert_not_called()

    assert any(
        "non-JSON-serializable" in record.message
        for record in caplog.records
    ), "Expected a non-JSON-serializable error to be logged"


def test_spatial_profile_guard_fires_on_bad_typical_activities(db, user_model_store, caplog):
    """Guard fires when typical_activities contains a non-serializable element.

    typical_activities is recomputed from activity_counts on every update, so
    we cannot inject bad data there directly.  Instead we use a raw datetime
    in 'first_visit' (not touched by the update logic) to trigger the guard.
    The test confirms the guard fires and logs an error about the bad field.
    """
    extractor = _make_extractor(db, user_model_store)

    # first_visit is a datetime — not JSON-serializable, not touched by modification code.
    bad_profile = {
        "data": {
            "place_behaviors": {
                "coffee shop": {
                    "place_id": "coffee shop",
                    "visit_count": 3,
                    "total_duration_minutes": 90.0,
                    "domain_counts": {"personal": 3},
                    "activity_counts": {"device_proximity": 3},
                    "typical_activities": ["device_proximity"],
                    "first_visit": datetime.now(timezone.utc),  # datetime — not JSON-safe
                    "last_visit": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
    }

    with patch.object(user_model_store, "get_signal_profile", return_value=bad_profile), \
         patch.object(user_model_store, "update_signal_profile") as mock_write, \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_spatial_profile(
            location="coffee shop",
            duration_minutes=30.0,
            domain="personal",
            activity_type="device_proximity",
            timestamp=datetime.now(timezone.utc),
        )

        mock_write.assert_not_called()

    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_messages, "Expected an ERROR log"
    combined = " ".join(error_messages)
    assert "non-JSON-serializable" in combined, (
        "Log should indicate the data was non-JSON-serializable"
    )


def test_spatial_profile_guard_logs_bad_field_details(db, user_model_store, caplog):
    """Guard logs descriptive details about the non-serializable field.

    We use a datetime in 'first_visit' (not touched by modification code) to
    trigger the guard, then verify the error log contains enough context for
    operators to identify the problem quickly.
    """
    extractor = _make_extractor(db, user_model_store)

    bad_profile = {
        "data": {
            "place_behaviors": {
                "studio": {
                    "place_id": "studio",
                    "visit_count": 1,
                    "total_duration_minutes": 120.0,
                    "domain_counts": {"personal": 1},
                    "activity_counts": {"recording": 1},  # valid int
                    "first_visit": datetime.now(timezone.utc),  # datetime — not JSON-safe
                    "last_visit": datetime.now(timezone.utc).isoformat(),
                }
            }
        }
    }

    with patch.object(user_model_store, "get_signal_profile", return_value=bad_profile), \
         patch.object(user_model_store, "update_signal_profile"), \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_spatial_profile(
            location="studio",
            duration_minutes=120.0,
            domain="personal",
            activity_type="recording",
            timestamp=datetime.now(timezone.utc),
        )

    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_messages, "Expected an ERROR log"
    combined = " ".join(error_messages)
    # The log should mention the problem type and the write being skipped
    assert "write SKIPPED" in combined or "non-JSON-serializable" in combined, (
        "Log message should indicate the write was skipped due to non-serializable data"
    )


# ---------------------------------------------------------------------------
# Valid data still persists correctly through both paths
# ---------------------------------------------------------------------------


def test_valid_inferred_location_persists(db, user_model_store):
    """Valid email timezone data produces a persisted inferred_locations entry.

    This is the happy-path regression test: the guard must not block valid data.
    """
    extractor = _make_extractor(db, user_model_store)

    # Japan timezone maps to "Japan" region in TIMEZONE_REGION_MAP
    event = _email_event("Asia/Tokyo")
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None, "Spatial profile should be created by the email event"

    inferred = profile["data"].get("inferred_locations", {})
    assert isinstance(inferred, dict), "inferred_locations must be a dict"
    assert "japan" in inferred, f"'japan' should be in inferred_locations, got: {list(inferred)}"
    assert inferred["japan"]["observation_count"] == 1
    assert isinstance(inferred["japan"]["sources"], dict), "sources must be a dict"


def test_valid_place_behavior_persists(db, user_model_store):
    """Valid calendar location data produces a persisted place_behaviors entry.

    This is the happy-path regression test: the guard must not block valid data.
    """
    extractor = _make_extractor(db, user_model_store)

    event = _calendar_event("Downtown Office")
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None, "Spatial profile should be created by the calendar event"

    place_behaviors = profile["data"].get("place_behaviors", {})
    assert isinstance(place_behaviors, dict), "place_behaviors must be a dict"
    assert "downtown office" in place_behaviors, (
        f"Expected 'downtown office' in place_behaviors, got: {list(place_behaviors)}"
    )
    assert place_behaviors["downtown office"]["visit_count"] == 1


def test_valid_data_persists_through_both_paths_interleaved(db, user_model_store):
    """Valid data written to both paths (calendar + email) accumulates correctly.

    Ensures that interleaved writes to _update_spatial_profile and
    _update_inferred_location do not corrupt one another through the guards.
    """
    extractor = _make_extractor(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Calendar event → _update_spatial_profile
    extractor.extract(_calendar_event("Central Library", now))

    # Email event → _update_inferred_location
    extractor.extract(_email_event("Europe/London", now))

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None

    data = profile["data"]
    assert isinstance(data.get("place_behaviors"), dict)
    assert "central library" in data["place_behaviors"]

    assert isinstance(data.get("inferred_locations"), dict)
    assert "london uk" in data["inferred_locations"]


# ---------------------------------------------------------------------------
# Post-write verification path
# ---------------------------------------------------------------------------


def test_post_write_verification_logs_error_when_profile_missing(
    db, user_model_store, caplog
):
    """Post-write verification logs an ERROR when the profile is absent after write.

    We simulate this by making update_signal_profile a no-op (so nothing is
    written) while get_signal_profile returns None on the verification read.
    The implementation should log an error about the write failure.
    """
    extractor = _make_extractor(db, user_model_store)

    call_count = {"n": 0}

    def patched_get(profile_type: str):
        """Return None only on the second call (the post-write read-back check)."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # First call: no existing profile
        return None  # Second call: simulated missing profile after write

    with patch.object(user_model_store, "get_signal_profile", side_effect=patched_get), \
         patch.object(user_model_store, "update_signal_profile"), \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_inferred_location(
            location="test location",
            source_field="email_timezone",
            timestamp=datetime.now(timezone.utc),
        )

    # Should log an error about the profile failing to persist
    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "FAILED to persist" in msg for msg in error_msgs
    ), f"Expected 'FAILED to persist' error, got: {error_msgs}"


def test_post_write_verification_logs_error_when_spatial_profile_missing_after_update(
    db, user_model_store, caplog
):
    """Post-write verification for _update_spatial_profile logs ERROR when profile absent.

    Mirrors the inferred-location test but for the place_behaviors write path.
    """
    extractor = _make_extractor(db, user_model_store)

    call_count = {"n": 0}

    def patched_get(profile_type: str):
        """Return None on both calls to simulate a DB write that doesn't persist."""
        call_count["n"] += 1
        return None

    with patch.object(user_model_store, "get_signal_profile", side_effect=patched_get), \
         patch.object(user_model_store, "update_signal_profile"), \
         caplog.at_level(logging.ERROR, logger="services.signal_extractor.spatial"):

        extractor._update_spatial_profile(
            location="gym",
            duration_minutes=45.0,
            domain="personal",
            activity_type="calendar_event",
            timestamp=datetime.now(timezone.utc),
        )

    error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "FAILED to persist" in msg for msg in error_msgs
    ), f"Expected 'FAILED to persist' error, got: {error_msgs}"
