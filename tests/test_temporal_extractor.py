"""
Tests for the Temporal Signal Extractor.

The TemporalExtractor builds a user's temporal profile by tracking activity
patterns by hour and day of week. This test suite verifies:
- Signal extraction from various event types
- Profile aggregation (hourly/daily activity counts)
- Scheduling pattern detection (advance planning horizon)
- Activity type classification
"""

import pytest
from datetime import datetime, timezone, timedelta
from services.signal_extractor.temporal import TemporalExtractor
from models.core import EventType


def test_temporal_extractor_can_process_relevant_events(db, user_model_store):
    """TemporalExtractor should process user-initiated events with timestamps."""
    extractor = TemporalExtractor(db, user_model_store)

    # Should process: user-initiated communication
    assert extractor.can_process({"type": EventType.EMAIL_SENT.value})
    assert extractor.can_process({"type": EventType.MESSAGE_SENT.value})

    # Should process: calendar and task events
    assert extractor.can_process({"type": EventType.CALENDAR_EVENT_CREATED.value})
    assert extractor.can_process({"type": EventType.TASK_CREATED.value})
    assert extractor.can_process({"type": EventType.TASK_COMPLETED.value})

    # Should process: direct user commands
    assert extractor.can_process({"type": EventType.USER_COMMAND.value})

    # Should process: inbound communication (temporal pattern signal)
    assert extractor.can_process({"type": EventType.EMAIL_RECEIVED.value})
    assert extractor.can_process({"type": EventType.MESSAGE_RECEIVED.value})

    # Should NOT process: system events
    assert not extractor.can_process({"type": EventType.CONNECTOR_SYNC_COMPLETE.value})


def test_temporal_extractor_extracts_hourly_activity_signals(db, user_model_store):
    """Temporal signals should capture hour and day of week for activity patterns."""
    extractor = TemporalExtractor(db, user_model_store)

    # Email sent at 2pm on a Wednesday
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": "2026-02-18T14:30:00Z",
        "payload": {"to_addresses": ["friend@example.com"]},
    }

    signals = extractor.extract(event)

    # Should extract one temporal activity signal
    assert len(signals) == 1
    assert signals[0]["type"] == "temporal_activity"
    assert signals[0]["hour"] == 14
    assert signals[0]["day_of_week"] == "wednesday"
    assert signals[0]["activity_type"] == "communication"


def test_temporal_extractor_classifies_activity_types(db, user_model_store):
    """Different event types should be classified into activity categories."""
    extractor = TemporalExtractor(db, user_model_store)
    base_time = "2026-02-18T10:00:00Z"

    # Communication activity
    email_event = {"type": EventType.EMAIL_SENT.value, "timestamp": base_time, "payload": {}}
    signals = extractor.extract(email_event)
    assert signals[0]["activity_type"] == "communication"

    # Planning activity
    task_event = {"type": EventType.TASK_CREATED.value, "timestamp": base_time, "payload": {}}
    signals = extractor.extract(task_event)
    assert signals[0]["activity_type"] == "planning"

    # Work activity
    complete_event = {"type": EventType.TASK_COMPLETED.value, "timestamp": base_time, "payload": {}}
    signals = extractor.extract(complete_event)
    assert signals[0]["activity_type"] == "work"

    # Command activity
    command_event = {"type": EventType.USER_COMMAND.value, "timestamp": base_time, "payload": {}}
    signals = extractor.extract(command_event)
    assert signals[0]["activity_type"] == "command"


def test_temporal_extractor_updates_profile_with_hourly_counts(db, user_model_store):
    """Profile should aggregate activity counts by hour and day."""
    extractor = TemporalExtractor(db, user_model_store)

    # Send 3 emails at different hours on the same day
    events = [
        {"type": EventType.EMAIL_SENT.value, "timestamp": "2026-02-18T09:00:00Z", "payload": {}},
        {"type": EventType.EMAIL_SENT.value, "timestamp": "2026-02-18T14:00:00Z", "payload": {}},
        {"type": EventType.EMAIL_SENT.value, "timestamp": "2026-02-18T14:30:00Z", "payload": {}},
    ]

    for event in events:
        extractor.extract(event)

    # Check profile data
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None
    # samples_count increments by 1 per update_signal_profile call (once per extract call)
    assert profile["samples_count"] == 3

    data = profile["data"]
    assert data["activity_by_hour"]["9"] == 1
    assert data["activity_by_hour"]["14"] == 2
    # Feb 18, 2026 is a Wednesday
    assert data["activity_by_day"]["wednesday"] == 3
    assert data["activity_by_type"]["communication"] == 3


def test_temporal_extractor_tracks_scheduled_event_patterns(db, user_model_store):
    """Calendar events should track both creation time and scheduled time."""
    extractor = TemporalExtractor(db, user_model_store)

    # Create a calendar event for 2pm, scheduled 3 days in advance
    creation_time = datetime(2026, 2, 18, 10, 0, 0, tzinfo=timezone.utc)
    event_time = datetime(2026, 2, 21, 14, 0, 0, tzinfo=timezone.utc)

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": creation_time.isoformat().replace("+00:00", "Z"),
        "payload": {
            "start_time": event_time.isoformat().replace("+00:00", "Z"),
            "title": "Team meeting",
        },
    }

    signals = extractor.extract(event)

    # Should extract TWO signals:
    # 1. Creation activity (10am Tuesday)
    # 2. Scheduled event pattern (2pm Friday, 3 days advance)
    assert len(signals) == 2

    creation_signal = [s for s in signals if s["type"] == "temporal_activity"][0]
    assert creation_signal["hour"] == 10
    # Feb 18, 2026 is a Wednesday
    assert creation_signal["day_of_week"] == "wednesday"
    assert creation_signal["activity_type"] == "planning"

    scheduled_signal = [s for s in signals if s["type"] == "temporal_scheduled_event"][0]
    assert scheduled_signal["scheduled_hour"] == 14
    # Feb 21, 2026 is a Saturday
    assert scheduled_signal["scheduled_day"] == "saturday"
    assert scheduled_signal["advance_planning_days"] == 3


def test_temporal_extractor_tracks_advance_planning_horizon(db, user_model_store):
    """Profile should track how far in advance events are scheduled."""
    extractor = TemporalExtractor(db, user_model_store)

    now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=timezone.utc)

    # Create events with different planning horizons
    planning_horizons = [1, 3, 7, 14, 30]  # days in advance
    for days_ahead in planning_horizons:
        event_time = now + timedelta(days=days_ahead)
        event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "payload": {
                "start_time": event_time.isoformat().replace("+00:00", "Z"),
            },
        }
        extractor.extract(event)

    # Check profile data
    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]

    # Should have collected all planning horizons
    assert len(data["advance_planning_days"]) == 5
    assert set(data["advance_planning_days"]) == set(planning_horizons)


def test_temporal_extractor_handles_past_events_gracefully(db, user_model_store):
    """Past events (negative planning horizon) should be ignored."""
    extractor = TemporalExtractor(db, user_model_store)

    now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=timezone.utc)
    past_time = now - timedelta(days=2)

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "payload": {
            "start_time": past_time.isoformat().replace("+00:00", "Z"),
        },
    }

    extractor.extract(event)

    # Should still extract creation activity signal
    profile = user_model_store.get_signal_profile("temporal")
    assert profile["samples_count"] == 1

    # But should NOT track negative planning horizon
    data = profile["data"]
    assert len(data["advance_planning_days"]) == 0


def test_temporal_extractor_builds_weekly_activity_pattern(db, user_model_store):
    """Profile should show activity distribution across days of the week."""
    extractor = TemporalExtractor(db, user_model_store)

    # Simulate a week of activity:
    # - Heavy Monday/Tuesday (work days)
    # - Light Wednesday/Thursday
    # - Weekend quiet
    week_pattern = {
        "monday": 10,
        "tuesday": 8,
        "wednesday": 3,
        "thursday": 2,
        "friday": 1,
        "saturday": 0,
        "sunday": 0,
    }

    base_date = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)  # Sunday
    for day_offset, (day_name, count) in enumerate(week_pattern.items()):
        for i in range(count):
            event_time = base_date + timedelta(days=day_offset, hours=i)
            event = {
                "type": EventType.EMAIL_SENT.value,
                "timestamp": event_time.isoformat().replace("+00:00", "Z"),
                "payload": {},
            }
            extractor.extract(event)

    # Check profile
    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]

    # Should show heavy Monday/Tuesday activity
    assert data["activity_by_day"]["monday"] == 10
    assert data["activity_by_day"]["tuesday"] == 8
    assert data["activity_by_day"]["wednesday"] == 3
    assert data["activity_by_day"].get("saturday", 0) == 0
    assert data["activity_by_day"].get("sunday", 0) == 0


def test_temporal_extractor_handles_malformed_timestamps(db, user_model_store):
    """Malformed timestamps should be gracefully skipped."""
    extractor = TemporalExtractor(db, user_model_store)

    # Event with invalid timestamp
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": "not-a-valid-timestamp",
        "payload": {},
    }

    signals = extractor.extract(event)

    # Should return empty signal list without crashing
    assert signals == []

    # Profile should not be created/updated
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is None


def test_temporal_extractor_handles_missing_payload_fields(db, user_model_store):
    """Events with missing payload fields should be handled gracefully."""
    extractor = TemporalExtractor(db, user_model_store)

    # Calendar event without start_time
    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": "2026-02-18T10:00:00Z",
        "payload": {},  # Missing start_time
    }

    signals = extractor.extract(event)

    # Should still extract creation activity signal
    assert len(signals) == 1
    assert signals[0]["type"] == "temporal_activity"

    # But no scheduled event signal (no start_time to analyze)
    assert all(s["type"] != "temporal_scheduled_event" for s in signals)


def test_temporal_extractor_caps_planning_horizon_list(db, user_model_store):
    """Planning horizon list should be capped at 1000 entries to prevent unbounded growth."""
    extractor = TemporalExtractor(db, user_model_store)

    now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=timezone.utc)

    # Create 1100 events (should cap at 1000)
    for i in range(1100):
        event_time = now + timedelta(days=1)
        event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "payload": {
                "start_time": event_time.isoformat().replace("+00:00", "Z"),
            },
        }
        extractor.extract(event)

    # Check profile
    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]

    # Should be capped at 1000
    assert len(data["advance_planning_days"]) == 1000


def test_temporal_extractor_incremental_profile_updates(db, user_model_store):
    """Profile should incrementally update with each new event batch."""
    extractor = TemporalExtractor(db, user_model_store)

    # First batch: 5 events
    for i in range(5):
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": f"2026-02-18T{10 + i}:00:00Z",
            "payload": {},
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("temporal")
    assert profile["samples_count"] == 5

    # Second batch: 3 more events
    for i in range(3):
        event = {
            "type": EventType.TASK_COMPLETED.value,
            "timestamp": f"2026-02-18T{15 + i}:00:00Z",
            "payload": {},
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("temporal")
    assert profile["samples_count"] == 8

    # Should have correct breakdown by activity type
    data = profile["data"]
    assert data["activity_by_type"]["communication"] == 5
    assert data["activity_by_type"]["work"] == 3


# --- Tests for _derive_behavioral_fields ---


def _build_profile_with_hourly_distribution(extractor, hour_counts: dict[int, int]):
    """
    Helper: feed events into the extractor to build a profile with the given
    hourly distribution.  hour_counts maps hour (int) -> number of events at that hour.
    All events land on Wednesday (2026-02-18).
    """
    for hour, count in hour_counts.items():
        for i in range(count):
            event = {
                "type": EventType.EMAIL_SENT.value,
                "timestamp": f"2026-02-18T{hour:02d}:{i:02d}:00Z",
                "payload": {},
            }
            extractor.extract(event)


def test_derive_chronotype_early_bird(db, user_model_store):
    """Profile with >30% of activity in hours 6-10 should derive chronotype='early_bird'."""
    extractor = TemporalExtractor(db, user_model_store)

    # 40% of activity in morning hours 6-10 (24 out of 60 events),
    # rest spread across midday hours
    hour_counts = {
        6: 5, 7: 5, 8: 5, 9: 5, 10: 4,  # 24 morning events
        12: 9, 13: 9, 14: 9, 15: 9,       # 36 midday events
    }
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert data["chronotype"] == "early_bird"


def test_derive_chronotype_night_owl(db, user_model_store):
    """Profile with >30% of activity in hours 20-23 should derive chronotype='night_owl'."""
    extractor = TemporalExtractor(db, user_model_store)

    # 40% of activity in evening hours 20-23 (24 out of 60 events)
    hour_counts = {
        12: 9, 13: 9, 14: 9, 15: 9,       # 36 midday events
        20: 6, 21: 6, 22: 6, 23: 6,        # 24 evening events
    }
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert data["chronotype"] == "night_owl"


def test_derive_chronotype_variable(db, user_model_store):
    """Even distribution across hours should derive chronotype='variable'."""
    extractor = TemporalExtractor(db, user_model_store)

    # Spread 60 events evenly across 12 hours (5 each), none dominant in morning/evening
    hour_counts = {h: 5 for h in range(8, 20)}  # 8am - 7pm, 12 hours * 5 = 60
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert data["chronotype"] == "variable"


def test_derive_peak_hours(db, user_model_store):
    """Profile with clear top-3 hours should derive peak_hours containing them."""
    extractor = TemporalExtractor(db, user_model_store)

    # Create a clear pattern: hours 10, 14, 16 dominate (8 events each),
    # everything else is 1 event (background noise)
    hour_counts = {h: 1 for h in range(8, 20)}  # 12 hours * 1 = 12 baseline
    hour_counts[10] = 8
    hour_counts[14] = 8
    hour_counts[16] = 8
    # Total = 12 + (8-1)*3 = 12 + 21 = 33
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert "peak_hours" in data
    assert set(data["peak_hours"]) == {10, 14, 16}


def test_derive_wake_sleep_hours(db, user_model_store):
    """Profile with activity from 7am-23pm should derive typical_wake_hour=7, typical_sleep_hour=23."""
    extractor = TemporalExtractor(db, user_model_store)

    # Activity from 7am to 11pm, each hour has 3 events (all above 2% threshold)
    # Total = 17 hours * 3 = 51 events; 2% of 51 ≈ 1.02
    hour_counts = {h: 3 for h in range(7, 24)}
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert data["typical_wake_hour"] == 7
    assert data["typical_sleep_hour"] == 23


def test_derive_skips_with_insufficient_samples(db, user_model_store):
    """Profile with <50 total activities should not have chronotype derived."""
    extractor = TemporalExtractor(db, user_model_store)

    # Only 10 events total — below the 50-event threshold for chronotype
    # and below the 20-event threshold for peak_hours
    hour_counts = {8: 5, 9: 5}
    _build_profile_with_hourly_distribution(extractor, hour_counts)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]

    # Chronotype needs 50+, so should not be present
    assert "chronotype" not in data
    # Peak hours needs 20+, so should not be present
    assert "peak_hours" not in data
    # Wake/sleep need 20+, so should not be present
    assert "typical_wake_hour" not in data


def test_derive_planning_horizon(db, user_model_store):
    """Profile with advance_planning_days [1,3,7,14,30] should derive median=7.0."""
    extractor = TemporalExtractor(db, user_model_store)

    now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=timezone.utc)
    planning_horizons = [1, 3, 7, 14, 30]

    for days_ahead in planning_horizons:
        event_time = now + timedelta(days=days_ahead)
        event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "payload": {
                "start_time": event_time.isoformat().replace("+00:00", "Z"),
            },
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("temporal")
    data = profile["data"]
    assert data["median_planning_horizon_days"] == 7.0
