"""
Life OS — Persistence tests for the Temporal Signal Extractor.

Verifies that:
1. Processing email.received events (the most common production event type)
   produces a temporal profile with correct sample counts.
2. Date-only all-day calendar events (whose start_time is an ISO date string
   with no time component, producing a naive datetime) do NOT crash the
   extractor via a TypeError in the timezone-aware comparison.
3. The temporal profile survives intact after the calendar event is processed
   (i.e., the fix does not regress the normal profile persistence path).
4. The _update_profile write-count / post-write verification path works
   correctly under normal conditions (no false CRITICAL logs for healthy DBs).

These tests exercise the full extraction and persistence path using real
db and user_model_store fixtures (no mocks), matching the approach used
by tests/test_temporal_extractor.py.
"""

import logging
from datetime import UTC, datetime, timedelta

from models.core import EventType
from services.signal_extractor.temporal import TemporalExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _email_received_event(timestamp: datetime) -> dict:
    """Build a minimal email.received event dict with the given timestamp.

    Args:
        timestamp: The event timestamp (should be timezone-aware).

    Returns:
        An event dict suitable for TemporalExtractor.extract().
    """
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "payload": {
            "from_address": "sender@example.com",
            "subject": "Test email",
            "body": "Hello world",
        },
    }


def _calendar_event_with_date_only_start(creation_ts: datetime, start_date: str) -> dict:
    """Build a calendar.event.created event whose start_time is a date-only string.

    All-day calendar events use ISO date strings like "2026-04-15" (no time
    component).  datetime.fromisoformat("2026-04-15") returns a naive datetime,
    which triggers TypeError when compared with a timezone-aware datetime via >=.

    Args:
        creation_ts: The event creation timestamp (timezone-aware).
        start_date: ISO date string, e.g. "2026-04-20" (no time component).

    Returns:
        A calendar event dict suitable for TemporalExtractor.extract().
    """
    return {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": creation_ts.isoformat().replace("+00:00", "Z"),
        "payload": {
            "summary": "All-day event",
            "start_time": start_date,  # intentionally date-only, no time/tz
            "end_time": start_date,
            "location": "",
            "attendees": [],
        },
    }


# ---------------------------------------------------------------------------
# Tests: email.received event produces temporal profile
# ---------------------------------------------------------------------------


def test_single_email_received_creates_profile(db, user_model_store):
    """Processing a single email.received event must create the temporal profile.

    This is the minimal reproduction of the production bug: 860+ email.received
    events flowed through the pipeline but the temporal profile had 0 samples.
    The most direct test is to process ONE event and immediately check the profile.
    """
    extractor = TemporalExtractor(db, user_model_store)

    ts = datetime(2026, 3, 10, 9, 0, 0, tzinfo=UTC)
    event = _email_received_event(ts)

    signals = extractor.extract(event)

    # The event should produce exactly one temporal_activity signal.
    assert len(signals) == 1
    assert signals[0]["type"] == "temporal_activity"
    assert signals[0]["activity_type"] == "email_inbound"
    assert signals[0]["hour"] == 9
    assert signals[0]["day_of_week"] == "tuesday"

    # The profile must exist immediately after the first extract() call.
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None, (
        "Temporal profile is None after processing a single email.received event. "
        "update_signal_profile() may be silently failing."
    )
    assert profile["samples_count"] == 1
    data = profile["data"]
    assert data["activity_by_hour"].get("9") == 1
    assert data["activity_by_day"].get("tuesday") == 1
    assert data["activity_by_type"].get("email_inbound") == 1


def test_ten_email_received_events_build_profile(db, user_model_store):
    """Processing 10 email.received events spread across days/hours builds a profile.

    Replicates the scale of the production scenario (860 events) at a smaller
    scope.  Events span different hours and different days of the week so that
    the aggregation logic is exercised fully.
    """
    extractor = TemporalExtractor(db, user_model_store)

    # Spread 10 events across 5 different days and several different hours.
    base = datetime(2026, 3, 9, 0, 0, 0, tzinfo=UTC)  # Monday
    timestamps = [
        base + timedelta(days=0, hours=8),  # Monday 08:00
        base + timedelta(days=0, hours=14),  # Monday 14:00
        base + timedelta(days=1, hours=9),  # Tuesday 09:00
        base + timedelta(days=1, hours=17),  # Tuesday 17:00
        base + timedelta(days=2, hours=11),  # Wednesday 11:00
        base + timedelta(days=2, hours=15),  # Wednesday 15:00
        base + timedelta(days=3, hours=10),  # Thursday 10:00
        base + timedelta(days=3, hours=13),  # Thursday 13:00
        base + timedelta(days=4, hours=8),  # Friday 08:00
        base + timedelta(days=4, hours=16),  # Friday 16:00
    ]

    for ts in timestamps:
        extractor.extract(_email_received_event(ts))

    # Profile must exist with at least 10 samples.
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None, "Temporal profile is None after processing 10 email.received events."
    assert profile["samples_count"] >= 10, f"Expected samples_count >= 10, got {profile['samples_count']}"

    data = profile["data"]
    # Should have accumulated activity across at least 5 different hours.
    assert len(data["activity_by_hour"]) >= 5, f"Expected >= 5 distinct hours, got {data['activity_by_hour']}"
    # Should have accumulated activity across the 5 distinct weekdays.
    assert len(data["activity_by_day"]) >= 5, f"Expected >= 5 distinct days, got {data['activity_by_day']}"
    # All events are email.received, so only 'email_inbound' should appear.
    assert data["activity_by_type"].get("email_inbound") == 10, (
        f"Expected email_inbound=10, got {data['activity_by_type']}"
    )
    # No other activity types should be present.
    assert set(data["activity_by_type"].keys()) == {"email_inbound"}


# ---------------------------------------------------------------------------
# Tests: date-only calendar event does NOT crash the extractor
# ---------------------------------------------------------------------------


def test_date_only_calendar_event_does_not_crash(db, user_model_store):
    """A date-only start_time ('2026-04-20') must not raise TypeError.

    Prior to the fix, datetime.fromisoformat('2026-04-20') returned a naive
    datetime.  Comparing it with the timezone-aware event dt via >= raised:
        TypeError: can't compare offset-naive and offset-aware datetimes
    The inner except (ValueError, AttributeError) did not catch TypeError, so
    it propagated to the outer except which also missed it, then the pipeline
    caught it and logged an error.

    After the fix, the naive datetime is normalized to UTC before comparison
    and TypeError is also listed in both except clauses.
    """
    extractor = TemporalExtractor(db, user_model_store)

    creation_ts = datetime(2026, 4, 10, 10, 0, 0, tzinfo=UTC)
    event = _calendar_event_with_date_only_start(creation_ts, "2026-04-20")

    # Must not raise any exception.
    signals = extractor.extract(event)

    # Should extract TWO signals: the creation activity + the scheduled event.
    temporal_signals = [s for s in signals if s["type"] == "temporal_activity"]
    scheduled_signals = [s for s in signals if s["type"] == "temporal_scheduled_event"]

    assert len(temporal_signals) == 1, f"Expected 1 temporal_activity signal, got {temporal_signals}"
    assert temporal_signals[0]["activity_type"] == "planning"

    assert len(scheduled_signals) == 1, (
        f"Expected 1 temporal_scheduled_event signal, got {scheduled_signals}. "
        "The date-only start_time should have been normalized to UTC, not skipped."
    )
    # Scheduled date: 2026-04-20 00:00 UTC → April 20, which is a Monday.
    assert scheduled_signals[0]["scheduled_hour"] == 0
    assert scheduled_signals[0]["scheduled_day"] == "monday"
    # Created on 2026-04-10, scheduled for 2026-04-20 → 10 days advance planning.
    assert scheduled_signals[0]["advance_planning_days"] == 10


def test_date_only_calendar_event_profile_survives(db, user_model_store):
    """Profile written from email events must survive a subsequent date-only calendar event.

    Scenario:
    1. Build a profile from 10 email.received events (samples_count = 10).
    2. Process one date-only calendar event (which used to crash extract()).
    3. Verify the profile still exists and has samples_count = 11.

    This guards against a regression where the bug fix accidentally discards
    previously accumulated profile data.
    """
    extractor = TemporalExtractor(db, user_model_store)

    # Step 1: process 10 email.received events.
    base = datetime(2026, 3, 9, 8, 0, 0, tzinfo=UTC)
    for i in range(10):
        extractor.extract(_email_received_event(base + timedelta(days=i % 5, hours=i)))

    profile_before = user_model_store.get_signal_profile("temporal")
    assert profile_before is not None, "Profile is None before calendar event — precondition failed"
    count_before = profile_before["samples_count"]
    assert count_before == 10, f"Expected 10 samples before calendar event, got {count_before}"

    # Step 2: process one date-only all-day calendar event.
    creation_ts = datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    calendar_event = _calendar_event_with_date_only_start(creation_ts, "2026-04-15")
    extractor.extract(calendar_event)

    # Step 3: profile must still exist with one more sample.
    profile_after = user_model_store.get_signal_profile("temporal")
    assert profile_after is not None, (
        "Temporal profile is None AFTER processing date-only calendar event. "
        "The bug fix may have introduced a crash in the write path."
    )
    assert profile_after["samples_count"] == 11, (
        f"Expected samples_count=11 after calendar event, got {profile_after['samples_count']}"
    )

    # The email_inbound count should still be 10 (calendar event adds 'planning').
    data = profile_after["data"]
    assert data["activity_by_type"].get("email_inbound") == 10, (
        f"email_inbound count changed unexpectedly: {data['activity_by_type']}"
    )
    assert data["activity_by_type"].get("planning") == 1, (
        f"Expected planning=1 from calendar event: {data['activity_by_type']}"
    )


# ---------------------------------------------------------------------------
# Tests: same-day and past-date calendar events are handled gracefully
# ---------------------------------------------------------------------------


def test_same_day_calendar_event_no_advance_planning(db, user_model_store):
    """All-day event on the same date as creation should have advance_planning_days=0.

    A date-only start_time matching the creation date (e.g. created and
    starting on 2026-04-10) should produce a scheduled event signal with
    advance_planning_days=0 and NOT be appended to advance_planning_days list.
    """
    extractor = TemporalExtractor(db, user_model_store)

    creation_ts = datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    # Same date as creation_ts — no advance planning
    event = _calendar_event_with_date_only_start(creation_ts, "2026-04-10")

    signals = extractor.extract(event)

    scheduled = [s for s in signals if s["type"] == "temporal_scheduled_event"]
    assert len(scheduled) == 1
    assert scheduled[0]["advance_planning_days"] == 0

    # Same-day events (advance_days == 0) must NOT be stored in advance_planning_days list.
    profile = user_model_store.get_signal_profile("temporal")
    assert profile["data"]["advance_planning_days"] == [], "Same-day advance_planning_days should be empty"


def test_past_date_only_calendar_event_graceful(db, user_model_store):
    """All-day event in the past relative to creation should produce advance_planning_days=0.

    When start_time is before creation time (already-passed all-day event),
    advance_planning_days should be 0 (negative is clamped to 0) and the
    event should NOT append to the planning horizon list.
    """
    extractor = TemporalExtractor(db, user_model_store)

    creation_ts = datetime(2026, 4, 10, 9, 0, 0, tzinfo=UTC)
    # Date in the past relative to creation — backlogged or historical event
    event = _calendar_event_with_date_only_start(creation_ts, "2026-04-05")

    # Must not crash
    signals = extractor.extract(event)

    scheduled = [s for s in signals if s["type"] == "temporal_scheduled_event"]
    assert len(scheduled) == 1
    assert scheduled[0]["advance_planning_days"] == 0

    profile = user_model_store.get_signal_profile("temporal")
    assert profile["data"]["advance_planning_days"] == [], "Past-date advance_planning_days should be empty"


# ---------------------------------------------------------------------------
# Tests: write-count / post-write verification counter
# ---------------------------------------------------------------------------


def test_profile_write_count_increments(db, user_model_store):
    """_profile_write_count must increment with each _update_profile call.

    This guards the defensive logic that fires a CRITICAL log when the count
    exceeds 1 but the profile is mysteriously absent.
    """
    extractor = TemporalExtractor(db, user_model_store)

    assert extractor._profile_write_count == 0, "Counter should start at 0"

    base = datetime(2026, 3, 9, 8, 0, 0, tzinfo=UTC)
    for i in range(5):
        extractor.extract(_email_received_event(base + timedelta(hours=i)))

    assert extractor._profile_write_count == 5, f"Expected write count 5, got {extractor._profile_write_count}"

    # Profile must exist and have the correct sample count.
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None
    assert profile["samples_count"] == 5


def test_critical_log_on_missing_profile_mid_stream(db, user_model_store, caplog):
    """A CRITICAL log must be emitted when the profile vanishes mid-stream.

    Simulate the scenario where update_signal_profile silently fails after the
    first successful write, by monkey-patching get_signal_profile to return
    None on the 10th post-write verification check.

    The test verifies:
    1. The CRITICAL message contains the expected text.
    2. Normal writes before the simulated failure are not affected.
    """
    extractor = TemporalExtractor(db, user_model_store)

    # The CRITICAL is emitted when _profile_write_count > 1 and count % 10 == 0.
    # So we need the write count to reach 10 before the patched read fires.
    original_get = user_model_store.get_signal_profile

    call_count = {"n": 0}

    def patched_get(profile_type):
        if profile_type == "temporal":
            call_count["n"] += 1
            # Only return None for the post-write verification call (the
            # verification is called only on count % 10 == 0, i.e. after the
            # 10th write).  The first 9 writes don't trigger a verification.
            if call_count["n"] > 1:
                return None
        return original_get(profile_type)

    user_model_store.get_signal_profile = patched_get

    base = datetime(2026, 3, 9, 8, 0, 0, tzinfo=UTC)
    with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.temporal"):
        for i in range(10):
            extractor.extract(_email_received_event(base + timedelta(hours=i)))

    # Restore original
    user_model_store.get_signal_profile = original_get

    critical_msgs = [
        r.message for r in caplog.records if r.levelno == logging.CRITICAL and "MISSING after" in r.message
    ]
    assert critical_msgs, (
        "Expected a CRITICAL log about profile being MISSING, but none was emitted. "
        "The defensive health-check in _update_profile may be broken."
    )
    assert "update_signal_profile() is silently failing" in critical_msgs[0]
