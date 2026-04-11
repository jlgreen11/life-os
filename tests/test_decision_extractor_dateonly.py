"""
Tests for DecisionExtractor — date-only start_time handling in calendar events.

Google Calendar stores all-day events with a date-only start_time
('2026-03-15') rather than a full ISO datetime string.  Parsing that string
with datetime.fromisoformat() produces a *naive* datetime (no tzinfo), while
the event's own created_at timestamp is always timezone-aware.  Subtracting
a naive datetime from an aware one raises TypeError, which was previously
silently swallowing all calendar.event.created signals and leaving the
decision profile at 0 samples.

These tests verify:
1. A date-only start_time is handled without crashing and produces a
   commitment_pattern signal.
2. A mix of date-only and full-datetime start_times both produce signals.
3. The decision profile is persisted with samples_count > 0 after processing
   date-only events.
4. Full-datetime calendar events (with timezone) still work correctly.
5. Edge cases: past date-only events (negative horizon) are skipped.
"""

import pytest
from datetime import datetime, timedelta, timezone

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calendar_event(start_time_str: str, created_offset_days: int = 0) -> dict:
    """Build a minimal calendar.event.created event dict.

    Args:
        start_time_str: ISO string for start_time — can be date-only ('2026-03-15')
            or full datetime ('2026-03-15T10:00:00+00:00').
        created_offset_days: Days before start_time_str that the event was created.
            Positive values place the creation before the event start (planning).
    """
    # Use a fixed aware base time so tests are deterministic
    created_at = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
    return {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": created_at.isoformat(),
        "payload": {
            "start_time": start_time_str,
            "summary": "Team standup",
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_date_only_start_time_does_not_crash(db, user_model_store):
    """A date-only start_time should not raise TypeError and must return signals."""
    extractor = DecisionExtractor(db, user_model_store)

    # '2026-03-15' is after the event creation timestamp of 2026-03-01 so the
    # planning horizon will be positive (event scheduled ~14 days out).
    event = _make_calendar_event("2026-03-15")

    # Must not raise any exception
    signals = extractor.extract(event)

    assert isinstance(signals, list), "extract() should always return a list"
    assert len(signals) == 1, f"Expected 1 commitment_pattern signal, got {len(signals)}: {signals}"
    assert signals[0]["type"] == "commitment_pattern"


def test_date_only_start_time_produces_correct_horizon_category(db, user_model_store):
    """A date-only start_time ~14 days out should be classified as long_term."""
    extractor = DecisionExtractor(db, user_model_store)

    # Created 2026-03-01, start date 2026-03-15 → ~14 days → 'long_term'
    event = _make_calendar_event("2026-03-15")
    signals = extractor.extract(event)

    assert signals[0]["horizon_category"] == "long_term", (
        f"Expected 'long_term' for ~14-day horizon, got {signals[0]['horizon_category']}"
    )
    # planning_horizon_seconds should be approximately 14 * 86400
    expected_horizon = 14 * 86400  # 14 days in seconds
    actual_horizon = signals[0]["planning_horizon_seconds"]
    assert abs(actual_horizon - expected_horizon) < 86400, (
        f"planning_horizon_seconds {actual_horizon} deviates more than 1 day from {expected_horizon}"
    )


def test_date_only_past_event_is_skipped(db, user_model_store):
    """A date-only start_time in the past (before created_at) should be skipped."""
    extractor = DecisionExtractor(db, user_model_store)

    # Created 2026-03-01, start date 2026-02-15 → past event → skip
    event = _make_calendar_event("2026-02-15")
    signals = extractor.extract(event)

    # Past event should produce no commitment_pattern signal
    assert signals == [], (
        f"Past date-only event should be skipped, but got: {signals}"
    )


def test_mix_of_date_only_and_full_datetime_events(db, user_model_store):
    """Both date-only and full-datetime start_times should produce signals."""
    extractor = DecisionExtractor(db, user_model_store)

    created_at = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

    # Date-only event: start 14 days out
    date_only_event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": created_at.isoformat(),
        "payload": {
            "start_time": "2026-03-15",
            "summary": "All-day conference",
        },
    }

    # Full datetime event: start 3 days out with timezone
    full_dt_start = (created_at + timedelta(days=3)).isoformat()
    full_dt_event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": created_at.isoformat(),
        "payload": {
            "start_time": full_dt_start,
            "summary": "Team sync call",
        },
    }

    # Z-suffix full datetime (common from Google Calendar)
    z_suffix_start = "2026-03-10T14:00:00Z"
    z_suffix_event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": created_at.isoformat(),
        "payload": {
            "start_time": z_suffix_start,
            "summary": "Product review",
        },
    }

    signals_date_only = extractor.extract(date_only_event)
    signals_full_dt = extractor.extract(full_dt_event)
    signals_z_suffix = extractor.extract(z_suffix_event)

    assert len(signals_date_only) == 1 and signals_date_only[0]["type"] == "commitment_pattern", (
        f"date-only event should produce commitment_pattern, got: {signals_date_only}"
    )
    assert len(signals_full_dt) == 1 and signals_full_dt[0]["type"] == "commitment_pattern", (
        f"full-datetime event should produce commitment_pattern, got: {signals_full_dt}"
    )
    assert len(signals_z_suffix) == 1 and signals_z_suffix[0]["type"] == "commitment_pattern", (
        f"Z-suffix datetime event should produce commitment_pattern, got: {signals_z_suffix}"
    )


def test_decision_profile_persisted_with_samples_after_date_only_event(db, user_model_store):
    """Processing a date-only calendar event should persist the decision profile."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_calendar_event("2026-03-15")
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None, "Decision profile should exist after processing a calendar event"
    assert profile["samples_count"] > 0, (
        f"samples_count should be > 0 after processing, got {profile['samples_count']}"
    )
    data = profile["data"]
    assert "risk_tolerance_by_domain" in data, (
        f"risk_tolerance_by_domain missing from profile. keys: {list(data.keys())}"
    )


def test_decision_profile_accumulates_samples_from_multiple_date_only_events(db, user_model_store):
    """Processing multiple date-only calendar events should accumulate profile samples."""
    extractor = DecisionExtractor(db, user_model_store)

    created_at = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)

    # Process 5 date-only events with different future dates
    # range(7, 57, 10) → [7, 17, 27, 37, 47] — exactly 5 values
    for day_offset in range(7, 57, 10):  # 7, 17, 27, 37, 47 days out
        future_date = (created_at + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": created_at.isoformat(),
            "payload": {
                "start_time": future_date,
                "summary": f"Event {day_offset} days out",
            },
        }
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert profile["samples_count"] == 5, (
        f"Expected 5 samples (one per event), got {profile['samples_count']}"
    )


def test_full_datetime_calendar_events_still_work(db, user_model_store):
    """Full-datetime start_times (the pre-existing code path) should remain unbroken."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event_start = now + timedelta(hours=2)

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "start_time": event_start.isoformat(),
            "summary": "Quick coffee catch-up",
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "commitment_pattern"
    # 2 hours = 7200s → NOT < 3600 (immediate), but < 86400 (same_day)
    assert signals[0]["horizon_category"] == "same_day"
    assert signals[0]["domain"] == "social"  # "coffee" → social


def test_date_only_with_negative_offset_handled_same_day(db, user_model_store):
    """A date-only start_time on the same day as created_at should be classified correctly.

    When created_at is 09:00 UTC and start_time is date-only '2026-03-01' (midnight UTC),
    the planning_horizon_seconds will be negative (midnight < 09:00), so the extractor
    should skip it as a past/imported event — consistent with full-datetime behavior.
    """
    extractor = DecisionExtractor(db, user_model_store)

    # Created at 09:00 UTC, all-day event on the same day (midnight UTC = earlier)
    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": "2026-03-01T09:00:00+00:00",
        "payload": {
            "start_time": "2026-03-01",  # midnight UTC < 09:00 UTC → negative horizon
            "summary": "All-day event created mid-morning",
        },
    }

    signals = extractor.extract(event)

    # Negative horizon → skipped (past/synced event)
    assert signals == [], (
        f"Same-day date-only event with negative horizon should be skipped, got: {signals}"
    )
