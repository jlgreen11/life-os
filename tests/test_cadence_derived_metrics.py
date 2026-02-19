"""
Tests for the three derived CadenceProfile metrics computed in CadenceExtractor:

  - peak_hours               — hours with above-average communication activity
  - quiet_hours_observed     — contiguous low-activity windows (sleep, quiet time)
  - avg_response_time_by_domain — reply latency bucketed by email domain

These metrics are derived from the raw hourly_activity histogram and the
per_contact_response_times dict that the extractor accumulates incrementally.
The derivation is re-run on every profile update so the derived values always
reflect the latest accumulated data.

Test strategy:
  - Inject a pre-built data dict into _compute_* directly (unit tests) to avoid
    storing thousands of events in the test DB for histogram setup.
  - End-to-end tests verify the derived values appear in the persisted profile
    after real extract() calls, exercising the full _update_profile path.
"""

from datetime import datetime, timezone, timedelta

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor(db, user_model_store):
    """CadenceExtractor wired to test databases."""
    return CadenceExtractor(db=db, user_model_store=user_model_store)


def _activity_data(hourly: dict[int, int]) -> dict:
    """Build a minimal cadence data dict with the given hourly histogram.

    Accepts integer hour keys for convenience; the cadence extractor stores
    string keys internally so this mirrors that representation.
    """
    return {
        "response_times": [],
        "hourly_activity": {str(h): v for h, v in hourly.items()},
        "daily_activity": {},
        "per_contact_response_times": {},
        "per_channel_response_times": {},
    }


# ---------------------------------------------------------------------------
# _compute_peak_hours unit tests
# ---------------------------------------------------------------------------


def test_peak_hours_identified_from_heavy_morning_activity(extractor):
    """Peak hours should include hours with above-average activity."""
    # Simulate a user active 9 AM–12 PM (high counts) with low activity elsewhere.
    hourly = {h: 1 for h in range(24)}
    for h in range(9, 13):
        hourly[h] = 20  # Heavy morning burst
    data = _activity_data(hourly)

    extractor._compute_peak_hours(data)

    assert "peak_hours" in data
    peak = data["peak_hours"]
    # All four heavy hours should be classified as peak.
    assert 9 in peak
    assert 10 in peak
    assert 11 in peak
    assert 12 in peak


def test_peak_hours_excludes_low_activity_hours(extractor):
    """Hours with activity well below the mean should not be labelled peak."""
    hourly = {h: 1 for h in range(24)}
    for h in range(9, 13):
        hourly[h] = 30
    data = _activity_data(hourly)

    extractor._compute_peak_hours(data)

    peak = data.get("peak_hours", [])
    # Quiet hours (count=1) should not appear in peak.
    assert 0 not in peak
    assert 3 not in peak
    assert 22 not in peak


def test_peak_hours_requires_50_total_samples(extractor):
    """With fewer than 50 total samples peak_hours should not be set."""
    hourly = {9: 10, 10: 8, 11: 12}  # Total = 30 < 50
    data = _activity_data(hourly)

    extractor._compute_peak_hours(data)

    # peak_hours should remain absent (not enough data for a reliable estimate).
    assert "peak_hours" not in data


def test_peak_hours_returned_sorted(extractor):
    """peak_hours list should be sorted in ascending order."""
    hourly = {h: 1 for h in range(24)}
    hourly[17] = 50
    hourly[9] = 50
    hourly[13] = 50
    # Make the total >= 50
    for h in range(24):
        if hourly.get(h, 0) == 1:
            hourly[h] = 5
    data = _activity_data(hourly)

    extractor._compute_peak_hours(data)

    peak = data.get("peak_hours", [])
    assert peak == sorted(peak)


def test_peak_hours_empty_histogram_is_a_noop(extractor):
    """An empty hourly_activity dict should leave peak_hours absent."""
    data = _activity_data({})
    extractor._compute_peak_hours(data)
    assert "peak_hours" not in data


# ---------------------------------------------------------------------------
# _compute_quiet_hours unit tests
# ---------------------------------------------------------------------------


def test_quiet_hours_detects_overnight_sleep_window(extractor):
    """A clear sleep window (10 PM – 6 AM) should be detected as quiet hours."""
    # Simulate typical daytime activity with silence overnight.
    hourly = {}
    for h in range(24):
        if 7 <= h <= 21:
            hourly[h] = 50  # Active during the day
        else:
            hourly[h] = 0   # Silent overnight (22, 23, 0-6)
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    assert "quiet_hours_observed" in data
    spans = data["quiet_hours_observed"]
    assert len(spans) >= 1

    # The span should cover the overnight silence (22 → 7 or similar).
    # At minimum, midnight should be inside the quiet window.
    all_quiet_hours = set()
    for start, end in spans:
        if end <= start:  # Wraps midnight
            for h in range(start, 24):
                all_quiet_hours.add(h)
            for h in range(0, end):
                all_quiet_hours.add(h)
        else:
            for h in range(start, end):
                all_quiet_hours.add(h)
    assert 0 in all_quiet_hours  # Midnight should be in a quiet window
    assert 3 in all_quiet_hours  # 3 AM should be in a quiet window


def test_quiet_hours_midnight_wrapping_detected_as_single_span(extractor):
    """A sleep window spanning midnight should appear as a single (start, end) pair."""
    hourly = {}
    for h in range(24):
        if 6 <= h <= 21:
            hourly[h] = 40
        else:
            hourly[h] = 0  # Quiet: 22, 23, 0, 1, 2, 3, 4, 5
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    spans = data.get("quiet_hours_observed", [])
    assert len(spans) == 1, (
        "Overnight quiet window crossing midnight should be a single span"
    )
    start, end = spans[0]
    # The span should start in the evening (hour >= 20) and end in the morning
    assert start >= 20 or start == 22
    # End should be in the morning
    assert end <= 10


def test_quiet_hours_minimum_3_hour_span_required(extractor):
    """Spans shorter than 3 hours should not be recorded as quiet windows."""
    hourly = {h: 50 for h in range(24)}
    # Create a 2-hour dip at hours 3 and 4 — too short to qualify.
    hourly[3] = 0
    hourly[4] = 0
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    # 2-hour gap should not be recorded.
    assert "quiet_hours_observed" not in data


def test_quiet_hours_requires_50_total_samples(extractor):
    """With fewer than 50 total samples quiet_hours should not be set."""
    hourly = {h: 0 for h in range(24)}
    hourly[9] = 10  # Total = 10 < 50
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    assert "quiet_hours_observed" not in data


def test_quiet_hours_empty_histogram_is_a_noop(extractor):
    """An empty hourly_activity dict should leave quiet_hours_observed absent."""
    data = _activity_data({})
    extractor._compute_quiet_hours(data)
    assert "quiet_hours_observed" not in data


def test_quiet_hours_no_quiet_periods_when_uniformly_active(extractor):
    """If all hours have similar high activity there should be no quiet windows."""
    # Uniform activity — no hour is below the 10% threshold of the peak.
    hourly = {h: 50 for h in range(24)}
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    # No quiet window should be recorded.
    assert "quiet_hours_observed" not in data


def test_quiet_hours_span_tuple_format(extractor):
    """Each quiet window should be a (start_hour, end_hour) tuple of ints 0-23."""
    hourly = {}
    for h in range(24):
        hourly[h] = 50 if 8 <= h <= 20 else 0
    data = _activity_data(hourly)

    extractor._compute_quiet_hours(data)

    for span in data.get("quiet_hours_observed", []):
        assert len(span) == 2
        start, end = span
        assert isinstance(start, int) and 0 <= start <= 23
        assert isinstance(end, int) and 0 <= end <= 23


# ---------------------------------------------------------------------------
# _compute_domain_response_times unit tests
# ---------------------------------------------------------------------------


def test_domain_response_times_computed_from_contact_data(extractor):
    """Average response times should be grouped by email domain."""
    data = _activity_data({})
    # Two contacts from the same gmail.com domain: 5 min + 10 min + 15 min = 10 min avg
    data["per_contact_response_times"] = {
        "alice@gmail.com": [300.0, 600.0, 900.0],   # 3 samples
        "bob@gmail.com":   [600.0, 600.0, 600.0],   # 3 samples → avg 600 s
        "carol@work.example.com": [7200.0, 3600.0, 5400.0],  # 3 samples
    }

    extractor._compute_domain_response_times(data)

    assert "avg_response_time_by_domain" in data
    domains = data["avg_response_time_by_domain"]

    assert "gmail.com" in domains
    assert "work.example.com" in domains

    # gmail.com: (300+600+900+600+600+600)/6 = 3600/6 = 600
    assert abs(domains["gmail.com"] - 600.0) < 1.0

    # work.example.com: (7200+3600+5400)/3 = 16200/3 = 5400
    assert abs(domains["work.example.com"] - 5400.0) < 1.0


def test_domain_response_times_requires_3_samples_per_domain(extractor):
    """Domains with fewer than 3 data points should be excluded."""
    data = _activity_data({})
    data["per_contact_response_times"] = {
        "alice@raredomain.com": [300.0, 600.0],  # Only 2 samples — excluded
        "bob@common.com": [100.0, 200.0, 300.0],  # 3 samples — included
    }

    extractor._compute_domain_response_times(data)

    domains = data.get("avg_response_time_by_domain", {})
    assert "raredomain.com" not in domains
    assert "common.com" in domains


def test_domain_response_times_skips_phone_numbers(extractor):
    """Phone number contact IDs (no '@') should be ignored for domain grouping."""
    data = _activity_data({})
    data["per_contact_response_times"] = {
        "+15551234567": [60.0, 90.0, 120.0],   # Phone number — skip
        "alice@email.com": [300.0, 300.0, 300.0],  # Email — include
    }

    extractor._compute_domain_response_times(data)

    domains = data.get("avg_response_time_by_domain", {})
    # No domain extracted from phone number.
    assert all("+" not in d for d in domains)
    assert "email.com" in domains


def test_domain_response_times_empty_contact_dict_is_a_noop(extractor):
    """Empty per_contact_response_times should leave the domain field absent."""
    data = _activity_data({})
    data["per_contact_response_times"] = {}

    extractor._compute_domain_response_times(data)

    assert "avg_response_time_by_domain" not in data


def test_domain_response_times_domain_key_is_lowercase(extractor):
    """Domain keys should be normalised to lowercase."""
    data = _activity_data({})
    data["per_contact_response_times"] = {
        "alice@GMAIL.COM": [300.0, 300.0, 300.0],
    }

    extractor._compute_domain_response_times(data)

    domains = data.get("avg_response_time_by_domain", {})
    assert "gmail.com" in domains
    assert "GMAIL.COM" not in domains


# ---------------------------------------------------------------------------
# End-to-end integration tests (full extract() → persisted profile)
# ---------------------------------------------------------------------------


def test_peak_hours_persisted_after_extract_calls(extractor, event_store, user_model_store):
    """peak_hours should appear in the stored profile after enough events."""
    # Generate 60 events concentrated at hour 10 and hour 14.
    for i in range(30):
        ts = datetime(2026, 2, 17, 10, i % 60, 0, tzinfo=timezone.utc).isoformat()
        extractor.extract({
            "id": f"peak-am-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": ts,
            "payload": {"to_addresses": ["test@example.com"], "body": "work"},
            "metadata": {},
        })
    for i in range(30):
        ts = datetime(2026, 2, 17, 14, i % 60, 0, tzinfo=timezone.utc).isoformat()
        extractor.extract({
            "id": f"peak-pm-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": ts,
            "payload": {"to_addresses": ["test@example.com"], "body": "work"},
            "metadata": {},
        })

    profile = user_model_store.get_signal_profile("cadence")
    assert "peak_hours" in profile["data"]
    assert 10 in profile["data"]["peak_hours"]
    assert 14 in profile["data"]["peak_hours"]


def test_domain_response_times_persisted_after_extract_calls(
    extractor, event_store, user_model_store
):
    """avg_response_time_by_domain should appear in profile after enough reply events."""
    base = datetime.now(timezone.utc)
    # Create 3 gmail replies at ~1 hour each and 3 work replies at ~4 hours each.
    for i in range(3):
        orig = {
            "id": f"orig-gmail-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": base.isoformat(),
            "payload": {
                "message_id": f"gmail-msg-{i}",
                "from_address": f"friend{i}@gmail.com",
                "body": "hey",
            },
            "metadata": {},
        }
        event_store.store_event(orig)
        extractor.extract({
            "id": f"reply-gmail-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (base + timedelta(hours=1)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"gmail-msg-{i}",
                "to_addresses": [f"friend{i}@gmail.com"],
                "body": "hey back",
            },
            "metadata": {},
        })
    for i in range(3):
        orig = {
            "id": f"orig-work-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": base.isoformat(),
            "payload": {
                "message_id": f"work-msg-{i}",
                "from_address": f"colleague{i}@corp.example.com",
                "body": "question",
            },
            "metadata": {},
        }
        event_store.store_event(orig)
        extractor.extract({
            "id": f"reply-work-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (base + timedelta(hours=4)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"work-msg-{i}",
                "to_addresses": [f"colleague{i}@corp.example.com"],
                "body": "answer",
            },
            "metadata": {},
        })

    profile = user_model_store.get_signal_profile("cadence")
    domains = profile["data"].get("avg_response_time_by_domain", {})

    assert "gmail.com" in domains
    assert "corp.example.com" in domains

    # Gmail replies ~1 hour = 3600 s; corp replies ~4 hours = 14400 s.
    assert abs(domains["gmail.com"] - 3600) < 10
    assert abs(domains["corp.example.com"] - 14400) < 10
