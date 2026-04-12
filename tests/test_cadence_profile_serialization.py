"""
Tests that CadenceExtractor correctly persists its profile to the database.

Regression suite for the defaultdict JSON-serialization bug: when no prior
cadence profile exists the extractor bootstrapped its data dict using
``defaultdict(int)`` and ``defaultdict(list)`` which raised
``TypeError: Object of type defaultdict is not JSON serializable`` inside
``UserModelStore.update_signal_profile()``.  The exception was silently
swallowed, so 13,519 qualifying events produced *zero* persisted profiles.

These tests verify:
  1. A single email.received event causes a profile to be written.
  2. The persisted profile contains the expected top-level keys.
  3. A second extract() call succeeds (write → read → update → write round-trip).
  4. Activity histograms are incremented correctly across multiple events.
  5. Per-contact inbound tracking accumulates across multiple events.
"""

from datetime import datetime, timezone, timedelta

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor


@pytest.fixture
def cadence(db, user_model_store):
    """CadenceExtractor wired to a temporary in-memory-like SQLite database."""
    return CadenceExtractor(db=db, user_model_store=user_model_store)


def _make_received_event(
    sender: str = "alice@example.com",
    ts: str | None = None,
    source: str = "proton_mail",
) -> dict:
    """Build a minimal email.received event for testing."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": "evt-test-001",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": source,
        "timestamp": ts,
        "payload": {
            "sender": sender,
            "subject": "Hello",
        },
    }


def _make_sent_event(
    recipient: str = "alice@example.com",
    ts: str | None = None,
    source: str = "proton_mail",
) -> dict:
    """Build a minimal email.sent event for testing."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "id": "evt-test-002",
        "type": EventType.EMAIL_SENT.value,
        "source": source,
        "timestamp": ts,
        "payload": {
            "to_addresses": [recipient],
            "subject": "Hello back",
            "is_reply": False,
        },
    }


# ---------------------------------------------------------------------------
# Core serialization regression test
# ---------------------------------------------------------------------------


def test_first_extract_persists_profile(cadence, user_model_store):
    """Processing a single event must write a cadence profile to the database.

    This is the primary regression test for the defaultdict serialization bug:
    previously the profile was NEVER written because json.dumps() raised
    TypeError on the defaultdict values.
    """
    event = _make_received_event()
    cadence.extract(event)

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None, (
        "cadence profile should be persisted after the first event — "
        "this likely means defaultdict JSON-serialization is still broken"
    )


def test_persisted_profile_has_expected_keys(cadence, user_model_store):
    """The persisted cadence profile data must contain all top-level keys."""
    event = _make_received_event()
    cadence.extract(event)

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    data = profile["data"]

    expected_keys = {
        "response_times",
        "hourly_activity",
        "daily_activity",
        "per_contact_response_times",
        "per_channel_response_times",
        "per_contact_initiations",
        "per_contact_inbound_count",
    }
    for key in expected_keys:
        assert key in data, f"expected key '{key}' missing from persisted profile"


def test_round_trip_write_read_update_write(cadence, user_model_store):
    """A second extract() call must succeed after the first has written a profile.

    Verifies the full round-trip:  write → read → merge → write.
    If the data deserialized from SQLite were incompatible with the update
    logic the second call would raise or silently skip.
    """
    ts1 = datetime(2026, 1, 15, 9, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    ts2 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    cadence.extract(_make_received_event(sender="bob@corp.com", ts=ts1))
    cadence.extract(_make_received_event(sender="carol@corp.com", ts=ts2))

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    data = profile["data"]

    # Both inbound contacts should appear in per_contact_inbound_count.
    assert "bob@corp.com" in data["per_contact_inbound_count"]
    assert "carol@corp.com" in data["per_contact_inbound_count"]
    assert data["per_contact_inbound_count"]["bob@corp.com"] == 1
    assert data["per_contact_inbound_count"]["carol@corp.com"] == 1


# ---------------------------------------------------------------------------
# Activity histogram accumulation
# ---------------------------------------------------------------------------


def test_hourly_activity_accumulates(cadence, user_model_store):
    """Multiple events at different hours must produce correct histogram counts."""
    base = datetime(2026, 3, 10, tzinfo=timezone.utc)

    for hour in [9, 9, 10, 11, 9]:
        ts = base.replace(hour=hour).isoformat().replace("+00:00", "Z")
        cadence.extract(_make_received_event(ts=ts))

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    hourly = profile["data"]["hourly_activity"]

    assert hourly.get("9", 0) == 3, "hour 9 should have count 3"
    assert hourly.get("10", 0) == 1, "hour 10 should have count 1"
    assert hourly.get("11", 0) == 1, "hour 11 should have count 1"


def test_daily_activity_accumulates(cadence, user_model_store):
    """Multiple events on different weekdays must produce correct histogram counts."""
    # 2026-03-09 is a Monday; 2026-03-10 is a Tuesday
    monday_ts = datetime(2026, 3, 9, 10, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    tuesday_ts = datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    cadence.extract(_make_received_event(ts=monday_ts))
    cadence.extract(_make_received_event(ts=monday_ts))
    cadence.extract(_make_received_event(ts=tuesday_ts))

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    daily = profile["data"]["daily_activity"]

    assert daily.get("monday", 0) == 2, "monday should have count 2"
    assert daily.get("tuesday", 0) == 1, "tuesday should have count 1"


# ---------------------------------------------------------------------------
# Outbound event accumulation
# ---------------------------------------------------------------------------


def test_outbound_events_accumulate_hourly_activity(cadence, user_model_store):
    """Sent-email events should also increment the activity histograms."""
    ts = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    cadence.extract(_make_sent_event(ts=ts))

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    hourly = profile["data"]["hourly_activity"]
    assert hourly.get("14", 0) == 1, "hour 14 should have count 1 from sent event"


# ---------------------------------------------------------------------------
# samples_count tracking
# ---------------------------------------------------------------------------


def test_samples_count_increments_on_each_write(cadence, user_model_store):
    """The samples_count column must increment with each profile write."""
    ts1 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    ts2 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    cadence.extract(_make_received_event(ts=ts1))
    after_first = user_model_store.get_signal_profile("cadence")
    assert after_first is not None
    count_after_first = after_first["samples_count"]
    assert count_after_first >= 1

    cadence.extract(_make_received_event(ts=ts2))
    after_second = user_model_store.get_signal_profile("cadence")
    assert after_second is not None
    assert after_second["samples_count"] > count_after_first
