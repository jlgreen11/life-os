"""
Tests for inbound event processing in the Temporal Signal Extractor.

Verifies that email.received and message.received events are accepted by
can_process(), produce correct temporal signals with distinct activity types,
and update the temporal profile alongside existing outbound event handling.
"""

import pytest

from models.core import EventType
from services.signal_extractor.temporal import TemporalExtractor


def test_can_process_accepts_email_received(db, user_model_store):
    """can_process() should return True for email.received events."""
    extractor = TemporalExtractor(db, user_model_store)
    assert extractor.can_process({"type": EventType.EMAIL_RECEIVED.value})


def test_can_process_accepts_message_received(db, user_model_store):
    """can_process() should return True for message.received events."""
    extractor = TemporalExtractor(db, user_model_store)
    assert extractor.can_process({"type": EventType.MESSAGE_RECEIVED.value})


def test_extract_email_received_produces_signal(db, user_model_store):
    """Extracting an email.received event should produce a temporal_activity signal with activity_type='email_inbound'."""
    extractor = TemporalExtractor(db, user_model_store)

    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": "2026-02-18T09:15:00Z",
        "payload": {"from_address": "colleague@example.com", "subject": "Meeting notes"},
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "temporal_activity"
    assert signals[0]["hour"] == 9
    assert signals[0]["day_of_week"] == "wednesday"
    assert signals[0]["activity_type"] == "email_inbound"
    assert signals[0]["event_type"] == EventType.EMAIL_RECEIVED.value


def test_extract_message_received_produces_signal(db, user_model_store):
    """Extracting a message.received event should produce a temporal_activity signal with activity_type='message_inbound'."""
    extractor = TemporalExtractor(db, user_model_store)

    event = {
        "type": EventType.MESSAGE_RECEIVED.value,
        "timestamp": "2026-02-18T20:30:00Z",
        "payload": {"sender": "+15551234567", "body": "Hey, are you free?"},
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "temporal_activity"
    assert signals[0]["hour"] == 20
    assert signals[0]["day_of_week"] == "wednesday"
    assert signals[0]["activity_type"] == "message_inbound"
    assert signals[0]["event_type"] == EventType.MESSAGE_RECEIVED.value


def test_classify_activity_inbound_types(db, user_model_store):
    """_classify_activity should return distinct types for inbound vs outbound events."""
    extractor = TemporalExtractor(db, user_model_store)

    # Inbound types are distinct from outbound
    assert extractor._classify_activity(EventType.EMAIL_RECEIVED.value, {}) == "email_inbound"
    assert extractor._classify_activity(EventType.MESSAGE_RECEIVED.value, {}) == "message_inbound"

    # Outbound types remain unchanged
    assert extractor._classify_activity(EventType.EMAIL_SENT.value, {}) == "communication"
    assert extractor._classify_activity(EventType.MESSAGE_SENT.value, {}) == "communication"


def test_inbound_events_update_temporal_profile(db, user_model_store):
    """Inbound events should populate the temporal profile's hourly/daily/type counters."""
    extractor = TemporalExtractor(db, user_model_store)

    events = [
        {"type": EventType.EMAIL_RECEIVED.value, "timestamp": "2026-02-18T08:00:00Z", "payload": {}},
        {"type": EventType.EMAIL_RECEIVED.value, "timestamp": "2026-02-18T08:30:00Z", "payload": {}},
        {"type": EventType.MESSAGE_RECEIVED.value, "timestamp": "2026-02-18T14:00:00Z", "payload": {}},
    ]

    for event in events:
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None
    assert profile["samples_count"] == 3

    data = profile["data"]
    # Hourly counts
    assert data["activity_by_hour"]["8"] == 2
    assert data["activity_by_hour"]["14"] == 1
    # Daily counts (all on Wednesday)
    assert data["activity_by_day"]["wednesday"] == 3
    # Type counts
    assert data["activity_by_type"]["email_inbound"] == 2
    assert data["activity_by_type"]["message_inbound"] == 1


def test_inbound_and_outbound_coexist_in_profile(db, user_model_store):
    """Inbound and outbound events should both contribute to the same temporal profile."""
    extractor = TemporalExtractor(db, user_model_store)

    events = [
        {"type": EventType.EMAIL_SENT.value, "timestamp": "2026-02-18T10:00:00Z", "payload": {}},
        {"type": EventType.EMAIL_RECEIVED.value, "timestamp": "2026-02-18T10:15:00Z", "payload": {}},
        {"type": EventType.MESSAGE_SENT.value, "timestamp": "2026-02-18T14:00:00Z", "payload": {}},
        {"type": EventType.MESSAGE_RECEIVED.value, "timestamp": "2026-02-18T14:30:00Z", "payload": {}},
    ]

    for event in events:
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("temporal")
    assert profile["samples_count"] == 4

    data = profile["data"]
    # Both outbound and inbound types recorded
    assert data["activity_by_type"]["communication"] == 2  # sent events
    assert data["activity_by_type"]["email_inbound"] == 1
    assert data["activity_by_type"]["message_inbound"] == 1
    # Hourly counts include both directions
    assert data["activity_by_hour"]["10"] == 2  # one sent + one received
    assert data["activity_by_hour"]["14"] == 2  # one sent + one received
