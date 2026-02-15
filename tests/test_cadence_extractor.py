"""
Tests for the CadenceExtractor service.

The CadenceExtractor tracks when and how quickly the user communicates,
revealing priorities, avoidance patterns, and natural activity rhythms.

These tests verify:
  - Correct event filtering (email/message sent/received)
  - Response time calculation from in_reply_to message lookups
  - Activity window detection (hour-of-day, day-of-week histograms)
  - Profile persistence with per-contact and per-channel bucketing
  - Capping of global response time lists to prevent unbounded growth
  - Fail-open handling of malformed timestamps and missing original messages
"""

from datetime import datetime, timezone, timedelta

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor


@pytest.fixture
def cadence_extractor(db, user_model_store):
    """Create a CadenceExtractor instance with a test database."""
    return CadenceExtractor(db=db, user_model_store=user_model_store)


# ---------------------------------------------------------------------------
# Event Filtering Tests
# ---------------------------------------------------------------------------


def test_can_process_email_sent(cadence_extractor):
    """Verify that sent emails are processed."""
    event = {
        "type": EventType.EMAIL_SENT.value,
        "payload": {"body": "Test"},
    }
    assert cadence_extractor.can_process(event) is True


def test_can_process_email_received(cadence_extractor):
    """Verify that received emails are processed."""
    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "payload": {"body": "Test"},
    }
    assert cadence_extractor.can_process(event) is True


def test_can_process_message_sent(cadence_extractor):
    """Verify that sent messages are processed."""
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "payload": {"body": "Test"},
    }
    assert cadence_extractor.can_process(event) is True


def test_can_process_message_received(cadence_extractor):
    """Verify that received messages are processed."""
    event = {
        "type": EventType.MESSAGE_RECEIVED.value,
        "payload": {"body": "Test"},
    }
    assert cadence_extractor.can_process(event) is True


def test_ignores_non_communication_events(cadence_extractor):
    """Verify that non-communication events are ignored."""
    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "payload": {},
    }
    assert cadence_extractor.can_process(event) is False


# ---------------------------------------------------------------------------
# Response Time Calculation Tests
# ---------------------------------------------------------------------------


def test_response_time_calculation_for_email_reply(
    cadence_extractor, event_store, user_model_store
):
    """Verify response time is calculated when replying to an email."""
    # Store the original inbound email
    original_time = datetime.now(timezone.utc)
    original_event = {
        "id": "evt-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-123",
            "from_address": "alice@example.com",
            "subject": "Question",
            "body": "What do you think?",
        },
        "metadata": {},
    }
    event_store.store_event(original_event)

    # Simulate the user replying 2 hours later
    reply_time = original_time + timedelta(hours=2)
    reply_event = {
        "id": "evt-2",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": reply_time.isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-123",
            "to_addresses": ["alice@example.com"],
            "body": "I think it's great!",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(reply_event)

    # Should produce a response-time signal
    response_signals = [s for s in signals if s["type"] == "cadence_response_time"]
    assert len(response_signals) == 1

    signal = response_signals[0]
    assert signal["contact_id"] == "alice@example.com"
    assert signal["channel"] == "email"
    # 2 hours = 7200 seconds
    assert abs(signal["response_time_seconds"] - 7200) < 1


def test_response_time_calculation_for_message_reply(
    cadence_extractor, event_store, user_model_store
):
    """Verify response time is calculated when replying to a message."""
    # Store the original inbound message
    original_time = datetime.now(timezone.utc)
    original_event = {
        "id": "evt-1",
        "type": EventType.MESSAGE_RECEIVED.value,
        "source": "imessage",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-456",
            "from": "+15551234567",
            "body": "Are we still on for lunch?",
        },
        "metadata": {},
    }
    event_store.store_event(original_event)

    # User replies 15 minutes later
    reply_time = original_time + timedelta(minutes=15)
    reply_event = {
        "id": "evt-2",
        "type": EventType.MESSAGE_SENT.value,
        "source": "imessage",
        "timestamp": reply_time.isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-456",
            "to_addresses": ["+15551234567"],
            "body": "Yes, see you at noon!",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(reply_event)

    response_signals = [s for s in signals if s["type"] == "cadence_response_time"]
    assert len(response_signals) == 1

    signal = response_signals[0]
    assert signal["contact_id"] == "+15551234567"
    assert signal["channel"] == "imessage"
    # 15 minutes = 900 seconds
    assert abs(signal["response_time_seconds"] - 900) < 1


def test_no_response_time_signal_when_original_message_not_found(
    cadence_extractor, user_model_store
):
    """Verify that missing original message returns None (fail-open)."""
    reply_time = datetime.now(timezone.utc)
    reply_event = {
        "id": "evt-2",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": reply_time.isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "nonexistent-msg-id",
            "to_addresses": ["alice@example.com"],
            "body": "Reply to nowhere",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(reply_event)

    # Should NOT produce a response-time signal
    response_signals = [s for s in signals if s["type"] == "cadence_response_time"]
    assert len(response_signals) == 0

    # Should still produce an activity signal
    activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
    assert len(activity_signals) == 1


def test_no_response_time_signal_for_non_reply_messages(
    cadence_extractor, user_model_store
):
    """Verify that non-reply outbound messages don't produce response-time signals."""
    event = {
        "id": "evt-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "is_reply": False,
            "to_addresses": ["bob@example.com"],
            "subject": "New thread",
            "body": "Starting a new conversation",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(event)

    # No response-time signal (not a reply)
    response_signals = [s for s in signals if s["type"] == "cadence_response_time"]
    assert len(response_signals) == 0

    # Should still have activity signal
    activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
    assert len(activity_signals) == 1


def test_negative_response_time_is_discarded(
    cadence_extractor, event_store, user_model_store
):
    """Verify that negative response times (clock skew) are discarded."""
    # Store original message with LATER timestamp
    original_time = datetime.now(timezone.utc) + timedelta(hours=1)
    original_event = {
        "id": "evt-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-future",
            "from_address": "alice@example.com",
            "body": "Future message",
        },
        "metadata": {},
    }
    event_store.store_event(original_event)

    # Reply with EARLIER timestamp (clock skew scenario)
    reply_time = datetime.now(timezone.utc)
    reply_event = {
        "id": "evt-2",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": reply_time.isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-future",
            "to_addresses": ["alice@example.com"],
            "body": "Past reply",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(reply_event)

    # Negative delta should be discarded (fail-open)
    response_signals = [s for s in signals if s["type"] == "cadence_response_time"]
    assert len(response_signals) == 0


# ---------------------------------------------------------------------------
# Activity Window Detection Tests
# ---------------------------------------------------------------------------


def test_activity_signal_captures_hour_and_day_of_week(
    cadence_extractor, user_model_store
):
    """Verify that activity signals capture hour and day-of-week correctly."""
    # Create a specific timestamp: Tuesday, 2:00 PM UTC
    timestamp = datetime(2026, 2, 17, 14, 0, 0, tzinfo=timezone.utc)
    event = {
        "id": "evt-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": timestamp.isoformat(),
        "payload": {
            "to_addresses": ["test@example.com"],
            "body": "Test",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(event)

    activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
    assert len(activity_signals) == 1

    signal = activity_signals[0]
    assert signal["hour"] == 14
    assert signal["day_of_week"] == "tuesday"
    assert signal["direction"] == "outbound"
    assert signal["channel"] == "email"


def test_activity_signal_distinguishes_inbound_outbound(
    cadence_extractor, user_model_store
):
    """Verify that activity signals distinguish inbound vs outbound."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Inbound message
    inbound_event = {
        "id": "evt-1",
        "type": EventType.MESSAGE_RECEIVED.value,
        "source": "imessage",
        "timestamp": timestamp,
        "payload": {"from": "+15551234567", "body": "Hey"},
        "metadata": {},
    }

    inbound_signals = cadence_extractor.extract(inbound_event)
    inbound_activity = [s for s in inbound_signals if s["type"] == "cadence_activity"]
    assert len(inbound_activity) == 1
    assert inbound_activity[0]["direction"] == "inbound"

    # Outbound message
    outbound_event = {
        "id": "evt-2",
        "type": EventType.MESSAGE_SENT.value,
        "source": "imessage",
        "timestamp": timestamp,
        "payload": {"to_addresses": ["+15551234567"], "body": "Hi"},
        "metadata": {},
    }

    outbound_signals = cadence_extractor.extract(outbound_event)
    outbound_activity = [s for s in outbound_signals if s["type"] == "cadence_activity"]
    assert len(outbound_activity) == 1
    assert outbound_activity[0]["direction"] == "outbound"


def test_activity_signal_handles_malformed_timestamp(
    cadence_extractor, user_model_store
):
    """Verify that malformed timestamps are handled gracefully (fail-open)."""
    event = {
        "id": "evt-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "not-a-valid-timestamp",
        "payload": {
            "to_addresses": ["test@example.com"],
            "body": "Test",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(event)

    # Should not produce an activity signal, but also should not crash
    activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
    assert len(activity_signals) == 0


def test_activity_signal_handles_missing_timestamp(
    cadence_extractor, user_model_store
):
    """Verify that missing timestamps are handled gracefully."""
    event = {
        "id": "evt-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        # No timestamp field
        "payload": {
            "to_addresses": ["test@example.com"],
            "body": "Test",
        },
        "metadata": {},
    }

    signals = cadence_extractor.extract(event)

    # Should not produce an activity signal, but also should not crash
    activity_signals = [s for s in signals if s["type"] == "cadence_activity"]
    assert len(activity_signals) == 0


# ---------------------------------------------------------------------------
# Profile Persistence Tests
# ---------------------------------------------------------------------------


def test_profile_persistence_basic(cadence_extractor, user_model_store):
    """Verify that signals are persisted into the cadence profile."""
    timestamp = datetime.now(timezone.utc).isoformat()
    event = {
        "id": "evt-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": timestamp,
        "payload": {
            "to_addresses": ["test@example.com"],
            "body": "Test",
        },
        "metadata": {},
    }

    cadence_extractor.extract(event)

    # Load the profile
    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None
    assert "hourly_activity" in profile["data"]
    assert "daily_activity" in profile["data"]


def test_profile_tracks_response_times_globally(
    cadence_extractor, event_store, user_model_store
):
    """Verify that response times are tracked in global list."""
    # Store original message
    original_time = datetime.now(timezone.utc)
    original_event = {
        "id": "evt-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-1",
            "from_address": "alice@example.com",
            "body": "Question?",
        },
        "metadata": {},
    }
    event_store.store_event(original_event)

    # Reply
    reply_time = original_time + timedelta(hours=1)
    reply_event = {
        "id": "evt-2",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": reply_time.isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-1",
            "to_addresses": ["alice@example.com"],
            "body": "Answer.",
        },
        "metadata": {},
    }

    cadence_extractor.extract(reply_event)

    # Check global response_times list
    profile = user_model_store.get_signal_profile("cadence")
    assert "response_times" in profile["data"]
    assert len(profile["data"]["response_times"]) == 1
    # 1 hour = 3600 seconds
    assert abs(profile["data"]["response_times"][0] - 3600) < 1


def test_profile_tracks_per_contact_response_times(
    cadence_extractor, event_store, user_model_store
):
    """Verify that response times are bucketed per contact."""
    # Store messages from two different contacts
    original_time = datetime.now(timezone.utc)

    # Message from Alice
    event_alice = {
        "id": "evt-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-alice",
            "from_address": "alice@example.com",
            "body": "Question from Alice?",
        },
        "metadata": {},
    }
    event_store.store_event(event_alice)

    # Message from Bob
    event_bob = {
        "id": "evt-2",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-bob",
            "from_address": "bob@example.com",
            "body": "Question from Bob?",
        },
        "metadata": {},
    }
    event_store.store_event(event_bob)

    # Reply to Alice (fast: 5 minutes)
    reply_alice = {
        "id": "evt-3",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": (original_time + timedelta(minutes=5)).isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-alice",
            "to_addresses": ["alice@example.com"],
            "body": "Quick reply to Alice",
        },
        "metadata": {},
    }
    cadence_extractor.extract(reply_alice)

    # Reply to Bob (slow: 2 hours)
    reply_bob = {
        "id": "evt-4",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": (original_time + timedelta(hours=2)).isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-bob",
            "to_addresses": ["bob@example.com"],
            "body": "Slow reply to Bob",
        },
        "metadata": {},
    }
    cadence_extractor.extract(reply_bob)

    # Check per_contact_response_times
    profile = user_model_store.get_signal_profile("cadence")
    per_contact = profile["data"]["per_contact_response_times"]

    assert "alice@example.com" in per_contact
    assert "bob@example.com" in per_contact

    # Alice: 5 minutes = 300 seconds
    assert abs(per_contact["alice@example.com"][0] - 300) < 1

    # Bob: 2 hours = 7200 seconds
    assert abs(per_contact["bob@example.com"][0] - 7200) < 1


def test_profile_tracks_per_channel_response_times(
    cadence_extractor, event_store, user_model_store
):
    """Verify that response times are bucketed per channel."""
    original_time = datetime.now(timezone.utc)

    # Email message
    event_email = {
        "id": "evt-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-email",
            "from_address": "alice@example.com",
            "body": "Email question?",
        },
        "metadata": {},
    }
    event_store.store_event(event_email)

    # iMessage message
    event_imessage = {
        "id": "evt-2",
        "type": EventType.MESSAGE_RECEIVED.value,
        "source": "imessage",
        "timestamp": original_time.isoformat(),
        "payload": {
            "message_id": "msg-imessage",
            "from": "+15551234567",
            "body": "Text question?",
        },
        "metadata": {},
    }
    event_store.store_event(event_imessage)

    # Reply to email (slow: 3 hours)
    reply_email = {
        "id": "evt-3",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": (original_time + timedelta(hours=3)).isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-email",
            "to_addresses": ["alice@example.com"],
            "body": "Email reply",
        },
        "metadata": {},
    }
    cadence_extractor.extract(reply_email)

    # Reply to iMessage (fast: 2 minutes)
    reply_imessage = {
        "id": "evt-4",
        "type": EventType.MESSAGE_SENT.value,
        "source": "imessage",
        "timestamp": (original_time + timedelta(minutes=2)).isoformat(),
        "payload": {
            "is_reply": True,
            "in_reply_to": "msg-imessage",
            "to_addresses": ["+15551234567"],
            "body": "Text reply",
        },
        "metadata": {},
    }
    cadence_extractor.extract(reply_imessage)

    # Check per_channel_response_times
    profile = user_model_store.get_signal_profile("cadence")
    per_channel = profile["data"]["per_channel_response_times"]

    assert "email" in per_channel
    assert "imessage" in per_channel

    # Email: 3 hours = 10800 seconds
    assert abs(per_channel["email"][0] - 10800) < 1

    # iMessage: 2 minutes = 120 seconds
    assert abs(per_channel["imessage"][0] - 120) < 1


def test_profile_caps_global_response_times_at_1000(
    cadence_extractor, event_store, user_model_store
):
    """Verify that global response_times list is capped at 1000 entries."""
    original_time = datetime.now(timezone.utc)

    # Create 1100 reply events
    for i in range(1100):
        # Store original message
        original_event = {
            "id": f"evt-orig-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": original_time.isoformat(),
            "payload": {
                "message_id": f"msg-{i}",
                "from_address": "alice@example.com",
                "body": f"Question {i}?",
            },
            "metadata": {},
        }
        event_store.store_event(original_event)

        # Reply
        reply_event = {
            "id": f"evt-reply-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (original_time + timedelta(seconds=i)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"msg-{i}",
                "to_addresses": ["alice@example.com"],
                "body": f"Answer {i}.",
            },
            "metadata": {},
        }
        cadence_extractor.extract(reply_event)

    # Check that list is capped at 1000
    profile = user_model_store.get_signal_profile("cadence")
    assert len(profile["data"]["response_times"]) == 1000

    # Should keep the MOST RECENT 1000 entries (indices 100-1099)
    # The oldest entry should correspond to i=100, which has response_time ~= 100 seconds
    assert profile["data"]["response_times"][0] >= 90  # Allow some tolerance


def test_profile_accumulates_hourly_activity_histogram(
    cadence_extractor, user_model_store
):
    """Verify that hourly activity histogram accumulates correctly."""
    # Send 3 emails at hour 14 (2 PM)
    for i in range(3):
        timestamp = datetime(2026, 2, 17, 14, i, 0, tzinfo=timezone.utc)
        event = {
            "id": f"evt-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": timestamp.isoformat(),
            "payload": {
                "to_addresses": ["test@example.com"],
                "body": f"Test {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    # Send 2 emails at hour 9 (9 AM)
    for i in range(2):
        timestamp = datetime(2026, 2, 17, 9, i, 0, tzinfo=timezone.utc)
        event = {
            "id": f"evt-morning-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": timestamp.isoformat(),
            "payload": {
                "to_addresses": ["test@example.com"],
                "body": f"Morning {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    # Check histogram
    profile = user_model_store.get_signal_profile("cadence")
    hourly = profile["data"]["hourly_activity"]

    assert hourly["14"] == 3
    assert hourly["9"] == 2


def test_profile_accumulates_daily_activity_histogram(
    cadence_extractor, user_model_store
):
    """Verify that daily activity histogram accumulates correctly."""
    # Monday
    for i in range(4):
        timestamp = datetime(2026, 2, 16, 10, i, 0, tzinfo=timezone.utc)  # Monday
        event = {
            "id": f"evt-mon-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": timestamp.isoformat(),
            "payload": {
                "to_addresses": ["test@example.com"],
                "body": f"Monday {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    # Friday
    for i in range(2):
        timestamp = datetime(2026, 2, 20, 10, i, 0, tzinfo=timezone.utc)  # Friday
        event = {
            "id": f"evt-fri-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": timestamp.isoformat(),
            "payload": {
                "to_addresses": ["test@example.com"],
                "body": f"Friday {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    # Check histogram
    profile = user_model_store.get_signal_profile("cadence")
    daily = profile["data"]["daily_activity"]

    assert daily["monday"] == 4
    assert daily["friday"] == 2
