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


# ---------------------------------------------------------------------------
# Inbound Message Tracking & read_not_replied Tests
# ---------------------------------------------------------------------------


def test_inbound_messages_increment_per_contact_inbound_count(
    cadence_extractor, user_model_store
):
    """Verify that inbound messages increment per_contact_inbound_count correctly."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # 3 inbound emails from Alice
    for i in range(3):
        event = {
            "id": f"evt-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "from_address": "alice@example.com",
                "body": f"Message {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    # 2 inbound messages from Bob
    for i in range(2):
        event = {
            "id": f"evt-bob-{i}",
            "type": EventType.MESSAGE_RECEIVED.value,
            "source": "imessage",
            "timestamp": timestamp,
            "payload": {
                "sender": "bob@example.com",
                "body": f"Hey {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    profile = user_model_store.get_signal_profile("cadence")
    inbound = profile["data"]["per_contact_inbound_count"]

    assert inbound["alice@example.com"] == 3
    assert inbound["bob@example.com"] == 2


def test_read_not_replied_computed_correctly(
    cadence_extractor, event_store, user_model_store
):
    """Verify read_not_replied is computed when some contacts have replies and others don't."""
    original_time = datetime.now(timezone.utc)
    timestamp = original_time.isoformat()

    # Alice sends 5 messages, user replies to 2
    for i in range(5):
        inbound = {
            "id": f"evt-alice-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "message_id": f"msg-alice-{i}",
                "from_address": "alice@example.com",
                "body": f"Alice msg {i}",
            },
            "metadata": {},
        }
        event_store.store_event(inbound)
        cadence_extractor.extract(inbound)

    # Reply to 2 of Alice's messages
    for i in range(2):
        reply = {
            "id": f"evt-alice-reply-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (original_time + timedelta(minutes=5)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"msg-alice-{i}",
                "to_addresses": ["alice@example.com"],
                "body": f"Reply {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(reply)

    # Bob sends 4 messages, user never replies
    for i in range(4):
        inbound = {
            "id": f"evt-bob-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "from_address": "bob@example.com",
                "body": f"Bob msg {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(inbound)

    profile = user_model_store.get_signal_profile("cadence")
    rnr = profile["data"]["read_not_replied"]

    # Both Alice (3 unreplied out of 5) and Bob (4 out of 4) should appear
    contacts = {entry["contact_id"]: entry for entry in rnr}

    assert "bob@example.com" in contacts
    assert contacts["bob@example.com"]["unreplied_count"] == 4
    assert contacts["bob@example.com"]["total_inbound"] == 4
    assert contacts["bob@example.com"]["unreplied_ratio"] == 1.0

    assert "alice@example.com" in contacts
    assert contacts["alice@example.com"]["unreplied_count"] == 3
    assert contacts["alice@example.com"]["total_inbound"] == 5
    assert abs(contacts["alice@example.com"]["unreplied_ratio"] - 0.6) < 0.01


def test_read_not_replied_excludes_low_volume_contacts(
    cadence_extractor, user_model_store
):
    """Verify contacts with fewer than 3 inbound messages are excluded."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Only 2 inbound messages from a contact (below threshold)
    for i in range(2):
        event = {
            "id": f"evt-low-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "from_address": "lowvol@example.com",
                "body": f"Message {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(event)

    profile = user_model_store.get_signal_profile("cadence")
    rnr = profile["data"].get("read_not_replied", [])

    # Should not include the low-volume contact
    contacts = [entry["contact_id"] for entry in rnr]
    assert "lowvol@example.com" not in contacts


def test_read_not_replied_ratio_is_correct(
    cadence_extractor, event_store, user_model_store
):
    """Verify unreplied_ratio = unreplied_count / total_inbound."""
    original_time = datetime.now(timezone.utc)
    timestamp = original_time.isoformat()

    # 10 inbound messages, user replies to 3
    for i in range(10):
        inbound = {
            "id": f"evt-ratio-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "message_id": f"msg-ratio-{i}",
                "from_address": "ratio@example.com",
                "body": f"Msg {i}",
            },
            "metadata": {},
        }
        event_store.store_event(inbound)
        cadence_extractor.extract(inbound)

    for i in range(3):
        reply = {
            "id": f"evt-ratio-reply-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (original_time + timedelta(minutes=1)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"msg-ratio-{i}",
                "to_addresses": ["ratio@example.com"],
                "body": f"Reply {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(reply)

    profile = user_model_store.get_signal_profile("cadence")
    rnr = profile["data"]["read_not_replied"]

    entry = next(e for e in rnr if e["contact_id"] == "ratio@example.com")
    assert entry["unreplied_count"] == 7
    assert entry["total_inbound"] == 10
    # 7 / 10 = 0.7
    assert abs(entry["unreplied_ratio"] - 0.7) < 0.01


def test_read_not_replied_sorted_by_ratio_descending(
    cadence_extractor, event_store, user_model_store
):
    """Verify the list is sorted by unreplied_ratio descending."""
    original_time = datetime.now(timezone.utc)
    timestamp = original_time.isoformat()

    # Contact A: 5 inbound, 4 replies → ratio 0.2
    for i in range(5):
        inbound = {
            "id": f"evt-a-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "message_id": f"msg-a-{i}",
                "from_address": "a@example.com",
                "body": f"A msg {i}",
            },
            "metadata": {},
        }
        event_store.store_event(inbound)
        cadence_extractor.extract(inbound)

    for i in range(4):
        reply = {
            "id": f"evt-a-reply-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (original_time + timedelta(minutes=1)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"msg-a-{i}",
                "to_addresses": ["a@example.com"],
                "body": f"Reply {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(reply)

    # Contact B: 5 inbound, 0 replies → ratio 1.0
    for i in range(5):
        inbound = {
            "id": f"evt-b-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "from_address": "b@example.com",
                "body": f"B msg {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(inbound)

    # Contact C: 4 inbound, 2 replies → ratio 0.5
    for i in range(4):
        inbound = {
            "id": f"evt-c-in-{i}",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "email",
            "timestamp": timestamp,
            "payload": {
                "message_id": f"msg-c-{i}",
                "from_address": "c@example.com",
                "body": f"C msg {i}",
            },
            "metadata": {},
        }
        event_store.store_event(inbound)
        cadence_extractor.extract(inbound)

    for i in range(2):
        reply = {
            "id": f"evt-c-reply-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": (original_time + timedelta(minutes=1)).isoformat(),
            "payload": {
                "is_reply": True,
                "in_reply_to": f"msg-c-{i}",
                "to_addresses": ["c@example.com"],
                "body": f"Reply {i}",
            },
            "metadata": {},
        }
        cadence_extractor.extract(reply)

    profile = user_model_store.get_signal_profile("cadence")
    rnr = profile["data"]["read_not_replied"]

    # Should be sorted: B (1.0) > C (0.5) > A (0.2)
    assert len(rnr) == 3
    assert rnr[0]["contact_id"] == "b@example.com"
    assert rnr[1]["contact_id"] == "c@example.com"
    assert rnr[2]["contact_id"] == "a@example.com"


def test_read_not_replied_backward_compat_missing_inbound_count(
    cadence_extractor, user_model_store
):
    """Verify that profiles without per_contact_inbound_count are handled gracefully."""
    # Simulate an old profile without per_contact_inbound_count by writing one directly
    user_model_store.update_signal_profile("cadence", {
        "response_times": [],
        "hourly_activity": {},
        "daily_activity": {},
        "per_contact_response_times": {},
        "per_channel_response_times": {},
        "per_contact_initiations": {},
        # Intentionally omitting per_contact_inbound_count
    })

    # Processing a new inbound event should work without errors
    timestamp = datetime.now(timezone.utc).isoformat()
    event = {
        "id": "evt-compat-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": timestamp,
        "payload": {
            "from_address": "compat@example.com",
            "body": "Backward compat test",
        },
        "metadata": {},
    }
    cadence_extractor.extract(event)

    profile = user_model_store.get_signal_profile("cadence")
    # Should have created the field and counted the message
    assert profile["data"]["per_contact_inbound_count"]["compat@example.com"] == 1


def test_inbound_signal_emitted_for_all_received_messages(
    cadence_extractor, user_model_store
):
    """Verify cadence_inbound_received is emitted for both replies and new conversations."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Non-reply inbound
    event_new = {
        "id": "evt-new-conv",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": timestamp,
        "payload": {
            "from_address": "sender@example.com",
            "body": "Starting a new conversation",
        },
        "metadata": {},
    }
    signals_new = cadence_extractor.extract(event_new)

    # Reply inbound
    event_reply = {
        "id": "evt-reply-conv",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": timestamp,
        "payload": {
            "is_reply": True,
            "in_reply_to": "some-msg-id",
            "from_address": "sender@example.com",
            "body": "Replying to your message",
        },
        "metadata": {},
    }
    signals_reply = cadence_extractor.extract(event_reply)

    # Both should have cadence_inbound_received signals
    inbound_new = [s for s in signals_new if s["type"] == "cadence_inbound_received"]
    inbound_reply = [s for s in signals_reply if s["type"] == "cadence_inbound_received"]
    assert len(inbound_new) == 1
    assert len(inbound_reply) == 1

    # Both should point to the same contact
    assert inbound_new[0]["contact_id"] == "sender@example.com"
    assert inbound_reply[0]["contact_id"] == "sender@example.com"

    # Count should be 2 total
    profile = user_model_store.get_signal_profile("cadence")
    assert profile["data"]["per_contact_inbound_count"]["sender@example.com"] == 2
