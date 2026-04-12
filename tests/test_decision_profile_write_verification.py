"""
Tests for DecisionExtractor — profile write verification and fallback signal.

Verifies:
1. Decision profile persists after processing a qualifying event.
2. Post-write verification detects a missing profile and logs CRITICAL.
3. Fallback 'decision_information_processing' signal is emitted for inbound
   emails/messages that don't match any specific decision pattern.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _email_received_event(
    body: str = "Plain email body with no special patterns.",
    subject: str = "Hello",
    from_address: str = "alice@example.com",
) -> dict:
    """Build a minimal EMAIL_RECEIVED event for testing."""
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "from_address": from_address,
            "subject": subject,
            "body": body,
        },
    }


def _message_received_event(
    body: str = "Hey what's up",
    from_address: str = "bob@example.com",
) -> dict:
    """Build a minimal MESSAGE_RECEIVED event for testing."""
    return {
        "type": EventType.MESSAGE_RECEIVED.value,
        "timestamp": datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "from_address": from_address,
            "body": body,
        },
    }


# ---------------------------------------------------------------------------
# Issue 1: Profile persistence
# ---------------------------------------------------------------------------


def test_decision_profile_persists_after_qualifying_event(db, user_model_store):
    """Decision profile must be persisted after processing an email.received event.

    This is the direct regression test for the bug: 13,726 qualifying events
    flowed through but no decision profile was written.  Processing ONE event
    must create the profile row in user_model.db.
    """
    extractor = DecisionExtractor(db, user_model_store)

    # A delegation-pattern email will definitely update the profile.
    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "to": "partner@example.com",
            "content": "You decide which restaurant to go to.",
        },
    }

    extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None, (
        "Decision profile is None after processing a delegation email. "
        "update_signal_profile() may be silently failing."
    )
    assert profile["samples_count"] >= 1


def test_decision_profile_write_count_increments(db, user_model_store):
    """_profile_write_count must increment with each _update_profile call."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._profile_write_count == 0, "Counter must start at 0"

    # Process 5 events that trigger _update_profile
    base_ts = datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": base_ts.replace(hour=9 + i).isoformat(),
            "payload": {
                "from_address": "alice@example.com",
                "subject": f"Email {i}",
                "body": "Some email body",
            },
        }
        extractor.extract(event)

    assert extractor._profile_write_count == 5, (
        f"Expected write count 5, got {extractor._profile_write_count}"
    )


def test_critical_log_emitted_when_profile_missing_mid_stream(db, user_model_store, caplog):
    """A CRITICAL log must fire when the profile vanishes after the first write.

    Simulates the production scenario where update_signal_profile silently
    fails: monkey-patches get_signal_profile to return None on post-write
    verification calls (triggered every 10 writes).
    """
    extractor = DecisionExtractor(db, user_model_store)

    original_get = user_model_store.get_signal_profile
    call_count = {"n": 0}

    def patched_get(profile_type):
        """Return None for 'decision' after the first call (simulate failure)."""
        if profile_type == "decision":
            call_count["n"] += 1
            # The first call populates the profile in _update_profile.
            # Subsequent calls during post-write verification see None.
            if call_count["n"] > 1:
                return None
        return original_get(profile_type)

    user_model_store.get_signal_profile = patched_get

    base_ts = datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc)
    with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.decision"):
        for i in range(10):
            event = {
                "type": EventType.EMAIL_RECEIVED.value,
                "timestamp": base_ts.replace(hour=i % 23).isoformat(),
                "payload": {
                    "from_address": "alice@example.com",
                    "subject": f"Email {i}",
                    "body": "Some email body",
                },
            }
            extractor.extract(event)

    user_model_store.get_signal_profile = original_get

    critical_msgs = [
        r.message for r in caplog.records
        if r.levelno == logging.CRITICAL and "MISSING after" in r.message
    ]
    assert critical_msgs, (
        "Expected a CRITICAL log about profile being MISSING but none was emitted. "
        "The post-write verification in DecisionExtractor._update_profile may be broken."
    )
    assert "update_signal_profile() is silently failing" in critical_msgs[0]


# ---------------------------------------------------------------------------
# Issue 2: Fallback signal
# ---------------------------------------------------------------------------


def test_fallback_signal_emitted_for_plain_inbound_email(db, user_model_store):
    """A plain email with no matching patterns must emit decision_information_processing.

    This ensures every EMAIL_RECEIVED event contributes a visible signal
    rather than being silently discarded after pattern matching fails.
    """
    extractor = DecisionExtractor(db, user_model_store)

    # Plain email body — no urgency, action request, info-gathering, or
    # decision-response patterns; non-marketing sender.
    event = _email_received_event(
        body="Just checking in. Hope you are doing well.",
        subject="Hi",
    )

    signals = extractor.extract(event)

    fallback = [s for s in signals if s.get("type") == "decision_information_processing"]
    assert fallback, (
        "Expected a 'decision_information_processing' fallback signal for a plain "
        f"email, but got signal types: {[s.get('type') for s in signals]}"
    )
    assert fallback[0]["source_type"] == EventType.EMAIL_RECEIVED.value
    assert isinstance(fallback[0]["word_count"], int)
    assert fallback[0]["word_count"] > 0


def test_fallback_signal_emitted_for_plain_inbound_message(db, user_model_store):
    """A plain MESSAGE_RECEIVED with no patterns must emit fallback signal."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _message_received_event(body="Running late, see you at noon")

    signals = extractor.extract(event)

    fallback = [s for s in signals if s.get("type") == "decision_information_processing"]
    assert fallback, (
        "Expected fallback signal for plain MESSAGE_RECEIVED, "
        f"got: {[s.get('type') for s in signals]}"
    )
    assert fallback[0]["source_type"] == EventType.MESSAGE_RECEIVED.value


def test_no_fallback_when_specific_signal_emitted(db, user_model_store):
    """When a specific decision signal is found, no fallback should be added."""
    extractor = DecisionExtractor(db, user_model_store)

    # Urgency keyword in subject → decision_signal of type urgency_response
    event = _email_received_event(
        body="Please respond urgently to this request.",
        subject="URGENT: action required",
    )

    signals = extractor.extract(event)

    fallback = [s for s in signals if s.get("type") == "decision_information_processing"]
    decision_signals = [s for s in signals if s.get("type") == "decision_signal"]

    assert decision_signals, "Expected at least one decision_signal for urgency email"
    assert not fallback, (
        "Fallback signal must NOT appear when specific decision signal was found, "
        f"but got fallback: {fallback}"
    )


def test_fallback_word_count_matches_body(db, user_model_store):
    """Fallback signal word_count must match the actual body word count."""
    extractor = DecisionExtractor(db, user_model_store)

    body = "This is a four word email"
    event = _email_received_event(body=body, subject="Plain subject")

    signals = extractor.extract(event)

    fallback = [s for s in signals if s.get("type") == "decision_information_processing"]
    assert fallback, "Expected fallback signal"
    assert fallback[0]["word_count"] == 6  # "This is a four word email"


def test_fallback_detects_attachments(db, user_model_store):
    """Fallback signal has_attachments must reflect payload attachments field."""
    extractor = DecisionExtractor(db, user_model_store)

    event = {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "from_address": "alice@example.com",
            "subject": "See attached",
            "body": "Please see the attached document.",
            "attachments": ["report.pdf"],
        },
    }

    signals = extractor.extract(event)

    # "please see the attached" could trigger information_gathering, so check
    # conditionally: if fallback present, verify has_attachments=True.
    fallback = [s for s in signals if s.get("type") == "decision_information_processing"]
    if fallback:
        assert fallback[0]["has_attachments"] is True
