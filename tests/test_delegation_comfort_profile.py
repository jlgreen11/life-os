"""
Tests for delegation_comfort and delegation_by_domain population in DecisionExtractor.

Prior to this fix the delegation_pattern handler in _update_profile only tracked
fatigue_time_of_day and silently discarded the delegation signal data.  These
tests verify that:

1. delegation_comfort is updated toward 1.0 when delegation messages arrive
2. delegation_comfort is pulled toward 0.0 by non-delegation outbound messages
3. delegation_by_domain is populated per recipient
4. defers_to is populated with recipient per inferred category
5. The EMA smoothing prevents single-message spikes
6. Non-delegation messages do NOT appear in the returned signal list
7. A mix of delegation and non-delegation messages produces a correct ratio
"""

import pytest
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sent_event(content: str, recipient: str, hour: int = 14) -> dict:
    """Build a MESSAGE_SENT event with the given content and recipient."""
    now = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
    return {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "content": content,
            "recipient": recipient,
        },
    }


def _make_email_sent(body: str, to: str, hour: int = 10) -> dict:
    """Build an EMAIL_SENT event with the given body and recipient."""
    now = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
    return {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "body": body,
            "to": to,
        },
    }


# ---------------------------------------------------------------------------
# delegation_comfort
# ---------------------------------------------------------------------------

def test_delegation_comfort_increases_on_full_delegation(db, user_model_store):
    """delegation_comfort should rise when a 'you decide' message is sent."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_sent_event("I don't care, you decide!", "partner")
    signals = extractor.extract(event)

    # Signal list should include the delegation_pattern signal
    assert any(s["type"] == "delegation_pattern" for s in signals)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    comfort = profile["data"]["delegation_comfort"]
    # Starting from 0.5 (default), EMA with ratio=1.0 → 0.7*0.5 + 0.3*1.0 = 0.65
    assert comfort == pytest.approx(0.65, abs=0.01)


def test_delegation_comfort_decreases_on_non_delegation(db, user_model_store):
    """delegation_comfort should fall when non-delegation messages dominate."""
    extractor = DecisionExtractor(db, user_model_store)

    # Send 5 decisive (non-delegation) messages
    decisive_messages = [
        "Let's meet at noon.",
        "I'll handle this project.",
        "Confirmed — booking the flight now.",
        "Setting up the meeting for Thursday.",
        "Going ahead with option A.",
    ]
    for msg in decisive_messages:
        extractor.extract(_make_sent_event(msg, "colleague"))

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    comfort = profile["data"]["delegation_comfort"]

    # After 5 non-delegation messages with no delegations, ratio=0 →
    # comfort should have drifted well below the 0.5 default.
    assert comfort < 0.5


def test_delegation_comfort_ratio_accuracy(db, user_model_store):
    """With 1 delegation out of 4 total messages, comfort should reflect ~25% ratio."""
    extractor = DecisionExtractor(db, user_model_store)

    # 3 decisive messages
    for msg in ["I'll handle it.", "Going with plan B.", "Confirmed."]:
        extractor.extract(_make_sent_event(msg, "boss"))

    # 1 delegation
    extractor.extract(_make_sent_event("Whatever you think is best, you decide.", "boss"))

    profile = user_model_store.get_signal_profile("decision")
    comfort = profile["data"]["delegation_comfort"]

    # EMA trace:
    #   step 1 (non-deleg): ratio=0/1=0.0  → 0.7*0.5  + 0.3*0.0  = 0.35
    #   step 2 (non-deleg): ratio=0/2=0.0  → 0.7*0.35 + 0.3*0.0  = 0.245
    #   step 3 (non-deleg): ratio=0/3=0.0  → 0.7*0.245+ 0.3*0.0  = 0.1715
    #   step 4 (deleg):     ratio=1/4=0.25 → 0.7*0.1715+0.3*0.25 = 0.1951
    # The EMA depresses comfort before the delegation arrives, which is the
    # correct behaviour (3 decisive messages do outweigh 1 delegation).
    assert comfort == pytest.approx(0.1951, abs=0.001)


def test_outbound_nondelegation_not_in_returned_signals(db, user_model_store):
    """Non-delegation outbound messages should NOT appear in the returned signal list."""
    extractor = DecisionExtractor(db, user_model_store)

    event = _make_sent_event("Let's meet at 3pm.", "coworker")
    signals = extractor.extract(event)

    # outbound_nondelegation is an internal accounting token and must be stripped
    assert all(s["type"] != "outbound_nondelegation" for s in signals)
    # No meaningful signals for a non-delegation message
    assert signals == []


def test_delegation_does_not_appear_in_nondelegation_counter(db, user_model_store):
    """A delegation message should increment both delegation and total counters."""
    extractor = DecisionExtractor(db, user_model_store)

    extractor.extract(_make_sent_event("You choose — I trust you.", "partner"))

    profile = user_model_store.get_signal_profile("decision")
    data = profile["data"]
    assert data["_delegation_event_count"] == 1
    assert data["_total_outbound_count"] == 1


def test_nondelegation_increments_only_total_counter(db, user_model_store):
    """A non-delegation message should increment only the total counter."""
    extractor = DecisionExtractor(db, user_model_store)

    extractor.extract(_make_sent_event("I'll take care of it.", "manager"))

    profile = user_model_store.get_signal_profile("decision")
    data = profile["data"]
    assert data.get("_delegation_event_count", 0) == 0
    assert data["_total_outbound_count"] == 1


# ---------------------------------------------------------------------------
# delegation_by_domain
# ---------------------------------------------------------------------------

def test_delegation_by_domain_populated_for_recipient(db, user_model_store):
    """delegation_by_domain should record delegation tendency per recipient."""
    extractor = DecisionExtractor(db, user_model_store)

    extractor.extract(_make_sent_event("Whatever you want to eat, you pick.", "partner"))

    profile = user_model_store.get_signal_profile("decision")
    by_domain = profile["data"].get("delegation_by_domain", {})
    assert "partner" in by_domain
    # First delegation → score = 1.0
    assert by_domain["partner"] == pytest.approx(1.0, abs=0.01)


def test_delegation_by_domain_ema_on_repeat_delegation(db, user_model_store):
    """Repeated delegation to the same recipient should apply EMA smoothing."""
    extractor = DecisionExtractor(db, user_model_store)

    extractor.extract(_make_sent_event("You decide.", "partner"))  # score = 1.0
    extractor.extract(_make_sent_event("Up to you.", "partner"))   # EMA: 0.7*1.0 + 0.3*1.0 = 1.0

    profile = user_model_store.get_signal_profile("decision")
    by_domain = profile["data"]["delegation_by_domain"]
    # Two full delegations → stays at 1.0
    assert by_domain["partner"] == pytest.approx(1.0, abs=0.01)


def test_delegation_by_domain_tracks_multiple_recipients(db, user_model_store):
    """delegation_by_domain should track each recipient independently."""
    extractor = DecisionExtractor(db, user_model_store)

    extractor.extract(_make_sent_event("You decide on dinner.", "partner"))
    extractor.extract(
        _make_email_sent("Any thoughts on the proposal?", "boss@work.com", hour=10)
    )

    profile = user_model_store.get_signal_profile("decision")
    by_domain = profile["data"]["delegation_by_domain"]
    assert "partner" in by_domain
    assert "boss@work.com" in by_domain


# ---------------------------------------------------------------------------
# defers_to
# ---------------------------------------------------------------------------

def test_defers_to_populated_evening_full_delegation(db, user_model_store):
    """Full delegation in the evening should add recipient to 'personal' category."""
    extractor = DecisionExtractor(db, user_model_store)

    # Hour 20 = 8pm → personal category for full delegation
    event = _make_sent_event("You decide, whatever you prefer.", "partner", hour=20)
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    defers_to = profile["data"].get("defers_to", {})
    assert "partner" in defers_to.get("personal", [])


def test_defers_to_populated_work_hours_seeking_input(db, user_model_store):
    """Seeking input during work hours should add recipient to 'work' category."""
    extractor = DecisionExtractor(db, user_model_store)

    # Hour 10 = 10am → work category for seeking_input
    event = _make_email_sent("What do you think about this approach?", "colleague@co.com", hour=10)
    extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    defers_to = profile["data"].get("defers_to", {})
    assert "colleague@co.com" in defers_to.get("work", [])


def test_defers_to_no_duplicate_entries(db, user_model_store):
    """Sending multiple delegation messages to the same contact should not duplicate defers_to entries."""
    extractor = DecisionExtractor(db, user_model_store)

    for _ in range(3):
        extractor.extract(_make_sent_event("You choose.", "partner", hour=18))

    profile = user_model_store.get_signal_profile("decision")
    defers_to = profile["data"].get("defers_to", {})
    # 'partner' should appear exactly once in 'personal'
    assert defers_to.get("personal", []).count("partner") == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_recipient_does_not_crash(db, user_model_store):
    """Delegation with no recipient should update comfort but skip per-contact tracking."""
    extractor = DecisionExtractor(db, user_model_store)

    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"body": "You decide.", "to": ""},
    }
    signals = extractor.extract(event)

    # Should produce a delegation_pattern signal
    assert any(s["type"] == "delegation_pattern" for s in signals)

    # delegation_by_domain should be empty (no recipient to track)
    profile = user_model_store.get_signal_profile("decision")
    by_domain = profile["data"].get("delegation_by_domain", {})
    # No entry for empty string
    assert "" not in by_domain


def test_persisted_counters_survive_multiple_calls(db, user_model_store):
    """Delegation and total counters should accumulate correctly across multiple extract() calls."""
    extractor = DecisionExtractor(db, user_model_store)

    # 2 non-delegation, 1 delegation
    extractor.extract(_make_sent_event("Confirmed.", "boss"))
    extractor.extract(_make_sent_event("Done.", "boss"))
    extractor.extract(_make_sent_event("Whatever you think, you decide.", "boss"))

    profile = user_model_store.get_signal_profile("decision")
    data = profile["data"]
    assert data["_delegation_event_count"] == 1
    assert data["_total_outbound_count"] == 3
