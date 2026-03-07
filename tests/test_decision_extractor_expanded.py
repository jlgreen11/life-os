"""
Tests for expanded DecisionExtractor — TRANSACTION_NEW and CALENDAR_EVENT_UPDATED support.

Verifies that the extractor correctly:
1. Processes TRANSACTION_NEW events as purchase decision signals
2. Processes CALENDAR_EVENT_UPDATED events as decision revision signals
3. Updates risk_tolerance_by_domain from transaction amounts
4. Updates mind_change_frequency from calendar revisions
5. Handles missing/malformed fields gracefully
6. Still processes original 5 event types (regression check)
"""

import pytest
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


def test_can_process_transaction_new(db, user_model_store):
    """can_process returns True for TRANSACTION_NEW events."""
    extractor = DecisionExtractor(db, user_model_store)
    assert extractor.can_process({"type": EventType.TRANSACTION_NEW.value})


def test_can_process_calendar_event_updated(db, user_model_store):
    """can_process returns True for CALENDAR_EVENT_UPDATED events."""
    extractor = DecisionExtractor(db, user_model_store)
    assert extractor.can_process({"type": EventType.CALENDAR_EVENT_UPDATED.value})


def test_can_process_original_event_types_regression(db, user_model_store):
    """Regression: can_process still returns True for all original 5 event types."""
    extractor = DecisionExtractor(db, user_model_store)
    original_types = [
        EventType.TASK_COMPLETED.value,
        EventType.TASK_CREATED.value,
        EventType.EMAIL_SENT.value,
        EventType.MESSAGE_SENT.value,
        EventType.CALENDAR_EVENT_CREATED.value,
    ]
    for event_type in original_types:
        assert extractor.can_process({"type": event_type}), f"Failed for {event_type}"


def test_transaction_new_produces_purchase_decision_signal(db, user_model_store):
    """TRANSACTION_NEW with amount and merchant produces a purchase_decision signal."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.TRANSACTION_NEW.value,
        "timestamp": now.isoformat(),
        "payload": {
            "amount": 45.99,
            "merchant_name": "Pizza Palace",
            "description": "Dinner purchase",
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "purchase_decision"
    assert signals[0]["amount"] == 45.99
    assert signals[0]["merchant"] == "Pizza Palace"
    assert signals[0]["domain"] == "dining"


def test_transaction_new_uses_description_fallback(db, user_model_store):
    """TRANSACTION_NEW uses description when merchant_name is missing."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.TRANSACTION_NEW.value,
        "timestamp": now.isoformat(),
        "payload": {
            "amount": 12.99,
            "description": "Netflix monthly subscription",
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["merchant"] == "Netflix monthly subscription"
    assert signals[0]["domain"] == "subscriptions"


def test_transaction_new_without_amount_returns_empty(db, user_model_store):
    """TRANSACTION_NEW without amount field gracefully returns empty signals."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.TRANSACTION_NEW.value,
        "timestamp": now.isoformat(),
        "payload": {
            "merchant_name": "Some Store",
        },
    }

    signals = extractor.extract(event)
    assert signals == []


def test_transaction_new_with_invalid_amount_returns_empty(db, user_model_store):
    """TRANSACTION_NEW with non-numeric amount gracefully returns empty signals."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.TRANSACTION_NEW.value,
        "timestamp": now.isoformat(),
        "payload": {
            "amount": "not-a-number",
            "merchant_name": "Some Store",
        },
    }

    signals = extractor.extract(event)
    assert signals == []


def test_transaction_new_updates_risk_tolerance(db, user_model_store):
    """TRANSACTION_NEW updates risk_tolerance_by_domain in the profile."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.TRANSACTION_NEW.value,
        "timestamp": now.isoformat(),
        "payload": {
            "amount": 150.0,
            "merchant_name": "Amazon purchase",
        },
    }

    extractor.extract(event)

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    risk = profile["data"]["risk_tolerance_by_domain"]
    assert "shopping" in risk
    # $150 / $200 = 0.75 risk score
    assert risk["shopping"] == pytest.approx(0.75, abs=0.01)


def test_calendar_event_updated_produces_decision_revision(db, user_model_store):
    """CALENDAR_EVENT_UPDATED produces a decision_revision signal."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.CALENDAR_EVENT_UPDATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "summary": "Team standup",
            "changes": {"start_time": "2026-03-08T10:00:00+00:00"},
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "decision_revision"
    assert signals[0]["revision_type"] == "time_change"
    assert signals[0]["event_summary"] == "Team standup"


def test_calendar_event_updated_location_change(db, user_model_store):
    """CALENDAR_EVENT_UPDATED with location change produces location_change revision."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.CALENDAR_EVENT_UPDATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "summary": "Lunch meeting",
            "changes": {"location": "New Restaurant"},
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["revision_type"] == "location_change"


def test_calendar_event_updated_no_changes_field(db, user_model_store):
    """CALENDAR_EVENT_UPDATED without changes field still produces a general revision."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event = {
        "type": EventType.CALENDAR_EVENT_UPDATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "summary": "Some event",
        },
    }

    signals = extractor.extract(event)

    assert len(signals) == 1
    assert signals[0]["type"] == "decision_revision"
    assert signals[0]["revision_type"] == "general"


def test_calendar_event_updated_increments_mind_change_frequency(db, user_model_store):
    """CALENDAR_EVENT_UPDATED increments mind_change_frequency in the profile."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    # First revision
    event1 = {
        "type": EventType.CALENDAR_EVENT_UPDATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "summary": "Meeting",
            "changes": {"start_time": "2026-03-08T10:00:00+00:00"},
        },
    }
    extractor.extract(event1)

    profile = user_model_store.get_signal_profile("decision")
    freq_after_1 = profile["data"]["mind_change_frequency"]
    # Default is 0.1, after one revision: 0.7 * 0.1 + 0.3 * 1.0 = 0.37
    assert freq_after_1 == pytest.approx(0.37, abs=0.01)

    # Second revision should push frequency higher
    event2 = {
        "type": EventType.CALENDAR_EVENT_UPDATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "summary": "Meeting again",
            "changes": {"location": "Room B"},
        },
    }
    extractor.extract(event2)

    profile = user_model_store.get_signal_profile("decision")
    freq_after_2 = profile["data"]["mind_change_frequency"]
    # 0.7 * 0.37 + 0.3 * 1.0 = 0.559
    assert freq_after_2 > freq_after_1


def test_classify_merchant_domain(db, user_model_store):
    """Merchant domain classification maps merchants to correct categories."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_merchant_domain("Pizza Palace Restaurant") == "dining"
    assert extractor._classify_merchant_domain("Amazon.com") == "shopping"
    assert extractor._classify_merchant_domain("Netflix") == "subscriptions"
    assert extractor._classify_merchant_domain("Uber ride") == "transport"
    assert extractor._classify_merchant_domain("Rent payment") == "housing"
    assert extractor._classify_merchant_domain("Random vendor") == "general"
