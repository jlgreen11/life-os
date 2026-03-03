"""
Tests for PredictionEngine relationship maintenance fallback to events.db.

When the user_model.db is corrupt or the 'relationships' signal profile is
unavailable, _check_relationship_maintenance should fall back to computing
contact interaction data directly from events.db email events.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_email_event(
    event_type: str,
    from_address: str,
    to_addresses: list[str],
    timestamp: datetime,
    source: str = "google",
) -> dict:
    """Build an event dict matching the Google connector's email payload schema."""
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": timestamp.isoformat(),
        "priority": "normal",
        "payload": {
            "message_id": f"msg-{uuid.uuid4().hex[:8]}",
            "from_address": from_address,
            "to_addresses": to_addresses,
            "subject": "Test email",
            "body": "Test body",
            "snippet": "Test body",
        },
        "metadata": {},
    }


def _seed_bidirectional_contact(event_store, contact_addr: str, user_addr: str, count: int, days_ago_start: int):
    """Seed a realistic bidirectional email conversation with a contact.

    Creates ``count`` email events alternating inbound/outbound, spaced
    evenly across the range [days_ago_start .. 0] days ago.
    """
    now = datetime.now(timezone.utc)
    interval = days_ago_start / max(count, 1)
    for i in range(count):
        ts = now - timedelta(days=days_ago_start - i * interval)
        if i % 2 == 0:
            # Inbound: contact -> user
            event_store.store_event(
                _make_email_event("email.received", contact_addr, [user_addr], ts)
            )
        else:
            # Outbound: user -> contact
            event_store.store_event(
                _make_email_event("email.sent", user_addr, [contact_addr], ts)
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine_no_signal_profile(db, user_model_store):
    """Create a PredictionEngine whose UserModelStore returns no signal profiles.

    Simulates the condition where user_model.db is corrupt or the
    'relationships' signal profile has never been populated.
    """
    # Override get_signal_profile to always return None
    user_model_store.get_signal_profile = MagicMock(return_value=None)
    return PredictionEngine(db=db, ums=user_model_store)


@pytest.fixture()
def engine_with_signal_profile(db, user_model_store):
    """Create a PredictionEngine with a working signal profile."""
    return PredictionEngine(db=db, ums=user_model_store)


# ---------------------------------------------------------------------------
# Tests: Fallback generates predictions from events.db
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_generates_predictions_from_events(db, event_store, engine_no_signal_profile):
    """When signal profile is None, predictions should be generated from events.db email data."""
    now = datetime.now(timezone.utc)

    # Create a contact with 10 interactions (5 inbound + 5 outbound) spread
    # over 60 days, but the last interaction was 30 days ago — well past the
    # 1.5x average gap threshold.
    contact = "alice@example.com"
    user = "me@example.com"
    for i in range(5):
        # Inbound emails: days 60, 50, 40, 30, 20 ago
        ts = now - timedelta(days=60 - i * 10)
        event_store.store_event(
            _make_email_event("email.received", contact, [user], ts)
        )
        # Outbound emails: days 55, 45, 35, 25, 15 ago
        ts = now - timedelta(days=55 - i * 10)
        event_store.store_event(
            _make_email_event("email.sent", user, [contact], ts)
        )

    predictions = await engine_no_signal_profile._check_relationship_maintenance({})

    # Should have generated at least one prediction for alice
    assert len(predictions) >= 1
    pred = predictions[0]
    assert pred.prediction_type == "opportunity"
    assert contact in pred.relevant_contacts
    assert pred.supporting_signals["contact_email"] == contact


@pytest.mark.asyncio
async def test_fallback_returns_empty_when_no_email_events(db, event_store, engine_no_signal_profile):
    """Fallback returns empty list when events.db has no email events."""
    predictions = await engine_no_signal_profile._check_relationship_maintenance({})
    assert predictions == []


@pytest.mark.asyncio
async def test_fallback_filters_marketing_contacts(db, event_store, engine_no_signal_profile):
    """Marketing/noreply contacts should be filtered even in fallback mode."""
    now = datetime.now(timezone.utc)
    marketing_addr = "noreply@notifications.example.com"
    user = "me@example.com"

    # Seed 10 interactions with marketing address (enough to pass count threshold)
    for i in range(10):
        ts = now - timedelta(days=80 - i * 5)
        event_store.store_event(
            _make_email_event("email.received", marketing_addr, [user], ts)
        )
        # Add some outbound so it doesn't get filtered by outbound_count=0
        ts = now - timedelta(days=78 - i * 5)
        event_store.store_event(
            _make_email_event("email.sent", user, [marketing_addr], ts)
        )

    predictions = await engine_no_signal_profile._check_relationship_maintenance({})

    # Marketing contacts should not produce predictions
    marketing_preds = [p for p in predictions if marketing_addr in p.relevant_contacts]
    assert len(marketing_preds) == 0


@pytest.mark.asyncio
async def test_fallback_skips_contacts_with_few_interactions(db, event_store, engine_no_signal_profile):
    """Contacts with fewer than 5 interactions should be excluded in fallback mode."""
    now = datetime.now(timezone.utc)
    contact = "sparse@example.com"
    user = "me@example.com"

    # Only 3 interactions — below the threshold
    for i in range(2):
        ts = now - timedelta(days=60 - i * 20)
        event_store.store_event(
            _make_email_event("email.received", contact, [user], ts)
        )
    event_store.store_event(
        _make_email_event("email.sent", user, [contact], now - timedelta(days=30))
    )

    predictions = await engine_no_signal_profile._check_relationship_maintenance({})

    sparse_preds = [p for p in predictions if contact in p.relevant_contacts]
    assert len(sparse_preds) == 0


@pytest.mark.asyncio
async def test_fallback_skips_inbound_only_contacts(db, event_store, engine_no_signal_profile):
    """Contacts where user has never sent a reply (outbound_count=0) should be skipped."""
    now = datetime.now(timezone.utc)
    contact = "sender-only@example.com"
    user = "me@example.com"

    # 10 inbound-only emails — user never replied
    for i in range(10):
        ts = now - timedelta(days=80 - i * 5)
        event_store.store_event(
            _make_email_event("email.received", contact, [user], ts)
        )

    predictions = await engine_no_signal_profile._check_relationship_maintenance({})

    inbound_preds = [p for p in predictions if contact in p.relevant_contacts]
    assert len(inbound_preds) == 0


@pytest.mark.asyncio
async def test_signal_profile_path_produces_compatible_predictions(db, event_store, user_model_store):
    """Both signal profile and fallback paths should produce predictions with the same format."""
    now = datetime.now(timezone.utc)

    # Set up a signal profile with one contact that should trigger a prediction
    contact = "bob@example.com"
    timestamps = [(now - timedelta(days=60 - i * 5)).isoformat() for i in range(8)]
    signal_data = {
        "contacts": {
            contact: {
                "interaction_count": 8,
                "last_interaction": (now - timedelta(days=30)).isoformat(),
                "outbound_count": 4,
                "interaction_timestamps": timestamps,
            }
        }
    }
    user_model_store.get_signal_profile = MagicMock(return_value={"data": signal_data})

    engine = PredictionEngine(db=db, ums=user_model_store)
    signal_preds = await engine._check_relationship_maintenance({})

    # Now test fallback path
    user_model_store.get_signal_profile = MagicMock(return_value=None)
    engine_fallback = PredictionEngine(db=db, ums=user_model_store)

    # Seed the same contact data in events.db
    user = "me@example.com"
    for i in range(8):
        ts = now - timedelta(days=60 - i * 5)
        if i % 2 == 0:
            event_store.store_event(
                _make_email_event("email.received", contact, [user], ts)
            )
        else:
            event_store.store_event(
                _make_email_event("email.sent", user, [contact], ts)
            )

    fallback_preds = await engine_fallback._check_relationship_maintenance({})

    # Both paths should produce predictions with the same required fields
    if signal_preds:
        sp = signal_preds[0]
        assert sp.prediction_type == "opportunity"
        assert hasattr(sp, "confidence")
        assert hasattr(sp, "confidence_gate")
        assert hasattr(sp, "relevant_contacts")
        assert hasattr(sp, "supporting_signals")
        assert "contact_email" in sp.supporting_signals
        assert "days_since_last_contact" in sp.supporting_signals

    if fallback_preds:
        fp = fallback_preds[0]
        assert fp.prediction_type == "opportunity"
        assert hasattr(fp, "confidence")
        assert hasattr(fp, "confidence_gate")
        assert hasattr(fp, "relevant_contacts")
        assert hasattr(fp, "supporting_signals")
        assert "contact_email" in fp.supporting_signals
        assert "days_since_last_contact" in fp.supporting_signals


# ---------------------------------------------------------------------------
# Tests: _build_contacts_from_events directly
# ---------------------------------------------------------------------------


def test_build_contacts_from_events_empty_db(db, engine_no_signal_profile):
    """Returns empty dict when events.db has no email events."""
    result = engine_no_signal_profile._build_contacts_from_events()
    assert result == {}


def test_build_contacts_from_events_filters_low_count(db, event_store, engine_no_signal_profile):
    """Contacts with fewer than 5 total interactions should be excluded."""
    now = datetime.now(timezone.utc)
    user = "me@example.com"

    # 3 inbound emails from sparse contact
    for i in range(3):
        ts = now - timedelta(days=30 - i * 5)
        event_store.store_event(
            _make_email_event("email.received", "sparse@example.com", [user], ts)
        )

    result = engine_no_signal_profile._build_contacts_from_events()
    assert "sparse@example.com" not in result


def test_build_contacts_from_events_includes_high_count(db, event_store, engine_no_signal_profile):
    """Contacts with 5+ interactions should be included with correct shape."""
    now = datetime.now(timezone.utc)
    contact = "frequent@example.com"
    user = "me@example.com"

    # 4 inbound + 3 outbound = 7 total interactions
    for i in range(4):
        ts = now - timedelta(days=40 - i * 5)
        event_store.store_event(
            _make_email_event("email.received", contact, [user], ts)
        )
    for i in range(3):
        ts = now - timedelta(days=38 - i * 5)
        event_store.store_event(
            _make_email_event("email.sent", user, [contact], ts)
        )

    result = engine_no_signal_profile._build_contacts_from_events()

    assert contact in result
    data = result[contact]
    assert data["interaction_count"] == 7
    assert data["outbound_count"] == 3
    assert data["last_interaction"] is not None
    assert isinstance(data["interaction_timestamps"], list)
    assert len(data["interaction_timestamps"]) > 0


def test_build_contacts_from_events_correct_outbound_attribution(db, event_store, engine_no_signal_profile):
    """Outbound emails should be attributed to the correct contact (to_addresses), not from_address."""
    now = datetime.now(timezone.utc)
    contact_a = "alice@example.com"
    contact_b = "bob@example.com"
    user = "me@example.com"

    # 5 inbound from alice
    for i in range(5):
        ts = now - timedelta(days=30 - i * 3)
        event_store.store_event(
            _make_email_event("email.received", contact_a, [user], ts)
        )

    # 3 outbound to alice (from user)
    for i in range(3):
        ts = now - timedelta(days=28 - i * 3)
        event_store.store_event(
            _make_email_event("email.sent", user, [contact_a], ts)
        )

    # 5 inbound from bob, no outbound
    for i in range(5):
        ts = now - timedelta(days=30 - i * 3)
        event_store.store_event(
            _make_email_event("email.received", contact_b, [user], ts)
        )

    result = engine_no_signal_profile._build_contacts_from_events()

    # Alice should have outbound_count=3 (from sent emails to her)
    assert contact_a in result
    assert result[contact_a]["outbound_count"] == 3

    # Bob should have outbound_count=0 (user never replied)
    assert contact_b in result
    assert result[contact_b]["outbound_count"] == 0
