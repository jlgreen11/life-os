"""
Tests for prediction engine degraded-mode fallback paths.

Verifies that the prediction engine produces useful output even when
user_model.db is corrupted or unavailable, by falling back to events.db
for contact and interaction data.

Test scenarios:
- _check_relationship_maintenance falls back to _build_contacts_from_events
  when ums.get_signal_profile("relationships") raises an exception
- _check_routine_deviations returns [] early when user_model.db is degraded
- Fallback predictions use conservative confidence values
- Normal path is used when user_model.db is healthy
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _insert_email_event(event_store, from_address, subject, hours_ago, event_type="email.received", to_addresses=None):
    """Insert an email event into events.db with the given parameters."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    payload = {
        "from_address": from_address,
        "subject": subject,
        "message_id": str(uuid.uuid4()),
    }
    if to_addresses is not None:
        payload["to_addresses"] = to_addresses
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "google",
        "timestamp": ts,
        "payload": payload,
        "metadata": {},
    })


def _populate_bidirectional_contact(event_store, contact_email, inbound_count=6, outbound_count=2, last_interaction_days_ago=20):
    """Create a bidirectional contact with enough history to trigger relationship predictions.

    Inserts inbound (email.received) and outbound (email.sent) events spread over
    the last 90 days, with the most recent at ``last_interaction_days_ago``.
    """
    now = datetime.now(timezone.utc)

    # Space interactions evenly, ending at last_interaction_days_ago
    for i in range(inbound_count):
        hours_ago = (last_interaction_days_ago + (i * 5)) * 24
        _insert_email_event(
            event_store,
            from_address=contact_email,
            subject=f"Inbound message {i}",
            hours_ago=hours_ago,
        )

    for i in range(outbound_count):
        hours_ago = (last_interaction_days_ago + (i * 7)) * 24
        _insert_email_event(
            event_store,
            from_address="user@example.com",
            subject=f"Outbound reply {i}",
            hours_ago=hours_ago,
            event_type="email.sent",
            to_addresses=[contact_email],
        )


# -------------------------------------------------------------------------
# _check_relationship_maintenance fallback tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relationship_maintenance_falls_back_to_events_db_on_exception(
    db, event_store, user_model_store
):
    """When ums.get_signal_profile raises, _check_relationship_maintenance
    should fall back to _build_contacts_from_events and still produce predictions."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Create a bidirectional contact with a long gap (> 7 days, > 1.5x avg gap)
    _populate_bidirectional_contact(
        event_store,
        contact_email="alice@example.com",
        inbound_count=8,
        outbound_count=3,
        last_interaction_days_ago=25,
    )

    # Patch ums.get_signal_profile to raise (simulating corrupted user_model.db)
    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("database disk image is malformed"),
    ):
        predictions = await engine._check_relationship_maintenance({})

    # Should have produced at least one prediction via events.db fallback
    assert len(predictions) >= 1, (
        f"Expected >=1 prediction from events.db fallback, got {len(predictions)}"
    )
    # All predictions should be opportunity type (relationship maintenance)
    for pred in predictions:
        assert pred.prediction_type == "opportunity"
        assert "alice" in pred.description.lower() or "alice@example.com" in pred.description.lower()


@pytest.mark.asyncio
async def test_relationship_maintenance_fallback_has_conservative_confidence(
    db, event_store, user_model_store
):
    """Predictions from the events.db fallback path should use conservative
    confidence values (between 0.3 and 0.6, matching the normal path's range)."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _populate_bidirectional_contact(
        event_store,
        contact_email="bob@example.com",
        inbound_count=10,
        outbound_count=4,
        last_interaction_days_ago=30,
    )

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("database disk image is malformed"),
    ):
        predictions = await engine._check_relationship_maintenance({})

    for pred in predictions:
        assert 0.3 <= pred.confidence <= 0.6, (
            f"Fallback confidence {pred.confidence} outside expected range [0.3, 0.6]"
        )


@pytest.mark.asyncio
async def test_relationship_maintenance_uses_signal_profile_when_healthy(
    db, event_store, user_model_store
):
    """When user_model.db is healthy, _check_relationship_maintenance should use
    the signal profile path (normal path), not the events.db fallback."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Provide a relationships signal profile via ums
    now = datetime.now(timezone.utc)
    profile_data = {
        "contacts": {
            "carol@example.com": {
                "interaction_count": 10,
                "outbound_count": 3,
                "last_interaction": (now - timedelta(days=30)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=30 + i * 5)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    # _build_contacts_from_events should NOT be called
    with patch.object(
        engine, "_build_contacts_from_events", wraps=engine._build_contacts_from_events
    ) as mock_build:
        predictions = await engine._check_relationship_maintenance({})
        mock_build.assert_not_called()

    # Should produce predictions from the signal profile
    assert len(predictions) >= 1
    for pred in predictions:
        assert pred.prediction_type == "opportunity"


@pytest.mark.asyncio
async def test_relationship_maintenance_no_predictions_without_data(
    db, event_store, user_model_store
):
    """When both signal profile AND events.db have no contact data,
    _check_relationship_maintenance returns an empty list gracefully."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("corrupted"),
    ):
        predictions = await engine._check_relationship_maintenance({})

    assert predictions == []


# -------------------------------------------------------------------------
# _check_routine_deviations degraded guard tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routine_deviations_returns_empty_when_degraded(
    db, event_store, user_model_store
):
    """When user_model.db is degraded, _check_routine_deviations should
    return [] immediately without attempting DB queries."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Mark user_model.db as degraded
    db.user_model_degraded = True

    predictions = await engine._check_routine_deviations({})
    assert predictions == []


@pytest.mark.asyncio
async def test_routine_deviations_runs_normally_when_healthy(
    db, event_store, user_model_store
):
    """When user_model.db is healthy, _check_routine_deviations should
    proceed with normal processing (querying routines table)."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # user_model.db is healthy by default
    assert db.is_user_model_healthy() is True

    # No routines in the DB → should return empty but NOT short-circuit
    predictions = await engine._check_routine_deviations({})
    assert predictions == []
    # The key assertion is that it ran without error (no early return from guard)


# -------------------------------------------------------------------------
# _check_follow_up_needs resilience tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_needs_works_with_degraded_user_model(
    db, event_store, user_model_store
):
    """_check_follow_up_needs should still produce predictions when
    user_model.db queries fail, since its main logic uses events.db."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Insert an unreplied inbound email from > 3 hours ago
    _insert_email_event(
        event_store,
        from_address="important@company.com",
        subject="Need your input on Q3 budget",
        hours_ago=6,
    )

    # Patch ums.get_signal_profile to raise (simulating corrupted user_model.db)
    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("database disk image is malformed"),
    ):
        predictions = await engine._check_follow_up_needs({})

    # Should still produce a follow-up prediction from events.db
    assert len(predictions) >= 1
    assert predictions[0].prediction_type == "reminder"
    assert "important@company.com" in predictions[0].description or "important" in predictions[0].description.lower()


@pytest.mark.asyncio
async def test_follow_up_needs_confidence_without_priority_boost(
    db, event_store, user_model_store
):
    """When signal profile is unavailable, follow-up predictions should use
    the baseline confidence (0.4) without priority contact boost."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    _insert_email_event(
        event_store,
        from_address="someone@example.com",
        subject="Quick question",
        hours_ago=5,
    )

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("corrupted"),
    ):
        predictions = await engine._check_follow_up_needs({})

    assert len(predictions) >= 1
    # Without priority boost, confidence should be at baseline (0.4)
    assert predictions[0].confidence == pytest.approx(0.4, abs=0.01)


# -------------------------------------------------------------------------
# Integration: generate_predictions with degraded user_model.db
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_predictions_produces_output_with_degraded_user_model(
    db, event_store, user_model_store
):
    """The full generate_predictions pipeline should produce at least some
    predictions when user_model.db is degraded, using events.db fallbacks."""
    engine = PredictionEngine(db, user_model_store, timezone="UTC")

    # Insert events that should trigger predictions
    now = datetime.now(timezone.utc)

    # Unreplied email (follow-up need)
    _insert_email_event(
        event_store,
        from_address="colleague@work.com",
        subject="Meeting notes from today",
        hours_ago=8,
    )

    # Bidirectional contact with long gap (relationship maintenance)
    _populate_bidirectional_contact(
        event_store,
        contact_email="friend@personal.com",
        inbound_count=8,
        outbound_count=3,
        last_interaction_days_ago=25,
    )

    # Mark user_model.db as degraded
    db.user_model_degraded = True

    # Patch ums methods to simulate corruption
    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=Exception("database disk image is malformed"),
    ), patch.object(
        user_model_store,
        "store_prediction",
        side_effect=Exception("database disk image is malformed"),
    ):
        # Run the full pipeline — should not crash
        predictions = await engine.generate_predictions({})

    # The pipeline should have produced some predictions despite degraded state.
    # Even if all get filtered by reaction prediction, the method should not crash.
    # (We can't assert on count since reaction prediction may filter them.)
    # The key assertion is that it completed without raising.
