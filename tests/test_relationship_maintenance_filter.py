"""
Tests for relationship maintenance marketing filter.

Verifies that relationship maintenance predictions correctly filter out
marketing/automated senders and only suggest reaching out to real human
relationships.
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from services.prediction_engine.engine import PredictionEngine
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


@pytest.fixture
def setup_relationship_test_data(db, user_model_store):
    """Create test data for relationship maintenance predictions."""

    # Create a relationships signal profile with both real contacts
    # and marketing senders
    now = datetime.now(timezone.utc)

    # Real human contact — should generate maintenance prediction if overdue
    alice_timestamps = [
        (now - timedelta(days=60)).isoformat(),
        (now - timedelta(days=50)).isoformat(),
        (now - timedelta(days=40)).isoformat(),
        (now - timedelta(days=30)).isoformat(),
        (now - timedelta(days=20)).isoformat(),
    ]

    # Marketing sender — frequent interactions, should be filtered out
    marketing_timestamps = []
    for i in range(100):
        marketing_timestamps.append((now - timedelta(hours=i * 6)).isoformat())

    # Another real contact who was contacted recently — should NOT generate prediction
    bob_timestamps = [
        (now - timedelta(days=30)).isoformat(),
        (now - timedelta(days=20)).isoformat(),
        (now - timedelta(days=10)).isoformat(),
        (now - timedelta(days=5)).isoformat(),
        (now - timedelta(days=1)).isoformat(),
    ]

    # Contact with exactly 5 interactions (edge case for minimum threshold)
    # Average gap: 15 days, last contact: 25 days ago
    # 25 > 1.5 * 15 (22.5), so should trigger
    charlie_timestamps = [
        (now - timedelta(days=85)).isoformat(),
        (now - timedelta(days=70)).isoformat(),
        (now - timedelta(days=55)).isoformat(),
        (now - timedelta(days=40)).isoformat(),
        (now - timedelta(days=25)).isoformat(),
    ]

    relationships_profile = {
        "contacts": {
            # Real human — last contact 20 days ago, typical gap ~10 days
            # 20 days > 1.5 * 10 days = 15 days, so should trigger
            "alice@example.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 250,
                "last_interaction": alice_timestamps[-1],
                "interaction_timestamps": alice_timestamps,
                "last_inbound_timestamp": alice_timestamps[-1],
            },

            # Marketing sender — should be filtered out by _is_marketing_or_noreply
            "hello@email.rei.com": {
                "interaction_count": 100,
                "inbound_count": 100,
                "outbound_count": 0,
                "channels_used": ["google"],
                "avg_message_length": 88000,
                "last_interaction": marketing_timestamps[0],
                "interaction_timestamps": marketing_timestamps[-10:],
                "last_inbound_timestamp": marketing_timestamps[0],
            },

            # Another marketing pattern — no-reply sender
            "noreply@service.com": {
                "interaction_count": 50,
                "inbound_count": 50,
                "outbound_count": 0,
                "channels_used": ["google"],
                "avg_message_length": 1500,
                "last_interaction": (now - timedelta(days=30)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=i)).isoformat() for i in range(30, 35)
                ],
                "last_inbound_timestamp": (now - timedelta(days=30)).isoformat(),
            },

            # Real human, contacted recently — should NOT trigger
            "bob@company.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 300,
                "last_interaction": bob_timestamps[-1],
                "interaction_timestamps": bob_timestamps,
                "last_inbound_timestamp": bob_timestamps[-1],
            },

            # Real human, exactly at minimum threshold, overdue
            "charlie@startup.io": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 200,
                "last_interaction": charlie_timestamps[-1],
                "interaction_timestamps": charlie_timestamps,
                "last_inbound_timestamp": charlie_timestamps[-1],
            },

            # Contact with too few interactions — should be skipped
            "david@email.com": {
                "interaction_count": 3,
                "inbound_count": 2,
                "outbound_count": 1,
                "channels_used": ["google"],
                "avg_message_length": 250,
                "last_interaction": (now - timedelta(days=30)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=30)).isoformat(),
                    (now - timedelta(days=20)).isoformat(),
                    (now - timedelta(days=10)).isoformat(),
                ],
                "last_inbound_timestamp": (now - timedelta(days=30)).isoformat(),
            },
        }
    }

    # Store the signal profile
    user_model_store.update_signal_profile(
        profile_type="relationships",
        data=relationships_profile,
    )

    return {
        "alice": "alice@example.com",
        "marketing_rei": "hello@email.rei.com",
        "marketing_noreply": "noreply@service.com",
        "bob": "bob@company.com",
        "charlie": "charlie@startup.io",
        "david": "david@email.com",
    }


@pytest.mark.asyncio
async def test_relationship_maintenance_filters_marketing(db, user_model_store, setup_relationship_test_data):
    """Relationship maintenance predictions should exclude marketing/automated senders."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_relationship_maintenance({})

    # Should generate predictions for Alice and Charlie (real contacts, overdue)
    # Should NOT generate predictions for:
    # - Marketing senders (email.rei.com domain, noreply@)
    # - Bob (contacted recently)
    # - David (too few interactions)

    pred_contacts = [p.relevant_contacts[0] for p in predictions if p.relevant_contacts]

    # Verify Alice is included (real contact, overdue)
    assert setup_relationship_test_data["alice"] in pred_contacts, \
        "Alice should generate a relationship maintenance prediction (real contact, overdue)"

    # Verify Charlie is included (real contact, overdue, at minimum threshold)
    assert setup_relationship_test_data["charlie"] in pred_contacts, \
        "Charlie should generate a prediction (real contact, overdue, exactly 5 interactions)"

    # Verify marketing senders are excluded
    assert setup_relationship_test_data["marketing_rei"] not in pred_contacts, \
        "Marketing sender (email.rei.com domain) should be filtered out"

    assert setup_relationship_test_data["marketing_noreply"] not in pred_contacts, \
        "No-reply sender should be filtered out"

    # Verify Bob is excluded (contacted recently)
    assert setup_relationship_test_data["bob"] not in pred_contacts, \
        "Bob should not generate a prediction (contacted recently)"

    # Verify David is excluded (too few interactions)
    assert setup_relationship_test_data["david"] not in pred_contacts, \
        "David should not generate a prediction (too few interactions)"

    # Verify prediction types are correct
    for pred in predictions:
        assert pred.prediction_type == "opportunity", \
            "Relationship maintenance predictions should be type 'opportunity'"
        assert pred.suggested_action.startswith("Reach out to"), \
            "Suggested action should recommend reaching out"


@pytest.mark.asyncio
async def test_relationship_maintenance_confidence_scaling(db, user_model_store, setup_relationship_test_data):
    """Confidence should scale based on how overdue the contact is."""
    engine = PredictionEngine(db, user_model_store)

    predictions = await engine._check_relationship_maintenance({})

    # Find Alice's prediction (she's more overdue than Charlie)
    alice_pred = next((p for p in predictions
                      if p.relevant_contacts and p.relevant_contacts[0] == setup_relationship_test_data["alice"]),
                     None)

    assert alice_pred is not None, "Alice should have a prediction"

    # Confidence should be above minimum threshold (0.3) since she's overdue
    assert alice_pred.confidence >= 0.3, \
        "Confidence should meet SUGGEST threshold for overdue contacts"

    # Confidence should be capped at 0.6 per the implementation
    assert alice_pred.confidence <= 0.6, \
        "Confidence should be capped at 0.6 for relationship maintenance"


@pytest.mark.asyncio
async def test_relationship_maintenance_empty_profile(db, user_model_store):
    """Should handle missing relationships profile gracefully."""
    engine = PredictionEngine(db, user_model_store)

    # No relationships profile exists yet
    predictions = await engine._check_relationship_maintenance({})

    # Should return empty list, not crash
    assert predictions == [], \
        "Should return empty list when no relationships profile exists"


@pytest.mark.asyncio
async def test_relationship_maintenance_no_overdue_contacts(db, user_model_store):
    """Should return no predictions when all contacts are current."""
    now = datetime.now(timezone.utc)

    # All contacts contacted very recently
    recent_timestamps = [
        (now - timedelta(days=i)).isoformat() for i in range(5)
    ]

    relationships_profile = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 250,
                "last_interaction": recent_timestamps[0],
                "interaction_timestamps": recent_timestamps,
                "last_inbound_timestamp": recent_timestamps[0],
            },
            "bob@company.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 300,
                "last_interaction": recent_timestamps[0],
                "interaction_timestamps": recent_timestamps,
                "last_inbound_timestamp": recent_timestamps[0],
            },
        }
    }

    user_model_store.update_signal_profile(
        profile_type="relationships",
        data=relationships_profile,
    )

    engine = PredictionEngine(db, user_model_store)
    predictions = await engine._check_relationship_maintenance({})

    assert len(predictions) == 0, \
        "Should not generate predictions when all contacts are current"


@pytest.mark.asyncio
async def test_relationship_maintenance_malformed_timestamps(db, user_model_store):
    """Should handle malformed timestamps gracefully without crashing."""
    now = datetime.now(timezone.utc)

    relationships_profile = {
        "contacts": {
            "alice@example.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,
                "channels_used": ["google"],
                "avg_message_length": 250,
                "last_interaction": "invalid-timestamp",  # Malformed
                "interaction_timestamps": [
                    "also-invalid",
                    (now - timedelta(days=10)).isoformat(),
                ],
                "last_inbound_timestamp": "invalid",
            },
        }
    }

    user_model_store.update_signal_profile(
        profile_type="relationships",
        data=relationships_profile,
    )

    engine = PredictionEngine(db, user_model_store)

    # Should not crash, should skip this contact due to parsing errors
    predictions = await engine._check_relationship_maintenance({})

    assert len(predictions) == 0, \
        "Should skip contacts with malformed timestamps"
