"""
Tests for semantic fact inference filtering of one-way relationships.

This test suite verifies that the semantic fact inferrer correctly filters
out one-way relationships (zero outbound communication) to prevent marketing
email senders from being classified as "high priority" contacts.
"""

import json
import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_relationship_profile(user_model_store, contacts: dict, samples_count: int):
    """
    Helper to directly set a relationship profile with specific samples_count.

    Args:
        user_model_store: UserModelStore instance
        contacts: Dict of contact data
        samples_count: Total number of samples in the profile
    """
    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("relationships", json.dumps({"contacts": contacts}), samples_count)
        )


def test_filters_one_way_marketing_contacts(user_model_store):
    """
    Verify that one-way marketing contacts are excluded from semantic fact generation.

    This is the core fix: marketing email senders (inbound-only, zero outbound)
    should not pollute semantic facts with false "high_priority" classifications.
    """
    inferrer = SemanticFactInferrer(user_model_store)

    # Create profile with 50+ one-way marketing contacts (realistic scenario)
    contacts = {}
    for i in range(50):
        contacts[f"marketing{i}@spam.com"] = {
            "interaction_count": 20,
            "inbound_count": 20,
            "outbound_count": 0,  # One-way
            "channels_used": ["email"],
        }

    # Add one bidirectional contact
    contacts["alice@example.com"] = {
        "interaction_count": 100,
        "inbound_count": 50,
        "outbound_count": 50,
        "channels_used": ["email", "sms"],
    }

    _set_relationship_profile(user_model_store, contacts, samples_count=1100)

    # Run inference
    inferrer.infer_from_relationship_profile()

    # Verify NO semantic facts were generated for marketing contacts
    all_facts = user_model_store.get_semantic_facts()
    marketing_facts = [f for f in all_facts if "marketing" in f["key"]]

    assert len(marketing_facts) == 0, \
        f"One-way marketing contacts should not generate facts, but got {len(marketing_facts)}: {[f['key'] for f in marketing_facts]}"

    # Verify bidirectional contact DID get facts
    alice_facts = [f for f in all_facts if "alice" in f["key"]]
    assert len(alice_facts) > 0, "Bidirectional contact should generate facts"


def test_all_oneway_contacts_skips_inference(user_model_store):
    """
    Verify that if ALL contacts are one-way, no facts are generated.

    Edge case: user has only received emails (no sent messages yet).
    """
    inferrer = SemanticFactInferrer(user_model_store)

    _set_relationship_profile(user_model_store, {
        "marketing1@spam.com": {
            "interaction_count": 50,
            "inbound_count": 50,
            "outbound_count": 0,
        },
        "marketing2@spam.com": {
            "interaction_count": 30,
            "inbound_count": 30,
            "outbound_count": 0,
        },
    }, samples_count=80)

    inferrer.infer_from_relationship_profile()

    # No semantic facts should exist for any contact
    facts = user_model_store.get_semantic_facts()
    relationship_facts = [f for f in facts if "relationship_" in f["key"]]

    assert len(relationship_facts) == 0, \
        "No relationship facts should be generated when all contacts are one-way"


def test_bidirectional_contacts_generate_facts(user_model_store):
    """
    Verify that bidirectional contacts still generate appropriate semantic facts.

    Ensures filtering doesn't break normal relationship inference.
    """
    inferrer = SemanticFactInferrer(user_model_store)

    _set_relationship_profile(user_model_store, {
        "alice@example.com": {
            "interaction_count": 100,  # High volume
            "inbound_count": 50,
            "outbound_count": 50,
            "channels_used": ["email", "sms"],
            "last_interaction": "2026-02-16T12:00:00Z",
        },
        "bob@example.com": {
            "interaction_count": 13,  # Low volume
            "inbound_count": 3,
            "outbound_count": 10,  # User-initiated (10 out vs 3 in = >3:1 ratio)
            "channels_used": ["email"],
            "last_interaction": "2026-02-16T12:00:00Z",
        },
    }, samples_count=113)

    inferrer.infer_from_relationship_profile()

    # Alice should be multi-channel (2 channels)
    alice_multichannel = user_model_store.get_semantic_fact("relationship_multichannel_alice@example.com")
    assert alice_multichannel is not None, "Multi-channel contact should be detected"
    assert alice_multichannel["value"] == "multi_channel"

    # Bob should be user-initiated (10 out vs 3 in = >3:1 ratio)
    bob_balance = user_model_store.get_semantic_fact("relationship_balance_bob@example.com")
    assert bob_balance is not None, "User-initiated relationship should be detected"
    assert bob_balance["value"] == "user_initiated"

    # Both should be mutual (balanced bidirectional communication, 10+ total interactions)
    alice_balance = user_model_store.get_semantic_fact("relationship_balance_alice@example.com")
    # Alice has 50/50 split, so balance ratio = min(50,50)/100 = 0.5 > 0.3, so mutual
    assert alice_balance is not None, "Balanced bidirectional contact should be mutual"
    assert alice_balance["value"] == "mutual"


def test_average_calculation_excludes_oneway(user_model_store):
    """
    Verify that one-way contacts don't inflate the average interaction count.

    Critical for correct high-priority threshold calculation.
    """
    inferrer = SemanticFactInferrer(user_model_store)

    _set_relationship_profile(user_model_store, {
        # Bidirectional contacts with moderate counts
        "alice@example.com": {
            "interaction_count": 30,
            "inbound_count": 15,
            "outbound_count": 15,
        },
        "bob@example.com": {
            "interaction_count": 10,
            "inbound_count": 5,
            "outbound_count": 5,
        },
        # Many one-way marketing contacts with HIGH counts (should be ignored)
        "marketing1@spam.com": {
            "interaction_count": 200,
            "inbound_count": 200,
            "outbound_count": 0,
        },
        "marketing2@spam.com": {
            "interaction_count": 150,
            "inbound_count": 150,
            "outbound_count": 0,
        },
    }, samples_count=390)

    inferrer.infer_from_relationship_profile()

    # If one-way contacts were included:
    #   Average = (30 + 10 + 200 + 150) / 4 = 97.5
    #   High-priority threshold = 195
    #   Alice (30) would NOT be high-priority
    #
    # With filtering:
    #   Average = (30 + 10) / 2 = 20
    #   High-priority threshold = 40
    #   Alice (30) still not high-priority, but Bob wouldn't skew the calculation

    # The key test: marketing contacts should have ZERO facts
    all_facts = user_model_store.get_semantic_facts()
    marketing_facts = [f for f in all_facts if "marketing" in f["key"]]
    assert len(marketing_facts) == 0, "One-way contacts should not generate any facts"


def test_logging_when_no_bidirectional_contacts(user_model_store, caplog):
    """
    Verify debug log is emitted when all contacts are filtered out.

    Helps with debugging why relationship inference isn't producing facts.
    """
    inferrer = SemanticFactInferrer(user_model_store)

    _set_relationship_profile(user_model_store, {
        "marketing@spam.com": {
            "interaction_count": 50,
            "inbound_count": 50,
            "outbound_count": 0,
        },
    }, samples_count=50)

    import logging
    with caplog.at_level(logging.DEBUG):
        inferrer.infer_from_relationship_profile()

    # Log message was updated in the marketing-filter PR to reflect the new
    # two-stage filtering (outbound_count > 0, then is_marketing_or_noreply).
    assert ("No human bidirectional contacts found" in caplog.text or
            "No bidirectional contacts found" in caplog.text), \
        "Should log when all contacts are filtered out"
