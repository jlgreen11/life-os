"""
Tests for semantic fact inferrer inbound-only inference path.

Validates that the inbound-only fallback in _infer_from_inbound_only_contacts
correctly handles users who are primarily email recipients (high inbound,
near-zero outbound).
"""

import logging

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _seed_relationship_profile(user_model_store, contacts: dict, samples_count: int = 100):
    """Seed a relationship signal profile with the given contacts dict."""
    user_model_store.update_signal_profile(
        "relationships",
        {"contacts": contacts},
    )
    # Patch samples_count to desired value (update_signal_profile increments by 1)
    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = 'relationships'",
            (samples_count,),
        )


def _get_semantic_facts(user_model_store):
    """Return all semantic facts as a list of dicts."""
    with user_model_store.db.get_connection("user_model") as conn:
        rows = conn.execute("SELECT * FROM semantic_facts").fetchall()
        return [dict(r) for r in rows]


class TestInboundOnlyThreshold:
    """Test that the lowered threshold (2) works correctly."""

    def test_two_contacts_runs_inference(self, user_model_store):
        """Inbound-only inference should run with just 2 non-marketing contacts."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        assert result.get("reason") is None
        facts = _get_semantic_facts(user_model_store)
        assert len(facts) > 0

    def test_one_contact_below_threshold(self, user_model_store):
        """With only 1 non-marketing contact, inference should skip."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        assert result["reason"] == "too_few_inbound_only"

    def test_zero_contacts_returns_gracefully(self, user_model_store):
        """With 0 contacts, inference should return gracefully."""
        _seed_relationship_profile(user_model_store, {})

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True


class TestCommunicationVolumeFact:
    """Test that the communication_volume_category fact is stored."""

    def test_stores_volume_category(self, user_model_store):
        """Inbound-only inference should store a communication_volume_category fact."""
        contacts = {
            f"person{i}@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            }
            for i in range(3)
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_semantic_facts(user_model_store)
        volume_facts = [f for f in facts if f["key"] == "communication_volume_category"]
        assert len(volume_facts) == 1
        # Values are JSON-encoded in SQLite
        assert "low_volume_email" in volume_facts[0]["value"]


class TestFrequentSenderFacts:
    """Test that frequent_sender facts are created for contacts with high inbound_count."""

    def test_frequent_sender_created_for_high_inbound(self, user_model_store):
        """Contacts with inbound_count >= 3 should get frequent_personal_sender facts."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 20,
                "outbound_count": 0,
                "interaction_count": 20,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "carol@example.com": {
                "inbound_count": 1,
                "outbound_count": 0,
                "interaction_count": 1,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_semantic_facts(user_model_store)
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]

        # alice (20) and bob (10) should qualify; carol (1) should not
        assert len(sender_facts) == 2
        keys = {f["key"] for f in sender_facts}
        assert "frequent_sender_alice@example.com" in keys
        assert "frequent_sender_bob@example.com" in keys

        for f in sender_facts:
            assert "frequent_personal_sender" in f["value"]

    def test_frequent_sender_not_created_below_threshold(self, user_model_store):
        """Contacts with inbound_count < 3 should not get frequent_sender facts."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 2,
                "outbound_count": 0,
                "interaction_count": 2,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 1,
                "outbound_count": 0,
                "interaction_count": 1,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_semantic_facts(user_model_store)
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]
        assert len(sender_facts) == 0


class TestMarketingFilter:
    """Test that marketing contacts are properly filtered out."""

    def test_marketing_contacts_excluded(self, user_model_store):
        """Marketing senders should be filtered out of inbound-only inference."""
        contacts = {
            # Marketing senders — should be filtered
            "newsletter@company.com": {
                "inbound_count": 100,
                "outbound_count": 0,
                "interaction_count": 100,
                "channels_used": ["email"],
            },
            "noreply@service.com": {
                "inbound_count": 50,
                "outbound_count": 0,
                "interaction_count": 50,
                "channels_used": ["email"],
            },
            # Human senders — should pass through
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        # Should succeed (2 human contacts >= threshold of 2)
        assert result["processed"] is True
        assert result.get("reason") is None

        facts = _get_semantic_facts(user_model_store)
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]

        # Only human senders should have facts
        sender_keys = {f["key"] for f in sender_facts}
        assert "frequent_sender_newsletter@company.com" not in sender_keys
        assert "frequent_sender_noreply@service.com" not in sender_keys
        assert "frequent_sender_alice@example.com" in sender_keys
        assert "frequent_sender_bob@example.com" in sender_keys

    def test_all_marketing_below_threshold(self, user_model_store):
        """If all contacts are marketing, should return too_few_inbound_only."""
        contacts = {
            "newsletter@company.com": {
                "inbound_count": 100,
                "outbound_count": 0,
                "interaction_count": 100,
                "channels_used": ["email"],
            },
            "noreply@service.com": {
                "inbound_count": 50,
                "outbound_count": 0,
                "interaction_count": 50,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["reason"] == "too_few_inbound_only"


class TestFilterFunnelLogging:
    """Test that filter funnel logging includes expected counts."""

    def test_funnel_logging_in_main_path(self, user_model_store, caplog):
        """Main relationship inference should log the filter funnel."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO):
            inferrer.infer_from_relationship_profile()

        funnel_msgs = [r for r in caplog.records if "Relationship inference funnel" in r.message]
        assert len(funnel_msgs) == 1
        assert "total_contacts=2" in funnel_msgs[0].message
        assert "bidirectional=0" in funnel_msgs[0].message
        assert "inbound_only_total=2" in funnel_msgs[0].message

    def test_funnel_logging_in_inbound_only(self, user_model_store, caplog):
        """Inbound-only inference should log filter funnel counts."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO):
            inferrer.infer_from_relationship_profile()

        inbound_msgs = [r for r in caplog.records if "Inbound-only inference:" in r.message]
        assert len(inbound_msgs) == 1
        assert "total_contacts=2" in inbound_msgs[0].message
        assert "after_marketing_filter=2" in inbound_msgs[0].message
        assert "threshold=2" in inbound_msgs[0].message


class TestFunnelInReturnDict:
    """Test that the return dict includes funnel diagnostic data."""

    def test_return_dict_includes_funnel(self, user_model_store):
        """The inbound-only inference result should include funnel counts."""
        contacts = {
            "alice@example.com": {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            },
            "bob@example.com": {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert "funnel" in result
        assert result["funnel"]["total_contacts"] == 2
        assert result["funnel"]["inbound_only_after_filter"] == 2
        assert result["facts_written"] >= 1
