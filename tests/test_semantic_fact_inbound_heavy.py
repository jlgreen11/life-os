"""
Tests for semantic fact inferrer in inbound-heavy environments.

Validates the supplementary fallback: when a few bidirectional contacts exist
but don't meet the main path's interaction thresholds, the inbound-only
fallback should also run to produce basic relationship facts.
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


class TestSupplementaryFallbackTriggered:
    """When bidirectional contacts exist but produce 0 facts, the fallback should run."""

    def test_fallback_runs_when_bidirectional_contacts_below_threshold(self, user_model_store):
        """
        100 inbound-only contacts + 2 contacts with outbound_count=1.

        The 2 bidirectional contacts have low interaction counts (1-2) so
        they don't meet the main path's interaction_count >= 3 threshold.
        The fallback should run and produce facts from the 100 inbound-only contacts.
        """
        contacts = {}
        # 100 inbound-only contacts with varying inbound counts
        for i in range(100):
            contacts[f"person{i}@example.com"] = {
                "inbound_count": 5 + (i % 20),
                "outbound_count": 0,
                "interaction_count": 5 + (i % 20),
                "channels_used": ["email"],
            }
        # 2 bidirectional contacts with very low interaction counts
        contacts["replied-once@example.com"] = {
            "inbound_count": 1,
            "outbound_count": 1,
            "interaction_count": 2,
            "channels_used": ["email"],
        }
        contacts["replied-twice@example.com"] = {
            "inbound_count": 0,
            "outbound_count": 1,
            "interaction_count": 1,
            "channels_used": ["email"],
        }

        _seed_relationship_profile(user_model_store, contacts, samples_count=3788)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        assert result.get("facts_written", 0) > 0, (
            "Should have produced facts via supplementary inbound-only fallback"
        )

        facts = _get_semantic_facts(user_model_store)
        # Should have at least a communication_volume_category fact
        volume_facts = [f for f in facts if f["key"] == "communication_volume_category"]
        assert len(volume_facts) == 1, "Expected communication_volume_category fact from fallback"
        assert "high_volume_email" in volume_facts[0]["value"]

        # Should also have frequent_sender facts
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]
        assert len(sender_facts) > 0, "Expected frequent_sender facts from fallback"

    def test_fallback_does_not_run_when_main_path_produces_facts(self, user_model_store):
        """
        When bidirectional contacts DO produce facts, the fallback should NOT run.

        We use a multi-channel contact (2+ channels) with enough interactions
        to trigger the multichannel and balance facts from the main path.
        """
        contacts = {}
        # 10 inbound-only contacts
        for i in range(10):
            contacts[f"person{i}@example.com"] = {
                "inbound_count": 5,
                "outbound_count": 0,
                "interaction_count": 5,
                "channels_used": ["email"],
            }
        # 1 bidirectional contact that produces multichannel + balance facts
        contacts["bestfriend@example.com"] = {
            "inbound_count": 50,
            "outbound_count": 50,
            "interaction_count": 100,
            "channels_used": ["email", "signal"],
        }

        _seed_relationship_profile(user_model_store, contacts, samples_count=500)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        assert result.get("facts_written", 0) > 0

        facts = _get_semantic_facts(user_model_store)
        # Should have multichannel and/or balance facts from main path
        main_facts = [f for f in facts if f["key"].startswith("relationship_")]
        assert len(main_facts) >= 1

        # Should NOT have communication_volume_category (fallback did not run)
        volume_facts = [f for f in facts if f["key"] == "communication_volume_category"]
        assert len(volume_facts) == 0, "Fallback should not run when main path produces facts"


class TestAllInboundOnlyProducesFacts:
    """When all contacts have outbound_count=0, fallback should produce facts."""

    def test_all_inbound_only_produces_volume_fact(self, user_model_store):
        """With all outbound_count=0, the direct fallback should produce facts."""
        contacts = {}
        for i in range(20):
            contacts[f"sender{i}@example.com"] = {
                "inbound_count": 10 + i,
                "outbound_count": 0,
                "interaction_count": 10 + i,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=200)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True

        facts = _get_semantic_facts(user_model_store)
        volume_facts = [f for f in facts if f["key"] == "communication_volume_category"]
        assert len(volume_facts) == 1
        assert "moderate_volume_email" in volume_facts[0]["value"]

        # Should also have frequent_sender facts for top senders
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]
        assert len(sender_facts) > 0


class TestMarketingContactsFilteredInFallback:
    """Marketing contacts should be excluded from inbound-only inference."""

    def test_marketing_contacts_excluded_from_supplementary_fallback(self, user_model_store):
        """
        When fallback runs via supplementary path, marketing contacts
        should still be filtered out.
        """
        contacts = {}
        # 1 bidirectional contact that won't meet thresholds
        contacts["replied-once@example.com"] = {
            "inbound_count": 1,
            "outbound_count": 1,
            "interaction_count": 2,
            "channels_used": ["email"],
        }
        # Marketing contacts (should be filtered)
        contacts["newsletter@marketing.com"] = {
            "inbound_count": 200,
            "outbound_count": 0,
            "interaction_count": 200,
            "channels_used": ["email"],
        }
        contacts["noreply@automated-service.com"] = {
            "inbound_count": 150,
            "outbound_count": 0,
            "interaction_count": 150,
            "channels_used": ["email"],
        }
        # 3 real human contacts (inbound-only)
        contacts["alice@example.com"] = {
            "inbound_count": 15,
            "outbound_count": 0,
            "interaction_count": 15,
            "channels_used": ["email"],
        }
        contacts["bob@example.com"] = {
            "inbound_count": 10,
            "outbound_count": 0,
            "interaction_count": 10,
            "channels_used": ["email"],
        }
        contacts["carol@example.com"] = {
            "inbound_count": 5,
            "outbound_count": 0,
            "interaction_count": 5,
            "channels_used": ["email"],
        }

        _seed_relationship_profile(user_model_store, contacts, samples_count=500)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True

        facts = _get_semantic_facts(user_model_store)
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]
        sender_keys = {f["key"] for f in sender_facts}

        # Marketing contacts should NOT have facts
        assert "frequent_sender_newsletter@marketing.com" not in sender_keys
        assert "frequent_sender_noreply@automated-service.com" not in sender_keys

        # Human contacts with inbound_count >= 3 should have facts
        assert "frequent_sender_alice@example.com" in sender_keys
        assert "frequent_sender_bob@example.com" in sender_keys
        assert "frequent_sender_carol@example.com" in sender_keys


class TestFactsWrittenInReturnDict:
    """Verify that facts_written is included in the return dict."""

    def test_return_dict_includes_facts_written(self, user_model_store):
        """The relationship inference result should include facts_written count."""
        contacts = {
            "bestfriend@example.com": {
                "inbound_count": 50,
                "outbound_count": 50,
                "interaction_count": 100,
                "channels_used": ["email", "signal"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts, samples_count=100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert "facts_written" in result, "Return dict should include facts_written"
        assert isinstance(result["facts_written"], int)

    def test_facts_written_zero_when_no_contacts_qualify(self, user_model_store):
        """
        When bidirectional contacts exist but don't meet thresholds,
        and there are no inbound-only contacts either, facts_written should be 0.
        """
        contacts = {
            "replied-once@example.com": {
                "inbound_count": 1,
                "outbound_count": 1,
                "interaction_count": 2,
                "channels_used": ["email"],
            },
        }
        _seed_relationship_profile(user_model_store, contacts, samples_count=50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        # Only 1 inbound-only contact (0 actually, since this one is bidirectional),
        # so fallback can't run either. facts_written should be 0.
        assert result.get("facts_written", 0) == 0


class TestSupplementaryFallbackLogging:
    """Verify that supplementary fallback logs diagnostic messages."""

    def test_logs_supplementary_fallback_message(self, user_model_store, caplog):
        """When supplementary fallback runs, it should log an informational message."""
        contacts = {}
        # 5 inbound-only contacts
        for i in range(5):
            contacts[f"person{i}@example.com"] = {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            }
        # 1 bidirectional contact that won't meet thresholds
        contacts["low-activity@example.com"] = {
            "inbound_count": 1,
            "outbound_count": 1,
            "interaction_count": 2,
            "channels_used": ["email"],
        }

        _seed_relationship_profile(user_model_store, contacts, samples_count=100)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO):
            inferrer.infer_from_relationship_profile()

        fallback_msgs = [
            r for r in caplog.records
            if "supplementary" in r.message.lower() and "inbound-only fallback" in r.message.lower()
        ]
        assert len(fallback_msgs) >= 1, "Should log supplementary fallback message"
