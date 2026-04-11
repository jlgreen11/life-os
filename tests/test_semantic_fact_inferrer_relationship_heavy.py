"""
Tests for SemanticFactInferrer with relationship-heavy datasets.

Validates that the inferrer produces meaningful facts when given large volumes
of relationship data (55K+ samples), many contacts, or rich topic/linguistic
profiles.  These tests address the observed production gap where only 9 facts
were extracted from 869 episodes and 55,091 relationship samples.

Test scenarios:
  - Relationship profile with 1000+ samples → at least 3 facts
  - Topic profile with 100+ samples → at least 1 fact
  - Inbound linguistic profile with 500+ samples → at least 1 fact
  - Aggregate relationship facts generated when 10+ contacts have 5+ interactions
"""

import json

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_relationship_profile(user_model_store, contacts: dict, samples_count: int):
    """Seed the relationship signal profile with given contacts and sample count."""
    user_model_store.update_signal_profile("relationships", {"contacts": contacts})
    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = 'relationships'",
            (samples_count,),
        )


def _seed_topic_profile(user_model_store, topic_counts: dict, samples_count: int):
    """Seed the topics signal profile with given topic_counts and sample count."""
    user_model_store.update_signal_profile("topics", {"topic_counts": topic_counts})
    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = 'topics'",
            (samples_count,),
        )


def _seed_inbound_linguistic_profile(
    user_model_store, per_contact_averages: dict, samples_count: int
):
    """Seed the inbound linguistic signal profile with given per-contact data."""
    user_model_store.update_signal_profile(
        "linguistic_inbound",
        {"per_contact_averages": per_contact_averages},
    )
    with user_model_store.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = 'linguistic_inbound'",
            (samples_count,),
        )


def _get_all_semantic_facts(user_model_store) -> list[dict]:
    """Return all semantic facts as a list of dicts with JSON-deserialized values.

    Mirrors the deserialization logic in UserModelStore.get_semantic_facts() so
    that ``fact["value"]`` is a native Python object (str, int, list) rather
    than a raw JSON string.
    """
    with user_model_store.db.get_connection("user_model") as conn:
        rows = conn.execute("SELECT * FROM semantic_facts").fetchall()
        facts = []
        for row in rows:
            fact = dict(row)
            fact["value"] = json.loads(fact["value"])
            facts.append(fact)
        return facts


# ---------------------------------------------------------------------------
# Relationship profile: large sample counts
# ---------------------------------------------------------------------------

class TestRelationshipHeavyDataset:
    """Relationship profile with 1000+ samples must produce at least 3 facts."""

    def test_relationship_1000_samples_produces_at_least_3_facts(self, user_model_store):
        """
        A realistic relationship profile with 1000+ samples across 30 active
        contacts should produce at least 3 semantic facts including aggregate
        network-level facts (relationship_network_size, regular_contact_count,
        communication_activity_level, etc.).
        """
        contacts = {}
        # 20 human bidirectional contacts with varying interaction counts
        for i in range(20):
            interaction_count = 15 + (i * 3)  # 15 to 72 interactions
            contacts[f"person{i}@work.com"] = {
                "inbound_count": interaction_count // 2,
                "outbound_count": interaction_count // 2,
                "interaction_count": interaction_count,
                "channels_used": ["email"],
            }
        # 10 additional inbound-only contacts
        for i in range(10):
            contacts[f"newsletter{i}@other.com"] = {
                "inbound_count": 5 + i,
                "outbound_count": 0,
                "interaction_count": 5 + i,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=1200)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        assert len(facts) >= 3, (
            f"Expected at least 3 facts from 1000+ samples, got {len(facts)}: "
            f"{[f['key'] for f in facts]}"
        )

    def test_relationship_55k_samples_many_contacts_produces_aggregate_facts(self, user_model_store):
        """
        Simulate the production scenario: 55K+ samples spread across 200 contacts,
        most with moderate individual counts.  The inferrer must produce aggregate
        network facts even when no single contact dominates the distribution.
        """
        contacts = {}
        # 150 bidirectional contacts with moderate interaction counts
        for i in range(150):
            contacts[f"contact{i}@domain{i % 15}.com"] = {
                "inbound_count": 5 + (i % 10),
                "outbound_count": 3 + (i % 5),
                "interaction_count": 8 + (i % 15),
                "channels_used": ["email"],
            }
        # 50 inbound-only contacts
        for i in range(50):
            contacts[f"newsletter{i}@service.com"] = {
                "inbound_count": 10,
                "outbound_count": 0,
                "interaction_count": 10,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=55000)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True

        facts = _get_all_semantic_facts(user_model_store)
        fact_keys = {f["key"] for f in facts}

        # Must have network size fact from the aggregate path
        assert "relationship_network_size" in fact_keys, (
            f"Expected relationship_network_size fact. Got keys: {sorted(fact_keys)}"
        )
        # At least 3 total facts
        assert len(facts) >= 3, (
            f"Expected >= 3 facts from 55K samples, got {len(facts)}: {sorted(fact_keys)}"
        )


# ---------------------------------------------------------------------------
# Aggregate relationship facts: 10+ contacts with 5+ interactions
# ---------------------------------------------------------------------------

class TestAggregateRelationshipFacts:
    """Aggregate facts must fire when 10+ contacts have 5+ interactions."""

    def test_10_contacts_with_5_interactions_produces_aggregate_facts(self, user_model_store):
        """
        Exactly 10 contacts each with 5+ interactions → relationship_network_size
        and regular_contact_count facts must be generated.
        """
        contacts = {}
        for i in range(10):
            contacts[f"person{i}@example.com"] = {
                "inbound_count": 5,
                "outbound_count": 5,
                "interaction_count": 10,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_all_semantic_facts(user_model_store)
        fact_keys = {f["key"] for f in facts}

        assert "relationship_network_size" in fact_keys, (
            f"Expected relationship_network_size. Got: {sorted(fact_keys)}"
        )
        assert "regular_contact_count" in fact_keys, (
            f"Expected regular_contact_count. Got: {sorted(fact_keys)}"
        )

        # Verify the network size value is extensive (10+ regular contacts)
        network_fact = next(f for f in facts if f["key"] == "relationship_network_size")
        assert network_fact["value"] == "extensive_network"

        # Verify regular_contact_count value is 10
        count_fact = next(f for f in facts if f["key"] == "regular_contact_count")
        assert count_fact["value"] == 10

    def test_5_contacts_with_5_interactions_produces_moderate_network_fact(self, user_model_store):
        """5 contacts with 5+ interactions → moderate_network fact."""
        contacts = {}
        for i in range(5):
            contacts[f"friend{i}@example.com"] = {
                "inbound_count": 8,
                "outbound_count": 7,
                "interaction_count": 15,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=75)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_all_semantic_facts(user_model_store)
        network_facts = [f for f in facts if f["key"] == "relationship_network_size"]
        assert len(network_facts) == 1
        assert network_facts[0]["value"] == "moderate_network"

    def test_diverse_domains_produces_network_breadth_fact(self, user_model_store):
        """
        Contacts from 10+ distinct non-marketing email domains should produce
        a contact_network_breadth fact.
        """
        contacts = {}
        domains = [
            "apple.com", "google.com", "microsoft.com", "amazon.com",
            "netflix.com", "tesla.com", "spacex.com", "stripe.com",
            "github.com", "cloudflare.com", "fastly.com",
        ]
        for i, domain in enumerate(domains):
            contacts[f"contact{i}@{domain}"] = {
                "inbound_count": 5 + i,
                "outbound_count": 3 + i,
                "interaction_count": 8 + i,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=200)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_all_semantic_facts(user_model_store)
        breadth_facts = [f for f in facts if f["key"] == "contact_network_breadth"]
        assert len(breadth_facts) == 1, (
            f"Expected contact_network_breadth fact. Got all keys: "
            f"{sorted(f['key'] for f in facts)}"
        )
        assert breadth_facts[0]["value"] == "diverse_multi_domain_network"

    def test_communication_activity_level_for_active_communicator(self, user_model_store):
        """
        Contacts with avg 20+ interactions per person → highly_active_communicator.
        """
        contacts = {}
        for i in range(8):
            contacts[f"colleague{i}@company.com"] = {
                "inbound_count": 15,
                "outbound_count": 15,
                "interaction_count": 30,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=240)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = _get_all_semantic_facts(user_model_store)
        activity_facts = [f for f in facts if f["key"] == "communication_activity_level"]
        assert len(activity_facts) == 1
        assert activity_facts[0]["value"] == "highly_active_communicator"


# ---------------------------------------------------------------------------
# Topic profile: 100+ samples must produce at least 1 fact
# ---------------------------------------------------------------------------

class TestTopicProfileHeavyDataset:
    """Topic profile with 100+ samples must produce at least 1 fact."""

    def test_topic_100_samples_with_clear_interest_produces_fact(self, user_model_store):
        """
        100 samples with a clear non-noise topic appearing 10+ times should
        produce at least 1 expertise or interest fact.
        """
        # python appears 10% of the time (10/100), above the 8% expertise threshold
        topic_counts = {
            "python": 10,
            "programming": 8,
            "software": 5,
            "javascript": 4,
            "database": 3,
        }
        _seed_topic_profile(user_model_store, topic_counts, samples_count=100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        expertise_or_interest = [
            f for f in facts
            if f["key"].startswith("expertise_") or f["key"].startswith("interest_")
        ]
        assert len(expertise_or_interest) >= 1, (
            f"Expected at least 1 expertise/interest fact from 100 samples. "
            f"Got keys: {[f['key'] for f in facts]}"
        )

    def test_topic_100_samples_fallback_with_sparse_counts(self, user_model_store):
        """
        100 samples where no topic meets the frequency threshold but 2+ non-noise
        topics exist → top-N fallback should produce at least 1 fact.
        """
        # No single topic exceeds 8%, but 3 meaningful topics exist
        topic_counts = {
            "cooking": 6,
            "travel": 5,
            "photography": 4,
            # Many other topics dilute frequency
            **{f"random_topic_{i}": 2 for i in range(40)},
        }
        _seed_topic_profile(user_model_store, topic_counts, samples_count=100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        # At minimum the fallback or diverse_interests fact should fire
        assert len(facts) >= 1, (
            f"Expected at least 1 fact from sparse 100-sample topic profile. "
            f"Got: {[f['key'] for f in facts]}"
        )

    def test_topic_860_samples_similar_to_production_produces_facts(self, user_model_store):
        """
        860 topic samples (production-like) with several non-noise topics above
        the 3% interest threshold should produce facts.
        """
        # Simulate a real inbox: a few genuine interest topics among some noise
        topic_counts = {
            "machine_learning": 45,  # 5.2% of 860 → interest (>3%)
            "python": 40,            # 4.7% → interest
            "finance": 35,           # 4.1% → interest
            "investment": 30,        # 3.5% → interest
            "technology": 25,        # 2.9% → below 3%, won't trigger interest
            "health": 20,
            "cooking": 15,
        }
        _seed_topic_profile(user_model_store, topic_counts, samples_count=860)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        interest_or_expertise = [
            f for f in facts
            if f["key"].startswith("interest_") or f["key"].startswith("expertise_")
        ]
        assert len(interest_or_expertise) >= 1, (
            f"Expected interest/expertise facts from 860-sample topic profile. "
            f"All facts: {[f['key'] for f in facts]}"
        )


# ---------------------------------------------------------------------------
# Inbound linguistic profile: 500+ samples must produce at least 1 fact
# ---------------------------------------------------------------------------

class TestInboundLinguisticHeavyDataset:
    """Inbound linguistic profile with 500+ samples must produce at least 1 fact."""

    def test_inbound_linguistic_500_samples_formal_produces_fact(self, user_model_store):
        """
        500 samples with high average formality (>0.7) across contacts should
        produce an inbound_communication_environment fact.
        """
        # Build per_contact_averages: 30 contacts all with formal communication
        per_contact_averages = {}
        for i in range(30):
            per_contact_averages[f"colleague{i}@corp.com"] = {
                "formality": 0.8 + (i % 3) * 0.03,  # 0.80–0.86
                "question_rate": 0.2,
                "hedge_rate": 0.05,
                "samples_count": 15 + i,
            }

        _seed_inbound_linguistic_profile(
            user_model_store, per_contact_averages, samples_count=500
        )

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        assert len(facts) >= 1, (
            f"Expected >= 1 fact from 500-sample formal inbound profile. "
            f"Got: {[f['key'] for f in facts]}"
        )

        env_facts = [f for f in facts if f["key"] == "inbound_communication_environment"]
        assert len(env_facts) == 1
        assert "formal" in env_facts[0]["value"]

    def test_inbound_linguistic_860_samples_casual_produces_fact(self, user_model_store):
        """
        860 inbound samples (production-like) with casual average formality
        should produce a casual_informal_environment fact.
        """
        per_contact_averages = {}
        for i in range(50):
            per_contact_averages[f"friend{i}@example.com"] = {
                "formality": 0.15 + (i % 5) * 0.02,  # 0.15–0.23 (casual)
                "question_rate": 0.3,
                "hedge_rate": 0.1,
                "samples_count": 10 + i,
            }

        _seed_inbound_linguistic_profile(
            user_model_store, per_contact_averages, samples_count=860
        )

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        assert len(facts) >= 1, (
            f"Expected >= 1 fact from 860-sample casual inbound profile. "
            f"Got: {[f['key'] for f in facts]}"
        )

    def test_inbound_linguistic_high_question_rate_produces_fact(self, user_model_store):
        """
        500+ inbound samples where contacts ask many questions (question_rate > 0.5)
        should produce an inbound_question_intensity fact indicating the user is
        frequently asked questions.
        """
        per_contact_averages = {}
        for i in range(20):
            per_contact_averages[f"asker{i}@questions.org"] = {
                "formality": 0.5,
                "question_rate": 0.7 + (i % 3) * 0.05,  # 0.70–0.80 (very high)
                "hedge_rate": 0.1,
                "samples_count": 25,
            }

        _seed_inbound_linguistic_profile(
            user_model_store, per_contact_averages, samples_count=500
        )

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_inbound_linguistic_profile()

        assert result["processed"] is True
        facts = _get_all_semantic_facts(user_model_store)
        question_facts = [f for f in facts if f["key"] == "inbound_question_intensity"]
        assert len(question_facts) == 1
        assert question_facts[0]["value"] == "frequently_asked_questions"


# ---------------------------------------------------------------------------
# End-to-end: run_all_inference with relationship-heavy data
# ---------------------------------------------------------------------------

class TestRunAllInferenceRelationshipHeavy:
    """run_all_inference with large relationship data must produce many facts."""

    def test_run_all_inference_relationship_heavy_minimum_facts(self, user_model_store):
        """
        With 55K+ relationship samples, run_all_inference must produce facts
        from the relationship profile (not 0 across all profiles).
        """
        contacts = {}
        for i in range(25):
            contacts[f"person{i}@domain{i % 8}.com"] = {
                "inbound_count": 20 + i,
                "outbound_count": 10 + i,
                "interaction_count": 30 + i,
                "channels_used": ["email"],
            }

        _seed_relationship_profile(user_model_store, contacts, samples_count=55000)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        facts = _get_all_semantic_facts(user_model_store)
        relationship_facts = [
            f for f in facts
            if any(
                f["key"].startswith(prefix)
                for prefix in [
                    "relationship_", "regular_contact_count",
                    "communication_activity_level", "contact_network_breadth",
                ]
            )
        ]
        assert len(relationship_facts) >= 2, (
            f"Expected >= 2 relationship-type facts from 55K samples. "
            f"Got: {[f['key'] for f in relationship_facts]}"
        )
