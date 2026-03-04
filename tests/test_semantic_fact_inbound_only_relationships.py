"""
Tests for SemanticFactInferrer — inbound-only relationship fallback.

When the user has almost no outbound messages, the main bidirectional
relationship inference path finds zero human_contacts and returns early.
The inbound-only fallback should still derive useful facts (communication
volume category, frequent senders) from contacts that are NOT marketing
or automated senders.
"""

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestInboundOnlyRelationshipFallback:
    """Tests for the inbound-only contact fallback in infer_from_relationship_profile."""

    def _build_inbound_contacts(self, count, base_interaction=10):
        """Build a dict of inbound-only contacts (outbound_count=0).

        Each contact is named person{i}@example.com with decreasing
        interaction counts so the sort order is predictable.
        """
        contacts = {}
        for i in range(count):
            contacts[f"person{i}@example.com"] = {
                "interaction_count": base_interaction + (count - i),
                "outbound_count": 0,
                "inbound_count": base_interaction + (count - i),
            }
        return contacts

    def test_inbound_only_produces_facts_when_no_bidirectional(self, user_model_store):
        """Inbound-only contacts should produce facts when human_contacts is empty."""
        contacts = self._build_inbound_contacts(10)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result is not None
        assert result["type"] == "relationship"
        assert result["processed"] is True
        assert result.get("facts_stored", 0) > 0

        # Check that communication_volume_category was stored
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        volume_fact = next((f for f in facts if f["key"] == "communication_volume_category"), None)
        assert volume_fact is not None
        assert volume_fact["value"] == "moderate_volume_email"  # 10 contacts -> moderate

    def test_marketing_contacts_excluded_from_inbound_only(self, user_model_store):
        """Marketing/noreply contacts should be filtered out of the inbound-only analysis."""
        # 3 human contacts + many marketing contacts = only 3 human, below threshold
        contacts = {
            "alice@example.com": {"interaction_count": 20, "outbound_count": 0, "inbound_count": 20},
            "bob@example.com": {"interaction_count": 15, "outbound_count": 0, "inbound_count": 15},
            "carol@example.com": {"interaction_count": 10, "outbound_count": 0, "inbound_count": 10},
            # Marketing contacts — should be excluded
            "noreply@amazon.com": {"interaction_count": 100, "outbound_count": 0, "inbound_count": 100},
            "newsletter@shop.com": {"interaction_count": 50, "outbound_count": 0, "inbound_count": 50},
            "marketing@bigcorp.com": {"interaction_count": 30, "outbound_count": 0, "inbound_count": 30},
            "updates@service.com": {"interaction_count": 25, "outbound_count": 0, "inbound_count": 25},
        }
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        # Only 3 human inbound-only contacts — below the 5 threshold
        assert result["type"] == "relationship"
        assert result["processed"] is True
        assert result.get("reason") == "too_few_inbound_only"

        # No frequent_sender facts should exist for marketing contacts
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        marketing_facts = [f for f in facts if "noreply@" in f["key"] or "newsletter@" in f["key"]]
        assert len(marketing_facts) == 0

    def test_volume_categories_correct(self, user_model_store):
        """Verify correct volume category assignment for different contact counts."""
        # Test high volume (>50 contacts)
        contacts = self._build_inbound_contacts(55)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        volume_fact = next((f for f in facts if f["key"] == "communication_volume_category"), None)
        assert volume_fact is not None
        assert volume_fact["value"] == "high_volume_email"

    def test_volume_category_low(self, user_model_store):
        """5-9 inbound-only contacts should produce low_volume_email."""
        contacts = self._build_inbound_contacts(7)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        volume_fact = next((f for f in facts if f["key"] == "communication_volume_category"), None)
        assert volume_fact is not None
        assert volume_fact["value"] == "low_volume_email"

    def test_bidirectional_contacts_skip_inbound_fallback(self, user_model_store):
        """When bidirectional human contacts exist, the inbound-only fallback does NOT run."""
        contacts = {
            # Bidirectional human contact
            "alice@example.com": {"interaction_count": 10, "outbound_count": 5, "inbound_count": 5},
            "bob@example.com": {"interaction_count": 8, "outbound_count": 3, "inbound_count": 5},
            # Many inbound-only contacts
            **self._build_inbound_contacts(20),
        }
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["type"] == "relationship"
        assert result["processed"] is True

        # The inbound-only fallback should NOT have run, so no volume category fact
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        volume_fact = next((f for f in facts if f["key"] == "communication_volume_category"), None)
        assert volume_fact is None

    def test_fewer_than_5_inbound_only_skips(self, user_model_store):
        """Fewer than 5 inbound-only human contacts skips inference (too little data)."""
        contacts = self._build_inbound_contacts(3)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["type"] == "relationship"
        assert result["processed"] is True
        assert result.get("reason") == "too_few_inbound_only"

        # No facts should have been stored
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        volume_fact = next((f for f in facts if f["key"] == "communication_volume_category"), None)
        assert volume_fact is None

    def test_top_5_frequent_senders_stored(self, user_model_store):
        """Top 5 frequent senders should each get a frequent_sender fact."""
        contacts = self._build_inbound_contacts(8, base_interaction=5)
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True
        # 1 volume fact + up to 5 sender facts = 6 total
        assert result.get("facts_stored", 0) == 6

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        sender_facts = [f for f in facts if f["key"].startswith("frequent_sender_")]
        assert len(sender_facts) == 5

        # Verify fact values contain interaction counts
        for fact in sender_facts:
            assert fact["value"].startswith("inbound_sender_count_")

    def test_frequent_sender_confidence_scales_with_interactions(self, user_model_store):
        """Frequent sender confidence should scale with interaction count."""
        contacts = {
            # High-interaction contact
            "heavy@example.com": {"interaction_count": 200, "outbound_count": 0, "inbound_count": 200},
            # Low-interaction contacts to fill the minimum
            **{f"filler{i}@example.com": {"interaction_count": 2, "outbound_count": 0, "inbound_count": 2}
               for i in range(6)},
        }
        user_model_store.update_signal_profile("relationships", {"contacts": contacts})
        _set_samples(user_model_store, "relationships", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        heavy_fact = next((f for f in facts if f["key"] == "frequent_sender_heavy@example.com"), None)
        filler_fact = next((f for f in facts if f["key"] == "frequent_sender_filler0@example.com"), None)

        assert heavy_fact is not None
        assert filler_fact is not None
        # Heavy sender (200 interactions) should have higher confidence than filler (2)
        assert heavy_fact["confidence"] > filler_fact["confidence"]
