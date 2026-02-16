"""
Tests for Semantic Fact Inference Service

Verifies that the SemanticFactInferrer correctly derives high-level semantic
facts from signal profiles across all five extractors.
"""

import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type)
        )


class TestSemanticFactInferrer:
    """Test suite for semantic fact inference across all signal profiles."""

    # -------------------------------------------------------------------
    # Linguistic Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_casual_communication_style(self, user_model_store):
        """Derive casual style preference from low formality scores."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual_fact = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert casual_fact is not None
        assert casual_fact["value"] == "casual"
        assert casual_fact["confidence"] >= 0.6

    def test_infer_formal_communication_style(self, user_model_store):
        """Derive formal style preference from high formality scores."""
        profile_data = {
            "averages": {"formality": 0.8, "emoji_rate": 0.0, "hedge_rate": 0.05, "exclamation_rate": 0.0},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        formal_fact = next((f for f in facts if f["key"] == "communication_style_formality"), None)
        assert formal_fact is not None
        assert formal_fact["value"] == "formal"
        assert formal_fact["confidence"] >= 0.6

    def test_infer_emoji_usage_preference(self, user_model_store):
        """Derive emoji usage preference from high emoji rate."""
        profile_data = {
            "averages": {"formality": 0.5, "emoji_rate": 0.06, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        emoji_fact = next((f for f in facts if f["key"] == "communication_style_emoji"), None)
        assert emoji_fact is not None
        assert emoji_fact["value"] == "expressive_with_emojis"

    def test_no_inference_with_insufficient_samples(self, user_model_store):
        """Skip inference when sample count is below threshold (20)."""
        profile_data = {"averages": {"formality": 0.1}}
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 10)  # Below 20-sample threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    # -------------------------------------------------------------------
    # Relationship Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_high_priority_contact(self, user_model_store):
        """Derive high-priority relationship from fast response times."""
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "message_count": 10,
                    "avg_response_time_seconds": 1800,  # 30 minutes
                }
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_fact = next((f for f in facts if f["key"] == "relationship_priority_alice@example.com"), None)
        assert priority_fact is not None
        assert priority_fact["value"] == "high_priority"

    def test_infer_low_priority_contact(self, user_model_store):
        """Derive low-priority relationship from slow response times."""
        profile_data = {
            "contacts": {
                "bob@example.com": {
                    "message_count": 8,
                    "avg_response_time_seconds": 172800,  # 48 hours
                }
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_fact = next((f for f in facts if f["key"] == "relationship_priority_bob@example.com"), None)
        assert priority_fact is not None
        assert priority_fact["value"] == "low_priority"

    # -------------------------------------------------------------------
    # Topic Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_expertise_from_frequent_topic(self, user_model_store):
        """Derive expertise from frequently discussed topics."""
        profile_data = {
            "topic_frequencies": {
                "python": 50,  # 50% of 100 samples
            }
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        python_fact = next((f for f in facts if f["key"] == "expertise_python"), None)
        assert python_fact is not None
        assert python_fact["value"] == "python"

    # -------------------------------------------------------------------
    # Cadence Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_strict_work_life_boundaries(self, user_model_store):
        """Derive strict boundaries from 90%+ business-hours messaging."""
        profile_data = {
            "hourly_distribution": {
                # 100 messages during business hours (9-17), 5 outside
                str(h): 100 if 9 <= h <= 17 else 0 for h in range(24)
            }
        }
        # Add a few off-hours to make it realistic but still >90%
        profile_data["hourly_distribution"]["8"] = 2
        profile_data["hourly_distribution"]["18"] = 3

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next((f for f in facts if f["key"] == "work_life_boundaries"), None)
        assert boundary_fact is not None
        assert boundary_fact["value"] == "strict_boundaries"

    # -------------------------------------------------------------------
    # Mood Profile Inference Tests
    # -------------------------------------------------------------------

    def test_infer_optimistic_baseline(self, user_model_store):
        """Derive optimistic baseline from positive average sentiment."""
        profile_data = {"avg_sentiment": 0.5}
        user_model_store.update_signal_profile("mood_signals", profile_data)
        _set_samples(user_model_store, "mood_signals", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_mood_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        mood_fact = next((f for f in facts if f["key"] == "emotional_baseline"), None)
        assert mood_fact is not None
        assert mood_fact["value"] == "optimistic"

    # -------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------

    def test_run_all_inference_across_profiles(self, user_model_store):
        """Run inference across all profiles in a single call."""
        # Set up minimal data for each profile type
        user_model_store.update_signal_profile("linguistic", {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        })
        _set_samples(user_model_store, "linguistic", 25)

        user_model_store.update_signal_profile("relationships", {
            "contacts": {"alice@example.com": {"message_count": 10, "avg_response_time_seconds": 1800}}
        })
        _set_samples(user_model_store, "relationships", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        # Assert: Facts created from multiple profiles
        facts = user_model_store.get_semantic_facts()
        assert len(facts) >= 2
        categories = {f["category"] for f in facts}
        assert "implicit_preference" in categories

    def test_confidence_growth_on_repeated_inference(self, user_model_store):
        """Semantic facts gain confidence with repeated observations."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        initial_confidence = next(f for f in facts if f["key"] == "communication_style_formality")["confidence"]

        # Second inference (same pattern observed again)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        updated_confidence = next(f for f in facts if f["key"] == "communication_style_formality")["confidence"]

        # Assert: Confidence increased by 0.05 (update_semantic_fact increment)
        assert updated_confidence > initial_confidence
        assert updated_confidence <= 1.0

    def test_no_facts_created_with_no_profiles(self, user_model_store):
        """Inference gracefully handles missing signal profiles."""
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference()

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_semantic_facts_include_provenance_fields(self, user_model_store):
        """Verify inferred facts include all required provenance fields."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts()
        fact = facts[0]

        # Assert: All required fields are present
        assert "key" in fact
        assert "category" in fact
        assert "value" in fact
        assert "confidence" in fact
        assert "source_episodes" in fact
        assert "first_observed" in fact
        assert "last_confirmed" in fact
        assert "times_confirmed" in fact
        assert "is_user_corrected" in fact
