"""
Tests for topic inference fallback when standard frequency thresholds produce zero facts.

When marketing emails dominate the topic profile, real user topics can be diluted
below the standard 8% (expertise) and 3% (interest) frequency thresholds. The
top-N relative fallback ensures at least the top non-noise topics are captured
as interest facts with reduced confidence.
"""

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestTopicSparseDataFallback:
    """Test suite for the top-N relative fallback in topic inference."""

    def test_standard_thresholds_still_work(self, user_model_store):
        """Standard thresholds produce facts when topics exceed frequency requirements (no regression).

        With 100 total samples, 'python' at 50 occurrences (50%) exceeds the 8%
        expertise threshold, and 'cooking' at 5 occurrences (5%) exceeds the 3%
        interest threshold.
        """
        topic_data = {
            "topic_counts": {
                "python": 50,   # 50% → well above 8% expertise threshold
                "cooking": 5,   # 5% → above 3% interest threshold
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["facts_written"] >= 2

        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]
        interest_keys = [f["key"] for f in facts if f["category"] == "implicit_preference"]

        assert "expertise_python" in expertise_keys
        assert "interest_cooking" in interest_keys

    def test_fallback_fires_when_no_standard_facts(self, user_model_store):
        """Fallback fires when no topics meet standard thresholds but non-noise topics exist with count >= 2.

        With 509 total samples, topics need 41+ occurrences for expertise (8%)
        or 15+ for interest (3%). All topics below those thresholds but above
        count >= 2 should trigger the fallback.
        """
        topic_data = {
            "topic_counts": {
                # Real topics — below standard thresholds at 509 samples
                # (need 41 for expertise, 15 for interest)
                "photography": 12,
                "hiking": 8,
                "woodworking": 6,
                "gardening": 4,
                "astronomy": 3,
                "pottery": 2,
                # Noise tokens — should be filtered even with high counts
                "email": 200,
                "more": 150,
                "click": 100,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 509)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        # Fallback should have created facts
        assert result["facts_written"] >= 1

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        fact_keys = {f["key"] for f in facts}

        # Top non-noise topics should be captured as interest facts
        assert "interest_photography" in fact_keys
        assert "interest_hiking" in fact_keys

        # Verify reduced confidence (fallback uses base_confidence * 0.6 + count/total)
        photo_fact = next(f for f in facts if f["key"] == "interest_photography")
        assert photo_fact["confidence"] <= 0.6, "Fallback facts should have reduced confidence"

    def test_fallback_does_not_fire_when_standard_succeeds(self, user_model_store):
        """Fallback does NOT fire when standard thresholds produce facts.

        If standard path succeeds, the fallback should be skipped entirely.
        We verify this by checking that lower-count topics are NOT captured
        as fallback interest facts.
        """
        topic_data = {
            "topic_counts": {
                "python": 50,       # 50% → expertise via standard path
                "birdwatching": 2,  # Below standard thresholds, would be fallback candidate
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 100)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts()
        fact_keys = {f["key"] for f in facts}

        # Standard path created expertise_python
        assert "expertise_python" in fact_keys
        # Fallback should NOT have fired — birdwatching not captured
        assert "interest_birdwatching" not in fact_keys

    def test_fallback_skips_noise_blocklisted_topics(self, user_model_store):
        """Fallback skips noise-blocklisted topics even with high counts.

        Even in fallback mode, noise tokens (HTML/CSS, stopwords, marketing
        vocabulary) must be filtered out.
        """
        topic_data = {
            "topic_counts": {
                # All noise tokens — should be filtered by blocklist
                "padding": 50,
                "margin": 40,
                "unsubscribe": 30,
                "nbsp": 25,
                "email": 20,
                # One real topic
                "cycling": 3,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 500)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        fact_keys = {f["key"] for f in facts}

        # Noise topics must not appear
        assert "interest_padding" not in fact_keys
        assert "interest_margin" not in fact_keys
        assert "interest_unsubscribe" not in fact_keys
        assert "interest_nbsp" not in fact_keys
        assert "interest_email" not in fact_keys

        # Real topic should be captured via fallback
        assert "interest_cycling" in fact_keys

    def test_fallback_caps_at_five_topics(self, user_model_store):
        """Fallback caps at 5 topics maximum even when more candidates exist."""
        # Create 10 non-noise topics, all below standard thresholds
        topic_data = {
            "topic_counts": {
                "photography": 10,
                "hiking": 9,
                "woodworking": 8,
                "gardening": 7,
                "astronomy": 6,
                "pottery": 5,
                "kayaking": 4,
                "calligraphy": 3,
                "origami": 3,
                "archery": 2,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics", 500)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        # Filter to only fallback interest facts (exclude diverse_interests if present)
        interest_facts = [f for f in facts if f["key"].startswith("interest_")]

        assert len(interest_facts) <= 5, f"Fallback should cap at 5, got {len(interest_facts)}"

        # Top 5 by count should be the ones captured
        fact_keys = {f["key"] for f in interest_facts}
        assert "interest_photography" in fact_keys
        assert "interest_hiking" in fact_keys
        assert "interest_woodworking" in fact_keys
        assert "interest_gardening" in fact_keys
        assert "interest_astronomy" in fact_keys

        # Topics beyond top 5 should NOT be captured
        assert "interest_kayaking" not in fact_keys
        assert "interest_calligraphy" not in fact_keys
