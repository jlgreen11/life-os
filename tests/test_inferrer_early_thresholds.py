"""
Tests for lowered semantic fact inference thresholds.

Verifies that the SemanticFactInferrer processes signal profiles at the new
(lowered) sample thresholds, still skips profiles below the new thresholds,
and uses reduced confidence for early inferences via _early_inference_confidence.
"""

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type),
        )


class TestEarlyInferenceConfidenceHelper:
    """Tests for the _early_inference_confidence scaling helper."""

    def test_returns_base_confidence_at_old_threshold(self, user_model_store):
        """At exactly old_threshold samples, return the full base_confidence."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer._early_inference_confidence(samples=10, old_threshold=10, base_confidence=0.5)
        assert result == 0.5

    def test_returns_base_confidence_above_old_threshold(self, user_model_store):
        """Above old_threshold samples, return the full base_confidence."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer._early_inference_confidence(samples=100, old_threshold=10, base_confidence=0.5)
        assert result == 0.5

    def test_scales_down_below_old_threshold(self, user_model_store):
        """Below old_threshold, confidence scales linearly between 0.3 and base."""
        inferrer = SemanticFactInferrer(user_model_store)
        # At 5 samples out of old_threshold=10: ratio = 0.5
        # 0.3 + (0.5 - 0.3) * 0.5 = 0.3 + 0.1 = 0.4
        result = inferrer._early_inference_confidence(samples=5, old_threshold=10, base_confidence=0.5)
        assert result == 0.4

    def test_at_zero_samples(self, user_model_store):
        """At zero samples, returns 0.3 (the minimum)."""
        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer._early_inference_confidence(samples=0, old_threshold=10, base_confidence=0.5)
        assert result == 0.3

    def test_midpoint_interpolation(self, user_model_store):
        """At midpoint between 0 and old_threshold, returns midpoint confidence."""
        inferrer = SemanticFactInferrer(user_model_store)
        # At 25 samples out of old_threshold=50: ratio = 0.5
        # 0.3 + (0.5 - 0.3) * 0.5 = 0.3 + 0.1 = 0.4
        result = inferrer._early_inference_confidence(samples=25, old_threshold=50, base_confidence=0.5)
        assert result == 0.4

    def test_custom_base_confidence(self, user_model_store):
        """Works correctly with a non-default base_confidence."""
        inferrer = SemanticFactInferrer(user_model_store)
        # At old_threshold, returns full base
        result = inferrer._early_inference_confidence(samples=20, old_threshold=20, base_confidence=0.7)
        assert result == 0.7
        # At 0 samples, returns 0.3
        result = inferrer._early_inference_confidence(samples=0, old_threshold=20, base_confidence=0.7)
        assert result == 0.3


class TestRelationshipEarlyThreshold:
    """Test relationship inference with lowered threshold (10 → 5)."""

    def test_relationship_inference_at_5_samples(self, user_model_store):
        """Relationship inference processes at 5 samples (was skipped at old threshold of 10)."""
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 10,
                    "avg_response_time_seconds": 1800,
                    "outbound_count": 5,
                },
                "bob@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,
                },
                "carol@example.com": {
                    "interaction_count": 1,
                    "outbound_count": 1,
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 5)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is True

    def test_still_skips_below_new_threshold(self, user_model_store):
        """Relationship inference still skips at 3 samples (below new threshold of 5)."""
        profile_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 10,
                    "outbound_count": 5,
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 3)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_relationship_profile()

        assert result["processed"] is False
        assert "insufficient samples (<5)" in result["reason"]


class TestCadenceEarlyThreshold:
    """Test cadence inference with lowered threshold (50 → 25)."""

    def test_cadence_inference_at_25_samples(self, user_model_store):
        """Cadence inference processes at 25 samples (was skipped at old threshold of 50)."""
        profile_data = {
            "hourly_activity": {
                str(h): 100 if 9 <= h <= 17 else 0 for h in range(24)
            }
        }
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is True

    def test_cadence_skips_below_new_threshold(self, user_model_store):
        """Cadence inference still skips at 20 samples (below new threshold of 25)."""
        profile_data = {
            "hourly_activity": {str(h): 10 for h in range(24)}
        }
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 20)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_cadence_profile()

        assert result["processed"] is False
        assert "insufficient samples (<25)" in result["reason"]


class TestTopicEarlyThreshold:
    """Test topic inference with lowered threshold (30 → 15)."""

    def test_topic_inference_at_15_samples(self, user_model_store):
        """Topic inference processes at 15 samples (was skipped at old threshold of 30)."""
        profile_data = {
            "topic_counts": {
                "python": 5,  # 5/15 = 33% frequency → above 8% expertise threshold
            }
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 15)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["processed"] is True

        # Should have created an expertise fact for python
        facts = user_model_store.get_semantic_facts(category="expertise")
        python_fact = next((f for f in facts if f["key"] == "expertise_python"), None)
        assert python_fact is not None
        assert python_fact["value"] == "python"

    def test_topic_skips_below_new_threshold(self, user_model_store):
        """Topic inference still skips at 10 samples (below new threshold of 15)."""
        profile_data = {"topic_counts": {"python": 5}}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 10)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert result["processed"] is False
        assert "insufficient samples (<15)" in result["reason"]


class TestMoodEarlyThreshold:
    """Test mood inference with lowered threshold (5 → 3)."""

    def test_mood_inference_at_3_samples(self, user_model_store):
        """Mood inference processes at 3 samples (was skipped at old threshold of 5)."""
        recent_signals = [
            {"signal_type": "positive_language", "value": 0.8}
            for _ in range(10)
        ]
        profile_data = {"recent_signals": recent_signals}
        user_model_store.update_signal_profile("mood_signals", profile_data)
        _set_samples(user_model_store, "mood_signals", 3)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["processed"] is True

    def test_mood_skips_below_new_threshold(self, user_model_store):
        """Mood inference still skips at 2 samples (below new threshold of 3)."""
        profile_data = {"recent_signals": [{"signal_type": "positive_language", "value": 0.8}]}
        user_model_store.update_signal_profile("mood_signals", profile_data)
        _set_samples(user_model_store, "mood_signals", 2)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_mood_profile()

        assert result["processed"] is False
        assert "insufficient samples (<3)" in result["reason"]


class TestTemporalEarlyThreshold:
    """Test temporal inference with lowered threshold (50 → 25)."""

    def test_temporal_inference_at_25_samples(self, user_model_store):
        """Temporal inference processes at 25 samples (was skipped at old threshold of 50)."""
        profile_data = {
            "activity_by_hour": {str(h): 10 if 6 <= h <= 10 else 1 for h in range(24)},
            "activity_by_day": {"monday": 20, "tuesday": 15, "wednesday": 15},
        }
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 25)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_temporal_profile()

        assert result["processed"] is True

    def test_temporal_skips_below_new_threshold(self, user_model_store):
        """Temporal inference still skips at 20 samples (below new threshold of 25)."""
        profile_data = {"activity_by_hour": {str(h): 10 for h in range(24)}}
        user_model_store.update_signal_profile("temporal", profile_data)
        _set_samples(user_model_store, "temporal", 20)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_temporal_profile()

        assert result["processed"] is False
        assert "insufficient samples (<25)" in result["reason"]


class TestSpatialEarlyThreshold:
    """Test spatial inference with lowered threshold (10 → 5)."""

    def test_spatial_inference_at_5_samples(self, user_model_store):
        """Spatial inference processes at 5 samples (was skipped at old threshold of 10)."""
        profile_data = {
            "place_behaviors": {
                "home_office": {"dominant_domain": "work", "visit_count": 5},
            }
        }
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 5)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_spatial_profile()

        assert result["processed"] is True

    def test_spatial_skips_below_new_threshold(self, user_model_store):
        """Spatial inference still skips at 3 samples (below new threshold of 5)."""
        profile_data = {"place_behaviors": {"home": {"visit_count": 1}}}
        user_model_store.update_signal_profile("spatial", profile_data)
        _set_samples(user_model_store, "spatial", 3)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_spatial_profile()

        assert result["processed"] is False
        assert "insufficient samples (<5)" in result["reason"]


class TestDecisionEarlyThreshold:
    """Test decision inference with lowered threshold (20 → 10)."""

    def test_decision_inference_at_10_samples(self, user_model_store):
        """Decision inference processes at 10 samples (was skipped at old threshold of 20)."""
        profile_data = {
            "decision_speed_by_domain": {"finance": 30},  # 30 seconds = quick
            "research_depth_by_domain": {"finance": 0.8},
        }
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 10)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_decision_profile()

        assert result["processed"] is True

    def test_decision_skips_below_new_threshold(self, user_model_store):
        """Decision inference still skips at 8 samples (below new threshold of 10)."""
        profile_data = {"decision_speed_by_domain": {"finance": 30}}
        user_model_store.update_signal_profile("decision", profile_data)
        _set_samples(user_model_store, "decision", 8)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_decision_profile()

        assert result["processed"] is False
        assert "insufficient samples (<10)" in result["reason"]


class TestEarlyConfidenceScaling:
    """Test that early inferences (between new and old thresholds) use reduced confidence."""

    def test_cadence_early_inference_has_lower_confidence(self, user_model_store):
        """Cadence facts at 25 samples have lower confidence than at 50 samples."""
        profile_data = {
            "hourly_activity": {
                str(h): 100 if 9 <= h <= 17 else 0 for h in range(24)
            }
        }

        # First: inference at 50 samples (above old threshold)
        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()
        facts_50 = user_model_store.get_semantic_facts(category="values")
        fact_50 = next((f for f in facts_50 if f["key"] == "work_life_boundaries"), None)

        # Clear facts for a clean test
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute("DELETE FROM semantic_facts")

        # Second: inference at 25 samples (at new threshold, below old)
        _set_samples(user_model_store, "cadence", 25)
        inferrer2 = SemanticFactInferrer(user_model_store)
        inferrer2.infer_from_cadence_profile()
        facts_25 = user_model_store.get_semantic_facts(category="values")
        fact_25 = next((f for f in facts_25 if f["key"] == "work_life_boundaries"), None)

        assert fact_50 is not None
        assert fact_25 is not None
        # Early inference should have lower confidence
        assert fact_25["confidence"] < fact_50["confidence"]
