"""
Tests for neutral/moderate-range semantic fact inference.

Verifies that the SemanticFactInferrer produces meaningful facts for users
whose signal metrics fall in the mid-range (0.3-0.7 formality, moderate
business hours ratio, diverse topic spread) rather than only at extremes.

These tests complement the existing test_semantic_fact_inferrer.py which
covers extreme-value paths. Together they ensure the inferrer produces
useful facts for ALL typical user profiles.
"""

import json
import logging

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type)
        )


def _insert_episode(ums):
    """Insert a minimal episode row for tests that need _get_recent_episodes to return data."""
    from datetime import datetime

    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO episodes (id, timestamp, event_id, interaction_type, "
            "content_summary, contacts_involved) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test-episode-neutral-001",
                datetime.now().isoformat(),
                "evt-neutral-001",
                "email_sent",
                "Test episode for neutral range inference",
                json.dumps(["test@example.com"]),
            ),
        )


class TestNeutralRangeLinguisticInference:
    """Tests for balanced/neutral-range linguistic profile inference."""

    def test_balanced_formality_produces_fact(self, user_model_store):
        """A formality of 0.5 with 50+ samples should produce a 'balanced' formality fact."""
        profile_data = {
            "averages": {"formality": 0.5, "emoji_rate": 0.01, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        balanced_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert balanced_fact is not None, "Expected a balanced formality fact for formality=0.5"
        assert balanced_fact["value"] == "balanced"
        assert balanced_fact["confidence"] > 0.3

    def test_balanced_formality_requires_minimum_samples(self, user_model_store):
        """Balanced formality fact requires 30+ samples to avoid premature inference."""
        profile_data = {
            "averages": {"formality": 0.5, "emoji_rate": 0.01, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 15)  # Below 30 threshold

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        balanced_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert balanced_fact is None, "Should NOT produce balanced fact with <30 samples"

    def test_balanced_formality_at_boundaries(self, user_model_store):
        """Formality at exactly 0.3 and 0.7 should still trigger balanced path."""
        for formality_val in [0.3, 0.7]:
            profile_data = {
                "averages": {"formality": formality_val, "emoji_rate": 0.01, "hedge_rate": 0.1, "exclamation_rate": 0.1},
            }
            user_model_store.update_signal_profile("linguistic", profile_data)
            _set_samples(user_model_store, "linguistic", 50)

            inferrer = SemanticFactInferrer(user_model_store)
            inferrer.infer_from_linguistic_profile()

            facts = user_model_store.get_semantic_facts(category="implicit_preference")
            balanced_fact = next(
                (f for f in facts if f["key"] == "communication_style_formality"), None
            )
            assert balanced_fact is not None, f"Expected balanced fact at formality={formality_val}"
            assert balanced_fact["value"] == "balanced"

    def test_balanced_confidence_scales_with_samples(self, user_model_store):
        """Balanced formality confidence should increase with more samples."""
        # First run with 30 samples
        profile_data = {
            "averages": {"formality": 0.5, "emoji_rate": 0.01, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 30)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        low_sample_conf = next(
            f for f in facts if f["key"] == "communication_style_formality"
        )["confidence"]

        # Re-run with 100 samples (need a fresh store to avoid confidence increment)
        _set_samples(user_model_store, "linguistic", 100)
        # Delete the existing fact to get a clean confidence value
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute("DELETE FROM semantic_facts WHERE key = 'communication_style_formality'")

        inferrer.infer_from_linguistic_profile()
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        high_sample_conf = next(
            f for f in facts if f["key"] == "communication_style_formality"
        )["confidence"]

        assert high_sample_conf > low_sample_conf, "Confidence should be higher with more samples"


class TestNeutralRangeInboundLinguisticInference:
    """Tests for mixed formality inbound environment inference."""

    def test_mixed_formality_environment(self, user_model_store):
        """Inbound formality of 0.5 with 30+ samples produces mixed_formality fact."""
        profile_data = {
            "per_contact_averages": {
                "alice@example.com": {"formality": 0.6, "question_rate": 0.1, "hedge_rate": 0.05, "samples_count": 20},
                "bob@example.com": {"formality": 0.4, "question_rate": 0.2, "hedge_rate": 0.1, "samples_count": 20},
            }
        }
        user_model_store.update_signal_profile("linguistic_inbound", profile_data)
        _set_samples(user_model_store, "linguistic_inbound", 40)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_inbound_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        env_fact = next(
            (f for f in facts if f["key"] == "inbound_communication_environment"), None
        )
        assert env_fact is not None, "Expected mixed formality environment fact"
        assert env_fact["value"] == "mixed_formality_environment"


class TestNeutralRangeCadenceInference:
    """Tests for moderate work-life boundaries and lowered peak hour threshold."""

    def test_moderate_boundaries_from_mixed_hours(self, user_model_store):
        """Business hours ratio of 0.65 should produce moderate_boundaries fact."""
        # 65 messages during business hours, 35 outside
        profile_data = {
            "hourly_activity": {
                str(h): 7 if 9 <= h <= 17 else 2 for h in range(24)
            }
        }
        # Adjust to get ~65% business hours
        # Business hours 9-17 = 9 hours * 7 = 63
        # Off hours = 15 * 2 = 30
        # Total = 93, ratio = 63/93 = 0.677 (within 0.3-0.9 range)

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next(
            (f for f in facts if f["key"] == "work_life_boundaries"), None
        )
        assert boundary_fact is not None, "Expected moderate_boundaries fact"
        assert boundary_fact["value"] == "moderate_boundaries"

    def test_lowered_peak_hour_threshold(self, user_model_store):
        """Peak hour at 15% of traffic (below old 20% threshold) should now produce a fact."""
        # Create a distribution where peak hour is ~15%
        profile_data = {
            "hourly_activity": {str(h): 5 for h in range(24)}
        }
        # Set hour 10 as the peak with 15% of total traffic
        # Total with uniform = 24 * 5 = 120
        # We want peak to be ~15% so peak_count / total >= 0.12
        # Set hour 10 to 22: total = 119 + 22 = 137, ratio = 22/137 = 0.16 > 0.12
        profile_data["hourly_activity"]["10"] = 22

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        peak_fact = next(
            (f for f in facts if f["key"] == "peak_communication_hour"), None
        )
        assert peak_fact is not None, "Expected peak_communication_hour fact with lowered threshold"
        assert peak_fact["value"] == 10


class TestNeutralRangeTopicInference:
    """Tests for diverse interests fact when no single topic dominates."""

    def test_diverse_topics_produce_diversity_fact(self, user_model_store):
        """8 topics each at 5% frequency (none >8%) should produce diverse_interests fact."""
        topic_counts = {
            "python": 25,
            "cooking": 25,
            "travel": 25,
            "music": 25,
            "photography": 25,
            "fitness": 25,
            "reading": 25,
            "gardening": 25,
        }
        # Total samples = 509 (from task description), each at ~5%
        profile_data = {"topic_counts": topic_counts}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 509)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        diversity_fact = next(
            (f for f in facts if f["key"] == "topic_breadth"), None
        )
        assert diversity_fact is not None, "Expected diverse_interests fact"
        assert diversity_fact["value"] == "diverse_interests"
        assert diversity_fact["confidence"] > 0.3

    def test_no_diversity_fact_when_dominant_topic_exists(self, user_model_store):
        """If one topic exceeds 8%, no diversity fact should be created."""
        topic_counts = {
            "python": 100,  # 100/500 = 20% > 8%
            "cooking": 25,
            "travel": 25,
            "music": 25,
            "photography": 25,
            "fitness": 25,
        }
        profile_data = {"topic_counts": topic_counts}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 500)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        diversity_fact = next(
            (f for f in facts if f["key"] == "topic_breadth"), None
        )
        assert diversity_fact is None, "Should NOT produce diversity fact when a dominant topic exists"

    def test_no_diversity_fact_with_fewer_than_5_topics(self, user_model_store):
        """Diversity fact requires at least 5 distinct non-noise topics."""
        topic_counts = {
            "python": 20,
            "cooking": 20,
            "travel": 20,
        }
        profile_data = {"topic_counts": topic_counts}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 500)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        diversity_fact = next(
            (f for f in facts if f["key"] == "topic_breadth"), None
        )
        assert diversity_fact is None, "Should NOT produce diversity fact with <5 topics"

    def test_topic_inference_returns_facts_written(self, user_model_store):
        """Topic inference result dict should include facts_written count."""
        topic_counts = {
            "python": 50,
            "cooking": 25,
            "travel": 25,
            "music": 25,
            "photography": 25,
            "fitness": 25,
            "reading": 25,
            "gardening": 25,
        }
        profile_data = {"topic_counts": topic_counts}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 509)

        inferrer = SemanticFactInferrer(user_model_store)
        result = inferrer.infer_from_topic_profile()

        assert "facts_written" in result, "Result should include facts_written count"
        assert result["facts_written"] >= 1, "Should have written at least 1 fact"


class TestInferenceSummaryFactsWritten:
    """Tests for facts_written reporting in _log_inference_summary."""

    def test_summary_includes_total_facts_written(self, user_model_store, caplog):
        """_log_inference_summary should report total_facts_written count."""
        # Set up topic profile with data that produces facts
        topic_counts = {
            "python": 100,  # 50% frequency — produces expertise fact
        }
        profile_data = {"topic_counts": topic_counts}
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        summary_lines = [
            r.message for r in caplog.records
            if "inference cycle complete" in r.message
        ]
        assert len(summary_lines) == 1
        assert "total_facts_written" in summary_lines[0]

    def test_summary_shows_zero_when_no_facts_produced(self, user_model_store, caplog):
        """When no profiles have data, total_facts_written should be 0."""
        inferrer = SemanticFactInferrer(user_model_store)
        with caplog.at_level(logging.INFO, logger="services.semantic_fact_inferrer.inferrer"):
            inferrer.run_all_inference()

        summary_lines = [
            r.message for r in caplog.records
            if "inference cycle complete" in r.message
        ]
        assert len(summary_lines) == 1
        assert "total_facts_written: 0" in summary_lines[0]


class TestExtremeValueRegression:
    """Regression tests: existing extreme-value facts still work after changes."""

    def test_casual_formality_still_works(self, user_model_store):
        """Low formality (<0.3) should still produce 'casual' fact."""
        profile_data = {
            "averages": {"formality": 0.2, "emoji_rate": 0.02, "hedge_rate": 0.1, "exclamation_rate": 0.1},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        casual_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert casual_fact is not None
        assert casual_fact["value"] == "casual"

    def test_formal_formality_still_works(self, user_model_store):
        """High formality (>0.7) should still produce 'formal' fact."""
        profile_data = {
            "averages": {"formality": 0.8, "emoji_rate": 0.0, "hedge_rate": 0.05, "exclamation_rate": 0.0},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        formal_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert formal_fact is not None
        assert formal_fact["value"] == "formal"

    def test_strict_boundaries_still_works(self, user_model_store):
        """Business hours ratio >0.9 should still produce 'strict_boundaries' fact."""
        profile_data = {
            "hourly_activity": {
                str(h): 100 if 9 <= h <= 17 else 0 for h in range(24)
            }
        }
        profile_data["hourly_activity"]["8"] = 2
        profile_data["hourly_activity"]["18"] = 3

        user_model_store.update_signal_profile("cadence", profile_data)
        _set_samples(user_model_store, "cadence", 50)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_cadence_profile()

        facts = user_model_store.get_semantic_facts(category="values")
        boundary_fact = next(
            (f for f in facts if f["key"] == "work_life_boundaries"), None
        )
        assert boundary_fact is not None
        assert boundary_fact["value"] == "strict_boundaries"

    def test_expertise_topic_still_works(self, user_model_store):
        """Topic with >8% frequency should still produce expertise fact."""
        profile_data = {
            "topic_counts": {"python": 50}
        }
        user_model_store.update_signal_profile("topics", profile_data)
        _set_samples(user_model_store, "topics", 200)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        python_fact = next(
            (f for f in facts if f["key"] == "expertise_python"), None
        )
        assert python_fact is not None
        assert python_fact["value"] == "python"
