"""
Tests for episode linkage in semantic fact inference.

Verifies that all semantic facts are properly linked to source episodes,
enabling the confidence growth loop and audit trail back to raw observations.
"""

import json
from datetime import datetime, timezone

import pytest

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


class TestEpisodeLinkageFix:
    """Tests for episode linkage in semantic fact inference."""

    @pytest.fixture
    def inferrer(self, user_model_store):
        """Create a semantic fact inferrer instance."""
        return SemanticFactInferrer(user_model_store)

    @pytest.fixture
    def sample_episodes(self, user_model_store, db):
        """
        Create sample episodes in the database to link against.

        Returns:
            List of episode IDs that can be used in tests
        """
        episodes = []

        # Create 10 communication episodes with timestamps spread out
        for i in range(10):
            episode = {
                "id": f"ep-comm-{i:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-{i:03d}",
                "interaction_type": "communication",
                "content_summary": f"Test communication {i}",
                "contacts_involved": json.dumps([f"contact{i}@example.com"]),
                "inferred_mood": json.dumps({"energy_level": 0.7, "stress_level": 0.3}),
                "active_domain": "personal",
            }
            user_model_store.store_episode(episode)
            episodes.append(episode["id"])

        # Create 5 calendar episodes
        for i in range(5):
            episode = {
                "id": f"ep-cal-{i:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-cal-{i:03d}",
                "interaction_type": "calendar",
                "content_summary": f"Test calendar event {i}",
                "contacts_involved": json.dumps([]),
                "active_domain": "work",
            }
            user_model_store.store_episode(episode)

        return episodes

    def test_topic_inference_links_episodes(self, inferrer, user_model_store, sample_episodes):
        """
        Test that topic inference properly links episodes to facts.

        Before the fix, topic facts had no episode linkage.
        After the fix, all topic facts should reference source episodes.
        """
        # Create a topic profile with sufficient samples
        # update_signal_profile only takes the data dict, and samples_count is auto-managed
        # We need to call it 100 times to get 100 samples, but that's inefficient.
        # Instead, update the profile once, then manually set samples_count in DB.
        topic_data = {
            "topic_frequencies": {
                "python": 50,  # 50% frequency → expertise
                "docker": 15,  # 15% frequency → interest
                "kubernetes": 3,  # 3% frequency → ignored (too low)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)

        # Manually set samples_count to 100 to meet threshold
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET samples_count = 100 WHERE profile_type = 'topics'"
            )

        # Run topic inference
        inferrer.infer_from_topic_profile()

        # Verify expertise fact has episode linkage
        expertise_facts = user_model_store.get_semantic_facts(category="expertise")
        assert len(expertise_facts) >= 1, "Should have at least 1 expertise fact"

        # Find the python fact (should have 50% frequency, highest)
        python_fact = [f for f in expertise_facts if f["key"] == "expertise_python"][0]
        assert python_fact["value"] == "python"

        # CRITICAL: Verify episode linkage exists
        assert "source_episodes" in python_fact
        assert isinstance(python_fact["source_episodes"], list)
        assert len(python_fact["source_episodes"]) > 0, \
            "Topic facts must link to source episodes for audit trail"

        # Episode should be one of our communication episodes
        linked_episode = python_fact["source_episodes"][0]
        assert linked_episode.startswith("ep-comm-"), \
            f"Expected communication episode, got {linked_episode}"

    def test_cadence_inference_links_episodes(self, inferrer, user_model_store, sample_episodes):
        """
        Test that cadence inference properly links episodes to facts.

        Before the fix, cadence facts had no episode linkage.
        After the fix, all cadence facts should reference source episodes.
        """
        # Create a cadence profile showing strong work-life boundaries
        cadence_data = {
            "hourly_distribution": {
                # 95% of messages during business hours (9-17)
                "9": 10, "10": 12, "11": 15, "12": 10,
                "13": 14, "14": 16, "15": 13, "16": 10,
                # 5% outside business hours
                "18": 2, "19": 1, "20": 1, "21": 1,
            }
        }
        user_model_store.update_signal_profile("cadence", cadence_data)

        # Manually set samples_count to 100 to meet threshold
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET samples_count = 100 WHERE profile_type = 'cadence'"
            )

        # Run cadence inference
        inferrer.infer_from_cadence_profile()

        # Verify work-life boundaries fact has episode linkage
        facts = user_model_store.get_semantic_facts(category="values")
        assert len(facts) >= 1, "Should have at least 1 value fact (work_life_boundaries)"

        boundary_fact = [f for f in facts if f["key"] == "work_life_boundaries"][0]
        assert boundary_fact["value"] == "strict_boundaries"

        # CRITICAL: Verify episode linkage exists
        assert "source_episodes" in boundary_fact
        assert isinstance(boundary_fact["source_episodes"], list)
        assert len(boundary_fact["source_episodes"]) > 0, \
            "Cadence facts must link to source episodes for audit trail"

        # Episode should be one of our communication episodes
        linked_episode = boundary_fact["source_episodes"][0]
        assert linked_episode.startswith("ep-comm-"), \
            f"Expected communication episode, got {linked_episode}"

    def test_mood_inference_links_episodes(self, inferrer, user_model_store, sample_episodes):
        """
        Test that mood inference properly links episodes to facts.

        Before the fix, mood facts had no episode linkage.
        After the fix, all mood facts should reference source episodes.
        """
        # Create a mood profile with optimistic baseline
        mood_data = {
            "avg_sentiment": 0.6,  # Positive baseline → optimistic
            "samples": [],
        }
        user_model_store.update_signal_profile("mood_signals", mood_data)

        # Manually set samples_count to 150 to meet threshold
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET samples_count = 150 WHERE profile_type = 'mood_signals'"
            )

        # Run mood inference
        inferrer.infer_from_mood_profile()

        # Verify emotional baseline fact has episode linkage
        facts = user_model_store.get_semantic_facts()
        emotional_facts = [f for f in facts if f["key"] == "emotional_baseline"]
        assert len(emotional_facts) == 1, "Should have 1 emotional baseline fact"

        baseline_fact = emotional_facts[0]
        assert baseline_fact["value"] == "optimistic"

        # CRITICAL: Verify episode linkage exists
        assert "source_episodes" in baseline_fact
        assert isinstance(baseline_fact["source_episodes"], list)
        assert len(baseline_fact["source_episodes"]) > 0, \
            "Mood facts must link to source episodes for audit trail"

        # Episode should be one of our communication episodes
        linked_episode = baseline_fact["source_episodes"][0]
        assert linked_episode.startswith("ep-comm-"), \
            f"Expected communication episode, got {linked_episode}"

    def test_all_fact_types_have_episode_linkage(self, inferrer, user_model_store, sample_episodes):
        """
        Integration test: Run all inference methods and verify every fact has episodes.

        This is the critical test that would have caught the bug before deployment.
        """
        # Create comprehensive profiles for all signal types
        profiles = {
            "linguistic": {
                "averages": {
                    "formality": 0.2,  # Casual
                    "emoji_rate": 0.08,  # Expressive
                }
            },
            "relationships": {
                "contacts": {
                    "contact0@example.com": {  # Match the contacts in sample_episodes
                        "interaction_count": 10,
                        "avg_response_time_seconds": 1800,  # 30 min → high priority
                    }
                }
            },
            "topics": {
                "topic_frequencies": {
                    "ai": 20,  # 33% → expertise
                }
            },
            "cadence": {
                "hourly_distribution": {
                    "10": 30, "11": 25, "14": 20, "15": 5,  # Peak at 10am
                }
            },
            "mood_signals": {
                "avg_sentiment": 0.5,  # Positive → optimistic
            },
        }

        # Store all profiles and set their samples_count
        samples_required = {
            "linguistic": 50,
            "relationships": 20,
            "topics": 60,
            "cadence": 80,
            "mood_signals": 120,
        }

        for profile_type, profile_data in profiles.items():
            user_model_store.update_signal_profile(profile_type, profile_data)
            # Manually set samples_count to meet thresholds
            with user_model_store.db.get_connection("user_model") as conn:
                conn.execute(
                    "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
                    (samples_required[profile_type], profile_type)
                )

        # Run all inference
        inferrer.run_all_inference()

        # Verify ALL facts have episode linkage
        all_facts = user_model_store.get_semantic_facts()
        assert len(all_facts) > 0, "Should have generated facts from all profiles"

        facts_without_episodes = []
        for fact in all_facts:
            if not fact.get("source_episodes") or len(fact["source_episodes"]) == 0:
                facts_without_episodes.append(fact["key"])

        assert len(facts_without_episodes) == 0, \
            f"All facts must have episode linkage. Missing: {facts_without_episodes}"

    def test_confidence_growth_loop_works_with_episodes(self, inferrer, user_model_store, sample_episodes):
        """
        Test that the confidence growth loop (+0.05 per re-confirmation) works correctly.

        This tests the end-to-end flow: fact created → re-confirmed → confidence increases.
        """
        # Create topic profile
        topic_data = {
            "topic_frequencies": {"rust": 20}  # 40% → expertise
        }
        user_model_store.update_signal_profile("topics", topic_data)

        # Manually set samples_count to 50 to meet threshold
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET samples_count = 50 WHERE profile_type = 'topics'"
            )

        # First inference run
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        initial_fact = [f for f in facts if f["key"] == "expertise_rust"][0]
        initial_confidence = initial_fact["confidence"]
        initial_times_confirmed = initial_fact["times_confirmed"]
        initial_episodes = len(initial_fact["source_episodes"])

        assert initial_times_confirmed == 1, "First observation should set times_confirmed=1"
        assert initial_episodes >= 1, "Should have at least one linked episode"

        # Simulate new episodes being created
        for i in range(3):
            episode = {
                "id": f"ep-new-{i:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-new-{i:03d}",
                "interaction_type": "communication",
                "content_summary": f"Discussion about Rust {i}",
                "contacts_involved": json.dumps(["dev@example.com"]),
                "active_domain": "work",
            }
            user_model_store.store_episode(episode)

        # Second inference run (re-confirmation)
        inferrer.infer_from_topic_profile()

        facts = user_model_store.get_semantic_facts(category="expertise")
        updated_fact = [f for f in facts if f["key"] == "expertise_rust"][0]
        updated_confidence = updated_fact["confidence"]
        updated_times_confirmed = updated_fact["times_confirmed"]

        # Verify confidence grew by +0.05
        assert updated_confidence > initial_confidence, \
            "Confidence should increase on re-confirmation"
        assert abs(updated_confidence - (initial_confidence + 0.05)) < 0.01, \
            "Confidence should grow by exactly +0.05 per re-confirmation"

        # Verify times_confirmed incremented
        assert updated_times_confirmed == initial_times_confirmed + 1, \
            "times_confirmed should increment on each re-confirmation"

    def test_no_episodes_graceful_degradation(self, inferrer, user_model_store, db):
        """
        Test that inference works even when no episodes exist (graceful degradation).

        Facts should still be created, but source_episodes will be an empty list.
        """
        # Don't create any episodes (no sample_episodes fixture)

        # Create topic profile
        topic_data = {
            "topic_frequencies": {"golang": 15}
        }
        user_model_store.update_signal_profile("topics", topic_data)

        # Manually set samples_count to 40 to meet threshold
        with user_model_store.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE signal_profiles SET samples_count = 40 WHERE profile_type = 'topics'"
            )

        # Run inference
        inferrer.infer_from_topic_profile()

        # Fact should still be created
        facts = user_model_store.get_semantic_facts(category="expertise")
        assert len(facts) == 1, "Should create fact even without episodes"

        golang_fact = facts[0]
        assert golang_fact["key"] == "expertise_golang"

        # source_episodes should exist but be empty
        assert "source_episodes" in golang_fact
        assert isinstance(golang_fact["source_episodes"], list)
        # It's okay to be empty when no episodes exist — system degrades gracefully
        # The important thing is the field exists and is a valid list
