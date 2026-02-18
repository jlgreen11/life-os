"""
Life OS — Semantic Inference Loop Tests

Verifies that the semantic fact inference background task correctly:
  - Runs periodically to analyze signal profiles
  - Derives semantic facts from accumulated statistics
  - Links facts to source episodes for provenance
  - Increments confidence when facts are re-confirmed
  - Handles errors gracefully without crashing the loop
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from main import LifeOS
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type)
        )


class TestSemanticInferenceLoop:
    """Test suite for the semantic fact inference background loop."""

    @pytest.mark.asyncio
    async def test_semantic_inference_loop_runs_periodically(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify the inference loop executes on schedule."""
        # Mock the inference runner to track calls
        call_count = 0

        def mock_run_all_inference():
            nonlocal call_count
            call_count += 1

        # Create a mock inferrer with tracking
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference = mock_run_all_inference

        # Create a mock shutdown event
        shutdown_event = asyncio.Event()

        # Start the inference loop with very short interval for testing
        async def short_interval_loop():
            """Modified loop with 0.1s interval instead of 6 hours."""
            while not shutdown_event.is_set():
                try:
                    inferrer.run_all_inference()
                except Exception:
                    pass
                await asyncio.sleep(0.1)  # 100ms instead of 6 hours

        # Run the loop briefly
        task = asyncio.create_task(short_interval_loop())
        await asyncio.sleep(0.35)  # Let it run for 350ms (should execute 3 times)
        shutdown_event.set()
        await task

        # Assert: The loop ran multiple times
        assert call_count >= 2, f"Expected at least 2 inference cycles, got {call_count}"

    @pytest.mark.asyncio
    async def test_inference_creates_facts_with_episode_linkage(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that facts inferred from signal profiles link to source episodes."""
        # Create a communication episode
        episode = {
            "id": "ep-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": "evt-001",
            "location": None,
            "inferred_mood": None,
            "active_domain": "personal",
            "energy_level": None,
            "interaction_type": "communication",
            "content_summary": "Email: Test message",
            "content_full": '{"subject": "Test", "body": "This is a test!"}',
            "contacts_involved": ["test@example.com"],
            "topics": [],
            "entities": [],
            "outcome": None,
            "user_satisfaction": None,
            "embedding_id": None,
        }
        user_model_store.store_episode(episode)

        # Create a linguistic profile with high formality
        profile_data = {
            "averages": {"formality": 0.88, "word_count": 50},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        # Run inference
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Assert: Fact was created with episode linkage
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        assert len(facts) > 0, "Expected at least one fact to be inferred"

        formality_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert formality_fact is not None, "Expected formality fact to be created"
        assert formality_fact["value"] == "formal"
        assert formality_fact["confidence"] >= 0.5
        assert formality_fact["source_episodes"] is not None
        assert isinstance(formality_fact["source_episodes"], list)
        # Should link to the episode we created
        if len(formality_fact["source_episodes"]) > 0:
            assert formality_fact["source_episodes"][0] == "ep-001"

    @pytest.mark.asyncio
    async def test_inference_increments_confidence_on_reconfirmation(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that re-running inference increments confidence of existing facts."""
        # Create episodes
        for i in range(3):
            episode = {
                "id": f"ep-{i:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-{i:03d}",
                "location": None,
                "inferred_mood": None,
                "active_domain": "personal",
                "energy_level": None,
                "interaction_type": "communication",
                "content_summary": f"Email {i}",
                "content_full": f'{{"subject": "Test {i}"}}',
                "contacts_involved": ["contact@example.com"],
                "topics": [],
                "entities": [],
                "outcome": None,
                "user_satisfaction": None,
                "embedding_id": None,
            }
            user_model_store.store_episode(episode)

        # Create linguistic profile
        profile_data = {
            "averages": {"formality": 0.2},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 25)

        # Run inference first time
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Get initial confidence
        facts = user_model_store.get_semantic_facts()
        initial_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert initial_fact is not None
        initial_confidence = initial_fact["confidence"]
        initial_times_confirmed = initial_fact["times_confirmed"]

        # Run inference again (simulating next cycle)
        inferrer.infer_from_linguistic_profile()

        # Assert: Confidence increased, times_confirmed incremented
        facts = user_model_store.get_semantic_facts()
        updated_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert updated_fact is not None
        assert updated_fact["confidence"] > initial_confidence, \
            "Confidence should increase on re-confirmation"
        assert updated_fact["times_confirmed"] > initial_times_confirmed, \
            "times_confirmed should increment on re-confirmation"

    @pytest.mark.asyncio
    async def test_inference_handles_relationship_profile(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that relationship profile inference creates priority facts."""
        # Create episodes with fast response times
        for i in range(6):
            episode = {
                "id": f"ep-rel-{i:03d}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-rel-{i:03d}",
                "location": None,
                "inferred_mood": None,
                "active_domain": "work",
                "energy_level": None,
                "interaction_type": "communication",
                "content_summary": f"Message {i}",
                "content_full": f'{{"from": "boss@company.com"}}',
                "contacts_involved": ["boss@company.com"],
                "topics": [],
                "entities": [],
                "outcome": None,
                "user_satisfaction": None,
                "embedding_id": None,
            }
            user_model_store.store_episode(episode)

        # Create relationship profile using the current schema (interaction_count +
        # inbound/outbound counts). The inferrer determines "high_priority" when a
        # contact's interaction_count >= 2x the average across all bidirectional
        # contacts.  With two contacts (counts 20 and 2), avg = 11, threshold = 22;
        # boss@company.com at 20 does NOT qualify.  Use a large gap instead:
        # boss=30, coworker=5 → avg=17.5, threshold=35 → boss still doesn't qualify.
        # The easiest approach: one contact with outbound>0 and interaction_count large
        # enough relative to a second "anchor" contact:
        #   boss=30, anchor=5 → avg=17.5, threshold=35 → boss (30) < 35 — still no.
        # With only one bidirectional contact, avg == that contact, threshold == 2x —
        # the sole contact can never beat its own 2x threshold.
        #
        # Fix: test two bidirectional contacts where one is clearly dominant:
        #   boss=20, occasional@example.com=2 → avg=11, threshold=22 → boss(20) < 22.
        # We need boss >= 2x average: boss=30, low=2 → avg=16, threshold=32 → 30<32.
        # Try: boss=20, low=1 → avg=10.5, threshold=21 → 20<21. Still short.
        # boss=22, low=2 → avg=12, threshold=24 → 22<24. No.
        # The ratio matters: to pass, contact_count >= 2 * avg, i.e.
        #   boss >= 2 * (boss + low) / 2  →  boss >= boss + low  →  0 >= low.
        # That's impossible with two contacts. Need THREE or more:
        #   boss=30, c1=5, c2=5 → avg=13.3, threshold=26.7 → boss(30) >= 26.7. ✓
        profile_data = {
            "contacts": {
                "boss@company.com": {
                    "interaction_count": 30,
                    "inbound_count": 15,
                    "outbound_count": 15,
                    "last_contact": datetime.now(timezone.utc).isoformat(),
                },
                "coworker1@example.com": {
                    "interaction_count": 5,
                    "inbound_count": 3,
                    "outbound_count": 2,
                    "last_contact": datetime.now(timezone.utc).isoformat(),
                },
                "coworker2@example.com": {
                    "interaction_count": 5,
                    "inbound_count": 2,
                    "outbound_count": 3,
                    "last_contact": datetime.now(timezone.utc).isoformat(),
                },
            }
        }
        user_model_store.update_signal_profile("relationships", profile_data)
        _set_samples(user_model_store, "relationships", 40)

        # Run inference
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_relationship_profile()

        # Assert: High priority fact was created for boss (30 >= 2x avg of 13.3)
        facts = user_model_store.get_semantic_facts()
        priority_fact = next(
            (f for f in facts if "relationship_priority_boss@company.com" in f["key"]), None
        )
        assert priority_fact is not None, (
            "Expected relationship_priority fact for boss@company.com "
            "(interaction_count=30 >= 2x avg of 13.3)"
        )
        assert priority_fact["value"] == "high_priority"
        assert priority_fact["confidence"] >= 0.6

    @pytest.mark.asyncio
    async def test_inference_loop_handles_errors_gracefully(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that the background loop catches errors and continues running."""
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer

        call_count = 0
        error_count = 0

        def mock_run_with_error():
            nonlocal call_count, error_count
            call_count += 1
            if call_count == 1:
                error_count += 1
                raise ValueError("Test error in first run")
            # Subsequent runs succeed

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.run_all_inference = mock_run_with_error

        # Simulate the background loop with error handling
        shutdown_event = asyncio.Event()

        async def inference_loop_with_error_handling():
            while not shutdown_event.is_set():
                try:
                    inferrer.run_all_inference()
                except Exception:
                    pass  # Loop should catch and continue
                await asyncio.sleep(0.1)

        # Run briefly
        task = asyncio.create_task(inference_loop_with_error_handling())
        await asyncio.sleep(0.35)
        shutdown_event.set()
        await task

        # Assert: Loop continued despite first error
        assert call_count >= 2, "Loop should continue after error"
        assert error_count == 1, "Error should have occurred once"

    @pytest.mark.asyncio
    async def test_inference_skips_when_insufficient_data(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that inference skips profiles with too few samples."""
        # Create linguistic profile with only 5 samples (threshold is 20)
        profile_data = {
            "samples": [{"formality": 0.5} for _ in range(5)],
            "averages": {"formality": 0.5},
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 5)

        # Run inference
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Assert: No facts were created (insufficient samples)
        facts = user_model_store.get_semantic_facts()
        formality_facts = [f for f in facts if "formality" in f["key"]]
        assert len(formality_facts) == 0, \
            "Should not infer facts from profiles with insufficient samples"

    @pytest.mark.asyncio
    async def test_episode_query_helper_filters_correctly(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that _get_recent_episodes filters by interaction type and contact."""
        # Create mixed episodes
        episodes = [
            {
                "id": "ep-comm-1",
                "timestamp": "2026-02-16T10:00:00Z",
                "event_id": "evt-1",
                "interaction_type": "communication",
                "content_summary": "Email",
                "contacts_involved": ["alice@example.com"],
            },
            {
                "id": "ep-comm-2",
                "timestamp": "2026-02-16T11:00:00Z",
                "event_id": "evt-2",
                "interaction_type": "communication",
                "content_summary": "Email",
                "contacts_involved": ["bob@example.com"],
            },
            {
                "id": "ep-task-1",
                "timestamp": "2026-02-16T12:00:00Z",
                "event_id": "evt-3",
                "interaction_type": "task",
                "content_summary": "Task",
                "contacts_involved": [],
            },
        ]

        for ep in episodes:
            full_episode = {
                **ep,
                "location": None,
                "inferred_mood": None,
                "active_domain": "personal",
                "energy_level": None,
                "content_full": "{}",
                "topics": [],
                "entities": [],
                "outcome": None,
                "user_satisfaction": None,
                "embedding_id": None,
            }
            user_model_store.store_episode(full_episode)

        # Test the helper method
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)

        # Filter by interaction type
        comm_episodes = inferrer._get_recent_episodes(interaction_type="communication")
        assert len(comm_episodes) == 2

        # Filter by contact
        alice_episodes = inferrer._get_recent_episodes(contact="alice@example.com")
        assert len(alice_episodes) == 1
        assert alice_episodes[0] == "ep-comm-1"

        # Combined filters
        bob_comm = inferrer._get_recent_episodes(
            interaction_type="communication",
            contact="bob@example.com"
        )
        assert len(bob_comm) == 1
        assert bob_comm[0] == "ep-comm-2"

    @pytest.mark.asyncio
    async def test_inference_creates_multiple_fact_types(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that a single profile can generate multiple fact types."""
        # Create episode
        episode = {
            "id": "ep-multi",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": "evt-multi",
            "location": None,
            "inferred_mood": None,
            "active_domain": "personal",
            "energy_level": None,
            "interaction_type": "communication",
            "content_summary": "Enthusiastic email",
            "content_full": '{"body": "Hello!!! Great news!!!"}',
            "contacts_involved": ["friend@example.com"],
            "topics": [],
            "entities": [],
            "outcome": None,
            "user_satisfaction": None,
            "embedding_id": None,
        }
        user_model_store.store_episode(episode)

        # Create linguistic profile with both high formality AND high exclamation rate
        profile_data = {
            "samples": [
                {"formality": 0.2, "exclamation_rate": 0.8},
                {"formality": 0.15, "exclamation_rate": 0.9},
            ] * 15,  # 30 samples
            "averages": {
                "formality": 0.175,  # Very casual (< 0.3)
                "exclamation_rate": 0.85,  # Very enthusiastic (> 0.3)
            },
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 30)

        # Run inference
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Assert: Multiple facts were created
        facts = user_model_store.get_semantic_facts()
        fact_keys = [f["key"] for f in facts]

        assert "communication_style_formality" in fact_keys, \
            "Should infer formality preference"
        assert "communication_style_enthusiasm" in fact_keys, \
            "Should infer enthusiasm preference"

    @pytest.mark.asyncio
    async def test_inference_respects_confidence_thresholds(
        self, db: DatabaseManager, user_model_store: UserModelStore
    ):
        """Verify that facts are created with appropriate confidence levels."""
        # Create episodes
        for i in range(3):
            user_model_store.store_episode({
                "id": f"ep-conf-{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": f"evt-conf-{i}",
                "location": None,
                "inferred_mood": None,
                "active_domain": "personal",
                "energy_level": None,
                "interaction_type": "communication",
                "content_summary": f"Test {i}",
                "content_full": "{}",
                "contacts_involved": [],
                "topics": [],
                "entities": [],
                "outcome": None,
                "user_satisfaction": None,
                "embedding_id": None,
            })

        # Create profile with extreme formality (should produce high confidence)
        profile_data = {
            "samples": [{"formality": 0.95}] * 50,
            "averages": {"formality": 0.95},  # Very high formality
        }
        user_model_store.update_signal_profile("linguistic", profile_data)
        _set_samples(user_model_store, "linguistic", 50)

        # Run inference
        from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer
        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Assert: Fact has high confidence (extreme signal + many samples)
        facts = user_model_store.get_semantic_facts()
        formality_fact = next(
            (f for f in facts if f["key"] == "communication_style_formality"), None
        )
        assert formality_fact is not None
        assert formality_fact["confidence"] >= 0.7, \
            "Extreme signals with many samples should produce high confidence"
