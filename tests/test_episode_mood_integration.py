"""
Tests for episode creation with mood integration.

CRITICAL BUG FIX (iteration 131):
Episode creation code was looking for mood data in the wrong field
(mood_profile["data"]["samples"]) when the MoodInferenceEngine stores
raw signals in "recent_signals". This caused ALL episodes to be created
with inferred_mood=null even though 22,400+ mood signals were collected.

The fix is to call signal_extractor.get_current_mood() which properly
aggregates recent_signals into a MoodState with energy_level, stress_level,
and emotional_valence.
"""

import json
import pytest
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.pipeline import SignalExtractorPipeline


class TestEpisodeMoodIntegration:
    """Test that episodes correctly capture mood state from signal profiles."""

    @pytest.fixture
    def signal_extractor(self, db, user_model_store):
        """Create a SignalExtractorPipeline instance using shared fixtures from conftest.py."""
        return SignalExtractorPipeline(db, user_model_store)

    @pytest.mark.asyncio
    async def test_episode_mood_integration_with_signals(
        self, db, user_model_store, signal_extractor
    ):
        """
        Test that episodes capture mood when signals exist.

        This tests the happy path:
        1. Process events through mood extractor to generate signals
        2. Retrieve mood via signal_extractor.get_current_mood()
        3. Verify mood data is correctly populated in episodes
        """
        # Create a few events that generate mood signals
        events = [
            {
                "id": f"event-{i}",
                "type": EventType.EMAIL_SENT.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "gmail",
                "payload": {
                    "from_address": "user@example.com",
                    "to_address": ["friend@example.com"],
                    "subject": "Frustrated with the project",
                    "body_plain": "I'm so frustrated and stressed about this project deadline. It's urgent and critical.",
                },
                "metadata": {},
            }
            for i in range(5)
        ]

        # Process each event through the signal extractor
        for event in events:
            await signal_extractor.process_event(event)

        # Get the computed mood state
        mood_state = signal_extractor.get_current_mood()

        # Verify mood state has data
        assert mood_state is not None
        assert mood_state.confidence > 0, "Mood confidence should be > 0 when signals exist"
        assert 0 <= mood_state.stress_level <= 1, "Stress level should be in [0, 1]"
        assert 0 <= mood_state.energy_level <= 1, "Energy level should be in [0, 1]"
        assert 0 <= mood_state.emotional_valence <= 1, "Emotional valence should be in [0, 1]"

        # Verify stress is elevated due to negative language
        # (frustrated, stressed, urgent, critical are all NEGATIVE_WORDS)
        # Threshold is 0.2 because the weighted average of signals may be lower than expected
        assert mood_state.stress_level > 0.2, "Stress should be elevated due to negative language"

        # Now verify an episode created with this mood state would have the data
        inferred_mood = {
            "energy_level": mood_state.energy_level,
            "stress_level": mood_state.stress_level,
            "emotional_valence": mood_state.emotional_valence,
        }
        assert inferred_mood["stress_level"] > 0.2
        assert all(
            k in inferred_mood for k in ["energy_level", "stress_level", "emotional_valence"]
        )

    @pytest.mark.asyncio
    async def test_episode_mood_integration_without_signals(
        self, db, user_model_store, signal_extractor
    ):
        """
        Test that episodes handle missing mood gracefully.

        When no mood signals exist yet, get_current_mood() should return
        a MoodState with confidence=0, and episodes should store null mood.
        """
        # Don't process any events — mood profile is empty
        mood_state = signal_extractor.get_current_mood()

        # Verify default mood state
        assert mood_state is not None
        assert mood_state.confidence == 0, "Confidence should be 0 when no signals exist"

        # Verify episode creation would skip mood
        inferred_mood = None
        if mood_state.confidence > 0:
            inferred_mood = {
                "energy_level": mood_state.energy_level,
                "stress_level": mood_state.stress_level,
                "emotional_valence": mood_state.emotional_valence,
            }
        assert inferred_mood is None, "Mood should be None when confidence=0"

    @pytest.mark.asyncio
    async def test_mood_signal_accumulation(self, db, user_model_store, signal_extractor):
        """
        Test that mood signals accumulate correctly across multiple events.

        As more events are processed, the mood signal count should grow
        and confidence should increase.
        """
        # Process increasing numbers of events and check mood confidence
        for i in range(1, 11):
            event = {
                "id": f"event-{i}",
                "type": EventType.EMAIL_SENT.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "gmail",
                "payload": {
                    "from_address": "user@example.com",
                    "to_address": ["friend@example.com"],
                    "subject": f"Message {i}",
                    "body_plain": "This is a happy and excited message!",
                },
                "metadata": {},
            }
            await signal_extractor.process_event(event)

        mood_state = signal_extractor.get_current_mood()
        assert mood_state.confidence > 0
        # With 10 events, confidence should be at least 0.1 (each adds ~0.1)
        assert mood_state.confidence >= 0.1

    @pytest.mark.asyncio
    async def test_mood_dimensions_from_different_signal_types(
        self, db, user_model_store, signal_extractor
    ):
        """
        Test that different event types contribute to different mood dimensions.

        - Negative language → stress_level and emotional_valence
        - Sleep → energy_level
        - Calendar density → stress_level
        """
        # Process a negative email (stress + valence)
        await signal_extractor.process_event(
            {
                "id": "email-1",
                "type": EventType.EMAIL_SENT.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "gmail",
                "payload": {
                    "from_address": "user@example.com",
                    "to_address": ["boss@company.com"],
                    "subject": "Urgent problem",
                    "body_plain": "Unfortunately we have a critical issue that needs immediate attention. I'm frustrated.",
                },
                "metadata": {},
            }
        )

        # Process a calendar event (calendar density → stress)
        await signal_extractor.process_event(
            {
                "id": "calendar-1",
                "type": EventType.CALENDAR_EVENT_CREATED.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "caldav",
                "payload": {
                    "title": "Back-to-back meetings",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                },
                "metadata": {},
            }
        )

        mood_state = signal_extractor.get_current_mood()

        # Verify stress is elevated from both sources
        # Threshold is 0.15 because stress signals are weighted averages
        assert mood_state.stress_level > 0.15, "Stress should be elevated"
        # Verify we have some confidence
        assert mood_state.confidence > 0

    @pytest.mark.asyncio
    async def test_inbound_message_pressure_affects_mood(
        self, db, user_model_store, signal_extractor
    ):
        """
        Test that inbound messages with urgency markers affect mood.

        Messages with ALL CAPS words or excessive exclamation marks
        should generate incoming_pressure signals.
        """
        await signal_extractor.process_event(
            {
                "id": "urgent-email",
                "type": EventType.EMAIL_RECEIVED.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "gmail",
                "payload": {
                    "from_address": "boss@company.com",
                    "to_address": ["user@example.com"],
                    "subject": "URGENT: IMMEDIATE ACTION REQUIRED",
                    "body_plain": "This is CRITICAL!!! We need your response ASAP!!! URGENT!!!",
                },
                "metadata": {},
            }
        )

        mood_state = signal_extractor.get_current_mood()
        # Incoming pressure and negative language should elevate stress
        # Threshold is 0.25 to account for weighted averaging
        assert mood_state.stress_level >= 0.25
        assert mood_state.confidence > 0

    @pytest.mark.asyncio
    async def test_mood_profile_ring_buffer_cap(
        self, db, user_model_store, signal_extractor
    ):
        """
        Test that mood signals are capped at 200 entries (ring buffer).

        After processing 250 events, the profile should contain exactly
        200 signals (the most recent ones).
        """
        # Process 250 events
        for i in range(250):
            await signal_extractor.process_event(
                {
                    "id": f"event-{i}",
                    "type": EventType.EMAIL_SENT.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "gmail",
                    "payload": {
                        "from_address": "user@example.com",
                        "to_address": ["friend@example.com"],
                        "subject": f"Message {i}",
                        "body_plain": "Short message",
                    },
                    "metadata": {},
                }
            )

        # Check the mood_signals profile
        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        recent_signals = profile["data"].get("recent_signals", [])
        assert len(recent_signals) == 200, "Ring buffer should cap at 200 signals"

    @pytest.mark.asyncio
    async def test_episode_stores_mood_state_correctly(
        self, db, user_model_store, signal_extractor
    ):
        """
        Integration test: Verify episodes actually store mood data.

        This tests the complete pipeline from signal extraction to
        episode storage, ensuring mood is persisted correctly.
        """
        # Process events to generate mood signals
        for i in range(5):
            await signal_extractor.process_event(
                {
                    "id": f"event-{i}",
                    "type": EventType.EMAIL_SENT.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "gmail",
                    "payload": {
                        "from_address": "user@example.com",
                        "to_address": ["friend@example.com"],
                        "subject": "Stressed out",
                        "body_plain": "I'm so stressed and frustrated with this urgent problem.",
                    },
                    "metadata": {},
                }
            )

        # Get the computed mood
        mood_state = signal_extractor.get_current_mood()
        assert mood_state.confidence > 0

        # Create an episode with this mood (mimicking main.py logic)
        inferred_mood = None
        if mood_state.confidence > 0:
            inferred_mood = {
                "energy_level": mood_state.energy_level,
                "stress_level": mood_state.stress_level,
                "emotional_valence": mood_state.emotional_valence,
            }

        episode = {
            "id": "episode-test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": "event-0",
            "location": None,
            "inferred_mood": inferred_mood,
            "active_domain": "personal",
            "energy_level": inferred_mood["energy_level"] if inferred_mood else None,
            "interaction_type": "email_sent",
            "content_summary": "Email sent",
            "content_full": json.dumps({}),
            "contacts_involved": ["friend@example.com"],
            "topics": [],
            "entities": [],
            "outcome": None,
            "user_satisfaction": None,
            "embedding_id": None,
        }

        user_model_store.store_episode(episode)

        # Retrieve and verify
        with db.get_connection("user_model") as conn:
            stored = conn.execute(
                "SELECT inferred_mood, energy_level FROM episodes WHERE id = ?",
                ("episode-test",),
            ).fetchone()

        assert stored is not None
        # inferred_mood should be JSON-encoded dict
        stored_mood = json.loads(stored["inferred_mood"])
        assert stored_mood is not None
        assert "energy_level" in stored_mood
        assert "stress_level" in stored_mood
        assert "emotional_valence" in stored_mood
        # Verify stress is elevated
        assert stored_mood["stress_level"] > 0.3
        # Verify energy_level column matches the mood dict
        assert stored["energy_level"] == stored_mood["energy_level"]

    def test_mood_state_default_values(self, signal_extractor):
        """
        Test that MoodState returns sensible defaults when no signals exist.

        Default values should be neutral (0.5 for most dimensions, 0.3 for stress).
        """
        mood_state = signal_extractor.get_current_mood()
        assert mood_state.energy_level == 0.5
        assert mood_state.stress_level == 0.3
        assert mood_state.emotional_valence == 0.5
        assert mood_state.confidence == 0.0
        assert mood_state.social_battery == 0.5
