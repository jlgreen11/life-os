"""
Test suite for UserModelStore.

Comprehensive coverage of all user model database operations including
episodes, semantic facts, signal profiles, mood tracking, predictions,
and communication templates. Tests both happy paths and edge cases to
ensure data integrity across the four-layer cognitive model.
"""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


@pytest.fixture
def user_model_store(db: DatabaseManager) -> UserModelStore:
    """Provide a UserModelStore instance with clean temp database."""
    return UserModelStore(db)


class TestEpisodicMemory:
    """Test Layer 1: Episodic memory operations."""

    def test_store_episode_minimal(self, user_model_store: UserModelStore):
        """Store episode with only required fields."""
        episode_id = str(uuid4())
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        episode = {
            "id": episode_id,
            "timestamp": timestamp,
            "event_id": event_id,
            "interaction_type": "email_received",
            "content_summary": "Meeting request from Alice",
        }

        user_model_store.store_episode(episode)

        # Verify episode was stored
        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            assert row is not None
            assert row["event_id"] == event_id
            assert row["interaction_type"] == "email_received"
            assert row["content_summary"] == "Meeting request from Alice"

    def test_store_episode_full(self, user_model_store: UserModelStore):
        """Store episode with all optional fields populated."""
        episode_id = str(uuid4())
        event_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        episode = {
            "id": episode_id,
            "timestamp": timestamp,
            "event_id": event_id,
            "location": "home",
            "inferred_mood": {
                "energy_level": 0.7,
                "stress_level": 0.3,
                "valence": 0.6,
            },
            "active_domain": "work",
            "energy_level": 0.7,
            "interaction_type": "email_sent",
            "content_summary": "Sent project update to team",
            "content_full": "Full email body with details...",
            "contacts_involved": ["alice@example.com", "bob@example.com"],
            "topics": ["project_apollo", "milestone_review"],
            "entities": ["Q3 deadline", "budget approval"],
            "outcome": "scheduled_meeting",
            "user_satisfaction": 0.8,
            "embedding_id": "embed_123",
        }

        user_model_store.store_episode(episode)

        # Verify all fields were stored correctly
        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            assert row["location"] == "home"
            assert json.loads(row["inferred_mood"]) == episode["inferred_mood"]
            assert row["active_domain"] == "work"
            assert row["energy_level"] == 0.7
            assert row["content_full"] == "Full email body with details..."
            assert json.loads(row["contacts_involved"]) == episode["contacts_involved"]
            assert json.loads(row["topics"]) == episode["topics"]
            assert json.loads(row["entities"]) == episode["entities"]
            assert row["outcome"] == "scheduled_meeting"
            assert row["user_satisfaction"] == 0.8
            assert row["embedding_id"] == "embed_123"

    def test_store_episode_idempotent(self, user_model_store: UserModelStore):
        """Storing the same episode ID twice should overwrite, not error."""
        episode_id = str(uuid4())

        episode_v1 = {
            "id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid4()),
            "interaction_type": "email_received",
            "content_summary": "Version 1",
        }

        episode_v2 = {
            "id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid4()),
            "interaction_type": "email_sent",
            "content_summary": "Version 2",
        }

        user_model_store.store_episode(episode_v1)
        user_model_store.store_episode(episode_v2)

        # Should have only one row with v2 data
        with user_model_store.db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["content_summary"] == "Version 2"
            assert rows[0]["interaction_type"] == "email_sent"


class TestSemanticMemory:
    """Test Layer 2: Semantic memory operations."""

    def test_update_semantic_fact_new(self, user_model_store: UserModelStore):
        """Create a new semantic fact."""
        user_model_store.update_semantic_fact(
            key="preferred_coffee",
            category="preference",
            value="dark roast",
            confidence=0.5,
            episode_id="ep_123",
        )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 1
        assert facts[0]["key"] == "preferred_coffee"
        assert facts[0]["category"] == "preference"
        assert json.loads(facts[0]["value"]) == "dark roast"
        assert facts[0]["confidence"] == 0.5
        assert json.loads(facts[0]["source_episodes"]) == ["ep_123"]
        assert facts[0]["times_confirmed"] == 1  # First observation counts as 1

    def test_update_semantic_fact_confidence_growth(self, user_model_store: UserModelStore):
        """Confirming an existing fact should increment confidence by 0.05."""
        # Create initial fact
        user_model_store.update_semantic_fact(
            key="meeting_preference",
            category="preference",
            value="mornings",
            confidence=0.5,
            episode_id="ep_1",
        )

        # Confirm it (re-observe the same fact)
        user_model_store.update_semantic_fact(
            key="meeting_preference",
            category="preference",
            value="mornings",
            confidence=0.5,  # Initial confidence ignored for existing facts
            episode_id="ep_2",
        )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 1
        assert facts[0]["confidence"] == 0.55  # 0.5 + 0.05
        assert facts[0]["times_confirmed"] == 2  # Initial + 1 confirmation
        assert json.loads(facts[0]["source_episodes"]) == ["ep_1", "ep_2"]

    def test_update_semantic_fact_confidence_cap(self, user_model_store: UserModelStore):
        """Confidence should never exceed 1.0."""
        key = "test_cap"

        # Create with high confidence
        user_model_store.update_semantic_fact(
            key=key, category="test", value="data", confidence=0.98
        )

        # Confirm multiple times
        for _ in range(10):
            user_model_store.update_semantic_fact(
                key=key, category="test", value="data", confidence=0.5
            )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 1
        assert facts[0]["confidence"] == 1.0  # Capped at 1.0, not 0.98 + 10*0.05

    def test_update_semantic_fact_no_duplicate_episodes(self, user_model_store: UserModelStore):
        """Re-confirming with the same episode ID shouldn't duplicate it."""
        user_model_store.update_semantic_fact(
            key="test_key", category="test", value="data", confidence=0.5, episode_id="ep_1"
        )
        user_model_store.update_semantic_fact(
            key="test_key", category="test", value="data", confidence=0.5, episode_id="ep_1"
        )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 1
        episodes = json.loads(facts[0]["source_episodes"])
        assert episodes == ["ep_1"]  # Not ["ep_1", "ep_1"]

    def test_get_semantic_facts_filter_by_category(self, user_model_store: UserModelStore):
        """Retrieve facts filtered by category."""
        user_model_store.update_semantic_fact(
            key="pref_1", category="preference", value="A", confidence=0.6
        )
        user_model_store.update_semantic_fact(
            key="fact_1", category="explicit", value="B", confidence=0.7
        )
        user_model_store.update_semantic_fact(
            key="pref_2", category="preference", value="C", confidence=0.8
        )

        prefs = user_model_store.get_semantic_facts(category="preference")
        assert len(prefs) == 2
        assert all(f["category"] == "preference" for f in prefs)

    def test_get_semantic_facts_filter_by_confidence(self, user_model_store: UserModelStore):
        """Retrieve facts above a minimum confidence threshold."""
        user_model_store.update_semantic_fact(
            key="low", category="test", value="A", confidence=0.3
        )
        user_model_store.update_semantic_fact(
            key="mid", category="test", value="B", confidence=0.6
        )
        user_model_store.update_semantic_fact(
            key="high", category="test", value="C", confidence=0.9
        )

        high_conf = user_model_store.get_semantic_facts(min_confidence=0.7)
        assert len(high_conf) == 1
        assert high_conf[0]["key"] == "high"

    def test_get_semantic_facts_sorted_by_confidence(self, user_model_store: UserModelStore):
        """Facts should be returned in descending confidence order."""
        user_model_store.update_semantic_fact(
            key="low", category="test", value="A", confidence=0.3
        )
        user_model_store.update_semantic_fact(
            key="high", category="test", value="B", confidence=0.9
        )
        user_model_store.update_semantic_fact(
            key="mid", category="test", value="C", confidence=0.6
        )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 3
        assert facts[0]["key"] == "high"
        assert facts[1]["key"] == "mid"
        assert facts[2]["key"] == "low"


class TestSignalProfiles:
    """Test signal profile storage and retrieval."""

    def test_update_signal_profile_new(self, user_model_store: UserModelStore):
        """Create a new signal profile."""
        profile_data = {
            "avg_word_count": 42.5,
            "vocabulary_richness": 0.75,
            "formality_score": 0.6,
        }

        user_model_store.update_signal_profile("linguistic", profile_data)

        profile = user_model_store.get_signal_profile("linguistic")
        assert profile is not None
        assert profile["profile_type"] == "linguistic"
        assert profile["data"] == profile_data
        assert profile["samples_count"] == 1

    def test_update_signal_profile_increment_samples(self, user_model_store: UserModelStore):
        """Updating a profile should increment samples_count."""
        data_v1 = {"metric": 1.0}
        data_v2 = {"metric": 2.0}

        user_model_store.update_signal_profile("cadence", data_v1)
        user_model_store.update_signal_profile("cadence", data_v2)

        profile = user_model_store.get_signal_profile("cadence")
        assert profile["data"] == data_v2  # Data replaced
        assert profile["samples_count"] == 2  # Counter incremented

    def test_update_signal_profile_multiple_types(self, user_model_store: UserModelStore):
        """Different profile types should be stored independently."""
        user_model_store.update_signal_profile("linguistic", {"lang": "data"})
        user_model_store.update_signal_profile("mood_signals", {"mood": "data"})
        user_model_store.update_signal_profile("cadence", {"cadence": "data"})

        ling = user_model_store.get_signal_profile("linguistic")
        mood = user_model_store.get_signal_profile("mood_signals")
        cadence = user_model_store.get_signal_profile("cadence")

        assert ling["data"] == {"lang": "data"}
        assert mood["data"] == {"mood": "data"}
        assert cadence["data"] == {"cadence": "data"}

    def test_get_signal_profile_nonexistent(self, user_model_store: UserModelStore):
        """Retrieving a profile that doesn't exist should return None."""
        profile = user_model_store.get_signal_profile("nonexistent")
        assert profile is None


class TestMoodTracking:
    """Test mood history logging."""

    def test_store_mood_minimal(self, user_model_store: UserModelStore):
        """Store mood with only some dimensions specified."""
        mood = {
            "energy_level": 0.7,
            "stress_level": 0.4,
        }

        user_model_store.store_mood(mood)

        # Verify mood was stored with defaults for unspecified dimensions
        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM mood_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            assert row["energy_level"] == 0.7
            assert row["stress_level"] == 0.4
            assert row["social_battery"] == 0.5  # Default
            assert row["cognitive_load"] == 0.3  # Default
            assert row["emotional_valence"] == 0.5  # Default
            assert row["confidence"] == 0.0  # Default
            assert row["trend"] == "stable"  # Default

    def test_store_mood_full(self, user_model_store: UserModelStore):
        """Store mood with all dimensions and metadata."""
        timestamp = datetime.now(timezone.utc).isoformat()
        mood = {
            "timestamp": timestamp,
            "energy_level": 0.8,
            "stress_level": 0.2,
            "social_battery": 0.9,
            "cognitive_load": 0.4,
            "emotional_valence": 0.7,
            "confidence": 0.85,
            "contributing_signals": ["signal_1", "signal_2", "signal_3"],
            "trend": "improving",
        }

        user_model_store.store_mood(mood)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM mood_history WHERE timestamp = ?", (timestamp,)
            ).fetchone()
            assert row["energy_level"] == 0.8
            assert row["stress_level"] == 0.2
            assert row["social_battery"] == 0.9
            assert row["cognitive_load"] == 0.4
            assert row["emotional_valence"] == 0.7
            assert row["confidence"] == 0.85
            assert json.loads(row["contributing_signals"]) == mood["contributing_signals"]
            assert row["trend"] == "improving"

    def test_store_mood_time_series(self, user_model_store: UserModelStore):
        """Multiple mood readings should accumulate as a time series."""
        for i in range(5):
            user_model_store.store_mood({"energy_level": 0.5 + i * 0.1})

        with user_model_store.db.get_connection("user_model") as conn:
            rows = conn.execute("SELECT * FROM mood_history").fetchall()
            assert len(rows) == 5


class TestPredictions:
    """Test prediction storage and resolution."""

    def test_store_prediction_minimal(self, user_model_store: UserModelStore):
        """Store prediction with only required fields."""
        pred_id = str(uuid4())
        prediction = {
            "id": pred_id,
            "prediction_type": "reminder",
            "description": "Reply to Alice's email",
            "confidence": 0.7,
            "confidence_gate": "SUGGEST",
        }

        user_model_store.store_prediction(prediction)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            assert row is not None
            assert row["prediction_type"] == "reminder"
            assert row["description"] == "Reply to Alice's email"
            assert row["confidence"] == 0.7
            assert row["confidence_gate"] == "SUGGEST"
            assert row["was_surfaced"] == 0  # Default False

    def test_store_prediction_full(self, user_model_store: UserModelStore):
        """Store prediction with all optional fields."""
        pred_id = str(uuid4())
        prediction = {
            "id": pred_id,
            "prediction_type": "meeting_scheduling",
            "description": "Schedule followup with Bob",
            "confidence": 0.85,
            "confidence_gate": "DEFAULT",
            "time_horizon": "24h",
            "suggested_action": "create_calendar_event",
            "supporting_signals": ["signal_a", "signal_b"],
            "was_surfaced": True,
        }

        user_model_store.store_prediction(prediction)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            assert row["time_horizon"] == "24h"
            assert row["suggested_action"] == "create_calendar_event"
            assert json.loads(row["supporting_signals"]) == ["signal_a", "signal_b"]
            assert row["was_surfaced"] == 1

    def test_resolve_prediction(self, user_model_store: UserModelStore):
        """Resolving a prediction should update was_accurate and resolved_at."""
        pred_id = str(uuid4())
        prediction = {
            "id": pred_id,
            "prediction_type": "reminder",
            "description": "Test prediction",
            "confidence": 0.6,
            "confidence_gate": "SUGGEST",
        }

        user_model_store.store_prediction(prediction)
        user_model_store.resolve_prediction(pred_id, was_accurate=True)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            assert row["was_accurate"] == 1
            assert row["resolved_at"] is not None
            assert row["user_response"] is None

    def test_resolve_prediction_with_feedback(self, user_model_store: UserModelStore):
        """Resolution can include user feedback text."""
        pred_id = str(uuid4())
        prediction = {
            "id": pred_id,
            "prediction_type": "reminder",
            "description": "Test prediction",
            "confidence": 0.6,
            "confidence_gate": "SUGGEST",
        }

        user_model_store.store_prediction(prediction)
        user_model_store.resolve_prediction(
            pred_id,
            was_accurate=False,
            user_response="This was not relevant to my workflow",
        )

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            assert row["was_accurate"] == 0
            assert row["user_response"] == "This was not relevant to my workflow"

    def test_resolve_prediction_idempotent(self, user_model_store: UserModelStore):
        """Resolving a prediction multiple times should update to latest value."""
        pred_id = str(uuid4())
        prediction = {
            "id": pred_id,
            "prediction_type": "reminder",
            "description": "Test prediction",
            "confidence": 0.6,
            "confidence_gate": "SUGGEST",
        }

        user_model_store.store_prediction(prediction)
        user_model_store.resolve_prediction(pred_id, was_accurate=True)
        user_model_store.resolve_prediction(pred_id, was_accurate=False, user_response="Changed my mind")

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (pred_id,)
            ).fetchone()
            # Latest resolution should win
            assert row["was_accurate"] == 0
            assert row["user_response"] == "Changed my mind"


class TestCommunicationTemplates:
    """Test Layer 3: Communication template storage."""

    def test_store_communication_template_minimal(self, user_model_store: UserModelStore):
        """Store template with only required fields."""
        template_id = str(uuid4())
        template = {
            "id": template_id,
            "context": "professional_email",
        }

        user_model_store.store_communication_template(template)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM communication_templates WHERE id = ?", (template_id,)
            ).fetchone()
            assert row is not None
            assert row["context"] == "professional_email"
            assert row["formality"] == 0.5  # Default
            assert row["typical_length"] == 50.0  # Default

    def test_store_communication_template_full(self, user_model_store: UserModelStore):
        """Store template with all style fields populated."""
        template_id = str(uuid4())
        template = {
            "id": template_id,
            "context": "casual_slack",
            "contact_id": "alice@example.com",
            "channel": "slack",
            "greeting": "Hey!",
            "closing": "Cheers",
            "formality": 0.2,
            "typical_length": 30.0,
            "uses_emoji": True,
            "common_phrases": ["sounds good", "let me know", "no worries"],
            "avoids_phrases": ["Dear", "Regards", "Please advise"],
            "tone_notes": ["casual", "friendly", "brief"],
            "example_message_ids": ["msg_1", "msg_2", "msg_3"],
            "samples_analyzed": 15,
        }

        user_model_store.store_communication_template(template)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM communication_templates WHERE id = ?", (template_id,)
            ).fetchone()
            assert row["contact_id"] == "alice@example.com"
            assert row["channel"] == "slack"
            assert row["greeting"] == "Hey!"
            assert row["closing"] == "Cheers"
            assert row["formality"] == 0.2
            assert row["typical_length"] == 30.0
            assert row["uses_emoji"] == 1
            assert json.loads(row["common_phrases"]) == template["common_phrases"]
            assert json.loads(row["avoids_phrases"]) == template["avoids_phrases"]
            assert json.loads(row["tone_notes"]) == template["tone_notes"]
            assert json.loads(row["example_message_ids"]) == template["example_message_ids"]
            assert row["samples_analyzed"] == 15

    def test_store_communication_template_idempotent(self, user_model_store: UserModelStore):
        """Re-storing a template with the same ID should overwrite."""
        template_id = str(uuid4())

        template_v1 = {
            "id": template_id,
            "context": "professional_email",
            "greeting": "Dear",
            "formality": 0.9,
        }

        template_v2 = {
            "id": template_id,
            "context": "casual_email",
            "greeting": "Hi",
            "formality": 0.3,
        }

        user_model_store.store_communication_template(template_v1)
        user_model_store.store_communication_template(template_v2)

        with user_model_store.db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT * FROM communication_templates WHERE id = ?", (template_id,)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["context"] == "casual_email"
            assert rows[0]["greeting"] == "Hi"
            assert rows[0]["formality"] == 0.3


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_json_serialization_special_chars(self, user_model_store: UserModelStore):
        """JSON fields should handle special characters correctly."""
        user_model_store.update_semantic_fact(
            key="test_unicode",
            category="test",
            value="emoji: 😊 quotes: \"nested\" backslash: \\",
            confidence=0.5,
        )

        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 1
        value = json.loads(facts[0]["value"])
        assert "😊" in value
        assert '"nested"' in value

    def test_empty_lists_serialization(self, user_model_store: UserModelStore):
        """Empty list fields should serialize correctly."""
        episode_id = str(uuid4())
        episode = {
            "id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid4()),
            "interaction_type": "test",
            "content_summary": "Test",
            "contacts_involved": [],
            "topics": [],
            "entities": [],
        }

        user_model_store.store_episode(episode)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            assert json.loads(row["contacts_involved"]) == []
            assert json.loads(row["topics"]) == []
            assert json.loads(row["entities"]) == []

    def test_null_optional_fields(self, user_model_store: UserModelStore):
        """Optional fields set to None should be stored as NULL."""
        episode_id = str(uuid4())
        episode = {
            "id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid4()),
            "interaction_type": "test",
            "content_summary": "Test",
            "location": None,
            "active_domain": None,
            "energy_level": None,
        }

        user_model_store.store_episode(episode)

        with user_model_store.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            assert row["location"] is None
            assert row["active_domain"] is None
            assert row["energy_level"] is None
