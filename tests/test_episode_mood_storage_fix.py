"""
Tests for episode mood storage fix (iteration 150).

CRITICAL BUG FIX:
All 31,534 episodes had NULL mood data despite 29,626 mood signals being
available and mood computation working perfectly (confidence=1.0).

ROOT CAUSE:
In user_model_store.py line 134, `episode.get("inferred_mood", {})` would
return None (not {}) when the key existed with None value, causing
json.dumps(None) to serialize as the string "null" instead of proper JSON.

This test verifies:
1. Episodes with mood data store it correctly as JSON
2. Episodes without mood data store {} (empty dict) not "null"
3. Mood retrieval from stored episodes works correctly
4. Edge cases (None, empty dict, partial mood) all serialize properly
"""

import json
import pytest
import uuid
from datetime import datetime, timezone

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def test_episode_with_full_mood_stores_correctly(db):
    """Episode with complete mood data should store all fields as JSON."""
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())
    episode = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-123",
        "location": "home",
        "inferred_mood": {
            "energy_level": 0.8,
            "stress_level": 0.3,
            "emotional_valence": 0.9,
        },
        "active_domain": "personal",
        "energy_level": 0.8,
        "interaction_type": "email_received",
        "content_summary": "Test episode with full mood data",
        "content_full": json.dumps({"test": "data"}),
        "contacts_involved": ["test@example.com"],
        "topics": ["testing", "mood"],
        "entities": [],
    }

    ums.store_episode(episode)

    # Verify the episode was stored correctly
    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT inferred_mood, energy_level FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "Episode should be stored"

        stored_mood = json.loads(row[0])
        assert stored_mood == {
            "energy_level": 0.8,
            "stress_level": 0.3,
            "emotional_valence": 0.9,
        }, "Mood should be stored as proper JSON dict"

        assert row[1] == 0.8, "Energy level should be stored separately"


def test_episode_with_none_mood_stores_empty_dict(db):
    """Episode with None mood should store {} not 'null' string.

    This is the PRIMARY bug fix — previously dict.get("inferred_mood", {})
    would return None when the key existed with None value, causing
    json.dumps(None) to produce "null" string instead of "{}".
    """
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())
    episode = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-456",
        "location": None,
        "inferred_mood": None,  # CRITICAL: None value, key exists
        "active_domain": "work",
        "energy_level": None,
        "interaction_type": "email_sent",
        "content_summary": "Test episode with None mood",
        "content_full": json.dumps({"test": "data"}),
        "contacts_involved": [],
        "topics": [],
        "entities": [],
    }

    ums.store_episode(episode)

    # Verify the episode stored {} not "null"
    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT inferred_mood, energy_level FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "Episode should be stored"

        stored_mood = json.loads(row[0])
        assert stored_mood == {}, "None mood should serialize as empty dict {}, not null"

        assert row[1] is None, "None energy level should stay None"


def test_episode_without_mood_key_stores_empty_dict(db):
    """Episode missing inferred_mood key should store {} not 'null'."""
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())
    episode = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-789",
        "interaction_type": "message_received",
        "content_summary": "Test episode without mood key",
        # inferred_mood key is completely missing
    }

    ums.store_episode(episode)

    # Verify the episode stored {} not "null"
    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT inferred_mood FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "Episode should be stored"

        stored_mood = json.loads(row[0])
        assert stored_mood == {}, "Missing mood key should default to empty dict {}"


def test_episode_with_partial_mood_stores_correctly(db):
    """Episode with partial mood data (missing some fields) should store as-is."""
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())
    episode = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-partial",
        "inferred_mood": {
            "energy_level": 0.6,
            # stress_level and emotional_valence missing
        },
        "energy_level": 0.6,
        "interaction_type": "calendar_event",
        "content_summary": "Test episode with partial mood",
    }

    ums.store_episode(episode)

    # Verify partial mood is preserved exactly
    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT inferred_mood FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "Episode should be stored"

        stored_mood = json.loads(row[0])
        assert stored_mood == {
            "energy_level": 0.6,
        }, "Partial mood should be stored as-is without defaults"


def test_episode_update_overwrites_mood(db):
    """Re-storing same episode ID should overwrite mood data (INSERT OR REPLACE)."""
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())

    # Store episode with no mood
    episode_v1 = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-update",
        "inferred_mood": None,
        "energy_level": None,
        "interaction_type": "email_received",
        "content_summary": "Version 1 without mood",
    }
    ums.store_episode(episode_v1)

    # Update same episode with mood data
    episode_v2 = {
        "id": episode_id,  # Same ID triggers INSERT OR REPLACE
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-event-update",
        "inferred_mood": {
            "energy_level": 0.7,
            "stress_level": 0.4,
            "emotional_valence": 0.6,
        },
        "energy_level": 0.7,
        "interaction_type": "email_received",
        "content_summary": "Version 2 with mood",
    }
    ums.store_episode(episode_v2)

    # Verify the mood was updated
    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT inferred_mood, energy_level FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "Episode should exist"

        stored_mood = json.loads(row[0])
        assert stored_mood == {
            "energy_level": 0.7,
            "stress_level": 0.4,
            "emotional_valence": 0.6,
        }, "Mood should be updated on re-store"

        assert row[1] == 0.7, "Energy level should be updated"


def test_json_roundtrip_for_all_mood_states(db):
    """Verify JSON serialization roundtrip for all mood value types."""
    ums = UserModelStore(db, event_bus=None)

    test_cases = [
        (None, {}),  # None serializes as empty dict
        ({}, {}),    # Empty dict stays empty dict
        ({"energy_level": 0.5}, {"energy_level": 0.5}),  # Partial mood
        (
            {
                "energy_level": 1.0,
                "stress_level": 0.0,
                "emotional_valence": 0.5,
            },
            {
                "energy_level": 1.0,
                "stress_level": 0.0,
                "emotional_valence": 0.5,
            },
        ),  # Full mood
    ]

    for input_mood, expected_output in test_cases:
        episode_id = str(uuid.uuid4())
        episode = {
            "id": episode_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": f"test-roundtrip-{episode_id}",
            "inferred_mood": input_mood,
            "interaction_type": "test",
            "content_summary": f"Testing mood: {input_mood}",
        }

        ums.store_episode(episode)

        with db.get_connection("user_model") as conn:
            cursor = conn.execute(
                "SELECT inferred_mood FROM episodes WHERE id = ?",
                (episode_id,)
            )
            row = cursor.fetchone()
            stored_mood = json.loads(row[0])

            assert stored_mood == expected_output, \
                f"Input {input_mood} should roundtrip to {expected_output}, got {stored_mood}"


def test_contacts_topics_entities_none_handling(db):
    """Verify lists (contacts, topics, entities) also handle None correctly."""
    ums = UserModelStore(db, event_bus=None)

    episode_id = str(uuid.uuid4())
    episode = {
        "id": episode_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_id": "test-lists-none",
        "inferred_mood": None,
        "contacts_involved": None,  # Should become []
        "topics": None,             # Should become []
        "entities": None,           # Should become []
        "interaction_type": "test",
        "content_summary": "Testing None lists",
    }

    ums.store_episode(episode)

    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT contacts_involved, topics, entities FROM episodes WHERE id = ?",
            (episode_id,)
        )
        row = cursor.fetchone()

        assert json.loads(row[0]) == [], "None contacts should become []"
        assert json.loads(row[1]) == [], "None topics should become []"
        assert json.loads(row[2]) == [], "None entities should become []"
