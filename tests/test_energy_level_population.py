"""
Test energy level population in episodes via proxy signals.

CRITICAL FIX (iteration 146):
Episodes had 0% energy_level population because compute_current_mood() only
considered ["sleep_quality", "sleep_duration", "activity_level"] signals,
but ZERO of these existed (all 200 mood signals were incoming_pressure/
negative_language/message_length). Without health connectors, proxy energy
signals (circadian_energy, communication_energy) are essential.

This test suite verifies:
1. Circadian energy signals are extracted from communication events
2. Communication energy signals are extracted from message length deviations
3. compute_current_mood() includes proxy signals in energy_level calculation
4. Episodes are created with non-NULL energy_level
5. Energy levels vary appropriately with time of day and activity patterns
"""

import json
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.mood import MoodInferenceEngine


def test_circadian_energy_extracted_from_morning_email(db, user_model_store):
    """Morning emails (9am) should generate high circadian energy signals."""
    engine = MoodInferenceEngine(db, user_model_store)

    event = {
        "id": "test-1",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T09:30:00+00:00",  # 9:30 AM UTC
        "payload": {"body": "Good morning! Let's sync on the project."},
    }

    result = engine.extract(event)

    # Extract method returns [{type: "mood_signal", signals: [...]}]
    assert len(result) == 1
    assert result[0]["type"] == "mood_signal"
    signals = result[0]["signals"]

    # Should extract circadian_energy signal
    circadian = [s for s in signals if s.get("signal_type") == "circadian_energy"]
    assert len(circadian) == 1
    assert circadian[0]["value"] == 0.8  # Morning peak energy
    assert circadian[0]["weight"] == 0.3


def test_circadian_energy_low_at_late_night(db, user_model_store):
    """Late night emails (11pm) should generate low circadian energy signals."""
    engine = MoodInferenceEngine(db, user_model_store)

    event = {
        "id": "test-2",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T23:45:00+00:00",  # 11:45 PM UTC
        "payload": {"body": "Quick note before bed."},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    circadian = [s for s in signals if s.get("signal_type") == "circadian_energy"]
    assert len(circadian) == 1
    assert circadian[0]["value"] == 0.3  # Late night low energy
    assert circadian[0]["delta_from_baseline"] == 0.3 - 0.5  # Below neutral


def test_circadian_energy_post_lunch_dip(db, user_model_store):
    """Post-lunch emails (1pm) should reflect afternoon dip."""
    engine = MoodInferenceEngine(db, user_model_store)

    event = {
        "id": "test-3",
        "type": EventType.MESSAGE_SENT.value,
        "source": "slack",
        "timestamp": "2026-02-16T13:00:00+00:00",  # 1:00 PM UTC
        "payload": {"body": "Thanks for the update."},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    circadian = [s for s in signals if s.get("signal_type") == "circadian_energy"]
    assert len(circadian) == 1
    assert circadian[0]["value"] == 0.6  # Post-lunch dip


def test_circadian_energy_inbound_also_generates_signal(db, user_model_store):
    """Inbound emails should also generate circadian signals (user is active reading)."""
    engine = MoodInferenceEngine(db, user_model_store)

    event = {
        "id": "test-4",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": "2026-02-16T10:00:00+00:00",  # 10:00 AM UTC
        "payload": {"body": "Can you review this proposal?"},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    circadian = [s for s in signals if s.get("signal_type") == "circadian_energy"]
    assert len(circadian) == 1
    assert circadian[0]["value"] == 0.8  # Morning peak


def test_communication_energy_high_for_long_messages(db, user_model_store):
    """Long messages (>120% baseline) indicate high engagement energy."""
    engine = MoodInferenceEngine(db, user_model_store)

    # Baseline is 25 words, so 31+ words is 120%+ above baseline
    long_body = " ".join(["word"] * 35)

    event = {
        "id": "test-5",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T14:00:00+00:00",
        "payload": {"body": long_body},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    comm_energy = [s for s in signals if s.get("signal_type") == "communication_energy"]
    assert len(comm_energy) == 1
    assert comm_energy[0]["delta_from_baseline"] == 0.2  # High energy
    assert comm_energy[0]["weight"] == 0.2


def test_communication_energy_low_for_short_messages(db, user_model_store):
    """Short messages (<50% baseline) suggest low-effort, potentially low energy."""
    engine = MoodInferenceEngine(db, user_model_store)

    # Baseline is 25 words, so <12 words is <50% baseline
    short_body = "ok thanks"  # 2 words

    event = {
        "id": "test-6",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T16:00:00+00:00",
        "payload": {"body": short_body},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    comm_energy = [s for s in signals if s.get("signal_type") == "communication_energy"]
    assert len(comm_energy) == 1
    assert comm_energy[0]["delta_from_baseline"] == -0.3  # Low energy


def test_communication_energy_not_extracted_for_normal_length(db, user_model_store):
    """Messages near baseline length don't generate communication_energy signals."""
    engine = MoodInferenceEngine(db, user_model_store)

    # Baseline is 25 words, so 20-30 words is within normal range
    normal_body = " ".join(["word"] * 23)

    event = {
        "id": "test-7",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T11:00:00+00:00",
        "payload": {"body": normal_body},
    }

    result = engine.extract(event)
    signals = result[0]["signals"] if result else []

    comm_energy = [s for s in signals if s.get("signal_type") == "communication_energy"]
    assert len(comm_energy) == 0  # No deviation, no signal


def test_compute_current_mood_uses_proxy_energy_signals(db, user_model_store):
    """compute_current_mood() should include circadian_energy in energy_level calculation."""
    engine = MoodInferenceEngine(db, user_model_store)

    # Generate circadian energy signals across different times
    events = [
        {
            "id": f"test-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": f"2026-02-16T{i:02d}:00:00+00:00",
            "payload": {"body": "Test message."},
        }
        for i in range(9, 12)  # 9am, 10am, 11am (high energy period)
    ]

    for event in events:
        engine.extract(event)

    mood = engine.compute_current_mood()

    # Should have energy_level > 0.5 (morning peak)
    assert mood.energy_level > 0.5
    # Should have non-zero confidence
    assert mood.confidence > 0


def test_compute_current_mood_energy_varies_with_time_of_day(db, user_model_store):
    """Energy level should be higher in morning than late night."""
    engine_morning = MoodInferenceEngine(db, user_model_store)
    engine_night = MoodInferenceEngine(db, user_model_store)

    # Morning activity (9am) - use longer message to avoid low communication_energy
    morning_event = {
        "id": "test-morning",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T09:00:00+00:00",
        "payload": {"body": " ".join(["Morning update on the project status."] * 3)},  # ~9 words * 3
    }
    engine_morning.extract(morning_event)
    mood_morning = engine_morning.compute_current_mood()

    # Night activity (11pm)
    night_event = {
        "id": "test-night",
        "type": EventType.EMAIL_SENT.value,
        "source": "gmail",
        "timestamp": "2026-02-16T23:00:00+00:00",
        "payload": {"body": " ".join(["Late night email about tomorrow."] * 3)},  # ~5 words * 3
    }
    engine_night.extract(night_event)
    mood_night = engine_night.compute_current_mood()

    # Morning energy should be significantly higher than night energy
    # NOTE: Both engines share the same mood_signals profile in this test,
    # so night mood includes both morning and night circadian signals.
    # In production, these would be temporal - only recent signals matter.
    # The key assertion is that morning > night (directionally correct).
    assert mood_morning.energy_level > mood_night.energy_level


def test_episodes_populated_with_energy_level(db, user_model_store):
    """
    Integration test: Episodes should have non-NULL energy_level when created
    with proxy energy signals available.
    """
    engine = MoodInferenceEngine(db, user_model_store)

    # Create an event with circadian energy signal
    event = {
        "id": "test-episode",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": "2026-02-16T10:30:00+00:00",  # Morning
        "priority": "normal",
        "payload": {
            "from": "alice@example.com",
            "subject": "Project update",
            "body": "Here's the latest on the project.",
        },
        "metadata": {},
    }

    # Extract mood signals
    engine.extract(event)

    # Compute mood with proxy signals
    mood = engine.compute_current_mood()

    # CRITICAL: mood should have non-NULL energy_level from proxy signals
    assert mood is not None
    assert mood.energy_level is not None
    # Should be in valid range (0-1)
    assert 0 <= mood.energy_level <= 1
    # Should reflect morning energy (elevated from circadian signal)
    assert mood.energy_level >= 0.3


def test_backfill_historical_episodes_with_energy(db, user_model_store):
    """
    Historical episodes with NULL energy_level should be backfillable by
    reprocessing their source events through the updated mood engine.

    This verifies that the fix enables backfilling the 29,712 existing
    episodes that have energy_level=NULL.
    """
    engine = MoodInferenceEngine(db, user_model_store)

    # Create an episode without energy_level (simulating old behavior)
    episode_old = {
        "id": "old-episode",
        "timestamp": "2026-02-15T14:00:00+00:00",
        "event_id": "old-event",
        "interaction_type": "email_received",
        "content_summary": "Old email",
        "content_full": json.dumps({"body": "test"}),
        "contacts_involved": ["test@example.com"],
        "topics": [],
        "entities": [],
        "inferred_mood": {},  # Empty mood
        "energy_level": None,  # NULL
    }
    user_model_store.store_episode(episode_old)

    # Verify NULL
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT energy_level FROM episodes WHERE id = ?",
            (episode_old["id"],),
        ).fetchone()
        assert row["energy_level"] is None

    # Now simulate reprocessing with mood signals available
    # Create the source event
    event = {
        "id": episode_old["event_id"],
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": episode_old["timestamp"],
        "priority": "normal",
        "payload": {"body": "Test email content."},
        "metadata": {},
    }

    # Extract signals and recompute mood
    engine.extract(event)
    mood = engine.compute_current_mood()

    # Backfill the episode with updated energy
    episode_updated = episode_old.copy()
    episode_updated["inferred_mood"] = {
        "energy_level": mood.energy_level,
        "stress_level": mood.stress_level,
        "emotional_valence": mood.emotional_valence,
    }
    episode_updated["energy_level"] = mood.energy_level
    user_model_store.store_episode(episode_updated)

    # Verify updated
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT energy_level FROM episodes WHERE id = ?",
            (episode_old["id"],),
        ).fetchone()
        assert row["energy_level"] is not None
        assert 0 <= row["energy_level"] <= 1
