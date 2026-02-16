"""
Tests for DecisionExtractor — tracks decision-making patterns.

Verifies that the extractor correctly:
1. Tracks decision speed from task creation → completion
2. Detects delegation patterns in outbound messages
3. Tracks commitment planning horizons from calendar events
4. Updates DecisionProfile with aggregated signals
5. Classifies domains correctly
"""

import pytest
from datetime import datetime, timedelta, timezone

from models.core import EventType
from services.signal_extractor.decision import DecisionExtractor


def test_decision_extractor_processes_relevant_events(db, user_model_store):
    """DecisionExtractor should process task, message, and calendar events."""
    extractor = DecisionExtractor(db, user_model_store)

    # Should process task completions
    assert extractor.can_process({"type": EventType.TASK_COMPLETED.value})

    # Should process task creation
    assert extractor.can_process({"type": EventType.TASK_CREATED.value})

    # Should process outbound messages
    assert extractor.can_process({"type": EventType.EMAIL_SENT.value})
    assert extractor.can_process({"type": EventType.MESSAGE_SENT.value})

    # Should process calendar event creation
    assert extractor.can_process({"type": EventType.CALENDAR_EVENT_CREATED.value})

    # Should NOT process inbound events
    assert not extractor.can_process({"type": EventType.EMAIL_RECEIVED.value})
    assert not extractor.can_process({"type": EventType.MESSAGE_RECEIVED.value})


def test_track_task_decision_speed_immediate(db, user_model_store, event_store):
    """Track immediate task completion (< 1 hour = impulsive decision)."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    task_id = "task_123"

    # Create task
    creation_event = {
        "id": "event_1",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": now.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
            "title": "Quick email reply",
            "domain": "work",
        },
        "metadata": {},
    }
    event_store.store_event(creation_event)

    # Complete task 30 minutes later
    completion_time = now + timedelta(minutes=30)
    completion_event = {
        "id": "event_2",
        "type": EventType.TASK_COMPLETED.value,
        "source": "test",
        "timestamp": completion_time.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
        },
        "metadata": {},
    }

    signals = extractor.extract(completion_event)

    # Should produce decision_speed signal
    assert len(signals) == 1
    assert signals[0]["type"] == "decision_speed"
    assert signals[0]["domain"] == "work"
    assert signals[0]["speed_category"] == "immediate"
    assert signals[0]["decision_time_seconds"] == pytest.approx(1800, abs=1)  # 30 min


def test_track_task_decision_speed_multi_day(db, user_model_store, event_store):
    """Track multi-day task completion (> 24 hours = deliberative decision)."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    task_id = "task_456"

    # Create task
    creation_event = {
        "id": "event_1",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": now.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
            "title": "Research new framework",
            "domain": "tech",
        },
        "metadata": {},
    }
    event_store.store_event(creation_event)

    # Complete task 3 days later
    completion_time = now + timedelta(days=3)
    completion_event = {
        "id": "event_2",
        "type": EventType.TASK_COMPLETED.value,
        "source": "test",
        "timestamp": completion_time.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
        },
        "metadata": {},
    }

    signals = extractor.extract(completion_event)

    # Should produce decision_speed signal
    assert len(signals) == 1
    assert signals[0]["type"] == "decision_speed"
    assert signals[0]["domain"] == "tech"
    assert signals[0]["speed_category"] == "multi_day"
    assert signals[0]["decision_time_seconds"] == pytest.approx(259200, abs=10)  # 3 days


def test_detect_delegation_full(db, user_model_store):
    """Detect full delegation patterns ('you decide', 'whatever you want')."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "content": "I don't really care where we eat, you decide!",
            "recipient": "partner",
        },
    }

    signals = extractor.extract(event)

    # Should detect delegation
    assert len(signals) == 1
    assert signals[0]["type"] == "delegation_pattern"
    assert signals[0]["delegation_type"] == "full"
    assert signals[0]["recipient"] == "partner"
    assert signals[0]["hour"] == now.hour


def test_detect_delegation_seeking_input(db, user_model_store):
    """Detect opinion-seeking patterns ('what do you think?', 'should I?')."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    event = {
        "type": EventType.EMAIL_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "body": "What do you think about this approach? Should I proceed or wait?",
            "to": "colleague@example.com",
        },
    }

    signals = extractor.extract(event)

    # Should detect opinion-seeking (soft delegation)
    assert len(signals) == 1
    assert signals[0]["type"] == "delegation_pattern"
    assert signals[0]["delegation_type"] == "seeking_input"
    assert signals[0]["recipient"] == "colleague@example.com"


def test_no_delegation_in_decisive_message(db, user_model_store):
    """Decisive messages should NOT trigger delegation detection."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "content": "Let's do the 7pm reservation at Chez Nous. See you there!",
            "recipient": "friend",
        },
    }

    signals = extractor.extract(event)

    # Should NOT detect delegation
    assert len(signals) == 0


def test_track_commitment_immediate(db, user_model_store):
    """Track immediate calendar commitment (< 1 hour = spontaneous)."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event_start = now + timedelta(minutes=30)

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "start_time": event_start.isoformat(),
            "summary": "Quick coffee catch-up",
        },
    }

    signals = extractor.extract(event)

    # Should detect immediate commitment
    assert len(signals) == 1
    assert signals[0]["type"] == "commitment_pattern"
    assert signals[0]["horizon_category"] == "immediate"
    assert signals[0]["domain"] == "social"  # "coffee" → social


def test_track_commitment_long_term(db, user_model_store):
    """Track long-term calendar commitment (> 1 week = deliberative)."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event_start = now + timedelta(weeks=3)

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "start_time": event_start.isoformat(),
            "summary": "Team quarterly review meeting",
        },
    }

    signals = extractor.extract(event)

    # Should detect long-term commitment
    assert len(signals) == 1
    assert signals[0]["type"] == "commitment_pattern"
    assert signals[0]["horizon_category"] == "long_term"
    assert signals[0]["domain"] == "work"  # "meeting" → work


def test_classify_event_domain_work(db, user_model_store):
    """Calendar events with work keywords should be classified as 'work'."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_event_domain("Team standup meeting") == "work"
    assert extractor._classify_event_domain("Client call review") == "work"
    assert extractor._classify_event_domain("Weekly sync") == "work"


def test_classify_event_domain_social(db, user_model_store):
    """Calendar events with social keywords should be classified as 'social'."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_event_domain("Dinner with friends") == "social"
    assert extractor._classify_event_domain("Coffee hangout") == "social"
    assert extractor._classify_event_domain("Birthday party") == "social"


def test_classify_event_domain_health(db, user_model_store):
    """Calendar events with health keywords should be classified as 'health'."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_event_domain("Dentist appointment") == "health"
    assert extractor._classify_event_domain("Gym workout session") == "health"
    assert extractor._classify_event_domain("Therapy session") == "health"


def test_classify_event_domain_finance(db, user_model_store):
    """Calendar events with finance keywords should be classified as 'finance'."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_event_domain("Bank meeting") == "finance"
    assert extractor._classify_event_domain("Tax preparation") == "finance"
    assert extractor._classify_event_domain("Budget review") == "finance"


def test_classify_event_domain_general(db, user_model_store):
    """Unclassifiable events should default to 'general'."""
    extractor = DecisionExtractor(db, user_model_store)

    assert extractor._classify_event_domain("Random event") == "general"
    assert extractor._classify_event_domain("") == "general"


def test_profile_update_decision_speed(db, user_model_store, event_store):
    """DecisionProfile should aggregate decision speed by domain."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    task_id = "task_speed_test"

    # Create task
    creation_event = {
        "id": "event_create",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": now.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
            "title": "Write report",
            "domain": "work",
        },
        "metadata": {},
    }
    event_store.store_event(creation_event)

    # Complete task 2 hours later
    completion_time = now + timedelta(hours=2)
    completion_event = {
        "id": "event_complete",
        "type": EventType.TASK_COMPLETED.value,
        "source": "test",
        "timestamp": completion_time.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
        },
        "metadata": {},
    }

    extractor.extract(completion_event)

    # Verify profile was updated
    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert "decision_speed_by_domain" in profile["data"]
    assert "work" in profile["data"]["decision_speed_by_domain"]
    # Should be approximately 7200 seconds (2 hours)
    assert profile["data"]["decision_speed_by_domain"]["work"] == pytest.approx(7200, abs=10)


def test_profile_update_risk_tolerance(db, user_model_store):
    """DecisionProfile should track risk tolerance via planning horizon."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)
    event_start = now + timedelta(hours=2)  # Short horizon = higher risk

    event = {
        "type": EventType.CALENDAR_EVENT_CREATED.value,
        "timestamp": now.isoformat(),
        "payload": {
            "start_time": event_start.isoformat(),
            "summary": "Last-minute dinner",
        },
    }

    extractor.extract(event)

    # Verify profile was updated
    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert "risk_tolerance_by_domain" in profile["data"]
    assert "social" in profile["data"]["risk_tolerance_by_domain"]
    # Short horizon should map to higher risk tolerance (closer to 1.0)
    assert profile["data"]["risk_tolerance_by_domain"]["social"] > 0.5


def test_profile_update_decision_fatigue(db, user_model_store):
    """DecisionProfile should detect decision fatigue patterns (late-night delegation)."""
    extractor = DecisionExtractor(db, user_model_store)

    # Send delegation message at 10pm (fatigue hour)
    late_time = datetime.now(timezone.utc).replace(hour=22, minute=0, second=0)

    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": late_time.isoformat(),
        "payload": {
            "content": "Whatever you want to do is fine with me",
            "recipient": "partner",
        },
    }

    extractor.extract(event)

    # Verify profile detected fatigue hour
    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert "fatigue_time_of_day" in profile["data"]
    assert profile["data"]["fatigue_time_of_day"] == 22


def test_profile_exponential_moving_average(db, user_model_store, event_store):
    """DecisionProfile should use EMA to smooth decision speed updates."""
    extractor = DecisionExtractor(db, user_model_store)

    now = datetime.now(timezone.utc)

    # First task: 1 hour decision time
    task_id_1 = "task_ema_1"
    creation_1 = {
        "id": "event_create_1",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": now.isoformat(),
        "priority": "normal",
        "payload": {"task_id": task_id_1, "title": "Task 1", "domain": "work"},
        "metadata": {},
    }
    event_store.store_event(creation_1)

    completion_1 = {
        "type": EventType.TASK_COMPLETED.value,
        "timestamp": (now + timedelta(hours=1)).isoformat(),
        "payload": {"task_id": task_id_1},
    }
    extractor.extract(completion_1)

    profile_after_1 = user_model_store.get_signal_profile("decision")
    first_speed = profile_after_1["data"]["decision_speed_by_domain"]["work"]
    assert first_speed == pytest.approx(3600, abs=10)  # 1 hour

    # Second task: 3 hours decision time
    task_id_2 = "task_ema_2"
    creation_2 = {
        "id": "event_create_2",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "priority": "normal",
        "payload": {"task_id": task_id_2, "title": "Task 2", "domain": "work"},
        "metadata": {},
    }
    event_store.store_event(creation_2)

    completion_2 = {
        "type": EventType.TASK_COMPLETED.value,
        "timestamp": (now + timedelta(hours=5)).isoformat(),
        "payload": {"task_id": task_id_2},
    }
    extractor.extract(completion_2)

    profile_after_2 = user_model_store.get_signal_profile("decision")
    second_speed = profile_after_2["data"]["decision_speed_by_domain"]["work"]

    # Should be EMA: 0.7 * 3600 + 0.3 * 10800 = 5760
    assert second_speed == pytest.approx(5760, abs=50)


def test_extractor_fail_open(db, user_model_store):
    """DecisionExtractor should fail-open on malformed events."""
    extractor = DecisionExtractor(db, user_model_store)

    # Malformed event (missing timestamp)
    malformed_event = {
        "type": EventType.TASK_COMPLETED.value,
        "payload": {"task_id": "missing_timestamp"},
    }

    # Should not crash, just return empty signals
    signals = extractor.extract(malformed_event)
    assert signals == []
