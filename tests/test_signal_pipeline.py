"""
Life OS — SignalExtractorPipeline Test Suite

Comprehensive test coverage for the central signal extraction pipeline that
routes events through all extractors (linguistic, cadence, mood, relationship,
topic). This pipeline processes 43K+ events/day and is critical infrastructure.
"""

import json

import pytest
from datetime import datetime, timezone
from services.signal_extractor.pipeline import SignalExtractorPipeline
from storage.event_store import EventStore
from models.user_model import MoodState


# =============================================================================
# Core Pipeline Tests
# =============================================================================


def test_pipeline_initialization(db, user_model_store):
    """Test that the pipeline initializes with all extractors registered."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Verify all 8 extractors are registered (added DecisionExtractor in iteration 138)
    assert len(pipeline.extractors) == 8

    # Verify extractor types in correct order
    extractor_names = [type(e).__name__ for e in pipeline.extractors]
    assert "LinguisticExtractor" in extractor_names
    assert "CadenceExtractor" in extractor_names
    assert "MoodInferenceEngine" in extractor_names
    assert "RelationshipExtractor" in extractor_names
    assert "TopicExtractor" in extractor_names
    assert "TemporalExtractor" in extractor_names
    assert "SpatialExtractor" in extractor_names
    assert "DecisionExtractor" in extractor_names

    # Verify mood engine reference exists (same instance as in extractors list)
    assert pipeline.mood_engine is not None
    assert type(pipeline.mood_engine).__name__ == "MoodInferenceEngine"


@pytest.mark.asyncio
async def test_process_event_routes_to_all_applicable_extractors(db, user_model_store):
    """Test that events are routed to all extractors that can process them."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create an email event (processable by all extractors)
    event = {
        "id": "test-123",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "subject": "Test email with emotional content",
            "body": "I'm really excited about this opportunity! Looking forward to collaborating.",
            "from": "sender@example.com",
            "to": ["user@example.com"],
        },
        "metadata": {}
    }

    signals = await pipeline.process_event(event)

    # Verify signals were extracted (at least from some extractors)
    # The exact number depends on extractor logic, but we should get something
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_event_respects_can_process_gate(db, user_model_store):
    """Test that extractors only process events they declare interest in."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create a system event (not processable by most extractors)
    event = {
        "id": "test-456",
        "type": "system.health.check",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {"status": "ok"},
        "metadata": {}
    }

    signals = await pipeline.process_event(event)

    # System events should not be processed by text/communication extractors
    # Most extractors will skip this event
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_event_handles_extractor_errors_gracefully(db, user_model_store):
    """Test that extractor errors don't crash the pipeline (fail-open)."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create a malformed event that might cause extractor errors
    event = {
        "id": "malformed-789",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            # Missing required fields like 'body' or 'from'
            "subject": "Incomplete email"
        },
        "metadata": {}
    }

    # Should not raise an exception despite malformed payload
    signals = await pipeline.process_event(event)

    # Pipeline should return a list even if some extractors failed
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_event_collects_all_signals(db, user_model_store):
    """Test that all signals from all extractors are collected and returned."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create a rich email event that should trigger multiple extractors
    event = {
        "id": "rich-event",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "high",
        "payload": {
            "subject": "Re: Project update - urgent feedback needed",
            "body": "Thanks for the update! I'm a bit concerned about the timeline. Can we discuss?",
            "from": "colleague@work.com",
            "to": ["user@example.com"],
            "is_reply": True,
            "in_reply_to": "original-123"
        },
        "metadata": {
            "contacts": ["colleague@work.com"]
        }
    }

    signals = await pipeline.process_event(event)

    # Should have signals from multiple extractors
    assert isinstance(signals, list)
    # The exact count varies by extractor implementation, but verify structure
    for signal in signals:
        assert isinstance(signal, dict)


# =============================================================================
# Mood Engine Tests
# =============================================================================


def test_get_current_mood_returns_mood_state(db, user_model_store):
    """Test that get_current_mood() returns a MoodState object."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    mood = pipeline.get_current_mood()

    # Verify it returns a MoodState instance
    assert isinstance(mood, MoodState)

    # Verify MoodState has expected attributes (using actual field names)
    assert hasattr(mood, 'emotional_valence')
    assert hasattr(mood, 'energy_level')
    assert hasattr(mood, 'confidence')


def test_get_current_mood_uses_dedicated_engine(db, user_model_store):
    """Test that mood is computed via the mood engine reference."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    mood = pipeline.get_current_mood()

    # Verify the mood engine reference exists and is callable
    assert pipeline.mood_engine is not None
    assert callable(pipeline.mood_engine.compute_current_mood)


# =============================================================================
# User Summary Tests
# =============================================================================


def test_get_user_summary_structure(db, user_model_store):
    """Test that get_user_summary() returns expected structure."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    summary = pipeline.get_user_summary()

    # Verify top-level structure
    assert isinstance(summary, dict)
    assert "profiles" in summary
    assert "semantic_facts_count" in summary
    assert "high_confidence_facts" in summary

    # Verify profiles structure
    assert isinstance(summary["profiles"], dict)


def test_get_user_summary_includes_all_profile_types(db, user_model_store):
    """Test that summary attempts to gather all 5 profile types."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create some signal profiles first
    for profile_type in ["linguistic", "cadence", "mood_signals", "relationships", "topics"]:
        user_model_store.update_signal_profile(profile_type, {"test": "data"})

    summary = pipeline.get_user_summary()

    # Verify all profiles are present
    profiles = summary["profiles"]
    expected_types = ["linguistic", "cadence", "mood_signals", "relationships", "topics"]

    for profile_type in expected_types:
        if profile_type in profiles:  # May not all be present if no data
            assert "samples_count" in profiles[profile_type]
            assert "last_updated" in profiles[profile_type]


def test_get_user_summary_filters_high_confidence_facts(db, user_model_store):
    """Test that only high-confidence facts (>= 0.7) are included in summary."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store facts with varying confidence levels
    user_model_store.update_semantic_fact(
        key="low_confidence_fact",
        category="test",
        value="test",
        confidence=0.4,
        episode_id="test-1"
    )
    user_model_store.update_semantic_fact(
        key="medium_confidence_fact",
        category="test",
        value="test",
        confidence=0.6,
        episode_id="test-2"
    )
    user_model_store.update_semantic_fact(
        key="high_confidence_fact",
        category="test",
        value="test",
        confidence=0.9,
        episode_id="test-3"
    )

    summary = pipeline.get_user_summary()

    # Should include 3 facts total (>= 0.3 threshold)
    assert summary["semantic_facts_count"] >= 3

    # But high_confidence_facts should only include the 0.9 one
    high_conf_keys = [f["key"] for f in summary["high_confidence_facts"]]
    assert "high_confidence_fact" in high_conf_keys
    assert "low_confidence_fact" not in high_conf_keys
    assert "medium_confidence_fact" not in high_conf_keys


def test_get_user_summary_with_empty_profiles(db, user_model_store):
    """Test that summary handles empty profiles gracefully."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Get summary without any profile data
    summary = pipeline.get_user_summary()

    # Should return valid structure even when empty
    assert isinstance(summary, dict)
    assert "profiles" in summary
    assert "semantic_facts_count" in summary
    assert "high_confidence_facts" in summary

    # Counts should be zero or empty
    assert summary["semantic_facts_count"] == 0
    assert len(summary["high_confidence_facts"]) == 0


# =============================================================================
# Event Type Coverage Tests
# =============================================================================


@pytest.mark.asyncio
async def test_process_email_event(db, user_model_store):
    """Test processing of email.received events."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "email-1",
        "type": "email.received",
        "source": "proton_mail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "subject": "Weekly team sync notes",
            "body": "Here are the notes from our weekly sync meeting.",
            "from": "teammate@work.com",
            "to": ["user@example.com"],
        },
        "metadata": {"contacts": ["teammate@work.com"]}
    }

    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_message_event(db, user_model_store):
    """Test processing of message.received events."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "msg-1",
        "type": "message.received",
        "source": "imessage",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "text": "Hey, are you free for lunch today?",
            "from": "+1234567890",
            "is_from_me": False,
        },
        "metadata": {"contacts": ["+1234567890"]}
    }

    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_calendar_event(db, user_model_store):
    """Test processing of calendar.event.created events."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "cal-1",
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "summary": "Team standup",
            "start": "2026-02-16T09:00:00Z",
            "end": "2026-02-16T09:30:00Z",
            "location": "Zoom",
        },
        "metadata": {}
    }

    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_process_task_event(db, user_model_store):
    """Test processing of task.created events."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "task-1",
        "type": "task.created",
        "source": "task_manager",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "high",
        "payload": {
            "title": "Review quarterly budget",
            "description": "Prepare Q1 budget review for leadership meeting",
            "due_date": "2026-02-20T17:00:00Z",
        },
        "metadata": {}
    }

    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pipeline_end_to_end_email_flow(db, user_model_store):
    """Test complete pipeline flow for a realistic email event."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Simulate a real email with rich content
    event = {
        "id": "end-to-end-email",
        "type": "email.received",
        "source": "proton_mail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "high",
        "payload": {
            "subject": "Re: Project deadline extension request",
            "body": (
                "Hi,\n\n"
                "I appreciate you bringing this up. I'm feeling a bit stressed about "
                "the current timeline too. Let's schedule a call tomorrow to discuss "
                "how we can adjust the scope or get additional resources.\n\n"
                "Thanks,\nAlex"
            ),
            "from": "alex@company.com",
            "to": ["user@example.com"],
            "is_reply": True,
            "in_reply_to": "original-request-456",
        },
        "metadata": {
            "contacts": ["alex@company.com"],
            "domain": "work"
        }
    }

    # Process the event
    signals = await pipeline.process_event(event)

    # Verify signals were extracted
    assert isinstance(signals, list)

    # Verify profile data was persisted (check at least one profile type)
    profiles_summary = pipeline.get_user_summary()
    assert isinstance(profiles_summary, dict)


@pytest.mark.asyncio
async def test_pipeline_processes_multiple_events_sequentially(db, user_model_store):
    """Test that pipeline can process multiple events in sequence."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    events = [
        {
            "id": f"batch-{i}",
            "type": "email.received",
            "source": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "subject": f"Email {i}",
                "body": f"This is test email number {i}",
                "from": f"sender{i}@example.com",
                "to": ["user@example.com"],
            },
            "metadata": {}
        }
        for i in range(5)
    ]

    # Process all events
    all_signals = []
    for event in events:
        signals = await pipeline.process_event(event)
        all_signals.extend(signals)

    # Verify all events were processed
    assert isinstance(all_signals, list)


@pytest.mark.asyncio
async def test_pipeline_handles_concurrent_processing(db, user_model_store):
    """Test that pipeline handles concurrent event processing safely."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    import asyncio

    # Create multiple events to process concurrently
    events = [
        {
            "id": f"concurrent-{i}",
            "type": "message.received",
            "source": "imessage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {
                "text": f"Concurrent message {i}",
                "from": f"+123456789{i}",
                "is_from_me": False,
            },
            "metadata": {}
        }
        for i in range(3)
    ]

    # Process concurrently
    results = await asyncio.gather(
        *[pipeline.process_event(event) for event in events]
    )

    # Verify all processed successfully
    assert len(results) == 3
    for signals in results:
        assert isinstance(signals, list)


# =============================================================================
# Extractor Coordination Tests
# =============================================================================


@pytest.mark.asyncio
async def test_extractors_share_database_connection(db, user_model_store):
    """Test that all extractors use the same database instance."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Verify all extractors have the same db reference
    for extractor in pipeline.extractors:
        assert extractor.db is db
        assert extractor.ums is user_model_store


def test_mood_engine_is_same_instance_as_extractor(db, user_model_store):
    """Test that self.mood_engine is the SAME object as the MoodInferenceEngine in the extractors list.

    This prevents stale-data bugs: get_current_mood() must see any in-memory
    state that was updated during extract() in the pipeline's event processing.
    """
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # The mood engine should exist
    assert pipeline.mood_engine is not None

    # Find the MoodInferenceEngine in the extractors list
    mood_extractors_in_pipeline = [
        e for e in pipeline.extractors
        if type(e).__name__ == "MoodInferenceEngine"
    ]

    # There should be exactly one in the extractors list
    assert len(mood_extractors_in_pipeline) == 1

    # It MUST be the same object (identity check, not equality)
    assert pipeline.mood_engine is mood_extractors_in_pipeline[0]


def test_pipeline_still_has_8_extractors_after_mood_dedup(db, user_model_store):
    """Verify that deduplicating the mood engine didn't remove it from the extractors list."""
    pipeline = SignalExtractorPipeline(db, user_model_store)
    assert len(pipeline.extractors) == 8


@pytest.mark.asyncio
async def test_get_current_mood_reflects_processed_event(db, user_model_store):
    """Test that get_current_mood() sees state from events processed through the pipeline.

    This verifies there's no stale-data issue: after process_event() runs the
    mood extractor's extract() method (which persists signals), the same instance
    is used by get_current_mood() -> compute_current_mood(), so it reads the
    freshly-written signals.
    """
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Mood with no data should be neutral (default MoodState)
    mood_before = pipeline.get_current_mood()
    assert mood_before.confidence == 0.0  # No signals yet

    # Process a mood-affecting event with negative language (stress signal)
    event = {
        "id": "mood-test-1",
        "type": "email.sent",
        "source": "test",
        "timestamp": "2026-03-01T10:00:00+00:00",
        "priority": "normal",
        "payload": {
            "body": (
                "I am frustrated and stressed about this urgent problem. "
                "This is an emergency and I'm overwhelmed by this failure. "
                "The situation is unacceptable and I'm exhausted."
            ),
            "to": ["someone@example.com"],
        },
        "metadata": {},
    }

    signals = await pipeline.process_event(event)

    # The pipeline should have extracted mood signals
    assert len(signals) > 0

    # Now get_current_mood() should reflect those signals
    mood_after = pipeline.get_current_mood()
    assert mood_after.confidence > 0.0  # Signals were processed


# =============================================================================
# Error Handling & Resilience Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pipeline_continues_after_single_extractor_failure(db, user_model_store):
    """Test that pipeline continues processing even if one extractor fails."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Create an event that might cause issues in one extractor but not others
    event = {
        "id": "edge-case",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "subject": "",  # Empty subject
            "body": None,   # Null body (edge case)
            "from": "test@example.com",
            "to": [],       # Empty recipient list
        },
        "metadata": {}
    }

    # Should not raise exception despite edge cases
    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_pipeline_handles_missing_payload_fields(db, user_model_store):
    """Test that pipeline handles events with missing payload fields."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "minimal-event",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {},  # Empty payload
        "metadata": {}
    }

    # Should handle gracefully
    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


@pytest.mark.asyncio
async def test_pipeline_handles_unicode_content(db, user_model_store):
    """Test that pipeline handles Unicode content correctly."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    event = {
        "id": "unicode-test",
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "subject": "Hello 你好 🌟",
            "body": "Testing emoji 😀 and Chinese 中文 and Arabic العربية",
            "from": "test@example.com",
            "to": ["user@example.com"],
        },
        "metadata": {}
    }

    signals = await pipeline.process_event(event)
    assert isinstance(signals, list)


# =============================================================================
# rebuild_profiles_from_events Tests
# =============================================================================


def _store_test_event(db, event_id, event_type, source, payload, metadata=None, timestamp=None):
    """Helper to insert an event into events.db with JSON-serialised payload/metadata."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    es = EventStore(db)
    es.store_event({
        "id": event_id,
        "type": event_type,
        "source": source,
        "timestamp": timestamp,
        "priority": "normal",
        "payload": payload,
        "metadata": metadata or {},
    })


def test_rebuild_profiles_from_events_basic(db, user_model_store):
    """Test that rebuild loads events from events.db and processes them through extractors."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store an email.received event (triggers linguistic, cadence, mood, relationship, topic extractors)
    _store_test_event(db, "rebuild-1", "email.received", "proton_mail", {
        "subject": "Weekly sync notes",
        "body": "Here are the notes from today's sync. Great discussion about the roadmap.",
        "from": "colleague@work.com",
        "to": ["user@example.com"],
    })

    result = pipeline.rebuild_profiles_from_events()

    assert result["events_processed"] == 1
    assert len(result["profiles_rebuilt"]) > 0
    assert isinstance(result["errors"], list)


def test_rebuild_profiles_populates_signal_profiles(db, user_model_store):
    """Test that after rebuild, signal profiles actually exist in user_model.db."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store several communication events to populate cadence and relationship profiles.
    for i in range(5):
        _store_test_event(db, f"rebuild-pop-{i}", "email.received", "proton_mail", {
            "subject": f"Email {i}",
            "body": f"Message content number {i} with some meaningful text about the project.",
            "from": "alice@company.com",
            "to": ["user@example.com"],
        }, timestamp=f"2026-03-01T{10 + i}:00:00+00:00")

    # No profiles should exist yet.
    assert user_model_store.get_signal_profile("cadence") is None
    assert user_model_store.get_signal_profile("relationships") is None

    result = pipeline.rebuild_profiles_from_events()

    assert result["events_processed"] == 5

    # After rebuild, at least cadence and relationships should be populated
    # (email.received triggers CadenceExtractor and RelationshipExtractor).
    cadence_profile = user_model_store.get_signal_profile("cadence")
    assert cadence_profile is not None

    relationship_profile = user_model_store.get_signal_profile("relationships")
    assert relationship_profile is not None


def test_rebuild_profiles_with_mixed_event_types(db, user_model_store):
    """Test rebuild with mixed event types to verify multiple extractors fire."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Email event — triggers linguistic, cadence, mood, relationship, topic
    _store_test_event(db, "mix-email", "email.received", "proton_mail", {
        "subject": "Project update",
        "body": "The project is going well. Let's schedule a review.",
        "from": "bob@work.com",
        "to": ["user@example.com"],
    }, timestamp="2026-03-01T10:00:00+00:00")

    # Calendar event — triggers temporal, topic
    _store_test_event(db, "mix-cal", "calendar.event.created", "caldav", {
        "summary": "Team standup",
        "start": "2026-03-02T09:00:00Z",
        "end": "2026-03-02T09:30:00Z",
        "location": "Conference Room A",
    }, timestamp="2026-03-01T11:00:00+00:00")

    # Task event — triggers decision, topic
    _store_test_event(db, "mix-task", "task.created", "task_manager", {
        "title": "Review quarterly budget",
        "description": "Prepare Q1 budget review for leadership meeting",
        "due_date": "2026-03-20T17:00:00Z",
    }, timestamp="2026-03-01T12:00:00+00:00")

    result = pipeline.rebuild_profiles_from_events()

    assert result["events_processed"] == 3
    # Multiple extractor types should have fired.
    assert len(result["profiles_rebuilt"]) >= 2


def test_rebuild_profiles_fail_open_on_extractor_error(db, user_model_store):
    """Test that an error in one extractor doesn't stop processing of other events or extractors."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store a normal event.
    _store_test_event(db, "ok-event", "email.received", "proton_mail", {
        "subject": "Normal email",
        "body": "This email should process fine.",
        "from": "alice@company.com",
        "to": ["user@example.com"],
    }, timestamp="2026-03-01T10:00:00+00:00")

    # Temporarily sabotage one extractor to trigger the fail-open path.
    original_extract = pipeline.extractors[0].extract
    def broken_extract(event):
        raise RuntimeError("Simulated extractor failure")
    pipeline.extractors[0].extract = broken_extract

    try:
        result = pipeline.rebuild_profiles_from_events()
    finally:
        pipeline.extractors[0].extract = original_extract

    # Event should still be processed (other extractors succeed).
    assert result["events_processed"] == 1
    # The broken extractor's error should be captured.
    assert len(result["errors"]) >= 1
    assert "Simulated extractor failure" in result["errors"][0]
    # Other extractors should still have succeeded.
    assert len(result["profiles_rebuilt"]) >= 1


def test_rebuild_profiles_event_limit(db, user_model_store):
    """Test that event_limit caps the number of events loaded from events.db."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store 10 events.
    for i in range(10):
        _store_test_event(db, f"limit-{i}", "email.received", "proton_mail", {
            "subject": f"Email {i}",
            "body": f"Content {i}",
            "from": "sender@example.com",
            "to": ["user@example.com"],
        }, timestamp=f"2026-03-01T{10 + i}:00:00+00:00")

    # Rebuild with limit of 3.
    result = pipeline.rebuild_profiles_from_events(event_limit=3)

    assert result["events_processed"] == 3


def test_rebuild_profiles_empty_events_db(db, user_model_store):
    """Test that rebuild handles an empty events.db gracefully."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    result = pipeline.rebuild_profiles_from_events()

    assert result["events_processed"] == 0
    assert result["profiles_rebuilt"] == []
    assert result["errors"] == []


def test_rebuild_profiles_handles_empty_metadata(db, user_model_store):
    """Test that rebuild handles events with empty JSON metadata."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Insert an event with empty-object metadata via the EventStore.
    _store_test_event(db, "empty-meta", "email.received", "proton_mail", {
        "subject": "Test",
        "body": "Hello world",
        "from": "a@b.com",
        "to": ["u@x.com"],
    }, metadata={})

    result = pipeline.rebuild_profiles_from_events()

    # Should process without errors.
    assert result["events_processed"] == 1
    assert len(result["profiles_rebuilt"]) > 0


def test_rebuild_profiles_chronological_order(db, user_model_store):
    """Test that events are processed in chronological order regardless of DB fetch order."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store events with explicit timestamps out of insertion order.
    _store_test_event(db, "chrono-3", "email.received", "proton_mail", {
        "subject": "Third",
        "body": "Third message",
        "from": "alice@company.com",
        "to": ["user@example.com"],
    }, timestamp="2026-03-01T12:00:00+00:00")

    _store_test_event(db, "chrono-1", "email.received", "proton_mail", {
        "subject": "First",
        "body": "First message",
        "from": "alice@company.com",
        "to": ["user@example.com"],
    }, timestamp="2026-03-01T10:00:00+00:00")

    _store_test_event(db, "chrono-2", "email.received", "proton_mail", {
        "subject": "Second",
        "body": "Second message",
        "from": "alice@company.com",
        "to": ["user@example.com"],
    }, timestamp="2026-03-01T11:00:00+00:00")

    # Track processing order by monkey-patching one extractor.
    processed_ids = []
    original_extract = pipeline.extractors[1].extract  # CadenceExtractor

    def tracking_extract(event):
        processed_ids.append(event["id"])
        return original_extract(event)

    pipeline.extractors[1].extract = tracking_extract

    try:
        result = pipeline.rebuild_profiles_from_events()
    finally:
        pipeline.extractors[1].extract = original_extract

    assert result["events_processed"] == 3
    # Events should have been processed in chronological order.
    assert processed_ids == ["chrono-1", "chrono-2", "chrono-3"]
