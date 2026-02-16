"""
Integration tests for DecisionExtractor in the SignalExtractorPipeline.

Verifies that:
1. DecisionExtractor is properly registered in the pipeline
2. Decision signals flow through the pipeline correctly
3. Profile is included in user summaries
"""

import pytest
from datetime import datetime, timedelta, timezone

from models.core import EventType
from services.signal_extractor.pipeline import SignalExtractorPipeline


@pytest.mark.asyncio
async def test_decision_extractor_in_pipeline(db, user_model_store):
    """DecisionExtractor should be registered in the signal pipeline."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Check that DecisionExtractor is in the pipeline
    extractor_types = [type(e).__name__ for e in pipeline.extractors]
    assert "DecisionExtractor" in extractor_types


@pytest.mark.asyncio
async def test_decision_signals_flow_through_pipeline(db, user_model_store, event_store):
    """Decision signals should flow through the pipeline correctly."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    now = datetime.now(timezone.utc)
    task_id = "pipeline_task"

    # Create task event
    creation_event = {
        "id": "event_create_pipeline",
        "type": EventType.TASK_CREATED.value,
        "source": "test",
        "timestamp": now.isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
            "title": "Pipeline test task",
            "domain": "work",
        },
        "metadata": {},
    }
    event_store.store_event(creation_event)

    # Complete task 1 hour later
    completion_event = {
        "id": "event_complete_pipeline",
        "type": EventType.TASK_COMPLETED.value,
        "source": "test",
        "timestamp": (now + timedelta(hours=1)).isoformat(),
        "priority": "normal",
        "payload": {
            "task_id": task_id,
        },
        "metadata": {},
    }

    # Process through pipeline
    signals = await pipeline.process_event(completion_event)

    # Should produce decision_speed signal
    decision_signals = [s for s in signals if s.get("type") == "decision_speed"]
    assert len(decision_signals) == 1
    assert decision_signals[0]["domain"] == "work"
    assert decision_signals[0]["speed_category"] in ["immediate", "same_day"]


@pytest.mark.asyncio
async def test_decision_profile_in_user_summary(db, user_model_store):
    """DecisionProfile should be included in pipeline user summary."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Generate a decision signal via pipeline
    event = {
        "type": EventType.MESSAGE_SENT.value,
        "timestamp": now.isoformat(),
        "payload": {
            "content": "You decide where we eat tonight!",
            "recipient": "partner",
        },
    }

    await pipeline.process_event(event)

    # Get user summary
    summary = pipeline.get_user_summary()

    # DecisionProfile should be in profiles
    assert "profiles" in summary
    assert "decision" in summary["profiles"]
    assert summary["profiles"]["decision"]["samples_count"] >= 0


@pytest.mark.asyncio
async def test_multiple_decision_signals_aggregate(db, user_model_store, event_store):
    """Multiple decision signals should aggregate into the profile correctly."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    now = datetime.now(timezone.utc)

    # Process multiple calendar events with different planning horizons
    for i in range(5):
        event_start = now + timedelta(days=i + 1)
        calendar_event = {
            "type": EventType.CALENDAR_EVENT_CREATED.value,
            "timestamp": now.isoformat(),
            "payload": {
                "start_time": event_start.isoformat(),
                "summary": f"Meeting {i}",
            },
        }
        await pipeline.process_event(calendar_event)

    # Verify profile has aggregated data
    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None
    assert profile["samples_count"] >= 5

    # Should have risk tolerance data
    assert "risk_tolerance_by_domain" in profile["data"]
    # At least one domain should have risk data
    assert len(profile["data"]["risk_tolerance_by_domain"]) > 0


@pytest.mark.asyncio
async def test_decision_extractor_fail_open_in_pipeline(db, user_model_store):
    """DecisionExtractor errors should not crash the pipeline."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Malformed event that might cause extraction errors
    malformed_event = {
        "type": EventType.TASK_COMPLETED.value,
        "timestamp": "invalid-timestamp",
        "payload": {},
    }

    # Pipeline should not crash
    signals = await pipeline.process_event(malformed_event)

    # Signals list should be returned (even if empty)
    assert isinstance(signals, list)
