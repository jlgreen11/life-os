"""
Tests for adaptive routine detection retry intervals.

Verifies that the routine detection loop uses appropriate retry intervals
based on how many patterns are detected:
  - 0 patterns: retry in 1 hour (cold start)
  - 1-2 patterns: retry in 3 hours (partial data)
  - 3+ patterns: retry in 12 hours (stable patterns)

This adaptive approach fixes the cold-start problem where routine detection
runs 60 seconds after startup (before connectors have synced) and finds 0
routines, then sleeps for 12 hours. With adaptive retry, it will retry in
1 hour and quickly detect routines once episodes accumulate.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_adaptive_retry_no_patterns_detected(db, user_model_store):
    """Test that detection retries in 1 hour when no patterns are found (cold start)."""
    from main import LifeOS

    # Mock config with minimal settings
    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    # Create LifeOS instance with injected dependencies
    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Mock the detectors to return 0 patterns
    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(return_value=[])
    lifeos.routine_detector.store_routines = MagicMock(return_value=0)

    lifeos.workflow_detector = MagicMock()
    lifeos.workflow_detector.detect_workflows = MagicMock(return_value=[])
    lifeos.workflow_detector.store_workflows = MagicMock(return_value=0)

    # Mock asyncio.sleep to track sleep duration
    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 2:  # Initial 60s + first retry
            lifeos.shutdown_event.set()  # Stop the loop
        await original_sleep(0.01)  # Small delay to yield control

    with patch("asyncio.sleep", side_effect=mock_sleep):
        # Run the detection loop
        await lifeos._routine_detection_loop()

    # Verify sleep durations
    assert len(sleep_durations) >= 2, "Should have at least 2 sleep calls"
    assert sleep_durations[0] == 60, "First sleep should be 60s startup delay"
    assert sleep_durations[1] == 3600, "Second sleep should be 1 hour (3600s) when 0 patterns detected"


@pytest.mark.asyncio
async def test_adaptive_retry_partial_patterns_detected(db, user_model_store):
    """Test that detection retries in 3 hours when 1-2 patterns are found."""
    from main import LifeOS

    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Mock the detectors to return 1 routine + 0 workflows (total = 1)
    mock_routine = {
        "name": "Morning routine",
        "trigger": "morning",
        "steps": [{"order": 0, "action": "check_email", "typical_duration_minutes": 5.0, "skip_rate": 0.1}],
        "typical_duration_minutes": 5.0,
        "consistency_score": 0.8,
        "times_observed": 10,
        "variations": [],
    }

    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(return_value=[mock_routine])
    lifeos.routine_detector.store_routines = MagicMock(return_value=1)

    lifeos.workflow_detector = MagicMock()
    lifeos.workflow_detector.detect_workflows = MagicMock(return_value=[])
    lifeos.workflow_detector.store_workflows = MagicMock(return_value=0)

    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 2:
            lifeos.shutdown_event.set()
        await original_sleep(0.01)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await lifeos._routine_detection_loop()

    assert len(sleep_durations) >= 2
    assert sleep_durations[0] == 60, "First sleep should be 60s startup delay"
    assert sleep_durations[1] == 10800, "Second sleep should be 3 hours (10800s) when 1 pattern detected"


@pytest.mark.asyncio
async def test_adaptive_retry_stable_patterns_detected(db, user_model_store):
    """Test that detection retries in 12 hours when 3+ patterns are found."""
    from main import LifeOS

    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Mock the detectors to return 2 routines + 1 workflow (total = 3)
    mock_routines = [
        {
            "name": "Morning routine",
            "trigger": "morning",
            "steps": [{"order": 0, "action": "check_email", "typical_duration_minutes": 5.0, "skip_rate": 0.1}],
            "typical_duration_minutes": 5.0,
            "consistency_score": 0.8,
            "times_observed": 10,
            "variations": [],
        },
        {
            "name": "Evening routine",
            "trigger": "evening",
            "steps": [{"order": 0, "action": "review_tasks", "typical_duration_minutes": 10.0, "skip_rate": 0.2}],
            "typical_duration_minutes": 10.0,
            "consistency_score": 0.7,
            "times_observed": 8,
            "variations": [],
        },
    ]

    mock_workflows = [
        {
            "name": "Code review workflow",
            "trigger": "pull_request_created",
            "steps": [{"order": 0, "action": "review_code", "typical_duration_minutes": 15.0}],
            "typical_duration_minutes": 15.0,
            "consistency_score": 0.9,
            "times_observed": 12,
        }
    ]

    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(return_value=mock_routines)
    lifeos.routine_detector.store_routines = MagicMock(return_value=2)

    lifeos.workflow_detector = MagicMock()
    lifeos.workflow_detector.detect_workflows = MagicMock(return_value=mock_workflows)
    lifeos.workflow_detector.store_workflows = MagicMock(return_value=1)

    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 2:
            lifeos.shutdown_event.set()
        await original_sleep(0.01)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await lifeos._routine_detection_loop()

    assert len(sleep_durations) >= 2
    assert sleep_durations[0] == 60, "First sleep should be 60s startup delay"
    assert sleep_durations[1] == 43200, "Second sleep should be 12 hours (43200s) when 3+ patterns detected"


@pytest.mark.asyncio
async def test_adaptive_retry_exactly_two_patterns(db, user_model_store):
    """Test boundary condition: exactly 2 patterns should use 3-hour retry."""
    from main import LifeOS

    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Mock 2 routines + 0 workflows (total = 2)
    mock_routines = [
        {
            "name": "Morning routine",
            "trigger": "morning",
            "steps": [],
            "typical_duration_minutes": 5.0,
            "consistency_score": 0.8,
            "times_observed": 10,
            "variations": [],
        },
        {
            "name": "Evening routine",
            "trigger": "evening",
            "steps": [],
            "typical_duration_minutes": 10.0,
            "consistency_score": 0.7,
            "times_observed": 8,
            "variations": [],
        },
    ]

    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(return_value=mock_routines)
    lifeos.routine_detector.store_routines = MagicMock(return_value=2)

    lifeos.workflow_detector = MagicMock()
    lifeos.workflow_detector.detect_workflows = MagicMock(return_value=[])
    lifeos.workflow_detector.store_workflows = MagicMock(return_value=0)

    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 2:
            lifeos.shutdown_event.set()
        await original_sleep(0.01)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await lifeos._routine_detection_loop()

    assert sleep_durations[1] == 10800, "Exactly 2 patterns should use 3-hour retry"


@pytest.mark.asyncio
async def test_adaptive_retry_after_error(db, user_model_store):
    """Test that detection retries in 1 hour after an error to avoid tight error loops."""
    from main import LifeOS

    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Mock the detectors to raise an exception
    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(side_effect=Exception("Test error"))

    lifeos.workflow_detector = MagicMock()

    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 2:
            lifeos.shutdown_event.set()
        await original_sleep(0.01)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await lifeos._routine_detection_loop()

    assert len(sleep_durations) >= 2
    assert sleep_durations[0] == 60, "First sleep should be 60s startup delay"
    assert sleep_durations[1] == 3600, "Second sleep should be 1 hour (3600s) after error"


@pytest.mark.asyncio
async def test_adaptive_retry_progression_cold_start_to_stable(db, user_model_store):
    """Test realistic progression: 0 patterns → 1 pattern → 3 patterns (adaptive retry intervals)."""
    from main import LifeOS

    config = {
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "connectors": {},
    }

    lifeos = LifeOS(config=config, db=db, user_model_store=user_model_store)
    lifeos.shutdown_event = asyncio.Event()

    # Simulate progression: first call finds 0, second finds 1, third finds 3
    detection_results = [
        ([], []),  # 0 patterns
        ([{"name": "Morning", "trigger": "morning", "steps": [], "typical_duration_minutes": 5, "consistency_score": 0.8, "times_observed": 10, "variations": []}], []),  # 1 pattern
        ([{"name": "Morning", "trigger": "morning", "steps": [], "typical_duration_minutes": 5, "consistency_score": 0.8, "times_observed": 10, "variations": []},
          {"name": "Evening", "trigger": "evening", "steps": [], "typical_duration_minutes": 10, "consistency_score": 0.7, "times_observed": 8, "variations": []}],
         [{"name": "Workflow1", "trigger": "event", "steps": [], "typical_duration_minutes": 15, "consistency_score": 0.9, "times_observed": 12}]),  # 3 patterns
    ]

    call_count = [0]

    def get_routines(lookback_days):
        result = detection_results[call_count[0]][0]
        return result

    def get_workflows(lookback_days):
        result = detection_results[call_count[0]][1]
        call_count[0] += 1
        return result

    lifeos.routine_detector = MagicMock()
    lifeos.routine_detector.detect_routines = MagicMock(side_effect=get_routines)
    lifeos.routine_detector.store_routines = MagicMock(side_effect=lambda r: len(r))

    lifeos.workflow_detector = MagicMock()
    lifeos.workflow_detector.detect_workflows = MagicMock(side_effect=get_workflows)
    lifeos.workflow_detector.store_workflows = MagicMock(side_effect=lambda w: len(w))

    sleep_durations = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_durations.append(duration)
        if len(sleep_durations) >= 4:  # Initial + 3 detection cycles
            lifeos.shutdown_event.set()
        await original_sleep(0.01)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        await lifeos._routine_detection_loop()

    # Verify progression
    assert len(sleep_durations) >= 4, "Should have 4 sleep calls (startup + 3 retries)"
    assert sleep_durations[0] == 60, "Initial startup delay"
    assert sleep_durations[1] == 3600, "1 hour retry after 0 patterns"
    assert sleep_durations[2] == 10800, "3 hour retry after 1 pattern"
    assert sleep_durations[3] == 43200, "12 hour retry after 3 patterns"
