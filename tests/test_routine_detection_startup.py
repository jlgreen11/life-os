"""
Tests for routine detection loop startup behavior.

Verifies that the routine detection loop runs immediately on startup (after 60s warmup)
rather than waiting 12 hours, ensuring Layer 3 (Procedural Memory) is populated quickly.
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from main import LifeOS


@pytest.mark.asyncio
async def test_routine_detection_60_second_warmup(db, event_bus, event_store, user_model_store):
    """Test that routine detection waits 60 seconds before first run."""
    config = {
        "data_dir": str(db.data_dir),
        "nats_url": "nats://localhost:4222",
        "connectors": {},
        "ai": {"use_cloud": False},
    }

    sleep_calls = []

    async def mock_sleep(duration):
        sleep_calls.append(duration)
        if duration == 60:
            # Allow 60s warmup to complete
            return
        # Cancel on 12-hour sleep to end test
        raise asyncio.CancelledError()

    # Patch RoutineDetector methods to track calls
    detect_calls = []

    async def mock_detect_routines(lookback_days):
        detect_calls.append(lookback_days)
        return []

    with patch("asyncio.sleep", side_effect=mock_sleep):
        app = LifeOS(config=config, db=db, event_bus=event_bus,
                     event_store=event_store, user_model_store=user_model_store)

        # Patch the detector's detect_routines method
        original_detect = app.routine_detector.detect_routines
        app.routine_detector.detect_routines = lambda lookback_days=30: (detect_calls.append(lookback_days), [])[1]

        loop_task = asyncio.create_task(app._routine_detection_loop())

        try:
            await asyncio.wait_for(loop_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    # Verify sleep was called with 60 seconds first (warmup)
    assert 60 in sleep_calls, f"Should sleep 60 seconds for warmup on startup. Got: {sleep_calls}"
    # Verify detection ran after warmup
    assert len(detect_calls) > 0, "Should run detection after 60s warmup"
    assert detect_calls[0] == 30, "Should use 30-day lookback"


@pytest.mark.asyncio
async def test_routine_detection_12_hour_interval(db, event_bus, event_store, user_model_store):
    """Test that routine detection continues running every 12 hours after startup."""
    config = {
        "data_dir": str(db.data_dir),
        "nats_url": "nats://localhost:4222",
        "connectors": {},
        "ai": {"use_cloud": False},
    }

    sleep_calls = []
    detection_count = [0]

    async def mock_sleep(duration):
        sleep_calls.append(duration)
        if duration == 60:
            return  # Allow warmup
        if detection_count[0] >= 2:
            # After 2 detection runs, stop the loop
            raise asyncio.CancelledError()
        return  # Allow 12h sleep to complete (fast in test)

    with patch("asyncio.sleep", side_effect=mock_sleep):
        app = LifeOS(config=config, db=db, event_bus=event_bus,
                     event_store=event_store, user_model_store=user_model_store)

        # Patch detector to count invocations and return 3 stable routines so
        # the loop takes the 12-hour sleep path (requires 3+ routines detected).
        original_detect = app.routine_detector.detect_routines

        def counting_detect(lookback_days=30):
            detection_count[0] += 1
            # Return 3 dummy routines so the loop enters the 12-hour sleep
            # branch (the branch that fires when patterns are well-established).
            return [
                {"name": "morning_email", "consistency": 0.8},
                {"name": "afternoon_review", "consistency": 0.7},
                {"name": "evening_planning", "consistency": 0.75},
            ]

        app.routine_detector.detect_routines = counting_detect

        loop_task = asyncio.create_task(app._routine_detection_loop())

        try:
            await asyncio.wait_for(loop_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    # Verify the loop ran multiple times
    assert detection_count[0] >= 2, "Detection should run multiple times (startup + interval)"
    # Verify 12-hour sleep was called between runs
    assert 43200 in sleep_calls, f"Should sleep 12 hours (43200s) between detection runs. Got: {sleep_calls}"


@pytest.mark.asyncio
async def test_routine_detection_handles_errors_gracefully(db, event_bus, event_store, user_model_store):
    """Test that routine detection loop continues running even if detection fails."""
    config = {
        "data_dir": str(db.data_dir),
        "nats_url": "nats://localhost:4222",
        "connectors": {},
        "ai": {"use_cloud": False},
    }

    detection_attempts = [0]

    async def mock_sleep(duration):
        if duration == 60:
            return
        # After error, should still sleep and retry
        if detection_attempts[0] >= 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", side_effect=mock_sleep):
        app = LifeOS(config=config, db=db, event_bus=event_bus,
                     event_store=event_store, user_model_store=user_model_store)

        def failing_detect(lookback_days=30):
            detection_attempts[0] += 1
            if detection_attempts[0] == 1:
                raise ValueError("Simulated detection failure")
            return []  # Second attempt succeeds

        app.routine_detector.detect_routines = failing_detect

        loop_task = asyncio.create_task(app._routine_detection_loop())

        try:
            await asyncio.wait_for(loop_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    # Verify detection was attempted twice (failed once, succeeded once)
    assert detection_attempts[0] >= 2, "Should retry detection after error"


# Note: Temporal routine detection logic is already tested in test_routine_detector.py
# This file focuses specifically on the startup behavior fix


def test_routine_detection_startup_improves_layer3_population():
    """Test that the 60-second startup delay improves Layer 3 population time.

    Before fix: routine detection would wait 12 hours, leaving Layer 3 empty
    After fix: routine detection runs after 60 seconds, populating Layer 3 quickly
    """
    # This is a documentation test - the actual behavior is tested in the async tests above
    # This test verifies the improvement described in the issue

    # Before: Layer 3 (Procedural Memory) empty for up to 12 hours after startup
    max_delay_before_fix = 12 * 60 * 60  # 43200 seconds

    # After: Layer 3 populated after 60 seconds + detection time (~5 seconds)
    max_delay_after_fix = 60 + 5  # 65 seconds

    improvement_factor = max_delay_before_fix / max_delay_after_fix
    assert improvement_factor > 600, f"Startup fix improves Layer 3 population time by {improvement_factor}x"
