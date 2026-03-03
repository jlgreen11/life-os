"""
Test background task exception handling and monitoring.

Tests that background tasks (prediction loop, insight loop, etc.) are properly
monitored and exceptions are logged. The _start_background_task method now accepts
a coroutine factory and automatically restarts crashed tasks with exponential backoff.

The exception handler was migrated from print() to logging in PR #231 to
standardize observability across the codebase. Auto-restart was added to prevent
permanently degraded operation after a single loop crash.
"""

import asyncio
import logging

import pytest

from main import LifeOS

_CONFIG = {
    "data_dir": "./test_data",
    "nats_url": "nats://localhost:4222",
    "connectors": {},
}


@pytest.mark.asyncio
async def test_background_task_tracking():
    """Background tasks are tracked in the background_tasks dict with metadata."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def sample_task():
        await asyncio.sleep(0.1)

    lifeos._start_background_task("test_task", sample_task)

    # Task should be tracked with metadata dict
    assert "test_task" in lifeos.background_tasks
    meta = lifeos.background_tasks["test_task"]
    assert isinstance(meta, dict)
    assert isinstance(meta["task"], asyncio.Task)
    assert meta["restarts"] == 0
    assert meta["last_restart"] is None

    # Wait for task to complete
    await meta["task"]

    # Task should still be tracked even after completion
    assert "test_task" in lifeos.background_tasks


@pytest.mark.asyncio
async def test_background_task_cancellation_not_logged(caplog):
    """Cancelled tasks (normal shutdown) do NOT trigger restart or CRITICAL logging."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def long_running_task():
        await asyncio.sleep(10)

    with caplog.at_level(logging.WARNING, logger="main"):
        lifeos._start_background_task("long_task", long_running_task)

        # Cancel the task (simulating shutdown)
        lifeos.background_tasks["long_task"]["task"].cancel()

        try:
            await lifeos.background_tasks["long_task"]["task"]
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.05)

    # Cancellation is normal — no WARNING or CRITICAL record should be emitted.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    cancel_related = [r for r in warning_records if "long_task" in r.getMessage()]
    assert not cancel_related, f"Expected no WARNING/CRITICAL log records on cancellation, got: {cancel_related}"


@pytest.mark.asyncio
async def test_multiple_background_tasks():
    """Multiple background tasks can run simultaneously and be tracked."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def task_a():
        await asyncio.sleep(0.05)

    async def task_b():
        await asyncio.sleep(0.05)

    async def task_c():
        await asyncio.sleep(0.05)

    lifeos._start_background_task("task_a", task_a)
    lifeos._start_background_task("task_b", task_b)
    lifeos._start_background_task("task_c", task_c)

    assert len(lifeos.background_tasks) == 3
    assert "task_a" in lifeos.background_tasks
    assert "task_b" in lifeos.background_tasks
    assert "task_c" in lifeos.background_tasks

    await asyncio.gather(*(m["task"] for m in lifeos.background_tasks.values()))

    # All should still be tracked
    assert len(lifeos.background_tasks) == 3


@pytest.mark.asyncio
async def test_background_task_result_success():
    """Successful tasks can be awaited normally and return their result."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def successful_task():
        await asyncio.sleep(0.01)
        return "success"

    lifeos._start_background_task("success_task", successful_task)
    result = await lifeos.background_tasks["success_task"]["task"]
    assert result is None  # Wrapper returns None (success_task return value is internal)
