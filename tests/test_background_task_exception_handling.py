"""
Test background task exception handling and monitoring.

Tests that background tasks (prediction loop, insight loop, etc.) are properly
monitored and exceptions are logged via logger.critical() instead of print().

The exception handler was migrated from print() to logging in PR #231 to
standardize observability across the codebase.
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
    """Background tasks are tracked in the background_tasks dict."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def sample_task():
        await asyncio.sleep(0.1)

    lifeos._start_background_task("test_task", sample_task())

    # Task should be tracked
    assert "test_task" in lifeos.background_tasks
    assert isinstance(lifeos.background_tasks["test_task"], asyncio.Task)

    # Wait for task to complete
    await lifeos.background_tasks["test_task"]

    # Task should still be tracked even after completion
    assert "test_task" in lifeos.background_tasks


@pytest.mark.asyncio
async def test_background_task_exception_logging(caplog):
    """Exceptions in background tasks are logged at CRITICAL level, not silently swallowed."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def crashing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Intentional test error")

    # Use caplog to capture structured log records (logger.critical uses logging, not print).
    with caplog.at_level(logging.CRITICAL, logger="main"):
        lifeos._start_background_task("crashing_task", crashing_task())

        # Wait for the task to crash (ValueError propagates out of the awaited task)
        try:
            await lifeos.background_tasks["crashing_task"]
        except ValueError:
            # Exception caught by the done callback; awaiting re-raises it
            pass

        # Give the done callback time to execute
        await asyncio.sleep(0.05)

    # Confirm the CRITICAL log record was emitted with the right content.
    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical_records, "Expected at least one CRITICAL log record"
    full_log = "\n".join(r.getMessage() for r in critical_records)
    assert "crashing_task" in full_log
    assert "crashed" in full_log or "Intentional test error" in full_log


@pytest.mark.asyncio
async def test_background_task_cancellation_not_logged(caplog):
    """Cancelled tasks (normal shutdown) do NOT trigger CRITICAL logging."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def long_running_task():
        await asyncio.sleep(10)

    with caplog.at_level(logging.CRITICAL, logger="main"):
        lifeos._start_background_task("long_task", long_running_task())

        # Cancel the task (simulating shutdown)
        lifeos.background_tasks["long_task"].cancel()

        try:
            await lifeos.background_tasks["long_task"]
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.05)

    # Cancellation is normal — no CRITICAL record should be emitted.
    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert not critical_records, (
        f"Expected no CRITICAL log records on cancellation, got: {critical_records}"
    )


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

    lifeos._start_background_task("task_a", task_a())
    lifeos._start_background_task("task_b", task_b())
    lifeos._start_background_task("task_c", task_c())

    assert len(lifeos.background_tasks) == 3
    assert "task_a" in lifeos.background_tasks
    assert "task_b" in lifeos.background_tasks
    assert "task_c" in lifeos.background_tasks

    await asyncio.gather(*lifeos.background_tasks.values())

    # All should still be tracked
    assert len(lifeos.background_tasks) == 3


@pytest.mark.asyncio
async def test_background_task_with_specific_exception_types(caplog):
    """Exception logging works for RuntimeError and KeyError."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def runtime_error_task():
        await asyncio.sleep(0.01)
        raise RuntimeError("Database connection lost")

    with caplog.at_level(logging.CRITICAL, logger="main"):
        lifeos._start_background_task("runtime_task", runtime_error_task())
        try:
            await lifeos.background_tasks["runtime_task"]
        except RuntimeError:
            pass
        await asyncio.sleep(0.05)

    # CRITICAL record should mention the exception
    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical_records, "Expected CRITICAL record for RuntimeError"
    assert any("runtime_task" in r.getMessage() for r in critical_records)

    # Reset for next task
    caplog.clear()

    async def key_error_task():
        await asyncio.sleep(0.01)
        raise KeyError("Missing configuration key")

    with caplog.at_level(logging.CRITICAL, logger="main"):
        lifeos._start_background_task("key_task", key_error_task())
        try:
            await lifeos.background_tasks["key_task"]
        except KeyError:
            pass
        await asyncio.sleep(0.05)

    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical_records, "Expected CRITICAL record for KeyError"
    assert any("key_task" in r.getMessage() for r in critical_records)


@pytest.mark.asyncio
async def test_background_task_exception_includes_exc_info(caplog):
    """Exception logging includes exc_info=True so the traceback is captured."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def nested_error_task():
        def inner_function():
            raise ValueError("Deep error in inner function")

        await asyncio.sleep(0.01)
        inner_function()

    with caplog.at_level(logging.CRITICAL, logger="main"):
        lifeos._start_background_task("nested_task", nested_error_task())
        try:
            await lifeos.background_tasks["nested_task"]
        except ValueError:
            pass
        await asyncio.sleep(0.05)

    # The record should have exc_info attached (logger.critical(..., exc_info=True))
    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical_records, "Expected at least one CRITICAL record"
    # exc_info is set on the LogRecord when exc_info=True is passed
    assert any(r.exc_info is not None for r in critical_records), (
        "Expected exc_info to be attached to the CRITICAL log record"
    )


@pytest.mark.asyncio
async def test_background_task_result_success():
    """Successful tasks can be awaited normally and return their result."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def successful_task():
        await asyncio.sleep(0.01)
        return "success"

    lifeos._start_background_task("success_task", successful_task())
    result = await lifeos.background_tasks["success_task"]
    assert result == "success"
