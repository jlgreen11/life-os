"""
Test background task exception handling and monitoring.

Tests that background tasks (prediction loop, insight loop, etc.) are properly
monitored and exceptions are logged instead of being silently swallowed.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from main import LifeOS


@pytest.mark.asyncio
async def test_background_task_tracking():
    """Test that background tasks are tracked in the background_tasks dict."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create a simple background task
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
async def test_background_task_exception_logging():
    """Test that exceptions in background tasks are logged, not silently swallowed."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create a task that will crash
    async def crashing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Intentional test error")

    # Patch print to capture exception logging
    with patch("builtins.print") as mock_print:
        lifeos._start_background_task("crashing_task", crashing_task())

        # Wait for task to crash
        try:
            await lifeos.background_tasks["crashing_task"]
        except ValueError:
            # Exception should be caught by the done callback
            pass

        # Give the done callback time to execute
        await asyncio.sleep(0.05)

        # Check that exception was logged
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("CRITICAL: Background task 'crashing_task' crashed" in call for call in print_calls)
        assert any("ValueError: Intentional test error" in call for call in print_calls)
        assert any("degraded mode" in call for call in print_calls)


@pytest.mark.asyncio
async def test_background_task_cancellation_not_logged():
    """Test that cancelled tasks (normal shutdown) don't trigger error logging."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create a long-running task
    async def long_running_task():
        await asyncio.sleep(10)

    # Patch print to verify no error logging
    with patch("builtins.print") as mock_print:
        lifeos._start_background_task("long_task", long_running_task())

        # Cancel the task (simulating shutdown)
        lifeos.background_tasks["long_task"].cancel()

        # Wait for cancellation to complete
        try:
            await lifeos.background_tasks["long_task"]
        except asyncio.CancelledError:
            pass

        # Give the done callback time to execute
        await asyncio.sleep(0.05)

        # Check that NO error was logged (cancellation is normal)
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert not any("CRITICAL" in call for call in print_calls)
        assert not any("crashed" in call for call in print_calls)


@pytest.mark.asyncio
async def test_multiple_background_tasks():
    """Test that multiple background tasks can run simultaneously and be tracked."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create multiple tasks
    async def task_a():
        await asyncio.sleep(0.05)

    async def task_b():
        await asyncio.sleep(0.05)

    async def task_c():
        await asyncio.sleep(0.05)

    lifeos._start_background_task("task_a", task_a())
    lifeos._start_background_task("task_b", task_b())
    lifeos._start_background_task("task_c", task_c())

    # All tasks should be tracked
    assert len(lifeos.background_tasks) == 3
    assert "task_a" in lifeos.background_tasks
    assert "task_b" in lifeos.background_tasks
    assert "task_c" in lifeos.background_tasks

    # Wait for all to complete
    await asyncio.gather(*lifeos.background_tasks.values())

    # All should still be tracked
    assert len(lifeos.background_tasks) == 3


@pytest.mark.asyncio
async def test_background_task_with_specific_exception_types():
    """Test exception logging for different exception types."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Test RuntimeError
    async def runtime_error_task():
        await asyncio.sleep(0.01)
        raise RuntimeError("Database connection lost")

    with patch("builtins.print") as mock_print:
        lifeos._start_background_task("runtime_task", runtime_error_task())

        try:
            await lifeos.background_tasks["runtime_task"]
        except RuntimeError:
            pass

        await asyncio.sleep(0.05)

        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("RuntimeError: Database connection lost" in call for call in print_calls)

    # Test KeyError
    async def key_error_task():
        await asyncio.sleep(0.01)
        raise KeyError("Missing configuration key")

    with patch("builtins.print") as mock_print:
        lifeos._start_background_task("key_task", key_error_task())

        try:
            await lifeos.background_tasks["key_task"]
        except KeyError:
            pass

        await asyncio.sleep(0.05)

        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("KeyError" in call for call in print_calls)


@pytest.mark.asyncio
async def test_background_task_exception_includes_traceback():
    """Test that exception logging includes full traceback for debugging."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create a task with a nested exception
    async def nested_error_task():
        def inner_function():
            raise ValueError("Deep error in inner function")

        await asyncio.sleep(0.01)
        inner_function()

    with patch("builtins.print") as mock_print:
        lifeos._start_background_task("nested_task", nested_error_task())

        try:
            await lifeos.background_tasks["nested_task"]
        except ValueError:
            pass

        await asyncio.sleep(0.05)

        # Check that traceback includes function names
        print_calls = [str(call) for call in mock_print.call_args_list]
        full_output = "\n".join(print_calls)
        assert "inner_function" in full_output
        assert "ValueError: Deep error in inner function" in full_output
        assert "Traceback" in full_output


@pytest.mark.asyncio
async def test_background_task_result_success():
    """Test that successful tasks can be awaited normally."""
    config = {
        "data_dir": "./test_data",
        "nats_url": "nats://localhost:4222",
        "connectors": {},
    }

    lifeos = LifeOS(config_path=None, config=config)

    # Create a task that returns a value
    async def successful_task():
        await asyncio.sleep(0.01)
        return "success"

    lifeos._start_background_task("success_task", successful_task())

    # Task should complete successfully
    result = await lifeos.background_tasks["success_task"]
    assert result == "success"
