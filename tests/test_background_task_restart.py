"""
Test background task auto-restart with exponential backoff.

Verifies that _start_background_task properly restarts crashed background loops
with exponential backoff (30s base, doubling, capped at 600s) and respects the
max_restarts limit. CancelledError (normal shutdown) must NOT trigger restarts.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from main import LifeOS

_CONFIG = {
    "data_dir": "./test_data",
    "nats_url": "nats://localhost:4222",
    "connectors": {},
}


@pytest.mark.asyncio
async def test_task_restarts_after_exception():
    """A task that crashes is automatically restarted by the wrapper."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def flaky_task():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError(f"Crash #{call_count}")
        # Third call succeeds and returns normally

    with patch("asyncio.sleep", new_callable=AsyncMock):
        lifeos._start_background_task("flaky", flaky_task, max_restarts=5)
        await lifeos.background_tasks["flaky"]["task"]

    assert call_count == 3, "Factory should have been called 3 times (2 crashes + 1 success)"
    assert lifeos.background_tasks["flaky"]["restarts"] == 2


@pytest.mark.asyncio
async def test_cancelled_error_does_not_trigger_restart():
    """CancelledError (normal shutdown) propagates without restarting."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def cancellable_task():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(100)  # Will be cancelled

    lifeos._start_background_task("cancel_me", cancellable_task, max_restarts=5)
    # Give the task a moment to start
    await asyncio.sleep(0.01)

    lifeos.background_tasks["cancel_me"]["task"].cancel()
    try:
        await lifeos.background_tasks["cancel_me"]["task"]
    except asyncio.CancelledError:
        pass

    assert call_count == 1, "Task should have run once and NOT restarted after cancel"
    assert lifeos.background_tasks["cancel_me"]["restarts"] == 0


@pytest.mark.asyncio
async def test_exponential_backoff_timing():
    """Backoff follows 30s * 2^(n-1), capped at 600s."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0
    backoff_values = []

    async def always_crash():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("always fails")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        lifeos._start_background_task("backoff_test", always_crash, max_restarts=6)
        await lifeos.background_tasks["backoff_test"]["task"]
        backoff_values = [call.args[0] for call in mock_sleep.call_args_list]

    # 6 restarts means 6 backoff sleeps, then the 7th crash exceeds max_restarts
    expected = [30, 60, 120, 240, 480, 600]
    assert backoff_values == expected, f"Expected backoff {expected}, got {backoff_values}"


@pytest.mark.asyncio
async def test_max_restarts_limit_respected():
    """After max_restarts consecutive failures, the task stops permanently."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def always_crash():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("permanent failure")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        lifeos._start_background_task("limited", always_crash, max_restarts=3)
        await lifeos.background_tasks["limited"]["task"]

    # 3 restarts + 1 final crash that exceeds the limit = 4 total calls
    assert call_count == 4
    assert lifeos.background_tasks["limited"]["restarts"] == 4


@pytest.mark.asyncio
async def test_max_restarts_logs_critical(caplog):
    """Exceeding max_restarts emits a CRITICAL log message."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def always_crash():
        raise RuntimeError("won't recover")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with caplog.at_level(logging.CRITICAL, logger="main"):
            lifeos._start_background_task("critical_test", always_crash, max_restarts=2)
            await lifeos.background_tasks["critical_test"]["task"]
            await asyncio.sleep(0.01)

    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert critical_records, "Expected CRITICAL log when max_restarts exceeded"
    msg = critical_records[0].getMessage()
    assert "critical_test" in msg
    assert "max restarts" in msg.lower() or "exceeded" in msg.lower()


@pytest.mark.asyncio
async def test_restart_logs_warning(caplog):
    """Each restart emits a WARNING log with the task name and restart count."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def crash_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("first-run crash")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with caplog.at_level(logging.WARNING, logger="main"):
            lifeos._start_background_task("warn_test", crash_once, max_restarts=5)
            await lifeos.background_tasks["warn_test"]["task"]

    warning_records = [
        r for r in caplog.records if r.levelno == logging.WARNING and r.name == "main" and "warn_test" in r.getMessage()
    ]
    assert warning_records, "Expected WARNING log on restart from main logger"
    msg = warning_records[0].getMessage()
    assert "warn_test" in msg
    assert "restart 1/5" in msg.lower() or "1/5" in msg


@pytest.mark.asyncio
async def test_restart_metadata_tracked():
    """Restart count and last_restart timestamp are updated in background_tasks."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def crash_twice():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError(f"crash {call_count}")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        lifeos._start_background_task("meta_test", crash_twice, max_restarts=5)
        await lifeos.background_tasks["meta_test"]["task"]

    meta = lifeos.background_tasks["meta_test"]
    assert meta["restarts"] == 2
    assert meta["last_restart"] is not None, "last_restart should be set after restart"


@pytest.mark.asyncio
async def test_normal_return_no_restart():
    """A task that returns normally is NOT restarted."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    call_count = 0

    async def run_once():
        nonlocal call_count
        call_count += 1

    lifeos._start_background_task("once_only", run_once, max_restarts=5)
    await lifeos.background_tasks["once_only"]["task"]

    assert call_count == 1, "Normal return should not trigger restart"
    assert lifeos.background_tasks["once_only"]["restarts"] == 0


@pytest.mark.asyncio
async def test_backoff_cap_at_600():
    """Backoff never exceeds 600 seconds regardless of restart count."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)

    async def always_crash():
        raise RuntimeError("fail")

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        lifeos._start_background_task("cap_test", always_crash, max_restarts=8)
        await lifeos.background_tasks["cap_test"]["task"]
        backoff_values = [call.args[0] for call in mock_sleep.call_args_list]

    # All values should be <= 600
    assert all(v <= 600 for v in backoff_values), f"Backoff exceeded cap: {backoff_values}"
    # Last few should be exactly 600
    assert backoff_values[-1] == 600


@pytest.mark.asyncio
async def test_factory_called_fresh_each_restart():
    """The coroutine factory is called fresh for each restart attempt."""
    lifeos = LifeOS(config_path=None, config=_CONFIG)
    factory_calls = []

    async def tracked_factory():
        factory_calls.append(len(factory_calls) + 1)
        if len(factory_calls) < 3:
            raise RuntimeError("not ready yet")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        lifeos._start_background_task("factory_test", tracked_factory, max_restarts=5)
        await lifeos.background_tasks["factory_test"]["task"]

    assert factory_calls == [1, 2, 3], f"Expected 3 factory calls, got {factory_calls}"
