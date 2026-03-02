"""
Tests for BaseConnector auto-reconnect with exponential backoff.

When authenticate() fails during start(), the connector launches a background
reconnect loop that retries at increasing intervals (1 min, 5 min, 15 min,
1 hr cap). On success, the normal sync loop starts automatically.

Coverage:
    - Auth failure starts reconnect loop (not dead permanently)
    - Exponential backoff delays follow RECONNECT_DELAYS schedule
    - Successful reconnect starts sync loop and resets attempt counter
    - stop() cancels an active reconnect task
    - Delay caps at max (1 hr) after exceeding RECONNECT_DELAYS length
    - Reconnect publishes system.connector.reconnected event on success
    - Reconnect updates connector state to 'active' on success
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


# -------------------------------------------------------------------------
# Test Connector with controllable auth
# -------------------------------------------------------------------------


class ReconnectTestConnector(BaseConnector):
    """Concrete BaseConnector subclass with controllable authentication."""

    CONNECTOR_ID = "reconnect_test"
    DISPLAY_NAME = "Reconnect Test Connector"
    SYNC_INTERVAL_SECONDS = 0.05  # Fast interval for tests

    # Use very short delays for tests (10ms each step)
    RECONNECT_DELAYS: list[int] = [0.01, 0.02, 0.03, 0.04]

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self.auth_results: list[bool] = []  # Pop from front on each call
        self.auth_call_count = 0
        self.sync_count = 0

    async def authenticate(self) -> bool:
        """Return next auth result from the queue, defaulting to False."""
        self.auth_call_count += 1
        if self.auth_results:
            return self.auth_results.pop(0)
        return False

    async def sync(self) -> int:
        """Track sync calls."""
        self.sync_count += 1
        return 0

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """No-op for testing."""
        return {"status": "ok"}

    async def health_check(self) -> dict[str, Any]:
        """No-op for testing."""
        return {"status": "ok"}


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def reconnect_connector(event_bus, db):
    """Create a ReconnectTestConnector with mock dependencies."""
    return ReconnectTestConnector(event_bus, db, {})


# -------------------------------------------------------------------------
# Tests: Auth failure starts reconnect loop
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_failure_starts_reconnect_loop(reconnect_connector):
    """When authenticate() fails, start() should launch a reconnect task."""
    reconnect_connector.auth_results = [False]  # First auth fails

    await reconnect_connector.start()

    # Connector should NOT be running (auth failed)
    assert reconnect_connector._running is False
    # But a reconnect task should be active
    assert reconnect_connector._reconnect_task is not None
    assert not reconnect_connector._reconnect_task.done()

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_auth_failure_sets_error_state(reconnect_connector, db):
    """When authenticate() fails, connector state should be 'error'."""
    reconnect_connector.auth_results = [False]

    await reconnect_connector.start()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status, last_error FROM connector_state WHERE connector_id = ?",
            (reconnect_connector.CONNECTOR_ID,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "error"
        assert "Authentication failed" in row["last_error"]

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_auth_failure_publishes_error_event(reconnect_connector, event_bus):
    """Auth failure should publish system.connector.error before starting reconnect."""
    reconnect_connector.auth_results = [False]

    await reconnect_connector.start()

    # Check that an error event was published
    error_events = [
        e for e in event_bus._published_events
        if e["type"] == "system.connector.error"
    ]
    assert len(error_events) >= 1
    assert error_events[0]["payload"]["error_type"] == "authentication"
    assert error_events[0]["payload"]["connector_id"] == reconnect_connector.CONNECTOR_ID

    await reconnect_connector.stop()


# -------------------------------------------------------------------------
# Tests: Successful reconnect starts sync loop
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_reconnect_starts_sync_loop(reconnect_connector):
    """After reconnect succeeds, sync loop should be running."""
    # First auth fails, second succeeds
    reconnect_connector.auth_results = [False, True]

    await reconnect_connector.start()
    # Wait for reconnect to succeed (delay is 0.01s)
    await asyncio.sleep(0.15)

    assert reconnect_connector._running is True
    assert reconnect_connector._task is not None
    assert not reconnect_connector._task.done()
    # Reconnect task should have completed and been cleared
    assert reconnect_connector._reconnect_task is None

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_successful_reconnect_resets_attempt_counter(reconnect_connector):
    """After reconnect succeeds, the attempt counter should be reset to 0."""
    # Fail twice, then succeed
    reconnect_connector.auth_results = [False, False, True]

    await reconnect_connector.start()
    await asyncio.sleep(0.2)

    assert reconnect_connector._reconnect_attempt == 0
    assert reconnect_connector._running is True

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_successful_reconnect_sets_active_state(reconnect_connector, db):
    """After reconnect succeeds, connector state should be 'active'."""
    reconnect_connector.auth_results = [False, True]

    await reconnect_connector.start()
    await asyncio.sleep(0.15)

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM connector_state WHERE connector_id = ?",
            (reconnect_connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["status"] == "active"

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_successful_reconnect_subscribes_to_actions(reconnect_connector, event_bus):
    """After reconnect succeeds, the connector should be subscribed to action events."""
    reconnect_connector.auth_results = [False, True]

    await reconnect_connector.start()
    await asyncio.sleep(0.15)

    # Verify subscription was created by checking the event_bus
    assert f"action.{reconnect_connector.CONNECTOR_ID}.*" in event_bus._subscribers

    await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_successful_reconnect_publishes_reconnected_event(reconnect_connector, event_bus):
    """After reconnect succeeds, a system.connector.reconnected event should be published."""
    reconnect_connector.auth_results = [False, True]

    await reconnect_connector.start()
    await asyncio.sleep(0.15)

    reconnected_events = [
        e for e in event_bus._published_events
        if e["type"] == "system.connector.reconnected"
    ]
    assert len(reconnected_events) == 1
    assert reconnected_events[0]["payload"]["connector_id"] == reconnect_connector.CONNECTOR_ID

    await reconnect_connector.stop()


# -------------------------------------------------------------------------
# Tests: Exponential backoff delays
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backoff_delays_follow_schedule(event_bus, db):
    """Reconnect should use increasing delays from RECONNECT_DELAYS."""
    connector = ReconnectTestConnector(event_bus, db, {})
    # All auth attempts fail — we'll track sleep calls
    connector.auth_results = [False, False, False, False]

    sleep_delays = []
    # Save a reference before patching so we can use it for outer waits
    _real_sleep = asyncio.sleep

    async def tracking_sleep(delay):
        """Track sleep durations called by the reconnect loop."""
        sleep_delays.append(delay)
        # Use near-zero sleep to keep tests fast
        await _real_sleep(0.001)

    with patch("connectors.base.connector.asyncio.sleep", side_effect=tracking_sleep):
        await connector.start()
        # Wait for several reconnect attempts using the real sleep
        await _real_sleep(0.1)

    await connector.stop()

    # Should have attempted reconnect at least once
    assert len(sleep_delays) >= 1
    # First delay should be RECONNECT_DELAYS[0]
    assert sleep_delays[0] == connector.RECONNECT_DELAYS[0]
    # Second delay (if present) should be RECONNECT_DELAYS[1]
    if len(sleep_delays) >= 2:
        assert sleep_delays[1] == connector.RECONNECT_DELAYS[1]


@pytest.mark.asyncio
async def test_backoff_caps_at_max_delay(event_bus, db):
    """After exhausting RECONNECT_DELAYS, delay should cap at the last value."""
    connector = ReconnectTestConnector(event_bus, db, {})
    # Fail many times — more than RECONNECT_DELAYS entries
    connector.auth_results = [False] * 10

    sleep_delays = []
    _real_sleep = asyncio.sleep

    async def tracking_sleep(delay):
        sleep_delays.append(delay)
        await _real_sleep(0.001)

    with patch("connectors.base.connector.asyncio.sleep", side_effect=tracking_sleep):
        await connector.start()
        await _real_sleep(0.15)

    await connector.stop()

    max_delay = connector.RECONNECT_DELAYS[-1]
    # Any delays beyond the length of RECONNECT_DELAYS should use the max
    for d in sleep_delays[len(connector.RECONNECT_DELAYS):]:
        assert d == max_delay


# -------------------------------------------------------------------------
# Tests: stop() cancels reconnect task
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_cancels_reconnect_task(reconnect_connector):
    """stop() should cancel an active reconnect task."""
    # Auth will always fail — reconnect loop runs indefinitely
    reconnect_connector.auth_results = [False]

    await reconnect_connector.start()
    assert reconnect_connector._reconnect_task is not None

    task = reconnect_connector._reconnect_task
    await reconnect_connector.stop()

    # Reconnect task should be cancelled and cleared
    assert reconnect_connector._reconnect_task is None
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_stop_sets_inactive_state_during_reconnect(reconnect_connector, db):
    """stop() during reconnect should set connector state to 'inactive'."""
    reconnect_connector.auth_results = [False]

    await reconnect_connector.start()
    await reconnect_connector.stop()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM connector_state WHERE connector_id = ?",
            (reconnect_connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["status"] == "inactive"


# -------------------------------------------------------------------------
# Tests: Happy path unchanged
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_auth_does_not_start_reconnect(reconnect_connector):
    """When authenticate() succeeds on first try, no reconnect task should be created."""
    reconnect_connector.auth_results = [True]

    await reconnect_connector.start()
    try:
        assert reconnect_connector._running is True
        assert reconnect_connector._reconnect_task is None
        assert reconnect_connector._task is not None
    finally:
        await reconnect_connector.stop()


@pytest.mark.asyncio
async def test_reconnect_handles_auth_exception(event_bus, db):
    """Reconnect loop should handle exceptions from authenticate() gracefully."""
    connector = ReconnectTestConnector(event_bus, db, {})

    call_count = 0

    async def exploding_auth():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False  # Initial auth failure triggers reconnect
        if call_count == 2:
            raise RuntimeError("Network timeout")  # Exception during reconnect
        return True  # Third attempt succeeds

    connector.authenticate = exploding_auth

    await connector.start()
    await asyncio.sleep(0.15)

    # Should have recovered despite the exception in attempt 2
    assert connector._running is True
    assert call_count >= 3

    await connector.stop()


@pytest.mark.asyncio
async def test_sync_runs_after_reconnect(reconnect_connector):
    """After successful reconnect, the sync loop should actually execute."""
    reconnect_connector.auth_results = [False, True]

    await reconnect_connector.start()
    # Wait for reconnect + a few sync cycles
    await asyncio.sleep(0.3)

    assert reconnect_connector.sync_count >= 1

    await reconnect_connector.stop()
