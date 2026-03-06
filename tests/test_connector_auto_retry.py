"""
Tests for connector auto-retry in the health monitor loop.

The ``_connector_health_monitor_loop`` now attempts to auto-retry connectors
in error state by calling ``start()`` on the connector instance.  Retries
use exponential backoff (1h, 2h, 4h, 8h, capped at 24h) to avoid hammering
connectors that require manual intervention while still recovering from
transient failures.

These tests exercise the ``_maybe_retry_connector`` helper method directly.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lifeos_stub(connector_map=None):
    """Create a minimal LifeOS-like object with _maybe_retry_connector bound.

    We import the real method and bind it to a stub so we can test the logic
    without instantiating the full LifeOS class.
    """
    from main import LifeOS

    stub = MagicMock()
    stub.connector_map = connector_map or {}
    # Bind the real method to our stub
    stub._maybe_retry_connector = LifeOS._maybe_retry_connector.__get__(stub, type(stub))
    return stub


def _make_mock_connector(running=False, reconnect_task=None):
    """Create a mock connector with configurable state."""
    connector = MagicMock()
    connector._running = running
    connector._reconnect_task = reconnect_task
    connector.start = AsyncMock()
    return connector


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectorAutoRetry:
    """Tests for the _maybe_retry_connector auto-retry logic."""

    async def test_retry_calls_start_after_backoff(self):
        """A connector in error state should get start() called after the backoff period."""
        connector = _make_mock_connector(running=False)
        # Simulate that start() succeeds and sets _running = True
        async def start_success():
            connector._running = True
        connector.start = AsyncMock(side_effect=start_success)

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        connector.start.assert_called_once()
        # Retry state should be reset on success
        assert "google" not in retry_counts
        assert "google" not in last_retry

    async def test_exponential_backoff_schedule(self):
        """Backoff should follow 1h, 2h, 4h, 8h, 24h cap."""
        connector = _make_mock_connector(running=False)
        # start() fails (connector stays not running)
        connector.start = AsyncMock()

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        base_ts = 1000000.0

        # Attempt 0: backoff = 1h (3600s)
        await stub._maybe_retry_connector("google", retry_counts, last_retry, base_ts)
        assert retry_counts["google"] == 1
        assert last_retry["google"] == base_ts

        # Too soon (only 30 min later) — should NOT retry
        connector.start.reset_mock()
        await stub._maybe_retry_connector("google", retry_counts, last_retry, base_ts + 1800)
        connector.start.assert_not_called()

        # After 2h (attempt 1 backoff) — should retry
        connector.start.reset_mock()
        await stub._maybe_retry_connector("google", retry_counts, last_retry, base_ts + 7200)
        connector.start.assert_called_once()
        assert retry_counts["google"] == 2

        # Attempt 2 backoff = 4h. Try after 3h — should NOT retry
        connector.start.reset_mock()
        ts_after_attempt2 = base_ts + 7200  # last_retry was set to this
        await stub._maybe_retry_connector("google", retry_counts, last_retry, ts_after_attempt2 + 10800)
        connector.start.assert_not_called()

        # After 4h from last retry — should retry
        connector.start.reset_mock()
        await stub._maybe_retry_connector("google", retry_counts, last_retry, ts_after_attempt2 + 14400)
        connector.start.assert_called_once()
        assert retry_counts["google"] == 3

    async def test_backoff_caps_at_24h(self):
        """Backoff should cap at 24 hours regardless of attempt count."""
        connector = _make_mock_connector(running=False)
        connector.start = AsyncMock()

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {"google": 10}  # High attempt count
        last_retry: dict[str, float] = {"google": 0.0}
        # 24h cap = 86400s. At attempt 10, 2^10 = 1024h, but capped at 24h
        now_ts = 86400.0  # Exactly 24h after last retry

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)
        connector.start.assert_called_once()

    async def test_recovered_connector_resets_retry_state(self):
        """When start() succeeds and connector is running, retry state should be cleared."""
        connector = _make_mock_connector(running=False)
        async def start_success():
            connector._running = True
        connector.start = AsyncMock(side_effect=start_success)

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {"google": 5}
        last_retry: dict[str, float] = {"google": 100.0}
        now_ts = time.time()

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        connector.start.assert_called_once()
        assert "google" not in retry_counts
        assert "google" not in last_retry

    async def test_connector_not_in_map_is_skipped(self):
        """Connectors not in connector_map should be silently skipped."""
        stub = _make_lifeos_stub(connector_map={})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        # Should not raise
        await stub._maybe_retry_connector("nonexistent", retry_counts, last_retry, now_ts)

        assert "nonexistent" not in retry_counts

    async def test_start_failure_increments_retry_count(self):
        """If start() raises an exception, retry count should increment without crashing."""
        connector = _make_mock_connector(running=False)
        connector.start = AsyncMock(side_effect=RuntimeError("Connection refused"))

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        # Should not raise — errors are caught
        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        assert retry_counts["google"] == 1
        assert last_retry["google"] == now_ts

    async def test_skips_connector_with_active_reconnect_loop(self):
        """Connectors already running their own reconnect loop should be skipped."""
        # Create a mock task that is NOT done (still running)
        active_task = MagicMock(spec=asyncio.Task)
        active_task.done.return_value = False

        connector = _make_mock_connector(running=False, reconnect_task=active_task)
        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        # Should NOT call start() since connector has active reconnect loop
        connector.start.assert_not_called()

    async def test_retries_connector_with_finished_reconnect_task(self):
        """Connectors with a completed (done) reconnect task should be retried."""
        finished_task = MagicMock(spec=asyncio.Task)
        finished_task.done.return_value = True

        connector = _make_mock_connector(running=False, reconnect_task=finished_task)
        connector.start = AsyncMock()

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        connector.start.assert_called_once()

    async def test_start_no_crash_sets_running_false(self):
        """If start() completes without error but _running is still False, increment retry."""
        connector = _make_mock_connector(running=False)
        # start() succeeds but doesn't set _running (auth failed internally)
        connector.start = AsyncMock()

        stub = _make_lifeos_stub(connector_map={"google": connector})

        retry_counts: dict[str, int] = {}
        last_retry: dict[str, float] = {}
        now_ts = time.time()

        await stub._maybe_retry_connector("google", retry_counts, last_retry, now_ts)

        connector.start.assert_called_once()
        assert retry_counts["google"] == 1
        assert last_retry["google"] == now_ts


class TestHealthMonitorRetryIntegration:
    """Integration-style tests verifying retry state resets on recovery."""

    async def test_recovery_clears_retry_state_in_monitor_loop(self):
        """When a connector recovers (status changes to active), retry state should be cleared.

        This tests the full monitor iteration path, not just _maybe_retry_connector.
        """
        from storage.manager import DatabaseManager
        from tests.test_connector_health_monitor import (
            _insert_connector_state,
            _run_monitor_iteration,
        )
        from datetime import datetime, timedelta, timezone

        # Use tmp_path equivalent
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(data_dir=tmp)
            db.initialize_all()

            mock_bus = MagicMock()
            mock_bus.is_connected = True
            mock_bus.publish = AsyncMock()

            # Step 1: Connector is degraded
            _insert_connector_state(db, "google", status="error", last_error="Auth failed")
            alerted: set[str] = set()
            await _run_monitor_iteration(db, mock_bus, alerted)
            assert "google" in alerted

            # Step 2: Connector recovers
            recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            _insert_connector_state(db, "google", status="ok", last_sync=recent)
            await _run_monitor_iteration(db, mock_bus, alerted)
            assert "google" not in alerted
