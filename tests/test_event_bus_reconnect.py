"""
Tests for EventBus NATS reconnection resilience.

Verifies that the EventBus configures nats-py with infinite reconnect attempts,
proper callbacks for disconnect/reconnect/error events, and exposes a
was_reconnected flag for health-check consumers.

All tests use unittest.mock to avoid requiring a live NATS server.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.event_bus.bus import EventBus


@pytest.fixture
def bus():
    """Provide an unconnected EventBus for unit testing."""
    return EventBus("nats://localhost:4222")


# ---------------------------------------------------------------------------
# Connection Configuration Tests
# ---------------------------------------------------------------------------


def _make_mock_nc():
    """Create a mock NATS connection with a synchronous jetstream() method.

    nats-py's jetstream() is a regular (non-async) method that returns a
    JetStream context object. We use MagicMock for the connection and set
    jetstream to return an AsyncMock (so its async methods like
    find_stream_name_by_subject work correctly with await).
    """
    mock_nc = MagicMock()
    mock_js = AsyncMock()
    mock_js.find_stream_name_by_subject = AsyncMock(return_value="LIFEOS")
    mock_nc.jetstream.return_value = mock_js
    return mock_nc


@pytest.mark.asyncio
async def test_connect_configures_infinite_reconnect(bus):
    """Verify connect() passes max_reconnect_attempts=-1 to nats.connect."""
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc) as mock_connect:
        await bus.connect()

        mock_connect.assert_called_once()
        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["max_reconnect_attempts"] == -1, (
            "EventBus must retry indefinitely (-1) for a 24/7 server"
        )


@pytest.mark.asyncio
async def test_connect_configures_reconnect_wait(bus):
    """Verify connect() sets reconnect_time_wait=5 seconds between attempts."""
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc) as mock_connect:
        await bus.connect()

        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["reconnect_time_wait"] == 5


@pytest.mark.asyncio
async def test_connect_registers_disconnect_callback(bus):
    """Verify connect() passes disconnected_cb to nats.connect."""
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc) as mock_connect:
        await bus.connect()

        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["disconnected_cb"] is not None
        assert callable(call_kwargs["disconnected_cb"])


@pytest.mark.asyncio
async def test_connect_registers_reconnect_callback(bus):
    """Verify connect() passes reconnected_cb to nats.connect."""
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc) as mock_connect:
        await bus.connect()

        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["reconnected_cb"] is not None
        assert callable(call_kwargs["reconnected_cb"])


@pytest.mark.asyncio
async def test_connect_registers_error_callback(bus):
    """Verify connect() passes error_cb to nats.connect."""
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc) as mock_connect:
        await bus.connect()

        call_kwargs = mock_connect.call_args.kwargs
        assert call_kwargs["error_cb"] is not None
        assert callable(call_kwargs["error_cb"])


# ---------------------------------------------------------------------------
# Callback Behavior Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_disconnect_logs_warning(bus, caplog):
    """Verify _on_disconnect() logs a warning about the paused pipeline."""
    with caplog.at_level(logging.WARNING, logger="services.event_bus.bus"):
        await bus._on_disconnect()

    assert any("NATS disconnected" in record.message for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.asyncio
async def test_on_reconnect_logs_info(bus, caplog):
    """Verify _on_reconnect() logs an info message about pipeline resumption."""
    # Set up a mock NATS connection so _on_reconnect can call jetstream()
    bus._nc = MagicMock()
    bus._nc.jetstream.return_value = MagicMock()

    with caplog.at_level(logging.INFO, logger="services.event_bus.bus"):
        await bus._on_reconnect()

    assert any("NATS reconnected" in record.message for record in caplog.records)
    assert any(record.levelno == logging.INFO for record in caplog.records)


@pytest.mark.asyncio
async def test_on_reconnect_reobtains_jetstream_context(bus):
    """Verify _on_reconnect() refreshes the JetStream context."""
    mock_nc = MagicMock()
    new_js = MagicMock()
    mock_nc.jetstream.return_value = new_js
    bus._nc = mock_nc

    old_js = bus._js  # None initially
    await bus._on_reconnect()

    # JetStream context should be refreshed
    mock_nc.jetstream.assert_called_once()
    assert bus._js is new_js
    assert bus._js is not old_js


@pytest.mark.asyncio
async def test_on_error_logs_error(bus, caplog):
    """Verify _on_error() logs the exception at ERROR level."""
    test_error = RuntimeError("test NATS error")

    with caplog.at_level(logging.ERROR, logger="services.event_bus.bus"):
        await bus._on_error(test_error)

    assert any("NATS async error" in record.message for record in caplog.records)
    assert any("test NATS error" in record.message for record in caplog.records)
    assert any(record.levelno == logging.ERROR for record in caplog.records)


# ---------------------------------------------------------------------------
# was_reconnected Flag Tests
# ---------------------------------------------------------------------------


def test_was_reconnected_initially_false(bus):
    """Verify was_reconnected is False before any reconnection."""
    assert bus.was_reconnected is False


@pytest.mark.asyncio
async def test_was_reconnected_true_after_reconnect(bus):
    """Verify was_reconnected returns True after _on_reconnect fires."""
    bus._nc = MagicMock()
    bus._nc.jetstream.return_value = MagicMock()

    await bus._on_reconnect()

    assert bus.was_reconnected is True


@pytest.mark.asyncio
async def test_was_reconnected_resets_after_read(bus):
    """Verify was_reconnected resets to False after being read once."""
    bus._nc = MagicMock()
    bus._nc.jetstream.return_value = MagicMock()

    await bus._on_reconnect()

    # First read should be True
    assert bus.was_reconnected is True
    # Second read should be False (flag resets on read)
    assert bus.was_reconnected is False


def test_reconnected_flag_initialized_in_constructor():
    """Verify the reconnected flag is initialized to False in __init__."""
    bus = EventBus("nats://test:4222")
    assert bus._reconnected_flag is False
