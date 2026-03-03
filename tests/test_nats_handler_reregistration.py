"""
Tests for NATS reconnect loop and event handler re-registration.

When NATS restarts independently of Life OS (common in Docker environments),
the event processing pipeline dies because JetStream subscriptions are lost.
The _nats_reconnect_loop background task detects this and re-registers handlers
so the system self-heals without a manual restart.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus with controllable is_connected property."""
    bus = MagicMock()
    bus.is_connected = False
    bus.subscribe_all = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def lifeos_instance(db, mock_event_bus, event_store, user_model_store):
    """Create a LifeOS instance with mocked event bus for testing reconnect logic."""
    from main import LifeOS

    config = {
        "data_dir": "./data",
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "web_host": "0.0.0.0",
        "embedding_model": "all-MiniLM-L6-v2",
        "ai": {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "mistral",
            "use_cloud": False,
        },
        "connectors": {},
    }
    instance = LifeOS(
        db=db,
        event_bus=mock_event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=config,
    )
    return instance


def _make_shutdown_sleep(shutdown_event: asyncio.Event):
    """Create an async sleep replacement that sets the shutdown event after the first call.

    This allows exactly one iteration of the reconnect loop before stopping.
    """

    async def _sleep(_duration):
        shutdown_event.set()

    return _sleep


class TestEventHandlersRegisteredFlag:
    """Tests for the _event_handlers_registered tracking flag."""

    def test_flag_starts_false(self, lifeos_instance):
        """The flag should be False on a fresh LifeOS instance."""
        assert lifeos_instance._event_handlers_registered is False

    async def test_flag_stays_false_when_nats_down_at_startup(self, lifeos_instance):
        """When NATS is not connected at startup, the flag should remain False.

        This simulates the degraded-mode startup path where event_bus.is_connected
        is False and _register_event_handlers is never called.
        """
        lifeos_instance.event_bus.is_connected = False
        # The flag should still be False because we never registered handlers
        assert lifeos_instance._event_handlers_registered is False


class TestNatsReconnectLoop:
    """Tests for the _nats_reconnect_loop background task."""

    async def test_registers_handlers_when_nats_becomes_available(self, lifeos_instance):
        """When NATS becomes connected and handlers aren't registered, the loop
        should call _register_event_handlers and set the flag to True."""
        lifeos_instance.event_bus.is_connected = True
        lifeos_instance._event_handlers_registered = False
        lifeos_instance._register_event_handlers = AsyncMock()

        sleep_fn = _make_shutdown_sleep(lifeos_instance.shutdown_event)
        with patch("asyncio.sleep", side_effect=sleep_fn):
            await lifeos_instance._nats_reconnect_loop()

        lifeos_instance._register_event_handlers.assert_called_once()
        assert lifeos_instance._event_handlers_registered is True

    async def test_resets_flag_when_nats_disconnects(self, lifeos_instance):
        """When NATS disconnects after handlers were registered, the loop should
        reset the flag so handlers will be re-registered on reconnection."""
        lifeos_instance.event_bus.is_connected = False
        lifeos_instance._event_handlers_registered = True

        sleep_fn = _make_shutdown_sleep(lifeos_instance.shutdown_event)
        with patch("asyncio.sleep", side_effect=sleep_fn):
            await lifeos_instance._nats_reconnect_loop()

        assert lifeos_instance._event_handlers_registered is False

    async def test_does_not_reregister_when_already_registered(self, lifeos_instance):
        """When handlers are already registered and NATS is connected, the loop
        should not call _register_event_handlers again (idempotency)."""
        lifeos_instance.event_bus.is_connected = True
        lifeos_instance._event_handlers_registered = True
        lifeos_instance._register_event_handlers = AsyncMock()

        sleep_fn = _make_shutdown_sleep(lifeos_instance.shutdown_event)
        with patch("asyncio.sleep", side_effect=sleep_fn):
            await lifeos_instance._nats_reconnect_loop()

        lifeos_instance._register_event_handlers.assert_not_called()
        assert lifeos_instance._event_handlers_registered is True

    async def test_does_nothing_when_nats_down_and_not_registered(self, lifeos_instance):
        """When NATS is down and handlers were never registered, the loop
        should do nothing (no error, no flag change)."""
        lifeos_instance.event_bus.is_connected = False
        lifeos_instance._event_handlers_registered = False
        lifeos_instance._register_event_handlers = AsyncMock()

        sleep_fn = _make_shutdown_sleep(lifeos_instance.shutdown_event)
        with patch("asyncio.sleep", side_effect=sleep_fn):
            await lifeos_instance._nats_reconnect_loop()

        lifeos_instance._register_event_handlers.assert_not_called()
        assert lifeos_instance._event_handlers_registered is False

    async def test_survives_handler_registration_error(self, lifeos_instance):
        """If _register_event_handlers raises an exception, the loop should
        catch it (fail-open) and continue running without crashing."""
        lifeos_instance.event_bus.is_connected = True
        lifeos_instance._event_handlers_registered = False
        lifeos_instance._register_event_handlers = AsyncMock(
            side_effect=Exception("NATS subscribe failed")
        )

        sleep_fn = _make_shutdown_sleep(lifeos_instance.shutdown_event)
        # Should not raise — the loop catches exceptions
        with patch("asyncio.sleep", side_effect=sleep_fn):
            await lifeos_instance._nats_reconnect_loop()

        lifeos_instance._register_event_handlers.assert_called_once()
        # Flag should NOT be set because registration failed
        assert lifeos_instance._event_handlers_registered is False

    async def test_full_disconnect_reconnect_cycle(self, lifeos_instance):
        """Simulate a full cycle: connected → disconnected → reconnected.
        Verify the flag transitions correctly through each state."""
        iteration = 0
        # Sequence: iteration 0 → connected (register), 1 → disconnected (reset),
        # 2 → reconnected (register again), then stop
        is_connected_sequence = [True, False, True]

        lifeos_instance._register_event_handlers = AsyncMock()
        lifeos_instance.event_bus.is_connected = is_connected_sequence[0]
        lifeos_instance._event_handlers_registered = False

        async def advancing_sleep(_duration):
            """Advance the is_connected state after each iteration."""
            nonlocal iteration
            iteration += 1
            if iteration >= len(is_connected_sequence):
                lifeos_instance.shutdown_event.set()
            else:
                lifeos_instance.event_bus.is_connected = is_connected_sequence[iteration]

        with patch("asyncio.sleep", side_effect=advancing_sleep):
            await lifeos_instance._nats_reconnect_loop()

        # Should have been called twice: once on initial connect, once on reconnect
        assert lifeos_instance._register_event_handlers.call_count == 2
        assert lifeos_instance._event_handlers_registered is True
