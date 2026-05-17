"""
Tests for EventBus throughput counters and get_metrics() introspection.

The bus is the central pipeline artery — when events stop flowing, operators
need to distinguish "connector not producing" from "bus dropping messages".
These tests verify that publish/consume counters, error counters, last-seen
timestamps, rolling rates, and the get_metrics() shape all behave correctly.

All tests mock the NATS layer to avoid requiring a live server.
"""

import asyncio
import logging
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.event_bus.bus import EventBus, METRICS_DEQUE_MAXLEN


def _make_mock_nc():
    """Build a mock NATS connection with a working jetstream() context.

    Mirrors the helper in test_event_bus_reconnect.py so these tests don't
    require a running NATS server.
    """
    mock_nc = MagicMock()
    mock_nc.is_connected = True
    mock_js = AsyncMock()
    mock_js.find_stream_name_by_subject = AsyncMock(return_value="LIFEOS")
    mock_js.publish = AsyncMock()
    mock_nc.jetstream.return_value = mock_js
    return mock_nc


@pytest.fixture
def bus():
    """Provide a fresh, unconnected EventBus for counter assertions."""
    return EventBus("nats://localhost:4222")


# ---------------------------------------------------------------------------
# Counter initialization
# ---------------------------------------------------------------------------


def test_counters_start_empty(bus):
    """Verify all metric structures are initialized empty on construction."""
    assert dict(bus.publish_total) == {}
    assert dict(bus.publish_errors) == {}
    assert dict(bus.consume_total) == {}
    assert bus.last_publish_at == {}
    assert bus.last_consume_at == {}


@pytest.mark.asyncio
async def test_connect_logs_metrics_enabled(bus, caplog):
    """Verify a single info log line confirms counters are active on startup."""
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        with caplog.at_level(logging.INFO, logger="services.event_bus.bus"):
            await bus.connect()

    enabled_logs = [r for r in caplog.records if "metrics enabled" in r.message]
    assert len(enabled_logs) == 1
    assert enabled_logs[0].levelno == logging.INFO


# ---------------------------------------------------------------------------
# Publish path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_bumps_publish_total_and_last_publish_at(bus):
    """Publishing increments per-subject totals and stamps the timestamp."""
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    await bus.publish("email.received", {"from": "alice"})
    await bus.publish("email.received", {"from": "bob"})
    await bus.publish("calendar.created", {"title": "lunch"})

    assert bus.publish_total["lifeos.email.received"] == 2
    assert bus.publish_total["lifeos.calendar.created"] == 1
    # Timestamps are ISO 8601 strings (JSON-serializable).
    assert isinstance(bus.last_publish_at["lifeos.email.received"], str)
    assert "T" in bus.last_publish_at["lifeos.email.received"]


@pytest.mark.asyncio
async def test_publish_failure_bumps_errors_and_reraises(bus):
    """A NATS publish failure increments publish_errors and propagates."""
    mock_nc = _make_mock_nc()
    # Make the JS publish raise the second time it's called.
    mock_js = mock_nc.jetstream.return_value
    mock_js.publish = AsyncMock(side_effect=[None, RuntimeError("broker down")])

    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    # First publish succeeds.
    await bus.publish("email.received", {"from": "alice"})
    assert bus.publish_total["lifeos.email.received"] == 1
    assert bus.publish_errors["lifeos.email.received"] == 0

    # Second publish raises — but the counter must still be bumped.
    with pytest.raises(RuntimeError, match="broker down"):
        await bus.publish("email.received", {"from": "bob"})

    assert bus.publish_errors["lifeos.email.received"] == 1
    # The successful-publish counter MUST NOT be bumped on failure.
    assert bus.publish_total["lifeos.email.received"] == 1


@pytest.mark.asyncio
async def test_serialization_failure_does_not_count_as_error(bus):
    """Non-JSON-serializable payloads raise before the bus is touched.

    These never reach NATS, so they're not 'bus errors' — they're caller
    bugs. The publish_errors counter is reserved for NATS-side failures.
    """
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    with pytest.raises(TypeError):
        await bus.publish("email.received", {"data": {1, 2, 3}})

    assert bus.publish_total == {}
    assert bus.publish_errors == {}


# ---------------------------------------------------------------------------
# Consume path (via the subscribe wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_bumps_counters_via_wrapped_handler(bus):
    """Verify the subscribe wrapper bumps consume_total + timestamp."""
    mock_nc = _make_mock_nc()
    mock_js = mock_nc.jetstream.return_value

    # Capture the wrapper callback that subscribe() registers with JetStream.
    captured = {}

    async def fake_subscribe(subject, cb=None, config=None):
        captured["cb"] = cb
        captured["subject"] = subject
        return MagicMock(unsubscribe=AsyncMock())

    mock_js.subscribe = fake_subscribe

    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    delivered = []

    async def user_handler(event):
        delivered.append(event)

    await bus.subscribe("email.*", user_handler, durable=False)

    # Simulate JetStream delivering a message: build a mock msg and invoke
    # the captured wrapper as JetStream would.
    msg = MagicMock()
    msg.subject = "lifeos.email.received"
    msg.data = b'{"id": "1", "type": "email.received", "payload": {}}'
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()

    await captured["cb"](msg)

    assert bus.consume_total["lifeos.email.received"] == 1
    assert isinstance(bus.last_consume_at["lifeos.email.received"], str)
    assert delivered and delivered[0]["id"] == "1"
    msg.ack.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_counter_bumps_even_when_handler_raises(bus):
    """A crashing user handler still counts as 'delivered' for observability.

    Operators distinguishing 'bus delivered nothing' from 'handler is buggy'
    need consume_total to reflect bus traffic, not handler success.
    """
    mock_nc = _make_mock_nc()
    mock_js = mock_nc.jetstream.return_value

    captured = {}

    async def fake_subscribe(subject, cb=None, config=None):
        captured["cb"] = cb
        return MagicMock(unsubscribe=AsyncMock())

    mock_js.subscribe = fake_subscribe

    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    async def failing_handler(event):
        raise ValueError("oops")

    await bus.subscribe("email.*", failing_handler, durable=False)

    msg = MagicMock()
    msg.subject = "lifeos.email.received"
    msg.data = b'{"id": "1", "payload": {}}'
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()

    await captured["cb"](msg)

    assert bus.consume_total["lifeos.email.received"] == 1
    msg.nak.assert_awaited_once()  # Failure path triggers NAK, not ACK.
    msg.ack.assert_not_called()


# ---------------------------------------------------------------------------
# get_metrics() shape and content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_metrics_returns_expected_shape(bus):
    """Verify get_metrics() returns the documented JSON-serializable shape."""
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    await bus.publish("email.received", {"from": "alice"})
    await bus.publish("email.received", {"from": "bob"})
    await bus.publish("calendar.created", {"title": "lunch"})

    metrics = await bus.get_metrics()

    # Top-level keys.
    assert set(metrics.keys()) == {"connected", "window_seconds", "subjects", "totals"}
    assert metrics["connected"] is True
    assert metrics["window_seconds"] == 60
    assert metrics["totals"]["publish_total"] == 3
    assert metrics["totals"]["publish_errors"] == 0
    assert metrics["totals"]["consume_total"] == 0

    # Per-subject breakdown.
    email = metrics["subjects"]["lifeos.email.received"]
    assert email["publish_total"] == 2
    assert email["publish_errors"] == 0
    assert email["consume_total"] == 0
    assert email["last_publish_at"] is not None
    assert email["last_consume_at"] is None
    assert email["publishes_per_minute"] > 0

    cal = metrics["subjects"]["lifeos.calendar.created"]
    assert cal["publish_total"] == 1


@pytest.mark.asyncio
async def test_get_metrics_is_json_serializable(bus):
    """The metrics dict must be JSON-serializable for /health responses."""
    import json
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    await bus.publish("email.received", {"from": "alice"})

    metrics = await bus.get_metrics()
    serialized = json.dumps(metrics)
    assert "lifeos.email.received" in serialized


@pytest.mark.asyncio
async def test_get_metrics_reports_connection_state(bus):
    """connected reflects the underlying NATS socket state."""
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    metrics = await bus.get_metrics()
    assert metrics["connected"] is True

    # Simulate the underlying socket dropping.
    mock_nc.is_connected = False
    metrics = await bus.get_metrics()
    assert metrics["connected"] is False


# ---------------------------------------------------------------------------
# Rolling rate window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rolling_rate_increases_within_window(bus):
    """Within the rolling window, each publish increases the per-minute rate.

    The rate is monotonically non-decreasing as long as we're still inside
    the window — older timestamps haven't aged out yet.
    """
    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    rates = []
    for _ in range(5):
        await bus.publish("email.received", {"x": 1})
        m = await bus.get_metrics()
        rates.append(m["subjects"]["lifeos.email.received"]["publishes_per_minute"])

    # Rate must be non-decreasing as long as we stay inside the window.
    for prev, curr in zip(rates, rates[1:]):
        assert curr >= prev, f"rate dropped within window: {rates}"
    # And after 5 publishes within a 60s window, the rate is strictly > 0.
    assert rates[-1] > 0


def test_rate_per_minute_drops_entries_outside_window(bus):
    """Timestamps older than the window must not contribute to the rate."""
    now = time.monotonic()
    window = deque(maxlen=METRICS_DEQUE_MAXLEN)
    # One old entry (well outside the 60s window), two fresh.
    window.append(now - 120)
    window.append(now - 1)
    window.append(now - 0.5)

    rate = bus._rate_per_minute(window, now)
    # Only 2 entries within the window — rate = 2 * (60/60) = 2.0/min.
    assert rate == pytest.approx(2.0)


def test_rate_per_minute_empty_window(bus):
    """An empty deque must return 0.0, not raise."""
    assert bus._rate_per_minute(deque(), time.monotonic()) == 0.0


def test_publish_window_deque_is_bounded(bus):
    """Per-subject deques must cap at METRICS_DEQUE_MAXLEN to bound memory."""
    # Accessing through defaultdict triggers the factory.
    d = bus._publish_window["lifeos.spam"]
    assert d.maxlen == METRICS_DEQUE_MAXLEN

    # Hammer past the cap — deque drops oldest, never grows past maxlen.
    for i in range(METRICS_DEQUE_MAXLEN + 50):
        d.append(float(i))
    assert len(d) == METRICS_DEQUE_MAXLEN


# ---------------------------------------------------------------------------
# Custom window configuration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_window_seconds():
    """The metrics window is configurable via constructor."""
    bus = EventBus("nats://localhost:4222", metrics_window_seconds=30)
    assert bus.metrics_window_seconds == 30

    mock_nc = _make_mock_nc()
    with patch("services.event_bus.bus.nats.connect",
               new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    metrics = await bus.get_metrics()
    assert metrics["window_seconds"] == 30
