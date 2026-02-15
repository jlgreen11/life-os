"""
Comprehensive test coverage for EventBus (NATS JetStream wrapper).

The EventBus is the central nervous system of Life OS, routing 530K+ events/day.
Tests verify connection lifecycle, publishing, subscriptions, and error handling.
"""

import asyncio
import json
import uuid

import pytest
import pytest_asyncio

from services.event_bus.bus import EventBus


@pytest_asyncio.fixture
async def event_bus():
    """Provide a connected EventBus instance with automatic cleanup."""
    bus = EventBus("nats://localhost:4222")
    await bus.connect()
    yield bus
    await bus.disconnect()


@pytest_asyncio.fixture
async def second_bus():
    """Provide a second EventBus for multi-client tests."""
    bus = EventBus("nats://localhost:4222")
    await bus.connect()
    yield bus
    await bus.disconnect()


def unique_subject():
    """Generate unique subject to avoid message replay."""
    return f"test{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Connection Lifecycle Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_establishes_connection():
    """Verify connect() creates a live NATS connection."""
    bus = EventBus("nats://localhost:4222")
    assert not bus.is_connected
    
    await bus.connect()
    assert bus.is_connected
    
    await bus.disconnect()


@pytest.mark.asyncio
async def test_connect_creates_jetstream_stream():
    """Verify connect() ensures the LIFEOS stream exists."""
    bus = EventBus("nats://localhost:4222")
    await bus.connect()
    
    stream_name = await bus._js.find_stream_name_by_subject("lifeos.>")
    assert stream_name == "LIFEOS"
    
    await bus.disconnect()


@pytest.mark.asyncio
async def test_disconnect_closes_connection():
    """Verify disconnect() tears down the NATS connection."""
    bus = EventBus("nats://localhost:4222")
    await bus.connect()
    assert bus.is_connected
    
    await bus.disconnect()
    assert not bus.is_connected


# ---------------------------------------------------------------------------
# Event Publishing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_returns_event_id(event_bus):
    """Verify publish() returns a unique event ID."""
    event_id = await event_bus.publish(unique_subject(), {"data": "test"})
    
    assert isinstance(event_id, str)
    assert len(event_id) == 36  # UUID4 format
    uuid.UUID(event_id)  # Raises if not valid


@pytest.mark.asyncio
async def test_publish_creates_event_envelope(event_bus):
    """Verify publish() creates a standard event envelope."""
    subject = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event)
    
    await event_bus.subscribe(subject, handler, durable=False)
    await asyncio.sleep(0.1)
    
    event_id = await event_bus.publish(
        subject,
        {"key": "value"},
        source="test-service",
        priority="high",
        metadata={"domain": "test"}
    )
    
    await asyncio.sleep(0.2)
    
    assert len(received) >= 1
    event = received[0]
    
    assert event["id"] == event_id
    assert event["type"] == subject
    assert event["source"] == "test-service"
    assert event["priority"] == "high"
    assert event["payload"] == {"key": "value"}
    assert event["metadata"] == {"domain": "test"}
    assert "timestamp" in event


@pytest.mark.asyncio
async def test_publish_default_values(event_bus):
    """Verify publish() uses sensible defaults."""
    subject = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event)
    
    await event_bus.subscribe(subject, handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(subject, {"data": "test"})
    await asyncio.sleep(0.2)
    
    event = received[0]
    assert event["source"] == "system"
    assert event["priority"] == "normal"
    assert event["metadata"] == {}


# ---------------------------------------------------------------------------
# Event Subscription Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_receives_published_events(event_bus):
    """Verify basic pub/sub delivery."""
    subject = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event)
    
    await event_bus.subscribe(subject, handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(subject, {"value": 42})
    await asyncio.sleep(0.2)
    
    assert len(received) >= 1
    assert received[0]["payload"]["value"] == 42


@pytest.mark.asyncio
async def test_subscribe_pattern_matching_wildcard(event_bus):
    """Verify * wildcard matches single segment."""
    prefix = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event["type"])
    
    await event_bus.subscribe(f"{prefix}.*", handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(f"{prefix}.one", {})
    await event_bus.publish(f"{prefix}.two", {})
    await event_bus.publish(f"{prefix}.deep.nested", {})
    
    await asyncio.sleep(0.2)
    
    assert f"{prefix}.one" in received
    assert f"{prefix}.two" in received
    assert f"{prefix}.deep.nested" not in received


@pytest.mark.asyncio
async def test_subscribe_pattern_matching_multi_wildcard(event_bus):
    """Verify > wildcard matches all segments."""
    prefix = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event["type"])
    
    await event_bus.subscribe(f"{prefix}.>", handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(f"{prefix}.one", {})
    await event_bus.publish(f"{prefix}.deep.nested", {})
    await event_bus.publish(f"{prefix}.very.deep.path", {})
    
    await asyncio.sleep(0.2)
    
    assert f"{prefix}.one" in received
    assert f"{prefix}.deep.nested" in received
    assert f"{prefix}.very.deep.path" in received


@pytest.mark.asyncio
async def test_subscribe_multiple_handlers_same_pattern(event_bus):
    """Verify multiple handlers can subscribe to the same pattern."""
    subject = unique_subject()
    received1 = []
    received2 = []
    
    async def handler1(event):
        received1.append(event)
    
    async def handler2(event):
        received2.append(event)
    
    await event_bus.subscribe(subject, handler1, consumer_name=f"h1{uuid.uuid4().hex[:8]}", durable=False)
    await event_bus.subscribe(subject, handler2, consumer_name=f"h2{uuid.uuid4().hex[:8]}", durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(subject, {"value": "shared"})
    await asyncio.sleep(0.2)
    
    assert len(received1) >= 1
    assert len(received2) >= 1
    assert received1[0]["payload"]["value"] == "shared"
    assert received2[0]["payload"]["value"] == "shared"


@pytest.mark.asyncio
async def test_subscribe_across_multiple_buses(second_bus, event_bus):
    """Verify pub/sub works across different connections."""
    subject = unique_subject()
    received = []
    
    async def handler(event):
        received.append(event)
    
    await event_bus.subscribe(subject, handler, durable=False)
    await asyncio.sleep(0.1)
    
    await second_bus.publish(subject, {"from": "bus2"})
    await asyncio.sleep(0.2)
    
    assert len(received) >= 1
    assert received[0]["payload"]["from"] == "bus2"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_exception_triggers_nak(event_bus):
    """Verify handler exceptions trigger NAK for redelivery."""
    subject = unique_subject()
    call_count = 0
    
    async def failing_handler(event):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated failure")
    
    await event_bus.subscribe(subject, failing_handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(subject, {"test": "retry"})
    await asyncio.sleep(6)  # Wait for NAK + redelivery
    
    assert call_count >= 2  # Initial + at least one retry


@pytest.mark.asyncio
async def test_publish_with_invalid_payload_raises_error(event_bus):
    """Verify publish rejects non-serializable payloads."""
    with pytest.raises(TypeError):
        await event_bus.publish(unique_subject(), {"data": {1, 2, 3}})  # Sets aren't JSON-serializable


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_connected_reflects_state():
    """Verify is_connected property tracks connection lifecycle."""
    bus = EventBus("nats://localhost:4222")
    
    assert not bus.is_connected
    
    await bus.connect()
    assert bus.is_connected
    
    await bus.disconnect()
    assert not bus.is_connected


@pytest.mark.asyncio
async def test_is_connected_false_before_connect():
    """Verify is_connected is False for uninitialized bus."""
    bus = EventBus("nats://localhost:4222")
    assert not bus.is_connected


# ---------------------------------------------------------------------------
# Message Delivery Guarantees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_published_events_persisted_to_stream(event_bus):
    """Verify published events are durably stored in JetStream."""
    await event_bus.publish(unique_subject(), {"data": "durable"})
    
    stream_info = await event_bus._js.stream_info("LIFEOS")
    assert stream_info.state.messages > 0


@pytest.mark.asyncio
async def test_acknowledgment_prevents_redelivery(event_bus):
    """Verify ACK'd messages are not redelivered."""
    subject = unique_subject()
    call_count = 0
    
    async def handler(event):
        nonlocal call_count
        call_count += 1
    
    await event_bus.subscribe(subject, handler, durable=False)
    await asyncio.sleep(0.1)
    
    await event_bus.publish(subject, {"test": "once"})
    await asyncio.sleep(6)  # Wait longer than NAK delay
    
    # Should be called exactly once (no redelivery after ACK)
    assert call_count == 1
