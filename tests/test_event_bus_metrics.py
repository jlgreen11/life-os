"""
Tests for EventBus throughput metrics instrumentation.

Verifies that publish() increments counters correctly, error conditions
increment the error counter without swallowing exceptions, get_metrics()
returns the expected shape and values, and the per-subject dict is capped
at 200 keys by evicting the least-used entry.

All tests use unittest.mock to avoid requiring a live NATS server.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from services.event_bus.bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus():
    """Provide an unconnected EventBus for unit testing."""
    return EventBus("nats://localhost:4222")


def _make_mock_nc():
    """Create a mock NATS connection backed by an AsyncMock JetStream context.

    nats-py's jetstream() is a synchronous method, so mock_nc uses MagicMock.
    The JetStream context is AsyncMock so that awaitable methods like
    find_stream_name_by_subject and publish work without a real server.
    """
    mock_nc = MagicMock()
    mock_js = AsyncMock()
    mock_js.find_stream_name_by_subject = AsyncMock(return_value="LIFEOS")
    mock_nc.jetstream.return_value = mock_js
    return mock_nc


@pytest_asyncio.fixture
async def connected_bus():
    """Provide a connected EventBus backed by a mock NATS connection.

    Patches nats.connect during connect() so the bus holds real references
    to mock_nc and mock_js.  After the patch exits the bus still works
    because its _nc and _js attributes point at the live mock objects.
    """
    bus = EventBus("nats://localhost:4222")
    mock_nc = _make_mock_nc()

    with patch("services.event_bus.bus.nats.connect", new_callable=AsyncMock, return_value=mock_nc):
        await bus.connect()

    return bus


# ---------------------------------------------------------------------------
# Publish Counter Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_increments_total_count(connected_bus):
    """Each publish() call increments _publish_count by exactly one."""
    assert connected_bus._publish_count == 0

    await connected_bus.publish("email.received", {"from": "alice@example.com"})
    assert connected_bus._publish_count == 1

    await connected_bus.publish("calendar.event", {"title": "Meeting"})
    assert connected_bus._publish_count == 2


@pytest.mark.asyncio
async def test_publish_increments_per_subject_counter(connected_bus):
    """publish() tracks counts broken down by full NATS subject."""
    await connected_bus.publish("email.received", {})
    await connected_bus.publish("email.received", {})
    await connected_bus.publish("calendar.event", {})

    assert connected_bus._publish_by_subject["lifeos.email.received"] == 2
    assert connected_bus._publish_by_subject["lifeos.calendar.event"] == 1


@pytest.mark.asyncio
async def test_publish_updates_last_publish_at(connected_bus):
    """publish() sets _last_publish_at to a timezone-aware ISO-8601 string."""
    assert connected_bus._last_publish_at is None

    await connected_bus.publish("test.event", {})

    assert connected_bus._last_publish_at is not None
    dt = datetime.fromisoformat(connected_bus._last_publish_at)
    assert dt.tzinfo is not None, "_last_publish_at must be timezone-aware"


@pytest.mark.asyncio
async def test_publish_last_publish_at_advances_over_time(connected_bus):
    """Successive publish() calls update _last_publish_at."""
    await connected_bus.publish("test.one", {})
    ts1 = connected_bus._last_publish_at

    await connected_bus.publish("test.two", {})
    ts2 = connected_bus._last_publish_at

    # ts2 >= ts1 (clock may not advance in fast tests, but value must not regress)
    assert ts2 >= ts1


# ---------------------------------------------------------------------------
# Error Counter Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_increments_error_count(connected_bus):
    """A JetStream-level failure increments _publish_error_count."""
    connected_bus._js.publish.side_effect = RuntimeError("NATS publish failed")

    with pytest.raises(RuntimeError):
        await connected_bus.publish("test.event", {})

    assert connected_bus._publish_error_count == 1


@pytest.mark.asyncio
async def test_successful_publish_does_not_increment_error_count(connected_bus):
    """A successful publish leaves _publish_error_count at zero."""
    await connected_bus.publish("test.event", {})

    assert connected_bus._publish_error_count == 0


@pytest.mark.asyncio
async def test_error_still_increments_publish_count(connected_bus):
    """A failed publish still increments _publish_count (attempted call)."""
    connected_bus._js.publish.side_effect = RuntimeError("NATS publish failed")

    with pytest.raises(RuntimeError):
        await connected_bus.publish("test.event", {})

    # Total count tracks attempts, not just successes.
    assert connected_bus._publish_count == 1


@pytest.mark.asyncio
async def test_error_re_raises_exception(connected_bus):
    """publish() re-raises the original exception after counting it."""
    connected_bus._js.publish.side_effect = ConnectionError("stream unavailable")

    with pytest.raises(ConnectionError, match="stream unavailable"):
        await connected_bus.publish("test.event", {})


@pytest.mark.asyncio
async def test_multiple_errors_accumulate(connected_bus):
    """Multiple failures accumulate in _publish_error_count."""
    connected_bus._js.publish.side_effect = RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await connected_bus.publish("test.event", {})

    assert connected_bus._publish_error_count == 3
    assert connected_bus._publish_count == 3


# ---------------------------------------------------------------------------
# get_metrics() Shape and Value Tests
# ---------------------------------------------------------------------------


def test_get_metrics_returns_expected_keys(bus):
    """get_metrics() returns a dict with exactly the documented keys."""
    metrics = bus.get_metrics()

    expected_keys = {
        "total_published",
        "total_errors",
        "active_subscriptions",
        "last_publish_at",
        "started_at",
        "uptime_seconds",
        "top_subjects",
        "error_rate",
        "is_connected",
    }
    assert set(metrics.keys()) == expected_keys


def test_get_metrics_initial_values(bus):
    """get_metrics() returns sensible zero/None defaults for a fresh bus."""
    metrics = bus.get_metrics()

    assert metrics["total_published"] == 0
    assert metrics["total_errors"] == 0
    assert metrics["active_subscriptions"] == 0
    assert metrics["last_publish_at"] is None
    assert metrics["started_at"] is None
    assert metrics["uptime_seconds"] == 0
    assert metrics["top_subjects"] == {}
    assert metrics["error_rate"] == 0.0
    assert metrics["is_connected"] is False


@pytest.mark.asyncio
async def test_get_metrics_reflects_publishes(connected_bus):
    """get_metrics() accurately reflects the counts from publish() calls."""
    await connected_bus.publish("email.received", {})
    await connected_bus.publish("email.received", {})
    await connected_bus.publish("calendar.event", {})

    metrics = connected_bus.get_metrics()

    assert metrics["total_published"] == 3
    assert metrics["total_errors"] == 0
    assert metrics["error_rate"] == 0.0
    assert metrics["top_subjects"].get("lifeos.email.received") == 2
    assert metrics["top_subjects"].get("lifeos.calendar.event") == 1


@pytest.mark.asyncio
async def test_get_metrics_error_rate_calculation(connected_bus):
    """get_metrics() computes error_rate as errors / total_published."""
    # Arrange: first call raises, next three succeed.
    connected_bus._js.publish.side_effect = [
        RuntimeError("fail"),
        None,
        None,
        None,
    ]

    with pytest.raises(RuntimeError):
        await connected_bus.publish("test.event", {})
    await connected_bus.publish("test.event", {})
    await connected_bus.publish("test.event", {})
    await connected_bus.publish("test.event", {})

    metrics = connected_bus.get_metrics()
    assert metrics["total_published"] == 4
    assert metrics["total_errors"] == 1
    assert metrics["error_rate"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_get_metrics_started_at_set_after_connect(connected_bus):
    """get_metrics() includes a parseable started_at after connect()."""
    metrics = connected_bus.get_metrics()

    assert metrics["started_at"] is not None
    dt = datetime.fromisoformat(metrics["started_at"])
    assert dt.tzinfo is not None, "started_at must be timezone-aware"


@pytest.mark.asyncio
async def test_get_metrics_uptime_seconds_nonnegative(connected_bus):
    """uptime_seconds is >= 0 immediately after connect()."""
    metrics = connected_bus.get_metrics()

    assert metrics["uptime_seconds"] >= 0.0


def test_get_metrics_uptime_zero_before_connect(bus):
    """uptime_seconds is 0 when the bus has never connected."""
    metrics = bus.get_metrics()

    assert metrics["uptime_seconds"] == 0


@pytest.mark.asyncio
async def test_get_metrics_top_subjects_limited_to_20(connected_bus):
    """top_subjects in get_metrics() contains at most 20 entries."""
    # Publish to 25 different subjects.
    for i in range(25):
        await connected_bus.publish(f"test.subject{i}", {})

    metrics = connected_bus.get_metrics()

    assert len(metrics["top_subjects"]) <= 20


@pytest.mark.asyncio
async def test_get_metrics_top_subjects_sorted_descending(connected_bus):
    """top_subjects values are ordered from highest to lowest count."""
    for _ in range(5):
        await connected_bus.publish("email.received", {})
    for _ in range(3):
        await connected_bus.publish("calendar.event", {})
    for _ in range(1):
        await connected_bus.publish("task.created", {})

    metrics = connected_bus.get_metrics()
    counts = list(metrics["top_subjects"].values())

    assert counts == sorted(counts, reverse=True)


@pytest.mark.asyncio
async def test_get_metrics_is_connected_reflects_state(connected_bus):
    """get_metrics() is_connected matches the bus is_connected property."""
    metrics = connected_bus.get_metrics()

    assert metrics["is_connected"] == connected_bus.is_connected


# ---------------------------------------------------------------------------
# Subject Counter Cap Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_by_subject_never_exceeds_200_keys(connected_bus):
    """_publish_by_subject is capped at 200 keys even with 250 distinct subjects."""
    for i in range(250):
        await connected_bus.publish(f"test.subject{i}", {})

    assert len(connected_bus._publish_by_subject) <= 200


@pytest.mark.asyncio
async def test_publish_by_subject_cap_evicts_least_used(connected_bus):
    """When the cap is enforced, the high-frequency subject is NOT evicted."""
    # Fill to exactly 200 subjects (each with count=1).
    for i in range(200):
        await connected_bus.publish(f"rare.subject{i}", {})

    assert len(connected_bus._publish_by_subject) == 200

    # Publish 10 times to a new "frequent" subject — this triggers the 201st
    # distinct key on the first call, forcing an eviction of the least-used
    # entry (one of the "rare" subjects with count=1).
    for _ in range(10):
        await connected_bus.publish("frequent.subject", {})

    # Cap is maintained.
    assert len(connected_bus._publish_by_subject) <= 200

    # The high-frequency subject must survive eviction.
    assert "lifeos.frequent.subject" in connected_bus._publish_by_subject
    assert connected_bus._publish_by_subject["lifeos.frequent.subject"] == 10


@pytest.mark.asyncio
async def test_total_count_unaffected_by_cap(connected_bus):
    """_publish_count grows unboundedly even when the subject dict is capped."""
    for i in range(250):
        await connected_bus.publish(f"test.subject{i}", {})

    # All 250 publish attempts are counted regardless of the dict cap.
    assert connected_bus._publish_count == 250
    # But the dict stays within bounds.
    assert len(connected_bus._publish_by_subject) <= 200


# ---------------------------------------------------------------------------
# Subscribe Counter Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_increments_subscribe_count(connected_bus):
    """subscribe() increments _subscribe_count for each successful call."""
    assert connected_bus._subscribe_count == 0

    async def handler(event):
        pass

    await connected_bus.subscribe("email.*", handler, durable=False)
    assert connected_bus._subscribe_count == 1

    await connected_bus.subscribe("calendar.*", handler, durable=False)
    assert connected_bus._subscribe_count == 2


@pytest.mark.asyncio
async def test_subscribe_all_increments_subscribe_count(connected_bus):
    """subscribe_all() increments _subscribe_count (delegates to subscribe)."""
    assert connected_bus._subscribe_count == 0

    async def handler(event):
        pass

    await connected_bus.subscribe_all(handler, consumer_name="test-all")
    assert connected_bus._subscribe_count == 1


@pytest.mark.asyncio
async def test_subscribe_count_reflected_in_metrics(connected_bus):
    """get_metrics() active_subscriptions matches _subscribe_count."""
    async def handler(event):
        pass

    await connected_bus.subscribe("email.*", handler, durable=False)
    await connected_bus.subscribe("calendar.*", handler, durable=False)

    metrics = connected_bus.get_metrics()
    assert metrics["active_subscriptions"] == 2
