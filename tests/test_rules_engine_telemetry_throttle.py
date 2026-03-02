"""
Life OS — Rules Engine Telemetry Throttle Tests

Verifies that system.rule.triggered telemetry events are throttled on a
per-rule basis to prevent the events table from being flooded with
high-frequency rule telemetry (previously 87% of all events).

The throttle only limits event-bus publication — the rules table
(times_triggered, last_triggered) is always updated on every trigger.
"""

from unittest.mock import AsyncMock, patch

import pytest

from services.rules_engine.engine import RulesEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(db, event_bus=None, throttle_seconds=300):
    """Create a RulesEngine with a configurable throttle interval."""
    return RulesEngine(
        db,
        event_bus=event_bus,
        config={"telemetry_throttle_seconds": throttle_seconds},
    )


def _mock_event_bus():
    """Create a minimal mock event bus with a trackable publish method."""
    bus = AsyncMock()
    bus.is_connected = True
    return bus


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_trigger_publishes_telemetry(db):
    """The first trigger for a rule should always publish a telemetry event."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    await engine.add_rule(
        name="Test rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    await engine.evaluate(event)

    # The bus should have received system.rule.triggered (plus the
    # system.rule.created from add_rule — filter for triggered only).
    triggered_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.triggered"
    ]
    assert len(triggered_calls) == 1, (
        f"Expected exactly 1 system.rule.triggered publish, got {len(triggered_calls)}"
    )


@pytest.mark.asyncio
async def test_second_trigger_within_window_is_throttled(db):
    """A second trigger within the throttle window should NOT publish telemetry."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    await engine.add_rule(
        name="Throttle test",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}

    # First trigger — should publish
    await engine.evaluate(event)
    # Second trigger — same rule, within 300s window — should be throttled
    event["id"] = "evt-2"
    await engine.evaluate(event)

    triggered_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.triggered"
    ]
    assert len(triggered_calls) == 1, (
        f"Expected 1 publish (second should be throttled), got {len(triggered_calls)}"
    )


@pytest.mark.asyncio
async def test_trigger_after_throttle_window_publishes(db):
    """A trigger after the throttle window expires should publish telemetry."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    await engine.add_rule(
        name="Window expiry test",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}

    # Trigger 1 at t=1000 — publishes
    with patch("time.monotonic", return_value=1000.0):
        # Set the monotonic return for _should_publish_telemetry
        await engine.evaluate(event)

    # Trigger 2 at t=1001 — within window — throttled
    event["id"] = "evt-2"
    with patch("time.monotonic", return_value=1001.0):
        await engine.evaluate(event)

    # Trigger 3 at t=1301 — past 300s window — publishes
    event["id"] = "evt-3"
    with patch("time.monotonic", return_value=1301.0):
        await engine.evaluate(event)

    triggered_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.triggered"
    ]
    assert len(triggered_calls) == 2, (
        f"Expected 2 publishes (first + after window), got {len(triggered_calls)}"
    )


@pytest.mark.asyncio
async def test_different_rules_have_independent_throttles(db):
    """Each rule has its own throttle timer — triggering rule A doesn't throttle rule B."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    await engine.add_rule(
        name="Rule A",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "a"}],
    )
    await engine.add_rule(
        name="Rule B",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "b"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    await engine.evaluate(event)

    # Both rules fired on the same event — each should get its own
    # first-trigger publish (2 triggered events total).
    triggered_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.triggered"
    ]
    assert len(triggered_calls) == 2, (
        f"Expected 2 publishes (one per rule), got {len(triggered_calls)}"
    )


@pytest.mark.asyncio
async def test_throttle_zero_disables_throttling(db):
    """Setting telemetry_throttle_seconds=0 should publish on every trigger."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=0)

    await engine.add_rule(
        name="No throttle rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )

    # Fire the same rule 3 times rapidly
    for i in range(3):
        event = {"id": f"evt-{i}", "type": "test.event", "payload": {}}
        await engine.evaluate(event)

    triggered_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.triggered"
    ]
    assert len(triggered_calls) == 3, (
        f"Expected 3 publishes (throttle disabled), got {len(triggered_calls)}"
    )


@pytest.mark.asyncio
async def test_db_trigger_record_updated_even_when_throttled(db):
    """The rules table (times_triggered, last_triggered) must update on EVERY
    trigger, regardless of whether the telemetry event was throttled."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    rule_id = await engine.add_rule(
        name="DB record test",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )

    # Trigger the rule 3 times (only the first should publish telemetry)
    for i in range(3):
        event = {"id": f"evt-{i}", "type": "test.event", "payload": {}}
        await engine.evaluate(event)

    # Verify times_triggered in the DB reflects all 3 triggers
    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT times_triggered, last_triggered FROM rules WHERE id = ?",
            (rule_id,),
        ).fetchone()

    assert row is not None
    assert row["times_triggered"] == 3, (
        f"Expected 3 DB trigger records, got {row['times_triggered']}"
    )
    assert row["last_triggered"] is not None


@pytest.mark.asyncio
async def test_default_throttle_is_300_seconds(db):
    """When no config is provided, the throttle defaults to 300 seconds."""
    engine = RulesEngine(db)
    assert engine._telemetry_throttle_seconds == 300.0


@pytest.mark.asyncio
async def test_config_overrides_throttle(db):
    """The telemetry_throttle_seconds config key overrides the default."""
    engine = RulesEngine(db, config={"telemetry_throttle_seconds": 60})
    assert engine._telemetry_throttle_seconds == 60.0


@pytest.mark.asyncio
async def test_other_telemetry_events_not_throttled(db):
    """Non-trigger telemetry (system.rule.created, system.rule.deactivated)
    should NOT be affected by the throttle."""
    bus = _mock_event_bus()
    engine = _make_engine(db, event_bus=bus, throttle_seconds=300)

    # add_rule publishes system.rule.created — do it twice
    await engine.add_rule(
        name="Rule 1",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "x"}],
    )
    await engine.add_rule(
        name="Rule 2",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "y"}],
    )

    created_calls = [
        c for c in bus.publish.call_args_list
        if c.args[0] == "system.rule.created"
    ]
    assert len(created_calls) == 2, (
        f"Expected 2 system.rule.created events (not throttled), got {len(created_calls)}"
    )
