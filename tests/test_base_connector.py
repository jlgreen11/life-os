"""
Tests for connectors/base/connector.py — BaseConnector framework.

BaseConnector is the foundation for all service integrations. It defines the
lifecycle (authenticate → sync → execute → health_check), manages async
polling loops, handles error tracking, and provides event bus integration.

Coverage:
    - Lifecycle management (start, stop, state transitions)
    - Sync loop with interval polling
    - Error handling and exponential backoff tracking
    - Action request handling via event bus
    - Sync cursor persistence for incremental syncing
    - Event publishing convenience methods
    - State tracking (active, error, inactive)
    - Authentication flow
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


# -------------------------------------------------------------------------
# Test Connector Implementation
# -------------------------------------------------------------------------


class TestConnector(BaseConnector):
    """Concrete implementation of BaseConnector for testing."""

    CONNECTOR_ID = "test_connector"
    DISPLAY_NAME = "Test Connector"
    SYNC_INTERVAL_SECONDS = 0.1  # Fast interval for tests

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self.auth_success = True
        self.sync_count = 0
        self.sync_events = []
        self.execute_calls = []
        self.health_status = {"status": "ok"}

    async def authenticate(self) -> bool:
        """Simulate authentication."""
        return self.auth_success

    async def sync(self) -> int:
        """Simulate syncing data."""
        self.sync_count += 1
        count = self.config.get("events_per_sync", 0)
        for i in range(count):
            await self.publish_event(
                "test.event",
                {"count": self.sync_count, "index": i},
            )
        self.sync_events.append({"timestamp": datetime.now(timezone.utc), "count": count})
        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Simulate executing an action."""
        self.execute_calls.append({"action": action, "params": params})
        if action == "fail":
            raise ValueError("Action failed")
        return {"status": "success", "action": action, "params": params}

    async def health_check(self) -> dict[str, Any]:
        """Return configured health status."""
        return self.health_status


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def test_connector(event_bus, db):
    """Create a TestConnector instance."""
    config = {"events_per_sync": 0}
    return TestConnector(event_bus, db, config)


# -------------------------------------------------------------------------
# Lifecycle Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_calls_authenticate(event_bus, db):
    """start() should call authenticate() before starting sync loop."""
    connector = TestConnector(event_bus, db, {})
    connector.auth_success = True

    await connector.start()
    try:
        assert connector._running is True
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_start_fails_if_authenticate_fails(event_bus, db):
    """start() should not start sync loop if authenticate() returns False."""
    connector = TestConnector(event_bus, db, {})
    connector.auth_success = False

    await connector.start()

    assert connector._running is False
    assert connector._task is None

    # Check that state was set to "error"
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status, last_error FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "error"
        assert "Authentication failed" in row["last_error"]


@pytest.mark.asyncio
async def test_start_sets_state_to_active(event_bus, db):
    """start() should set connector state to 'active' after successful auth."""
    connector = TestConnector(event_bus, db, {})
    connector.auth_success = True

    await connector.start()
    try:
        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT status FROM connector_state WHERE connector_id = ?",
                (connector.CONNECTOR_ID,),
            ).fetchone()
            assert row is not None
            assert row["status"] == "active"
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_start_creates_sync_task(event_bus, db):
    """start() should create a background task for the sync loop."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    try:
        assert connector._task is not None
        assert not connector._task.done()
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_start_subscribes_to_action_events(event_bus, db):
    """start() should subscribe to action.{connector_id}.* events."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    try:
        # Publish an action event and verify it gets handled
        await event_bus.publish(
            f"action.{connector.CONNECTOR_ID}.test",
            {"action": "test_action", "params": {"key": "value"}},
        )
        await asyncio.sleep(0.05)
        assert len(connector.execute_calls) == 1
        assert connector.execute_calls[0]["action"] == "test_action"
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_stop_cancels_sync_task(event_bus, db):
    """stop() should cancel the background sync task."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    task = connector._task

    await connector.stop()

    assert connector._running is False
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_stop_sets_state_to_inactive(event_bus, db):
    """stop() should set connector state to 'inactive'."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    await connector.stop()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "inactive"


@pytest.mark.asyncio
async def test_start_is_idempotent(event_bus, db):
    """Calling start() multiple times should not create multiple tasks."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    first_task = connector._task

    await connector.start()  # Call again
    second_task = connector._task

    assert first_task is second_task
    await connector.stop()


# -------------------------------------------------------------------------
# Sync Loop Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_loop_calls_sync_periodically(event_bus, db):
    """The sync loop should call sync() at the configured interval."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 0})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    await connector.start()
    await asyncio.sleep(0.15)  # Wait for ~3 sync cycles
    await connector.stop()

    assert connector.sync_count >= 2


@pytest.mark.asyncio
async def test_sync_loop_publishes_sync_complete_when_events_exist(event_bus, db):
    """sync loop should publish system.connector.sync_complete when sync() returns > 0."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 5})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    received_events = []

    async def capture_sync_complete(event):
        received_events.append(event)

    await event_bus.subscribe("system.connector.sync_complete", capture_sync_complete)

    await connector.start()
    await asyncio.sleep(0.15)
    await connector.stop()

    # Should have received at least one sync_complete event
    assert len(received_events) >= 1
    assert received_events[0]["payload"]["connector_id"] == connector.CONNECTOR_ID
    assert received_events[0]["payload"]["events_count"] == 5


@pytest.mark.asyncio
async def test_sync_loop_does_not_publish_sync_complete_when_no_events(event_bus, db):
    """sync loop should not publish sync_complete when sync() returns 0."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 0})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    received_events = []

    async def capture_sync_complete(event):
        received_events.append(event)

    await event_bus.subscribe("system.connector.sync_complete", capture_sync_complete)

    await connector.start()
    await asyncio.sleep(0.15)
    await connector.stop()

    # Should not have received any sync_complete events
    assert len(received_events) == 0


@pytest.mark.asyncio
async def test_sync_loop_resets_error_count_on_success(event_bus, db):
    """Successful sync should reset error_count to 0."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 1})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    # Manually insert an error state
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO connector_state
               (connector_id, status, error_count, last_error, updated_at)
               VALUES (?, 'error', 5, 'Previous error', ?)""",
            (connector.CONNECTOR_ID, datetime.now(timezone.utc).isoformat()),
        )

    await connector.start()
    await asyncio.sleep(0.12)

    # Check BEFORE stopping (while still active)
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT error_count, status FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["error_count"] == 0
        assert row["status"] == "active"

    await connector.stop()


# -------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_error_publishes_error_event(event_bus, db):
    """Sync errors should publish system.connector.error events."""
    connector = TestConnector(event_bus, db, {})

    # Override sync to raise an error
    original_sync = connector.sync

    async def failing_sync():
        raise ValueError("Sync failed")

    connector.sync = failing_sync

    received_errors = []

    async def capture_error(event):
        received_errors.append(event)

    await event_bus.subscribe("system.connector.error", capture_error)

    await connector.start()
    await asyncio.sleep(0.15)
    await connector.stop()

    assert len(received_errors) >= 1
    assert "Sync failed" in received_errors[0]["payload"]["error"]
    assert received_errors[0]["payload"]["connector_id"] == connector.CONNECTOR_ID


@pytest.mark.asyncio
async def test_sync_error_increments_error_count(event_bus, db):
    """Sync errors should increment error_count in connector_state."""
    connector = TestConnector(event_bus, db, {})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    # Override sync to raise an error
    async def failing_sync():
        raise ValueError("Sync failed")

    connector.sync = failing_sync

    await connector.start()
    await asyncio.sleep(0.15)  # Let it fail a few times

    # Check BEFORE stopping (while errors are occurring)
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT error_count, status, last_error FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["error_count"] >= 2
        assert row["status"] == "error"
        assert "Sync failed" in row["last_error"]

    await connector.stop()


@pytest.mark.asyncio
async def test_sync_error_does_not_stop_loop(event_bus, db):
    """Sync errors should not stop the sync loop."""
    connector = TestConnector(event_bus, db, {})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    error_count = 0

    original_sync = connector.sync

    async def failing_sync():
        nonlocal error_count
        error_count += 1
        if error_count < 3:
            raise ValueError("Temporary error")
        return await original_sync()

    connector.sync = failing_sync

    await connector.start()
    await asyncio.sleep(0.2)
    await connector.stop()

    # Should have attempted sync multiple times despite errors
    assert error_count >= 3


# -------------------------------------------------------------------------
# Action Handling Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_request_calls_execute(event_bus, db):
    """Action requests should be routed to execute()."""
    connector = TestConnector(event_bus, db, {})
    await connector.start()
    try:
        await event_bus.publish(
            f"action.{connector.CONNECTOR_ID}.send",
            {"action": "send_message", "params": {"to": "user@example.com"}},
        )
        await asyncio.sleep(0.05)

        assert len(connector.execute_calls) == 1
        assert connector.execute_calls[0]["action"] == "send_message"
        assert connector.execute_calls[0]["params"]["to"] == "user@example.com"
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_action_success_publishes_action_taken_event(event_bus, db):
    """Successful action execution should publish system.ai.action_taken."""
    connector = TestConnector(event_bus, db, {})

    received_events = []

    async def capture_action_taken(event):
        received_events.append(event)

    await event_bus.subscribe("system.ai.action_taken", capture_action_taken)

    await connector.start()
    try:
        await event_bus.publish(
            f"action.{connector.CONNECTOR_ID}.test",
            {"action": "test_action", "params": {"key": "value"}},
        )
        await asyncio.sleep(0.05)

        assert len(received_events) == 1
        assert received_events[0]["payload"]["connector_id"] == connector.CONNECTOR_ID
        assert received_events[0]["payload"]["action"] == "test_action"
        assert received_events[0]["payload"]["success"] is True
    finally:
        await connector.stop()


@pytest.mark.asyncio
async def test_action_failure_publishes_error_event(event_bus, db):
    """Failed action execution should publish system.connector.error."""
    connector = TestConnector(event_bus, db, {})

    received_errors = []

    async def capture_error(event):
        received_errors.append(event)

    await event_bus.subscribe("system.connector.error", capture_error)

    await connector.start()
    try:
        await event_bus.publish(
            f"action.{connector.CONNECTOR_ID}.test",
            {"action": "fail", "params": {}},
        )
        await asyncio.sleep(0.05)

        assert len(received_errors) == 1
        assert received_errors[0]["payload"]["connector_id"] == connector.CONNECTOR_ID
        assert received_errors[0]["payload"]["action"] == "fail"
        assert "Action failed" in received_errors[0]["payload"]["error"]
    finally:
        await connector.stop()


# -------------------------------------------------------------------------
# Sync Cursor Tests
# -------------------------------------------------------------------------


def test_get_sync_cursor_returns_none_when_not_set(db):
    """get_sync_cursor() should return None if no cursor is stored."""
    connector = TestConnector(EventBus(), db, {})
    cursor = connector.get_sync_cursor()
    assert cursor is None


def test_set_and_get_sync_cursor(db):
    """set_sync_cursor() should persist the cursor for get_sync_cursor()."""
    connector = TestConnector(EventBus(), db, {})

    # First insert a state record
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO connector_state (connector_id, status, updated_at) VALUES (?, 'active', ?)",
            (connector.CONNECTOR_ID, datetime.now(timezone.utc).isoformat()),
        )

    connector.set_sync_cursor("cursor_value_123")
    cursor = connector.get_sync_cursor()
    assert cursor == "cursor_value_123"


def test_set_sync_cursor_updates_timestamp(db):
    """set_sync_cursor() should update the updated_at timestamp."""
    connector = TestConnector(EventBus(), db, {})

    # First insert a state record
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO connector_state (connector_id, status, updated_at) VALUES (?, 'active', ?)",
            (connector.CONNECTOR_ID, "2020-01-01T00:00:00Z"),
        )

    connector.set_sync_cursor("new_cursor")

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT updated_at FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        # Should have updated to a recent timestamp
        assert row["updated_at"] > "2020-01-01T00:00:00Z"


# -------------------------------------------------------------------------
# Event Publishing Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_event_sets_source_to_connector_id(event_bus, db):
    """publish_event() should automatically set the source field."""
    connector = TestConnector(event_bus, db, {})

    received_events = []

    async def capture_event(event):
        received_events.append(event)

    await event_bus.subscribe("test.event", capture_event)

    await connector.publish_event("test.event", {"data": "value"})
    await asyncio.sleep(0.05)

    assert len(received_events) == 1
    assert received_events[0]["source"] == connector.CONNECTOR_ID


@pytest.mark.asyncio
async def test_publish_event_accepts_priority(event_bus, db):
    """publish_event() should forward the priority parameter."""
    connector = TestConnector(event_bus, db, {})

    received_events = []

    async def capture_event(event):
        received_events.append(event)

    await event_bus.subscribe("test.event", capture_event)

    await connector.publish_event("test.event", {"data": "value"}, priority="high")
    await asyncio.sleep(0.05)

    assert len(received_events) == 1
    assert received_events[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_publish_event_accepts_metadata(event_bus, db):
    """publish_event() should forward the metadata parameter."""
    connector = TestConnector(event_bus, db, {})

    received_events = []

    async def capture_event(event):
        received_events.append(event)

    await event_bus.subscribe("test.event", capture_event)

    await connector.publish_event(
        "test.event",
        {"data": "value"},
        metadata={"domain": "work", "related_contacts": ["user@example.com"]},
    )
    await asyncio.sleep(0.05)

    assert len(received_events) == 1
    assert received_events[0]["metadata"]["domain"] == "work"


# -------------------------------------------------------------------------
# State Management Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_state_creates_record_if_not_exists(db):
    """_update_state() should create a new state record if none exists."""
    connector = TestConnector(EventBus(), db, {})
    await connector._update_state("active")

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "active"


@pytest.mark.asyncio
async def test_update_state_with_error_sets_error_fields(db):
    """_update_state() with error should set last_error and increment error_count."""
    connector = TestConnector(EventBus(), db, {})
    await connector._update_state("error", "Test error message")

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status, last_error, error_count FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["status"] == "error"
        assert row["last_error"] == "Test error message"
        assert row["error_count"] == 1


@pytest.mark.asyncio
async def test_update_state_with_error_increments_existing_count(db):
    """_update_state() with error should increment existing error_count."""
    connector = TestConnector(EventBus(), db, {})

    # First error
    await connector._update_state("error", "First error")
    # Second error
    await connector._update_state("error", "Second error")

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT error_count FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["error_count"] == 2


@pytest.mark.asyncio
async def test_update_state_with_reset_clears_error_count(db):
    """_update_state() with error_count_reset=True should reset error_count to 0."""
    connector = TestConnector(EventBus(), db, {})

    # Create an error state
    await connector._update_state("error", "Test error")

    # Reset with success
    await connector._update_state("active", error_count_reset=True)

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT error_count, status FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["error_count"] == 0
        assert row["status"] == "active"


@pytest.mark.asyncio
async def test_update_state_with_reset_sets_last_sync(db):
    """_update_state() with error_count_reset=True should update last_sync timestamp."""
    connector = TestConnector(EventBus(), db, {})

    before = datetime.now(timezone.utc).isoformat()
    await connector._update_state("active", error_count_reset=True)
    after = datetime.now(timezone.utc).isoformat()

    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT last_sync FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["last_sync"] >= before
        assert row["last_sync"] <= after


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_lifecycle_start_sync_stop(event_bus, db):
    """Test complete lifecycle: start → sync → stop."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 3})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    received_events = []

    async def capture_event(event):
        received_events.append(event)

    await event_bus.subscribe("test.event", capture_event)

    # Start
    await connector.start()
    assert connector._running is True

    # Wait for syncs
    await asyncio.sleep(0.15)

    # Stop
    await connector.stop()
    assert connector._running is False

    # Should have received events from multiple sync cycles
    assert len(received_events) >= 3

    # State should be inactive
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row["status"] == "inactive"


@pytest.mark.asyncio
async def test_connector_recovers_from_transient_errors(event_bus, db):
    """Connector should continue syncing after transient errors."""
    connector = TestConnector(event_bus, db, {"events_per_sync": 1})
    connector.SYNC_INTERVAL_SECONDS = 0.05

    error_on_sync = [True, True, False]  # Fail first 2, succeed on 3rd

    original_sync = connector.sync

    async def sometimes_failing_sync():
        if error_on_sync:
            should_fail = error_on_sync.pop(0)
            if should_fail:
                raise ValueError("Transient error")
        return await original_sync()

    connector.sync = sometimes_failing_sync

    await connector.start()
    await asyncio.sleep(0.25)  # Wait for recovery

    # Check BEFORE stopping (while recovered and active)
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT status, error_count FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        # Should have recovered and reset error count
        assert row["status"] == "active"
        assert row["error_count"] == 0  # Reset after success

    await connector.stop()
