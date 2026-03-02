"""
Tests for auth failure error event publishing in BaseConnector.start().

When a connector fails authentication, start() should publish a
'system.connector.error' event to the event bus so the failure is visible
to the dashboard event feed, health endpoint, and notification system —
not just silently recorded in the connector_state table.

Coverage:
    - Auth failure publishes error event with connector-specific message
    - Auth failure uses fallback message when _auth_error is not set
    - Successful auth does not publish an error event
"""

import asyncio

import pytest
from typing import Any

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


# -------------------------------------------------------------------------
# Test Connector Implementations
# -------------------------------------------------------------------------


class FailingAuthConnector(BaseConnector):
    """Connector whose authenticate() always fails with a specific error."""

    CONNECTOR_ID = "auth_fail_test"
    DISPLAY_NAME = "Auth Fail Test Connector"
    SYNC_INTERVAL_SECONDS = 60

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any],
                 auth_error: str | None = None):
        """Initialize with an optional auth_error message.

        Args:
            event_bus: Event bus for publishing events.
            db: Database manager for state persistence.
            config: Connector configuration dict.
            auth_error: If provided, set as self._auth_error before returning
                False from authenticate(). If None, _auth_error is not set,
                testing the fallback path.
        """
        super().__init__(event_bus, db, config)
        self._custom_auth_error = auth_error

    async def authenticate(self) -> bool:
        """Simulate authentication failure, optionally setting _auth_error."""
        if self._custom_auth_error is not None:
            self._auth_error = self._custom_auth_error
        return False

    async def sync(self) -> int:
        return 0

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "not_implemented"}

    async def health_check(self) -> dict[str, Any]:
        return {"status": "error", "details": "Not authenticated"}


class SuccessAuthConnector(BaseConnector):
    """Connector whose authenticate() always succeeds."""

    CONNECTOR_ID = "auth_success_test"
    DISPLAY_NAME = "Auth Success Test Connector"
    SYNC_INTERVAL_SECONDS = 60

    async def authenticate(self) -> bool:
        """Simulate successful authentication."""
        return True

    async def sync(self) -> int:
        return 0

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok"}

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_failure_publishes_error_event(event_bus, db):
    """Auth failure should publish a system.connector.error event with the
    connector-specific error message from _auth_error."""
    connector = FailingAuthConnector(event_bus, db, {}, auth_error="Token expired")

    received_errors = []

    async def capture_error(event):
        received_errors.append(event)

    await event_bus.subscribe("system.connector.error", capture_error)

    await connector.start()
    await asyncio.sleep(0.05)

    assert len(received_errors) == 1
    payload = received_errors[0]["payload"]
    assert payload["connector_id"] == "auth_fail_test"
    assert payload["error"] == "Token expired"
    assert payload["error_type"] == "authentication"
    assert payload["display_name"] == "Auth Fail Test Connector"

    # Connector should NOT be running
    assert connector._running is False
    assert connector._task is None


@pytest.mark.asyncio
async def test_auth_failure_uses_fallback_message(event_bus, db):
    """When _auth_error is not set by the subclass, the published error event
    should use the default 'Authentication failed' message."""
    # auth_error=None means _auth_error will NOT be set on the connector
    connector = FailingAuthConnector(event_bus, db, {}, auth_error=None)

    received_errors = []

    async def capture_error(event):
        received_errors.append(event)

    await event_bus.subscribe("system.connector.error", capture_error)

    await connector.start()
    await asyncio.sleep(0.05)

    assert len(received_errors) == 1
    payload = received_errors[0]["payload"]
    assert payload["connector_id"] == "auth_fail_test"
    assert payload["error"] == "Authentication failed"
    assert payload["error_type"] == "authentication"

    # Also verify the database state uses the same fallback message
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT last_error FROM connector_state WHERE connector_id = ?",
            (connector.CONNECTOR_ID,),
        ).fetchone()
        assert row is not None
        assert row["last_error"] == "Authentication failed"


@pytest.mark.asyncio
async def test_successful_auth_does_not_publish_error(event_bus, db):
    """Successful authentication should NOT publish a system.connector.error
    event. Only sync_complete or action events should appear."""
    connector = SuccessAuthConnector(event_bus, db, {})

    received_errors = []

    async def capture_error(event):
        received_errors.append(event)

    await event_bus.subscribe("system.connector.error", capture_error)

    await connector.start()
    await asyncio.sleep(0.05)

    try:
        # No error events should have been published
        assert len(received_errors) == 0

        # Connector should be running normally
        assert connector._running is True
        assert connector._task is not None
    finally:
        await connector.stop()
