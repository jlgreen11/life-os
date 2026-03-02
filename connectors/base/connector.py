"""
Life OS — Base Connector Framework

Every external service (email, calendar, messaging, finance, etc.) is 
integrated through a Connector. This module defines the base class and
the lifecycle that all connectors follow.

Connector Lifecycle:
    1. authenticate()  — Establish connection to the external service
    2. sync()          — Pull new data, publish events to the bus  
    3. execute(action)  — Perform an action (send message, create event, etc.)
    4. health_check()   — Verify the connection is alive

Each connector runs in its own async loop, polling at a configurable interval.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract base class for all service connectors."""

    # Override these in subclasses
    CONNECTOR_ID: str = "base"
    DISPLAY_NAME: str = "Base Connector"
    SYNC_INTERVAL_SECONDS: int = 60

    # Exponential backoff delays for auto-reconnect (seconds):
    # 1 min → 5 min → 15 min → 1 hr (cap)
    RECONNECT_DELAYS: list[int] = [60, 300, 900, 3600]

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        self.bus = event_bus
        self.db = db
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempt: int = 0

    # -------------------------------------------------------------------
    # Lifecycle methods — Override these in subclasses
    # -------------------------------------------------------------------

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Establish connection to the external service.
        Returns True if authentication succeeded.
        """
        raise NotImplementedError

    @abstractmethod
    async def sync(self) -> int:
        """
        Pull new data from the external service and publish events.
        Returns the number of new events published.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Perform an outbound action (send message, create event, etc.).
        Returns a result dict.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check if the connector is alive and functioning.
        Returns {"status": "ok|error", "details": ...}
        """
        raise NotImplementedError

    # -------------------------------------------------------------------
    # Runtime management
    # -------------------------------------------------------------------

    async def start(self):
        """Start the connector's sync loop."""
        if self._running:
            return

        # Startup sequence: authenticate first to verify the external service
        # is reachable. Only after a successful auth do we spin up the
        # background sync loop and subscribe to inbound action requests.
        success = await self.authenticate()
        if not success:
            # Use descriptive error from subclass (e.g. GoogleConnector sets
            # self._auth_error) with a generic fallback for connectors that
            # don't set the attribute.
            auth_error = getattr(self, "_auth_error", None) or "Authentication failed"
            await self._update_state("error", auth_error)
            await self.bus.publish(
                "system.connector.error",
                {
                    "connector_id": self.CONNECTOR_ID,
                    "error": auth_error,
                    "error_type": "authentication",
                    "display_name": self.DISPLAY_NAME,
                },
                source=self.CONNECTOR_ID,
            )
            logger.warning(
                "[%s] Authentication failed — starting reconnect loop. Error: %s",
                self.CONNECTOR_ID,
                auth_error,
            )
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            return

        self._running = True
        await self._update_state("active")
        self._task = asyncio.create_task(self._sync_loop())

        # Subscribe to action requests for this connector
        await self.bus.subscribe(
            f"action.{self.CONNECTOR_ID}.*",
            self._handle_action_request,
            consumer_name=f"connector-{self.CONNECTOR_ID}",
        )

    async def stop(self):
        """Stop the connector's sync loop and any reconnect attempts."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        await self._update_state("inactive")

    async def _reconnect_loop(self):
        """Background loop that retries authentication with exponential backoff.

        Started automatically when start() fails to authenticate. Retries
        at increasing intervals (1 min, 5 min, 15 min, 1 hr cap) until
        authentication succeeds, then starts the normal sync loop.
        """
        while not self._running:
            delay = self.RECONNECT_DELAYS[min(self._reconnect_attempt, len(self.RECONNECT_DELAYS) - 1)]
            logger.info(
                "[%s] Reconnect attempt %d in %ds",
                self.CONNECTOR_ID,
                self._reconnect_attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)
            self._reconnect_attempt += 1
            try:
                success = await self.authenticate()
                if success:
                    logger.info(
                        "[%s] Reconnected successfully after %d attempt(s)",
                        self.CONNECTOR_ID,
                        self._reconnect_attempt,
                    )
                    self._running = True
                    self._reconnect_attempt = 0
                    self._reconnect_task = None
                    await self._update_state("active")
                    self._task = asyncio.create_task(self._sync_loop())
                    await self.bus.subscribe(
                        f"action.{self.CONNECTOR_ID}.*",
                        self._handle_action_request,
                        consumer_name=f"connector-{self.CONNECTOR_ID}",
                    )
                    await self.bus.publish(
                        "system.connector.reconnected",
                        {
                            "connector_id": self.CONNECTOR_ID,
                            "attempts": self._reconnect_attempt,
                        },
                        source=self.CONNECTOR_ID,
                    )
                    return
                else:
                    auth_error = getattr(self, "_auth_error", None) or "Authentication failed"
                    logger.warning(
                        "[%s] Reconnect attempt %d failed: %s",
                        self.CONNECTOR_ID,
                        self._reconnect_attempt,
                        auth_error,
                    )
                    await self._update_state("error", auth_error)
            except Exception as e:
                logger.error(
                    "[%s] Reconnect attempt %d error: %s",
                    self.CONNECTOR_ID,
                    self._reconnect_attempt,
                    e,
                )

    async def _sync_loop(self):
        """Main polling loop."""
        # Poll-based sync: call the subclass's sync() at a fixed interval.
        # When sync() returns count > 0, we publish a sync_complete event so
        # downstream consumers (AI agents, dashboards) know new data arrived.
        while self._running:
            try:
                count = await self.sync()
                if count > 0:
                    await self.bus.publish(
                        "system.connector.sync_complete",
                        {
                            "connector_id": self.CONNECTOR_ID,
                            "events_count": count,
                        },
                        source=self.CONNECTOR_ID,
                    )
                await self._update_state("active", error_count_reset=True)
            except Exception as e:
                await self._handle_sync_error(e)

            await asyncio.sleep(self.SYNC_INTERVAL_SECONDS)

    async def _handle_action_request(self, event: dict):
        """Handle an inbound action request from the event bus."""
        action = event.get("payload", {}).get("action", "")
        params = event.get("payload", {}).get("params", {})

        try:
            result = await self.execute(action, params)
            await self.bus.publish(
                f"system.ai.action_taken",
                {
                    "connector_id": self.CONNECTOR_ID,
                    "action": action,
                    "result": result,
                    "success": True,
                },
                source=self.CONNECTOR_ID,
            )
        except Exception as e:
            await self.bus.publish(
                "system.connector.error",
                {
                    "connector_id": self.CONNECTOR_ID,
                    "action": action,
                    "error": str(e),
                },
                source=self.CONNECTOR_ID,
            )

    async def _handle_sync_error(self, error: Exception):
        """Handle sync errors with exponential backoff tracking."""
        # Error tracking is persisted in the database via _update_state, which
        # increments the error_count column. This allows the UI and health
        # checks to surface repeated failures without in-memory state.
        error_msg = str(error)
        logger.error("[%s] Sync error: %s", self.CONNECTOR_ID, error_msg)

        await self.bus.publish(
            "system.connector.error",
            {
                "connector_id": self.CONNECTOR_ID,
                "error": error_msg,
            },
            source=self.CONNECTOR_ID,
        )
        await self._update_state("error", error_msg)

    async def _update_state(self, status: str, error: Optional[str] = None,
                            error_count_reset: bool = False):
        """Update connector state in the database, preserving existing config."""
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("state") as conn:
            # Branch 1 — Success reset: after a successful sync, reset the
            # error_count back to 0 and record the last_sync timestamp.
            if error_count_reset:
                conn.execute(
                    """INSERT OR REPLACE INTO connector_state
                       (connector_id, status, last_sync, error_count, updated_at)
                       VALUES (?, ?, ?, 0, ?)""",
                    (
                        self.CONNECTOR_ID,
                        status,
                        datetime.now(timezone.utc).isoformat(),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            elif error:
                # Branch 2 — Error increment: on failure, upsert the row and
                # increment error_count. Uses ON CONFLICT so the very first
                # error creates the row while subsequent errors just bump the
                # counter and update last_error.
                conn.execute(
                    """INSERT INTO connector_state (connector_id, status, last_error, error_count, updated_at)
                       VALUES (?, ?, ?, 1, ?)
                       ON CONFLICT(connector_id) DO UPDATE SET
                           status = ?, last_error = ?,
                           error_count = error_count + 1,
                           updated_at = ?""",
                    (
                        self.CONNECTOR_ID, status, error,
                        datetime.now(timezone.utc).isoformat(),
                        status, error,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            else:
                # Branch 3 — Simple status update: just change the status
                # (e.g., "active" or "inactive") without touching error state.
                conn.execute(
                    """INSERT OR REPLACE INTO connector_state
                       (connector_id, status, updated_at)
                       VALUES (?, ?, ?)""",
                    (self.CONNECTOR_ID, status, datetime.now(timezone.utc).isoformat()),
                )

    # -------------------------------------------------------------------
    # Helpers for subclasses
    # -------------------------------------------------------------------

    def get_sync_cursor(self) -> Optional[str]:
        """Get the last sync cursor for incremental syncing."""
        with self.db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT sync_cursor FROM connector_state WHERE connector_id = ?",
                (self.CONNECTOR_ID,),
            ).fetchone()
            return row["sync_cursor"] if row else None

    def set_sync_cursor(self, cursor: str):
        """Store the sync cursor for next run."""
        with self.db.get_connection("state") as conn:
            conn.execute(
                """UPDATE connector_state SET sync_cursor = ?, updated_at = ?
                   WHERE connector_id = ?""",
                (cursor, datetime.now(timezone.utc).isoformat(), self.CONNECTOR_ID),
            )

    async def publish_event(self, event_type: str, payload: dict,
                            priority: str = "normal", metadata: Optional[dict] = None) -> str:
        """Convenience method to publish an event from this connector."""
        # Auto-sets the source field to this connector's CONNECTOR_ID so
        # downstream consumers always know which connector produced the event.
        return await self.bus.publish(
            event_type,
            payload,
            source=self.CONNECTOR_ID,
            priority=priority,
            metadata=metadata,
        )
