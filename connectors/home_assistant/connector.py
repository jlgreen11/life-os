"""
Life OS — Home Assistant Connector

Connects to a Home Assistant instance for smart home awareness:
presence detection, device states, automations, and environment data.

Configuration:
    connectors:
      home_assistant:
        url: "http://homeassistant.local:8123"
        token: "your-long-lived-access-token"
        sync_interval: 30
        watched_entities:
          - "person.jay"
          - "sensor.living_room_temperature"
          - "binary_sensor.front_door"
          - "light.living_room"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class HomeAssistantConnector(BaseConnector):
    """Connector that polls Home Assistant's REST API for entity state changes.

    Design decisions:
        - Uses the REST API (``/api/states/<entity_id>``) rather than the
          WebSocket API to keep the implementation simple and stateless between
          sync cycles.
        - Maintains an in-memory cache (``_last_states``) of the most recent
          state per entity so that only *changed* states are published to the
          event bus, avoiding duplicate / noisy events.
        - State changes are classified into semantic event types (e.g.,
          ``home.arrived``, ``home.departed``) via ``_classify_state_change``
          to let downstream automations react with appropriate priority.
    """

    CONNECTOR_ID = "home_assistant"
    DISPLAY_NAME = "Home Assistant"
    # 30-second polling keeps smart-home state reasonably fresh without
    # hammering the HA instance.
    SYNC_INTERVAL_SECONDS = 30

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        # Base URL of the Home Assistant instance (including port).
        self._url = config.get("url", "http://homeassistant.local:8123")
        # Long-lived access token generated in HA's user profile page.
        self._token = config.get("token", "")
        # List of entity IDs to track (e.g., "person.jay", "sensor.temp").
        self._watched = config.get("watched_entities", [])
        # In-memory cache mapping entity_id -> last known state string.
        # Used for change detection so we only emit events on transitions.
        self._last_states: dict[str, Any] = {}

    async def authenticate(self) -> bool:
        """Verify connectivity to the Home Assistant instance.

        Hits the ``/api/`` endpoint which returns a simple JSON message when
        the long-lived access token is valid.  This also confirms that the HA
        instance is reachable over the network.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._url}/api/",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    async def sync(self) -> int:
        """Poll each watched entity and publish events for state transitions.

        For every entity in the watch list we:
            1. GET its current state from the HA REST API.
            2. Compare with the cached ``_last_states`` value.
            3. If changed, classify the transition (arrival, departure, generic
               device change, etc.) and publish the appropriate event.

        Returns the number of state-change events published this cycle.
        """
        count = 0

        # A single httpx client is reused across all entity requests in this
        # cycle to benefit from HTTP keep-alive.
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {self._token}"}

            for entity_id in self._watched:
                try:
                    # Fetch the full state object for this entity, which
                    # includes the state value and a rich attributes dict.
                    resp = await client.get(
                        f"{self._url}/api/states/{entity_id}",
                        headers=headers,
                    )
                    resp.raise_for_status()
                    state = resp.json()

                    current_state = state.get("state")
                    last_state = self._last_states.get(entity_id)

                    # ---- Change Detection ----
                    # Only publish if state changed to avoid flooding the bus
                    # with redundant events every 30 seconds.
                    if current_state != last_state:
                        # Update the cache before publishing so we don't
                        # re-emit the same transition on failure/retry.
                        self._last_states[entity_id] = current_state

                        # ---- Event Classification ----
                        # Map the raw state transition to a semantic event type
                        # (e.g., "home.arrived") and priority level.
                        event_type, priority = self._classify_state_change(
                            entity_id, last_state, current_state
                        )

                        payload = {
                            "entity_id": entity_id,
                            "state": current_state,
                            "previous_state": last_state,
                            # HA attributes include device-specific data such
                            # as brightness, temperature, friendly_name, etc.
                            "attributes": state.get("attributes", {}),
                            "last_changed": state.get("last_changed"),
                        }

                        await self.publish_event(
                            event_type, payload, priority=priority,
                        )
                        count += 1

                except Exception as e:
                    # Log individual entity failures without aborting the loop
                    # so that other entities still get polled.
                    logger.warning("Error reading entity %s: %s", entity_id, e)

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Control Home Assistant entities via the service-call REST API.

        Supported actions:
            - ``turn_on``  / ``turn_off`` -- convenience wrappers around the
              ``homeassistant.turn_on`` / ``homeassistant.turn_off`` services.
            - ``call_service`` -- generic passthrough that lets the agent call
              any HA service by specifying ``domain``, ``service``, and ``data``.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {self._token}",
                       "Content-Type": "application/json"}

            if action == "turn_on":
                # POST to the generic turn_on service; HA resolves the correct
                # domain (light, switch, etc.) from the entity_id prefix.
                resp = await client.post(
                    f"{self._url}/api/services/homeassistant/turn_on",
                    headers=headers,
                    json={"entity_id": params["entity_id"]},
                )
                resp.raise_for_status()
                return {"status": "turned_on", "entity": params["entity_id"]}

            elif action == "turn_off":
                resp = await client.post(
                    f"{self._url}/api/services/homeassistant/turn_off",
                    headers=headers,
                    json={"entity_id": params["entity_id"]},
                )
                resp.raise_for_status()
                return {"status": "turned_off", "entity": params["entity_id"]}

            elif action == "call_service":
                # Generic service call -- the caller specifies domain (e.g.,
                # "light"), service (e.g., "turn_on"), and an optional data
                # dict with service-specific parameters (e.g., brightness).
                resp = await client.post(
                    f"{self._url}/api/services/{params['domain']}/{params['service']}",
                    headers=headers,
                    json=params.get("data", {}),
                )
                resp.raise_for_status()
                return {"status": "service_called"}

        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
        """Ping the HA API root to verify the instance is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self._url}/api/",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                resp.raise_for_status()
                return {"status": "ok", "connector": self.CONNECTOR_ID,
                        "entities_watched": len(self._watched)}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    def _classify_state_change(self, entity_id: str, old: Any, new: Any) -> tuple[str, str]:
        """Classify a state change into a semantic event type and priority.

        The classification uses HA entity-id naming conventions:
            - ``person.*`` entities track user presence.  Transitioning *to*
              "home" means arrival; transitioning *from* "home" means departure.
            - ``binary_sensor.*door*`` entities represent physical entry points
              and get normal priority so the agent can react to security-
              relevant events.
            - Everything else (lights, sensors, switches) is a generic device
              state change at low priority.

        Returns:
            A ``(event_type, priority)`` tuple.
        """
        if entity_id.startswith("person."):
            # Presence detection — the most actionable smart-home signal.
            if new == "home":
                return "home.arrived", "normal"
            elif old == "home":
                return "home.departed", "normal"
            # Person moved between non-home zones (e.g., work -> gym).
            return "location.changed", "low"

        elif entity_id.startswith("binary_sensor.") and "door" in entity_id:
            # Door open/close events are security-relevant, keep normal priority.
            return "home.device.state_changed", "normal"

        # Default: generic device state change at low priority to avoid noise.
        return "home.device.state_changed", "low"
