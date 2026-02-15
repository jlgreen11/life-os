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

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class HomeAssistantConnector(BaseConnector):

    CONNECTOR_ID = "home_assistant"
    DISPLAY_NAME = "Home Assistant"
    SYNC_INTERVAL_SECONDS = 30

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._url = config.get("url", "http://homeassistant.local:8123")
        self._token = config.get("token", "")
        self._watched = config.get("watched_entities", [])
        self._last_states: dict[str, Any] = {}

    async def authenticate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self._url}/api/",
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            print(f"[home_assistant] Auth failed: {e}")
            return False

    async def sync(self) -> int:
        count = 0

        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {self._token}"}

            for entity_id in self._watched:
                try:
                    resp = await client.get(
                        f"{self._url}/api/states/{entity_id}",
                        headers=headers,
                    )
                    resp.raise_for_status()
                    state = resp.json()

                    current_state = state.get("state")
                    last_state = self._last_states.get(entity_id)

                    # Only publish if state changed
                    if current_state != last_state:
                        self._last_states[entity_id] = current_state

                        # Classify the event
                        event_type, priority = self._classify_state_change(
                            entity_id, last_state, current_state
                        )

                        payload = {
                            "entity_id": entity_id,
                            "state": current_state,
                            "previous_state": last_state,
                            "attributes": state.get("attributes", {}),
                            "last_changed": state.get("last_changed"),
                        }

                        await self.publish_event(
                            event_type, payload, priority=priority,
                        )
                        count += 1

                except Exception as e:
                    print(f"[home_assistant] Error reading {entity_id}: {e}")

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Control Home Assistant entities."""
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"Authorization": f"Bearer {self._token}",
                       "Content-Type": "application/json"}

            if action == "turn_on":
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
                resp = await client.post(
                    f"{self._url}/api/services/{params['domain']}/{params['service']}",
                    headers=headers,
                    json=params.get("data", {}),
                )
                resp.raise_for_status()
                return {"status": "service_called"}

        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
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
        """Classify a state change into an event type and priority."""
        if entity_id.startswith("person."):
            if new == "home":
                return "home.arrived", "normal"
            elif old == "home":
                return "home.departed", "normal"
            return "location.changed", "low"

        elif entity_id.startswith("binary_sensor.") and "door" in entity_id:
            return "home.device.state_changed", "normal"

        return "home.device.state_changed", "low"
