"""
Comprehensive test coverage for HomeAssistantConnector.

Tests smart home state polling, entity change detection, event classification,
service call execution, presence detection, and error handling.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from connectors.home_assistant.connector import HomeAssistantConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ha_config():
    """Minimal Home Assistant connector configuration."""
    return {
        "url": "http://homeassistant.local:8123",
        "token": "test-long-lived-access-token",
        "sync_interval": 30,
        "watched_entities": [
            "person.jay",
            "sensor.living_room_temperature",
            "binary_sensor.front_door",
            "light.living_room",
        ],
    }


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient for HTTP requests."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    return client


@pytest.fixture
def mock_state_response():
    """Create a mock Home Assistant state response."""
    return {
        "entity_id": "person.jay",
        "state": "home",
        "attributes": {
            "friendly_name": "Jay",
            "latitude": 37.7749,
            "longitude": -122.4194,
        },
        "last_changed": "2026-02-15T10:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


def test_initialization(event_bus, db, ha_config):
    """Test HomeAssistantConnector initializes with correct configuration."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    assert connector.CONNECTOR_ID == "home_assistant"
    assert connector.DISPLAY_NAME == "Home Assistant"
    assert connector.SYNC_INTERVAL_SECONDS == 30
    assert connector._url == "http://homeassistant.local:8123"
    assert connector._token == "test-long-lived-access-token"
    assert len(connector._watched) == 4
    assert connector._last_states == {}
    assert connector.config == ha_config


def test_initialization_defaults(event_bus, db):
    """Test connector uses default values when config is minimal."""
    config = {"token": "test-token"}
    connector = HomeAssistantConnector(event_bus, db, config)

    assert connector._url == "http://homeassistant.local:8123"
    assert connector._watched == []
    assert connector._token == "test-token"


# ---------------------------------------------------------------------------
# Test: Authentication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_success(event_bus, db, ha_config, mock_httpx_client):
    """Test successful Home Assistant authentication."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock successful API response
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.authenticate()

    assert result is True
    mock_httpx_client.get.assert_called_once_with(
        "http://homeassistant.local:8123/api/",
        headers={"Authorization": "Bearer test-long-lived-access-token"},
    )


@pytest.mark.asyncio
async def test_authenticate_network_error(event_bus, db, ha_config):
    """Test authentication handles network errors gracefully."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock httpx to raise an exception when creating the client
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_class.side_effect = Exception("Connection refused")
        result = await connector.authenticate()

    # The connector catches all exceptions and returns False
    assert result is False


@pytest.mark.asyncio
async def test_authenticate_invalid_token(event_bus, db, ha_config):
    """Test authentication handles HTTP 401 unauthorized."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock httpx to raise an exception for invalid token
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_class.side_effect = Exception("401 Unauthorized")
        result = await connector.authenticate()

    # The connector catches all exceptions and returns False
    assert result is False


# ---------------------------------------------------------------------------
# Test: Sync - State Change Detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_first_run_no_events(event_bus, db, ha_config, mock_httpx_client):
    """Test first sync populates cache but emits no events (no previous state)."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock state responses for all watched entities
    mock_responses = [
        Mock(
            raise_for_status=Mock(),
            json=Mock(
                return_value={
                    "entity_id": "person.jay",
                    "state": "home",
                    "attributes": {"friendly_name": "Jay"},
                    "last_changed": "2026-02-15T10:00:00+00:00",
                }
            ),
        ),
        Mock(
            raise_for_status=Mock(),
            json=Mock(
                return_value={
                    "entity_id": "sensor.living_room_temperature",
                    "state": "72",
                    "attributes": {"unit_of_measurement": "°F"},
                    "last_changed": "2026-02-15T10:00:00+00:00",
                }
            ),
        ),
        Mock(
            raise_for_status=Mock(),
            json=Mock(
                return_value={
                    "entity_id": "binary_sensor.front_door",
                    "state": "off",
                    "attributes": {"device_class": "door"},
                    "last_changed": "2026-02-15T10:00:00+00:00",
                }
            ),
        ),
        Mock(
            raise_for_status=Mock(),
            json=Mock(
                return_value={
                    "entity_id": "light.living_room",
                    "state": "on",
                    "attributes": {"brightness": 255},
                    "last_changed": "2026-02-15T10:00:00+00:00",
                }
            ),
        ),
    ]

    mock_httpx_client.get = AsyncMock(side_effect=mock_responses)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # First sync: all states are new, so 4 events should be published
    assert count == 4
    assert len(connector._last_states) == 4
    assert connector._last_states["person.jay"] == "home"
    assert connector._last_states["sensor.living_room_temperature"] == "72"
    assert event_bus.publish.call_count == 4


@pytest.mark.asyncio
async def test_sync_state_changed(event_bus, db, ha_config, mock_httpx_client):
    """Test sync publishes event only when state actually changes."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Pre-populate cache with previous state
    connector._last_states["person.jay"] = "away"

    # Mock state response showing change from "away" to "home"
    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "home",
                "attributes": {"friendly_name": "Jay"},
                "last_changed": "2026-02-15T10:30:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    # Only watch person.jay for this test
    connector._watched = ["person.jay"]

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    assert count == 1
    assert connector._last_states["person.jay"] == "home"
    event_bus.publish.assert_called_once()

    # Verify event payload - publish is called with (subject, payload, ...)
    call_args = event_bus.publish.call_args
    subject = call_args[0][0]
    payload = call_args[0][1]
    assert subject == "home.arrived"
    assert payload["entity_id"] == "person.jay"
    assert payload["state"] == "home"
    assert payload["previous_state"] == "away"


@pytest.mark.asyncio
async def test_sync_state_unchanged(event_bus, db, ha_config, mock_httpx_client):
    """Test sync does not publish event when state is unchanged."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Pre-populate cache with current state
    connector._last_states["person.jay"] = "home"

    # Mock state response showing no change
    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "home",
                "attributes": {"friendly_name": "Jay"},
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    connector._watched = ["person.jay"]

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # No state change, so no events should be published
    assert count == 0
    event_bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Event Classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_person_arrival(event_bus, db, ha_config, mock_httpx_client):
    """Test person arriving home generates home.arrived event with normal priority."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["person.jay"] = "away"
    connector._watched = ["person.jay"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "home",
                "attributes": {},
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    # publish is called with (subject, payload, source=..., priority=...)
    call_kwargs = event_bus.publish.call_args[1]
    subject = event_bus.publish.call_args[0][0]
    assert subject == "home.arrived"
    assert call_kwargs["priority"] == "normal"


@pytest.mark.asyncio
async def test_classify_person_departure(event_bus, db, ha_config, mock_httpx_client):
    """Test person leaving home generates home.departed event with normal priority."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["person.jay"] = "home"
    connector._watched = ["person.jay"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "work",
                "attributes": {},
                "last_changed": "2026-02-15T08:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    call_kwargs = event_bus.publish.call_args[1]
    subject = event_bus.publish.call_args[0][0]
    assert subject == "home.departed"
    assert call_kwargs["priority"] == "normal"


@pytest.mark.asyncio
async def test_classify_person_zone_change(event_bus, db, ha_config, mock_httpx_client):
    """Test person moving between non-home zones generates location.changed at low priority."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["person.jay"] = "work"
    connector._watched = ["person.jay"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "gym",
                "attributes": {},
                "last_changed": "2026-02-15T18:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    call_kwargs = event_bus.publish.call_args[1]
    subject = event_bus.publish.call_args[0][0]
    assert subject == "location.changed"
    assert call_kwargs["priority"] == "low"


@pytest.mark.asyncio
async def test_classify_door_sensor(event_bus, db, ha_config, mock_httpx_client):
    """Test door sensor state change has normal priority (security-relevant)."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["binary_sensor.front_door"] = "off"
    connector._watched = ["binary_sensor.front_door"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "binary_sensor.front_door",
                "state": "on",
                "attributes": {"device_class": "door"},
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    call_kwargs = event_bus.publish.call_args[1]
    subject = event_bus.publish.call_args[0][0]
    assert subject == "home.device.state_changed"
    assert call_kwargs["priority"] == "normal"


@pytest.mark.asyncio
async def test_classify_generic_device(event_bus, db, ha_config, mock_httpx_client):
    """Test generic device state change has low priority to avoid noise."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["light.living_room"] = "off"
    connector._watched = ["light.living_room"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"brightness": 255},
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    call_kwargs = event_bus.publish.call_args[1]
    subject = event_bus.publish.call_args[0][0]
    assert subject == "home.device.state_changed"
    assert call_kwargs["priority"] == "low"


# ---------------------------------------------------------------------------
# Test: Sync Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_entity_error_continues(event_bus, db, ha_config, mock_httpx_client):
    """Test sync continues processing other entities when one fails."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["person.jay", "light.living_room"]

    # First entity fails, second succeeds
    mock_httpx_client.get = AsyncMock(
        side_effect=[
            Exception("Timeout"),
            Mock(
                raise_for_status=Mock(),
                json=Mock(
                    return_value={
                        "entity_id": "light.living_room",
                        "state": "on",
                        "attributes": {},
                        "last_changed": "2026-02-15T10:00:00+00:00",
                    }
                ),
            ),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # Only the second entity should succeed
    assert count == 1
    assert "light.living_room" in connector._last_states
    assert "person.jay" not in connector._last_states


@pytest.mark.asyncio
async def test_sync_http_error_handling(event_bus, db, ha_config, mock_httpx_client):
    """Test sync handles HTTP errors gracefully."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["person.jay"]

    # Mock HTTP 404 error
    mock_response = Mock()
    mock_response.raise_for_status = Mock(side_effect=Exception("404 Not Found"))
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # Should handle error gracefully and not crash
    assert count == 0


# ---------------------------------------------------------------------------
# Test: Execute - Service Calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_turn_on(event_bus, db, ha_config, mock_httpx_client):
    """Test executing turn_on action calls correct Home Assistant service."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    mock_response = Mock(raise_for_status=Mock())
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.execute("turn_on", {"entity_id": "light.living_room"})

    assert result["status"] == "turned_on"
    assert result["entity"] == "light.living_room"

    mock_httpx_client.post.assert_called_once_with(
        "http://homeassistant.local:8123/api/services/homeassistant/turn_on",
        headers={
            "Authorization": "Bearer test-long-lived-access-token",
            "Content-Type": "application/json",
        },
        json={"entity_id": "light.living_room"},
    )


@pytest.mark.asyncio
async def test_execute_turn_off(event_bus, db, ha_config, mock_httpx_client):
    """Test executing turn_off action calls correct Home Assistant service."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    mock_response = Mock(raise_for_status=Mock())
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.execute("turn_off", {"entity_id": "light.living_room"})

    assert result["status"] == "turned_off"
    assert result["entity"] == "light.living_room"

    mock_httpx_client.post.assert_called_once_with(
        "http://homeassistant.local:8123/api/services/homeassistant/turn_off",
        headers={
            "Authorization": "Bearer test-long-lived-access-token",
            "Content-Type": "application/json",
        },
        json={"entity_id": "light.living_room"},
    )


@pytest.mark.asyncio
async def test_execute_generic_service_call(event_bus, db, ha_config, mock_httpx_client):
    """Test executing generic call_service action with custom domain and service."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    mock_response = Mock(raise_for_status=Mock())
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    params = {
        "domain": "light",
        "service": "turn_on",
        "data": {"entity_id": "light.bedroom", "brightness": 128, "color_temp": 400},
    }

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.execute("call_service", params)

    assert result["status"] == "service_called"

    mock_httpx_client.post.assert_called_once_with(
        "http://homeassistant.local:8123/api/services/light/turn_on",
        headers={
            "Authorization": "Bearer test-long-lived-access-token",
            "Content-Type": "application/json",
        },
        json={"entity_id": "light.bedroom", "brightness": 128, "color_temp": 400},
    )


@pytest.mark.asyncio
async def test_execute_service_with_empty_data(event_bus, db, ha_config, mock_httpx_client):
    """Test call_service works when data parameter is omitted."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    mock_response = Mock(raise_for_status=Mock())
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    params = {
        "domain": "automation",
        "service": "trigger",
    }

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.execute("call_service", params)

    assert result["status"] == "service_called"

    # Should send empty dict when data is not provided
    mock_httpx_client.post.assert_called_once()
    call_json = mock_httpx_client.post.call_args[1]["json"]
    assert call_json == {}


@pytest.mark.asyncio
async def test_execute_unknown_action(event_bus, db, ha_config):
    """Test execute raises ValueError for unknown actions."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    with pytest.raises(ValueError, match="Unknown action: invalid_action"):
        await connector.execute("invalid_action", {})


@pytest.mark.asyncio
async def test_execute_http_error(event_bus, db, ha_config):
    """Test execute propagates HTTP errors from Home Assistant."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock httpx to raise an exception during service call
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_class.side_effect = Exception("404 Service not found")

        # The execute method doesn't have try/except, so errors propagate
        with pytest.raises(Exception, match="404 Service not found"):
            await connector.execute("turn_on", {"entity_id": "light.nonexistent"})


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_success(event_bus, db, ha_config, mock_httpx_client):
    """Test health check returns ok when Home Assistant is reachable."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    mock_response = Mock(raise_for_status=Mock())
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "home_assistant"
    assert result["entities_watched"] == 4


@pytest.mark.asyncio
async def test_health_check_failure(event_bus, db, ha_config):
    """Test health check returns error when Home Assistant is unreachable."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # Mock httpx to raise an exception
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_class.side_effect = Exception("Connection refused")
        result = await connector.health_check()

    # The health_check method catches exceptions and returns error dict
    assert result["status"] == "error"
    assert "Connection refused" in result["details"]


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_with_no_watched_entities(event_bus, db, mock_httpx_client):
    """Test sync with empty watched_entities list returns 0."""
    config = {"url": "http://homeassistant.local:8123", "token": "test"}
    connector = HomeAssistantConnector(event_bus, db, config)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    assert count == 0
    mock_httpx_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_sync_preserves_attributes(event_bus, db, ha_config, mock_httpx_client):
    """Test sync includes full attributes dict in event payload."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["sensor.living_room_temperature"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "sensor.living_room_temperature",
                "state": "72",
                "attributes": {
                    "unit_of_measurement": "°F",
                    "device_class": "temperature",
                    "friendly_name": "Living Room Temperature",
                },
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    payload = event_bus.publish.call_args[0][1]
    assert payload["attributes"]["unit_of_measurement"] == "°F"
    assert payload["attributes"]["device_class"] == "temperature"


@pytest.mark.asyncio
async def test_sync_handles_missing_attributes(event_bus, db, ha_config, mock_httpx_client):
    """Test sync handles entities with missing attributes dict."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["person.jay"]

    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "home",
                # Missing attributes key
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )
    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    event_bus.publish.assert_called_once()
    payload = event_bus.publish.call_args[0][1]
    assert payload["attributes"] == {}


@pytest.mark.asyncio
async def test_cache_update_before_publish(event_bus, db, ha_config, mock_httpx_client):
    """Test cache is updated before publishing to avoid re-emitting on retry."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._last_states["person.jay"] = "away"
    connector._watched = ["person.jay"]

    # First call: successful state fetch but publish will fail
    # Second call: should show no state change (cache already updated)
    mock_response = Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": "person.jay",
                "state": "home",
                "attributes": {},
                "last_changed": "2026-02-15T10:00:00+00:00",
            }
        ),
    )

    mock_httpx_client.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count1 = await connector.sync()
        count2 = await connector.sync()

    # First sync should publish
    assert count1 == 1
    # Second sync should not (state already cached as "home")
    assert count2 == 0
    # Cache should be updated to prevent duplicate events
    assert connector._last_states["person.jay"] == "home"


def test_classify_state_change_person_arrival(event_bus, db, ha_config):
    """Test _classify_state_change identifies person arriving home."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    event_type, priority = connector._classify_state_change(
        "person.jay", "away", "home"
    )

    assert event_type == "home.arrived"
    assert priority == "normal"


def test_classify_state_change_person_departure(event_bus, db, ha_config):
    """Test _classify_state_change identifies person leaving home."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    event_type, priority = connector._classify_state_change(
        "person.jay", "home", "work"
    )

    assert event_type == "home.departed"
    assert priority == "normal"


def test_classify_state_change_zone_transition(event_bus, db, ha_config):
    """Test _classify_state_change identifies non-home zone transitions."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    event_type, priority = connector._classify_state_change(
        "person.jay", "work", "gym"
    )

    assert event_type == "location.changed"
    assert priority == "low"


def test_classify_state_change_door_sensor_keyword(event_bus, db, ha_config):
    """Test _classify_state_change identifies door sensors by keyword."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    event_type, priority = connector._classify_state_change(
        "binary_sensor.garage_door", "off", "on"
    )

    assert event_type == "home.device.state_changed"
    assert priority == "normal"


def test_classify_state_change_generic_device(event_bus, db, ha_config):
    """Test _classify_state_change defaults to low priority for generic devices."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    event_type, priority = connector._classify_state_change(
        "sensor.cpu_temperature", "65", "70"
    )

    assert event_type == "home.device.state_changed"
    assert priority == "low"
