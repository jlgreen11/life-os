"""
Tests for Home Assistant connector state persistence across restarts.

Verifies that entity state is persisted to state.db so that restarting
the connector does not flood the event bus with false state-change events.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

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
        "token": "test-token",
        "watched_entities": [
            "person.jay",
            "light.living_room",
            "binary_sensor.front_door",
        ],
    }


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient for HTTP requests."""
    client = Mock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    return client


def _make_state_response(entity_id: str, state: str, attributes: dict | None = None):
    """Helper to create a mock HA state API response."""
    return Mock(
        raise_for_status=Mock(),
        json=Mock(
            return_value={
                "entity_id": entity_id,
                "state": state,
                "attributes": attributes or {},
                "last_changed": "2026-03-01T10:00:00+00:00",
            }
        ),
    )


# ---------------------------------------------------------------------------
# Test: States are persisted to state.db after sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_persists_states_to_db(event_bus, db, ha_config, mock_httpx_client):
    """After sync() detects state changes, the cache is persisted to kv_store."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["person.jay"]

    mock_httpx_client.get = AsyncMock(
        return_value=_make_state_response("person.jay", "home")
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    assert count == 1

    # Verify persisted to kv_store
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            ("home_assistant:entity_states",),
        ).fetchone()

    assert row is not None
    persisted = json.loads(row["value"])
    assert persisted["person.jay"] == "home"


@pytest.mark.asyncio
async def test_sync_does_not_persist_when_no_changes(event_bus, db, ha_config, mock_httpx_client):
    """When no state changes occur, _persist_states is not called (no unnecessary writes)."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["person.jay"]
    # Pre-populate cache so state appears unchanged
    connector._last_states["person.jay"] = "home"

    mock_httpx_client.get = AsyncMock(
        return_value=_make_state_response("person.jay", "home")
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    assert count == 0

    # kv_store should have no entry (never persisted)
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            ("home_assistant:entity_states",),
        ).fetchone()

    assert row is None


# ---------------------------------------------------------------------------
# Test: Persisted states are loaded on re-initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persisted_states_loaded_on_init(event_bus, db, ha_config):
    """When kv_store has persisted entity states, they are loaded into _last_states on init."""
    # Seed the kv_store with prior states
    states = {"person.jay": "home", "light.living_room": "on"}
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("home_assistant:entity_states", json.dumps(states)),
        )

    # Create a new connector — it should load the persisted states
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    assert connector._last_states == states
    assert connector._last_states["person.jay"] == "home"
    assert connector._last_states["light.living_room"] == "on"


# ---------------------------------------------------------------------------
# Test: Restart does NOT publish false events for unchanged entities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_no_false_events_for_unchanged_entities(
    event_bus, db, ha_config, mock_httpx_client
):
    """After a restart with persisted state, unchanged entities produce zero events."""
    # Simulate prior run: persist states to DB
    prior_states = {
        "person.jay": "home",
        "light.living_room": "on",
        "binary_sensor.front_door": "off",
    }
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("home_assistant:entity_states", json.dumps(prior_states)),
        )

    # Create a "restarted" connector — it loads persisted states
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    assert connector._last_states == prior_states

    # All entities return the same state as before
    mock_httpx_client.get = AsyncMock(
        side_effect=[
            _make_state_response("person.jay", "home"),
            _make_state_response("light.living_room", "on"),
            _make_state_response("binary_sensor.front_door", "off"),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # No state changes — no events published
    assert count == 0
    event_bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Genuine state changes after restart ARE still published
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_publishes_genuine_state_changes(
    event_bus, db, ha_config, mock_httpx_client
):
    """After a restart, entities that genuinely changed state still produce events."""
    # Persist prior state
    prior_states = {
        "person.jay": "away",
        "light.living_room": "on",
        "binary_sensor.front_door": "off",
    }
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("home_assistant:entity_states", json.dumps(prior_states)),
        )

    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # person.jay changed from "away" to "home"; others unchanged
    mock_httpx_client.get = AsyncMock(
        side_effect=[
            _make_state_response("person.jay", "home"),
            _make_state_response("light.living_room", "on"),
            _make_state_response("binary_sensor.front_door", "off"),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # Only the genuine change should be published
    assert count == 1
    event_bus.publish.assert_called_once()

    call_args = event_bus.publish.call_args
    assert call_args[0][0] == "home.arrived"
    assert call_args[0][1]["entity_id"] == "person.jay"
    assert call_args[0][1]["previous_state"] == "away"
    assert call_args[0][1]["state"] == "home"


# ---------------------------------------------------------------------------
# Test: First-run (no persisted state) still works correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_no_persisted_state(event_bus, db, ha_config, mock_httpx_client):
    """On first run with no persisted state, all entities are published (expected behavior)."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)

    # No prior state in DB — _last_states should be empty
    assert connector._last_states == {}

    mock_httpx_client.get = AsyncMock(
        side_effect=[
            _make_state_response("person.jay", "home"),
            _make_state_response("light.living_room", "on"),
            _make_state_response("binary_sensor.front_door", "off"),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count = await connector.sync()

    # All 3 entities are new — all should be published
    assert count == 3
    assert event_bus.publish.call_count == 3

    # And now the states should be persisted for next run
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            ("home_assistant:entity_states",),
        ).fetchone()

    assert row is not None
    persisted = json.loads(row["value"])
    assert persisted == {
        "person.jay": "home",
        "light.living_room": "on",
        "binary_sensor.front_door": "off",
    }


# ---------------------------------------------------------------------------
# Test: Full round-trip (sync → persist → new connector → sync)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_round_trip_persist_and_reload(
    event_bus, db, ha_config, mock_httpx_client
):
    """End-to-end: sync persists states, new connector loads them, no false events."""
    # --- First "run": connector syncs and persists states ---
    connector1 = HomeAssistantConnector(event_bus, db, ha_config)

    mock_httpx_client.get = AsyncMock(
        side_effect=[
            _make_state_response("person.jay", "home"),
            _make_state_response("light.living_room", "on"),
            _make_state_response("binary_sensor.front_door", "off"),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count1 = await connector1.sync()

    assert count1 == 3  # First run: all new

    # --- Simulate restart: create a new connector with the same DB ---
    event_bus.publish.reset_mock()

    connector2 = HomeAssistantConnector(event_bus, db, ha_config)

    # States should be loaded from DB
    assert connector2._last_states["person.jay"] == "home"
    assert connector2._last_states["light.living_room"] == "on"
    assert connector2._last_states["binary_sensor.front_door"] == "off"

    # Same states returned by HA API
    mock_httpx_client.get = AsyncMock(
        side_effect=[
            _make_state_response("person.jay", "home"),
            _make_state_response("light.living_room", "on"),
            _make_state_response("binary_sensor.front_door", "off"),
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        count2 = await connector2.sync()

    # No false events — states match persisted cache
    assert count2 == 0
    event_bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Persistence updates existing kv_store row (not duplicates)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_updates_existing_row(event_bus, db, ha_config, mock_httpx_client):
    """Multiple sync cycles with changes update the same kv_store row, not create new ones."""
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    connector._watched = ["light.living_room"]

    # First sync: light turns on
    mock_httpx_client.get = AsyncMock(
        return_value=_make_state_response("light.living_room", "on")
    )
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    # Second sync: light turns off
    mock_httpx_client.get = AsyncMock(
        return_value=_make_state_response("light.living_room", "off")
    )
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        await connector.sync()

    # Should only have one row in kv_store, not two
    with db.get_connection("state") as conn:
        rows = conn.execute(
            "SELECT COUNT(*) as cnt FROM kv_store WHERE key = ?",
            ("home_assistant:entity_states",),
        ).fetchone()

    assert rows["cnt"] == 1

    # And the value should reflect the latest state
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            ("home_assistant:entity_states",),
        ).fetchone()

    persisted = json.loads(row["value"])
    assert persisted["light.living_room"] == "off"


# ---------------------------------------------------------------------------
# Test: Load failure is non-fatal (fail-open)
# ---------------------------------------------------------------------------


def test_load_failure_is_non_fatal(event_bus, db, ha_config):
    """If loading persisted states fails, the connector starts with empty cache."""
    # Corrupt the kv_store entry
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            ("home_assistant:entity_states", "not-valid-json{{{"),
        )

    # Should not raise — fails open with empty cache
    connector = HomeAssistantConnector(event_bus, db, ha_config)
    assert connector._last_states == {}
