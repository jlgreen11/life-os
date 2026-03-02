"""
Tests that the /health endpoint surfaces connector errors and DB degradation
so the dashboard status bar can display them prominently.

Verifies:
- Connector error details are returned in health.connectors[]
- DB degradation is reflected in health.db_status
- Multiple simultaneous connector errors are all reported
- Mixed healthy/unhealthy connectors are correctly classified
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with all required services."""
    life_os = Mock()

    # Mock database manager
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock())
    )
    life_os.db.get_database_health = Mock(
        return_value={
            "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
            "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
            "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
            "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
            "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
        }
    )

    # Mock event bus
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True

    # Mock event store
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=100)

    # Mock vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 50, "dimensions": 384})

    # No connectors by default
    life_os.connectors = []

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Connector error surfacing
# ---------------------------------------------------------------------------


def test_health_returns_connector_error_details(client, mock_life_os):
    """Connector health_check failure should appear in health.connectors with
    status='error' and a details string describing the problem."""
    failing = Mock()
    failing.CONNECTOR_ID = "google"
    failing.health_check = AsyncMock(side_effect=Exception("Authentication failed"))
    mock_life_os.connectors = [failing]

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert len(data["connectors"]) == 1
    conn = data["connectors"][0]
    assert conn["connector"] == "google"
    assert conn["status"] == "error"
    assert "Authentication failed" in conn["details"]


def test_health_returns_multiple_connector_errors(client, mock_life_os):
    """When several connectors fail, all errors should be reported."""
    google = Mock()
    google.CONNECTOR_ID = "google"
    google.health_check = AsyncMock(side_effect=Exception("Authentication failed"))

    signal = Mock()
    signal.CONNECTOR_ID = "signal"
    signal.health_check = AsyncMock(side_effect=Exception("signal-cli not found"))

    mock_life_os.connectors = [google, signal]

    data = client.get("/health").json()

    error_connectors = [c for c in data["connectors"] if c["status"] == "error"]
    assert len(error_connectors) == 2
    names = {c["connector"] for c in error_connectors}
    assert names == {"google", "signal"}


def test_health_mixed_healthy_and_failing_connectors(client, mock_life_os):
    """Healthy connectors should not be marked as errors; only failing ones
    should have status='error'."""
    healthy = Mock()
    healthy.CONNECTOR_ID = "caldav"
    healthy.health_check = AsyncMock(
        return_value={"connector": "caldav", "status": "ok", "details": "synced"}
    )

    failing = Mock()
    failing.CONNECTOR_ID = "google"
    failing.health_check = AsyncMock(side_effect=Exception("Token expired"))

    mock_life_os.connectors = [healthy, failing]

    data = client.get("/health").json()

    assert len(data["connectors"]) == 2
    statuses = {c["connector"]: c["status"] for c in data["connectors"]}
    assert statuses["caldav"] == "ok"
    assert statuses["google"] == "error"


def test_health_connector_timeout_reported_as_error(client, mock_life_os):
    """A connector that times out should appear as status='error' with
    details='timeout'."""
    import asyncio

    slow = Mock()
    slow.CONNECTOR_ID = "slow_service"
    # Simulate a timeout by raising asyncio.TimeoutError (the health route
    # catches this with wait_for).
    slow.health_check = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_life_os.connectors = [slow]

    data = client.get("/health").json()

    assert len(data["connectors"]) == 1
    conn = data["connectors"][0]
    assert conn["status"] == "error"
    assert conn["details"] == "timeout"


# ---------------------------------------------------------------------------
# DB degradation surfacing
# ---------------------------------------------------------------------------


def test_health_db_status_ok_when_all_healthy(client, mock_life_os):
    """When every database passes integrity checks, db_status should be 'ok'."""
    data = client.get("/health").json()
    assert data["db_status"] == "ok"


def test_health_db_status_degraded_when_corrupted(client, mock_life_os):
    """A corrupted database should cause db_status='degraded'."""
    mock_life_os.db.get_database_health.return_value = {
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {
            "status": "error",
            "errors": ["database disk image is malformed"],
            "path": "/tmp/user_model.db",
            "size_bytes": 1024,
        },
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    }

    data = client.get("/health").json()

    assert data["db_status"] == "degraded"
    assert data["db_health"]["user_model"]["status"] == "error"
    assert "malformed" in data["db_health"]["user_model"]["errors"][0]


def test_health_db_status_degraded_multiple_dbs(client, mock_life_os):
    """Multiple corrupted databases should still produce db_status='degraded'."""
    mock_life_os.db.get_database_health.return_value = {
        "events": {
            "status": "error",
            "errors": ["database disk image is malformed"],
            "path": "/tmp/events.db",
            "size_bytes": 1024,
        },
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {
            "status": "error",
            "errors": ["database disk image is malformed"],
            "path": "/tmp/user_model.db",
            "size_bytes": 1024,
        },
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    }

    data = client.get("/health").json()
    assert data["db_status"] == "degraded"


# ---------------------------------------------------------------------------
# Combined connector errors + DB degradation
# ---------------------------------------------------------------------------


def test_health_reports_both_connector_errors_and_db_degradation(client, mock_life_os):
    """The health endpoint should surface connector errors AND DB degradation
    simultaneously when both conditions exist."""
    failing = Mock()
    failing.CONNECTOR_ID = "google"
    failing.health_check = AsyncMock(side_effect=Exception("Auth expired"))
    mock_life_os.connectors = [failing]

    mock_life_os.db.get_database_health.return_value = {
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {
            "status": "error",
            "errors": ["database disk image is malformed"],
            "path": "/tmp/user_model.db",
            "size_bytes": 1024,
        },
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    }

    data = client.get("/health").json()

    # Both issues should be visible in the response
    assert data["db_status"] == "degraded"
    error_conns = [c for c in data["connectors"] if c["status"] == "error"]
    assert len(error_conns) == 1
    assert error_conns[0]["connector"] == "google"
