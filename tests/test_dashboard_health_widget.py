"""
Tests for the dashboard system-status sidebar widget.

Verifies:
- The dashboard template contains the System Status sidebar section
- The updateSidebarConnectorStatus() JS function is defined
- The loadStatus() function calls updateSidebarConnectorStatus()
- The /health endpoint returns all fields needed by the sidebar widget
- Periodic polling is wired up via setInterval(loadStatus, ...)
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app
from web.template import HTML_TEMPLATE


@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with all required services."""
    life_os = Mock()

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

    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True

    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=100)

    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 50, "dimensions": 384})

    life_os.connectors = []

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Template structure tests
# ---------------------------------------------------------------------------


def test_template_contains_system_status_section():
    """The sidebar should contain a 'System Status' widget section."""
    assert "System Status" in HTML_TEMPLATE
    assert 'id="systemStatusContent"' in HTML_TEMPLATE


def test_template_contains_connector_dot_css():
    """CSS classes for connector status dots must be defined."""
    assert ".connector-dot.ok" in HTML_TEMPLATE
    assert ".connector-dot.error" in HTML_TEMPLATE
    assert ".connector-dot.unknown" in HTML_TEMPLATE


def test_template_contains_connector_row_css():
    """CSS for connector rows must be defined."""
    assert ".connector-row" in HTML_TEMPLATE


def test_template_defines_update_sidebar_function():
    """The updateSidebarConnectorStatus() JS function must be defined."""
    assert "function updateSidebarConnectorStatus" in HTML_TEMPLATE


def test_template_load_status_calls_sidebar_update():
    """loadStatus() must call updateSidebarConnectorStatus to populate
    the sidebar widget from the same /health fetch."""
    assert "updateSidebarConnectorStatus(health)" in HTML_TEMPLATE


def test_template_has_periodic_status_polling():
    """loadStatus should be called on a setInterval for periodic refresh."""
    assert "setInterval(loadStatus," in HTML_TEMPLATE


# ---------------------------------------------------------------------------
# API contract tests — /health returns fields needed by sidebar widget
# ---------------------------------------------------------------------------


def test_health_returns_connectors_list(client):
    """The /health response must include a 'connectors' list."""
    data = client.get("/health").json()
    assert "connectors" in data
    assert isinstance(data["connectors"], list)


def test_health_returns_db_status_field(client):
    """The /health response must include a 'db_status' field."""
    data = client.get("/health").json()
    assert "db_status" in data
    assert data["db_status"] in ("ok", "degraded")


def test_health_connector_entry_has_required_fields(client, mock_life_os):
    """Each connector entry should have 'connector' and 'status' fields."""
    healthy = Mock()
    healthy.CONNECTOR_ID = "caldav"
    healthy.health_check = AsyncMock(
        return_value={"connector": "caldav", "status": "ok", "details": "synced"}
    )
    mock_life_os.connectors = [healthy]

    data = client.get("/health").json()
    assert len(data["connectors"]) == 1
    entry = data["connectors"][0]
    assert "connector" in entry
    assert "status" in entry


def test_health_error_connector_has_details(client, mock_life_os):
    """A failing connector's entry should include a 'details' field."""
    failing = Mock()
    failing.CONNECTOR_ID = "google"
    failing.health_check = AsyncMock(side_effect=Exception("Token expired"))
    mock_life_os.connectors = [failing]

    data = client.get("/health").json()
    entry = data["connectors"][0]
    assert entry["status"] == "error"
    assert "details" in entry
    assert "Token expired" in entry["details"]


def test_health_db_health_per_database(client):
    """The /health response should include per-database health info in
    db_health so the sidebar can identify which DB is degraded."""
    data = client.get("/health").json()
    assert "db_health" in data
    assert "user_model" in data["db_health"]
    assert "status" in data["db_health"]["user_model"]
