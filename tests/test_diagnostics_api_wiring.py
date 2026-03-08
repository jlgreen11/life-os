"""
Life OS — Tests for diagnostics API wiring of source_weight_manager and rules_engine.

Verifies that the /api/diagnostics/user-model endpoint correctly includes
diagnostics from source_weight_manager and rules_engine services, and that
the health assessment detects source weight feedback loop issues.
"""

import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


@pytest.fixture
def life_os_mock(db, event_store, user_model_store):
    """Create a minimal LifeOS mock with real db components."""
    mock = MagicMock()
    mock.db = db
    mock.event_store = event_store
    mock.user_model_store = user_model_store
    mock.signal_extractor = MagicMock()
    mock.vector_store = MagicMock()
    mock.event_bus = MagicMock()
    mock.event_bus.is_connected = False
    mock.connectors = []
    mock.notification_manager = MagicMock()
    mock.feedback_collector = MagicMock()
    mock.rules_engine = MagicMock()
    mock.task_manager = MagicMock()
    mock.ai_engine = MagicMock()
    mock.browser_orchestrator = MagicMock()
    mock.onboarding = MagicMock()
    # Remove get_diagnostics from services that shouldn't have it by default
    # so only the ones we explicitly set up will report
    for attr in ("prediction_engine", "routine_detector", "workflow_detector",
                 "semantic_fact_inferrer", "insight_engine", "behavioral_tracker",
                 "task_completion_detector", "conflict_detector"):
        setattr(mock, attr, None)
    return mock


@pytest.fixture
def client(life_os_mock):
    """Create a test client with the mocked LifeOS."""
    from web.app import create_web_app
    app = create_web_app(life_os_mock)
    return TestClient(app)


def test_diagnostics_includes_source_weight_manager(life_os_mock, client):
    """Diagnostics endpoint includes source_weight_manager when get_diagnostics() exists."""
    life_os_mock.source_weight_manager = MagicMock()
    life_os_mock.source_weight_manager.get_diagnostics.return_value = {
        "total_sources": 16,
        "total_interactions": 500,
        "feedback_loop_health": "healthy",
    }

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    assert "source_weight_manager" in data
    assert data["source_weight_manager"]["total_sources"] == 16
    assert data["source_weight_manager"]["feedback_loop_health"] == "healthy"


def test_diagnostics_includes_rules_engine(life_os_mock, client):
    """Diagnostics endpoint includes rules_engine when get_diagnostics() exists."""
    life_os_mock.rules_engine = MagicMock()
    life_os_mock.rules_engine.get_diagnostics.return_value = {
        "total_rules": 5,
        "active_rules": 3,
        "health": "healthy",
    }

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    assert "rules_engine" in data
    assert data["rules_engine"]["total_rules"] == 5


def test_diagnostics_skips_service_without_get_diagnostics(life_os_mock, client):
    """Services without get_diagnostics() method are gracefully skipped."""
    # Create a plain object without get_diagnostics
    plain_service = object()
    life_os_mock.source_weight_manager = plain_service

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    # source_weight_manager should not appear since it lacks get_diagnostics()
    assert "source_weight_manager" not in data


def test_diagnostics_detects_broken_source_weight_feedback(life_os_mock, client):
    """Health assessment flags broken source weight feedback loop."""
    life_os_mock.source_weight_manager = MagicMock()
    life_os_mock.source_weight_manager.get_diagnostics.return_value = {
        "total_sources": 16,
        "total_interactions": 200,
        "total_engagements": 0,
        "total_dismissals": 0,
        "feedback_loop_health": "broken",
    }

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    assert data["health"] == "degraded"
    assert any("Source weight feedback loop" in issue for issue in data["issues"])


def test_diagnostics_detects_no_feedback_source_weight(life_os_mock, client):
    """Health assessment flags no_feedback source weight status."""
    life_os_mock.source_weight_manager = MagicMock()
    life_os_mock.source_weight_manager.get_diagnostics.return_value = {
        "feedback_loop_health": "no_feedback",
    }

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    assert data["health"] == "degraded"
    assert any("Source weight feedback loop" in issue for issue in data["issues"])


def test_diagnostics_healthy_source_weight_no_issue(life_os_mock, client):
    """Healthy source weight feedback loop does not add issues."""
    life_os_mock.source_weight_manager = MagicMock()
    life_os_mock.source_weight_manager.get_diagnostics.return_value = {
        "feedback_loop_health": "healthy",
    }

    response = client.get("/api/diagnostics/user-model")
    assert response.status_code == 200
    data = response.json()
    assert not any("Source weight feedback loop" in issue for issue in data["issues"])
