"""
Life OS — Tests for /api/insights/summary endpoint.

Verifies the insights aggregation endpoint returns 200 with the correct
structure, including when signal profiles are empty or missing.
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
    mock.signal_extractor.get_user_summary.return_value = {"profiles": {}}
    mock.vector_store = MagicMock()
    mock.event_bus = MagicMock()
    mock.event_bus.is_connected = False
    mock.connectors = []
    mock.notification_manager = MagicMock()
    mock.notification_manager.get_stats.return_value = {}
    mock.feedback_collector = MagicMock()
    mock.feedback_collector.get_feedback_summary.return_value = {}
    mock.rules_engine = MagicMock()
    mock.task_manager = MagicMock()
    mock.ai_engine = MagicMock()
    mock.browser_orchestrator = MagicMock()
    mock.onboarding = MagicMock()
    return mock


@pytest.fixture
def client(life_os_mock):
    from web.app import create_web_app
    app = create_web_app(life_os_mock)
    return TestClient(app)


def test_insights_summary_returns_200(client):
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert isinstance(data["insights"], list)
    assert "generated_at" in data
