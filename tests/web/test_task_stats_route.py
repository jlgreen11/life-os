"""Tests for the GET /api/tasks/stats endpoint.

Verifies that the task statistics route correctly delegates to
TaskManager.get_task_stats() and returns the expected response shape.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for task-stats tests."""
    life_os = Mock()

    # Database mock
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.execute.return_value = mock_cursor
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / event store
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager — the focus of these tests
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()
    life_os.task_manager.get_task_stats = Mock(
        return_value={
            "pending": 0,
            "completed_today": 0,
            "overdue": 0,
            "by_domain": {},
        }
    )

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable",
        )
    )

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    # User model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector / browser stubs
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Endpoint existence and response shape
# ---------------------------------------------------------------------------


def test_task_stats_endpoint_exists(client):
    """GET /api/tasks/stats returns 200."""
    response = client.get("/api/tasks/stats")
    assert response.status_code == 200


def test_task_stats_returns_expected_keys(client):
    """Response contains all four expected stat keys."""
    response = client.get("/api/tasks/stats")
    data = response.json()
    assert "pending" in data
    assert "completed_today" in data
    assert "overdue" in data
    assert "by_domain" in data


def test_task_stats_empty(client):
    """Stats endpoint returns zeroes when no tasks exist."""
    response = client.get("/api/tasks/stats")
    data = response.json()
    assert data["pending"] == 0
    assert data["completed_today"] == 0
    assert data["overdue"] == 0
    assert data["by_domain"] == {}


# ---------------------------------------------------------------------------
# Stats reflect task state
# ---------------------------------------------------------------------------


def test_task_stats_with_tasks(mock_life_os, client):
    """Stats reflect actual task counts from the manager."""
    mock_life_os.task_manager.get_task_stats = Mock(
        return_value={
            "pending": 5,
            "completed_today": 2,
            "overdue": 1,
            "by_domain": {"work": 3, "personal": 2},
        }
    )
    response = client.get("/api/tasks/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["pending"] == 5
    assert data["completed_today"] == 2
    assert data["overdue"] == 1
    assert data["by_domain"]["work"] == 3
    assert data["by_domain"]["personal"] == 2


def test_task_stats_delegates_to_manager(mock_life_os, client):
    """The route calls TaskManager.get_task_stats() exactly once."""
    client.get("/api/tasks/stats")
    mock_life_os.task_manager.get_task_stats.assert_called_once()


# ---------------------------------------------------------------------------
# Route ordering: /api/tasks/stats must not collide with /api/tasks/{task_id}
# ---------------------------------------------------------------------------


def test_task_stats_not_captured_by_task_id_route(client):
    """Ensure /api/tasks/stats is not matched as /api/tasks/{task_id='stats'}."""
    response = client.get("/api/tasks/stats")
    # If route ordering is wrong, this would 404 or return a task-not-found error
    assert response.status_code == 200
    data = response.json()
    # Confirm we got stats, not a single-task response
    assert "pending" in data
