"""
Tests for the /api/dashboard/badges endpoint.

The badges endpoint replaces 5 separate full-feed requests (each fetching up
to 100 full items) with a single lightweight count-only response.  These tests
verify the response shape, count computation logic, and graceful error handling.
"""

from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for badge-count tests."""
    life_os = Mock()

    # Database mock that returns zero calendar events by default
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

    # Event bus / event store (required by create_web_app)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager — empty by default
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager — empty by default
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    # Signal extractor (used by health check and mood endpoints)
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable"
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
# Response shape tests
# ---------------------------------------------------------------------------


def test_badges_endpoint_exists(client):
    """GET /api/dashboard/badges returns 200."""
    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200


def test_badges_returns_badges_dict(client):
    """Response must contain a 'badges' key mapping topic IDs to integers."""
    response = client.get("/api/dashboard/badges")
    data = response.json()
    assert "badges" in data
    badges = data["badges"]
    assert isinstance(badges, dict)


def test_badges_has_all_expected_topics(client):
    """Badge counts are returned for all six main topics including insights."""
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    for topic in ("inbox", "messages", "email", "calendar", "tasks", "insights"):
        assert topic in badges, f"Missing badge count for topic '{topic}'"


def test_badges_values_are_integers(client):
    """Every badge count value must be a non-negative integer."""
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    for topic, count in badges.items():
        assert isinstance(count, int), f"Badge count for '{topic}' is not an int"
        assert count >= 0, f"Badge count for '{topic}' is negative"


# ---------------------------------------------------------------------------
# Count correctness tests
# ---------------------------------------------------------------------------


def test_badges_all_zero_when_empty(client):
    """All badge counts are 0 when there are no notifications, tasks, or events."""
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    for topic, count in badges.items():
        assert count == 0, f"Expected 0 for '{topic}' but got {count}"


def test_badges_task_count(mock_life_os, client):
    """Task count reflects the number of pending tasks."""
    mock_life_os.task_manager.get_pending_tasks = Mock(
        return_value=[{"id": f"t-{i}", "title": f"Task {i}"} for i in range(5)]
    )
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    assert badges["tasks"] == 5


def test_badges_email_count(mock_life_os, client):
    """Email badge counts only notifications whose domain contains 'email'."""
    mock_life_os.notification_manager.get_pending = Mock(
        return_value=[
            {"id": "n1", "domain": "email", "title": "Email 1"},
            {"id": "n2", "domain": "email", "title": "Email 2"},
            {"id": "n3", "domain": "message", "title": "Message"},
        ]
    )
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    assert badges["email"] == 2


def test_badges_messages_count(mock_life_os, client):
    """Messages badge counts notifications whose domain contains 'message' or 'signal'."""
    mock_life_os.notification_manager.get_pending = Mock(
        return_value=[
            {"id": "n1", "domain": "message", "title": "Signal msg"},
            {"id": "n2", "domain": "message", "title": "iMessage"},
            {"id": "n3", "domain": "email", "title": "Email"},
        ]
    )
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    assert badges["messages"] == 2


def test_badges_inbox_aggregates_all(mock_life_os, client):
    """Inbox count = all notifications + tasks + calendar events."""
    mock_life_os.notification_manager.get_pending = Mock(
        return_value=[
            {"id": "n1", "domain": "email", "title": "Email"},
            {"id": "n2", "domain": "email", "title": "Email 2"},
        ]
    )
    mock_life_os.task_manager.get_pending_tasks = Mock(
        return_value=[{"id": "t1", "title": "Task"}]
    )
    # Simulate 3 upcoming calendar events from the DB
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (3,)
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    # 2 notifications + 1 task + 3 calendar = 6
    assert badges["inbox"] == 6
    assert badges["tasks"] == 1
    assert badges["calendar"] == 3


# ---------------------------------------------------------------------------
# Graceful error handling
# ---------------------------------------------------------------------------


def test_badges_handles_notification_error(mock_life_os, client):
    """Endpoint returns 200 even if the notification manager raises."""
    mock_life_os.notification_manager.get_pending = Mock(
        side_effect=RuntimeError("DB unavailable")
    )
    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200
    badges = response.json()["badges"]
    # Notifications should default to 0 on error
    assert badges["email"] == 0
    assert badges["messages"] == 0


def test_badges_handles_task_error(mock_life_os, client):
    """Endpoint returns 200 even if the task manager raises."""
    mock_life_os.task_manager.get_pending_tasks = Mock(
        side_effect=RuntimeError("Task DB error")
    )
    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200
    assert response.json()["badges"]["tasks"] == 0


def test_badges_handles_calendar_db_error(mock_life_os, client):
    """Endpoint returns 200 even if the calendar DB query raises."""
    mock_life_os.db.get_connection = Mock(
        side_effect=RuntimeError("Calendar DB error")
    )
    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200
    assert response.json()["badges"]["calendar"] == 0


def test_badges_insights_count_zero_when_empty(client):
    """Insights badge count is 0 when the DB returns no active insights."""
    # The default mock returns (0,) for all DB queries, including insights.
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    assert badges["insights"] == 0


def test_badges_insights_count_from_db(mock_life_os, client):
    """Insights badge reflects the number of active (non-dismissed, non-expired) insights."""
    # Simulate 7 active insights in user_model.db
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (7,)
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )
    response = client.get("/api/dashboard/badges")
    badges = response.json()["badges"]
    assert badges["insights"] == 7


def test_badges_insights_handles_db_error(mock_life_os, client):
    """Insights badge defaults to 0 when the user_model DB query fails."""
    # Calendar and insights both use db.get_connection; make it raise.
    mock_life_os.db.get_connection = Mock(
        side_effect=RuntimeError("user_model DB unavailable")
    )
    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200
    assert response.json()["badges"]["insights"] == 0
