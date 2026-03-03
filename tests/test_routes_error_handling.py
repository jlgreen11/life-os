"""
Tests for error handling and graceful degradation in web/routes.py

Covers:
1. Diagnostics endpoint returns correct 'actionable' count (not 'unread')
2. User-model endpoints return graceful error JSON on service failure
3. Task endpoints return error JSON on service failure
4. Notification endpoints return error JSON on service failure
5. Preference and feedback endpoints return error JSON on DB/service failure
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with all required services.

    Mirrors the fixture in test_web_routes.py with the minimum set of mocks
    needed for the endpoints under test.
    """
    life_os = Mock()

    # Mock database manager with context manager support
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock())
    )
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # Mock event bus
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # Mock event store
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=100)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-123")

    # Mock vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 50, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # Mock signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable"
    ))

    # Mock notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 5, "delivered": 100})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.mark_read = AsyncMock()
    life_os.notification_manager.dismiss = AsyncMock()
    life_os.notification_manager.mark_acted_on = AsyncMock()
    life_os.notification_manager.get_digest = AsyncMock(return_value="Daily digest")

    # Mock feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 10})
    life_os.feedback_collector.process_explicit_feedback = AsyncMock()

    # Mock AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="Morning briefing")
    life_os.ai_engine.draft_reply = AsyncMock(return_value="Draft message")
    life_os.ai_engine.search_life = AsyncMock(return_value="Search result")

    # Mock task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = AsyncMock(return_value="task-123")
    life_os.task_manager.update_task = AsyncMock()
    life_os.task_manager.complete_task = AsyncMock()
    life_os.task_manager.get_task_stats = Mock(return_value={
        "pending": 3, "completed_today": 1, "overdue": 0, "by_domain": {}
    })

    # Mock rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = AsyncMock(return_value="rule-123")
    life_os.rules_engine.remove_rule = AsyncMock()

    # Mock user model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Mock connectors
    life_os.connectors = []

    # Mock browser orchestrator
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Mock onboarding manager
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Mock connector management methods
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = AsyncMock(return_value={"success": True})
    life_os.enable_connector = AsyncMock(return_value={"status": "started"})
    life_os.disable_connector = AsyncMock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug 1: Diagnostics 'actionable' count (was 'unread')
# ---------------------------------------------------------------------------


def test_diagnostics_notifications_uses_actionable_status(client, mock_life_os):
    """Test that diagnostics endpoint queries 'pending'/'delivered' statuses, not 'unread'.

    The old code queried WHERE status = 'unread' which doesn't exist in the schema.
    The fix queries WHERE status IN ('pending', 'delivered') and returns the count
    under the 'actionable' key instead of 'unread'.
    """
    # Set up mock DB to track which queries are executed
    mock_conn = Mock()
    call_results = {}

    def execute_side_effect(query, *args):
        """Return different counts based on the query."""
        mock_row = Mock()
        if "status IN ('pending', 'delivered')" in query:
            mock_row.__getitem__ = lambda self, key: 7
            call_results["actionable_query_called"] = True
        elif "status = 'unread'" in query:
            mock_row.__getitem__ = lambda self, key: 0
            call_results["unread_query_called"] = True
        elif "COUNT(*) as c FROM notifications WHERE" in query:
            # last_24h query
            mock_row.__getitem__ = lambda self, key: 2
        elif "COUNT(*) as c FROM notifications" in query:
            mock_row.__getitem__ = lambda self, key: 15
        elif "COUNT(*) as c FROM predictions" in query:
            mock_row.__getitem__ = lambda self, key: 10
        elif "MAX(" in query:
            mock_row.__getitem__ = lambda self, key: "2026-03-03T10:00:00"
        elif "COUNT(*) as c FROM events" in query:
            mock_row.__getitem__ = lambda self, key: 50
        else:
            mock_row.__getitem__ = lambda self, key: 0
        result = Mock()
        result.fetchone = Mock(return_value=mock_row)
        return result

    mock_conn.execute = Mock(side_effect=execute_side_effect)
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Also mock event store and user_model_store for the diagnostics endpoint
    mock_life_os.event_store.get_event_count = Mock(return_value=50)
    mock_life_os.user_model_store.get_signal_profile = Mock(return_value=None)

    response = client.get("/api/diagnostics/pipeline")
    assert response.status_code == 200
    data = response.json()

    # Verify the notifications section uses 'actionable', not 'unread'
    notifs = data.get("notifications", {})
    if not notifs.get("error"):
        assert "actionable" in notifs, f"Expected 'actionable' key in notifications, got: {notifs.keys()}"
        assert "unread" not in notifs, f"'unread' key should not exist in notifications"

    # Verify the correct query was called
    assert call_results.get("actionable_query_called"), "Should query for pending/delivered statuses"
    assert not call_results.get("unread_query_called"), "Should NOT query for 'unread' status"


# ---------------------------------------------------------------------------
# User-model endpoint error handling
# ---------------------------------------------------------------------------


def test_user_model_returns_error_on_service_failure(client, mock_life_os):
    """GET /api/user-model returns JSON error (not 500 crash) when signal extractor fails."""
    mock_life_os.signal_extractor.get_user_summary = Mock(
        side_effect=Exception("user_model.db is corrupted")
    )

    response = client.get("/api/user-model")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "corrupted" in data["error"]


def test_user_model_facts_returns_error_on_failure(client, mock_life_os):
    """GET /api/user-model/facts returns empty list + error on DB failure."""
    mock_life_os.user_model_store.get_semantic_facts = Mock(
        side_effect=Exception("database disk image is malformed")
    )

    response = client.get("/api/user-model/facts")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["facts"] == []


def test_delete_fact_returns_error_on_failure(client, mock_life_os):
    """DELETE /api/user-model/facts/{key} returns error JSON on DB failure."""
    mock_conn = Mock()
    mock_conn.execute = Mock(side_effect=Exception("database is locked"))
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    response = client.delete("/api/user-model/facts/some-key")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "locked" in data["error"]


# ---------------------------------------------------------------------------
# Task endpoint error handling
# ---------------------------------------------------------------------------


def test_task_stats_returns_error_on_failure(client, mock_life_os):
    """GET /api/tasks/stats returns error JSON when task manager fails."""
    mock_life_os.task_manager.get_task_stats = Mock(
        side_effect=Exception("state.db is corrupted")
    )

    response = client.get("/api/tasks/stats")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "corrupted" in data["error"]


def test_update_task_returns_error_on_failure(client, mock_life_os):
    """PATCH /api/tasks/{task_id} returns error JSON when update fails."""
    mock_life_os.task_manager.update_task = AsyncMock(
        side_effect=Exception("database disk image is malformed")
    )

    response = client.patch("/api/tasks/task-123", json={"status": "completed"})
    assert response.status_code == 500
    data = response.json()
    assert "error" in data


def test_complete_task_returns_error_on_failure(client, mock_life_os):
    """POST /api/tasks/{task_id}/complete returns error JSON when completion fails."""
    mock_life_os.task_manager.complete_task = AsyncMock(
        side_effect=Exception("task not found")
    )

    response = client.post("/api/tasks/task-999/complete")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Notification endpoint error handling
# ---------------------------------------------------------------------------


def test_list_notifications_returns_error_on_failure(client, mock_life_os):
    """GET /api/notifications returns empty list + error on service failure."""
    mock_life_os.notification_manager.get_pending = Mock(
        side_effect=Exception("state.db is corrupted")
    )

    response = client.get("/api/notifications")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["notifications"] == []


def test_mark_read_returns_error_on_failure(client, mock_life_os):
    """POST /api/notifications/{id}/read returns error JSON on failure."""
    mock_life_os.notification_manager.mark_read = AsyncMock(
        side_effect=Exception("notification not found")
    )

    response = client.post("/api/notifications/notif-123/read")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data


def test_notification_digest_returns_error_on_failure(client, mock_life_os):
    """GET /api/notifications/digest returns error JSON on failure."""
    mock_life_os.notification_manager.get_digest = AsyncMock(
        side_effect=Exception("digest generation failed")
    )

    response = client.get("/api/notifications/digest")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["digest"] is None


# ---------------------------------------------------------------------------
# Preference endpoint error handling
# ---------------------------------------------------------------------------


def test_get_preferences_returns_error_on_failure(client, mock_life_os):
    """GET /api/preferences returns empty list + error on DB failure."""
    mock_conn = Mock()
    mock_conn.execute = Mock(side_effect=Exception("preferences.db is corrupted"))
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    response = client.get("/api/preferences")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["preferences"] == []


def test_update_preference_returns_error_on_failure(client, mock_life_os):
    """PUT /api/preferences returns error JSON on DB failure."""
    mock_conn = Mock()
    mock_conn.execute = Mock(side_effect=Exception("database is locked"))
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    response = client.put("/api/preferences", json={"key": "theme", "value": "dark"})
    assert response.status_code == 500
    data = response.json()
    assert "error" in data


# ---------------------------------------------------------------------------
# Feedback endpoint error handling
# ---------------------------------------------------------------------------


def test_submit_feedback_returns_error_on_failure(client, mock_life_os):
    """POST /api/feedback returns error JSON on service failure."""
    mock_life_os.feedback_collector.process_explicit_feedback = AsyncMock(
        side_effect=Exception("feedback processing failed")
    )

    response = client.post("/api/feedback", json={"message": "test feedback"})
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "feedback processing failed" in data["error"]


# ---------------------------------------------------------------------------
# Happy-path checks (ensure error handling doesn't break normal operation)
# ---------------------------------------------------------------------------


def test_user_model_works_normally(client, mock_life_os):
    """GET /api/user-model still works when service is healthy."""
    response = client.get("/api/user-model")
    assert response.status_code == 200
    data = response.json()
    assert "facts" in data


def test_task_stats_works_normally(client, mock_life_os):
    """GET /api/tasks/stats still works when service is healthy."""
    response = client.get("/api/tasks/stats")
    assert response.status_code == 200
    data = response.json()
    assert "pending" in data


def test_notifications_works_normally(client, mock_life_os):
    """GET /api/notifications still works when service is healthy."""
    response = client.get("/api/notifications")
    assert response.status_code == 200
    data = response.json()
    assert "notifications" in data


def test_preferences_works_normally(client, mock_life_os):
    """GET /api/preferences still works when service is healthy."""
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[])))
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock())
    )

    response = client.get("/api/preferences")
    assert response.status_code == 200
    data = response.json()
    assert "preferences" in data
