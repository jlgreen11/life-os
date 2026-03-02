"""
Comprehensive test suite for web/routes.py

Tests all REST API endpoints, including health checks, command bar, dashboard feed,
briefing, search, tasks, notifications, drafts, rules, user model, insights,
preferences, feedback, events, connectors, context ingestion, admin endpoints,
and the main UI routes.

Coverage: 40+ endpoints, ~1,000 LOC
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with all required services."""
    life_os = Mock()

    # Mock database manager with context manager support
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock()
    life_os.db.get_connection = Mock(return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock()))
    # Return a healthy per-database status dict so the /health endpoint can
    # iterate it without hitting the real filesystem.
    life_os.db.get_database_health = Mock(return_value={
        "events":      {"status": "ok", "errors": [], "path": "/tmp/events.db",      "size_bytes": 1024},
        "entities":    {"status": "ok", "errors": [], "path": "/tmp/entities.db",    "size_bytes": 1024},
        "state":       {"status": "ok", "errors": [], "path": "/tmp/state.db",       "size_bytes": 1024},
        "user_model":  {"status": "ok", "errors": [], "path": "/tmp/user_model.db",  "size_bytes": 1024},
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

    # Mock task manager — get_tasks() is the method the route now calls;
    # get_pending_tasks() is kept for backward-compat shim tests.
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = AsyncMock(return_value="task-123")
    life_os.task_manager.update_task = AsyncMock()
    life_os.task_manager.complete_task = AsyncMock()

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
def app(mock_life_os):
    """Create a FastAPI test app with mocked dependencies."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

def test_health_endpoint(client, mock_life_os):
    """Test /health returns system health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert data["event_bus"] is True
    assert data["events_stored"] == 100
    assert "vector_store" in data
    assert "connectors" in data


def test_health_with_connector_failure(client, mock_life_os):
    """Test /health handles individual connector failures gracefully."""
    failing_connector = Mock()
    failing_connector.CONNECTOR_ID = "failing"
    failing_connector.health_check = AsyncMock(side_effect=Exception("Connection failed"))
    mock_life_os.connectors = [failing_connector]

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["connectors"]) == 1
    assert data["connectors"][0]["status"] == "error"


def test_status_endpoint(client, mock_life_os):
    """Test /api/status returns detailed system status."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["event_count"] == 100
    assert "vector_store" in data
    assert "user_model" in data
    assert "notification_stats" in data
    assert "feedback_summary" in data


# ---------------------------------------------------------------------------
# Command Bar
# ---------------------------------------------------------------------------

def test_command_search(client, mock_life_os):
    """Test command bar handles search commands."""
    mock_life_os.vector_store.search.return_value = [
        {"text": "Result 1", "score": 0.9}
    ]

    response = client.post("/api/command", json={"text": "search meetings"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "search_results"
    assert len(data["results"]) == 1
    mock_life_os.vector_store.search.assert_called_once_with("meetings", limit=10)


def test_command_find(client, mock_life_os):
    """Test command bar handles find commands (alias for search)."""
    response = client.post("/api/command", json={"text": "find emails from john"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "search_results"
    mock_life_os.vector_store.search.assert_called_once()


def test_command_task(client, mock_life_os):
    """Test command bar creates tasks from task/todo commands."""
    response = client.post("/api/command", json={"text": "task Buy groceries"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "task_created"
    assert data["task_id"] == "task-123"
    mock_life_os.task_manager.create_task.assert_called_once_with(title="Buy groceries")


def test_command_todo(client, mock_life_os):
    """Test command bar handles todo commands (alias for task)."""
    response = client.post("/api/command", json={"text": "todo Review PR"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "task_created"


def test_command_briefing(client, mock_life_os):
    """Test command bar generates briefing."""
    response = client.post("/api/command", json={"text": "briefing"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "briefing"
    assert data["content"] == "Morning briefing"
    mock_life_os.ai_engine.generate_briefing.assert_called_once()


def test_command_morning_briefing(client, mock_life_os):
    """Test command bar handles 'morning briefing' variant."""
    response = client.post("/api/command", json={"text": "morning briefing"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "briefing"


def test_command_draft(client, mock_life_os):
    """Test command bar generates draft messages."""
    response = client.post("/api/command", json={"text": "draft Reply to John's email"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "draft"
    assert data["content"] == "Draft message"
    mock_life_os.ai_engine.draft_reply.assert_called_once()


def test_command_ai_query(client, mock_life_os):
    """Test command bar falls back to AI for unrecognized commands."""
    response = client.post("/api/command", json={"text": "What's my schedule today?"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "ai_response"
    assert data["content"] == "Search result"
    mock_life_os.ai_engine.search_life.assert_called_once()


def test_command_empty(client):
    """Test command bar rejects empty commands."""
    response = client.post("/api/command", json={"text": ""})
    assert response.status_code == 400


def test_command_publishes_telemetry(client, mock_life_os):
    """Test command bar publishes telemetry events."""
    client.post("/api/command", json={"text": "search test"})
    mock_life_os.event_bus.publish.assert_called_once()
    args = mock_life_os.event_bus.publish.call_args
    assert args[0][0] == "system.user.command"
    assert args[1]["source"] == "web_api"


# ---------------------------------------------------------------------------
# Dashboard Feed
# ---------------------------------------------------------------------------

def test_dashboard_feed_default(client, mock_life_os):
    """Test dashboard feed returns unified inbox by default."""
    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "count" in data
    assert data["topic"] == "inbox"


def test_dashboard_feed_with_notifications(client, mock_life_os):
    """Test dashboard feed includes notifications."""
    mock_life_os.notification_manager.get_pending.return_value = [
        {
            "id": "n1",
            "title": "Test notification",
            "body": "Body text",
            "priority": "high",
            "created_at": "2026-02-15T12:00:00Z",
            "domain": "email",
            "source_event_id": "evt-1",
        }
    ]

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["kind"] == "notification"
    assert data["items"][0]["channel"] == "email"


def test_dashboard_feed_with_tasks(client, mock_life_os):
    """Test dashboard feed includes tasks."""
    mock_life_os.task_manager.get_pending_tasks.return_value = [
        {
            "id": "t1",
            "title": "Test task",
            "description": "Task description",
            "priority": "normal",
            "created_at": "2026-02-15T12:00:00Z",
            "domain": "work"
        }
    ]

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["kind"] == "task"
    assert data["items"][0]["channel"] == "task"


def test_dashboard_feed_topic_filter(client, mock_life_os):
    """Test dashboard feed filters by topic."""
    response = client.get("/api/dashboard/feed?topic=tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "tasks"


def test_dashboard_feed_limit(client, mock_life_os):
    """Test dashboard feed respects limit parameter."""
    response = client.get("/api/dashboard/feed?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 10


def test_dashboard_feed_priority_sorting(client, mock_life_os):
    """Test dashboard feed sorts by priority."""
    mock_life_os.notification_manager.get_pending.return_value = [
        {"id": "n1", "priority": "normal", "created_at": "2026-02-15T12:00:00Z",
         "title": "Normal", "body": "", "source": "test", "metadata": {}},
        {"id": "n2", "priority": "critical", "created_at": "2026-02-15T11:00:00Z",
         "title": "Critical", "body": "", "source": "test", "metadata": {}},
        {"id": "n3", "priority": "high", "created_at": "2026-02-15T13:00:00Z",
         "title": "High", "body": "", "source": "test", "metadata": {}}
    ]

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()
    # Should be sorted: critical, high, normal
    assert data["items"][0]["priority"] == "critical"
    assert data["items"][1]["priority"] == "high"
    assert data["items"][2]["priority"] == "normal"


# ---------------------------------------------------------------------------
# Briefing
# ---------------------------------------------------------------------------

def test_get_briefing(client, mock_life_os):
    """Test GET /api/briefing generates morning briefing."""
    response = client.get("/api/briefing")
    assert response.status_code == 200
    data = response.json()
    assert data["briefing"] == "Morning briefing"
    assert "generated_at" in data
    mock_life_os.ai_engine.generate_briefing.assert_called_once()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search(client, mock_life_os):
    """Test POST /api/search performs semantic search."""
    mock_life_os.vector_store.search.return_value = [
        {"text": "Result 1", "score": 0.9}
    ]

    response = client.post("/api/search", json={"query": "test", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test"
    assert len(data["results"]) == 1
    assert data["count"] == 1
    mock_life_os.vector_store.search.assert_called_once_with(
        "test", limit=5, filter_metadata=None
    )


def test_search_with_filters(client, mock_life_os):
    """Test search with metadata filters."""
    response = client.post("/api/search", json={
        "query": "test",
        "limit": 10,
        "filters": {"source": "email"}
    })
    assert response.status_code == 200
    args = mock_life_os.vector_store.search.call_args
    assert args[1]["filter_metadata"] == {"source": "email"}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def test_list_tasks(client, mock_life_os):
    """Test GET /api/tasks lists pending tasks via get_tasks(status='pending')."""
    # The route now calls get_tasks() instead of the old get_pending_tasks() stub.
    mock_life_os.task_manager.get_tasks.return_value = [
        {"id": "t1", "title": "Task 1"}
    ]

    response = client.get("/api/tasks")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tasks"]) == 1
    assert data["count"] == 1
    mock_life_os.task_manager.get_tasks.assert_called_once_with(
        status="pending", limit=50
    )


def test_create_task(client, mock_life_os):
    """Test POST /api/tasks creates a new task."""
    response = client.post("/api/tasks", json={
        "title": "Buy groceries",
        "description": "Milk, eggs, bread",
        "priority": "high"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "task-123"
    mock_life_os.task_manager.create_task.assert_called_once()


def test_update_task(client, mock_life_os):
    """Test PATCH /api/tasks/{task_id} updates a task."""
    response = client.patch("/api/tasks/task-123", json={
        "status": "in_progress",
        "priority": "high"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    mock_life_os.task_manager.update_task.assert_called_once()


def test_complete_task(client, mock_life_os):
    """Test POST /api/tasks/{task_id}/complete marks task as done."""
    response = client.post("/api/tasks/task-123/complete")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    mock_life_os.task_manager.complete_task.assert_called_once_with("task-123")


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def test_list_notifications(client, mock_life_os):
    """Test GET /api/notifications lists pending notifications."""
    mock_life_os.notification_manager.get_pending.return_value = [
        {"id": "n1", "title": "Notification 1"}
    ]

    response = client.get("/api/notifications")
    assert response.status_code == 200
    data = response.json()
    assert len(data["notifications"]) == 1


def test_mark_notification_read(client, mock_life_os):
    """Test POST /api/notifications/{id}/read marks notification as read."""
    response = client.post("/api/notifications/n1/read")
    assert response.status_code == 200
    assert response.json()["status"] == "read"
    mock_life_os.notification_manager.mark_read.assert_called_once_with("n1")


def test_dismiss_notification(client, mock_life_os):
    """Test POST /api/notifications/{id}/dismiss dismisses notification."""
    response = client.post("/api/notifications/n1/dismiss")
    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"
    mock_life_os.notification_manager.dismiss.assert_called_once_with("n1")


def test_act_on_notification(client, mock_life_os):
    """Test POST /api/notifications/{id}/act marks notification as acted on."""
    response = client.post("/api/notifications/n1/act")
    assert response.status_code == 200
    assert response.json()["status"] == "acted_on"
    mock_life_os.notification_manager.mark_acted_on.assert_called_once_with("n1")


def test_get_digest(client, mock_life_os):
    """Test GET /api/notifications/digest returns notification summary."""
    response = client.get("/api/notifications/digest")
    assert response.status_code == 200
    data = response.json()
    assert data["digest"] == "Daily digest"


# ---------------------------------------------------------------------------
# Draft Messages
# ---------------------------------------------------------------------------

def test_draft_message(client, mock_life_os):
    """Test POST /api/draft generates message draft."""
    response = client.post("/api/draft", json={
        "contact_id": "c1",
        "channel": "email",
        "incoming_message": "Can we meet tomorrow?"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["draft"] == "Draft message"
    mock_life_os.ai_engine.draft_reply.assert_called_once()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def test_list_rules(client, mock_life_os):
    """Test GET /api/rules lists all automation rules."""
    mock_life_os.rules_engine.get_all_rules.return_value = [
        {"id": "r1", "name": "Rule 1"}
    ]

    response = client.get("/api/rules")
    assert response.status_code == 200
    data = response.json()
    assert len(data["rules"]) == 1


def test_create_rule(client, mock_life_os):
    """Test POST /api/rules creates a new automation rule."""
    response = client.post("/api/rules", json={
        "name": "Auto-tag work emails",
        "trigger_event": "email.received",
        "conditions": [{"field": "sender", "op": "contains", "value": "@work.com"}],
        "actions": [{"type": "tag", "value": "work"}]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["rule_id"] == "rule-123"
    mock_life_os.rules_engine.add_rule.assert_called_once()


def test_delete_rule(client, mock_life_os):
    """Test DELETE /api/rules/{rule_id} removes a rule."""
    response = client.delete("/api/rules/rule-123")
    assert response.status_code == 200
    assert response.json()["status"] == "deactivated"
    mock_life_os.rules_engine.remove_rule.assert_called_once_with("rule-123")


# ---------------------------------------------------------------------------
# User Model
# ---------------------------------------------------------------------------

def test_get_user_model(client, mock_life_os):
    """Test GET /api/user-model returns full user model summary."""
    response = client.get("/api/user-model")
    assert response.status_code == 200
    data = response.json()
    assert "facts" in data


def test_get_facts(client, mock_life_os):
    """Test GET /api/user-model/facts returns semantic facts."""
    mock_life_os.user_model_store.get_semantic_facts.return_value = [
        {"key": "favorite_color", "value": "blue", "confidence": 0.8}
    ]

    response = client.get("/api/user-model/facts")
    assert response.status_code == 200
    data = response.json()
    assert len(data["facts"]) == 1


def test_get_facts_with_min_confidence(client, mock_life_os):
    """Test GET /api/user-model/facts filters by confidence."""
    response = client.get("/api/user-model/facts?min_confidence=0.7")
    assert response.status_code == 200
    mock_life_os.user_model_store.get_semantic_facts.assert_called_with(min_confidence=0.7)


def test_delete_fact(client, mock_life_os):
    """Test DELETE /api/user-model/facts/{key} removes a fact."""
    mock_conn = Mock()
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.delete("/api/user-model/facts/favorite_color")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    mock_conn.execute.assert_called_once()


def test_get_mood(client, mock_life_os):
    """Test GET /api/user-model/mood returns current mood state."""
    # Use a simple object that doesn't have .dict() to test manual serialization
    class SimpleMood:
        energy_level = 0.7
        stress_level = 0.3
        social_battery = 0.8
        cognitive_load = 0.4
        emotional_valence = 0.6
        confidence = 0.75
        trend = "stable"

    mock_life_os.signal_extractor.get_current_mood.return_value = SimpleMood()

    response = client.get("/api/user-model/mood")
    assert response.status_code == 200
    data = response.json()
    assert "mood" in data
    assert data["mood"]["energy_level"] == 0.7
    assert data["mood"]["stress_level"] == 0.3


def test_get_mood_with_pydantic_model(client, mock_life_os):
    """Test GET /api/user-model/mood serialises via model_dump() (Pydantic V2)."""
    pydantic_mood = Mock()
    # The route now checks for model_dump (Pydantic V2) rather than dict (V1).
    pydantic_mood.model_dump.return_value = {"energy_level": 0.8, "stress_level": 0.2}
    mock_life_os.signal_extractor.get_current_mood.return_value = pydantic_mood

    response = client.get("/api/user-model/mood")
    assert response.status_code == 200
    data = response.json()
    assert data["mood"]["energy_level"] == 0.8


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def test_insights_summary(client, mock_life_os):
    """Test GET /api/insights/summary aggregates signal profiles."""
    response = client.get("/api/insights/summary")
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data
    assert "generated_at" in data
    assert isinstance(data["insights"], list)


def test_list_insights(client, mock_life_os):
    """Test GET /api/insights returns recent insights from InsightEngine."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"id": "i1", "type": "pattern", "summary": "Pattern detected", "evidence": '{"count": 5}'}
    ]
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert len(data["insights"]) == 1
    assert data["insights"][0]["evidence"]["count"] == 5


def test_insight_feedback(client, mock_life_os):
    """Test POST /api/insights/{id}/feedback records user feedback."""
    from unittest.mock import MagicMock
    mock_conn = MagicMock()
    # The route fetches the insight row and accesses row["category"].
    # Use a dict-like return value so subscript access works correctly.
    mock_conn.execute.return_value.fetchone.return_value = {"category": "contact_gap", "entity": "friend@example.com"}
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.post("/api/insights/i1/feedback?feedback=useful")
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


def test_insight_feedback_invalid(client, mock_life_os):
    """Test POST /api/insights/{id}/feedback rejects invalid feedback."""
    response = client.post("/api/insights/i1/feedback?feedback=invalid")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def test_get_preferences(client, mock_life_os):
    """Test GET /api/preferences returns all user preferences."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"key": "theme", "value": "dark", "set_by": "user", "updated_at": "2026-02-15T12:00:00Z"}
    ]
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/preferences")
    assert response.status_code == 200
    data = response.json()
    assert len(data["preferences"]) == 1


def test_update_preference(client, mock_life_os):
    """Test PUT /api/preferences updates a preference."""
    mock_conn = Mock()
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.put("/api/preferences", json={"key": "theme", "value": "dark"})
    assert response.status_code == 200
    assert response.json()["status"] == "updated"
    mock_conn.execute.assert_called_once()


def test_update_preference_publishes_event(client, mock_life_os):
    """Test PUT /api/preferences publishes telemetry event."""
    mock_conn = Mock()
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    client.put("/api/preferences", json={"key": "theme", "value": "dark"})
    mock_life_os.event_bus.publish.assert_called_once()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def test_submit_feedback(client, mock_life_os):
    """Test POST /api/feedback submits user feedback."""
    response = client.post("/api/feedback", json={"message": "Great feature!"})
    assert response.status_code == 200
    assert response.json()["status"] == "received"
    mock_life_os.feedback_collector.process_explicit_feedback.assert_called_once_with("Great feature!")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_list_events(client, mock_life_os):
    """Test GET /api/events lists events with optional filters."""
    mock_life_os.event_store.get_events.return_value = [
        {"id": "e1", "type": "email.received", "timestamp": "2026-02-15T12:00:00Z"}
    ]

    response = client.get("/api/events?event_type=email.received&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 1
    assert data["count"] == 1


# ---------------------------------------------------------------------------
# Connectors
# ---------------------------------------------------------------------------

def test_list_connectors(client, mock_life_os):
    """Test GET /api/connectors lists all connectors with health status."""
    mock_connector = Mock()
    mock_connector.CONNECTOR_ID = "test"
    mock_connector.DISPLAY_NAME = "Test Connector"
    mock_connector.health_check = AsyncMock(return_value={"status": "ok"})
    mock_life_os.connectors = [mock_connector]

    response = client.get("/api/connectors")
    assert response.status_code == 200
    data = response.json()
    assert len(data["connectors"]) == 1
    assert data["connectors"][0]["id"] == "test"


def test_browser_status(client, mock_life_os):
    """Test GET /api/browser/status returns browser orchestrator status."""
    response = client.get("/api/browser/status")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is False


def test_browser_vault_sites(client, mock_life_os):
    """Test GET /api/browser/vault returns vault sites."""
    mock_life_os.browser_orchestrator.get_vault_sites.return_value = ["reddit.com"]

    response = client.get("/api/browser/vault")
    assert response.status_code == 200
    data = response.json()
    assert "sites" in data


# ---------------------------------------------------------------------------
# Context API (iOS)
# ---------------------------------------------------------------------------

def test_submit_context_event(client, mock_life_os):
    """Test POST /api/context/event ingests a single context event."""
    response = client.post("/api/context/event", json={
        "type": "context.location",
        "source": "ios_app",
        "payload": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "place_name": "San Francisco"
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["event_id"] == "evt-123"
    mock_life_os.event_store.store_event.assert_called_once()


def test_submit_context_batch(client, mock_life_os):
    """Test POST /api/context/batch ingests multiple context events."""
    response = client.post("/api/context/batch", json={
        "events": [
            {
                "type": "context.location",
                "source": "ios_app",
                "payload": {"latitude": 37.7749, "longitude": -122.4194}
            },
            {
                "type": "context.device_nearby",
                "source": "ios_app",
                "payload": {"device_name": "iPhone", "signal_strength": -50}
            }
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["count"] == 2
    assert len(data["event_ids"]) == 2


def test_get_context_summary(client, mock_life_os):
    """Test GET /api/context/summary aggregates recent context data."""
    mock_life_os.event_store.get_events.return_value = [
        {
            "id": "e1",
            "payload": {"latitude": 37.7749, "longitude": -122.4194, "place_name": "SF"},
            "metadata": {"mobile_event_type": "context.location"},
            "timestamp": "2026-02-15T12:00:00Z"
        }
    ]

    response = client.get("/api/context/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "context_summary"
    assert len(data["locations"]) == 1


def test_get_context_places(client, mock_life_os):
    """Test GET /api/context/places returns learned places."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"id": "p1", "name": "Home", "visit_count": 100}
    ]
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/context/places")
    assert response.status_code == 200
    data = response.json()
    assert len(data["places"]) == 1


# ---------------------------------------------------------------------------
# Admin — Connector Management
# ---------------------------------------------------------------------------

def test_admin_connector_registry(client, mock_life_os):
    """Test GET /api/admin/connectors/registry returns connector schemas."""
    from dataclasses import dataclass

    @dataclass
    class MockConnectorDef:
        connector_id: str = "test"
        display_name: str = "Test"
        description: str = "Test connector"
        category: str = "email"
        module_path: str = "test.path"
        class_name: str = "TestConnector"
        config_schema: dict = None

    with patch("connectors.registry.CONNECTOR_REGISTRY", {"test": MockConnectorDef()}):
        response = client.get("/api/admin/connectors/registry")
        assert response.status_code == 200
        data = response.json()
        assert "registry" in data


def test_admin_list_connectors(client, mock_life_os):
    """Test GET /api/admin/connectors returns all connectors with config."""
    from dataclasses import dataclass

    @dataclass
    class MockConnectorDef:
        connector_id: str = "test"
        display_name: str = "Test"
        description: str = "Test connector"
        category: str = "email"
        module_path: str = "test.path"
        class_name: str = "TestConnector"
        config_schema: dict = None

    with patch("connectors.registry.CONNECTOR_REGISTRY", {"test": MockConnectorDef()}):
        response = client.get("/api/admin/connectors")
        assert response.status_code == 200
        data = response.json()
        assert "connectors" in data


def test_admin_save_config(client, mock_life_os):
    """Test PUT /api/admin/connectors/{id}/config saves configuration."""
    response = client.put("/api/admin/connectors/test/config", json={
        "config": {"api_key": "secret"}
    })
    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    mock_life_os.save_connector_config.assert_called_once()


def test_admin_test_connector(client, mock_life_os):
    """Test POST /api/admin/connectors/{id}/test validates credentials."""
    response = client.post("/api/admin/connectors/test/test", json={
        "config": {"api_key": "secret"}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_admin_enable_connector(client, mock_life_os):
    """Test POST /api/admin/connectors/{id}/enable starts connector."""
    response = client.post("/api/admin/connectors/test/enable")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"


def test_admin_disable_connector(client, mock_life_os):
    """Test POST /api/admin/connectors/{id}/disable stops connector."""
    response = client.post("/api/admin/connectors/test/disable")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "stopped"


# ---------------------------------------------------------------------------
# Admin — Database Viewer
# ---------------------------------------------------------------------------

def test_admin_db_schema(client, mock_life_os):
    """Test GET /api/admin/db returns database schema."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.return_value = [{"name": "events"}]
    mock_conn.execute.return_value.fetchone.return_value = {"c": 100}
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/admin/db")
    assert response.status_code == 200
    data = response.json()
    assert "databases" in data


def test_admin_db_query(client, mock_life_os):
    """Test GET /api/admin/db/{db}/{table} queries table data."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.side_effect = [
        [{"name": "events"}],  # Table list
        [{"name": "id"}, {"name": "type"}],  # Columns
        [{"id": "e1", "type": "email.received"}]  # Rows
    ]
    mock_conn.execute.return_value.fetchone.return_value = {"c": 1}
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/admin/db/events/events?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "rows" in data
    assert data["total"] == 1


# ---------------------------------------------------------------------------
# Setup / Onboarding
# ---------------------------------------------------------------------------

def test_setup_status(client, mock_life_os):
    """Test GET /api/setup/status checks onboarding completion."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/setup/status")
    assert response.status_code == 200
    data = response.json()
    assert data["completed"] is False
    assert "answers" in data


def test_setup_flow(client, mock_life_os):
    """Test GET /api/setup/flow returns onboarding phases."""
    with patch("services.onboarding.manager.ONBOARDING_PHASES", [
        {"id": "welcome", "title": "Welcome", "options": [{"label": "Next", "value": "next"}]}
    ]):
        response = client.get("/api/setup/flow")
        assert response.status_code == 200
        data = response.json()
        assert "phases" in data


def test_setup_submit(client, mock_life_os):
    """Test POST /api/setup/submit saves an onboarding answer."""
    response = client.post("/api/setup/submit", json={
        "step_id": "name",
        "value": "John Doe"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    mock_life_os.onboarding.submit_answer.assert_called_once()


def test_setup_finalize(client, mock_life_os):
    """Test POST /api/setup/finalize completes onboarding."""
    mock_life_os.onboarding.finalize.return_value = {"priority_contacts": [], "vaults": []}

    response = client.post("/api/setup/finalize")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Web UI Routes
# ---------------------------------------------------------------------------

def test_index_redirects_to_setup_when_not_onboarded(client, mock_life_os):
    """Test GET / redirects to /setup if onboarding incomplete."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/setup"


def test_index_shows_dashboard_when_onboarded(client, mock_life_os):
    """Test GET / shows dashboard if onboarding complete."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchone.return_value = {"value": "true"}
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_admin_page(client):
    """Test GET /admin returns admin HTML."""
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_admin_db_page(client):
    """Test GET /admin/db returns database viewer HTML."""
    response = client.get("/admin/db")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_setup_page(client):
    """Test GET /setup returns setup HTML."""
    response = client.get("/setup")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Calendar Events API  (/api/calendar/events)
# ---------------------------------------------------------------------------

def _make_cal_rows(*payloads):
    """Build mock DB row dicts for calendar.event.created events.

    Each payload is JSON-encoded into a ``payload`` key, matching the format
    returned by the ``events`` table.  The ``id`` field uses the event's own
    ``event_id`` value for convenience.
    """
    return [{"id": p.get("event_id", f"db-{i}"), "payload": json.dumps(p), "timestamp": "2026-02-28T10:00:00Z"}
            for i, p in enumerate(payloads)]


def _set_cal_rows(mock_life_os, *payloads):
    """Configure mock DB to return the given calendar event rows."""
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchall.return_value = _make_cal_rows(*payloads)
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn


def test_calendar_events_returns_events(client, mock_life_os):
    """Test GET /api/calendar/events returns events overlapping the date range."""
    _set_cal_rows(
        mock_life_os,
        {
            "event_id": "evt-1",
            "title": "Team Standup",
            "start_time": "2026-03-01T09:00:00+00:00",
            "end_time": "2026-03-01T09:30:00+00:00",
            "is_all_day": False,
            "location": "Zoom",
            "attendees": ["alice@example.com"],
            "description": "Daily standup",
            "calendar_id": "work",
        },
    )

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-02")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["events"]) == 1
    evt = data["events"][0]
    assert evt["id"] == "evt-1"
    assert evt["title"] == "Team Standup"
    assert evt["location"] == "Zoom"
    assert evt["attendees"] == ["alice@example.com"]
    assert evt["calendar_id"] == "work"


def test_calendar_events_deduplicates_by_event_id(client, mock_life_os):
    """Test /api/calendar/events keeps only the most-recent sync per event_id.

    The DB may have multiple rows for the same calendar event (from successive
    syncs).  The endpoint deduplicates by event_id, keeping the first row
    since rows are ordered timestamp DESC.
    """
    # Two rows with the same event_id but different titles (simulating re-sync)
    _set_cal_rows(
        mock_life_os,
        {
            "event_id": "evt-dup",
            "title": "Updated Title",  # Most recent (first row, DESC order)
            "start_time": "2026-03-01T10:00:00+00:00",
            "end_time": "2026-03-01T11:00:00+00:00",
            "is_all_day": False,
        },
        {
            "event_id": "evt-dup",
            "title": "Original Title",  # Older row — should be discarded
            "start_time": "2026-03-01T10:00:00+00:00",
            "end_time": "2026-03-01T11:00:00+00:00",
            "is_all_day": False,
        },
    )

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-02")
    assert response.status_code == 200
    data = response.json()
    # Only one event should be returned (deduplicated)
    assert data["count"] == 1
    assert data["events"][0]["title"] == "Updated Title"


def test_calendar_events_filters_outside_range(client, mock_life_os):
    """Test /api/calendar/events excludes events not overlapping [start, end)."""
    _set_cal_rows(
        mock_life_os,
        {
            "event_id": "in-range",
            "title": "In Range",
            "start_time": "2026-03-01T09:00:00+00:00",
            "end_time": "2026-03-01T10:00:00+00:00",
            "is_all_day": False,
        },
        {
            "event_id": "before-range",
            "title": "Before Range",
            "start_time": "2026-02-28T09:00:00+00:00",
            "end_time": "2026-02-28T10:00:00+00:00",
            "is_all_day": False,
        },
        {
            "event_id": "after-range",
            "title": "After Range",
            "start_time": "2026-03-05T09:00:00+00:00",
            "end_time": "2026-03-05T10:00:00+00:00",
            "is_all_day": False,
        },
    )

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-02")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["events"][0]["id"] == "in-range"


def test_calendar_events_empty(client, mock_life_os):
    """Test /api/calendar/events returns empty list when no events exist."""
    _set_cal_rows(mock_life_os)  # No rows

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-02")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["events"] == []


def test_calendar_events_sorted_by_start_time(client, mock_life_os):
    """Test /api/calendar/events returns events sorted by start_time ascending."""
    _set_cal_rows(
        mock_life_os,
        {
            "event_id": "evt-afternoon",
            "title": "Afternoon",
            "start_time": "2026-03-01T14:00:00+00:00",
            "end_time": "2026-03-01T15:00:00+00:00",
            "is_all_day": False,
        },
        {
            "event_id": "evt-morning",
            "title": "Morning",
            "start_time": "2026-03-01T09:00:00+00:00",
            "end_time": "2026-03-01T10:00:00+00:00",
            "is_all_day": False,
        },
    )

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-02")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    # Should be sorted morning → afternoon
    assert data["events"][0]["id"] == "evt-morning"
    assert data["events"][1]["id"] == "evt-afternoon"


def test_calendar_events_all_day(client, mock_life_os):
    """Test /api/calendar/events correctly returns all-day events."""
    _set_cal_rows(
        mock_life_os,
        {
            "event_id": "all-day-evt",
            "title": "Team Offsite",
            "start_time": "2026-03-01",
            "end_time": "2026-03-02",
            "is_all_day": True,
        },
    )

    response = client.get("/api/calendar/events?start=2026-03-01&end=2026-03-05")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["events"][0]["is_all_day"] is True


def test_calendar_events_missing_required_params(client, mock_life_os):
    """Test /api/calendar/events returns 422 when start or end is missing."""
    response = client.get("/api/calendar/events?start=2026-03-01")
    assert response.status_code == 422  # FastAPI validation error

    response = client.get("/api/calendar/events?end=2026-03-05")
    assert response.status_code == 422


def test_dashboard_feed_calendar_topic(client, mock_life_os):
    """Test dashboard feed topic=calendar returns upcoming calendar events."""
    # Set up the DB to return one future calendar event
    mock_conn = Mock()
    future_payload = json.dumps({
        "event_id": "cal-1",
        "title": "Sprint Review",
        "start_time": "2026-03-05T14:00:00+00:00",
        "end_time": "2026-03-05T15:00:00+00:00",
        "is_all_day": False,
        "description": "Review sprint results",
        "location": "Conference Room A",
        "attendees": [],
    })
    mock_conn.execute.return_value.fetchall.return_value = [
        {"id": "db-cal-1", "payload": future_payload, "timestamp": "2026-02-28T10:00:00Z"}
    ]
    mock_life_os.db.get_connection.return_value.__enter__.return_value = mock_conn

    response = client.get("/api/dashboard/feed?topic=calendar")
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "calendar"
    # Calendar events should appear in the feed
    calendar_items = [i for i in data["items"] if i["channel"] == "calendar"]
    assert len(calendar_items) == 1
    assert calendar_items[0]["title"] == "Sprint Review"
    assert calendar_items[0]["kind"] == "event"
