"""
Tests for POST /api/command error handling when AI engine is unavailable.

Verifies that AI-dependent command branches (briefing, draft, fallback)
return graceful error responses (200 with type='error') instead of 500s
when the AI engine raises exceptions.
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

    # Mock vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 50, "dimensions": 384})

    # Mock signal extractor
    life_os.signal_extractor = Mock()

    # Mock notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 5, "delivered": 100})

    # Mock task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_stats = Mock(return_value={"total": 0})

    # Mock feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 10})

    # Mock user model store
    life_os.user_model_store = Mock()

    # Mock prediction engine
    life_os.prediction_engine = Mock()
    life_os.prediction_engine.get_stats = Mock(return_value={"total": 0})

    # Mock insight engine
    life_os.insight_engine = Mock()

    # Mock AI engine — defaults to working; tests override with side_effect
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="Morning briefing")
    life_os.ai_engine.draft_reply = AsyncMock(return_value="Draft message")
    life_os.ai_engine.search_life = AsyncMock(return_value="Search result")

    # Mock connectors
    life_os.connectors = {}

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Briefing command — AI failure
# ---------------------------------------------------------------------------


def test_command_briefing_returns_error_on_ai_failure(client, mock_life_os):
    """POST /api/command with 'briefing' returns type='error' when AI engine fails."""
    mock_life_os.ai_engine.generate_briefing = AsyncMock(
        side_effect=ConnectionError("Ollama connection refused")
    )
    response = client.post("/api/command", json={"text": "briefing"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "temporarily unavailable" in data["content"]
    assert "AI engine" in data["content"]


def test_command_briefing_returns_error_on_runtime_error(client, mock_life_os):
    """POST /api/command with 'briefing' handles RuntimeError gracefully."""
    mock_life_os.ai_engine.generate_briefing = AsyncMock(
        side_effect=RuntimeError("model not found")
    )
    response = client.post("/api/command", json={"text": "briefing"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "unavailable" in data["content"]


# ---------------------------------------------------------------------------
# Draft command — AI failure
# ---------------------------------------------------------------------------


def test_command_draft_returns_error_on_ai_failure(client, mock_life_os):
    """POST /api/command with 'draft' returns type='error' when AI engine fails."""
    mock_life_os.ai_engine.draft_reply = AsyncMock(
        side_effect=ConnectionError("Ollama connection refused")
    )
    response = client.post("/api/command", json={"text": "draft reply to John about the meeting"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "temporarily unavailable" in data["content"]
    assert "AI engine" in data["content"]


def test_command_draft_returns_error_on_timeout(client, mock_life_os):
    """POST /api/command with 'draft' handles TimeoutError gracefully."""
    mock_life_os.ai_engine.draft_reply = AsyncMock(
        side_effect=TimeoutError("request timed out")
    )
    response = client.post("/api/command", json={"text": "draft reply to the email"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "unavailable" in data["content"]


# ---------------------------------------------------------------------------
# Fallback AI command — AI failure
# ---------------------------------------------------------------------------


def test_command_fallback_returns_error_on_ai_failure(client, mock_life_os):
    """POST /api/command fallback returns type='error' when AI engine fails."""
    mock_life_os.ai_engine.search_life = AsyncMock(
        side_effect=ConnectionError("Ollama connection refused")
    )
    response = client.post("/api/command", json={"text": "what happened last week?"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "temporarily unavailable" in data["content"]
    assert "AI engine" in data["content"]


def test_command_fallback_returns_error_on_value_error(client, mock_life_os):
    """POST /api/command fallback handles ValueError gracefully."""
    mock_life_os.ai_engine.search_life = AsyncMock(
        side_effect=ValueError("invalid model response")
    )
    response = client.post("/api/command", json={"text": "summarize my day"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "error"
    assert "unavailable" in data["content"]


# ---------------------------------------------------------------------------
# Verify happy path still works
# ---------------------------------------------------------------------------


def test_command_briefing_still_works_when_ai_available(client, mock_life_os):
    """POST /api/command with 'briefing' returns normal response when AI works."""
    response = client.post("/api/command", json={"text": "briefing"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "briefing"
    assert data["content"] == "Morning briefing"


def test_command_draft_still_works_when_ai_available(client, mock_life_os):
    """POST /api/command with 'draft' returns normal response when AI works."""
    response = client.post("/api/command", json={"text": "draft reply to the email"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "draft"
    assert data["content"] == "Draft message"


def test_command_fallback_still_works_when_ai_available(client, mock_life_os):
    """POST /api/command fallback returns normal response when AI works."""
    response = client.post("/api/command", json={"text": "what's on my calendar?"})

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "ai_response"
    assert data["content"] == "Search result"
