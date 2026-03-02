"""
Tests for dashboard_feed() structured warning logging and sections_loaded tracking.

Validates that:
1. All 5 sections are tracked in sections_loaded on success.
2. A failing section does not prevent other sections from loading.
3. Failed sections are excluded from sections_loaded.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture()
def mock_life_os():
    """Create a mock LifeOS instance with all services returning valid data."""
    life_os = Mock()

    # --- Database ---
    life_os.db = Mock()
    mock_conn = Mock()
    # Return empty result sets by default (no events in DB)
    mock_conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[])))
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

    # --- Event bus ---
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # --- Event store ---
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])

    # --- Vector store ---
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # --- Signal extractor ---
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable",
    ))

    # --- Notification manager (notifications section) ---
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])

    # --- Task manager (tasks section) ---
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])

    # --- Feedback collector ---
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})

    # --- AI engine ---
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="")
    life_os.ai_engine.search_life = AsyncMock(return_value="")

    # --- Rules engine ---
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])

    # --- User model store ---
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)

    # --- Connectors ---
    life_os.connectors = []

    # --- Browser orchestrator ---
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # --- Onboarding ---
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # --- Connector management ---
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = AsyncMock(return_value={"success": True})
    life_os.enable_connector = AsyncMock(return_value={"status": "started"})
    life_os.disable_connector = AsyncMock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture()
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test: All sections load successfully
# ---------------------------------------------------------------------------

def test_all_sections_loaded_on_success(client, mock_life_os):
    """When all sections succeed, sections_loaded contains all 5 names."""
    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    assert "sections_loaded" in data
    assert sorted(data["sections_loaded"]) == [
        "calendar", "email", "messages", "notifications", "tasks",
    ]


# ---------------------------------------------------------------------------
# Test: One section failing doesn't break others
# ---------------------------------------------------------------------------

def test_tasks_failure_still_returns_other_sections(client, mock_life_os):
    """When the tasks section raises, the endpoint still returns 200 and
    the other sections still load successfully."""
    # Make tasks section blow up
    mock_life_os.task_manager.get_pending_tasks.side_effect = RuntimeError("DB locked")

    # Add a notification so we can verify notifications still loads
    mock_life_os.notification_manager.get_pending.return_value = [
        {
            "id": "n1",
            "title": "Test",
            "body": "body",
            "priority": "normal",
            "created_at": "2026-02-15T12:00:00Z",
            "source": "email.received",
            "metadata": {},
        }
    ]

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    # Notifications still loaded
    assert len(data["items"]) >= 1
    assert any(item["kind"] == "notification" for item in data["items"])

    # Other sections loaded despite tasks failure
    assert "notifications" in data["sections_loaded"]
    assert "calendar" in data["sections_loaded"]
    assert "email" in data["sections_loaded"]
    assert "messages" in data["sections_loaded"]


# ---------------------------------------------------------------------------
# Test: Failed section is NOT in sections_loaded
# ---------------------------------------------------------------------------

def test_failed_section_excluded_from_sections_loaded(client, mock_life_os):
    """A section that raises an exception is not listed in sections_loaded."""
    mock_life_os.task_manager.get_pending_tasks.side_effect = RuntimeError("DB locked")

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    assert "tasks" not in data["sections_loaded"]
    # The other 4 sections should still be present
    assert "notifications" in data["sections_loaded"]
    assert "calendar" in data["sections_loaded"]
    assert "email" in data["sections_loaded"]
    assert "messages" in data["sections_loaded"]


def test_notifications_failure_excluded_from_sections_loaded(client, mock_life_os):
    """Notifications failure is excluded; other sections still load."""
    mock_life_os.notification_manager.get_pending.side_effect = RuntimeError("connection reset")

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    assert "notifications" not in data["sections_loaded"]
    assert "tasks" in data["sections_loaded"]
    assert "calendar" in data["sections_loaded"]
    assert "email" in data["sections_loaded"]
    assert "messages" in data["sections_loaded"]


def test_calendar_db_failure_excluded_from_sections_loaded(client, mock_life_os):
    """When the DB raises inside the calendar section, calendar is excluded."""
    # The calendar section calls life_os.db.get_connection("events") as a
    # context manager.  Make the __enter__ raise for the "events" DB.
    original_get_connection = mock_life_os.db.get_connection

    def failing_get_connection(db_name):
        """Raise only for the 'events' DB to simulate calendar/email/message failure."""
        if db_name == "events":
            raise RuntimeError("events DB unavailable")
        return original_get_connection(db_name)

    mock_life_os.db.get_connection = Mock(side_effect=failing_get_connection)

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    # Calendar, email, and messages all use the events DB, so all fail
    assert "calendar" not in data["sections_loaded"]
    assert "email" not in data["sections_loaded"]
    assert "messages" not in data["sections_loaded"]

    # Notifications and tasks use different paths (not the events DB)
    assert "notifications" in data["sections_loaded"]
    assert "tasks" in data["sections_loaded"]


def test_warning_logged_on_section_failure(client, mock_life_os, caplog):
    """Verify that a warning is logged when a section fails."""
    import logging

    mock_life_os.task_manager.get_pending_tasks.side_effect = RuntimeError("test error")

    with caplog.at_level(logging.WARNING, logger="web.routes"):
        response = client.get("/api/dashboard/feed")

    assert response.status_code == 200

    # Check that the warning message was logged for the tasks section
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tasks" in msg and "test error" in msg for msg in warning_messages), (
        f"Expected a warning about 'tasks' section failure, got: {warning_messages}"
    )
