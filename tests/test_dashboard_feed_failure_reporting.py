"""
Tests for sections_failed tracking in the /api/dashboard/feed endpoint.

The dashboard feed loads data from multiple sections (notifications, tasks,
calendar, email, messages).  When a section fails, the error should be
tracked in `sections_failed` in the response so the frontend can display
diagnostic messages and help troubleshoot missing data.

Coverage:
- Response always includes sections_failed key (even when empty)
- A failing section appears in sections_failed with error details
- Other sections still load successfully when one fails
- Action-item enrichment failures are tracked (previously swallowed by bare pass)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.app import create_web_app
from web.routes import register_routes


# ---------------------------------------------------------------------------
# Fixtures — Mock-based (for isolated section failure tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for dashboard feed tests."""
    life_os = Mock()

    # Database mock returning empty results by default
    mock_conn = Mock()
    mock_conn.execute.return_value = Mock(fetchall=Mock(return_value=[]))
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

    # Signal extractor
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


def test_dashboard_feed_includes_sections_failed_key(client):
    """Response always includes 'sections_failed', even when all sections succeed."""
    resp = client.get("/api/dashboard/feed")
    assert resp.status_code == 200
    data = resp.json()
    assert "sections_failed" in data
    assert isinstance(data["sections_failed"], list)


def test_dashboard_feed_sections_failed_empty_on_success(client):
    """When all sections load successfully, sections_failed is an empty list."""
    resp = client.get("/api/dashboard/feed")
    data = resp.json()
    assert data["sections_failed"] == []


# ---------------------------------------------------------------------------
# Section failure tracking tests
# ---------------------------------------------------------------------------


def test_dashboard_feed_reports_failed_tasks_section(mock_life_os, client):
    """When the task manager raises, 'tasks' appears in sections_failed."""
    mock_life_os.task_manager.get_pending_tasks = Mock(
        side_effect=RuntimeError("Tasks DB corrupted")
    )
    resp = client.get("/api/dashboard/feed?topic=tasks")
    assert resp.status_code == 200
    data = resp.json()

    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "tasks" in failed_sections

    # Verify error message is included
    tasks_failure = next(s for s in data["sections_failed"] if s["section"] == "tasks")
    assert "Tasks DB corrupted" in tasks_failure["error"]


def test_dashboard_feed_reports_failed_notifications_section(mock_life_os, client):
    """When notification_manager.get_pending raises, 'notifications' appears in sections_failed."""
    mock_life_os.notification_manager.get_pending = Mock(
        side_effect=RuntimeError("Notification DB unavailable")
    )
    resp = client.get("/api/dashboard/feed?topic=inbox")
    assert resp.status_code == 200
    data = resp.json()

    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "notifications" in failed_sections


def test_dashboard_feed_reports_failed_calendar_section(mock_life_os, client):
    """When the calendar DB query raises, 'calendar' appears in sections_failed."""
    mock_life_os.db.get_connection = Mock(
        side_effect=RuntimeError("Calendar DB error")
    )
    resp = client.get("/api/dashboard/feed?topic=calendar")
    assert resp.status_code == 200
    data = resp.json()

    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "calendar" in failed_sections


def test_dashboard_feed_still_returns_other_sections_on_failure(mock_life_os, client):
    """When one section fails, other sections still load and return items."""
    # Make tasks fail
    mock_life_os.task_manager.get_pending_tasks = Mock(
        side_effect=RuntimeError("Tasks DB error")
    )
    # Make notifications succeed with real data
    mock_life_os.notification_manager.get_pending = Mock(
        return_value=[
            {"id": "n1", "domain": "email", "title": "Test Email", "body": "Hello",
             "priority": "normal", "created_at": "2026-01-01T00:00:00Z"},
        ]
    )

    resp = client.get("/api/dashboard/feed?topic=inbox")
    assert resp.status_code == 200
    data = resp.json()

    # Tasks should be in sections_failed
    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "tasks" in failed_sections

    # Notifications should still be in sections_loaded
    assert "notifications" in data["sections_loaded"]

    # Items from notifications should still be present
    assert data["count"] > 0


def test_dashboard_feed_multiple_sections_can_fail(mock_life_os, client):
    """Multiple sections can fail independently and all appear in sections_failed."""
    mock_life_os.notification_manager.get_pending = Mock(
        side_effect=RuntimeError("Notification error")
    )
    mock_life_os.task_manager.get_pending_tasks = Mock(
        side_effect=RuntimeError("Task error")
    )

    resp = client.get("/api/dashboard/feed?topic=inbox")
    assert resp.status_code == 200
    data = resp.json()

    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "notifications" in failed_sections
    assert "tasks" in failed_sections


# ---------------------------------------------------------------------------
# Action-item enrichment failure tracking (previously bare pass)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_feed_action_item_failure_tracked(db):
    """When action-item enrichment fails, 'action_items' appears in sections_failed."""
    # Insert a real email event so that the enrichment block is triggered
    event_id = str(uuid.uuid4())
    payload = json.dumps({
        "from_address": "sender@example.com",
        "from_name": "Sender",
        "subject": "Test Email",
        "body": "Test body",
        "snippet": "Test snippet",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, 'email.received', 'google', datetime('now'), 'normal', ?, '{}')""",
            (event_id, payload),
        )

    app = FastAPI()
    life_os = MagicMock()
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.user_model_store = MagicMock()

    # Use real DB for events (so email events are found) but make the
    # state DB connection raise an error for the action-item enrichment query.
    real_get_connection = db.get_connection

    def patched_get_connection(db_name):
        """Return real connections except for 'state' which raises on execute."""
        if db_name == "state":
            mock_conn = MagicMock()
            mock_conn.execute.side_effect = RuntimeError("State DB unavailable")
            # Return a context manager that yields the mock
            return MagicMock(
                __enter__=MagicMock(return_value=mock_conn),
                __exit__=MagicMock(return_value=False),
            )
        return real_get_connection(db_name)

    life_os.db = MagicMock()
    life_os.db.get_connection = patched_get_connection

    register_routes(app, life_os)
    test_client = TestClient(app)

    resp = test_client.get("/api/dashboard/feed?topic=email&limit=10")
    assert resp.status_code == 200
    data = resp.json()

    failed_sections = [s["section"] for s in data["sections_failed"]]
    assert "action_items" in failed_sections

    # Verify the email section still loaded successfully
    assert "email" in data["sections_loaded"]
    # Email items should still be present despite enrichment failure
    email_items = [it for it in data["items"] if it.get("kind") == "email"]
    assert len(email_items) > 0
