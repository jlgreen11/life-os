"""
Life OS — Tests for create_task rule action notification & WebSocket broadcast.

When the rules engine fires a create_task action, the pipeline should:
1. Create the task in state.db
2. Create a user-facing notification so the user knows about the auto-created task
3. Broadcast the notification via WebSocket for real-time dashboard updates

These tests verify all three behaviours using real DatabaseManager instances
(via the conftest.py fixtures) and a minimal LifeOS shell.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.notification_manager.manager import NotificationManager
from services.task_manager.manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_life_os(db, event_bus):
    """Build a minimal LifeOS shell with real TaskManager and NotificationManager.

    Uses object.__new__ to skip LifeOS.__init__ (which requires NATS, Ollama,
    etc.) and wires only the attributes that _execute_rule_action touches.
    """
    from main import LifeOS

    lo = object.__new__(LifeOS)
    lo.db = db
    lo.task_manager = TaskManager(db, ai_engine=None, event_bus=event_bus)
    lo.notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Wire bound methods that the action handler calls
    lo._infer_domain_from_event_type = LifeOS._infer_domain_from_event_type.__get__(lo, LifeOS)

    return lo


def _dummy_event(event_type="email.received", source="google", event_id="evt-123"):
    """Create a minimal event dict for testing."""
    return {
        "id": event_id,
        "type": event_type,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"subject": "Invoice #42"},
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_task_action_creates_notification(db, event_bus):
    """create_task action should insert a notification into state.db."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task", "title": "Follow up on invoice", "priority": "high"}
    event = _dummy_event()

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock()
        await lo._execute_rule_action(action, event)

    # Verify task was created
    with db.get_connection("state") as conn:
        tasks = conn.execute("SELECT * FROM tasks WHERE source = 'rule'").fetchall()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Follow up on invoice"
    assert tasks[0]["priority"] == "high"

    # Verify notification was created
    with db.get_connection("state") as conn:
        notifs = conn.execute("SELECT * FROM notifications").fetchall()
    assert len(notifs) == 1
    assert "Task created" in notifs[0]["title"]
    assert "Follow up on invoice" in notifs[0]["title"]
    assert notifs[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_create_task_action_notification_contains_source(db, event_bus):
    """Notification body should mention the event source for traceability."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task", "title": "Review document", "priority": "normal"}
    event = _dummy_event(source="proton_mail")

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock()
        await lo._execute_rule_action(action, event)

    with db.get_connection("state") as conn:
        notif = conn.execute("SELECT * FROM notifications").fetchone()
    assert notif is not None
    assert "proton_mail" in notif["body"]


@pytest.mark.asyncio
async def test_create_task_action_broadcasts_websocket(db, event_bus):
    """create_task action should broadcast a WebSocket notification."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task", "title": "Call vendor", "priority": "normal"}
    event = _dummy_event(event_id="evt-456")

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock()
        await lo._execute_rule_action(action, event)

        mock_ws.broadcast.assert_called_once()
        broadcast_data = mock_ws.broadcast.call_args[0][0]
        assert broadcast_data["type"] == "notification"
        assert "Call vendor" in broadcast_data["title"]
        assert broadcast_data["source_event_id"] == "evt-456"


@pytest.mark.asyncio
async def test_create_task_action_default_title_and_priority(db, event_bus):
    """create_task action should use defaults when title/priority are absent."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task"}
    event = _dummy_event()

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock()
        await lo._execute_rule_action(action, event)

    with db.get_connection("state") as conn:
        task = conn.execute("SELECT * FROM tasks WHERE source = 'rule'").fetchone()
        notif = conn.execute("SELECT * FROM notifications").fetchone()

    assert task["title"] == "Auto-created task"
    assert task["priority"] == "normal"
    assert notif["title"] == "Task created: Auto-created task"
    assert notif["priority"] == "normal"


@pytest.mark.asyncio
async def test_create_task_action_infers_domain(db, event_bus):
    """Notification domain should be inferred from the event type."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task", "title": "Schedule meeting"}
    event = _dummy_event(event_type="calendar.event.created")

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock()
        await lo._execute_rule_action(action, event)

    with db.get_connection("state") as conn:
        notif = conn.execute("SELECT * FROM notifications").fetchone()
    assert notif["domain"] == "calendar"


@pytest.mark.asyncio
async def test_create_task_ws_broadcast_failure_does_not_block(db, event_bus):
    """WebSocket broadcast failure should not prevent task/notification creation."""
    lo = _make_life_os(db, event_bus)

    action = {"type": "create_task", "title": "Important task"}
    event = _dummy_event()

    with patch("main.ws_manager") as mock_ws:
        mock_ws.broadcast = AsyncMock(side_effect=RuntimeError("ws down"))
        await lo._execute_rule_action(action, event)

    # Task and notification should still exist despite WS failure
    with db.get_connection("state") as conn:
        tasks = conn.execute("SELECT * FROM tasks WHERE source = 'rule'").fetchall()
        notifs = conn.execute("SELECT * FROM notifications").fetchall()
    assert len(tasks) == 1
    assert len(notifs) == 1
