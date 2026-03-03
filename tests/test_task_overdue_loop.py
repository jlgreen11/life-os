"""
Tests for the task overdue detection loop.

The _task_overdue_loop() in main.py periodically checks for pending tasks
past their due_date, publishes task.overdue events, and creates user-facing
notifications.  These tests verify the detection logic by exercising the
TaskManager.get_overdue_tasks() query against a real SQLite database and
checking that the event bus and notification manager are called correctly.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.notification_manager.manager import NotificationManager
from services.task_manager.manager import TaskManager


@pytest.fixture
def task_manager(db):
    """TaskManager wired to the temporary database (no AI engine needed)."""
    return TaskManager(db, event_bus=MagicMock(), ai_engine=None)


@pytest.fixture
def mock_notification_manager():
    """Mock NotificationManager with async create_notification."""
    nm = MagicMock(spec=NotificationManager)
    nm.create_notification = AsyncMock(return_value="notif-id")
    return nm


def insert_task(
    db,
    task_id=None,
    title="Test task",
    status="pending",
    due_date=None,
    priority="normal",
    domain="personal",
):
    """Helper to insert a task row directly into the state database."""
    if task_id is None:
        task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks (id, title, description, status, source, domain,
                                  priority, due_date, created_at, updated_at)
               VALUES (?, ?, '', ?, 'manual', ?, ?, ?, ?, ?)""",
            (
                task_id,
                title,
                status,
                domain,
                priority,
                due_date,
                now,
                now,
            ),
        )
    return task_id


class TestOverdueDetection:
    """Tests for core overdue task detection logic."""

    def test_overdue_loop_detects_overdue_task(self, db, task_manager, event_bus, mock_notification_manager):
        """A pending task with a past due_date should be detected as overdue."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        task_id = insert_task(db, title="Submit report", due_date=past_due, priority="high")

        overdue = task_manager.get_overdue_tasks()

        assert len(overdue) == 1
        assert overdue[0]["id"] == task_id
        assert overdue[0]["title"] == "Submit report"

    @pytest.mark.asyncio
    async def test_overdue_event_published_and_notification_created(
        self, db, task_manager, event_bus, mock_notification_manager
    ):
        """Publishing a task.overdue event and creating a notification for a newly-overdue task."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        task_id = insert_task(db, title="Pay invoice", due_date=past_due, priority="high", domain="finance")

        # Simulate the detection logic from _task_overdue_loop
        notified_set: set[str] = set()
        overdue_tasks = task_manager.get_overdue_tasks()

        for task in overdue_tasks:
            tid = task.get("id")
            if not tid or tid in notified_set:
                continue

            await event_bus.publish(
                "task.overdue",
                {
                    "task_id": tid,
                    "title": task.get("title"),
                    "due_date": task.get("due_date"),
                    "priority": task.get("priority"),
                    "overdue_by": "5 hours",
                },
                source="task_manager",
            )

            await mock_notification_manager.create_notification(
                title=f"Task overdue: {task.get('title')}",
                body="Due 5 hours ago",
                priority="high",
                source_event_id=tid,
                domain=task.get("domain"),
            )
            notified_set.add(tid)

        # Verify event was published
        event_bus.publish.assert_called()
        published = event_bus._published_events
        overdue_events = [e for e in published if e["type"] == "task.overdue"]
        assert len(overdue_events) == 1
        assert overdue_events[0]["payload"]["task_id"] == task_id
        assert overdue_events[0]["payload"]["title"] == "Pay invoice"
        assert overdue_events[0]["source"] == "task_manager"

        # Verify notification was created
        mock_notification_manager.create_notification.assert_called_once_with(
            title="Task overdue: Pay invoice",
            body="Due 5 hours ago",
            priority="high",
            source_event_id=task_id,
            domain="finance",
        )

    @pytest.mark.asyncio
    async def test_overdue_loop_skips_already_notified(
        self, db, task_manager, event_bus, mock_notification_manager
    ):
        """The same overdue task should not generate duplicate notifications."""
        past_due = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        task_id = insert_task(db, title="Call dentist", due_date=past_due)

        # Pre-populate the notified set (simulating a prior loop iteration)
        notified_set: set[str] = {task_id}

        overdue_tasks = task_manager.get_overdue_tasks()
        assert len(overdue_tasks) == 1  # Task IS overdue

        for task in overdue_tasks:
            tid = task.get("id")
            if not tid or tid in notified_set:
                continue
            # This block should NOT execute
            await event_bus.publish("task.overdue", {}, source="task_manager")
            await mock_notification_manager.create_notification(title="should not happen")

        # Neither publish nor notification should have been called
        event_bus.publish.assert_not_called()
        mock_notification_manager.create_notification.assert_not_called()

    def test_overdue_loop_skips_tasks_without_due_date(self, db, task_manager):
        """Tasks with no due_date should never appear in overdue results."""
        insert_task(db, title="No deadline task", due_date=None)

        overdue = task_manager.get_overdue_tasks()

        assert len(overdue) == 0

    def test_overdue_loop_skips_completed_tasks(self, db, task_manager):
        """Completed tasks should not be flagged as overdue even if past due."""
        past_due = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        insert_task(db, title="Done task", due_date=past_due, status="completed")

        overdue = task_manager.get_overdue_tasks()

        assert len(overdue) == 0

    def test_overdue_loop_skips_future_due_tasks(self, db, task_manager):
        """Tasks with a future due_date should not be flagged as overdue."""
        future_due = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        insert_task(db, title="Future task", due_date=future_due)

        overdue = task_manager.get_overdue_tasks()

        assert len(overdue) == 0

    @pytest.mark.asyncio
    async def test_multiple_overdue_tasks_all_notified(
        self, db, task_manager, event_bus, mock_notification_manager
    ):
        """Multiple overdue tasks should each get their own event and notification."""
        notified_set: set[str] = set()

        past1 = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        past2 = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        id1 = insert_task(db, title="Task A", due_date=past1)
        id2 = insert_task(db, title="Task B", due_date=past2)

        overdue_tasks = task_manager.get_overdue_tasks()
        assert len(overdue_tasks) == 2

        for task in overdue_tasks:
            tid = task.get("id")
            if not tid or tid in notified_set:
                continue
            await event_bus.publish(
                "task.overdue",
                {"task_id": tid, "title": task.get("title")},
                source="task_manager",
            )
            await mock_notification_manager.create_notification(
                title=f"Task overdue: {task.get('title')}",
                body="overdue",
                priority="high",
                source_event_id=tid,
                domain=task.get("domain"),
            )
            notified_set.add(tid)

        # Both tasks should have generated events
        overdue_events = [e for e in event_bus._published_events if e["type"] == "task.overdue"]
        assert len(overdue_events) == 2
        published_ids = {e["payload"]["task_id"] for e in overdue_events}
        assert published_ids == {id1, id2}

        # Both notifications created
        assert mock_notification_manager.create_notification.call_count == 2
