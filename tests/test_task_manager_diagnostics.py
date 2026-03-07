"""Tests for TaskManager.get_diagnostics() observability method."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.task_manager.manager import TaskManager


@pytest.fixture
def task_manager(db):
    """A TaskManager wired to the temporary DatabaseManager with no AI engine."""
    return TaskManager(db, ai_engine=None, event_bus=None)


class TestGetDiagnosticsEmpty:
    """Tests against an empty task table."""

    def test_returns_expected_keys(self, task_manager):
        """get_diagnostics() returns all expected top-level keys."""
        result = task_manager.get_diagnostics()
        expected_keys = {
            "total_tasks", "by_status", "tasks_due_soon_24h",
            "tasks_due_soon_72h", "stale_pending_count",
            "recent_completions_24h", "top_domains",
            "ai_extraction_available", "health",
        }
        assert set(result.keys()) == expected_keys

    def test_empty_table_defaults(self, task_manager):
        """Empty task table should return zeroed-out diagnostics."""
        result = task_manager.get_diagnostics()
        assert result["total_tasks"] == 0
        assert result["by_status"] == {}
        assert result["tasks_due_soon_24h"] == 0
        assert result["tasks_due_soon_72h"] == 0
        assert result["stale_pending_count"] == 0
        assert result["recent_completions_24h"] == 0
        assert result["top_domains"] == {}
        assert result["health"] == "healthy"

    def test_ai_extraction_unavailable(self, task_manager):
        """ai_extraction_available should be False when no AI engine."""
        result = task_manager.get_diagnostics()
        assert result["ai_extraction_available"] is False

    def test_ai_extraction_available(self, db):
        """ai_extraction_available should be True when AI engine is set."""
        fake_ai = object()
        tm = TaskManager(db, ai_engine=fake_ai, event_bus=None)
        result = tm.get_diagnostics()
        assert result["ai_extraction_available"] is True


class TestGetDiagnosticsWithTasks:
    """Tests with tasks in the database."""

    @pytest.mark.asyncio
    async def test_total_tasks_count(self, task_manager):
        """Total tasks should reflect the number of created tasks."""
        await task_manager.create_task(title="Task 1")
        await task_manager.create_task(title="Task 2")
        await task_manager.create_task(title="Task 3")

        result = task_manager.get_diagnostics()
        assert result["total_tasks"] == 3

    @pytest.mark.asyncio
    async def test_by_status_reflects_mix(self, task_manager):
        """by_status should correctly count tasks per status."""
        task_id_1 = await task_manager.create_task(title="Pending task")
        task_id_2 = await task_manager.create_task(title="To complete")
        await task_manager.create_task(title="Another pending")

        await task_manager.complete_task(task_id_2)

        result = task_manager.get_diagnostics()
        assert result["by_status"]["pending"] == 2
        assert result["by_status"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_recent_completions_24h(self, task_manager):
        """recent_completions_24h should count tasks completed recently."""
        task_id = await task_manager.create_task(title="Done task")
        await task_manager.complete_task(task_id)

        result = task_manager.get_diagnostics()
        assert result["recent_completions_24h"] == 1

    @pytest.mark.asyncio
    async def test_top_domains(self, task_manager):
        """top_domains should return domain counts sorted by frequency."""
        await task_manager.create_task(title="Work 1", domain="work")
        await task_manager.create_task(title="Work 2", domain="work")
        await task_manager.create_task(title="Health 1", domain="health")
        await task_manager.create_task(title="No domain")

        result = task_manager.get_diagnostics()
        assert result["top_domains"]["work"] == 2
        assert result["top_domains"]["health"] == 1
        assert "unassigned" not in result["top_domains"]  # NULL domains excluded

    @pytest.mark.asyncio
    async def test_tasks_due_soon(self, task_manager):
        """tasks_due_soon counts should reflect tasks with upcoming due dates."""
        now = datetime.now(timezone.utc)
        due_12h = (now + timedelta(hours=12)).isoformat()
        due_48h = (now + timedelta(hours=48)).isoformat()
        due_96h = (now + timedelta(hours=96)).isoformat()

        await task_manager.create_task(title="Due in 12h", due_date=due_12h)
        await task_manager.create_task(title="Due in 48h", due_date=due_48h)
        await task_manager.create_task(title="Due in 96h", due_date=due_96h)

        result = task_manager.get_diagnostics()
        assert result["tasks_due_soon_24h"] == 1
        assert result["tasks_due_soon_72h"] == 2


class TestGetDiagnosticsStalePending:
    """Tests for stale pending task detection and health assessment."""

    def _insert_old_task(self, db, days_old: int, status: str = "pending"):
        """Helper to insert a task with an old created_at timestamp."""
        task_id = str(uuid.uuid4())
        old_date = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id, title, status, source, priority, tags,
                    related_contacts, related_events, created_at, updated_at)
                   VALUES (?, ?, ?, 'user', 'normal', '[]', '[]', '[]', ?, ?)""",
                (task_id, f"Old task {task_id[:8]}", status, old_date, old_date),
            )
        return task_id

    def test_stale_pending_count(self, task_manager, db):
        """stale_pending_count should identify tasks pending for > 7 days."""
        # Insert tasks with various ages
        self._insert_old_task(db, days_old=10)  # stale
        self._insert_old_task(db, days_old=14)  # stale
        self._insert_old_task(db, days_old=3)   # not stale
        self._insert_old_task(db, days_old=20, status="completed")  # not pending

        result = task_manager.get_diagnostics()
        assert result["stale_pending_count"] == 2

    def test_health_healthy_when_few_stale(self, task_manager, db):
        """health should be 'healthy' when stale count is below threshold."""
        for _ in range(5):
            self._insert_old_task(db, days_old=10)

        result = task_manager.get_diagnostics()
        assert result["health"] == "healthy"

    def test_health_degraded_when_many_stale(self, task_manager, db):
        """health should be 'degraded' when stale count >= 20."""
        for _ in range(20):
            self._insert_old_task(db, days_old=10)

        result = task_manager.get_diagnostics()
        assert result["stale_pending_count"] == 20
        assert result["health"] == "degraded"
