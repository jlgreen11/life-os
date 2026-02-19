"""
Tests for TaskManager.get_tasks() and the /api/tasks status-filter fix.

Before this fix, GET /api/tasks accepted a ``status`` query parameter but
silently ignored it — always returning pending tasks regardless of what the
caller requested.  The root cause was that the route called
``get_pending_tasks()`` (which hard-coded ``status='pending'``) instead of
a flexible ``get_tasks(status=...)`` method.

These tests verify:
  1. ``TaskManager.get_tasks()`` correctly filters by every valid status.
  2. ``get_tasks()`` passes domain, priority, and limit filters through.
  3. Unknown status values are normalised to "pending" with a log warning.
  4. ``get_pending_tasks()`` still works (backward-compat shim).
  5. The /api/tasks route now delegates to ``get_tasks()`` and returns the
     correct rows for completed, in_progress, and pending status values.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from services.task_manager.manager import TaskManager
from web.app import create_web_app


# ---------------------------------------------------------------------------
# Route-level fixtures (minimal copies of test_web_routes.py fixtures).
# These are defined here because pytest fixtures are module-scoped and those
# in test_web_routes.py are not available to tests in this file.
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_life_os():
    """Minimal mock LifeOS instance sufficient for the task route tests."""
    life_os = Mock()

    # DB mock with context-manager support (required by web/app.py startup)
    mock_conn = Mock()
    mock_conn.execute = Mock()
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(),
        )
    )

    # Required services that routes.py references
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-123")

    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable",
    ))

    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.mark_read = AsyncMock()
    life_os.notification_manager.dismiss = AsyncMock()
    life_os.notification_manager.mark_acted_on = AsyncMock()
    life_os.notification_manager.get_digest = AsyncMock(return_value="")

    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = AsyncMock()

    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="briefing")
    life_os.ai_engine.draft_reply = AsyncMock(return_value="draft")
    life_os.ai_engine.search_life = AsyncMock(return_value="results")

    # Task manager — get_tasks() is the method the route now calls.
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = AsyncMock(return_value="task-123")
    life_os.task_manager.update_task = AsyncMock()
    life_os.task_manager.complete_task = AsyncMock()

    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = AsyncMock(return_value="rule-123")
    life_os.rules_engine.remove_rule = AsyncMock()

    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = AsyncMock(return_value={"success": True})
    life_os.enable_connector = AsyncMock(return_value={"status": "started"})
    life_os.disable_connector = AsyncMock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture
def app(mock_life_os):
    """FastAPI test app wired to the mock LifeOS instance."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Synchronous test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# TaskManager unit-test helpers
# ---------------------------------------------------------------------------

def _make_task(db, *, status: str = "pending", domain: str = "work",
               priority: str = "normal", title: str | None = None,
               completed_at: str | None = None) -> str:
    """Insert a task row into the temp DB and return its ID.

    Bypasses the async TaskManager.create_task() coroutine so we can seed
    the database synchronously during test setup.
    """
    task_id = str(uuid.uuid4())
    if title is None:
        title = f"Task {task_id[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, priority, domain, created_at, updated_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, title, status, priority, domain, now, now, completed_at),
        )
    return task_id


@pytest.fixture()
def manager(db, event_bus):
    """TaskManager wired to the real temp DB and a mock event bus."""
    return TaskManager(db=db, event_bus=event_bus)


# ---------------------------------------------------------------------------
# TaskManager.get_tasks() — default behaviour
# ---------------------------------------------------------------------------

class TestGetTasksDefaultsToPending:
    """get_tasks() with no arguments returns only pending tasks."""

    def test_returns_pending_when_no_status_given(self, manager, db):
        """Default call (no status arg) returns only pending tasks."""
        pending_id = _make_task(db, status="pending")
        _make_task(db, status="completed")
        _make_task(db, status="in_progress")

        results = manager.get_tasks()

        ids = [r["id"] for r in results]
        assert pending_id in ids
        assert len(results) == 1

    def test_returns_empty_when_no_pending(self, manager, db):
        """Returns an empty list when there are no pending tasks."""
        _make_task(db, status="completed")
        assert manager.get_tasks() == []


# ---------------------------------------------------------------------------
# TaskManager.get_tasks() — status filtering
# ---------------------------------------------------------------------------

class TestGetTasksByStatus:
    """get_tasks(status=X) returns only tasks with that exact status."""

    def test_completed_status(self, manager, db):
        """status='completed' returns only completed tasks."""
        done_id = _make_task(db, status="completed", title="Done task")
        _make_task(db, status="pending", title="Still pending")

        results = manager.get_tasks(status="completed")

        ids = [r["id"] for r in results]
        assert done_id in ids
        assert len(results) == 1

    def test_in_progress_status(self, manager, db):
        """status='in_progress' returns only in-progress tasks."""
        wip_id = _make_task(db, status="in_progress", title="WIP task")
        _make_task(db, status="pending", title="Not started yet")

        results = manager.get_tasks(status="in_progress")

        ids = [r["id"] for r in results]
        assert wip_id in ids
        assert len(results) == 1

    def test_archived_status(self, manager, db):
        """status='archived' returns only archived tasks."""
        arch_id = _make_task(db, status="archived", title="Old task")
        _make_task(db, status="pending")

        results = manager.get_tasks(status="archived")

        ids = [r["id"] for r in results]
        assert arch_id in ids
        assert len(results) == 1

    def test_cancelled_status(self, manager, db):
        """status='cancelled' returns only cancelled tasks."""
        canc_id = _make_task(db, status="cancelled", title="Dropped task")
        _make_task(db, status="pending")

        results = manager.get_tasks(status="cancelled")

        ids = [r["id"] for r in results]
        assert canc_id in ids
        assert len(results) == 1

    def test_each_status_returns_correct_subset(self, manager, db):
        """Multiple statuses in DB — only the requested ones come back."""
        ids = {
            "pending":     _make_task(db, status="pending"),
            "completed":   _make_task(db, status="completed"),
            "in_progress": _make_task(db, status="in_progress"),
            "archived":    _make_task(db, status="archived"),
        }

        for status, expected_id in ids.items():
            results = manager.get_tasks(status=status)
            returned_ids = [r["id"] for r in results]
            assert expected_id in returned_ids, f"Expected {status} task in results"
            # No other statuses should leak through
            for other_status, other_id in ids.items():
                if other_status != status:
                    assert other_id not in returned_ids, (
                        f"{other_status} task leaked into {status} query results"
                    )


# ---------------------------------------------------------------------------
# TaskManager.get_tasks() — unknown status normalisation
# ---------------------------------------------------------------------------

class TestGetTasksUnknownStatus:
    """Unknown status values are safely normalised to 'pending'."""

    def test_unknown_status_falls_back_to_pending(self, manager, db):
        """An unrecognised status string returns pending tasks (safe default)."""
        pending_id = _make_task(db, status="pending")
        _make_task(db, status="completed")

        # "totally_invalid_status" is not in VALID_STATUSES; should return pending
        results = manager.get_tasks(status="totally_invalid_status")

        ids = [r["id"] for r in results]
        assert pending_id in ids

    def test_unknown_status_logs_warning(self, manager, db):
        """An unrecognised status emits logger.warning (not a silent swallow)."""
        _make_task(db, status="pending")

        with patch("services.task_manager.manager.logger") as mock_logger:
            manager.get_tasks(status="bad_value")

        mock_logger.warning.assert_called_once()
        # Confirm the warning references the bad value so it's debuggable
        warning_call = mock_logger.warning.call_args
        assert "bad_value" in str(warning_call)


# ---------------------------------------------------------------------------
# TaskManager.get_tasks() — optional filters
# ---------------------------------------------------------------------------

class TestGetTasksFilters:
    """domain, priority, and limit filters narrow the result set correctly."""

    def test_domain_filter(self, manager, db):
        """Only tasks matching the requested domain are returned."""
        work_id   = _make_task(db, status="pending", domain="work")
        health_id = _make_task(db, status="pending", domain="health")

        results = manager.get_tasks(status="pending", domain="work")
        ids = [r["id"] for r in results]

        assert work_id in ids
        assert health_id not in ids

    def test_priority_filter(self, manager, db):
        """Only tasks matching the requested priority are returned."""
        high_id   = _make_task(db, status="pending", priority="high")
        normal_id = _make_task(db, status="pending", priority="normal")

        results = manager.get_tasks(status="pending", priority="high")
        ids = [r["id"] for r in results]

        assert high_id in ids
        assert normal_id not in ids

    def test_domain_and_priority_combined(self, manager, db):
        """Domain + priority filters are ANDed together."""
        match_id   = _make_task(db, status="pending", domain="work",   priority="high")
        wrong_prio = _make_task(db, status="pending", domain="work",   priority="normal")
        wrong_dom  = _make_task(db, status="pending", domain="health", priority="high")

        results = manager.get_tasks(status="pending", domain="work", priority="high")
        ids = [r["id"] for r in results]

        assert match_id in ids
        assert wrong_prio not in ids
        assert wrong_dom not in ids

    def test_limit_caps_result_count(self, manager, db):
        """limit parameter caps the number of returned rows."""
        for _ in range(5):
            _make_task(db, status="pending")

        results = manager.get_tasks(status="pending", limit=2)
        assert len(results) == 2

    def test_completed_domain_filter(self, manager, db):
        """Domain filter works correctly for non-pending statuses too."""
        work_done   = _make_task(db, status="completed", domain="work")
        health_done = _make_task(db, status="completed", domain="health")

        results = manager.get_tasks(status="completed", domain="work")
        ids = [r["id"] for r in results]

        assert work_done in ids
        assert health_done not in ids


# ---------------------------------------------------------------------------
# TaskManager.get_tasks() — sort ordering
# ---------------------------------------------------------------------------

class TestGetTasksSorting:
    """Result ordering is correct for each status type."""

    def test_pending_sorted_by_priority_then_due_date(self, manager, db):
        """Pending tasks: critical > high > normal > low."""
        low_id    = _make_task(db, status="pending", priority="low")
        high_id   = _make_task(db, status="pending", priority="high")
        crit_id   = _make_task(db, status="pending", priority="critical")
        normal_id = _make_task(db, status="pending", priority="normal")

        results = manager.get_tasks(status="pending")
        ids = [r["id"] for r in results]

        assert ids.index(crit_id) < ids.index(high_id)
        assert ids.index(high_id) < ids.index(normal_id)
        assert ids.index(normal_id) < ids.index(low_id)

    def test_completed_sorted_most_recent_first(self, manager, db):
        """Completed tasks: most-recently completed appears first."""
        old_id   = _make_task(db, status="completed",
                               completed_at="2026-01-01T00:00:00+00:00")
        new_id   = _make_task(db, status="completed",
                               completed_at="2026-02-15T12:00:00+00:00")
        newer_id = _make_task(db, status="completed",
                               completed_at="2026-02-18T08:00:00+00:00")

        results = manager.get_tasks(status="completed")
        ids = [r["id"] for r in results]

        assert ids.index(newer_id) < ids.index(new_id) < ids.index(old_id)


# ---------------------------------------------------------------------------
# get_pending_tasks() backward-compat shim
# ---------------------------------------------------------------------------

class TestGetPendingTasksBackwardCompat:
    """get_pending_tasks() still works as a thin shim over get_tasks('pending')."""

    def test_returns_only_pending_tasks(self, manager, db):
        """get_pending_tasks() returns the same result as get_tasks('pending')."""
        pending_id = _make_task(db, status="pending")
        _make_task(db, status="completed")

        old_api = manager.get_pending_tasks()
        new_api = manager.get_tasks(status="pending")

        assert [r["id"] for r in old_api] == [r["id"] for r in new_api]
        assert any(r["id"] == pending_id for r in old_api)

    def test_domain_filter_passes_through(self, manager, db):
        """get_pending_tasks(domain=X) still filters correctly."""
        work_id   = _make_task(db, status="pending", domain="work")
        health_id = _make_task(db, status="pending", domain="health")

        results = manager.get_pending_tasks(domain="work")
        ids = [r["id"] for r in results]

        assert work_id in ids
        assert health_id not in ids


# ---------------------------------------------------------------------------
# /api/tasks route integration tests
# ---------------------------------------------------------------------------

class TestListTasksRoute:
    """GET /api/tasks now delegates to get_tasks() and honours the status param."""

    def test_default_calls_get_tasks_with_pending(self, client, mock_life_os):
        """GET /api/tasks with no params calls get_tasks(status='pending')."""
        mock_life_os.task_manager.get_tasks.return_value = [
            {"id": "t1", "title": "Pending task"}
        ]

        response = client.get("/api/tasks")

        assert response.status_code == 200
        mock_life_os.task_manager.get_tasks.assert_called_once_with(
            status="pending", limit=50
        )

    def test_status_completed_is_passed_through(self, client, mock_life_os):
        """GET /api/tasks?status=completed calls get_tasks(status='completed')."""
        mock_life_os.task_manager.get_tasks.return_value = [
            {"id": "c1", "title": "Done task", "status": "completed"}
        ]

        response = client.get("/api/tasks?status=completed")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["tasks"][0]["id"] == "c1"
        mock_life_os.task_manager.get_tasks.assert_called_once_with(
            status="completed", limit=50
        )

    def test_status_in_progress_is_passed_through(self, client, mock_life_os):
        """GET /api/tasks?status=in_progress calls get_tasks(status='in_progress')."""
        mock_life_os.task_manager.get_tasks.return_value = [
            {"id": "w1", "title": "WIP task", "status": "in_progress"}
        ]

        response = client.get("/api/tasks?status=in_progress")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        mock_life_os.task_manager.get_tasks.assert_called_once_with(
            status="in_progress", limit=50
        )

    def test_limit_parameter_forwarded(self, client, mock_life_os):
        """GET /api/tasks?limit=5 forwards limit=5 to get_tasks()."""
        mock_life_os.task_manager.get_tasks.return_value = []

        client.get("/api/tasks?limit=5")

        mock_life_os.task_manager.get_tasks.assert_called_once_with(
            status="pending", limit=5
        )

    def test_count_matches_returned_tasks(self, client, mock_life_os):
        """The count field always matches the number of task objects in the response."""
        mock_life_os.task_manager.get_tasks.return_value = [
            {"id": f"t{i}", "title": f"Task {i}"} for i in range(3)
        ]

        response = client.get("/api/tasks")

        data = response.json()
        assert data["count"] == 3
        assert len(data["tasks"]) == 3

    def test_empty_completed_result_is_valid(self, client, mock_life_os):
        """No completed tasks returns an empty list with count=0 (not 404)."""
        mock_life_os.task_manager.get_tasks.return_value = []

        response = client.get("/api/tasks?status=completed")

        assert response.status_code == 200
        data = response.json()
        assert data["tasks"] == []
        assert data["count"] == 0

    def test_status_and_limit_combined(self, client, mock_life_os):
        """GET /api/tasks?status=completed&limit=10 passes both parameters through."""
        mock_life_os.task_manager.get_tasks.return_value = []

        client.get("/api/tasks?status=completed&limit=10")

        mock_life_os.task_manager.get_tasks.assert_called_once_with(
            status="completed", limit=10
        )
