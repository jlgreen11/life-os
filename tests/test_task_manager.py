"""
Comprehensive test coverage for TaskManager.

Tests all public methods across multiple scenarios:
- Task CRUD operations (create, complete, update)
- Query methods (get_pending_tasks, get_overdue_tasks, get_tasks_due_soon)
- Context assembly (get_task_context with related events and contacts)
- AI extraction integration (ingest_ai_extracted_tasks)
- Task statistics (get_task_stats)
- Edge cases (invalid IDs, empty results, JSON serialization)
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_manager.manager import TaskManager


# --- Test: Task Creation ---


@pytest.mark.asyncio
async def test_create_task_minimal(db):
    """Test creating a task with only required fields."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(title="Test Task")

    assert task_id is not None
    assert isinstance(task_id, str)

    # Verify task was persisted with defaults
    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row is not None
        assert row["title"] == "Test Task"
        assert row["description"] is None
        assert row["source"] == "user"
        assert row["priority"] == "normal"
        assert row["status"] == "pending"
        assert row["created_at"] is not None


@pytest.mark.asyncio
async def test_create_task_full_metadata(db):
    """Test creating a task with all optional fields populated."""
    manager = TaskManager(db=db)

    due_date = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    reminder_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    task_id = await manager.create_task(
        title="Complex Task",
        description="Detailed description",
        source="ai_extracted",
        source_event_id="evt-123",
        source_context="Email from boss",
        domain="work",
        priority="high",
        tags=["urgent", "report"],
        due_date=due_date,
        reminder_at=reminder_at,
        estimated_minutes=60,
        related_contacts=["contact-1", "contact-2"],
        related_events=["event-1"],
    )

    # Verify all fields were persisted correctly
    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "Complex Task"
        assert row["description"] == "Detailed description"
        assert row["source"] == "ai_extracted"
        assert row["source_event_id"] == "evt-123"
        assert row["source_context"] == "Email from boss"
        assert row["domain"] == "work"
        assert row["priority"] == "high"
        assert row["due_date"] == due_date
        assert row["reminder_at"] == reminder_at
        assert row["estimated_minutes"] == 60

        # List fields are JSON-serialized
        assert json.loads(row["tags"]) == ["urgent", "report"]
        assert json.loads(row["related_contacts"]) == ["contact-1", "contact-2"]
        assert json.loads(row["related_events"]) == ["event-1"]


@pytest.mark.asyncio
async def test_create_task_with_event_bus(db):
    """Test that task creation publishes telemetry events."""
    mock_bus = MagicMock()
    mock_bus.is_connected = True
    mock_bus.publish = AsyncMock()

    manager = TaskManager(db=db, event_bus=mock_bus)

    task_id = await manager.create_task(
        title="Task with Event",
        source="rule",
        priority="critical",
        tags=["automated"],
    )

    # Verify telemetry event was published
    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args
    assert call_args[0][0] == "task.created"  # event type
    payload = call_args[0][1]
    assert payload["task_id"] == task_id
    assert payload["title"] == "Task with Event"
    assert payload["source"] == "rule"
    assert payload["priority"] == "critical"
    assert payload["tags"] == ["automated"]
    assert call_args[1]["source"] == "task_manager"


# --- Test: Task Completion ---


@pytest.mark.asyncio
async def test_complete_task(db):
    """Test completing a pending task."""
    manager = TaskManager(db=db)

    # Create a task
    task_id = await manager.create_task(title="Task to Complete")

    # Complete it
    await manager.complete_task(task_id)

    # Verify status changed and completed_at was set
    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["status"] == "completed"
        assert row["completed_at"] is not None
        assert row["updated_at"] is not None


@pytest.mark.asyncio
async def test_complete_task_with_event_bus(db):
    """Test that task completion publishes telemetry events."""
    mock_bus = MagicMock()
    mock_bus.is_connected = True
    mock_bus.publish = AsyncMock()

    manager = TaskManager(db=db, event_bus=mock_bus)

    task_id = await manager.create_task(title="Task to Complete", domain="personal")
    mock_bus.publish.reset_mock()  # Clear creation event

    await manager.complete_task(task_id)

    # Verify completion event was published
    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args
    assert call_args[0][0] == "task.completed"
    payload = call_args[0][1]
    assert payload["task_id"] == task_id
    assert payload["completed_at"] is not None
    assert payload["title"] == "Task to Complete"
    assert payload["domain"] == "personal"


@pytest.mark.asyncio
async def test_complete_nonexistent_task(db):
    """Test completing a task that doesn't exist (no-op, no error)."""
    manager = TaskManager(db=db)

    # Should not raise an error
    await manager.complete_task("nonexistent-id")


# --- Test: Task Updates ---


@pytest.mark.asyncio
async def test_update_task_single_field(db):
    """Test updating a single field on a task."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(title="Original Title", priority="normal")

    await manager.update_task(task_id, title="Updated Title")

    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "Updated Title"
        assert row["priority"] == "normal"  # Unchanged
        assert row["updated_at"] is not None


@pytest.mark.asyncio
async def test_update_task_multiple_fields(db):
    """Test updating multiple fields at once."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(title="Task", priority="normal")

    due_date = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    await manager.update_task(
        task_id,
        title="New Title",
        description="New description",
        priority="high",
        due_date=due_date,
        tags=["updated", "test"],
    )

    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "New Title"
        assert row["description"] == "New description"
        assert row["priority"] == "high"
        assert row["due_date"] == due_date
        assert json.loads(row["tags"]) == ["updated", "test"]


@pytest.mark.asyncio
async def test_update_task_filters_disallowed_fields(db):
    """Test that update_task silently ignores attempts to modify protected fields."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(title="Task", source="user")

    # Attempt to update protected fields (should be ignored)
    await manager.update_task(
        task_id,
        source="hacked",  # Not allowed
        created_at="2020-01-01T00:00:00Z",  # Not allowed
        id="new-id",  # Not allowed
        title="Allowed Update",  # Allowed
    )

    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "Allowed Update"  # This was allowed
        assert row["source"] == "user"  # This was protected
        assert row["id"] == task_id  # This was protected
        assert row["created_at"] != "2020-01-01T00:00:00Z"  # This was protected


@pytest.mark.asyncio
async def test_update_task_no_fields(db):
    """Test that update_task with no valid fields is a no-op."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(title="Task")

    # No allowed fields provided
    await manager.update_task(task_id, invalid_field="value")

    # Task should be unchanged (no error)
    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row["title"] == "Task"


@pytest.mark.asyncio
async def test_update_task_with_event_bus(db):
    """Test that task updates publish telemetry events."""
    mock_bus = MagicMock()
    mock_bus.is_connected = True
    mock_bus.publish = AsyncMock()

    manager = TaskManager(db=db, event_bus=mock_bus)

    task_id = await manager.create_task(title="Task")
    mock_bus.publish.reset_mock()

    await manager.update_task(task_id, title="Updated", priority="high")

    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args
    assert call_args[0][0] == "task.updated"
    payload = call_args[0][1]
    assert payload["task_id"] == task_id
    assert set(payload["updated_fields"]) == {"title", "priority"}


# --- Test: Query Methods ---


@pytest.mark.asyncio
async def test_get_pending_tasks_no_filters(db):
    """Test retrieving all pending tasks."""
    manager = TaskManager(db=db)

    # Create multiple tasks
    await manager.create_task(title="Task 1", priority="normal")
    await manager.create_task(title="Task 2", priority="high")
    task3_id = await manager.create_task(title="Task 3", priority="critical")

    # Complete one task
    await manager.complete_task(task3_id)

    tasks = manager.get_pending_tasks()

    # Should only return pending tasks (Task 1 and Task 2)
    assert len(tasks) == 2
    assert all(t["status"] == "pending" for t in tasks)

    # Should be sorted by priority (high before normal)
    assert tasks[0]["priority"] == "high"
    assert tasks[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_get_pending_tasks_priority_ordering(db):
    """Test that pending tasks are ordered by priority correctly."""
    manager = TaskManager(db=db)

    # Create tasks in random order
    await manager.create_task(title="Low", priority="low")
    await manager.create_task(title="Normal", priority="normal")
    await manager.create_task(title="Critical", priority="critical")
    await manager.create_task(title="High", priority="high")

    tasks = manager.get_pending_tasks()

    # Should be sorted: critical, high, normal, low
    assert tasks[0]["priority"] == "critical"
    assert tasks[1]["priority"] == "high"
    assert tasks[2]["priority"] == "normal"
    assert tasks[3]["priority"] == "low"


@pytest.mark.asyncio
async def test_get_pending_tasks_due_date_ordering(db):
    """Test that tasks with same priority are ordered by due date."""
    manager = TaskManager(db=db)

    now = datetime.now(timezone.utc)

    # Create tasks with same priority but different due dates
    await manager.create_task(
        title="Due in 3 days",
        priority="normal",
        due_date=(now + timedelta(days=3)).isoformat(),
    )
    await manager.create_task(
        title="Due in 1 day",
        priority="normal",
        due_date=(now + timedelta(days=1)).isoformat(),
    )
    await manager.create_task(
        title="No due date",
        priority="normal",
    )

    tasks = manager.get_pending_tasks()

    # Should be sorted by due date (earliest first, None last)
    assert tasks[0]["title"] == "Due in 1 day"
    assert tasks[1]["title"] == "Due in 3 days"
    assert tasks[2]["title"] == "No due date"


@pytest.mark.asyncio
async def test_get_pending_tasks_with_domain_filter(db):
    """Test filtering pending tasks by domain."""
    manager = TaskManager(db=db)

    await manager.create_task(title="Work Task", domain="work")
    await manager.create_task(title="Personal Task", domain="personal")
    await manager.create_task(title="Another Work Task", domain="work")

    work_tasks = manager.get_pending_tasks(domain="work")

    assert len(work_tasks) == 2
    assert all(t["domain"] == "work" for t in work_tasks)


@pytest.mark.asyncio
async def test_get_pending_tasks_with_priority_filter(db):
    """Test filtering pending tasks by priority."""
    manager = TaskManager(db=db)

    await manager.create_task(title="Critical Task", priority="critical")
    await manager.create_task(title="Normal Task", priority="normal")
    await manager.create_task(title="Another Critical", priority="critical")

    critical_tasks = manager.get_pending_tasks(priority="critical")

    assert len(critical_tasks) == 2
    assert all(t["priority"] == "critical" for t in critical_tasks)


@pytest.mark.asyncio
async def test_get_pending_tasks_with_limit(db):
    """Test limiting the number of returned tasks."""
    manager = TaskManager(db=db)

    # Create 10 tasks
    for i in range(10):
        await manager.create_task(title=f"Task {i}")

    tasks = manager.get_pending_tasks(limit=5)

    assert len(tasks) == 5


@pytest.mark.asyncio
async def test_get_overdue_tasks(db):
    """Test retrieving tasks past their due date."""
    manager = TaskManager(db=db)

    now = datetime.now(timezone.utc)

    # Create tasks with various due dates
    await manager.create_task(
        title="Overdue by 2 days",
        due_date=(now - timedelta(days=2)).isoformat(),
    )
    await manager.create_task(
        title="Overdue by 1 hour",
        due_date=(now - timedelta(hours=1)).isoformat(),
    )
    await manager.create_task(
        title="Due tomorrow",
        due_date=(now + timedelta(days=1)).isoformat(),
    )
    await manager.create_task(title="No due date")

    overdue = manager.get_overdue_tasks()

    assert len(overdue) == 2
    assert all("Overdue" in t["title"] for t in overdue)
    # Should be sorted by due date (oldest first)
    assert overdue[0]["title"] == "Overdue by 2 days"


@pytest.mark.asyncio
async def test_get_tasks_due_soon_default_24h(db):
    """Test retrieving tasks due within the next 24 hours."""
    manager = TaskManager(db=db)

    now = datetime.now(timezone.utc)

    # Create tasks with various due dates
    await manager.create_task(
        title="Due in 12 hours",
        due_date=(now + timedelta(hours=12)).isoformat(),
    )
    await manager.create_task(
        title="Due in 30 hours",
        due_date=(now + timedelta(hours=30)).isoformat(),
    )
    await manager.create_task(
        title="Due in 1 hour",
        due_date=(now + timedelta(hours=1)).isoformat(),
    )

    due_soon = manager.get_tasks_due_soon()  # Default 24 hours

    assert len(due_soon) == 2
    assert due_soon[0]["title"] == "Due in 1 hour"
    assert due_soon[1]["title"] == "Due in 12 hours"


@pytest.mark.asyncio
async def test_get_tasks_due_soon_custom_window(db):
    """Test retrieving tasks due within a custom time window."""
    manager = TaskManager(db=db)

    now = datetime.now(timezone.utc)

    await manager.create_task(
        title="Due in 30 hours",
        due_date=(now + timedelta(hours=30)).isoformat(),
    )
    await manager.create_task(
        title="Due in 50 hours",
        due_date=(now + timedelta(hours=50)).isoformat(),
    )

    due_soon = manager.get_tasks_due_soon(hours=48)

    assert len(due_soon) == 1
    assert due_soon[0]["title"] == "Due in 30 hours"


# --- Test: Task Context Assembly ---


@pytest.mark.asyncio
async def test_get_task_context_basic(db):
    """Test retrieving basic task context without relations."""
    manager = TaskManager(db=db)

    task_id = await manager.create_task(
        title="Standalone Task",
        description="No relations",
    )

    context = manager.get_task_context(task_id)

    assert "task" in context
    assert context["task"]["id"] == task_id
    assert context["task"]["title"] == "Standalone Task"
    assert "source_event" not in context
    assert "related_contacts" not in context


@pytest.mark.asyncio
async def test_get_task_context_with_source_event(db):
    """Test that task context includes the originating event."""
    manager = TaskManager(db=db)

    # Create a source event in the events database
    event_id = str(uuid.uuid4())
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload)
               VALUES (?, ?, ?, ?, ?)""",
            (event_id, "email.received", "gmail", datetime.now(timezone.utc).isoformat(), "{}"),
        )

    # Create task referencing the event
    task_id = await manager.create_task(
        title="Task from Email",
        source_event_id=event_id,
    )

    context = manager.get_task_context(task_id)

    assert "source_event" in context
    assert context["source_event"]["id"] == event_id
    assert context["source_event"]["type"] == "email.received"


@pytest.mark.asyncio
async def test_get_task_context_with_related_contacts(db):
    """Test that task context includes full contact details."""
    manager = TaskManager(db=db)

    # Create contacts in the entities database
    contact1_id = str(uuid.uuid4())
    contact2_id = str(uuid.uuid4())
    with db.get_connection("entities") as conn:
        conn.execute(
            """INSERT INTO contacts (id, name, emails)
               VALUES (?, ?, ?)""",
            (contact1_id, "Alice", '["alice@example.com"]'),
        )
        conn.execute(
            """INSERT INTO contacts (id, name, emails)
               VALUES (?, ?, ?)""",
            (contact2_id, "Bob", '["bob@example.com"]'),
        )

    # Create task with related contacts
    task_id = await manager.create_task(
        title="Team Task",
        related_contacts=[contact1_id, contact2_id],
    )

    context = manager.get_task_context(task_id)

    assert "related_contacts" in context
    assert len(context["related_contacts"]) == 2

    contact_names = {c["name"] for c in context["related_contacts"]}
    assert contact_names == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_get_task_context_nonexistent_task(db):
    """Test that context for nonexistent task returns empty dict."""
    manager = TaskManager(db=db)

    context = manager.get_task_context("nonexistent-id")

    assert context == {}


# --- Test: AI Extraction Integration ---


@pytest.mark.asyncio
async def test_ingest_ai_extracted_tasks(db):
    """Test ingesting tasks extracted by the AI engine."""
    manager = TaskManager(db=db)

    source_event_id = "email-123"

    extracted_tasks = [
        {
            "title": "Send report by Friday",
            "priority": "high",
            "due_hint": "2026-02-21T17:00:00Z",
        },
        {
            "title": "Review proposal",
            "priority": "normal",
        },
    ]

    await manager.ingest_ai_extracted_tasks(extracted_tasks, source_event_id)

    # Verify both tasks were created
    with db.get_connection("state") as conn:
        rows = conn.execute(
            """SELECT * FROM tasks WHERE source = 'ai_extracted'
               ORDER BY CASE priority
                   WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                   WHEN 'normal' THEN 3 ELSE 4
               END"""
        ).fetchall()

        assert len(rows) == 2

        # First task (high priority)
        assert rows[0]["title"] == "Send report by Friday"
        assert rows[0]["source"] == "ai_extracted"
        assert rows[0]["source_event_id"] == source_event_id
        assert rows[0]["priority"] == "high"
        assert rows[0]["due_date"] == "2026-02-21T17:00:00Z"

        # Second task (normal priority)
        assert rows[1]["title"] == "Review proposal"
        assert rows[1]["source_event_id"] == source_event_id
        assert rows[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_ingest_ai_extracted_tasks_with_defaults(db):
    """Test that AI-extracted tasks use defaults for missing fields."""
    manager = TaskManager(db=db)

    # Task with minimal data
    extracted_tasks = [{"title": "Minimal Task"}]

    await manager.ingest_ai_extracted_tasks(extracted_tasks, "event-id")

    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE title = 'Minimal Task'").fetchone()
        assert row["source"] == "ai_extracted"
        assert row["priority"] == "normal"  # Default
        assert row["due_date"] is None


@pytest.mark.asyncio
async def test_ingest_ai_extracted_tasks_empty_title(db):
    """Test that tasks with no title get 'Untitled task' as default."""
    manager = TaskManager(db=db)

    extracted_tasks = [{"priority": "high"}]  # No title

    await manager.ingest_ai_extracted_tasks(extracted_tasks, "event-id")

    with db.get_connection("state") as conn:
        row = conn.execute("SELECT * FROM tasks WHERE source = 'ai_extracted'").fetchone()
        assert row["title"] == "Untitled task"


# --- Test: Task Statistics ---


@pytest.mark.asyncio
async def test_get_task_stats_empty(db):
    """Test task statistics when no tasks exist."""
    manager = TaskManager(db=db)

    stats = manager.get_task_stats()

    assert stats["pending"] == 0
    assert stats["completed_today"] == 0
    assert stats["overdue"] == 0
    assert stats["by_domain"] == {}


@pytest.mark.asyncio
async def test_get_task_stats_with_tasks(db):
    """Test task statistics with various task states."""
    manager = TaskManager(db=db)

    now = datetime.now(timezone.utc)

    # Create pending tasks in different domains
    await manager.create_task(title="Work Task 1", domain="work")
    await manager.create_task(title="Work Task 2", domain="work")
    await manager.create_task(title="Personal Task", domain="personal")

    # Create an overdue task
    overdue_id = await manager.create_task(
        title="Overdue",
        due_date=(now - timedelta(days=1)).isoformat(),
    )

    # Create and complete a task
    completed_id = await manager.create_task(title="Completed")
    await manager.complete_task(completed_id)

    stats = manager.get_task_stats()

    assert stats["pending"] == 4  # 3 regular + 1 overdue
    assert stats["completed_today"] == 1
    assert stats["overdue"] == 1
    assert stats["by_domain"]["work"] == 2
    assert stats["by_domain"]["personal"] == 1
    assert stats["by_domain"]["unassigned"] == 1  # The overdue task


# --- Test: Row Serialization ---


def test_row_to_dict_deserializes_json_fields(db):
    """Test that _row_to_dict properly deserializes JSON list fields."""
    manager = TaskManager(db=db)

    # Create a mock row with JSON fields
    class MockRow:
        def __init__(self):
            self.data = {
                "id": "task-1",
                "title": "Test",
                "tags": '["tag1", "tag2"]',
                "related_contacts": '["contact-1"]',
                "related_events": '[]',
                "description": "Normal field",
            }

        def keys(self):
            return self.data.keys()

        def __getitem__(self, key):
            return self.data[key]

    row = MockRow()
    result = manager._row_to_dict(row)

    assert result["tags"] == ["tag1", "tag2"]
    assert result["related_contacts"] == ["contact-1"]
    assert result["related_events"] == []
    assert result["description"] == "Normal field"


def test_row_to_dict_handles_malformed_json(db):
    """Test that _row_to_dict handles malformed JSON gracefully."""
    manager = TaskManager(db=db)

    class MockRow:
        def __init__(self):
            self.data = {
                "id": "task-1",
                "tags": "invalid-json{",  # Malformed - should fallback to []
                "related_contacts": "also-invalid",  # Malformed - should fallback to []
                "related_events": '["valid"]',  # Valid JSON
                "other_field": "normal",  # Non-JSON field
            }

        def keys(self):
            return self.data.keys()

        def __getitem__(self, key):
            return self.data[key]

    row = MockRow()
    result = manager._row_to_dict(row)

    # Malformed JSON should fallback to empty lists
    assert result["tags"] == []
    assert result["related_contacts"] == []
    # Valid JSON should deserialize correctly
    assert result["related_events"] == ["valid"]
    # Other fields should pass through unchanged
    assert result["other_field"] == "normal"
