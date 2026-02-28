"""
Tests for the backfill task extraction script.

Validates that the backfill script correctly processes historical events
and extracts tasks using the AI engine.
"""

import pytest
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add scripts directory to path to import the backfill module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from backfill_task_extraction import backfill_tasks, _count_tasks
from services.task_manager.manager import TaskManager


@pytest.mark.asyncio
async def test_backfill_finds_all_actionable_events(db, event_store):
    """
    Backfill should identify all email, message, and calendar events.

    The script queries for five specific event types that can contain
    action items (both inbound and outbound). Other event types should be ignored.
    """
    # Create mix of actionable and non-actionable events
    now = datetime.now(timezone.utc).isoformat()

    # Actionable events (should be processed)
    event_store.store_event({
        "id": "email-recv",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Please finish the report by Friday."},
        "metadata": {}
    })
    event_store.store_event({
        "id": "email-sent",
        "type": "email.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "I finished the analysis report yesterday."},
        "metadata": {}
    })
    event_store.store_event({
        "id": "msg-recv",
        "type": "message.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Don't forget to book the hotel for the conference."},
        "metadata": {}
    })
    event_store.store_event({
        "id": "msg-sent",
        "type": "message.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Booked the hotel and sent confirmation."},
        "metadata": {}
    })
    event_store.store_event({
        "id": "cal-1",
        "type": "calendar.event.created",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"description": "TODO: Prepare slides and handouts."},
        "metadata": {}
    })

    # Non-actionable events (should be ignored)
    event_store.store_event({
        "id": "pred-1",
        "type": "usermodel.prediction.generated",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {},
        "metadata": {}
    })
    event_store.store_event({
        "id": "notif-1",
        "type": "notification.created",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {},
        "metadata": {}
    })

    # Mock AI engine that always finds one task
    class MockAIEngine:
        async def extract_action_items(self, text, source):
            if len(text) > 20:  # Mimic the real threshold
                return [{"title": "Mock task", "due_hint": None, "priority": "normal"}]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    # Run backfill (dry run to count events)
    stats = await backfill_tasks(db, task_manager, dry_run=True)

    # Should find exactly 5 actionable events (2 email + 2 message + 1 calendar)
    assert stats["total_events"] == 5


@pytest.mark.asyncio
async def test_backfill_extracts_tasks_from_emails(db, event_store):
    """
    Backfill should extract action items from email bodies.

    Emails often contain requests, commitments, and action items that
    the AI engine can identify and convert into tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Email with clear action item
    event_store.store_event({
        "id": "email-task-1",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Q4 Planning",
            "body": "Can you please prepare the budget proposal by end of week?"
        },
        "metadata": {}
    })

    # Mock AI engine that extracts the task
    class MockAIEngine:
        async def extract_action_items(self, text, source):
            if "budget proposal" in text:
                return [{
                    "title": "Prepare budget proposal",
                    "due_hint": "end of week",
                    "priority": "high"
                }]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    tasks_before = _count_tasks(db)
    stats = await backfill_tasks(db, task_manager)
    tasks_after = _count_tasks(db)

    assert stats["total_events"] == 1
    assert tasks_after == tasks_before + 1


@pytest.mark.asyncio
async def test_backfill_handles_html_email_bodies(db, event_store):
    """
    Backfill should strip HTML from email bodies before extraction.

    Many email connectors store raw HTML. The AI engine needs plain text
    to extract action items effectively.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Email with HTML body
    event_store.store_event({
        "id": "html-email-1",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {
            "body": "<html><body><p>Please <b>review</b> the contract by Monday.</p></body></html>"
        },
        "metadata": {}
    })

    # Mock AI engine that checks for plain text (not HTML tags)
    class MockAIEngine:
        async def extract_action_items(self, text, source):
            # Should receive "Please review the contract by Monday."
            # not "<html><body><p>Please <b>review</b>..."
            assert "<html>" not in text
            assert "<b>" not in text
            assert "review the contract" in text
            return [{"title": "Review contract", "due_hint": "Monday", "priority": "normal"}]

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    stats = await backfill_tasks(db, task_manager)
    assert stats["tasks_extracted"] == 1


@pytest.mark.asyncio
async def test_backfill_skips_trivial_messages(db, event_store):
    """
    Backfill should skip messages that are too short to contain action items.

    Short messages like "ok", "thanks", "got it" don't need AI processing.
    The 20-character threshold filters these out to reduce LLM load.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Short, trivial messages (should be skipped)
    for i, body in enumerate(["ok", "thanks", "got it", "👍"]):
        event_store.store_event({
            "id": f"trivial-{i}",
            "type": "message.received",
            "source": "test",
            "timestamp": now,
            "priority": "normal",
            "payload": {"body": body},
            "metadata": {}
        })

    # One substantial message (should be processed)
    event_store.store_event({
        "id": "substantial-1",
        "type": "message.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Can you send me the slides from yesterday's meeting?"},
        "metadata": {}
    })

    # Mock AI engine that tracks how many times it was called
    class MockAIEngine:
        def __init__(self):
            self.call_count = 0

        async def extract_action_items(self, text, source):
            self.call_count += 1
            return [{"title": "Send meeting slides", "due_hint": None, "priority": "normal"}]

    ai_engine = MockAIEngine()
    task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

    stats = await backfill_tasks(db, task_manager)

    # Should process only 1 event (the substantial one), skipping the 4 trivial ones
    assert ai_engine.call_count == 1
    assert stats["tasks_extracted"] == 1


@pytest.mark.asyncio
async def test_backfill_handles_calendar_descriptions(db, event_store):
    """
    Backfill should extract action items from calendar event descriptions.

    Calendar events often contain meeting agendas with TODO items, prep tasks,
    or follow-up actions embedded in the description field.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_store.store_event({
        "id": "cal-with-todos",
        "type": "calendar.event.created",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {
            "summary": "Project Kickoff Meeting",
            "description": "Agenda: Review timeline. TODO: Set up Slack channel. TODO: Book conference room."
        },
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            if "Set up Slack channel" in text and "Book conference room" in text:
                return [
                    {"title": "Set up Slack channel", "due_hint": None, "priority": "normal"},
                    {"title": "Book conference room", "due_hint": None, "priority": "normal"}
                ]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    stats = await backfill_tasks(db, task_manager)

    # Should extract 2 tasks from the calendar description
    assert stats["tasks_extracted"] == 2


@pytest.mark.asyncio
async def test_backfill_limit_parameter(db, event_store):
    """
    Backfill should respect the --limit parameter for testing.

    When limit=N is specified, only the first N events should be processed.
    This is useful for testing the script on a subset of events.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Create 10 events
    for i in range(10):
        event_store.store_event({
            "id": f"email-{i}",
            "type": "email.received",
            "source": "test",
            "timestamp": now,
            "priority": "normal",
            "payload": {"body": f"This is email number {i} with enough text to pass the threshold."},
            "metadata": {}
        })

    class MockAIEngine:
        """Returns a unique task title per call to avoid deduplication blocking."""
        def __init__(self):
            self._call = 0

        async def extract_action_items(self, text, source):
            self._call += 1
            return [{"title": f"Task {self._call}", "due_hint": None, "priority": "normal"}]

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    # Run with limit=3
    stats = await backfill_tasks(db, task_manager, limit=3)

    # Should process only 3 events, extract 3 tasks
    assert stats["total_events"] == 3
    assert stats["tasks_extracted"] == 3


@pytest.mark.asyncio
async def test_backfill_dry_run_mode(db, event_store):
    """
    Backfill dry run should count events but not create tasks.

    The --dry-run flag shows what would be processed without making changes.
    This is useful for previewing the backfill scope.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_store.store_event({
        "id": "email-dry-run",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "This email contains an action item."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            return [{"title": "Task", "due_hint": None, "priority": "normal"}]

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    tasks_before = _count_tasks(db)
    stats = await backfill_tasks(db, task_manager, dry_run=True)
    tasks_after = _count_tasks(db)

    # Should report 1 event found, but 0 tasks extracted (dry run)
    assert stats["total_events"] == 1
    assert stats["tasks_extracted"] == 0
    # No tasks should be created
    assert tasks_after == tasks_before


@pytest.mark.asyncio
async def test_backfill_error_handling(db, event_store):
    """
    Backfill should continue processing after AI engine errors.

    If the AI engine fails on one event (model down, parsing error, etc.),
    the TaskManager's error handling will catch it and log it, then the
    backfill continues with remaining events. The error is handled at the
    TaskManager level, so the backfill script doesn't count it as an error.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Create 3 events
    for i in range(3):
        event_store.store_event({
            "id": f"email-error-{i}",
            "type": "email.received",
            "source": "test",
            "timestamp": now,
            "priority": "normal",
            "payload": {"body": f"Email {i} with sufficient text for processing."},
            "metadata": {}
        })

    # Mock AI engine that fails on the second event
    class FailingAIEngine:
        def __init__(self):
            self.call_count = 0

        async def extract_action_items(self, text, source):
            self.call_count += 1
            if self.call_count == 2:
                raise Exception("Simulated AI engine failure")
            return [{"title": f"Task {self.call_count}", "due_hint": None, "priority": "normal"}]

    ai_engine = FailingAIEngine()
    task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

    stats = await backfill_tasks(db, task_manager)

    # Should process all 3 events despite the error
    assert stats["total_events"] == 3
    # Should extract 2 tasks (skipping the failed one)
    assert stats["tasks_extracted"] == 2
    # TaskManager catches the error internally, so backfill doesn't see it
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_backfill_is_idempotent(db, event_store):
    """
    Backfill should be safe to run multiple times.

    Running the script twice on the same events should not create duplicate tasks.
    Each task is linked to a source_event_id, and the script processes all events
    regardless of whether they've been processed before, but the AI engine may
    return [] for events that don't contain actionable items.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_store.store_event({
        "id": "email-idem-1",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Please complete the onboarding checklist."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            return [{"title": "Complete onboarding", "due_hint": None, "priority": "normal"}]

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    # Run backfill twice
    stats1 = await backfill_tasks(db, task_manager)
    tasks_after_first = _count_tasks(db)

    stats2 = await backfill_tasks(db, task_manager)
    tasks_after_second = _count_tasks(db)

    # First run should extract 1 task
    assert stats1["tasks_extracted"] == 1
    # Second run: deduplication suppresses the re-creation.
    # The task already exists from the first run, so the AI's output is silently
    # skipped — true idempotency: re-running the backfill does not accumulate
    # duplicate tasks that the user would have to dismiss one by one.
    assert stats2["tasks_extracted"] == 0
    # Total task count must not grow on the second run
    assert tasks_after_second == tasks_after_first


@pytest.mark.asyncio
async def test_backfill_processes_newest_first(db, event_store):
    """
    Backfill should process events in reverse chronological order.

    The script queries events with ORDER BY timestamp DESC, so the most
    recent events (most likely to contain still-relevant tasks) are
    processed first. This is important if the backfill is interrupted.
    """
    # Create events with different timestamps
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    event_store.store_event({
        "id": "old-email",
        "type": "email.received",
        "source": "test",
        "timestamp": base_time.isoformat(),
        "priority": "normal",
        "payload": {"body": "Old email from January."},
        "metadata": {}
    })

    new_time = datetime(2026, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
    event_store.store_event({
        "id": "new-email",
        "type": "email.received",
        "source": "test",
        "timestamp": new_time.isoformat(),
        "priority": "normal",
        "payload": {"body": "Recent email from February."},
        "metadata": {}
    })

    # Mock AI engine that tracks which event was processed first
    class OrderTrackingAIEngine:
        def __init__(self):
            self.order = []

        async def extract_action_items(self, text, source):
            if "February" in text:
                self.order.append("new")
            elif "January" in text:
                self.order.append("old")
            return []

    ai_engine = OrderTrackingAIEngine()
    task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

    await backfill_tasks(db, task_manager)

    # New event should be processed before old event
    assert ai_engine.order == ["new", "old"]


@pytest.mark.asyncio
async def test_backfill_links_tasks_to_source_events(db, event_store):
    """
    Backfill should link each task to its source event.

    Tasks have a source_event_id field that maintains full provenance back
    to the original email/message/calendar event. This enables the user to
    see where each task came from.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_id = "source-event-123"
    event_store.store_event({
        "id": event_id,
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Please review the quarterly report by Friday."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            return [{"title": "Review quarterly report", "due_hint": "Friday", "priority": "high"}]

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    await backfill_tasks(db, task_manager)

    # Verify the task was created with correct source_event_id
    with db.get_connection("state") as conn:
        task = conn.execute("SELECT * FROM tasks WHERE source_event_id = ?", (event_id,)).fetchone()

    assert task is not None
    assert task["title"] == "Review quarterly report"
    assert task["source"] == "ai_extracted"
    assert task["source_event_id"] == event_id


@pytest.mark.asyncio
async def test_backfill_without_ai_engine(db, event_store):
    """
    Backfill should gracefully skip extraction if AI engine is not available.

    If the task manager is initialized without an AI engine (e.g., in a
    minimal deployment), the backfill should complete without errors.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_store.store_event({
        "id": "no-ai-1",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Email that would normally trigger extraction."},
        "metadata": {}
    })

    # Task manager without AI engine
    task_manager = TaskManager(db, event_bus=None, ai_engine=None)

    stats = await backfill_tasks(db, task_manager)

    # Should process the event but not extract any tasks
    assert stats["total_events"] == 1
    assert stats["tasks_extracted"] == 0
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_count_tasks_helper(db):
    """
    The _count_tasks helper should accurately count tasks in the database.

    This helper is used to compute tasks_extracted by comparing counts
    before and after the backfill.
    """
    # Initially should be 0
    assert _count_tasks(db) == 0

    # Create some tasks manually
    with db.get_connection("state") as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO tasks (id, title, source, created_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (f"task-{i}", f"Task {i}", "test")
            )

    assert _count_tasks(db) == 5


@pytest.mark.asyncio
async def test_backfill_processes_sent_events(db, event_store):
    """
    Backfill should process both received AND sent events.

    INBOUND events (email.received, message.received) contain pending tasks.
    OUTBOUND events (email.sent, message.sent) may contain completion reports
    like "I finished the report" or "I sent the document yesterday".

    This enables workflow detection from historical data by generating
    task.completed events instead of waiting 7+ days for task aging.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Create both received and sent events
    event_store.store_event({
        "id": "email-received",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Can you please send the Q4 report by Friday?"},
        "metadata": {}
    })

    event_store.store_event({
        "id": "email-sent",
        "type": "email.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "I finished the Q4 report and sent it yesterday."},
        "metadata": {}
    })

    event_store.store_event({
        "id": "message-sent",
        "type": "message.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Submitted the expense report this morning."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            # Simulate AI detecting tasks and their completion status
            if "send the Q4 report" in text:
                return [{"title": "Send Q4 report", "priority": "high", "completed": False}]
            elif "finished the Q4 report" in text:
                return [{"title": "Q4 report", "priority": "normal", "completed": True}]
            elif "Submitted the expense report" in text:
                return [{"title": "Submit expense report", "priority": "normal", "completed": True}]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    stats = await backfill_tasks(db, task_manager)

    # Should process all 3 events (1 received + 2 sent)
    assert stats["total_events"] == 3
    # Should extract 3 tasks total
    assert stats["tasks_extracted"] == 3


@pytest.mark.asyncio
async def test_backfill_detects_completed_tasks(db, event_store):
    """
    Backfill should detect and mark tasks as completed when AI identifies them.

    When the AI detects that a task was already completed (e.g., "I sent the
    report yesterday"), the task should be created with status='completed'
    and a task.completed event should be published for workflow detection.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Sent email reporting completion
    event_store.store_event({
        "id": "sent-completion",
        "type": "email.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {
            "body": "Hi team, I completed the security audit and sent the findings to IT."
        },
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            # AI detects this is a completed task
            if "completed the security audit" in text:
                return [{
                    "title": "Complete security audit",
                    "priority": "high",
                    "completed": True
                }]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    tasks_before = _count_tasks(db)
    stats = await backfill_tasks(db, task_manager)
    tasks_after = _count_tasks(db)

    # Should create 1 task
    assert stats["tasks_extracted"] == 1
    assert tasks_after == tasks_before + 1

    # Verify the task was marked as completed
    with db.get_connection("state") as conn:
        task = conn.execute(
            "SELECT * FROM tasks WHERE source_event_id = ?",
            ("sent-completion",)
        ).fetchone()

    assert task is not None
    assert task["title"] == "Complete security audit"
    assert task["status"] == "completed"
    assert task["completed_at"] is not None


@pytest.mark.asyncio
async def test_backfill_generates_completion_events(db, event_store):
    """
    Backfill should publish task.completed events for completed tasks.

    When a task is detected as already completed, the backfill script
    must publish a task.completed event to events.db. This enables the
    workflow detector to learn multi-step patterns from historical data.
    """
    now = datetime.now(timezone.utc).isoformat()

    event_store.store_event({
        "id": "completed-task-source",
        "type": "message.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "I deployed the hotfix to production this morning."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            if "deployed the hotfix" in text:
                return [{
                    "title": "Deploy hotfix to production",
                    "priority": "critical",
                    "completed": True
                }]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    # Count task.completed events before backfill
    with db.get_connection("events") as conn:
        before_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE type = 'task.completed'"
        ).fetchone()[0]

    await backfill_tasks(db, task_manager)

    # Count task.completed events after backfill
    with db.get_connection("events") as conn:
        after_count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE type = 'task.completed'"
        ).fetchone()[0]

    # Should have published 1 task.completed event
    assert after_count == before_count + 1

    # Verify the event has correct payload
    with db.get_connection("events") as conn:
        completion_event = conn.execute(
            "SELECT * FROM events WHERE type = 'task.completed' ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

    assert completion_event is not None
    payload = json.loads(completion_event["payload"])
    assert payload["title"] == "Deploy hotfix to production"
    assert "task_id" in payload
    assert "completed_at" in payload


@pytest.mark.asyncio
async def test_backfill_mixed_pending_and_completed_tasks(db, event_store):
    """
    Backfill should handle a mix of pending and completed tasks correctly.

    Some events contain pending tasks ("please do X"), others contain
    completion reports ("I did X"). The AI should detect the difference
    and mark them appropriately.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Pending task (received email)
    event_store.store_event({
        "id": "pending-task",
        "type": "email.received",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "Please review the pull request when you get a chance."},
        "metadata": {}
    })

    # Completed task (sent email)
    event_store.store_event({
        "id": "completed-task",
        "type": "email.sent",
        "source": "test",
        "timestamp": now,
        "priority": "normal",
        "payload": {"body": "I reviewed and approved the pull request."},
        "metadata": {}
    })

    class MockAIEngine:
        async def extract_action_items(self, text, source):
            if "Please review the pull request" in text:
                return [{"title": "Review pull request", "priority": "normal", "completed": False}]
            elif "reviewed and approved the pull request" in text:
                return [{"title": "Review pull request", "priority": "normal", "completed": True}]
            return []

    task_manager = TaskManager(db, event_bus=None, ai_engine=MockAIEngine())

    await backfill_tasks(db, task_manager)

    # Should extract 2 tasks total
    assert _count_tasks(db) == 2

    # Check statuses
    with db.get_connection("state") as conn:
        pending_task = conn.execute(
            "SELECT * FROM tasks WHERE source_event_id = ?",
            ("pending-task",)
        ).fetchone()
        completed_task = conn.execute(
            "SELECT * FROM tasks WHERE source_event_id = ?",
            ("completed-task",)
        ).fetchone()

    assert pending_task["status"] == "pending"
    assert pending_task["completed_at"] is None

    assert completed_task["status"] == "completed"
    assert completed_task["completed_at"] is not None
