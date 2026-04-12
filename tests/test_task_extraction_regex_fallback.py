"""
Tests for TaskManager regex-based task extraction fallback.

When the AI engine is unavailable (ai_engine=None) or raises an exception,
process_event should fall through to _regex_extract_tasks() to capture
obvious action items from the message text rather than silently discarding
all events.

Coverage:
  1. Fallback triggers when ai_engine is None
  2. Fallback triggers when AI raises an exception
  3. Regex detects TODO/ACTION/TASK markers
  4. Urgency markers elevate priority to 'high'
  5. Completion signals set completed=True on extracted tasks
  6. FYI / informational messages produce no tasks
  7. Telemetry counters (_tasks_regex_extracted, _events_regex_fallback) increment
  8. Marketing emails are still filtered before regex extraction
"""

from unittest.mock import AsyncMock

import pytest

from services.task_manager.manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_email_event(body: str, from_address: str = "colleague@company.com", event_id: str = "evt-test") -> dict:
    """Build a minimal email.received event dict for testing."""
    return {
        "id": event_id,
        "type": "email.received",
        "payload": {
            "from_address": from_address,
            "body": body,
        },
    }


# ---------------------------------------------------------------------------
# Fixture: task manager with NO AI engine
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager_no_ai(db):
    """TaskManager with no AI engine — every event hits the regex fallback."""
    return TaskManager(db=db, event_bus=None, ai_engine=None)


# ---------------------------------------------------------------------------
# Test 1: Fallback triggers when ai_engine is None
# ---------------------------------------------------------------------------


async def test_fallback_triggers_when_ai_engine_is_none(manager_no_ai):
    """With ai_engine=None, a clear action-item email should produce a task
    via the regex fallback.  The due_hint should reflect the deadline phrase."""
    await manager_no_ai.process_event(
        make_email_event("Please review the budget proposal by Friday.")
    )

    tasks = manager_no_ai.get_pending_tasks()
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}: {tasks}"
    task = tasks[0]
    assert "review" in task["title"].lower() or "budget" in task["title"].lower()
    # ingest_ai_extracted_tasks stores due_hint as due_date in the tasks table
    assert task["due_date"] is not None
    assert "friday" in task["due_date"].lower()
    assert task["source"] == "ai_extracted"  # ingest_ai_extracted_tasks always sets this


# ---------------------------------------------------------------------------
# Test 2: Fallback triggers when AI raises an exception
# ---------------------------------------------------------------------------


async def test_fallback_triggers_on_ai_exception(db):
    """When the AI engine raises, the except block should call regex fallback
    so that obvious action items are not lost."""
    mock_engine = AsyncMock()
    mock_engine.extract_action_items = AsyncMock(side_effect=RuntimeError("Model unavailable"))

    manager = TaskManager(db=db, event_bus=None, ai_engine=mock_engine)

    await manager.process_event(
        make_email_event("Please send me the updated project timeline by tomorrow.")
    )

    # AI was called (and raised), regex fallback took over
    mock_engine.extract_action_items.assert_called_once()

    tasks = manager.get_pending_tasks()
    assert len(tasks) >= 1, "Regex fallback should have created at least one task"
    titles_lower = [t["title"].lower() for t in tasks]
    assert any("send" in t or "timeline" in t or "project" in t for t in titles_lower)


# ---------------------------------------------------------------------------
# Test 3: Regex detects TODO/ACTION/TASK markers
# ---------------------------------------------------------------------------


async def test_regex_detects_todo_marker(manager_no_ai):
    """'TODO: Update the CI pipeline config' should produce a task with the
    title extracted after the marker."""
    await manager_no_ai.process_event(
        make_email_event("Hi, just a quick note. TODO: Update the CI pipeline config")
    )

    tasks = manager_no_ai.get_pending_tasks()
    assert len(tasks) >= 1
    titles_lower = [t["title"].lower() for t in tasks]
    assert any("update the ci pipeline config" in t for t in titles_lower), (
        f"Expected TODO task title in {titles_lower}"
    )


# ---------------------------------------------------------------------------
# Test 4: Urgency markers elevate priority to 'high'
# ---------------------------------------------------------------------------


async def test_regex_detects_urgency(manager_no_ai):
    """Messages containing urgency markers ('urgent', 'ASAP', etc.) should
    produce tasks with priority='high'."""
    await manager_no_ai.process_event(
        make_email_event("This is urgent — please submit the expense report immediately.")
    )

    tasks = manager_no_ai.get_pending_tasks()
    assert len(tasks) >= 1, f"Expected at least one task, got {tasks}"
    assert all(t["priority"] == "high" for t in tasks), (
        f"Expected priority='high' for all tasks, got {[t['priority'] for t in tasks]}"
    )


# ---------------------------------------------------------------------------
# Test 5: Completion signals set completed=True
# ---------------------------------------------------------------------------


async def test_regex_detects_completed_signal(manager_no_ai):
    """When a message contains both a task-trigger pattern (e.g. 'please
    acknowledge') AND a completion signal ('I've already submitted'), the
    extracted task should be immediately marked as completed.

    The completion flag is applied globally to all tasks extracted from the
    same message — if the sender is reporting past work while also asking for
    acknowledgment, the task reflects the completion state of that exchange.
    """
    # This message has BOTH a task-trigger ('please acknowledge') AND a
    # completion signal ('I've already submitted').  The regex should extract
    # a task and mark it completed because of the completion phrase.
    await manager_no_ai.process_event(
        make_email_event(
            "I've already submitted the quarterly report. "
            "Please acknowledge receipt when you get a chance."
        )
    )

    # Completed tasks land in 'completed' status, not pending
    with manager_no_ai.db.get_connection("state") as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE status = 'completed'"
        ).fetchall()

    assert len(rows) >= 1, (
        "Completion signal should mark the extracted task as completed. "
        "Check that the message contains a task-trigger pattern (please/TODO/etc.) "
        "alongside the completion signal."
    )


# ---------------------------------------------------------------------------
# Test 6: FYI / informational messages produce no tasks
# ---------------------------------------------------------------------------


async def test_regex_skips_fyi_messages(manager_no_ai):
    """Messages explicitly flagged as informational (FYI, no action required,
    etc.) should never generate tasks."""
    fyi_messages = [
        "FYI - the server was restarted last night.",
        "Just letting you know the report is available.",
        "For your information, the meeting has been rescheduled.",
        "No action required — this is just a status update.",
        "Just a heads up — the deadline was extended.",
    ]
    for body in fyi_messages:
        await manager_no_ai.process_event(make_email_event(body, event_id=f"evt-{body[:10]}"))

    tasks = manager_no_ai.get_pending_tasks()
    assert len(tasks) == 0, (
        f"FYI messages should produce no tasks, but got: {[t['title'] for t in tasks]}"
    )


# ---------------------------------------------------------------------------
# Test 7: Telemetry counters increment
# ---------------------------------------------------------------------------


async def test_fallback_telemetry_counters(manager_no_ai):
    """_tasks_regex_extracted and _events_regex_fallback should both increment
    when the regex fallback successfully extracts tasks."""
    # Initial state
    assert manager_no_ai._tasks_regex_extracted == 0
    assert manager_no_ai._events_regex_fallback == 0

    await manager_no_ai.process_event(
        make_email_event("Please review the attached document by end of week.", event_id="evt-telemetry-1")
    )

    assert manager_no_ai._events_regex_fallback >= 1, (
        "_events_regex_fallback should increment when regex extracts tasks"
    )
    assert manager_no_ai._tasks_regex_extracted >= 1, (
        "_tasks_regex_extracted should reflect the number of tasks extracted by regex"
    )

    # Also verify the counters appear in get_diagnostics()
    diag = manager_no_ai.get_diagnostics()
    et = diag["extraction_telemetry"]
    assert "events_regex_fallback" in et
    assert "tasks_regex_extracted" in et
    assert et["events_regex_fallback"] == manager_no_ai._events_regex_fallback
    assert et["tasks_regex_extracted"] == manager_no_ai._tasks_regex_extracted


# ---------------------------------------------------------------------------
# Test 8: Marketing filter still applies before regex extraction
# ---------------------------------------------------------------------------


async def test_fallback_respects_marketing_filter(manager_no_ai):
    """Marketing emails must be filtered out before regex runs — even with
    no AI engine, promotional content should never produce tasks."""
    await manager_no_ai.process_event({
        "id": "evt-marketing-regex",
        "type": "email.received",
        "payload": {
            "from_address": "noreply@marketing.example.com",
            "subject": "Big Sale This Weekend!",
            "body": (
                "Please hurry! Our biggest sale of the year starts TODAY. "
                "Don't miss out — act now before the deals expire! "
                "Unsubscribe from our list by clicking here."
            ),
        },
    })

    tasks = manager_no_ai.get_pending_tasks()
    assert len(tasks) == 0, (
        f"Marketing emails should be filtered before regex, but got tasks: "
        f"{[t['title'] for t in tasks]}"
    )
    # Verify the marketing skip counter incremented (not the regex fallback counter)
    assert manager_no_ai._events_skipped_marketing >= 1
    assert manager_no_ai._events_regex_fallback == 0
