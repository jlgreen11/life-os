"""
Tests for calendar.event.updated task extraction and AI engine skip counter.

Covers:
- calendar.event.updated events are processed for task extraction
- Duplicate tasks from created+updated for the same calendar event are deduplicated
- AI engine skip counter logs at events 1, 100, 200
- Skip counter log messages include the running count
"""

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_manager.manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_calendar_event(event_type: str, description: str, summary: str = "Meeting") -> dict:
    """Build a minimal calendar event dict for testing."""
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "payload": {
            "description": description,
            "summary": summary,
        },
        "metadata": {},
    }


def _mock_ai_engine(tasks_to_return: list[dict] | None = None):
    """Create a mock AI engine that returns the given tasks from extract_action_items."""
    engine = MagicMock()
    engine.extract_action_items = AsyncMock(
        return_value=tasks_to_return or []
    )
    return engine


# ---------------------------------------------------------------------------
# Test: calendar.event.updated processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calendar_event_updated_triggers_task_extraction(db):
    """calendar.event.updated events should be processed for task extraction."""
    ai_engine = _mock_ai_engine([
        {"title": "Prepare slides for presentation", "priority": "normal"},
    ])
    manager = TaskManager(db=db, ai_engine=ai_engine)

    event = _make_calendar_event(
        "calendar.event.updated",
        description="Prepare slides for presentation before Thursday",
        summary="Q1 Review",
    )

    await manager.process_event(event)

    # AI engine should have been called with combined description + summary text
    ai_engine.extract_action_items.assert_called_once()
    call_text = ai_engine.extract_action_items.call_args[0][0]
    assert "Prepare slides" in call_text
    assert "Q1 Review" in call_text

    # Task should have been created in the database
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE title = 'Prepare slides for presentation'"
        ).fetchone()
        assert row is not None
        assert row["source"] == "ai_extracted"
        assert row["source_event_id"] == event["id"]


@pytest.mark.asyncio
async def test_calendar_event_created_also_includes_summary(db):
    """calendar.event.created should also combine description + summary text."""
    ai_engine = _mock_ai_engine([
        {"title": "Review Q1 budget", "priority": "high"},
    ])
    manager = TaskManager(db=db, ai_engine=ai_engine)

    event = _make_calendar_event(
        "calendar.event.created",
        description="Review Q1 budget before the meeting",
        summary="Budget Review",
    )

    await manager.process_event(event)

    call_text = ai_engine.extract_action_items.call_args[0][0]
    assert "Review Q1 budget" in call_text
    assert "Budget Review" in call_text


@pytest.mark.asyncio
async def test_calendar_event_updated_short_text_skipped(db):
    """calendar.event.updated with text < 20 chars should be skipped."""
    ai_engine = _mock_ai_engine()
    manager = TaskManager(db=db, ai_engine=ai_engine)

    event = _make_calendar_event(
        "calendar.event.updated",
        description="",
        summary="Hi",
    )

    await manager.process_event(event)

    # AI engine should NOT have been called for trivial text
    ai_engine.extract_action_items.assert_not_called()


@pytest.mark.asyncio
async def test_calendar_event_updated_no_tasks_found(db):
    """AI returning empty list for calendar.event.updated is a no-op."""
    ai_engine = _mock_ai_engine([])  # No tasks extracted
    manager = TaskManager(db=db, ai_engine=ai_engine)

    event = _make_calendar_event(
        "calendar.event.updated",
        description="Just a casual check-in, nothing actionable here at all",
        summary="Team Sync",
    )

    await manager.process_event(event)

    ai_engine.extract_action_items.assert_called_once()

    # No tasks should exist
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 0


# ---------------------------------------------------------------------------
# Test: Deduplication between created and updated events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_task_from_created_and_updated_is_deduplicated(db):
    """A task extracted from calendar.event.created should not be re-created
    when the same title appears in a subsequent calendar.event.updated event."""
    ai_engine = _mock_ai_engine([
        {"title": "Prepare slides for presentation", "priority": "normal"},
    ])
    manager = TaskManager(db=db, ai_engine=ai_engine)

    # First: calendar.event.created extracts a task
    created_event = _make_calendar_event(
        "calendar.event.created",
        description="Prepare slides for presentation",
        summary="Q1 Review",
    )
    await manager.process_event(created_event)

    # Verify one task was created
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 1

    # Second: calendar.event.updated with same action item
    updated_event = _make_calendar_event(
        "calendar.event.updated",
        description="Prepare slides for presentation (updated details)",
        summary="Q1 Review",
    )
    await manager.process_event(updated_event)

    # _is_duplicate_task should have prevented a second task
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 1  # Still only one task


@pytest.mark.asyncio
async def test_different_task_from_updated_event_is_created(db):
    """A genuinely new task from calendar.event.updated should be created
    even if a different task was already extracted from the created event."""
    manager = TaskManager(db=db, ai_engine=_mock_ai_engine([
        {"title": "Review budget spreadsheet", "priority": "normal"},
    ]))

    created_event = _make_calendar_event(
        "calendar.event.created",
        description="Review budget spreadsheet",
        summary="Planning",
    )
    await manager.process_event(created_event)

    # Now the updated event has a different task
    manager.ai_engine = _mock_ai_engine([
        {"title": "Send agenda to participants", "priority": "normal"},
    ])

    updated_event = _make_calendar_event(
        "calendar.event.updated",
        description="Send agenda to participants before meeting",
        summary="Planning",
    )
    await manager.process_event(updated_event)

    # Both tasks should exist — they have different titles
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 2


# ---------------------------------------------------------------------------
# Test: AI engine skip counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ai_engine_skip_counter_logs_on_first_event(db, caplog):
    """The skip counter should log a warning on the very first skipped event."""
    manager = TaskManager(db=db, ai_engine=None)

    event = _make_calendar_event(
        "email.received",
        description="Please send me the report by Friday afternoon",
        summary="",
    )
    # Override type since helper defaults to calendar event
    event["type"] = "email.received"
    event["payload"]["body"] = "Please send me the report by Friday afternoon"

    with caplog.at_level(logging.WARNING, logger="services.task_manager.manager"):
        await manager.process_event(event)

    assert manager._ai_engine_skip_count == 1
    assert any("1 events skipped so far" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_ai_engine_skip_counter_logs_at_100(db, caplog):
    """The skip counter should log again at every 100th event."""
    manager = TaskManager(db=db, ai_engine=None)

    event = {
        "id": "test-event",
        "type": "email.received",
        "payload": {"body": "This is a sufficiently long message for processing"},
        "metadata": {},
    }

    with caplog.at_level(logging.WARNING, logger="services.task_manager.manager"):
        for _ in range(100):
            await manager.process_event(event)

    assert manager._ai_engine_skip_count == 100

    # Should have two log messages: at event 1 and event 100
    warning_messages = [
        msg for msg in caplog.messages
        if "AI engine not available" in msg
    ]
    assert len(warning_messages) == 2
    assert "1 events skipped" in warning_messages[0]
    assert "100 events skipped" in warning_messages[1]


@pytest.mark.asyncio
async def test_ai_engine_skip_counter_logs_at_200(db, caplog):
    """The skip counter should log at event 200 as well."""
    manager = TaskManager(db=db, ai_engine=None)

    event = {
        "id": "test-event",
        "type": "message.received",
        "payload": {"body": "This is a sufficiently long message for testing"},
        "metadata": {},
    }

    with caplog.at_level(logging.WARNING, logger="services.task_manager.manager"):
        for _ in range(200):
            await manager.process_event(event)

    assert manager._ai_engine_skip_count == 200

    warning_messages = [
        msg for msg in caplog.messages
        if "AI engine not available" in msg
    ]
    # Should have 3 log messages: at event 1, 100, and 200
    assert len(warning_messages) == 3
    assert "200 events skipped" in warning_messages[2]


@pytest.mark.asyncio
async def test_ai_engine_skip_counter_silent_between_milestones(db, caplog):
    """Events between milestones (e.g., 2-99) should NOT produce log output."""
    manager = TaskManager(db=db, ai_engine=None)

    event = {
        "id": "test-event",
        "type": "email.received",
        "payload": {"body": "A message long enough to pass the length check easily"},
        "metadata": {},
    }

    with caplog.at_level(logging.WARNING, logger="services.task_manager.manager"):
        # Process 50 events (only event 1 should log)
        for _ in range(50):
            await manager.process_event(event)

    assert manager._ai_engine_skip_count == 50

    warning_messages = [
        msg for msg in caplog.messages
        if "AI engine not available" in msg
    ]
    # Only the first event should have logged
    assert len(warning_messages) == 1
    assert "1 events skipped" in warning_messages[0]
