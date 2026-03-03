"""
Tests for the AI-powered task extraction pipeline.

This module verifies that the TaskManager correctly processes events,
calls the AI engine to extract action items, and persists the extracted
tasks with full provenance tracking.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.task_manager.manager import TaskManager


# -------------------------------------------------------------------
# Task Extraction Integration Tests
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_event_extracts_tasks_from_email(db, event_store):
    """TaskManager should extract and persist tasks from actionable emails."""
    # Mock AI engine that returns action items
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[
        {"title": "Review PR #123", "due_hint": "by Friday", "priority": "high"},
        {"title": "Update documentation", "due_hint": None, "priority": "normal"},
    ])

    manager = TaskManager(db, ai_engine=mock_ai)

    # Incoming email with action items
    event = {
        "id": "evt-001",
        "type": "email.received",
        "payload": {
            "subject": "Code review needed",
            "body": "Can you review PR #123 by Friday? Also please update the documentation.",
            "from_address": "alice@example.com",
        },
    }

    await manager.process_event(event)

    # Verify AI engine was called with the email body
    mock_ai.extract_action_items.assert_called_once_with(
        "Can you review PR #123 by Friday? Also please update the documentation.",
        "email.received"
    )

    # Verify tasks were persisted to the database
    with db.get_connection("state") as conn:
        tasks = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        assert len(tasks) == 2

        # First task: high-priority code review
        task1 = dict(tasks[0])
        assert task1["title"] == "Review PR #123"
        assert task1["source"] == "ai_extracted"
        assert task1["source_event_id"] == "evt-001"
        assert task1["priority"] == "high"
        assert task1["due_date"] == "by Friday"
        assert task1["status"] == "pending"

        # Second task: normal-priority documentation
        task2 = dict(tasks[1])
        assert task2["title"] == "Update documentation"
        assert task2["source"] == "ai_extracted"
        assert task2["source_event_id"] == "evt-001"
        assert task2["priority"] == "normal"
        assert task2["due_date"] is None


@pytest.mark.asyncio
async def test_process_event_extracts_tasks_from_message(db):
    """TaskManager should extract tasks from direct messages."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[
        {"title": "Send expense report", "due_hint": "today", "priority": "high"},
    ])

    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "msg-001",
        "type": "message.received",
        "payload": {
            "body": "Hey, can you send me the expense report today? Thanks!",
            "from": "Bob",
        },
    }

    await manager.process_event(event)

    # Verify AI extraction was called
    mock_ai.extract_action_items.assert_called_once_with(
        "Hey, can you send me the expense report today? Thanks!",
        "message.received"
    )

    # Verify task was created
    with db.get_connection("state") as conn:
        tasks = conn.execute("SELECT * FROM tasks").fetchall()
        assert len(tasks) == 1
        task = dict(tasks[0])
        assert task["title"] == "Send expense report"
        assert task["due_date"] == "today"


@pytest.mark.asyncio
async def test_process_event_extracts_tasks_from_calendar_event(db):
    """TaskManager should extract tasks from calendar event descriptions."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[
        {"title": "Prepare slides", "due_hint": None, "priority": "normal"},
        {"title": "Book conference room", "due_hint": None, "priority": "normal"},
    ])

    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "cal-001",
        "type": "calendar.event.created",
        "payload": {
            "summary": "Team meeting",
            "description": "Agenda: Q4 planning. TODO: Prepare slides. Book conference room.",
        },
    }

    await manager.process_event(event)

    # Verify AI extraction was called with description + summary
    mock_ai.extract_action_items.assert_called_once_with(
        "Agenda: Q4 planning. TODO: Prepare slides. Book conference room. Team meeting",
        "calendar.event.created"
    )

    # Verify both tasks were created
    with db.get_connection("state") as conn:
        tasks = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        assert len(tasks) == 2


@pytest.mark.asyncio
async def test_process_event_skips_non_actionable_types(db):
    """TaskManager should ignore system events and predictions."""
    mock_ai = AsyncMock()
    manager = TaskManager(db, ai_engine=mock_ai)

    # System events should not trigger extraction
    non_actionable_types = [
        "usermodel.prediction.generated",
        "notification.created",
        "system.connector.sync_complete",
        "usermodel.signal_profile.updated",
    ]

    for event_type in non_actionable_types:
        event = {
            "id": f"evt-{event_type}",
            "type": event_type,
            "payload": {"text": "Some content here"},
        }
        await manager.process_event(event)

    # AI engine should never have been called
    mock_ai.extract_action_items.assert_not_called()

    # No tasks should be created
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 0


@pytest.mark.asyncio
async def test_process_event_skips_short_messages(db):
    """TaskManager should skip trivial messages to avoid wasting LLM cycles."""
    mock_ai = AsyncMock()
    manager = TaskManager(db, ai_engine=mock_ai)

    # Short messages like "ok", "thanks", etc. should be ignored
    short_messages = ["ok", "thanks", "got it", "👍", "lol", "k"]

    for msg in short_messages:
        event = {
            "id": f"msg-{msg}",
            "type": "message.received",
            "payload": {"body": msg},
        }
        await manager.process_event(event)

    # AI engine should never be called for trivial messages
    mock_ai.extract_action_items.assert_not_called()


@pytest.mark.asyncio
async def test_process_event_requires_minimum_20_chars(db):
    """TaskManager should require at least 20 characters to trigger extraction."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[])
    manager = TaskManager(db, ai_engine=mock_ai)

    # 19 characters - should be skipped
    short_event = {
        "id": "short",
        "type": "email.received",
        "payload": {"body": "Call me tomorrow!"},  # 17 chars
    }
    await manager.process_event(short_event)
    assert mock_ai.extract_action_items.call_count == 0

    # 20+ characters - should be processed
    long_event = {
        "id": "long",
        "type": "email.received",
        "payload": {"body": "Can you call me tomorrow at 3pm?"},  # 33 chars
    }
    await manager.process_event(long_event)
    assert mock_ai.extract_action_items.call_count == 1


@pytest.mark.asyncio
async def test_process_event_uses_body_over_snippet(db):
    """TaskManager should prioritize full email body over snippet."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[])
    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-001",
        "type": "email.received",
        "payload": {
            "subject": "Quick question",
            "snippet": "Can you review the...",
            "body": "Can you review the contract and send it back by Friday?",
        },
    }

    await manager.process_event(event)

    # Should extract from body, not snippet
    call_args = mock_ai.extract_action_items.call_args[0]
    assert call_args[0] == "Can you review the contract and send it back by Friday?"


@pytest.mark.asyncio
async def test_process_event_falls_back_to_snippet(db):
    """TaskManager should use snippet if body is missing."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[])
    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-002",
        "type": "email.received",
        "payload": {
            "subject": "Quick question",
            "snippet": "Can you review the contract by Friday?",
            # No body field
        },
    }

    await manager.process_event(event)

    # Should extract from snippet
    call_args = mock_ai.extract_action_items.call_args[0]
    assert call_args[0] == "Can you review the contract by Friday?"


@pytest.mark.asyncio
async def test_process_event_falls_back_to_subject(db):
    """TaskManager should use subject if body and snippet are missing."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[])
    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-003",
        "type": "email.received",
        "payload": {
            "subject": "Review contract by Friday please",
            # No body or snippet
        },
    }

    await manager.process_event(event)

    # Should extract from subject
    call_args = mock_ai.extract_action_items.call_args[0]
    assert call_args[0] == "Review contract by Friday please"


@pytest.mark.asyncio
async def test_process_event_skips_when_ai_engine_missing(db):
    """TaskManager should gracefully skip extraction if AI engine not wired."""
    # No AI engine provided
    manager = TaskManager(db, ai_engine=None)

    event = {
        "id": "email-004",
        "type": "email.received",
        "payload": {
            "body": "Can you review this contract by Friday?",
        },
    }

    # Should not crash, just skip extraction
    await manager.process_event(event)

    # No tasks should be created
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 0


@pytest.mark.asyncio
async def test_process_event_handles_ai_engine_failure(db):
    """TaskManager should gracefully handle AI engine failures."""
    mock_ai = AsyncMock()
    # AI engine raises an exception
    mock_ai.extract_action_items = AsyncMock(side_effect=Exception("Model timeout"))
    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-005",
        "type": "email.received",
        "payload": {
            "body": "Can you review this contract by Friday?",
        },
    }

    # Should not crash, just log and continue
    await manager.process_event(event)

    # No tasks should be created (extraction failed)
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 0


@pytest.mark.asyncio
async def test_process_event_handles_empty_extraction_result(db):
    """TaskManager should handle case where AI finds no action items."""
    mock_ai = AsyncMock()
    # AI returns empty list (no action items found)
    mock_ai.extract_action_items = AsyncMock(return_value=[])
    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-006",
        "type": "email.received",
        "payload": {
            "body": "Thanks for your help yesterday! Everything went great.",
        },
    }

    await manager.process_event(event)

    # AI was called but found nothing
    mock_ai.extract_action_items.assert_called_once()

    # No tasks should be created
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 0


@pytest.mark.asyncio
async def test_process_event_preserves_event_provenance(db):
    """TaskManager should link tasks back to source events for full traceability."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(return_value=[
        {"title": "Send report", "due_hint": "Friday", "priority": "high"},
    ])

    manager = TaskManager(db, ai_engine=mock_ai)

    event = {
        "id": "email-unique-id-12345",
        "type": "email.received",
        "payload": {
            "body": "Please send me the quarterly report by Friday.",
            "from_address": "manager@example.com",
        },
    }

    await manager.process_event(event)

    # Verify task has full provenance
    with db.get_connection("state") as conn:
        task = conn.execute("SELECT * FROM tasks").fetchone()
        assert task["source"] == "ai_extracted"
        assert task["source_event_id"] == "email-unique-id-12345"


@pytest.mark.asyncio
async def test_process_event_handles_multiple_extraction_calls(db):
    """TaskManager should correctly handle multiple events in sequence."""
    mock_ai = AsyncMock()
    mock_ai.extract_action_items = AsyncMock(side_effect=[
        [{"title": "Task 1", "due_hint": None, "priority": "normal"}],
        [{"title": "Task 2", "due_hint": None, "priority": "normal"}],
        [],  # Third event has no action items
    ])

    manager = TaskManager(db, ai_engine=mock_ai)

    # Process three events
    for i in range(3):
        event = {
            "id": f"evt-{i}",
            "type": "email.received",
            "payload": {"body": f"Email content {i} with enough characters to process"},
        }
        await manager.process_event(event)

    # Verify AI was called 3 times
    assert mock_ai.extract_action_items.call_count == 3

    # Verify only 2 tasks were created (third event had no action items)
    with db.get_connection("state") as conn:
        count = conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()["cnt"]
        assert count == 2
