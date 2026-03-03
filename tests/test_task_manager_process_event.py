"""
Test coverage for TaskManager.process_event — the primary entry point for
automatic task extraction from the event pipeline.

process_event is called from main.py for every event and has 5 distinct code
paths:
  1. Event type filter (only actionable types pass through)
  2. Text extraction (varies by event type)
  3. HTML stripping (emails often contain raw HTML)
  4. Marketing email filter (skip promotional content)
  5. AI engine invocation (extract action items from text)

Plus 3 early-return guards:
  - No AI engine configured
  - Non-actionable event type
  - Short/empty text content
"""

from unittest.mock import AsyncMock, patch

import pytest

from services.task_manager.manager import TaskManager


@pytest.fixture()
def mock_ai_engine():
    """AI engine mock that returns an empty action items list by default."""
    engine = AsyncMock()
    engine.extract_action_items = AsyncMock(return_value=[])
    return engine


@pytest.fixture()
def task_manager(db, mock_ai_engine):
    """TaskManager with a real DB and mocked AI engine."""
    return TaskManager(db=db, event_bus=None, ai_engine=mock_ai_engine)


# --- Test 1: Non-actionable event types are skipped ---


async def test_process_event_skips_non_actionable_types(task_manager, mock_ai_engine):
    """Events that are not emails, messages, or calendar events should be
    ignored entirely — the AI engine must never be called for them."""
    non_actionable_types = [
        "system.rule.triggered",
        "usermodel.signal_profile.updated",
        "prediction.generated",
        "notification.created",
        "task.created",
        "context.location.updated",
    ]
    for event_type in non_actionable_types:
        await task_manager.process_event({
            "id": f"evt-{event_type}",
            "type": event_type,
            "payload": {"body": "This is a substantial body text that is well over twenty characters."},
        })

    mock_ai_engine.extract_action_items.assert_not_called()


# --- Test 2: Short text is skipped ---


async def test_process_event_skips_short_text(task_manager, mock_ai_engine):
    """Messages shorter than 20 characters should not reach the AI engine."""
    await task_manager.process_event({
        "id": "evt-short",
        "type": "email.received",
        "payload": {"body": "ok", "from_address": "alice@example.com"},
    })

    mock_ai_engine.extract_action_items.assert_not_called()


# --- Test 3: Marketing emails are skipped ---


async def test_process_event_skips_marketing_email(task_manager, mock_ai_engine):
    """Marketing/promotional emails should be filtered out before AI processing."""
    await task_manager.process_event({
        "id": "evt-marketing",
        "type": "email.received",
        "payload": {
            "from_address": "noreply@marketing.example.com",
            "subject": "50% off everything this week!",
            "body": (
                "Shop our biggest sale of the year! "
                "Amazing deals on all categories. "
                "Click here to browse our selection. "
                "If you no longer wish to receive these emails, "
                "click here to Unsubscribe."
            ),
        },
    })

    mock_ai_engine.extract_action_items.assert_not_called()


# --- Test 4: HTML is stripped before AI processing ---


async def test_process_event_strips_html(task_manager, mock_ai_engine):
    """HTML email bodies should be converted to plain text before sending to
    the AI engine."""
    html_body = (
        "<html><body>"
        "<h1>Meeting Notes</h1>"
        "<p>Please review the attached report and send feedback by Friday.</p>"
        "</body></html>"
    )

    await task_manager.process_event({
        "id": "evt-html",
        "type": "email.received",
        "payload": {
            "from_address": "colleague@company.com",
            "body": html_body,
        },
    })

    mock_ai_engine.extract_action_items.assert_called_once()
    text_arg = mock_ai_engine.extract_action_items.call_args[0][0]

    # Plain text content should be preserved
    assert "Meeting Notes" in text_arg
    assert "Please review" in text_arg
    assert "send feedback by Friday" in text_arg

    # HTML tags should be stripped
    assert "<html>" not in text_arg
    assert "<body>" not in text_arg
    assert "<h1>" not in text_arg
    assert "<p>" not in text_arg
    assert "</p>" not in text_arg


# --- Test 5: Well-formed email.received calls AI and ingests tasks ---


async def test_process_event_email_received_calls_ai(task_manager, mock_ai_engine):
    """A substantial, non-marketing email.received event should trigger AI
    extraction and task ingestion."""
    extracted = [{"title": "Review report", "due_hint": "Friday", "priority": "normal"}]
    mock_ai_engine.extract_action_items.return_value = extracted

    await task_manager.process_event({
        "id": "evt-email-recv",
        "type": "email.received",
        "payload": {
            "from_address": "boss@company.com",
            "subject": "Q3 Report",
            "body": "Hi, please review the Q3 report and send me your feedback by end of week.",
        },
    })

    # AI engine was called
    mock_ai_engine.extract_action_items.assert_called_once()
    call_args = mock_ai_engine.extract_action_items.call_args
    assert call_args[0][1] == "email.received"  # event_type passed as 2nd arg

    # Task was ingested into the database
    tasks = task_manager.get_pending_tasks()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Review report"
    assert tasks[0]["source"] == "ai_extracted"
    assert tasks[0]["source_event_id"] == "evt-email-recv"


# --- Test 6: email.sent events are processed ---


async def test_process_event_email_sent(task_manager, mock_ai_engine):
    """Sent emails should also reach the AI engine — they may contain
    completion signals like 'I sent the report yesterday'."""
    await task_manager.process_event({
        "id": "evt-email-sent",
        "type": "email.sent",
        "payload": {
            "from_address": "me@company.com",
            "body": "Hi Alice, I have completed the quarterly financial review and attached the report.",
        },
    })

    mock_ai_engine.extract_action_items.assert_called_once()
    assert mock_ai_engine.extract_action_items.call_args[0][1] == "email.sent"


# --- Test 7: message.received and message.sent are processed ---


async def test_process_event_message_types(task_manager, mock_ai_engine):
    """Both inbound and outbound messages should reach the AI engine."""
    await task_manager.process_event({
        "id": "evt-msg-recv",
        "type": "message.received",
        "payload": {"body": "Can you send me the project timeline by tomorrow morning?"},
    })

    await task_manager.process_event({
        "id": "evt-msg-sent",
        "type": "message.sent",
        "payload": {"body": "Sure, I will prepare the project timeline and send it tonight."},
    })

    assert mock_ai_engine.extract_action_items.call_count == 2

    # Verify both event types were passed correctly
    calls = mock_ai_engine.extract_action_items.call_args_list
    assert calls[0][0][1] == "message.received"
    assert calls[1][0][1] == "message.sent"


# --- Test 8: calendar.event.created uses description field ---


async def test_process_event_calendar_event(task_manager, mock_ai_engine):
    """Calendar events should extract action items from the description field."""
    await task_manager.process_event({
        "id": "evt-cal",
        "type": "calendar.event.created",
        "payload": {
            "summary": "Sprint Planning",
            "description": "Agenda: Review backlog items, assign story points, identify blockers for next sprint.",
        },
    })

    mock_ai_engine.extract_action_items.assert_called_once()
    text_arg = mock_ai_engine.extract_action_items.call_args[0][0]
    assert "Review backlog items" in text_arg
    assert "assign story points" in text_arg


# --- Test 9: AI engine failure is handled gracefully (fail-open) ---


async def test_process_event_ai_engine_failure(task_manager, mock_ai_engine):
    """If the AI engine raises an exception, process_event must NOT re-raise.
    This ensures the event pipeline continues processing subsequent events."""
    mock_ai_engine.extract_action_items.side_effect = RuntimeError("Model unavailable")

    # Should NOT raise — fail-open behavior
    await task_manager.process_event({
        "id": "evt-fail",
        "type": "email.received",
        "payload": {
            "from_address": "colleague@company.com",
            "body": "Please complete the compliance training module by end of month.",
        },
    })

    mock_ai_engine.extract_action_items.assert_called_once()


# --- Test 10: No AI engine configured returns immediately ---


async def test_process_event_no_ai_engine(db):
    """When ai_engine is None, process_event should return immediately
    without error (the warning log path)."""
    manager = TaskManager(db=db, event_bus=None, ai_engine=None)

    # Should NOT raise — early return when ai_engine is None
    await manager.process_event({
        "id": "evt-no-ai",
        "type": "email.received",
        "payload": {
            "from_address": "alice@example.com",
            "body": "This is a real email with enough text to pass the length filter easily.",
        },
    })

    # No crash, no tasks created
    tasks = manager.get_pending_tasks()
    assert len(tasks) == 0
