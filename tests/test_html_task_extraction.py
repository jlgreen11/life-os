"""
Tests for HTML stripping in task extraction pipeline.

This test suite validates that the TaskManager correctly strips HTML markup
from email bodies before passing them to the AI engine for action item extraction.

Prior to this fix, 68K+ emails were processed but 0 tasks were created because
the AI engine was receiving raw HTML instead of plain text.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from services.task_manager.manager import TaskManager, HTMLStripper


class TestHTMLStripper:
    """Test the HTMLStripper helper class used to clean email bodies."""

    def test_strip_simple_html(self):
        """Should remove basic HTML tags and return plain text."""
        html = "<p>Hello <b>world</b>!</p>"
        stripper = HTMLStripper()
        stripper.feed(html)
        # .strip() at the end removes trailing newline
        assert stripper.get_text() == "Hello world!"

    def test_preserve_paragraph_breaks(self):
        """Should preserve paragraph structure with newlines."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        assert "First paragraph" in text
        assert "Second paragraph" in text
        assert "\n" in text  # Should have line breaks between paragraphs

    def test_strip_nested_tags(self):
        """Should handle deeply nested HTML structures."""
        html = """
        <div>
            <p>Outer <span>with <strong>nested <em>tags</em></strong></span></p>
        </div>
        """
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        assert "Outer with nested tags" in text
        assert "<" not in text  # No tags should remain

    def test_handle_html_entities(self):
        """Should decode HTML entities like &amp; and &nbsp;."""
        html = "<p>Coffee &amp; Tea&nbsp;Co.</p>"
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        assert "&amp;" not in text
        assert "&nbsp;" not in text
        assert "Coffee & Tea" in text

    def test_collapse_multiple_newlines(self):
        """Should collapse excessive whitespace into clean paragraphs."""
        html = "<div><br><br><br><p>Text</p><br><br></div>"
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in text
        assert "Text" in text

    def test_collapse_multiple_spaces(self):
        """Should normalize excessive spacing within text."""
        html = "<p>Too     many     spaces</p>"
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()
        assert "  " not in text  # No double spaces
        assert "Too many spaces" in text

    def test_handle_empty_html(self):
        """Should handle empty HTML gracefully."""
        html = "<div></div>"
        stripper = HTMLStripper()
        stripper.feed(html)
        assert stripper.get_text() == ""

    def test_handle_malformed_html(self):
        """Should handle unclosed tags without crashing."""
        html = "<p>Unclosed paragraph<div>Unclosed div"
        stripper = HTMLStripper()
        # Should not raise an exception
        stripper.feed(html)
        text = stripper.get_text()
        assert "Unclosed paragraph" in text
        assert "Unclosed div" in text

    def test_real_world_email_html(self):
        """Should extract readable text from typical email HTML."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Email Title</title></head>
        <body>
            <div style="font-family: Arial">
                <p>Hi Jeremy,</p>
                <p>Can you please send me the <b>Q4 report</b> by Friday?</p>
                <p>Thanks,<br>Alice</p>
            </div>
        </body>
        </html>
        """
        stripper = HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()

        # Should extract the actual message content
        assert "Hi Jeremy" in text
        assert "Q4 report" in text
        assert "by Friday" in text
        assert "Alice" in text

        # Should not have HTML artifacts
        assert "<" not in text
        assert "DOCTYPE" not in text
        assert "style=" not in text


class TestTaskManagerHTMLProcessing:
    """Test TaskManager's integration of HTML stripping in the extraction pipeline."""

    @pytest.mark.asyncio
    async def test_strips_html_from_email_body(self):
        """Should strip HTML from email.received events before AI extraction."""
        # Setup mock dependencies
        db = Mock()
        event_bus = Mock()
        event_bus.is_connected = True
        ai_engine = AsyncMock()
        # Return empty list to avoid task creation path
        ai_engine.extract_action_items = AsyncMock(return_value=[])

        task_manager = TaskManager(db, event_bus, ai_engine)

        # Simulate an email event with HTML body
        event = {
            "id": "evt_123",
            "type": "email.received",
            "payload": {
                "subject": "Report request",
                "body": "<html><body><p>Can you send the <b>Q4 report</b> by Friday?</p></body></html>",
                "from_address": "alice@example.com"
            },
            "timestamp": "2026-02-15T10:00:00Z"
        }

        await task_manager.process_event(event)

        # Verify AI engine was called with plain text (no HTML tags)
        ai_engine.extract_action_items.assert_called_once()
        call_args = ai_engine.extract_action_items.call_args
        extracted_text = call_args[0][0]  # First positional arg

        assert "<html>" not in extracted_text
        assert "<body>" not in extracted_text
        assert "<p>" not in extracted_text
        assert "Q4 report" in extracted_text
        assert "by Friday" in extracted_text

    @pytest.mark.asyncio
    async def test_preserves_plain_text_emails(self):
        """Should pass through plain text emails without modification."""
        db = Mock()
        ai_engine = AsyncMock()
        ai_engine.extract_action_items = AsyncMock(return_value=[])

        task_manager = TaskManager(db, None, ai_engine)

        # Plain text email (no HTML)
        event = {
            "id": "evt_456",
            "type": "email.received",
            "payload": {
                "subject": "Plain text",
                "body": "Just a simple message with no HTML tags."
            }
        }

        await task_manager.process_event(event)

        # Verify the text was passed through unchanged
        call_args = ai_engine.extract_action_items.call_args
        extracted_text = call_args[0][0]
        assert extracted_text == "Just a simple message with no HTML tags."

    @pytest.mark.asyncio
    async def test_falls_back_to_snippet_if_no_body(self):
        """Should use snippet when body is missing, stripping HTML if needed."""
        db = Mock()
        ai_engine = AsyncMock()
        ai_engine.extract_action_items = AsyncMock(return_value=[])

        task_manager = TaskManager(db, None, ai_engine)

        event = {
            "id": "evt_789",
            "type": "email.received",
            "payload": {
                "subject": "Subject line",
                # Snippet needs to be long enough (20+ chars) to trigger extraction
                "snippet": "<p>Short preview text that is long enough</p>"
            }
        }

        await task_manager.process_event(event)

        # Verify AI was called (snippet was long enough)
        ai_engine.extract_action_items.assert_called_once()
        call_args = ai_engine.extract_action_items.call_args
        extracted_text = call_args[0][0]
        assert "<p>" not in extracted_text
        assert "Short preview text" in extracted_text

    @pytest.mark.asyncio
    async def test_skips_extraction_for_short_text(self):
        """Should not waste LLM cycles on trivial messages (< 20 chars)."""
        db = Mock()
        ai_engine = AsyncMock()

        task_manager = TaskManager(db, None, ai_engine)

        # Very short message
        event = {
            "id": "evt_short",
            "type": "email.received",
            "payload": {"body": "ok"}
        }

        await task_manager.process_event(event)

        # Should NOT have called the AI engine
        ai_engine.extract_action_items.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_html_parsing_failure_gracefully(self):
        """Should fall back to original text if HTML parsing fails."""
        db = Mock()
        ai_engine = AsyncMock()
        ai_engine.extract_action_items = AsyncMock(return_value=[])

        task_manager = TaskManager(db, None, ai_engine)

        # Extremely malformed HTML that might cause parser issues
        event = {
            "id": "evt_malformed",
            "type": "email.received",
            "payload": {
                "body": "< This is not valid HTML but contains angle brackets >"
            }
        }

        # Should not crash
        await task_manager.process_event(event)

        # AI engine should still be called (fail-open behavior)
        ai_engine.extract_action_items.assert_called_once()

    @pytest.mark.asyncio
    async def test_strips_html_from_calendar_descriptions(self):
        """Should strip HTML from calendar event descriptions too."""
        db = Mock()
        ai_engine = AsyncMock()
        ai_engine.extract_action_items = AsyncMock(return_value=[])

        task_manager = TaskManager(db, None, ai_engine)

        event = {
            "id": "evt_cal",
            "type": "calendar.event.created",
            "payload": {
                "description": "<p>Meeting agenda:<br>1. Review Q4<br>2. Plan Q1</p>"
            }
        }

        await task_manager.process_event(event)

        call_args = ai_engine.extract_action_items.call_args
        extracted_text = call_args[0][0]
        assert "<p>" not in extracted_text
        assert "<br>" not in extracted_text
        assert "Review Q4" in extracted_text
        assert "Plan Q1" in extracted_text

    @pytest.mark.asyncio
    async def test_creates_tasks_from_extracted_items(self):
        """Should create tasks when AI extracts action items from clean text."""
        # Setup database mock
        db = Mock()
        db.get_connection = Mock(return_value=Mock(__enter__=Mock(return_value=Mock(execute=Mock())), __exit__=Mock()))

        ai_engine = AsyncMock()
        ai_engine.extract_action_items = AsyncMock(return_value=[
            {"title": "Review Q4 financials", "priority": "high", "due_hint": "2026-02-20"},
            {"title": "Schedule team meeting", "priority": "normal"}
        ])

        task_manager = TaskManager(db, None, ai_engine)

        event = {
            "id": "evt_tasks",
            "type": "email.received",
            "payload": {
                "body": "<p>Please review Q4 financials by Feb 20 and schedule a team meeting.</p>"
            }
        }

        await task_manager.process_event(event)

        # Verify tasks were created (check database calls)
        assert db.get_connection.call_count >= 2  # At least 2 task inserts
