"""
Life OS — Rule Notification Content Tests

Tests for _build_notification_content, which extracts meaningful titles and
bodies from event payloads for rule-triggered notifications.  Previously the
notify action hard-coded the rule name as the title and only read a ``snippet``
payload field for the body, leaving most notifications empty.

Test approach:
    - Uses the same _make_life_os helper pattern as test_rule_actions_extended.py
    - Each test constructs a specific event type (email, calendar, finance,
      message) and verifies the title/body contain the expected content.
    - Also tests the full round-trip through _execute_rule_action to verify
      the notification manager receives the enriched content.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_life_os(db, event_store, user_model_store):
    """
    Build a minimal LifeOS shell with notification_manager and task_manager
    stubs.  Follows the same pattern as test_rule_actions_extended.py.
    """
    from main import LifeOS

    lo = object.__new__(LifeOS)
    lo.db = db
    lo.event_store = event_store
    lo.user_model_store = user_model_store
    lo.connector_map = {}

    nm = MagicMock()
    nm.create_notification = AsyncMock()
    lo.notification_manager = nm

    tm = MagicMock()
    tm.create_task = AsyncMock()
    lo.task_manager = tm

    # Wire real methods that the notify action path uses
    lo._infer_domain_from_event_type = LifeOS._infer_domain_from_event_type.__get__(
        lo, LifeOS
    )
    lo._build_notification_content = LifeOS._build_notification_content.__get__(
        lo, LifeOS
    )
    return lo


def _make_event(event_type, **payload_fields):
    """Create an event dict with a specific type and payload."""
    return {
        "id": "evt-test",
        "type": event_type,
        "source": "test_source",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload_fields,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Unit tests for _build_notification_content
# ---------------------------------------------------------------------------

class TestBuildNotificationContentEmail:
    """Email events should use subject as title and body_plain/snippet as body."""

    def test_email_title_from_subject(self, db, event_store, user_model_store):
        """Title is extracted from payload.subject for email events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Invoice from Acme Corp",
            body_plain="Please find attached...",
            from_address="billing@acme.com",
        )
        action = {"rule_name": "Important Email"}

        title, body = lo._build_notification_content(event, action)

        assert title == "Invoice from Acme Corp"

    def test_email_body_from_snippet(self, db, event_store, user_model_store):
        """Body prefers snippet when available."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Test",
            snippet="This is a short preview",
            body_plain="This is the full plain text body",
            from_address="alice@example.com",
        )
        action = {"rule_name": "Test Rule"}

        title, body = lo._build_notification_content(event, action)

        # snippet is tried first, so it should appear in the body (after From: prefix)
        assert "This is a short preview" in body

    def test_email_body_from_body_plain(self, db, event_store, user_model_store):
        """Body falls back to body_plain when snippet is absent."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Meeting notes",
            body_plain="Here are the notes from today's meeting...",
            from_address="bob@example.com",
        )
        action = {"rule_name": "Test Rule"}

        title, body = lo._build_notification_content(event, action)

        assert "Here are the notes from today's meeting" in body

    def test_email_from_address_prepended(self, db, event_store, user_model_store):
        """Email notifications include 'From: <address>' at the start of the body."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Hello",
            body_plain="Hi there!",
            from_address="sender@example.com",
        )
        action = {"rule_name": "Test Rule"}

        title, body = lo._build_notification_content(event, action)

        assert body.startswith("From: sender@example.com")

    def test_email_from_field_fallback(self, db, event_store, user_model_store):
        """Falls back to 'from' field when 'from_address' is absent."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Hello",
            body_plain="Hi there!",
        )
        # Use 'from' instead of 'from_address'
        event["payload"]["from"] = "fallback@example.com"
        action = {"rule_name": "Test Rule"}

        title, body = lo._build_notification_content(event, action)

        assert "From: fallback@example.com" in body


class TestBuildNotificationContentCalendar:
    """Calendar events should use summary as title."""

    def test_calendar_title_from_summary(self, db, event_store, user_model_store):
        """Title is extracted from payload.summary for calendar events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "calendar.event.created",
            summary="Team Standup",
            location="Room 42",
            start_time="2026-03-02T09:00:00",
        )
        action = {"rule_name": "Calendar Alert"}

        title, body = lo._build_notification_content(event, action)

        assert title == "Team Standup"

    def test_calendar_body_from_description(self, db, event_store, user_model_store):
        """Body is extracted from payload.description for calendar events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "calendar.event.created",
            summary="Sprint Planning",
            description="Review backlog items and plan sprint goals",
        )
        action = {"rule_name": "Calendar Alert"}

        title, body = lo._build_notification_content(event, action)

        assert "Review backlog items" in body


class TestBuildNotificationContentFinance:
    """Finance events should use merchant_name as title with amount."""

    def test_finance_title_from_merchant_name(self, db, event_store, user_model_store):
        """Title shows merchant_name for finance events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "finance.transaction",
            merchant_name="Whole Foods",
            amount=87.50,
            description="Grocery purchase",
        )
        action = {"rule_name": "Large Transaction"}

        title, body = lo._build_notification_content(event, action)

        assert "Whole Foods" in title

    def test_finance_title_includes_amount(self, db, event_store, user_model_store):
        """Finance title includes the dollar amount when both merchant and amount are present."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "finance.transaction",
            merchant_name="Amazon",
            amount=142.99,
            description="Online purchase",
        )
        action = {"rule_name": "Large Transaction"}

        title, body = lo._build_notification_content(event, action)

        assert title == "Amazon — $142.99"

    def test_finance_body_from_description(self, db, event_store, user_model_store):
        """Body is extracted from payload.description for finance events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "finance.transaction",
            merchant_name="Starbucks",
            amount=5.75,
            description="Coffee and pastry",
        )
        action = {"rule_name": "Purchase Alert"}

        title, body = lo._build_notification_content(event, action)

        assert "Coffee and pastry" in body


class TestBuildNotificationContentMessage:
    """Message events should use content/body field for the body."""

    def test_message_body_from_content(self, db, event_store, user_model_store):
        """Body is extracted from payload.content for message events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "message.received",
            content="Hey, are you free for lunch today?",
            sender="Alice",
        )
        action = {"rule_name": "Message Alert"}

        title, body = lo._build_notification_content(event, action)

        assert "Hey, are you free for lunch today?" in body

    def test_message_body_from_body_field(self, db, event_store, user_model_store):
        """Body falls back to payload.body for message events."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "message.received",
            body="Can you review the PR?",
            sender="Bob",
        )
        action = {"rule_name": "Message Alert"}

        title, body = lo._build_notification_content(event, action)

        assert "Can you review the PR?" in body


class TestBuildNotificationContentFallbacks:
    """Fallback behavior when no suitable payload fields exist."""

    def test_fallback_title_to_rule_name(self, db, event_store, user_model_store):
        """Title falls back to 'Rule: <name>' when no payload title fields exist."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event("system.heartbeat")
        action = {"rule_name": "Heartbeat Monitor"}

        title, body = lo._build_notification_content(event, action)

        assert title == "Rule: Heartbeat Monitor"

    def test_fallback_title_to_unknown(self, db, event_store, user_model_store):
        """Title falls back to 'Rule: Unknown' when no rule_name is provided."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event("system.heartbeat")
        action = {}

        title, body = lo._build_notification_content(event, action)

        assert title == "Rule: Unknown"

    def test_empty_body_when_no_fields(self, db, event_store, user_model_store):
        """Body is empty string when no recognized payload fields exist."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event("system.heartbeat", some_irrelevant_field="data")
        action = {"rule_name": "Test"}

        title, body = lo._build_notification_content(event, action)

        assert body == ""

    def test_empty_payload(self, db, event_store, user_model_store):
        """Handles events with no payload gracefully."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = {"id": "evt-1", "type": "system.test", "payload": {}, "metadata": {}}
        action = {"rule_name": "Empty Test"}

        title, body = lo._build_notification_content(event, action)

        assert title == "Rule: Empty Test"
        assert body == ""


class TestBuildNotificationContentTruncation:
    """Body content is truncated at 200 characters."""

    def test_body_truncated_at_200_chars(self, db, event_store, user_model_store):
        """Long body text is truncated to 200 characters with ellipsis."""
        lo = _make_life_os(db, event_store, user_model_store)
        long_text = "A" * 300
        event = _make_event("message.received", content=long_text)
        action = {"rule_name": "Test"}

        title, body = lo._build_notification_content(event, action)

        assert len(body) <= 200
        assert body.endswith("...")

    def test_body_not_truncated_when_short(self, db, event_store, user_model_store):
        """Short body text is not truncated or modified."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event("message.received", content="Short message")
        action = {"rule_name": "Test"}

        title, body = lo._build_notification_content(event, action)

        assert body == "Short message"
        assert "..." not in body

    def test_email_body_with_from_truncated(self, db, event_store, user_model_store):
        """Email body with From: prefix is truncated as a whole to 200 chars."""
        lo = _make_life_os(db, event_store, user_model_store)
        long_body = "B" * 300
        event = _make_event(
            "email.received",
            subject="Test",
            body_plain=long_body,
            from_address="sender@example.com",
        )
        action = {"rule_name": "Test"}

        title, body = lo._build_notification_content(event, action)

        assert len(body) <= 200
        assert body.startswith("From: sender@example.com")
        assert body.endswith("...")


# ---------------------------------------------------------------------------
# Integration: verify _execute_rule_action passes enriched content through
# ---------------------------------------------------------------------------

class TestNotifyActionUsesEnrichedContent:
    """End-to-end: _execute_rule_action notify type uses _build_notification_content."""

    @pytest.mark.asyncio
    async def test_email_notification_has_subject_title(self, db, event_store, user_model_store):
        """A notify action for an email event passes the subject as the title."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "email.received",
            subject="Urgent: Server down",
            body_plain="Production server is not responding.",
            from_address="ops@company.com",
        )
        action = {"type": "notify", "rule_name": "Critical Email", "priority": "high"}

        await lo._execute_rule_action(action, event)

        lo.notification_manager.create_notification.assert_awaited_once()
        call_kwargs = lo.notification_manager.create_notification.call_args[1]
        assert call_kwargs["title"] == "Urgent: Server down"
        assert "From: ops@company.com" in call_kwargs["body"]
        assert "Production server is not responding" in call_kwargs["body"]

    @pytest.mark.asyncio
    async def test_calendar_notification_has_summary_title(self, db, event_store, user_model_store):
        """A notify action for a calendar event passes the summary as the title."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "calendar.event.created",
            summary="Board Meeting",
            description="Quarterly review with the board",
        )
        action = {"type": "notify", "rule_name": "Calendar Alert"}

        await lo._execute_rule_action(action, event)

        call_kwargs = lo.notification_manager.create_notification.call_args[1]
        assert call_kwargs["title"] == "Board Meeting"
        assert "Quarterly review" in call_kwargs["body"]

    @pytest.mark.asyncio
    async def test_finance_notification_has_merchant_title(self, db, event_store, user_model_store):
        """A notify action for a finance event passes merchant + amount as the title."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event(
            "finance.transaction",
            merchant_name="Best Buy",
            amount=999.99,
            description="Electronics purchase",
        )
        action = {"type": "notify", "rule_name": "Large Transaction"}

        await lo._execute_rule_action(action, event)

        call_kwargs = lo.notification_manager.create_notification.call_args[1]
        assert call_kwargs["title"] == "Best Buy — $999.99"
        assert "Electronics purchase" in call_kwargs["body"]

    @pytest.mark.asyncio
    async def test_empty_payload_falls_back_to_rule_name(self, db, event_store, user_model_store):
        """A notify action with empty payload uses rule name as title."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _make_event("system.heartbeat")
        action = {"type": "notify", "rule_name": "Heartbeat Monitor"}

        await lo._execute_rule_action(action, event)

        call_kwargs = lo.notification_manager.create_notification.call_args[1]
        assert call_kwargs["title"] == "Rule: Heartbeat Monitor"
