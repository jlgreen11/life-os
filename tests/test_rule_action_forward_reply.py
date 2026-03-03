"""
Life OS — Tests for forward and auto_reply rule action handlers.

Verifies that _execute_rule_action() calls the correct connector action
names ("send_email" for forward, "reply_email" for auto_reply) and that
edge cases (missing params, missing connector, connector errors) are
handled gracefully with warning/error logs rather than raised exceptions.

Uses the same integration pattern as test_rule_event_cascade.py: real
DatabaseManager + EventStore/UserModelStore with temporary SQLite, mock
EventBus, and AsyncMock connectors injected via connector_map.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from models.core import EventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lifeos_config():
    """Minimal config dict for LifeOS in test mode."""
    return {
        "web_port": 8080,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "mistral",
    }


@pytest.fixture()
async def lifeos(db, event_bus, event_store, user_model_store, lifeos_config):
    """Create a LifeOS instance with the master_event_handler wired up."""
    from main import LifeOS

    los = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=lifeos_config,
    )

    await los._register_event_handlers()

    # Extract the handler from the mock's call args for direct calls.
    handler = event_bus.subscribe_all.call_args[0][0]
    los.master_event_handler = handler

    return los


def _make_email_event(source="proton_mail", **payload_overrides):
    """Build a well-formed email event dict with sensible defaults.

    Args:
        source: The connector source (e.g. "proton_mail", "google").
        **payload_overrides: Fields merged into the default email payload.

    Returns:
        A complete event dict suitable for _execute_rule_action.
    """
    payload = {
        "message_id": "<msg-123@example.com>",
        "subject": "Test Subject",
        "body": "Hello, this is the email body.",
        "snippet": "Hello, this is...",
        "sender": "alice@example.com",
        "from": "alice@example.com",
    }
    payload.update(payload_overrides)
    return {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": payload,
        "metadata": {},
    }


def _make_mock_connector():
    """Create a mock connector with an AsyncMock execute method."""
    connector = AsyncMock()
    connector.execute = AsyncMock(return_value={"status": "sent"})
    return connector


# ---------------------------------------------------------------------------
# Forward action tests
# ---------------------------------------------------------------------------


class TestForwardAction:
    """Tests for the 'forward' rule action handler."""

    async def test_forward_calls_send_email(self, lifeos):
        """Forward action should call connector.execute with 'send_email'."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-1",
        }

        await lifeos._execute_rule_action(action, event)

        connector.execute.assert_called_once()
        call_args = connector.execute.call_args
        assert call_args[0][0] == "send_email"

    async def test_forward_params_shape(self, lifeos):
        """Forward action should pass correct params to send_email."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-1",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        assert params["to"] == ["bob@example.com"]
        assert params["subject"] == "Fwd: Test Subject"
        assert params["body"] == "Hello, this is the email body."
        assert params["forwarded_from"] == event["id"]

    async def test_forward_preserves_existing_fwd_prefix(self, lifeos):
        """Forward action should not double-prepend 'Fwd: ' to subjects."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event(subject="Fwd: Already Forwarded")
        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-1",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        assert params["subject"] == "Fwd: Already Forwarded"

    async def test_forward_missing_target_logs_warning(self, lifeos):
        """Forward action with no target address should log warning and return."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "forward",
            "rule_id": "rule-2",
            # No "value" or "to" — missing target
        }

        with patch("main.logger") as mock_logger:
            await lifeos._execute_rule_action(action, event)

        connector.execute.assert_not_called()

    async def test_forward_no_connector_logs_warning(self, lifeos):
        """Forward action with no matching connector should log warning and return."""
        # Don't register any connector in connector_map
        event = _make_email_event(source="unknown_connector")
        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-3",
        }

        # Should not raise — just log a warning
        await lifeos._execute_rule_action(action, event)

    async def test_forward_connector_error_is_caught(self, lifeos):
        """Forward action should catch and log connector.execute errors."""
        connector = _make_mock_connector()
        connector.execute = AsyncMock(side_effect=RuntimeError("SMTP down"))
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-4",
        }

        # Should not raise — error is caught and logged
        await lifeos._execute_rule_action(action, event)

    async def test_forward_uses_snippet_when_no_body(self, lifeos):
        """Forward action should fall back to snippet when body is missing."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event(body="", snippet="Short preview text")
        # Remove the body key entirely to test the fallback chain
        del event["payload"]["body"]

        action = {
            "type": "forward",
            "value": "bob@example.com",
            "rule_id": "rule-5",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        assert params["body"] == "Short preview text"


# ---------------------------------------------------------------------------
# Auto-reply action tests
# ---------------------------------------------------------------------------


class TestAutoReplyAction:
    """Tests for the 'auto_reply' rule action handler."""

    async def test_auto_reply_calls_reply_email(self, lifeos):
        """Auto-reply action should call connector.execute with 'reply_email'."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "auto_reply",
            "value": "Thanks for your email, I'll get back to you soon.",
            "rule_id": "rule-10",
        }

        await lifeos._execute_rule_action(action, event)

        connector.execute.assert_called_once()
        call_args = connector.execute.call_args
        assert call_args[0][0] == "reply_email"

    async def test_auto_reply_params_shape(self, lifeos):
        """Auto-reply action should pass correct params to reply_email."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "auto_reply",
            "value": "Auto-reply body text",
            "rule_id": "rule-10",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        assert params["to"] == ["alice@example.com"]
        assert params["in_reply_to"] == "<msg-123@example.com>"
        assert params["original_subject"] == "Test Subject"
        assert params["body"] == "Auto-reply body text"

    async def test_auto_reply_uses_message_key(self, lifeos):
        """Auto-reply should also accept message text from the 'message' key."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "auto_reply",
            "message": "Reply from message key",
            "rule_id": "rule-11",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        assert params["body"] == "Reply from message key"

    async def test_auto_reply_missing_message_logs_warning(self, lifeos):
        """Auto-reply with no message text should log warning and return."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "auto_reply",
            "rule_id": "rule-12",
            # No "value" or "message" — missing reply text
        }

        with patch("main.logger") as mock_logger:
            await lifeos._execute_rule_action(action, event)

        connector.execute.assert_not_called()

    async def test_auto_reply_no_connector_logs_warning(self, lifeos):
        """Auto-reply with no matching connector should log warning and return."""
        event = _make_email_event(source="unknown_connector")
        action = {
            "type": "auto_reply",
            "value": "Thanks!",
            "rule_id": "rule-13",
        }

        # Should not raise — just log a warning
        await lifeos._execute_rule_action(action, event)

    async def test_auto_reply_connector_error_is_caught(self, lifeos):
        """Auto-reply should catch and log connector.execute errors."""
        connector = _make_mock_connector()
        connector.execute = AsyncMock(side_effect=ConnectionError("Network down"))
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        action = {
            "type": "auto_reply",
            "value": "Thanks for reaching out!",
            "rule_id": "rule-14",
        }

        # Should not raise — error is caught and logged
        await lifeos._execute_rule_action(action, event)

    async def test_auto_reply_falls_back_to_event_id(self, lifeos):
        """Auto-reply should use event ID as in_reply_to when message_id key is absent."""
        connector = _make_mock_connector()
        lifeos.connector_map["proton_mail"] = connector

        event = _make_email_event()
        # Remove message_id from payload entirely so .get() falls back
        del event["payload"]["message_id"]
        action = {
            "type": "auto_reply",
            "value": "Got it!",
            "rule_id": "rule-15",
        }

        await lifeos._execute_rule_action(action, event)

        call_args = connector.execute.call_args
        params = call_args[0][1]
        # Should fall back to event["id"] when message_id key is absent
        assert params["in_reply_to"] == event["id"]
