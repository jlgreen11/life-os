"""
Life OS — Extended Rule Action Tests

Tests for the three newly-implemented rule action types: archive, forward,
and auto_reply. These actions were documented in the rules engine but
previously fell through to the "unhandled action" warning path.

Test approach:
    - archive: uses real EventStore (from the db fixture) to verify tags
    - forward/auto_reply: use a mock connector to verify execute() dispatch
    - Error paths: verify graceful failure with proper logging
"""

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_life_os(db, event_store, user_model_store, connector_map=None):
    """
    Build a minimal LifeOS shell that exposes _execute_rule_action without
    spinning up NATS, Ollama, or any connector.

    Uses the same pattern as test_rule_action_observability.py: allocate the
    object without __init__ and wire only the attributes that the method
    touches.
    """
    from main import LifeOS

    lo = object.__new__(LifeOS)
    lo.db = db
    lo.event_store = event_store
    lo.user_model_store = user_model_store
    lo.connector_map = connector_map or {}

    # Stub services that other action types (notify, create_task) use
    nm = MagicMock()
    nm.create_notification = AsyncMock()
    lo.notification_manager = nm

    tm = MagicMock()
    tm.create_task = AsyncMock()
    lo.task_manager = tm

    # Wire the real _infer_domain_from_event_type method
    lo._infer_domain_from_event_type = LifeOS._infer_domain_from_event_type.__get__(
        lo, LifeOS
    )
    return lo


def _dummy_event(event_type="email.received", source="proton_mail", **payload_overrides):
    """Create a minimal event dict with sensible defaults for testing."""
    payload = {
        "subject": "Test subject",
        "body": "Test body content",
        "sender": "alice@example.com",
        "from": "alice@example.com",
        "message_id": "msg-123",
        "snippet": "Test snippet",
    }
    payload.update(payload_overrides)
    return {
        "id": "evt-1",
        "type": event_type,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "metadata": {},
    }


def _mock_connector():
    """Create a mock connector with an async execute method."""
    connector = MagicMock()
    connector.execute = AsyncMock(return_value={"status": "ok"})
    return connector


# ---------------------------------------------------------------------------
# Archive action tests
# ---------------------------------------------------------------------------

class TestArchiveAction:
    """Tests for the 'archive' rule action type."""

    @pytest.mark.asyncio
    async def test_archive_sets_suppressed_flag(self, db, event_store, user_model_store):
        """Archive action sets the _suppressed in-memory flag on the event."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _dummy_event()
        action = {"type": "archive", "rule_id": "rule-arch-1"}

        # Store the event first so add_tag has a valid event_id
        event_store.store_event(event)
        await lo._execute_rule_action(action, event)

        assert event.get("_suppressed") is True

    @pytest.mark.asyncio
    async def test_archive_adds_system_archived_tag(self, db, event_store, user_model_store):
        """Archive action persists a 'system:archived' tag via EventStore."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _dummy_event()
        action = {"type": "archive", "rule_id": "rule-arch-2"}

        event_store.store_event(event)
        await lo._execute_rule_action(action, event)

        tags = event_store.get_tags(event["id"])
        assert "system:archived" in tags

    @pytest.mark.asyncio
    async def test_archive_suppresses_subsequent_notify(self, db, event_store, user_model_store):
        """After archive, a notify action on the same event should be skipped."""
        lo = _make_life_os(db, event_store, user_model_store)
        event = _dummy_event()

        event_store.store_event(event)

        # Archive first
        await lo._execute_rule_action(
            {"type": "archive", "rule_id": "rule-arch-3"}, event
        )
        # Then try to notify — should be skipped because _suppressed is True
        await lo._execute_rule_action(
            {"type": "notify", "rule_name": "Test", "rule_id": "rule-n-1"}, event
        )

        lo.notification_manager.create_notification.assert_not_called()


# ---------------------------------------------------------------------------
# Forward action tests
# ---------------------------------------------------------------------------

class TestForwardAction:
    """Tests for the 'forward' rule action type."""

    @pytest.mark.asyncio
    async def test_forward_calls_connector_execute(self, db, event_store, user_model_store):
        """Forward action dispatches to connector.execute('send_email', ...) with correct params."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {"type": "forward", "to": "bob@example.com", "rule_id": "rule-fwd-1"}

        await lo._execute_rule_action(action, event)

        connector.execute.assert_awaited_once()
        call_args = connector.execute.call_args
        # Uses "send_email" — the standard action name implemented by connectors
        assert call_args[0][0] == "send_email"
        params = call_args[0][1]
        assert params["to"] == ["bob@example.com"]
        assert params["forwarded_from"] == "evt-1"
        assert params["subject"] == "Fwd: Test subject"
        assert params["body"] == "Test body content"

    @pytest.mark.asyncio
    async def test_forward_uses_value_field(self, db, event_store, user_model_store):
        """Forward action reads the target address from 'value' as a fallback."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {"type": "forward", "value": "charlie@example.com", "rule_id": "rule-fwd-2"}

        await lo._execute_rule_action(action, event)

        params = connector.execute.call_args[0][1]
        assert params["to"] == ["charlie@example.com"]

    @pytest.mark.asyncio
    async def test_forward_missing_to_logs_warning(self, db, event_store, user_model_store, caplog):
        """Forward action with no target address logs a warning and returns."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "forward", "rule_id": "rule-fwd-3"}

        with caplog.at_level(logging.WARNING, logger="main"):
            await lo._execute_rule_action(action, _dummy_event())

        assert any("forward action missing target address" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_forward_unknown_source_logs_warning(self, db, event_store, user_model_store, caplog):
        """Forward action with no matching connector logs a warning and returns."""
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={},  # no connectors registered
        )
        event = _dummy_event(source="unknown_connector")
        action = {"type": "forward", "to": "bob@example.com", "rule_id": "rule-fwd-4"}

        with caplog.at_level(logging.WARNING, logger="main"):
            await lo._execute_rule_action(action, event)

        assert any("no connector for source" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_forward_connector_error_logged(self, db, event_store, user_model_store, caplog):
        """If connector.execute raises, the error is logged and the action fails gracefully."""
        connector = _mock_connector()
        connector.execute = AsyncMock(side_effect=NotImplementedError("forward not supported"))
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {"type": "forward", "to": "bob@example.com", "rule_id": "rule-fwd-5"}

        with caplog.at_level(logging.ERROR, logger="main"):
            await lo._execute_rule_action(action, event)

        assert any("forward action failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_forward_uses_snippet_when_no_body(self, db, event_store, user_model_store):
        """Forward action falls back to 'snippet' when 'body' is absent from the payload."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        # Remove 'body' from payload so it falls back to 'snippet'
        del event["payload"]["body"]
        action = {"type": "forward", "to": "bob@example.com", "rule_id": "rule-fwd-6"}

        await lo._execute_rule_action(action, event)

        params = connector.execute.call_args[0][1]
        assert params["body"] == "Test snippet"


# ---------------------------------------------------------------------------
# Auto-reply action tests
# ---------------------------------------------------------------------------

class TestAutoReplyAction:
    """Tests for the 'auto_reply' rule action type."""

    @pytest.mark.asyncio
    async def test_auto_reply_calls_connector_execute(self, db, event_store, user_model_store):
        """Auto-reply action dispatches to connector.execute('reply_email', ...) with correct params."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {
            "type": "auto_reply",
            "value": "I'm currently away.",
            "rule_id": "rule-ar-1",
        }

        await lo._execute_rule_action(action, event)

        connector.execute.assert_awaited_once()
        call_args = connector.execute.call_args
        # Uses "reply_email" — the standard action name implemented by connectors
        assert call_args[0][0] == "reply_email"
        params = call_args[0][1]
        assert params["to"] == ["alice@example.com"]
        assert params["body"] == "I'm currently away."
        assert params["original_subject"] == "Test subject"
        assert params["in_reply_to"] == "msg-123"

    @pytest.mark.asyncio
    async def test_auto_reply_uses_message_field(self, db, event_store, user_model_store):
        """Auto-reply reads the reply text from 'message' as a fallback for 'value'."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {
            "type": "auto_reply",
            "message": "Thanks for your email!",
            "rule_id": "rule-ar-2",
        }

        await lo._execute_rule_action(action, event)

        params = connector.execute.call_args[0][1]
        assert params["body"] == "Thanks for your email!"

    @pytest.mark.asyncio
    async def test_auto_reply_missing_message_logs_warning(self, db, event_store, user_model_store, caplog):
        """Auto-reply with no message text logs a warning and returns."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "auto_reply", "rule_id": "rule-ar-3"}

        with caplog.at_level(logging.WARNING, logger="main"):
            await lo._execute_rule_action(action, _dummy_event())

        assert any("auto_reply action missing message text" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_auto_reply_unknown_source_logs_warning(self, db, event_store, user_model_store, caplog):
        """Auto-reply with no matching connector logs a warning and returns."""
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={},
        )
        event = _dummy_event(source="unknown_connector")
        action = {
            "type": "auto_reply",
            "value": "Out of office",
            "rule_id": "rule-ar-4",
        }

        with caplog.at_level(logging.WARNING, logger="main"):
            await lo._execute_rule_action(action, event)

        assert any("no connector for source" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_auto_reply_connector_error_logged(self, db, event_store, user_model_store, caplog):
        """If connector.execute raises, the error is logged gracefully."""
        connector = _mock_connector()
        connector.execute = AsyncMock(side_effect=RuntimeError("Connection lost"))
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        action = {
            "type": "auto_reply",
            "value": "Auto-reply text",
            "rule_id": "rule-ar-5",
        }

        with caplog.at_level(logging.ERROR, logger="main"):
            await lo._execute_rule_action(action, event)

        assert any("auto_reply action failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_auto_reply_uses_event_id_when_no_message_id(self, db, event_store, user_model_store):
        """Auto-reply falls back to event['id'] for in_reply_to when payload has no message_id."""
        connector = _mock_connector()
        lo = _make_life_os(
            db, event_store, user_model_store,
            connector_map={"proton_mail": connector},
        )
        event = _dummy_event(source="proton_mail")
        del event["payload"]["message_id"]
        action = {
            "type": "auto_reply",
            "value": "Got it!",
            "rule_id": "rule-ar-6",
        }

        await lo._execute_rule_action(action, event)

        params = connector.execute.call_args[0][1]
        assert params["in_reply_to"] == "evt-1"
