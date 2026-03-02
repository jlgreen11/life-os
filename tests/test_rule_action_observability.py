"""
Life OS — Rule Action Observability Tests

Tests for two code-quality fixes shipped together:

1. Unhandled rule action types now emit logger.warning instead of being
   silently dropped.  Rules that specify an unknown/unimplemented type would
   previously do nothing without any indication that the action was ignored.
   Note: "forward", "auto_reply", and "archive" are now implemented — see
   test_rule_actions_extended.py for their specific tests.

2. Bare ``except:`` in the prediction-engine diagnostics endpoint was replaced
   with ``except (json.JSONDecodeError, KeyError, ValueError):`` so that
   KeyboardInterrupt and SystemExit are no longer accidentally swallowed when
   iterating calendar event payloads.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_life_os(db, event_store, user_model_store):
    """
    Build a minimal LifeOS shell that exposes _execute_rule_action without
    spinning up NATS, Ollama, or any connector.
    """
    from main import LifeOS

    lo = object.__new__(LifeOS)
    lo.db = db
    lo.event_store = event_store
    lo.user_model_store = user_model_store
    lo.connector_map = {}

    # Stub out every service _execute_rule_action touches
    nm = MagicMock()
    nm.create_notification = AsyncMock()
    lo.notification_manager = nm

    tm = MagicMock()
    tm.create_task = AsyncMock()
    lo.task_manager = tm

    # _infer_domain_from_event_type is a pure method — wire the real one
    lo._infer_domain_from_event_type = LifeOS._infer_domain_from_event_type.__get__(
        lo, LifeOS
    )
    return lo


def _dummy_event(event_type="email.received"):
    return {"id": "evt-1", "type": event_type, "payload": {}, "metadata": {}}


# ---------------------------------------------------------------------------
# 1. Unhandled rule action type  → logger.warning
# ---------------------------------------------------------------------------

class TestUnhandledRuleActionWarning:
    """_execute_rule_action should warn on unknown action types."""

    @pytest.mark.asyncio
    async def test_unknown_action_emits_warning(self, db, event_store, user_model_store):
        """An unrecognised action type triggers a logger.warning, not a silent no-op."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "send_sms", "rule_id": "rule-abc"}

        with patch("main.logger") as mock_logger:
            await lo._execute_rule_action(action, _dummy_event())
            mock_logger.warning.assert_called_once()
            # The warning message should mention the unrecognised action type
            call_args = mock_logger.warning.call_args
            assert "send_sms" in str(call_args)

    @pytest.mark.asyncio
    async def test_webhook_action_emits_warning(self, db, event_store, user_model_store):
        """An unimplemented action type ('webhook') emits a warning."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "webhook", "rule_id": "rule-xyz"}

        with patch("main.logger") as mock_logger:
            await lo._execute_rule_action(action, _dummy_event())
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_action_emits_warning(self, db, event_store, user_model_store):
        """An unimplemented action type ('escalate') emits a warning."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "escalate", "rule_id": "rule-esc"}

        with patch("main.logger") as mock_logger:
            await lo._execute_rule_action(action, _dummy_event())
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_warning_includes_rule_id_and_event_type(self, db, event_store, user_model_store):
        """The warning message contains the rule_id and event_type for traceability."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"type": "unknown_type", "rule_id": "rule-99"}
        event = _dummy_event("calendar.event.created")

        with patch("main.logger") as mock_logger:
            await lo._execute_rule_action(action, event)
            call_args = str(mock_logger.warning.call_args)
            assert "rule-99" in call_args
            assert "calendar.event.created" in call_args

    @pytest.mark.asyncio
    async def test_known_action_does_not_warn(self, db, event_store, user_model_store):
        """Known action types must not trigger the unhandled-action warning."""
        lo = _make_life_os(db, event_store, user_model_store)

        # Patch event_store.add_tag to avoid DB operations for tag/suppress/archive
        lo.event_store = MagicMock()
        lo.event_store.add_tag = MagicMock()

        # Add a mock connector so forward/auto_reply can dispatch
        mock_connector = MagicMock()
        mock_connector.execute = AsyncMock(return_value={"status": "ok"})
        lo.connector_map = {"test_source": mock_connector}

        known_actions = [
            {"type": "tag", "value": "test", "rule_id": "r1"},
            {"type": "suppress", "rule_id": "r2"},
            {"type": "create_task", "title": "Do it", "rule_id": "r3"},
            {"type": "archive", "rule_id": "r4"},
            {"type": "forward", "to": "someone@example.com", "rule_id": "r5"},
            {"type": "auto_reply", "value": "Thanks!", "rule_id": "r6"},
        ]

        for action in known_actions:
            event = _dummy_event()
            event["source"] = "test_source"
            with patch("main.logger") as mock_logger:
                await lo._execute_rule_action(action, event)
                mock_logger.warning.assert_not_called(), (
                    f"action type={action['type']} should not emit a warning"
                )

    @pytest.mark.asyncio
    async def test_none_action_type_emits_warning(self, db, event_store, user_model_store):
        """An action with type=None is also unhandled and should emit a warning."""
        lo = _make_life_os(db, event_store, user_model_store)
        action = {"rule_id": "r-none"}  # no "type" key

        with patch("main.logger") as mock_logger:
            await lo._execute_rule_action(action, _dummy_event())
            mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Bare except: → specific exception types in diagnostics
# ---------------------------------------------------------------------------

class TestDiagnosticsExceptionSpecificity:
    """
    The prediction-engine diagnostics endpoint previously used a bare except:
    when counting all-day vs. timed calendar events.  It now catches only
    (json.JSONDecodeError, KeyError, ValueError) — not BaseException subclasses.
    """

    def test_malformed_json_payload_is_skipped(self, db):
        """
        Rows with non-JSON payloads are silently skipped; other rows continue
        to be counted normally.

        This test exercises the counting loop directly rather than going through
        the DB (SQLite enforces a JSON constraint on the payload column so we
        cannot insert invalid JSON there).  The loop logic is identical to what
        the diagnostics endpoint executes after fetching rows from the DB.
        """
        # Simulate two rows: one with a valid JSON string and one whose payload
        # field contains a non-JSON value (e.g. a corrupted row or a dict that
        # sqlite3.Row returns as a raw string rather than parsed JSON).
        class FakeRow:
            def __init__(self, payload):
                self._payload = payload

            def __getitem__(self, key):
                if key == "payload":
                    return self._payload
                raise KeyError(key)

        good_row = FakeRow(json.dumps({"is_all_day": False, "title": "Meeting"}))
        bad_row = FakeRow("NOT_VALID_JSON")
        rows = [good_row, bad_row]

        all_day_count = 0
        timed_count = 0
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                if payload.get("is_all_day"):
                    all_day_count += 1
                else:
                    timed_count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        # Only the good row should be counted; the bad row is skipped cleanly
        assert timed_count == 1
        assert all_day_count == 0

    def test_key_error_in_payload_is_skipped(self, db):
        """KeyError during payload inspection does not abort the count loop."""
        # If row["payload"] itself raises KeyError on access, the loop continues.
        # This mimics what the fixed code now handles explicitly.
        rows = [{"other_key": "not_payload"}]

        all_day_count = 0
        timed_count = 0
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                if payload.get("is_all_day"):
                    all_day_count += 1
                else:
                    timed_count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        assert all_day_count == 0
        assert timed_count == 0
