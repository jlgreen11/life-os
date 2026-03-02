"""
Tests for RulesEngine error isolation and logging.

These tests verify the fail-open behavior added to the rules engine:
malformed rules, type errors in conditions, and DB failures during
trigger recording are all handled gracefully without crashing the
evaluation of other rules.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from services.rules_engine.engine import RulesEngine


@pytest.mark.asyncio
async def test_load_rules_skips_malformed_json(db, caplog):
    """Rules with invalid JSON in conditions/actions are skipped without crashing."""
    engine = RulesEngine(db)

    # Insert a valid rule through the normal API
    await engine.add_rule(
        name="Valid rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "good"}],
    )

    # Insert a rule with malformed JSON directly into the DB
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            ("bad-rule-1", "Bad JSON rule", "test.event", "{not valid json", "[]", "test"),
        )

    # Force cache reload
    engine._cache_loaded_at = None

    with caplog.at_level(logging.WARNING, logger="services.rules_engine.engine"):
        event = {"id": "evt-1", "type": "test.event", "payload": {}}
        actions = await engine.evaluate(event)

    # The valid rule should still fire
    assert len(actions) == 1
    assert actions[0]["value"] == "good"

    # A warning should have been logged for the malformed rule
    assert any("Bad JSON rule" in msg for msg in caplog.messages)
    assert any("malformed JSON" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_load_rules_skips_malformed_actions(db, caplog):
    """Rules with malformed JSON in the actions column are skipped gracefully."""
    engine = RulesEngine(db)

    # Insert a valid rule
    await engine.add_rule(
        name="Valid rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "ok"}],
    )

    # Insert a rule with malformed JSON in the actions column
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            ("bad-actions", "Bad actions rule", "test.event", "[]", "not-valid-json{{{", "test"),
        )

    engine._cache_loaded_at = None

    with caplog.at_level(logging.WARNING, logger="services.rules_engine.engine"):
        event = {"id": "evt-1", "type": "test.event", "payload": {}}
        actions = await engine.evaluate(event)

    # Valid rule fires, broken rule is skipped
    assert len(actions) == 1
    assert actions[0]["value"] == "ok"
    assert any("Bad actions rule" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_evaluate_isolates_per_rule_errors(db, caplog):
    """An error in one rule's evaluation doesn't prevent other rules from firing.

    We inject a rule whose condition has `expected=None` for the 'contains'
    operator — before the fix this would crash with AttributeError. The
    per-rule try/except catches this and continues to the next rule.
    """
    engine = RulesEngine(db)

    # Rule 1: will fail — 'contains' with expected=None used to crash
    await engine.add_rule(
        name="Broken rule",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": None},
        ],
        actions=[{"type": "tag", "value": "broken"}],
    )

    # Rule 2: should still fire
    await engine.add_rule(
        name="Good rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "good"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"subject": "hello world"},
    }
    actions = await engine.evaluate(event)

    # The good rule must fire regardless of the broken rule
    assert len(actions) == 1
    assert actions[0]["value"] == "good"


@pytest.mark.asyncio
async def test_contains_operator_handles_none_expected(db):
    """The 'contains' operator returns False when expected value is None."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Contains None",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.text", "op": "contains", "value": None},
        ],
        actions=[{"type": "tag", "value": "should-not-fire"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "some text"},
    }
    # Should not crash, should not match
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_operator_handles_int_expected(db):
    """The 'contains' operator returns False when expected value is an integer."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Contains int",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.text", "op": "contains", "value": 42},
        ],
        actions=[{"type": "tag", "value": "should-not-fire"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "answer is 42"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_any_handles_non_iterable(db):
    """The 'contains_any' operator returns False when expected is not a list."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Contains any string",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.body", "op": "contains_any", "value": "just-a-string"},
        ],
        actions=[{"type": "tag", "value": "should-not-fire"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"body": "just-a-string is here"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_any_handles_none_expected(db):
    """The 'contains_any' operator returns False when expected is None."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Contains any None",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.body", "op": "contains_any", "value": None},
        ],
        actions=[{"type": "tag", "value": "should-not-fire"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"body": "some text"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_any_handles_non_string_items_in_list(db):
    """The 'contains_any' operator skips non-string items in the expected list."""
    engine = RulesEngine(db)

    # List contains a mix of strings and non-strings
    await engine.add_rule(
        name="Mixed list",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.body", "op": "contains_any", "value": [None, 42, "hello"]},
        ],
        actions=[{"type": "tag", "value": "found"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"body": "say hello world"},
    }
    actions = await engine.evaluate(event)
    # Should match on "hello" despite None and 42 in the list
    assert len(actions) == 1
    assert actions[0]["value"] == "found"


@pytest.mark.asyncio
async def test_record_trigger_failure_doesnt_crash(db):
    """A DB failure during _record_trigger doesn't prevent actions from being returned."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Action rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "notify", "priority": "high"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}

    # Patch _record_trigger to simulate a DB failure
    original_record = engine._record_trigger

    async def failing_record(*args, **kwargs):
        """Simulate a DB write failure."""
        raise Exception("DB write failed")

    engine._record_trigger = failing_record

    # The evaluation should still return actions despite the trigger recording failure.
    # The per-rule try/except in evaluate() catches the _record_trigger exception.
    actions = await engine.evaluate(event)
    assert len(actions) == 1
    assert actions[0]["type"] == "notify"
    assert actions[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_evaluate_logs_debug_summary(db, caplog):
    """The evaluate method logs a debug-level summary of evaluation results."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Debug test rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "test"}],
    )

    with caplog.at_level(logging.DEBUG, logger="services.rules_engine.engine"):
        event = {"id": "evt-1", "type": "test.event", "payload": {}}
        await engine.evaluate(event)

    # Check for the debug summary line
    assert any("Evaluated" in msg and "test.event" in msg and "1 matched" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_load_rules_logs_cache_refresh(db, caplog):
    """The load_rules method logs an INFO message about the cache refresh."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Log test",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "test"}],
    )

    # Force a fresh reload
    engine._cache_loaded_at = None

    with caplog.at_level(logging.INFO, logger="services.rules_engine.engine"):
        engine.load_rules()

    assert any("Rules cache refreshed" in msg for msg in caplog.messages)
    assert any("1 active rules loaded" in msg for msg in caplog.messages)
