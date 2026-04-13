"""
Tests for rules engine edge cases — regex compilation safety and numeric type guards.

Validates that invalid regex patterns and type mismatches in numeric comparisons
are handled gracefully (return False) instead of raising exceptions that cause
entire rules to be skipped.

Note: ``add_rule()`` now rejects invalid regex patterns at creation time.  Tests
that exercise runtime tolerance for malformed patterns insert rules directly
into the DB to simulate patterns that arrived through an older code path or a
manual DB edit — this is intentional and explicitly documented below.
"""

import json

import pytest

from services.rules_engine.engine import RulesEngine


def _insert_rule_with_invalid_regex(db, name: str, trigger_event: str, pattern: str) -> str:
    """Insert a rule with a regex pattern that bypasses add_rule() validation.

    This helper is used only in tests that verify runtime tolerance: that the
    evaluation path handles pre-existing malformed patterns gracefully rather
    than crashing.  It should not be used for anything else.
    """
    import uuid
    rule_id = str(uuid.uuid4())
    conditions = json.dumps([{"field": "payload.text", "op": "regex", "value": pattern}])
    actions = json.dumps([{"type": "tag", "value": "matched"}])
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by)
               VALUES (?, ?, ?, ?, ?, 'test')""",
            (rule_id, name, trigger_event, conditions, actions),
        )
    return rule_id


# -------------------------------------------------------------------
# Regex operator edge cases
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regex_invalid_pattern_unclosed_bracket(db):
    """Invalid regex pattern '[' already in the DB should return False, not raise re.error.

    add_rule() now rejects invalid patterns at creation time, so this test
    inserts the rule directly to simulate a legacy or manually-edited record.
    """
    engine = RulesEngine(db)
    _insert_rule_with_invalid_regex(db, "Bad regex bracket", "test.event", "[")
    engine.load_rules()

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "anything"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_regex_invalid_pattern_unmatched_group(db):
    """Invalid regex pattern '(' already in the DB should return False, not raise re.error.

    add_rule() now rejects invalid patterns at creation time, so this test
    inserts the rule directly to simulate a legacy or manually-edited record.
    """
    engine = RulesEngine(db)
    _insert_rule_with_invalid_regex(db, "Bad regex paren", "test.event", "(")
    engine.load_rules()

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "anything"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_regex_valid_pattern_still_works(db):
    """Valid regex patterns should continue to match correctly."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Email regex",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.text", "op": "regex", "value": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"},
        ],
        actions=[{"type": "tag", "value": "has-email"}],
    )

    # Match: text contains an email address
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "Contact me at user@example.com please"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: no email address
    event["payload"]["text"] = "No email here"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_regex_non_string_actual_returns_false(db):
    """Regex operator should return False when the field value is not a string."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Regex on number",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.count", "op": "regex", "value": r"\d+"},
        ],
        actions=[{"type": "tag", "value": "matched"}],
    )

    # actual is an integer, not a string
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"count": 42},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # actual is None (field missing)
    event["payload"] = {}
    actions = await engine.evaluate(event)
    assert len(actions) == 0


# -------------------------------------------------------------------
# Numeric operator type guard edge cases
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gt_string_vs_numeric_returns_false(db):
    """gt operator should return False when actual is a non-numeric string."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="GT string mismatch",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.priority", "op": "gt", "value": 100},
        ],
        actions=[{"type": "tag", "value": "matched"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"priority": "high"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_lt_string_vs_numeric_returns_false(db):
    """lt operator should return False when actual is a non-numeric string."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="LT string mismatch",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.level", "op": "lt", "value": 50},
        ],
        actions=[{"type": "tag", "value": "matched"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"level": "low"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_gte_string_vs_numeric_returns_false(db):
    """gte operator should return False when actual is a non-numeric string."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="GTE string mismatch",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.score", "op": "gte", "value": 80},
        ],
        actions=[{"type": "tag", "value": "matched"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"score": "excellent"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_lte_string_vs_numeric_returns_false(db):
    """lte operator should return False when actual is a non-numeric string."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="LTE string mismatch",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.rating", "op": "lte", "value": 5},
        ],
        actions=[{"type": "tag", "value": "matched"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"rating": "good"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_numeric_string_actual_converts_and_compares(db):
    """Numeric operators should convert numeric strings (e.g., '42') and compare correctly."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="GT with string number",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.amount", "op": "gt", "value": 40},
        ],
        actions=[{"type": "tag", "value": "gt-matched"}],
    )

    await engine.add_rule(
        name="LT with string number",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.amount", "op": "lt", "value": 50},
        ],
        actions=[{"type": "tag", "value": "lt-matched"}],
    )

    await engine.add_rule(
        name="GTE with string number",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.amount", "op": "gte", "value": 42},
        ],
        actions=[{"type": "tag", "value": "gte-matched"}],
    )

    await engine.add_rule(
        name="LTE with string number",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.amount", "op": "lte", "value": 42},
        ],
        actions=[{"type": "tag", "value": "lte-matched"}],
    )

    # actual is "42" as a string — should be converted to float for comparison
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"amount": "42"},
    }
    actions = await engine.evaluate(event)
    values = {a["value"] for a in actions}

    # 42 > 40: True, 42 < 50: True, 42 >= 42: True, 42 <= 42: True
    assert "gt-matched" in values
    assert "lt-matched" in values
    assert "gte-matched" in values
    assert "lte-matched" in values


@pytest.mark.asyncio
async def test_numeric_operators_none_actual_returns_false(db):
    """All numeric operators should return False when actual is None."""
    engine = RulesEngine(db)

    for op in ["gt", "lt", "gte", "lte"]:
        await engine.add_rule(
            name=f"None {op}",
            trigger_event="test.event",
            conditions=[
                {"field": "payload.missing_field", "op": op, "value": 10},
            ],
            actions=[{"type": "tag", "value": f"{op}-matched"}],
        )

    # missing_field is not present, so _resolve_field returns None
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


# -------------------------------------------------------------------
# Rule isolation — bad conditions don't affect other rules
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_regex_does_not_skip_other_rules(db):
    """A rule with an invalid regex should fail gracefully; other rules should still evaluate.

    The bad-regex rule is inserted directly to simulate a legacy record, since
    add_rule() now rejects invalid patterns at creation time.
    """
    engine = RulesEngine(db)

    # Rule 1: invalid regex inserted directly — should fail its condition (return False)
    _insert_rule_with_invalid_regex(db, "Bad regex rule", "test.event", "[invalid")

    # Rule 2: valid rule added normally — should still fire
    await engine.add_rule(
        name="Good rule",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.text", "op": "contains", "value": "hello"},
        ],
        actions=[{"type": "tag", "value": "good-rule"}],
    )

    # Force a cache reload so both rules are visible to evaluate()
    engine.load_rules()

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"text": "hello world"},
    }
    actions = await engine.evaluate(event)

    # Only the good rule should fire; the bad regex rule should return False
    assert len(actions) == 1
    assert actions[0]["value"] == "good-rule"


@pytest.mark.asyncio
async def test_type_mismatch_does_not_skip_other_rules(db):
    """A rule with a numeric type mismatch should fail gracefully; other rules still evaluate."""
    engine = RulesEngine(db)

    # Rule 1: type mismatch — 'high' > 100 should return False
    await engine.add_rule(
        name="Bad comparison rule",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.priority", "op": "gt", "value": 100},
        ],
        actions=[{"type": "tag", "value": "bad-compare"}],
    )

    # Rule 2: valid rule — should still fire
    await engine.add_rule(
        name="Valid rule",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.status", "op": "eq", "value": "active"},
        ],
        actions=[{"type": "tag", "value": "valid-rule"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"priority": "high", "status": "active"},
    }
    actions = await engine.evaluate(event)

    # Only the valid rule should fire
    assert len(actions) == 1
    assert actions[0]["value"] == "valid-rule"
