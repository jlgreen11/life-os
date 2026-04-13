"""
Tests for RulesEngine input validation and diagnostics.

Covers:
- ValueError raised when an action is missing the required "type" field.
- ValueError raised when a regex condition contains an invalid pattern.
- logger.warning emitted (but no exception) for rules with empty conditions.
- get_diagnostics() structure and content.
- get_diagnostics() correctly flags empty-condition rules.
- get_diagnostics() correctly flags invalid regex patterns already in the DB.
"""

import json
import logging
from unittest.mock import patch

import pytest

from services.rules_engine.engine import RulesEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(db) -> RulesEngine:
    """Return a RulesEngine wired to a fresh temporary database."""
    return RulesEngine(db)


async def _add_basic_rule(engine: RulesEngine, **kwargs) -> str:
    """Add a minimal valid rule, overriding fields via kwargs."""
    defaults = dict(
        name="Test rule",
        trigger_event="email.received",
        conditions=[{"field": "payload.subject", "op": "eq", "value": "hello"}],
        actions=[{"type": "tag", "value": "test"}],
    )
    defaults.update(kwargs)
    return await engine.add_rule(**defaults)


# ---------------------------------------------------------------------------
# Validation: invalid regex pattern raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_rule_invalid_regex_raises(db):
    """add_rule() must raise ValueError when a regex condition has a broken pattern."""
    engine = _make_engine(db)

    with pytest.raises(ValueError, match="Invalid regex"):
        await engine.add_rule(
            name="Bad regex rule",
            trigger_event="email.received",
            conditions=[
                {"field": "payload.subject", "op": "regex", "value": "[unclosed"},
            ],
            actions=[{"type": "notify"}],
        )


@pytest.mark.asyncio
async def test_add_rule_invalid_regex_mentions_field(db):
    """The ValueError message includes the field name to aid debugging."""
    engine = _make_engine(db)

    with pytest.raises(ValueError, match="payload.from_address"):
        await engine.add_rule(
            name="Bad regex field info",
            trigger_event="email.received",
            conditions=[
                {"field": "payload.from_address", "op": "regex", "value": "(?P<bad"},
            ],
            actions=[{"type": "tag", "value": "x"}],
        )


@pytest.mark.asyncio
async def test_add_rule_valid_regex_does_not_raise(db):
    """A well-formed regex pattern must be accepted without error."""
    engine = _make_engine(db)

    # Should complete without raising
    rule_id = await engine.add_rule(
        name="Valid regex rule",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "regex", "value": r"^\[URGENT\]"},
        ],
        actions=[{"type": "notify"}],
    )
    assert rule_id  # non-empty UUID


# ---------------------------------------------------------------------------
# Validation: empty conditions triggers logger.warning (not an exception)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_rule_empty_conditions_warns(db):
    """add_rule() must emit a logger.warning — not raise — for empty conditions."""
    engine = _make_engine(db)

    with patch.object(
        logging.getLogger("services.rules_engine.engine"),
        "warning",
    ) as mock_warn:
        rule_id = await engine.add_rule(
            name="Catch-all rule",
            trigger_event="calendar.conflict.detected",
            conditions=[],  # intentionally empty
            actions=[{"type": "notify", "priority": "high"}],
        )

    # Rule must still be created successfully
    assert rule_id

    # Warning must have been emitted and must mention the rule name
    mock_warn.assert_called_once()
    call_args = mock_warn.call_args
    # First positional arg is the format string; remaining args are format params
    formatted = call_args[0][0] % call_args[0][1:]
    assert "Catch-all rule" in formatted
    assert "calendar.conflict.detected" in formatted


@pytest.mark.asyncio
async def test_add_rule_nonempty_conditions_does_not_warn(db):
    """add_rule() must NOT warn when conditions are provided."""
    engine = _make_engine(db)

    with patch.object(
        logging.getLogger("services.rules_engine.engine"),
        "warning",
    ) as mock_warn:
        await _add_basic_rule(engine)

    mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Validation: missing action 'type' raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_rule_missing_action_type_raises(db):
    """add_rule() must raise ValueError when an action lacks the 'type' key."""
    engine = _make_engine(db)

    with pytest.raises(ValueError, match='missing required "type" field'):
        await engine.add_rule(
            name="No-type action",
            trigger_event="email.received",
            conditions=[{"field": "payload.subject", "op": "eq", "value": "hi"}],
            actions=[{"value": "urgent"}],  # 'type' key absent
        )


@pytest.mark.asyncio
async def test_add_rule_missing_action_type_mentions_index(db):
    """The ValueError message includes the zero-based action index."""
    engine = _make_engine(db)

    with pytest.raises(ValueError, match="Action 1"):
        await engine.add_rule(
            name="Second action broken",
            trigger_event="email.received",
            conditions=[{"field": "payload.subject", "op": "eq", "value": "hi"}],
            actions=[
                {"type": "tag", "value": "ok"},
                {"value": "missing_type"},  # index 1 is broken
            ],
        )


# ---------------------------------------------------------------------------
# get_diagnostics() — structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_diagnostics_structure(db):
    """get_diagnostics() must return a dict with all required top-level keys."""
    engine = _make_engine(db)
    engine.load_rules()

    diag = engine.get_diagnostics()

    required_keys = {
        "total_rules",
        "cache_loaded_at",
        "rules_by_trigger",
        "empty_condition_rules",
        "invalid_regex_rules",
        "health",
    }
    assert required_keys.issubset(set(diag.keys()))
    assert isinstance(diag["total_rules"], int)
    assert isinstance(diag["rules_by_trigger"], dict)
    assert isinstance(diag["empty_condition_rules"], list)
    assert isinstance(diag["invalid_regex_rules"], list)
    assert diag["health"] in ("ok", "degraded")


@pytest.mark.asyncio
async def test_get_diagnostics_empty_engine(db):
    """Diagnostics on an engine with no rules should report zero totals and 'ok' health."""
    engine = _make_engine(db)
    engine.load_rules()

    diag = engine.get_diagnostics()

    assert diag["total_rules"] == 0
    assert diag["rules_by_trigger"] == {}
    assert diag["empty_condition_rules"] == []
    assert diag["invalid_regex_rules"] == []
    assert diag["health"] == "ok"
    assert "recommendations" not in diag


# ---------------------------------------------------------------------------
# get_diagnostics() — flags empty-condition rules
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_diagnostics_flags_empty_conditions(db):
    """A rule with empty conditions must appear in diagnostics['empty_condition_rules']."""
    engine = _make_engine(db)

    rule_id = await engine.add_rule(
        name="Wildcard notify",
        trigger_event="system.connector.error",
        conditions=[],
        actions=[{"type": "notify", "priority": "high"}],
    )

    diag = engine.get_diagnostics()

    assert diag["total_rules"] == 1
    assert len(diag["empty_condition_rules"]) == 1
    entry = diag["empty_condition_rules"][0]
    assert entry["id"] == rule_id
    assert entry["name"] == "Wildcard notify"
    assert entry["trigger"] == "system.connector.error"

    # Should also generate a human-readable recommendation
    assert "recommendations" in diag
    assert any("Wildcard notify" in rec for rec in diag["recommendations"])


@pytest.mark.asyncio
async def test_get_diagnostics_does_not_flag_normal_rules(db):
    """Rules with non-empty conditions must NOT appear in empty_condition_rules."""
    engine = _make_engine(db)

    await _add_basic_rule(engine)

    diag = engine.get_diagnostics()

    assert diag["empty_condition_rules"] == []
    assert "recommendations" not in diag


@pytest.mark.asyncio
async def test_get_diagnostics_counts_by_trigger(db):
    """rules_by_trigger must correctly tally rules per trigger event type."""
    engine = _make_engine(db)

    await _add_basic_rule(engine, name="Email 1", trigger_event="email.received")
    await _add_basic_rule(engine, name="Email 2", trigger_event="email.received")
    await _add_basic_rule(engine, name="Calendar", trigger_event="calendar.created")

    diag = engine.get_diagnostics()

    assert diag["rules_by_trigger"]["email.received"] == 2
    assert diag["rules_by_trigger"]["calendar.created"] == 1
    assert diag["total_rules"] == 3


# ---------------------------------------------------------------------------
# get_diagnostics() — flags invalid regex in existing DB rules
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_diagnostics_flags_invalid_regex(db):
    """A rule whose regex pattern is invalid must appear in diagnostics['invalid_regex_rules'].

    Because add_rule() now rejects invalid patterns, this test inserts the
    broken rule directly via SQL (simulating a rule that arrived via an older
    code path or manual DB edit) and then checks that diagnostics surfaces it.
    """
    engine = _make_engine(db)

    # Bypass add_rule() validation by writing directly to the DB
    bad_conditions = json.dumps([
        {"field": "payload.subject", "op": "regex", "value": "[bad-pattern"},
    ])
    bad_actions = json.dumps([{"type": "notify"}])
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by)
               VALUES ('bad-rule-id', 'Legacy bad regex', 'email.received', ?, ?, 'system')""",
            (bad_conditions, bad_actions),
        )

    engine.load_rules()
    diag = engine.get_diagnostics()

    assert len(diag["invalid_regex_rules"]) == 1
    entry = diag["invalid_regex_rules"][0]
    assert entry["rule_id"] == "bad-rule-id"
    assert entry["rule_name"] == "Legacy bad regex"
    assert entry["field"] == "payload.subject"
    assert entry["pattern"] == "[bad-pattern"
    assert entry["error"]  # non-empty error message

    # Health should be degraded when invalid regex rules exist
    assert diag["health"] == "degraded"


@pytest.mark.asyncio
async def test_get_diagnostics_health_ok_when_no_invalid_regex(db):
    """Health must remain 'ok' when all regex patterns are valid."""
    engine = _make_engine(db)

    await engine.add_rule(
        name="Good regex",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "regex", "value": r"^\[URGENT\]"},
        ],
        actions=[{"type": "notify"}],
    )

    diag = engine.get_diagnostics()

    assert diag["health"] == "ok"
    assert diag["invalid_regex_rules"] == []
