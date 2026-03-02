"""
Tests for the RulesEngine cache TTL reload logic.

Regression tests for a bug where ``timedelta.seconds`` was used instead of
``timedelta.total_seconds()`` in the cache staleness check.  ``timedelta.seconds``
returns only the *seconds component* (0-86399), so a delta of exactly 1 hour
gives ``.seconds == 0``, which incorrectly skips the reload.

These tests verify the fix by manipulating ``_cache_loaded_at`` to simulate
elapsed time and checking whether the rules cache is refreshed on the next
``evaluate()`` call.
"""

import json

import pytest
from datetime import datetime, timedelta, timezone

from services.rules_engine.engine import RulesEngine


def _insert_rule_directly(db, rule_id: str, name: str, trigger: str = "test.event"):
    """Insert a rule directly into the DB, bypassing the engine cache."""
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rule_id, name, trigger, json.dumps([]), json.dumps([{"type": "tag", "value": name}]), "test"),
        )


DUMMY_EVENT = {"id": "evt-1", "type": "test.event", "payload": {}}


@pytest.mark.asyncio
async def test_cache_reload_within_60s(db):
    """Cache should NOT reload when less than 60 seconds have elapsed."""
    engine = RulesEngine(db)

    # Prime the cache with an initial load (empty rules)
    engine.load_rules()
    assert engine._rules_cache == []

    # Insert a rule directly into the DB (engine doesn't know about it yet)
    _insert_rule_directly(db, "rule-fresh", "Fresh Rule")

    # Set cache loaded at 30 seconds ago — within the 60s TTL window
    engine._cache_loaded_at = datetime.now(timezone.utc) - timedelta(seconds=30)

    # evaluate() should NOT reload the cache, so the new rule stays invisible
    actions = await engine.evaluate(DUMMY_EVENT)
    assert len(actions) == 0, "Cache should not have reloaded within the 60s TTL window"


@pytest.mark.asyncio
async def test_cache_reload_after_60s(db):
    """Cache SHOULD reload when more than 60 seconds have elapsed."""
    engine = RulesEngine(db)

    # Prime the cache (empty)
    engine.load_rules()
    assert engine._rules_cache == []

    # Insert a rule directly into the DB
    _insert_rule_directly(db, "rule-stale", "Stale Rule")

    # Set cache loaded at 90 seconds ago — past the 60s TTL
    engine._cache_loaded_at = datetime.now(timezone.utc) - timedelta(seconds=90)

    # evaluate() should reload and pick up the new rule
    actions = await engine.evaluate(DUMMY_EVENT)
    assert len(actions) == 1, "Cache should have reloaded after 90 seconds"
    assert actions[0]["value"] == "Stale Rule"


@pytest.mark.asyncio
async def test_cache_reload_at_hour_boundary(db):
    """Cache SHOULD reload when exactly 1 hour has elapsed.

    This is the specific regression test for the ``.seconds`` bug.
    ``timedelta(hours=1).seconds == 0``, so the old code evaluated
    ``0 > 60`` which is ``False`` — incorrectly skipping the reload.
    With ``.total_seconds()``, the delta is 3600 which correctly
    triggers a reload.
    """
    engine = RulesEngine(db)

    # Prime the cache (empty)
    engine.load_rules()
    assert engine._rules_cache == []

    # Insert a rule directly
    _insert_rule_directly(db, "rule-hour", "Hour Rule")

    # Set cache loaded at exactly 1 hour ago
    engine._cache_loaded_at = datetime.now(timezone.utc) - timedelta(hours=1)

    # evaluate() should reload — this would FAIL with the .seconds bug
    actions = await engine.evaluate(DUMMY_EVENT)
    assert len(actions) == 1, (
        "Cache should have reloaded after 1 hour — "
        "timedelta(hours=1).total_seconds() == 3600 > 60"
    )
    assert actions[0]["value"] == "Hour Rule"


@pytest.mark.asyncio
async def test_cache_reload_at_multi_hour_boundary(db):
    """Cache SHOULD reload when exactly 2 hours have elapsed.

    Another boundary case: ``timedelta(hours=2).seconds == 0`` with the
    old bug, but ``.total_seconds() == 7200`` which correctly triggers.
    """
    engine = RulesEngine(db)

    # Prime the cache (empty)
    engine.load_rules()
    assert engine._rules_cache == []

    # Insert a rule directly
    _insert_rule_directly(db, "rule-multi-hour", "Multi Hour Rule")

    # Set cache loaded at exactly 2 hours ago
    engine._cache_loaded_at = datetime.now(timezone.utc) - timedelta(hours=2)

    # evaluate() should reload
    actions = await engine.evaluate(DUMMY_EVENT)
    assert len(actions) == 1, (
        "Cache should have reloaded after 2 hours — "
        "timedelta(hours=2).total_seconds() == 7200 > 60"
    )
    assert actions[0]["value"] == "Multi Hour Rule"
