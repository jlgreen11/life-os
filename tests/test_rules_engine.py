"""
Tests for the RulesEngine — deterministic event automation system.

The RulesEngine evaluates rules against every event on the bus and executes
matching actions. This test suite validates:

1. Trigger matching — exact, wildcard, and glob patterns
2. Condition evaluation — all 11 operators (eq, neq, contains, contains_any,
   in, not_in, gt, lt, gte, lte, exists, not_exists, regex)
3. Action collection — rule metadata attachment
4. Rule management — add, remove, cache reload
5. Edge cases — nested field resolution, type mismatches, unknown operators
"""

import json
import pytest
from datetime import datetime, timezone
from services.rules_engine.engine import RulesEngine, install_default_rules


@pytest.mark.asyncio
async def test_exact_trigger_match(db):
    """Test that rules only fire for exact event type matches."""
    engine = RulesEngine(db)

    # Add rule that triggers on email.received
    await engine.add_rule(
        name="Email handler",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify"}],
    )

    # Matching event type -> should trigger
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"from": "test@example.com"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1
    assert actions[0]["type"] == "notify"
    assert actions[0]["rule_name"] == "Email handler"

    # Non-matching event type -> should not trigger
    event["type"] = "email.sent"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_wildcard_trigger(db):
    """Test that '*' trigger matches all event types."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Catch-all",
        trigger_event="*",
        conditions=[],
        actions=[{"type": "tag", "value": "all"}],
    )

    # All event types should match
    for event_type in ["email.received", "calendar.created", "finance.transaction", "custom.event"]:
        event = {"id": "evt-1", "type": event_type, "payload": {}}
        actions = await engine.evaluate(event)
        assert len(actions) == 1
        assert actions[0]["value"] == "all"


@pytest.mark.asyncio
async def test_glob_trigger_pattern(db):
    """Test that glob patterns like 'email.*' match correctly."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="All email events",
        trigger_event="email.*",
        conditions=[],
        actions=[{"type": "tag", "value": "email"}],
    )

    # Should match any email.* event
    for suffix in ["received", "sent", "archived", "deleted"]:
        event = {"id": "evt-1", "type": f"email.{suffix}", "payload": {}}
        actions = await engine.evaluate(event)
        assert len(actions) == 1

    # Should NOT match other event types
    event = {"id": "evt-1", "type": "calendar.created", "payload": {}}
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_eq_operator(db):
    """Test the 'eq' (equality) operator."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Priority email",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.priority", "op": "eq", "value": "high"},
        ],
        actions=[{"type": "notify"}],
    )

    # Match: priority == "high"
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"priority": "high"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: priority == "low"
    event["payload"]["priority"] = "low"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_neq_operator(db):
    """Test the 'neq' (not equal) operator."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Not spam",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.folder", "op": "neq", "value": "spam"},
        ],
        actions=[{"type": "tag", "value": "not-spam"}],
    )

    # Match: folder != "spam"
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"folder": "inbox"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: folder == "spam"
    event["payload"]["folder"] = "spam"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_operator(db):
    """Test the 'contains' operator (case-insensitive substring)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Urgent subject",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "urgent"},
        ],
        actions=[{"type": "notify"}],
    )

    # Match: case-insensitive substring
    for subject in ["URGENT: Meeting", "Please handle urgently", "urgent"]:
        event = {
            "id": "evt-1",
            "type": "email.received",
            "payload": {"subject": subject},
        }
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match: {subject}"

    # No match: substring not present
    event["payload"]["subject"] = "Normal email"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_contains_any_operator(db):
    """Test the 'contains_any' operator (matches if ANY value is present)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Marketing detector",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.body", "op": "contains_any",
             "value": ["unsubscribe", "opt out", "manage preferences"]},
        ],
        actions=[{"type": "tag", "value": "marketing"}],
    )

    # Match: any of the keywords present (case-insensitive)
    for body in ["Click to UNSUBSCRIBE", "opt out here", "Manage preferences at bottom"]:
        event = {
            "id": "evt-1",
            "type": "email.received",
            "payload": {"body": body},
        }
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match: {body}"

    # No match: none of the keywords
    event["payload"]["body"] = "Normal email content"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_in_operator(db):
    """Test the 'in' operator (value is member of list)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="VIP sender",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.from", "op": "in",
             "value": ["boss@company.com", "ceo@company.com", "founder@startup.com"]},
        ],
        actions=[{"type": "notify", "priority": "high"}],
    )

    # Match: sender in VIP list
    for sender in ["boss@company.com", "ceo@company.com", "founder@startup.com"]:
        event = {
            "id": "evt-1",
            "type": "email.received",
            "payload": {"from": sender},
        }
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match: {sender}"

    # No match: sender not in VIP list
    event["payload"]["from"] = "random@example.com"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_not_in_operator(db):
    """Test the 'not_in' operator (value is NOT in list)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="External sender",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.from", "op": "not_in",
             "value": ["internal@company.com", "team@company.com"]},
        ],
        actions=[{"type": "tag", "value": "external"}],
    )

    # Match: sender not in internal list
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"from": "external@example.com"},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: sender is in internal list
    event["payload"]["from"] = "internal@company.com"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_gt_lt_operators(db):
    """Test numeric comparison operators (gt, lt, gte, lte)."""
    engine = RulesEngine(db)

    # Rule: transactions > $500
    await engine.add_rule(
        name="Large transaction",
        trigger_event="finance.transaction",
        conditions=[
            {"field": "payload.amount", "op": "gt", "value": 500},
        ],
        actions=[{"type": "notify"}],
    )

    # Match: amount > 500
    event = {
        "id": "evt-1",
        "type": "finance.transaction",
        "payload": {"amount": 1000},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: amount <= 500
    event["payload"]["amount"] = 500
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # Rule: transactions < 10
    await engine.add_rule(
        name="Small transaction",
        trigger_event="finance.transaction",
        conditions=[
            {"field": "payload.amount", "op": "lt", "value": 10},
        ],
        actions=[{"type": "tag", "value": "micro"}],
    )

    event["payload"]["amount"] = 5
    actions = await engine.evaluate(event)
    # Should match the "Small transaction" rule, not "Large transaction"
    assert len(actions) == 1
    assert actions[0]["value"] == "micro"


@pytest.mark.asyncio
async def test_gte_lte_operators(db):
    """Test greater-than-or-equal and less-than-or-equal operators."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Valid age",
        trigger_event="user.profile.updated",
        conditions=[
            {"field": "payload.age", "op": "gte", "value": 18},
            {"field": "payload.age", "op": "lte", "value": 120},
        ],
        actions=[{"type": "tag", "value": "valid-age"}],
    )

    # Match: 18 <= age <= 120
    for age in [18, 50, 120]:
        event = {
            "id": "evt-1",
            "type": "user.profile.updated",
            "payload": {"age": age},
        }
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match age: {age}"

    # No match: age < 18 or age > 120
    event["payload"]["age"] = 17
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    event["payload"]["age"] = 121
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_exists_operator(db):
    """Test the 'exists' operator (field is present and not None)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Has attachments",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.attachments", "op": "exists"},
        ],
        actions=[{"type": "tag", "value": "has-attachments"}],
    )

    # Match: field exists and is not None
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"attachments": ["file1.pdf", "file2.jpg"]},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # Match: field exists and is empty list (still not None)
    event["payload"]["attachments"] = []
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: field is None
    event["payload"]["attachments"] = None
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # No match: field doesn't exist
    event["payload"] = {}
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_not_exists_operator(db):
    """Test the 'not_exists' operator (field is absent or None)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="No reply-to",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.reply_to", "op": "not_exists"},
        ],
        actions=[{"type": "tag", "value": "no-reply"}],
    )

    # Match: field is None
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"reply_to": None},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # Match: field doesn't exist
    event["payload"] = {}
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: field exists and is not None
    event["payload"]["reply_to"] = "reply@example.com"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_regex_operator(db):
    """Test the 'regex' operator (case-insensitive regex search)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Invoice subject",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "regex", "value": r"invoice #?\d+"},
        ],
        actions=[{"type": "tag", "value": "invoice"}],
    )

    # Match: regex pattern found (case-insensitive)
    for subject in ["Invoice #12345", "INVOICE 67890", "Your invoice #999"]:
        event = {
            "id": "evt-1",
            "type": "email.received",
            "payload": {"subject": subject},
        }
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match: {subject}"

    # No match: pattern not found
    event["payload"]["subject"] = "Receipt for your purchase"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_nested_field_resolution(db):
    """Test that nested field paths like 'payload.user.email' resolve correctly."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Admin user",
        trigger_event="user.action",
        conditions=[
            {"field": "payload.user.role", "op": "eq", "value": "admin"},
        ],
        actions=[{"type": "tag", "value": "admin-action"}],
    )

    # Match: nested field resolves to "admin"
    event = {
        "id": "evt-1",
        "type": "user.action",
        "payload": {
            "user": {
                "role": "admin",
                "email": "admin@company.com",
            },
        },
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: nested field resolves to different value
    event["payload"]["user"]["role"] = "user"
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_broken_field_path(db):
    """Test that broken field paths (missing intermediate keys) return None gracefully."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Deep field",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.user.profile.email", "op": "exists"},
        ],
        actions=[{"type": "tag", "value": "has-email"}],
    )

    # No match: intermediate key 'profile' doesn't exist
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {
            "user": {
                "name": "Test",
                # 'profile' key is missing
            },
        },
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # No match: 'user' key doesn't exist
    event["payload"] = {}
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_multiple_conditions_and_logic(db):
    """Test that all conditions must pass (AND logic)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Important email from boss",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.from", "op": "eq", "value": "boss@company.com"},
            {"field": "payload.subject", "op": "contains", "value": "urgent"},
            {"field": "payload.has_attachments", "op": "eq", "value": True},
        ],
        actions=[{"type": "notify", "priority": "high"}],
    )

    # Match: all three conditions pass
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {
            "from": "boss@company.com",
            "subject": "Urgent: Please review",
            "has_attachments": True,
        },
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # No match: first condition fails
    event["payload"]["from"] = "other@company.com"
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # No match: second condition fails (subject doesn't contain "urgent")
    event["payload"]["from"] = "boss@company.com"
    event["payload"]["subject"] = "Normal email"
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # No match: third condition fails (no attachments)
    event["payload"]["subject"] = "Urgent matter"
    event["payload"]["has_attachments"] = False
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_multiple_actions(db):
    """Test that rules can emit multiple actions."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Marketing email",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.body", "op": "contains", "value": "unsubscribe"},
        ],
        actions=[
            {"type": "tag", "value": "marketing"},
            {"type": "suppress"},
            {"type": "archive"},
        ],
    )

    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"body": "Click to unsubscribe"},
    }
    actions = await engine.evaluate(event)

    # Should receive all three actions
    assert len(actions) == 3
    assert actions[0]["type"] == "tag"
    assert actions[0]["value"] == "marketing"
    assert actions[1]["type"] == "suppress"
    assert actions[2]["type"] == "archive"

    # All actions should include rule metadata
    for action in actions:
        assert action["rule_name"] == "Marketing email"
        assert "rule_id" in action


@pytest.mark.asyncio
async def test_multiple_rules_trigger(db):
    """Test that multiple rules can trigger on the same event."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Tag all emails",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "tag", "value": "email"}],
    )

    await engine.add_rule(
        name="Tag VIP sender",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.from", "op": "eq", "value": "vip@company.com"},
        ],
        actions=[{"type": "tag", "value": "vip"}],
    )

    # Event from VIP sender should trigger both rules
    event = {
        "id": "evt-1",
        "type": "email.received",
        "payload": {"from": "vip@company.com"},
    }
    actions = await engine.evaluate(event)

    assert len(actions) == 2
    values = {a["value"] for a in actions}
    assert values == {"email", "vip"}


@pytest.mark.asyncio
async def test_rule_cache_reload(db):
    """Test that the rule cache auto-reloads after 60 seconds."""
    engine = RulesEngine(db)

    # Initial load
    await engine.add_rule(
        name="Rule 1",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "r1"}],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    actions = await engine.evaluate(event)
    assert len(actions) == 1
    assert actions[0]["value"] == "r1"

    # Add a new rule directly to the database (bypassing engine.add_rule)
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT INTO rules (id, name, trigger_event, conditions, actions, created_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "rule-2-id",
                "Rule 2",
                "test.event",
                json.dumps([]),
                json.dumps([{"type": "tag", "value": "r2"}]),
                "test",
            ),
        )

    # Force cache reload by setting cache timestamp to None
    engine._cache_loaded_at = None

    # Next evaluation should reload the cache and pick up the new rule
    actions = await engine.evaluate(event)
    assert len(actions) == 2
    values = {a["value"] for a in actions}
    assert values == {"r1", "r2"}


@pytest.mark.asyncio
async def test_remove_rule(db):
    """Test that removing a rule deactivates it (soft delete)."""
    engine = RulesEngine(db)

    rule_id = await engine.add_rule(
        name="Temporary rule",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "temp"}],
    )

    # Rule should trigger
    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # Remove the rule
    await engine.remove_rule(rule_id)

    # Rule should no longer trigger
    actions = await engine.evaluate(event)
    assert len(actions) == 0

    # Rule should still exist in database but with is_active=0
    all_rules = engine.get_all_rules()
    removed_rule = next(r for r in all_rules if r["id"] == rule_id)
    assert removed_rule["is_active"] == 0


@pytest.mark.asyncio
async def test_unknown_operator(db):
    """Test that unknown operators fail closed (return False)."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Unknown op rule",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.value", "op": "unknown_operator", "value": "test"},
        ],
        actions=[{"type": "tag", "value": "should-not-trigger"}],
    )

    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"value": "test"},
    }
    actions = await engine.evaluate(event)

    # Unknown operator should fail closed (not trigger)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_type_mismatch_graceful_failure(db):
    """Test that type mismatches (e.g., contains on non-string) don't crash."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="Type mismatch",
        trigger_event="test.event",
        conditions=[
            {"field": "payload.count", "op": "contains", "value": "123"},
        ],
        actions=[{"type": "tag", "value": "should-not-trigger"}],
    )

    # payload.count is an integer, not a string — "contains" should return False
    event = {
        "id": "evt-1",
        "type": "test.event",
        "payload": {"count": 12345},
    }
    actions = await engine.evaluate(event)
    assert len(actions) == 0


@pytest.mark.asyncio
async def test_default_rules_installation(db):
    """Test that default rules are installed idempotently."""
    # Install default rules
    await install_default_rules(db)

    engine = RulesEngine(db)
    rules = engine.get_all_rules()

    # Should have exactly 8 default rules
    assert len(rules) == 8
    rule_names = {r["name"] for r in rules}
    assert rule_names == {
        "Archive marketing emails",
        "Flag emails with attachments",
        "High priority: calendar conflict",
        "Alert on large transactions",
        "Notify on urgent emails",
        "Notify on direct reply requests",
        "Notify on connector errors",
        "Alert on degraded connector",
    }

    # All default rules should be created by "system"
    for rule in rules:
        assert rule["created_by"] == "system"

    # Running install again should be idempotent (no duplicates)
    await install_default_rules(db)
    rules_after_second_install = engine.get_all_rules()
    assert len(rules_after_second_install) == 8


@pytest.mark.asyncio
async def test_trigger_count_tracking(db):
    """Test that times_triggered increments correctly."""
    engine = RulesEngine(db)

    rule_id = await engine.add_rule(
        name="Counter test",
        trigger_event="test.event",
        conditions=[],
        actions=[{"type": "tag", "value": "test"}],
    )

    # Trigger the rule 5 times
    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    for _ in range(5):
        await engine.evaluate(event)

    # Check that times_triggered incremented
    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT times_triggered FROM rules WHERE id = ?",
            (rule_id,)
        ).fetchone()
        assert row["times_triggered"] == 5


@pytest.mark.asyncio
async def test_empty_conditions_always_match(db):
    """Test that rules with empty conditions array always match their trigger."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="No conditions",
        trigger_event="test.event",
        conditions=[],  # Empty conditions list
        actions=[{"type": "tag", "value": "always"}],
    )

    # Any event with matching type should trigger
    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    actions = await engine.evaluate(event)
    assert len(actions) == 1

    # Even with no payload
    event = {"id": "evt-2", "type": "test.event"}
    actions = await engine.evaluate(event)
    assert len(actions) == 1


@pytest.mark.asyncio
async def test_rule_metadata_in_actions(db):
    """Test that all actions include rule_id and rule_name metadata."""
    engine = RulesEngine(db)

    rule_id = await engine.add_rule(
        name="Metadata test",
        trigger_event="test.event",
        conditions=[],
        actions=[
            {"type": "notify"},
            {"type": "tag", "value": "test"},
        ],
    )

    event = {"id": "evt-1", "type": "test.event", "payload": {}}
    actions = await engine.evaluate(event)

    assert len(actions) == 2
    for action in actions:
        assert action["rule_id"] == rule_id
        assert action["rule_name"] == "Metadata test"
        assert "type" in action


@pytest.mark.asyncio
async def test_complex_glob_patterns(db):
    """Test complex glob patterns with multiple wildcards."""
    engine = RulesEngine(db)

    await engine.add_rule(
        name="User model events",
        trigger_event="usermodel.*.*",
        conditions=[],
        actions=[{"type": "tag", "value": "usermodel"}],
    )

    # Should match usermodel.<category>.<action> patterns
    for event_type in [
        "usermodel.prediction.generated",
        "usermodel.signal_profile.updated",
        "usermodel.episode.stored",
    ]:
        event = {"id": "evt-1", "type": event_type, "payload": {}}
        actions = await engine.evaluate(event)
        assert len(actions) == 1, f"Failed to match: {event_type}"

    # Should NOT match different patterns
    for event_type in ["usermodel.single", "other.prediction.generated"]:
        event = {"id": "evt-1", "type": event_type, "payload": {}}
        actions = await engine.evaluate(event)
        assert len(actions) == 0, f"Incorrectly matched: {event_type}"
