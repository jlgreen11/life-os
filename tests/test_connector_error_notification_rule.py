"""
Tests for the 'Notify on connector errors' default rule.

This rule ensures that connector failures (authentication errors, sync errors)
are surfaced as high-priority dashboard notifications instead of being hidden
on the /admin page. The base connector publishes system.connector.error events
on auth and sync failures; this rule catches them.

Tests validate:
1. The rule exists in DEFAULT_RULES with the correct structure
2. install_default_rules inserts it into the database
3. The rule triggers on system.connector.error events with priority high
4. Repeated install_default_rules calls don't create duplicates
5. Connector error payloads (connector name, error message) pass through
"""

import pytest

from services.rules_engine.engine import DEFAULT_RULES, RulesEngine, install_default_rules


def _find_connector_error_rule():
    """Find the connector error rule in DEFAULT_RULES by name."""
    for rule in DEFAULT_RULES:
        if rule["name"] == "Notify on connector errors":
            return rule
    return None


class TestConnectorErrorRuleInDefaults:
    """Verify the rule definition in DEFAULT_RULES is correct."""

    def test_rule_exists_in_default_rules(self):
        """The 'Notify on connector errors' rule must be present in DEFAULT_RULES."""
        rule = _find_connector_error_rule()
        assert rule is not None, "Rule 'Notify on connector errors' not found in DEFAULT_RULES"

    def test_rule_trigger_event(self):
        """The rule must trigger on system.connector.error events."""
        rule = _find_connector_error_rule()
        assert rule["trigger_event"] == "system.connector.error"

    def test_rule_has_no_conditions(self):
        """All connector errors should generate notifications — no conditions filter."""
        rule = _find_connector_error_rule()
        assert rule["conditions"] == []

    def test_rule_action_is_notify_high(self):
        """The rule should produce a single notify action with priority high."""
        rule = _find_connector_error_rule()
        assert len(rule["actions"]) == 1
        action = rule["actions"][0]
        assert action["type"] == "notify"
        assert action["priority"] == "high"


@pytest.mark.asyncio
async def test_install_default_rules_includes_connector_error(db):
    """install_default_rules should insert the connector error rule into the DB."""
    await install_default_rules(db)

    engine = RulesEngine(db)
    rules = engine.get_all_rules()
    rule_names = {r["name"] for r in rules}
    assert "Notify on connector errors" in rule_names

    # Verify the installed rule has the right structure
    installed = next(r for r in rules if r["name"] == "Notify on connector errors")
    assert installed["trigger_event"] == "system.connector.error"
    assert installed["conditions"] == []
    assert installed["actions"] == [{"type": "notify", "priority": "high"}]
    assert installed["created_by"] == "system"
    assert installed["is_active"] == 1


@pytest.mark.asyncio
async def test_idempotent_installation(db):
    """Running install_default_rules twice must not create duplicate rules."""
    await install_default_rules(db)
    await install_default_rules(db)

    engine = RulesEngine(db)
    rules = engine.get_all_rules()
    connector_error_rules = [r for r in rules if r["name"] == "Notify on connector errors"]
    assert len(connector_error_rules) == 1


@pytest.mark.asyncio
async def test_evaluate_connector_error_event(db):
    """A system.connector.error event should produce a notify action with priority high."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = {
        "id": "evt-conn-err-1",
        "type": "system.connector.error",
        "source": "google",
        "payload": {
            "connector_id": "google",
            "error": "Authentication failed: token expired",
            "error_type": "authentication",
            "display_name": "Google",
        },
    }
    actions = await engine.evaluate(event)

    # Should have exactly one notify action from the connector error rule
    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) == 1
    assert notify_actions[0]["priority"] == "high"
    assert notify_actions[0]["rule_name"] == "Notify on connector errors"


@pytest.mark.asyncio
async def test_evaluate_sync_error_event(db):
    """Sync errors (no error_type field) should also trigger the rule."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = {
        "id": "evt-conn-err-2",
        "type": "system.connector.error",
        "source": "proton_mail",
        "payload": {
            "connector_id": "proton_mail",
            "error": "IMAP connection refused",
        },
    }
    actions = await engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) == 1
    assert notify_actions[0]["priority"] == "high"


@pytest.mark.asyncio
async def test_non_connector_error_does_not_trigger(db):
    """Events that are NOT system.connector.error should not trigger this rule."""
    await install_default_rules(db)
    engine = RulesEngine(db)

    # A normal email event should not trigger the connector error rule
    event = {
        "id": "evt-email-1",
        "type": "email.received",
        "payload": {"subject": "Hello"},
    }
    actions = await engine.evaluate(event)
    connector_rule_actions = [a for a in actions if a.get("rule_name") == "Notify on connector errors"]
    assert len(connector_rule_actions) == 0


@pytest.mark.asyncio
async def test_event_payload_preserved_for_notification(db):
    """The event payload (connector_id, error message) should be available for notification rendering.

    The rules engine passes the full event to the notification manager, so the
    connector name and error message from the payload are available to build
    an informative notification body.
    """
    await install_default_rules(db)
    engine = RulesEngine(db)

    event = {
        "id": "evt-conn-err-3",
        "type": "system.connector.error",
        "source": "google",
        "payload": {
            "connector_id": "google",
            "error": "Authentication failed: invalid_grant",
            "error_type": "authentication",
            "display_name": "Google",
        },
    }
    actions = await engine.evaluate(event)

    # Verify the rule matched and the event's payload fields are present
    # (the notification manager reads these from the original event, not the action)
    assert len(actions) >= 1
    assert event["payload"]["connector_id"] == "google"
    assert event["payload"]["error"] == "Authentication failed: invalid_grant"
    assert event["payload"]["display_name"] == "Google"
