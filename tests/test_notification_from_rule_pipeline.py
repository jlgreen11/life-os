"""
Life OS — Rule-to-Notification Pipeline Integration Tests

Verifies the FULL chain from event arrival → rule evaluation → notify action
execution → notification creation → notification visible in get_pending().

Individual components (rules engine, notification manager) are well-tested
in isolation. These tests verify the *glue* between them — the pipeline that
main.py:_execute_rule_action() implements — using real temporary databases
to catch integration bugs that unit tests miss.

Each test instantiates a RulesEngine and NotificationManager with real
(temporary) SQLite databases, installs default rules, evaluates an event,
and then mimics the _execute_rule_action() logic to create notifications
from matching rule actions.
"""

import uuid
from datetime import datetime, timezone

import pytest

from services.notification_manager.manager import NotificationManager
from services.rules_engine.engine import RulesEngine, install_default_rules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str, source: str = "test", priority: str = "normal",
                payload: dict | None = None, metadata: dict | None = None) -> dict:
    """Build a well-formed event dict for pipeline testing.

    Args:
        event_type: The event type string (e.g., "system.connector.error").
        source: Event source identifier.
        priority: Event priority level.
        payload: Event payload dict (defaults to empty).
        metadata: Event metadata dict (defaults to empty).

    Returns:
        A complete event dict matching the NATS envelope format.
    """
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": priority,
        "payload": payload or {},
        "metadata": metadata or {},
    }


def _infer_domain(event_type: str) -> str:
    """Infer notification domain from event type, matching main.py logic.

    Args:
        event_type: The event type string.

    Returns:
        The inferred domain (first segment before the dot).
    """
    if not event_type or "." not in event_type:
        return "system"
    return event_type.split(".")[0]


def _build_notification_content(event: dict, action: dict) -> tuple[str, str]:
    """Build title and body from event + action, matching main.py logic.

    Mirrors LifeOS._build_notification_content() so tests exercise the same
    content-generation logic as the production pipeline.

    Args:
        event: The event dict with payload and type fields.
        action: The rule action dict with rule_name.

    Returns:
        A (title, body) tuple.
    """
    payload = event.get("payload", {})

    # Title: pick the most descriptive field available
    title = (
        payload.get("subject")
        or payload.get("summary")
        or payload.get("merchant_name")
        or f"Rule: {action.get('rule_name', 'Unknown')}"
    )

    # Finance events: append amount to merchant name
    if payload.get("merchant_name") and payload.get("amount") is not None:
        title = f"{payload['merchant_name']} — ${payload['amount']}"

    # Body: try payload fields in order of usefulness
    body = (
        payload.get("snippet")
        or payload.get("body_plain")
        or payload.get("body")
        or payload.get("description")
        or payload.get("content")
        or ""
    )

    # Email events: prepend the sender
    from_address = payload.get("from_address") or payload.get("from")
    if from_address and event.get("type", "").startswith("email"):
        body = f"From: {from_address}\n{body}" if body else f"From: {from_address}"

    # Truncate to 200 characters
    if len(body) > 200:
        body = body[:197] + "..."

    return title, body


async def _execute_notify_actions(notification_manager: NotificationManager,
                                  actions: list[dict], event: dict):
    """Execute notify actions from the rules engine, mimicking main.py logic.

    For each action with type='notify', creates a notification via the
    notification manager using the same content-building logic as
    LifeOS._execute_rule_action().

    Args:
        notification_manager: The NotificationManager instance.
        actions: List of actions returned by rules_engine.evaluate().
        event: The original event dict.
    """
    for action in actions:
        if action["type"] != "notify":
            continue

        # Check suppress flag (matches main.py behavior)
        if event.get("_suppressed"):
            continue

        domain = event.get("metadata", {}).get("domain")
        if not domain:
            domain = _infer_domain(event.get("type", ""))

        title, body = _build_notification_content(event, action)
        await notification_manager.create_notification(
            title=title,
            body=body,
            priority=action.get("priority", "normal"),
            source_event_id=event.get("id"),
            domain=domain,
        )


# ---------------------------------------------------------------------------
# Test 1: Connector error → rule match → notification created and visible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connector_error_produces_notification(db, event_bus):
    """A system.connector.error event must flow through the full pipeline:
    rule evaluation → notify action → notification creation → visible in get_pending().

    This is the most critical integration test: it verifies the glue between
    the rules engine and notification manager that main.py:_execute_rule_action
    implements.
    """
    # Set up services with real temp DBs
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")

    # Install default rules (includes "Notify on connector errors")
    await install_default_rules(db)

    # Create a connector error event
    event = _make_event(
        event_type="system.connector.error",
        source="google",
        priority="high",
        payload={
            "connector_id": "google",
            "error": "Authentication failed: token expired",
            "error_type": "authentication",
            "display_name": "Google",
        },
    )

    # Evaluate rules against the event
    actions = await rules_engine.evaluate(event)

    # Assert at least one notify action with high priority
    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) >= 1, (
        f"Expected at least one notify action for connector error, got: {actions}"
    )
    assert any(a["priority"] == "high" for a in notify_actions), (
        f"Expected a high-priority notify action, got priorities: "
        f"{[a.get('priority') for a in notify_actions]}"
    )

    # Execute notify actions (mimicking _execute_rule_action)
    await _execute_notify_actions(notification_manager, actions, event)

    # Verify notification is visible in get_pending()
    pending = notification_manager.get_pending()
    assert len(pending) >= 1, (
        "Expected at least one pending notification after connector error pipeline"
    )

    # Verify the notification's title references the rule name (since connector
    # error payloads don't have 'subject' or 'summary' fields, the title falls
    # back to "Rule: <rule_name>")
    titles = [n["title"].lower() for n in pending]
    assert any("connector" in t or "error" in t for t in titles), (
        f"Expected notification title to reference connector/error, got: {titles}"
    )


# ---------------------------------------------------------------------------
# Test 2: Urgent email → notify + tag actions → notification visible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_urgent_email_produces_notification(db, event_bus):
    """An email with urgent keywords must produce both a notify action (high
    priority) and a tag action ('urgent'), and the notification must appear
    in get_pending().
    """
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")
    await install_default_rules(db)

    event = _make_event(
        event_type="email.received",
        source="google",
        priority="normal",
        payload={
            "subject": "URGENT: Action Required - Account Review",
            "from_address": "boss@example.com",
            "body_plain": "Please review the attached document ASAP.",
            "has_attachments": False,
        },
    )

    actions = await rules_engine.evaluate(event)

    # Should produce a notify action with high priority
    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) >= 1, (
        f"Expected at least one notify action for urgent email, got: {actions}"
    )
    assert any(a["priority"] == "high" for a in notify_actions), (
        f"Expected high-priority notify, got: {[a.get('priority') for a in notify_actions]}"
    )

    # Should also produce a tag action with value "urgent"
    tag_actions = [a for a in actions if a["type"] == "tag"]
    tag_values = [a.get("value") for a in tag_actions]
    assert "urgent" in tag_values, (
        f"Expected 'urgent' tag action, got tag values: {tag_values}"
    )

    # Execute notify actions and verify notification appears
    await _execute_notify_actions(notification_manager, actions, event)

    pending = notification_manager.get_pending()
    assert len(pending) >= 1, (
        "Expected at least one pending notification after urgent email pipeline"
    )

    # Verify the notification title uses the email subject
    assert any("URGENT" in n["title"] for n in pending), (
        f"Expected notification title to contain 'URGENT', got: "
        f"{[n['title'] for n in pending]}"
    )


# ---------------------------------------------------------------------------
# Test 3: Marketing email → suppress + tag, NO notification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_marketing_email_suppressed_no_notification(db, event_bus):
    """A marketing email (contains 'unsubscribe' in body) must produce suppress
    and tag actions but NO notify action. get_pending() must be empty.
    """
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")
    await install_default_rules(db)

    event = _make_event(
        event_type="email.received",
        source="google",
        priority="normal",
        payload={
            "subject": "Weekly Newsletter - Spring Collection",
            "from_address": "marketing@store.com",
            "body_plain": "Check out our new items! Click to unsubscribe from this list.",
            "has_attachments": False,
        },
    )

    actions = await rules_engine.evaluate(event)

    # Should have suppress and tag actions
    action_types = [a["type"] for a in actions]
    assert "suppress" in action_types, (
        f"Expected a suppress action for marketing email, got: {action_types}"
    )
    assert "tag" in action_types, (
        f"Expected a tag action for marketing email, got: {action_types}"
    )

    # Marketing email might also match other rules (e.g., "reply requests"
    # if body contains matching phrases). But the suppress action means
    # the _execute_rule_action logic in main.py sets _suppressed=True before
    # processing notify actions. Mimic that: execute suppress first, then
    # check that no notify actions fire.
    # In main.py, suppress actions are processed first (the suppress action
    # sets event["_suppressed"] = True), then notify actions check this flag.
    for action in actions:
        if action["type"] == "suppress":
            event["_suppressed"] = True

    # Now execute remaining actions — notify should be skipped due to suppress
    await _execute_notify_actions(notification_manager, actions, event)

    # Verify no notifications were created
    pending = notification_manager.get_pending()
    assert len(pending) == 0, (
        f"Expected no pending notifications for suppressed marketing email, "
        f"got {len(pending)}: {[n['title'] for n in pending]}"
    )


# ---------------------------------------------------------------------------
# Test 4: Notify action includes rule metadata (rule_id and rule_name)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_includes_rule_metadata(db, event_bus):
    """The rules engine must attach rule_id and rule_name to every action it
    returns. This metadata is critical for audit trails and the rules
    management UI.
    """
    rules_engine = RulesEngine(db)
    await install_default_rules(db)

    event = _make_event(
        event_type="system.connector.error",
        source="google",
        payload={
            "connector_id": "google",
            "error": "Auth failed",
            "error_type": "authentication",
        },
    )

    actions = await rules_engine.evaluate(event)

    # Every action must have rule_id and rule_name
    for action in actions:
        assert "rule_id" in action, (
            f"Action missing rule_id: {action}"
        )
        assert "rule_name" in action, (
            f"Action missing rule_name: {action}"
        )
        assert action["rule_id"] is not None, (
            f"Action rule_id is None: {action}"
        )
        assert action["rule_name"] is not None, (
            f"Action rule_name is None: {action}"
        )

    # Specifically for connector error, the rule_name should match
    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert any(a["rule_name"] == "Notify on connector errors" for a in notify_actions), (
        f"Expected 'Notify on connector errors' rule_name, got: "
        f"{[a.get('rule_name') for a in notify_actions]}"
    )


# ---------------------------------------------------------------------------
# Test 5: High-priority notification delivered immediately (not batched)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_priority_notification_delivered_immediately(db, event_bus):
    """A high-priority notification from a connector error must be delivered
    immediately (status='delivered'), not left as 'pending'. This confirms
    that critical alerts bypass batching regardless of notification_mode.
    """
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")
    await install_default_rules(db)

    event = _make_event(
        event_type="system.connector.error",
        source="proton_mail",
        priority="high",
        payload={
            "connector_id": "proton_mail",
            "error": "IMAP connection refused",
            "error_type": "connection",
        },
    )

    actions = await rules_engine.evaluate(event)
    await _execute_notify_actions(notification_manager, actions, event)

    # get_pending() returns notifications with status IN ('pending', 'delivered')
    pending = notification_manager.get_pending()
    assert len(pending) >= 1, (
        "Expected at least one notification after connector error"
    )

    # High-priority notifications should be delivered immediately, which means
    # the notification manager calls _deliver_notification() setting status='delivered'
    # (rather than leaving it as 'pending' for batch delivery).
    delivered = [n for n in pending if n["status"] == "delivered"]
    assert len(delivered) >= 1, (
        f"Expected high-priority notification to have status='delivered', "
        f"got statuses: {[n['status'] for n in pending]}"
    )


# ---------------------------------------------------------------------------
# Test 6: Calendar conflict → notification visible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calendar_conflict_produces_notification(db, event_bus):
    """A calendar.conflict.detected event should trigger the 'High priority:
    calendar conflict' rule and produce a visible notification.
    """
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")
    await install_default_rules(db)

    event = _make_event(
        event_type="calendar.conflict.detected",
        source="caldav",
        priority="high",
        payload={
            "summary": "Team standup conflicts with dentist appointment",
            "conflicting_events": ["standup-123", "dentist-456"],
        },
    )

    actions = await rules_engine.evaluate(event)

    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) >= 1, (
        f"Expected notify action for calendar conflict, got: {actions}"
    )

    await _execute_notify_actions(notification_manager, actions, event)

    pending = notification_manager.get_pending()
    assert len(pending) >= 1, (
        "Expected notification after calendar conflict"
    )

    # Title should use the summary from the payload
    assert any("standup" in n["title"].lower() or "conflict" in n["title"].lower()
               for n in pending), (
        f"Expected notification title to reference the conflict, got: "
        f"{[n['title'] for n in pending]}"
    )


# ---------------------------------------------------------------------------
# Test 7: Large transaction → notify + tag pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_large_transaction_produces_notification(db, event_bus):
    """A finance.transaction.new event over $500 should trigger the 'Alert on
    large transactions' rule, producing both a notify and tag action.
    """
    rules_engine = RulesEngine(db)
    notification_manager = NotificationManager(db, event_bus, config={}, timezone="UTC")
    await install_default_rules(db)

    event = _make_event(
        event_type="finance.transaction.new",
        source="plaid",
        priority="normal",
        payload={
            "merchant_name": "Best Buy",
            "amount": 1299.99,
            "category": "electronics",
            "account_id": "checking-001",
        },
    )

    actions = await rules_engine.evaluate(event)

    # Should have both notify and tag actions
    notify_actions = [a for a in actions if a["type"] == "notify"]
    tag_actions = [a for a in actions if a["type"] == "tag"]
    assert len(notify_actions) >= 1, f"Expected notify action, got: {actions}"
    assert any(a.get("value") == "large-transaction" for a in tag_actions), (
        f"Expected 'large-transaction' tag, got: {tag_actions}"
    )

    await _execute_notify_actions(notification_manager, actions, event)

    pending = notification_manager.get_pending()
    assert len(pending) >= 1, (
        "Expected notification after large transaction"
    )

    # Title should include the merchant name and amount (from _build_notification_content)
    assert any("Best Buy" in n["title"] for n in pending), (
        f"Expected 'Best Buy' in notification title, got: "
        f"{[n['title'] for n in pending]}"
    )
