"""
Life OS — Notification Delivery Pipeline Integration Tests

Verifies the end-to-end flow from rule evaluation through notification
creation to API retrieval:

  1. Event arrives → rules_engine.evaluate(event) returns actions
  2. _execute_rule_action() with type='notify' calls notification_manager.create_notification()
  3. NotificationManager stores the notification in state.db
  4. GET /api/notifications retrieves stored notifications

These tests bridge the gap between the existing RulesEngine unit tests
(28 tests) and NotificationManager unit tests (46 tests) by verifying
the two services work together end-to-end through the LifeOS pipeline.
"""

import json
import uuid
from datetime import datetime, timezone

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
    """Create a LifeOS instance with the master_event_handler wired up.

    Uses the same injection pattern as test_master_event_handler_pipeline.py.
    Registers event handlers so the pipeline is fully functional.
    """
    from main import LifeOS

    los = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=lifeos_config,
    )

    await los._register_event_handlers()

    # Extract the handler from the mock so tests can invoke it directly.
    handler = event_bus.subscribe_all.call_args[0][0]
    los.master_event_handler = handler

    return los


def _make_event(event_type: str = "email.received", **payload_overrides) -> dict:
    """Build a well-formed event dict with sensible defaults.

    Args:
        event_type: The event type string (e.g., "email.received").
        **payload_overrides: Fields merged into the default payload.

    Returns:
        A complete event dict ready for master_event_handler.
    """
    payload = {
        "from_address": "alice@example.com",
        "to_addresses": ["user@example.com"],
        "subject": "Meeting tomorrow",
        "body_plain": "Hi, can we meet at 3pm tomorrow to discuss the project?",
        "message_id": f"<test-msg-{uuid.uuid4().hex[:8]}>",
    }
    payload.update(payload_overrides)

    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": payload,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Test 1: Notify action creates a notification in the database
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_action_creates_notification(lifeos, db):
    """A rule with a notify action should create a notification in state.db.

    Verifies the core path: _execute_rule_action(notify) →
    notification_manager.create_notification() → INSERT into notifications.
    """
    # Add a rule that notifies on all email.received events
    await lifeos.rules_engine.add_rule(
        name="Notify on all emails",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "priority": "high", "message": "Test alert"}],
    )

    event = _make_event(
        "email.received",
        subject="Integration test notification",
        body_plain="This email should trigger a notification.",
    )

    await lifeos.master_event_handler(event)

    # Verify the notification was created in state.db
    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1, "A notification should be created for the event"
    notif = dict(notifs[0])

    # Verify notification fields
    assert notif["priority"] == "high", "Priority should match the rule action"
    assert notif["source_event_id"] == event["id"], "source_event_id should link to the event"
    assert notif["title"], "Notification should have a title"


# ---------------------------------------------------------------------------
# Test 2: Suppressed event skips notification creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suppressed_event_skips_notification(lifeos, db):
    """When event['_suppressed'] is True, _execute_rule_action should skip
    creating a notification even if the action type is 'notify'.

    This tests the suppress→notify ordering documented in the pipeline:
    suppress actions run first, setting _suppressed=True, and subsequent
    notify actions check this flag before creating notifications.
    """
    # Add both a suppress rule and a notify rule matching the same event
    await lifeos.rules_engine.add_rule(
        name="Suppress suppression-test emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "suppression-test"},
        ],
        actions=[{"type": "suppress"}],
    )

    await lifeos.rules_engine.add_rule(
        name="Notify on suppression-test emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "suppression-test"},
        ],
        actions=[{"type": "notify", "message": "Should not appear"}],
    )

    event = _make_event(
        "email.received",
        subject="This is a suppression-test email",
        body_plain="The suppress rule should prevent the notification from being created.",
    )

    await lifeos.master_event_handler(event)

    # Verify the event was marked as suppressed
    assert event.get("_suppressed") is True, "Event should be marked as suppressed"

    # Verify NO notification was created
    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) == 0, "Suppressed event should NOT generate a notification"


# ---------------------------------------------------------------------------
# Test 3: Rule evaluation triggers notification for matching event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rule_evaluation_triggers_notification_for_matching_event(lifeos, db):
    """Create a rule that matches email.received events with a specific subject,
    evaluate the event, and verify both:
    1. The rules engine returns a notify action
    2. The notification actually exists in the database after pipeline execution
    """
    # Add a rule with a condition matching specific subject keywords
    await lifeos.rules_engine.add_rule(
        name="Alert on urgent emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "critical-alert"},
        ],
        actions=[{"type": "notify", "priority": "high"}],
    )

    event = _make_event(
        "email.received",
        subject="This is a critical-alert from the monitoring system",
        body_plain="Server CPU usage exceeded 90% threshold.",
    )

    # Step 1: Verify the rules engine returns the correct action
    actions = await lifeos.rules_engine.evaluate(event)
    notify_actions = [a for a in actions if a["type"] == "notify"]
    assert len(notify_actions) >= 1, "Rules engine should return a notify action"
    assert notify_actions[0]["priority"] == "high"

    # Step 2: Process through the full pipeline and verify notification exists
    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1, "Notification should exist after pipeline execution"
    notif = dict(notifs[0])
    assert notif["priority"] == "high"


# ---------------------------------------------------------------------------
# Test 4: Notification retrievable via API after creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_retrievable_via_api_after_creation(lifeos, db):
    """After creating a notification via the pipeline, GET /api/notifications
    should return the notification in the response JSON.
    """
    from fastapi.testclient import TestClient
    from web.app import create_web_app

    # Add a rule that notifies on all email.received events
    await lifeos.rules_engine.add_rule(
        name="Notify for API test",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "priority": "high"}],
    )

    event = _make_event(
        "email.received",
        subject="API retrieval test",
        body_plain="This notification should be retrievable via the API.",
    )

    await lifeos.master_event_handler(event)

    # Create the FastAPI test client
    app = create_web_app(lifeos)
    client = TestClient(app)

    # Retrieve notifications via the API
    response = client.get("/api/notifications")
    assert response.status_code == 200

    data = response.json()
    assert "notifications" in data

    # Find the notification for our event
    matching = [
        n for n in data["notifications"]
        if n.get("source_event_id") == event["id"]
    ]
    assert len(matching) >= 1, (
        f"Notification for event {event['id']} should be retrievable via "
        f"GET /api/notifications. Got {len(data['notifications'])} notifications total."
    )

    notif = matching[0]
    assert notif["priority"] == "high"
    assert notif["source_event_id"] == event["id"]


# ---------------------------------------------------------------------------
# Test 5: Multiple rules create multiple notifications
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_rules_create_multiple_notifications(lifeos, db):
    """An event matching 2 rules with notify actions should create 2 separate
    notifications.
    """
    # Add two rules that both match email.received events
    await lifeos.rules_engine.add_rule(
        name="Rule A: Notify on emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "multi-rule-test"},
        ],
        actions=[{"type": "notify", "priority": "high", "message": "Alert A"}],
    )

    await lifeos.rules_engine.add_rule(
        name="Rule B: Also notify on emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "multi-rule-test"},
        ],
        actions=[{"type": "notify", "priority": "normal", "message": "Alert B"}],
    )

    event = _make_event(
        "email.received",
        subject="This is a multi-rule-test email",
        body_plain="Both rules should fire and create separate notifications.",
    )

    await lifeos.master_event_handler(event)

    # Verify both notifications were created
    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) == 2, (
        f"Expected 2 notifications from 2 matching rules, got {len(notifs)}"
    )

    # Verify different priorities (confirming they came from different rules)
    priorities = {dict(n)["priority"] for n in notifs}
    assert "high" in priorities, "One notification should have high priority"
    assert "normal" in priorities, "One notification should have normal priority"


# ---------------------------------------------------------------------------
# Test 6: Notify action infers domain from event type
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_action_infers_domain_from_event_type(lifeos, db):
    """When the event has no explicit domain in metadata, _execute_rule_action
    should infer the domain from the event type (e.g., 'email.received' → 'email').

    This verifies _infer_domain_from_event_type() is called in the notify path.
    """
    await lifeos.rules_engine.add_rule(
        name="Notify for domain inference test",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "priority": "normal"}],
    )

    event = _make_event(
        "email.received",
        subject="Domain inference test",
        body_plain="The notification domain should be inferred as 'email'.",
    )
    # Ensure no explicit domain in metadata
    event["metadata"] = {}

    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1, "Notification should be created"
    notif = dict(notifs[0])
    assert notif["domain"] == "email", (
        f"Domain should be inferred as 'email' from event type 'email.received', "
        f"got '{notif['domain']}'"
    )


# ---------------------------------------------------------------------------
# Test 7: Notification content built from event payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_content_built_from_event_payload(lifeos, db):
    """Verify _build_notification_content() extracts meaningful title and body
    from the event payload (subject for title, body_plain with sender for body).
    """
    await lifeos.rules_engine.add_rule(
        name="Notify for content test",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "priority": "normal"}],
    )

    event = _make_event(
        "email.received",
        from_address="boss@company.com",
        subject="Q1 Budget Review",
        body_plain="Please review the attached budget document before Friday.",
    )

    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1
    notif = dict(notifs[0])

    # Title should come from the email subject
    assert "Q1 Budget Review" in notif["title"], (
        f"Title should contain the email subject, got '{notif['title']}'"
    )

    # Body should include the sender (for email events, _build_notification_content
    # prepends "From: <address>")
    assert "boss@company.com" in notif["body"], (
        f"Body should include the sender address, got '{notif['body']}'"
    )


# ---------------------------------------------------------------------------
# Test 8: Non-matching rule does not create notification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_matching_rule_does_not_create_notification(lifeos, db):
    """A rule whose conditions don't match the event should not create a
    notification. This verifies the condition evaluation path.
    """
    await lifeos.rules_engine.add_rule(
        name="Only match finance events",
        trigger_event="finance.transaction.new",
        conditions=[
            {"field": "payload.amount", "op": "gt", "value": 500},
        ],
        actions=[{"type": "notify", "priority": "high"}],
    )

    # Send an email event (not a finance event) — rule should NOT match
    event = _make_event(
        "email.received",
        subject="Non-matching event test",
        body_plain="This should not trigger the finance rule.",
    )

    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) == 0, "Non-matching rule should not create a notification"


# ---------------------------------------------------------------------------
# Test 9: Calendar event domain inferred correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calendar_event_domain_inferred_correctly(lifeos, db):
    """Verify domain inference works for non-email event types (e.g., calendar)."""
    await lifeos.rules_engine.add_rule(
        name="Notify on calendar conflicts",
        trigger_event="calendar.conflict.detected",
        conditions=[],
        actions=[{"type": "notify", "priority": "high"}],
    )

    event = {
        "id": str(uuid.uuid4()),
        "type": "calendar.conflict.detected",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "high",
        "payload": {
            "summary": "Meeting overlap detected",
            "description": "Two meetings at 2pm on Tuesday",
        },
        "metadata": {},
    }

    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1, "Notification should be created for calendar conflict"
    notif = dict(notifs[0])
    assert notif["domain"] == "calendar", (
        f"Domain should be 'calendar' for calendar.conflict.detected, "
        f"got '{notif['domain']}'"
    )


# ---------------------------------------------------------------------------
# Test 10: Explicit metadata domain takes precedence over inference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_metadata_domain_takes_precedence(lifeos, db):
    """When event metadata includes an explicit domain, it should be used
    instead of inferring from the event type.
    """
    await lifeos.rules_engine.add_rule(
        name="Notify for domain precedence test",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "priority": "normal"}],
    )

    event = _make_event(
        "email.received",
        subject="Domain precedence test",
        body_plain="Metadata domain should take precedence.",
    )
    # Set explicit domain in metadata
    event["metadata"] = {"domain": "prediction"}

    await lifeos.master_event_handler(event)

    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?",
            (event["id"],),
        ).fetchall()

    assert len(notifs) >= 1
    notif = dict(notifs[0])
    assert notif["domain"] == "prediction", (
        f"Explicit metadata domain 'prediction' should take precedence over "
        f"inferred domain 'email', got '{notif['domain']}'"
    )
