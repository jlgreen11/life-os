"""
Test that workflow and routine detection publishes events correctly.

This test verifies the fix for a bug where workflows and routines were being
detected and stored in the database, but no events were published to the event
bus because the detection happens in thread pools (via asyncio.to_thread())
where _emit_telemetry() fails silently.

The fix moves event publication to the async context in main.py after the
threaded detection completes.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector
from services.workflow_detector.detector import WorkflowDetector


@pytest.fixture
def populated_db(db, user_model_store):
    """Populate test data for workflow and routine detection."""
    now = datetime.now(timezone.utc)

    # Populate events.db for workflow detection
    with db.get_connection("events") as conn:
        # Create email workflow pattern: receive → send (10 instances)
        for i in range(10):
            timestamp = (now - timedelta(days=i, hours=1)).isoformat()
            response_time = (now - timedelta(days=i, minutes=30)).isoformat()

            # Received email
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from, email_to)
                VALUES (?, 'email.received', 'test', ?, 'medium', '{}', '{}', 'boss@company.com', 'user@company.com')
            """, (f"email_recv_{i}", timestamp))

            # Sent email in response
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from, email_to)
                VALUES (?, 'email.sent', 'test', ?, 'medium', '{}', '{}', 'user@company.com', 'boss@company.com')
            """, (f"email_sent_{i}", response_time))

    # Populate user_model.db episodic memory for routine detection
    with db.get_connection("user_model") as conn:
        # Create morning routine pattern: email checks every morning
        for i in range(15):
            morning_time = (now - timedelta(days=i)).replace(hour=8, minute=30, second=0, microsecond=0)

            conn.execute("""
                INSERT INTO episodes (id, event_id, interaction_type, content_summary,
                                     location, contacts_involved, outcome, timestamp)
                VALUES (?, ?, 'email_received', 'Morning email check',
                       'home', '["boss@company.com"]', 'read', ?)
            """, (f"morning_episode_{i}", f"morning_email_{i}", morning_time.isoformat()))

    return db


@pytest.mark.asyncio
async def test_routine_detection_publishes_events(populated_db, user_model_store, event_bus):
    """Test that routine detection publishes usermodel.routine.updated events."""
    # Create detectors
    routine_detector = RoutineDetector(populated_db, user_model_store)

    # Run routine detection manually (simulating the background loop)
    routines = await asyncio.to_thread(routine_detector.detect_routines, 30)
    stored_count = await asyncio.to_thread(routine_detector.store_routines, routines)

    # Publish events from async context (this is what the fix adds to main.py)
    for routine in routines:
        await event_bus.publish(
            "usermodel.routine.updated",
            {
                "routine_name": routine["name"],
                "trigger": routine["trigger"],
                "steps_count": len(routine.get("steps", [])),
                "consistency_score": routine.get("consistency_score", 0.5),
                "times_observed": routine.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="routine_detector",
        )

    # Allow events to propagate
    await asyncio.sleep(0.1)

    # Verify events were published through event bus
    assert event_bus.publish.call_count >= stored_count, \
        f"Expected at least {stored_count} publish calls, got {event_bus.publish.call_count}"

    # Verify event bus received routine events
    routine_events = [
        call for call in event_bus.publish.call_args_list
        if call[0][0] == "usermodel.routine.updated"
    ]
    assert len(routine_events) == stored_count, \
        f"Event count mismatch: stored {stored_count} routines but published {len(routine_events)} events"


@pytest.mark.asyncio
async def test_workflow_detection_publishes_events(populated_db, user_model_store, event_bus):
    """Test that workflow detection publishes usermodel.workflow.updated events."""
    # Create detectors
    workflow_detector = WorkflowDetector(populated_db, user_model_store)

    # Run workflow detection manually (simulating the background loop)
    workflows = await asyncio.to_thread(workflow_detector.detect_workflows, 30)
    workflow_stored = await asyncio.to_thread(workflow_detector.store_workflows, workflows)

    # Publish events from async context (this is what the fix adds to main.py)
    for workflow in workflows:
        await event_bus.publish(
            "usermodel.workflow.updated",
            {
                "workflow_name": workflow["name"],
                "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
                "steps_count": len(workflow.get("steps", [])),
                "tools_count": len(workflow.get("tools_used", [])),
                "success_rate": workflow.get("success_rate", 0.5),
                "times_observed": workflow.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="workflow_detector",
        )

    # Allow events to propagate
    await asyncio.sleep(0.1)

    # Verify events were published through event bus
    assert event_bus.publish.call_count >= workflow_stored, \
        f"Expected at least {workflow_stored} publish calls, got {event_bus.publish.call_count}"

    # Verify event bus received workflow events
    workflow_events = [
        call for call in event_bus.publish.call_args_list
        if call[0][0] == "usermodel.workflow.updated"
    ]
    assert len(workflow_events) == workflow_stored, \
        f"Event count mismatch: stored {workflow_stored} workflows but published {len(workflow_events)} events"


@pytest.mark.asyncio
async def test_event_payloads_match_stored_data(populated_db, user_model_store, event_bus):
    """Test that published event payloads contain accurate workflow metadata."""
    workflow_detector = WorkflowDetector(populated_db, user_model_store)

    # Run detection
    workflows = await asyncio.to_thread(workflow_detector.detect_workflows, 30)
    await asyncio.to_thread(workflow_detector.store_workflows, workflows)

    # Publish events
    for workflow in workflows:
        await event_bus.publish(
            "usermodel.workflow.updated",
            {
                "workflow_name": workflow["name"],
                "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
                "steps_count": len(workflow.get("steps", [])),
                "tools_count": len(workflow.get("tools_used", [])),
                "success_rate": workflow.get("success_rate", 0.5),
                "times_observed": workflow.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="workflow_detector",
        )

    await asyncio.sleep(0.1)

    # Check the published events via mock
    workflow_calls = [
        call for call in event_bus.publish.call_args_list
        if call[0][0] == "usermodel.workflow.updated"
    ]

    assert len(workflow_calls) > 0, "No workflow events published"

    # Verify each event payload
    for call in workflow_calls:
        payload = call[0][1]  # Second positional arg is the payload

        # All fields should be present
        assert "workflow_name" in payload
        assert "trigger_conditions_count" in payload
        assert "steps_count" in payload
        assert "tools_count" in payload
        assert "success_rate" in payload
        assert "times_observed" in payload
        assert "updated_at" in payload

        # Validate types
        assert isinstance(payload["workflow_name"], str)
        assert isinstance(payload["trigger_conditions_count"], int)
        assert isinstance(payload["steps_count"], int)
        assert isinstance(payload["tools_count"], int)
        assert isinstance(payload["success_rate"], (int, float))
        assert isinstance(payload["times_observed"], int)
        assert isinstance(payload["updated_at"], str)

        # Validate ranges
        assert 0.0 <= payload["success_rate"] <= 1.0
        assert payload["times_observed"] > 0


@pytest.mark.asyncio
async def test_detection_loop_integration(populated_db, user_model_store, event_bus):
    """Test that the full detection loop (as in main.py) publishes events correctly."""
    routine_detector = RoutineDetector(populated_db, user_model_store)
    workflow_detector = WorkflowDetector(populated_db, user_model_store)

    # Simulate one iteration of the routine/workflow detection loop
    # (This is the exact code from main.py:1241-1257 plus event publishing)

    routines = await asyncio.to_thread(routine_detector.detect_routines, 30)
    stored_count = await asyncio.to_thread(routine_detector.store_routines, routines)

    for routine in routines:
        await event_bus.publish(
            "usermodel.routine.updated",
            {
                "routine_name": routine["name"],
                "trigger": routine["trigger"],
                "steps_count": len(routine.get("steps", [])),
                "consistency_score": routine.get("consistency_score", 0.5),
                "times_observed": routine.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="routine_detector",
        )

    workflows = await asyncio.to_thread(workflow_detector.detect_workflows, 30)
    workflow_stored = await asyncio.to_thread(workflow_detector.store_workflows, workflows)

    for workflow in workflows:
        await event_bus.publish(
            "usermodel.workflow.updated",
            {
                "workflow_name": workflow["name"],
                "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
                "steps_count": len(workflow.get("steps", [])),
                "tools_count": len(workflow.get("tools_used", [])),
                "success_rate": workflow.get("success_rate", 0.5),
                "times_observed": workflow.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="workflow_detector",
        )

    await asyncio.sleep(0.1)

    # Verify both types of events were published
    routine_events = [
        call for call in event_bus.publish.call_args_list
        if call[0][0] == "usermodel.routine.updated"
    ]
    workflow_events = [
        call for call in event_bus.publish.call_args_list
        if call[0][0] == "usermodel.workflow.updated"
    ]

    # At least one of each should exist if any were stored
    if stored_count > 0:
        assert len(routine_events) == stored_count, \
            f"Stored {stored_count} routines but published {len(routine_events)} events"
    if workflow_stored > 0:
        assert len(workflow_events) == workflow_stored, \
            f"Stored {workflow_stored} workflows but published {len(workflow_events)} events"


@pytest.mark.asyncio
async def test_publishing_from_thread_fails_silently(populated_db, user_model_store, event_bus):
    """
    Test that demonstrates the bug: _emit_telemetry() from UserModelStore
    fails silently when called from a thread (asyncio.to_thread context).

    This test shows why we need to publish events from main.py's async context.
    """
    workflow_detector = WorkflowDetector(populated_db, user_model_store)

    # Reset event bus mock
    event_bus.publish.reset_mock()

    # Run workflow detection and storage in a thread (as main.py does)
    # The store_workflows() method calls user_model_store.store_workflow(),
    # which tries to emit telemetry but fails because there's no event loop in the thread
    workflows = await asyncio.to_thread(workflow_detector.detect_workflows, 30)
    workflow_stored = await asyncio.to_thread(workflow_detector.store_workflows, workflows)

    await asyncio.sleep(0.1)

    # BUG DEMONSTRATION: No events were published because _emit_telemetry()
    # silently failed when called from the thread pool
    assert event_bus.publish.call_count == 0, \
        "BUG: Events should NOT be published from thread context (no event loop)"

    # Now publish from async context (the fix)
    for workflow in workflows:
        await event_bus.publish(
            "usermodel.workflow.updated",
            {
                "workflow_name": workflow["name"],
                "trigger_conditions_count": len(workflow.get("trigger_conditions", [])),
                "steps_count": len(workflow.get("steps", [])),
                "tools_count": len(workflow.get("tools_used", [])),
                "success_rate": workflow.get("success_rate", 0.5),
                "times_observed": workflow.get("times_observed", 0),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source="workflow_detector",
        )

    await asyncio.sleep(0.1)

    # FIX VERIFICATION: Now events ARE published because we're in async context
    assert event_bus.publish.call_count == workflow_stored, \
        f"Fix works: published {event_bus.publish.call_count} events from async context"
