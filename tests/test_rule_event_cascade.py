"""
Life OS — Rule Event Cascade Prevention Tests

Verifies that system telemetry events (system.rule.triggered,
system.connector.sync_complete, etc.) are stored in events.db but
skip pipeline stages 2–6 (signal extraction, rules evaluation, task
extraction, vector embedding, episode creation).

This prevents:
  - Wasted CPU on stages that produce no useful output for telemetry.
  - Event amplification: system.rule.triggered events re-evaluated by
    the rules engine would create more system events (2.5x today,
    potentially infinite with a wildcard rule).

Uses the same integration-test pattern as
test_master_event_handler_pipeline.py: real DatabaseManager +
EventStore/UserModelStore with temporary SQLite, mock EventBus.
"""

import uuid
from datetime import datetime, timezone

import pytest

from models.core import EventType


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_master_event_handler_pipeline.py)
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

    Registers event handlers and exposes the master_event_handler as a
    callable attribute for direct invocation in tests.
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

    # Extract the handler from the mock's call args for direct calls.
    handler = event_bus.subscribe_all.call_args[0][0]
    los.master_event_handler = handler

    return los


def _make_event(event_type: str, **payload_overrides) -> dict:
    """Build a well-formed event dict with sensible defaults.

    Args:
        event_type: The event type string (e.g., "system.rule.triggered").
        **payload_overrides: Fields merged into the default payload.

    Returns:
        A complete event dict ready for master_event_handler.
    """
    payload = {"detail": "test-payload"}
    payload.update(payload_overrides)

    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "rules_engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": payload,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Test 1: system.rule.triggered events skip rules evaluation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_rule_triggered_skips_rules_evaluation(lifeos, db):
    """A system.rule.triggered event must NOT be re-evaluated by the rules engine.

    This is the core cascade prevention: rules engine telemetry must never
    re-enter rules evaluation, otherwise each rule match would generate
    another system.rule.triggered event in an amplification loop.
    """
    evaluate_called_with = []
    original_evaluate = lifeos.rules_engine.evaluate

    async def tracking_evaluate(event):
        """Track calls to rules_engine.evaluate()."""
        evaluate_called_with.append(event["type"])
        return await original_evaluate(event)

    lifeos.rules_engine.evaluate = tracking_evaluate

    event = _make_event(
        EventType.RULE_TRIGGERED.value,
        rule_name="Test Rule",
        matched_event_id="some-event-id",
    )

    await lifeos.master_event_handler(event)

    assert len(evaluate_called_with) == 0, (
        "rules_engine.evaluate() should NOT be called for system.rule.triggered events, "
        f"but was called with: {evaluate_called_with}"
    )

    lifeos.rules_engine.evaluate = original_evaluate


# ---------------------------------------------------------------------------
# Test 2: system.connector.* events skip signal extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_connector_event_skips_signal_extraction(lifeos, db):
    """A system.connector.sync_complete event must NOT run through signal extraction.

    Connector telemetry carries no user content — processing it through the
    signal extractor wastes CPU and produces empty/meaningless profiles.
    """
    process_called_with = []
    original_process = lifeos.signal_extractor.process_event

    async def tracking_process(event):
        """Track calls to signal_extractor.process_event()."""
        process_called_with.append(event["type"])
        return await original_process(event)

    lifeos.signal_extractor.process_event = tracking_process

    event = _make_event(
        EventType.CONNECTOR_SYNC_COMPLETE.value,
        connector_id="google-gmail",
    )

    await lifeos.master_event_handler(event)

    assert len(process_called_with) == 0, (
        "signal_extractor.process_event() should NOT be called for "
        "system.connector.sync_complete events, "
        f"but was called with: {process_called_with}"
    )

    lifeos.signal_extractor.process_event = original_process


# ---------------------------------------------------------------------------
# Test 3: Real events still processed through all stages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_events_still_processed(lifeos, db):
    """An email.received event must still run through ALL pipeline stages.

    Guards against the system-event skip being too broad — only events with
    the 'system.' prefix should be skipped.
    """
    stages_reached = []

    # Track signal extraction (stage 2)
    original_process = lifeos.signal_extractor.process_event

    async def tracking_process(event):
        stages_reached.append("signal_extraction")
        return await original_process(event)

    lifeos.signal_extractor.process_event = tracking_process

    # Track rules evaluation (stage 3)
    original_evaluate = lifeos.rules_engine.evaluate

    async def tracking_evaluate(event):
        stages_reached.append("rules_evaluation")
        return await original_evaluate(event)

    lifeos.rules_engine.evaluate = tracking_evaluate

    # Track task extraction (stage 4)
    original_task_process = lifeos.task_manager.process_event

    async def tracking_task_process(event):
        stages_reached.append("task_extraction")
        return await original_task_process(event)

    lifeos.task_manager.process_event = tracking_task_process

    # Track embedding (stage 5)
    original_embed = lifeos._embed_event

    async def tracking_embed(event):
        stages_reached.append("embedding")
        return await original_embed(event)

    lifeos._embed_event = tracking_embed

    # Track episode creation (stage 6)
    original_episode = lifeos._create_episode

    async def tracking_episode(event):
        stages_reached.append("episode_creation")
        return await original_episode(event)

    lifeos._create_episode = tracking_episode

    event = _make_event("email.received")
    event["source"] = "google"
    event["payload"] = {
        "from_address": "alice@example.com",
        "to_addresses": ["user@example.com"],
        "subject": "Cascade prevention test",
        "body_plain": "This email should flow through every stage.",
        "message_id": f"<test-{uuid.uuid4().hex[:8]}>",
    }

    await lifeos.master_event_handler(event)

    assert "signal_extraction" in stages_reached, "Stage 2 should run for real events"
    assert "rules_evaluation" in stages_reached, "Stage 3 should run for real events"
    assert "task_extraction" in stages_reached, "Stage 4 should run for real events"
    assert "embedding" in stages_reached, "Stage 5 should run for real events"
    assert "episode_creation" in stages_reached, "Stage 6 should run for real events"

    # Restore originals
    lifeos.signal_extractor.process_event = original_process
    lifeos.rules_engine.evaluate = original_evaluate
    lifeos.task_manager.process_event = original_task_process
    lifeos._embed_event = original_embed
    lifeos._create_episode = original_episode


# ---------------------------------------------------------------------------
# Test 4: System events are still stored in events.db
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_events_still_stored(lifeos, db):
    """System events must still be persisted in events.db (stage 1).

    The skip guard only affects stages 2–6. Stage 1 (event storage)
    runs unconditionally so system events remain in the audit trail.
    """
    event = _make_event(
        EventType.RULE_TRIGGERED.value,
        rule_name="Audit Trail Rule",
        matched_event_id="original-event-123",
    )

    await lifeos.master_event_handler(event)

    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()

    assert row is not None, "System event must be stored in events.db"
    assert dict(row)["type"] == EventType.RULE_TRIGGERED.value


# ---------------------------------------------------------------------------
# Test 5: usermodel.* events are still processed normally
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usermodel_events_still_processed(lifeos, db):
    """Events with the 'usermodel.' prefix must still go through all stages.

    usermodel.signal_profile.updated and similar events carry valuable
    data about user behavior — they must not be caught by the system-event
    skip guard.
    """
    stages_reached = []

    original_process = lifeos.signal_extractor.process_event

    async def tracking_process(event):
        stages_reached.append("signal_extraction")
        return await original_process(event)

    lifeos.signal_extractor.process_event = tracking_process

    original_evaluate = lifeos.rules_engine.evaluate

    async def tracking_evaluate(event):
        stages_reached.append("rules_evaluation")
        return await original_evaluate(event)

    lifeos.rules_engine.evaluate = tracking_evaluate

    event = _make_event("usermodel.signal_profile.updated")
    event["source"] = "signal_extractor"
    event["payload"] = {
        "profile_type": "linguistic",
        "samples_count": 42,
    }

    await lifeos.master_event_handler(event)

    assert "signal_extraction" in stages_reached, (
        "usermodel.* events should still run through signal extraction"
    )
    assert "rules_evaluation" in stages_reached, (
        "usermodel.* events should still run through rules evaluation"
    )

    # Restore originals
    lifeos.signal_extractor.process_event = original_process
    lifeos.rules_engine.evaluate = original_evaluate
