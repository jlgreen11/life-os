"""
Life OS — Master Event Handler Pipeline Integration Tests

Verifies that the master_event_handler() in main.py correctly routes events
through the full processing pipeline:

  1. Store (events.db)
  2. Feedback processing (notification responses)
  3. Source weight tracking
  4. Episode creation (user_model.db)
  5. Signal extraction (linguistic, cadence, mood, etc.)
  6. Rules evaluation (deterministic automation)
  7. Task extraction (AI-identified action items)
  8. Vector embedding (semantic search)

Individual stages have their own unit tests; these integration tests verify
the stages execute together in sequence and that errors in one stage do not
prevent subsequent stages from running (fail-open convention).
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

    Follows the same injection pattern used in test_episodic_memory.py,
    then additionally calls _register_event_handlers() so the
    master_event_handler local function is subscribed to the event bus.
    We extract the handler and attach it as an attribute for direct calls.
    """
    from main import LifeOS

    los = LifeOS(
        db=db,
        event_bus=event_bus,
        event_store=event_store,
        user_model_store=user_model_store,
        config=lifeos_config,
    )

    # Wire up the pipeline — this registers master_event_handler via
    # event_bus.subscribe_all(handler).
    await los._register_event_handlers()

    # Extract the handler from the mock's call args so tests can invoke
    # it directly without needing to publish through the bus.
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
# Test 1: Full pipeline stores event and creates episode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_email_event_stored_and_episode_created(lifeos, db):
    """Process an email.received event end-to-end and verify storage + episode."""
    event = _make_event(
        "email.received",
        from_address="alice@example.com",
        to_addresses=["user@example.com"],
        subject="Quarterly review meeting",
        body_plain="Let's discuss the Q1 results next Tuesday.",
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event persisted in events.db
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored in events.db"
    assert dict(row)["type"] == "email.received"

    # Stage 1.5: episode created in user_model.db
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 1, "Exactly one episode should be created"

    episode = dict(episodes[0])
    assert episode["interaction_type"] == "email_received"

    # Contacts should include the sender
    contacts = json.loads(episode["contacts_involved"])
    assert "alice@example.com" in contacts


# ---------------------------------------------------------------------------
# Test 2: Signal extraction updates linguistic profile
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_email_event_triggers_signal_extraction(lifeos, db):
    """Process an email.sent event and verify the linguistic signal profile is updated."""
    event = _make_event(
        "email.sent",
        from_address="user@example.com",
        to_addresses=["bob@example.com"],
        subject="Re: Project update",
        body_plain="Thanks for the update. I think we should move forward with option A.",
    )

    await lifeos.master_event_handler(event)

    # The linguistic extractor should have written a profile
    profile = lifeos.user_model_store.get_signal_profile("linguistic")
    assert profile is not None, "Linguistic profile should exist after processing"
    assert profile.get("samples_count", 0) >= 1, "At least one sample should be recorded"


# ---------------------------------------------------------------------------
# Test 3: Source weight tracking records an interaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_source_weight_tracking_records_interaction(lifeos, db):
    """Process an event and verify source weight tracking increments the counter.

    record_interaction() does an UPDATE, so default source weight rows must
    exist first (normally seeded at startup via seed_defaults()).
    """
    # Seed default weights — in production this happens during LifeOS.start()
    lifeos.source_weight_manager.seed_defaults()

    event = _make_event("email.received")

    # Check the interaction count before processing
    weights_before = lifeos.source_weight_manager.get_all_weights()
    source_key = lifeos.source_weight_manager.classify_event(event)
    before_count = 0
    for w in weights_before:
        if w["source_key"] == source_key:
            before_count = w.get("interactions", 0)
            break

    await lifeos.master_event_handler(event)

    # Verify the interaction counter was incremented
    weights_after = lifeos.source_weight_manager.get_all_weights()
    after_count = 0
    for w in weights_after:
        if w["source_key"] == source_key:
            after_count = w.get("interactions", 0)
            break

    assert after_count == before_count + 1, (
        f"Interaction count for '{source_key}' should increment by 1 "
        f"(was {before_count}, now {after_count})"
    )


# ---------------------------------------------------------------------------
# Test 4: Vector embedding stores the document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vector_embedding_stage_reached(lifeos, db):
    """Verify the pipeline reaches the embedding stage (stage 5).

    We instrument _embed_event with a tracking flag to confirm the pipeline
    calls it, regardless of whether the embedding model is available in the
    test environment (sentence-transformers may not be installed).
    """
    embed_called_with = []
    original_embed = lifeos._embed_event

    async def tracking_embed(event):
        embed_called_with.append(event["id"])
        return await original_embed(event)

    lifeos._embed_event = tracking_embed

    event = _make_event(
        "email.received",
        subject="Important project discussion about the quarterly budget review",
        body_plain="We need to review the quarterly budget and make adjustments for next quarter.",
    )

    await lifeos.master_event_handler(event)

    assert event["id"] in embed_called_with, "Pipeline should call _embed_event for the event"

    # Restore
    lifeos._embed_event = original_embed


# ---------------------------------------------------------------------------
# Test 5: Pipeline error isolation — broken signal extractor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_error_isolation_signal_extractor(lifeos, db):
    """If signal extraction fails, event storage and episode creation still work.

    This validates the fail-open convention: each pipeline stage is wrapped
    in its own try/except, so a failure in one stage cannot prevent others.
    """
    # Sabotage the signal extractor to force an error
    original_process = lifeos.signal_extractor.process_event

    async def broken_process(event):
        raise RuntimeError("Simulated signal extractor failure")

    lifeos.signal_extractor.process_event = broken_process

    event = _make_event(
        "email.received",
        from_address="test@example.com",
        subject="Test error isolation",
        body_plain="This message should still be stored despite signal extraction failure.",
    )

    # Should NOT raise — the pipeline catches errors per stage
    await lifeos.master_event_handler(event)

    # Stage 1: event should still be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored despite signal extraction failure"

    # Stage 1.5: episode should still be created (runs before signal extraction)
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) >= 1, "Episode should be created despite signal extraction failure"

    # Restore the original method
    lifeos.signal_extractor.process_event = original_process


# ---------------------------------------------------------------------------
# Test 6: Pipeline error isolation — broken rules engine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_error_isolation_rules_engine(lifeos, db):
    """If the rules engine fails, earlier stages (store, episode, signals) still work."""
    # Sabotage the rules engine
    original_evaluate = lifeos.rules_engine.evaluate

    async def broken_evaluate(event):
        raise RuntimeError("Simulated rules engine failure")

    lifeos.rules_engine.evaluate = broken_evaluate

    event = _make_event(
        "email.sent",
        from_address="user@example.com",
        to_addresses=["colleague@example.com"],
        subject="Error isolation test for rules",
        body_plain="Even if rules fail, signals should still be extracted.",
    )

    await lifeos.master_event_handler(event)

    # Event should be stored (stage 1)
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored despite rules engine failure"

    # Episode should be created (stage 1.5)
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) >= 1, "Episode should be created despite rules engine failure"

    # Signal extraction should still have run (stage 2)
    profile = lifeos.user_model_store.get_signal_profile("linguistic")
    assert profile is not None, "Signal extraction should run despite rules engine failure"

    # Restore
    lifeos.rules_engine.evaluate = original_evaluate


# ---------------------------------------------------------------------------
# Test 7: Rule match produces actions during pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rule_match_executes_during_pipeline(lifeos, db):
    """Add a rule, process an event, and verify the rule fired and tagged the event."""
    # Add a rule that tags all email.received events
    await lifeos.rules_engine.add_rule(
        name="Tag all emails",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "tag", "value": "pipeline-test-tag"}],
    )

    event = _make_event(
        "email.received",
        subject="Rule integration test",
        body_plain="This email should trigger the tagging rule.",
    )

    await lifeos.master_event_handler(event)

    # Verify the event was stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None

    # Verify the tag action was executed — event_tags lives in events.db
    with db.get_connection("events") as conn:
        tags = conn.execute(
            "SELECT * FROM event_tags WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(tags) >= 1, "Rule should have tagged the event"
    tag_values = [dict(t)["tag"] for t in tags]
    assert "pipeline-test-tag" in tag_values


# ---------------------------------------------------------------------------
# Test 8: Suppress action prevents notification creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suppress_action_prevents_notification(lifeos, db):
    """When suppress + notify rules both match, suppress prevents the notification."""
    # Add a suppress rule that matches email.received
    await lifeos.rules_engine.add_rule(
        name="Suppress test emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "suppress-me"},
        ],
        actions=[{"type": "suppress"}],
    )

    # Also add a notify rule that matches all emails
    await lifeos.rules_engine.add_rule(
        name="Notify on all emails",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "message": "New email received"}],
    )

    event = _make_event(
        "email.received",
        subject="Please suppress-me in pipeline",
        body_plain="This email should be suppressed and not generate a notification.",
    )

    await lifeos.master_event_handler(event)

    # Event should be stored regardless
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None

    # No notification should have been created because suppress runs first
    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?", (event["id"],)
        ).fetchall()
    assert len(notifs) == 0, "Suppressed event should NOT generate a notification"


# ---------------------------------------------------------------------------
# Test 9: Multiple events processed sequentially
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_events_sequential_pipeline(lifeos, db):
    """Process several events in sequence and verify each is stored with an episode."""
    events = [
        _make_event("email.received", subject=f"Sequential test email {i}",
                     body_plain=f"Body of sequential test email number {i}.")
        for i in range(3)
    ]

    for event in events:
        await lifeos.master_event_handler(event)

    # All events should be stored
    with db.get_connection("events") as conn:
        for event in events:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (event["id"],)
            ).fetchone()
            assert row is not None, f"Event {event['id']} should be stored"

    # All events should have episodes
    with db.get_connection("user_model") as conn:
        for event in events:
            episodes = conn.execute(
                "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
            ).fetchall()
            assert len(episodes) == 1, f"Event {event['id']} should have exactly one episode"


# ---------------------------------------------------------------------------
# Test 10: System events are stored but skip episode creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_event_stored_but_no_episode(lifeos, db):
    """System/internal events should be stored in events.db but NOT create episodes."""
    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.CONNECTOR_SYNC_COMPLETE.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {"connector_id": "google-gmail"},
        "metadata": {},
    }

    await lifeos.master_event_handler(event)

    # Event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "System event should still be stored"

    # But no episode should be created (system events are skipped)
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 0, "System event should NOT create an episode"
