"""
Life OS — Master Event Handler Pipeline Integration Tests

Verifies that the master_event_handler() in main.py correctly routes events
through the full processing pipeline:

  1. Store (events.db)
  2. Feedback processing (notification responses)
  3. Source weight tracking
  4. Signal extraction (linguistic, cadence, mood, etc.)
  5. Rules evaluation (deterministic automation)
  6. Task extraction (AI-identified action items)
  7. Vector embedding (semantic search)
  8. Episode creation (user_model.db) — after signals so mood data is current

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

    # Stage 6: episode created in user_model.db
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

    # Stage 6: episode should still be created (runs after signal extraction,
    # but signal extraction failure doesn't block it due to fail-open)
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

    # Episode should be created (stage 6)
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


# ---------------------------------------------------------------------------
# Test 11: Episode creation runs AFTER signal extraction (pipeline ordering)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_episode_creation_runs_after_signal_extraction(lifeos, db):
    """Verify episode creation (Stage 6) executes after signal extraction (Stage 2).

    This is critical because _create_episode() calls
    signal_extractor.get_current_mood() to attach mood context to the episode.
    If episode creation ran before signal extraction, the mood data would be
    stale by one event.
    """
    call_order = []

    # Wrap signal_extractor.process_event to track call order
    original_process = lifeos.signal_extractor.process_event

    async def tracking_process(event):
        call_order.append("signal_extraction")
        return await original_process(event)

    lifeos.signal_extractor.process_event = tracking_process

    # Wrap _create_episode to track call order
    original_create_episode = lifeos._create_episode

    async def tracking_create_episode(event):
        call_order.append("episode_creation")
        return await original_create_episode(event)

    lifeos._create_episode = tracking_create_episode

    event = _make_event(
        "email.received",
        from_address="ordering-test@example.com",
        subject="Pipeline ordering test",
        body_plain="This event tests that signal extraction runs before episode creation.",
    )

    await lifeos.master_event_handler(event)

    # Both stages should have run
    assert "signal_extraction" in call_order, "Signal extraction should have been called"
    assert "episode_creation" in call_order, "Episode creation should have been called"

    # Signal extraction must come before episode creation
    signal_idx = call_order.index("signal_extraction")
    episode_idx = call_order.index("episode_creation")
    assert signal_idx < episode_idx, (
        f"Signal extraction (index {signal_idx}) must run before "
        f"episode creation (index {episode_idx}). "
        f"Actual call order: {call_order}"
    )

    # Restore originals
    lifeos.signal_extractor.process_event = original_process
    lifeos._create_episode = original_create_episode


# ---------------------------------------------------------------------------
# Test 12: WebSocket mood broadcast after email event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_email_event_triggers_mood_websocket_broadcast(lifeos, db):
    """Processing an email.received event should trigger a mood_update WebSocket broadcast.

    After signal extraction runs on content-bearing events (email.*, message.*,
    chat.*), the pipeline broadcasts a mood_update message so the dashboard mood
    widget refreshes instantly.  See main.py lines 2101-2108.
    """
    from unittest.mock import AsyncMock, patch
    from web.websocket import ws_manager

    event = _make_event(
        "email.received",
        from_address="alice@example.com",
        subject="Mood broadcast test",
        body_plain="This email should trigger a mood WebSocket broadcast.",
    )

    with patch.object(ws_manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
        await lifeos.master_event_handler(event)

    # Verify at least one broadcast call was a mood_update
    mood_calls = [
        call for call in mock_broadcast.call_args_list
        if call[0][0].get("type") == "mood_update"
    ]
    assert len(mood_calls) >= 1, (
        f"Expected at least one mood_update broadcast for email.received event. "
        f"All broadcast calls: {mock_broadcast.call_args_list}"
    )


# ---------------------------------------------------------------------------
# Test 13: Notification triggers WebSocket broadcast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_triggers_websocket_broadcast(lifeos, db):
    """When a rule action creates a notification, the pipeline should broadcast
    a WebSocket message with type 'notification'.  See main.py lines 2292-2300.
    """
    from unittest.mock import AsyncMock, patch
    from web.websocket import ws_manager

    # Add a notify rule matching all email.received events
    await lifeos.rules_engine.add_rule(
        name="Notify on all emails for WS test",
        trigger_event="email.received",
        conditions=[],
        actions=[{"type": "notify", "message": "New email arrived"}],
    )

    event = _make_event(
        "email.received",
        subject="Notification broadcast test",
        body_plain="This should trigger both a notification and a WebSocket broadcast.",
    )

    with patch.object(ws_manager, "broadcast", new_callable=AsyncMock) as mock_broadcast:
        await lifeos.master_event_handler(event)

    # Verify a notification-type WebSocket broadcast was sent
    notif_calls = [
        call for call in mock_broadcast.call_args_list
        if call[0][0].get("type") == "notification"
    ]
    assert len(notif_calls) >= 1, (
        f"Expected at least one notification WebSocket broadcast. "
        f"All broadcast calls: {mock_broadcast.call_args_list}"
    )

    # Verify the notification broadcast includes the source_event_id
    notif_payload = notif_calls[0][0][0]
    assert notif_payload.get("source_event_id") == event["id"]


# ---------------------------------------------------------------------------
# Test 14: WebSocket broadcast failure does not crash the pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_websocket_broadcast_failure_does_not_crash_pipeline(lifeos, db):
    """If ws_manager.broadcast raises, the pipeline should continue without error.

    The pipeline wraps all WebSocket broadcasts in bare try/except with pass
    (main.py lines 2031-2038 and 2104-2108) to ensure a WebSocket failure never
    blocks event processing.
    """
    from unittest.mock import AsyncMock, patch
    from web.websocket import ws_manager

    event = _make_event(
        "email.received",
        from_address="resilience@example.com",
        subject="WebSocket resilience test",
        body_plain="Pipeline should survive a WebSocket broadcast failure.",
    )

    with patch.object(
        ws_manager, "broadcast",
        new_callable=AsyncMock,
        side_effect=RuntimeError("WebSocket broadcast failed"),
    ):
        # Should NOT raise — the pipeline catches broadcast errors
        await lifeos.master_event_handler(event)

    # Stage 1: event should still be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored despite WebSocket broadcast failure"

    # Stage 6: episode should still be created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) >= 1, "Episode should be created despite WebSocket broadcast failure"


# ---------------------------------------------------------------------------
# Test 15: System event skips signal extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_event_skips_signal_extraction(lifeos, db):
    """System events (type starting with 'system.') skip stages 2-6 including
    signal extraction.  The existing test_system_event_stored_but_no_episode
    checks episode creation; this test verifies signal_extractor.process_event
    is NOT called.  See main.py lines 2078-2091.
    """
    # Track whether signal_extractor.process_event is called
    process_event_called = []
    original_process = lifeos.signal_extractor.process_event

    async def tracking_process(event):
        process_event_called.append(event["id"])
        return await original_process(event)

    lifeos.signal_extractor.process_event = tracking_process

    event = {
        "id": str(uuid.uuid4()),
        "type": "system.rule.triggered",
        "source": "rules_engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {"rule_id": "test-rule", "action": "tag"},
        "metadata": {},
    }

    await lifeos.master_event_handler(event)

    # Signal extractor should NOT have been called for a system event
    assert event["id"] not in process_event_called, (
        "signal_extractor.process_event should NOT be called for system events "
        "(the pipeline guard at main.py:2085-2091 should skip stages 2-6)"
    )

    # But the event should still be stored (stage 1 runs before the guard)
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "System event should still be stored in events.db"

    # Restore
    lifeos.signal_extractor.process_event = original_process


# ---------------------------------------------------------------------------
# Test 16: System event still records source weight
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_event_still_records_source_weight(lifeos, db):
    """System events skip stages 2-6 but source weight tracking (stage 1.3)
    runs before the guard, so record_interaction should still be called.
    See main.py lines 2068-2076 (before the guard at 2085-2091).
    """
    # Track calls to record_interaction
    record_calls = []
    original_record = lifeos.source_weight_manager.record_interaction

    def tracking_record(source_key):
        record_calls.append(source_key)
        return original_record(source_key)

    lifeos.source_weight_manager.record_interaction = tracking_record

    event = {
        "id": str(uuid.uuid4()),
        "type": "system.rule.triggered",
        "source": "rules_engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {"rule_id": "test-rule", "action": "notify"},
        "metadata": {},
    }

    await lifeos.master_event_handler(event)

    # record_interaction should have been called despite this being a system event
    assert len(record_calls) >= 1, (
        "source_weight_manager.record_interaction should be called for system events "
        "(stage 1.3 runs before the system event guard at main.py:2085)"
    )

    # The source key for system.rule.triggered should be "system.general"
    assert "system.general" in record_calls, (
        f"Expected 'system.general' source key for system.rule.triggered, "
        f"got: {record_calls}"
    )

    # Restore
    lifeos.source_weight_manager.record_interaction = original_record


# ---------------------------------------------------------------------------
# Test 17: Suppress action sets _suppressed flag on event dict
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suppress_action_sets_suppressed_flag_on_event(lifeos, db):
    """When the rules engine returns a suppress action, the pipeline should set
    event['_suppressed'] = True on the event dict.  The existing
    test_suppress_action_prevents_notification checks notification suppression
    but not the in-memory flag itself.  See main.py lines 2309-2315.
    """
    # Add a suppress rule matching email.received with a specific subject
    await lifeos.rules_engine.add_rule(
        name="Suppress flagged emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "flag-test"},
        ],
        actions=[{"type": "suppress"}],
    )

    event = _make_event(
        "email.received",
        subject="Please flag-test this email",
        body_plain="The suppress action should set _suppressed = True.",
    )

    # Confirm _suppressed is not set before processing
    assert "_suppressed" not in event

    await lifeos.master_event_handler(event)

    # The suppress action should have set the in-memory flag
    assert event.get("_suppressed") is True, (
        "event['_suppressed'] should be True after suppress action executes"
    )

    # Also verify the persistent tag was written
    with db.get_connection("events") as conn:
        tags = conn.execute(
            "SELECT * FROM event_tags WHERE event_id = ? AND tag = 'system:suppressed'",
            (event["id"],),
        ).fetchall()
    assert len(tags) >= 1, "Persistent 'system:suppressed' tag should be written"


# ---------------------------------------------------------------------------
# Test 18: Suppress runs before notify in same rule set (ordering)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_suppress_runs_before_notify_in_same_rule_set(lifeos, db):
    """When the rules engine returns both suppress and notify actions (from
    different rules), the pipeline sorts suppress actions first so the
    _suppressed flag is set before the notify action checks it.

    This tests the explicit sort at main.py lines 2120-2122:
        suppress_actions = [a for a in actions if a['type'] == 'suppress']
        other_actions = [a for a in actions if a['type'] != 'suppress']
        for action in suppress_actions + other_actions:

    The notify rule is added FIRST to ensure the ordering logic (not insertion
    order) is what prevents the notification from being created.
    """
    # Add the NOTIFY rule FIRST — if the sort didn't exist, notify would run
    # before suppress and the notification would be created.
    await lifeos.rules_engine.add_rule(
        name="Notify on order-test emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "order-test"},
        ],
        actions=[{"type": "notify", "message": "Should be suppressed"}],
    )

    # Add the SUPPRESS rule SECOND
    await lifeos.rules_engine.add_rule(
        name="Suppress order-test emails",
        trigger_event="email.received",
        conditions=[
            {"field": "payload.subject", "op": "contains", "value": "order-test"},
        ],
        actions=[{"type": "suppress"}],
    )

    event = _make_event(
        "email.received",
        subject="This is an order-test email",
        body_plain="Both suppress and notify rules match, suppress should win.",
    )

    await lifeos.master_event_handler(event)

    # The suppress action should have set the flag
    assert event.get("_suppressed") is True, (
        "Suppress action should have set _suppressed = True"
    )

    # No notification should have been created because suppress ran first
    with db.get_connection("state") as conn:
        notifs = conn.execute(
            "SELECT * FROM notifications WHERE source_event_id = ?", (event["id"],)
        ).fetchall()
    assert len(notifs) == 0, (
        "Notification should NOT be created when suppress runs in the same action set — "
        "the suppress-before-notify sort at main.py:2120-2122 should prevent it"
    )


# ---------------------------------------------------------------------------
# Test 19: Task manager error does not block embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_manager_error_does_not_block_embedding(lifeos, db):
    """If task_manager.process_event (stage 4) raises, the pipeline should
    still reach _embed_event (stage 5).  Existing tests cover signal_extractor
    and rules_engine isolation but not task_manager.
    """
    # Sabotage the task manager
    original_process = lifeos.task_manager.process_event

    async def broken_process(event):
        raise RuntimeError("Simulated task manager failure")

    lifeos.task_manager.process_event = broken_process

    # Track whether _embed_event is called
    embed_called_with = []
    original_embed = lifeos._embed_event

    async def tracking_embed(event):
        embed_called_with.append(event["id"])
        return await original_embed(event)

    lifeos._embed_event = tracking_embed

    event = _make_event(
        "email.received",
        subject="Task manager isolation test",
        body_plain="Embedding should still run even if task extraction fails.",
    )

    # Should NOT raise — the pipeline catches per-stage errors
    await lifeos.master_event_handler(event)

    # _embed_event should have been called despite task_manager failure
    assert event["id"] in embed_called_with, (
        "_embed_event (stage 5) should still run when task_manager.process_event "
        "(stage 4) fails — each stage is independently wrapped in try/except"
    )

    # Restore
    lifeos.task_manager.process_event = original_process
    lifeos._embed_event = original_embed


# ---------------------------------------------------------------------------
# Test 20: Embedding error does not block episode creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embedding_error_does_not_block_episode_creation(lifeos, db):
    """If _embed_event (stage 5) raises, the pipeline should still reach
    _create_episode (stage 6).  This validates fail-open isolation between
    the final two pipeline stages.
    """
    # Sabotage _embed_event
    original_embed = lifeos._embed_event

    async def broken_embed(event):
        raise RuntimeError("Simulated embedding failure")

    lifeos._embed_event = broken_embed

    event = _make_event(
        "email.received",
        from_address="embedding-fail@example.com",
        subject="Embedding isolation test",
        body_plain="Episode creation should still run even if embedding fails.",
    )

    # Should NOT raise
    await lifeos.master_event_handler(event)

    # Event should be stored (stage 1)
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored despite embedding failure"

    # Episode should be created (stage 6) despite embedding failure (stage 5)
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) >= 1, (
        "Episode (stage 6) should be created despite _embed_event (stage 5) failure — "
        "each pipeline stage is independently wrapped in try/except"
    )

    # Restore
    lifeos._embed_event = original_embed
