"""
Life OS — WebSocket Broadcast Integration Tests

Verifies that ws_manager.broadcast() is called at the correct pipeline
points so the dashboard receives real-time updates instead of relying
on polling.

Broadcast calls are tested by patching the module-level ``ws_manager``
singleton in ``web.websocket``.  Every broadcast site is wrapped in
try/except, so these tests also verify that a broadcast failure never
crashes the event pipeline (fail-open convention).
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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

    Follows the same injection pattern used in
    test_master_event_handler_pipeline.py.
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

    # Extract the handler for direct invocation in tests.
    handler = event_bus.subscribe_all.call_args[0][0]
    los.master_event_handler = handler

    return los


def _make_event(event_type: str = "email.received", **payload_overrides) -> dict:
    """Build a well-formed event dict with sensible defaults."""
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
# Test 1: Broadcast on new event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_on_new_event(lifeos):
    """Processing an email event should trigger a broadcast with type='event'."""
    mock_broadcast = AsyncMock()
    event = _make_event("email.received")

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos.master_event_handler(event)

    # The first broadcast call should be the 'event' broadcast fired
    # right after stage 1 (event storage).
    event_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "event"
    ]
    assert len(event_calls) >= 1, "broadcast with type='event' should be called"
    msg = event_calls[0][0][0]
    assert msg["event_type"] == "email.received"
    assert msg["event_id"] == event["id"]


# ---------------------------------------------------------------------------
# Test 2: No broadcast for system events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_on_system_event(lifeos):
    """System events still trigger the event broadcast (it fires before the guard).

    The broadcast fires after stage 1 (storage) and before the system-event
    guard that skips stages 2-6.  So system events DO get the initial
    'event' broadcast but do NOT get a 'mood_update' broadcast.
    """
    mock_broadcast = AsyncMock()
    event = {
        "id": str(uuid.uuid4()),
        "type": EventType.CONNECTOR_SYNC_COMPLETE.value,
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": {"connector_id": "google-gmail"},
        "metadata": {},
    }

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos.master_event_handler(event)

    # System events get the 'event' broadcast (fires before guard)
    event_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "event"
    ]
    assert len(event_calls) >= 1, "System events still get an 'event' broadcast"

    # But they should NOT get a 'mood_update' broadcast (guard skips stages 2-6)
    mood_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "mood_update"
    ]
    assert len(mood_calls) == 0, "System events should NOT trigger a mood_update broadcast"


# ---------------------------------------------------------------------------
# Test 3: Broadcast on notification action
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_on_notification_action(lifeos):
    """_execute_rule_action with a 'notify' action should broadcast type='notification'."""
    mock_broadcast = AsyncMock()
    event = _make_event("email.received")
    action = {"type": "notify", "message": "New email received"}

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos._execute_rule_action(action, event)

    notif_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "notification"
    ]
    assert len(notif_calls) == 1, "broadcast with type='notification' should be called"
    msg = notif_calls[0][0][0]
    assert "title" in msg
    assert msg["source_event_id"] == event["id"]


# ---------------------------------------------------------------------------
# Test 4: Suppressed event does NOT broadcast notification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_suppressed_notification(lifeos):
    """_execute_rule_action on a suppressed event should NOT broadcast a notification."""
    mock_broadcast = AsyncMock()
    event = _make_event("email.received")
    event["_suppressed"] = True
    action = {"type": "notify", "message": "Should not fire"}

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos._execute_rule_action(action, event)

    notif_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "notification"
    ]
    assert len(notif_calls) == 0, "Suppressed events should NOT broadcast a notification"


# ---------------------------------------------------------------------------
# Test 5: Broadcast failure does not crash the pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_failure_does_not_crash_pipeline(lifeos, db):
    """If ws_manager.broadcast() raises, the rest of the pipeline still executes."""
    mock_broadcast = AsyncMock(side_effect=RuntimeError("WebSocket dead"))
    event = _make_event("email.received")

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        # Should NOT raise — the pipeline catches broadcast errors
        await lifeos.master_event_handler(event)

    # Stage 1: event should still be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "Event should be stored despite broadcast failure"

    # Stage 6: episode should still be created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) >= 1, "Episode should be created despite broadcast failure"


# ---------------------------------------------------------------------------
# Test 6: Mood update broadcast for content-bearing events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mood_update_broadcast_for_email(lifeos):
    """Content-bearing events (email.*) should trigger a mood_update broadcast."""
    mock_broadcast = AsyncMock()
    event = _make_event("email.received")

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos.master_event_handler(event)

    mood_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "mood_update"
    ]
    assert len(mood_calls) >= 1, "email events should trigger a mood_update broadcast"


# ---------------------------------------------------------------------------
# Test 7: No mood update broadcast for non-content events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_mood_update_for_non_content_event(lifeos):
    """Events that are not email/message/chat should NOT trigger mood_update."""
    mock_broadcast = AsyncMock()
    event = _make_event("calendar.event_created")

    with patch("main.ws_manager") as patched:
        patched.broadcast = mock_broadcast
        await lifeos.master_event_handler(event)

    mood_calls = [
        c for c in mock_broadcast.call_args_list
        if c[0][0].get("type") == "mood_update"
    ]
    assert len(mood_calls) == 0, "calendar events should NOT trigger a mood_update broadcast"
