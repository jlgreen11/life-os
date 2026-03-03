"""
Life OS — System Event Guard Tests

Verifies that internal/system events are correctly guarded from re-entering
the full pipeline stages 2-6 (signal extraction, rules evaluation, task
extraction, vector embedding, episode creation).

The guard at main.py:master_event_handler skips stages 2-6 for events whose
type starts with any prefix in _SYSTEM_EVENT_PREFIXES:
  - "system."        — rule triggered, connector sync, AI actions
  - "notification."  — notification.created, notification.dismissed
  - "task."          — task.created, task.completed
  - "usermodel."     — signal_profile.updated, episode.stored

These events are still persisted (stage 1) and tracked for source weights
(stage 1.3) because those stages run before the guard.

Without this guard, internal meta-events would wastefully pass through signal
extraction, rules evaluation, and task extraction — consuming CPU and risking
cascade loops if any rule matches broad event patterns.
"""

import uuid
from datetime import datetime, timezone

import pytest

from models.core import EventType


# ---------------------------------------------------------------------------
# Fixtures (reused from test_master_event_handler_pipeline.py pattern)
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

    # Extract the handler from the mock's call args so tests can invoke
    # it directly without needing to publish through the bus.
    handler = event_bus.subscribe_all.call_args[0][0]
    los.master_event_handler = handler

    return los


def _make_event(event_type: str, **payload_overrides) -> dict:
    """Build a well-formed event dict with sensible defaults.

    Args:
        event_type: The event type string (e.g., "notification.created").
        **payload_overrides: Fields merged into the default payload.

    Returns:
        A complete event dict ready for master_event_handler.
    """
    payload = {"detail": "test event"}
    payload.update(payload_overrides)

    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "internal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "low",
        "payload": payload,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Helper: instrument signal_extractor to detect whether it was called
# ---------------------------------------------------------------------------

def _track_signal_extractor(lifeos):
    """Wrap signal_extractor.process_event to record which event IDs it sees.

    Returns:
        A list that accumulates event IDs as they are processed.
        Call this before invoking master_event_handler, then check the list.
    """
    called_with = []
    original = lifeos.signal_extractor.process_event

    async def tracking(event):
        called_with.append(event["id"])
        return await original(event)

    lifeos.signal_extractor.process_event = tracking
    return called_with


# ---------------------------------------------------------------------------
# Test 1: notification.created events are skipped by stages 2-6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notification_created_skips_signal_extraction(lifeos, db):
    """Events with type 'notification.created' should be stored (stage 1)
    but NOT processed through signal extraction (stage 2) or later stages.
    """
    called_with = _track_signal_extractor(lifeos)

    event = _make_event(
        "notification.created",
        notification_id="notif-123",
        message="You have a new email",
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "notification.created event should be stored in events.db"

    # Stage 2: signal extractor should NOT have been called
    assert event["id"] not in called_with, (
        "signal_extractor.process_event should NOT be called for notification.created — "
        "the system event guard should skip stages 2-6"
    )

    # No episode should be created (stage 6 is also skipped)
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 0, "notification.created should NOT create an episode"


# ---------------------------------------------------------------------------
# Test 2: task.created events are skipped by stages 2-6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_created_skips_signal_extraction(lifeos, db):
    """Events with type 'task.created' should be stored (stage 1)
    but NOT processed through signal extraction or later stages.
    """
    called_with = _track_signal_extractor(lifeos)

    event = _make_event(
        "task.created",
        task_id="task-456",
        title="Review quarterly report",
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "task.created event should be stored in events.db"

    # Stage 2: signal extractor should NOT be called
    assert event["id"] not in called_with, (
        "signal_extractor.process_event should NOT be called for task.created — "
        "the system event guard should skip stages 2-6"
    )

    # No episode should be created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 0, "task.created should NOT create an episode"


# ---------------------------------------------------------------------------
# Test 3: usermodel.signal_profile.updated events are skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usermodel_signal_profile_updated_skips_pipeline(lifeos, db):
    """Events with type 'usermodel.signal_profile.updated' should be stored
    but NOT processed through stages 2-6.
    """
    called_with = _track_signal_extractor(lifeos)

    event = _make_event(
        "usermodel.signal_profile.updated",
        profile_type="linguistic",
        samples_count=42,
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "usermodel.* event should be stored in events.db"

    # Stage 2: signal extractor should NOT be called
    assert event["id"] not in called_with, (
        "signal_extractor.process_event should NOT be called for usermodel.* events — "
        "the system event guard should skip stages 2-6"
    )

    # No episode should be created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 0, "usermodel.* events should NOT create an episode"


# ---------------------------------------------------------------------------
# Test 4: email.received still passes through all stages normally
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_email_received_passes_through_full_pipeline(lifeos, db):
    """Events with type 'email.received' are user content and must NOT be
    blocked by the system event guard — they should pass through all stages.
    """
    called_with = _track_signal_extractor(lifeos)

    event = _make_event(
        "email.received",
        from_address="alice@example.com",
        to_addresses=["user@example.com"],
        subject="Guard bypass test",
        body_plain="This is a real user email that should be fully processed.",
        message_id="<guard-test-msg>",
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "email.received event should be stored"

    # Stage 2: signal extractor SHOULD be called for user content events
    assert event["id"] in called_with, (
        "signal_extractor.process_event SHOULD be called for email.received — "
        "the system event guard must not block user content events"
    )

    # Stage 6: episode should be created for user content events
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 1, "email.received should create an episode"


# ---------------------------------------------------------------------------
# Test 5: system.connector.sync_complete is still skipped (existing behavior)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_system_connector_sync_complete_still_skipped(lifeos, db):
    """Events with type 'system.connector.sync_complete' should continue to be
    skipped by the guard (this was the original behavior before the expansion).
    """
    called_with = _track_signal_extractor(lifeos)

    event = _make_event(
        EventType.CONNECTOR_SYNC_COMPLETE.value,
        connector_id="google-gmail",
        events_synced=15,
    )

    await lifeos.master_event_handler(event)

    # Stage 1: event should be stored
    with db.get_connection("events") as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event["id"],)
        ).fetchone()
    assert row is not None, "system.connector.sync_complete should be stored"

    # Stage 2: signal extractor should NOT be called
    assert event["id"] not in called_with, (
        "signal_extractor.process_event should NOT be called for "
        "system.connector.sync_complete — existing guard behavior must be preserved"
    )

    # No episode should be created
    with db.get_connection("user_model") as conn:
        episodes = conn.execute(
            "SELECT * FROM episodes WHERE event_id = ?", (event["id"],)
        ).fetchall()
    assert len(episodes) == 0, "system.connector.sync_complete should NOT create an episode"
