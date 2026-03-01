"""
Tests for automatic episode backfill from events.db on startup.

Verifies that LifeOS._backfill_episodes_from_events_if_needed() correctly:
- Is a no-op when episodes already exist in user_model.db
- Is a no-op when events.db has no episodic event types
- Creates episodes when user_model.db is empty but events.db has data

This auto-trigger is critical because after a user_model.db rebuild (corruption
repair), episodes are empty and the entire cognitive pipeline silently fails:
routine detection, semantic fact inference, and prediction accuracy all depend
on episodes existing.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from storage.manager import DatabaseManager


def _insert_event(db: DatabaseManager, event_type: str, payload: dict | None = None) -> str:
    """Insert a fake event into events.db and return its ID.

    Args:
        db: DatabaseManager with initialized schemas
        event_type: Event type string (e.g., "email.received")
        payload: Optional payload dict (defaults to a minimal email payload)

    Returns:
        The generated event ID
    """
    event_id = str(uuid.uuid4())
    if payload is None:
        payload = {
            "from_address": "test@example.com",
            "subject": f"Test event {event_id[:8]}",
            "body_plain": "Hello, this is a test.",
        }
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                event_type,
                "test",
                datetime.now(timezone.utc).isoformat(),
                "normal",
                json.dumps(payload),
                json.dumps({}),
            ),
        )
    return event_id


def _insert_episode(db: DatabaseManager, event_id: str | None = None) -> str:
    """Insert a fake episode into user_model.db and return its ID.

    Args:
        db: DatabaseManager with initialized schemas
        event_id: Optional event_id to link to (defaults to a random UUID)

    Returns:
        The generated episode ID
    """
    episode_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary, content_full,
                contacts_involved, topics, entities, inferred_mood)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id,
                datetime.now(timezone.utc).isoformat(),
                event_id or str(uuid.uuid4()),
                "email_received",
                "Test episode",
                json.dumps({"subject": "test"}),
                json.dumps(["test@example.com"]),
                json.dumps([]),
                json.dumps([]),
                json.dumps({}),
            ),
        )
    return episode_id


def _count_episodes(db: DatabaseManager) -> int:
    """Return the number of episodes in user_model.db."""
    with db.get_connection("user_model") as conn:
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


@pytest.mark.asyncio
async def test_noop_when_episodes_already_exist(db: DatabaseManager):
    """Backfill should be a no-op when episodes already exist in user_model.db."""
    # Insert an event and an episode so the guard triggers
    event_id = _insert_event(db, "email.received")
    _insert_episode(db, event_id)

    assert _count_episodes(db) == 1

    # Also insert another event that would be eligible for backfill
    _insert_event(db, "email.sent", {
        "to_addresses": ["bob@example.com"],
        "subject": "Re: test",
    })

    # Run the backfill method
    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)
    await app._backfill_episodes_from_events_if_needed()

    # Episode count should remain 1 — the second event should NOT be backfilled
    # because the guard sees episodes already exist
    assert _count_episodes(db) == 1


@pytest.mark.asyncio
async def test_noop_when_no_episodic_events(db: DatabaseManager):
    """Backfill should be a no-op when events.db has no episodic event types."""
    # Insert a non-episodic event type
    _insert_event(db, "system.health.check", {"status": "ok"})

    assert _count_episodes(db) == 0

    # Run the backfill method
    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)
    await app._backfill_episodes_from_events_if_needed()

    # No episodes should be created because there are no episodic events
    assert _count_episodes(db) == 0


@pytest.mark.asyncio
async def test_backfills_when_episodes_empty_and_events_exist(db: DatabaseManager):
    """Backfill should create episodes when user_model.db is empty but events.db has data."""
    # Insert several episodic events
    event_ids = []
    event_ids.append(_insert_event(db, "email.received", {
        "from_address": "alice@example.com",
        "subject": "Hello from Alice",
        "body_plain": "Hi there!",
    }))
    event_ids.append(_insert_event(db, "email.sent", {
        "to_addresses": ["bob@example.com"],
        "subject": "Meeting notes",
        "body_plain": "Here are the notes from today.",
    }))
    event_ids.append(_insert_event(db, "message.received", {
        "from_address": "carol@example.com",
        "snippet": "Can we chat?",
    }))
    event_ids.append(_insert_event(db, "task.created", {
        "title": "Review PR #42",
    }))

    assert _count_episodes(db) == 0

    # Run the backfill method
    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)
    await app._backfill_episodes_from_events_if_needed()

    # All 4 events should have generated episodes
    assert _count_episodes(db) == 4

    # Verify the episodes have correct event_id linkage
    with db.get_connection("user_model") as conn:
        cursor = conn.execute("SELECT event_id FROM episodes ORDER BY timestamp")
        episode_event_ids = {row[0] for row in cursor.fetchall()}

    for eid in event_ids:
        assert eid in episode_event_ids, f"Event {eid} should have a corresponding episode"


@pytest.mark.asyncio
async def test_backfill_is_idempotent(db: DatabaseManager):
    """Running the backfill twice should not create duplicate episodes."""
    _insert_event(db, "email.received", {
        "from_address": "alice@example.com",
        "subject": "Test idempotency",
    })

    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)

    # Run twice
    await app._backfill_episodes_from_events_if_needed()
    count_after_first = _count_episodes(db)
    assert count_after_first == 1

    # Manually clear the episode count guard by removing the episode and re-running
    # (simulates a scenario where the guard wouldn't trigger)
    # Actually, the second run will see episodes exist and return early — that's correct
    await app._backfill_episodes_from_events_if_needed()
    assert _count_episodes(db) == count_after_first


@pytest.mark.asyncio
async def test_backfill_episode_has_correct_interaction_type(db: DatabaseManager):
    """Backfilled episodes should have granular interaction types, not generic ones."""
    _insert_event(db, "email.received", {
        "from_address": "alice@example.com",
        "subject": "Test classification",
    })
    _insert_event(db, "task.completed", {
        "title": "Finish report",
    })

    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)
    await app._backfill_episodes_from_events_if_needed()

    with db.get_connection("user_model") as conn:
        cursor = conn.execute(
            "SELECT interaction_type FROM episodes ORDER BY timestamp"
        )
        types = [row[0] for row in cursor.fetchall()]

    assert "email_received" in types
    assert "task_completed" in types


@pytest.mark.asyncio
async def test_backfill_handles_empty_payload(db: DatabaseManager):
    """Backfill should handle events with empty payloads gracefully."""
    # Insert a good event
    _insert_event(db, "email.received", {
        "from_address": "alice@example.com",
        "subject": "Good event",
    })

    # Insert an event with an empty JSON payload (no useful fields)
    _insert_event(db, "email.received", {})

    from main import LifeOS

    app = LifeOS(config={"data_dir": db.data_dir}, db=db)

    # Should not raise — events with empty payloads produce episodes with defaults
    await app._backfill_episodes_from_events_if_needed()

    # Both events should have produced episodes (empty payload still produces an episode)
    assert _count_episodes(db) == 2
