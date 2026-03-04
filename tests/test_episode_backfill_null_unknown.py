"""
Tests for episode classification backfill handling NULL and 'unknown' interaction types.

Verifies that LifeOS._backfill_episode_classification_if_needed() correctly:
- Reclassifies episodes with interaction_type=NULL based on linked event type
- Reclassifies episodes with interaction_type='unknown'
- Still reclassifies episodes with interaction_type='communication' (existing behavior)
- Skips episodes whose linked events don't exist in events.db
- Does NOT touch episodes already classified with granular types
- Is idempotent — running twice produces the same result
- Skips UPDATE when reclassification still yields 'unknown'

These are critical for routine detection, which excludes NULL, 'unknown', and
'communication' episodes from its primary query.

Note: The current schema has NOT NULL on interaction_type. NULL episodes exist
in production databases that predate this constraint. Tests for NULL use a
helper that relaxes the constraint to simulate pre-migration data.
"""

import json
import uuid
from datetime import datetime, timezone

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


def _insert_episode(
    db: DatabaseManager,
    event_id: str,
    interaction_type: str = "communication",
) -> str:
    """Insert an episode into user_model.db with a specific interaction_type.

    Args:
        db: DatabaseManager with initialized schemas
        event_id: Event ID to link this episode to
        interaction_type: The interaction_type value

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
                event_id,
                interaction_type,
                "Test episode",
                json.dumps({"subject": "test"}),
                json.dumps(["test@example.com"]),
                json.dumps([]),
                json.dumps([]),
                json.dumps({}),
            ),
        )
    return episode_id


def _relax_not_null_constraint(db: DatabaseManager):
    """Recreate the episodes table without NOT NULL on interaction_type.

    Simulates a pre-migration database where the constraint didn't exist.
    Must be called BEFORE inserting any episodes that need NULL interaction_type.
    """
    with db.get_connection("user_model") as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes_backup AS SELECT * FROM episodes;
            DROP TABLE episodes;
            CREATE TABLE episodes (
                id                  TEXT PRIMARY KEY,
                timestamp           TEXT NOT NULL,
                event_id            TEXT NOT NULL,
                location            TEXT,
                inferred_mood       TEXT,
                active_domain       TEXT,
                energy_level        REAL,
                interaction_type    TEXT,
                content_summary     TEXT NOT NULL,
                content_full        TEXT,
                contacts_involved   TEXT DEFAULT '[]',
                topics              TEXT DEFAULT '[]',
                entities            TEXT DEFAULT '[]',
                outcome             TEXT,
                user_satisfaction   REAL,
                embedding_id        TEXT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            INSERT INTO episodes SELECT * FROM episodes_backup;
            DROP TABLE episodes_backup;
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(interaction_type);
            CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(active_domain);
        """)


def _insert_null_episode(db: DatabaseManager, event_id: str) -> str:
    """Insert an episode with NULL interaction_type (requires relaxed constraint).

    Args:
        db: DatabaseManager with relaxed NOT NULL constraint on episodes
        event_id: Event ID to link this episode to

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
                event_id,
                None,
                "Test episode",
                json.dumps({"subject": "test"}),
                json.dumps(["test@example.com"]),
                json.dumps([]),
                json.dumps([]),
                json.dumps({}),
            ),
        )
    return episode_id


def _get_interaction_type(db: DatabaseManager, episode_id: str) -> str | None:
    """Return the interaction_type for a specific episode."""
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT interaction_type FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        return row[0] if row else None


def _make_app(db: DatabaseManager):
    """Create a LifeOS instance wired to the test database."""
    from main import LifeOS

    return LifeOS(config={"data_dir": db.data_dir}, db=db)


@pytest.mark.asyncio
async def test_null_interaction_type_is_reclassified(db: DatabaseManager):
    """Episodes with interaction_type=NULL should be reclassified based on their linked event."""
    # Relax the NOT NULL constraint to simulate a pre-migration database
    _relax_not_null_constraint(db)

    event_id = _insert_event(db, "email.received")
    episode_id = _insert_null_episode(db, event_id)

    assert _get_interaction_type(db, episode_id) is None

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, episode_id) == "email_received"


@pytest.mark.asyncio
async def test_unknown_interaction_type_is_reclassified(db: DatabaseManager):
    """Episodes with interaction_type='unknown' should be reclassified."""
    event_id = _insert_event(db, "email.sent", {
        "to_addresses": ["bob@example.com"],
        "subject": "Test",
    })
    episode_id = _insert_episode(db, event_id, interaction_type="unknown")

    assert _get_interaction_type(db, episode_id) == "unknown"

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, episode_id) == "email_sent"


@pytest.mark.asyncio
async def test_communication_interaction_type_still_reclassified(db: DatabaseManager):
    """Episodes with interaction_type='communication' should still be reclassified (existing behavior)."""
    event_id = _insert_event(db, "message.received", {
        "from_address": "alice@example.com",
        "snippet": "Hey!",
    })
    episode_id = _insert_episode(db, event_id, interaction_type="communication")

    assert _get_interaction_type(db, episode_id) == "communication"

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, episode_id) == "message_received"


@pytest.mark.asyncio
async def test_missing_event_is_skipped_gracefully(db: DatabaseManager):
    """Episodes whose linked events don't exist in events.db should be skipped."""
    # Create an episode pointing to a non-existent event
    fake_event_id = str(uuid.uuid4())
    episode_id = _insert_episode(db, fake_event_id, interaction_type="unknown")

    app = _make_app(db)
    # Should not raise
    await app._backfill_episode_classification_if_needed()

    # interaction_type remains 'unknown' since the event was not found and
    # the reclassification couldn't run
    assert _get_interaction_type(db, episode_id) == "unknown"


@pytest.mark.asyncio
async def test_missing_event_skipped_for_null_episodes(db: DatabaseManager):
    """NULL episodes whose linked events don't exist should remain NULL."""
    _relax_not_null_constraint(db)

    fake_event_id = str(uuid.uuid4())
    episode_id = _insert_null_episode(db, fake_event_id)

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    # Should still be NULL — event was not found
    assert _get_interaction_type(db, episode_id) is None


@pytest.mark.asyncio
async def test_granular_types_are_not_touched(db: DatabaseManager):
    """Episodes already classified with granular types (e.g., 'email_received') should NOT be modified."""
    event_id = _insert_event(db, "email.received")
    episode_id = _insert_episode(db, event_id, interaction_type="email_received")

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    # Should still be email_received — backfill should not have touched it
    assert _get_interaction_type(db, episode_id) == "email_received"


@pytest.mark.asyncio
async def test_backfill_is_idempotent(db: DatabaseManager):
    """Running the backfill twice should produce the same result."""
    event_id_1 = _insert_event(db, "email.received")
    event_id_2 = _insert_event(db, "task.created", {"title": "Test task"})

    ep_id_1 = _insert_episode(db, event_id_1, interaction_type="unknown")
    ep_id_2 = _insert_episode(db, event_id_2, interaction_type="communication")

    app = _make_app(db)

    # First run
    await app._backfill_episode_classification_if_needed()
    type_1_after_first = _get_interaction_type(db, ep_id_1)
    type_2_after_first = _get_interaction_type(db, ep_id_2)

    assert type_1_after_first == "email_received"
    assert type_2_after_first == "task_created"

    # Second run — should be a no-op (stale_count == 0)
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, ep_id_1) == type_1_after_first
    assert _get_interaction_type(db, ep_id_2) == type_2_after_first


@pytest.mark.asyncio
async def test_backfill_idempotent_with_null(db: DatabaseManager):
    """Idempotency also holds for NULL episodes (with relaxed constraint)."""
    _relax_not_null_constraint(db)

    event_id = _insert_event(db, "email.received")
    ep_id = _insert_null_episode(db, event_id)

    app = _make_app(db)

    await app._backfill_episode_classification_if_needed()
    assert _get_interaction_type(db, ep_id) == "email_received"

    # Second run — no-op
    await app._backfill_episode_classification_if_needed()
    assert _get_interaction_type(db, ep_id) == "email_received"


@pytest.mark.asyncio
async def test_mixed_stale_types_all_reclassified(db: DatabaseManager):
    """A mix of NULL, 'unknown', and 'communication' episodes should all be reclassified."""
    _relax_not_null_constraint(db)

    ev1 = _insert_event(db, "email.received")
    ev2 = _insert_event(db, "email.sent", {"to_addresses": ["x@y.com"], "subject": "Hi"})
    ev3 = _insert_event(db, "task.completed", {"title": "Done"})

    ep1 = _insert_null_episode(db, ev1)
    ep2 = _insert_episode(db, ev2, interaction_type="unknown")
    ep3 = _insert_episode(db, ev3, interaction_type="communication")

    app = _make_app(db)
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, ep1) == "email_received"
    assert _get_interaction_type(db, ep2) == "email_sent"
    assert _get_interaction_type(db, ep3) == "task_completed"


@pytest.mark.asyncio
async def test_noop_when_no_stale_episodes(db: DatabaseManager):
    """Backfill should be a no-op when all episodes already have granular types."""
    event_id = _insert_event(db, "email.received")
    episode_id = _insert_episode(db, event_id, interaction_type="email_received")

    app = _make_app(db)
    # Should return immediately without errors
    await app._backfill_episode_classification_if_needed()

    assert _get_interaction_type(db, episode_id) == "email_received"
