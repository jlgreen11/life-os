"""Tests for RoutineDetector._backfill_stale_interaction_types().

Verifies that episodes with stale interaction_type values (NULL, 'unknown',
'communication') are batch-updated using derived types from their linked
events in events.db.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.routine_detector.detector import RoutineDetector


def _insert_event(db, event_id: str, event_type: str, timestamp: str | None = None):
    """Insert a minimal event row into events.db."""
    ts = timestamp or datetime.now(UTC).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority) VALUES (?, ?, ?, ?, ?)",
            (event_id, event_type, "test", ts, "normal"),
        )
        conn.commit()


def _relax_interaction_type_constraint(db):
    """Drop the NOT NULL constraint on episodes.interaction_type.

    Production databases may contain NULL interaction_type values from
    episodes created before the constraint was added.  The test schema
    enforces NOT NULL, so we rebuild the table without it to simulate
    legacy data conditions.  This only needs to be called once per test
    database.
    """
    with db.get_connection("user_model") as conn:
        # Check if we already relaxed it (idempotent)
        col_info = conn.execute("PRAGMA table_info(episodes)").fetchall()
        for col in col_info:
            if col[1] == "interaction_type" and col[3] == 0:  # notnull == 0
                return  # Already relaxed
        # SQLite doesn't support ALTER COLUMN; we must recreate the table.
        # Get the full CREATE TABLE statement and modify it.
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='episodes'"
        ).fetchone()
        if not row:
            return
        original_sql = row[0]
        # Replace 'interaction_type    TEXT NOT NULL' with 'interaction_type    TEXT'
        relaxed_sql = original_sql.replace(
            "interaction_type    TEXT NOT NULL",
            "interaction_type    TEXT",
        )
        if relaxed_sql == original_sql:
            # Try without extra spaces
            relaxed_sql = original_sql.replace(
                "interaction_type TEXT NOT NULL",
                "interaction_type TEXT",
            )
        conn.execute("ALTER TABLE episodes RENAME TO episodes_backup")
        conn.execute(relaxed_sql)
        conn.execute("INSERT INTO episodes SELECT * FROM episodes_backup")
        conn.execute("DROP TABLE episodes_backup")
        conn.commit()


def _insert_episode(
    db,
    event_id: str,
    interaction_type: str | None,
    timestamp: str | None = None,
    episode_id: str | None = None,
):
    """Insert a minimal episode row into user_model.db.

    When ``interaction_type`` is None, the NOT NULL constraint is first
    relaxed (via table rebuild) to simulate legacy production data.
    """
    ep_id = episode_id or str(uuid.uuid4())
    ts = timestamp or datetime.now(UTC).isoformat()
    if interaction_type is None:
        _relax_interaction_type_constraint(db)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary)
               VALUES (?, ?, ?, ?, ?)""",
            (ep_id, ts, event_id, interaction_type, "test episode"),
        )
        conn.commit()
    return ep_id


def _get_episode_interaction_type(db, episode_id: str) -> str | None:
    """Read interaction_type for a given episode."""
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT interaction_type FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
    return row[0] if row else None


@pytest.fixture()
def detector(db, user_model_store):
    """A RoutineDetector wired to the temporary DatabaseManager."""
    return RoutineDetector(db, user_model_store, timezone="UTC")


class TestBackfillUpdatesNullInteractionTypes:
    """Episodes with NULL interaction_type should be backfilled."""

    def test_backfill_updates_null_interaction_types(self, db, detector):
        """Episodes with NULL types get updated from linked event types."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "email.received")
        ep_id = _insert_episode(db, ev_id, interaction_type=None)

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == "email_received"


class TestBackfillUpdatesCommunicationTypes:
    """Episodes with 'communication' interaction_type should be backfilled."""

    def test_backfill_updates_communication_types(self, db, detector):
        """Episodes with 'communication' type get updated from linked event types."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "message.received")
        ep_id = _insert_episode(db, ev_id, interaction_type="communication")

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == "message_received"

    def test_backfill_updates_unknown_types(self, db, detector):
        """Episodes with 'unknown' type get updated from linked event types."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "task.created")
        ep_id = _insert_episode(db, ev_id, interaction_type="unknown")

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == "task_created"


class TestBackfillPreservesGoodTypes:
    """Episodes with already-classified types should NOT be modified."""

    def test_backfill_preserves_good_types(self, db, detector):
        """Episodes with 'email_received' should NOT be changed."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "email.received")
        ep_id = _insert_episode(db, ev_id, interaction_type="email_received")

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == "email_received"

    def test_backfill_only_touches_stale_episodes(self, db, detector):
        """Mix of stale and good episodes — only stale ones get updated."""
        ev_id_1 = str(uuid.uuid4())
        ev_id_2 = str(uuid.uuid4())
        _insert_event(db, ev_id_1, "email.received")
        _insert_event(db, ev_id_2, "calendar.event.created")

        ep_good = _insert_episode(db, ev_id_1, interaction_type="email_received")
        ep_stale = _insert_episode(db, ev_id_2, interaction_type=None)

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_good) == "email_received"
        assert _get_episode_interaction_type(db, ep_stale) == "calendar_event_created"


class TestBackfillHandlesMissingEvents:
    """Episodes whose event_id doesn't exist in events.db should be skipped."""

    def test_backfill_handles_missing_events(self, db, detector):
        """Episodes with nonexistent event_ids are skipped gracefully."""
        missing_ev_id = str(uuid.uuid4())
        real_ev_id = str(uuid.uuid4())
        _insert_event(db, real_ev_id, "task.completed")

        ep_missing = _insert_episode(db, missing_ev_id, interaction_type=None)
        ep_real = _insert_episode(db, real_ev_id, interaction_type="communication")

        detector._backfill_stale_interaction_types(lookback_days=30)

        # Missing event episode stays NULL
        assert _get_episode_interaction_type(db, ep_missing) is None
        # Real event episode gets updated
        assert _get_episode_interaction_type(db, ep_real) == "task_completed"


class TestBackfillMapsEventTypesCorrectly:
    """Verify dot-to-underscore conversion for various event types."""

    @pytest.mark.parametrize(
        "event_type,expected",
        [
            ("email.received", "email_received"),
            ("email.sent", "email_sent"),
            ("calendar.event.created", "calendar_event_created"),
            ("message.received", "message_received"),
            ("browser.page_visited", "browser_page_visited"),
            ("task.created", "task_created"),
            ("system.connector.sync_complete", "system_connector_sync_complete"),
        ],
    )
    def test_backfill_maps_event_types_correctly(self, db, detector, event_type, expected):
        """Event types are converted with dot-to-underscore, not EVENT_TYPE_TO_ACTIVITY."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, event_type)
        ep_id = _insert_episode(db, ev_id, interaction_type=None)

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == expected


class TestBackfillSkipsOutOfWindow:
    """Episodes outside the lookback window should not be backfilled."""

    def test_old_episodes_not_backfilled(self, db, detector):
        """Episodes older than lookback_days are left alone."""
        ev_id = str(uuid.uuid4())
        old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        _insert_event(db, ev_id, "email.received", timestamp=old_ts)
        ep_id = _insert_episode(db, ev_id, interaction_type=None, timestamp=old_ts)

        detector._backfill_stale_interaction_types(lookback_days=30)

        # Should still be NULL — outside lookback window
        assert _get_episode_interaction_type(db, ep_id) is None


class TestFallbackUsesCorrectDerivation:
    """The episode-based fallback should use _derive_interaction_type_from_event,
    not _classify_event_type_to_activity, to avoid type name fragmentation."""

    def test_fallback_uses_dot_to_underscore(self, db, detector):
        """Verify the fallback path produces 'email_received' not 'email_check'."""
        # _derive_interaction_type_from_event uses dot→underscore
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "email.received")

        result = detector._derive_interaction_type_from_event(ev_id)
        assert result == "email_received"

        # _classify_event_type_to_activity uses EVENT_TYPE_TO_ACTIVITY mapping
        result_old = detector._classify_event_type_to_activity(ev_id)
        assert result_old == "email_check"  # This is the WRONG mapping for this context

        # Confirm they produce DIFFERENT results — proving the bug matters
        assert result != result_old
