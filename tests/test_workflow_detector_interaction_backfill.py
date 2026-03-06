"""Tests for WorkflowDetector._backfill_stale_interaction_types().

Verifies that episodes with stale interaction_type values (NULL, 'unknown',
'communication') are batch-updated using derived types from their linked
events in events.db — identical logic to the RoutineDetector backfill but
invoked by the WorkflowDetector before workflow detection runs.
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from services.workflow_detector.detector import WorkflowDetector


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
    legacy data conditions.
    """
    with db.get_connection("user_model") as conn:
        col_info = conn.execute("PRAGMA table_info(episodes)").fetchall()
        for col in col_info:
            if col[1] == "interaction_type" and col[3] == 0:  # notnull == 0
                return  # Already relaxed
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='episodes'"
        ).fetchone()
        if not row:
            return
        original_sql = row[0]
        relaxed_sql = original_sql.replace(
            "interaction_type    TEXT NOT NULL",
            "interaction_type    TEXT",
        )
        if relaxed_sql == original_sql:
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
    """Insert a minimal episode row into user_model.db."""
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
    """A WorkflowDetector wired to the temporary DatabaseManager."""
    return WorkflowDetector(db, user_model_store)


class TestBackfillUpdatesNullInteractionTypes:
    """Episodes with NULL interaction_type should be backfilled."""

    def test_backfill_updates_null_interaction_types(self, db, detector):
        """Episodes with NULL types get updated from linked event types."""
        ev_id = str(uuid.uuid4())
        _insert_event(db, ev_id, "email.received")
        ep_id = _insert_episode(db, ev_id, interaction_type=None)

        detector._backfill_stale_interaction_types(lookback_days=30)

        assert _get_episode_interaction_type(db, ep_id) == "email_received"


class TestBackfillUpdatesStaleTypes:
    """Episodes with 'unknown' or 'communication' should be backfilled."""

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


class TestBackfillChunkedProcessing:
    """Chunked processing works for >900 episodes."""

    def test_backfill_processes_more_than_chunk_size(self, db, detector):
        """Over 900 stale episodes are processed in multiple chunks."""
        _relax_interaction_type_constraint(db)
        episode_ids = []
        # Create 950 episodes (above the 900 chunk size)
        for i in range(950):
            ev_id = str(uuid.uuid4())
            _insert_event(db, ev_id, "email.received")
            ep_id = _insert_episode(db, ev_id, interaction_type=None)
            episode_ids.append(ep_id)

        detector._backfill_stale_interaction_types(lookback_days=30)

        # Spot-check first, middle, and last
        assert _get_episode_interaction_type(db, episode_ids[0]) == "email_received"
        assert _get_episode_interaction_type(db, episode_ids[475]) == "email_received"
        assert _get_episode_interaction_type(db, episode_ids[949]) == "email_received"


class TestBackfillCalledBeforeDetection:
    """Backfill is called before detection strategies in detect_workflows()."""

    def test_detect_workflows_calls_backfill(self, db, detector):
        """detect_workflows() calls _backfill_stale_interaction_types before strategies."""
        call_order = []

        original_backfill = detector._backfill_stale_interaction_types
        original_email = detector._detect_email_workflows

        def mock_backfill(lookback_days):
            call_order.append("backfill")
            return original_backfill(lookback_days)

        def mock_email(lookback_days):
            call_order.append("email")
            return original_email(lookback_days)

        with patch.object(detector, "_backfill_stale_interaction_types", side_effect=mock_backfill), \
             patch.object(detector, "_detect_email_workflows", side_effect=mock_email):
            detector.detect_workflows(lookback_days=30)

        assert "backfill" in call_order
        assert call_order.index("backfill") < call_order.index("email")


class TestBackfillFailOpen:
    """DB errors during backfill don't crash detect_workflows()."""

    def test_backfill_failure_does_not_crash_detection(self, db, detector):
        """If backfill raises an exception, detect_workflows() still runs."""
        with patch.object(
            detector,
            "_backfill_stale_interaction_types",
            side_effect=RuntimeError("simulated DB error"),
        ):
            # Should not raise — fail-open behavior
            workflows = detector.detect_workflows(lookback_days=30)
            assert isinstance(workflows, list)


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
        """Event types are converted with dot-to-underscore."""
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

        assert _get_episode_interaction_type(db, ep_id) is None
