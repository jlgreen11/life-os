"""
Tests for episode outcome tracking — verifying that episodes get updated
when tasks linked to them are completed.

Covers:
- UserModelStore.update_episode() COALESCE semantics
- TaskCompletionDetector wiring episode outcomes on task completion
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_completion_detector.detector import TaskCompletionDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_episode(db, episode_id, event_id, timestamp=None):
    """Insert a minimal episode row into user_model.db for testing."""
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary,
                contacts_involved, topics, entities, inferred_mood)
               VALUES (?, ?, ?, 'test', 'test summary', '[]', '[]', '[]', '{}')""",
            (episode_id, ts, event_id),
        )


def _get_episode(db, episode_id):
    """Fetch an episode row by ID."""
    with db.get_connection("user_model") as conn:
        row = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    return dict(row) if row else None


def _create_task(db, task_id, title, source=None, created_at=None, status="pending"):
    """Insert a task into state.db."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks (id, title, description, status, source, domain,
                                 priority, created_at, updated_at)
               VALUES (?, ?, '', ?, ?, 'personal', 'normal', ?, ?)""",
            (task_id, title, status, source or "ai", created_at.isoformat(), created_at.isoformat()),
        )


def _create_event(db, event_id, event_type, payload, timestamp=None):
    """Insert an event into events.db."""
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, 'test', ?, 'normal', ?, '{}')""",
            (event_id, event_type, ts, json.dumps(payload)),
        )


# ---------------------------------------------------------------------------
# UserModelStore.update_episode tests
# ---------------------------------------------------------------------------

class TestUpdateEpisode:
    """Tests for UserModelStore.update_episode() method."""

    def test_updates_outcome_field(self, db, user_model_store):
        """update_episode should set the outcome on an existing episode."""
        ep_id = str(uuid.uuid4())
        ev_id = str(uuid.uuid4())
        _store_episode(db, ep_id, ev_id)

        result = user_model_store.update_episode(ep_id, outcome="activity_match")

        assert result is True
        row = _get_episode(db, ep_id)
        assert row["outcome"] == "activity_match"

    def test_updates_user_satisfaction_field(self, db, user_model_store):
        """update_episode should set user_satisfaction on an existing episode."""
        ep_id = str(uuid.uuid4())
        ev_id = str(uuid.uuid4())
        _store_episode(db, ep_id, ev_id)

        result = user_model_store.update_episode(ep_id, user_satisfaction=0.85)

        assert result is True
        row = _get_episode(db, ep_id)
        assert row["user_satisfaction"] == pytest.approx(0.85)

    def test_updates_embedding_id_field(self, db, user_model_store):
        """update_episode should set embedding_id on an existing episode."""
        ep_id = str(uuid.uuid4())
        ev_id = str(uuid.uuid4())
        _store_episode(db, ep_id, ev_id)

        emb_id = str(uuid.uuid4())
        result = user_model_store.update_episode(ep_id, embedding_id=emb_id)

        assert result is True
        row = _get_episode(db, ep_id)
        assert row["embedding_id"] == emb_id

    def test_coalesce_preserves_existing_fields(self, db, user_model_store):
        """Updating one field must NOT overwrite previously-set fields."""
        ep_id = str(uuid.uuid4())
        ev_id = str(uuid.uuid4())
        _store_episode(db, ep_id, ev_id)

        # First: set outcome
        user_model_store.update_episode(ep_id, outcome="activity_match")
        # Second: set satisfaction (outcome should be preserved)
        user_model_store.update_episode(ep_id, user_satisfaction=0.9)

        row = _get_episode(db, ep_id)
        assert row["outcome"] == "activity_match"
        assert row["user_satisfaction"] == pytest.approx(0.9)

    def test_returns_false_for_nonexistent_episode(self, db, user_model_store):
        """update_episode should return False when the episode_id doesn't exist."""
        result = user_model_store.update_episode("nonexistent-id", outcome="stale")
        assert result is False

    def test_emits_telemetry(self, db, event_store, event_bus):
        """update_episode should emit a usermodel.episode.updated telemetry event.

        _emit_telemetry falls back to writing directly to the EventStore when
        no async loop is running (synchronous test context). We wire both
        event_bus and event_store so the fallback path persists the telemetry
        event to events.db where we can verify it.
        """
        from storage.user_model_store import UserModelStore

        ums = UserModelStore(db, event_bus=event_bus, event_store=event_store)

        ep_id = str(uuid.uuid4())
        ev_id = str(uuid.uuid4())
        _store_episode(db, ep_id, ev_id)

        ums.update_episode(ep_id, outcome="inactivity")

        # Telemetry falls back to event_store when no async loop is running.
        with db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT type, payload FROM events WHERE type = 'usermodel.episode.updated'"
            ).fetchall()

        assert len(rows) >= 1
        payload = json.loads(rows[0]["payload"])
        assert payload["episode_id"] == ep_id
        assert "outcome" in payload["fields_set"]


# ---------------------------------------------------------------------------
# TaskCompletionDetector episode-wiring tests
# ---------------------------------------------------------------------------

class TestDetectorEpisodeWiring:
    """Tests that TaskCompletionDetector updates episode outcomes on task completion."""

    @pytest.fixture
    def detector_with_ums(self, db, event_bus, user_model_store):
        """Detector wired with a real UserModelStore."""
        task_manager = MagicMock()
        task_manager.complete_task = AsyncMock()
        return TaskCompletionDetector(db, task_manager, event_bus, user_model_store=user_model_store)

    @pytest.mark.asyncio
    async def test_activity_completion_updates_episode(self, detector_with_ums, db):
        """Activity-based completion should set episode outcome to 'activity_match'."""
        base_time = datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())
        ep_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        # Create the originating event
        _create_event(db, event_id, "email.received", {
            "subject": "Invoice for Acme Corporation services",
            "body_plain": "Please send the invoice for Acme Corporation services.",
        }, timestamp=base_time - timedelta(hours=3))

        # Create episode linked to that event
        _store_episode(db, ep_id, event_id, timestamp=base_time - timedelta(hours=3))

        # Create task whose source is the event
        _create_task(db, task_id, "Send invoice to Acme Corporation",
                     source=event_id, created_at=base_time - timedelta(hours=2))

        # Create a sent email with enough keyword overlap + completion signal
        _create_event(db, str(uuid.uuid4()), "email.sent", {
            "to_addresses": ["billing@acme.com"],
            "subject": "Invoice for Acme - Q4",
            "body_plain": "The invoice for Acme Corporation has been sent and completed.",
        }, timestamp=base_time - timedelta(minutes=30))

        await detector_with_ums.detect_completions()

        row = _get_episode(db, ep_id)
        assert row["outcome"] == "activity_match"

    @pytest.mark.asyncio
    async def test_inactivity_completion_updates_episode(self, detector_with_ums, db):
        """Inactivity-based completion should set episode outcome to 'inactivity'."""
        base_time = datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())
        ep_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        _create_event(db, event_id, "email.received", {
            "subject": "Old email",
        }, timestamp=base_time - timedelta(days=10))

        _store_episode(db, ep_id, event_id, timestamp=base_time - timedelta(days=10))

        _create_task(db, task_id, "Old inactive task",
                     source=event_id, created_at=base_time - timedelta(days=10))

        await detector_with_ums.detect_completions()

        row = _get_episode(db, ep_id)
        assert row["outcome"] == "inactivity"

    @pytest.mark.asyncio
    async def test_stale_completion_updates_episode(self, detector_with_ums, db):
        """Stale task cleanup should set episode outcome to 'stale'."""
        base_time = datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())
        ep_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        _create_event(db, event_id, "email.received", {
            "subject": "Very old email",
        }, timestamp=base_time - timedelta(days=35))

        _store_episode(db, ep_id, event_id, timestamp=base_time - timedelta(days=35))

        _create_task(db, task_id, "Very old stale task",
                     source=event_id, created_at=base_time - timedelta(days=35))

        await detector_with_ums.detect_completions()

        row = _get_episode(db, ep_id)
        # Stale tasks are also older than inactivity threshold, so outcome
        # might be either 'inactivity' or 'stale' depending on which detector
        # fires first. Both are correct — the important thing is it's not None.
        assert row["outcome"] in ("inactivity", "stale")

    @pytest.mark.asyncio
    async def test_no_episode_gracefully_skipped(self, detector_with_ums, db):
        """When no episode is linked to the task, completion should still succeed."""
        base_time = datetime.now(timezone.utc)
        task_id = str(uuid.uuid4())

        # Task with no matching episode — source is an event_id with no episode
        _create_task(db, task_id, "Task with no episode",
                     source="nonexistent-event-id", created_at=base_time - timedelta(days=10))

        # Should complete without error
        completed = await detector_with_ums.detect_completions()
        assert completed >= 1

    @pytest.mark.asyncio
    async def test_detector_without_ums_still_works(self, db, event_bus):
        """Detector created without user_model_store should complete tasks normally."""
        task_manager = MagicMock()
        task_manager.complete_task = AsyncMock()
        detector = TaskCompletionDetector(db, task_manager, event_bus)

        base_time = datetime.now(timezone.utc)
        task_id = str(uuid.uuid4())
        _create_task(db, task_id, "Task without UMS", created_at=base_time - timedelta(days=10))

        completed = await detector.detect_completions()
        assert completed >= 1
        task_manager.complete_task.assert_called()
