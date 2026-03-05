"""
Tests for TaskCompletionDetector detection strategies — comprehensive coverage.

Each detection strategy (activity-based, inactivity-based, stale cleanup) is
tested in isolation with boundary conditions, then verified together in
integration scenarios.  These tests complement the base test file by covering
edge cases: stem matching, boundary thresholds, episode outcome updates,
event bus telemetry, and cross-strategy interaction.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.task_completion_detector.detector import TaskCompletionDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def task_manager(db):
    """Mock task manager that also updates the real DB on complete_task().

    The real TaskManager.complete_task() updates the tasks table so that
    subsequent strategies don't re-process already-completed tasks.  Our mock
    replicates this side effect so integration tests see accurate counts.
    """
    mgr = MagicMock()

    async def _complete_task(task_id):
        """Mark the task as completed in the database (side effect)."""
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("state") as conn:
            conn.execute(
                "UPDATE tasks SET status = 'completed', completed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, task_id),
            )

    mgr.complete_task = AsyncMock(side_effect=_complete_task)
    return mgr


@pytest.fixture
def detector(db, task_manager, event_bus):
    """TaskCompletionDetector wired to real DB, mock task_manager, mock event bus."""
    return TaskCompletionDetector(db, task_manager, event_bus)


@pytest.fixture
def detector_with_ums(db, task_manager, event_bus, user_model_store):
    """Detector with a real UserModelStore for episode-outcome tests."""
    return TaskCompletionDetector(db, task_manager, event_bus, user_model_store)


@pytest.fixture
def now():
    """Fixed 'now' timestamp for deterministic time arithmetic."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_task(db, title, *, task_id=None, description="", status="pending",
                 created_at=None, source=None):
    """Insert a task row directly into state.db and return the task ID."""
    task_id = task_id or str(uuid.uuid4())
    created_at = created_at or datetime.now(timezone.utc)
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, description, status, source, domain, priority,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'personal', 'normal', ?, ?)""",
            (task_id, title, description, status,
             source or "ai", created_at.isoformat(), created_at.isoformat()),
        )
    return task_id


def _insert_event(db, event_type, payload, *, timestamp=None, event_id=None):
    """Insert an event row directly into events.db and return the event ID."""
    event_id = event_id or str(uuid.uuid4())
    timestamp = timestamp or datetime.now(timezone.utc)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, 'test', ?, 'normal', ?, '{}')""",
            (event_id, event_type, timestamp.isoformat(), json.dumps(payload)),
        )
    return event_id


# ===================================================================
# Strategy 1 — Activity-based completion
# ===================================================================

class TestActivityBasedCompletion:
    """Tests for _detect_activity_based_completion."""

    @pytest.mark.asyncio
    async def test_matching_email_with_task_keywords_completes_task(self, detector, db, now):
        """An email.sent event whose body overlaps with the task title and
        contains a completion keyword should mark the task complete."""
        task_id = _insert_task(
            db, "Send quarterly report to manager",
            created_at=now - timedelta(hours=3),
        )
        _insert_event(db, "email.sent", {
            "subject": "Quarterly report",
            "body_plain": (
                "Hi, the quarterly report for the manager has been sent. "
                "Please confirm receipt."
            ),
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()

        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_unrelated_email_does_not_complete_task(self, detector, db, now):
        """An email that shares no keywords with the task should not trigger."""
        _insert_task(
            db, "Send quarterly report to manager",
            created_at=now - timedelta(hours=3),
        )
        _insert_event(db, "email.sent", {
            "subject": "Lunch plans",
            "body_plain": "Hey, want to grab lunch tomorrow? I'm done with my workout.",
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_old_task_excluded_from_activity_detection(self, detector, db, now):
        """Tasks older than inactivity_days are NOT checked by this strategy.

        The activity-based query filters ``created_at >= cutoff`` so that very
        old tasks are left for the inactivity/stale strategies instead.
        """
        _insert_task(
            db, "Send quarterly report to manager",
            created_at=now - timedelta(days=10),  # older than 7-day window
        )
        _insert_event(db, "email.sent", {
            "subject": "Quarterly report",
            "body_plain": "The quarterly report for the manager has been sent.",
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stem_matching_catches_word_variants(self, detector, db, now):
        """Stem matching (first 4 chars) should match word variants like
        'reporting' against a task containing 'report'."""
        task_id = _insert_task(
            db, "Complete project reporting deliverables",
            created_at=now - timedelta(hours=4),
        )
        # "proj" matches "project", "repo" matches "reporting",
        # "deli" matches "deliverables" — via stem overlap.
        # "completed" is a completion keyword.
        _insert_event(db, "email.sent", {
            "subject": "Project deliverables",
            "body_plain": "All project reporting deliverables are completed.",
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_completion_keyword_required(self, detector, db, now):
        """Even with strong keyword overlap, missing a completion keyword
        means the task is NOT completed."""
        _insert_task(
            db, "Update project timeline document",
            created_at=now - timedelta(hours=2),
        )
        # High keyword overlap ("project", "timeline", "document") but NO
        # completion signal word.
        _insert_event(db, "email.sent", {
            "subject": "Project timeline",
            "body_plain": "I'm still working on the project timeline document. Will update tomorrow.",
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 0

    @pytest.mark.asyncio
    async def test_message_sent_event_also_triggers(self, detector, db, now):
        """message.sent events should be checked alongside email.sent."""
        task_id = _insert_task(
            db, "Review contract with legal team",
            created_at=now - timedelta(hours=4),
        )
        _insert_event(db, "message.sent", {
            "body_plain": (
                "Finished the contract review with legal. Everything resolved."
            ),
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_event_before_task_creation_ignored(self, detector, db, now):
        """Sent events timestamped before the task was created must be ignored."""
        _insert_task(
            db, "Send invoice to Acme Corporation",
            created_at=now - timedelta(hours=2),
        )
        # Event happened BEFORE the task was created
        _insert_event(db, "email.sent", {
            "subject": "Invoice for Acme",
            "body_plain": "Sent the Acme Corporation invoice. Done!",
        }, timestamp=now - timedelta(hours=5))

        count = await detector._detect_activity_based_completion()
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_title_task_skipped(self, detector, db, now):
        """Tasks with very short titles (no extractable keywords) are skipped."""
        _insert_task(db, "Go", created_at=now - timedelta(hours=1))
        _insert_event(db, "email.sent", {
            "body_plain": "Let's go! Done!",
        }, timestamp=now - timedelta(minutes=30))

        count = await detector._detect_activity_based_completion()
        # "Go" is only 2 chars — below the 4-char threshold for keyword extraction
        assert count == 0

    @pytest.mark.asyncio
    async def test_payload_with_no_text_fields_skipped(self, detector, db, now):
        """Events whose payload has no recognized text fields should be skipped."""
        _insert_task(
            db, "Send quarterly report to manager",
            created_at=now - timedelta(hours=2),
        )
        # Payload with no subject/body_plain/snippet/description/summary/title
        _insert_event(db, "email.sent", {
            "to_addresses": ["someone@example.com"],
            "attachment_count": 3,
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_activity_based_completion()
        assert count == 0


# ===================================================================
# Strategy 2 — Inactivity-based completion
# ===================================================================

class TestInactivityBasedCompletion:
    """Tests for _detect_inactivity_based_completion."""

    @pytest.mark.asyncio
    async def test_task_with_no_activity_past_threshold_completed(self, detector, db, now):
        """A pending task older than inactivity_days with no related events
        should be marked complete."""
        task_id = _insert_task(
            db, "Follow up with client",
            created_at=now - timedelta(days=8),
        )

        count = await detector._detect_inactivity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_recent_task_not_completed(self, detector, db, now):
        """A task created 2 days ago should remain pending."""
        _insert_task(
            db, "Follow up with client",
            created_at=now - timedelta(days=2),
        )

        count = await detector._detect_inactivity_based_completion()
        assert count == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_boundary_task_within_threshold_not_completed(self, detector, db, now):
        """A task created well within the inactivity window should NOT be completed.

        The query uses ``created_at < cutoff``.  A task created 6 days ago
        (1 day inside the 7-day window) must remain pending.
        """
        _insert_task(
            db, "Still fresh task",
            created_at=now - timedelta(days=6),
        )

        count = await detector._detect_inactivity_based_completion()
        assert count == 0

    @pytest.mark.asyncio
    async def test_boundary_one_second_past_threshold(self, detector, db, now):
        """A task one second older than the threshold SHOULD be completed."""
        cutoff = now - timedelta(days=detector.inactivity_days)
        task_id = _insert_task(
            db, "Just past boundary",
            created_at=cutoff - timedelta(seconds=1),
        )

        count = await detector._detect_inactivity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_already_completed_tasks_ignored(self, detector, db, now):
        """Tasks with status != 'pending' are not re-processed."""
        _insert_task(
            db, "Old completed task",
            status="completed",
            created_at=now - timedelta(days=10),
        )

        count = await detector._detect_inactivity_based_completion()
        assert count == 0

    @pytest.mark.asyncio
    async def test_unrelated_system_activity_does_not_prevent_completion(self, detector, db, now):
        """Global system activity (other emails arriving) should NOT keep an
        old dormant task alive.  Only task age matters."""
        task_id = _insert_task(
            db, "Dormant task nobody touched",
            created_at=now - timedelta(days=9),
        )
        # Lots of unrelated recent activity
        for i in range(10):
            _insert_event(db, "email.received", {
                "from_address": f"news{i}@example.com",
                "body_plain": "Totally unrelated newsletter content.",
            }, timestamp=now - timedelta(hours=i + 1))

        count = await detector._detect_inactivity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_multiple_inactive_tasks(self, detector, db, now):
        """All qualifying inactive tasks should be completed in one pass."""
        ids = []
        for i in range(5):
            tid = _insert_task(
                db, f"Old task {i}",
                created_at=now - timedelta(days=8 + i),
            )
            ids.append(tid)

        count = await detector._detect_inactivity_based_completion()
        assert count == 5
        completed_ids = [c.args[0] for c in detector.task_manager.complete_task.call_args_list]
        for tid in ids:
            assert tid in completed_ids


# ===================================================================
# Strategy 3 — Stale task cleanup
# ===================================================================

class TestStaleTaskCleanup:
    """Tests for _detect_stale_tasks."""

    @pytest.mark.asyncio
    async def test_task_past_stale_threshold_archived(self, detector, db, now):
        """A pending task 31+ days old should be archived."""
        task_id = _insert_task(
            db, "Ancient task",
            created_at=now - timedelta(days=31),
        )

        count = await detector._detect_stale_tasks()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_task_under_stale_threshold_not_archived(self, detector, db, now):
        """A pending task 29 days old should NOT be archived by the stale strategy."""
        _insert_task(
            db, "Not yet stale",
            created_at=now - timedelta(days=29),
        )

        count = await detector._detect_stale_tasks()
        assert count == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_within_stale_threshold_not_archived(self, detector, db, now):
        """A task well within the stale threshold (25 days) should NOT be archived."""
        _insert_task(
            db, "Not stale yet",
            created_at=now - timedelta(days=25),
        )

        count = await detector._detect_stale_tasks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_stale_task_with_recent_activity_still_archived(self, detector, db, now):
        """The stale strategy uses age alone — recent activity does not prevent
        archival.  (This is the current design: tasks older than 30 days are
        unconditionally archived by this strategy.)"""
        task_id = _insert_task(
            db, "Old task with recent email",
            created_at=now - timedelta(days=35),
        )
        # Recent email that references the task — doesn't matter for staleness
        _insert_event(db, "email.sent", {
            "subject": "Old task update",
            "body_plain": "Still working on the old task with recent email discussion.",
        }, timestamp=now - timedelta(hours=1))

        count = await detector._detect_stale_tasks()
        assert count == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_already_completed_stale_task_ignored(self, detector, db, now):
        """Completed tasks older than stale_days should not be re-processed."""
        _insert_task(
            db, "Done and dusted",
            status="completed",
            created_at=now - timedelta(days=40),
        )

        count = await detector._detect_stale_tasks()
        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_stale_tasks(self, detector, db, now):
        """All stale tasks should be cleaned up in one pass."""
        ids = []
        for i in range(3):
            tid = _insert_task(
                db, f"Stale task {i}",
                created_at=now - timedelta(days=35 + i * 5),
            )
            ids.append(tid)

        count = await detector._detect_stale_tasks()
        assert count == 3
        completed_ids = [c.args[0] for c in detector.task_manager.complete_task.call_args_list]
        for tid in ids:
            assert tid in completed_ids


# ===================================================================
# Integration — detect_completions() orchestrates all strategies
# ===================================================================

class TestDetectCompletionsIntegration:
    """Tests for the top-level detect_completions() method."""

    @pytest.mark.asyncio
    async def test_all_strategies_fire_and_total_is_correct(self, detector, db, now):
        """Create tasks matching different strategies and verify the total."""
        # Strategy 1 — activity match (recent task + matching email)
        t1 = _insert_task(
            db, "Send quarterly report to manager",
            created_at=now - timedelta(hours=4),
        )
        _insert_event(db, "email.sent", {
            "subject": "Quarterly report",
            "body_plain": "The quarterly report for the manager is sent and completed.",
        }, timestamp=now - timedelta(hours=1))

        # Strategy 2 — inactivity (8-day-old task, no matching sent events)
        t2 = _insert_task(
            db, "Unique inactive task alpha",
            created_at=now - timedelta(days=8),
        )

        # Strategy 3 — stale (35-day-old task)
        t3 = _insert_task(
            db, "Very old abandoned task beta",
            created_at=now - timedelta(days=35),
        )

        total = await detector.detect_completions()

        # With the side-effecting mock, each task is completed exactly once:
        # - t1 by activity, t2 by inactivity, t3 by inactivity (hits before stale)
        # or t3 by stale — but since the mock updates status, the later strategy
        # won't re-find it.
        assert total >= 3
        completed_ids = [c.args[0] for c in detector.task_manager.complete_task.call_args_list]
        assert t1 in completed_ids
        assert t2 in completed_ids
        assert t3 in completed_ids

    @pytest.mark.asyncio
    async def test_no_tasks_returns_zero(self, detector, db):
        """Empty database yields zero completions."""
        total = await detector.detect_completions()
        assert total == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_pending_tasks_processed(self, detector, db, now):
        """Completed and cancelled tasks should be untouched."""
        _insert_task(db, "Already done", status="completed",
                     created_at=now - timedelta(days=10))
        _insert_task(db, "Cancelled task", status="cancelled",
                     created_at=now - timedelta(days=10))

        total = await detector.detect_completions()
        assert total == 0

    @pytest.mark.asyncio
    async def test_task_completed_events_published_via_task_manager(self, detector, db, now):
        """When a task is completed, task_manager.complete_task() is called,
        which is responsible for publishing task.completed events.  Verify
        the call happens for each detected completion."""
        _insert_task(
            db, "Stale ignored task",
            created_at=now - timedelta(days=35),
        )
        _insert_task(
            db, "Another stale ignored task",
            created_at=now - timedelta(days=40),
        )

        await detector.detect_completions()

        # complete_task should have been called for each task
        assert detector.task_manager.complete_task.call_count >= 2


# ===================================================================
# Episode outcome updates
# ===================================================================

class TestEpisodeOutcomeUpdates:
    """Tests for _update_episode_outcome integration."""

    def test_no_user_model_store_is_safe(self, detector, db):
        """Calling _update_episode_outcome with no user_model_store is a no-op."""
        # detector fixture has user_model_store=None
        detector._update_episode_outcome("some-source-id", "activity_match")
        # Should not raise

    def test_no_source_is_safe(self, detector_with_ums, db):
        """Calling with source=None is a no-op."""
        detector_with_ums._update_episode_outcome(None, "activity_match")

    def test_missing_episode_is_safe(self, detector_with_ums, db):
        """Calling with a source that has no episode is a no-op."""
        detector_with_ums._update_episode_outcome("nonexistent-event-id", "activity_match")

    def test_updates_episode_outcome_when_episode_exists(self, detector_with_ums, db):
        """When an episode exists for the task's source event, the outcome
        field should be updated to the completion method."""
        source_event_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO episodes
                   (id, event_id, interaction_type, content_summary, timestamp, created_at)
                   VALUES (?, ?, 'email', 'Test episode', ?, ?)""",
                (episode_id, source_event_id, now, now),
            )

        detector_with_ums._update_episode_outcome(source_event_id, "activity_match")

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT outcome FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()

        assert row is not None
        assert row["outcome"] == "activity_match"


# ===================================================================
# Text extraction edge cases
# ===================================================================

class TestTextExtraction:
    """Edge cases for _extract_text_content."""

    def test_all_fields_concatenated(self, detector):
        """All recognized text fields should be concatenated."""
        payload = {
            "subject": "Subject line",
            "body_plain": "Body content",
            "snippet": "Snippet text",
            "description": "Description field",
            "summary": "Summary field",
            "title": "Title field",
        }
        text = detector._extract_text_content(payload)
        for expected in ("Subject line", "Body content", "Snippet text",
                         "Description field", "Summary field", "Title field"):
            assert expected in text

    def test_empty_payload_returns_empty_string(self, detector):
        """An empty payload should return an empty string."""
        assert detector._extract_text_content({}) == ""

    def test_none_values_skipped(self, detector):
        """Fields with None values should not contribute to the output."""
        payload = {"subject": None, "body_plain": "Hello"}
        text = detector._extract_text_content(payload)
        assert "Hello" in text
        assert "None" not in text
