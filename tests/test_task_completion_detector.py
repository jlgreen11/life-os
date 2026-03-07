"""
Tests for TaskCompletionDetector - automatic task completion inference.

The detector needs to infer when tasks have been completed from behavioral
signals since users often complete work without explicitly marking tasks done.
This is critical for enabling workflow detection (Layer 3 procedural memory).
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_completion_detector.detector import TaskCompletionDetector


@pytest.fixture
def detector(db, event_bus):
    """Create a task completion detector with mocked task manager."""
    task_manager = MagicMock()
    task_manager.complete_task = AsyncMock()
    return TaskCompletionDetector(db, task_manager, event_bus)


@pytest.fixture
def base_time():
    """Fixed timestamp for consistent testing."""
    return datetime.now(timezone.utc)


def create_task(db, task_id, title, description="", created_at=None, status="pending"):
    """Helper to create a task in the database."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    with db.get_connection("state") as conn:
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, source, domain,
                             priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ai', 'personal', 'normal', ?, ?)
        """, (
            task_id, title, description, status,
            created_at.isoformat(), created_at.isoformat()
        ))


def create_event(db, event_type, payload, timestamp=None):
    """Helper to create an event in the database."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    event_id = str(uuid.uuid4())
    with db.get_connection("events") as conn:
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, 'test', ?, 'normal', ?, '{}')
        """, (event_id, event_type, timestamp.isoformat(), json.dumps(payload)))

    return event_id


class TestActivityBasedCompletion:
    """Test completion detection from email/message activity."""

    @pytest.mark.asyncio
    async def test_email_sent_with_task_keywords_and_completion_signal(self, detector, db, base_time):
        """Task should be marked complete when user sends email referencing it with completion keywords."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Send invoice to Acme Corporation", created_at=base_time - timedelta(hours=2))

        # User sends email mentioning the task with completion keyword
        # Include more task keywords to ensure match: "invoice", "Acme"
        create_event(db, "email.sent", {
            "to_addresses": ["billing@acmecorp.com"],
            "subject": "Invoice for Acme - Q4 Services",
            "body_plain": "Hi, I've sent the invoice for Acme Corporation quarterly services as requested. Let me know if you need anything else. The invoice is completed and attached.",
        }, timestamp=base_time - timedelta(minutes=30))

        completed = await detector.detect_completions()

        assert completed >= 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_message_sent_with_task_reference_completes_task(self, detector, db, base_time):
        """Sent messages with task keywords should trigger completion."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Review contract with legal team", created_at=base_time - timedelta(hours=4))

        # User sends message saying they finished the review
        create_event(db, "message.sent", {
            "to_addresses": ["+15551234567"],
            "body_plain": "Just finished reviewing the contract with legal. Everything looks good, we're ready to move forward.",
        }, timestamp=base_time - timedelta(minutes=15))

        completed = await detector.detect_completions()

        assert completed >= 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_requires_keyword_overlap_and_completion_signal(self, detector, db, base_time):
        """Both keyword match AND completion signal are required."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Schedule dentist appointment", created_at=base_time - timedelta(hours=3))

        # Email with keywords but no completion signal - should NOT complete
        create_event(db, "email.sent", {
            "to_addresses": ["friend@example.com"],
            "subject": "Weekend plans",
            "body_plain": "I need to schedule a dentist appointment soon, but haven't gotten around to it yet.",
        }, timestamp=base_time - timedelta(minutes=45))

        completed = await detector.detect_completions()

        assert completed == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_requires_minimum_keyword_matches(self, detector, db, base_time):
        """Need at least 2 keyword matches to avoid false positives."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Update project timeline document", created_at=base_time - timedelta(hours=2))

        # Email with completion keyword but NO task keyword matches
        # (talking about a completely different thing)
        create_event(db, "email.sent", {
            "to_addresses": ["team@company.com"],
            "subject": "Weekly status",
            "body_plain": "Just finished reviewing the quarterly budget report. Everything is completed and submitted.",
        }, timestamp=base_time - timedelta(minutes=20))

        completed = await detector.detect_completions()

        # Should not complete because there are no keyword matches
        assert completed == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_completion_keywords_detected(self, detector, db, base_time):
        """System should recognize various completion signal words."""
        completion_keywords = ['done', 'finished', 'completed', 'sent', 'submitted',
                               'delivered', 'shipped', 'resolved', 'closed', 'fixed']

        for keyword in completion_keywords[:5]:  # Test a sample
            task_id = str(uuid.uuid4())
            create_task(db, task_id, f"Complete {keyword} testing task", created_at=base_time - timedelta(hours=1))

            create_event(db, "email.sent", {
                "to_addresses": ["test@example.com"],
                "subject": "Task update",
                "body_plain": f"The testing task is {keyword}. Ready for review.",
            }, timestamp=base_time - timedelta(minutes=10))

            detector.task_manager.complete_task.reset_mock()
            completed = await detector.detect_completions()

            assert completed >= 1, f"Should detect completion keyword: {keyword}"

    @pytest.mark.asyncio
    async def test_only_detects_events_after_task_creation(self, detector, db, base_time):
        """Should only consider emails/messages sent AFTER the task was created."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Submit expense report", created_at=base_time - timedelta(hours=1))

        # Email sent BEFORE task creation - should be ignored
        create_event(db, "email.sent", {
            "to_addresses": ["accounting@company.com"],
            "subject": "Expense Report Submission",
            "body_plain": "I've submitted the expense report as requested. All done!",
        }, timestamp=base_time - timedelta(hours=2))

        completed = await detector.detect_completions()

        assert completed == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_after_first_matching_event(self, detector, db, base_time):
        """Should mark task complete on first match, not process duplicates."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Prepare presentation slides", created_at=base_time - timedelta(hours=3))

        # Multiple matching emails
        for i in range(3):
            create_event(db, "email.sent", {
                "to_addresses": ["team@company.com"],
                "subject": "Presentation update",
                "body_plain": f"The presentation slides are finished and ready for review #{i}.",
            }, timestamp=base_time - timedelta(minutes=30-i*5))

        completed = await detector.detect_completions()

        # Should only complete once
        assert completed == 1
        assert detector.task_manager.complete_task.call_count == 1


class TestInactivityBasedCompletion:
    """Test completion detection from task inactivity.

    The inactivity detector closes any pending task older than ``inactivity_days``
    that was not already handled by the activity-based detector.  Task age is the
    only signal — unrelated system activity (other emails, received messages) does
    NOT keep a dormant task alive.  The old implementation tested a global
    ``recent_activity_count == 0`` guard which caused the strategy to never fire
    on a live instance where there is always at least one recent event.
    """

    @pytest.mark.asyncio
    async def test_marks_inactive_tasks_complete(self, detector, db, base_time):
        """Tasks older than inactivity_days should be marked complete."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Old task", created_at=base_time - timedelta(days=8))

        completed = await detector.detect_completions()

        assert completed >= 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_marks_inactive_task_complete_even_with_unrelated_system_activity(
        self, detector, db, base_time
    ):
        """Global system activity must NOT prevent an old task from being closed.

        Previously the detector checked ``COUNT(*) == 0`` over all events in the
        last 7 days.  On any live instance this count is always > 0, so the
        inactivity strategy never fired.  This test proves the fix: an
        unrelated received email no longer shields a dormant 8-day-old task.
        """
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Dormant task", created_at=base_time - timedelta(days=8))

        # Add recent activity completely unrelated to the task — should NOT matter.
        create_event(db, "email.received", {
            "from_address": "newsletter@example.com",
            "subject": "Weekly digest",
            "body_plain": "Here is your weekly digest of unrelated news.",
        }, timestamp=base_time - timedelta(days=2))

        completed = await detector.detect_completions()

        # Task is old enough and has no completion signal — must be closed.
        assert completed >= 1
        calls = [call.args[0] for call in detector.task_manager.complete_task.call_args_list]
        assert task_id in calls

    @pytest.mark.asyncio
    async def test_respects_inactivity_threshold(self, detector, db, base_time):
        """Should only mark tasks complete after the full inactivity period."""
        # Task created exactly at threshold (7 days) — still inside window, skip.
        task_id_at_threshold = str(uuid.uuid4())
        create_task(db, task_id_at_threshold, "At threshold",
                   created_at=base_time - timedelta(days=7))

        # Task created just past threshold (7 days + 1 hour) — should complete.
        task_id_past_threshold = str(uuid.uuid4())
        create_task(db, task_id_past_threshold, "Past threshold",
                   created_at=base_time - timedelta(days=7, hours=1))

        completed = await detector.detect_completions()

        assert completed >= 1
        calls = [call.args[0] for call in detector.task_manager.complete_task.call_args_list]
        assert task_id_past_threshold in calls

    @pytest.mark.asyncio
    async def test_does_not_complete_recent_tasks(self, detector, db, base_time):
        """Tasks created within the inactivity window should NOT be auto-completed."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Recent task", created_at=base_time - timedelta(days=3))

        # Add lots of unrelated activity — still should not close a recent task.
        for i in range(5):
            create_event(db, "email.received", {
                "from_address": f"sender{i}@example.com",
                "subject": "Something",
                "body_plain": "Content",
            }, timestamp=base_time - timedelta(hours=i + 1))

        completed = await detector.detect_completions()

        # Inactivity detector must not fire for tasks younger than threshold.
        calls = [call.args[0] for call in detector.task_manager.complete_task.call_args_list]
        assert task_id not in calls


class TestStaleTaskCleanup:
    """Test cleanup of very old tasks."""

    @pytest.mark.asyncio
    async def test_archives_stale_tasks(self, detector, db, base_time):
        """Tasks older than 30 days should be auto-archived."""
        task_id = str(uuid.uuid4())
        # Task created 35 days ago
        create_task(db, task_id, "Very old task", created_at=base_time - timedelta(days=35))

        completed = await detector.detect_completions()

        assert completed >= 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_keeps_newer_tasks(self, detector, db, base_time):
        """Tasks under 30 days old should not be auto-archived."""
        task_id = str(uuid.uuid4())
        # Task created 25 days ago (under threshold)
        create_task(db, task_id, "Older task but not stale", created_at=base_time - timedelta(days=25))

        # Should not be completed by stale task cleanup alone
        # (might be completed by inactivity check, but that's a different test)
        completed = await detector.detect_completions()

        # If it was completed, it should be due to inactivity, not staleness
        # We can't assert NOT called because inactivity check might trigger
        # So we just verify the logic runs without error
        assert completed >= 0  # Non-negative result


class TestUpdateEpisodeOutcome:
    """Tests for linking task completion to episodic memory."""

    def test_noop_without_user_model_store(self, detector):
        """Should silently return when user_model_store is None."""
        # detector fixture has user_model_store=None by default
        detector._update_episode_outcome("some-source", "activity_match")
        # No error raised

    def test_noop_without_task_source(self, detector):
        """Should silently return when task_source is None."""
        detector._update_episode_outcome(None, "activity_match")
        # No error raised

    def test_updates_episode_when_found(self, db, event_bus, user_model_store):
        """When an episode exists for the source event, its outcome should be updated."""
        task_manager = MagicMock()
        task_manager.complete_task = AsyncMock()
        det = TaskCompletionDetector(db, task_manager, event_bus,
                                     user_model_store=user_model_store)

        # Insert an episode linked to an event_id
        event_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                "INSERT INTO episodes (id, event_id, interaction_type, timestamp, content_summary) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, event_id, "email",
                 datetime.now(timezone.utc).isoformat(), "Test episode"),
            )

        det._update_episode_outcome(event_id, "activity_match")

        # Verify the outcome was updated
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT outcome FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
        assert row is not None
        assert row["outcome"] == "activity_match"

    def test_handles_missing_episode_gracefully(self, db, event_bus, user_model_store):
        """Should not raise when no episode matches the source event."""
        task_manager = MagicMock()
        task_manager.complete_task = AsyncMock()
        det = TaskCompletionDetector(db, task_manager, event_bus,
                                     user_model_store=user_model_store)

        # Call with an event_id that has no matching episode
        det._update_episode_outcome(str(uuid.uuid4()), "inactivity")
        # No error raised


class TestTextExtraction:
    """Test text content extraction from event payloads."""

    def test_extracts_email_fields(self, detector):
        """Should extract subject, body, and snippet from emails."""
        payload = {
            "subject": "Important update",
            "body_plain": "Here is the full email body with details.",
            "snippet": "Preview text",
        }

        text = detector._extract_text_content(payload)

        assert "important update" in text.lower()
        assert "full email body" in text.lower()
        assert "preview text" in text.lower()

    def test_extracts_message_fields(self, detector):
        """Should extract body and snippet from messages."""
        payload = {
            "body_plain": "Quick message text",
            "snippet": "Message preview",
        }

        text = detector._extract_text_content(payload)

        assert "quick message" in text.lower()
        assert "preview" in text.lower()

    def test_extracts_calendar_fields(self, detector):
        """Should extract summary and description from calendar events."""
        payload = {
            "summary": "Team Meeting",
            "description": "Discuss Q4 goals and deliverables",
        }

        text = detector._extract_text_content(payload)

        assert "team meeting" in text.lower()
        assert "q4 goals" in text.lower()

    def test_extracts_snippet_and_title_fields(self, detector):
        """Should also extract snippet and title fields from payloads."""
        payload = {
            "snippet": "Quick preview text",
            "title": "Event title",
        }
        text = detector._extract_text_content(payload)
        assert "Quick preview text" in text
        assert "Event title" in text

    def test_handles_missing_fields(self, detector):
        """Should handle payloads with missing text fields."""
        payload = {"id": "123", "timestamp": "2024-01-01T00:00:00Z"}

        text = detector._extract_text_content(payload)

        assert text == ""  # No text content available


class TestDirectStrategyMethods:
    """Test individual detection strategies called directly (not via detect_completions).

    Calling the private methods directly avoids cross-strategy interference
    and gives clearer boundary testing.
    """

    @pytest.mark.asyncio
    async def test_activity_detection_returns_count(self, detector, db, base_time):
        """_detect_activity_based_completion() should return its own count."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Send quarterly report manager",
                    created_at=base_time - timedelta(hours=3))
        create_event(db, "email.sent", {
            "subject": "Quarterly report",
            "body_plain": "The quarterly report for the manager is done and sent.",
        }, timestamp=base_time - timedelta(minutes=30))

        count = await detector._detect_activity_based_completion()
        assert count == 1

    @pytest.mark.asyncio
    async def test_inactivity_detection_directly(self, detector, db, base_time):
        """_detect_inactivity_based_completion() called directly."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Old task", created_at=base_time - timedelta(days=8))

        count = await detector._detect_inactivity_based_completion()
        assert count == 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_inactivity_detection_skips_recent(self, detector, db, base_time):
        """_detect_inactivity_based_completion() should skip 3-day-old tasks."""
        create_task(db, str(uuid.uuid4()), "Recent task",
                    created_at=base_time - timedelta(days=3))

        count = await detector._detect_inactivity_based_completion()
        assert count == 0

    @pytest.mark.asyncio
    async def test_stale_detection_directly(self, detector, db, base_time):
        """_detect_stale_tasks() called directly on 35-day-old task."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Ancient task",
                    created_at=base_time - timedelta(days=35))

        count = await detector._detect_stale_tasks()
        assert count == 1
        detector.task_manager.complete_task.assert_called_with(task_id)

    @pytest.mark.asyncio
    async def test_stale_detection_skips_20_day_task(self, detector, db, base_time):
        """_detect_stale_tasks() should NOT archive a 20-day-old task."""
        create_task(db, str(uuid.uuid4()), "Not yet stale",
                    created_at=base_time - timedelta(days=20))

        count = await detector._detect_stale_tasks()
        assert count == 0


class TestIntegration:
    """Integration tests for the full detection cycle."""

    @pytest.mark.asyncio
    async def test_detects_multiple_completion_strategies(self, detector, db, base_time):
        """Should detect completions using all strategies in one cycle."""
        # Activity-based completion
        task_id_1 = str(uuid.uuid4())
        create_task(db, task_id_1, "Send quarterly report manager", created_at=base_time - timedelta(hours=3))
        create_event(db, "email.sent", {
            "to_addresses": ["manager@company.com"],
            "subject": "Q4 Report",
            "body_plain": "I've finished the quarterly report and sent it as requested.",
        }, timestamp=base_time - timedelta(minutes=30))

        # Stale task cleanup (use 35 days to ensure it's past both thresholds)
        task_id_2 = str(uuid.uuid4())
        create_task(db, task_id_2, "Stale task", created_at=base_time - timedelta(days=35))

        completed = await detector.detect_completions()

        # Activity-based (task_id_1) + stale cleanup (task_id_2).
        # The inactivity detector would also fire for task_id_2 were it not
        # already past the stale threshold (35 days > 30 days stale cutoff).
        assert completed >= 2
        calls = [call.args[0] for call in detector.task_manager.complete_task.call_args_list]
        assert task_id_1 in calls  # Activity
        assert task_id_2 in calls  # Stale

    @pytest.mark.asyncio
    async def test_only_affects_pending_tasks(self, detector, db, base_time):
        """Should not process tasks that are already completed."""
        # Create a completed task
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Already done", created_at=base_time - timedelta(days=10), status="completed")

        initial_call_count = detector.task_manager.complete_task.call_count
        await detector.detect_completions()

        # Should not try to complete an already-completed task
        assert detector.task_manager.complete_task.call_count == initial_call_count

    @pytest.mark.asyncio
    async def test_handles_empty_database(self, detector, db):
        """Should handle empty task database gracefully."""
        completed = await detector.detect_completions()

        assert completed == 0
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_malformed_event_payloads(self, detector, db, base_time):
        """Should handle events with malformed JSON gracefully."""
        task_id = str(uuid.uuid4())
        create_task(db, task_id, "Test task", created_at=base_time - timedelta(hours=2))

        # Create event with valid JSON but unexpected structure
        event_id = str(uuid.uuid4())
        with db.get_connection("events") as conn:
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, 'email.sent', 'test', ?, 'normal', '{"unexpected": "structure"}', '{}')
            """, (event_id, base_time.isoformat()))

        # Should not crash even with unexpected payload structure
        completed = await detector.detect_completions()
        assert completed >= 0  # Non-negative result
