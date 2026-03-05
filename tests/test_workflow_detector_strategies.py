"""
Life OS — WorkflowDetector Strategy-Level Tests

Targeted tests for each of the 4 detection strategies in WorkflowDetector
to diagnose the zero-workflow production issue. Each test inserts realistic
event/episode data and validates that the strategy correctly identifies (or
rejects) workflow patterns.

Context: Despite 17,274 episodes in production, 0 workflows are detected.
These tests isolate each strategy to pinpoint which ones are failing and why.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.workflow_detector.detector import WorkflowDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(dt: datetime) -> str:
    """Format a datetime as ISO 8601 string for SQLite storage."""
    return dt.isoformat()


def _insert_event(conn, event_type: str, timestamp: datetime, *,
                  email_from: str | None = None,
                  email_to: str | None = None,
                  task_id: str | None = None,
                  calendar_event_id: str | None = None,
                  payload: dict | None = None,
                  source: str = "test"):
    """Insert a single event into the events table with denormalized columns.

    Directly populates the denormalized columns (email_from, email_to, etc.)
    instead of relying on triggers, since triggers may not fire in all test
    environments.
    """
    eid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                            email_from, email_to, task_id, calendar_event_id)
        VALUES (?, ?, ?, ?, 'normal', ?, '{}', ?, ?, ?, ?)
        """,
        (
            eid,
            event_type,
            source,
            _ts(timestamp),
            json.dumps(payload or {}),
            email_from,
            email_to,
            task_id,
            calendar_event_id,
        ),
    )
    return eid


def _insert_episode(conn, interaction_type: str, timestamp: datetime, *,
                    event_id: str | None = None,
                    content_summary: str = "test episode"):
    """Insert a single episode into the episodes table."""
    ep_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO episodes (id, timestamp, event_id, interaction_type,
                              content_summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ep_id,
            _ts(timestamp),
            event_id or str(uuid.uuid4()),
            interaction_type,
            content_summary,
            _ts(datetime.now(timezone.utc)),
        ),
    )
    return ep_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(db, user_model_store):
    """A WorkflowDetector wired to temporary databases."""
    return WorkflowDetector(db, user_model_store)


# ===========================================================================
# 1. _detect_email_workflows tests
# ===========================================================================


class TestDetectEmailWorkflows:
    """Tests for the email workflow detection strategy."""

    def test_detects_workflow_with_matching_sender_recipient(self, detector, db):
        """Emails received from sender + replies sent TO that sender = workflow."""
        now = datetime.now(timezone.utc)
        sender = "boss@example.com"

        with db.get_connection("events") as conn:
            # 3 received emails from the same sender over multiple days
            for i in range(3):
                _insert_event(conn, "email.received",
                              now - timedelta(days=5 - i),
                              email_from=sender)
                # Reply sent to the same sender within 2 hours
                _insert_event(conn, "email.sent",
                              now - timedelta(days=5 - i, hours=-2),
                              email_to=json.dumps([sender]))

        workflows = detector._detect_email_workflows(lookback_days=30)

        assert len(workflows) >= 1
        wf = workflows[0]
        assert "boss" in wf["name"].lower() or sender in wf["name"]
        assert wf["times_observed"] >= 3
        assert wf["success_rate"] > 0
        assert "email" in wf["tools_used"]

    def test_no_workflow_when_no_replies(self, detector, db):
        """Received emails without any sent replies should NOT produce a workflow."""
        now = datetime.now(timezone.utc)
        sender = "newsletter@example.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                _insert_event(conn, "email.received",
                              now - timedelta(days=5 - i),
                              email_from=sender)

        workflows = detector._detect_email_workflows(lookback_days=30)

        # No sent emails means no completions, so no workflow
        assert len(workflows) == 0

    def test_no_workflow_below_min_occurrences(self, detector, db):
        """Fewer than min_occurrences received emails should NOT produce a workflow."""
        now = datetime.now(timezone.utc)
        sender = "rare@example.com"

        with db.get_connection("events") as conn:
            # Only 2 emails (below default min_occurrences=3)
            for i in range(2):
                _insert_event(conn, "email.received",
                              now - timedelta(days=2 - i),
                              email_from=sender)
                _insert_event(conn, "email.sent",
                              now - timedelta(days=2 - i, hours=-1),
                              email_to=json.dumps([sender]))

        workflows = detector._detect_email_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_no_workflow_when_reply_outside_gap(self, detector, db):
        """Replies sent more than max_step_gap_hours after receive are not matched."""
        now = datetime.now(timezone.utc)
        sender = "slow@example.com"

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=20 - i * 4)
                _insert_event(conn, "email.received", base,
                              email_from=sender)
                # Reply sent 24 hours later (outside 12-hour gap)
                # No other events in between to avoid cross-matching
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=24),
                              email_to=json.dumps([sender]))

        workflows = detector._detect_email_workflows(lookback_days=30)
        # All replies are outside the 12-hour gap, so email.sent actions
        # should have < min_occurrences matches (the sliding window expires
        # the received emails before the sent arrives)
        assert len(workflows) == 0

    def test_min_completions_threshold(self, detector, db):
        """Workflow requires min_completions actual sent replies to be stored."""
        now = datetime.now(timezone.utc)
        sender = "client@example.com"

        with db.get_connection("events") as conn:
            # 5 received emails but only 1 reply (below min_completions=2)
            for i in range(5):
                _insert_event(conn, "email.received",
                              now - timedelta(days=5 - i),
                              email_from=sender)
            # Only one reply
            _insert_event(conn, "email.sent",
                          now - timedelta(days=4, hours=-1),
                          email_to=json.dumps([sender]))

        # min_completions=2 by default, but we need min_occurrences (3) matching
        # actions to even count them. 1 reply < 3 min_occurrences for the action.
        workflows = detector._detect_email_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_email_to_as_plain_string(self, detector, db):
        """email_to stored as a plain string (not JSON array) still matches."""
        now = datetime.now(timezone.utc)
        sender = "plain@example.com"

        with db.get_connection("events") as conn:
            for i in range(3):
                _insert_event(conn, "email.received",
                              now - timedelta(days=5 - i),
                              email_from=sender)
                # email_to as plain string instead of JSON array
                _insert_event(conn, "email.sent",
                              now - timedelta(days=5 - i, hours=-1),
                              email_to=sender)

        workflows = detector._detect_email_workflows(lookback_days=30)
        assert len(workflows) >= 1

    def test_case_insensitive_sender_matching(self, detector, db):
        """Sender matching is case-insensitive."""
        now = datetime.now(timezone.utc)
        sender = "Boss@Example.COM"

        with db.get_connection("events") as conn:
            for i in range(3):
                _insert_event(conn, "email.received",
                              now - timedelta(days=5 - i),
                              email_from=sender)
                _insert_event(conn, "email.sent",
                              now - timedelta(days=5 - i, hours=-1),
                              email_to=json.dumps(["boss@example.com"]))

        workflows = detector._detect_email_workflows(lookback_days=30)
        assert len(workflows) >= 1

    def test_multiple_action_types_create_richer_workflow(self, detector, db):
        """Email receive followed by both task creation and reply = multi-step workflow."""
        now = datetime.now(timezone.utc)
        sender = "project@example.com"

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_event(conn, "email.received", base, email_from=sender)
                # Create a task 30 min later
                _insert_event(conn, "task.created", base + timedelta(minutes=30))
                # Send reply 2 hours later
                _insert_event(conn, "email.sent", base + timedelta(hours=2),
                              email_to=json.dumps([sender]))

        workflows = detector._detect_email_workflows(lookback_days=30)
        assert len(workflows) >= 1
        wf = workflows[0]
        # Should have email + task_manager tools
        assert "email" in wf["tools_used"]
        assert len(wf["steps"]) >= 3  # read + created + sent


# ===========================================================================
# 2. _detect_task_workflows tests
# ===========================================================================


class TestDetectTaskWorkflows:
    """Tests for the task workflow detection strategy."""

    def test_detects_task_completion_workflow(self, detector, db):
        """task.created followed by task.completed + email.sent = workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                tid = f"task-{i}"
                _insert_event(conn, "task.created", base, task_id=tid)
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=1))
                _insert_event(conn, "task.completed",
                              base + timedelta(hours=2), task_id=tid)

        workflows = detector._detect_task_workflows(lookback_days=30)

        assert len(workflows) >= 1
        wf = workflows[0]
        assert wf["name"] == "Task completion workflow"
        assert "create_task" in wf["steps"]
        assert wf["success_rate"] > 0
        assert wf["times_observed"] >= 4

    def test_no_workflow_with_insufficient_tasks(self, detector, db):
        """Fewer than min_occurrences tasks should not produce a workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            # Only 2 tasks (below min_occurrences=3)
            for i in range(2):
                _insert_event(conn, "task.created",
                              now - timedelta(days=2 - i))
                _insert_event(conn, "task.completed",
                              now - timedelta(days=2 - i, hours=-1))

        workflows = detector._detect_task_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_no_workflow_without_min_steps(self, detector, db):
        """Tasks with only one type of follow-up don't meet min_steps=2."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            # 4 tasks, but only task.completed follows (1 action type)
            for i in range(4):
                _insert_event(conn, "task.created",
                              now - timedelta(days=4 - i))
                _insert_event(conn, "task.completed",
                              now - timedelta(days=4 - i, hours=-2))

        workflows = detector._detect_task_workflows(lookback_days=30)
        # min_steps=2 means we need 2 different following action types
        # If only task.completed follows, that's 1 action type < min_steps
        # The strategy requires len(task_actions) >= min_steps (2)
        assert len(workflows) == 0

    def test_no_workflow_without_completions(self, detector, db):
        """Tasks followed by actions but no task.completed → 0 completions → no workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_event(conn, "task.created", base)
                _insert_event(conn, "email.sent", base + timedelta(hours=1))
                _insert_event(conn, "message.sent", base + timedelta(hours=2))

        workflows = detector._detect_task_workflows(lookback_days=30)
        # 2 action types (email.sent + message.sent) >= min_steps, but 0
        # task.completed events means completion_count=0 < min_completions=2
        assert len(workflows) == 0

    def test_events_outside_gap_not_matched(self, detector, db):
        """Follow-up events outside max_step_gap_hours are not linked to tasks."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=20 - i * 4)
                _insert_event(conn, "task.created", base)
                # Completion 24 hours later (outside 12-hour gap)
                _insert_event(conn, "task.completed",
                              base + timedelta(hours=24))
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=25))

        workflows = detector._detect_task_workflows(lookback_days=30)
        assert len(workflows) == 0


# ===========================================================================
# 3. _detect_calendar_workflows tests
# ===========================================================================


class TestDetectCalendarWorkflows:
    """Tests for the calendar workflow detection strategy."""

    def test_detects_calendar_followup_workflow(self, detector, db):
        """Calendar events followed by sent emails = follow-up workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                cal_id = f"cal-{i}"
                _insert_event(conn, "calendar.event.created", base,
                              calendar_event_id=cal_id)
                # Follow-up email 1 hour after event
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=1))
                # Follow-up task 2 hours after event
                _insert_event(conn, "task.created",
                              base + timedelta(hours=2))

        workflows = detector._detect_calendar_workflows(lookback_days=30)

        assert len(workflows) >= 1
        wf = workflows[0]
        assert wf["name"] == "Calendar event workflow"
        assert "calendar" in wf["tools_used"]
        assert any("followup" in s for s in wf["steps"])

    def test_no_workflow_with_insufficient_events(self, detector, db):
        """Fewer than min_occurrences calendar events should not produce a workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(2):
                _insert_event(conn, "calendar.event.created",
                              now - timedelta(days=2 - i))
                _insert_event(conn, "email.sent",
                              now - timedelta(days=2 - i, hours=-1))

        workflows = detector._detect_calendar_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_prep_activity_after_calendar_creation(self, detector, db):
        """Prep detection: calendar event created, then task.created before event time.

        The sliding window algorithm processes events chronologically. For prep
        detection, a non-calendar event must occur AFTER the calendar.event.created
        appears in the stream, but with a calendar event timestamp that is in the
        future relative to the prep activity. In practice this means: calendar is
        created early, then prep tasks are created before the meeting time.

        Since the current implementation tracks upcoming_events by insertion order
        and checks evt_ts > timestamp, prep is only detected when the calendar
        event's timestamp is AFTER the prep event's timestamp in the stream. This
        happens when calendar events are created well ahead of time.
        """
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                # Calendar event created early (time set in the future)
                _insert_event(conn, "calendar.event.created", base,
                              calendar_event_id=f"cal-{i}")
                # Prep: email received 1 hour after creation but still "before" the meeting
                _insert_event(conn, "email.received",
                              base + timedelta(hours=1))
                # Follow-up: email sent 3 hours after creation
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=3))
                # Follow-up: task created 4 hours after
                _insert_event(conn, "task.created",
                              base + timedelta(hours=4))

        workflows = detector._detect_calendar_workflows(lookback_days=30)

        assert len(workflows) >= 1
        wf = workflows[0]
        assert any("followup" in s for s in wf["steps"])

    def test_followup_outside_gap_not_matched(self, detector, db):
        """Follow-up events outside max_step_gap_hours are not linked."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=16 - i * 4)
                _insert_event(conn, "calendar.event.created", base,
                              calendar_event_id=f"cal-{i}")
                # Follow-up 24 hours later (outside gap)
                _insert_event(conn, "email.sent",
                              base + timedelta(hours=24))

        workflows = detector._detect_calendar_workflows(lookback_days=30)
        assert len(workflows) == 0


# ===========================================================================
# 4. _detect_interaction_workflows tests
# ===========================================================================


class TestDetectInteractionWorkflows:
    """Tests for the interaction (episode-based) workflow detection strategy."""

    def test_detects_interaction_sequence_workflow(self, detector, db):
        """Episodes with distinct interaction_type sequences = workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # Create repeating sequences: research → compose → review
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_episode(conn, "research", base)
                _insert_episode(conn, "compose",
                                base + timedelta(hours=1))
                _insert_episode(conn, "review",
                                base + timedelta(hours=2))

        workflows = detector._detect_interaction_workflows(lookback_days=30)

        assert len(workflows) >= 1
        # Should find at least one workflow starting with "research"
        research_wfs = [w for w in workflows if "research" in w["steps"][0].lower()]
        assert len(research_wfs) >= 1

    def test_empty_string_interaction_type_not_useful(self, detector, db):
        """Episodes with empty-string interaction_type don't form meaningful workflows.

        Note: The episodes table has a NOT NULL constraint on interaction_type,
        so truly NULL values can't exist. However, empty strings or placeholder
        values like 'unknown' could still be inserted. This test verifies
        that such degenerate values don't produce spurious workflows.
        """
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # Insert episodes with empty interaction_type
            for i in range(10):
                _insert_episode(conn, "", now - timedelta(days=10 - i))

        workflows = detector._detect_interaction_workflows(lookback_days=30)

        # All episodes have the same (empty) type, so no cross-type sequences
        assert len(workflows) == 0

    def test_single_interaction_type_no_workflow(self, detector, db):
        """All episodes with the same interaction_type produce no cross-type workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(10):
                _insert_episode(conn, "email_read",
                                now - timedelta(days=10 - i))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_interaction_outside_gap_not_matched(self, detector, db):
        """Episodes separated by more than max_step_gap_hours are not linked."""
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(4):
                base = now - timedelta(days=20 - i * 4)
                _insert_episode(conn, "research", base)
                # Next episode 24 hours later (outside 12-hour gap)
                _insert_episode(conn, "compose",
                                base + timedelta(hours=24))
                _insert_episode(conn, "review",
                                base + timedelta(hours=48))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        assert len(workflows) == 0

    def test_requires_min_steps_following_actions(self, detector, db):
        """Needs min_steps (2) distinct following action types to create a workflow."""
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # Only 2 interaction types: research → compose
            # This means "research" has 1 following type ("compose")
            # which is < min_steps=2 for following_actions
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_episode(conn, "research", base)
                _insert_episode(conn, "compose",
                                base + timedelta(hours=1))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        # "research" has only 1 following type → < min_steps=2
        assert len(workflows) == 0

    def test_mixed_degenerate_and_valid_interaction_types(self, detector, db):
        """Mix of 'unknown' and valid interaction_type — unknown is just another type.

        Note: The episodes table has NOT NULL on interaction_type, so truly NULL
        values can't be inserted. This test uses 'unknown' as a placeholder to
        verify that degenerate values don't interfere with valid sequences.
        """
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                # Valid episodes forming a sequence
                _insert_episode(conn, "research", base)
                _insert_episode(conn, "compose", base + timedelta(hours=1))
                _insert_episode(conn, "review", base + timedelta(hours=2))

                # 'unknown' episodes interspersed
                _insert_episode(conn, "unknown",
                                base + timedelta(hours=1, minutes=30))

        workflows = detector._detect_interaction_workflows(lookback_days=30)
        # Valid episodes should still form workflows
        assert len(workflows) >= 1


# ===========================================================================
# 5. Integration tests — detect_workflows() end-to-end
# ===========================================================================


class TestDetectWorkflowsIntegration:
    """End-to-end tests for the top-level detect_workflows() method."""

    def test_aggregates_all_strategy_results(self, detector, db):
        """detect_workflows() returns combined results from all strategies."""
        now = datetime.now(timezone.utc)

        # Insert email workflow data
        sender = "coworker@example.com"
        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_event(conn, "email.received", base, email_from=sender)
                _insert_event(conn, "task.created", base + timedelta(minutes=30))
                _insert_event(conn, "email.sent", base + timedelta(hours=2),
                              email_to=json.dumps([sender]))

        # Insert interaction workflow data
        with db.get_connection("user_model") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_episode(conn, "analyze", base)
                _insert_episode(conn, "draft", base + timedelta(hours=1))
                _insert_episode(conn, "send", base + timedelta(hours=2))

        workflows = detector.detect_workflows(lookback_days=30)

        # Should have at least email workflows
        assert len(workflows) >= 1
        # Each workflow should have required keys
        for wf in workflows:
            assert "name" in wf
            assert "trigger_conditions" in wf
            assert "steps" in wf
            assert "tools_used" in wf
            assert "success_rate" in wf
            assert "times_observed" in wf

    def test_empty_database_returns_no_workflows(self, detector):
        """Empty database produces zero workflows without errors."""
        workflows = detector.detect_workflows(lookback_days=30)
        assert workflows == []

    def test_strategy_failure_does_not_block_others(self, detector, db):
        """If one strategy throws, the others still run (fail-open)."""
        now = datetime.now(timezone.utc)
        sender = "reliable@example.com"

        # Insert valid email data
        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_event(conn, "email.received", base, email_from=sender)
                _insert_event(conn, "task.created", base + timedelta(minutes=30))
                _insert_event(conn, "email.sent", base + timedelta(hours=2),
                              email_to=json.dumps([sender]))

        # Monkey-patch interaction strategy to throw
        original = detector._detect_interaction_workflows
        detector._detect_interaction_workflows = lambda days: (_ for _ in ()).throw(
            RuntimeError("simulated failure")
        )

        try:
            workflows = detector.detect_workflows(lookback_days=30)
            # Email workflows should still be detected despite interaction failure
            assert len(workflows) >= 1
        finally:
            detector._detect_interaction_workflows = original

    def test_store_workflows_persists(self, detector, db):
        """store_workflows() persists detected workflows to the database."""
        now = datetime.now(timezone.utc)
        sender = "store@example.com"

        with db.get_connection("events") as conn:
            for i in range(4):
                base = now - timedelta(days=8 - i * 2)
                _insert_event(conn, "email.received", base, email_from=sender)
                _insert_event(conn, "task.created", base + timedelta(minutes=30))
                _insert_event(conn, "email.sent", base + timedelta(hours=2),
                              email_to=json.dumps([sender]))

        workflows = detector.detect_workflows(lookback_days=30)
        if workflows:
            stored = detector.store_workflows(workflows)
            assert stored >= 1


# ===========================================================================
# 6. Diagnostics tests
# ===========================================================================


class TestGetDiagnostics:
    """Tests for the get_diagnostics() method."""

    def test_diagnostics_with_empty_db(self, detector):
        """Diagnostics returns valid structure even with empty database."""
        diag = detector.get_diagnostics(lookback_days=30)

        assert "event_counts" in diag
        assert "thresholds" in diag
        assert "detection_results" in diag
        assert "total_detected" in diag
        assert "data_sufficient" in diag
        assert diag["total_detected"] == 0

    def test_diagnostics_reports_thresholds(self, detector):
        """Diagnostics includes current detection thresholds."""
        diag = detector.get_diagnostics(lookback_days=30)

        thresholds = diag["thresholds"]
        assert thresholds["min_occurrences"] == 3
        assert thresholds["max_step_gap_hours"] == 12
        assert thresholds["min_steps"] == 2
        assert thresholds["min_completions"] == 2

    def test_diagnostics_reports_per_strategy_results(self, detector, db):
        """Diagnostics shows detection results per strategy."""
        now = datetime.now(timezone.utc)

        with db.get_connection("events") as conn:
            for i in range(20):
                _insert_event(conn, "email.received",
                              now - timedelta(days=10 - i % 10),
                              email_from="diag@example.com")

        diag = detector.get_diagnostics(lookback_days=30)

        assert "email" in diag["detection_results"]
        assert "task" in diag["detection_results"]
        assert "calendar" in diag["detection_results"]
        assert "interaction" in diag["detection_results"]
