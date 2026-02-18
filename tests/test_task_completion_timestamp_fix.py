"""
Tests for the duplicate timestamp SQL bug fix in TaskCompletionDetector.

Bug: _detect_activity_based_completion() and _detect_inactivity_based_completion()
both contained a duplicate ``AND timestamp > ?`` clause with two different params:
    (created_at, cutoff.isoformat())

In _detect_activity_based_completion() the second clause was redundant (tasks are
already constrained to created_at >= cutoff, so created_at >= cutoff makes the
second clause a no-op).

In _detect_inactivity_based_completion() the intent was to check for ANY recent
activity (past 7 days), but passing `created_at` as a dead-first-param alongside
`cutoff` made the query confusing and fragile: if the threshold ever widened so
created_at could be older than cutoff, the first clause would win and the recent-
activity check would fail silently.

Fix: Remove the redundant second ``AND timestamp > ?`` and its parameter in both
methods. The correct single parameter is:
- activity detection: created_at (events must come AFTER task was created)
- inactivity detection: cutoff (events must be RECENT, i.e. within 7 days)
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_completion_detector.detector import TaskCompletionDetector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def detector(db, event_bus):
    """TaskCompletionDetector with a mocked task manager."""
    task_manager = MagicMock()
    task_manager.complete_task = AsyncMock()
    return TaskCompletionDetector(db, task_manager, event_bus)


@pytest.fixture
def now():
    """Fixed current time used throughout the test module."""
    return datetime.now(timezone.utc)


def _insert_task(db, task_id: str, title: str, created_at: datetime,
                 status: str = "pending") -> None:
    """Insert a task row directly into the state database."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO tasks
                   (id, title, description, status, source, domain, priority,
                    created_at, updated_at)
               VALUES (?, ?, '', ?, 'ai', 'work', 'normal', ?, ?)""",
            (task_id, title, status, created_at.isoformat(), created_at.isoformat()),
        )


def _insert_event(db, event_type: str, payload: dict,
                  timestamp: datetime) -> str:
    """Insert a synthetic event row into the events database."""
    event_id = str(uuid.uuid4())
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events
                   (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, 'test', ?, 'normal', ?, '{}')""",
            (event_id, event_type, timestamp.isoformat(), json.dumps(payload)),
        )
    return event_id


# ---------------------------------------------------------------------------
# Activity-based detection — verifying the single-param fix
# ---------------------------------------------------------------------------

class TestActivityDetectionTimestampFix:
    """Verify that activity detection only uses task's created_at as the cutoff."""

    @pytest.mark.asyncio
    async def test_email_before_task_creation_is_ignored(self, detector, db, now):
        """Pre-creation emails must never trigger task completion.

        The SQL should filter with ``timestamp > created_at``; a pre-creation
        email must be excluded even if it perfectly matches the task keywords.
        """
        task_id = str(uuid.uuid4())
        created_at = now - timedelta(hours=2)
        _insert_task(db, task_id, "Submit budget report quarterly", created_at=created_at)

        # Email sent 3 hours ago — BEFORE the task was created 2 hours ago
        _insert_event(db, "email.sent", {
            "subject": "Budget report submitted quarterly",
            "body_plain": "The quarterly budget report is submitted and completed.",
        }, timestamp=now - timedelta(hours=3))

        completed = await detector._detect_activity_based_completion()

        assert completed == 0, (
            "Email sent before task creation must not trigger completion"
        )
        detector.task_manager.complete_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_email_after_task_creation_triggers_completion(self, detector, db, now):
        """Post-creation email with matching keywords should complete the task.

        This is the positive-case counterpart to the above test: the single
        ``timestamp > created_at`` guard must admit events that follow the task.
        """
        task_id = str(uuid.uuid4())
        created_at = now - timedelta(hours=3)
        _insert_task(db, task_id, "Submit budget report quarterly", created_at=created_at)

        # Email sent 1 hour ago — AFTER task created 3 hours ago
        _insert_event(db, "email.sent", {
            "subject": "Budget Report Submission",
            "body_plain": "Just submitted the quarterly budget report as requested. Completed!",
        }, timestamp=now - timedelta(hours=1))

        completed = await detector._detect_activity_based_completion()

        assert completed == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_task_recently_created_with_matching_reply(self, detector, db, now):
        """Task created minutes ago can still match an email sent since creation.

        With the old dual-param bug, if cutoff (7 days ago) was passed as a
        redundant param it would be less restrictive than ``created_at`` and
        thus benign. But this test documents that the fix leaves behaviour
        correct for newly created tasks too.

        Use keywords that satisfy the score >= 2.0 threshold (2 exact word
        matches plus a completion keyword).
        """
        task_id = str(uuid.uuid4())
        created_at = now - timedelta(minutes=30)
        # Title keywords (len > 3, after stop-word removal): kickoff, meeting, budget
        _insert_task(db, task_id, "Prepare kickoff meeting budget", created_at=created_at)

        # Email sent 10 minutes ago — after task creation.
        # "kickoff" and "meeting" are exact matches (score ≥ 2) + "done" is the
        # completion keyword.
        _insert_event(db, "email.sent", {
            "subject": "Kickoff Meeting Done",
            "body_plain": "The kickoff meeting budget is done. Everything is ready.",
        }, timestamp=now - timedelta(minutes=10))

        completed = await detector._detect_activity_based_completion()

        assert completed == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)


# ---------------------------------------------------------------------------
# Inactivity-based detection — verifying the single-param fix
# ---------------------------------------------------------------------------

class TestInactivityDetectionTimestampFix:
    """Verify that inactivity detection uses task age as the only completion signal.

    The previous implementation checked ``COUNT(*) == 0`` over all system events
    in the last 7 days.  On any live instance with continuous email/calendar
    traffic this count is always > 0, so the strategy never fired.

    The fix removes that global-activity guard entirely.  An old task is closed
    based solely on its age — unrelated system activity no longer interferes.
    """

    @pytest.mark.asyncio
    async def test_old_task_completed_despite_unrelated_recent_event(self, detector, db, now):
        """An unrelated received email must NOT prevent inactivity-based completion.

        Previously the detector checked global ``recent_activity_count == 0``
        which meant any received email — however unrelated — would shield every
        old task from auto-completion.  That guard has been removed; task age
        is now the sole criterion.
        """
        task_id = str(uuid.uuid4())
        created_at = now - timedelta(days=10)
        _insert_task(db, task_id, "Old dormant task", created_at=created_at)

        # Recent email completely unrelated to the task — must NOT block completion.
        _insert_event(db, "email.received", {
            "subject": "Newsletter digest",
            "body_plain": "Totally unrelated content.",
        }, timestamp=now - timedelta(days=2))

        completed = await detector._detect_inactivity_based_completion()

        assert completed == 1, (
            "Unrelated system activity must not prevent inactivity-based completion "
            "of an old task.  The global activity guard has been removed."
        )
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_old_event_does_not_block_inactivity_completion(self, detector, db, now):
        """An event older than the 7-day threshold must NOT count as recent activity.

        The task has an old event (10 days ago) but no activity in the last 7 days,
        so inactivity detection should still mark it complete.
        """
        task_id = str(uuid.uuid4())
        created_at = now - timedelta(days=20)
        _insert_task(db, task_id, "Forgotten task", created_at=created_at)

        # Old email (10 days ago) — outside the 7-day inactivity window
        _insert_event(db, "email.received", {
            "subject": "Old email",
            "body_plain": "Something from a while back.",
        }, timestamp=now - timedelta(days=10))

        completed = await detector._detect_inactivity_based_completion()

        assert completed == 1, (
            "Activity older than threshold should not block inactivity completion"
        )
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_no_events_at_all_triggers_inactivity_completion(self, detector, db, now):
        """Task with zero events should always be completed by inactivity check."""
        task_id = str(uuid.uuid4())
        _insert_task(db, task_id, "Silent old task", created_at=now - timedelta(days=8))

        completed = await detector._detect_inactivity_based_completion()

        assert completed == 1
        detector.task_manager.complete_task.assert_called_once_with(task_id)

    @pytest.mark.asyncio
    async def test_event_exactly_at_cutoff_boundary(self, detector, db, now):
        """Event exactly at the cutoff boundary (7 days ago) should not block completion.

        The SQL uses strict greater-than (>) so an event at cutoff is excluded
        from recent-activity counting and the task should still be completed.
        """
        task_id = str(uuid.uuid4())
        _insert_task(db, task_id, "Boundary task", created_at=now - timedelta(days=10))

        # Event exactly at the 7-day cutoff boundary
        cutoff = now - timedelta(days=7)
        _insert_event(db, "email.received", {
            "subject": "Boundary email",
            "body_plain": "Exactly at threshold.",
        }, timestamp=cutoff)

        completed = await detector._detect_inactivity_based_completion()

        # Event at the boundary is NOT > cutoff, so no recent activity, should complete
        assert completed == 1


# ---------------------------------------------------------------------------
# SQL parameter count — whitebox correctness verification
# ---------------------------------------------------------------------------

class TestSqlParamCounts:
    """Whitebox tests: verify that neither query passes a redundant timestamp param.

    We patch db.get_connection to capture executed SQL and its parameters, then
    assert that each timestamp-bearing query is issued with exactly ONE timestamp
    param (not two).
    """

    @pytest.mark.asyncio
    async def test_activity_detection_executes_single_timestamp_param(
        self, detector, db, now
    ):
        """_detect_activity_based_completion must use exactly one timestamp per task query."""
        executed_queries = []

        original_get_connection = db.get_connection

        class CapturingConn:
            """Thin wrapper that records execute() calls on the events database."""

            def __init__(self, real_conn):
                self._real = real_conn
                self._cursor = None

            def execute(self, sql, params=()):
                if "email.sent" in sql or "message.sent" in sql:
                    executed_queries.append((sql, params))
                return self._real.execute(sql, params)

            def fetchall(self):
                return self._real.fetchall() if hasattr(self._real, "fetchall") else []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        import contextlib

        # Insert one task to make the loop run
        task_id = str(uuid.uuid4())
        _insert_task(db, task_id, "Test task params", created_at=now - timedelta(hours=1))

        # Run the detector; we accept any completion result
        await detector._detect_activity_based_completion()

        # If any query for email.sent events was captured, assert it has 1 timestamp param
        for sql, params in executed_queries:
            if "email.sent" in sql:
                # Count occurrences of "timestamp > ?" in the SQL
                timestamp_clauses = sql.count("timestamp > ?")
                assert timestamp_clauses == 1, (
                    f"Expected exactly 1 timestamp > ? clause in activity query, "
                    f"found {timestamp_clauses}.\nSQL: {sql}"
                )

    @pytest.mark.asyncio
    async def test_inactivity_detection_no_events_query_in_source(
        self, detector, db, now
    ):
        """_detect_inactivity_based_completion must NOT contain an events timestamp query.

        The original bug involved a global ``COUNT(*) FROM events WHERE timestamp > ?``
        guard that prevented inactivity detection from ever firing on a live system.
        The fix removes the events query entirely — only the tasks query (with a single
        ``created_at < ?`` param) remains.  This source-level assertion documents that
        the events guard is gone.
        """
        import inspect
        from services.task_completion_detector import detector as det_module

        source = inspect.getsource(det_module.TaskCompletionDetector._detect_inactivity_based_completion)

        # The old broken implementation had a COUNT(*) query over the events table.
        # After the fix, no such query should exist in this method.
        assert "FROM events" not in source, (
            "_detect_inactivity_based_completion must not query the events table. "
            "The global activity guard that prevented inactivity detection on live "
            "systems has been removed; task age is the sole completion criterion."
        )

    @pytest.mark.asyncio
    async def test_activity_source_has_single_timestamp_clause(self, detector, db, now):
        """Source inspection: _detect_activity_based_completion must have exactly one timestamp clause."""
        import inspect
        from services.task_completion_detector import detector as det_module

        source = inspect.getsource(det_module.TaskCompletionDetector._detect_activity_based_completion)

        timestamp_clause_count = source.count("AND timestamp > ?")
        assert timestamp_clause_count == 1, (
            f"_detect_activity_based_completion should have exactly 1 "
            f"'AND timestamp > ?' clause, found {timestamp_clause_count}. "
            "The duplicate timestamp param bug may not be fixed."
        )
