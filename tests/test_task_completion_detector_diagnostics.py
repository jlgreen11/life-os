"""
Tests for the per-strategy diagnostics added to TaskCompletionDetector.

Diagnostics let operators see, after each detect_completions() run, how many
pending tasks each strategy considered, how many keyword matches the
activity strategy found, and how many tasks each strategy actually closed.
Without this visibility, "no completions today" is indistinguishable from
"the detector is broken" — see the data quality report where tasks.by_status
shows {} and there is no way to tell which case it is.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.task_completion_detector.detector import TaskCompletionDetector


@pytest.fixture
def detector(db, event_bus):
    """A TaskCompletionDetector with a mocked task_manager so we don't actually mutate tasks."""
    task_manager = MagicMock()
    task_manager.complete_task = AsyncMock()
    return TaskCompletionDetector(db, task_manager, event_bus)


@pytest.fixture
def base_time():
    """A stable "now" reference for the test cases."""
    return datetime.now(timezone.utc)


def _insert_task(db, task_id, title, created_at, description="", status="pending"):
    """Insert a single task row directly via SQL, matching the existing tests' style."""
    with db.get_connection("state") as conn:
        conn.execute(
            """
            INSERT INTO tasks (id, title, description, status, source, domain,
                             priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'ai', 'personal', 'normal', ?, ?)
            """,
            (
                task_id,
                title,
                description,
                status,
                created_at.isoformat(),
                created_at.isoformat(),
            ),
        )


def _insert_event(db, event_type, payload, timestamp):
    """Insert a single event row directly via SQL."""
    event_id = str(uuid.uuid4())
    with db.get_connection("events") as conn:
        conn.execute(
            """
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, ?, 'test', ?, 'normal', ?, '{}')
            """,
            (event_id, event_type, timestamp.isoformat(), json.dumps(payload)),
        )
    return event_id


class TestDiagnosticsShape:
    """The diagnostics dict must always have a predictable shape after a run."""

    @pytest.mark.asyncio
    async def test_get_last_run_diagnostics_empty_before_first_run(self, detector):
        """No detect_completions() yet → diagnostics is an empty dict."""
        assert detector.get_last_run_diagnostics() == {}

    @pytest.mark.asyncio
    async def test_zero_pending_tasks_yields_all_zero_counters(self, detector):
        """With an empty tasks table every counter should be 0 and total_completions=0."""
        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()

        # Top-level shape
        assert set(diag.keys()) == {
            'last_run_at',
            'pending_tasks_examined',
            'activity',
            'inactivity',
            'stale',
            'total_completions',
        }
        assert diag['pending_tasks_examined'] == 0
        assert diag['total_completions'] == 0

        # Per-strategy counters all zero
        assert diag['activity'] == {
            'tasks_examined': 0,
            'sent_events_examined': 0,
            'keyword_matches': 0,
            'completions': 0,
        }
        assert diag['inactivity'] == {'tasks_examined': 0, 'completions': 0}
        assert diag['stale'] == {'tasks_examined': 0, 'completions': 0}

        # last_run_at must be a parseable ISO8601 string
        datetime.fromisoformat(diag['last_run_at'])

    @pytest.mark.asyncio
    async def test_get_last_run_diagnostics_returns_copy(self, detector):
        """Caller mutations to the returned dict must not affect future runs."""
        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()
        diag['total_completions'] = 999
        # Internal state untouched
        assert detector._last_run_diagnostics['total_completions'] == 0


class TestActivityCounters:
    """Counters for the activity-based strategy."""

    @pytest.mark.asyncio
    async def test_pending_task_with_matching_sent_event_increments_counters(
        self, detector, db, base_time
    ):
        """A task whose keywords appear in a completion-signal email should bump
        activity.completions and activity.keyword_matches both to 1."""
        task_id = str(uuid.uuid4())
        _insert_task(
            db,
            task_id,
            "Send quarterly report manager",
            created_at=base_time - timedelta(hours=3),
        )
        # Sent email contains task keywords ("quarterly", "report", "manager")
        # AND a completion keyword ("sent"/"done"), so the activity strategy
        # should both record the keyword match and mark the task complete.
        _insert_event(
            db,
            "email.sent",
            {
                "subject": "Quarterly report",
                "body_plain": "The quarterly report for the manager is done and sent.",
            },
            timestamp=base_time - timedelta(minutes=30),
        )

        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()

        assert diag['activity']['tasks_examined'] == 1
        assert diag['activity']['sent_events_examined'] == 1
        assert diag['activity']['keyword_matches'] == 1
        assert diag['activity']['completions'] == 1
        assert diag['total_completions'] >= 1

    @pytest.mark.asyncio
    async def test_keyword_match_without_completion_signal_records_match_only(
        self, detector, db, base_time
    ):
        """Keyword overlap with no completion word should bump keyword_matches but NOT completions.

        This is the diagnostic that catches "the detector saw references to
        the task but never closed it" — exactly the silent-failure mode the
        old aggregate log line obscured.
        """
        task_id = str(uuid.uuid4())
        _insert_task(
            db,
            task_id,
            "Schedule dentist appointment",
            created_at=base_time - timedelta(hours=3),
        )
        # Mentions "dentist" and "schedule" but contains NO completion keyword.
        _insert_event(
            db,
            "email.sent",
            {
                "subject": "Weekend plans",
                "body_plain": (
                    "I should schedule a dentist appointment for next week, "
                    "but haven't gotten around to it."
                ),
            },
            timestamp=base_time - timedelta(minutes=30),
        )

        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()

        assert diag['activity']['tasks_examined'] == 1
        assert diag['activity']['sent_events_examined'] == 1
        assert diag['activity']['keyword_matches'] == 1
        assert diag['activity']['completions'] == 0


class TestInactivityAndStaleCounters:
    """Counters for the inactivity and stale strategies."""

    @pytest.mark.asyncio
    async def test_old_pending_task_with_no_activity_records_stale_completion(
        self, detector, db, base_time
    ):
        """A 35-day-old pending task should count as one stale completion.

        Note: the same task is also past the 7-day inactivity threshold, so
        the inactivity strategy will examine it too. We assert on the stale
        counter specifically since the task description called this out as
        the stale-cleanup case.
        """
        task_id = str(uuid.uuid4())
        _insert_task(
            db,
            task_id,
            "Very old forgotten task",
            created_at=base_time - timedelta(days=35),
        )

        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()

        assert diag['stale']['tasks_examined'] == 1
        assert diag['stale']['completions'] == 1
        # The task is also closed once by the inactivity strategy because both
        # strategies run independently and there's no de-dup between them.
        # Total completions is therefore the sum across strategies.
        assert diag['total_completions'] == (
            diag['activity']['completions']
            + diag['inactivity']['completions']
            + diag['stale']['completions']
        )

    @pytest.mark.asyncio
    async def test_recent_task_does_not_increment_inactivity_or_stale(
        self, detector, db, base_time
    ):
        """A 1-day-old task is too young for either inactivity or stale to act on it."""
        _insert_task(
            db,
            str(uuid.uuid4()),
            "Fresh task",
            created_at=base_time - timedelta(days=1),
        )

        await detector.detect_completions()
        diag = detector.get_last_run_diagnostics()

        assert diag['pending_tasks_examined'] == 1
        assert diag['inactivity']['tasks_examined'] == 0
        assert diag['inactivity']['completions'] == 0
        assert diag['stale']['tasks_examined'] == 0
        assert diag['stale']['completions'] == 0
