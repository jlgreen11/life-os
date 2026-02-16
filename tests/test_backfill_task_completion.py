"""
Tests for scripts/backfill_task_completion.py

Verifies that the task completion backfill script correctly identifies
and marks completed tasks based on historical behavioral signals.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

# Import the backfill module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from backfill_task_completion import TaskCompletionBackfill


@pytest.fixture
def backfill(db):
    """Create a TaskCompletionBackfill instance with test database."""
    return TaskCompletionBackfill(db)


def create_task(db, title: str, description: str = "", created_at: datetime = None) -> str:
    """Helper to create a test task.

    Args:
        db: Database manager
        title: Task title
        description: Task description (optional)
        created_at: Task creation timestamp (defaults to now)

    Returns:
        Task ID
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    task_id = str(uuid4())

    with db.get_connection("state") as conn:
        conn.execute("""
            INSERT INTO tasks (id, title, description, status, created_at, source)
            VALUES (?, ?, ?, 'pending', ?, 'test')
        """, (task_id, title, description, created_at.isoformat()))

    return task_id


def create_sent_email(
    db,
    subject: str,
    body: str,
    timestamp: datetime = None
) -> str:
    """Helper to create a sent email event.

    Args:
        db: Database manager
        subject: Email subject
        body: Email body text
        timestamp: Email timestamp (defaults to now)

    Returns:
        Event ID
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    event_id = str(uuid4())

    payload = {
        'subject': subject,
        'body_plain': body,
        'to_address': 'recipient@example.com',
        'from_address': 'user@example.com'
    }

    with db.get_connection("events") as conn:
        conn.execute("""
            INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
            VALUES (?, 'email.sent', 'test', ?, 'normal', ?, '{}')
        """, (event_id, timestamp.isoformat(), json.dumps(payload)))

    return event_id


def get_task_status(db, task_id: str) -> str:
    """Get the status of a task.

    Args:
        db: Database manager
        task_id: Task ID

    Returns:
        Task status ('pending' or 'completed')
    """
    with db.get_connection("state") as conn:
        cursor = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def count_completion_events(db) -> int:
    """Count task.completed events in the database.

    Args:
        db: Database manager

    Returns:
        Number of task.completed events
    """
    with db.get_connection("events") as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM events WHERE type = 'task.completed'"
        )
        return cursor.fetchone()[0]


class TestTaskCompletionBackfill:
    """Test suite for task completion backfill."""

    def test_no_tasks(self, backfill, db):
        """Should handle empty task list gracefully."""
        stats = backfill.run()

        assert stats['tasks_checked'] == 0
        assert stats['tasks_completed'] == 0
        assert stats['events_published'] == 0

    def test_task_with_no_sent_messages(self, backfill, db):
        """Should leave task pending if no sent messages reference it."""
        task_id = create_task(db, "Review quarterly report")

        stats = backfill.run()

        assert stats['tasks_checked'] == 1
        assert stats['tasks_completed'] == 0
        assert get_task_status(db, task_id) == 'pending'

    def test_task_completed_exact_match(self, backfill, db):
        """Should mark task complete when sent email has exact keyword match + completion signal."""
        # Create task at T0
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(
            db,
            "Submit expense report",
            created_at=created_at
        )

        # Create sent email at T0 + 1 hour with exact match
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="RE: Expense Report",
            body="Hi, I've submitted the expense report. It's done and ready for review.",
            timestamp=sent_at
        )

        stats = backfill.run()

        assert stats['tasks_checked'] == 1
        assert stats['tasks_completed'] == 1
        assert get_task_status(db, task_id) == 'completed'
        assert count_completion_events(db) == 1

    def test_task_completed_stem_match(self, backfill, db):
        """Should detect completion via stem matching (submit/submitted)."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=3)
        task_id = create_task(
            db,
            "Submit quarterly presentation",
            created_at=created_at
        )

        # Email uses "submitted" (past tense) vs task "submit" (present)
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Quarterly presentation submitted",
            body="The presentation has been submitted and is ready for the meeting. Done!",
            timestamp=sent_at
        )

        stats = backfill.run()

        assert stats['tasks_completed'] == 1
        assert get_task_status(db, task_id) == 'completed'

    def test_task_not_completed_missing_completion_keyword(self, backfill, db):
        """Should NOT mark complete if email lacks completion signal words."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(
            db,
            "Review budget proposal",
            created_at=created_at
        )

        # Email references task but has no completion keyword
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="RE: Budget Proposal",
            body="I'm working on reviewing the budget proposal. Will send updates soon.",
            timestamp=sent_at
        )

        stats = backfill.run()

        assert stats['tasks_completed'] == 0
        assert get_task_status(db, task_id) == 'pending'

    def test_task_not_completed_insufficient_keyword_overlap(self, backfill, db):
        """Should NOT mark complete if keyword overlap < 2.0."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(
            db,
            "Review quarterly budget proposal",
            created_at=created_at
        )

        # Email has only 1 matching keyword ("review") + completion word
        # Not enough overlap to avoid false positives
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Meeting done",
            body="The review meeting is done.",
            timestamp=sent_at
        )

        stats = backfill.run()

        assert stats['tasks_completed'] == 0
        assert get_task_status(db, task_id) == 'pending'

    def test_task_not_completed_sent_before_task_created(self, backfill, db):
        """Should NOT mark complete if sent email predates task creation."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Email sent BEFORE task was created
        sent_at = created_at - timedelta(hours=1)
        create_sent_email(
            db,
            subject="Invoice processed",
            body="The invoice processing is done and sent to accounting.",
            timestamp=sent_at
        )

        # Task created later
        task_id = create_task(
            db,
            "Process monthly invoice",
            created_at=created_at
        )

        stats = backfill.run()

        assert stats['tasks_completed'] == 0
        assert get_task_status(db, task_id) == 'pending'

    def test_multiple_tasks_partial_completion(self, backfill, db):
        """Should handle mixed scenario: some tasks complete, others pending."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=3)

        # Task 1: Will be completed
        task1_id = create_task(db, "Send client proposal", created_at=created_at)
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Client proposal sent",
            body="Hi team, the client proposal has been sent. Done!",
            timestamp=sent_at
        )

        # Task 2: Will remain pending (no sent message)
        task2_id = create_task(db, "Review legal documents", created_at=created_at)

        # Task 3: Will be completed
        task3_id = create_task(db, "Update project timeline", created_at=created_at)
        sent_at2 = created_at + timedelta(hours=2)
        create_sent_email(
            db,
            subject="RE: Project Timeline",
            body="I've updated the project timeline. Finished and shared with the team.",
            timestamp=sent_at2
        )

        stats = backfill.run()

        assert stats['tasks_checked'] == 3
        assert stats['tasks_completed'] == 2
        assert get_task_status(db, task1_id) == 'completed'
        assert get_task_status(db, task2_id) == 'pending'
        assert get_task_status(db, task3_id) == 'completed'
        assert count_completion_events(db) == 2

    def test_completion_event_payload(self, backfill, db):
        """Should publish task.completed event with correct payload."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(db, "Deploy new feature", created_at=created_at)

        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Feature deployed",
            body="The new feature has been deployed to production. Done!",
            timestamp=sent_at
        )

        backfill.run()

        # Check event was published correctly
        with db.get_connection("events") as conn:
            cursor = conn.execute("""
                SELECT payload, metadata
                FROM events
                WHERE type = 'task.completed'
            """)
            row = cursor.fetchone()

        assert row is not None
        payload = json.loads(row[0])
        metadata = json.loads(row[1])

        assert payload['task_id'] == task_id
        assert payload['title'] == "Deploy new feature"
        assert payload['backfill'] is True
        assert metadata['backfill_run'] is True
        assert metadata['detection_method'] == 'behavioral_signal'

    def test_stop_words_filtered(self, backfill, db):
        """Should filter out stop words when matching keywords."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Task with many stop words
        task_id = create_task(
            db,
            "Review the quarterly report for the finance team",
            created_at=created_at
        )

        # Email with some matches (quarterly, report, finance)
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Finance quarterly report",
            body="The quarterly finance report review is done and sent to the team.",
            timestamp=sent_at
        )

        stats = backfill.run()

        # Should match because stop words are filtered correctly
        assert stats['tasks_completed'] == 1
        assert get_task_status(db, task_id) == 'completed'

    def test_completion_keywords_variety(self, backfill, db):
        """Should recognize various completion signal keywords."""
        completion_words = [
            'done', 'finished', 'completed', 'sent', 'submitted',
            'delivered', 'shipped', 'resolved', 'closed', 'fixed',
            'merged', 'deployed', 'published', 'launched', 'ready'
        ]

        created_at = datetime.now(timezone.utc) - timedelta(hours=3)

        for idx, keyword in enumerate(completion_words[:5]):  # Test subset
            task_id = create_task(
                db,
                f"Complete task number {idx}",
                created_at=created_at
            )

            sent_at = created_at + timedelta(minutes=idx * 10)
            create_sent_email(
                db,
                subject=f"Task number {idx}",
                body=f"The task number {idx} is {keyword}.",
                timestamp=sent_at
            )

        stats = backfill.run()

        assert stats['tasks_completed'] == 5

    def test_task_with_description_matching(self, backfill, db):
        """Should extract keywords from task description as well as title."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Keywords in description, not title
        task_id = create_task(
            db,
            "Weekly report",
            description="Submit the quarterly financial analysis to accounting team",
            created_at=created_at
        )

        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Financial analysis",
            body="The quarterly financial analysis has been submitted. Done!",
            timestamp=sent_at
        )

        # Note: Current implementation only uses title for matching,
        # so this test documents that behavior. Could be enhanced to use description.
        stats = backfill.run()

        # With current implementation, this should NOT complete because
        # keywords are in description, not title
        # If we enhance to use description, change this assertion
        assert stats['tasks_completed'] == 0

    def test_idempotency(self, backfill, db):
        """Should be idempotent - running twice doesn't create duplicate events."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(db, "Finalize contract", created_at=created_at)

        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Contract finalized",
            body="The contract has been finalized and sent. Done!",
            timestamp=sent_at
        )

        # Run backfill twice
        stats1 = backfill.run()
        stats2 = backfill.run()

        # First run completes the task
        assert stats1['tasks_completed'] == 1

        # Second run finds no pending tasks (already completed)
        assert stats2['tasks_completed'] == 0

        # Only one completion event exists (INSERT OR IGNORE prevents duplicates)
        assert count_completion_events(db) == 1

    def test_message_sent_detection(self, backfill, db):
        """Should detect completion via message.sent events, not just email.sent."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        task_id = create_task(db, "Respond to customer inquiry", created_at=created_at)

        # Create message.sent event (e.g., Signal, iMessage)
        event_id = str(uuid4())
        sent_at = created_at + timedelta(hours=1)
        payload = {
            'body_plain': 'Hi! The customer inquiry response has been sent. Done!',
            'to_contact': 'Customer'
        }

        with db.get_connection("events") as conn:
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, 'message.sent', 'test', ?, 'normal', ?, '{}')
            """, (event_id, sent_at.isoformat(), json.dumps(payload)))

        stats = backfill.run()

        assert stats['tasks_completed'] == 1
        assert get_task_status(db, task_id) == 'completed'

    def test_short_task_title_no_keywords(self, backfill, db):
        """Should handle tasks with short titles that have no extractable keywords."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Very short title with only stop words
        task_id = create_task(db, "Do it now", created_at=created_at)

        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="All done",
            body="Everything is finished and complete!",
            timestamp=sent_at
        )

        stats = backfill.run()

        # Should remain pending because no keywords to match
        assert stats['tasks_completed'] == 0
        assert get_task_status(db, task_id) == 'pending'

    def test_case_insensitive_matching(self, backfill, db):
        """Should perform case-insensitive keyword matching."""
        created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        task_id = create_task(db, "SEND INVOICE TO CLIENT", created_at=created_at)

        # Email in mixed case
        sent_at = created_at + timedelta(hours=1)
        create_sent_email(
            db,
            subject="Invoice Sent",
            body="The invoice has been sent to the client. Done!",
            timestamp=sent_at
        )

        stats = backfill.run()

        assert stats['tasks_completed'] == 1
        assert get_task_status(db, task_id) == 'completed'

    def test_limit_100_sent_events(self, backfill, db):
        """Should limit search to 100 most recent sent events to avoid performance issues."""
        created_at = datetime.now(timezone.utc) - timedelta(days=30)
        task_id = create_task(db, "Old task review", created_at=created_at)

        # Create 150 sent events, only the first one matches
        for i in range(150):
            sent_at = created_at + timedelta(hours=i)
            if i == 0:
                # First email matches
                create_sent_email(
                    db,
                    subject="Task review complete",
                    body="The old task review is done!",
                    timestamp=sent_at
                )
            else:
                # Others don't match
                create_sent_email(
                    db,
                    subject=f"Unrelated email {i}",
                    body=f"Some other content {i}",
                    timestamp=sent_at
                )

        stats = backfill.run()

        # Should find the match even with 150 events because it's in the first 100
        assert stats['tasks_completed'] == 1
