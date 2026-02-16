#!/usr/bin/env python3
"""
Life OS — Backfill Task Completion Detection

Retroactively detects which historical tasks should already be marked complete
based on behavioral signals (sent emails/messages that reference the task).

The task completion detector runs on a schedule, but it only processes tasks
created more than 7 days ago. For tasks extracted from historical emails via
backfill_task_extraction.py, we need to check if the user ALREADY completed
them before the task was even extracted.

Common pattern: User receives email → completes work → sends confirmation →
days/weeks later backfill extracts the task. The task is marked "pending" but
should be "completed" because the sent event already happened.

This script:
1. Finds all pending tasks
2. For each task, searches for sent emails/messages AFTER task creation
3. Checks if sent content references the task + contains completion keywords
4. Marks matching tasks complete and publishes task.completed events
5. Enables workflow detection by closing the task lifecycle loop

Usage:
    python scripts/backfill_task_completion.py

Safety:
    - Only marks tasks complete, never deletes
    - Requires keyword overlap >= 2.0 to avoid false positives
    - Requires completion signal words (done, finished, sent, etc.)
    - Logs all completion decisions for audit
"""

import asyncio
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaskCompletionBackfill:
    """Backfills task completion status from historical behavioral signals.

    Analyzes all pending tasks to determine if they were already completed
    based on sent emails/messages that reference the task. This closes the
    task lifecycle loop and enables workflow detection (Layer 3 procedural memory).
    """

    def __init__(self, db: DatabaseManager):
        """Initialize the backfill processor.

        Args:
            db: Database manager for accessing tasks and events
        """
        self.db = db

        # Completion signal keywords to look for in sent email/message content
        self.completion_keywords = {
            'done', 'finished', 'completed', 'sent', 'submitted',
            'delivered', 'shipped', 'resolved', 'closed', 'fixed',
            'merged', 'deployed', 'published', 'launched', 'ready'
        }

        # Stop words to filter out when extracting task keywords
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'with', 'from', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'each', 'other', 'some', 'such', 'only', 'own', 'same',
            'than', 'too', 'very', 'just', 'should', 'would', 'could', 'will'
        }

    def run(self) -> dict:
        """Run the backfill process.

        Returns:
            Statistics about the backfill run (tasks processed, completed, etc.)
        """
        logger.info("Starting task completion backfill...")

        stats = {
            'tasks_checked': 0,
            'tasks_completed': 0,
            'events_published': 0,
            'start_time': datetime.now(timezone.utc).isoformat(),
        }

        # Get all pending tasks
        pending_tasks = self._get_pending_tasks()
        stats['tasks_checked'] = len(pending_tasks)

        logger.info(f"Found {len(pending_tasks)} pending tasks to check")

        # Check each task for completion signals
        for task in pending_tasks:
            if self._is_task_completed(task):
                self._mark_task_complete(task)
                stats['tasks_completed'] += 1
                stats['events_published'] += 1

        stats['end_time'] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Backfill complete: {stats['tasks_completed']}/{stats['tasks_checked']} "
            f"tasks marked complete"
        )

        return stats

    def _get_pending_tasks(self) -> list[dict]:
        """Fetch all pending tasks from the database.

        Returns:
            List of task dictionaries with id, title, description, created_at
        """
        with self.db.get_connection("state") as conn:
            cursor = conn.execute("""
                SELECT id, title, description, created_at, source
                FROM tasks
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def _is_task_completed(self, task: dict) -> bool:
        """Check if a task should be marked complete based on behavioral signals.

        Searches for sent emails/messages after task creation that:
        1. Reference the task (keyword overlap >= 2.0)
        2. Contain completion signal words (done, finished, sent, etc.)

        Args:
            task: Task dictionary with id, title, description, created_at

        Returns:
            True if the task shows completion signals, False otherwise
        """
        task_id = task['id']
        task_title = task['title'].lower() if task['title'] else ''
        task_desc = task['description'].lower() if task['description'] else ''
        created_at = task['created_at']

        # Extract meaningful keywords from task title for matching
        # Remove stop words and use stemming (first 4 chars) for variant matching
        title_words = {
            word for word in re.findall(r'\w+', task_title)
            if len(word) > 3 and word not in self.stop_words
        }
        title_stems = {word[:4] for word in title_words if len(word) >= 4}

        if not title_words and not title_stems:
            # No meaningful keywords to match against
            return False

        # Search for sent emails/messages after task creation
        # Limit to 100 most recent to avoid scanning entire history for old tasks
        with self.db.get_connection("events") as conn:
            cursor = conn.execute("""
                SELECT id, type, payload, timestamp
                FROM events
                WHERE type IN ('email.sent', 'message.sent')
                  AND timestamp >= ?
                ORDER BY timestamp ASC
                LIMIT 100
            """, (created_at,))

            sent_events = cursor.fetchall()

        # Check each sent event for task reference + completion keywords
        for event_row in sent_events:
            event_id, event_type, payload_json, timestamp = event_row

            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                continue

            # Extract text content from the payload
            text_content = self._extract_text_content(payload).lower()

            if not text_content:
                continue

            # Count keyword matches between task and email content
            # Match both full words and stems for better coverage
            text_words = set(re.findall(r'\w+', text_content))
            text_stems = {word[:4] for word in text_words if len(word) >= 4}

            # Count exact word matches
            exact_matches = len(title_words & text_words)
            # Count stem matches (catch send/sent, complete/completed, etc.)
            stem_matches = len(title_stems & text_stems)
            # Total score is weighted sum (exact matches worth more)
            keyword_overlap = exact_matches + (stem_matches * 0.5)

            # Check for completion signal keywords
            has_completion_keyword = any(
                keyword in text_content for keyword in self.completion_keywords
            )

            # If we have both keyword overlap AND completion signals, mark complete
            # Require score >= 2.0 to avoid false positives
            if keyword_overlap >= 2.0 and has_completion_keyword:
                logger.info(
                    f"Task '{task['title'][:50]}' completed via {event_type} "
                    f"({keyword_overlap:.1f} keyword matches, timestamp={timestamp})"
                )
                return True

        return False

    def _extract_text_content(self, payload: dict) -> str:
        """Extract searchable text from an event payload.

        Pulls out subject lines, body content, and any other text fields
        that might reference a task.

        Args:
            payload: Event payload dictionary

        Returns:
            Combined text content as a single string
        """
        text_parts = []

        # Email/message fields
        if payload.get('subject'):
            text_parts.append(payload['subject'])
        if payload.get('body_plain'):
            text_parts.append(payload['body_plain'])
        if payload.get('snippet'):
            text_parts.append(payload['snippet'])
        if payload.get('description'):
            text_parts.append(payload['description'])

        # Calendar event fields
        if payload.get('summary'):
            text_parts.append(payload['summary'])
        if payload.get('title'):
            text_parts.append(payload['title'])

        return ' '.join(text_parts)

    def _mark_task_complete(self, task: dict):
        """Mark a task as completed and publish a task.completed event.

        Updates the task status in state.db and publishes a task.completed
        event to enable workflow detection.

        Args:
            task: Task dictionary with id, title, etc.
        """
        task_id = task['id']

        # Update task status to completed
        with self.db.get_connection("state") as conn:
            conn.execute("""
                UPDATE tasks
                SET status = 'completed',
                    completed_at = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), task_id))

        # Publish task.completed event for workflow detection
        event = {
            'id': f"{task_id}-completion",  # Stable ID for idempotency
            'type': 'task.completed',
            'source': 'backfill.task_completion',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'priority': 'normal',
            'payload': {
                'task_id': task_id,
                'title': task['title'],
                'source': task.get('source', 'unknown'),
                'backfill': True  # Flag this as a backfilled completion
            },
            'metadata': {
                'backfill_run': True,
                'detection_method': 'behavioral_signal'
            }
        }

        # Store the event in events.db
        with self.db.get_connection("events") as conn:
            conn.execute("""
                INSERT OR IGNORE INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event['id'],
                event['type'],
                event['source'],
                event['timestamp'],
                event['priority'],
                json.dumps(event['payload']),
                json.dumps(event['metadata'])
            ))

        logger.debug(f"Marked task {task_id} as completed")


def main():
    """Main entry point for the backfill script."""
    # Initialize database manager
    db = DatabaseManager(data_dir="./data")

    # Run backfill
    backfill = TaskCompletionBackfill(db)
    stats = backfill.run()

    # Print summary
    print("\n" + "="*60)
    print("TASK COMPLETION BACKFILL SUMMARY")
    print("="*60)
    print(f"Tasks checked:    {stats['tasks_checked']}")
    print(f"Tasks completed:  {stats['tasks_completed']}")
    print(f"Events published: {stats['events_published']}")
    print(f"Start time:       {stats['start_time']}")
    print(f"End time:         {stats['end_time']}")
    print("="*60)

    return 0 if stats['tasks_completed'] >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
