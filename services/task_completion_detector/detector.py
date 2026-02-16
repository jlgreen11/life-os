"""
Life OS — Task Completion Detector

Automatically detects when tasks have been completed based on behavioral signals.
Since many tasks are completed outside the system (in real life, via email, etc.),
we need to infer completion from user behavior rather than explicit UI actions.

Detection strategies:
1. **Activity matching** — User sends email/message mentioning the task
2. **Completion keywords** — Communication contains "done", "finished", "completed"
3. **Inactivity aging** — Task has no activity for 7+ days (likely completed or abandoned)
4. **Stale task cleanup** — Tasks older than 30 days with zero activity are auto-archived

This enables workflow detection (Layer 3 procedural memory) by ensuring task.created
events have corresponding task.completed events when the work is actually done.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


class TaskCompletionDetector:
    """Detects task completion from behavioral signals.

    Tasks are often completed without explicit UI interaction - users finish
    work by sending emails, attending meetings, or simply doing the task offline.
    This detector watches for those signals and automatically marks tasks complete.

    Detection heuristics:
    - Email/message sent that references the task title/description
    - Communication contains completion keywords ("done", "finished", "sent")
    - Task has been inactive for 7+ days (assumed completed or abandoned)
    - Task is older than 30 days with zero related activity (stale, auto-archive)

    Each completed task publishes a task.completed event so the workflow detector
    can learn multi-step task completion patterns (read email → research → execute
    → confirm completion).
    """

    def __init__(self, db: DatabaseManager, task_manager: Any, event_bus: Any):
        """Initialize the task completion detector.

        Args:
            db: Database manager for querying tasks and events
            task_manager: Task manager for marking tasks complete
            event_bus: Event bus for publishing completion events
        """
        self.db = db
        self.task_manager = task_manager
        self.event_bus = event_bus

        # Detection thresholds
        self.inactivity_days = 7  # Mark tasks inactive after 7 days with no signals
        self.stale_days = 30  # Archive tasks older than 30 days with zero activity

        # Completion keywords to look for in email/message content
        self.completion_keywords = {
            'done', 'finished', 'completed', 'sent', 'submitted',
            'delivered', 'shipped', 'resolved', 'closed', 'fixed',
            'merged', 'deployed', 'published', 'launched'
        }

    async def detect_completions(self) -> int:
        """Scan all pending tasks and detect which ones are completed.

        Runs multiple detection strategies:
        1. Activity-based: tasks referenced in recent emails/messages
        2. Keyword-based: communication with completion signals
        3. Inactivity-based: tasks with no activity for threshold period
        4. Stale task cleanup: very old tasks with zero activity

        Returns:
            Number of tasks marked complete
        """
        completed_count = 0

        # Strategy 1: Activity-based completion detection
        activity_completions = await self._detect_activity_based_completion()
        completed_count += activity_completions

        # Strategy 2: Inactivity-based completion (tasks that went silent)
        inactivity_completions = await self._detect_inactivity_based_completion()
        completed_count += inactivity_completions

        # Strategy 3: Stale task cleanup (very old tasks, likely abandoned)
        stale_completions = await self._detect_stale_tasks()
        completed_count += stale_completions

        if completed_count > 0:
            logger.info(
                f"Auto-detected {completed_count} completed tasks "
                f"({activity_completions} activity, {inactivity_completions} inactive, "
                f"{stale_completions} stale)"
            )

        return completed_count

    async def _detect_activity_based_completion(self) -> int:
        """Detect task completion from email/message activity.

        Looks for tasks where the user sent an email or message that:
        1. References the task title/description (keyword overlap)
        2. Contains completion signal words ("done", "finished", "sent")
        3. Was sent after the task was created

        This catches the common pattern: receive email → create task → do work
        → send email confirming completion.

        Returns:
            Number of tasks marked complete via activity signals
        """
        completed_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.inactivity_days)

        # Get all pending tasks
        with self.db.get_connection("state") as conn:
            cursor = conn.execute("""
                SELECT id, title, description, created_at, source
                FROM tasks
                WHERE status = 'pending'
                  AND created_at >= ?
                ORDER BY created_at DESC
            """, (cutoff.isoformat(),))

            tasks = [dict(row) for row in cursor.fetchall()]

        # For each task, look for sent emails/messages after creation that reference it
        for task in tasks:
            task_id = task['id']
            task_title = task['title'].lower() if task['title'] else ''
            task_desc = task['description'].lower() if task['description'] else ''
            created_at = task['created_at']

            # Extract keywords from task title and description for matching
            # Remove common stop words and keep meaningful terms
            # Use simple stemming by taking first 4 characters to match word variants
            # (send/sent/sending, complete/completed/completing, etc.)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'with', 'from', 'about', 'into', 'through', 'during', 'before',
                         'after', 'above', 'below', 'between', 'under', 'again', 'further',
                         'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                         'all', 'each', 'other', 'some', 'such', 'only', 'own', 'same',
                         'than', 'too', 'very', 'just', 'should', 'would', 'could'}
            title_words = {
                word for word in re.findall(r'\w+', task_title)
                if len(word) > 3 and word not in stop_words
            }
            # Also extract stems (first 4-5 chars) for better matching
            title_stems = {word[:4] for word in title_words if len(word) >= 4}

            if not title_words and not title_stems:
                continue  # No meaningful keywords to match against

            # Search for sent emails/messages after task creation
            with self.db.get_connection("events") as conn:
                cursor = conn.execute("""
                    SELECT id, type, payload, timestamp
                    FROM events
                    WHERE type IN ('email.sent', 'message.sent')
                      AND timestamp > ?
                      AND timestamp > ?
                    ORDER BY timestamp ASC
                    LIMIT 50
                """, (created_at, cutoff.isoformat()))

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
                # Require score >= 2.0 to avoid false positives (e.g., 2 exact or 1 exact + 2 stems)
                if keyword_overlap >= 2.0 and has_completion_keyword:
                    await self.task_manager.complete_task(task_id)
                    completed_count += 1
                    logger.debug(
                        f"Auto-completed task '{task['title']}' based on sent "
                        f"{event_type} with {keyword_overlap} keyword matches"
                    )
                    break  # Move to next task

        return completed_count

    async def _detect_inactivity_based_completion(self) -> int:
        """Detect completion from task inactivity.

        Tasks that haven't had any related activity for {inactivity_days} days
        are likely completed or abandoned. We mark them complete to close the loop.

        This handles cases where users complete tasks but don't explicitly mark
        them done or send confirmation emails (e.g., personal tasks, quick fixes).

        Returns:
            Number of tasks marked complete due to inactivity
        """
        completed_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.inactivity_days)

        # Find tasks created more than {inactivity_days} ago that are still pending
        with self.db.get_connection("state") as conn:
            cursor = conn.execute("""
                SELECT id, title, created_at
                FROM tasks
                WHERE status = 'pending'
                  AND created_at < ?
                ORDER BY created_at ASC
            """, (cutoff.isoformat(),))

            inactive_tasks = [dict(row) for row in cursor.fetchall()]

        # For each inactive task, check if there's been ANY recent activity
        for task in inactive_tasks:
            task_id = task['id']
            created_at = task['created_at']

            # Check for any events that might reference this task
            # (emails, messages, calendar events created after the task)
            with self.db.get_connection("events") as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM events
                    WHERE type IN ('email.sent', 'email.received', 'message.sent',
                                   'message.received', 'calendar.event.created')
                      AND timestamp > ?
                      AND timestamp > ?
                """, (created_at, cutoff.isoformat()))

                recent_activity_count = cursor.fetchone()[0]

            # If there's been activity in the lookback window, the user is still
            # engaged with this area of work, so keep the task pending
            # If there's been NO activity at all recently, mark complete (or abandoned)
            if recent_activity_count == 0:
                await self.task_manager.complete_task(task_id)
                completed_count += 1
                logger.debug(
                    f"Auto-completed inactive task '{task['title']}' "
                    f"(no activity for {self.inactivity_days}+ days)"
                )

        return completed_count

    async def _detect_stale_tasks(self) -> int:
        """Archive very old tasks with zero activity.

        Tasks older than {stale_days} days that never had any related activity
        are likely data quality issues (extracted incorrectly, duplicates, etc.)
        or tasks that became irrelevant. Mark them complete to clean up the list.

        Returns:
            Number of stale tasks archived
        """
        completed_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.stale_days)

        # Find very old pending tasks
        with self.db.get_connection("state") as conn:
            cursor = conn.execute("""
                SELECT id, title, created_at
                FROM tasks
                WHERE status = 'pending'
                  AND created_at < ?
                ORDER BY created_at ASC
            """, (cutoff.isoformat(),))

            stale_tasks = [dict(row) for row in cursor.fetchall()]

        # Mark them all complete - they're too old to be actionable
        for task in stale_tasks:
            await self.task_manager.complete_task(task['id'])
            completed_count += 1
            logger.debug(
                f"Archived stale task '{task['title']}' "
                f"(created {task['created_at']}, {self.stale_days}+ days old)"
            )

        return completed_count

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
