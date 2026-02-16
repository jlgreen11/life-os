"""
Life OS — Workflow Detector

Analyzes event sequences to detect multi-step task-completion workflows.
Unlike routines (time/location-triggered patterns), workflows are goal-driven
processes that span multiple interaction types and tools.

Workflow types detected:
- Communication workflows (read → research → draft → review → send)
- Task completion workflows (receive → analyze → execute → confirm)
- Research workflows (search → collect → synthesize → save)
- Decision workflows (gather info → evaluate → decide → act)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from storage.manager import DatabaseManager
    from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


class WorkflowDetector:
    """Detects multi-step task-completion workflows from event sequences.

    Analyzes episodes and events to find recurring goal-driven processes.
    Workflows are characterized by:
    - Trigger conditions (event types, contexts that initiate the workflow)
    - Sequence of steps (actions taken to complete the goal)
    - Tools used (email, calendar, browser, apps involved)
    - Success rate (how often the workflow completes vs. abandons)
    - Timing (typical duration from start to completion)

    Example workflows:
    - Responding to boss: read email → draft → check tone → send
    - Planning trip: search → book hotel → book flight → calendar → notify
    - Weekly review: check completed tasks → review calendar → plan next week
    - Handle invoice: receive → review → approve → forward to accounting
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        """Initialize the workflow detector.

        Args:
            db: Database manager for querying events and episodes
            user_model_store: Store for persisting detected workflows
        """
        self.db = db
        self.user_model_store = user_model_store

        # Detection thresholds
        self.min_occurrences = 3  # Need at least 3 instances to identify a workflow
        self.max_step_gap_hours = 4  # Steps within 4h can be part of same workflow
        self.min_steps = 2  # Workflows must have at least 2 distinct steps
        # Lower success threshold to 1% to handle realistic email response rates.
        # Most emails (marketing, newsletters, notifications) don't require responses.
        # With 77K emails received and 229 sent, the response rate is ~0.3%, so a 40%
        # threshold would block all workflow detection. A 1% threshold allows detection
        # of workflows that actually happen (e.g., "respond to boss emails") without
        # requiring unrealistic response rates.
        self.success_threshold = 0.01  # 1% success rate minimum to store workflow

    def detect_workflows(self, lookback_days: int = 30) -> list[dict[str, Any]]:
        """Detect all workflows from recent event history.

        Runs multiple detection strategies:
        1. Email-response workflows (receive → draft → send patterns)
        2. Task-completion workflows (identified → worked → completed sequences)
        3. Calendar-based workflows (event created → prepared → attended → follow-up)
        4. Multi-tool research workflows (search → collect → synthesize patterns)

        Args:
            lookback_days: How many days of history to analyze (default 30)

        Returns:
            List of detected workflows with metadata
        """
        workflows = []

        # Strategy 1: Email response workflows
        email_workflows = self._detect_email_workflows(lookback_days)
        workflows.extend(email_workflows)

        # Strategy 2: Task completion workflows
        task_workflows = self._detect_task_workflows(lookback_days)
        workflows.extend(task_workflows)

        # Strategy 3: Calendar event workflows
        calendar_workflows = self._detect_calendar_workflows(lookback_days)
        workflows.extend(calendar_workflows)

        # Strategy 4: Sequential interaction patterns
        interaction_workflows = self._detect_interaction_workflows(lookback_days)
        workflows.extend(interaction_workflows)

        logger.info(
            f"Workflow detection complete: {len(workflows)} workflows found "
            f"({len(email_workflows)} email, {len(task_workflows)} task, "
            f"{len(calendar_workflows)} calendar, {len(interaction_workflows)} interaction)"
        )

        return workflows

    def _detect_email_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows triggered by incoming emails.

        Identifies patterns like:
        - From boss: read → draft response → send (within 2 hours)
        - From client: read → research → draft proposal → send
        - Newsletter: read → save links → archive

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of email-triggered workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Find email.received events and track what happens after them
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    json_extract(payload, '$.from_address') as sender_category,
                    COUNT(*) as receive_count
                FROM events
                WHERE type = 'email.received'
                  AND julianday(timestamp) > julianday(?)
                  AND json_extract(payload, '$.from_address') IS NOT NULL
                GROUP BY sender_category
                HAVING receive_count >= ?
                ORDER BY receive_count DESC
                LIMIT 20
            """, (cutoff.isoformat(), self.min_occurrences))

            sender_categories = cursor.fetchall()

        for sender_category, receive_count in sender_categories:
            if not sender_category:
                continue

            # For each sender category, find the sequence of actions that follow.
            # FIXED: Query now counts TOTAL occurrences of following actions across
            # all emails from this sender, not per-email (which would require 3+
            # responses to the same email, which is unrealistic).
            #
            # IMPORTANT: Use julianday() for ALL timestamp comparisons to avoid
            # timezone string comparison bugs. SQLite datetime() strips timezone
            # suffixes, causing '2026-02-01T08:00:00+00:00' > '2026-02-01 12:00:00'
            # in string comparison even though it's earlier in time.
            with self.db.get_connection("events") as conn:
                cursor = conn.cursor()
                # Find events that occur within max_step_gap_hours after email receipt
                cursor.execute("""
                    SELECT
                        e2.type,
                        COUNT(DISTINCT e2.id) as occurrence_count,
                        AVG(julianday(e2.timestamp) - julianday(e1.timestamp)) * 24 as avg_hours_after
                    FROM events e1
                    JOIN events e2 ON
                        julianday(e2.timestamp) > julianday(e1.timestamp)
                        AND julianday(e2.timestamp) < julianday(e1.timestamp) + (CAST(? AS REAL) / 24.0)
                        AND e2.type != e1.type
                    WHERE e1.type = 'email.received'
                      AND julianday(e1.timestamp) > julianday(?)
                      AND json_extract(e1.payload, '$.from_address') = ?
                      AND e2.type IN ('email.sent', 'task.created', 'calendar.event.created',
                                      'message.sent')
                    GROUP BY e2.type
                    HAVING occurrence_count >= ?
                    ORDER BY avg_hours_after ASC
                """, (self.max_step_gap_hours, cutoff.isoformat(), sender_category, self.min_occurrences))

                following_actions = cursor.fetchall()

            # A workflow needs at least min_steps total steps. Since we're adding
            # the trigger event (read_email) as the first step, we only need
            # (min_steps - 1) following actions. For min_steps=2, this means
            # a simple "receive → respond" workflow (1 following action) is valid.
            if len(following_actions) >= (self.min_steps - 1):
                # Build workflow from detected pattern
                steps = ["read_email_from_" + sender_category.replace(" ", "_").lower()]
                tools = ["email"]

                for action_type, count, avg_hours in following_actions:
                    # Extract action verb from event type
                    action = action_type.split(".")[-1]  # e.g., "sent" from "email.sent"
                    source = action_type.split(".")[0]   # e.g., "email" from "email.sent"
                    steps.append(action)
                    if source not in tools:
                        tools.append(source)

                # Calculate success rate (assume email.sent means completion)
                completion_count = sum(count for event_type, count, _ in following_actions if event_type == "email.sent")
                success_rate = min(1.0, completion_count / receive_count)

                # Calculate typical duration
                if following_actions:
                    typical_duration = sum(avg_hours for _, _, avg_hours in following_actions) * 60  # Convert to minutes
                else:
                    typical_duration = None

                if success_rate >= self.success_threshold:
                    workflow = {
                        "name": f"Responding to {sender_category}",
                        "trigger_conditions": [f"email.received.from.{sender_category}"],
                        "steps": steps,
                        "typical_duration_minutes": typical_duration,
                        "tools_used": tools,
                        "success_rate": success_rate,
                        "times_observed": receive_count,
                    }
                    workflows.append(workflow)
                    logger.debug(f"Detected email workflow for {sender_category}: {len(steps)} steps, {success_rate:.2f} success rate")

        return workflows

    def _detect_task_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows for task completion patterns.

        Identifies sequences like:
        - Task created → researched → worked → completed
        - Task assigned → delegated → tracked → verified
        - Task received → clarified → executed → reported

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of task-completion workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Find task.created events and track completion patterns
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT DATE(timestamp)) as days_with_tasks,
                    COUNT(*) as total_tasks
                FROM events
                WHERE type = 'task.created'
                  AND julianday(timestamp) > julianday(?)
            """, (cutoff.isoformat(),))

            result = cursor.fetchone()
            if not result or result[1] < self.min_occurrences:
                return workflows

            days_with_tasks, total_tasks = result

        # Find events that commonly occur between task creation and completion
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    e2.type,
                    COUNT(*) as occurrence_count,
                    AVG(julianday(e3.timestamp) - julianday(e1.timestamp)) * 24 as avg_hours_to_complete
                FROM events e1
                JOIN events e2 ON
                    julianday(e2.timestamp) > julianday(e1.timestamp)
                    AND julianday(e2.timestamp) < julianday(e1.timestamp) + (CAST(? AS REAL) / 24.0)
                LEFT JOIN events e3 ON
                    e3.type = 'task.completed'
                    AND julianday(e3.timestamp) > julianday(e1.timestamp)
                    AND json_extract(e3.payload, '$.task_id') = json_extract(e1.payload, '$.task_id')
                WHERE e1.type = 'task.created'
                  AND julianday(e1.timestamp) > julianday(?)
                  AND e2.type != 'task.created'
                  AND e2.type IN ('email.sent', 'email.received',
                                  'calendar.event.created', 'message.sent', 'task.completed')
                GROUP BY e2.type
                HAVING occurrence_count >= ?
                ORDER BY occurrence_count DESC
            """, (self.max_step_gap_hours, cutoff.isoformat(), self.min_occurrences))

            task_actions = cursor.fetchall()

        if len(task_actions) >= self.min_steps:
            steps = ["create_task"]
            tools = ["task_manager"]

            for action_type, count, avg_hours in task_actions:
                action = action_type.split(".")[-1]
                source = action_type.split(".")[0]
                steps.append(action)
                if source not in tools:
                    tools.append(source)

            # Calculate success rate (tasks that reached completion)
            completion_count = sum(count for event_type, count, _ in task_actions if event_type == "task.completed")
            success_rate = min(1.0, completion_count / total_tasks) if total_tasks > 0 else 0.0

            # Calculate typical duration
            typical_duration = task_actions[0][2] * 60 if task_actions and task_actions[0][2] else None

            if success_rate >= self.success_threshold:
                workflow = {
                    "name": "Task completion workflow",
                    "trigger_conditions": ["task.created"],
                    "steps": steps,
                    "typical_duration_minutes": typical_duration,
                    "tools_used": tools,
                    "success_rate": success_rate,
                    "times_observed": total_tasks,
                }
                workflows.append(workflow)
                logger.debug(f"Detected task workflow: {len(steps)} steps, {success_rate:.2f} success rate")

        return workflows

    def _detect_calendar_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows around calendar events.

        Identifies patterns like:
        - Meeting scheduled → prep → attend → follow-up email
        - Event created → invite sent → reminder → attended
        - Recurring meeting → review agenda → attend → update notes

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of calendar-based workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Find calendar.event.created and track preparation/follow-up patterns
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as event_count
                FROM events
                WHERE type = 'calendar.event.created'
                  AND julianday(timestamp) > julianday(?)
            """, (cutoff.isoformat(),))

            result = cursor.fetchone()
            if not result or result[0] < self.min_occurrences:
                return workflows

            event_count = result[0]

        # Find events before and after calendar events
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            # Look for preparation activities (before event)
            cursor.execute("""
                SELECT
                    e2.type,
                    COUNT(*) as occurrence_count,
                    'before' as timing
                FROM events e1
                JOIN events e2 ON
                    julianday(e2.timestamp) < julianday(e1.timestamp)
                    AND julianday(e2.timestamp) > julianday(e1.timestamp) - (CAST(? AS REAL) / 24.0)
                WHERE e1.type = 'calendar.event.created'
                  AND julianday(e1.timestamp) > julianday(?)
                  AND e2.type IN ('email.received', 'task.created')
                GROUP BY e2.type
                HAVING occurrence_count >= ?
                UNION ALL
                SELECT
                    e3.type,
                    COUNT(*) as occurrence_count,
                    'after' as timing
                FROM events e1
                JOIN events e3 ON
                    julianday(e3.timestamp) > julianday(e1.timestamp)
                    AND julianday(e3.timestamp) < julianday(e1.timestamp) + (CAST(? AS REAL) / 24.0)
                WHERE e1.type = 'calendar.event.created'
                  AND julianday(e1.timestamp) > julianday(?)
                  AND e3.type IN ('email.sent', 'task.created', 'message.sent')
                GROUP BY e3.type
                HAVING occurrence_count >= ?
                ORDER BY timing DESC, occurrence_count DESC
            """, (
                self.max_step_gap_hours, cutoff.isoformat(), self.min_occurrences,
                self.max_step_gap_hours, cutoff.isoformat(), self.min_occurrences
            ))

            calendar_actions = cursor.fetchall()

        if len(calendar_actions) >= self.min_steps:
            steps = []
            tools = ["calendar"]

            # Sort by timing (before → during → after)
            for action_type, count, timing in calendar_actions:
                action = action_type.split(".")[-1]
                source = action_type.split(".")[0]
                if timing == "before":
                    steps.insert(0, f"prep_{action}")
                else:
                    steps.append(f"followup_{action}")
                if source not in tools:
                    tools.append(source)

            # Add the calendar event itself in the middle
            steps.insert(len([s for s in steps if s.startswith("prep_")]), "attend_event")

            # Estimate success rate (if follow-up actions occurred)
            followup_count = sum(count for _, count, timing in calendar_actions if timing == "after")
            success_rate = min(1.0, followup_count / event_count) if event_count > 0 else 0.0

            if success_rate >= self.success_threshold:
                workflow = {
                    "name": "Calendar event workflow",
                    "trigger_conditions": ["calendar.event.created"],
                    "steps": steps,
                    "typical_duration_minutes": None,  # Spans event duration
                    "tools_used": tools,
                    "success_rate": success_rate,
                    "times_observed": event_count,
                }
                workflows.append(workflow)
                logger.debug(f"Detected calendar workflow: {len(steps)} steps, {success_rate:.2f} success rate")

        return workflows

    def _detect_interaction_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows from episodic interaction sequences.

        Uses the episodes table to find recurring multi-step interaction patterns
        that don't fit into email/task/calendar categories.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of interaction-based workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Find common interaction sequences from episodes
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            # Find pairs of interaction types that frequently occur together
            cursor.execute("""
                SELECT
                    e1.interaction_type as first_action,
                    e2.interaction_type as second_action,
                    COUNT(*) as sequence_count
                FROM episodes e1
                JOIN episodes e2 ON
                    julianday(e2.timestamp) > julianday(e1.timestamp)
                    AND julianday(e2.timestamp) < julianday(e1.timestamp) + (CAST(? AS REAL) / 24.0)
                    AND e1.interaction_type != e2.interaction_type
                WHERE julianday(e1.timestamp) > julianday(?)
                  AND e1.interaction_type IS NOT NULL
                  AND e2.interaction_type IS NOT NULL
                GROUP BY e1.interaction_type, e2.interaction_type
                HAVING sequence_count >= ?
                ORDER BY sequence_count DESC
                LIMIT 10
            """, (self.max_step_gap_hours, cutoff.isoformat(), self.min_occurrences))

            interaction_pairs = cursor.fetchall()

        # Group by first action to find multi-step sequences
        workflow_sequences = defaultdict(list)
        for first, second, count in interaction_pairs:
            workflow_sequences[first].append((second, count))

        for first_action, following_actions in workflow_sequences.items():
            if len(following_actions) >= self.min_steps:
                # Build workflow from interaction sequence
                steps = [first_action] + [action for action, _ in following_actions]
                total_observations = sum(count for _, count in following_actions)
                avg_observations = total_observations / len(following_actions)

                # Estimate success rate (assume if all steps occurred, it succeeded)
                success_rate = min(1.0, following_actions[-1][1] / following_actions[0][1]) if following_actions else 0.5

                if success_rate >= self.success_threshold:
                    workflow = {
                        "name": f"{first_action.replace('_', ' ').title()} workflow",
                        "trigger_conditions": [first_action],
                        "steps": steps,
                        "typical_duration_minutes": None,
                        "tools_used": [],  # Would need to extract from event metadata
                        "success_rate": success_rate,
                        "times_observed": int(avg_observations),
                    }
                    workflows.append(workflow)
                    logger.debug(f"Detected interaction workflow: {first_action} → {len(steps)} steps")

        return workflows

    def store_workflows(self, workflows: list[dict[str, Any]]) -> int:
        """Persist detected workflows to the database.

        Uses UPSERT logic: if a workflow with the same name already exists,
        updates its statistics (success_rate, times_observed, tools_used).

        Args:
            workflows: List of workflow dictionaries to store

        Returns:
            Number of workflows stored
        """
        stored_count = 0
        for workflow in workflows:
            try:
                self.user_model_store.store_workflow(workflow)
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store workflow '{workflow.get('name')}': {e}")

        logger.info(f"Stored {stored_count}/{len(workflows)} workflows")
        return stored_count
