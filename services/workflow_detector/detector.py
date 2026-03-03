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

import json
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

        PERFORMANCE FIX (v3): Replaced O(n×m) range JOINs with O(n) sliding window
        algorithm. Instead of joining every trigger event with every response event,
        we now:
        1. Fetch events in chronological order (single table scan)
        2. Maintain a sliding window of recent trigger events
        3. Match response events to triggers as we scan forward
        4. Expire old triggers outside the time window

        This reduces 800K×800K = 640B comparisons to a single pass over 800K events,
        completing in <1s instead of 30s+.

        Args:
            lookback_days: How many days of history to analyze

        Returns:
            List of detected workflows (email, task, calendar, interaction-based)
        """
        workflows = []

        # Each strategy is wrapped in try/except so that a failure in one
        # (e.g. corrupted user_model.db) does not prevent the others from
        # running.  This follows the same fail-open pattern used by the
        # RoutineDetector and InsightEngine correlators.
        email_workflows = []
        try:
            email_workflows = self._detect_email_workflows(lookback_days)
            workflows.extend(email_workflows)
        except Exception:
            logger.exception("WorkflowDetector: email workflow detection failed")

        task_workflows = []
        try:
            task_workflows = self._detect_task_workflows(lookback_days)
            workflows.extend(task_workflows)
        except Exception:
            logger.exception("WorkflowDetector: task workflow detection failed")

        calendar_workflows = []
        try:
            calendar_workflows = self._detect_calendar_workflows(lookback_days)
            workflows.extend(calendar_workflows)
        except Exception:
            logger.exception("WorkflowDetector: calendar workflow detection failed")

        interaction_workflows = []
        try:
            interaction_workflows = self._detect_interaction_workflows(lookback_days)
            workflows.extend(interaction_workflows)
        except Exception:
            logger.exception("WorkflowDetector: interaction workflow detection failed")

        logger.info(
            f"Detected {len(workflows)} workflows from {lookback_days} days of history "
            f"({len(email_workflows)} email, {len(task_workflows)} task, "
            f"{len(calendar_workflows)} calendar, {len(interaction_workflows)} interaction)"
        )
        return workflows

    def _detect_email_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows triggered by incoming emails using sliding window algorithm.

        Identifies patterns like:
        - From boss: read → draft response → send (within 2 hours)
        - From client: read → research → draft proposal → send
        - Newsletter: read → save links → archive

        SLIDING WINDOW ALGORITHM (O(n) instead of O(n×m)):
        1. Fetch all relevant events in timestamp order (single scan)
        2. Maintain a dict of "active" received emails by sender (sliding window)
        3. As we encounter response events, match them to recent received emails
        4. Expire old received emails outside the time window
        5. Aggregate statistics as we go

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of email-triggered workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        max_gap = timedelta(hours=self.max_step_gap_hours)

        # Statistics: sender → {receive_count, following_actions → {action_type → [delays]}}
        sender_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
            'receive_count': 0,
            'following_actions': defaultdict(list)
        })

        # Sliding window: sender → [(timestamp, email_id), ...]
        # Tracks recent received emails that might trigger responses
        active_received: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

        # Fetch all relevant events in chronological order (single scan, O(n))
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, timestamp, email_from, email_to
                FROM events
                WHERE julianday(timestamp) > julianday(?)
                  AND type IN ('email.received', 'email.sent', 'task.created',
                               'calendar.event.created', 'message.sent')
                  AND (
                      (type = 'email.received' AND email_from IS NOT NULL)
                      OR type != 'email.received'
                  )
                ORDER BY timestamp ASC
            """, (cutoff.isoformat(),))

            # Process events in chronological order
            for event_id, event_type, timestamp_str, email_from, email_to in cursor:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                if event_type == 'email.received':
                    # Track this received email in the sliding window
                    sender_stats[email_from]['receive_count'] += 1
                    active_received[email_from].append((timestamp, event_id))

                else:
                    # This is a potential response event (sent, task, calendar, message)
                    # Find all active received emails that this could be responding to

                    # Expire old received emails outside the time window
                    expired_senders = []
                    for sender, received_list in active_received.items():
                        # Remove emails older than max_gap from this response
                        active_received[sender] = [
                            (recv_ts, recv_id) for recv_ts, recv_id in received_list
                            if timestamp - recv_ts <= max_gap
                        ]
                        if not active_received[sender]:
                            expired_senders.append(sender)

                    for sender in expired_senders:
                        del active_received[sender]

                    # Match this response to active received emails
                    for sender, received_list in active_received.items():
                        if received_list:
                            # Check if this response is relevant to this sender
                            is_match = False

                            if event_type == 'email.sent':
                                # For sent emails, check if recipient matches sender
                                if email_to and sender in email_to:
                                    is_match = True
                            else:
                                # For task/calendar/message, allow without recipient check
                                # (these are often created in response to any email)
                                is_match = True

                            if is_match:
                                # Calculate average delay from all active received emails
                                for recv_ts, recv_id in received_list:
                                    delay_hours = (timestamp - recv_ts).total_seconds() / 3600
                                    sender_stats[sender]['following_actions'][event_type].append(delay_hours)

        # Build workflows from aggregated statistics
        for sender, stats in sender_stats.items():
            receive_count = stats['receive_count']
            if receive_count < self.min_occurrences:
                continue

            # Aggregate following actions: count occurrences and average delay
            following_actions = []
            for action_type, delays in stats['following_actions'].items():
                count = len(delays)
                if count >= self.min_occurrences:
                    avg_hours = sum(delays) / count
                    following_actions.append((action_type, count, avg_hours))

            # Sort by average timing (actions that happen sooner are earlier in workflow)
            following_actions.sort(key=lambda x: x[2])

            if len(following_actions) >= (self.min_steps - 1):
                # Build workflow from detected pattern
                steps = ["read_email_from_" + sender.replace(" ", "_").replace("@", "_at_")]
                tools = ["email"]

                for action_type, count, avg_hours in following_actions:
                    action = action_type.split(".")[-1]
                    source = action_type.split(".")[0]
                    steps.append(action)
                    if source not in tools:
                        tools.append(source)

                # Calculate success rate (assume email.sent means completion)
                completion_count = sum(count for event_type, count, _ in following_actions if event_type == "email.sent")
                success_rate = min(1.0, completion_count / receive_count)

                # Calculate typical duration
                if following_actions:
                    typical_duration = sum(avg_hours for _, _, avg_hours in following_actions) * 60
                else:
                    typical_duration = None

                if success_rate >= self.success_threshold:
                    workflow = {
                        "name": f"Responding to {sender}",
                        "trigger_conditions": [f"email.received.from.{sender}"],
                        "steps": steps,
                        "typical_duration_minutes": typical_duration,
                        "tools_used": tools,
                        "success_rate": success_rate,
                        "times_observed": receive_count,
                    }
                    workflows.append(workflow)
                    logger.debug(f"Detected email workflow for {sender}: {len(steps)} steps, {success_rate:.2f} success rate")

        # Cap at top 20 senders by email volume to keep workflow storage manageable
        # and focus on the most significant communication patterns.  On systems
        # with hundreds of unique senders (mailing lists, newsletters, etc.) an
        # unlimited result set wastes storage and dilutes signal.
        workflows.sort(key=lambda w: w["times_observed"], reverse=True)
        workflows = workflows[:20]

        return workflows

    def _detect_task_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows for task completion patterns using sliding window.

        Identifies sequences like:
        - Task created → researched → worked → completed
        - Task assigned → delegated → tracked → verified
        - Task received → clarified → executed → reported

        SLIDING WINDOW ALGORITHM:
        Processes events chronologically, tracking task.created events and
        matching subsequent events that occur within max_gap_hours.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of task-completion workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        max_gap = timedelta(hours=self.max_step_gap_hours)

        # Statistics: {following_actions → {action_type → [delays]}}
        task_stats = {
            'total_tasks': 0,
            'following_actions': defaultdict(list)
        }

        # Sliding window: list of (timestamp, task_id) for recently created tasks
        active_tasks: list[tuple[datetime, str]] = []

        # Fetch all relevant events in chronological order
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, timestamp, task_id
                FROM events
                WHERE julianday(timestamp) > julianday(?)
                  AND type IN ('task.created', 'email.sent', 'email.received',
                               'calendar.event.created', 'message.sent', 'task.completed')
                ORDER BY timestamp ASC
            """, (cutoff.isoformat(),))

            for event_id, event_type, timestamp_str, task_id in cursor:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                if event_type == 'task.created':
                    # Track this task in the sliding window
                    task_stats['total_tasks'] += 1
                    active_tasks.append((timestamp, task_id or event_id))

                else:
                    # This is a potential follow-up action
                    # Expire old tasks outside the time window
                    active_tasks = [
                        (task_ts, t_id) for task_ts, t_id in active_tasks
                        if timestamp - task_ts <= max_gap
                    ]

                    # Match this event to active tasks
                    if active_tasks:
                        # Calculate average delay from all active tasks
                        for task_ts, t_id in active_tasks:
                            delay_hours = (timestamp - task_ts).total_seconds() / 3600
                            task_stats['following_actions'][event_type].append(delay_hours)

        if task_stats['total_tasks'] < self.min_occurrences:
            return workflows

        # Aggregate following actions
        task_actions = []
        for action_type, delays in task_stats['following_actions'].items():
            count = len(delays)
            if count >= self.min_occurrences:
                avg_hours = sum(delays) / count
                task_actions.append((action_type, count, avg_hours))

        # Sort by occurrence count (most common actions first)
        task_actions.sort(key=lambda x: x[1], reverse=True)

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
            success_rate = min(1.0, completion_count / task_stats['total_tasks'])

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
                    "times_observed": task_stats['total_tasks'],
                }
                workflows.append(workflow)
                logger.debug(f"Detected task workflow: {len(steps)} steps, {success_rate:.2f} success rate")

        return workflows

    def _detect_calendar_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows around calendar events using sliding window.

        Identifies patterns like:
        - Meeting scheduled → prep → attend → follow-up email
        - Event created → invite sent → reminder → attended
        - Recurring meeting → review agenda → attend → update notes

        SLIDING WINDOW ALGORITHM:
        Processes events chronologically, tracking calendar.event.created and
        matching events before (prep) and after (follow-up) within time window.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of calendar-based workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        max_gap = timedelta(hours=self.max_step_gap_hours)

        # Statistics: {event_count, prep_actions, followup_actions}
        calendar_stats = {
            'event_count': 0,
            'prep_actions': defaultdict(list),   # action_type → [delays before event]
            'followup_actions': defaultdict(list)  # action_type → [delays after event]
        }

        # Two sliding windows:
        # - Upcoming calendar events (for detecting prep activities before them)
        # - Recent calendar events (for detecting follow-up activities after them)
        upcoming_events: list[tuple[datetime, str]] = []
        recent_events: list[tuple[datetime, str]] = []

        # Fetch all relevant events in chronological order
        with self.db.get_connection("events") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, type, timestamp, calendar_event_id
                FROM events
                WHERE julianday(timestamp) > julianday(?)
                  AND type IN ('calendar.event.created', 'email.received', 'email.sent',
                               'task.created', 'message.sent')
                ORDER BY timestamp ASC
            """, (cutoff.isoformat(),))

            for event_id, event_type, timestamp_str, calendar_event_id in cursor:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                if event_type == 'calendar.event.created':
                    # This is a calendar event - track it in both windows
                    calendar_stats['event_count'] += 1
                    event_key = (timestamp, calendar_event_id or event_id)
                    upcoming_events.append(event_key)
                    recent_events.append(event_key)

                    # Expire old upcoming events (no longer relevant for prep detection)
                    upcoming_events = [
                        (evt_ts, evt_id) for evt_ts, evt_id in upcoming_events
                        if evt_ts - timestamp <= max_gap
                    ]

                else:
                    # This could be prep (before upcoming events) or follow-up (after recent events)

                    # Check if this is prep for any upcoming calendar events
                    prep_matches = [
                        (evt_ts, evt_id) for evt_ts, evt_id in upcoming_events
                        if evt_ts > timestamp and evt_ts - timestamp <= max_gap
                    ]
                    if prep_matches and event_type in ('email.received', 'task.created'):
                        for evt_ts, evt_id in prep_matches:
                            delay_hours = (evt_ts - timestamp).total_seconds() / 3600
                            calendar_stats['prep_actions'][event_type].append(delay_hours)

                    # Check if this is follow-up to any recent calendar events
                    followup_matches = [
                        (evt_ts, evt_id) for evt_ts, evt_id in recent_events
                        if timestamp > evt_ts and timestamp - evt_ts <= max_gap
                    ]
                    if followup_matches and event_type in ('email.sent', 'task.created', 'message.sent'):
                        for evt_ts, evt_id in followup_matches:
                            delay_hours = (timestamp - evt_ts).total_seconds() / 3600
                            calendar_stats['followup_actions'][event_type].append(delay_hours)

                    # Expire old recent events (no longer relevant for follow-up detection)
                    recent_events = [
                        (evt_ts, evt_id) for evt_ts, evt_id in recent_events
                        if timestamp - evt_ts <= max_gap
                    ]

        if calendar_stats['event_count'] < self.min_occurrences:
            return workflows

        # Aggregate calendar actions
        calendar_actions = []
        for action_type, delays in calendar_stats['prep_actions'].items():
            count = len(delays)
            if count >= self.min_occurrences:
                calendar_actions.append((action_type, count, 'before'))

        for action_type, delays in calendar_stats['followup_actions'].items():
            count = len(delays)
            if count >= self.min_occurrences:
                calendar_actions.append((action_type, count, 'after'))

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
            success_rate = min(1.0, followup_count / calendar_stats['event_count'])

            if success_rate >= self.success_threshold:
                workflow = {
                    "name": "Calendar event workflow",
                    "trigger_conditions": ["calendar.event.created"],
                    "steps": steps,
                    "typical_duration_minutes": None,  # Spans event duration
                    "tools_used": tools,
                    "success_rate": success_rate,
                    "times_observed": calendar_stats['event_count'],
                }
                workflows.append(workflow)
                logger.debug(f"Detected calendar workflow: {len(steps)} steps, {success_rate:.2f} success rate")

        return workflows

    def _detect_interaction_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows from episodic interaction sequences using sliding window.

        Uses the episodes table to find recurring multi-step interaction patterns
        that don't fit into email/task/calendar categories.

        SLIDING WINDOW ALGORITHM:
        Processes episodes chronologically, tracking interaction types and
        matching sequences that occur within max_gap_hours.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of interaction-based workflows
        """
        workflows = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        max_gap = timedelta(hours=self.max_step_gap_hours)

        # Statistics: interaction_type → {following_type → [delays]}
        interaction_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # Sliding window: interaction_type → [(timestamp, episode_id), ...]
        active_interactions: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

        # Fetch all episodes in chronological order.
        # Wrapped in try/except so a corrupted user_model.db returns an empty
        # list instead of crashing the entire workflow detection pipeline.
        try:
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, interaction_type, timestamp
                    FROM episodes
                    WHERE julianday(timestamp) > julianday(?)
                      AND interaction_type IS NOT NULL
                    ORDER BY timestamp ASC
                """, (cutoff.isoformat(),))

                for episode_id, interaction_type, timestamp_str in cursor:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                    # Track this interaction in the sliding window
                    active_interactions[interaction_type].append((timestamp, episode_id))

                    # Check if this interaction follows any other active interactions
                    expired_types = []
                    for prev_type, prev_list in active_interactions.items():
                        if prev_type == interaction_type:
                            continue  # Skip same-type sequences

                        # Remove old interactions outside the time window
                        active_interactions[prev_type] = [
                            (prev_ts, prev_id) for prev_ts, prev_id in prev_list
                            if timestamp - prev_ts <= max_gap
                        ]

                        if not active_interactions[prev_type]:
                            expired_types.append(prev_type)
                        else:
                            # This interaction follows the previous type
                            for prev_ts, prev_id in active_interactions[prev_type]:
                                delay_hours = (timestamp - prev_ts).total_seconds() / 3600
                                interaction_stats[prev_type][interaction_type].append(delay_hours)

                    for prev_type in expired_types:
                        del active_interactions[prev_type]
        except Exception:
            logger.warning("WorkflowDetector: user_model.db unavailable for interaction workflows")
            return []

        # Build workflows from interaction sequences
        for first_action, following_types in interaction_stats.items():
            # Aggregate following actions
            following_actions = []
            for second_action, delays in following_types.items():
                count = len(delays)
                if count >= self.min_occurrences:
                    following_actions.append((second_action, count))

            # Sort by occurrence count
            following_actions.sort(key=lambda x: x[1], reverse=True)

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
