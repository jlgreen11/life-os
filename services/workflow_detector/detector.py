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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

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

    # Prefixes that identify internal telemetry episode types (not real user
    # activity).  These dominate episode counts and create false workflow
    # patterns if not excluded from detection queries.
    INTERNAL_TYPE_PREFIXES = ("usermodel_", "system_", "test")

    # SQL WHERE clause fragment to exclude internal telemetry types from
    # workflow detection queries.  Append after existing WHERE conditions.
    INTERNAL_TYPE_SQL_FILTER = (
        "AND interaction_type NOT LIKE 'usermodel_%' "
        "AND interaction_type NOT LIKE 'system_%' "
        "AND interaction_type NOT LIKE 'test%'"
    )

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
        self.max_step_gap_hours = 12  # Steps within 12h can be part of same workflow (covers workday spans)
        self.min_steps = 2  # Workflows must have at least 2 distinct steps
        # Require at least 3 actual completions to identify a workflow, regardless
        # of overall rate.  This handles the email asymmetry where most received
        # emails are automated/marketing and don't need replies.  A user who has
        # replied to their boss 15 times but received 5000 total emails has a 0.3%
        # rate, but those 15 replies clearly represent a real workflow.
        self.min_completions = 2  # Absolute completion count minimum to store workflow

    def _backfill_stale_interaction_types(self, lookback_days: int) -> None:
        """Batch-update episodes with stale interaction_type values.

        Episodes created before granular interaction_type classification was
        deployed have ``interaction_type`` set to NULL, 'unknown', or the
        generic 'communication'.  These get filtered out by the interaction
        workflow detection query (``AND interaction_type IS NOT NULL``),
        leaving 0 usable episodes even when thousands exist.

        This method runs a one-time batch backfill at the start of
        ``detect_workflows()`` by:
        1. Querying stale episode ``event_id`` values from user_model.db
        2. Looking up corresponding event types from events.db
        3. Converting dotted event types to underscored interaction types
           (e.g. 'email.received' -> 'email_received')
        4. Updating the episodes table in user_model.db

        The two databases cannot be JOINed in SQLite, so the work is done
        in three separate queries with chunked IN-clauses (SQLite variable
        limit ~999).

        Args:
            lookback_days: Only backfill episodes within this lookback window.
        """
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Step 1: Find stale episodes in user_model.db
        try:
            with self.db.get_connection("user_model") as conn:
                stale_rows = conn.execute(
                    """SELECT id, event_id FROM episodes
                       WHERE timestamp > ?
                         AND (interaction_type IS NULL
                              OR interaction_type IN ('unknown', 'communication'))
                         AND event_id IS NOT NULL""",
                    (cutoff.isoformat(),),
                ).fetchall()
        except Exception:
            logger.warning("Backfill: failed to query stale episodes — skipping")
            return

        if not stale_rows:
            return

        logger.info("Backfill: %d episodes with stale interaction_type in lookback window", len(stale_rows))

        # Build mapping: event_id -> episode_id(s)
        event_id_to_episode_ids: dict[str, list[str]] = defaultdict(list)
        for ep_id, ev_id in stale_rows:
            event_id_to_episode_ids[ev_id].append(ep_id)

        all_event_ids = list(event_id_to_episode_ids.keys())

        # Step 2: Look up event types from events.db in chunks
        event_type_map: dict[str, str] = {}  # event_id -> event.type
        chunk_size = 900  # Stay under SQLite's ~999 variable limit
        try:
            with self.db.get_connection("events") as conn:
                for i in range(0, len(all_event_ids), chunk_size):
                    chunk = all_event_ids[i : i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT id, type FROM events WHERE id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for ev_id, ev_type in rows:
                        if ev_type:
                            event_type_map[ev_id] = ev_type
        except Exception:
            logger.warning("Backfill: failed to query events.db — skipping")
            return

        if not event_type_map:
            logger.info("Backfill: no matching events found in events.db — skipping")
            return

        # Step 3: Update episodes in user_model.db
        updates: list[tuple[str, str]] = []  # (interaction_type, episode_id)
        for ev_id, ev_type in event_type_map.items():
            derived_type = ev_type.replace(".", "_")
            for ep_id in event_id_to_episode_ids[ev_id]:
                updates.append((derived_type, ep_id))

        try:
            with self.db.get_connection("user_model") as conn:
                for i in range(0, len(updates), chunk_size):
                    chunk = updates[i : i + chunk_size]
                    conn.executemany(
                        "UPDATE episodes SET interaction_type = ? WHERE id = ?",
                        chunk,
                    )
                conn.commit()
        except Exception:
            logger.warning("Backfill: failed to update episodes — skipping")
            return

        logger.info("Backfill: updated %d episodes with derived interaction_type values", len(updates))

    # Event types that indicate real user activity — used by adaptive lookback
    # to decide whether the default window has sufficient data for detection.
    QUALIFYING_EVENT_TYPES: tuple[str, ...] = (
        "email.received",
        "email.sent",
        "message.sent",
        "message.received",
        "task.created",
        "task.completed",
        "calendar.event.created",
    )

    # Adaptive lookback fires when fewer than this many qualifying events are
    # found in the default window.  50 is the same floor used by the routine
    # detector's min_episodes_for_detection and provides a reasonable baseline
    # for multi-step workflow pattern recognition.
    MIN_QUALIFYING_EVENTS = 50

    # Hard ceiling on how far back adaptive lookback will ever extend.
    # 365 days covers a full year of behavioral data while keeping query
    # costs bounded on systems with millions of historical events.
    MAX_ADAPTIVE_LOOKBACK_DAYS = 365

    def _compute_adaptive_lookback_days(self, lookback_days: int) -> int:
        """Extend the lookback window when qualifying events are sparse.

        When a connector outage (or any data gap) moves all qualifying events
        outside the default window, ``detect_workflows()`` would return 0
        results despite rich historical data existing in events.db.  This
        method detects that scenario by:

        1. Counting qualifying events (email, task, calendar, message) within
           the current ``lookback_days`` window.
        2. If count < MIN_QUALIFYING_EVENTS, doubling the window and repeating.
        3. Stopping once count >= MIN_QUALIFYING_EVENTS or the window hits
           MAX_ADAPTIVE_LOOKBACK_DAYS (365 days).
        4. Logging the extension so operators can see when it fires.

        The extension only fires when the window is sparse (count < 50) —
        having even a small number of recent events means the system is still
        receiving data normally; in that case, no extension is applied.

        Args:
            lookback_days: The requested lookback window in days.

        Returns:
            Effective lookback window in days (>= lookback_days,
            <= MAX_ADAPTIVE_LOOKBACK_DAYS).
        """
        original_lookback = lookback_days
        effective_lookback = lookback_days
        count = 0  # Track count for the final log message

        try:
            placeholders = ",".join("?" * len(self.QUALIFYING_EVENT_TYPES))

            while True:
                cutoff = (datetime.now(UTC) - timedelta(days=effective_lookback)).isoformat()
                with self.db.get_connection("events") as conn:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM events "
                        f"WHERE julianday(timestamp) > julianday(?) "
                        f"AND type IN ({placeholders})",
                        (cutoff, *self.QUALIFYING_EVENT_TYPES),
                    ).fetchone()
                count = row[0] if row else 0

                if count >= self.MIN_QUALIFYING_EVENTS:
                    # Sufficient events found — no further extension needed.
                    break

                if effective_lookback >= self.MAX_ADAPTIVE_LOOKBACK_DAYS:
                    # Already at the hard ceiling — cannot extend further.
                    break

                # Double the window (capped at the maximum).
                effective_lookback = min(effective_lookback * 2, self.MAX_ADAPTIVE_LOOKBACK_DAYS)

            if effective_lookback != original_lookback:
                logger.info(
                    "WorkflowDetector: extended lookback from %dd to %dd (%d qualifying events)",
                    original_lookback,
                    effective_lookback,
                    count,
                )

            return effective_lookback

        except Exception:
            logger.warning(
                "WorkflowDetector: adaptive lookback computation failed — falling back to %dd",
                lookback_days,
            )
            return lookback_days

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
        # Adaptive lookback: when the default window contains fewer than
        # MIN_QUALIFYING_EVENTS qualifying events (because a connector outage
        # pushed all data outside the window), double the window repeatedly up
        # to MAX_ADAPTIVE_LOOKBACK_DAYS.  This mirrors the same pattern used
        # by RoutineDetector._compute_adaptive_lookback_days().
        try:
            effective_lookback_days = self._compute_adaptive_lookback_days(lookback_days)
        except Exception:
            logger.warning(
                "WorkflowDetector: adaptive lookback raised unexpectedly — using default %dd",
                lookback_days,
            )
            effective_lookback_days = lookback_days

        # Backfill stale interaction_type values before any detection strategy
        # runs, so interaction workflows benefit from properly classified episodes.
        try:
            self._backfill_stale_interaction_types(effective_lookback_days)
        except Exception:
            logger.exception("Interaction type backfill failed — continuing with detection")

        workflows = []

        # Each strategy is wrapped in try/except so that a failure in one
        # (e.g. corrupted user_model.db) does not prevent the others from
        # running.  This follows the same fail-open pattern used by the
        # RoutineDetector and InsightEngine correlators.
        email_workflows = []
        try:
            email_workflows = self._detect_email_workflows(effective_lookback_days)
            workflows.extend(email_workflows)
        except Exception:
            logger.exception("WorkflowDetector: email workflow detection failed")

        task_workflows = []
        try:
            task_workflows = self._detect_task_workflows(effective_lookback_days)
            workflows.extend(task_workflows)
        except Exception:
            logger.exception("WorkflowDetector: task workflow detection failed")

        calendar_workflows = []
        try:
            calendar_workflows = self._detect_calendar_workflows(effective_lookback_days)
            workflows.extend(calendar_workflows)
        except Exception:
            logger.exception("WorkflowDetector: calendar workflow detection failed")

        interaction_workflows = []
        try:
            interaction_workflows = self._detect_interaction_workflows(effective_lookback_days)
            workflows.extend(interaction_workflows)
        except Exception:
            logger.exception("WorkflowDetector: interaction workflow detection failed")

        recurring_inbound_workflows = []
        try:
            recurring_inbound_workflows = self._detect_recurring_inbound_patterns(effective_lookback_days)
            workflows.extend(recurring_inbound_workflows)
        except Exception:
            logger.exception("WorkflowDetector: recurring inbound detection failed")

        logger.info(
            f"Detected {len(workflows)} workflows from {effective_lookback_days} days of history "
            f"({len(email_workflows)} email, {len(task_workflows)} task, "
            f"{len(calendar_workflows)} calendar, {len(interaction_workflows)} interaction, "
            f"{len(recurring_inbound_workflows)} recurring_inbound)"
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
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
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
                                # For sent emails, check if recipient matches sender.
                                # email_to is populated from json_extract(payload, '$.to_addresses')
                                # which stores a JSON array string like '["user@example.com"]'.
                                # Parse it properly and do case-insensitive exact matching.
                                try:
                                    if email_to and email_to.startswith('['):
                                        recipients = json.loads(email_to)
                                    elif email_to:
                                        recipients = [email_to]
                                    else:
                                        recipients = []
                                except (json.JSONDecodeError, TypeError):
                                    recipients = [email_to] if email_to else []

                                sender_lower = sender.lower()
                                if any(r and r.lower() == sender_lower for r in recipients):
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

                if completion_count >= self.min_completions:
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
                    logger.debug(f"Detected email workflow for {sender}: {len(steps)} steps, {completion_count} completions")
                else:
                    # Log why this sender was rejected for debugging workflow detection issues
                    if not following_actions:
                        logger.debug("Workflow candidate '%s' rejected: %d receives but 0 following actions", sender, receive_count)
                    elif completion_count < self.min_completions:
                        logger.debug(
                            "Workflow candidate '%s' rejected: %d receives, %d completions < min %d",
                            sender, receive_count, completion_count, self.min_completions,
                        )

        # Cap at top 20 senders by email volume to keep workflow storage manageable
        # and focus on the most significant communication patterns.  On systems
        # with hundreds of unique senders (mailing lists, newsletters, etc.) an
        # unlimited result set wastes storage and dilutes signal.
        workflows.sort(key=lambda w: w["times_observed"], reverse=True)
        workflows = workflows[:20]

        return workflows

    def _detect_recurring_inbound_patterns(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect recurring inbound email patterns without requiring replies.

        Unlike _detect_email_workflows which requires bidirectional communication
        (receive + reply), this strategy identifies recurring emails from the same
        sender that arrive on a predictable schedule. This is critical for systems
        where inbound email vastly outnumbers outbound (e.g. 12,429 received vs
        10 sent), which would otherwise produce 0 email workflows.

        Uses a dynamic volume threshold that scales with total email count to
        avoid incorrectly filtering legitimate high-volume senders (colleagues,
        daily reports) in email-heavy environments.  Also excludes senders whose
        events are majority-tagged as 'marketing' or 'system:suppressed'.

        Detection criteria:
        1. Group email.received events by sender (email_from)
        2. For senders with >= min_occurrences emails, check temporal regularity
        3. Filter out senders tagged as marketing via event_tags
        4. Filter out extreme-volume senders using a dynamic threshold that
           scales with total email volume (prevents excluding legitimate
           high-traffic senders in email-heavy environments)
        5. Detect cadence: daily (same hour ±2h, including weekday-only),
           weekly (same day of week), or consistent interval (within 20% variance)

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of recurring inbound email workflows
        """
        workflows = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Query all received emails grouped by sender within the lookback window
        sender_timestamps: dict[str, list[datetime]] = defaultdict(list)
        # Also track event IDs per sender for marketing tag lookup
        sender_event_ids: dict[str, list[str]] = defaultdict(list)

        with self.db.get_connection("events") as conn:
            cursor = conn.execute(
                """
                SELECT id, email_from, timestamp
                FROM events
                WHERE julianday(timestamp) > julianday(?)
                  AND type = 'email.received'
                  AND email_from IS NOT NULL
                ORDER BY timestamp ASC
                """,
                (cutoff.isoformat(),),
            )
            for event_id, email_from, timestamp_str in cursor:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                sender_timestamps[email_from].append(ts)
                sender_event_ids[email_from].append(event_id)

            # Build set of event IDs tagged as marketing/suppressed
            marketing_event_ids: set[str] = set()
            try:
                tag_cursor = conn.execute(
                    """
                    SELECT DISTINCT event_id FROM event_tags
                    WHERE tag IN ('marketing', 'system:suppressed')
                    """,
                )
                marketing_event_ids = {row[0] for row in tag_cursor}
            except Exception:
                # event_tags table may not exist in older schemas; proceed without
                pass

        # Dynamic max_volume: scales with total email count to avoid excluding
        # legitimate senders in email-heavy environments.  Floor of 200 ensures
        # small-volume mailboxes still filter spam; ceiling of 20% of unique
        # senders' average handles large-volume cases.
        total_emails = sum(len(ts) for ts in sender_timestamps.values())
        num_senders = len(sender_timestamps)
        if num_senders > 0:
            max_volume = max(200, total_emails // 5)
        else:
            max_volume = 200

        for sender, timestamps in sender_timestamps.items():
            count = len(timestamps)

            # Skip senders below minimum occurrence threshold
            if count < self.min_occurrences:
                continue

            # Skip senders where majority of events are tagged as marketing
            sender_ids = sender_event_ids.get(sender, [])
            if sender_ids and marketing_event_ids:
                tagged_count = sum(1 for eid in sender_ids if eid in marketing_event_ids)
                if tagged_count / len(sender_ids) > 0.5:
                    continue

            # Skip extreme-volume senders (dynamic threshold)
            if count > max_volume:
                continue

            cadence = self._detect_cadence(timestamps)
            if cadence is None:
                continue

            sender_label = sender.replace("@", "_at_").replace(".", "_")
            workflow = {
                "name": f"Recurring email from {sender}",
                "trigger_conditions": [f"email.received.from.{sender}"],
                "steps": ["email_received"],
                "typical_duration_minutes": None,
                "tools_used": ["email"],
                "success_rate": 1.0,  # Inbound-only; always "completes"
                "times_observed": count,
                "metadata": {
                    "cadence": cadence["type"],
                    "sender": sender,
                    "avg_interval_hours": cadence.get("avg_interval_hours"),
                },
            }
            workflows.append(workflow)
            logger.debug(
                "Detected recurring inbound pattern from %s: %s cadence, %d occurrences",
                sender,
                cadence["type"],
                count,
            )

        # Cap at top 20 senders by volume
        workflows.sort(key=lambda w: w["times_observed"], reverse=True)
        return workflows[:20]

    @staticmethod
    def _detect_cadence(timestamps: list[datetime]) -> dict[str, Any] | None:
        """Determine if a list of timestamps exhibits a regular cadence.

        Checks for four patterns in order:
        1. Daily — emails arrive at roughly the same hour (±2h), on different days
        2. Weekday-only — like daily but only Mon-Fri (avg interval ~33.6h due to
           weekend gaps)
        3. Weekly — emails arrive on the same day of week
        4. Consistent interval — intervals between emails have ≤20% coefficient of variation

        Args:
            timestamps: Sorted list of email timestamps for a single sender

        Returns:
            Dict with 'type' key ('daily', 'weekday_daily', 'weekly', 'interval')
            and details, or None if no regular pattern is found.
        """
        if len(timestamps) < 3:
            return None

        # Calculate intervals in hours between consecutive emails
        intervals_hours = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
            if delta > 0:
                intervals_hours.append(delta)

        if len(intervals_hours) < 2:
            return None

        avg_interval = sum(intervals_hours) / len(intervals_hours)

        # Check daily pattern: most emails arrive at a similar hour (±2h)
        hours = [ts.hour for ts in timestamps]
        median_hour = sorted(hours)[len(hours) // 2]
        within_window = sum(1 for h in hours if abs(h - median_hour) <= 2 or abs(h - median_hour) >= 22)
        hour_consistent = within_window / len(hours) >= 0.7

        if hour_consistent and 18 <= avg_interval <= 30:
            return {
                "type": "daily",
                "hour": median_hour,
                "avg_interval_hours": round(avg_interval, 1),
            }

        # Check weekday-only daily pattern: emails arrive Mon-Fri at a similar
        # hour, with weekend gaps pushing avg_interval to ~33.6h.  Require that
        # almost all emails land on weekdays (0=Mon .. 4=Fri).
        weekdays = [ts.weekday() for ts in timestamps]
        weekday_ratio = sum(1 for d in weekdays if d < 5) / len(weekdays)
        if hour_consistent and weekday_ratio >= 0.85 and 24 <= avg_interval <= 50:
            return {
                "type": "weekday_daily",
                "hour": median_hour,
                "avg_interval_hours": round(avg_interval, 1),
            }

        # Check weekly pattern: most emails arrive on the same day of week
        weekdays = [ts.weekday() for ts in timestamps]
        most_common_day = max(set(weekdays), key=weekdays.count)
        day_match_ratio = weekdays.count(most_common_day) / len(weekdays)
        if day_match_ratio >= 0.7 and 120 <= avg_interval <= 210:
            return {
                "type": "weekly",
                "day_of_week": most_common_day,
                "avg_interval_hours": round(avg_interval, 1),
            }

        # Check consistent interval: coefficient of variation ≤ 20%
        if avg_interval > 0:
            variance = sum((x - avg_interval) ** 2 for x in intervals_hours) / len(intervals_hours)
            std_dev = variance**0.5
            cv = std_dev / avg_interval
            if cv <= 0.20:
                return {
                    "type": "interval",
                    "avg_interval_hours": round(avg_interval, 1),
                    "cv": round(cv, 3),
                }

        return None

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
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
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

            if completion_count >= self.min_completions:
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
                logger.debug(f"Detected task workflow: {len(steps)} steps, {completion_count} completions")

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
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
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

            if followup_count >= self.min_completions:
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
                logger.debug(f"Detected calendar workflow: {len(steps)} steps, {followup_count} completions")

        return workflows

    def _detect_interaction_workflows_from_events(self, lookback_days: int) -> list[dict[str, Any]]:
        """Fallback: detect interaction workflows by querying events.db directly.

        Called by _detect_interaction_workflows() when the episodes table has 0
        qualifying rows (e.g. after a database rebuild or connector outage that
        prevented episode creation).

        Algorithm:
        1. Query email.received events grouped by sender (email_from column).
        2. Segment each sender's events into *sessions* — consecutive emails
           that arrive within max_step_gap_hours of each other.
        3. A session with >= min_steps emails counts as one multi-step instance.
        4. A sender with >= min_occurrences such instances is reported as a
           recurring email-thread communication workflow.
        5. min_completions is also applied as an absolute floor.

        The INTERNAL_TYPE_PREFIXES filter is adapted for the events.type column
        (the original INTERNAL_TYPE_SQL_FILTER targets the episodes.interaction_type
        column, so we replicate the same pattern here).

        Args:
            lookback_days: Days of history to analyze.

        Returns:
            List of interaction workflow dicts with the same schema as other
            detection strategies.
        """
        workflows: list[dict[str, Any]] = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        max_gap = timedelta(hours=self.max_step_gap_hours)

        # Events-table adaptation of INTERNAL_TYPE_SQL_FILTER.
        # INTERNAL_TYPE_SQL_FILTER references `interaction_type` (episodes column);
        # here we replicate the same prefix exclusions for `type` (events column).
        events_internal_filter = (
            "AND type NOT LIKE 'usermodel_%' "
            "AND type NOT LIKE 'system_%' "
            "AND type NOT LIKE 'test%'"
        )

        # Query qualifying email.received events from events.db.
        try:
            with self.db.get_connection("events") as conn:
                cursor = conn.execute(
                    f"""
                    SELECT id, type, timestamp, email_from
                    FROM events
                    WHERE julianday(timestamp) > julianday(?)
                      AND type = 'email.received'
                      AND email_from IS NOT NULL
                      {events_internal_filter}
                    ORDER BY email_from, timestamp ASC
                    """,
                    (cutoff.isoformat(),),
                )
                rows = cursor.fetchall()
        except Exception:
            logger.warning("WorkflowDetector: fallback events.db query failed — returning empty list")
            return []

        n_events = len(rows)
        logger.info(
            "WorkflowDetector: 0 episodes, falling back to events.db sequence analysis (%d qualifying events)",
            n_events,
        )

        if n_events == 0:
            return []

        # Group email timestamps by sender.
        sender_timestamps: dict[str, list[datetime]] = defaultdict(list)
        for _event_id, _event_type, timestamp_str, email_from in rows:
            try:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                sender_timestamps[email_from].append(ts)
            except (ValueError, AttributeError):
                continue

        # For each sender, segment their timestamps into sessions.
        # A session is a run of consecutive emails each within max_gap of the
        # previous one.  Sessions with >= min_steps emails represent a genuine
        # multi-step thread pattern.
        for sender, timestamps in sender_timestamps.items():
            timestamps.sort()

            # Segment into sessions using a single linear pass.
            sessions: list[list[datetime]] = []
            current_session: list[datetime] = [timestamps[0]]
            for i in range(1, len(timestamps)):
                if timestamps[i] - current_session[-1] <= max_gap:
                    current_session.append(timestamps[i])
                else:
                    sessions.append(current_session)
                    current_session = [timestamps[i]]
            sessions.append(current_session)

            # Keep only sessions that satisfy the min_steps threshold.
            multi_step_sessions = [s for s in sessions if len(s) >= self.min_steps]
            completion_count = len(multi_step_sessions)

            # Apply both occurrence and completion thresholds.
            if completion_count < self.min_occurrences:
                continue
            if completion_count < self.min_completions:
                continue

            # Build workflow steps from the median session depth.
            # Cap at 5 named steps to keep workflow names readable.
            max_steps_seen = max(len(s) for s in multi_step_sessions)
            steps = ["email_received"]
            for i in range(1, min(max_steps_seen, 5)):
                steps.append(f"email_received_thread_{i}")

            workflow: dict[str, Any] = {
                "name": f"Email thread from {sender}",
                "trigger_conditions": [f"email.received.from.{sender}"],
                "steps": steps,
                "typical_duration_minutes": None,
                "tools_used": ["email"],
                "success_rate": 1.0,
                "times_observed": completion_count,
            }
            workflows.append(workflow)
            logger.debug(
                "WorkflowDetector: fallback detected email thread workflow for %s: "
                "%d sessions, %d steps",
                sender,
                completion_count,
                len(steps),
            )

        # Return top 20 senders by session count.
        workflows.sort(key=lambda w: w["times_observed"], reverse=True)
        return workflows[:20]

    def _detect_interaction_workflows(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect workflows from episodic interaction sequences using sliding window.

        Uses the episodes table to find recurring multi-step interaction patterns
        that don't fit into email/task/calendar categories.

        When the episodes table contains 0 qualifying rows (e.g. after a DB
        rebuild), automatically falls back to
        _detect_interaction_workflows_from_events() which queries events.db
        directly for recurring email-thread patterns.

        SLIDING WINDOW ALGORITHM:
        Processes episodes chronologically, tracking interaction types and
        matching sequences that occur within max_gap_hours.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of interaction-based workflows
        """
        workflows = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
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
                # Count qualifying episodes first.  If there are none, fall back
                # to events.db sequence analysis so that the procedural user model
                # layer is still populated even after a database rebuild.
                count_row = conn.execute(
                    f"""
                    SELECT COUNT(*) FROM episodes
                    WHERE julianday(timestamp) > julianday(?)
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                    """,
                    (cutoff.isoformat(),),
                ).fetchone()
                episode_count = count_row[0] if count_row else 0

                if episode_count == 0:
                    # No episodes available — use the events.db fallback path.
                    return self._detect_interaction_workflows_from_events(lookback_days)

                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT id, interaction_type, timestamp
                    FROM episodes
                    WHERE julianday(timestamp) > julianday(?)
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                    ORDER BY timestamp ASC
                """, (cutoff.isoformat(),))

                for episode_id, interaction_type, timestamp_str in cursor:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Episodes may store date-only strings (e.g. "2026-02-16") which
                    # parse as naive datetimes.  Make them UTC-aware so the sliding
                    # window subtraction (timestamp - prev_ts) does not raise a
                    # TypeError when compared against the UTC-aware cutoff.
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)

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
                            # This interaction follows the previous type.
                            # Append only ONCE per B-event occurrence, using the most
                            # recent active A as the delay reference.  Appending once
                            # per active A (the old behaviour) produces combinatorial
                            # inflation in email-heavy environments where hundreds of
                            # email_received episodes can be active simultaneously --
                            # a single calendar event would then contribute thousands
                            # of entries, making times_observed meaningless.
                            most_recent_ts = active_interactions[prev_type][-1][0]
                            delay_hours = (timestamp - most_recent_ts).total_seconds() / 3600
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
                # Use the least frequent following action count as completion count
                completion_count = following_actions[-1][1] if following_actions else 0

                if completion_count >= self.min_completions:
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

    def get_diagnostics(self, lookback_days: int = 30) -> dict:
        """Return workflow detector diagnostic information for monitoring.

        Reports data availability, detection thresholds, and per-strategy results
        so operators can understand why workflows are or aren't being detected.
        Follows the same pattern as RoutineDetector.get_diagnostics() and
        PredictionEngine.get_diagnostics().

        Each section is queried independently with try/except so that a single
        DB failure doesn't prevent the rest of the diagnostics from returning.

        Args:
            lookback_days: How many days of history to analyze (default 30).

        Returns:
            Dict with keys: event_counts, thresholds, detection_results,
            total_detected, episode_interaction_types, data_sufficient.
        """
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        result: dict[str, Any] = {}

        # 1. Event counts by type in the lookback window
        try:
            with self.db.get_connection("events") as conn:
                rows = conn.execute(
                    """
                    SELECT type, COUNT(*) as cnt
                    FROM events
                    WHERE julianday(timestamp) > julianday(?)
                    GROUP BY type
                    """,
                    (cutoff.isoformat(),),
                ).fetchall()
            result["event_counts"] = {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.warning("get_diagnostics: event_counts query failed: %s", e)
            result["event_counts"] = {"error": str(e)}

        # 2. Detection thresholds
        try:
            result["thresholds"] = {
                "min_occurrences": self.min_occurrences,
                "max_step_gap_hours": self.max_step_gap_hours,
                "min_steps": self.min_steps,
                "min_completions": self.min_completions,
            }
        except Exception as e:
            logger.warning("get_diagnostics: thresholds failed: %s", e)
            result["thresholds"] = {"error": str(e)}

        # 3. Per-strategy detection results (run each strategy, count results)
        detection_results: dict[str, dict[str, int]] = {}
        total_detected = 0

        strategies = {
            "email": self._detect_email_workflows,
            "task": self._detect_task_workflows,
            "calendar": self._detect_calendar_workflows,
            "interaction": self._detect_interaction_workflows,
            "recurring_inbound": self._detect_recurring_inbound_patterns,
        }
        for name, strategy_fn in strategies.items():
            try:
                detected = strategy_fn(lookback_days)
                detection_results[name] = {"detected": len(detected)}
                total_detected += len(detected)
            except Exception as e:
                logger.warning("get_diagnostics: %s strategy failed: %s", name, e)
                detection_results[name] = {"detected": 0, "error": str(e)}

        result["detection_results"] = detection_results
        result["total_detected"] = total_detected

        # 4. Episode interaction_type distribution
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    f"""
                    SELECT interaction_type, COUNT(*) as count
                    FROM episodes
                    WHERE timestamp > datetime('now', '-' || ? || ' days')
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                    GROUP BY interaction_type
                    ORDER BY count DESC
                    """,
                    (lookback_days,),
                ).fetchall()
            type_counts = {row[0]: row[1] for row in rows}
            episode_total = sum(type_counts.values())
            type_diversity = len(type_counts)
            # At least 3 distinct types with 3+ episodes each matches min_occurrences threshold
            types_with_enough = sum(1 for c in type_counts.values() if c >= self.min_occurrences)
            result["episode_interaction_types"] = {
                "distribution": type_counts,
                "episode_total": episode_total,
                "type_diversity": type_diversity,
                "sufficient_for_interaction_workflows": types_with_enough >= 3,
            }
        except Exception as e:
            logger.warning("get_diagnostics: episode_interaction_types query failed: %s", e)
            result["episode_interaction_types"] = {
                "distribution": {},
                "episode_total": 0,
                "type_diversity": 0,
                "sufficient_for_interaction_workflows": False,
                "error": str(e),
            }

        # 5. Data availability — count key event types for workflow detection
        try:
            event_counts = result.get("event_counts", {})
            if isinstance(event_counts, dict) and "error" not in event_counts:
                email_count = event_counts.get("email.received", 0) + event_counts.get("email.sent", 0)
            else:
                # Re-query if event_counts failed above
                with self.db.get_connection("events") as conn:
                    row = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM events
                        WHERE julianday(timestamp) > julianday(?)
                          AND type IN ('email.received', 'email.sent')
                        """,
                        (cutoff.isoformat(),),
                    ).fetchone()
                email_count = row[0] if row else 0
            # Need at least 10 email events for meaningful workflow detection
            result["data_sufficient"] = email_count >= 10
        except Exception as e:
            logger.warning("get_diagnostics: data_sufficient check failed: %s", e)
            result["data_sufficient"] = False

        return result

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
