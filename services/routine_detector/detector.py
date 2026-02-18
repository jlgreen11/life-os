"""
Life OS — Routine Detector

Analyzes episodic memory to detect recurring behavioral patterns (routines).
Looks for sequences of actions that occur regularly at similar times or after
specific triggers.

Routine types detected:
- Temporal routines (morning routine, end of day)
- Location-based routines (arrive at work, arrive home)
- Event-triggered routines (after meetings, before travel)
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


class RoutineDetector:
    """Detects recurring behavioral patterns from episodic memory.

    Analyzes episodes to find sequences of actions that repeat with regularity.
    Routines are characterized by:
    - Trigger (time of day, location arrival, event type)
    - Sequence of steps (actions taken in order)
    - Consistency (how often the pattern is followed)
    - Timing (typical duration, time of day)

    Example routines:
    - Morning: wake up → check email → review calendar → coffee
    - End of work: calendar review → inbox zero → update task list
    - Arrive home: turn on lights → check mail → change clothes
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        """Initialize the routine detector.

        Args:
            db: Database manager for querying episodic memory
            user_model_store: Store for persisting detected routines
        """
        self.db = db
        self.user_model_store = user_model_store

        # Detection thresholds
        self.min_occurrences = 3  # Need at least 3 instances to call it a routine
        self.time_window_hours = 2  # Actions within 2h can be part of same routine
        self.consistency_threshold = 0.6  # 60% of instances must match for it to be a routine

    def detect_routines(self, lookback_days: int = 30) -> list[dict[str, Any]]:
        """Detect all routines from recent episodic memory.

        Runs multiple detection strategies in parallel:
        1. Time-of-day routines (morning, afternoon, evening)
        2. Location-based routines (arrive/depart patterns)
        3. Event-triggered routines (after-meeting patterns)

        Args:
            lookback_days: How many days of history to analyze (default 30)

        Returns:
            List of detected routines with metadata
        """
        routines = []

        # Strategy 1: Time-of-day routines
        temporal_routines = self._detect_temporal_routines(lookback_days)
        routines.extend(temporal_routines)

        # Strategy 2: Location-based routines
        location_routines = self._detect_location_routines(lookback_days)
        routines.extend(location_routines)

        # Strategy 3: Event-triggered routines (e.g., post-meeting patterns)
        event_routines = self._detect_event_triggered_routines(lookback_days)
        routines.extend(event_routines)

        logger.info(
            f"Routine detection complete: {len(routines)} routines found "
            f"({len(temporal_routines)} temporal, {len(location_routines)} location-based, "
            f"{len(event_routines)} event-triggered)"
        )

        return routines

    def _count_active_days(self, cutoff: datetime) -> int:
        """Count distinct calendar days with at least one episode since the cutoff.

        Used to normalize consistency scores against the actual span of observed
        data rather than the full lookback window.  Without this, a 10-day dataset
        queried over a 30-day window would always score consistency ≤ 0.33, making
        it impossible to reach the 0.6 threshold even for a perfect daily pattern.

        Args:
            cutoff: Only count days after this timestamp

        Returns:
            Number of distinct days with episode data, or 1 to avoid division by zero
        """
        with self.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT DATE(timestamp)) FROM episodes WHERE timestamp > ?",
                (cutoff.isoformat(),),
            ).fetchone()
        return max(1, row[0] if row and row[0] else 1)

    def _compute_step_duration_map(self, cutoff: datetime) -> dict[str, float]:
        """Compute average gap (in minutes) from each interaction type to the next.

        For each interaction type observed since *cutoff*, measures the average
        time to the immediately-following episode on the same calendar day.  This
        replaces the old hardcoded 5-minute placeholder with real observed
        inter-step durations, giving every detection strategy (temporal,
        location-based, event-triggered) accurate timing data.

        Implementation details:
        - Uses a window-function CTE to rank episodes within each day, then
          self-joins each episode to the one immediately after it (rn + 1).
        - JULIANDAY arithmetic converts the gap to fractional days, which we
          multiply by 24 * 60 to get minutes.
        - Both sides of the gap join use datetime() so that SQLite compares
          normalised ``YYYY-MM-DD HH:MM:SS`` strings (stored timestamps may
          carry a ``+00:00`` suffix that breaks plain string comparison).

        Args:
            cutoff: Only consider episodes after this timestamp.

        Returns:
            Dict mapping interaction_type → average gap in minutes to the
            next episode.  Types with no measurable successor are absent;
            callers should use a sensible fallback (e.g. 5.0 minutes).

        Example::

            map = detector._compute_step_duration_map(cutoff)
            duration = map.get("check_email", 5.0)  # → ~15.0 if observed
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute("""
                WITH ranked AS (
                    SELECT
                        interaction_type,
                        DATE(timestamp) as day,
                        datetime(timestamp) as ts,
                        ROW_NUMBER() OVER (
                            PARTITION BY DATE(timestamp)
                            ORDER BY datetime(timestamp)
                        ) as rn
                    FROM episodes
                    WHERE timestamp > ? AND interaction_type IS NOT NULL
                )
                SELECT
                    a.interaction_type,
                    AVG(
                        (JULIANDAY(b.ts) - JULIANDAY(a.ts)) * 24 * 60
                    ) as avg_gap_minutes
                FROM ranked a
                JOIN ranked b ON a.day = b.day AND b.rn = a.rn + 1
                GROUP BY a.interaction_type
            """, (cutoff.isoformat(),)).fetchall()
        return {row[0]: row[1] for row in rows if row[1] is not None}

    def _detect_temporal_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines that occur at similar times each day.

        Groups episodes into time-of-day buckets (morning: 5-11am, afternoon: 11am-5pm,
        evening: 5-11pm, night: 11pm-5am) and looks for recurring action sequences.

        Consistency is measured as (avg occurrences per active day) rather than
        (avg occurrences / full lookback window).  The full-window denominator
        systematically underestimates consistency when data spans only a fraction
        of the lookback period — e.g., 10 days of perfect data in a 30-day window
        would score 0.33 instead of 1.0.

        Step durations are measured from actual observed inter-episode gaps via
        ``_compute_step_duration_map()``, with a 5-minute fallback for types
        where no successor was observed, and a 15-minute default for the final
        step of every routine (no successor to measure against).

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of temporal routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Number of distinct days with any episode data in the window.
        # This is the denominator for consistency: an action that fires on 8 of
        # 10 active days has 80% consistency, regardless of the lookback window.
        active_days = self._count_active_days(cutoff)

        # Fetch episodes grouped by time-of-day bucket.
        # strftime('%H', timestamp) extracts the UTC hour (0-23).
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    strftime('%H', timestamp) as hour,
                    interaction_type,
                    COUNT(DISTINCT DATE(timestamp)) as day_count
                FROM episodes
                WHERE timestamp > ? AND interaction_type IS NOT NULL
                GROUP BY hour, interaction_type
                HAVING day_count >= ?
                ORDER BY hour, day_count DESC
            """, (cutoff.isoformat(), self.min_occurrences))

            hour_actions = cursor.fetchall()

        if not hour_actions:
            return routines

        # Group actions by time-of-day bucket
        buckets = {
            "morning": (5, 11),    # 5am-11am
            "midday": (11, 14),    # 11am-2pm
            "afternoon": (14, 17), # 2pm-5pm
            "evening": (17, 23),   # 5pm-11pm
            "night": (23, 5),      # 11pm-5am (wraps around midnight)
        }

        # Compute actual measured inter-step durations once, reuse for all buckets.
        # The final step of each routine falls back to LAST_STEP_DEFAULT_MINUTES
        # because there is no subsequent step to measure a gap against.
        LAST_STEP_DEFAULT_MINUTES = 15.0
        step_duration_map = self._compute_step_duration_map(cutoff)

        bucket_actions: dict[str, list] = defaultdict(list)
        for hour_str, interaction_type, day_count in hour_actions:
            hour = int(hour_str)
            # Look up the measured gap duration; placeholder 5.0 if not available.
            duration = step_duration_map.get(interaction_type, 5.0)
            for bucket_name, (start, end) in buckets.items():
                if start < end:
                    if start <= hour < end:
                        bucket_actions[bucket_name].append((interaction_type, day_count, duration))
                else:  # Night bucket wraps around midnight
                    if hour >= start or hour < end:
                        bucket_actions[bucket_name].append((interaction_type, day_count, duration))

        # Create routines for buckets with at least one recurring action.
        # A single consistent action at a fixed time (e.g., morning coffee at 7am)
        # is already a valid behavioral routine worth surfacing.
        for bucket_name, actions in bucket_actions.items():
            if len(actions) >= 1:
                # Sort by recurrence (most days first)
                actions.sort(key=lambda x: x[1], reverse=True)

                # Consistency = fraction of active days where this action appeared.
                # avg_day_count is the mean across all actions in the bucket so
                # that actions missing on a few days don't dominate the score.
                avg_day_count = sum(dc for _, dc, _ in actions) / len(actions)
                consistency = min(1.0, avg_day_count / active_days)

                if consistency >= self.consistency_threshold:
                    steps = actions[:10]  # Cap at 10 steps
                    # Compute total routine duration: sum of measured gap durations
                    # for all-but-last step, plus the default for the last step.
                    step_durations = [d for _, _, d in steps]
                    if step_durations:
                        # Last step has no measured gap to a successor, so use default.
                        step_durations[-1] = LAST_STEP_DEFAULT_MINUTES
                    total_duration = sum(step_durations) if step_durations else LAST_STEP_DEFAULT_MINUTES

                    routine = {
                        "name": f"{bucket_name.capitalize()} routine",
                        "trigger": bucket_name,
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": dur,
                                # skip_rate = fraction of active days the step was absent
                                "skip_rate": max(0.0, 1.0 - (dc / active_days)),
                            }
                            for i, (action, dc, dur) in enumerate(steps)
                        ],
                        "typical_duration_minutes": total_duration,
                        "consistency_score": consistency,
                        "times_observed": int(avg_day_count),
                        "variations": [],  # Could add variation detection in future
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected %s routine with %d steps, consistency %.2f",
                        bucket_name, len(actions), consistency,
                    )

        return routines

    def _detect_location_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines triggered by location changes (arrive/depart patterns).

        Looks for action sequences that consistently follow location transitions
        (e.g., arrive_home → turn on lights → check mail).  A location routine
        requires only 1 recurring interaction type rather than 2, because a
        reliable single action at a location (e.g., always checking mail on
        arriving home) is itself a meaningful behavioral signal.

        Consistency is normalized against actual active days, not the full
        lookback window, for the same reason as temporal routines.

        Step durations use actual measured inter-episode gaps from
        ``_compute_step_duration_map()``, with a 5-minute fallback for types
        where no successor was observed.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of location-based routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Reuse active-day count computed once for the whole detection pass.
        active_days = self._count_active_days(cutoff)

        # Use actual measured inter-step durations instead of a hardcoded
        # placeholder.  Falls back to 5.0 minutes for interaction types where
        # no same-day successor was observed.
        step_duration_map = self._compute_step_duration_map(cutoff)
        STEP_DURATION_FALLBACK = 5.0

        # Fetch recurring (location, interaction_type) pairs.
        # day_count = distinct days where this action occurred at this location,
        # which is the correct denominator for the consistency fraction.
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    location,
                    interaction_type,
                    COUNT(DISTINCT DATE(timestamp)) as day_count
                FROM episodes
                WHERE timestamp > ?
                  AND location IS NOT NULL
                  AND interaction_type IS NOT NULL
                GROUP BY location, interaction_type
                HAVING day_count >= ?
                ORDER BY location, day_count DESC
            """, (cutoff.isoformat(), self.min_occurrences))

            location_actions = cursor.fetchall()

        if not location_actions:
            return routines

        # Group recurring actions by location, attaching measured durations.
        location_groups: dict[str, list] = defaultdict(list)
        for location, interaction_type, day_count in location_actions:
            duration = step_duration_map.get(interaction_type, STEP_DURATION_FALLBACK)
            location_groups[location].append((interaction_type, day_count, duration))

        # Create a routine for every location that has at least one recurring action.
        # We use >= 1 (not >= 2) because even a single reliable action at a
        # location constitutes a behaviorally meaningful pattern.
        for location, actions in location_groups.items():
            if len(actions) >= 1:
                actions.sort(key=lambda x: x[1], reverse=True)

                avg_day_count = sum(dc for _, dc, _ in actions) / len(actions)
                consistency = min(1.0, avg_day_count / active_days)

                if consistency >= self.consistency_threshold:
                    routine = {
                        "name": f"Arrive at {location}",
                        "trigger": f"arrive_{location.lower().replace(' ', '_')}",
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": duration,
                                "skip_rate": max(0.0, 1.0 - (dc / active_days)),
                            }
                            for i, (action, dc, duration) in enumerate(actions[:10])
                        ],
                        "typical_duration_minutes": sum(d for _, _, d in actions),
                        "consistency_score": consistency,
                        "times_observed": int(avg_day_count),
                        "variations": [],
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected location routine for %s with %d steps, consistency %.2f",
                        location, len(actions), consistency,
                    )

        return routines

    def _detect_event_triggered_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines triggered by specific event types.

        Identifies patterns like:
        - After meetings: update task list, send follow-up email
        - After receiving invoice: review, approve, forward to accounting
        - Friday afternoon: week review, inbox cleanup, plan next week

        Timezone-safe timestamp comparison:
        Episode timestamps are stored with UTC offset (e.g., ``+00:00``), but
        SQLite's ``datetime()`` modifier returns timestamps WITHOUT a timezone
        suffix.  String comparison between ``'2026-02-09T08:05:00+00:00'`` and
        ``'2026-02-09 10:05:00'`` (returned by ``datetime(ts, '+2 hours')``) is
        lexicographically wrong because ASCII ``T`` (84) > ASCII `` `` (32), so
        every tz-aware timestamp compares as *greater* than the ``datetime()``
        output, making the window check always false.

        Fix: wrap both sides in ``datetime()`` so SQLite normalises both to
        the same ``YYYY-MM-DD HH:MM:SS`` format before comparing.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of event-triggered routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Look for interaction types that occur on enough distinct days to be
        # candidates for routine triggers.
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    interaction_type,
                    COUNT(DISTINCT DATE(timestamp)) as days_occurred
                FROM episodes
                WHERE timestamp > ? AND interaction_type IS NOT NULL
                GROUP BY interaction_type
                HAVING days_occurred >= ?
                ORDER BY days_occurred DESC
            """, (cutoff.isoformat(), self.min_occurrences))

            trigger_events = cursor.fetchall()

        # Compute measured inter-step durations once, shared across all trigger types.
        # Falls back to 5.0 minutes for interaction types where no same-day successor
        # was observed in the lookback window.
        step_duration_map = self._compute_step_duration_map(cutoff)
        STEP_DURATION_FALLBACK = 5.0

        for interaction_type, days_occurred in trigger_events:
            # For each candidate trigger, find interaction types that commonly
            # follow it within a 2-hour window on the same calendar day.
            #
            # IMPORTANT: both timestamp operands are wrapped in datetime() so
            # that SQLite compares ``YYYY-MM-DD HH:MM:SS`` strings on both
            # sides.  Without this, stored timestamps (ISO 8601 with +00:00)
            # compare as always-greater than datetime() output (no TZ suffix),
            # causing the window filter to silently drop every match.
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        e2.interaction_type,
                        COUNT(DISTINCT DATE(e1.timestamp)) as day_count
                    FROM episodes e1
                    JOIN episodes e2 ON DATE(e1.timestamp) = DATE(e2.timestamp)
                        AND datetime(e2.timestamp) > datetime(e1.timestamp)
                        AND datetime(e2.timestamp) < datetime(e1.timestamp, '+2 hours')
                        AND e1.interaction_type != e2.interaction_type
                    WHERE e1.interaction_type = ?
                      AND e1.timestamp > ?
                    GROUP BY e2.interaction_type
                    HAVING day_count >= ?
                    ORDER BY day_count DESC
                """, (interaction_type, cutoff.isoformat(), self.min_occurrences))

                following_actions = cursor.fetchall()

            if len(following_actions) >= 2:  # At least 2 following steps for a sequence
                avg_day_count = sum(dc for _, dc in following_actions) / len(following_actions)
                # Consistency = fraction of trigger-event days where the follow-up
                # actions also appeared.  Use days_occurred as the denominator
                # (number of days the trigger fired) rather than total active days.
                consistency = min(1.0, avg_day_count / days_occurred)

                if consistency >= self.consistency_threshold:
                    trigger_name = interaction_type.replace("_", " ").title()

                    # Attach measured durations to each following action.
                    # Replaces the old hardcoded 5.0-minute value with actual
                    # observed inter-step gaps for more accurate routine timing.
                    steps_with_duration = [
                        (action, dc, step_duration_map.get(action, STEP_DURATION_FALLBACK))
                        for action, dc in following_actions[:10]
                    ]
                    total_duration = sum(d for _, _, d in steps_with_duration)

                    routine = {
                        "name": f"After {trigger_name}",
                        "trigger": f"after_{interaction_type}",
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": dur,
                                "skip_rate": max(0.0, 1.0 - (dc / days_occurred)),
                            }
                            for i, (action, dc, dur) in enumerate(steps_with_duration)
                        ],
                        "typical_duration_minutes": total_duration,
                        "consistency_score": consistency,
                        "times_observed": int(avg_day_count),
                        "variations": [],
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected event-triggered routine after %s, consistency %.2f",
                        interaction_type, consistency,
                    )

        return routines

    def store_routines(self, routines: list[dict[str, Any]]) -> int:
        """Persist detected routines to the database.

        Uses UPSERT logic: if a routine with the same name already exists,
        updates its statistics (consistency_score, times_observed). New steps
        are merged with existing steps.

        Args:
            routines: List of routine dictionaries to store

        Returns:
            Number of routines stored
        """
        stored_count = 0
        for routine in routines:
            try:
                self.user_model_store.store_routine(routine)
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store routine '{routine.get('name')}': {e}")

        logger.info(f"Stored {stored_count}/{len(routines)} routines")
        return stored_count
