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

    def _detect_temporal_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines that occur at similar times each day.

        Groups episodes into time-of-day buckets (morning: 5-11am, afternoon: 11am-5pm,
        evening: 5-11pm, night: 11pm-5am) and looks for recurring action sequences.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of temporal routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Fetch episodes grouped by time-of-day bucket
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    strftime('%H', timestamp) as hour,
                    interaction_type,
                    COUNT(*) as frequency
                FROM episodes
                WHERE timestamp > ? AND interaction_type IS NOT NULL
                GROUP BY hour, interaction_type
                HAVING frequency >= ?
                ORDER BY hour, frequency DESC
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
            "night": (23, 5),      # 11pm-5am (wraps around)
        }

        bucket_actions = defaultdict(list)
        for hour_str, interaction_type, frequency in hour_actions:
            hour = int(hour_str)
            for bucket_name, (start, end) in buckets.items():
                if start < end:
                    if start <= hour < end:
                        bucket_actions[bucket_name].append((interaction_type, frequency, 5.0))
                else:  # Night bucket wraps around midnight
                    if hour >= start or hour < end:
                        bucket_actions[bucket_name].append((interaction_type, frequency, 5.0))

        # Create routines for buckets with multiple recurring actions
        for bucket_name, actions in bucket_actions.items():
            if len(actions) >= 2:  # Need at least 2 steps for a routine
                # Sort by frequency (most common first)
                actions.sort(key=lambda x: x[1], reverse=True)

                # Calculate consistency score (avg frequency / total days)
                avg_frequency = sum(freq for _, freq, _ in actions) / len(actions)
                consistency = min(1.0, avg_frequency / lookback_days)

                if consistency >= self.consistency_threshold:
                    routine = {
                        "name": f"{bucket_name.capitalize()} routine",
                        "trigger": bucket_name,
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": duration,
                                "skip_rate": 1.0 - (freq / avg_frequency) if avg_frequency > 0 else 0.0,
                            }
                            for i, (action, freq, duration) in enumerate(actions[:10])  # Max 10 steps
                        ],
                        "typical_duration_minutes": sum(d for _, _, d in actions),
                        "consistency_score": consistency,
                        "times_observed": int(avg_frequency),
                        "variations": [],  # Could add variation detection in future
                    }
                    routines.append(routine)
                    logger.debug(f"Detected {bucket_name} routine with {len(actions)} steps, consistency {consistency:.2f}")

        return routines

    def _detect_location_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines triggered by location changes (arrive/depart patterns).

        Looks for action sequences that consistently follow location transitions
        (e.g., arrive_home → turn on lights → check mail).

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of location-based routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Fetch episodes with location context
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    location,
                    interaction_type,
                    COUNT(*) as frequency
                FROM episodes
                WHERE timestamp > ?
                  AND location IS NOT NULL
                  AND interaction_type IS NOT NULL
                GROUP BY location, interaction_type
                HAVING frequency >= ?
                ORDER BY location, frequency DESC
            """, (cutoff.isoformat(), self.min_occurrences))

            location_actions = cursor.fetchall()

        if not location_actions:
            return routines

        # Group actions by location
        location_groups = defaultdict(list)
        for location, interaction_type, frequency in location_actions:
            location_groups[location].append((interaction_type, frequency, 5.0))

        # Create routines for locations with multiple recurring actions
        for location, actions in location_groups.items():
            if len(actions) >= 2:  # Need at least 2 steps
                actions.sort(key=lambda x: x[1], reverse=True)

                avg_frequency = sum(freq for _, freq, _ in actions) / len(actions)
                consistency = min(1.0, avg_frequency / lookback_days)

                if consistency >= self.consistency_threshold:
                    routine = {
                        "name": f"Arrive at {location}",
                        "trigger": f"arrive_{location.lower().replace(' ', '_')}",
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": duration,
                                "skip_rate": 1.0 - (freq / avg_frequency) if avg_frequency > 0 else 0.0,
                            }
                            for i, (action, freq, duration) in enumerate(actions[:10])
                        ],
                        "typical_duration_minutes": sum(d for _, _, d in actions),
                        "consistency_score": consistency,
                        "times_observed": int(avg_frequency),
                        "variations": [],
                    }
                    routines.append(routine)
                    logger.debug(f"Detected location routine for {location} with {len(actions)} steps")

        return routines

    def _detect_event_triggered_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines triggered by specific event types.

        Identifies patterns like:
        - After meetings: update task list, send follow-up email
        - After receiving invoice: review, approve, forward to accounting
        - Friday afternoon: week review, inbox cleanup, plan next week

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of event-triggered routines
        """
        routines = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        # Look for sequences of actions following specific event types
        # This requires temporal ordering within episodes
        with self.db.get_connection("user_model") as conn:
            cursor = conn.cursor()

            # Find common interaction types that might trigger routines
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

        for interaction_type, days_occurred in trigger_events:
            # For each potential trigger, find interaction types that commonly follow
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        e2.interaction_type,
                        COUNT(*) as frequency
                    FROM episodes e1
                    JOIN episodes e2 ON DATE(e1.timestamp) = DATE(e2.timestamp)
                        AND e2.timestamp > e1.timestamp
                        AND e2.timestamp < datetime(e1.timestamp, '+2 hours')
                        AND e1.interaction_type != e2.interaction_type
                    WHERE e1.interaction_type = ?
                      AND e1.timestamp > ?
                    GROUP BY e2.interaction_type
                    HAVING frequency >= ?
                    ORDER BY frequency DESC
                """, (interaction_type, cutoff.isoformat(), self.min_occurrences))

                following_actions = cursor.fetchall()

            if len(following_actions) >= 2:  # At least 2 steps
                avg_frequency = sum(freq for _, freq in following_actions) / len(following_actions)
                consistency = min(1.0, avg_frequency / days_occurred)

                if consistency >= self.consistency_threshold:
                    # Clean up interaction type for display
                    trigger_name = interaction_type.replace("_", " ").title()

                    routine = {
                        "name": f"After {trigger_name}",
                        "trigger": f"after_{interaction_type}",
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": 5.0,
                                "skip_rate": 1.0 - (freq / avg_frequency) if avg_frequency > 0 else 0.0,
                            }
                            for i, (action, freq) in enumerate(following_actions[:10])
                        ],
                        "typical_duration_minutes": len(following_actions) * 5.0,
                        "consistency_score": consistency,
                        "times_observed": int(avg_frequency),
                        "variations": [],
                    }
                    routines.append(routine)
                    logger.debug(f"Detected event-triggered routine after {interaction_type}")

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
