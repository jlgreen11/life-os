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
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from storage.manager import DatabaseManager
    from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)

# Mapping from event type prefixes to activity classifications used when
# signal profile data is unavailable.  The fallback episode-based routine
# detector uses this to classify raw episodes by their linked event type
# instead of relying on the temporal signal profile's interaction_type.
EVENT_TYPE_TO_ACTIVITY: dict[str, str] = {
    "email.received": "email_check",
    "email.sent": "email_compose",
    "calendar.event.created": "calendar_review",
    "calendar.event.updated": "calendar_review",
    "notification.created": "notification_check",
    "system.connector.sync_complete": "system_maintenance",
    "message.received": "message_check",
    "message.sent": "message_compose",
    "task.created": "task_management",
    "task.completed": "task_management",
    "browser.page_visited": "web_browsing",
}


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

    # Prefixes that identify internal telemetry episode types (not real user
    # activity).  These dominate episode counts and skew consistency scores
    # if not excluded from routine detection queries.
    INTERNAL_TYPE_PREFIXES = ("usermodel_", "system_", "test")

    # SQL WHERE clause fragment to exclude internal telemetry types from
    # routine detection queries.  Append after existing NOT IN filters.
    INTERNAL_TYPE_SQL_FILTER = (
        "AND interaction_type NOT LIKE 'usermodel_%' "
        "AND interaction_type NOT LIKE 'system_%' "
        "AND interaction_type NOT LIKE 'test%'"
    )

    # Interaction types that represent high-volume passive events whose arrival
    # patterns are driven by external senders rather than the user's deliberate
    # behavior.  Email arrives when others send it, not when the user acts, so
    # a 60% consistency threshold is too strict: an inbox with 30 messages/day
    # spread across morning and afternoon won't reliably hit the morning bucket
    # on more than ~50-60% of days even with a strong morning skew.
    HIGH_VOLUME_PASSIVE_TYPES: frozenset[str] = frozenset(
        {
            "email_received",
            "email_sent",
            "message_received",
            "notification_received",
        }
    )

    # Upper-bound consistency threshold applied to HIGH_VOLUME_PASSIVE_TYPES.
    # 0.4 means "this bucket must be active on at least 40% of all active days"
    # — achievable for genuine morning-email patterns while still excluding
    # uniformly distributed noise.  Applied as a cap (min) so that cold-start
    # scaling (which may already be lower) is never overridden upward.
    PASSIVE_TYPE_CONSISTENCY_THRESHOLD: float = 0.4

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore, timezone: str = "UTC"):
        """Initialize the routine detector.

        Args:
            db: Database manager for querying episodic memory
            user_model_store: Store for persisting detected routines
            timezone: IANA timezone string (e.g. 'America/New_York') for
                converting UTC episode timestamps to local time before
                time-of-day bucketing. Defaults to 'UTC' (fail-open).
        """
        self.db = db
        self.user_model_store = user_model_store
        self._tz = ZoneInfo(timezone)

        # Detection thresholds
        self.min_occurrences = 3  # Need at least 3 instances to call it a routine
        self.time_window_hours = 2  # Actions within 2h can be part of same routine
        self.consistency_threshold = 0.6  # 60% of instances must match for it to be a routine
        self.min_episodes_for_detection = 50  # Minimum episodes before skipping fallback

        # Diagnostics: cached result from the most recent detect_routines() call
        self._last_detection_count: int | None = None
        self._last_detection_time: str | None = None

    def _effective_consistency_threshold(
        self,
        active_days: int,
        interaction_type: str | None = None,
    ) -> float:
        """Return a consistency threshold scaled by data maturity and interaction type.

        During cold-start (few active days), the base threshold of 0.6 is too
        strict — a pattern must appear on 60 % of active days, which is nearly
        impossible when data is sparse or unevenly distributed.  This method
        applies a graduated scale so that the system can surface provisional
        routines early while converging to full strictness as data matures.

        Additionally, ``HIGH_VOLUME_PASSIVE_TYPES`` (e.g. ``email_received``)
        receive a lower cap (``PASSIVE_TYPE_CONSISTENCY_THRESHOLD = 0.4``)
        because their arrival time is driven by external senders, not by the
        user's deliberate schedule.  Thirty emails per day spread across morning
        and afternoon will never achieve 60 % morning-bucket consistency even
        when the user genuinely has a morning-email habit; the lower cap makes
        these patterns detectable.

        Scaling tiers (applied before the type-aware cap):
            active_days < 7   → 0.3  (very lenient — system just started)
            active_days < 14  → 0.4  (moderate — building data)
            active_days < 30  → 0.5  (approaching maturity)
            active_days >= 30 → self.consistency_threshold (0.6 — full strictness)

        Type-aware cap (applied after day-scaling):
            interaction_type in HIGH_VOLUME_PASSIVE_TYPES →
                min(day_tier_threshold, PASSIVE_TYPE_CONSISTENCY_THRESHOLD)
            This ensures cold-start scaling (already lower) is never overridden.

        Args:
            active_days: Number of distinct calendar days with episode data.
            interaction_type: The dominant interaction type for the current
                detection bucket.  ``None`` means no type-aware override.

        Returns:
            The effective threshold to use for consistency filtering.
        """
        if active_days < 7:
            threshold = 0.3
        elif active_days < 14:
            threshold = 0.4
        elif active_days < 30:
            threshold = 0.5
        else:
            threshold = self.consistency_threshold

        # For high-volume passive types, cap the threshold to avoid requiring
        # unreachable consistency for bursty external-signal types.  We use
        # min() so cold-start scaling (already at or below 0.4) is preserved.
        if interaction_type in self.HIGH_VOLUME_PASSIVE_TYPES:
            threshold = min(threshold, self.PASSIVE_TYPE_CONSISTENCY_THRESHOLD)

        logger.info(
            "Effective consistency threshold: %.2f (active_days=%d, base=%.2f, type=%s)",
            threshold,
            active_days,
            self.consistency_threshold,
            interaction_type or "none",
        )
        return threshold

    def _effective_min_episodes(self, episode_count: int, data_age_days: int) -> int:
        """Return an adaptive minimum-episodes threshold scaled by data maturity.

        During cold-start (first few days of data), requiring 50 episodes before
        triggering the fallback path is too strict — email-only installations may
        take weeks to accumulate that many typed episodes.  This method scales the
        threshold down during early operation so that cold-start systems can still
        detect routines.

        This mirrors the scaling tiers used by ``_effective_consistency_threshold()``
        so that both gates open and close at the same data-maturity milestones:

        Scaling tiers:
            data_age_days < 7   → 10  (very lenient — first week)
            data_age_days < 14  → 20  (second week)
            data_age_days < 30  → 35  (first month)
            data_age_days >= 30 → self.min_episodes_for_detection (50 — mature system)

        Args:
            episode_count: Number of episodes that passed the interaction_type filter.
                Included for diagnostic logging; not used in the threshold calculation.
            data_age_days: Age of the oldest typed episode in days, used to
                determine data maturity tier.  Pass 0 when no typed episodes exist
                (cold-start assumption — most lenient tier).

        Returns:
            Effective minimum episode count before the fallback path is triggered.
        """
        if data_age_days < 7:
            threshold = 10
        elif data_age_days < 14:
            threshold = 20
        elif data_age_days < 30:
            threshold = 35
        else:
            threshold = self.min_episodes_for_detection

        logger.info(
            "Effective min_episodes threshold: %d (data_age_days=%d, primary_episodes=%d, base=%d)",
            threshold,
            data_age_days,
            episode_count,
            self.min_episodes_for_detection,
        )
        return threshold

    def _compute_adaptive_lookback_days(self, lookback_days: int) -> int:
        """Compute an adaptive lookback window that extends when recent episodes are sparse.

        When a connector outage moves all episodes outside the default lookback window,
        routine detection returns 0 results despite thousands of historical episodes.
        This method detects that scenario and extends the window to cover available data.

        Algorithm:
        1. Count episodes in the default window.
        2. If count > 0, the system has recent data — return lookback_days unchanged.
        3. If count == 0 (connector outage scenario), find the oldest timestamp among
           the 200 most recent episodes to discover where the data is.
        4. If that timestamp predates the current cutoff, extend lookback_days to cover
           it (adding a 1-day buffer), capped at MAX_ADAPTIVE_LOOKBACK_DAYS (180).
        5. On any DB error, fall back silently to the original lookback_days.

        The extension fires ONLY when the default window is completely empty (0 episodes).
        Having any recent episodes means the system is still receiving data normally;
        extending in that case would incorrectly include pre-outage data alongside fresh
        data and violate the lookback boundary guarantee.

        Args:
            lookback_days: The requested lookback window in days.

        Returns:
            Effective lookback window in days (>= lookback_days, <= 180).
        """
        MAX_ADAPTIVE_LOOKBACK_DAYS = 180
        try:
            now = datetime.now(UTC)
            cutoff = now - timedelta(days=lookback_days)
            cutoff_iso = cutoff.isoformat()

            # Step 1: Count episodes in the default window.
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE timestamp > ?",
                    (cutoff_iso,),
                ).fetchone()
            episode_count = row[0] if row else 0

            # Step 2: Episodes exist in the default window — no extension needed.
            # We only extend when the window is completely empty (count == 0),
            # which is the connector-outage scenario where ALL data has aged out.
            # Having even a small number of recent episodes means the system is
            # still receiving data; extending in that case would incorrectly pull
            # in pre-outage episodes alongside fresh ones and break the lookback
            # filtering guarantee that tests like test_lookback_period_filtering
            # rely on.
            if episode_count > 0:
                return lookback_days

            # Step 3: Find the oldest timestamp among the 200 most recent episodes.
            # This tells us how far back we need to look to get meaningful data.
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT MIN(timestamp)
                       FROM (SELECT timestamp FROM episodes ORDER BY timestamp DESC LIMIT 200)""",
                ).fetchone()

            if not row or not row[0]:
                # No episodes at all — extension would be pointless.
                return lookback_days

            try:
                oldest_dt = datetime.fromisoformat(row[0])
                if oldest_dt.tzinfo is None:
                    oldest_dt = oldest_dt.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                return lookback_days

            # Step 4: If the oldest recent episode predates the current cutoff,
            # extend the window to cover it (plus a 1-day safety buffer).
            if oldest_dt < cutoff:
                needed_days = int((now - oldest_dt).total_seconds() / 86400) + 1
                effective_days = min(needed_days, MAX_ADAPTIVE_LOOKBACK_DAYS)
                logger.info(
                    "Adaptive lookback: extended from %d to %d days (%d episodes in default window)",
                    lookback_days,
                    effective_days,
                    episode_count,
                )
                return effective_days

            return lookback_days

        except Exception:
            logger.warning(
                "Adaptive lookback computation failed — falling back to default %d days",
                lookback_days,
            )
            return lookback_days

    @staticmethod
    def _hour_to_bucket(hour: int) -> str:
        """Map an hour (0–23) to a time-of-day bucket name.

        Bucket boundaries:
            morning   :  5 – 10
            midday    : 11 – 13
            afternoon : 14 – 16
            evening   : 17 – 22
            night     :  0 –  4 and 23

        Args:
            hour: Hour of the day in the user's local timezone (0–23).

        Returns:
            One of 'morning', 'midday', 'afternoon', 'evening', 'night'.
        """
        if 5 <= hour <= 10:
            return "morning"
        if 11 <= hour <= 13:
            return "midday"
        if 14 <= hour <= 16:
            return "afternoon"
        if 17 <= hour <= 22:
            return "evening"
        return "night"

    def _backfill_stale_interaction_types(self, lookback_days: int) -> None:
        """Batch-update episodes with stale interaction_type values.

        Episodes created before the granular ``_classify_interaction_type()``
        logic was deployed have ``interaction_type`` set to NULL, 'unknown',
        or the generic 'communication'.  These get filtered out by the primary
        temporal detection query, leaving 0 usable episodes even when thousands
        exist.

        This method runs a one-time batch backfill at the start of
        ``detect_routines()`` by:
        1. Querying stale episode ``event_id`` values from user_model.db
        2. Looking up corresponding event types from events.db
        3. Converting dotted event types to underscored interaction types
           (e.g. 'email.received' → 'email_received')
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

        # Build mapping: event_id → episode_id(s)
        event_id_to_episode_ids: dict[str, list[str]] = defaultdict(list)
        for ep_id, ev_id in stale_rows:
            event_id_to_episode_ids[ev_id].append(ep_id)

        all_event_ids = list(event_id_to_episode_ids.keys())

        # Step 2: Look up event types from events.db in chunks
        event_type_map: dict[str, str] = {}  # event_id → event.type
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

        updated_count = 0
        try:
            with self.db.get_connection("user_model") as conn:
                for i in range(0, len(updates), chunk_size):
                    chunk = updates[i : i + chunk_size]
                    conn.executemany(
                        "UPDATE episodes SET interaction_type = ? WHERE id = ?",
                        chunk,
                    )
                conn.commit()
                updated_count = len(updates)
        except Exception:
            logger.warning("Backfill: failed to update episodes — skipping")
            return

        logger.info("Backfill: updated %d episodes with derived interaction_type values", updated_count)

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
        # Adaptive lookback: when the default window contains fewer episodes than
        # the detection minimum (e.g., because a connector outage pushed all
        # episodes older than 30 days), extend the window to cover the most recent
        # available data.  This runs BEFORE backfill and detection so every
        # downstream step benefits from the same extended window.
        # Wrapped in try/except so a transient DB failure doesn't block detection.
        try:
            effective_lookback_days = self._compute_adaptive_lookback_days(lookback_days)
        except Exception:
            logger.warning(
                "Adaptive lookback computation raised unexpectedly — using default %d days",
                lookback_days,
            )
            effective_lookback_days = lookback_days

        # Backfill stale interaction_type values before any detection strategy
        # runs, so all strategies benefit from properly classified episodes.
        try:
            self._backfill_stale_interaction_types(effective_lookback_days)
        except Exception:
            logger.exception("Interaction type backfill failed — continuing with detection")

        routines = []

        # Each strategy is wrapped in try/except so that a failure in one
        # (e.g. corrupted user_model.db) does not prevent the others from
        # running.  This follows the same fail-open pattern used by the
        # InsightEngine correlators.
        temporal_routines = []
        try:
            temporal_routines = self._detect_temporal_routines(effective_lookback_days)
            routines.extend(temporal_routines)
        except Exception:
            logger.exception("Temporal routine detection failed (possible DB corruption)")

        # Fallback: when primary temporal detection finds nothing, try
        # detecting routines directly from raw episode data.  This handles
        # scenarios where signal profiles are missing, corrupted, or where
        # primary detection returns 0 despite thousands of episodes.
        if not temporal_routines:
            logger.info(
                "Temporal detection returned 0 routines — trying episode-based fallback",
            )
            try:
                fallback_routines = self._detect_routines_from_episodes_fallback(effective_lookback_days)
                temporal_routines = fallback_routines
                routines.extend(fallback_routines)
            except Exception:
                logger.exception("Episode-based fallback routine detection failed")

        location_routines = []
        try:
            location_routines = self._detect_location_routines(effective_lookback_days)
            routines.extend(location_routines)
        except Exception:
            logger.exception("Location routine detection failed (possible DB corruption)")

        event_routines = []
        try:
            event_routines = self._detect_event_triggered_routines(effective_lookback_days)
            routines.extend(event_routines)
        except Exception:
            logger.exception("Event-triggered routine detection failed (possible DB corruption)")

        logger.info(
            f"Routine detection complete: {len(routines)} routines found "
            f"({len(temporal_routines)} temporal, {len(location_routines)} location-based, "
            f"{len(event_routines)} event-triggered)"
        )

        # Prune stored routines that are no longer being detected.
        # This prevents abandoned patterns from accumulating and generating
        # false routine_deviation predictions in the prediction engine.
        # Guard: skip pruning when detection returned 0 routines to avoid
        # nuking all stored routines due to a transient detection failure.
        if routines:
            self.prune_stale_routines(routines)
        else:
            logger.info("Skipping prune: 0 routines detected (preserving existing routines)")

        # Cache result for diagnostics observability
        self._last_detection_count = len(routines)
        self._last_detection_time = datetime.now(UTC).isoformat()

        return routines

    def get_diagnostics(self, lookback_days: int = 30) -> dict:
        """Return routine detector diagnostic information for monitoring.

        Reports data availability, detection thresholds, and pipeline health
        so operators can understand why routines are or aren't being detected.
        Follows the same pattern as InsightEngine.get_diagnostics() and
        PredictionEngine.get_diagnostics().

        Each field is queried independently with try/except so that a single
        DB failure doesn't prevent the rest of the diagnostics from returning.

        Args:
            lookback_days: How many days of history to analyze (default 30).

        Returns:
            Dict with keys: episode_count, active_days, effective_consistency_threshold,
            distinct_interaction_types, episodes_per_day, time_bucket_distribution,
            usable_episode_count, interaction_type_counts, candidate_pairs_count,
            pairs_meeting_min_occurrences, stored_routines_count,
            last_detection_count, last_detection_time, health.
        """
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        result: dict = {}

        # 1. Episode count in lookback window
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM episodes WHERE timestamp >= ?",
                    (cutoff.isoformat(),),
                ).fetchone()
            result["episode_count"] = row[0] if row else 0
        except Exception as e:
            logger.warning("get_diagnostics: episode_count query failed: %s", e)
            result["episode_count"] = {"error": str(e)}

        # 2. Active days (distinct calendar days with episode data)
        try:
            result["active_days"] = self._count_active_days(cutoff)
        except Exception as e:
            logger.warning("get_diagnostics: active_days query failed: %s", e)
            result["active_days"] = {"error": str(e)}

        # 3. Effective consistency threshold for the current data maturity
        try:
            active_days_val = result["active_days"] if isinstance(result["active_days"], int) else 1
            result["effective_consistency_threshold"] = self._effective_consistency_threshold(active_days_val)
        except Exception as e:
            logger.warning("get_diagnostics: threshold computation failed: %s", e)
            result["effective_consistency_threshold"] = {"error": str(e)}

        # 4. Distinct interaction types in the lookback window
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    "SELECT DISTINCT interaction_type FROM episodes WHERE timestamp >= ?",
                    (cutoff.isoformat(),),
                ).fetchall()
            result["distinct_interaction_types"] = [row[0] for row in rows if row[0]]
        except Exception as e:
            logger.warning("get_diagnostics: distinct_interaction_types query failed: %s", e)
            result["distinct_interaction_types"] = {"error": str(e)}

        # 5. Episodes per active day
        try:
            ep_count = result["episode_count"] if isinstance(result["episode_count"], int) else 0
            ad = result["active_days"] if isinstance(result["active_days"], int) else 1
            result["episodes_per_day"] = round(ep_count / max(1, ad), 2)
        except Exception as e:
            logger.warning("get_diagnostics: episodes_per_day computation failed: %s", e)
            result["episodes_per_day"] = {"error": str(e)}

        # 6. Time bucket distribution (episodes per time-of-day bucket)
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    "SELECT timestamp FROM episodes WHERE timestamp >= ?",
                    (cutoff.isoformat(),),
                ).fetchall()
            buckets: dict[str, int] = {"morning": 0, "midday": 0, "afternoon": 0, "evening": 0, "night": 0}
            for (ts_str,) in rows:
                try:
                    dt_utc = datetime.fromisoformat(ts_str)
                    if dt_utc.tzinfo is None:
                        dt_utc = dt_utc.replace(tzinfo=UTC)
                    dt_local = dt_utc.astimezone(self._tz)
                    bucket = self._hour_to_bucket(dt_local.hour)
                    buckets[bucket] += 1
                except (ValueError, TypeError):
                    continue
            result["time_bucket_distribution"] = buckets
        except Exception as e:
            logger.warning("get_diagnostics: time_bucket_distribution query failed: %s", e)
            result["time_bucket_distribution"] = {"error": str(e)}

        # 7. Usable episode count (episodes surviving the internal-type filter)
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    f"""SELECT COUNT(*) FROM episodes
                        WHERE timestamp >= ?
                          AND interaction_type IS NOT NULL
                          AND interaction_type NOT IN ('unknown', 'communication')
                          {self.INTERNAL_TYPE_SQL_FILTER}""",
                    (cutoff.isoformat(),),
                ).fetchone()
            result["usable_episode_count"] = row[0] if row else 0
        except Exception as e:
            logger.warning("get_diagnostics: usable_episode_count query failed: %s", e)
            result["usable_episode_count"] = {"error": str(e)}

        # 8. Top interaction types by count
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT interaction_type, COUNT(*) as cnt FROM episodes
                       WHERE timestamp >= ?
                         AND interaction_type IS NOT NULL
                       GROUP BY interaction_type
                       ORDER BY cnt DESC
                       LIMIT 20""",
                    (cutoff.isoformat(),),
                ).fetchall()
            result["interaction_type_counts"] = {row[0]: row[1] for row in rows}
        except Exception as e:
            logger.warning("get_diagnostics: interaction_type_counts query failed: %s", e)
            result["interaction_type_counts"] = {"error": str(e)}

        # 9. Candidate pairs and pairs meeting min_occurrences
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    f"""SELECT timestamp, interaction_type FROM episodes
                        WHERE timestamp >= ?
                          AND interaction_type IS NOT NULL
                          AND interaction_type NOT IN ('unknown', 'communication')
                          {self.INTERNAL_TYPE_SQL_FILTER}""",
                    (cutoff.isoformat(),),
                ).fetchall()
            bucket_day_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
            for ts_str, interaction_type in rows:
                try:
                    dt_utc = datetime.fromisoformat(ts_str)
                    if dt_utc.tzinfo is None:
                        dt_utc = dt_utc.replace(tzinfo=UTC)
                    dt_local = dt_utc.astimezone(self._tz)
                    bucket = self._hour_to_bucket(dt_local.hour)
                    local_date = dt_local.strftime("%Y-%m-%d")
                    bucket_day_sets[(bucket, interaction_type)].add(local_date)
                except (ValueError, TypeError):
                    continue
            result["candidate_pairs_count"] = len(bucket_day_sets)
            result["pairs_meeting_min_occurrences"] = sum(
                1 for days in bucket_day_sets.values() if len(days) >= self.min_occurrences
            )
        except Exception as e:
            logger.warning("get_diagnostics: candidate_pairs query failed: %s", e)
            result["candidate_pairs_count"] = {"error": str(e)}
            result["pairs_meeting_min_occurrences"] = {"error": str(e)}

        # 10. Stored routines count
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute("SELECT COUNT(*) FROM routines").fetchone()
            result["stored_routines_count"] = row[0] if row else 0
        except Exception as e:
            logger.warning("get_diagnostics: stored_routines_count query failed: %s", e)
            result["stored_routines_count"] = {"error": str(e)}

        # 8. Last detection result (cached from most recent detect_routines() call)
        result["last_detection_count"] = self._last_detection_count
        result["last_detection_time"] = self._last_detection_time

        # 9. Health indicator
        if self._last_detection_count is None:
            result["health"] = "no_data"
        elif isinstance(result.get("episode_count"), int) and result["episode_count"] == 0:
            result["health"] = "no_data"
        elif self._last_detection_count == 0 and isinstance(result.get("episode_count"), int) and result["episode_count"] > 0:
            result["health"] = "degraded"
        else:
            result["health"] = "ok"

        return result

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
        try:
            with self.db.get_connection("user_model") as conn:
                # Fetch raw timestamps and convert to local timezone before
                # extracting dates.  This matches the bucketing logic in
                # _detect_temporal_routines() which converts to self._tz
                # before computing local dates.  The old approach used
                # DATE(timestamp) which operates in UTC and miscounts days
                # near the UTC midnight boundary for non-UTC timezones.
                rows = conn.execute(
                    f"""SELECT DISTINCT timestamp FROM episodes
                        WHERE timestamp > ?
                          AND interaction_type IS NOT NULL
                          AND interaction_type NOT IN ('unknown', 'communication')
                          {self.INTERNAL_TYPE_SQL_FILTER}""",
                    (cutoff.isoformat(),),
                ).fetchall()
            local_dates: set[str] = set()
            for row in rows:
                if row[0]:
                    try:
                        dt_utc = datetime.fromisoformat(row[0])
                        if dt_utc.tzinfo is None:
                            dt_utc = dt_utc.replace(tzinfo=UTC)
                        dt_local = dt_utc.astimezone(self._tz)
                        local_dates.add(dt_local.strftime("%Y-%m-%d"))
                    except (ValueError, TypeError):
                        continue
            return max(1, len(local_dates))
        except sqlite3.DatabaseError as e:
            logger.warning("_count_active_days: user_model.db query failed: %s", e)
            return 1

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
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT interaction_type, timestamp
                       FROM episodes
                       WHERE timestamp > ? AND interaction_type IS NOT NULL
                       ORDER BY timestamp""",
                    (cutoff.isoformat(),),
                ).fetchall()

            # Group episodes by local date (using self._tz) instead of
            # SQL-level DATE() which operates in UTC.  Same fix as
            # _count_active_days() — episodes near the UTC midnight boundary
            # were grouped into the wrong day, corrupting gap calculations.
            day_episodes: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
            for row in rows:
                itype, ts_str = row[0], row[1]
                try:
                    dt_utc = datetime.fromisoformat(ts_str)
                    if dt_utc.tzinfo is None:
                        dt_utc = dt_utc.replace(tzinfo=UTC)
                    dt_local = dt_utc.astimezone(self._tz)
                    local_date = dt_local.strftime("%Y-%m-%d")
                    day_episodes[local_date].append((itype, dt_utc))
                except (ValueError, TypeError):
                    continue

            # Compute average gap from each interaction type to the next
            # episode on the same local day.
            gap_sums: dict[str, float] = defaultdict(float)
            gap_counts: dict[str, int] = defaultdict(int)
            for _day, episodes in day_episodes.items():
                # Episodes are already sorted by timestamp from the SQL query
                for i in range(len(episodes) - 1):
                    itype = episodes[i][0]
                    gap_minutes = (episodes[i + 1][1] - episodes[i][1]).total_seconds() / 60.0
                    gap_sums[itype] += gap_minutes
                    gap_counts[itype] += 1

            return {
                itype: gap_sums[itype] / gap_counts[itype]
                for itype in gap_counts
            }
        except sqlite3.DatabaseError as e:
            logger.warning("_compute_step_duration_map: user_model.db query failed: %s", e)
            return {}

    def _derive_interaction_type_from_event(self, event_id: str) -> str | None:
        """Derive an interaction type from the linked event when episode has no usable type.

        Looks up the original event by event_id in the events database and maps
        the event's ``type`` field (e.g. 'email.received') to a routine-friendly
        interaction type string (e.g. 'email_received').

        This fallback ensures episodes created before the granular classification
        was introduced still contribute to routine detection.

        Args:
            event_id: The event ID linking the episode to its source event.

        Returns:
            A derived interaction type string, or None if the event cannot be found
            or its type cannot be mapped.
        """
        try:
            with self.db.get_connection("events") as conn:
                row = conn.execute("SELECT type FROM events WHERE id = ?", (event_id,)).fetchone()
            if not row or not row[0]:
                return None
            # Convert dotted event type to underscored interaction type
            # e.g. "email.received" → "email_received"
            return row[0].replace(".", "_")
        except Exception:
            return None

    def _fallback_temporal_episodes(self, cutoff: datetime) -> list[tuple[str, str]]:
        """Fallback query for temporal detection when no episodes have usable interaction_type.

        Re-queries episodes WITHOUT the ``interaction_type IS NOT NULL`` filter,
        then derives an interaction type for each row from the linked event's
        ``type`` field in the events database.  Rows whose event_id cannot be
        resolved or whose derived type is empty are skipped.

        This handles the common production scenario where episodes were created
        before the granular classification logic was deployed, leaving
        interaction_type as NULL, 'unknown', or the old generic 'communication'.

        Args:
            cutoff: Only include episodes after this timestamp.

        Returns:
            List of (timestamp, derived_interaction_type) tuples, matching
            the format expected by the caller.
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT timestamp, event_id, interaction_type
                   FROM episodes
                   WHERE timestamp > ?
                   ORDER BY timestamp""",
                (cutoff.isoformat(),),
            ).fetchall()

        if not rows:
            return []

        logger.info(
            "Temporal detection fallback: %d total episodes in window (ignoring interaction_type filter)",
            len(rows),
        )

        results: list[tuple[str, str]] = []
        for ts, event_id, existing_type in rows:
            # Use existing type if it's non-null and not a useless placeholder
            if existing_type and existing_type not in (None, "unknown", "communication"):
                # Skip internal telemetry types
                if any(existing_type.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                    continue
                results.append((ts, existing_type))
                continue
            # Otherwise derive from the linked event
            derived = self._derive_interaction_type_from_event(event_id)
            if derived and not any(derived.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                results.append((ts, derived))

        logger.info(
            "Temporal detection fallback: %d episodes recovered with derived interaction types",
            len(results),
        )
        return results

    def _fallback_location_episodes(self, cutoff: datetime) -> list[tuple[str, str, str]]:
        """Fallback query for location detection when no episodes have usable interaction_type.

        Re-queries episodes that have a non-NULL location WITHOUT the
        ``interaction_type IS NOT NULL`` filter, then derives an interaction type
        for each row from the linked event's ``type`` field in the events database.
        Rows whose event_id cannot be resolved or whose derived type is empty are
        skipped.

        This handles the common production scenario where episodes were created
        before the granular classification logic was deployed, leaving
        interaction_type as NULL, 'unknown', or the old generic 'communication'.

        Args:
            cutoff: Only include episodes after this timestamp.

        Returns:
            List of (location, derived_interaction_type, timestamp) tuples.
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT location, interaction_type, event_id, timestamp
                   FROM episodes
                   WHERE timestamp > ?
                     AND location IS NOT NULL
                   ORDER BY timestamp""",
                (cutoff.isoformat(),),
            ).fetchall()

        if not rows:
            return []

        logger.info(
            "Location detection fallback: %d total episodes with location in window (ignoring interaction_type filter)",
            len(rows),
        )

        results: list[tuple[str, str, str]] = []
        for location, existing_type, event_id, ts in rows:
            # Use existing type if it's non-null and not a useless placeholder
            if existing_type and existing_type not in (None, "unknown", "communication"):
                # Skip internal telemetry types
                if any(existing_type.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                    continue
                results.append((location, existing_type, ts))
                continue
            # Otherwise derive from the linked event
            derived = self._derive_interaction_type_from_event(event_id)
            if derived and not any(derived.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                results.append((location, derived, ts))

        logger.info(
            "Location detection fallback: %d episodes recovered with derived interaction types",
            len(results),
        )
        return results

    def _fallback_event_triggered_episodes(self, cutoff: datetime) -> list[tuple[str, str]]:
        """Fallback query for event-triggered detection when no episodes have usable interaction_type.

        Re-queries episodes WITHOUT the ``interaction_type IS NOT NULL`` filter,
        then derives an interaction type for each row from the linked event's
        ``type`` field in the events database.  Rows whose event_id cannot be
        resolved or whose derived type is empty are skipped.

        This handles the common production scenario where episodes were created
        before the granular classification logic was deployed, leaving
        interaction_type as NULL, 'unknown', or the old generic 'communication'.

        Args:
            cutoff: Only include episodes after this timestamp.

        Returns:
            List of (interaction_type, timestamp) tuples.
        """
        with self.db.get_connection("user_model") as conn:
            rows = conn.execute(
                """SELECT interaction_type, event_id, timestamp
                   FROM episodes
                   WHERE timestamp > ?
                   ORDER BY timestamp""",
                (cutoff.isoformat(),),
            ).fetchall()

        if not rows:
            return []

        logger.info(
            "Event-triggered detection fallback: %d total episodes in window (ignoring interaction_type filter)",
            len(rows),
        )

        results: list[tuple[str, str]] = []
        for existing_type, event_id, ts in rows:
            # Use existing type if it's non-null and not a useless placeholder
            if existing_type and existing_type not in (None, "unknown", "communication"):
                # Skip internal telemetry types
                if any(existing_type.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                    continue
                results.append((existing_type, ts))
                continue
            # Otherwise derive from the linked event
            derived = self._derive_interaction_type_from_event(event_id)
            if derived and not any(derived.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                results.append((derived, ts))

        logger.info(
            "Event-triggered detection fallback: %d episodes recovered with derived interaction types",
            len(results),
        )
        return results

    def _fallback_follow_up_actions(
        self,
        trigger_timestamps: list[str],
        trigger_type: str,
        all_fallback_rows: list[tuple[str, str]],
    ) -> list[tuple[str, int]]:
        """Find follow-up actions for fallback-derived trigger episodes.

        When trigger episodes have NULL/unknown stored interaction_type (so the
        standard SQL JOIN can't match them by type), this method uses the known
        trigger timestamps and the full set of derived (type, timestamp) pairs
        from the fallback to find follow-up patterns entirely in Python.

        This avoids re-querying the DB (where interaction types are still
        NULL/unknown) and uses the already-derived types from the fallback.

        Args:
            trigger_timestamps: ISO timestamps of the trigger episodes.
            trigger_type: The derived interaction type (used only for exclusion).
            all_fallback_rows: All (derived_type, timestamp) pairs from the
                fallback, including both triggers and potential follow-ups.

        Returns:
            List of (follow_up_interaction_type, day_count) tuples, matching
            the format of the standard follow-up query result.
        """
        from datetime import datetime as _dt

        # Parse trigger timestamps into (date_str, datetime) pairs for matching.
        # Normalize all datetimes to UTC-aware to avoid "can't compare
        # offset-naive and offset-aware datetimes" when timestamps have
        # mixed tz formats (some with +00:00 suffix, some without).
        trigger_parsed: list[tuple[str, _dt]] = []
        for ts in trigger_timestamps:
            try:
                dt = _dt.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                trigger_parsed.append((ts[:10], dt))
            except (ValueError, TypeError):
                continue

        # Parse all fallback rows into (type, date_str, datetime) for matching
        all_parsed: list[tuple[str, str, _dt]] = []
        for itype, ts in all_fallback_rows:
            try:
                dt = _dt.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                all_parsed.append((itype, ts[:10], dt))
            except (ValueError, TypeError):
                continue

        # For each trigger, find follow-up types within 2 hours on the same day
        follow_up_day_sets: dict[str, set[str]] = defaultdict(set)
        two_hours = timedelta(hours=2)

        for trig_date, trig_dt in trigger_parsed:
            for ftype, fdate, fdt in all_parsed:
                if ftype == trigger_type:
                    continue
                if fdate != trig_date:
                    continue
                if fdt > trig_dt and (fdt - trig_dt) < two_hours:
                    follow_up_day_sets[ftype].add(fdate)

        # Filter to follow-up types meeting min_occurrences
        results = [
            (follow_type, len(days))
            for follow_type, days in follow_up_day_sets.items()
            if len(days) >= self.min_occurrences
        ]
        results.sort(key=lambda x: -x[1])
        return results

    def _detect_temporal_routines(self, lookback_days: int) -> list[dict[str, Any]]:
        """Detect routines that occur at similar times each day.

        Groups episodes into time-of-day buckets (morning: 5–10am, midday: 11am–1pm,
        afternoon: 2–4pm, evening: 5–10pm, night: 11pm–4am) and looks for recurring
        action sequences.

        Timestamps are converted from UTC to the user's configured local timezone
        (``self._tz``) before extracting the hour, so that a 7 AM local activity
        stored as 12:00 UTC is correctly bucketed as 'morning' rather than 'midday'.
        The bucketing is performed in pure Python via ``zoneinfo.ZoneInfo`` for
        reliable cross-timezone behaviour (SQLite's timezone support is limited).

        Consistency is measured as (avg occurrences per active day) rather than
        (avg occurrences / full lookback window).  The full-window denominator
        systematically underestimates consistency when data spans only a fraction
        of the lookback period — e.g., 10 days of perfect data in a 30-day window
        would score 0.33 instead of 1.0.

        Step durations are measured from actual observed inter-episode gaps via
        ``_compute_step_duration_map()``, with a 5-minute fallback for types
        where no successor was observed, and a 15-minute default for the final
        step of every routine (no successor to measure against).

        When all episodes lack a usable ``interaction_type`` (NULL or 'unknown'),
        a fallback re-queries without the type filter and derives a classification
        from the linked event's ``type`` field via the events database.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of temporal routines
        """
        routines = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Number of distinct days with any episode data in the window.
        # This is the denominator for consistency: an action that fires on 8 of
        # 10 active days has 80% consistency, regardless of the lookback window.
        # Wrapped in try/except so a corrupted DB returns a safe default (1).
        try:
            active_days = self._count_active_days(cutoff)
        except Exception:
            logger.exception("_count_active_days failed in temporal detection; using default 1")
            active_days = 1

        # Fetch raw episode timestamps and interaction types so that we can
        # convert UTC → local timezone in Python before bucketing.
        # Exclude placeholder types ('unknown', 'communication') that provide
        # no signal for routine detection — they were set before the granular
        # classification logic was deployed.
        try:
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT timestamp, interaction_type
                    FROM episodes
                    WHERE timestamp > ?
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                    ORDER BY timestamp
                """,
                    (cutoff.isoformat(),),
                )
                raw_episodes = cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("_detect_temporal_routines: user_model.db query failed: %s", e)
            return []

        logger.info(
            "Temporal detection: %d episodes with usable interaction_type in lookback window",
            len(raw_episodes),
        )

        # Compute data_age_days from the earliest typed episode in the query results.
        # When raw_episodes is empty (full cold-start), default to 0 so the most
        # lenient tier (threshold=10) applies — we're definitely in cold-start.
        if raw_episodes:
            try:
                earliest_dt = datetime.fromisoformat(raw_episodes[0][0])
                if earliest_dt.tzinfo is None:
                    earliest_dt = earliest_dt.replace(tzinfo=UTC)
                data_age_days = (datetime.now(UTC) - earliest_dt).days
            except (ValueError, TypeError):
                data_age_days = 0
        else:
            # No typed episodes → cold-start assumption
            data_age_days = 0

        effective_min_episodes = self._effective_min_episodes(len(raw_episodes), data_age_days)

        # Fallback: if too few episodes have a usable interaction_type, re-query
        # WITHOUT the filter and derive classification from the linked event.
        # The threshold is adaptive (see _effective_min_episodes) so that cold-start
        # email-dominated installations trigger the fallback sooner.
        if len(raw_episodes) < effective_min_episodes:
            fallback_reason = "full fallback (0 primary)" if not raw_episodes else (
                f"supplemental fallback ({len(raw_episodes)} primary < {effective_min_episodes} threshold)"
            )
            try:
                fallback_episodes = self._fallback_temporal_episodes(cutoff)
                if fallback_episodes:
                    logger.info(
                        "Temporal detection: %s — recovered %d episodes via event_type derivation",
                        fallback_reason,
                        len(fallback_episodes),
                    )
                    # Merge primary and fallback, deduplicating by (timestamp, interaction_type)
                    seen = {(ts, it) for ts, it in raw_episodes}
                    for ep in fallback_episodes:
                        if ep not in seen:
                            raw_episodes.append(ep)
                            seen.add(ep)
            except Exception:
                logger.exception("Temporal detection: fallback query failed")

        if not raw_episodes:
            logger.info("Temporal detection: 0 usable episodes after fallback — skipping")
            return routines

        # Bucket episodes by local time-of-day. Track distinct local dates per
        # (bucket, interaction_type) pair so we can compute day_count.
        # Key: (time_bucket, interaction_type) → set of local date strings.
        bucket_day_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
        for ts_str, interaction_type in raw_episodes:
            try:
                # Parse the stored ISO 8601 timestamp and convert to local tz.
                dt_utc = datetime.fromisoformat(ts_str)
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=UTC)
                dt_local = dt_utc.astimezone(self._tz)
                bucket = self._hour_to_bucket(dt_local.hour)
                local_date = dt_local.strftime("%Y-%m-%d")
                bucket_day_sets[(bucket, interaction_type)].add(local_date)
            except (ValueError, TypeError):
                # Skip episodes with unparseable timestamps.
                continue

        logger.info(
            "Temporal detection: %d (bucket, type) pairs found",
            len(bucket_day_sets),
        )

        # Filter to (bucket, type) pairs meeting min_occurrences.
        hour_actions = [
            (bucket, itype, len(days))
            for (bucket, itype), days in bucket_day_sets.items()
            if len(days) >= self.min_occurrences
        ]
        # Sort by bucket then day_count descending to match previous output order.
        hour_actions.sort(key=lambda x: (x[0], -x[2]))

        logger.info(
            "Temporal detection: %d pairs meet min_occurrences=%d",
            len(hour_actions),
            self.min_occurrences,
        )

        if not hour_actions:
            return routines

        # Compute actual measured inter-step durations once, reuse for all buckets.
        # The final step of each routine falls back to LAST_STEP_DEFAULT_MINUTES
        # because there is no subsequent step to measure a gap against.
        # Wrapped in try/except so a corrupted DB returns an empty map.
        LAST_STEP_DEFAULT_MINUTES = 15.0
        try:
            step_duration_map = self._compute_step_duration_map(cutoff)
        except Exception:
            logger.exception("_compute_step_duration_map failed in temporal detection; using empty map")
            step_duration_map = {}

        # Group by time_bucket for routine construction.
        # Each tuple is (time_bucket, interaction_type, day_count).
        bucket_actions: dict[str, list] = defaultdict(list)
        for time_bucket, interaction_type, day_count in hour_actions:
            # Look up the measured gap duration; placeholder 5.0 if not available.
            duration = step_duration_map.get(interaction_type, 5.0)
            bucket_actions[time_bucket].append((interaction_type, day_count, duration))

        # Create routines for buckets with at least one recurring action.
        # A single consistent action at a fixed time (e.g., morning coffee at 7am)
        # is already a valid behavioral routine worth surfacing.
        for bucket_name, actions in bucket_actions.items():
            if len(actions) >= 1:
                # Sort by recurrence (most days first)
                actions.sort(key=lambda x: x[1], reverse=True)

                # Consistency = fraction of active days where the *most consistent*
                # action in this bucket appeared.  Using the max day_count means
                # "does the user reliably do SOMETHING in this time bucket?" rather
                # than "does the user reliably do EVERYTHING in this time bucket?".
                #
                # The old avg_day_count approach caused dominant patterns (e.g.
                # email_received on 39/39 days) to be dragged below the 0.6
                # threshold by rare co-occurring types (e.g. meeting_scheduled on
                # 3/39 days): avg=(39+3)/2=21, consistency=21/39=0.54 → FAIL.
                # With max: max=39, consistency=39/39=1.0 → PASS.
                max_day_count = max(dc for _, dc, _ in actions)
                consistency = min(1.0, max_day_count / active_days)

                # Pass the dominant (most-frequent) interaction type so that the
                # threshold can be lowered for high-volume passive types such as
                # email_received, whose arrival time is externally driven and
                # therefore won't achieve the default 0.6 consistency.
                dominant_type = actions[0][0] if actions else None
                effective_threshold = self._effective_consistency_threshold(active_days, dominant_type)
                is_cold_start = effective_threshold < self.consistency_threshold

                logger.info(
                    "Temporal routine detection: %d active days, effective threshold=%.2f (base=%.2f)",
                    active_days,
                    effective_threshold,
                    self.consistency_threshold,
                )
                logger.info(
                    "Temporal detection: bucket %s consistency=%.2f (threshold=%.2f) %s",
                    bucket_name,
                    consistency,
                    effective_threshold,
                    "PASS" if consistency >= effective_threshold else "FAIL",
                )

                if consistency >= effective_threshold:
                    steps = actions[:10]  # Cap at 10 steps
                    # Compute total routine duration: sum of measured gap durations
                    # for all-but-last step, plus the default for the last step.
                    step_durations = [d for _, _, d in steps]
                    if step_durations:
                        # Last step has no measured gap to a successor, so use default.
                        step_durations[-1] = LAST_STEP_DEFAULT_MINUTES
                    total_duration = sum(step_durations) if step_durations else LAST_STEP_DEFAULT_MINUTES

                    # Scale confidence down for cold-start detections that would
                    # have failed the full base threshold.  This signals to
                    # downstream consumers that the routine is provisional.
                    would_fail_base = consistency < self.consistency_threshold
                    confidence = consistency * 0.7 if (is_cold_start and would_fail_base) else consistency

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
                        "consistency_score": confidence,
                        "times_observed": int(max_day_count),
                        "variations": [],  # Could add variation detection in future
                        "cold_start": is_cold_start and would_fail_base,
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected %s routine with %d steps, consistency %.2f",
                        bucket_name,
                        len(actions),
                        consistency,
                    )

        return routines

    def _get_temporal_profile_sample_count(self) -> int:
        """Return the sample count from the temporal signal profile.

        Used to decide whether the primary temporal detection has enough
        upstream data to classify episodes reliably.  When the temporal
        profile has fewer than 25 samples, the episode-based fallback
        should be tried instead.

        Returns:
            Number of samples in the temporal signal profile, or 0 if
            the profile is missing or the query fails.
        """
        try:
            profile = self.user_model_store.get_signal_profile("temporal")
            if profile and "samples_count" in profile:
                return int(profile["samples_count"])
            return 0
        except Exception:
            logger.warning("Failed to read temporal signal profile sample count")
            return 0

    def _detect_routines_from_episodes_fallback(self, lookback_days: int) -> list[dict[str, Any]]:
        """Fallback routine detection that works directly from raw episode data.

        When signal profiles are unavailable (e.g., due to user_model.db
        corruption or insufficient upstream data), the primary temporal
        detection may return 0 results despite thousands of episodes.

        This method bypasses signal profiles entirely by:
        1. Querying episodes directly from user_model.db
        2. Classifying episode types using the event_type field from the
           linked event record via the EVENT_TYPE_TO_ACTIVITY mapping
        3. Grouping episodes by time-of-day bucket and activity type
        4. Applying the same consistency threshold as primary detection

        The output format matches ``_detect_temporal_routines()`` so that
        routines from either path can be stored and consumed identically.

        Args:
            lookback_days: Days of history to analyze

        Returns:
            List of detected routines in the same format as
            ``_detect_temporal_routines()``.
        """
        routines: list[dict[str, Any]] = []
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Count active days for consistency normalization
        try:
            active_days = self._count_active_days(cutoff)
        except Exception:
            logger.exception("_count_active_days failed in episode fallback; using default 1")
            active_days = 1

        # Fetch episodes with their linked event_ids
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT timestamp, event_id, interaction_type
                       FROM episodes
                       WHERE timestamp > ?
                       ORDER BY timestamp""",
                    (cutoff.isoformat(),),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("Episode fallback: user_model.db query failed: %s", e)
            return []

        if not rows:
            logger.info("Episode fallback: 0 episodes in lookback window")
            return []

        logger.info("Episode fallback: %d episodes in lookback window", len(rows))

        # Classify each episode — prefer existing interaction_type, fall back
        # to EVENT_TYPE_TO_ACTIVITY mapping via the linked event's type field.
        classified: list[tuple[str, str]] = []  # (timestamp, activity_type)
        for ts, event_id, existing_type in rows:
            # Use existing type if it's meaningful and not internal telemetry
            if existing_type and existing_type not in (None, "unknown", "communication"):
                if any(existing_type.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                    continue
                classified.append((ts, existing_type))
                continue

            # Derive interaction type from the linked event using the same
            # dot-to-underscore conversion as the primary temporal detection,
            # NOT the incompatible EVENT_TYPE_TO_ACTIVITY mapping.
            activity = self._derive_interaction_type_from_event(event_id)
            if activity:
                # Apply the same internal-type filter as the existing-type path
                if any(activity.startswith(p) for p in self.INTERNAL_TYPE_PREFIXES):
                    continue
                classified.append((ts, activity))

        if not classified:
            logger.info("Episode fallback: 0 episodes could be classified")
            return []

        logger.info("Episode fallback: %d episodes classified", len(classified))

        # Bucket by time-of-day and activity type — same logic as primary detection
        bucket_day_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
        for ts_str, activity_type in classified:
            try:
                dt_utc = datetime.fromisoformat(ts_str)
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=UTC)
                dt_local = dt_utc.astimezone(self._tz)
                bucket = self._hour_to_bucket(dt_local.hour)
                local_date = dt_local.strftime("%Y-%m-%d")
                bucket_day_sets[(bucket, activity_type)].add(local_date)
            except (ValueError, TypeError):
                continue

        # Filter to pairs meeting min_occurrences
        hour_actions = [
            (bucket, atype, len(days))
            for (bucket, atype), days in bucket_day_sets.items()
            if len(days) >= self.min_occurrences
        ]
        hour_actions.sort(key=lambda x: (x[0], -x[2]))

        if not hour_actions:
            logger.info("Episode fallback: no (bucket, type) pairs meet min_occurrences=%d", self.min_occurrences)
            return []

        # Compute step durations
        LAST_STEP_DEFAULT_MINUTES = 15.0
        try:
            step_duration_map = self._compute_step_duration_map(cutoff)
        except Exception:
            logger.exception("Episode fallback: _compute_step_duration_map failed; using empty map")
            step_duration_map = {}

        # Group by time bucket and build routines
        bucket_actions: dict[str, list] = defaultdict(list)
        for time_bucket, activity_type, day_count in hour_actions:
            duration = step_duration_map.get(activity_type, 5.0)
            bucket_actions[time_bucket].append((activity_type, day_count, duration))

        effective_threshold = self._effective_consistency_threshold(active_days)
        is_cold_start = effective_threshold < self.consistency_threshold

        for bucket_name, actions in bucket_actions.items():
            if len(actions) >= 1:
                actions.sort(key=lambda x: x[1], reverse=True)

                # Consistency = fraction of active days where the *most consistent*
                # action in this bucket appeared.  Using max day_count answers
                # "does the user reliably do SOMETHING here?" rather than
                # "does the user reliably do EVERYTHING here?".
                # (Same fix as in _detect_temporal_routines — see that docstring.)
                max_day_count = max(dc for _, dc, _ in actions)
                consistency = min(1.0, max_day_count / active_days)

                logger.info(
                    "Episode fallback: bucket %s consistency=%.2f (threshold=%.2f) %s",
                    bucket_name,
                    consistency,
                    effective_threshold,
                    "PASS" if consistency >= effective_threshold else "FAIL",
                )

                if consistency >= effective_threshold:
                    steps = actions[:10]
                    step_durations = [d for _, _, d in steps]
                    if step_durations:
                        step_durations[-1] = LAST_STEP_DEFAULT_MINUTES
                    total_duration = sum(step_durations) if step_durations else LAST_STEP_DEFAULT_MINUTES

                    would_fail_base = consistency < self.consistency_threshold
                    confidence = consistency * 0.7 if (is_cold_start and would_fail_base) else consistency

                    routine = {
                        "name": f"{bucket_name.capitalize()} routine",
                        "trigger": bucket_name,
                        "steps": [
                            {
                                "order": i,
                                "action": action,
                                "typical_duration_minutes": dur,
                                "skip_rate": max(0.0, 1.0 - (dc / active_days)),
                            }
                            for i, (action, dc, dur) in enumerate(steps)
                        ],
                        "typical_duration_minutes": total_duration,
                        "consistency_score": confidence,
                        "times_observed": int(max_day_count),
                        "variations": [],
                        "cold_start": is_cold_start and would_fail_base,
                        "detection_method": "episode_fallback",
                    }
                    routines.append(routine)
                    logger.info(
                        "Episode fallback: detected %s routine with %d steps, consistency %.2f",
                        bucket_name,
                        len(steps),
                        consistency,
                    )

        logger.info(
            "Episode fallback: evaluated %d (bucket, type) pairs, %d passed min_occurrences, produced %d routines",
            len(bucket_day_sets),
            len(hour_actions),
            len(routines),
        )

        return routines

    def _classify_event_type_to_activity(self, event_id: str) -> str | None:
        """Map an episode's linked event type to an activity classification.

        Looks up the event by ID in the events database and maps its ``type``
        field to an activity name using the ``EVENT_TYPE_TO_ACTIVITY`` mapping.
        Falls back to prefix matching if an exact match isn't found.

        Args:
            event_id: The event ID linked from the episode.

        Returns:
            Activity classification string, or None if unmappable.
        """
        if not event_id:
            return None
        try:
            with self.db.get_connection("events") as conn:
                row = conn.execute(
                    "SELECT type FROM events WHERE id = ?",
                    (event_id,),
                ).fetchone()
            if not row or not row[0]:
                return None

            event_type = row[0]

            # Exact match first
            if event_type in EVENT_TYPE_TO_ACTIVITY:
                return EVENT_TYPE_TO_ACTIVITY[event_type]

            # Prefix match: e.g. "email.received.important" → "email.received"
            for prefix, activity in EVENT_TYPE_TO_ACTIVITY.items():
                if event_type.startswith(prefix):
                    return activity

            return None
        except Exception:
            return None

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
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Reuse active-day count computed once for the whole detection pass.
        # Wrapped in try/except so a corrupted DB returns a safe default (1).
        try:
            active_days = self._count_active_days(cutoff)
        except Exception:
            logger.exception("_count_active_days failed in location detection; using default 1")
            active_days = 1

        # Use actual measured inter-step durations instead of a hardcoded
        # placeholder.  Falls back to 5.0 minutes for interaction types where
        # no same-day successor was observed.
        # Wrapped in try/except so a corrupted DB returns an empty map.
        try:
            step_duration_map = self._compute_step_duration_map(cutoff)
        except Exception:
            logger.exception("_compute_step_duration_map failed in location detection; using empty map")
            step_duration_map = {}
        STEP_DURATION_FALLBACK = 5.0

        # Fetch recurring (location, interaction_type) pairs.
        # day_count = distinct local-timezone days where this action occurred at
        # this location.  We fetch raw timestamps and group by local date in
        # Python to avoid the DATE() UTC-midnight bug (see PR #640).
        try:
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        location,
                        interaction_type,
                        timestamp
                    FROM episodes
                    WHERE timestamp > ?
                      AND location IS NOT NULL
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                """,
                    (cutoff.isoformat(),),
                )
                raw_rows = cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("_detect_location_routines: user_model.db query failed: %s", e)
            return []

        # Group by (location, interaction_type) and count distinct local days.
        loc_type_days: dict[tuple[str, str], set[str]] = defaultdict(set)
        for location, itype, ts in raw_rows:
            try:
                dt_utc = datetime.fromisoformat(ts)
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=UTC)
                dt_local = dt_utc.astimezone(self._tz)
                loc_type_days[(location, itype)].add(dt_local.strftime("%Y-%m-%d"))
            except (ValueError, TypeError):
                continue

        # Apply min_occurrences threshold and build result tuples.
        location_actions = [
            (loc, itype, len(days))
            for (loc, itype), days in loc_type_days.items()
            if len(days) >= self.min_occurrences
        ]
        location_actions.sort(key=lambda x: (x[0], -x[2]))

        logger.info(
            "Location detection: %d (location, type) pairs meet min_occurrences=%d",
            len(location_actions) if location_actions else 0,
            self.min_occurrences,
        )

        # Fallback: if too few (location, type) pairs have a usable interaction_type,
        # re-query WITHOUT the filter and derive classification from the linked event.
        # Location actions are already aggregated, so use a lower threshold of 3.
        min_location_pairs = 3
        if len(location_actions) < min_location_pairs:
            fallback_reason = "full fallback (0 primary)" if not location_actions else (
                f"supplemental fallback ({len(location_actions)} primary < {min_location_pairs} threshold)"
            )
            try:
                fallback_rows = self._fallback_location_episodes(cutoff)
                if fallback_rows:
                    logger.info(
                        "Location detection: %s — recovered %d episodes via event_type derivation",
                        fallback_reason,
                        len(fallback_rows),
                    )
                    # Re-aggregate by (location, interaction_type) with local-tz day counts.
                    # Convert timestamps to local dates to avoid UTC-midnight grouping bug.
                    loc_type_days: dict[tuple[str, str], set[str]] = defaultdict(set)
                    for location, itype, ts in fallback_rows:
                        try:
                            dt_utc = datetime.fromisoformat(ts)
                            if dt_utc.tzinfo is None:
                                dt_utc = dt_utc.replace(tzinfo=UTC)
                            dt_local = dt_utc.astimezone(self._tz)
                            loc_type_days[(location, itype)].add(dt_local.strftime("%Y-%m-%d"))
                        except (ValueError, TypeError):
                            continue
                    # Build fallback actions and merge with primary
                    fallback_actions = [
                        (loc, itype, len(days))
                        for (loc, itype), days in loc_type_days.items()
                        if len(days) >= self.min_occurrences
                    ]
                    # Merge: add fallback pairs not already in primary results
                    existing_pairs = {(loc, itype) for loc, itype, _ in location_actions}
                    for loc, itype, day_count in fallback_actions:
                        if (loc, itype) not in existing_pairs:
                            location_actions.append((loc, itype, day_count))
                            existing_pairs.add((loc, itype))
                    location_actions.sort(key=lambda x: (x[0], -x[2]))
                    logger.info(
                        "Location detection: after merge %d (location, type) pairs meeting min_occurrences=%d",
                        len(location_actions),
                        self.min_occurrences,
                    )
            except Exception:
                logger.exception("Location detection: fallback query failed")

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

                effective_threshold = self._effective_consistency_threshold(active_days)
                is_cold_start = effective_threshold < self.consistency_threshold

                logger.info(
                    "Location routine detection: %d active days, effective threshold=%.2f (base=%.2f)",
                    active_days,
                    effective_threshold,
                    self.consistency_threshold,
                )
                logger.info(
                    "Location detection: %s consistency=%.2f (threshold=%.2f) %s",
                    location,
                    consistency,
                    effective_threshold,
                    "PASS" if consistency >= effective_threshold else "FAIL",
                )

                if consistency >= effective_threshold:
                    # Scale confidence down for cold-start detections that would
                    # have failed the full base threshold.
                    would_fail_base = consistency < self.consistency_threshold
                    confidence = consistency * 0.7 if (is_cold_start and would_fail_base) else consistency

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
                        "consistency_score": confidence,
                        "times_observed": int(avg_day_count),
                        "variations": [],
                        "cold_start": is_cold_start and would_fail_base,
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected location routine for %s with %d steps, consistency %.2f",
                        location,
                        len(actions),
                        consistency,
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
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)

        # Look for interaction types that occur on enough distinct local-tz days
        # to be candidates for routine triggers.  We fetch raw timestamps and
        # group by local date in Python to avoid the DATE() UTC-midnight bug
        # (see PR #640).
        try:
            with self.db.get_connection("user_model") as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    SELECT
                        interaction_type,
                        timestamp
                    FROM episodes
                    WHERE timestamp > ?
                      AND interaction_type IS NOT NULL
                      AND interaction_type NOT IN ('unknown', 'communication')
                      {self.INTERNAL_TYPE_SQL_FILTER}
                """,
                    (cutoff.isoformat(),),
                )
                trigger_raw_rows = cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logger.warning("_detect_event_triggered_routines: user_model.db trigger query failed: %s", e)
            return []

        # Group by interaction_type and count distinct local days.
        trigger_type_days: dict[str, set[str]] = defaultdict(set)
        for itype, ts in trigger_raw_rows:
            try:
                dt_utc = datetime.fromisoformat(ts)
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=UTC)
                dt_local = dt_utc.astimezone(self._tz)
                trigger_type_days[itype].add(dt_local.strftime("%Y-%m-%d"))
            except (ValueError, TypeError):
                continue

        # Apply min_occurrences threshold and build result tuples.
        trigger_events = [
            (itype, len(days))
            for itype, days in trigger_type_days.items()
            if len(days) >= self.min_occurrences
        ]
        trigger_events.sort(key=lambda x: -x[1])

        logger.info(
            "Event-triggered detection: %d candidate trigger types meet min_occurrences=%d",
            len(trigger_events) if trigger_events else 0,
            self.min_occurrences,
        )

        # Fallback: if no episodes have a usable interaction_type, re-query
        # WITHOUT the filter and derive classification from the linked event.
        # Track timestamps per derived type so the follow-up query can find
        # trigger episodes by timestamp (since their stored interaction_type
        # is NULL/unknown, the standard follow-up JOIN won't match them).
        # Also keep the full fallback rows for the follow-up matching.
        # Fallback: if too few trigger event types have a usable interaction_type,
        # re-query WITHOUT the filter and derive classification from the linked event.
        # Use a lower threshold of 3 since trigger_events are already aggregated by type.
        min_trigger_types = 3
        fallback_trigger_timestamps: dict[str, list[str]] = {}
        all_fallback_rows: list[tuple[str, str]] = []
        if len(trigger_events) < min_trigger_types:
            fallback_reason = "full fallback (0 primary)" if not trigger_events else (
                f"supplemental fallback ({len(trigger_events)} primary < {min_trigger_types} threshold)"
            )
            try:
                all_fallback_rows = self._fallback_event_triggered_episodes(cutoff)
                if all_fallback_rows:
                    logger.info(
                        "Event-triggered detection: %s — recovered %d episodes via event_type derivation",
                        fallback_reason,
                        len(all_fallback_rows),
                    )
                    # Re-aggregate by interaction_type with local-tz day counts,
                    # and also store timestamps per type for the follow-up query.
                    # Convert timestamps to local dates to avoid UTC-midnight grouping bug.
                    type_days: dict[str, set[str]] = defaultdict(set)
                    type_timestamps: dict[str, list[str]] = defaultdict(list)
                    for itype, ts in all_fallback_rows:
                        try:
                            dt_utc = datetime.fromisoformat(ts)
                            if dt_utc.tzinfo is None:
                                dt_utc = dt_utc.replace(tzinfo=UTC)
                            dt_local = dt_utc.astimezone(self._tz)
                            type_days[itype].add(dt_local.strftime("%Y-%m-%d"))
                            type_timestamps[itype].append(ts)
                        except (ValueError, TypeError):
                            continue
                    # Build fallback trigger events and merge with primary
                    fallback_triggers = [
                        (itype, len(days))
                        for itype, days in type_days.items()
                        if len(days) >= self.min_occurrences
                    ]
                    # Merge: add fallback types not already in primary results
                    existing_types = {itype for itype, _ in trigger_events}
                    for itype, day_count in fallback_triggers:
                        if itype not in existing_types:
                            trigger_events.append((itype, day_count))
                            existing_types.add(itype)
                    trigger_events.sort(key=lambda x: -x[1])
                    # Only keep timestamps for fallback-derived types
                    fallback_trigger_timestamps = {
                        itype: type_timestamps[itype]
                        for itype, _ in fallback_triggers
                        if itype in {it for it, _ in trigger_events}
                    }
                    logger.info(
                        "Event-triggered detection: after merge %d candidate trigger types meeting min_occurrences=%d",
                        len(trigger_events),
                        self.min_occurrences,
                    )
            except Exception:
                logger.exception("Event-triggered detection: fallback query failed")

        # Compute measured inter-step durations once, shared across all trigger types.
        # Falls back to 5.0 minutes for interaction types where no same-day successor
        # was observed in the lookback window.
        # Wrapped in try/except so a corrupted DB returns an empty map.
        try:
            step_duration_map = self._compute_step_duration_map(cutoff)
        except Exception:
            logger.exception("_compute_step_duration_map failed in event-triggered detection; using empty map")
            step_duration_map = {}
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
            try:
                if interaction_type in fallback_trigger_timestamps:
                    # Fallback mode: trigger episodes have NULL/unknown stored
                    # interaction_type, so we use their known timestamps and the
                    # full fallback-derived data to find follow-ups in Python.
                    following_actions = self._fallback_follow_up_actions(
                        fallback_trigger_timestamps[interaction_type],
                        interaction_type,
                        all_fallback_rows,
                    )
                else:
                    # Fetch all trigger episodes and potential follow-up episodes,
                    # then pair them in Python using local-timezone dates to avoid
                    # the DATE() UTC-midnight bug (see PR #640).
                    with self.db.get_connection("user_model") as conn:
                        cursor = conn.cursor()
                        # Get trigger episodes
                        cursor.execute(
                            """
                            SELECT timestamp
                            FROM episodes
                            WHERE interaction_type = ?
                              AND timestamp > ?
                            """,
                            (interaction_type, cutoff.isoformat()),
                        )
                        trigger_rows = cursor.fetchall()

                        # Get all candidate follow-up episodes in the lookback window
                        cursor.execute(
                            """
                            SELECT interaction_type, timestamp
                            FROM episodes
                            WHERE interaction_type IS NOT NULL
                              AND interaction_type != ?
                              AND timestamp > ?
                            """,
                            (interaction_type, cutoff.isoformat()),
                        )
                        followup_rows = cursor.fetchall()

                    # Parse trigger timestamps and group by local date.
                    trigger_by_local_date: dict[str, list[datetime]] = defaultdict(list)
                    for (ts,) in trigger_rows:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            local_date = dt.astimezone(self._tz).strftime("%Y-%m-%d")
                            trigger_by_local_date[local_date].append(dt)
                        except (ValueError, TypeError):
                            continue

                    # Parse follow-up timestamps and group by local date.
                    followup_by_local_date: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
                    for f_itype, f_ts in followup_rows:
                        try:
                            dt = datetime.fromisoformat(f_ts)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            local_date = dt.astimezone(self._tz).strftime("%Y-%m-%d")
                            followup_by_local_date[local_date].append((f_itype, dt))
                        except (ValueError, TypeError):
                            continue

                    # For each follow-up type, count distinct local days where
                    # it occurred within 2 hours after a trigger on the same day.
                    followup_type_days: dict[str, set[str]] = defaultdict(set)
                    two_hours = timedelta(hours=2)
                    for local_date, triggers in trigger_by_local_date.items():
                        followups = followup_by_local_date.get(local_date, [])
                        for f_itype, f_dt in followups:
                            for t_dt in triggers:
                                if t_dt < f_dt <= t_dt + two_hours:
                                    followup_type_days[f_itype].add(local_date)
                                    break  # One match per follow-up per day is enough

                    # Apply min_occurrences threshold and build result tuples.
                    following_actions = [
                        (f_itype, len(days))
                        for f_itype, days in followup_type_days.items()
                        if len(days) >= self.min_occurrences
                    ]
                    following_actions.sort(key=lambda x: -x[1])
            except sqlite3.DatabaseError as e:
                logger.warning(
                    "_detect_event_triggered_routines: user_model.db follow-up query failed for %s: %s",
                    interaction_type,
                    e,
                )
                continue

            if len(following_actions) >= 1:  # At least 1 following step for a sequence
                avg_day_count = sum(dc for _, dc in following_actions) / len(following_actions)
                # Consistency = fraction of trigger-event days where the follow-up
                # actions also appeared.  Use days_occurred as the denominator
                # (number of days the trigger fired) rather than total active days.
                consistency = min(1.0, avg_day_count / days_occurred)

                # Event-triggered routines use days_occurred (trigger days) as
                # the maturity signal, since the trigger may not fire every day.
                effective_threshold = self._effective_consistency_threshold(days_occurred)
                is_cold_start = effective_threshold < self.consistency_threshold

                logger.info(
                    "Event-triggered routine detection: %d trigger days, effective threshold=%.2f (base=%.2f)",
                    days_occurred,
                    effective_threshold,
                    self.consistency_threshold,
                )
                logger.info(
                    "Event-triggered detection: after_%s consistency=%.2f (threshold=%.2f) %s",
                    interaction_type,
                    consistency,
                    effective_threshold,
                    "PASS" if consistency >= effective_threshold else "FAIL",
                )

                if consistency >= effective_threshold:
                    trigger_name = interaction_type.replace("_", " ").title()

                    # Attach measured durations to each following action.
                    # Replaces the old hardcoded 5.0-minute value with actual
                    # observed inter-step gaps for more accurate routine timing.
                    steps_with_duration = [
                        (action, dc, step_duration_map.get(action, STEP_DURATION_FALLBACK))
                        for action, dc in following_actions[:10]
                    ]
                    total_duration = sum(d for _, _, d in steps_with_duration)

                    # Scale confidence down for cold-start detections that would
                    # have failed the full base threshold.
                    would_fail_base = consistency < self.consistency_threshold
                    confidence = consistency * 0.7 if (is_cold_start and would_fail_base) else consistency

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
                        "consistency_score": confidence,
                        "times_observed": int(avg_day_count),
                        "variations": [],
                        "cold_start": is_cold_start and would_fail_base,
                    }
                    routines.append(routine)
                    logger.debug(
                        "Detected event-triggered routine after %s, consistency %.2f",
                        interaction_type,
                        consistency,
                    )

        return routines

    def prune_stale_routines(self, detected_routines: list[dict[str, Any]], max_stale_days: int = 14) -> int:
        """Remove stored routines that were not re-detected and are older than the stale threshold.

        A routine is considered stale when:
        1. It was NOT found in the current detection run (not in detected_routines), AND
        2. Its updated_at timestamp is older than max_stale_days ago.

        This prevents abandoned patterns from accumulating in the database and
        generating false routine_deviation predictions in the prediction engine.

        Args:
            detected_routines: List of routine dicts from the current detection run.
                Each must have a "name" key.
            max_stale_days: Number of days after which an un-redetected routine is
                pruned. Default 14 days.

        Returns:
            Number of routines pruned.
        """
        detected_names = {r["name"] for r in detected_routines}
        pruned = 0
        try:
            cutoff = (datetime.now(UTC) - timedelta(days=max_stale_days)).isoformat()
            with self.db.get_connection("user_model") as conn:
                stored = conn.execute("SELECT name, updated_at FROM routines").fetchall()
                for row in stored:
                    name = row["name"]
                    updated_at = row["updated_at"] or ""
                    if name not in detected_names and updated_at < cutoff:
                        conn.execute("DELETE FROM routines WHERE name = ?", (name,))
                        pruned += 1
                        logger.info(
                            "Pruned stale routine: %s (last updated %s, cutoff %s)",
                            name,
                            updated_at,
                            cutoff,
                        )
            if pruned:
                logger.info("Pruned %d stale routine(s) from database", pruned)
        except Exception as e:
            logger.warning("prune_stale_routines failed (non-critical): %s", e)
        return pruned

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
