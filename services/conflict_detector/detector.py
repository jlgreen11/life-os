"""
Life OS — Calendar Conflict Detector

Scans calendar events for scheduling overlaps and publishes
``calendar.conflict.detected`` events so the rules engine can
notify the user.

The default rule "High priority: calendar conflict" in the rules
engine listens for these events but nothing previously emitted them.
This service closes that gap.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detects overlapping calendar events and publishes conflict events.

    Constructor follows the project's dependency-injection pattern: accepts
    a ``DatabaseManager`` instance (no global singletons).
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        # Track already-published conflicts so we don't re-publish on every run.
        # Key is a frozenset of (event_a_id, event_b_id) to handle pair ordering.
        # Populated from state.db on init so restarts don't lose dedup state.
        self._published_conflicts: set[frozenset[str]] = set()
        # Tracks when the last automatic cleanup ran so we only clean once per day.
        self._last_cleanup: Optional[datetime] = None
        self._load_published_conflicts()

    def _load_published_conflicts(self):
        """Load previously-published conflict pairs from state.db.

        Populates the in-memory ``_published_conflicts`` set so that conflicts
        detected before a process restart are not re-published as new events.
        """
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    "SELECT event_id_a, event_id_b FROM published_conflicts WHERE source = 'conflict_detector'"
                ).fetchall()
                for row in rows:
                    self._published_conflicts.add(frozenset([row[0], row[1]]))
                logger.debug("Loaded %d published conflict pairs from state.db", len(rows))
        except Exception as e:
            logger.warning("Could not load published conflicts: %s", e)

    def _persist_conflict(self, pair_key: frozenset[str]):
        """Persist a newly-published conflict pair to state.db.

        Uses sorted ordering so (A,B) and (B,A) always map to the same row.
        INSERT OR IGNORE avoids errors on duplicate inserts.
        """
        ids = sorted(pair_key)
        try:
            with self.db.get_connection("state") as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO published_conflicts (event_id_a, event_id_b, source) VALUES (?, ?, ?)",
                    (ids[0], ids[1], "conflict_detector"),
                )
        except Exception as e:
            logger.warning("Could not persist conflict pair: %s", e)

    def cleanup_old_conflicts(self, days: int = 30) -> int:
        """Remove conflict pairs older than *days* from state.db.

        Prevents the published_conflicts table from growing without bound.
        Only removes entries from the 'conflict_detector' source.

        Returns:
            The number of rows deleted.
        """
        try:
            with self.db.get_connection("state") as conn:
                cursor = conn.execute(
                    "DELETE FROM published_conflicts WHERE source = 'conflict_detector' AND detected_at < datetime('now', ?)",
                    (f"-{days} days",),
                )
                return cursor.rowcount
        except Exception as e:
            logger.warning("Could not clean up old conflicts: %s", e)
            return 0

    def detect_conflicts(self, forward_hours: int = 48) -> list[dict[str, Any]]:
        """Find overlapping calendar events in the upcoming time window.

        Queries ``events.db`` for all ``calendar.event.created`` events, then
        filters to those whose start/end times fall within the upcoming
        *forward_hours* window.  This forward-looking approach catches conflicts
        regardless of when events were originally synced — a critical difference
        from filtering by sync timestamp, which misses ~99.9% of real conflicts.

        Args:
            forward_hours: How far ahead (in hours) to look for upcoming
                calendar events.  Defaults to 48.

        Returns:
            A list of conflict dicts, each containing:
                - ``event_a_id``: ID of the first event
                - ``event_b_id``: ID of the second event
                - ``overlap_minutes``: Duration of the overlap in minutes
                - ``event_a_summary``: Title of the first event
                - ``event_b_summary``: Title of the second event
        """
        try:
            return self._detect_conflicts_impl(forward_hours)
        except Exception:
            logger.exception("conflict_detector: error during conflict detection")
            return []

    def _detect_conflicts_impl(self, forward_hours: int) -> list[dict[str, Any]]:
        """Core implementation of conflict detection (unwrapped for testing clarity).

        Fetches recent calendar events and filters in Python to those whose
        start_time falls within the upcoming *forward_hours* window.  A 90-day
        SQL-level lookback on ``timestamp`` bounds the query so we don't scan
        the entire events table, while the Python filter handles the forward-
        looking calendar window.
        """
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(hours=forward_hours)
        # 90-day lookback bounds the SQL query — calendar events synced more
        # than 90 days ago are unlikely to represent upcoming appointments.
        timestamp_cutoff = (now - timedelta(days=90)).isoformat()

        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                """SELECT id, payload, timestamp FROM events
                   WHERE type = 'calendar.event.created'
                     AND timestamp > ?
                   ORDER BY timestamp DESC
                   LIMIT 2000""",
                (timestamp_cutoff,),
            ).fetchall()

        if len(rows) < 2:
            logger.debug(
                "conflict_detector: %d calendar events found (need ≥2) — skipping",
                len(rows),
            )
            return []

        # Parse each event's payload for start/end times, filtering to
        # events in the upcoming forward window.
        parsed: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
                # Handle double-encoded JSON payloads
                if isinstance(payload, str):
                    payload = json.loads(payload)

                # Support both naming conventions for start/end times
                start_str = payload.get("start_time") or payload.get("start", "")
                end_str = payload.get("end_time") or payload.get("end", "")

                if not start_str or not end_str:
                    continue

                start_dt = self._parse_datetime(start_str)
                end_dt = self._parse_datetime(end_str)
                if start_dt is None or end_dt is None:
                    continue

                # Forward-looking filter: skip events that have already ended
                # or that start after the look-ahead window.
                if end_dt < now or start_dt > window_end:
                    continue

                # Skip all-day events for conflict detection — multiple all-day
                # markers on the same day are normal (e.g., birthdays + holidays).
                is_all_day = payload.get("is_all_day", False)
                if is_all_day:
                    continue

                parsed.append({
                    "id": row["id"],
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "summary": payload.get("summary") or payload.get("title", "(no title)"),
                })
            except Exception:
                logger.debug("conflict_detector: skipping unparseable event %s", row["id"])
                continue

        # Compare all unique pairs for overlaps.
        # Two events overlap when: A.start < B.end AND B.start < A.end
        # Back-to-back events (A.end == B.start) are NOT conflicts.
        conflicts: list[dict[str, Any]] = []
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                a = parsed[i]
                b = parsed[j]

                if a["start_dt"] < b["end_dt"] and b["start_dt"] < a["end_dt"]:
                    # Calculate overlap duration
                    overlap_start = max(a["start_dt"], b["start_dt"])
                    overlap_end = min(a["end_dt"], b["end_dt"])
                    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60

                    conflicts.append({
                        "event_a_id": a["id"],
                        "event_b_id": b["id"],
                        "overlap_minutes": round(overlap_minutes, 1),
                        "event_a_summary": a["summary"],
                        "event_b_summary": b["summary"],
                    })

        logger.info(
            "conflict_detector: checked %d timed events, found %d conflicts",
            len(parsed),
            len(conflicts),
        )
        return conflicts

    async def check_and_publish(self, event_bus: Any) -> int:
        """Detect conflicts and publish new ones to the event bus.

        Calls :meth:`detect_conflicts` (which scans the upcoming 48-hour
        window regardless of when events were originally synced) and publishes
        a ``calendar.conflict.detected`` event for each conflict that has not
        already been published in this process lifetime.

        Args:
            event_bus: The NATS event bus instance (or mock) with an async
                ``publish(event_type, payload, source=..., priority=..., metadata=...)``
                method.

        Returns:
            The number of newly published conflict events.
        """
        # Run daily cleanup of stale conflict pairs (at most once per 24 hours).
        now = datetime.now(timezone.utc)
        if self._last_cleanup is None or (now - self._last_cleanup).total_seconds() > 86400:
            try:
                removed = self.cleanup_old_conflicts(days=30)
                self._last_cleanup = now
                if removed:
                    logger.info("ConflictDetector: cleaned up %d stale conflicts", removed)
            except Exception as e:
                logger.warning("ConflictDetector: cleanup failed (non-fatal): %s", e)

        conflicts = self.detect_conflicts()
        published_count = 0

        for conflict in conflicts:
            pair_key = frozenset({conflict["event_a_id"], conflict["event_b_id"]})
            if pair_key in self._published_conflicts:
                continue  # Already published this conflict

            try:
                await event_bus.publish(
                    "calendar.conflict.detected",
                    {
                        "event_a_id": conflict["event_a_id"],
                        "event_b_id": conflict["event_b_id"],
                        "overlap_minutes": conflict["overlap_minutes"],
                        "event_a_summary": conflict["event_a_summary"],
                        "event_b_summary": conflict["event_b_summary"],
                    },
                    source="conflict_detector",
                    priority="high",
                    metadata={
                        "related_events": [conflict["event_a_id"], conflict["event_b_id"]],
                    },
                )
                self._published_conflicts.add(pair_key)
                self._persist_conflict(pair_key)
                published_count += 1
            except Exception:
                logger.exception(
                    "conflict_detector: failed to publish conflict between %s and %s",
                    conflict["event_a_id"],
                    conflict["event_b_id"],
                )

        if published_count:
            logger.info("conflict_detector: published %d new conflict events", published_count)

        return published_count

    def get_diagnostics(self) -> dict:
        """Return conflict detector diagnostic information for monitoring.

        Reports published conflict count, last cleanup time, calendar event
        counts, and conflict pairs so operators can verify the service is
        working correctly.

        Each field is queried independently with try/except so that a single
        DB failure doesn't prevent the rest of the diagnostics from returning.

        Returns:
            Dict with keys: published_conflicts_count, last_cleanup,
            calendar_events_in_window, conflict_pairs, state_db_table_exists.
        """
        result: dict = {}

        # 1. In-memory published conflicts count
        try:
            result["published_conflicts_count"] = len(self._published_conflicts)
        except Exception as e:
            logger.warning("get_diagnostics: published_conflicts_count failed: %s", e)
            result["published_conflicts_count"] = {"error": str(e)}

        # 2. Last cleanup timestamp
        try:
            result["last_cleanup"] = self._last_cleanup.isoformat() if self._last_cleanup else None
        except Exception as e:
            logger.warning("get_diagnostics: last_cleanup failed: %s", e)
            result["last_cleanup"] = {"error": str(e)}

        # 3. Calendar events in upcoming 48h window
        try:
            now = datetime.now(timezone.utc)
            window_end = now + timedelta(hours=48)
            with self.db.get_connection("events") as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE type LIKE 'calendar.event.%' AND timestamp > ? AND timestamp < ?",
                    (now.isoformat(), window_end.isoformat()),
                ).fetchone()
            result["calendar_events_in_window"] = row[0] if row else 0
        except Exception as e:
            logger.warning("get_diagnostics: calendar_events_in_window failed: %s", e)
            result["calendar_events_in_window"] = {"error": str(e)}

        # 4. Published conflict pairs (frozensets converted to sorted lists, capped at 20)
        try:
            pairs = [sorted(pair) for pair in self._published_conflicts]
            result["conflict_pairs"] = pairs[:20]
        except Exception as e:
            logger.warning("get_diagnostics: conflict_pairs failed: %s", e)
            result["conflict_pairs"] = {"error": str(e)}

        # 5. Whether the published_conflicts table exists in state.db
        try:
            with self.db.get_connection("state") as conn:
                conn.execute("SELECT 1 FROM published_conflicts LIMIT 1")
            result["state_db_table_exists"] = True
        except Exception:
            result["state_db_table_exists"] = False

        return result

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        """Parse an ISO 8601 timestamp string into a timezone-aware datetime.

        Handles both ``Z`` suffix and ``+00:00`` format.  Returns ``None``
        for unparseable values so callers can skip gracefully.
        """
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # Ensure timezone-aware (date-only strings parse as naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            return None
