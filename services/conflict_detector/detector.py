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
from typing import Any

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
        self._published_conflicts: set[frozenset[str]] = set()

    def detect_conflicts(self, time_window_hours: int = 24) -> list[dict[str, Any]]:
        """Find overlapping calendar events within the given time window.

        Queries ``events.db`` for ``calendar.event.created`` events whose
        payload timestamps fall within *time_window_hours* from now, then
        checks every unique pair for time overlaps.

        Args:
            time_window_hours: How far back to look for calendar events
                that were synced. The method then filters to events whose
                actual start/end times overlap.

        Returns:
            A list of conflict dicts, each containing:
                - ``event_a_id``: ID of the first event
                - ``event_b_id``: ID of the second event
                - ``overlap_minutes``: Duration of the overlap in minutes
                - ``event_a_summary``: Title of the first event
                - ``event_b_summary``: Title of the second event
        """
        try:
            return self._detect_conflicts_impl(time_window_hours)
        except Exception:
            logger.exception("conflict_detector: error during conflict detection")
            return []

    def _detect_conflicts_impl(self, time_window_hours: int) -> list[dict[str, Any]]:
        """Core implementation of conflict detection (unwrapped for testing clarity)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=time_window_hours)).isoformat()

        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                """SELECT * FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?
                   ORDER BY timestamp DESC""",
                (cutoff,),
            ).fetchall()

        if len(rows) < 2:
            logger.debug(
                "conflict_detector: %d calendar events found (need ≥2) — skipping",
                len(rows),
            )
            return []

        # Parse each event's payload for start/end times.
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

        Calls :meth:`detect_conflicts` and publishes a
        ``calendar.conflict.detected`` event for each conflict that has not
        already been published in this process lifetime.

        Args:
            event_bus: The NATS event bus instance (or mock) with an async
                ``publish(event_type, payload, source=..., priority=..., metadata=...)``
                method.

        Returns:
            The number of newly published conflict events.
        """
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
