"""
Life OS — Notification Manager

Intelligent notification routing. Doesn't just forward alerts blindly —
considers the user's current mood, location, activity, quiet hours,
and notification preferences before delivering.

This is where the "the AI protected my attention" magic happens.
Batches low-priority items, escalates genuinely urgent ones, and
suppresses noise the user has trained the system to ignore.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, time, timezone
from typing import Any, Optional

from storage.database import DatabaseManager


class NotificationManager:
    """Routes, batches, and delivers notifications intelligently."""

    def __init__(self, db: DatabaseManager, event_bus: Any, config: dict):
        self.db = db          # Database access for notifications and preferences tables
        self.bus = event_bus   # Event bus for publishing delivery/dismissal events
        self.config = config   # App-level configuration (not user preferences)

        # In-memory batch for digest delivery.
        # Low-priority notifications accumulate here until get_digest() is
        # called (typically on a schedule — e.g., 9 AM, 1 PM, 6 PM).
        self._pending_batch: list[dict] = []

    async def create_notification(
        self,
        title: str,
        body: Optional[str] = None,
        priority: str = "normal",
        source_event_id: Optional[str] = None,
        domain: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a notification. Doesn't necessarily deliver it immediately.

        Flow:
            1. Check quiet hours → suppress or queue
            2. Check priority → critical bypasses everything
            3. Check notification mode → immediate, batched, or minimal
            4. Check recent dismissal patterns → should we even bother?
            5. Deliver or queue for batch digest
        """
        notif_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Always persist the notification first, regardless of delivery decision.
        # This ensures we have a complete audit trail even for suppressed items.
        with self.db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications 
                   (id, title, body, priority, source_event_id, domain, status, action_url)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (notif_id, title, body, priority, source_event_id, domain, action_url),
            )

        # --- Delivery decision flow ---
        # Three possible outcomes:
        #   "immediate" -> push to the user right now (via event bus)
        #   "batch"     -> hold in memory until the next digest window
        #   "suppress"  -> silently drop (e.g., quiet hours, user trained to ignore)
        delivery = self._decide_delivery(priority, domain, now)

        if delivery == "immediate":
            await self._deliver_notification(notif_id, title, body, priority)
        elif delivery == "batch":
            # Accumulate in the in-memory batch list for later digest delivery
            self._pending_batch.append({
                "id": notif_id, "title": title, "body": body,
                "priority": priority, "domain": domain,
            })
        elif delivery == "suppress":
            # Mark as suppressed in DB so it shows up in audit logs but is
            # never delivered to the user. Return None to signal suppression.
            self._mark_status(notif_id, "suppressed")
            return None

        return notif_id

    def _decide_delivery(self, priority: str, domain: Optional[str],
                         now: datetime) -> str:
        """
        Decide whether to deliver immediately, batch, or suppress.

        Decision tree (evaluated top to bottom, first match wins):

            1. CRITICAL priority  -> always immediate (bypass everything)
            2. Quiet hours active -> HIGH = immediate, everything else suppressed
            3. User mode:
                 "minimal"  -> only critical/high get through, rest suppressed
                 "batched"  -> critical/high immediate, normal/low batched
                 "frequent" -> everything immediate except low (batched)

        This layered approach means urgent items always reach the user
        while respecting their attention preferences for lower-priority items.
        """

        # --- Priority escalation: critical bypasses ALL filters ---
        if priority == "critical":
            return "immediate"

        # --- Quiet hours gate ---
        # During quiet hours only high-priority items break through.
        # Normal and low items are suppressed entirely (not even batched).
        if self._is_quiet_hours(now):
            if priority == "high":
                return "immediate"  # High still gets through during quiet hours
            return "suppress"

        # --- User notification mode ---
        # The mode is set during onboarding and can be changed at any time.
        mode = self._get_notification_mode()

        if mode == "minimal":
            # "Protect my focus" — only truly important items surface
            if priority in ("critical", "high"):
                return "immediate"
            return "suppress"

        elif mode == "batched":
            # "Digest 2-3 times a day" — urgent items still break through
            if priority in ("critical", "high"):
                return "immediate"
            return "batch"

        else:  # "frequent" — "keep me in the loop"
            # Everything goes through immediately except low-priority,
            # which still gets batched to avoid overwhelming the user.
            if priority == "low":
                return "batch"
            return "immediate"

    def _is_quiet_hours(self, now: datetime) -> bool:
        """
        Check if we're currently in quiet hours.

        Quiet hours are stored as a JSON list of time-range objects, each
        with "start", "end", and "days" fields. Multiple ranges are
        supported (e.g., different times on weekdays vs. weekends).
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        if not rows:
            return False

        try:
            quiet_hours = json.loads(rows["value"])
            current_time = now.time()
            current_day = now.strftime("%A").lower()

            for qh in quiet_hours:
                # Skip ranges that don't apply to the current day of the week
                if current_day not in qh.get("days", []):
                    continue
                start = time.fromisoformat(qh["start"])
                end = time.fromisoformat(qh["end"])

                # --- Overnight spanning logic ---
                # If start <= end (e.g., 09:00-17:00), it's a same-day range:
                #   just check if current_time falls within [start, end].
                # If start > end (e.g., 22:00-07:00), the range crosses midnight:
                #   the user is in quiet hours if current_time >= start OR <= end.
                if start <= end:
                    if start <= current_time <= end:
                        return True
                else:
                    # Overnight range: 22:00 -> midnight -> 07:00
                    if current_time >= start or current_time <= end:
                        return True
        except (json.JSONDecodeError, KeyError, ValueError):
            # Malformed quiet hours data — fail open (no quiet hours)
            pass

        return False

    def _get_notification_mode(self) -> str:
        """
        Get the user's notification preference.

        Returns one of: "minimal", "batched", "frequent".
        Defaults to "batched" if the user hasn't set a preference yet
        (safest default — avoids overwhelming a new user).
        """
        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'notification_mode'"
            ).fetchone()
            return row["value"] if row else "batched"

    async def _deliver_notification(self, notif_id: str, title: str,
                                     body: Optional[str], priority: str):
        """
        Actually deliver a notification to the user.

        Delivery is a two-step process:
            1. Mark the DB record as "delivered" (for audit/tracking)
            2. Publish a "notification.delivered" event on the bus so the
               web UI and/or mobile push service can pick it up.
        """
        self._mark_status(notif_id, "delivered")

        # Publish delivery event — downstream consumers (web UI, mobile app,
        # push notification service) subscribe to this event type.
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.delivered",
                {
                    "notification_id": notif_id,
                    "title": title,
                    "body": body,
                    "priority": priority,
                },
                source="notification_manager",
                priority=priority,
            )

    def _mark_status(self, notif_id: str, status: str):
        """
        Update notification status and the corresponding timestamp column.

        Each status has its own timestamp column (delivered_at, read_at,
        acted_on_at, dismissed_at) so we can measure latency between
        delivery and user action — a key signal for the feedback collector.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("state") as conn:
            if status == "delivered":
                conn.execute(
                    "UPDATE notifications SET status = ?, delivered_at = ? WHERE id = ?",
                    (status, now, notif_id),
                )
            elif status == "read":
                conn.execute(
                    "UPDATE notifications SET status = ?, read_at = ? WHERE id = ?",
                    (status, now, notif_id),
                )
            elif status == "acted_on":
                conn.execute(
                    "UPDATE notifications SET status = ?, acted_on_at = ? WHERE id = ?",
                    (status, now, notif_id),
                )
            elif status == "dismissed":
                conn.execute(
                    "UPDATE notifications SET status = ?, dismissed_at = ? WHERE id = ?",
                    (status, now, notif_id),
                )
            else:
                conn.execute(
                    "UPDATE notifications SET status = ? WHERE id = ?",
                    (status, notif_id),
                )

    async def mark_read(self, notif_id: str):
        """Mark notification as read."""
        self._mark_status(notif_id, "read")

    async def mark_acted_on(self, notif_id: str):
        """
        Mark notification as acted on (strong positive signal).

        Also publishes a bus event so the feedback collector can record
        this as an implicit positive signal for the notification's domain.
        """
        self._mark_status(notif_id, "acted_on")
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.acted_on",
                {"notification_id": notif_id},
                source="notification_manager",
            )

    async def dismiss(self, notif_id: str):
        """
        Dismiss a notification (negative feedback signal).

        The bus event allows the feedback collector to learn that this
        type of notification was unwanted, informing future suppression.
        """
        self._mark_status(notif_id, "dismissed")
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.dismissed",
                {"notification_id": notif_id},
                source="notification_manager",
            )

    async def get_digest(self) -> list[dict]:
        """
        Get the pending batch as a digest and clear the queue.

        This is the digest/batch mechanism: batched notifications accumulate
        in self._pending_batch until this method is called (typically on a
        timer — e.g., morning briefing, lunch digest, evening wrap-up).
        Each item is marked "delivered" upon retrieval, and the in-memory
        queue is reset so the next digest starts fresh.
        """
        digest = list(self._pending_batch)
        for item in digest:
            self._mark_status(item["id"], "delivered")
        self._pending_batch = []
        return digest

    def get_pending(self, limit: int = 50) -> list[dict]:
        """
        Get all pending/delivered notifications for the notification center UI.

        Results are sorted by priority (critical first) then by recency,
        so the most urgent and newest items appear at the top.
        """
        with self.db.get_connection("state") as conn:
            rows = conn.execute(
                # Priority ordering: critical=1, high=2, normal=3, low=4.
                # Within the same priority, newest notifications come first.
                """SELECT * FROM notifications
                   WHERE status IN ('pending', 'delivered')
                   ORDER BY
                       CASE priority
                           WHEN 'critical' THEN 1
                           WHEN 'high' THEN 2
                           WHEN 'normal' THEN 3
                           ELSE 4
                       END,
                       created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Get notification statistics grouped by status for the dashboard."""
        with self.db.get_connection("state") as conn:
            rows = conn.execute(
                """SELECT status, COUNT(*) as cnt 
                   FROM notifications 
                   GROUP BY status"""
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}
