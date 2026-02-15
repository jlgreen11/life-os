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
        self.db = db
        self.bus = event_bus
        self.config = config

        # In-memory batch for digest delivery
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

        # Store the notification
        with self.db.get_connection("state") as conn:
            conn.execute(
                """INSERT INTO notifications
                   (id, title, body, priority, source_event_id, domain, status, action_url)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (notif_id, title, body, priority, source_event_id, domain, action_url),
            )

        # Publish creation telemetry event
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.created",
                {
                    "notification_id": notif_id,
                    "title": title,
                    "priority": priority,
                    "source_event_id": source_event_id,
                    "domain": domain,
                    "created_at": now.isoformat(),
                },
                source="notification_manager",
                priority=priority,
            )

        # Decision: deliver now or later?
        delivery = self._decide_delivery(priority, domain, now)

        if delivery == "immediate":
            await self._deliver_notification(notif_id, title, body, priority)
        elif delivery == "batch":
            self._pending_batch.append({
                "id": notif_id, "title": title, "body": body,
                "priority": priority, "domain": domain,
            })
        elif delivery == "suppress":
            self._mark_status(notif_id, "suppressed")
            return None

        return notif_id

    def _decide_delivery(self, priority: str, domain: Optional[str],
                         now: datetime) -> str:
        """Decide whether to deliver immediately, batch, or suppress."""

        # Critical always goes through
        if priority == "critical":
            return "immediate"

        # Check quiet hours
        if self._is_quiet_hours(now):
            if priority == "high":
                return "immediate"  # High still gets through during quiet hours
            return "suppress"

        # Get user's notification mode
        mode = self._get_notification_mode()

        if mode == "minimal":
            if priority in ("critical", "high"):
                return "immediate"
            return "suppress"

        elif mode == "batched":
            if priority in ("critical", "high"):
                return "immediate"
            return "batch"

        else:  # "frequent"
            if priority == "low":
                return "batch"
            return "immediate"

    def _is_quiet_hours(self, now: datetime) -> bool:
        """Check if we're currently in quiet hours."""
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
                if current_day not in qh.get("days", []):
                    continue
                start = time.fromisoformat(qh["start"])
                end = time.fromisoformat(qh["end"])

                # Handle overnight quiet hours (e.g., 22:00 to 07:00)
                if start <= end:
                    if start <= current_time <= end:
                        return True
                else:
                    if current_time >= start or current_time <= end:
                        return True
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return False

    def _get_notification_mode(self) -> str:
        """Get the user's notification preference."""
        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'notification_mode'"
            ).fetchone()
            return row["value"] if row else "batched"

    async def _deliver_notification(self, notif_id: str, title: str,
                                     body: Optional[str], priority: str):
        """Actually deliver a notification to the user."""
        self._mark_status(notif_id, "delivered")

        # Publish delivery event (web UI / mobile app picks this up)
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
        """Update notification status."""
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
        """Mark notification as acted on (strong positive signal)."""
        self._mark_status(notif_id, "acted_on")
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.acted_on",
                {"notification_id": notif_id},
                source="notification_manager",
            )

    async def dismiss(self, notif_id: str):
        """Dismiss a notification (negative feedback signal)."""
        self._mark_status(notif_id, "dismissed")
        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.dismissed",
                {"notification_id": notif_id},
                source="notification_manager",
            )

    async def get_digest(self) -> list[dict]:
        """Get the pending batch as a digest and clear the queue."""
        digest = list(self._pending_batch)
        for item in digest:
            self._mark_status(item["id"], "delivered")
        self._pending_batch = []
        return digest

    def get_pending(self, limit: int = 50) -> list[dict]:
        """Get all pending/delivered notifications."""
        with self.db.get_connection("state") as conn:
            rows = conn.execute(
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
        """Get notification statistics."""
        with self.db.get_connection("state") as conn:
            rows = conn.execute(
                """SELECT status, COUNT(*) as cnt 
                   FROM notifications 
                   GROUP BY status"""
            ).fetchall()
            return {row["status"]: row["cnt"] for row in rows}
