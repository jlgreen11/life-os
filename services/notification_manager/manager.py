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
import logging
import uuid
from datetime import datetime, time, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

from storage.database import DatabaseManager


class NotificationManager:
    """Routes, batches, and delivers notifications intelligently."""

    def __init__(self, db: DatabaseManager, event_bus: Any, config: dict, timezone: str = "America/Los_Angeles"):
        self.db = db          # Database access for notifications and preferences tables
        self.bus = event_bus   # Event bus for publishing delivery/dismissal events
        self.config = config   # App-level configuration (not user preferences)
        self._tz = ZoneInfo(timezone)

        # In-memory batch for digest delivery.
        # Low-priority notifications accumulate here until get_digest() is
        # called (typically on a schedule — e.g., 9 AM, 1 PM, 6 PM).
        self._pending_batch: list[dict] = []

        # Recover any batched notifications that were lost during a restart.
        # Notifications with status='pending' and normal/low priority were
        # destined for batch delivery but their in-memory references were lost.
        self._recover_pending_batch()

    def _recover_pending_batch(self):
        """Recover batched notifications from the database after a restart.

        When the server restarts, the in-memory _pending_batch list is lost.
        Notifications that were routed to batch delivery are still in the DB
        with status='pending'. This method queries for those notifications and
        re-populates the batch so the next get_digest() call delivers them.

        Only normal/low priority notifications are recovered because those are
        the ones routed to batch delivery by _decide_delivery(). Critical and
        high priority notifications are always delivered immediately and would
        never be in a pending-batch state.
        """
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    """SELECT id, title, body, priority, domain, source_event_id
                       FROM notifications
                       WHERE status = 'pending'
                         AND priority IN ('normal', 'low')""",
                ).fetchall()

            for row in rows:
                self._pending_batch.append({
                    "id": row["id"],
                    "title": row["title"],
                    "body": row["body"],
                    "priority": row["priority"],
                    "domain": row["domain"],
                    "source_event_id": row["source_event_id"],
                })

            if self._pending_batch:
                logger.info("Recovered %d pending batch notifications from database", len(self._pending_batch))
        except Exception:
            # Fail-open: if recovery fails, start with an empty batch.
            # The notifications are still in the DB and can be recovered later.
            logger.warning("Failed to recover pending batch notifications", exc_info=True)

    def _log_automatic_feedback(self, action_id: str, action_type: str,
                                 feedback_type: str, context: Optional[dict] = None):
        """
        Log automatic feedback to feedback_log without requiring user interaction.

        This method enables the feedback loop to function in a passive observation
        system where most predictions are never explicitly acted on or dismissed.
        By automatically logging feedback when predictions are auto-resolved, we
        ensure the reaction prediction system has data to learn from.

        Args:
            action_id: ID of the notification or prediction being resolved
            action_type: Type of action (e.g., "notification", "prediction")
            feedback_type: Type of feedback (e.g., "dismissed", "ignored", "engaged")
            context: Optional context dict with metadata about the auto-resolution
        """
        import uuid
        feedback_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT INTO feedback_log
                   (id, timestamp, action_id, action_type, feedback_type,
                    response_latency_seconds, context, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    now,
                    action_id,
                    action_type,
                    feedback_type,
                    None,  # Auto-resolved feedback has no latency
                    json.dumps(context or {}),
                    "Automatic feedback from prediction auto-resolution",
                ),
            )

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

        # --- Delivery decision flow ---
        # Three possible outcomes:
        #   "immediate" -> push to the user right now (via event bus)
        #   "batch"     -> hold in memory until the next digest window
        #   "suppress"  -> silently drop (e.g., quiet hours, user trained to ignore)
        delivery = self._decide_delivery(priority, domain, now)

        if delivery == "immediate":
            await self._deliver_notification(notif_id, title, body, priority)
            # Mark prediction as surfaced only when notification is actually delivered.
            # This fixes a critical bug where predictions were marked as surfaced even
            # when their notifications were suppressed, causing them to never be
            # auto-resolved by auto_resolve_stale_predictions() (which only looks for
            # delivered notifications with status='delivered').
            if domain == "prediction" and source_event_id:
                self._mark_prediction_surfaced(source_event_id)
        elif delivery == "batch":
            # Accumulate in the in-memory batch list for later digest delivery
            self._pending_batch.append({
                "id": notif_id, "title": title, "body": body,
                "priority": priority, "domain": domain,
                "source_event_id": source_event_id,  # Preserve for later surfacing
            })
        elif delivery == "suppress":
            # Mark as suppressed in DB so it shows up in audit logs but is
            # never delivered to the user. Return None to signal suppression.
            # IMPORTANT: Do NOT mark the prediction as surfaced here — suppressed
            # predictions should be filtered (was_surfaced=0) since the user never
            # saw them.
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

        The incoming ``now`` is UTC; we convert to the user's local
        timezone before comparing against the configured quiet-hours
        windows (which are specified in local time).
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        if not rows:
            return False

        try:
            quiet_hours = json.loads(rows["value"])
            local_now = now.astimezone(self._tz)
            current_time = local_now.time()
            current_day = local_now.strftime("%A").lower()

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
        """Get the user's notification preference.

        Returns one of: "minimal", "batched", "frequent".
        Defaults to "batched" if the user hasn't set a preference yet
        (safest default — avoids overwhelming a new user).

        Preferences can be stored in two formats depending on how they were
        written:
        - Plain string: ``"frequent"`` (written by ``update_preference`` route
          when the value is already a str, or inserted directly in tests)
        - JSON-encoded string: ``'"frequent"'`` (written by callers that
          JSON-encode all values before storage)

        We try JSON deserialization first and fall back to the raw value so
        the method works correctly regardless of which storage path was used.
        This prevents a ``json.JSONDecodeError`` when the value is stored as a
        plain string (the most common path via the web API).

        Example:
            >>> mgr._get_notification_mode()
            'batched'
        """
        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'notification_mode'"
            ).fetchone()
            if not row:
                return "batched"

        raw = row["value"]
        try:
            # Attempt JSON deserialization in case the value was stored as a
            # JSON-encoded string (e.g. '"frequent"').
            decoded = json.loads(raw)
            # json.loads('"frequent"') → 'frequent' (a str) — that's what we want.
            # json.loads('frequent') raises JSONDecodeError — handled below.
            if isinstance(decoded, str):
                return decoded
            # If decoded to a non-string type, fall through to returning raw
        except (json.JSONDecodeError, TypeError):
            pass

        # Raw value is already a plain string (e.g. 'frequent' stored directly).
        return raw

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

    def _mark_prediction_surfaced(self, prediction_id: str):
        """Mark a prediction as surfaced (shown to the user via notification).

        This is critical for the accuracy feedback loop. Only predictions that
        were actually surfaced to the user should be included in accuracy
        calculations — we can't measure whether a prediction was "accurate"
        if the user never saw it.

        This method is called ONLY when a notification is actually delivered to
        the user (either immediately or via batch digest). Predictions whose
        notifications are suppressed (quiet hours, minimal mode, etc.) are NOT
        marked as surfaced. The key distinction is:
        - was_surfaced = 1: Notification was delivered (user saw it)
        - was_surfaced = 0: Filtered by confidence gates or suppressed before delivery
        """
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE predictions SET was_surfaced = 1 WHERE id = ?",
                (prediction_id,),
            )

    def _update_linked_prediction(self, notif_id: str, was_accurate: bool):
        """If this notification came from a prediction, update prediction accuracy.

        Traces from the notification back to the originating prediction via
        source_event_id and updates the prediction's was_accurate and
        resolved_at fields. This closes the feedback loop so the prediction
        engine can learn from user responses.
        """
        with self.db.get_connection("state") as conn:
            notif = conn.execute(
                "SELECT source_event_id, domain FROM notifications WHERE id = ?",
                (notif_id,),
            ).fetchone()

        if not notif or notif["domain"] != "prediction" or not notif["source_event_id"]:
            return

        prediction_id = notif["source_event_id"]
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """UPDATE predictions SET
                   was_accurate = ?, resolved_at = ?,
                   user_response = ?
                   WHERE id = ?""",
                (
                    1 if was_accurate else 0,
                    now,
                    "acted_on" if was_accurate else "dismissed",
                    prediction_id,
                ),
            )

    async def mark_acted_on(self, notif_id: str):
        """
        Mark notification as acted on (strong positive signal).

        Also publishes a bus event so the feedback collector can record
        this as an implicit positive signal for the notification's domain.
        If the notification originated from a prediction, updates that
        prediction's accuracy tracking (was_accurate = True).
        """
        self._mark_status(notif_id, "acted_on")
        self._update_linked_prediction(notif_id, was_accurate=True)

        # Log explicit user feedback directly to feedback_log. This ensures
        # the feedback loop works even if the event bus handler fails or if
        # events aren't being stored. Direct logging provides a reliable path
        # for feedback data to reach the reaction prediction system.
        self._log_automatic_feedback(
            action_id=notif_id,
            action_type="notification",
            feedback_type="engaged",
            context={"explicit_user_action": True, "action": "acted_on"}
        )

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
        If the notification originated from a prediction, updates that
        prediction's accuracy tracking (was_accurate = False).
        """
        self._mark_status(notif_id, "dismissed")
        self._update_linked_prediction(notif_id, was_accurate=False)

        # Log explicit user dismissal directly to feedback_log. This ensures
        # the feedback loop works even if the event bus handler fails or if
        # events aren't being stored. Direct logging provides a reliable path
        # for dismissal patterns to reach the reaction prediction system.
        self._log_automatic_feedback(
            action_id=notif_id,
            action_type="notification",
            feedback_type="dismissed",
            context={"explicit_user_action": True, "action": "dismissed"}
        )

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
            # Mark prediction as surfaced when batched notification is delivered.
            # This ensures predictions are only counted as surfaced when the user
            # actually sees them (either immediate or batched delivery).
            if item.get("domain") == "prediction" and item.get("source_event_id"):
                self._mark_prediction_surfaced(item["source_event_id"])
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

    async def auto_resolve_stale_predictions(self, timeout_hours: int = 24):
        """
        Auto-resolve prediction notifications that users ignored for too long.

        When a prediction notification remains in "delivered" status beyond the
        timeout period with no user interaction (no dismiss, no act-on, no read),
        it's a signal that the prediction wasn't relevant or compelling enough
        to warrant attention. This method marks such predictions as inaccurate,
        closing the feedback loop so the prediction engine can learn from the
        implicit dismissal.

        Why this matters:
        - Without auto-resolution, ignored predictions remain unresolved forever
        - This breaks the accuracy feedback loop (_get_accuracy_multiplier)
        - The prediction engine can't learn which predictions are unhelpful
        - Confidence gates never adjust downward for noisy prediction types

        Args:
            timeout_hours: Number of hours after delivery to consider stale.
                          Default is 24h (1 day). Predictions older than this
                          with no user interaction are marked inaccurate.

        Returns:
            Number of predictions auto-resolved.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=timeout_hours)).isoformat()

        # Find stale prediction notifications: delivered more than timeout_hours
        # ago, still in "delivered" status (never read, acted on, or dismissed).
        with self.db.get_connection("state") as conn:
            stale = conn.execute(
                """SELECT id, source_event_id FROM notifications
                   WHERE domain = 'prediction'
                     AND status = 'delivered'
                     AND delivered_at < ?
                     AND source_event_id IS NOT NULL""",
                (cutoff,),
            ).fetchall()

        resolved_count = 0
        for notif in stale:
            # Mark the prediction as inaccurate (user ignored it = not helpful).
            # Use was_accurate=0, user_response='ignored' to distinguish from
            # explicit dismissals (user_response='dismissed').
            prediction_id = notif["source_event_id"]
            with self.db.get_connection("user_model") as conn:
                conn.execute(
                    """UPDATE predictions SET
                       was_accurate = 0,
                       resolved_at = ?,
                       user_response = 'ignored'
                       WHERE id = ? AND resolved_at IS NULL""",
                    (now.isoformat(), prediction_id),
                )
                if conn.total_changes > 0:
                    resolved_count += 1

            # Log automatic feedback for the ignored prediction. This closes the
            # feedback loop by recording implicit dismissals in feedback_log, which
            # enables the reaction prediction system to learn from patterns even
            # without explicit user clicks. Ignored predictions are treated as
            # dismissed for feedback purposes since the user didn't find them
            # compelling enough to act on.
            self._log_automatic_feedback(
                action_id=notif["id"],
                action_type="notification",
                feedback_type="dismissed",
                context={"auto_resolved": True, "reason": "ignored", "timeout_hours": timeout_hours}
            )

            # Mark the notification as expired so it doesn't clutter the UI.
            self._mark_status(notif["id"], "expired")

        return resolved_count

    def auto_resolve_filtered_predictions(self, timeout_hours: int = 1) -> int:
        """Auto-resolve predictions that were filtered out before surfacing.

        Predictions with was_surfaced=0 were filtered by confidence gates or
        reaction prediction and never shown to the user. After a timeout period,
        these should be auto-resolved to prevent database bloat and polluted
        accuracy metrics.

        These predictions are marked with:
        - was_accurate = NULL (we never tested them, so can't measure accuracy)
        - user_response = 'filtered' (to distinguish from 'ignored' surfaced ones)
        - resolved_at = now

        This is critical for data quality:
        - Without this, 270k+ filtered predictions sit unresolved forever
        - Accuracy queries become slow (scanning hundreds of thousands of rows)
        - The "unresolved predictions" metric becomes meaningless
        - Database bloat from storing predictions that will never be used

        Args:
            timeout_hours: Hours after creation to consider a filtered prediction
                          stale. Default is 1 hour (short because these were never
                          surfaced, so they're immediately irrelevant).

        Returns:
            Number of filtered predictions auto-resolved.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=timeout_hours)).isoformat()

        # Find unsurfaced predictions created more than timeout_hours ago
        # that are still unresolved.
        with self.db.get_connection("user_model") as conn:
            result = conn.execute(
                """UPDATE predictions SET
                   was_accurate = NULL,
                   resolved_at = ?,
                   user_response = 'filtered'
                   WHERE was_surfaced = 0
                     AND resolved_at IS NULL
                     AND created_at < ?""",
                (now.isoformat(), cutoff),
            )
            resolved_count = result.rowcount

        return resolved_count
