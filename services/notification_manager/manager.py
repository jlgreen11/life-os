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
import sqlite3
import uuid
from datetime import UTC, datetime, time, timedelta
from typing import Any
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

    def expire_stale_notifications(self, max_age_hours: int = 48) -> tuple[int, list[str]]:
        """Expire pending notifications that are older than max_age_hours.

        Stale pending notifications (e.g., "You have a meeting at 2pm" from
        days ago) create noise in digests and erode user trust. This method
        marks them as 'expired' so they are excluded from future delivery.

        For each expired notification, automatic 'ignored' feedback is logged
        so the FeedbackCollector can learn from user non-interaction (e.g.,
        reducing notifications the user never reads).

        Only affects notifications with status='pending' or 'batched' —
        delivered, read, acted-on, and dismissed notifications are left untouched.

        Args:
            max_age_hours: Maximum age in hours before a pending notification
                          is considered stale. Default is 48 hours.

        Returns:
            Tuple of (count of expired notifications, list of expired notification IDs).
        """
        try:
            # Format must match SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            # where %f = SS.SSS (seconds with 3-digit fractional part).
            # Sub-second precision is irrelevant for a 48-hour cutoff.
            cutoff = (datetime.now(UTC) - timedelta(hours=max_age_hours)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )
            with self.db.get_connection("state") as conn:
                # Collect IDs before the UPDATE so we can log feedback for each.
                # Include both 'pending' (immediate-delivery queue) and 'batched'
                # (digest-delivery queue) so all undelivered notifications age out.
                expired_ids = [
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM notifications WHERE status IN ('pending', 'batched') AND created_at < ?",
                        (cutoff,),
                    ).fetchall()
                ]

                result = conn.execute(
                    """UPDATE notifications
                       SET status = 'expired'
                       WHERE status IN ('pending', 'batched')
                         AND created_at < ?""",
                    (cutoff,),
                )
                expired_count = result.rowcount

            if expired_count > 0:
                logger.info("Expired %d stale pending notifications (older than %dh)", expired_count, max_age_hours)

            # Log 'ignored' feedback for each expired notification so the
            # FeedbackCollector's _learn_from_ignore() can adjust notification
            # behaviour (channel weights, contact preferences, type suppression).
            for nid in expired_ids:
                try:
                    self._log_automatic_feedback(
                        action_id=nid,
                        action_type="notification",
                        feedback_type="ignored",
                        context={"explicit_user_action": False, "action": "expired_ignored"},
                    )
                except Exception:
                    logger.debug("Failed to log ignored feedback for expired notification %s", nid)

            return expired_count, expired_ids
        except Exception:
            # Fail-open: expiry failures must never block notification delivery.
            logger.warning("Failed to expire stale notifications", exc_info=True)
            return 0, []

    async def auto_deliver_stale_batch(
        self,
        max_pending_hours: float = 1,
        high_priority_hours: float = 0.5,
    ) -> int:
        """Auto-deliver pending notifications that have been waiting too long.

        Notifications sit in 'pending' or 'batched' status until the user
        visits a digest endpoint.  If the user doesn't check the dashboard,
        they expire after 48 hours and are never seen — previously an 86%
        expiry rate in practice.

        This method bridges the gap with a **graduated delivery strategy**:

        - ``high`` and ``critical`` priority notifications are delivered after
          *high_priority_hours* (default 0.5h / 30 min).  Urgent signals
          should reach the user quickly regardless of dashboard visits.
        - ``normal`` and ``low`` priority notifications are delivered after
          *max_pending_hours* (default 1h).  If the user hasn't checked in
          one hour they're unlikely to see a digest soon.

        Called periodically from the _notification_expiry_loop in main.py,
        BEFORE the expiry step, so notifications get a chance at delivery
        before they age out.

        Args:
            max_pending_hours: Deliver normal/low priority notifications older
                               than this many hours.  Default 1 hour.
            high_priority_hours: Deliver high/critical priority notifications
                                 older than this many hours.  Default 0.5 hours
                                 (30 minutes).

        Returns:
            Count of notifications auto-delivered.
        """
        try:
            now = datetime.now(UTC)
            # Cutoff timestamps for each priority tier.
            normal_cutoff = (now - timedelta(hours=max_pending_hours)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )
            high_cutoff = (now - timedelta(hours=high_priority_hours)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"
            )

            with self.db.get_connection("state") as conn:
                # Fetch notifications eligible under either threshold.
                # high/critical: older than high_priority_hours
                # normal/low: older than max_pending_hours
                rows = conn.execute(
                    """SELECT id, title, body, priority, domain, source_event_id
                       FROM notifications
                       WHERE status IN ('pending', 'batched')
                         AND (
                           (priority IN ('high', 'critical') AND created_at < ?)
                           OR
                           (priority NOT IN ('high', 'critical') AND created_at < ?)
                         )
                       ORDER BY created_at""",
                    (high_cutoff, normal_cutoff),
                ).fetchall()

            delivered = 0
            for row in rows:
                try:
                    await self._deliver_notification(
                        row["id"], row["title"], row["body"], row["priority"]
                    )
                    # Mark prediction as surfaced so accuracy tracking works.
                    if row["domain"] == "prediction" and row["source_event_id"]:
                        self._mark_prediction_surfaced(row["source_event_id"])
                    delivered += 1
                except Exception:
                    logger.warning(
                        "Failed to auto-deliver notification %s", row["id"], exc_info=True
                    )

            if delivered:
                logger.info(
                    "Auto-delivered %d stale notification(s) (high_priority_hours=%.1f, "
                    "normal_hours=%.1f)",
                    delivered, high_priority_hours, max_pending_hours,
                )

            return delivered
        except Exception:
            # Fail-open: auto-delivery failures must never crash the expiry loop.
            logger.warning("Failed to auto-deliver stale batched notifications", exc_info=True)
            return 0

    def _check_dismissal_suppression(self, domain: str | None) -> bool:
        """Check if notifications from this domain have been frequently dismissed.

        Returns True if the notification should be suppressed based on
        recent dismissal patterns (70%+ dismissal rate with 3+ data points).
        """
        if not domain:
            return False

        try:
            with self.db.get_connection("preferences") as conn:
                stats = conn.execute(
                    """SELECT
                           SUM(CASE WHEN feedback_type = 'dismissed' THEN 1 ELSE 0 END) as dismissed,
                           COUNT(*) as total
                       FROM feedback_log
                       WHERE action_type = 'notification'
                         AND json_extract(context, '$.domain') = ?
                         AND timestamp > datetime('now', '-7 days')""",
                    (domain,),
                ).fetchone()

                if not stats or not stats["total"]:
                    return False

                dismissed = stats["dismissed"] or 0
                total = stats["total"]

                # Suppress if 70%+ of recent notifications from this domain were dismissed
                # AND at least 3 data points exist (avoid suppressing on sparse data)
                if total >= 3 and dismissed / total >= 0.7:
                    logger.info(
                        "Suppressing notification for domain %s: %d/%d dismissed (%.0f%%)",
                        domain, dismissed, total, 100 * dismissed / total,
                    )
                    return True
        except Exception:
            pass  # Fail-open: if the check fails, don't suppress

        return False

    def _log_automatic_feedback(self, action_id: str, action_type: str,
                                 feedback_type: str, context: dict | None = None):
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
        now = datetime.now(UTC).isoformat()

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

    async def _publish_notification_ignored_events(self, notification_ids: list[str]):
        """Publish notification.ignored events for expired notifications.

        Called after expire_stale_notifications() in async contexts (e.g.,
        get_digest) so the FeedbackCollector's _learn_from_ignore() handler
        can process the event bus signal alongside the direct DB feedback.
        """
        if not self.bus or not self.bus.is_connected:
            return
        for nid in notification_ids:
            try:
                await self.bus.publish(
                    "notification.ignored",
                    {"notification_id": nid},
                    source="notification_manager",
                )
            except Exception:
                logger.debug("Failed to publish notification.ignored for %s", nid)

    async def create_notification(
        self,
        title: str,
        body: str | None = None,
        priority: str = "normal",
        source_event_id: str | None = None,
        domain: str | None = None,
        action_url: str | None = None,
    ) -> str | None:
        """
        Create a notification. Doesn't necessarily deliver it immediately.

        Flow:
            0. Deduplication check → return existing ID if duplicate found
            1. Check quiet hours → suppress or queue
            2. Check priority → critical bypasses everything
            3. Check notification mode → immediate, batched, or minimal
            4. Check recent dismissal patterns → should we even bother?
            5. Deliver or queue for batch digest
        """
        # --- Deduplication: source_event_id check (highest priority) ---
        # If the same source event already has a pending notification, suppress
        # the duplicate. This prevents repeated notifications when connectors
        # sync duplicate events or the prediction loop re-generates predictions.
        if source_event_id:
            try:
                with self.db.get_connection("state") as conn:
                    existing = conn.execute(
                        """SELECT id FROM notifications
                           WHERE source_event_id = ?
                             AND status IN ('pending', 'batched')
                           LIMIT 1""",
                        (source_event_id,),
                    ).fetchone()
                    if existing:
                        logger.debug(
                            "Suppressed duplicate notification for event %s (existing=%s)",
                            source_event_id,
                            existing["id"],
                        )
                        return existing["id"]
            except Exception:
                pass  # Fail-open: if the dedup check fails, create the notification anyway

        # --- Deduplication: title + domain within a 10-minute window ---
        # Catches duplicates that lack a source_event_id or have different
        # source_event_ids but represent the same logical notification.
        try:
            with self.db.get_connection("state") as conn:
                recent = conn.execute(
                    """SELECT id FROM notifications
                       WHERE title = ? AND domain = ?
                         AND created_at > datetime('now', '-10 minutes')
                       LIMIT 1""",
                    (title, domain),
                ).fetchone()
                if recent:
                    logger.debug(
                        "Suppressed duplicate notification: %s (domain=%s, existing=%s)",
                        title,
                        domain,
                        recent["id"],
                    )
                    return recent["id"]
        except Exception:
            pass  # Fail-open: if the dedup check fails, create the notification anyway

        # --- Step 4: Dismissal pattern suppression ---
        # If the user has been consistently dismissing notifications from this
        # domain, don't even bother creating one. Critical notifications always
        # get through regardless of dismissal history.
        if priority != "critical" and self._check_dismissal_suppression(domain):
            logger.debug(
                "Notification suppressed by dismissal pattern: %s (domain=%s)",
                title, domain,
            )
            return None

        notif_id = str(uuid.uuid4())
        now = datetime.now(UTC)

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
            # Mark as 'batched' in the DB so it persists across restarts and is
            # picked up by get_digest() when the next digest window fires.
            # Using a distinct status (rather than 'pending') makes batch-routed
            # notifications clearly identifiable and durable.
            self._mark_status(notif_id, "batched")
        elif delivery == "suppress":
            # Mark as suppressed in DB so it shows up in audit logs but is
            # never delivered to the user. Return None to signal suppression.
            # IMPORTANT: Do NOT mark the prediction as surfaced here — suppressed
            # predictions should be filtered (was_surfaced=0) since the user never
            # saw them.
            self._mark_status(notif_id, "suppressed")
            return None

        return notif_id

    def _decide_delivery(self, priority: str, domain: str | None,
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
        Defaults to "immediate" if the user hasn't set a preference yet,
        ensuring notifications are delivered out-of-the-box on fresh
        installations that haven't completed onboarding.

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
            'immediate'
        """
        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'notification_mode'"
            ).fetchone()
            if not row:
                logger.info("No notification_mode preference set — defaulting to 'immediate'")
                return "immediate"

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
                                     body: str | None, priority: str):
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
        now = datetime.now(UTC).isoformat()
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
        try:
            with self.db.get_connection("user_model") as conn:
                conn.execute(
                    "UPDATE predictions SET was_surfaced = 1 WHERE id = ?",
                    (prediction_id,),
                )
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            # Fail-open: surfacing tracking is secondary to notification delivery.
            # A corrupt user_model.db should not prevent notifications from reaching the user.
            logger.warning(
                "Failed to mark prediction %s as surfaced (user_model.db may be corrupted)",
                prediction_id, exc_info=True,
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
        now = datetime.now(UTC).isoformat()
        try:
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
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            # Fail-open: prediction accuracy tracking is secondary to notification lifecycle.
            # mark_acted_on/dismiss should still succeed even if user_model.db is corrupt.
            logger.warning(
                "Failed to update linked prediction for notification %s (user_model.db may be corrupted)",
                notif_id, exc_info=True,
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

        # Look up the notification's domain so we can include it in feedback.
        # This allows _check_dismissal_suppression() to query by domain later.
        notif_domain = None
        try:
            with self.db.get_connection("state") as conn:
                notif = conn.execute(
                    "SELECT domain FROM notifications WHERE id = ?", (notif_id,)
                ).fetchone()
                notif_domain = notif["domain"] if notif else None
        except Exception:
            pass  # Fail-open: missing domain won't break the dismissal log

        # Log explicit user dismissal directly to feedback_log. This ensures
        # the feedback loop works even if the event bus handler fails or if
        # events aren't being stored. Direct logging provides a reliable path
        # for dismissal patterns to reach the reaction prediction system.
        self._log_automatic_feedback(
            action_id=notif_id,
            action_type="notification",
            feedback_type="dismissed",
            context={"explicit_user_action": True, "action": "dismissed", "domain": notif_domain},
        )

        if self.bus and self.bus.is_connected:
            await self.bus.publish(
                "notification.dismissed",
                {"notification_id": notif_id},
                source="notification_manager",
            )

    async def get_digest(self) -> list[dict]:
        """
        Get the pending batch as a digest and deliver all batched notifications.

        Batch-routed notifications are stored with status='batched' in the DB
        (durable across restarts) until this method is called (typically on a
        timer — e.g., morning briefing, lunch digest, evening wrap-up).
        Each item is marked "delivered" upon retrieval.

        Before flushing, stale notifications are expired so that items that
        aged out during a long-running session don't appear in the digest.
        """
        # Expire stale notifications that accumulated during a long session.
        # Publish notification.ignored events so the FeedbackCollector can
        # learn from user non-interaction via the event bus.
        try:
            _expired_count, _expired_ids = self.expire_stale_notifications()
            if _expired_ids:
                await self._publish_notification_ignored_events(_expired_ids)
        except Exception:
            logger.warning("Failed to expire stale notifications during digest", exc_info=True)

        # Read the digest directly from the DB — all batch-routed notifications
        # are durably stored with status='batched', so there is no in-memory
        # list to consult.  This makes the digest crash-safe: a restart between
        # notification creation and get_digest() call loses nothing.
        digest = []
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    "SELECT id, title, body, priority, domain, source_event_id, action_url "
                    "FROM notifications WHERE status = 'batched' ORDER BY created_at"
                ).fetchall()
            digest = [dict(row) for row in rows]
        except Exception:
            logger.warning("Failed to load batched notifications in get_digest", exc_info=True)

        for item in digest:
            self._mark_status(item["id"], "delivered")
            # Mark prediction as surfaced when batched notification is delivered.
            # This ensures predictions are only counted as surfaced when the user
            # actually sees them (either immediate or batched delivery).
            if item.get("domain") == "prediction" and item.get("source_event_id"):
                self._mark_prediction_surfaced(item["source_event_id"])
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
                   WHERE status IN ('pending', 'batched', 'delivered')
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

    def delivery_health(self) -> dict:
        """Return a snapshot of the notification delivery pipeline health.

        Provides operators and diagnostics with a single-call view into how
        many notifications are being created, delivered, and lost to expiry.
        The *delivery_rate* is the primary signal: a healthy system should
        trend upward toward 1.0 as the graduated auto-delivery strategy keeps
        more notifications from expiring unseen.

        Returns:
            A dict with the following keys:

            - ``total_created``   — All notifications ever created.
            - ``delivered``       — Notifications marked 'delivered'.
            - ``expired``         — Notifications that aged out before delivery.
            - ``pending``         — Still waiting for delivery (pending/batched).
            - ``other``           — All other statuses (read, acted_on, dismissed,
                                    suppressed, …).
            - ``delivery_rate``   — delivered / total_created (0.0 if none created).
        """
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    """SELECT status, COUNT(*) as cnt
                       FROM notifications
                       GROUP BY status"""
                ).fetchall()
        except Exception:
            logger.warning("Failed to query delivery health stats", exc_info=True)
            return {
                "total_created": 0,
                "delivered": 0,
                "expired": 0,
                "pending": 0,
                "other": 0,
                "delivery_rate": 0.0,
            }

        counts: dict[str, int] = {row["status"]: row["cnt"] for row in rows}
        delivered = counts.get("delivered", 0)
        expired = counts.get("expired", 0)
        # Both 'pending' and 'batched' are awaiting delivery.
        pending = counts.get("pending", 0) + counts.get("batched", 0)
        # Everything that isn't delivered/expired/pending/batched (read, acted_on, etc.)
        other = sum(
            v for k, v in counts.items()
            if k not in ("delivered", "expired", "pending", "batched")
        )
        total_created = delivered + expired + pending + other
        delivery_rate = delivered / total_created if total_created > 0 else 0.0

        return {
            "total_created": total_created,
            "delivered": delivered,
            "expired": expired,
            "pending": pending,
            "other": other,
            "delivery_rate": round(delivery_rate, 4),
        }

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

        now = datetime.now(UTC)
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
            was_resolved = False
            try:
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
                        was_resolved = True
            except (sqlite3.DatabaseError, sqlite3.OperationalError):
                # Fail-open: skip this prediction but continue processing remaining items.
                # Skip feedback logging on DB error — we don't know whether the prediction
                # was already resolved or not.
                logger.warning(
                    "Failed to auto-resolve stale prediction %s (user_model.db may be corrupted)",
                    prediction_id, exc_info=True,
                )

            # Only log automatic feedback when we actually resolved the prediction.
            # If the prediction was already resolved (e.g., user acted on it), the
            # UPDATE matched zero rows and we must NOT log a spurious 'dismissed'
            # entry — doing so would falsely depress the accuracy multiplier.
            if was_resolved:
                self._log_automatic_feedback(
                    action_id=notif["id"],
                    action_type="notification",
                    feedback_type="dismissed",
                    context={"auto_resolved": True, "reason": "ignored", "timeout_hours": timeout_hours}
                )

            # Always mark the notification as expired so it doesn't clutter the UI,
            # regardless of whether the prediction was already resolved.
            self._mark_status(notif["id"], "expired")

        return resolved_count

    def get_diagnostics(self) -> dict:
        """Comprehensive notification pipeline diagnostics.

        Returns a detailed snapshot of the notification system's health including
        status counts, delivery statistics, domain breakdown, and actionable
        recommendations. Follows the same diagnostic pattern used by
        PredictionEngine.get_diagnostics() and InsightEngine.get_diagnostics().

        Returns:
            Dictionary with structure:
            {
                "status_counts": {"pending": int, "delivered": int, ...},
                "db_batch_depth": int,
                "delivery_mode": str,
                "recent_activity": {
                    "created_24h": int,
                    "delivered_24h": int,
                    "expired_24h": int,
                    "read_rate_7d": float
                },
                "domain_breakdown": {"prediction": int, "rule": int, ...},
                "oldest_pending_hours": float | None,
                "health": "ok" | "degraded" | "noisy",
                "recommendations": [str]
            }
        """
        diagnostics: dict = {
            "status_counts": {},
            "db_batch_depth": 0,  # Populated below from DB (replaces in-memory list)
            "delivery_mode": getattr(self, "_delivery_mode", "unknown"),
            "recent_activity": {},
            "domain_breakdown": {},
            "oldest_pending_hours": None,
            "health": "ok",
            "recommendations": [],
        }

        now = datetime.now(UTC)
        day_ago = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # --- Status counts ---
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM notifications GROUP BY status"
                ).fetchall()
            diagnostics["status_counts"] = {row["status"]: row["cnt"] for row in rows}
            # Derive batch depth from the 'batched' status bucket (DB-backed, crash-safe).
            diagnostics["db_batch_depth"] = diagnostics["status_counts"].get("batched", 0)
        except Exception:
            logger.warning("Diagnostics: failed to query status counts", exc_info=True)

        # --- Recent activity (24h) ---
        try:
            with self.db.get_connection("state") as conn:
                created_24h = conn.execute(
                    "SELECT COUNT(*) as cnt FROM notifications WHERE created_at > ?",
                    (day_ago,),
                ).fetchone()["cnt"]

                delivered_24h = conn.execute(
                    "SELECT COUNT(*) as cnt FROM notifications WHERE delivered_at > ?",
                    (day_ago,),
                ).fetchone()["cnt"]

                expired_24h = conn.execute(
                    "SELECT COUNT(*) as cnt FROM notifications WHERE status = 'expired' AND created_at > ?",
                    (day_ago,),
                ).fetchone()["cnt"]

            diagnostics["recent_activity"] = {
                "created_24h": created_24h,
                "delivered_24h": delivered_24h,
                "expired_24h": expired_24h,
            }
        except Exception:
            logger.warning("Diagnostics: failed to query recent activity", exc_info=True)

        # --- Read rate over last 7 days ---
        try:
            with self.db.get_connection("state") as conn:
                # Denominator: notifications that reached the user (delivered, read, acted_on, dismissed)
                engagement_rows = conn.execute(
                    """SELECT status, COUNT(*) as cnt FROM notifications
                       WHERE status IN ('delivered', 'read', 'acted_on', 'dismissed')
                         AND created_at > ?
                       GROUP BY status""",
                    (week_ago,),
                ).fetchall()

            engagement = {row["status"]: row["cnt"] for row in engagement_rows}
            read_count = engagement.get("read", 0) + engagement.get("acted_on", 0)
            total_engaged = sum(engagement.values())
            read_rate = round(read_count / total_engaged, 3) if total_engaged > 0 else 0.0
            diagnostics["recent_activity"]["read_rate_7d"] = read_rate
        except Exception:
            logger.warning("Diagnostics: failed to compute read rate", exc_info=True)
            diagnostics["recent_activity"].setdefault("read_rate_7d", 0.0)

        # --- Domain breakdown ---
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    "SELECT domain, COUNT(*) as cnt FROM notifications GROUP BY domain"
                ).fetchall()
            diagnostics["domain_breakdown"] = {
                (row["domain"] or "unknown"): row["cnt"] for row in rows
            }
        except Exception:
            logger.warning("Diagnostics: failed to query domain breakdown", exc_info=True)

        # --- Oldest pending notification age ---
        try:
            with self.db.get_connection("state") as conn:
                oldest = conn.execute(
                    "SELECT MIN(created_at) as oldest FROM notifications WHERE status = 'pending'"
                ).fetchone()
            if oldest and oldest["oldest"]:
                oldest_dt = datetime.fromisoformat(oldest["oldest"].replace("Z", "+00:00"))
                age_hours = round((now - oldest_dt).total_seconds() / 3600, 1)
                diagnostics["oldest_pending_hours"] = age_hours
        except Exception:
            logger.warning("Diagnostics: failed to compute oldest pending age", exc_info=True)

        # --- Health assessment ---
        status_counts = diagnostics["status_counts"]
        pending_count = status_counts.get("pending", 0)
        recent = diagnostics.get("recent_activity", {})
        read_rate = recent.get("read_rate_7d", 0.0)
        created_24h = recent.get("created_24h", 0)
        expired_24h = recent.get("expired_24h", 0)
        oldest_hours = diagnostics["oldest_pending_hours"]

        recommendations = []

        # Check for noisy pipeline (most notifications ignored)
        if created_24h > 0 and expired_24h > created_24h * 0.7:
            diagnostics["health"] = "noisy"
            recommendations.append(
                f"High expiry rate: {expired_24h}/{created_24h} notifications expired in 24h. "
                "Consider tightening prediction confidence gates or notification suppression rules."
            )

        # Check for degraded state (only flag low read rate if there are delivered notifications)
        total_notifications = sum(status_counts.values())
        if read_rate < 0.1 and total_notifications > 0 and diagnostics["health"] != "noisy":
            diagnostics["health"] = "degraded"
            recommendations.append(
                f"Low read rate ({read_rate:.1%} over 7d). Notifications may not be reaching users "
                "or content may not be compelling enough."
            )

        if pending_count > 50:
            diagnostics["health"] = "degraded"
            recommendations.append(
                f"{pending_count} pending notifications — batch queue may be stuck. "
                "Check digest delivery schedule."
            )

        if oldest_hours is not None and oldest_hours > 48:
            diagnostics["health"] = "degraded"
            recommendations.append(
                f"Oldest pending notification is {oldest_hours:.0f}h old. "
                "Run expire_stale_notifications() or check the digest schedule."
            )

        diagnostics["recommendations"] = recommendations

        # --- Quiet hours state ---
        try:
            diagnostics["quiet_hours_active"] = self._is_quiet_hours(now)
        except Exception:
            logger.warning("Diagnostics: failed to check quiet hours", exc_info=True)
            diagnostics["quiet_hours_active"] = False

        return diagnostics

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

        now = datetime.now(UTC)
        cutoff = (now - timedelta(hours=timeout_hours)).isoformat()

        # Find unsurfaced predictions created more than timeout_hours ago
        # that are still unresolved.
        try:
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
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            # Fail-open: filtered prediction cleanup is a maintenance task.
            # A corrupt user_model.db should not crash the background loop.
            logger.warning(
                "Failed to auto-resolve filtered predictions (user_model.db may be corrupted)",
                exc_info=True,
            )
            return 0

        return resolved_count
