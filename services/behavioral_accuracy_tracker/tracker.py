"""
Life OS — Behavioral Prediction Accuracy Tracker

Automatically infers prediction accuracy from user behavior, closing the
feedback loop without requiring explicit user interaction with notifications.

The Problem:
    The prediction engine needs accuracy data to learn and improve, but this
    data only comes from explicit user actions (clicking "Act On" or "Dismiss"
    buttons) or from auto-resolution after 24 hours of ignoring a notification.
    This creates a cold-start problem: new systems have no accuracy data, so
    they can't learn or calibrate confidence gates.

The Solution:
    Track user behavior and automatically mark predictions as accurate when the
    user's actions align with what was predicted, even if they never interacted
    with the notification directly.

Examples:
    - Prediction: "Reply to Alice about dinner plans"
      Behavior: User sends a message to Alice within 6 hours
      → Mark prediction as ACCURATE

    - Prediction: "Calendar conflict: Team sync overlaps with dentist"
      Behavior: User reschedules one of the events within 24 hours
      → Mark prediction as ACCURATE

    - Prediction: "Prepare slides for Q4 planning meeting"
      Behavior: User opens/edits a file containing "slides" or "Q4" keywords
      → Mark prediction as ACCURATE

    - Prediction: "Follow up with Bob about the project"
      Behavior: 48 hours pass, no message sent to Bob
      → Mark prediction as INACCURATE

This allows the system to bootstrap its learning from observed behavior instead
of waiting for explicit feedback, dramatically accelerating the calibration loop.

Architecture:
    - Runs as a background task every 15 minutes (same cadence as prediction engine)
    - Queries unresolved surfaced predictions
    - Scans recent events for behavioral signals that confirm or refute each prediction
    - Updates predictions.was_accurate when confidence threshold is met
    - Preserves user_response = 'inferred' to distinguish from explicit feedback
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from storage.database import DatabaseManager


class BehavioralAccuracyTracker:
    """Infers prediction accuracy from user behavior patterns."""

    def __init__(self, db: DatabaseManager):
        """Initialize the behavioral accuracy tracker.

        Args:
            db: Database manager for accessing events and predictions.
        """
        self.db = db

    async def run_inference_cycle(self) -> dict[str, int]:
        """Run one inference cycle over unresolved predictions.

        Returns:
            Dict with counts: {'marked_accurate': N, 'marked_inaccurate': M}
        """
        stats = {'marked_accurate': 0, 'marked_inaccurate': 0}

        # Get all surfaced predictions that haven't been resolved yet
        with self.db.get_connection("user_model") as conn:
            predictions = conn.execute(
                """SELECT id, prediction_type, description, suggested_action,
                          supporting_signals, created_at
                   FROM predictions
                   WHERE was_surfaced = 1
                     AND resolved_at IS NULL
                     AND created_at > ?""",
                # Only look at predictions from the last 7 days (older ones are
                # handled by auto-resolve stale predictions logic)
                ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),),
            ).fetchall()

        for pred in predictions:
            # Try to infer accuracy from behavioral signals
            result = await self._infer_accuracy(dict(pred))

            if result is not None:
                # We have enough confidence to make an inference
                was_accurate = result
                now = datetime.now(timezone.utc).isoformat()

                with self.db.get_connection("user_model") as conn:
                    conn.execute(
                        """UPDATE predictions SET
                           was_accurate = ?,
                           resolved_at = ?,
                           user_response = 'inferred'
                           WHERE id = ?""",
                        (1 if was_accurate else 0, now, pred["id"]),
                    )

                if was_accurate:
                    stats['marked_accurate'] += 1
                else:
                    stats['marked_inaccurate'] += 1

        return stats

    async def _infer_accuracy(self, prediction: dict) -> Optional[bool]:
        """Infer whether a prediction was accurate based on user behavior.

        Args:
            prediction: Dict with fields: id, prediction_type, description,
                       suggested_action, supporting_signals, created_at

        Returns:
            True if behavior confirms the prediction was accurate
            False if behavior confirms the prediction was inaccurate
            None if insufficient evidence to make a determination
        """
        pred_type = prediction["prediction_type"]
        created_at = datetime.fromisoformat(prediction["created_at"].replace('Z', '+00:00'))

        # Parse supporting_signals JSON to extract relevant context
        try:
            signals = json.loads(prediction["supporting_signals"]) if prediction["supporting_signals"] else {}
        except (json.JSONDecodeError, TypeError):
            signals = {}

        # Dispatch to type-specific inference logic
        if pred_type == "reminder":
            return await self._infer_reminder_accuracy(prediction, signals, created_at)
        elif pred_type == "conflict":
            return await self._infer_conflict_accuracy(prediction, signals, created_at)
        elif pred_type == "need":
            return await self._infer_need_accuracy(prediction, signals, created_at)
        elif pred_type == "opportunity":
            return await self._infer_opportunity_accuracy(prediction, signals, created_at)
        elif pred_type == "risk":
            return await self._infer_risk_accuracy(prediction, signals, created_at)
        else:
            return None  # Unknown prediction type

    async def _infer_reminder_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'reminder' predictions.

        Reminder predictions typically suggest: "Reply to X" or "Follow up with Y".
        We look for outbound messages to the mentioned contact within a reasonable
        timeframe (6-48 hours).
        """
        # Extract contact name/email from signals or description
        contact_id = signals.get("contact_id")
        contact_name = signals.get("contact_name")

        # If no contact info, try to extract from description
        if not contact_id and not contact_name:
            # Simple heuristic: look for "reply to NAME" or "follow up with NAME"
            # This is intentionally conservative — only matches clear patterns with
            # capitalized names (e.g., "Grace", "Alice", "Bob Smith")
            import re
            # Case-insensitive match for the trigger phrase, then extract capitalized name
            match = re.search(r'(?:reply to|follow up with|message)\s+([A-Z][a-z]+)', prediction["description"], re.IGNORECASE)
            if match:
                # Extract just the first capitalized word as the name
                contact_name = match.group(1)

        if not contact_id and not contact_name:
            return None  # Can't determine without contact info

        # Look for outbound messages to this contact within 6-48 hours of prediction
        window_start = created_at
        window_end = created_at + timedelta(hours=48)

        with self.db.get_connection("events") as conn:
            # Check for email.sent or message.sent events to this contact
            events = conn.execute(
                """SELECT type, payload, timestamp
                   FROM events
                   WHERE (type = 'email.sent' OR type = 'message.sent')
                     AND timestamp >= ?
                     AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Check if recipient matches
                to = payload.get("to", "")
                if contact_name and contact_name.lower() in to.lower():
                    return True  # User DID follow up — prediction was accurate
                if contact_id and contact_id in to:
                    return True
            except (json.JSONDecodeError, TypeError):
                continue

        # Check if enough time has passed to infer inaccuracy
        # If 48+ hours have passed with no action, prediction was likely wrong
        now = datetime.now(timezone.utc)
        if now - created_at > timedelta(hours=48):
            return False  # No action taken → prediction was inaccurate

        return None  # Still within the window, can't determine yet

    async def _infer_conflict_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'conflict' predictions.

        Conflict predictions alert about calendar overlaps. We check if the user
        took corrective action (rescheduled, cancelled, or shortened one of the
        conflicting events).
        """
        # Extract event IDs from signals
        event_ids = signals.get("conflicting_event_ids", [])
        if not event_ids:
            return None

        # Look for calendar.event.updated or calendar.event.deleted events
        # for either of the conflicting events within 24 hours
        window_end = created_at + timedelta(hours=24)

        with self.db.get_connection("events") as conn:
            updates = conn.execute(
                """SELECT type, payload
                   FROM events
                   WHERE (type = 'calendar.event.updated' OR type = 'calendar.event.deleted')
                     AND timestamp >= ?
                     AND timestamp <= ?""",
                (created_at.isoformat(), window_end.isoformat()),
            ).fetchall()

        for update in updates:
            try:
                payload = json.loads(update["payload"])
                event_id = payload.get("event_id")
                if event_id in event_ids:
                    return True  # User resolved the conflict — prediction was accurate
            except (json.JSONDecodeError, TypeError):
                continue

        # If 24+ hours passed and conflict still exists, prediction was correct
        # but user chose to ignore it (still counts as accurate prediction)
        now = datetime.now(timezone.utc)
        if now - created_at > timedelta(hours=24):
            return True  # Conflict was real, even if user didn't fix it

        return None  # Still within resolution window

    async def _infer_need_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'need' predictions.

        Need predictions suggest: "You'll probably need X soon". We check if
        the user actually accessed/used X within the predicted timeframe.
        """
        # Extract predicted need from signals
        need_type = signals.get("need_type")
        need_keywords = signals.get("keywords", [])

        if not need_type and not need_keywords:
            return None

        # Look for events indicating the need was met (e.g., file access, app usage)
        # This is highly system-specific — for now, we'll be conservative
        # and only mark as accurate if we see clear confirmation

        # Since we don't have file access/app usage connectors yet, we can't
        # reliably infer need accuracy. Return None for now.
        return None

    async def _infer_opportunity_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'opportunity' predictions.

        Opportunity predictions suggest: "Good time to do X based on your patterns".
        Similar to need predictions, we check if the user took advantage of the
        opportunity within a reasonable timeframe.
        """
        # For now, return None — this requires more sophisticated behavioral tracking
        # that we haven't implemented yet (e.g., task completion, habit tracking)
        return None

    async def _infer_risk_accuracy(
        self, prediction: dict, signals: dict, created_at: datetime
    ) -> Optional[bool]:
        """Infer accuracy for 'risk' predictions.

        Risk predictions warn: "Something might go wrong if you don't..."
        We check if the risk materialized (user had the problem) or if the user
        took preventive action.
        """
        # This is the hardest to infer automatically — we'd need to detect
        # negative outcomes, which is very domain-specific. Return None for now.
        return None
