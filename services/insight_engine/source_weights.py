"""
Life OS -- Source Weight Manager

Manages the dual-weight system that controls how much influence each data
source has on insights, predictions, and signal extraction.

Every source has two components:
    user_weight  — set explicitly by the user (0.0 = ignore, 1.0 = max influence)
    ai_drift     — learned adjustment that the AI shifts over time based on
                   engagement patterns, bounded to [-0.3, +0.3]

    effective_weight = clamp(user_weight + ai_drift, 0.0, 1.0)

The user always retains primary control: they can see both values, override
the user_weight at any time, and reset the AI drift.  The AI drift is
transparent — its reason and history are stored and surfaced in the UI.

Source Classification:
    Events are classified into source_keys based on their type, payload, and
    metadata.  For example, an email.received event may be classified as
    "email.marketing", "email.personal", or "email.work" based on sender
    domain, headers, and content heuristics.

AI Drift Algorithm:
    - On each insight feedback (engagement/dismissal), the relevant source's
      drift is nudged by ±DRIFT_STEP (0.02).
    - Drift is bounded to [-MAX_DRIFT, +MAX_DRIFT] (±0.3).
    - Drift decays toward zero with a 28-day half-life when no feedback
      is received, preventing stale adjustments from persisting.
    - A minimum of MIN_INTERACTIONS (5) must occur before drift is applied,
      so the AI doesn't overreact to sparse data.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)

# -- Drift tuning constants --------------------------------------------------

DRIFT_STEP = 0.02        # How much each feedback signal nudges the drift
MAX_DRIFT = 0.3           # Absolute cap on AI drift (user stays in control)
MIN_INTERACTIONS = 5      # Minimum events before drift kicks in
DECAY_HALF_LIFE_DAYS = 28 # Drift decays toward zero with this half-life

# -- Default source weight seeds ---------------------------------------------
# These are created on first run.  Users can adjust; AI will drift from here.

DEFAULT_WEIGHTS: list[dict[str, Any]] = [
    # Email
    {
        "source_key": "email.personal",
        "category": "email",
        "label": "Personal Email",
        "description": "Email from known personal contacts",
        "user_weight": 0.8,
    },
    {
        "source_key": "email.work",
        "category": "email",
        "label": "Work Email",
        "description": "Email from work domains and colleagues",
        "user_weight": 0.7,
    },
    {
        "source_key": "email.marketing",
        "category": "email",
        "label": "Marketing & Promotions",
        "description": "Newsletters, promotions, and bulk marketing emails",
        "user_weight": 0.15,
    },
    {
        "source_key": "email.transactional",
        "category": "email",
        "label": "Receipts & Confirmations",
        "description": "Order confirmations, shipping notifications, receipts",
        "user_weight": 0.3,
    },
    {
        "source_key": "email.newsletter",
        "category": "email",
        "label": "Newsletters",
        "description": "Subscribed newsletters and digests",
        "user_weight": 0.35,
    },
    # Messaging
    {
        "source_key": "messaging.direct",
        "category": "messaging",
        "label": "Direct Messages",
        "description": "1-on-1 messages from contacts",
        "user_weight": 0.9,
    },
    {
        "source_key": "messaging.group",
        "category": "messaging",
        "label": "Group Chats",
        "description": "Group conversations and channels",
        "user_weight": 0.5,
    },
    {
        "source_key": "messaging.bot",
        "category": "messaging",
        "label": "Automated / Bot Messages",
        "description": "Automated notifications from bots and integrations",
        "user_weight": 0.2,
    },
    # Calendar
    {
        "source_key": "calendar.meetings",
        "category": "calendar",
        "label": "Meetings",
        "description": "Scheduled meetings and calls",
        "user_weight": 0.7,
    },
    {
        "source_key": "calendar.reminders",
        "category": "calendar",
        "label": "Reminders & Events",
        "description": "Personal reminders and non-meeting events",
        "user_weight": 0.6,
    },
    # Finance
    {
        "source_key": "finance.transactions",
        "category": "finance",
        "label": "Transactions",
        "description": "Purchase and payment activity",
        "user_weight": 0.5,
    },
    {
        "source_key": "finance.subscriptions",
        "category": "finance",
        "label": "Subscriptions",
        "description": "Recurring subscription charges",
        "user_weight": 0.4,
    },
    # Health
    {
        "source_key": "health.sleep",
        "category": "health",
        "label": "Sleep Data",
        "description": "Sleep duration and quality metrics",
        "user_weight": 0.8,
    },
    {
        "source_key": "health.activity",
        "category": "health",
        "label": "Activity & Exercise",
        "description": "Exercise sessions and activity metrics",
        "user_weight": 0.6,
    },
    # Location
    {
        "source_key": "location.visits",
        "category": "location",
        "label": "Location Visits",
        "description": "Place visits and movement patterns",
        "user_weight": 0.4,
    },
    # Smart Home
    {
        "source_key": "home.devices",
        "category": "home",
        "label": "Smart Home Devices",
        "description": "Device state changes and automations",
        "user_weight": 0.3,
    },
]

# -- Marketing / bulk-mail heuristics ----------------------------------------

_MARKETING_HEADERS = {"list-unsubscribe", "x-mailer", "x-campaign"}
_MARKETING_SENDER_PATTERNS = [
    re.compile(r"(no-?reply|newsletter|marketing|promo|info|updates|news)@", re.I),
    re.compile(r"@(mailchimp|sendgrid|constantcontact|hubspot|campaign)", re.I),
]
_TRANSACTIONAL_PATTERNS = [
    re.compile(r"(receipt|invoice|order.confirm|shipping|delivery|payment)", re.I),
]


class SourceWeightManager:
    """Manages the dual-weight (user + AI drift) system for data sources."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def seed_defaults(self):
        """Insert default source weights if the table is empty.

        Called during system startup.  Uses INSERT OR IGNORE so existing
        user-customized rows are never overwritten.
        """
        with self.db.get_connection("preferences") as conn:
            for w in DEFAULT_WEIGHTS:
                conn.execute(
                    """INSERT OR IGNORE INTO source_weights
                       (source_key, category, label, description, user_weight)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        w["source_key"],
                        w["category"],
                        w["label"],
                        w.get("description", ""),
                        w["user_weight"],
                    ),
                )

    # ------------------------------------------------------------------
    # Source Classification
    # ------------------------------------------------------------------

    def classify_event(self, event: dict) -> str:
        """Classify an event into a source_key for weight lookup.

        Returns a dotted key like "email.marketing" or "calendar.meetings".
        Falls back to a generic key based on the event type prefix if no
        specific classification rule matches.
        """
        event_type = event.get("type", "")
        payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
        metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}

        # --- Email classification ---
        if event_type in ("email.received", "email.sent"):
            return self._classify_email(payload, metadata)

        # --- Messaging classification ---
        if event_type in ("message.received", "message.sent"):
            return self._classify_message(payload, metadata)

        # --- Calendar ---
        if event_type.startswith("calendar."):
            attendees = payload.get("attendees", [])
            if attendees or "meeting" in payload.get("title", "").lower():
                return "calendar.meetings"
            return "calendar.reminders"

        # --- Finance ---
        if event_type == "finance.transaction.new":
            return "finance.transactions"
        if event_type == "finance.subscription.detected":
            return "finance.subscriptions"

        # --- Health ---
        if event_type == "sleep.recorded":
            return "health.sleep"
        if event_type in ("exercise.recorded", "health.metric.updated"):
            return "health.activity"

        # --- Location ---
        if event_type.startswith("location."):
            return "location.visits"

        # --- Smart Home ---
        if event_type.startswith("home."):
            return "home.devices"

        # Fallback: use the first segment of the event type
        prefix = event_type.split(".")[0] if "." in event_type else event_type
        return f"{prefix}.general"

    def _classify_email(self, payload: dict, metadata: dict) -> str:
        """Sub-classify an email event."""
        sender = (payload.get("from", "") or payload.get("sender", "")).lower()
        subject = (payload.get("subject", "")).lower()
        headers = payload.get("headers", {})

        # Check for transactional first (most specific)
        for pattern in _TRANSACTIONAL_PATTERNS:
            if pattern.search(subject) or pattern.search(sender):
                return "email.transactional"

        # Check for newsletter before generic marketing (newsletter is a
        # subset of marketing but users may want to weight them differently)
        if "newsletter" in sender or "digest" in subject or "weekly" in subject:
            return "email.newsletter"

        # Check for marketing indicators (headers, sender patterns)
        if isinstance(headers, dict):
            header_keys = {k.lower() for k in headers}
            if header_keys & _MARKETING_HEADERS:
                return "email.marketing"

        for pattern in _MARKETING_SENDER_PATTERNS:
            if pattern.search(sender):
                return "email.marketing"

        # Distinguish work vs personal by domain heuristics
        domain = metadata.get("domain", "")
        if not domain and "@" in sender:
            domain = sender.split("@")[-1]

        # Common personal email domains
        personal_domains = {
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
            "icloud.com", "me.com", "aol.com", "protonmail.com",
            "proton.me", "fastmail.com",
        }
        if domain in personal_domains:
            return "email.personal"

        # If the domain matches the user's work domain (stored in metadata
        # or contacts), classify as work.  Default to work for unknown domains.
        return "email.work"

    def _classify_message(self, payload: dict, metadata: dict) -> str:
        """Sub-classify a messaging event."""
        channel_type = payload.get("channel_type", "")
        is_group = payload.get("is_group", False)
        sender = (payload.get("from", "") or payload.get("sender", "")).lower()

        # Bot detection
        if payload.get("is_bot") or "bot" in sender:
            return "messaging.bot"

        if is_group or channel_type in ("group", "channel"):
            return "messaging.group"

        return "messaging.direct"

    # ------------------------------------------------------------------
    # Weight Retrieval
    # ------------------------------------------------------------------

    def get_effective_weight(self, source_key: str) -> float:
        """Get the effective weight for a source, applying time-decay to drift.

        Returns a float in [0.0, 1.0].  If the source_key is unknown, returns
        0.5 (neutral default).
        """
        row = self._get_weight_row(source_key)
        if not row:
            return 0.5

        user_weight = row["user_weight"]
        ai_drift = row["ai_drift"]

        # Apply time-decay to drift
        ai_drift = self._decay_drift(ai_drift, row.get("ai_updated_at"))

        return max(0.0, min(1.0, user_weight + ai_drift))

    def get_all_weights(self) -> list[dict]:
        """Return all source weights with computed effective values."""
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                """SELECT * FROM source_weights ORDER BY category, source_key"""
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            ai_drift = self._decay_drift(d["ai_drift"], d.get("ai_updated_at"))
            d["effective_weight"] = max(0.0, min(1.0, d["user_weight"] + ai_drift))
            d["ai_drift_decayed"] = round(ai_drift, 4)
            # Parse JSON fields
            try:
                d["drift_history"] = json.loads(d.get("drift_history", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["drift_history"] = []
            results.append(d)

        return results

    def get_weights_by_category(self) -> dict[str, list[dict]]:
        """Return weights grouped by category for UI display."""
        all_weights = self.get_all_weights()
        grouped: dict[str, list[dict]] = {}
        for w in all_weights:
            cat = w["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(w)
        return grouped

    def _get_weight_row(self, source_key: str) -> Optional[dict]:
        """Fetch a single weight row by source_key."""
        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT * FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # User Weight Updates
    # ------------------------------------------------------------------

    def set_user_weight(self, source_key: str, weight: float) -> dict:
        """Set the user-controlled weight for a source.

        Args:
            source_key: The source identifier (e.g. "email.marketing")
            weight: New weight value, clamped to [0.0, 1.0]

        Returns:
            The updated weight record.
        """
        weight = max(0.0, min(1.0, weight))
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            # Verify the source exists
            existing = conn.execute(
                "SELECT * FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()

            if not existing:
                raise ValueError(f"Unknown source_key: {source_key}")

            conn.execute(
                """UPDATE source_weights
                   SET user_weight = ?, user_set_at = ?
                   WHERE source_key = ?""",
                (weight, now, source_key),
            )

        return self._get_weight_row(source_key)

    def reset_ai_drift(self, source_key: str) -> dict:
        """Reset the AI drift to zero for a source.

        Called when the user wants to clear the AI's learned adjustment,
        typically after changing their own weight significantly.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            existing = conn.execute(
                "SELECT * FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()

            if not existing:
                raise ValueError(f"Unknown source_key: {source_key}")

            # Record the reset in drift history
            history = json.loads(existing["drift_history"] or "[]")
            history.append({
                "timestamp": now,
                "old_drift": existing["ai_drift"],
                "new_drift": 0.0,
                "reason": "user_reset",
            })
            # Keep last 50 history entries
            history = history[-50:]

            conn.execute(
                """UPDATE source_weights
                   SET ai_drift = 0.0, drift_reason = 'reset by user',
                       drift_history = ?, ai_updated_at = ?
                   WHERE source_key = ?""",
                (json.dumps(history), now, source_key),
            )

        return self._get_weight_row(source_key)

    def add_source(self, source_key: str, category: str, label: str,
                   description: str = "", user_weight: float = 0.5) -> dict:
        """Add a new custom source weight (user-created).

        Users can create custom source classifications beyond the defaults,
        e.g., "email.client_acme" for a specific business contact.
        """
        user_weight = max(0.0, min(1.0, user_weight))

        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT OR IGNORE INTO source_weights
                   (source_key, category, label, description, user_weight)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_key, category, label, description, user_weight),
            )

        return self._get_weight_row(source_key)

    # ------------------------------------------------------------------
    # AI Drift Learning
    # ------------------------------------------------------------------

    def record_interaction(self, source_key: str):
        """Record that an event from this source was processed.

        Called by the event pipeline for every event, building the
        interaction count that gates when AI drift becomes active.
        """
        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """UPDATE source_weights
                   SET interactions = interactions + 1
                   WHERE source_key = ?""",
                (source_key,),
            )

    def record_engagement(self, source_key: str):
        """Record that the user engaged with an insight from this source.

        Engagement = user clicked, acted on, or marked an insight as useful.
        Nudges the AI drift upward.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT * FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()

            if not row:
                return

            conn.execute(
                "UPDATE source_weights SET engagements = engagements + 1 WHERE source_key = ?",
                (source_key,),
            )

            # Only apply drift after minimum interactions threshold
            if row["interactions"] < MIN_INTERACTIONS:
                return

            old_drift = row["ai_drift"]
            new_drift = min(MAX_DRIFT, old_drift + DRIFT_STEP)

            # Check for drift saturation and warn if user preference becomes invisible
            saturation = self._check_drift_saturation(source_key, row["user_weight"], new_drift)
            if saturation:
                logger.warning(
                    "Source weight drift saturated for %s: drift=%s, user_weight=%s, "
                    "effective_weight=%s. User preference is no longer visible.",
                    source_key,
                    new_drift,
                    row["user_weight"],
                    saturation["effective_weight"],
                )

            history = json.loads(row["drift_history"] or "[]")
            entry: dict = {
                "timestamp": now,
                "old_drift": round(old_drift, 4),
                "new_drift": round(new_drift, 4),
                "reason": "engagement",
            }
            if saturation:
                entry["saturated"] = True
            history.append(entry)
            history = history[-50:]

            conn.execute(
                """UPDATE source_weights
                   SET ai_drift = ?, drift_reason = 'increased due to engagement',
                       drift_history = ?, ai_updated_at = ?
                   WHERE source_key = ?""",
                (new_drift, json.dumps(history), now, source_key),
            )

    def record_dismissal(self, source_key: str):
        """Record that the user dismissed an insight from this source.

        Dismissal = user swiped away, dismissed, or marked as not useful.
        Nudges the AI drift downward.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT * FROM source_weights WHERE source_key = ?",
                (source_key,),
            ).fetchone()

            if not row:
                return

            conn.execute(
                "UPDATE source_weights SET dismissals = dismissals + 1 WHERE source_key = ?",
                (source_key,),
            )

            if row["interactions"] < MIN_INTERACTIONS:
                return

            old_drift = row["ai_drift"]
            new_drift = max(-MAX_DRIFT, old_drift - DRIFT_STEP)

            # Check for drift saturation and warn if user preference becomes invisible
            saturation = self._check_drift_saturation(source_key, row["user_weight"], new_drift)
            if saturation:
                logger.warning(
                    "Source weight drift saturated for %s: drift=%s, user_weight=%s, "
                    "effective_weight=%s. User preference is no longer visible.",
                    source_key,
                    new_drift,
                    row["user_weight"],
                    saturation["effective_weight"],
                )

            history = json.loads(row["drift_history"] or "[]")
            entry: dict = {
                "timestamp": now,
                "old_drift": round(old_drift, 4),
                "new_drift": round(new_drift, 4),
                "reason": "dismissal",
            }
            if saturation:
                entry["saturated"] = True
            history.append(entry)
            history = history[-50:]

            conn.execute(
                """UPDATE source_weights
                   SET ai_drift = ?, drift_reason = 'decreased due to dismissal',
                       drift_history = ?, ai_updated_at = ?
                   WHERE source_key = ?""",
                (new_drift, json.dumps(history), now, source_key),
            )

    def bulk_recalculate_drift(self):
        """Recalculate AI drift for all sources based on engagement ratios.

        Called periodically (e.g. daily) to adjust drift based on overall
        patterns rather than individual feedback events.  This catches
        sources where the user consistently ignores content but never
        explicitly dismisses it.

        Algorithm:
            engagement_rate = engagements / max(interactions, 1)
            global_avg_rate = sum(all engagement_rates) / count(sources)

            If a source's rate is significantly below average, drift down.
            If significantly above average, drift up.
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                "SELECT * FROM source_weights WHERE interactions >= ?",
                (MIN_INTERACTIONS,),
            ).fetchall()

        if not rows:
            return

        # Compute per-source engagement rates
        rates = []
        for row in rows:
            total = row["engagements"] + row["dismissals"]
            if total > 0:
                rate = row["engagements"] / total
            else:
                rate = 0.5  # Neutral if no feedback at all
            rates.append((dict(row), rate))

        if not rates:
            return

        avg_rate = sum(r for _, r in rates) / len(rates)
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            for row_dict, rate in rates:
                deviation = rate - avg_rate

                # Only nudge if the deviation is meaningful (> 10%)
                if abs(deviation) < 0.1:
                    continue

                old_drift = row_dict["ai_drift"]
                # Nudge proportionally to deviation, but cap the step
                nudge = max(-DRIFT_STEP * 2, min(DRIFT_STEP * 2, deviation * 0.1))
                new_drift = max(-MAX_DRIFT, min(MAX_DRIFT, old_drift + nudge))

                if abs(new_drift - old_drift) < 0.001:
                    continue

                # Check for drift saturation and warn if user preference becomes invisible
                saturation = self._check_drift_saturation(
                    row_dict["source_key"], row_dict["user_weight"], new_drift
                )
                if saturation:
                    logger.warning(
                        "Source weight drift saturated for %s: drift=%s, user_weight=%s, "
                        "effective_weight=%s. User preference is no longer visible.",
                        row_dict["source_key"],
                        new_drift,
                        row_dict["user_weight"],
                        saturation["effective_weight"],
                    )

                history = json.loads(row_dict.get("drift_history", "[]"))
                reason = "above" if deviation > 0 else "below"
                entry: dict = {
                    "timestamp": now,
                    "old_drift": round(old_drift, 4),
                    "new_drift": round(new_drift, 4),
                    "reason": f"bulk_recalc: engagement rate {reason} average",
                }
                if saturation:
                    entry["saturated"] = True
                history.append(entry)
                history = history[-50:]

                conn.execute(
                    """UPDATE source_weights
                       SET ai_drift = ?,
                           drift_reason = ?,
                           drift_history = ?,
                           ai_updated_at = ?
                       WHERE source_key = ?""",
                    (
                        new_drift,
                        f"engagement rate {rate:.0%} vs avg {avg_rate:.0%}",
                        json.dumps(history),
                        now,
                        row_dict["source_key"],
                    ),
                )

    # ------------------------------------------------------------------
    # Drift Saturation Check
    # ------------------------------------------------------------------

    def _check_drift_saturation(
        self, source_key: str, user_weight: float, new_drift: float
    ) -> dict:
        """Check whether the given drift saturates the effective weight.

        Saturation occurs when:
          - abs(new_drift) == MAX_DRIFT (AI drift is at its hard cap), OR
          - the resulting effective weight is clamped to exactly 0.0 or 1.0
            (meaning the user's explicit weight preference becomes invisible).

        At saturation the user-controlled weight no longer affects the effective
        weight, which undermines the design guarantee that the user stays in
        primary control.

        Args:
            source_key:  The source identifier (used for logging context).
            user_weight: The user-set weight for the source.
            new_drift:   The candidate new drift value (post-nudge, pre-save).

        Returns:
            A non-empty dict with saturation details if saturated, empty dict
            otherwise.  Keys when saturated: ``source_key``, ``user_weight``,
            ``drift``, ``effective_weight``, ``saturated`` (always True).
        """
        effective = max(0.0, min(1.0, user_weight + new_drift))
        is_saturated = (
            abs(new_drift) >= MAX_DRIFT
            or user_weight + new_drift <= 0.0
            or user_weight + new_drift >= 1.0
        )
        if not is_saturated:
            return {}
        return {
            "saturated": True,
            "source_key": source_key,
            "user_weight": user_weight,
            "drift": new_drift,
            "effective_weight": effective,
        }

    # ------------------------------------------------------------------
    # Drift Decay
    # ------------------------------------------------------------------

    def _decay_drift(self, drift: float, last_updated: Optional[str]) -> float:
        """Apply exponential time-decay to an AI drift value.

        Drift decays toward zero with a half-life of DECAY_HALF_LIFE_DAYS.
        This prevents stale drift from persisting forever when a source
        stops generating feedback.
        """
        if not last_updated or drift == 0.0:
            return drift

        try:
            updated_dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_elapsed = (now - updated_dt).total_seconds() / 86400

            if days_elapsed <= 0:
                return drift

            # Exponential decay: drift * 0.5^(days / half_life)
            decay_factor = math.pow(0.5, days_elapsed / DECAY_HALF_LIFE_DAYS)
            return drift * decay_factor
        except (ValueError, TypeError):
            return drift

    # ------------------------------------------------------------------
    # Statistics & Debugging
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """Return diagnostic information about the source weight system.

        Provides observability into the feedback loop health, drift activity,
        and per-source statistics. Follows the same try/except pattern as
        other service diagnostics (PredictionEngine, RoutineDetector, etc.)
        so a single DB failure doesn't prevent other diagnostics from returning.

        Returns:
            Dict with keys: total_sources, total_interactions, total_engagements,
            total_dismissals, sources_with_drift, feedback_loop_health,
            per_source, stale_sources, drift_active.
        """
        result: dict = {
            "total_sources": 0,
            "total_interactions": 0,
            "total_engagements": 0,
            "total_dismissals": 0,
            "sources_with_drift": 0,
            "feedback_loop_health": "healthy",
            "per_source": [],
            "stale_sources": [],
            "drift_active": False,
            # Saturation diagnostics — see _check_drift_saturation for details
            "saturated_sources": [],
            "drift_health": "inactive",
        }

        # Aggregate counts
        try:
            with self.db.get_connection("preferences") as conn:
                row = conn.execute(
                    """SELECT
                           COUNT(*) AS total_sources,
                           COALESCE(SUM(interactions), 0) AS total_interactions,
                           COALESCE(SUM(engagements), 0) AS total_engagements,
                           COALESCE(SUM(dismissals), 0) AS total_dismissals,
                           COALESCE(SUM(CASE WHEN ai_drift != 0 THEN 1 ELSE 0 END), 0) AS sources_with_drift
                       FROM source_weights"""
                ).fetchone()

            result["total_sources"] = row["total_sources"]
            result["total_interactions"] = row["total_interactions"]
            result["total_engagements"] = row["total_engagements"]
            result["total_dismissals"] = row["total_dismissals"]
            result["sources_with_drift"] = row["sources_with_drift"]
            result["drift_active"] = row["sources_with_drift"] > 0
        except Exception:
            logger.warning("Failed to fetch source weight aggregate counts", exc_info=True)

        # Feedback loop health assessment
        try:
            total_feedback = result["total_engagements"] + result["total_dismissals"]
            if result["total_interactions"] > 100 and total_feedback == 0:
                result["feedback_loop_health"] = "broken"
            elif result["total_engagements"] > 0 and result["total_dismissals"] > 0:
                result["feedback_loop_health"] = "healthy"
            elif total_feedback > 0:
                result["feedback_loop_health"] = "partial"
            else:
                result["feedback_loop_health"] = "partial"
        except Exception:
            logger.warning("Failed to assess feedback loop health", exc_info=True)

        # Per-source details and stale source detection
        try:
            with self.db.get_connection("preferences") as conn:
                rows = conn.execute(
                    "SELECT * FROM source_weights ORDER BY source_key"
                ).fetchall()

            stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            per_source = []
            stale_sources = []
            saturated_sources = []

            for row in rows:
                d = dict(row)
                effective = max(0.0, min(1.0, d["user_weight"] + d["ai_drift"]))
                # Use ai_updated_at if available, otherwise fall back to created_at
                last_updated = d.get("ai_updated_at") or d.get("created_at")
                per_source.append({
                    "source_key": d["source_key"],
                    "interactions": d["interactions"],
                    "engagements": d["engagements"],
                    "dismissals": d["dismissals"],
                    "user_weight": d["user_weight"],
                    "ai_drift": d["ai_drift"],
                    "effective_weight": round(effective, 4),
                    "updated_at": last_updated,
                })

                # Stale: has interactions but hasn't been updated in 7+ days
                if d["interactions"] > 0 and last_updated and last_updated < stale_cutoff:
                    stale_sources.append(d["source_key"])

                # Saturated: drift at cap or effective weight is pinned at 0 or 1,
                # meaning the user's explicit weight preference has no visible effect.
                if (
                    abs(d["ai_drift"]) >= MAX_DRIFT
                    or effective == 0.0
                    or effective == 1.0
                ):
                    saturated_sources.append(d["source_key"])

            result["per_source"] = per_source
            result["stale_sources"] = stale_sources
            result["saturated_sources"] = saturated_sources

            # drift_health summarises the overall saturation state for dashboards
            if saturated_sources:
                result["drift_health"] = "saturated"
            elif result.get("drift_active"):
                result["drift_health"] = "active"
            else:
                result["drift_health"] = "inactive"
        except Exception:
            logger.warning("Failed to fetch per-source diagnostics", exc_info=True)

        return result

    def get_source_stats(self, source_key: str) -> Optional[dict]:
        """Get detailed statistics for a single source weight."""
        row = self._get_weight_row(source_key)
        if not row:
            return None

        ai_drift_decayed = self._decay_drift(row["ai_drift"], row.get("ai_updated_at"))
        effective = max(0.0, min(1.0, row["user_weight"] + ai_drift_decayed))

        total_feedback = row["engagements"] + row["dismissals"]
        engagement_rate = row["engagements"] / total_feedback if total_feedback > 0 else None

        try:
            drift_history = json.loads(row.get("drift_history", "[]"))
        except (json.JSONDecodeError, TypeError):
            drift_history = []

        return {
            "source_key": row["source_key"],
            "category": row["category"],
            "label": row["label"],
            "description": row.get("description", ""),
            "user_weight": row["user_weight"],
            "ai_drift_raw": row["ai_drift"],
            "ai_drift_decayed": round(ai_drift_decayed, 4),
            "effective_weight": round(effective, 4),
            "drift_reason": row["drift_reason"],
            "interactions": row["interactions"],
            "engagements": row["engagements"],
            "dismissals": row["dismissals"],
            "engagement_rate": round(engagement_rate, 3) if engagement_rate is not None else None,
            "drift_active": row["interactions"] >= MIN_INTERACTIONS,
            "drift_history": drift_history[-10:],  # Last 10 entries
            "user_set_at": row.get("user_set_at"),
            "ai_updated_at": row.get("ai_updated_at"),
        }
