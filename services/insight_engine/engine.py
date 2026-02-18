"""
Life OS -- Insight Engine

Backward-looking cross-signal correlator.  Unlike the PredictionEngine
(forward-looking guesses), the InsightEngine discovers patterns that
already exist in collected data and surfaces them as human-readable
insights.

Runs hourly. Deduplicates insights so the same pattern is not resurfaced
until evidence changes or the staleness TTL expires.

Insight Types:
    behavioral_pattern          -- Repeated behaviors (e.g. "You visit Cafe X 5x/week")
    actionable_alert            -- Something the user should act on now
    relationship_intelligence   -- Social-graph discoveries
    communication_style         -- Writing-style observations from linguistic profile
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.insight_engine.models import Insight
from services.insight_engine.source_weights import SourceWeightManager
from services.signal_extractor.marketing_filter import is_marketing_or_noreply
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


class InsightEngine:
    """Cross-correlates signal profiles to produce human-readable insights."""

    def __init__(self, db: DatabaseManager, ums: UserModelStore,
                 source_weight_manager: Optional[SourceWeightManager] = None):
        self.db = db
        self.ums = ums
        self.swm = source_weight_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_insights(self) -> list[Insight]:
        """Main loop: run all correlators, deduplicate, store, return new insights."""
        raw: list[Insight] = []

        # Each correlator handles its own errors gracefully and returns
        # an empty list when there is insufficient data.
        try:
            raw.extend(self._place_frequency_insights())
        except Exception:
            logger.exception("place_frequency correlator failed")

        try:
            raw.extend(self._contact_gap_insights())
        except Exception:
            logger.exception("contact_gap correlator failed")

        try:
            raw.extend(self._email_volume_insights())
        except Exception:
            logger.exception("email_volume correlator failed")

        try:
            raw.extend(self._communication_style_insights())
        except Exception:
            logger.exception("communication_style correlator failed")

        try:
            raw.extend(self._actionable_alert_insights())
        except Exception:
            logger.exception("actionable_alert correlator failed")

        # Ensure every insight has a dedup key
        for insight in raw:
            if not insight.dedup_key:
                insight.compute_dedup_key()

        # Apply source weights: modulate each insight's confidence by the
        # effective weight for its source category.  This is how user tuning
        # and AI drift influence which insights are surfaced.
        raw = self._apply_source_weights(raw)

        # Remove insights that are still within their staleness window
        fresh = self._deduplicate(raw)

        # Persist the survivors
        for insight in fresh:
            self._store_insight(insight)

        return fresh

    # ------------------------------------------------------------------
    # Source Weight Application
    # ------------------------------------------------------------------

    def _apply_source_weights(self, insights: list[Insight]) -> list[Insight]:
        """Modulate insight confidence by the effective source weight.

        Each insight's category maps to a source_key used to look up the
        user+AI effective weight.  The insight's confidence is multiplied
        by this weight, so low-weight sources produce lower-confidence
        insights that are less likely to be surfaced.

        Insights whose weighted confidence drops below 0.1 are filtered out
        entirely — no point surfacing something the user has deprioritized.
        """
        if not self.swm:
            return insights

        # Map insight categories to source weight keys.
        # actionable_alert categories are intentionally excluded: overdue tasks
        # and upcoming calendar events should always be surfaced regardless of
        # source-weight tuning — they represent direct user obligations, not
        # inferred patterns that can be deprioritized.
        category_to_source = {
            "place": "location.visits",
            "contact_gap": "messaging.direct",
            "email_volume": "email.work",
            "communication_style": "messaging.direct",
        }

        weighted: list[Insight] = []
        for insight in insights:
            source_key = category_to_source.get(insight.category)
            if source_key:
                weight = self.swm.get_effective_weight(source_key)
                # Store original confidence for transparency
                insight.evidence.append(f"source_weight={weight:.2f}")
                insight.confidence = insight.confidence * weight

            # Filter out insights that have been effectively silenced
            if insight.confidence >= 0.1:
                weighted.append(insight)

        return weighted

    # ------------------------------------------------------------------
    # Correlator: Place Frequency
    # ------------------------------------------------------------------

    def _place_frequency_insights(self) -> list[Insight]:
        """Discover places the user visits frequently (visit_count > 3)."""
        insights: list[Insight] = []

        try:
            with self.db.get_connection("entities") as conn:
                rows = conn.execute(
                    "SELECT name, visit_count, place_type FROM places WHERE visit_count > 3"
                ).fetchall()
        except Exception:
            return []

        for row in rows:
            name = row["name"]
            count = row["visit_count"]
            place_type = row["place_type"] or "place"

            insight = Insight(
                type="behavioral_pattern",
                summary=f"You visit {name} frequently ({count} visits recorded).",
                confidence=min(0.9, 0.5 + count * 0.05),
                evidence=[f"visit_count={count}", f"place_type={place_type}"],
                category="place",
                entity=name,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Correlator: Contact Gap
    # ------------------------------------------------------------------

    def _contact_gap_insights(self) -> list[Insight]:
        """Find contacts overdue relative to their usual interaction interval.

        Uses the relationships signal profile, which stores per-contact
        interaction_timestamps.  For each contact with sufficient history
        (>= 5 interactions) we compute the average gap and flag when the
        current gap exceeds 1.5x the average and is at least 7 days.

        Filtering:
            - Marketing/automated senders are excluded via the shared
              ``is_marketing_or_noreply`` filter.  Without this filter, the
              relationships profile (170K+ samples) contains many automated
              mailers (newsletters, no-reply accounts, brokerage alerts, etc.)
              that produce ``relationship_intelligence`` insights the user can
              never act on — the same root cause that drove opportunity prediction
              accuracy to 19% before PRs #127–#189 fixed the prediction engine.
            - Inbound-only contacts (outbound_count == 0) are skipped.  If the
              user has never sent a message to someone, there is no established
              bidirectional relationship to maintain.  This mirrors the filter
              added to ``_check_relationship_maintenance`` in PR #204.

        Gap calculation:
            Uses fractional days (``total_seconds() / 86400``) rather than the
            integer ``.days`` attribute.  For contacts who interact daily or
            multiple times per day, ``.days`` truncates sub-24-hour gaps to 0,
            making ``avg_gap = 0`` and causing the threshold condition
            ``days_since > 0 * 1.5`` to always fire for anyone unseen >7 days —
            a false-positive generator.  This is the same integer-truncation bug
            fixed in the prediction engine in PR #166.

        Examples:
            Contact with avg daily email interaction:
                Old: gaps = [0, 0, 0, 0]  →  avg_gap = 0  →  any gap > 7d fires
                New: gaps = [0.9, 1.1, 0.8, …]  →  avg_gap ≈ 1.0  →  fires only at 1.5d
        """
        insights: list[Insight] = []

        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return []

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)
        skipped_marketing = 0
        skipped_inbound_only = 0

        for addr, data in contacts.items():
            last = data.get("last_interaction")
            count = data.get("interaction_count", 0)

            if not last or count < 5:
                continue

            # Skip marketing/automated senders — the shared filter checks for
            # noreply, newsletter, bulk-mail patterns, financial senders, etc.
            # These generate structurally unfulfillable insights because the user
            # cannot reach out to an automated mailer.
            if is_marketing_or_noreply(addr):
                skipped_marketing += 1
                continue

            # Skip inbound-only contacts (user has never messaged them).
            # A one-sided follow from a mailing list or cold-email sender is not
            # an established relationship; there is nothing to "maintain".
            if data.get("outbound_count", 0) == 0:
                skipped_inbound_only += 1
                continue

            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) < 3:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                # Fractional days so same-day contacts don't appear stale
                days_since = (now - last_dt).total_seconds() / 86400
            except (ValueError, TypeError):
                continue

            try:
                dts = sorted([
                    datetime.fromisoformat(t.replace("Z", "+00:00"))
                    for t in timestamps[-10:]
                ])
                # Fractional days: avoids avg_gap=0 for high-frequency contacts
                # (daily emailers, instant-message threads).  The .days attribute
                # truncates to integers, turning a 6-hour gap into 0 days.
                gaps = [
                    (dts[i + 1] - dts[i]).total_seconds() / 86400
                    for i in range(len(dts) - 1)
                ]
                avg_gap = sum(gaps) / len(gaps) if gaps else 30
            except (ValueError, TypeError):
                avg_gap = 30

            if days_since > avg_gap * 1.5 and days_since > 7:
                confidence = min(0.8, 0.4 + (days_since / max(avg_gap, 1) - 1.5) * 0.15)
                insight = Insight(
                    type="relationship_intelligence",
                    summary=(
                        f"It has been {int(days_since)} days since you last contacted {addr} "
                        f"(usual interval ~{int(avg_gap)} days)."
                    ),
                    confidence=confidence,
                    evidence=[
                        f"days_since_last={int(days_since)}",
                        f"avg_gap_days={int(avg_gap)}",
                        f"interaction_count={count}",
                    ],
                    category="contact_gap",
                    entity=addr,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        logger.debug(
            "contact_gap_insights: %d insights generated "
            "(skipped_marketing=%d, skipped_inbound_only=%d)",
            len(insights),
            skipped_marketing,
            skipped_inbound_only,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Email Volume by Day of Week
    # ------------------------------------------------------------------

    def _email_volume_insights(self) -> list[Insight]:
        """Identify the user's busiest email day of the week.

        Queries email.received and email.sent events from the last 30 days,
        buckets by weekday, and surfaces the peak day if it is significantly
        busier than the average.
        """
        insights: list[Insight] = []

        try:
            with self.db.get_connection("events") as conn:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                rows = conn.execute(
                    """SELECT timestamp FROM events
                       WHERE type IN ('email.received', 'email.sent')
                       AND timestamp > ?""",
                    (cutoff,),
                ).fetchall()
        except Exception:
            return []

        if len(rows) < 7:
            return []  # Need at least a week of data

        day_counts: Counter[str] = Counter()
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                day_counts[dt.strftime("%A")] += 1
            except (ValueError, TypeError):
                continue

        if not day_counts:
            return []

        busiest_day, busiest_count = day_counts.most_common(1)[0]
        avg_count = sum(day_counts.values()) / max(len(day_counts), 1)

        # Only surface if the busiest day is at least 1.5x the average
        if busiest_count >= avg_count * 1.5:
            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your busiest email day is {busiest_day} "
                    f"({busiest_count} emails vs ~{int(avg_count)} average)."
                ),
                confidence=min(0.85, 0.5 + (busiest_count / max(avg_count, 1) - 1.0) * 0.2),
                evidence=[
                    f"busiest_day={busiest_day}",
                    f"busiest_count={busiest_count}",
                    f"avg_count={int(avg_count)}",
                ],
                category="email_volume",
                entity=busiest_day,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Correlator: Actionable Alerts
    # ------------------------------------------------------------------

    def _actionable_alert_insights(self) -> list[Insight]:
        """Surface time-sensitive items the user should act on now.

        Generates ``actionable_alert`` insights from two sources:

        **Overdue / soon-due tasks:**
            Queries the ``tasks`` table for open tasks whose ``due_date`` is
            in the past (overdue) or within the next 24 hours (due soon).
            Tasks with ``status`` of ``pending`` or ``in_progress`` are
            considered; completed/cancelled tasks are excluded.  Each
            qualifying task generates one insight so the user is reminded
            even if they are not looking at the task list.

        **Upcoming calendar events requiring preparation:**
            Queries ``calendar.event.created`` events from the last 24 hours
            that start within the next 24 hours.  Events that start in less
            than one hour get a ``high_urgency`` flag and higher confidence.
            This mirrors the logic in the prediction engine's
            ``_check_preparation_needs`` but surfaces the alert directly as
            an insight (immediately visible) rather than a prediction that
            must clear a confidence gate before being notified.

        Dedup strategy:
            Category ``overdue_task`` + entity = task id (changes each time
            the task is touched, so the insight refreshes naturally).
            Category ``upcoming_calendar`` + entity = calendar event id.
            Both use the default 168-hour (7-day) staleness TTL, which means
            a calendar-event insight won't re-fire for 7 days after first
            generation — appropriate because the event itself is a one-off.

        Example insights generated:
            "Task 'Submit Q1 report' is overdue (due 3 days ago)."
            "Upcoming event 'Team standup' starts in 45 minutes — consider preparing."

        Returns:
            list[Insight]: Zero or more ``actionable_alert`` insights.
        """
        insights: list[Insight] = []
        now = datetime.now(timezone.utc)

        # ----------------------------------------------------------------
        # Source 1: Overdue and soon-due tasks
        # ----------------------------------------------------------------
        try:
            with self.db.get_connection("state") as conn:
                rows = conn.execute(
                    """SELECT id, title, due_date, priority
                       FROM tasks
                       WHERE status IN ('pending', 'in_progress')
                         AND due_date IS NOT NULL
                         AND due_date <= ?
                       ORDER BY due_date ASC""",
                    # Include tasks due within the next 24 hours
                    ((now + timedelta(hours=24)).isoformat(),),
                ).fetchall()
        except Exception:
            logger.exception("actionable_alert: failed to query tasks")
            rows = []

        for row in rows:
            try:
                due_dt = datetime.fromisoformat(
                    row["due_date"].replace("Z", "+00:00")
                )
                hours_until_due = (due_dt - now).total_seconds() / 3600

                if hours_until_due < 0:
                    # Overdue
                    days_overdue = abs(hours_until_due) / 24
                    summary = (
                        f"Task '{row['title']}' is overdue "
                        f"(due {int(days_overdue) + 1} day(s) ago)."
                    )
                    # Confidence increases with how overdue the task is,
                    # capped at 0.9 to leave room for very long-overdue tasks.
                    confidence = min(0.9, 0.6 + days_overdue * 0.05)
                    evidence = [
                        "status=overdue",
                        f"days_overdue={int(days_overdue)}",
                        f"priority={row['priority']}",
                    ]
                else:
                    # Due within 24 hours
                    summary = (
                        f"Task '{row['title']}' is due in "
                        f"{int(hours_until_due) + 1} hour(s)."
                    )
                    # Higher urgency for tasks due very soon
                    confidence = min(0.85, 0.5 + (24 - hours_until_due) / 24 * 0.3)
                    evidence = [
                        "status=due_soon",
                        f"hours_until_due={int(hours_until_due)}",
                        f"priority={row['priority']}",
                    ]

                insight = Insight(
                    type="actionable_alert",
                    summary=summary,
                    confidence=confidence,
                    evidence=evidence,
                    category="overdue_task",
                    entity=row["id"],
                    # Shorter staleness for task alerts: re-surface after 6 hours
                    # so persistent overdue tasks stay visible without being annoying.
                    staleness_ttl_hours=6,
                )
                insight.compute_dedup_key()
                insights.append(insight)
            except (ValueError, TypeError):
                # Skip tasks with malformed due_date
                continue

        # ----------------------------------------------------------------
        # Source 2: Calendar events starting within the next 24 hours
        # ----------------------------------------------------------------
        try:
            cutoff_past = (now - timedelta(hours=24)).isoformat()
            cutoff_future = (now + timedelta(hours=24)).isoformat()

            with self.db.get_connection("events") as conn:
                cal_rows = conn.execute(
                    """SELECT payload FROM events
                       WHERE type = 'calendar.event.created'
                         AND timestamp > ?
                         AND timestamp <= ?
                       ORDER BY timestamp DESC
                       LIMIT 200""",
                    (cutoff_past, cutoff_future),
                ).fetchall()
        except Exception:
            logger.exception("actionable_alert: failed to query calendar events")
            cal_rows = []

        seen_cal_ids: set[str] = set()
        for row in cal_rows:
            try:
                payload = json.loads(row["payload"])
                event_id = payload.get("event_id") or payload.get("id") or ""
                title = payload.get("title") or payload.get("summary") or "Untitled event"
                start_str = payload.get("start_time") or payload.get("start") or ""

                if not start_str or event_id in seen_cal_ids:
                    continue
                seen_cal_ids.add(event_id)

                # Parse start time
                start_dt = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                )

                # Only surface events that start in the future
                hours_until_start = (start_dt - now).total_seconds() / 3600
                if hours_until_start < 0 or hours_until_start > 24:
                    continue

                if hours_until_start < 1:
                    urgency_label = "starts very soon"
                    confidence = 0.85
                elif hours_until_start < 4:
                    urgency_label = f"starts in {int(hours_until_start) + 1} hour(s)"
                    confidence = 0.75
                else:
                    urgency_label = f"starts in {int(hours_until_start)} hours"
                    confidence = 0.65

                summary = (
                    f"Upcoming event '{title}' {urgency_label} — consider preparing."
                )
                evidence = [
                    f"hours_until_start={int(hours_until_start)}",
                    f"event_id={event_id}",
                ]
                if hours_until_start < 1:
                    evidence.append("high_urgency=true")

                insight = Insight(
                    type="actionable_alert",
                    summary=summary,
                    confidence=confidence,
                    evidence=evidence,
                    category="upcoming_calendar",
                    entity=event_id or title,
                    # Use 12-hour staleness so the insight refreshes mid-day
                    # for events that were created overnight.
                    staleness_ttl_hours=12,
                )
                insight.compute_dedup_key()
                insights.append(insight)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

        logger.debug(
            "actionable_alert_insights: %d insights generated "
            "(tasks=%d, calendar=%d)",
            len(insights),
            sum(1 for i in insights if i.category == "overdue_task"),
            sum(1 for i in insights if i.category == "upcoming_calendar"),
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Communication Style
    # ------------------------------------------------------------------

    def _communication_style_insights(self) -> list[Insight]:
        """Surface formality observations from the linguistic signal profile.

        Reads the linguistic profile's ``averages.formality`` score and
        translates it into a human-readable insight.
        """
        insights: list[Insight] = []

        profile = self.ums.get_signal_profile("linguistic")
        if not profile:
            return []

        averages = profile["data"].get("averages", {})
        formality = averages.get("formality")

        if formality is None:
            return []

        if formality >= 0.7:
            style_label = "formal"
        elif formality <= 0.3:
            style_label = "casual"
        else:
            style_label = "balanced"

        samples_count = profile.get("samples_count", 0)
        if samples_count < 3:
            return []  # Not enough data for a meaningful observation

        insight = Insight(
            type="communication_style",
            summary=(
                f"Your overall writing style is {style_label} "
                f"(formality score {formality:.2f}, based on {samples_count} messages)."
            ),
            confidence=min(0.85, 0.4 + samples_count * 0.02),
            evidence=[
                f"formality={formality:.2f}",
                f"samples_count={samples_count}",
                f"style={style_label}",
            ],
            category="communication_style",
            entity=style_label,
        )
        insight.compute_dedup_key()
        insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, insights: list[Insight]) -> list[Insight]:
        """Remove insights whose dedup_key already exists within staleness TTL.

        An insight is considered stale (and therefore skipped) if the insights
        table already contains a row with the same dedup_key whose created_at
        is within ``staleness_ttl_hours`` of now.
        """
        fresh: list[Insight] = []
        now = datetime.now(timezone.utc)

        for insight in insights:
            if not insight.dedup_key:
                insight.compute_dedup_key()

            try:
                with self.db.get_connection("user_model") as conn:
                    row = conn.execute(
                        "SELECT created_at FROM insights WHERE dedup_key = ? ORDER BY created_at DESC LIMIT 1",
                        (insight.dedup_key,),
                    ).fetchone()
            except Exception:
                # If the table doesn't exist yet or any DB error, treat as fresh
                fresh.append(insight)
                continue

            if row is None:
                fresh.append(insight)
                continue

            try:
                created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                age_hours = (now - created_at).total_seconds() / 3600
                if age_hours >= insight.staleness_ttl_hours:
                    fresh.append(insight)
                # else: skip — still within staleness window
            except (ValueError, TypeError):
                fresh.append(insight)

        return fresh

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_insight(self, insight: Insight):
        """INSERT a new insight row into the user_model insights table."""
        with self.db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO insights
                   (id, type, summary, confidence, evidence, category,
                    entity, staleness_ttl_hours, dedup_key, feedback, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    insight.id,
                    insight.type,
                    insight.summary,
                    insight.confidence,
                    json.dumps(insight.evidence),
                    insight.category,
                    insight.entity,
                    insight.staleness_ttl_hours,
                    insight.dedup_key,
                    insight.feedback,
                    insight.created_at,
                ),
            )
