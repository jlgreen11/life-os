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
    temporal_pattern            -- Chronotype and productive-hour insights from temporal profile
    mood_trend                  -- Mood trajectory insights derived from mood_history
    spending_pattern            -- Financial behavioral patterns from transaction history
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

        try:
            raw.extend(self._temporal_pattern_insights())
        except Exception:
            logger.exception("temporal_pattern correlator failed")

        try:
            raw.extend(self._mood_trend_insights())
        except Exception:
            logger.exception("mood_trend correlator failed")

        try:
            raw.extend(self._spending_pattern_insights())
        except Exception:
            logger.exception("spending_pattern correlator failed")

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
            # Temporal and mood insights derive from all communication signals,
            # so they are weighted against the broadest applicable source key.
            "chronotype": "email.work",
            "peak_hour": "email.work",
            "busiest_day": "email.work",
            "mood_trajectory": "messaging.direct",
            # Spending pattern insights derive from finance connector data.
            "top_spending_category": "finance.transactions",
            "spending_increase": "finance.transactions",
            "spending_decrease": "finance.transactions",
            "recurring_subscription": "finance.transactions",
        }

        weighted: list[Insight] = []
        for insight in insights:
            source_key = category_to_source.get(insight.category)
            if source_key:
                weight = self.swm.get_effective_weight(source_key)
                # Store original confidence for transparency
                insight.evidence.append(f"source_weight={weight:.2f}")
                insight.confidence = insight.confidence * weight

                # Record that this source produced an insight so the interaction
                # counter advances toward the MIN_INTERACTIONS gate.  Drift
                # (record_engagement / record_dismissal) is only applied after
                # the gate is crossed, preventing drift from wild-swinging on
                # sparse data.  Without this call the counter stays at 0
                # forever and AI drift never activates.
                try:
                    self.swm.record_interaction(source_key)
                except Exception:
                    pass  # Non-critical; never break insight delivery for this

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
    # Correlator: Temporal Patterns
    # ------------------------------------------------------------------

    def _temporal_pattern_insights(self) -> list[Insight]:
        """Surface chronotype and productive-hour insights from the temporal profile.

        Reads the ``temporal`` signal profile (built by ``TemporalExtractor``) and
        translates raw activity-by-hour and activity-by-day histograms into two
        human-readable insight types:

        **Chronotype (early bird / night owl / mixed):**
            Compares activity density in the morning window (5–11 h) versus the
            evening window (18–23 h).  At least ``MIN_SAMPLES`` data points must
            exist before the correlator fires.  A morning-biased user whose morning
            activity exceeds evening by ``CHRONOTYPE_RATIO`` or more is labelled an
            "early bird"; the inverse yields "night owl"; otherwise "mixed".

        **Peak productive hour:**
            Identifies the single hour with the most recorded activity events.
            Only hours in the productive day (6–22 h) are considered.  At least
            ``MIN_PEAK_ACTIVITY`` events must be recorded for the peak hour before
            an insight is generated.

        **Busiest day of the week:**
            Finds the calendar day with the highest activity count.  Requires at
            least ``MIN_DAY_SAMPLES`` total day-level samples to avoid noise from
            sparse data.

        Dedup strategy:
            Each insight sub-type uses a stable ``entity`` derived from the
            detected pattern label (e.g. "early_bird", "peak_hour_9", "tuesday").
            This means the insight is resurfaced only when the label itself changes
            (i.e. the pattern shifts), not just because a new sample was added.
            Staleness TTL is 168 hours (7 days) — appropriate for slow-moving
            behavioral traits.

        Returns:
            list[Insight]: Zero, one, two, or three insights depending on how much
            temporal data is available and how clear the patterns are.

        Example insights generated::

            "You tend to be most active in the morning, suggesting you're an
            early bird (60% of activity between 05:00–11:00)."

            "Your most productive hour is 9 AM — consider scheduling deep-work
            blocks then."

            "Tuesday is your busiest day (32% more activity than your weekly
            average)."
        """
        # Minimum data requirements to avoid noise from sparse profiles
        MIN_SAMPLES = 50          # total activity samples required for chronotype
        MIN_PEAK_ACTIVITY = 10   # events in the busiest hour for a peak-hour insight
        MIN_DAY_SAMPLES = 30     # total day-level samples for busiest-day insight
        CHRONOTYPE_RATIO = 1.5   # morning vs evening activity ratio threshold

        profile = self.ums.get_signal_profile("temporal")
        if not profile:
            return []

        data = profile.get("data", {})
        total_samples = profile.get("samples_count", 0)

        activity_by_hour: dict[str, int] = data.get("activity_by_hour", {})
        activity_by_day: dict[str, int] = data.get("activity_by_day", {})

        if not activity_by_hour or total_samples < MIN_SAMPLES:
            return []

        insights: list[Insight] = []

        # ----------------------------------------------------------------
        # Sub-insight 1: Chronotype (early bird / night owl / mixed)
        # ----------------------------------------------------------------
        # Morning window: 05:00–10:59; evening window: 18:00–22:59
        morning_hours = [str(h) for h in range(5, 11)]
        evening_hours = [str(h) for h in range(18, 23)]

        morning_count = sum(activity_by_hour.get(h, 0) for h in morning_hours)
        evening_count = sum(activity_by_hour.get(h, 0) for h in evening_hours)
        total_windowed = morning_count + evening_count

        if total_windowed >= 10:  # enough window samples to compare
            if morning_count >= evening_count * CHRONOTYPE_RATIO:
                chronotype = "early_bird"
                pct = int(100 * morning_count / total_windowed)
                description = "early bird"
                window_label = "05:00–11:00"
            elif evening_count >= morning_count * CHRONOTYPE_RATIO:
                chronotype = "night_owl"
                pct = int(100 * evening_count / total_windowed)
                description = "night owl"
                window_label = "18:00–23:00"
            else:
                chronotype = "mixed"
                pct = int(100 * morning_count / total_windowed)
                description = "mixed (morning and evening activity are balanced)"
                window_label = "morning and evening windows"

            if chronotype in ("early_bird", "night_owl"):
                summary = (
                    f"You tend to be most active in the {'morning' if chronotype == 'early_bird' else 'evening'}, "
                    f"suggesting you're an {description} ({pct}% of windowed activity between {window_label})."
                )
            else:
                summary = (
                    f"Your activity is fairly balanced between morning and evening ({description})."
                )

            insight = Insight(
                type="temporal_pattern",
                summary=summary,
                # Confidence ramps with sample count; reaches 0.85 at 500+ samples
                confidence=min(0.85, 0.4 + total_samples / 1000),
                evidence=[
                    f"chronotype={chronotype}",
                    f"morning_count={morning_count}",
                    f"evening_count={evening_count}",
                    f"total_samples={total_samples}",
                ],
                category="chronotype",
                entity=chronotype,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 2: Peak productive hour
        # ----------------------------------------------------------------
        # Only consider daytime hours (06:00–21:59) to avoid fringe outliers
        productive_hours = {h: activity_by_hour.get(str(h), 0) for h in range(6, 22)}
        if productive_hours:
            peak_hour = max(productive_hours, key=lambda h: productive_hours[h])
            peak_count = productive_hours[peak_hour]

            if peak_count >= MIN_PEAK_ACTIVITY:
                # Format hour as human-readable "9 AM", "2 PM", etc.
                hour_label = datetime.now().replace(hour=peak_hour).strftime("%-I %p")
                summary = (
                    f"Your most active hour is {hour_label} — consider scheduling "
                    f"important work or focus blocks during this window "
                    f"({peak_count} activity events recorded)."
                )
                insight = Insight(
                    type="temporal_pattern",
                    summary=summary,
                    confidence=min(0.80, 0.35 + peak_count / 100),
                    evidence=[
                        f"peak_hour={peak_hour}",
                        f"peak_count={peak_count}",
                        f"total_samples={total_samples}",
                    ],
                    category="peak_hour",
                    # Entity encodes the specific hour so the insight refreshes
                    # only if the peak hour shifts to a different time slot.
                    entity=f"peak_hour_{peak_hour}",
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 3: Busiest day of the week
        # ----------------------------------------------------------------
        total_day_samples = sum(activity_by_day.values())
        if activity_by_day and total_day_samples >= MIN_DAY_SAMPLES:
            busiest_day = max(activity_by_day, key=lambda d: activity_by_day[d])
            busiest_count = activity_by_day[busiest_day]
            avg_count = total_day_samples / max(len(activity_by_day), 1)

            # Only surface if the busiest day is meaningfully above average
            if busiest_count >= avg_count * 1.3:
                pct_above = int(100 * (busiest_count / avg_count - 1))
                summary = (
                    f"{busiest_day.capitalize()} is your busiest day of the week "
                    f"({pct_above}% more activity than your daily average)."
                )
                insight = Insight(
                    type="temporal_pattern",
                    summary=summary,
                    confidence=min(0.80, 0.40 + total_day_samples / 500),
                    evidence=[
                        f"busiest_day={busiest_day}",
                        f"busiest_count={busiest_count}",
                        f"avg_count={avg_count:.1f}",
                        f"pct_above_avg={pct_above}",
                    ],
                    category="busiest_day",
                    entity=busiest_day,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        logger.debug(
            "temporal_pattern_insights: %d insights generated "
            "(samples=%d, hours=%d, days=%d)",
            len(insights),
            total_samples,
            len(activity_by_hour),
            len(activity_by_day),
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Mood Trend
    # ------------------------------------------------------------------

    def _mood_trend_insights(self) -> list[Insight]:
        """Surface mood trajectory insights from the mood_history time-series.

        Reads the ``mood_history`` table and computes a composite mood score for
        two windows (recent vs. baseline) to determine whether the user's overall
        mood is improving, declining, or stable.  This mirrors the logic in
        ``MoodExtractor.detect_mood_trend()`` but converts the trend into a
        persistent, deduplicated insight rather than a transient signal.

        **Composite score formula:**
            ``composite = energy_level + emotional_valence - stress_level``

        This combines the three most meaningful mood dimensions into a single
        scalar on approximately the range −1 to 2.  A higher score is better.

        **Window design:**
            - *Recent* window: most recent 3 mood_history rows
            - *Baseline* window: rows 4–12 (next-most-recent 9 rows)

        At least 6 rows are required to compute a meaningful comparison.

        **Thresholds:**
            - ``> 0.15`` delta → "improving"  (positive trajectory)
            - ``< -0.15`` delta → "declining" (negative trajectory)
            - Otherwise → "stable"

        Only "improving" and "declining" trends generate insights — stable mood
        is the expected baseline and not worth surfacing.  The staleness TTL is
        set to 48 hours so that mood trends refresh twice a day, keeping the
        insight relevant as mood fluctuates.

        Dedup strategy:
            Entity is the trend label ("improving" or "declining"), so the insight
            stays fresh while the trend persists and is replaced if the label changes.

        Returns:
            list[Insight]: Zero or one insight (only for non-stable trends with
            sufficient data).

        Example insights generated::

            "Your mood has been improving over the past few days — energy and
            positivity are trending upward."

            "Your mood appears to be declining recently — stress levels are elevated
            compared to your recent baseline."
        """
        # Minimum rows needed to compute a meaningful recent-vs-baseline comparison
        MIN_ROWS = 6
        TREND_THRESHOLD = 0.15  # minimum composite-score delta to call a trend

        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT energy_level, stress_level, emotional_valence
                       FROM mood_history
                       ORDER BY timestamp DESC
                       LIMIT 12""",
                ).fetchall()
        except Exception:
            logger.exception("mood_trend_insights: failed to query mood_history")
            return []

        if len(rows) < MIN_ROWS:
            return []

        def _composite(subset: list) -> float:
            """Compute average composite mood score (energy + valence − stress)."""
            total = 0.0
            for row in subset:
                energy = row["energy_level"] if row["energy_level"] is not None else 0.5
                stress = row["stress_level"] if row["stress_level"] is not None else 0.5
                valence = row["emotional_valence"] if row["emotional_valence"] is not None else 0.5
                total += energy + valence - stress
            return total / len(subset)

        # Recent: most recent 3 rows; baseline: next 9 rows
        recent_rows = rows[:3]
        baseline_rows = rows[3:]

        if not recent_rows or not baseline_rows:
            return []

        recent_score = _composite(recent_rows)
        baseline_score = _composite(baseline_rows)
        delta = recent_score - baseline_score

        if delta > TREND_THRESHOLD:
            trend = "improving"
            summary = (
                "Your mood has been improving over the past few days — "
                "energy and positivity appear to be trending upward."
            )
            # Higher confidence for larger positive swings
            confidence = min(0.80, 0.50 + delta * 0.5)
        elif delta < -TREND_THRESHOLD:
            trend = "declining"
            summary = (
                "Your mood appears to be declining recently — "
                "stress levels are elevated compared to your recent baseline. "
                "Consider whether workload or sleep quality has changed."
            )
            confidence = min(0.80, 0.50 + abs(delta) * 0.5)
        else:
            # Stable mood is the expected baseline; not worth surfacing as an insight.
            logger.debug(
                "mood_trend_insights: trend=stable (delta=%.3f), skipping insight",
                delta,
            )
            return []

        insight = Insight(
            type="mood_trend",
            summary=summary,
            confidence=confidence,
            evidence=[
                f"trend={trend}",
                f"recent_composite={recent_score:.3f}",
                f"baseline_composite={baseline_score:.3f}",
                f"delta={delta:.3f}",
                f"rows_analyzed={len(rows)}",
            ],
            category="mood_trajectory",
            entity=trend,
            # Refresh every 48 hours so mood trends stay current without flooding
            staleness_ttl_hours=48,
        )
        insight.compute_dedup_key()

        logger.debug(
            "mood_trend_insights: trend=%s delta=%.3f recent=%.3f baseline=%.3f",
            trend, delta, recent_score, baseline_score,
        )
        return [insight]

    # ------------------------------------------------------------------
    # Correlator: Spending Patterns
    # ------------------------------------------------------------------

    def _spending_pattern_insights(self) -> list[Insight]:
        """Surface financial behavioral patterns from transaction history.

        Reads ``finance.transaction.new`` events and produces up to three
        categories of insight:

        **1. Top spending category (behavioral_pattern)**
            The single category that consumed the most of the user's budget
            over the last 30 days.  Only fires when the top category accounts
            for ≥25% of total spend AND at least $100 absolute — below those
            thresholds the signal is noise.  Confidence scales with the
            category's share of total spend.

            Example: "FOOD_AND_DRINK is your largest spending category this
            month at $430 (34% of total)."

        **2. Month-over-month category change (spending_pattern)**
            Compares each category's 30-day total against the prior 30 days.
            Surfaces the single category with the largest *absolute* dollar
            change when the change exceeds $100 AND 30% of the prior-period
            amount.  Both a notable increase and a notable decrease generate
            an insight.  This avoids flooding the user with incremental noise
            while surfacing genuinely significant budget shifts.

            Example: "Your TRAVEL spending increased by $280 this month
            ($130 → $410, +215%)."

        **3. Recurring subscription detection (behavioral_pattern)**
            Groups transactions by (merchant, rounded-amount) bucket and
            flags any combination that appears in ≥2 distinct calendar months
            within the last 90 days.  Buckets amounts to the nearest $1 so
            small rounding differences don't break the match.  Only fires for
            amounts ≥$5 to ignore micro-transactions.

            Example: "Recurring subscription detected: 'Netflix' charges
            ~$15 every month (3 occurrences in the last 90 days)."

        **Data requirements:**
            - Methods 1 & 2 require ≥5 transactions in the last 30 days.
            - Method 3 requires ≥90 days of transaction history.

        **Staleness TTL:**
            - Top-category and MoM-change insights expire after 7 days (168h)
              so they refresh on the next monthly billing cycle.
            - Subscription insights expire after 30 days (720h) — they are
              expected to be stable and re-surfacing them weekly would be
              noisy.

        **Dedup strategy:**
            Category is the spending insight sub-type (``top_spending_category``,
            ``spending_increase``, ``spending_decrease``, ``recurring_subscription``).
            Entity is the merchant/category name so each distinct merchant or
            category generates its own dedup key and can refresh independently.

        Returns:
            list[Insight]: Zero or more spending-related insights.
        """
        insights: list[Insight] = []
        now = datetime.now(timezone.utc)

        # ----------------------------------------------------------------
        # Load 30-day and 60-day transaction windows
        # ----------------------------------------------------------------
        cutoff_30 = (now - timedelta(days=30)).isoformat()
        cutoff_60 = (now - timedelta(days=60)).isoformat()
        cutoff_90 = (now - timedelta(days=90)).isoformat()

        try:
            with self.db.get_connection("events") as conn:
                recent_rows = conn.execute(
                    """SELECT payload, timestamp FROM events
                       WHERE type = 'finance.transaction.new'
                         AND timestamp > ?
                       ORDER BY timestamp DESC""",
                    (cutoff_30,),
                ).fetchall()

                prior_rows = conn.execute(
                    """SELECT payload, timestamp FROM events
                       WHERE type = 'finance.transaction.new'
                         AND timestamp > ?
                         AND timestamp <= ?
                       ORDER BY timestamp DESC""",
                    (cutoff_60, cutoff_30),
                ).fetchall()

                subscription_rows = conn.execute(
                    """SELECT payload, timestamp FROM events
                       WHERE type = 'finance.transaction.new'
                         AND timestamp > ?
                       ORDER BY timestamp DESC""",
                    (cutoff_90,),
                ).fetchall()
        except Exception:
            logger.exception("spending_pattern_insights: failed to query transactions")
            return []

        # ----------------------------------------------------------------
        # Parse transaction payloads into structured dicts
        # ----------------------------------------------------------------
        def _parse_txns(rows: list) -> list[dict]:
            """Parse raw DB rows into (amount, category, merchant, timestamp) dicts.

            Skips rows with malformed JSON or missing/zero amounts since they
            provide no signal.  Amount is always taken as absolute value so that
            Plaid's sign convention (negative = outflow) doesn't affect aggregation.
            """
            result = []
            for row in rows:
                try:
                    payload = json.loads(row["payload"])
                    amount = abs(payload.get("amount", 0))
                    if amount <= 0:
                        # Income or zero-value event — not a spending transaction
                        continue
                    result.append({
                        "amount": amount,
                        "category": (payload.get("category") or "uncategorized").strip(),
                        "merchant": (
                            payload.get("merchant") or payload.get("name") or "unknown"
                        ).strip(),
                        "timestamp": row["timestamp"],
                    })
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
            return result

        recent_txns = _parse_txns(recent_rows)
        prior_txns = _parse_txns(prior_rows)
        sub_txns = _parse_txns(subscription_rows)

        # ----------------------------------------------------------------
        # Insight 1: Top spending category (last 30 days)
        # Requires ≥5 transactions so averages are meaningful.
        # ----------------------------------------------------------------
        if len(recent_txns) >= 5:
            # Aggregate spend per category
            by_category: dict[str, float] = {}
            for txn in recent_txns:
                cat = txn["category"]
                by_category[cat] = by_category.get(cat, 0) + txn["amount"]

            total = sum(by_category.values())

            if total > 0:
                # Find the dominant category
                top_cat, top_amt = max(by_category.items(), key=lambda x: x[1])
                top_pct = top_amt / total

                # Only surface when the category is both a large fraction AND
                # a meaningful absolute amount — avoids noise for sparse data.
                if top_pct >= 0.25 and top_amt >= 100:
                    confidence = min(0.80, 0.50 + top_pct * 0.60)
                    insight = Insight(
                        type="spending_pattern",
                        summary=(
                            f"{top_cat.replace('_', ' ').title()} is your largest "
                            f"spending category this month at ${top_amt:.0f} "
                            f"({top_pct * 100:.0f}% of ${total:.0f} total)."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"top_category={top_cat}",
                            f"top_amount=${top_amt:.2f}",
                            f"top_pct={top_pct * 100:.1f}%",
                            f"total_spend=${total:.2f}",
                            f"transaction_count={len(recent_txns)}",
                            f"categories_count={len(by_category)}",
                        ],
                        category="top_spending_category",
                        entity=top_cat,
                        # Refresh weekly — spending categories shift on billing cycles
                        staleness_ttl_hours=168,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

            # ----------------------------------------------------------------
            # Insight 2: Month-over-month change (largest absolute shift)
            # Requires data in both windows to compare.
            # ----------------------------------------------------------------
            if len(prior_txns) >= 5:
                prior_by_category: dict[str, float] = {}
                for txn in prior_txns:
                    cat = txn["category"]
                    prior_by_category[cat] = prior_by_category.get(cat, 0) + txn["amount"]

                # Find the category with the biggest absolute dollar change
                all_cats = set(by_category.keys()) | set(prior_by_category.keys())
                best_change_cat: Optional[str] = None
                best_change_abs = 0.0
                best_change_delta = 0.0
                best_change_prior = 0.0
                best_change_recent = 0.0

                for cat in all_cats:
                    recent_amt = by_category.get(cat, 0)
                    prior_amt = prior_by_category.get(cat, 0)
                    delta = recent_amt - prior_amt
                    abs_delta = abs(delta)

                    # Require both absolute change ≥$100 and relative change ≥30%
                    # to avoid surfacing noise from low-spend categories.
                    if prior_amt > 0:
                        pct_change = abs_delta / prior_amt
                    else:
                        # New category with no prior history — only surface if
                        # it's a significant new expense (≥$100).
                        pct_change = 1.0 if recent_amt >= 100 else 0.0

                    if abs_delta >= 100 and pct_change >= 0.30:
                        if abs_delta > best_change_abs:
                            best_change_abs = abs_delta
                            best_change_cat = cat
                            best_change_delta = delta
                            best_change_prior = prior_amt
                            best_change_recent = recent_amt

                if best_change_cat is not None:
                    sign = "increased" if best_change_delta > 0 else "decreased"
                    pct_str = ""
                    if best_change_prior > 0:
                        pct = abs(best_change_delta) / best_change_prior * 100
                        pct_str = f" (+{pct:.0f}%)" if best_change_delta > 0 else f" (-{pct:.0f}%)"
                    cat_display = best_change_cat.replace("_", " ").title()
                    cat_key = "spending_increase" if best_change_delta > 0 else "spending_decrease"
                    # Confidence scales with the size of the change relative to prior spend
                    if best_change_prior > 0:
                        rel_change = min(1.0, abs(best_change_delta) / best_change_prior)
                    else:
                        rel_change = 1.0
                    confidence = min(0.80, 0.50 + rel_change * 0.30)

                    insight = Insight(
                        type="spending_pattern",
                        summary=(
                            f"Your {cat_display} spending {sign} by "
                            f"${abs(best_change_delta):.0f} this month "
                            f"(${best_change_prior:.0f} → ${best_change_recent:.0f}"
                            f"{pct_str})."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"category={best_change_cat}",
                            f"prior_30d=${best_change_prior:.2f}",
                            f"recent_30d=${best_change_recent:.2f}",
                            f"delta=${best_change_delta:+.2f}",
                            f"abs_delta=${best_change_abs:.2f}",
                        ],
                        category=cat_key,
                        entity=best_change_cat,
                        staleness_ttl_hours=168,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

        # ----------------------------------------------------------------
        # Insight 3: Recurring subscription detection (90-day window)
        # ----------------------------------------------------------------
        if len(sub_txns) >= 3:
            # Group by (merchant, rounded-amount) to find recurring charges.
            # Rounding to nearest $1 handles minor Plaid rounding differences.
            # month_key = YYYY-MM so each calendar month contributes one hit.
            from collections import defaultdict
            bucket_months: dict[tuple[str, int], set[str]] = defaultdict(set)

            for txn in sub_txns:
                if txn["amount"] < 5:
                    # Ignore micro-transactions (cents-level fees, etc.)
                    continue
                merchant = txn["merchant"]
                rounded_amt = round(txn["amount"])
                try:
                    ts = datetime.fromisoformat(
                        txn["timestamp"].replace("Z", "+00:00")
                    )
                    month_key = ts.strftime("%Y-%m")
                except (ValueError, AttributeError):
                    continue
                bucket_months[(merchant, rounded_amt)].add(month_key)

            # Surface any (merchant, amount) pair that appeared in ≥2 distinct months
            for (merchant, rounded_amt), months in bucket_months.items():
                if len(months) < 2:
                    continue
                occurrence_count = len(months)
                confidence = min(0.80, 0.50 + occurrence_count * 0.10)
                insight = Insight(
                    type="spending_pattern",
                    summary=(
                        f"Recurring subscription detected: '{merchant}' charges "
                        f"~${rounded_amt} every month "
                        f"({occurrence_count} occurrences in the last 90 days)."
                    ),
                    confidence=confidence,
                    evidence=[
                        f"merchant={merchant}",
                        f"rounded_amount=${rounded_amt}",
                        f"months_seen={occurrence_count}",
                        f"calendar_months={sorted(months)}",
                    ],
                    category="recurring_subscription",
                    entity=f"{merchant}_{rounded_amt}",
                    # Refresh monthly — subscriptions are stable over time
                    staleness_ttl_hours=720,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        logger.debug(
            "spending_pattern_insights: %d insights generated "
            "(recent_txns=%d, prior_txns=%d, sub_txns=%d)",
            len(insights),
            len(recent_txns),
            len(prior_txns),
            len(sub_txns),
        )
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
