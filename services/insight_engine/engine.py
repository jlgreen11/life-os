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
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


class InsightEngine:
    """Cross-correlates signal profiles to produce human-readable insights."""

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        self.db = db
        self.ums = ums

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

        # Ensure every insight has a dedup key
        for insight in raw:
            if not insight.dedup_key:
                insight.compute_dedup_key()

        # Remove insights that are still within their staleness window
        fresh = self._deduplicate(raw)

        # Persist the survivors
        for insight in fresh:
            self._store_insight(insight)

        return fresh

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
        """
        insights: list[Insight] = []

        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return []

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            last = data.get("last_interaction")
            count = data.get("interaction_count", 0)

            if not last or count < 5:
                continue

            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) < 3:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            try:
                dts = sorted([
                    datetime.fromisoformat(t.replace("Z", "+00:00"))
                    for t in timestamps[-10:]
                ])
                gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 30
            except (ValueError, TypeError):
                avg_gap = 30

            if days_since > avg_gap * 1.5 and days_since > 7:
                confidence = min(0.8, 0.4 + (days_since / max(avg_gap, 1) - 1.5) * 0.15)
                insight = Insight(
                    type="relationship_intelligence",
                    summary=(
                        f"It has been {days_since} days since you last contacted {addr} "
                        f"(usual interval ~{int(avg_gap)} days)."
                    ),
                    confidence=confidence,
                    evidence=[
                        f"days_since_last={days_since}",
                        f"avg_gap_days={int(avg_gap)}",
                        f"interaction_count={count}",
                    ],
                    category="contact_gap",
                    entity=addr,
                )
                insight.compute_dedup_key()
                insights.append(insight)

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
