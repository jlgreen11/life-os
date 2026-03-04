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
    relationship_intelligence   -- Social-graph discoveries (contact_gap, reciprocity_imbalance, fast_responder)
    communication_style         -- Writing-style observations from linguistic profile
    temporal_pattern            -- Chronotype and productive-hour insights from temporal profile
    mood_trend                  -- Mood trajectory insights derived from mood_history
    spending_pattern            -- Financial behavioral patterns from transaction history
    decision_pattern            -- Decision-making style from speed, delegation, and fatigue signals
    topic_interest              -- Dominant interests and trending topics from topic signal profile
    cadence_response            -- Reply-latency baseline, priority contacts, and peak hours from cadence profile
    routine_pattern             -- Recurring behavioral sequences detected in procedural memory (Layer 3)
    spatial_location            -- Location-behavioral patterns from spatial signal profile (visit frequency, work/personal split)
    workflow_pattern            -- Goal-driven multi-step processes detected by WorkflowDetector (email response, task completion, meeting prep)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from services.insight_engine.models import Insight
from services.insight_engine.source_weights import SourceWeightManager
from services.signal_extractor.marketing_filter import is_marketing_or_noreply
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _routine_trigger_label(trigger: str) -> str:
    """Convert a machine trigger key to a human-readable phrase.

    Args:
        trigger: Trigger key stored by RoutineDetector, e.g. ``"morning"``,
                 ``"arrive_home"``, ``"after_meeting"``.

    Returns:
        A readable phrase suitable for insertion into an insight summary,
        e.g. ``"morning routine"`` or ``"arrival routine at home"``.

    Examples::

        >>> _routine_trigger_label("morning")
        'morning routine'
        >>> _routine_trigger_label("arrive_home")
        'arrival routine at home'
        >>> _routine_trigger_label("after_meeting")
        'post-meeting routine'
        >>> _routine_trigger_label("custom_label")
        'custom_label routine'
    """
    known: dict[str, str] = {
        "morning": "morning routine",
        "midday": "midday routine",
        "afternoon": "afternoon routine",
        "evening": "evening routine",
        "night": "night routine",
    }
    if trigger in known:
        return known[trigger]

    # "arrive_<location>" → "arrival routine at <location>"
    if trigger.startswith("arrive_"):
        location = trigger[len("arrive_"):].replace("_", " ")
        return f"arrival routine at {location}"

    # "after_<event>" → "post-<event> routine"
    if trigger.startswith("after_"):
        event = trigger[len("after_"):].replace("_", " ")
        return f"post-{event} routine"

    # Fallback: use the trigger string verbatim
    return f"{trigger.replace('_', ' ')} routine"


class InsightEngine:
    """Cross-correlates signal profiles to produce human-readable insights."""

    def __init__(self, db: DatabaseManager, ums: UserModelStore,
                 source_weight_manager: Optional[SourceWeightManager] = None,
                 timezone: str = "America/Los_Angeles",
                 cache_ttl_seconds: float = 300.0):
        self.db = db
        self.ums = ums
        self.swm = source_weight_manager
        self._tz = ZoneInfo(timezone)
        self._insight_cache_ttl: float = cache_ttl_seconds
        self._last_insight_run: float = 0.0

        # Diagnostic counters for monitoring insight pipeline health.
        # Updated at the end of each generate_insights() cycle and
        # queryable via get_diagnostics().
        self._total_runs: int = 0
        self._last_run_at: str | None = None
        self._last_correlator_stats: dict[str, int | str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_data_sufficiency_report(self) -> dict:
        """Report which correlators have sufficient data to produce insights.

        Checks each signal profile and data source that the correlators depend
        on and returns a structured dict mapping correlator name to its data
        readiness status.

        Returns:
            A dict keyed by correlator method name. Each value contains:
            - ``profile`` or ``source``: the data source checked
            - ``status``: one of ``'ready'``, ``'partial'``, or ``'no_data'``
            - ``samples`` or ``count``: current data point count
            - ``min_required``: minimum samples needed for ``'ready'`` status
              (profile-based correlators only)
        """
        report: dict[str, dict] = {}

        # Profile-based correlators: each needs a signal profile with enough
        # samples to produce meaningful insights.
        profiles_to_check = [
            ("relationships", "_contact_gap_insights", 10),
            ("linguistic", "_communication_style_insights", 10),
            ("linguistic_inbound", "_inbound_style_insights", 5),
            ("cadence", "_cadence_response_insights", 10),
            ("temporal", "_temporal_pattern_insights", 7),
            ("mood_signals", "_mood_trend_insights", 5),
            ("topics", "_topic_interest_insights", 10),
            ("spatial", "_spatial_insights", 10),
            ("decision", "_decision_pattern_insights", 20),
        ]
        for profile_type, correlator_name, min_samples in profiles_to_check:
            try:
                profile = self.ums.get_signal_profile(profile_type)
                samples = profile["samples_count"] if profile else 0
            except Exception:
                samples = -1  # DB or deserialization error
            if samples < 0:
                status = "error"
            elif samples >= min_samples:
                status = "ready"
            elif samples > 0:
                status = "partial"
            else:
                status = "no_data"
            report[correlator_name] = {
                "profile": profile_type,
                "status": status,
                "samples": samples,
                "min_required": min_samples,
            }

        # Episode-based correlator (_place_frequency_insights)
        try:
            with self.db.get_connection("user_model") as conn:
                ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except Exception:
            ep_count = -1
        report["_place_frequency_insights"] = {
            "source": "episodes",
            "status": "ready" if ep_count >= 7 else ("error" if ep_count < 0 else "no_data"),
            "count": ep_count,
            "min_required": 7,
        }

        # Routine-based correlator (_routine_insights)
        try:
            with self.db.get_connection("user_model") as conn:
                routine_count = conn.execute("SELECT COUNT(*) FROM routines").fetchone()[0]
        except Exception:
            routine_count = -1
        report["_routine_insights"] = {
            "source": "routines",
            "status": "ready" if routine_count > 0 else ("error" if routine_count < 0 else "no_data"),
            "count": routine_count,
            "min_required": 1,
        }

        # Events-DB correlators: need minimum event counts
        events_correlators = [
            ("_email_volume_insights", "email.received", 7),
            ("_email_peak_hour_insights", "email.received", 50),
            ("_meeting_density_insights", "calendar.event.created", 10),
        ]
        for correlator_name, event_type, min_count in events_correlators:
            try:
                with self.db.get_connection("events") as conn:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM events WHERE type = ?",
                        (event_type,),
                    ).fetchone()[0]
            except Exception:
                count = -1
            report[correlator_name] = {
                "source": f"events({event_type})",
                "status": "ready" if count >= min_count else ("error" if count < 0 else ("partial" if count > 0 else "no_data")),
                "count": count,
                "min_required": min_count,
            }

        return report

    async def generate_insights(self) -> list[Insight]:
        """Main loop: run all correlators, deduplicate, store, return new insights.

        Skips correlator execution if the last successful run was within
        ``_insight_cache_ttl`` seconds.  The caller (e.g. the ``/api/insights/summary``
        route) still reads stored insights from the database, so returning an
        empty list here simply means "no new insights computed this call".
        """
        if self._insight_cache_ttl > 0 and (time.monotonic() - self._last_insight_run) < self._insight_cache_ttl:
            logger.debug("Skipping correlator run — last run %.1fs ago (TTL %.0fs)",
                         time.monotonic() - self._last_insight_run, self._insight_cache_ttl)
            return []

        raw: list[Insight] = []
        correlator_stats: dict[str, int | str] = {}

        # Each correlator handles its own errors gracefully and returns
        # an empty list when there is insufficient data.  Per-correlator
        # result counts are tracked for diagnostics.
        correlators = [
            ("place_frequency", self._place_frequency_insights),
            ("contact_gap", self._contact_gap_insights),
            ("relationship_intelligence", self._relationship_intelligence_insights),
            ("email_volume", self._email_volume_insights),
            ("email_peak_hour", self._email_peak_hour_insights),
            ("meeting_density", self._meeting_density_insights),
            ("communication_style", self._communication_style_insights),
            ("inbound_style", self._inbound_style_insights),
            ("actionable_alert", self._actionable_alert_insights),
            ("temporal_pattern", self._temporal_pattern_insights),
            ("mood_trend", self._mood_trend_insights),
            ("spending_pattern", self._spending_pattern_insights),
            ("decision_pattern", self._decision_pattern_insights),
            ("topic_interest", self._topic_interest_insights),
            ("cadence_response", self._cadence_response_insights),
            ("routine", self._routine_insights),
            ("spatial", self._spatial_insights),
            ("workflow_pattern", self._workflow_pattern_insights),
        ]

        for name, method in correlators:
            try:
                results = method()
                correlator_stats[name] = len(results)
                raw.extend(results)
            except Exception:
                correlator_stats[name] = "error"
                logger.exception("%s correlator failed", name)

        logger.info("InsightEngine: correlator results — %s", correlator_stats)

        # Ensure every insight has a dedup key
        for insight in raw:
            if not insight.dedup_key:
                insight.compute_dedup_key()

        # Apply source weights: modulate each insight's confidence by the
        # effective weight for its source category.  This is how user tuning
        # and AI drift influence which insights are surfaced.
        raw = self._apply_source_weights(raw)

        # Remove insights that are still within their staleness window
        before_dedup = len(raw)
        fresh = self._deduplicate(raw)
        if before_dedup > len(fresh):
            logger.debug(
                "Deduplication removed %d of %d insights",
                before_dedup - len(fresh),
                before_dedup,
            )

        # Persist the survivors — wrap each call so a corrupted user_model.db
        # does not discard insights that were computed from healthy sources
        # (events.db correlators like email_volume, meeting_density, etc.).
        for insight in fresh:
            try:
                self._store_insight(insight)
            except Exception as e:
                logger.warning(
                    "Failed to persist insight %s (user_model.db may be corrupted): %s",
                    insight.id,
                    e,
                )

        # When no insights are produced, log a data sufficiency report so
        # operators can see exactly which correlators are blocked and why.
        if not fresh:
            try:
                sufficiency = await self.get_data_sufficiency_report()
                ready_count = sum(1 for v in sufficiency.values() if v.get("status") == "ready")
                total = len(sufficiency)
                blocked = {k: v for k, v in sufficiency.items() if v.get("status") != "ready"}
                logger.info(
                    "Insight engine produced 0 insights: %d/%d correlators have sufficient data. "
                    "Blocked correlators: %s",
                    ready_count,
                    total,
                    blocked,
                )
            except Exception as e:
                logger.warning("Failed to generate data sufficiency report: %s", e)

        # Update diagnostic counters for observability.
        self._total_runs += 1
        self._last_run_at = datetime.now(timezone.utc).isoformat()
        self._last_correlator_stats = correlator_stats

        # Mark successful run so subsequent calls within the TTL are skipped.
        self._last_insight_run = time.monotonic()

        return fresh

    def get_diagnostics(self) -> dict:
        """Return insight engine diagnostic information for monitoring.

        Provides per-correlator execution stats from the last run, overall
        run counts, stored insight totals, and a health indicator.  Designed
        for the admin dashboard and data-quality endpoint to quickly assess
        whether the insight pipeline is producing results or stalled.

        Returns:
            dict with keys: ``total_runs``, ``last_run_at``,
            ``last_correlator_stats``, ``total_insights_stored``,
            ``insights_by_type``, and ``health``.
        """
        total_stored = 0
        by_type: dict[str, int] = {}
        try:
            with self.db.get_connection("user_model") as conn:
                total_stored = conn.execute(
                    "SELECT COUNT(*) FROM insights"
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT type, COUNT(*) as cnt FROM insights GROUP BY type"
                ).fetchall()
                by_type = {row["type"]: row["cnt"] for row in rows}
        except Exception as e:
            logger.warning("get_diagnostics: failed to query insights table: %s", e)

        # Health: no_data if never run, degraded if last run produced all
        # zeros or errors, ok otherwise.
        if self._total_runs == 0:
            health = "no_data"
        elif all(v == 0 or v == "error" for v in self._last_correlator_stats.values()):
            health = "degraded"
        else:
            health = "ok"

        return {
            "total_runs": self._total_runs,
            "last_run_at": self._last_run_at,
            "last_correlator_stats": self._last_correlator_stats,
            "total_insights_stored": total_stored,
            "insights_by_type": by_type,
            "health": health,
        }

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
            # Relationship intelligence insights derive from the relationships
            # signal profile (inbound/outbound counts, response time ring buffers).
            # Weighted against direct messaging — the source with the most relationship
            # signal data.
            "reciprocity_imbalance": "messaging.direct",
            "fast_responder": "messaging.direct",
            "email_volume": "email.work",
            "communication_style": "messaging.direct",
            # Inbound style mismatch insights derive from comparing received
            # messages against the user's own outbound style baseline — weighted
            # against the same messaging source as the outbound style insight.
            "style_mismatch": "messaging.direct",
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
            # Decision pattern insights: speed/fatigue come from task/calendar
            # signals; delegation comes from outbound message patterns.
            "decision_speed": "email.work",
            "delegation_tendency": "messaging.direct",
            "decision_fatigue": "messaging.direct",
            # Topic interest insights derive from all communication events (email
            # and messages), weighted against the broadest applicable source key.
            "top_interests": "email.work",
            "trending_topic": "email.work",
            # Cadence insights derive from email and messaging response-time data.
            # Baseline and peak-hours are weighted against email (broadest source);
            # per-contact fast replies are weighted against direct messaging.
            "response_time_baseline": "email.work",
            "fastest_contacts": "messaging.direct",
            "communication_peak_hours": "email.work",
            "channel_cadence": "email.work",
            # Routine pattern insights derive from episodic memory (calendar events,
            # email events, location signals), weighted against the broadest applicable
            # source key.  Routines shift slowly, so a 7-day staleness TTL is appropriate.
            "routine_pattern": "email.work",
            # Email timing insights derive from email send/receive timestamps,
            # weighted against work email (broadest applicable email source key).
            "email_timing": "email.work",
            # Meeting density insights derive from calendar event frequency,
            # weighted against calendar meetings — the source that produces them.
            "meeting_density": "calendar.meetings",
            # Spatial insights derive from the spatial signal profile (calendar location
            # fields, iOS context updates, explicit location events).  All three sub-types
            # use the location.visits source key — the same key used by the
            # _place_frequency_insights() correlator.
            "spatial_top_location": "location.visits",
            "spatial_work_location": "location.visits",
            "spatial_location_diversity": "location.visits",
            # Workflow pattern insights (from _workflow_pattern_insights).  Email and task
            # workflow patterns are derived from email-processing behavior; calendar workflow
            # patterns from meeting-prep signals; interaction patterns from direct messaging
            # cadence (batch replies, quick responses).
            "workflow_pattern_email": "email.work",
            "workflow_pattern_task": "email.work",
            "workflow_pattern_calendar": "email.work",
            "workflow_pattern_interaction": "messaging.direct",
        }

        weighted: list[Insight] = []
        dropped_count = 0
        for insight in insights:
            source_key = category_to_source.get(insight.category)
            if source_key:
                weight = self.swm.get_effective_weight(source_key)
                original_confidence = insight.confidence
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
            else:
                dropped_count += 1
                logger.debug(
                    "Insight dropped by source weight: type=%s source=%s "
                    "original_confidence=%.3f weighted_confidence=%.3f threshold=0.1",
                    getattr(insight, "insight_type", insight.type),
                    source_key or "unmapped",
                    original_confidence if source_key else insight.confidence,
                    insight.confidence,
                )

        if dropped_count:
            logger.info(
                "Source weight filtering: kept %d insights, dropped %d below 0.1 threshold",
                len(weighted),
                dropped_count,
            )

        return weighted

    # ------------------------------------------------------------------
    # Contact name resolution helpers
    # ------------------------------------------------------------------

    def _load_contact_name_map(self) -> dict[str, str]:
        """Load all email-to-contact-name mappings from the entities database.

        Performs a single JOIN across ``contact_identifiers`` and ``contacts``
        to build a complete email → display-name dictionary.  Called once per
        correlator that needs to humanize e-mail addresses in insight summaries
        (``_contact_gap_insights``, ``_inbound_style_insights``, and
        ``_cadence_response_insights``).

        The query is intentionally a full scan: the contacts table typically
        has < 1 000 rows and we amortise the cost across all contacts processed
        in a single correlator pass.

        Returns:
            Dict mapping **lowercase** e-mail addresses to their contact
            display names.  Returns an empty dict if the entities database is
            unavailable or the tables do not yet exist (e.g. first-run before
            any connector has synced contacts).

        Examples::

            >>> engine._load_contact_name_map()
            {'alice@example.com': 'Alice Smith', 'bob@example.com': 'Bob Jones'}
        """
        try:
            with self.db.get_connection("entities") as conn:
                rows = conn.execute(
                    """SELECT ci.identifier, c.name
                       FROM contact_identifiers ci
                       JOIN contacts c ON c.id = ci.contact_id
                       WHERE ci.identifier_type = 'email'"""
                ).fetchall()
                return {row["identifier"].lower(): row["name"] for row in rows}
        except Exception:
            logger.debug("Could not load contact name map from entities DB")
            return {}

    def _display_name(self, email: str, name_map: dict[str, str]) -> str:
        """Resolve an e-mail address to a human-readable contact name.

        Looks up *email* (case-insensitive) in *name_map*, which should be the
        dict returned by ``_load_contact_name_map()``.  Falls back to the raw
        e-mail address when no contact record exists, so insight summaries
        remain informative even when the contacts table is sparse.

        Args:
            email:    E-mail address key used in the signal profiles.
            name_map: Dict from ``_load_contact_name_map()``, mapping lowercase
                      e-mail addresses to display names.

        Returns:
            Display name string (e.g. ``"Alice Smith"``) or, on a miss, the
            original e-mail address (e.g. ``"alice@example.com"``).

        Examples::

            >>> engine._display_name(
            ...     "alice@example.com",
            ...     {"alice@example.com": "Alice Smith"},
            ... )
            'Alice Smith'
            >>> engine._display_name("unknown@example.com", {})
            'unknown@example.com'
        """
        return name_map.get(email.lower(), email)

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
            # Fallback: build contact-gap insights directly from events.db
            # when user_model.db is corrupted or has no relationship profile.
            return self._contact_gap_insights_from_events()

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)
        # Load email → display-name map once; correlator iterates many contacts.
        name_map = self._load_contact_name_map()
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
                # Use display name when a contact record exists; fall back to
                # the raw email address so the insight remains actionable when
                # the contacts table hasn't been populated yet.
                label = self._display_name(addr, name_map)
                # Enrich with last-discussed topics from episodic memory.
                # Transforms "12 days since last contact" → "12 days since
                # last contact. Last topics: budget-review, q1-planning."
                # giving the user concrete context about what to follow up on.
                last_topics = self._get_contact_last_topics(addr)
                topic_suffix = (
                    f" Last topics: {', '.join(last_topics)}."
                    if last_topics
                    else ""
                )
                insight = Insight(
                    type="relationship_intelligence",
                    summary=(
                        f"It has been {int(days_since)} days since you last contacted {label} "
                        f"(usual interval ~{int(avg_gap)} days).{topic_suffix}"
                    ),
                    confidence=confidence,
                    evidence=[
                        f"days_since_last={int(days_since)}",
                        f"avg_gap_days={int(avg_gap)}",
                        f"interaction_count={count}",
                        *(f"last_topic={t}" for t in last_topics),
                    ],
                    category="contact_gap",
                    # Keep the email address as the entity so the dedup key
                    # is stable even if the display name changes in contacts.
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

    def _contact_gap_insights_from_events(self) -> list[Insight]:
        """Fallback contact-gap correlator using events.db directly.

        Called when the relationships signal profile is unavailable (e.g.
        user_model.db is corrupted or has not been populated yet).  Mirrors
        the approach in ``PredictionEngine._build_contacts_from_events()``
        by querying inbound email events from the last 90 days.

        For each sender with >= 5 inbound emails, checks whether the user
        has replied in the last 14 days.  If not, generates a contact_gap
        insight.  Marketing/no-reply addresses are filtered out via the
        shared ``is_marketing_or_noreply`` helper.

        Returns:
            A list of contact_gap :class:`Insight` objects, or an empty list
            on any database error.
        """
        insights: list[Insight] = []
        try:
            with self.db.get_connection("events") as conn:
                # Find contacts with 5+ inbound emails in the last 90 days
                rows = conn.execute(
                    """SELECT
                           json_extract(payload, '$.from_address') AS sender,
                           COUNT(*) AS cnt,
                           MAX(timestamp) AS last_seen
                       FROM events
                       WHERE type = 'email.received'
                         AND timestamp > datetime('now', '-90 days')
                         AND json_extract(payload, '$.from_address') IS NOT NULL
                       GROUP BY sender
                       HAVING cnt >= 5"""
                ).fetchall()

                now = datetime.now(timezone.utc)
                name_map = self._load_contact_name_map()
                skipped_marketing = 0

                for row in rows:
                    addr = row["sender"]
                    if not addr:
                        continue

                    # Skip marketing/automated senders
                    if is_marketing_or_noreply(addr):
                        skipped_marketing += 1
                        continue

                    # Check if user has replied to this contact recently
                    reply_row = conn.execute(
                        """SELECT 1 FROM events
                           WHERE type = 'email.sent'
                             AND json_extract(payload, '$.to_address') = ?
                             AND timestamp > datetime('now', '-14 days')
                           LIMIT 1""",
                        (addr,),
                    ).fetchone()

                    if reply_row:
                        continue  # User has been in touch recently

                    # Compute days since last interaction
                    try:
                        last_dt = datetime.fromisoformat(
                            row["last_seen"].replace("Z", "+00:00")
                        )
                        days_since = (now - last_dt).total_seconds() / 86400
                    except (ValueError, TypeError):
                        continue

                    # Only flag contacts silent for 14+ days
                    if days_since < 14:
                        continue

                    confidence = min(0.7, 0.35 + (days_since / 30) * 0.1)
                    label = self._display_name(addr, name_map)
                    insight = Insight(
                        type="relationship_intelligence",
                        summary=(
                            f"It has been {int(days_since)} days since you last "
                            f"heard from {label} ({row['cnt']} emails in 90 days). "
                            f"Consider reaching out."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"days_since_last={int(days_since)}",
                            f"inbound_count_90d={row['cnt']}",
                            "source=events_db_fallback",
                        ],
                        category="contact_gap",
                        entity=addr,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

            logger.debug(
                "contact_gap_insights_from_events: %d insights generated "
                "(skipped_marketing=%d)",
                len(insights),
                skipped_marketing,
            )
        except Exception as e:
            logger.warning(
                "contact_gap_insights_from_events fallback failed: %s", e
            )

        return insights

    def _get_contact_last_topics(self, email_addr: str, limit: int = 3) -> list[str]:
        """Return the most-recently-discussed topics for a given contact.

        Queries the episodes table for the most recent episode that includes
        ``email_addr`` in its ``contacts_involved`` JSON array and has at
        least one topic tag.  Returns an empty list on any error so callers
        treat missing episode data as a no-op.

        The LIKE pattern ``'%' || email_addr || '%'`` is an approximation of
        JSON-array membership that works across all SQLite versions without
        the json1 extension.  It can produce false positives only when one
        email address is a substring of another (extremely unlikely given
        RFC 5321 format), so the trade-off is acceptable here.

        Args:
            email_addr: The contact's email address to search for.
            limit: Maximum number of topic strings to return (default 3).

        Returns:
            A list of topic strings from the most recent episode, or an
            empty list if no matching episode with topic data is found.

        Examples::

            engine._get_contact_last_topics("alice@example.com", limit=3)
            # → ["budget-review", "q1-planning", "headcount"]

            engine._get_contact_last_topics("nobody@example.com")
            # → []  (no episodes involving this address)
        """
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT topics
                       FROM episodes
                       WHERE contacts_involved LIKE '%' || ? || '%'
                         AND topics IS NOT NULL
                         AND topics != '[]'
                       ORDER BY timestamp DESC
                       LIMIT 1""",
                    (email_addr,),
                ).fetchone()
        except Exception:
            # Fail-open: topic enrichment is best-effort, never fatal.
            return []

        if not row:
            return []

        try:
            topics = json.loads(row["topics"])
            # Filter empty / whitespace-only strings before capping.
            return [str(t) for t in topics if t and str(t).strip()][:limit]
        except (json.JSONDecodeError, TypeError):
            return []

    # ------------------------------------------------------------------
    # Correlator: Relationship Intelligence (reciprocity & fast responders)
    # ------------------------------------------------------------------

    def _relationship_intelligence_insights(self) -> list[Insight]:
        """Surface social-graph discoveries from the relationships signal profile.

        Produces two sub-categories of ``relationship_intelligence`` insights:

        **reciprocity_imbalance** — flags relationships with a strong directional
        asymmetry in who initiates conversations.  When the user sends ≥ 85% of
        all messages with a contact (outbound/(inbound+outbound) ≥ 0.85), the
        relationship is one-sided: the user always reaches out but the contact
        rarely reciprocates.  The inverse (≤ 15% outbound) flags contacts who
        consistently reach out but rarely hear back from the user.

        **fast_responder** — identifies contacts the user responds to unusually
        quickly (avg response time < 30 minutes across ≥ 5 measured replies).
        Fast response time is an implicit signal of high relationship priority
        even when the contact has not been explicitly tagged as important.

        Filtering:
            - Marketing/automated senders are excluded via
              ``is_marketing_or_noreply``.  These have structural outbound==0
              patterns that would otherwise flood the reciprocity list.
            - Contacts with fewer than 10 total interactions are skipped for
              reciprocity (not enough data to establish a pattern).
            - Contacts with fewer than 5 measured response times are skipped
              for fast_responder (one quick reply doesn't establish a pattern).

        Deduplication:
            Each insight uses ``type:category:entity`` as its dedup key, so
            the same contact cannot generate duplicate insights within the
            staleness window (default 7 days / 168 hours).

        Examples::

            # Alice always emails first — user rarely initiates
            "Almost all messages from Alice Smith go unanswered — she reaches
             out first 92% of the time. Consider initiating contact."

            # Bob gets very fast replies
            "You respond to Bob Jones in under 8 minutes on average — he may
             be a high-priority contact."
        """
        insights: list[Insight] = []

        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return []

        contacts = rel_profile["data"].get("contacts", {})
        if not contacts:
            return []

        # Build name map once for all contacts in this pass.
        name_map = self._load_contact_name_map()

        for addr, data in contacts.items():
            # Skip marketing/automated senders — they always have outbound==0
            # patterns that would pollute the reciprocity insight with
            # "newsletter@company.com always initiates" false alarms.
            if is_marketing_or_noreply(addr):
                continue

            inbound = data.get("inbound_count", 0)
            outbound = data.get("outbound_count", 0)
            total = inbound + outbound
            label = self._display_name(addr, name_map)

            # --- Sub-insight 1: Reciprocity imbalance ---
            # Only fire when there is enough evidence to establish a pattern
            # (10+ interactions) and the imbalance is clear (≥ 85% one-sided).
            if total >= 10:
                ratio = outbound / total  # fraction of messages the user sent
                if ratio >= 0.85:
                    # User initiates ≥ 85% of conversations — one-sided relationship.
                    pct = int(round(ratio * 100))
                    confidence = min(0.8, 0.5 + (ratio - 0.85) * 2.0)
                    insight = Insight(
                        type="relationship_intelligence",
                        summary=(
                            f"You initiate {pct}% of conversations with {label}. "
                            f"They rarely reach out first — consider whether this "
                            f"relationship is balanced."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"outbound_count={outbound}",
                            f"inbound_count={inbound}",
                            f"outbound_ratio={ratio:.2f}",
                            f"total_interactions={total}",
                        ],
                        category="reciprocity_imbalance",
                        entity=addr,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

                elif ratio <= 0.15 and outbound > 0:
                    # Contact initiates ≥ 85% of conversations — user is mostly
                    # a responder.  Only surface when the user has sent at least
                    # one message (outbound > 0) to confirm it is a real two-way
                    # channel, not just a mailing list the filter missed.
                    pct = int(round((1 - ratio) * 100))
                    confidence = min(0.75, 0.45 + (0.15 - ratio) * 2.0)
                    insight = Insight(
                        type="relationship_intelligence",
                        summary=(
                            f"{label} initiates {pct}% of your conversations. "
                            f"You rarely reach out first — they may value this "
                            f"relationship more than your message history suggests."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"outbound_count={outbound}",
                            f"inbound_count={inbound}",
                            f"outbound_ratio={ratio:.2f}",
                            f"total_interactions={total}",
                        ],
                        category="reciprocity_imbalance",
                        entity=addr,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

            # --- Sub-insight 2: Fast responder ---
            # Surface when the user has at least 5 measured response times and
            # their average response to this contact is under 30 minutes.
            # A 30-minute threshold is conservative: it captures genuine
            # high-priority responders while filtering out contacts the user
            # responds to "fairly quickly" (1–2 hours).
            resp_times = data.get("response_times_seconds", [])
            if len(resp_times) >= 5:
                avg_seconds = sum(resp_times) / len(resp_times)
                fast_threshold_seconds = 1800  # 30 minutes
                if avg_seconds < fast_threshold_seconds:
                    avg_minutes = int(avg_seconds / 60)
                    # Higher confidence when the average is very fast (< 5 min)
                    # vs. merely fast (< 30 min).
                    confidence = min(0.8, 0.55 + (1 - avg_seconds / fast_threshold_seconds) * 0.25)
                    insight = Insight(
                        type="relationship_intelligence",
                        summary=(
                            f"You respond to {label} in under {max(1, avg_minutes)} minutes "
                            f"on average — they appear to be a high-priority contact."
                        ),
                        confidence=confidence,
                        evidence=[
                            f"avg_response_time_seconds={int(avg_seconds)}",
                            f"response_time_samples={len(resp_times)}",
                            f"avg_response_minutes={avg_minutes}",
                        ],
                        category="fast_responder",
                        entity=addr,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

        logger.debug(
            "relationship_intelligence_insights: %d insights generated "
            "(reciprocity + fast_responder)",
            len(insights),
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
                # Use 30-day window to match docstring and give a full month of
                # data — a 7-day window was too short: on Sundays the comparison
                # days (Saturday–Monday prior week) fall outside the window,
                # leaving only 1 weekday bucket which can never reach the 1.5x
                # threshold relative to itself.
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
    # Correlator: Email Peak Hour
    # ------------------------------------------------------------------

    def _email_peak_hour_insights(self) -> list[Insight]:
        """Identify the user's peak email hours.

        Queries email.received events from the last 30 days, buckets by
        hour-of-day in the user's timezone, and surfaces the peak hour if
        it is significantly busier than the average.
        """
        insights: list[Insight] = []

        try:
            with self.db.get_connection("events") as conn:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                rows = conn.execute(
                    """SELECT timestamp FROM events
                       WHERE type = 'email.received'
                       AND timestamp > ?""",
                    (cutoff,),
                ).fetchall()
        except Exception:
            return []

        if len(rows) < 50:
            return []  # Need at least 50 emails for meaningful hourly analysis

        hour_counts: Counter[int] = Counter()
        for row in rows:
            try:
                dt = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                local_dt = dt.astimezone(self._tz)
                hour_counts[local_dt.hour] += 1
            except (ValueError, TypeError):
                continue

        if not hour_counts:
            return []

        peak_hour, peak_count = hour_counts.most_common(1)[0]
        avg_count = sum(hour_counts.values()) / max(len(hour_counts), 1)

        # Only surface if the peak hour is at least 2.0x the average
        if peak_count >= avg_count * 2.0:
            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your email peaks between {peak_hour}:00-{peak_hour + 1}:00 "
                    f"({peak_count} emails vs ~{int(avg_count)} average)"
                ),
                confidence=min(0.85, 0.5 + (peak_count / max(avg_count, 1) - 1.0) * 0.15),
                evidence=[
                    f"peak_hour={peak_hour}",
                    f"peak_count={peak_count}",
                    f"avg_hourly={int(avg_count)}",
                ],
                category="email_timing",
                entity=str(peak_hour),
            )
            insight.compute_dedup_key()
            insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Correlator: Meeting Density by Day of Week
    # ------------------------------------------------------------------

    def _meeting_density_insights(self) -> list[Insight]:
        """Identify the user's meeting-heaviest day of the week.

        Queries calendar events from the last 30 days using the actual
        calendar event start time (from the JSON payload ``start_time``
        field) rather than the sync timestamp.  Includes both
        ``calendar.event.created`` and ``calendar.event.updated`` events.
        Buckets by day-of-week and surfaces the busiest day.
        """
        insights: list[Insight] = []

        try:
            with self.db.get_connection("events") as conn:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                rows = conn.execute(
                    """SELECT json_extract(payload, '$.start_time') as start_time
                       FROM events
                       WHERE type IN ('calendar.event.created', 'calendar.event.updated')
                       AND timestamp > ?
                       AND json_extract(payload, '$.start_time') IS NOT NULL""",
                    (cutoff,),
                ).fetchall()
        except Exception:
            return []

        if len(rows) < 10:
            return []  # Need at least 10 calendar events

        day_counts: Counter[str] = Counter()
        for row in rows:
            try:
                start_time = row["start_time"]
                # Date-only strings (e.g. '2026-03-05') represent all-day
                # events — treat as midnight UTC on that day.
                if len(start_time) <= 10:
                    dt = datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc)
                else:
                    dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                local_dt = dt.astimezone(self._tz)
                day_counts[local_dt.strftime("%A")] += 1
            except (ValueError, TypeError):
                continue

        if not day_counts:
            return []

        peak_day, peak_count = day_counts.most_common(1)[0]
        avg_count = sum(day_counts.values()) / max(len(day_counts), 1)

        # Only surface if peak day is at least 1.5x the average
        if peak_count >= avg_count * 1.5:
            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your meeting-heaviest day is {peak_day} "
                    f"({peak_count} meetings vs ~{int(avg_count)} average)"
                ),
                confidence=min(0.8, 0.45 + (peak_count / max(avg_count, 1) - 1.0) * 0.2),
                evidence=[
                    f"peak_day={peak_day}",
                    f"peak_count={peak_count}",
                    f"avg_daily={int(avg_count)}",
                ],
                category="meeting_density",
                entity=peak_day,
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
            # Use a 90-day lower bound on timestamp to avoid scanning the
            # entire events table, but filter on the *payload* start_time
            # so events synced long ago but starting within 24h are found.
            long_cutoff = (now - timedelta(days=90)).isoformat()
            cutoff_now = now.isoformat()
            cutoff_future = (now + timedelta(hours=24)).isoformat()

            with self.db.get_connection("events") as conn:
                cal_rows = conn.execute(
                    """SELECT payload FROM events
                       WHERE type = 'calendar.event.created'
                         AND timestamp > ?
                         AND (
                           COALESCE(json_extract(payload, '$.start_time'),
                                    json_extract(payload, '$.start')) > ?
                           AND COALESCE(json_extract(payload, '$.start_time'),
                                        json_extract(payload, '$.start')) <= ?
                         )
                       ORDER BY timestamp DESC
                       LIMIT 200""",
                    (long_cutoff, cutoff_now, cutoff_future),
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
        """Surface communication-style observations from the linguistic signal profile.

        Reads ``averages`` from the ``linguistic`` profile and generates up to five
        complementary insight types:

        1. **Formality** — formal / balanced / casual writing register.
        2. **Question rate** — inquisitive (question-heavy) vs. declarative style.
        3. **Hedge rate** — tentative ("maybe", "I think") vs. confident phrasing.
        4. **Emoji rate** — expressive emoji-using style vs. plain-text communication.
        5. **Vocabulary diversity** — rich word choice (high TTR) vs. simple/repetitive.

        Minimum 5 samples required for all insights; formality fires at 3.
        Confidence scales with sample count, capped at 0.85.

        Example::

            # With 200 outbound messages analysed, may return:
            # - "Your overall writing style is balanced (formality 0.48, 200 msgs)"
            # - "You frequently pepper messages with questions (rate 0.42/sentence)"
            # - "You tend to hedge often ('maybe', 'I think') — tentative phrasing"
        """
        insights: list[Insight] = []

        profile = self.ums.get_signal_profile("linguistic")
        if not profile:
            return []

        averages = profile["data"].get("averages", {})
        formality = averages.get("formality")
        samples_count = profile.get("samples_count", 0)

        if formality is None or samples_count < 3:
            return []

        # --- Insight 1: Formality ---
        if formality >= 0.7:
            style_label = "formal"
        elif formality <= 0.3:
            style_label = "casual"
        else:
            style_label = "balanced"

        confidence = min(0.85, 0.4 + samples_count * 0.02)

        insight = Insight(
            type="communication_style",
            summary=(
                f"Your overall writing style is {style_label} "
                f"(formality score {formality:.2f}, based on {samples_count} messages)."
            ),
            confidence=confidence,
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

        # The remaining insights require at least 5 samples to be meaningful.
        if samples_count < 5:
            return insights

        # --- Insight 2: Question rate ---
        # A rate ≥ 0.35 means at least 1 question per 3 sentences on average,
        # which marks a clearly inquisitive style. ≤ 0.05 marks an assertive,
        # declarative communicator.
        question_rate = averages.get("question_rate")
        if question_rate is not None:
            if question_rate >= 0.35:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"You frequently ask questions in your messages "
                        f"({question_rate:.2f} questions/sentence). "
                        "Your communication style is inquisitive and dialogue-oriented."
                    ),
                    confidence=confidence,
                    evidence=[f"question_rate={question_rate:.3f}"],
                    category="communication_style",
                    entity="inquisitive",
                )
                insight.compute_dedup_key()
                insights.append(insight)
            elif question_rate <= 0.05:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"You rarely phrase things as questions "
                        f"({question_rate:.2f} questions/sentence). "
                        "Your communication style is direct and declarative."
                    ),
                    confidence=confidence,
                    evidence=[f"question_rate={question_rate:.3f}"],
                    category="communication_style",
                    entity="declarative",
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # --- Insight 3: Hedge rate ---
        # ≥ 0.5 hedges/sentence ("maybe", "I think", "sort of") indicates
        # a consistently tentative style; ≤ 0.05 marks confident, direct writing.
        hedge_rate = averages.get("hedge_rate")
        if hedge_rate is not None:
            if hedge_rate >= 0.5:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"You frequently use hedge words like 'maybe', 'I think', "
                        f"or 'sort of' ({hedge_rate:.2f}/sentence). "
                        "Your writing comes across as thoughtful but sometimes tentative."
                    ),
                    confidence=confidence,
                    evidence=[f"hedge_rate={hedge_rate:.3f}"],
                    category="communication_style",
                    entity="tentative",
                )
                insight.compute_dedup_key()
                insights.append(insight)
            elif hedge_rate <= 0.05:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"You rarely hedge in your messages "
                        f"({hedge_rate:.2f} hedges/sentence). "
                        "Your writing style is confident and direct."
                    ),
                    confidence=confidence,
                    evidence=[f"hedge_rate={hedge_rate:.3f}"],
                    category="communication_style",
                    entity="confident",
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # --- Insight 4: Emoji rate ---
        # ≥ 0.05 emojis per word means roughly 1 emoji per 20 words, which
        # is a noticeably expressive pattern.
        emoji_rate = averages.get("emoji_rate")
        if emoji_rate is not None and emoji_rate >= 0.05:
            insight = Insight(
                type="communication_style",
                summary=(
                    f"You use emojis frequently in your messages "
                    f"({emoji_rate:.3f} emojis/word). "
                    "Your writing style is visually expressive."
                ),
                confidence=confidence,
                evidence=[f"emoji_rate={emoji_rate:.3f}"],
                category="communication_style",
                entity="expressive",
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # --- Insight 5: Vocabulary diversity (type-token ratio) ---
        # TTR ≥ 0.75 indicates a notably rich, varied vocabulary across messages.
        # TTR ≤ 0.40 indicates repetitive or simple word choice.
        unique_word_ratio = averages.get("unique_word_ratio")
        if unique_word_ratio is not None:
            if unique_word_ratio >= 0.75:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"Your vocabulary is notably diverse "
                        f"(type-token ratio {unique_word_ratio:.2f}). "
                        "You use a wide range of words across your messages."
                    ),
                    confidence=confidence,
                    evidence=[f"unique_word_ratio={unique_word_ratio:.3f}"],
                    category="communication_style",
                    entity="rich_vocabulary",
                )
                insight.compute_dedup_key()
                insights.append(insight)
            elif unique_word_ratio <= 0.40:
                insight = Insight(
                    type="communication_style",
                    summary=(
                        f"Your writing uses a consistent, focused vocabulary "
                        f"(type-token ratio {unique_word_ratio:.2f}). "
                        "You tend to use the same words reliably across messages."
                    ),
                    confidence=confidence,
                    evidence=[f"unique_word_ratio={unique_word_ratio:.3f}"],
                    category="communication_style",
                    entity="focused_vocabulary",
                )
                insight.compute_dedup_key()
                insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Correlator: Inbound Style Mismatch
    # ------------------------------------------------------------------

    def _inbound_style_insights(self) -> list[Insight]:
        """Surface formality-mismatch insights from inbound linguistic profiles.

        The ``linguistic_inbound`` profile accumulates per-contact writing-style
        averages from every inbound message the system has processed (100K+
        samples).  By comparing each contact's average inbound formality against
        the user's own outbound formality baseline, the system can flag contacts
        whose communication style differs significantly from the user's own voice.

        This makes 101K+ dormant inbound samples actionable: a contact who always
        writes casually (formality ≤ 0.3) while the user's baseline is formal
        (≥ 0.5) suggests the user could try a warmer tone; the reverse suggests
        matching their professional register.

        Filtering:
            - Marketing/automated senders are excluded: the user cannot adjust
              their style with a newsletter or a no-reply address.
            - Contacts with fewer than 5 inbound samples are skipped (unreliable
              style reading).
            - Only fire when the formality gap exceeds 0.3 to avoid surfacing
              trivial style differences.

        Capping:
            At most 10 insights are returned, ranked by gap size descending, so
            the most actionable mismatches are surfaced first.

        Generated insight type: ``communication_style``,
        category ``style_mismatch``, entity = contact e-mail address.
        Staleness TTL: 7 days (default) — re-fires weekly if gap persists.

        Example:
            >>> engine._inbound_style_insights()
            [Insight(summary="alice@example.com writes casually (formality 0.20),
             while your baseline is 0.72. Consider a warmer, more relaxed tone.")]
        """
        insights: list[Insight] = []

        # Require both the inbound contact-style profile and the user's own
        # outbound baseline to make a meaningful comparison.
        inbound_profile = self.ums.get_signal_profile("linguistic_inbound")
        if not inbound_profile:
            return []

        outbound_profile = self.ums.get_signal_profile("linguistic")
        # Default to 0.5 (neutral baseline) if outbound profile is missing.
        user_formality: float = (
            outbound_profile["data"].get("averages", {}).get("formality", 0.5)
            if outbound_profile
            else 0.5
        )

        per_contact_avgs = inbound_profile["data"].get("per_contact_averages", {})

        # Load email → display-name map once for the entire correlator pass.
        name_map = self._load_contact_name_map()

        # Collect all qualifying mismatches before sorting so we can cap cleanly.
        # Each tuple: (gap, contact_email, averages_dict)
        mismatches: list[tuple[float, str, dict]] = []

        for contact_email, avgs in per_contact_avgs.items():
            samples_count = avgs.get("samples_count", 0)

            # Skip contacts with too few samples for a reliable style reading.
            if samples_count < 5:
                continue

            # Skip marketing/automated senders — the user cannot meaningfully
            # adjust their communication style with an automated newsletter.
            if is_marketing_or_noreply(contact_email):
                continue

            contact_formality = avgs.get("formality", 0.5)
            gap = abs(user_formality - contact_formality)

            # Only surface meaningful divergences (> 0.3 on the 0–1 scale).
            if gap > 0.3:
                mismatches.append((gap, contact_email, avgs))

        # Sort by largest gap first so the most actionable insights appear first
        # when the list is capped at 10.
        mismatches.sort(reverse=True)

        for gap, contact_email, avgs in mismatches[:10]:
            contact_formality = avgs["formality"]
            samples_count = avgs["samples_count"]

            if contact_formality < user_formality:
                # Contact writes more casually than the user's baseline.
                direction = "casually"
                advice = "Consider a warmer, more relaxed tone with them."
            else:
                # Contact writes more formally than the user's baseline.
                direction = "formally"
                advice = "Consider matching their professional register."

            # Resolve email to display name; fall back to raw address on miss.
            label = self._display_name(contact_email, name_map)

            insight = Insight(
                type="communication_style",
                summary=(
                    f"{label} writes {direction} "
                    f"(formality {contact_formality:.2f}), "
                    f"while your baseline is {user_formality:.2f}. "
                    f"{advice}"
                ),
                # Confidence scales with gap size: a 0.3 gap → 0.54, 0.7 gap → 0.8.
                confidence=min(0.80, 0.40 + gap * 0.57),
                evidence=[
                    f"contact_formality={contact_formality:.2f}",
                    f"user_formality={user_formality:.2f}",
                    f"gap={gap:.2f}",
                    f"samples_count={samples_count}",
                ],
                category="style_mismatch",
                entity=contact_email,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        logger.debug(
            "_inbound_style_insights: %d insights generated from %d qualifying contacts "
            "(%d total contacts in inbound profile)",
            len(insights),
            len([
                c for c, a in per_contact_avgs.items()
                if a.get("samples_count", 0) >= 5 and not is_marketing_or_noreply(c)
            ]),
            len(per_contact_avgs),
        )
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

        activity_by_hour_utc: dict[str, int] = data.get("activity_by_hour", {})
        activity_by_day: dict[str, int] = data.get("activity_by_day", {})

        if not activity_by_hour_utc or total_samples < MIN_SAMPLES:
            return []

        # Convert UTC hour buckets to user-local hour buckets so that
        # chronotype and peak-hour insights reflect the user's actual
        # day, not UTC offsets.
        now_utc = datetime.now(timezone.utc)
        utc_offset_hours = now_utc.astimezone(self._tz).utcoffset().total_seconds() / 3600
        activity_by_hour: dict[str, int] = {}
        for utc_h_str, count in activity_by_hour_utc.items():
            local_h = (int(utc_h_str) + int(utc_offset_hours)) % 24
            activity_by_hour[str(local_h)] = activity_by_hour.get(str(local_h), 0) + count

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
                hour_label = datetime.now(timezone.utc).astimezone(self._tz).replace(hour=peak_hour).strftime("%-I %p")
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
    # Correlator: Decision Patterns
    # ------------------------------------------------------------------

    def _decision_pattern_insights(self) -> list[Insight]:
        """Surface decision-making behavioral patterns from the decision signal profile.

        Reads the ``decision`` signal profile (built by ``DecisionExtractor``) and
        translates raw speed, delegation, and fatigue data into up to three insight
        sub-types.  The profile accumulates signals from task completions, outbound
        messages, and calendar events, so it captures a full picture of how the user
        approaches choices.

        **1. Decision speed comparison across domains (``decision_speed``):**
            Reads ``decision_speed_by_domain`` (time in seconds from task creation
            to completion, per domain) and surfaces a comparison between the fastest
            and slowest domains.  Only fires when at least two domains have data and
            the slowest domain takes at least 2× longer than the fastest — otherwise
            there is nothing meaningful to compare.  Seconds are translated into
            human-readable labels (under an hour / same day / multiple days).

            Example: "You complete work tasks quickly (under an hour) but take
            longer on finance decisions (avg 2.3 days)."

        **2. Delegation tendency (``delegation_tendency``):**
            Reads ``delegation_comfort`` (0 = micromanage everything, 1 = fully
            delegated) and ``_total_outbound_count`` (denominator used to compute
            the score).  Only fires when the score departs from the neutral 0.5
            baseline by more than ``DELEGATION_THRESHOLD`` (0.15), so balanced
            delegators receive no misleading label.  Requires at least 10 outbound
            messages to avoid noise from sparse early data.

            Example: "You prefer to handle decisions yourself (delegation score
            0.23, based on 85 messages)."

        **3. Decision fatigue time (``decision_fatigue``):**
            Reads ``fatigue_time_of_day`` — the hour at which the user consistently
            starts delegating decisions.  Only fires when the field is set (the
            DecisionExtractor writes it when it detects late-evening delegation
            events ≥ 20:00).

            Example: "Your decision fatigue tends to set in after 8 PM — consider
            front-loading complex choices earlier in the day."

        **Data requirements:**
            All sub-types require ``samples_count >= MIN_SAMPLES`` on the decision
            profile to guard against noise from the first few events.

        **Staleness TTL:**
            168 hours (7 days) for all sub-types.  Decision-making habits are
            slow-moving traits; daily refresh would be noisy.

        **Dedup strategy:**
            ``category`` is the sub-type (``decision_speed``, ``delegation_tendency``,
            ``decision_fatigue``).  ``entity`` is a stable string derived from the
            detected pattern label so the insight only re-surfaces when the
            underlying pattern changes, not on every new data point.

        Returns:
            list[Insight]: Zero to three decision-pattern insights.
        """
        # Low minimum: even 5 samples give a reasonable delegation ratio.
        MIN_SAMPLES = 5
        # Must depart from neutral 0.5 by this much to earn a label.
        DELEGATION_THRESHOLD = 0.15
        # Minimum outbound messages before delegation score is trustworthy.
        MIN_OUTBOUND = 10

        profile = self.ums.get_signal_profile("decision")
        if not profile:
            return []

        data = profile.get("data", {})
        total_samples = profile.get("samples_count", 0)

        if total_samples < MIN_SAMPLES:
            return []

        insights: list[Insight] = []

        # ----------------------------------------------------------------
        # Sub-insight 1: Decision speed comparison across domains
        # ----------------------------------------------------------------
        speed_by_domain: dict[str, float] = data.get("decision_speed_by_domain", {})

        if len(speed_by_domain) >= 2:
            def _speed_label(seconds: float) -> str:
                """Convert raw seconds into a human-readable speed description."""
                if seconds < 3600:
                    return "quickly (under an hour)"
                elif seconds < 28800:  # 8 hours
                    return f"within the same day (avg {seconds / 3600:.1f} hours)"
                elif seconds < 86400:  # 24 hours
                    return f"within a day (avg {seconds / 3600:.1f} hours)"
                else:
                    days = seconds / 86400
                    return f"over multiple days (avg {days:.1f} days)"

            sorted_domains = sorted(speed_by_domain.items(), key=lambda x: x[1])
            fastest_domain, fastest_seconds = sorted_domains[0]
            slowest_domain, slowest_seconds = sorted_domains[-1]

            # Only surface when there is a meaningful contrast (at least 2×).
            if slowest_seconds >= fastest_seconds * 2:
                summary = (
                    f"You complete {fastest_domain} tasks {_speed_label(fastest_seconds)} "
                    f"but take longer on {fastest_domain if fastest_domain != slowest_domain else slowest_domain} "
                    f"— your {slowest_domain} decisions take {_speed_label(slowest_seconds)}."
                )
                insight = Insight(
                    type="behavioral_pattern",
                    summary=(
                        f"You complete {fastest_domain} tasks {_speed_label(fastest_seconds)}, "
                        f"but your {slowest_domain} decisions take {_speed_label(slowest_seconds)}."
                    ),
                    confidence=min(0.80, 0.50 + total_samples * 0.003),
                    evidence=[
                        f"fastest_domain={fastest_domain}",
                        f"fastest_seconds={fastest_seconds:.0f}",
                        f"slowest_domain={slowest_domain}",
                        f"slowest_seconds={slowest_seconds:.0f}",
                        f"samples={total_samples}",
                    ],
                    category="decision_speed",
                    entity=f"{fastest_domain}_vs_{slowest_domain}",
                    staleness_ttl_hours=168,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 2: Delegation tendency
        # ----------------------------------------------------------------
        delegation_comfort = data.get("delegation_comfort")
        total_outbound = data.get("_total_outbound_count", 0)

        if delegation_comfort is not None and total_outbound >= MIN_OUTBOUND:
            if delegation_comfort >= 0.5 + DELEGATION_THRESHOLD:
                tendency = "high"
                description = (
                    f"You tend to delegate decisions freely to others "
                    f"(delegation score {delegation_comfort:.2f}, based on "
                    f"{total_outbound} outbound messages)."
                )
            elif delegation_comfort <= 0.5 - DELEGATION_THRESHOLD:
                tendency = "low"
                description = (
                    f"You prefer to handle decisions yourself "
                    f"(delegation score {delegation_comfort:.2f}, based on "
                    f"{total_outbound} outbound messages)."
                )
            else:
                tendency = None
                description = None

            if tendency is not None:
                insight = Insight(
                    type="behavioral_pattern",
                    summary=description,
                    confidence=min(0.80, 0.40 + total_outbound * 0.005),
                    evidence=[
                        f"delegation_comfort={delegation_comfort:.2f}",
                        f"total_outbound={total_outbound}",
                        f"tendency={tendency}",
                    ],
                    category="delegation_tendency",
                    entity=tendency,
                    staleness_ttl_hours=168,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 3: Decision fatigue time
        # ----------------------------------------------------------------
        fatigue_hour = data.get("fatigue_time_of_day")

        if fatigue_hour is not None:
            # Convert to 12-hour clock label for readability.
            if fatigue_hour == 0:
                hour_label = "midnight"
            elif fatigue_hour < 12:
                hour_label = f"{fatigue_hour} AM"
            elif fatigue_hour == 12:
                hour_label = "noon"
            else:
                hour_label = f"{fatigue_hour - 12} PM"

            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your decision fatigue tends to set in after {hour_label} — "
                    f"consider front-loading complex choices earlier in the day."
                ),
                confidence=0.65,
                evidence=[
                    f"fatigue_hour={fatigue_hour}",
                    f"samples={total_samples}",
                ],
                category="decision_fatigue",
                entity=str(fatigue_hour),
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        logger.debug(
            "decision_pattern_insights: %d insights generated "
            "(speed_domains=%d, delegation_comfort=%s, fatigue_hour=%s)",
            len(insights),
            len(speed_by_domain),
            f"{delegation_comfort:.2f}" if delegation_comfort is not None else "None",
            fatigue_hour,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Topic Interest
    # ------------------------------------------------------------------

    def _topic_interest_insights(self) -> list[Insight]:
        """Surface dominant-interest and trending-topic insights from the topics profile.

        Reads the ``topics`` signal profile (built by ``TopicExtractor``) which
        accumulates two structures:

          - ``topic_counts``: all-time keyword → message-count frequency map built
            incrementally as every email/message event is processed.
          - ``recent_topics``: ring buffer (capped at 500) of timestamped topic
            lists, one entry per processed event.

        This correlator translates those raw counts into two human-readable insight
        sub-types.

        **1. Top interests (``top_interests``):**
            Reads ``topic_counts`` and surfaces the user's five most-mentioned topics
            as a compact interest summary.  Requires at least ``MIN_SAMPLES`` (50)
            profile updates and at least ``MIN_TOP_COUNT`` (5) occurrences for the
            top topic, so that a sparse early profile does not produce misleading
            "top interests".  The ``entity`` fingerprint is derived from the top-3
            topic names so the insight is only re-surfaced when the dominant interest
            composition actually changes (e.g., a new topic displaces one of the
            top 3), not just because a new email arrived.

            Example::

                "Your most-engaged topics: work (847), project (634), team (523),
                email (441), meeting (312). Across 12,480 message observations."

        **2. Trending topic (``trending_topic``):**
            Compares topic frequencies in the most recent ``MIN_RECENT`` (50) entries
            of the ring buffer against the all-time ``topic_counts`` baseline.  A
            topic is "trending" when its recent-window rate is at least
            ``TRENDING_RATIO`` (2.0×) higher than its historical average and it
            appears at least 3 times in the recent window (avoids surfacing a topic
            that appeared only once as "trending").  The topic with the highest
            ratio is selected when multiple candidates exist.  Only fires when there
            are at least ``MIN_RECENT`` ring-buffer entries so the comparison is
            meaningful.

            Example::

                "You've been engaging significantly more with 'budget' topics
                recently — 3.2× above your usual rate (8 mentions in your last
                50 messages)."

        **Data requirements:**
            Both sub-types require the ``topics`` profile to exist with at least
            ``MIN_SAMPLES`` (50) updates to guard against noise from the first few
            events processed by the extractor.

        **Staleness TTL:**
            ``top_interests``: 168 hours (7 days) — dominant interests shift slowly.
            ``trending_topic``: 48 hours (2 days) — trending topics can change quickly.

        **Dedup strategy:**
            ``top_interests`` uses the concatenated top-3 topic names as ``entity``
            so the insight re-surfaces only when the dominant interest composition
            changes.  ``trending_topic`` uses the topic name itself as ``entity``
            so each unique trending topic gets its own dedup lifecycle.

        Returns:
            list[Insight]: Zero, one, or two insights depending on data availability
            and pattern strength.
        """
        # Minimum profile updates before we trust the data enough to surface insights.
        MIN_SAMPLES = 50
        # The top topic must appear in at least this many messages to be meaningful.
        MIN_TOP_COUNT = 5
        # Minimum recent_topics ring-buffer entries for trending-topic detection.
        MIN_RECENT = 50
        # Recent-rate / historical-rate ratio threshold for "trending" classification.
        TRENDING_RATIO = 2.0
        # A trending topic must appear at least this many times in the recent window.
        MIN_TRENDING_COUNT = 3
        # Number of top topics to include in the summary sentence.
        TOP_N = 5

        profile = self.ums.get_signal_profile("topics")
        if not profile:
            return []

        data = profile.get("data", {})
        total_samples = profile.get("samples_count", 0)

        if total_samples < MIN_SAMPLES:
            return []

        topic_counts: dict[str, int] = data.get("topic_counts", {})
        recent_topics: list[dict] = data.get("recent_topics", [])

        if not topic_counts:
            return []

        insights: list[Insight] = []

        # ----------------------------------------------------------------
        # Sub-insight 1: Top interests
        # ----------------------------------------------------------------
        # Sort topics by all-time frequency descending and take the top N.
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [(t, c) for t, c in sorted_topics[:TOP_N] if c >= MIN_TOP_COUNT]

        if top_topics:
            # Format as "topic (count)" pairs for readability in the summary.
            topic_list = ", ".join(f"{t} ({c:,})" for t, c in top_topics)
            # Total observation count across all topics provides context for scale.
            n_total = sum(c for _, c in sorted_topics)

            # Stable fingerprint: concatenate top-3 names so the dedup key
            # only changes when the composition of dominant interests shifts.
            top3_names = "_".join(t for t, _ in sorted_topics[:3])

            # Confidence grows with sample count, capped at 0.80.
            confidence = min(0.80, 0.50 + total_samples * 0.001)

            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your most-engaged topics: {topic_list}. "
                    f"Across {n_total:,} message observations."
                ),
                confidence=confidence,
                evidence=[
                    f"top_topic={sorted_topics[0][0]}",
                    f"top_count={sorted_topics[0][1]}",
                    f"unique_topics={len(topic_counts)}",
                    f"samples={total_samples}",
                ],
                category="top_interests",
                entity=top3_names,
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 2: Trending topic
        # ----------------------------------------------------------------
        if len(recent_topics) >= MIN_RECENT:
            # Tally topic frequencies within the most recent MIN_RECENT entries.
            recent_window = recent_topics[-MIN_RECENT:]
            recent_counts: dict[str, int] = {}
            for entry in recent_window:
                for topic in entry.get("topics", []):
                    recent_counts[topic] = recent_counts.get(topic, 0) + 1

            if recent_counts:
                # Total all-time mentions across the full corpus for the baseline rate.
                total_all_time = sum(topic_counts.values())

                best_trending: tuple[str, int, float] | None = None
                best_ratio = 0.0

                for topic, recent_count in recent_counts.items():
                    historical_count = topic_counts.get(topic, 0)
                    # Skip topics with no historical baseline — no rate to compare.
                    if historical_count == 0:
                        continue
                    # Skip topics that appear only once or twice in the recent window
                    # to prevent single-occurrence noise from registering as "trending".
                    if recent_count < MIN_TRENDING_COUNT:
                        continue

                    # Normalise each count by its corpus size to get comparable rates.
                    # recent_rate  = fraction of recent-window messages mentioning this topic
                    # historical_rate = fraction of all-time messages mentioning this topic
                    recent_rate = recent_count / len(recent_window)
                    historical_rate = historical_count / total_all_time

                    if historical_rate <= 0:
                        continue

                    ratio = recent_rate / historical_rate
                    if ratio > best_ratio and ratio >= TRENDING_RATIO:
                        best_ratio = ratio
                        best_trending = (topic, recent_count, ratio)

                if best_trending is not None:
                    topic, count, ratio = best_trending
                    insight = Insight(
                        type="behavioral_pattern",
                        summary=(
                            f"You've been engaging significantly more with '{topic}' "
                            f"recently — {ratio:.1f}× above your usual rate "
                            f"({count} mentions in your last {MIN_RECENT} messages)."
                        ),
                        confidence=min(0.75, 0.40 + ratio * 0.05),
                        evidence=[
                            f"trending_topic={topic}",
                            f"recent_count={count}",
                            f"ratio={ratio:.2f}",
                            f"recent_window_size={len(recent_window)}",
                        ],
                        category="trending_topic",
                        entity=topic,
                        staleness_ttl_hours=48,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

        logger.debug(
            "topic_interest_insights: %d insights generated "
            "(unique_topics=%d, recent_entries=%d, samples=%d)",
            len(insights),
            len(topic_counts),
            len(recent_topics),
            total_samples,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Cadence Response Patterns
    # ------------------------------------------------------------------

    def _cadence_response_insights(self) -> list[Insight]:
        """Surface communication-cadence insights from the cadence signal profile.

        Reads the ``cadence`` signal profile (built by ``CadenceExtractor``) which
        accumulates four running aggregates:

          - ``response_times``:             global list of reply-latency deltas in
            seconds (capped at 1,000 most-recent entries).
          - ``per_contact_response_times``: dict mapping contact addresses to their
            per-contact response-time lists.
          - ``per_channel_response_times``: dict mapping channel names (e.g.
            ``proton_mail``, ``imessage``) to their per-channel response-time lists.
          - ``hourly_activity``:            histogram of communication events by
            hour-of-day (string keys ``"0"``–``"23"``).

        This correlator translates those raw numbers into up to four human-readable
        insight sub-types.

        **1. Response time baseline (``response_time_baseline``):**
            Reports the user's overall average reply latency across all contacts
            and channels.  Requires at least ``MIN_RT_SAMPLES`` (10) global
            response-time observations.  Re-surfaces at most once every 7 days
            since average response time shifts slowly.

            Example::

                "Your average reply time across all contacts is 3.2 hours.
                Based on 427 response-time observations."

        **2. Priority-contact fast replies (``fastest_contacts``):**
            Identifies up to ``MAX_CONTACTS`` (3) contacts the user consistently
            replies to faster than half the global average — a strong signal of
            high priority or relationship closeness.  Requires ≥ ``MIN_CT_SAMPLES``
            (3) samples per contact.  Marketing and no-reply addresses are excluded
            via ``is_marketing_or_noreply()`` so automated senders never pollute
            the priority list.

            Example::

                "You reply to alice@example.com notably faster than average
                (0.5h vs 3.2h overall) — a strong priority signal."

        **3. Peak communication hours (``communication_peak_hours``):**
            Identifies the top three hours of day with the most communication
            activity (both inbound and outbound).  Requires ≥ ``MIN_HOURLY`` (30)
            total hourly counts.  Uses a 72-hour staleness TTL so it refreshes
            more frequently than slowly-shifting structural patterns.

            Example::

                "Your most active communication hours are 9:00, 10:00, 14:00."

        **4. Channel cadence comparison (``channel_cadence``):**
            Compares average response times across communication channels to surface
            the user's fastest and slowest-responding channel.  Only fires when at
            least two channels each have ≥ ``MIN_CT_SAMPLES`` (3) response-time
            samples and the fastest is at least 2× faster than the slowest.

            Example::

                "You respond fastest on imessage (18m avg) and slowest on
                proton_mail (4.1h avg). 2 channels compared."

        **Data requirements:**
            All sub-types require the ``cadence`` profile to exist.  Each sub-type
            has its own minimum sample threshold to prevent noise from sparse data.

        **Staleness TTL:**
            ``response_time_baseline``:   168 hours (7 days) — shifts slowly.
            ``fastest_contacts``:         168 hours (7 days) — priority signals are stable.
            ``communication_peak_hours``:  72 hours (3 days) — refreshes more often.
            ``channel_cadence``:          168 hours (7 days) — channel habits are stable.

        **Dedup strategy:**
            ``response_time_baseline`` uses the fixed entity ``"global_avg"`` so only
            one baseline insight exists at a time.  ``fastest_contacts`` uses the
            contact address as entity.  ``communication_peak_hours`` uses the
            concatenated top-hour strings so it re-surfaces when peak hours shift.
            ``channel_cadence`` uses ``"fastest:slowest"`` channel names as entity.

        **Marketing filter:**
            ``fastest_contacts`` excludes automated senders, mailing lists, and
            no-reply addresses via the shared ``is_marketing_or_noreply()`` helper.

        Returns:
            list[Insight]: Zero to four insights depending on data availability
            and pattern strength.
        """
        # Minimum global response-time observations before surfacing baseline.
        MIN_RT_SAMPLES = 10
        # Minimum per-contact or per-channel samples for per-entity insights.
        MIN_CT_SAMPLES = 3
        # Fast-reply threshold: contact avg < this fraction of global avg.
        FAST_RATIO = 0.5
        # Minimum total hourly-activity counts for peak-hours insight.
        MIN_HOURLY = 30
        # Maximum priority-contact (fastest_contacts) insights to surface at once.
        MAX_CONTACTS = 3

        profile = self.ums.get_signal_profile("cadence")
        if not profile:
            return []

        data = profile.get("data", {})
        if not data:
            return []

        global_rts: list[float] = data.get("response_times", [])
        per_contact: dict[str, list] = data.get("per_contact_response_times", {})
        per_channel: dict[str, list] = data.get("per_channel_response_times", {})
        hourly: dict[str, int] = data.get("hourly_activity", {})

        insights: list[Insight] = []

        # Load email → display-name map once; used in Sub-insight 2 to replace
        # raw email addresses with human-readable contact names in summaries.
        name_map = self._load_contact_name_map()

        # ----------------------------------------------------------------
        # Sub-insight 1: Response time baseline
        # ----------------------------------------------------------------
        if len(global_rts) >= MIN_RT_SAMPLES:
            avg_seconds = sum(global_rts) / len(global_rts)
            avg_hours = avg_seconds / 3600.0
            n = len(global_rts)

            # Format a human-readable duration: show minutes for sub-hour averages.
            if avg_hours < 1.0:
                duration_str = f"{int(avg_seconds / 60)} minutes"
            else:
                duration_str = f"{avg_hours:.1f} hours"

            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your average reply time across all contacts is {duration_str}. "
                    f"Based on {n:,} response-time observations."
                ),
                confidence=min(0.85, 0.50 + n * 0.003),
                evidence=[
                    f"avg_response_hours={avg_hours:.2f}",
                    f"n_samples={n}",
                ],
                category="response_time_baseline",
                entity="global_avg",
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 2: Priority-contact fast replies
        # ----------------------------------------------------------------
        if global_rts and per_contact:
            global_avg = sum(global_rts) / len(global_rts)

            # Collect contacts that reply faster than FAST_RATIO × global avg.
            fast_candidates: list[tuple[str, float, int]] = []
            for contact_id, rts in per_contact.items():
                if len(rts) < MIN_CT_SAMPLES:
                    continue
                # Exclude automated senders — only human contacts matter here.
                if is_marketing_or_noreply(contact_id):
                    continue
                avg_ct = sum(rts) / len(rts)
                if global_avg > 0 and (avg_ct / global_avg) < FAST_RATIO:
                    fast_candidates.append((contact_id, avg_ct, len(rts)))

            # Surface at most MAX_CONTACTS insights, ordered fastest-first.
            fast_candidates.sort(key=lambda x: x[1])
            for contact_id, avg_ct, n_rts in fast_candidates[:MAX_CONTACTS]:
                ct_hours = avg_ct / 3600.0
                global_hours = global_avg / 3600.0

                if ct_hours < 1.0:
                    ct_str = f"{int(avg_ct / 60)}m"
                else:
                    ct_str = f"{ct_hours:.1f}h"

                # Resolve the email to a display name when a contact record
                # exists; fall back to the raw address so the insight remains
                # actionable in sparse-contacts deployments.
                label = self._display_name(contact_id, name_map)

                insight = Insight(
                    type="behavioral_pattern",
                    summary=(
                        f"You reply to {label} notably faster than average "
                        f"({ct_str} vs {global_hours:.1f}h overall) — a strong "
                        f"priority signal."
                    ),
                    confidence=min(0.80, 0.45 + n_rts * 0.03),
                    evidence=[
                        f"contact={contact_id}",
                        f"avg_response_hours={ct_hours:.2f}",
                        f"global_avg_hours={global_hours:.2f}",
                        f"n_samples={n_rts}",
                    ],
                    category="fastest_contacts",
                    # Preserve the email as the entity so the dedup key is
                    # stable even if the contact name changes.
                    entity=contact_id,
                    staleness_ttl_hours=168,
                )
                insight.compute_dedup_key()
                insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 3: Peak communication hours
        # ----------------------------------------------------------------
        if hourly:
            total_counts = sum(hourly.values())
            if total_counts >= MIN_HOURLY:
                sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_hours[:3]
                if top_3:
                    hour_labels = [f"{int(h)}:00" for h, _ in top_3]
                    # Entity fingerprint changes when the dominant hours shift.
                    entity_key = "_".join(h for h, _ in top_3)

                    insight = Insight(
                        type="behavioral_pattern",
                        summary=(
                            f"Your most active communication hours are "
                            f"{', '.join(hour_labels)}."
                        ),
                        confidence=min(0.85, 0.50 + total_counts * 0.002),
                        evidence=[
                            f"top_hours={','.join(h for h, _ in top_3)}",
                            f"total_activity_events={total_counts}",
                        ],
                        category="communication_peak_hours",
                        entity=entity_key,
                        staleness_ttl_hours=72,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 4: Channel cadence comparison
        # ----------------------------------------------------------------
        if per_channel:
            # Build a list of (channel, avg_seconds, n_samples) for channels
            # with enough data to be statistically meaningful.
            channel_avgs: list[tuple[str, float, int]] = []
            for channel, rts in per_channel.items():
                if len(rts) >= MIN_CT_SAMPLES:
                    channel_avgs.append((channel, sum(rts) / len(rts), len(rts)))

            if len(channel_avgs) >= 2:
                channel_avgs.sort(key=lambda x: x[1])
                fastest_ch, fastest_avg, fastest_n = channel_avgs[0]
                slowest_ch, slowest_avg, slowest_n = channel_avgs[-1]

                # Only surface when the gap is substantial (fastest < 50% of slowest).
                if slowest_avg > 0 and (fastest_avg / slowest_avg) < 0.5:
                    fastest_hours = fastest_avg / 3600.0
                    slowest_hours = slowest_avg / 3600.0

                    if fastest_hours < 1.0:
                        fastest_str = f"{int(fastest_avg / 60)}m"
                    else:
                        fastest_str = f"{fastest_hours:.1f}h"

                    # Entity encodes both channel names so the dedup key changes
                    # only when the relative ordering of fastest/slowest shifts.
                    entity_key = f"{fastest_ch}:{slowest_ch}"

                    insight = Insight(
                        type="behavioral_pattern",
                        summary=(
                            f"You respond fastest on {fastest_ch} ({fastest_str} avg) "
                            f"and slowest on {slowest_ch} ({slowest_hours:.1f}h avg). "
                            f"{len(channel_avgs)} channels compared."
                        ),
                        confidence=min(0.75, 0.40 + (fastest_n + slowest_n) * 0.02),
                        evidence=[
                            f"fastest_channel={fastest_ch}",
                            f"fastest_avg_hours={fastest_hours:.2f}",
                            f"slowest_channel={slowest_ch}",
                            f"slowest_avg_hours={slowest_hours:.2f}",
                            f"channels_compared={len(channel_avgs)}",
                        ],
                        category="channel_cadence",
                        entity=entity_key,
                        staleness_ttl_hours=168,
                    )
                    insight.compute_dedup_key()
                    insights.append(insight)

        logger.debug(
            "cadence_response_insights: %d insights generated "
            "(global_rt_samples=%d, per_contact=%d, hourly_total=%d)",
            len(insights),
            len(global_rts),
            len(per_contact),
            sum(hourly.values()) if hourly else 0,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Routine Patterns (Layer 3 Procedural Memory)
    # ------------------------------------------------------------------

    def _routine_insights(self) -> list[Insight]:
        """Surface high-consistency routines detected in procedural memory.

        Reads the ``routines`` table (populated by ``RoutineDetector``) and
        generates one insight per qualifying routine.  A routine qualifies when
        it has been observed at least ``MIN_OBSERVATIONS`` times **and** its
        ``consistency_score`` meets or exceeds ``MIN_CONSISTENCY``.  Low-quality
        routines (rare or highly variable) are silently skipped to keep the
        insight surface actionable.

        Insight confidence scales linearly with consistency:
            ``confidence = min(0.85, 0.50 + consistency_score * 0.40)``

        This gives a 0.70 baseline for a perfectly consistent routine and
        rewards stronger patterns, capping at 0.85 to leave headroom for
        source-weight modulation.

        **Dedup strategy:**
            Each routine uses its name as the entity, so the dedup key is stable
            across runs.  Staleness TTL is 7 days (168 h) — routines shift
            slowly and re-surfacing weekly is appropriate.

        **Trigger labels:**
            - ``morning``         → "morning routine"
            - ``midday``          → "midday routine"
            - ``afternoon``       → "afternoon routine"
            - ``evening``         → "evening routine"
            - ``night``           → "night routine"
            - ``arrive_*``        → "arrival routine at <location>"
            - ``after_*``         → "post-<event> routine"
            - anything else       → the trigger string verbatim

        Returns:
            list[Insight]: One ``behavioral_pattern`` insight per qualifying
            routine, ordered by consistency_score descending (highest first).

        Example output::

            "You have a consistent morning routine (5 steps, seen 14 times,
             87% consistency, ~28 min)."
        """
        # Minimum times a routine must be observed before surfacing it.
        MIN_OBSERVATIONS = 3
        # Minimum consistency fraction (0–1) required to surface the insight.
        MIN_CONSISTENCY = 0.70

        try:
            routines = self.ums.get_routines()
        except Exception:
            logger.exception("routine_insights: failed to fetch routines")
            return []

        if not routines:
            logger.debug("routine_insights: no routines stored — skipping")
            return []

        insights: list[Insight] = []

        for routine in routines:
            consistency = routine.get("consistency_score", 0.0)
            observed = routine.get("times_observed", 0)
            name = routine.get("name", "")
            trigger = routine.get("trigger", "")
            steps = routine.get("steps", [])
            duration_min = routine.get("typical_duration_minutes", 0.0)

            if not name or observed < MIN_OBSERVATIONS or consistency < MIN_CONSISTENCY:
                continue

            # Build a human-readable trigger label.
            trigger_label = _routine_trigger_label(trigger)

            # Round duration to the nearest minute for readability.
            duration_str = f", ~{round(duration_min)} min" if duration_min > 0 else ""
            steps_str = f"{len(steps)} steps" if steps else "multiple steps"
            pct = round(consistency * 100)

            summary = (
                f"You have a consistent {trigger_label} "
                f"({steps_str}, seen {observed} times, "
                f"{pct}% consistency{duration_str})."
            )

            confidence = min(0.85, 0.50 + consistency * 0.40)

            insight = Insight(
                type="behavioral_pattern",
                summary=summary,
                confidence=confidence,
                evidence=[
                    f"routine_name={name}",
                    f"trigger={trigger}",
                    f"consistency_score={consistency:.2f}",
                    f"times_observed={observed}",
                    f"steps_count={len(steps)}",
                    f"typical_duration_min={round(duration_min, 1)}",
                ],
                category="routine_pattern",
                entity=name,
                staleness_ttl_hours=168,  # 7 days — routines shift slowly
            )
            insight.compute_dedup_key()
            insights.append(insight)

        logger.debug(
            "routine_insights: %d insights generated "
            "(total_routines=%d, qualifying=%d, min_observations=%d, min_consistency=%.2f)",
            len(insights),
            len(routines),
            len(insights),
            MIN_OBSERVATIONS,
            MIN_CONSISTENCY,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Spatial Patterns (Location Behavioral Profile)
    # ------------------------------------------------------------------

    def _spatial_insights(self) -> list[Insight]:
        """Surface location-behavioral patterns from the spatial signal profile.

        Reads the ``spatial`` signal profile (built by ``SpatialExtractor``) and
        translates per-place visit counts, domain distributions, and durations
        into up to three human-readable insight sub-types.

        The spatial profile accumulates signals from calendar events with location
        fields, iOS context updates, and explicit location updates.  It stores a
        ``place_behaviors`` dict keyed by normalized location name, with visit
        counts, domain tallies, and average durations per place.

        **1. Most-frequented location (``spatial_top_location``):**
            Surfaces the single place with the highest overall visit count.
            Includes average time-at-location when duration data is available.
            Requires at least ``MIN_VISITS`` (3) visits for the top place.

            Example::

                "Your most-visited location is 'conference room' (18 visits, avg 55 min)."

        **2. Primary work location (``spatial_work_location``):**
            Finds the location with the most work-domain events (as recorded in
            ``domain_counts["work"]``).  Only surfaces when at least one location
            has >= ``MIN_WORK_VISITS`` (3) work-tagged events.  Detects the
            home-office pattern when the normalized location name contains "home".

            Example::

                "You primarily work from home (23 work events recorded at 'home')."
                "Your most frequent work location is 'office' (15 work events recorded)."

        **3. Location diversity (``spatial_location_diversity``):**
            Counts distinct locations with >= ``MIN_VISITS`` (3) visits and breaks
            them down by dominant domain (work vs personal).  Only surfaces when
            the user has >= 2 distinct frequent locations.

            Example::

                "You frequent 5 distinct locations: 3 work-related, 2 personal
                (based on 678 location observations)."

        **Staleness TTL:**
            168 hours (7 days) for all sub-types.  Location patterns shift slowly
            and weekly refresh is appropriate.

        **Dedup strategy:**
            Category + entity encode the specific pattern so the insight refreshes
            only when the dominant place, work location, or split actually changes,
            not just because a new calendar event arrived.

        Returns:
            list[Insight]: Zero to three location-behavioral insights.
        """
        # Minimum visits for a place to be considered "frequent enough" to surface.
        MIN_VISITS = 3
        # Minimum work-tagged visits before surfacing a primary work location insight.
        MIN_WORK_VISITS = 3

        profile = self.ums.get_signal_profile("spatial")
        if not profile:
            return []

        data = profile.get("data", {})
        total_samples = profile.get("samples_count", 0)

        # Spatial profile stores place_behaviors as a JSON-encoded string inside
        # the "data" blob (see SpatialExtractor._update_spatial_profile).
        # Tolerate both pre-serialized string and a native dict (e.g. in tests).
        place_behaviors_raw = data.get("place_behaviors", {})
        if isinstance(place_behaviors_raw, str):
            try:
                place_behaviors = json.loads(place_behaviors_raw)
            except (json.JSONDecodeError, ValueError):
                logger.warning("spatial_insights: could not parse place_behaviors JSON")
                return []
        else:
            place_behaviors = place_behaviors_raw if place_behaviors_raw else {}

        if not place_behaviors:
            logger.debug("spatial_insights: place_behaviors is empty — skipping")
            return []

        insights: list[Insight] = []

        # ----------------------------------------------------------------
        # Sub-insight 1: Most-frequented location
        # ----------------------------------------------------------------
        # Find the single place with the highest recorded visit count.
        top_name, top_data = max(
            place_behaviors.items(),
            key=lambda x: x[1].get("visit_count", 0),
        )
        top_visits = top_data.get("visit_count", 0)

        if top_visits >= MIN_VISITS:
            avg_dur = top_data.get("average_duration_minutes")
            # Include average duration when the spatial extractor captured event times.
            dur_str = f", avg {round(avg_dur)} min" if avg_dur and avg_dur > 0 else ""
            # Truncate very long normalized location strings for readability.
            display_name = top_name if len(top_name) <= 40 else top_name[:37] + "…"

            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"Your most-visited location is '{display_name}' "
                    f"({top_visits} visits{dur_str})."
                ),
                confidence=min(0.85, 0.45 + top_visits * 0.01),
                evidence=[
                    f"top_location={top_name}",
                    f"visit_count={top_visits}",
                    f"avg_duration_min={round(avg_dur, 1) if avg_dur else 'n/a'}",
                    f"total_locations={len(place_behaviors)}",
                ],
                category="spatial_top_location",
                # Entity is the location name — dedup key changes only when
                # a different place becomes the most-visited location.
                entity=top_name[:80],
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 2: Primary work location
        # ----------------------------------------------------------------
        # Identify the place with the highest count of work-domain events.
        best_work_name: str | None = None
        best_work_count = 0

        for loc_name, loc_data in place_behaviors.items():
            dc = loc_data.get("domain_counts", {})
            work_count = dc.get("work", 0)
            if work_count > best_work_count:
                best_work_count = work_count
                best_work_name = loc_name

        if best_work_name and best_work_count >= MIN_WORK_VISITS:
            display_work = (
                best_work_name if len(best_work_name) <= 40 else best_work_name[:37] + "…"
            )

            # Detect home-office pattern: location name contains "home" or common
            # residential keywords in the normalized string.
            home_keywords = {"home", "house", "apartment", "residence", "flat"}
            is_home_office = (
                "home" in best_work_name.lower()
                or best_work_name.lower() in home_keywords
            )

            if is_home_office:
                summary = (
                    f"You primarily work from home "
                    f"({best_work_count} work events recorded at '{display_work}')."
                )
            else:
                summary = (
                    f"Your most frequent work location is '{display_work}' "
                    f"({best_work_count} work events recorded)."
                )

            insight = Insight(
                type="behavioral_pattern",
                summary=summary,
                confidence=min(0.80, 0.40 + best_work_count * 0.02),
                evidence=[
                    f"work_location={best_work_name}",
                    f"work_visit_count={best_work_count}",
                    f"is_home_office={is_home_office}",
                ],
                category="spatial_work_location",
                # Entity anchored to the location name; changes if a new work
                # location accumulates more events than the current leader.
                entity=best_work_name[:80],
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # ----------------------------------------------------------------
        # Sub-insight 3: Location diversity (work vs personal split)
        # ----------------------------------------------------------------
        # Count places the user visits frequently and split by dominant domain.
        frequent_places = [
            (name, d)
            for name, d in place_behaviors.items()
            if d.get("visit_count", 0) >= MIN_VISITS
        ]

        # Only surface when there are at least 2 frequent locations — a single
        # place provides no meaningful "diversity" comparison.
        if len(frequent_places) >= 2:
            work_count = sum(
                1 for _, d in frequent_places if d.get("dominant_domain") == "work"
            )
            personal_count = len(frequent_places) - work_count
            n_total = len(frequent_places)

            # Entity encodes the distribution; refreshes when the split changes.
            entity_key = f"total{n_total}_work{work_count}_personal{personal_count}"

            insight = Insight(
                type="behavioral_pattern",
                summary=(
                    f"You frequent {n_total} distinct locations: "
                    f"{work_count} work-related, {personal_count} personal "
                    f"(based on {total_samples} location observations)."
                ),
                confidence=min(0.80, 0.40 + total_samples * 0.001),
                evidence=[
                    f"frequent_location_count={n_total}",
                    f"work_locations={work_count}",
                    f"personal_locations={personal_count}",
                    f"total_samples={total_samples}",
                ],
                category="spatial_location_diversity",
                entity=entity_key,
                staleness_ttl_hours=168,
            )
            insight.compute_dedup_key()
            insights.append(insight)

        logger.debug(
            "spatial_insights: %d insights generated "
            "(total_locations=%d, frequent_locations=%d, total_samples=%d)",
            len(insights),
            len(place_behaviors),
            len([d for d in place_behaviors.values() if d.get("visit_count", 0) >= MIN_VISITS]),
            total_samples,
        )
        return insights

    # ------------------------------------------------------------------
    # Correlator: Workflow Patterns (Procedural Memory — Layer 3)
    # ------------------------------------------------------------------

    def _workflow_pattern_insights(self) -> list[Insight]:
        """Surface goal-driven workflow patterns detected by WorkflowDetector.

        Reads the ``workflows`` table (populated by ``WorkflowDetector``) and
        generates one insight per qualifying workflow.  A workflow qualifies when
        it has been observed at least ``MIN_OBSERVATIONS`` times **and** its
        ``success_rate`` meets ``MIN_SUCCESS_RATE``.

        Three workflow sub-types are surfaced with tailored summaries:

        **Email workflows** (name starts with "Responding to"):
            "You consistently respond to emails from <sender>: seen N times,
             X% reply rate."  Marketing/automated senders are filtered out
             using the shared ``is_marketing_or_noreply()`` helper so the
             correlator only surfaces genuine human communication patterns.

        **Task workflows** (name contains "Task completion"):
            "Your task management workflow spans N steps across M tools,
             with X% completion rate (seen K times)."  Surfaces the user's
             productivity pattern around task handling.

        **Calendar workflows** (name contains "Calendar"):
            "Your meetings consistently trigger prep/follow-up actions:
             N calendar events observed, with X% follow-up rate."

        **Interaction workflows** (any other trigger-based workflow):
            "You have a recurring <trigger> workflow pattern:
             N steps, seen K times, X% success rate."

        Confidence formula::

            confidence = min(0.85, 0.50
                            + min(observations, 50) / 50 * 0.20
                            + success_rate * 0.15)

        - 0.50 base  — all qualifying workflows carry at least moderate signal
        - 0.20 bonus — scales linearly with observation count, capping at 50 obs
        - 0.15 bonus — scales with success_rate (0=no bonus, 1=full bonus)
        - 0.85 cap   — leaves headroom for source-weight modulation

        **Dedup strategy:**
            Entity encodes the workflow name so the dedup key is stable across
            runs.  Staleness TTL is 7 days (168 h) — workflows are re-detected
            weekly by the WorkflowDetector.

        Returns:
            list[Insight]: One ``behavioral_pattern`` insight per qualifying
            workflow, ordered by times_observed descending.

        Example outputs::

            "You consistently respond to emails from alice@work.com
             (seen 47 times, 92% reply rate)."

            "Your task management workflow spans 4 steps across email + calendar
             (seen 7,232 times, 25% completion rate)."

            "Your meetings consistently trigger email follow-ups
             (seen 2,638 calendar events, 68% follow-up rate)."
        """
        # Minimum times a workflow must be observed before surfacing it.
        # Set to 3 to match WorkflowDetector.min_occurrences — if the detector
        # stored it, it already met this bar, but guard defensively.
        MIN_OBSERVATIONS = 3
        # Minimum success rate: 1% (same as WorkflowDetector.success_threshold).
        # We use 1% because email workflows are inherently low-rate (most emails
        # don't get replies), and the interesting signal is the *pattern*, not
        # the frequency.
        MIN_SUCCESS_RATE = 0.01

        try:
            workflows = self.ums.get_workflows()
        except Exception:
            logger.exception("workflow_pattern_insights: failed to fetch workflows")
            return []

        if not workflows:
            logger.debug("workflow_pattern_insights: no workflows stored — skipping")
            return []

        insights: list[Insight] = []

        for workflow in workflows:
            name: str = workflow.get("name", "")
            observed: int = workflow.get("times_observed", 0)
            success_rate: float = workflow.get("success_rate", 0.0)
            steps: list = workflow.get("steps", [])
            tools: list = workflow.get("tools_used", [])
            duration_min: float = workflow.get("typical_duration_minutes") or 0.0

            if not name or observed < MIN_OBSERVATIONS or success_rate < MIN_SUCCESS_RATE:
                continue

            # --- Build summary and classify workflow type ---
            if name.startswith("Responding to "):
                # Extract the sender address from the workflow name.
                # WorkflowDetector formats these as "Responding to <addr>".
                sender = name[len("Responding to "):]

                # Skip automated/marketing senders — these workflows are
                # noise (e.g., "Responding to newsletter@company.com" with
                # success_rate=0.001 because the user never replies).
                if is_marketing_or_noreply(sender, {}):
                    logger.debug(
                        "workflow_pattern_insights: skipping marketing workflow '%s'", name
                    )
                    continue

                reply_pct = round(success_rate * 100)
                summary = (
                    f"You consistently respond to emails from {sender} "
                    f"(seen {observed} times, {reply_pct}% reply rate)."
                )
                category = "workflow_pattern_email"

            elif "Task completion" in name or "task" in name.lower():
                step_count = len(steps)
                tools_str = " + ".join(tools[:3]) if tools else "multiple tools"
                completion_pct = round(success_rate * 100)
                summary = (
                    f"Your task management workflow spans {step_count} steps "
                    f"across {tools_str} "
                    f"(seen {observed} times, {completion_pct}% completion rate)."
                )
                category = "workflow_pattern_task"

            elif "Calendar" in name or "calendar" in name.lower():
                followup_pct = round(success_rate * 100)
                summary = (
                    f"Your meetings consistently trigger prep or follow-up actions "
                    f"(seen {observed} calendar events, {followup_pct}% follow-up rate)."
                )
                category = "workflow_pattern_calendar"

            else:
                # Generic interaction-based workflow (e.g., "Email Received Workflow")
                step_count = len(steps)
                success_pct = round(success_rate * 100)
                duration_str = f", ~{round(duration_min)} min avg" if duration_min > 0 else ""
                summary = (
                    f"You have a recurring workflow: {name} "
                    f"({step_count} steps, seen {observed} times, "
                    f"{success_pct}% success rate{duration_str})."
                )
                category = "workflow_pattern_interaction"

            # Confidence formula: base 0.50, up to 0.20 bonus from observations
            # (capped at 50 to avoid over-confidence on high-volume marketing),
            # up to 0.15 bonus from success_rate.
            confidence = min(
                0.85,
                0.50
                + min(observed, 50) / 50 * 0.20
                + success_rate * 0.15,
            )

            insight = Insight(
                type="behavioral_pattern",
                summary=summary,
                confidence=confidence,
                evidence=[
                    f"workflow_name={name}",
                    f"times_observed={observed}",
                    f"success_rate={success_rate:.3f}",
                    f"steps_count={len(steps)}",
                    f"tools_used={','.join(tools[:3])}",
                    f"typical_duration_min={round(duration_min, 1)}",
                ],
                category=category,
                # Entity is the workflow name — stable dedup key across runs.
                entity=name,
                staleness_ttl_hours=168,  # 7 days — matches WorkflowDetector cadence
            )
            insight.compute_dedup_key()
            insights.append(insight)

        # Sort descending by times_observed so the most-seen workflows
        # appear first in the briefing context.
        insights.sort(key=lambda i: -(i.evidence and
                                       int(next((e.split("=")[1] for e in i.evidence
                                                 if e.startswith("times_observed=")), "0"))))

        logger.debug(
            "workflow_pattern_insights: %d insights generated "
            "(total_workflows=%d, qualifying=%d)",
            len(insights),
            len(workflows),
            len(insights),
        )
        return insights

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self, insights: list[Insight]) -> list[Insight]:
        """Remove insights whose dedup_key already exists within staleness TTL.

        An insight is considered stale (and therefore skipped) if the insights
        table already contains a row with the same dedup_key whose created_at
        is within the effective TTL window.

        Feedback-aware TTL scaling:
        - dismissed / not_relevant: TTL * 4 (suppress longer to respect rejection)
        - useful: TTL * 0.5 (allow resurfacing sooner)
        - None or unknown: original TTL unchanged (fail-open)
        """
        fresh: list[Insight] = []
        now = datetime.now(timezone.utc)

        for insight in insights:
            if not insight.dedup_key:
                insight.compute_dedup_key()

            try:
                with self.db.get_connection("user_model") as conn:
                    row = conn.execute(
                        "SELECT created_at, feedback FROM insights WHERE dedup_key = ? ORDER BY created_at DESC LIMIT 1",
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

                # Apply feedback-aware TTL scaling
                feedback = row["feedback"] if "feedback" in row.keys() else None
                effective_ttl = insight.staleness_ttl_hours
                if feedback in ("dismissed", "not_relevant"):
                    effective_ttl = insight.staleness_ttl_hours * 4
                elif feedback == "useful":
                    effective_ttl = insight.staleness_ttl_hours * 0.5

                if age_hours >= effective_ttl:
                    fresh.append(insight)
                else:
                    if feedback in ("dismissed", "not_relevant"):
                        logger.debug(
                            "Suppressing insight (dedup_key=%s) — user feedback '%s' extends TTL to %dh (age=%dh)",
                            insight.dedup_key,
                            feedback,
                            effective_ttl,
                            int(age_hours),
                        )
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
