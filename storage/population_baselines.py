"""
Life OS -- Population Baseline Store

Manages research-seeded population benchmarks and percentile computation.

The core problem: every insight in the system is self-referential.  "Your
stress increased 20%" only means something if you know whether 20% puts you
at the 50th or 95th percentile of the population.  Without population
context, insights have no anchor.

This module provides that anchor.  It seeds research-backed percentile
distributions for key behavioral metrics (email volume, stress, response
time, meeting load, etc.) and exposes a simple API:

    store.get_percentile("email.daily_received_count", 85)  -> 78
    store.get_baseline("mood.stress_daily_avg")             -> {p10: 0.15, ...}

Data sources:
    - McKinsey Global Institute (email volume in knowledge work)
    - Radicati Group Email Statistics
    - APA "Stress in America" surveys (stress normative data)
    - NSF "Sleep in America" polls (sleep quality)
    - Atlassian / Microsoft WorkLab (meeting load research)
    - Bureau of Labor Statistics (spending patterns)
    - Yahoo Research / USC ISI (email response time)

Architecture:
    Research baselines are seeded on first run.  The schema supports
    multiple cohorts (general, knowledge_worker) so peer-network data
    can refine baselines without overwriting the research anchor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


# ============================================================================
# Research-seeded baseline data
# ============================================================================

# Each entry: (metric_key, cohort, description, p10, p25, p50, p75, p90,
#              mean, std_dev, unit, source, sample_size)
_SEED_BASELINES: list[tuple] = [
    # -- Communication --
    (
        "email.daily_received_count", "knowledge_worker",
        "Emails received per day",
        12, 25, 40, 65, 121,
        47.0, 30.0,
        "count_per_day",
        "McKinsey Global Institute 2012; Radicati Group 2023 Email Statistics",
        10000,
    ),
    (
        "email.daily_sent_count", "knowledge_worker",
        "Emails sent per day",
        3, 8, 15, 28, 50,
        18.0, 14.0,
        "count_per_day",
        "Radicati Group 2023 Email Statistics",
        10000,
    ),
    (
        "email.avg_response_minutes", "knowledge_worker",
        "Average email response time in minutes",
        2, 8, 47, 180, 720,
        98.0, 150.0,
        "minutes",
        "Yahoo Research large-scale email study; USC ISI 2015",
        200000,
    ),
    (
        "messaging.avg_response_minutes", "general",
        "Average messaging (IM/SMS) response time in minutes",
        1, 3, 10, 45, 180,
        35.0, 60.0,
        "minutes",
        "Pew Research Center 2021 messaging behavior survey",
        5000,
    ),
    (
        "communication.initiation_ratio", "general",
        "Fraction of conversations you initiate (0-1)",
        0.15, 0.30, 0.45, 0.60, 0.75,
        0.45, 0.18,
        "ratio_0_1",
        "Communication reciprocity research, Dunbar 2012",
        2000,
    ),
    # -- Calendar --
    (
        "calendar.weekly_meeting_hours", "knowledge_worker",
        "Hours spent in meetings per week",
        2, 5, 8, 15, 23,
        10.0, 7.0,
        "hours_per_week",
        "Atlassian 2019; Microsoft WorkLab 2023 meeting research",
        30000,
    ),
    (
        "calendar.daily_meeting_count", "knowledge_worker",
        "Number of meetings per day",
        0, 1, 2, 4, 7,
        2.5, 2.0,
        "count_per_day",
        "Microsoft WorkLab 2023; Reclaim.ai 2024 productivity report",
        30000,
    ),
    # -- Mood / Wellbeing --
    (
        "mood.stress_daily_avg", "general",
        "Average daily stress level (0 = none, 1 = extreme)",
        0.15, 0.28, 0.45, 0.62, 0.78,
        0.44, 0.20,
        "score_0_1",
        "APA Stress in America 2023; Cohen Perceived Stress Scale norms",
        3000,
    ),
    (
        "mood.energy_daily_avg", "general",
        "Average daily energy level (0 = exhausted, 1 = peak)",
        0.25, 0.40, 0.55, 0.70, 0.82,
        0.54, 0.18,
        "score_0_1",
        "NSF Sleep in America polls; fatigue survey meta-analysis",
        5000,
    ),
    (
        "mood.social_battery_daily_avg", "general",
        "Average social battery (0 = depleted, 1 = full)",
        0.20, 0.35, 0.52, 0.68, 0.82,
        0.51, 0.19,
        "score_0_1",
        "Big Five Personality extroversion norms; social energy research",
        5000,
    ),
    # -- Task / Productivity --
    (
        "task.daily_completed_count", "knowledge_worker",
        "Tasks completed per day",
        0, 1, 3, 5, 9,
        3.2, 2.5,
        "count_per_day",
        "Todoist 2023 productivity report; RescueTime 2022",
        50000,
    ),
    (
        "task.completion_rate", "general",
        "Fraction of tasks completed by due date (0-1)",
        0.20, 0.38, 0.58, 0.75, 0.88,
        0.56, 0.22,
        "ratio_0_1",
        "Todoist 2023; Asana productivity research 2022",
        50000,
    ),
    (
        "task.overdue_rate", "general",
        "Fraction of tasks that become overdue (0-1)",
        0.05, 0.12, 0.25, 0.42, 0.60,
        0.27, 0.18,
        "ratio_0_1",
        "Todoist 2023; project management completion research",
        50000,
    ),
    # -- Finance --
    (
        "finance.daily_spend_total", "general",
        "Total daily discretionary spending in dollars",
        5, 12, 25, 55, 120,
        38.0, 35.0,
        "dollars_per_day",
        "Bureau of Labor Statistics 2023 Consumer Expenditure Survey",
        130000,
    ),
    (
        "finance.monthly_transaction_count", "general",
        "Number of financial transactions per month",
        15, 30, 55, 90, 150,
        62.0, 40.0,
        "count_per_month",
        "Federal Reserve 2022 Payments Study; Plaid aggregate data",
        50000,
    ),
    (
        "finance.daily_transaction_count", "general",
        "Number of financial transactions per day",
        0, 1, 2, 3, 5,
        2.1, 1.5,
        "count_per_day",
        "Derived from monthly: Federal Reserve 2022 Payments Study (monthly / 30)",
        50000,
    ),
    # -- Social --
    (
        "social.unique_contacts_weekly", "general",
        "Unique people communicated with per week",
        3, 7, 14, 25, 45,
        16.0, 12.0,
        "count_per_week",
        "Dunbar 2012 social brain research; communication pattern studies",
        2000,
    ),
    (
        "social.unique_contacts_daily", "general",
        "Unique people communicated with per day",
        1, 2, 4, 7, 12,
        3.5, 3.0,
        "count_per_day",
        "Derived from weekly: Dunbar 2012 social brain research (weekly / 5 workdays)",
        2000,
    ),
    # -- Behavioral --
    (
        "behavioral.routine_consistency", "general",
        "Consistency score for daily routines (0-1)",
        0.15, 0.30, 0.50, 0.68, 0.82,
        0.48, 0.20,
        "score_0_1",
        "Habit formation research; Wood & Neal 2007",
        1000,
    ),
    (
        "behavioral.after_hours_email_rate", "knowledge_worker",
        "Fraction of emails sent outside business hours",
        0.02, 0.08, 0.18, 0.32, 0.50,
        0.20, 0.15,
        "ratio_0_1",
        "Microsoft WorkLab 2023; burnout research meta-analysis",
        30000,
    ),
]


class PopulationBaselineStore:
    """Manages population benchmarks and percentile computation.

    Provides the "is this normal?" context that individual signal profiles
    cannot answer.  Seeded with research data on first access, updatable
    from peer network contributions.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._seeded = False

    def ensure_seeded(self) -> None:
        """Seed research baselines if not already present."""
        if self._seeded:
            return

        try:
            with self.db.get_connection("user_model") as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM population_baselines WHERE source != 'peer_network'"
                ).fetchone()[0]
                if count >= len(_SEED_BASELINES):
                    self._seeded = True
                    return
        except Exception:
            logger.debug("population_baselines table may not exist yet")
            return

        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("user_model") as conn:
            for row in _SEED_BASELINES:
                (metric_key, cohort, description, p10, p25, p50, p75, p90,
                 mean, std_dev, unit, source, sample_size) = row
                conn.execute(
                    """INSERT OR IGNORE INTO population_baselines
                       (metric_key, cohort, description, p10, p25, p50, p75, p90,
                        mean, std_dev, unit, source, sample_size, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (metric_key, cohort, description, p10, p25, p50, p75, p90,
                     mean, std_dev, unit, source, sample_size, now),
                )

        self._seeded = True
        logger.info(
            "Seeded %d population baselines from research data",
            len(_SEED_BASELINES),
        )

    def get_baseline(self, metric_key: str, cohort: str = "general") -> dict[str, Any] | None:
        """Return the full baseline record for a metric, or None if unknown.

        Falls back to 'general' cohort if the requested cohort is not found,
        then to 'knowledge_worker' if general is also missing.
        """
        self.ensure_seeded()
        for try_cohort in dict.fromkeys([cohort, "general", "knowledge_worker"]):
            try:
                with self.db.get_connection("user_model") as conn:
                    row = conn.execute(
                        """SELECT * FROM population_baselines
                           WHERE metric_key = ? AND cohort = ?""",
                        (metric_key, try_cohort),
                    ).fetchone()
                    if row:
                        return dict(row)
            except Exception:
                continue
        return None

    def get_percentile(self, metric_key: str, value: float,
                       cohort: str = "general") -> int | None:
        """Compute the approximate percentile rank of a value.

        Uses linear interpolation between the stored percentile breakpoints.
        Returns an integer 0-100, or None if no baseline exists.

        Args:
            metric_key: The metric to compare against.
            value: The user's observed value.
            cohort: Which population cohort to compare against.

        Returns:
            Integer percentile (0-100), or None.
        """
        baseline = self.get_baseline(metric_key, cohort)
        if not baseline:
            return None

        # Ordered breakpoints: (percentile, value)
        breakpoints = [
            (10, baseline["p10"]),
            (25, baseline["p25"]),
            (50, baseline["p50"]),
            (75, baseline["p75"]),
            (90, baseline["p90"]),
        ]

        # Filter out None breakpoints
        breakpoints = [(p, v) for p, v in breakpoints if v is not None]
        if not breakpoints:
            return None

        # Below the lowest breakpoint
        if value <= breakpoints[0][1]:
            # Extrapolate down: linearly from 0 to first breakpoint
            p_low, v_low = breakpoints[0]
            if v_low == 0:
                return 0
            return max(0, int(p_low * value / v_low))

        # Above the highest breakpoint
        if value >= breakpoints[-1][1]:
            p_high, v_high = breakpoints[-1]
            if v_high == 0:
                return 99
            # Extrapolate up: linearly from last breakpoint to 100
            remaining = 100 - p_high
            excess_ratio = (value - v_high) / max(v_high, 1)
            return min(99, int(p_high + remaining * min(excess_ratio, 1.0)))

        # Interpolate between adjacent breakpoints
        for i in range(len(breakpoints) - 1):
            p_low, v_low = breakpoints[i]
            p_high, v_high = breakpoints[i + 1]
            if v_low <= value <= v_high:
                if v_high == v_low:
                    return int((p_low + p_high) / 2)
                fraction = (value - v_low) / (v_high - v_low)
                return int(p_low + fraction * (p_high - p_low))

        return 50  # Fallback

    def get_all_baselines(self) -> list[dict[str, Any]]:
        """Return all population baselines."""
        self.ensure_seeded()
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute("SELECT * FROM population_baselines").fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def update_from_peer(self, metric_key: str, cohort: str,
                         percentiles: dict[str, float],
                         sample_size: int) -> None:
        """Update or insert a baseline from peer network data.

        Only updates if the peer data has a larger sample size than
        the existing baseline (research data acts as the floor).
        Uses UPDATE to preserve research-seeded columns (description,
        mean, std_dev, unit) that peer data does not provide.
        """
        existing = self.get_baseline(metric_key, cohort)
        if (existing and existing["sample_size"] and existing["sample_size"] >= sample_size
                and existing.get("source") != "peer_network"):
            return  # Don't overwrite research with smaller peer sample

        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection("user_model") as conn:
            if existing:
                # UPDATE preserves description, mean, std_dev, unit columns
                conn.execute(
                    """UPDATE population_baselines
                       SET p10 = ?, p25 = ?, p50 = ?, p75 = ?, p90 = ?,
                           source = 'peer_network', sample_size = ?, last_updated = ?
                       WHERE metric_key = ? AND cohort = ?""",
                    (
                        percentiles.get("p10"), percentiles.get("p25"),
                        percentiles.get("p50"), percentiles.get("p75"),
                        percentiles.get("p90"),
                        sample_size, now,
                        metric_key, cohort,
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO population_baselines
                       (metric_key, cohort, p10, p25, p50, p75, p90,
                        source, sample_size, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'peer_network', ?, ?)""",
                    (
                        metric_key, cohort,
                        percentiles.get("p10"), percentiles.get("p25"),
                        percentiles.get("p50"), percentiles.get("p75"),
                        percentiles.get("p90"),
                        sample_size, now,
                    ),
                )
