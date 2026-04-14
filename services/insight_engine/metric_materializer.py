"""
Life OS -- Metric Materializer

Pre-aggregates raw events into a queryable time-series and discovers
cross-metric correlations automatically.

The problem:
    Every correlator in the InsightEngine scans events.db independently,
    parsing JSON payloads and bucketing by timestamp.  With 800K+ events
    this is slow and wasteful.  Worse, cross-metric correlations (does
    stress correlate with spending?) require joining two independent scans
    which no correlator currently does.

The solution:
    MetricMaterializer runs periodically (after the insight loop) and:

    1. Aggregates raw events into daily metric rows in metric_time_series.
    2. Pulls mood_history, task, and signal profile data into the same
       time-series format.
    3. Computes pairwise Pearson correlations between all metric pairs
       and stores significant ones (|r| >= 0.4) in metric_correlations.

    Correlators then read from metric_time_series instead of raw events,
    and the new _cross_metric_correlation_insights() correlator reads
    from metric_correlations to surface discovered relationships.

Metric key taxonomy:
    email.daily_received_count      email.daily_sent_count
    email.avg_response_minutes      messaging.daily_received_count
    messaging.daily_sent_count      calendar.daily_meeting_count
    task.daily_completed_count      task.daily_created_count
    finance.daily_spend_total       finance.daily_transaction_count
    mood.stress_daily_avg           mood.energy_daily_avg
    mood.valence_daily_avg          mood.social_battery_daily_avg
    mood.cognitive_load_daily_avg   social.unique_contacts_daily
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


class MetricMaterializer:
    """Aggregates raw events into metric_time_series and computes correlations."""

    def __init__(self, db: DatabaseManager, lookback_days: int = 90):
        self.db = db
        self._lookback_days = lookback_days

    def materialize(self) -> int:
        """Aggregate recent events and mood data into metric_time_series rows.

        Returns the number of metric rows upserted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._lookback_days)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        total = 0

        total += self._materialize_event_counts(cutoff, now)
        total += self._materialize_mood_metrics(cutoff, now)
        total += self._materialize_task_metrics(cutoff, now)
        total += self._materialize_finance_metrics(cutoff, now)
        total += self._materialize_social_metrics(cutoff, now)

        if total:
            logger.info("MetricMaterializer: upserted %d time-series rows", total)
        return total

    def compute_correlations(self, min_samples: int = 14) -> int:
        """Compute pairwise Pearson correlations between all metric pairs.

        Only stores correlations where |r| >= 0.4 and sample_size >= min_samples.
        Returns the number of correlation rows stored.
        """
        # Load all daily metric time-series keyed by (metric_key, period)
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT metric_key, period, value
                       FROM metric_time_series
                       WHERE granularity = 'daily'
                       ORDER BY metric_key, period"""
                ).fetchall()
        except Exception:
            logger.debug("compute_correlations: could not query metric_time_series")
            return 0

        if not rows:
            return 0

        # Build dict: metric_key -> {period: value}
        series: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            series[row["metric_key"]][row["period"]] = row["value"]

        metric_keys = sorted(series.keys())
        now = datetime.now(timezone.utc).isoformat()
        stored = 0

        with self.db.get_connection("user_model") as conn:
            # Clean up stale correlations that are no longer recomputed
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            conn.execute(
                "DELETE FROM metric_correlations WHERE computed_at < ?",
                (stale_cutoff,),
            )
            for i, key_a in enumerate(metric_keys):
                for key_b in metric_keys[i + 1:]:
                    # Find overlapping periods
                    common_periods = sorted(
                        set(series[key_a].keys()) & set(series[key_b].keys())
                    )
                    if len(common_periods) < min_samples:
                        continue

                    xs = [series[key_a][p] for p in common_periods]
                    ys = [series[key_b][p] for p in common_periods]

                    r = _pearson_r(xs, ys)
                    if r is None or abs(r) < 0.4:
                        continue

                    n = len(common_periods)
                    p_value = _approximate_p_value(r, n)

                    conn.execute(
                        """INSERT OR REPLACE INTO metric_correlations
                           (metric_a, metric_b, correlation, lag_periods,
                            granularity, sample_size, p_value, computed_at)
                           VALUES (?, ?, ?, 0, 'daily', ?, ?, ?)""",
                        (key_a, key_b, r, n, p_value, now),
                    )
                    stored += 1

        if stored:
            logger.info("MetricMaterializer: stored %d significant correlations", stored)
        return stored

    # ------------------------------------------------------------------
    # Private: event-type materializers
    # ------------------------------------------------------------------

    def _materialize_event_counts(self, cutoff: str, now: str) -> int:
        """Count daily events by type and store as time-series."""
        event_metrics = [
            ("email.daily_received_count", "email.received"),
            ("email.daily_sent_count", "email.sent"),
            ("messaging.daily_received_count", "message.received"),
            ("messaging.daily_sent_count", "message.sent"),
            ("calendar.daily_meeting_count", "calendar.event.created"),
        ]
        total = 0
        try:
            # Fetch all event data first, then close events connection
            fetched: list[tuple[str, list]] = []
            with self.db.get_connection("events") as econn:
                for metric_key, event_type in event_metrics:
                    rows = econn.execute(
                        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
                           FROM events
                           WHERE type = ? AND timestamp > ?
                           GROUP BY DATE(timestamp)""",
                        (event_type, cutoff),
                    ).fetchall()
                    fetched.append((metric_key, rows))

            # Write to user_model without holding events connection
            with self.db.get_connection("user_model") as uconn:
                for metric_key, rows in fetched:
                    for row in rows:
                        uconn.execute(
                            """INSERT OR REPLACE INTO metric_time_series
                               (metric_key, period, granularity, value,
                                sample_count, computed_at)
                               VALUES (?, ?, 'daily', ?, ?, ?)""",
                            (metric_key, row["day"], row["cnt"], row["cnt"], now),
                        )
                        total += 1
        except Exception:
            logger.debug("_materialize_event_counts: could not query events")
        return total

    def _materialize_mood_metrics(self, cutoff: str, now: str) -> int:
        """Aggregate mood_history into daily averages."""
        total = 0
        try:
            with self.db.get_connection("user_model") as conn:
                rows = conn.execute(
                    """SELECT DATE(timestamp) as day,
                              AVG(stress_level) as avg_stress,
                              AVG(energy_level) as avg_energy,
                              AVG(emotional_valence) as avg_valence,
                              AVG(social_battery) as avg_social,
                              AVG(cognitive_load) as avg_cognitive,
                              COUNT(*) as cnt
                       FROM mood_history
                       WHERE timestamp > ?
                       GROUP BY DATE(timestamp)""",
                    (cutoff,),
                ).fetchall()

                mood_metrics = [
                    ("mood.stress_daily_avg", "avg_stress"),
                    ("mood.energy_daily_avg", "avg_energy"),
                    ("mood.valence_daily_avg", "avg_valence"),
                    ("mood.social_battery_daily_avg", "avg_social"),
                    ("mood.cognitive_load_daily_avg", "avg_cognitive"),
                ]
                for row in rows:
                    for metric_key, col in mood_metrics:
                        val = row[col]
                        if val is not None:
                            conn.execute(
                                """INSERT OR REPLACE INTO metric_time_series
                                   (metric_key, period, granularity, value,
                                    sample_count, computed_at)
                                   VALUES (?, ?, 'daily', ?, ?, ?)""",
                                (metric_key, row["day"], val, row["cnt"], now),
                            )
                            total += 1
        except Exception:
            logger.debug("_materialize_mood_metrics: could not query mood_history")
        return total

    def _materialize_task_metrics(self, cutoff: str, now: str) -> int:
        """Count daily task creations and completions."""
        total = 0
        try:
            fetched: list[tuple[str, list]] = []
            with self.db.get_connection("events") as econn:
                for metric_key, event_type in [
                    ("task.daily_created_count", "task.created"),
                    ("task.daily_completed_count", "task.completed"),
                ]:
                    rows = econn.execute(
                        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
                           FROM events
                           WHERE type = ? AND timestamp > ?
                           GROUP BY DATE(timestamp)""",
                        (event_type, cutoff),
                    ).fetchall()
                    fetched.append((metric_key, rows))

            with self.db.get_connection("user_model") as uconn:
                for metric_key, rows in fetched:
                    for row in rows:
                        uconn.execute(
                            """INSERT OR REPLACE INTO metric_time_series
                               (metric_key, period, granularity, value,
                                sample_count, computed_at)
                               VALUES (?, ?, 'daily', ?, ?, ?)""",
                            (metric_key, row["day"], row["cnt"], row["cnt"], now),
                        )
                        total += 1
        except Exception:
            logger.debug("_materialize_task_metrics: could not query events")
        return total

    def _materialize_finance_metrics(self, cutoff: str, now: str) -> int:
        """Aggregate daily spend totals and transaction counts."""
        total = 0
        try:
            with self.db.get_connection("events") as econn:
                rows = econn.execute(
                    """SELECT DATE(timestamp) as day,
                              SUM(ABS(CAST(json_extract(payload, '$.amount') AS REAL))) as total_spend,
                              COUNT(*) as cnt
                       FROM events
                       WHERE type = 'finance.transaction.new'
                         AND timestamp > ?
                         AND json_extract(payload, '$.amount') IS NOT NULL
                       GROUP BY DATE(timestamp)""",
                    (cutoff,),
                ).fetchall()

            # Write to user_model without holding events connection
            with self.db.get_connection("user_model") as uconn:
                for row in rows:
                    if row["total_spend"] is not None:
                        uconn.execute(
                            """INSERT OR REPLACE INTO metric_time_series
                               (metric_key, period, granularity, value,
                                sample_count, computed_at)
                               VALUES (?, ?, 'daily', ?, ?, ?)""",
                            ("finance.daily_spend_total", row["day"],
                             row["total_spend"], row["cnt"], now),
                        )
                        total += 1
                    uconn.execute(
                        """INSERT OR REPLACE INTO metric_time_series
                           (metric_key, period, granularity, value,
                            sample_count, computed_at)
                           VALUES (?, ?, 'daily', ?, ?, ?)""",
                        ("finance.daily_transaction_count", row["day"],
                         row["cnt"], row["cnt"], now),
                    )
                    total += 1
        except Exception:
            logger.debug("_materialize_finance_metrics: could not query events")
        return total

    def _materialize_social_metrics(self, cutoff: str, now: str) -> int:
        """Count unique contacts per day from email and messaging events."""
        total = 0
        try:
            with self.db.get_connection("events") as econn:
                rows = econn.execute(
                    """SELECT DATE(timestamp) as day,
                              COUNT(DISTINCT COALESCE(
                                  json_extract(payload, '$.from'),
                                  json_extract(payload, '$.sender'),
                                  json_extract(payload, '$.contact')
                              )) as unique_contacts
                       FROM events
                       WHERE type IN ('email.received', 'email.sent',
                                      'message.received', 'message.sent')
                         AND timestamp > ?
                       GROUP BY DATE(timestamp)""",
                    (cutoff,),
                ).fetchall()

            with self.db.get_connection("user_model") as uconn:
                for row in rows:
                    uconn.execute(
                        """INSERT OR REPLACE INTO metric_time_series
                           (metric_key, period, granularity, value,
                            sample_count, computed_at)
                           VALUES (?, ?, 'daily', ?, ?, ?)""",
                        ("social.unique_contacts_daily", row["day"],
                         row["unique_contacts"], row["unique_contacts"], now),
                    )
                    total += 1
        except Exception:
            logger.debug("_materialize_social_metrics: could not query events")
        return total


# ============================================================================
# Statistics helpers (no numpy dependency)
# ============================================================================

def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation coefficient without numpy.

    Returns None if the standard deviation of either series is zero
    (constant series cannot correlate).
    """
    n = len(xs)
    if n < 3:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


def _approximate_p_value(r: float, n: int) -> float | None:
    """Approximate two-tailed p-value for Pearson r using t-distribution.

    Uses the t = r * sqrt((n-2) / (1-r^2)) transformation.  For large n
    the t-distribution approaches normal, so we use a simple approximation.
    Returns None if the computation is undefined.
    """
    if abs(r) >= 1.0 or n < 4:
        return None

    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    # Approximate p-value using the survival function of a normal distribution
    # (conservative for small n, increasingly accurate as n grows)
    p = 2 * (1 - _normal_cdf(t))
    return max(p, 1e-10)  # Floor to avoid zero


def _normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the Abramowitz & Stegun formula."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
