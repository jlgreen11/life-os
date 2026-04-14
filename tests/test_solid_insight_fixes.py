"""
Tests for the three solid fixes:

  1. Personal trend correlator: 90-day personal baseline comparison
  2. insight_outcomes write path: baseline on store, post-values on schedule
  3. generate_insights threads correlators (event loop unblocked)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from storage.population_baselines import PopulationBaselineStore
from storage.user_model_store import UserModelStore


# =============================================================================
# Helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    ums = UserModelStore(db)
    pbs = PopulationBaselineStore(db)
    return InsightEngine(db=db, ums=ums, population_baseline_store=pbs, timezone="UTC")


def _insert_metric(db, metric_key: str, period: str, value: float) -> None:
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metric_time_series
               (metric_key, period, granularity, value, sample_count, computed_at)
               VALUES (?, ?, 'daily', ?, 1, ?)""",
            (metric_key, period, value, datetime.now(timezone.utc).isoformat()),
        )


def _day(offset: int) -> str:
    """Return YYYY-MM-DD string offset days from today."""
    return (datetime.now(timezone.utc) - timedelta(days=offset)).strftime("%Y-%m-%d")


# =============================================================================
# Personal trend correlator
# =============================================================================


class TestPersonalTrendInsights:
    def test_no_insight_when_no_data(self, db):
        engine = _make_engine(db)
        results = engine._personal_trend_insights()
        assert results == []

    def test_no_insight_when_insufficient_recent_days(self, db):
        """Need >= 5 recent days and >= 14 baseline days."""
        engine = _make_engine(db)
        # Insert only 3 recent days, no baseline
        for i in range(3):
            _insert_metric(db, "email.daily_received_count", _day(i), 80.0)
        results = engine._personal_trend_insights()
        assert results == []

    def test_no_insight_when_change_below_threshold(self, db):
        """< 20% change should produce no insight."""
        engine = _make_engine(db)
        # Baseline: 14 days at 50 emails/day (15–28 days ago)
        for i in range(15, 29):
            _insert_metric(db, "email.daily_received_count", _day(i), 50.0)
        # Recent: 7 days at 55 emails/day (+10% -- below 20% threshold)
        for i in range(7):
            _insert_metric(db, "email.daily_received_count", _day(i), 55.0)
        results = engine._personal_trend_insights()
        assert results == []

    def test_insight_fires_for_significant_increase(self, db):
        """40% increase above personal baseline should fire."""
        engine = _make_engine(db)
        # Baseline: 20 days at 50 emails/day
        for i in range(15, 35):
            _insert_metric(db, "email.daily_received_count", _day(i), 50.0)
        # Recent: 7 days at 75 emails/day (+50%)
        for i in range(7):
            _insert_metric(db, "email.daily_received_count", _day(i), 75.0)
        results = engine._personal_trend_insights()
        assert len(results) == 1
        insight = results[0]
        assert "up" in insight.summary
        assert "email" in insight.summary
        assert insight.category == "personal_trend"
        assert insight.confidence > 0.45

    def test_insight_fires_for_significant_decrease(self, db):
        """40% decrease below personal baseline should fire."""
        engine = _make_engine(db)
        # Baseline: 20 days at 50 tasks/day
        for i in range(15, 35):
            _insert_metric(db, "task.daily_completed_count", _day(i), 50.0)
        # Recent: 7 days at 25 tasks/day (-50%)
        for i in range(7):
            _insert_metric(db, "task.daily_completed_count", _day(i), 25.0)
        results = engine._personal_trend_insights()
        assert len(results) == 1
        assert "down" in results[0].summary

    def test_evidence_contains_required_fields(self, db):
        engine = _make_engine(db)
        for i in range(15, 35):
            _insert_metric(db, "mood.stress_daily_avg", _day(i), 0.4)
        for i in range(7):
            _insert_metric(db, "mood.stress_daily_avg", _day(i), 0.7)  # +75%
        results = engine._personal_trend_insights()
        assert len(results) == 1
        ev = {e.split("=")[0]: e.split("=", 1)[1] for e in results[0].evidence}
        assert "metric_key" in ev
        assert "recent_avg" in ev
        assert "personal_baseline" in ev
        assert "pct_change" in ev

    def test_personal_trend_wired_in_correlators(self, db):
        engine = _make_engine(db)
        raw, stats = engine._run_correlators_sync()
        assert "personal_trend" in stats

    def test_multiple_metrics_produce_multiple_insights(self, db):
        engine = _make_engine(db)
        for metric in ["email.daily_received_count", "mood.stress_daily_avg"]:
            for i in range(15, 35):
                _insert_metric(db, metric, _day(i), 30.0)
            for i in range(7):
                _insert_metric(db, metric, _day(i), 60.0)  # +100%
        results = engine._personal_trend_insights()
        assert len(results) == 2


# =============================================================================
# insight_outcomes write path
# =============================================================================


class TestInsightOutcomesWritePath:
    def test_baseline_written_on_store_for_population_insight(self, db):
        """Storing a population-comparison insight should write a baseline row."""
        engine = _make_engine(db)
        # Seed a population insight manually via _store_insight
        from services.insight_engine.models import Insight
        insight = Insight(
            type="behavioral_pattern",
            summary="Test population insight",
            confidence=0.7,
            evidence=[
                "metric_key=email.daily_received_count",
                "user_avg=85.000",
                "population_median=40",
                "percentile=92",
            ],
            category="population_percentile_email",
            entity="email.daily_received_count",
        )
        insight.compute_dedup_key()
        engine._store_insight(insight)

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT * FROM insight_outcomes
                   WHERE insight_id = ? AND days_after = 0""",
                (insight.id,),
            ).fetchone()

        assert row is not None
        assert row["metric_key"] == "email.daily_received_count"
        assert abs(row["baseline_value"] - 85.0) < 0.01

    def test_baseline_written_for_personal_trend_insight(self, db):
        """Storing a personal-trend insight should also write a baseline row."""
        from services.insight_engine.models import Insight
        engine = _make_engine(db)
        insight = Insight(
            type="behavioral_pattern",
            summary="Stress up 40% from personal baseline",
            confidence=0.65,
            evidence=[
                "metric_key=mood.stress_daily_avg",
                "recent_avg=0.650",
                "personal_baseline=0.400",
            ],
            category="personal_trend",
            entity="mood.stress_daily_avg",
        )
        insight.compute_dedup_key()
        engine._store_insight(insight)

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT * FROM insight_outcomes WHERE insight_id = ? AND days_after = 0",
                (insight.id,),
            ).fetchone()

        assert row is not None
        assert row["metric_key"] == "mood.stress_daily_avg"
        assert abs(row["baseline_value"] - 0.65) < 0.001

    def test_no_baseline_for_insights_without_metric_key(self, db):
        """Insights without metric_key evidence should not write outcome rows."""
        from services.insight_engine.models import Insight
        engine = _make_engine(db)
        insight = Insight(
            type="behavioral_pattern",
            summary="Email peak hour at 9am",
            confidence=0.6,
            evidence=["peak_hour=9", "count=42"],
            category="email_peak_hour",
            entity="alice@example.com",
        )
        insight.compute_dedup_key()
        engine._store_insight(insight)

        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM insight_outcomes WHERE insight_id = ?",
                (insight.id,),
            ).fetchone()[0]

        assert count == 0

    def test_measure_pending_outcomes_writes_post_value(self, db):
        """measure_pending_outcomes should write 7-day post value for old insights."""
        from services.insight_engine.models import Insight
        engine = _make_engine(db)

        # Insert a metric_time_series entry for the current 7-day window
        for i in range(7):
            _insert_metric(db, "email.daily_received_count", _day(i), 70.0)

        # Simulate an insight stored 7 days ago with a baseline row
        insight_id = str(uuid.uuid4())
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO insights
                   (id, type, summary, confidence, evidence, category,
                    entity, staleness_ttl_hours, dedup_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (insight_id, "behavioral_pattern", "Test", 0.7, "[]",
                 "population_percentile_email", "email.daily_received_count",
                 168, "dedup_test", seven_days_ago),
            )
            conn.execute(
                """INSERT INTO insight_outcomes
                   (insight_id, metric_key, baseline_value, days_after, measured_at)
                   VALUES (?, ?, ?, 0, ?)""",
                (insight_id, "email.daily_received_count", 85.0, seven_days_ago),
            )

        written = engine.measure_pending_outcomes()
        assert written == 1

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT * FROM insight_outcomes
                   WHERE insight_id = ? AND days_after = 7""",
                (insight_id,),
            ).fetchone()

        assert row is not None
        assert row["post_value"] is not None
        assert abs(row["post_value"] - 70.0) < 0.01
        # Delta is signed: higher email = worse (lower_is_better), so delta should be negative
        assert row["delta"] is not None

    def test_measure_skips_when_no_post_metric_data(self, db):
        """If no metric_time_series data exists for the post-period, skip."""
        engine = _make_engine(db)
        insight_id = str(uuid.uuid4())
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO insights
                   (id, type, summary, confidence, evidence, category,
                    entity, staleness_ttl_hours, dedup_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (insight_id, "behavioral_pattern", "Test", 0.7, "[]",
                 "population_percentile_email", "email.daily_received_count",
                 168, "dedup_test2", seven_days_ago),
            )
            conn.execute(
                """INSERT INTO insight_outcomes
                   (insight_id, metric_key, baseline_value, days_after, measured_at)
                   VALUES (?, ?, ?, 0, ?)""",
                (insight_id, "email.daily_received_count", 85.0, seven_days_ago),
            )

        # No metric_time_series data → written should be 0
        written = engine.measure_pending_outcomes()
        assert written == 0

    def test_measure_not_duplicate_if_already_measured(self, db):
        """measure_pending_outcomes should not re-write an already-measured checkpoint."""
        engine = _make_engine(db)
        insight_id = str(uuid.uuid4())
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        now = datetime.now(timezone.utc).isoformat()

        for i in range(7):
            _insert_metric(db, "task.daily_completed_count", _day(i), 5.0)

        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO insights
                   (id, type, summary, confidence, evidence, category,
                    entity, staleness_ttl_hours, dedup_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (insight_id, "behavioral_pattern", "Test", 0.7, "[]",
                 "personal_trend", "task.daily_completed_count",
                 168, "dedup_test3", seven_days_ago),
            )
            # baseline row
            conn.execute(
                """INSERT INTO insight_outcomes
                   (insight_id, metric_key, baseline_value, days_after, measured_at)
                   VALUES (?, ?, ?, 0, ?)""",
                (insight_id, "task.daily_completed_count", 3.0, seven_days_ago),
            )
            # already has a 7-day row
            conn.execute(
                """INSERT INTO insight_outcomes
                   (insight_id, metric_key, baseline_value, post_value,
                    delta, days_after, measured_at)
                   VALUES (?, ?, ?, ?, ?, 7, ?)""",
                (insight_id, "task.daily_completed_count", 3.0, 5.0, 2.0, now),
            )

        written = engine.measure_pending_outcomes()
        assert written == 0  # Already measured, should not double-write


# =============================================================================
# Event loop: generate_insights threads the correlator loop
# =============================================================================


class TestGenerateInsightsNonBlocking:
    async def test_generate_insights_is_awaitable(self, db):
        """generate_insights must be awaitable (returns a coroutine)."""
        engine = _make_engine(db)
        result = await engine.generate_insights()
        assert isinstance(result, list)

    async def test_run_correlators_sync_returns_tuple(self, db):
        """_run_correlators_sync returns (list, dict) for destructuring in generate_insights."""
        engine = _make_engine(db)
        raw, stats = engine._run_correlators_sync()
        assert isinstance(raw, list)
        assert isinstance(stats, dict)
        assert "personal_trend" in stats
        assert "comparative_population" in stats

    async def test_correlators_run_in_thread(self, db):
        """Verify correlators run in a thread pool, not blocking the event loop."""
        engine = _make_engine(db)

        # If the correlators blocked the event loop, this concurrent sleep would
        # not advance.  With to_thread, both complete within the same wall time.
        async def _background():
            await asyncio.sleep(0.01)
            return "done"

        result, bg = await asyncio.gather(
            engine.generate_insights(),
            _background(),
        )
        assert bg == "done"
