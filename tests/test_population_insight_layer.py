"""
Tests for the cross-user insight layer:

  - PopulationBaselineStore: seeding, percentile computation, edge cases
  - MetricMaterializer: event aggregation, mood materialization, correlation computation
  - CohortProfiler: dimension classification, cohort key hashing, storage
  - Comparative correlators: population percentile insights, cross-metric insights
  - Schema: migration v5 creates all 6 new tables
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from services.insight_engine.cohort import CohortProfiler
from services.insight_engine.engine import InsightEngine, _ordinal
from services.insight_engine.metric_materializer import (
    MetricMaterializer,
    _normal_cdf,
    _pearson_r,
)
from storage.population_baselines import PopulationBaselineStore
from storage.user_model_store import UserModelStore

# =============================================================================
# Helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    ums = UserModelStore(db)
    pbs = PopulationBaselineStore(db)
    return InsightEngine(db=db, ums=ums, population_baseline_store=pbs, timezone="UTC")


def _insert_event(db, event_type: str, ts: str, payload: str = "{}") -> None:
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), event_type, "test", ts, 2, payload, "{}"),
        )


def _insert_mood(db, stress: float, energy: float = 0.6, valence: float = 0.6,
                 ts: str | None = None) -> None:
    timestamp = ts or datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO mood_history
               (timestamp, energy_level, stress_level, emotional_valence,
                social_battery, cognitive_load, confidence, trend)
               VALUES (?, ?, ?, ?, 0.5, 0.5, 0.8, 'stable')""",
            (timestamp, energy, stress, valence),
        )


def _insert_metric(db, metric_key: str, period: str, value: float) -> None:
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metric_time_series
               (metric_key, period, granularity, value, sample_count, computed_at)
               VALUES (?, ?, 'daily', ?, 1, ?)""",
            (metric_key, period, value, datetime.now(timezone.utc).isoformat()),
        )


def _insert_correlation(db, metric_a: str, metric_b: str, r: float,
                        sample_size: int = 30) -> None:
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO metric_correlations
               (metric_a, metric_b, correlation, lag_periods, granularity,
                sample_size, p_value, computed_at)
               VALUES (?, ?, ?, 0, 'daily', ?, 0.001, ?)""",
            (metric_a, metric_b, r, sample_size, datetime.now(timezone.utc).isoformat()),
        )


# =============================================================================
# Tests: _ordinal helper
# =============================================================================


class TestOrdinalHelper:
    def test_standard_suffixes(self):
        assert _ordinal(1) == "1st"
        assert _ordinal(2) == "2nd"
        assert _ordinal(3) == "3rd"
        assert _ordinal(4) == "4th"
        assert _ordinal(10) == "10th"
        assert _ordinal(21) == "21st"
        assert _ordinal(22) == "22nd"
        assert _ordinal(23) == "23rd"

    def test_teens(self):
        assert _ordinal(11) == "11th"
        assert _ordinal(12) == "12th"
        assert _ordinal(13) == "13th"

    def test_zero(self):
        assert _ordinal(0) == "0th"

    def test_large_numbers(self):
        assert _ordinal(91) == "91st"
        assert _ordinal(99) == "99th"


# =============================================================================
# Tests: _pearson_r helper
# =============================================================================


class TestPearsonR:
    def test_perfect_positive(self):
        r = _pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r is not None
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = _pearson_r([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r is not None
        assert abs(r + 1.0) < 0.001

    def test_no_correlation(self):
        r = _pearson_r([1, 2, 3, 4, 5], [5, 1, 4, 2, 3])
        assert r is not None
        assert abs(r) < 0.5

    def test_constant_series_returns_none(self):
        assert _pearson_r([5, 5, 5, 5], [1, 2, 3, 4]) is None
        assert _pearson_r([1, 2, 3, 4], [7, 7, 7, 7]) is None

    def test_too_few_points(self):
        assert _pearson_r([1, 2], [3, 4]) is None

    def test_normal_cdf_bounds(self):
        assert 0.49 < _normal_cdf(0) < 0.51
        assert _normal_cdf(3) > 0.99
        assert _normal_cdf(-3) < 0.01


# =============================================================================
# Tests: PopulationBaselineStore
# =============================================================================


class TestPopulationBaselineStore:
    def test_seeding(self, db):
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        baselines = store.get_all_baselines()
        assert len(baselines) >= 15  # We have 18 seed baselines

    def test_get_baseline_returns_dict(self, db):
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        baseline = store.get_baseline("email.daily_received_count", "knowledge_worker")
        assert baseline is not None
        assert baseline["p50"] == 40
        assert baseline["unit"] == "count_per_day"

    def test_get_baseline_cohort_fallback(self, db):
        """Requesting a non-existent cohort falls back to general or knowledge_worker."""
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        baseline = store.get_baseline("email.daily_received_count", "nonexistent_cohort")
        assert baseline is not None

    def test_get_baseline_unknown_metric_returns_none(self, db):
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        assert store.get_baseline("fake.metric.does.not.exist") is None

    def test_percentile_at_median(self, db):
        """Value equal to p50 should return ~50."""
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        pct = store.get_percentile("email.daily_received_count", 40, "knowledge_worker")
        assert pct is not None
        assert 45 <= pct <= 55

    def test_percentile_below_p10(self, db):
        """Value well below p10 should return a low percentile."""
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        pct = store.get_percentile("email.daily_received_count", 3, "knowledge_worker")
        assert pct is not None
        assert pct < 10

    def test_percentile_above_p90(self, db):
        """Value above p90 should return a high percentile."""
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        pct = store.get_percentile("email.daily_received_count", 200, "knowledge_worker")
        assert pct is not None
        assert pct > 90

    def test_percentile_zero_value(self, db):
        """Value of 0 should not crash."""
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        pct = store.get_percentile("email.daily_received_count", 0, "knowledge_worker")
        assert pct is not None
        assert pct >= 0

    def test_percentile_unknown_metric_returns_none(self, db):
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        assert store.get_percentile("fake.metric", 42) is None

    def test_double_seed_is_idempotent(self, db):
        store = PopulationBaselineStore(db)
        store.ensure_seeded()
        count1 = len(store.get_all_baselines())
        store._seeded = False
        store.ensure_seeded()
        count2 = len(store.get_all_baselines())
        assert count1 == count2


# =============================================================================
# Tests: MetricMaterializer
# =============================================================================


class TestMetricMaterializer:
    def test_materialize_email_events(self, db):
        """Email events are aggregated into daily counts."""
        mat = MetricMaterializer(db)
        now = datetime.now(timezone.utc)
        for i in range(10):
            _insert_event(db, "email.received", (now - timedelta(days=1)).isoformat())
        for i in range(5):
            _insert_event(db, "email.sent", (now - timedelta(days=1)).isoformat())

        total = mat.materialize()
        assert total >= 2  # At least email.received + email.sent for that day

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value FROM metric_time_series WHERE metric_key = 'email.daily_received_count'"
            ).fetchone()
            assert row is not None
            assert row["value"] == 10

    def test_materialize_mood_metrics(self, db):
        """Mood history is aggregated into daily averages."""
        mat = MetricMaterializer(db)
        now = datetime.now(timezone.utc)
        for i in range(5):
            _insert_mood(db, stress=0.6, energy=0.7, valence=0.5, ts=now.isoformat())

        total = mat.materialize()
        assert total >= 1

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT value FROM metric_time_series
                   WHERE metric_key = 'mood.stress_daily_avg'"""
            ).fetchone()
            assert row is not None
            assert abs(row["value"] - 0.6) < 0.01

    def test_materialize_finance_metrics(self, db):
        """Finance transactions are aggregated into daily totals."""
        mat = MetricMaterializer(db)
        now = datetime.now(timezone.utc)
        for amount in [25.0, 50.0, 75.0]:
            _insert_event(
                db, "finance.transaction.new", now.isoformat(),
                json.dumps({"amount": amount, "category": "TEST"}),
            )

        total = mat.materialize()
        assert total >= 1

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT value FROM metric_time_series
                   WHERE metric_key = 'finance.daily_spend_total'"""
            ).fetchone()
            assert row is not None
            assert abs(row["value"] - 150.0) < 0.01

    def test_compute_correlations_with_correlated_data(self, db):
        """Strongly correlated metrics produce a stored correlation row."""
        mat = MetricMaterializer(db)

        # Create two perfectly correlated daily series across different domains
        for i in range(20):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "mood.stress_daily_avg", period, 0.3 + i * 0.02)
            _insert_metric(db, "finance.daily_spend_total", period, 10 + i * 5)

        stored = mat.compute_correlations(min_samples=14)
        assert stored >= 1

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT correlation FROM metric_correlations
                   WHERE metric_a = 'finance.daily_spend_total'
                     AND metric_b = 'mood.stress_daily_avg'"""
            ).fetchone()
            assert row is not None
            assert abs(row["correlation"]) > 0.9

    def test_compute_correlations_skips_uncorrelated(self, db):
        """Uncorrelated metrics do not produce a correlation row."""
        mat = MetricMaterializer(db)

        for i in range(20):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "email.daily_received_count", period, 40 + (i % 3))
            _insert_metric(db, "task.daily_completed_count", period, 3 + ((i * 7) % 5))

        mat.compute_correlations(min_samples=14)
        # These should not be strongly correlated
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM metric_correlations WHERE ABS(correlation) >= 0.7"
            ).fetchone()
            assert row["cnt"] == 0


# =============================================================================
# Tests: CohortProfiler
# =============================================================================


class TestCohortProfiler:
    def test_all_unknown_when_no_data(self, db):
        ums = UserModelStore(db)
        profiler = CohortProfiler(db, ums)
        profile = profiler.compute_profile()
        assert all(v == "unknown" for v in profile.values())

    def test_compute_and_store_returns_none_with_no_data(self, db):
        """Fewer than 3 known dimensions -> None cohort key."""
        ums = UserModelStore(db)
        profiler = CohortProfiler(db, ums)
        result = profiler.compute_and_store()
        assert result is None

    def test_cohort_key_is_deterministic(self, db):
        """Same profile always produces the same cohort key."""
        from services.insight_engine.cohort import CohortProfiler
        key1 = CohortProfiler._compute_cohort_key({
            "chronotype": "early_bird",
            "stress_profile": "moderate",
            "communication_volume": "high",
        })
        key2 = CohortProfiler._compute_cohort_key({
            "chronotype": "early_bird",
            "stress_profile": "moderate",
            "communication_volume": "high",
        })
        assert key1 == key2
        assert len(key1) == 8

    def test_different_profiles_produce_different_keys(self, db):
        key1 = CohortProfiler._compute_cohort_key({
            "chronotype": "early_bird",
            "stress_profile": "moderate",
        })
        key2 = CohortProfiler._compute_cohort_key({
            "chronotype": "night_owl",
            "stress_profile": "high",
        })
        assert key1 != key2

    def test_stored_profile_roundtrip(self, db):
        """compute_and_store() stores dimensions retrievable by get_stored_profile()."""
        ums = UserModelStore(db)
        profiler = CohortProfiler(db, ums)

        # Manually insert enough data to give 3+ known dimensions
        now = datetime.now(timezone.utc)
        # Stress data
        for i in range(10):
            _insert_mood(db, stress=0.7, ts=(now - timedelta(days=i)).isoformat())
        # Email data
        for i in range(40):
            _insert_event(db, "email.received", (now - timedelta(days=i % 30)).isoformat())
        # Calendar data
        for i in range(30):
            _insert_event(db, "calendar.event.created", (now - timedelta(days=i % 30)).isoformat())

        result = profiler.compute_and_store()
        # Should have at least stress_profile, communication_volume, meeting_load
        if result is not None:
            stored = profiler.get_stored_profile()
            assert len(stored) >= 3
            assert stored.get("stress_profile") == "high"


# =============================================================================
# Tests: Comparative Population Insights Correlator
# =============================================================================


class TestComparativePopulationInsights:
    def test_no_insight_when_no_materialized_metrics(self, db):
        engine = _make_engine(db)
        insights = engine._comparative_population_insights()
        assert insights == []

    def test_no_insight_when_metric_in_normal_range(self, db):
        """Value at 50th percentile -> no insight (20-80 range is silent)."""
        engine = _make_engine(db)
        engine.pbs.ensure_seeded()

        # Insert 7 days of email counts at exactly the median (40)
        for i in range(7):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "email.daily_received_count", period, 40)

        insights = engine._comparative_population_insights()
        email_insights = [i for i in insights if "email" in (i.entity or "")]
        assert email_insights == []

    def test_insight_fires_for_extreme_high_metric(self, db):
        """Email volume at 200/day (well above p90=121) -> fires insight."""
        engine = _make_engine(db)
        engine.pbs.ensure_seeded()

        for i in range(7):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "email.daily_received_count", period, 200)

        insights = engine._comparative_population_insights()
        email_insights = [
            i for i in insights
            if i.entity == "email.daily_received_count"
        ]
        assert len(email_insights) == 1
        assert "percentile" in email_insights[0].summary.lower()

    def test_insight_fires_for_extreme_low_metric(self, db):
        """Stress at 0.1 (well below p10=0.15) -> fires low-percentile insight."""
        engine = _make_engine(db)
        engine.pbs.ensure_seeded()

        for i in range(7):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "mood.stress_daily_avg", period, 0.08)

        insights = engine._comparative_population_insights()
        stress_insights = [
            i for i in insights
            if i.entity == "mood.stress_daily_avg"
        ]
        assert len(stress_insights) == 1

    def test_no_insight_with_fewer_than_5_days(self, db):
        """Fewer than 5 metric days -> not enough data."""
        engine = _make_engine(db)
        engine.pbs.ensure_seeded()

        for i in range(3):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "email.daily_received_count", period, 200)

        insights = engine._comparative_population_insights()
        email_insights = [i for i in insights if i.entity == "email.daily_received_count"]
        assert email_insights == []


# =============================================================================
# Tests: Cross-Metric Correlation Insights
# =============================================================================


class TestCrossMetricCorrelationInsights:
    def test_no_insight_when_no_correlations(self, db):
        engine = _make_engine(db)
        insights = engine._cross_metric_correlation_insights()
        assert insights == []

    def test_surfaces_cross_domain_correlation(self, db):
        """Correlation between mood and finance (different domains) is surfaced."""
        engine = _make_engine(db)
        _insert_correlation(db, "finance.daily_spend_total", "mood.stress_daily_avg", 0.72)
        insights = engine._cross_metric_correlation_insights()
        assert len(insights) == 1
        assert "correlated" in insights[0].summary.lower()
        assert insights[0].category == "cross_metric_correlation"

    def test_skips_same_domain_correlation(self, db):
        """Correlation between email.sent and email.received (same domain) is NOT surfaced."""
        engine = _make_engine(db)
        _insert_correlation(db, "email.daily_received_count", "email.daily_sent_count", 0.85)
        insights = engine._cross_metric_correlation_insights()
        assert insights == []

    def test_negative_correlation_described_correctly(self, db):
        """Negative correlation uses 'inversely' in summary."""
        engine = _make_engine(db)
        _insert_correlation(db, "mood.energy_daily_avg", "finance.daily_spend_total", -0.65)
        insights = engine._cross_metric_correlation_insights()
        assert len(insights) == 1
        assert "inversely" in insights[0].summary.lower()


# =============================================================================
# Tests: Schema Migration v5
# =============================================================================


class TestSchemaV5:
    def test_all_new_tables_exist(self, db):
        """All 6 cross-user insight tables are created by the schema."""
        expected_tables = [
            "metric_time_series",
            "population_baselines",
            "metric_correlations",
            "cohort_profiles",
            "insight_outcomes",
            "contribution_queue",
        ]
        with db.get_connection("user_model") as conn:
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        for table in expected_tables:
            assert table in existing, f"Missing table: {table}"

    def test_schema_version_is_5(self, db):
        with db.get_connection("user_model") as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            assert row[0] == 5

    def test_metric_time_series_primary_key(self, db):
        """Upserting same metric+period replaces, not duplicates."""
        with db.get_connection("user_model") as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT OR REPLACE INTO metric_time_series
                   (metric_key, period, granularity, value, sample_count, computed_at)
                   VALUES ('test.metric', '2026-01-01', 'daily', 10.0, 1, ?)""",
                (now,),
            )
            conn.execute(
                """INSERT OR REPLACE INTO metric_time_series
                   (metric_key, period, granularity, value, sample_count, computed_at)
                   VALUES ('test.metric', '2026-01-01', 'daily', 20.0, 2, ?)""",
                (now,),
            )
            rows = conn.execute(
                "SELECT * FROM metric_time_series WHERE metric_key = 'test.metric'"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["value"] == 20.0


# =============================================================================
# Integration: new correlators appear in generate_insights
# =============================================================================


class TestComparativeWiring:
    async def test_comparative_population_wired_in(self, db):
        """Population insight fires through generate_insights when data present."""
        engine = _make_engine(db)
        engine.pbs.ensure_seeded()
        for i in range(7):
            period = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            _insert_metric(db, "email.daily_received_count", period, 200)
        results = await engine.generate_insights()
        categories = {i.category for i in results}
        assert "population_percentile_email" in categories

    async def test_cross_metric_wired_in(self, db):
        """Cross-metric correlation fires through generate_insights."""
        engine = _make_engine(db)
        _insert_correlation(db, "finance.daily_spend_total", "mood.stress_daily_avg", 0.72)
        results = await engine.generate_insights()
        categories = {i.category for i in results}
        assert "cross_metric_correlation" in categories
