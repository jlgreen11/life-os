"""
Tests for the InsightEngine ``_temporal_pattern_insights`` and
``_mood_trend_insights`` correlators.

``_temporal_pattern_insights`` converts the temporal signal profile
(activity_by_hour, activity_by_day) into three possible insight sub-types:
- chronotype (early bird / night owl / mixed)
- peak_hour (most active hour during daytime)
- busiest_day (day with the most activity vs. weekly average)

``_mood_trend_insights`` reads the mood_history time-series and surfaces
improving or declining mood trajectories based on composite score deltas.

This test suite validates:
- No insights when temporal profile is absent or has too few samples
- Chronotype detection: early bird, night owl, and mixed / insufficient window data
- Peak-hour detection and suppression when peak count is below threshold
- Busiest-day detection and suppression when busiest day is not significantly above average
- No mood insight when mood_history has fewer than MIN_ROWS rows
- Stable mood (delta within threshold) generates no insight
- Improving mood generates an "improving" insight with correct metadata
- Declining mood generates a "declining" insight with correct metadata
- Both correlators are wired into generate_insights()
- New insight types appear in generated output after being wired in
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore


# =============================================================================
# Helpers
# =============================================================================


def _make_ums(db) -> UserModelStore:
    """Return a UserModelStore wired to the temp DatabaseManager."""
    return UserModelStore(db)


def _make_engine(db) -> InsightEngine:
    """Return an InsightEngine wired to the temp DatabaseManager."""
    ums = _make_ums(db)
    return InsightEngine(db=db, ums=ums)


def _set_temporal_profile(ums: UserModelStore, activity_by_hour: dict[str, int],
                          activity_by_day: dict[str, int],
                          samples_count: int) -> None:
    """Write a temporal signal profile with specified histogram data.

    Calls update_signal_profile() once per artificial sample so the
    samples_count field is driven purely by the supplied value.
    We achieve this by calling upsert once with the target samples_count
    written directly via SQL.
    """
    data = {
        "activity_by_hour": activity_by_hour,
        "activity_by_day": activity_by_day,
        "activity_by_type": {},
        "scheduled_hours": {},
        "advance_planning_days": [],
    }
    # Update once to create the row
    ums.update_signal_profile("temporal", data)
    # Force the samples_count to the desired value directly
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (samples_count, "temporal"),
        )


def _insert_mood_row(db, energy: float, stress: float, valence: float,
                     ts: str | None = None) -> None:
    """Insert a row into mood_history."""
    timestamp = ts or datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO mood_history
               (timestamp, energy_level, stress_level, emotional_valence, confidence, trend)
               VALUES (?, ?, ?, ?, 0.8, 'stable')""",
            (timestamp, energy, stress, valence),
        )


# =============================================================================
# Tests: _temporal_pattern_insights
# =============================================================================


class TestTemporalPatternInsights:
    """Unit tests for the temporal-pattern correlator."""

    def test_returns_empty_when_no_temporal_profile(self, db):
        """No temporal profile → no insights."""
        engine = _make_engine(db)
        insights = engine._temporal_pattern_insights()
        assert insights == []

    def test_returns_empty_when_too_few_samples(self, db):
        """Fewer than MIN_SAMPLES (50) total samples → no insights emitted."""
        ums = _make_ums(db)
        # Provide a plausible hour histogram but only 10 samples
        hours = {str(h): 5 for h in range(6, 12)}  # morning-only activity
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=10)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        assert insights == []

    def test_early_bird_chronotype(self, db):
        """Heavy morning activity → 'early_bird' chronotype insight."""
        ums = _make_ums(db)
        # 5–10 h dominant; 18–22 h minimal
        hours = {str(h): 20 for h in range(5, 11)}  # 120 morning events
        hours.update({str(h): 3 for h in range(18, 23)})  # 15 evening events
        days = {"monday": 20, "tuesday": 18, "wednesday": 15, "thursday": 17,
                "friday": 15, "saturday": 10, "sunday": 5}
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day=days,
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        chrono_insights = [i for i in insights if i.category == "chronotype"]
        assert len(chrono_insights) == 1
        insight = chrono_insights[0]
        assert insight.type == "temporal_pattern"
        assert "early_bird" in insight.entity
        assert "early bird" in insight.summary.lower() or "morning" in insight.summary.lower()
        assert any("chronotype=early_bird" in e for e in insight.evidence)
        assert 0.0 < insight.confidence <= 1.0

    def test_night_owl_chronotype(self, db):
        """Heavy evening activity → 'night_owl' chronotype insight."""
        ums = _make_ums(db)
        hours = {str(h): 3 for h in range(5, 11)}   # 18 morning events
        hours.update({str(h): 25 for h in range(18, 23)})  # 125 evening events
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        chrono = [i for i in insights if i.category == "chronotype"]
        assert len(chrono) == 1
        assert chrono[0].entity == "night_owl"

    def test_mixed_chronotype_no_dominant_window(self, db):
        """Balanced morning/evening activity → 'mixed' chronotype."""
        ums = _make_ums(db)
        # Morning ~100, evening ~100 — ratio < 1.5
        hours = {str(h): 17 for h in range(5, 11)}   # ~102 morning
        hours.update({str(h): 16 for h in range(18, 23)})  # 80 evening
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        chrono = [i for i in insights if i.category == "chronotype"]
        assert len(chrono) == 1
        assert chrono[0].entity == "mixed"

    def test_peak_hour_insight(self, db):
        """Hour with high activity count → 'peak_hour' insight."""
        ums = _make_ums(db)
        # Hour 9 has the most activity (50 events — well above MIN_PEAK_ACTIVITY=10)
        hours: dict[str, int] = {}
        for h in range(6, 22):
            hours[str(h)] = 5  # baseline
        hours["9"] = 50  # peak
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=300)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        peak_insights = [i for i in insights if i.category == "peak_hour"]
        assert len(peak_insights) == 1
        insight = peak_insights[0]
        assert insight.entity == "peak_hour_9"
        assert "9" in insight.summary
        assert any("peak_hour=9" in e for e in insight.evidence)
        assert 0.0 < insight.confidence <= 1.0

    def test_peak_hour_suppressed_when_below_threshold(self, db):
        """Peak hour count below MIN_PEAK_ACTIVITY → no peak_hour insight."""
        ums = _make_ums(db)
        # All hours have only 5 events — below MIN_PEAK_ACTIVITY (10)
        hours = {str(h): 5 for h in range(6, 22)}
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        assert not any(i.category == "peak_hour" for i in insights)

    def test_busiest_day_insight(self, db):
        """Day with significantly above-average activity → 'busiest_day' insight."""
        ums = _make_ums(db)
        hours = {str(h): 10 for h in range(6, 22)}  # provide enough hour data
        # Tuesday has 2× the average
        days = {
            "monday": 20, "tuesday": 60, "wednesday": 20,
            "thursday": 22, "friday": 18, "saturday": 10, "sunday": 10,
        }
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day=days,
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        day_insights = [i for i in insights if i.category == "busiest_day"]
        assert len(day_insights) == 1
        assert day_insights[0].entity == "tuesday"
        assert "tuesday" in day_insights[0].summary.lower()

    def test_busiest_day_suppressed_when_not_above_average(self, db):
        """Uniform day distribution → no busiest_day insight."""
        ums = _make_ums(db)
        hours = {str(h): 10 for h in range(6, 22)}
        # All days equal — ratio = 1.0, below 1.3 threshold
        days = {d: 20 for d in
                ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day=days,
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        assert not any(i.category == "busiest_day" for i in insights)

    def test_busiest_day_suppressed_when_too_few_day_samples(self, db):
        """Fewer than MIN_DAY_SAMPLES (30) day-level samples → no busiest_day insight."""
        ums = _make_ums(db)
        hours = {str(h): 10 for h in range(6, 22)}
        # Only 10 total day samples (2+2+2+1+1+1+1)
        days = {"monday": 5, "tuesday": 3, "wednesday": 2}
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day=days,
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._temporal_pattern_insights()
        assert not any(i.category == "busiest_day" for i in insights)

    def test_dedup_key_is_stable_for_same_label(self, db):
        """Two correlator runs with the same pattern → same dedup_key."""
        ums = _make_ums(db)
        hours = {str(h): 20 for h in range(5, 11)}
        hours.update({str(h): 3 for h in range(18, 23)})
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day={},
                              samples_count=200)
        engine = InsightEngine(db=db, ums=ums)
        run1 = engine._temporal_pattern_insights()
        run2 = engine._temporal_pattern_insights()
        keys1 = {i.dedup_key for i in run1 if i.category == "chronotype"}
        keys2 = {i.dedup_key for i in run2 if i.category == "chronotype"}
        assert keys1 == keys2, "Dedup key should be deterministic for same label"


# =============================================================================
# Tests: _mood_trend_insights
# =============================================================================


class TestMoodTrendInsights:
    """Unit tests for the mood-trend correlator."""

    def test_returns_empty_when_no_mood_history(self, db):
        """Empty mood_history table → no insight."""
        engine = _make_engine(db)
        insights = engine._mood_trend_insights()
        assert insights == []

    def test_returns_empty_when_too_few_rows(self, db):
        """Fewer than 6 mood_history rows → no insight."""
        engine = _make_engine(db)
        # Insert only 4 rows
        for _ in range(4):
            _insert_mood_row(db, energy=0.7, stress=0.3, valence=0.8)
        insights = engine._mood_trend_insights()
        assert insights == []

    def test_stable_mood_generates_no_insight(self, db):
        """Stable mood (delta within ±0.15) → no insight.

        Insert baseline first (older) then recent (newer) — both with nearly
        identical composites so delta stays within ±0.15.
        """
        engine = _make_engine(db)
        # Baseline 9 (inserted first → older): composite ≈ 0.68+0.78-0.28 = 1.18
        for _ in range(9):
            _insert_mood_row(db, energy=0.68, stress=0.28, valence=0.78)
        # Recent 3 (inserted last → newer): composite ≈ 0.70+0.80-0.30 = 1.20 (delta ≈ 0.02)
        for _ in range(3):
            _insert_mood_row(db, energy=0.70, stress=0.30, valence=0.80)
        insights = engine._mood_trend_insights()
        assert insights == []

    def test_improving_mood_generates_insight(self, db):
        """Recent mood significantly better than baseline → 'improving' insight.

        The mood_history query returns rows ORDER BY timestamp DESC, so rows
        inserted last are "recent" and rows inserted first are "baseline".
        """
        engine = _make_engine(db)
        # Baseline 9 (inserted first → older timestamps): low energy, high stress
        for _ in range(9):
            _insert_mood_row(db, energy=0.3, stress=0.7, valence=0.5)
        # Recent 3 (inserted last → newer timestamps): high energy, low stress
        for _ in range(3):
            _insert_mood_row(db, energy=0.9, stress=0.1, valence=0.8)
        insights = engine._mood_trend_insights()
        assert len(insights) == 1
        insight = insights[0]
        assert insight.type == "mood_trend"
        assert insight.category == "mood_trajectory"
        assert insight.entity == "improving"
        assert "improving" in insight.summary.lower()
        assert any("trend=improving" in e for e in insight.evidence)
        assert insight.staleness_ttl_hours == 48
        assert 0.0 < insight.confidence <= 1.0

    def test_declining_mood_generates_insight(self, db):
        """Recent mood significantly worse than baseline → 'declining' insight.

        Insert baseline first (older) then recent (newer) so the DESC query
        picks recent rows as rows[0:3] and baseline as rows[3:].
        """
        engine = _make_engine(db)
        # Baseline 9 (inserted first → older): high energy, low stress
        for _ in range(9):
            _insert_mood_row(db, energy=0.9, stress=0.1, valence=0.8)
        # Recent 3 (inserted last → newer): low energy, high stress
        for _ in range(3):
            _insert_mood_row(db, energy=0.3, stress=0.8, valence=0.4)
        insights = engine._mood_trend_insights()
        assert len(insights) == 1
        insight = insights[0]
        assert insight.entity == "declining"
        assert "declining" in insight.summary.lower()
        assert any("trend=declining" in e for e in insight.evidence)

    def test_mood_insight_includes_numeric_evidence(self, db):
        """Mood insight evidence should include composite scores and delta."""
        engine = _make_engine(db)
        # Baseline first (older), recent last (newer)
        for _ in range(9):
            _insert_mood_row(db, energy=0.4, stress=0.6, valence=0.4)
        for _ in range(3):
            _insert_mood_row(db, energy=0.9, stress=0.1, valence=0.9)
        insights = engine._mood_trend_insights()
        assert len(insights) == 1
        ev = insights[0].evidence
        assert any("recent_composite=" in e for e in ev)
        assert any("baseline_composite=" in e for e in ev)
        assert any("delta=" in e for e in ev)
        assert any("rows_analyzed=" in e for e in ev)

    def test_mood_trend_dedup_key_stable(self, db):
        """Same trend → same dedup_key on two separate calls."""
        engine = _make_engine(db)
        # Baseline first, recent last
        for _ in range(9):
            _insert_mood_row(db, energy=0.3, stress=0.8, valence=0.3)
        for _ in range(3):
            _insert_mood_row(db, energy=0.9, stress=0.1, valence=0.9)
        r1 = engine._mood_trend_insights()
        r2 = engine._mood_trend_insights()
        assert r1[0].dedup_key == r2[0].dedup_key

    def test_null_mood_fields_handled_gracefully(self, db):
        """NULL mood_history columns fall back to 0.5 without crashing.

        The query returns rows ORDER BY timestamp DESC, so the 9 baseline rows
        (inserted first → older) become rows[3:], and the 3 NULL rows (inserted
        last → newer) become rows[0:3].  NULL → 0.5 defaults make the recent
        composite 0.5+0.5-0.5=0.5, while the baseline composite is 1.7 → declining.
        """
        engine = _make_engine(db)
        # Baseline 9 (inserted first → older): high composite
        with db.get_connection("user_model") as conn:
            for _ in range(9):
                conn.execute(
                    "INSERT INTO mood_history (timestamp, energy_level, stress_level, "
                    "emotional_valence, confidence, trend) VALUES (?, 0.9, 0.1, 0.9, 0.8, 'stable')",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            # Recent 3 (inserted last → newer): NULL fields → fall back to 0.5
            for _ in range(3):
                conn.execute(
                    "INSERT INTO mood_history (timestamp, confidence, trend) "
                    "VALUES (?, 0.5, 'stable')",
                    (datetime.now(timezone.utc).isoformat(),),
                )
        # Should not raise; NULL → 0.5 defaults are applied
        insights = engine._mood_trend_insights()
        # NULL rows: composite = 0.5+0.5-0.5 = 0.5
        # Baseline rows: composite = 0.9+0.9-0.1 = 1.7  → delta = 0.5-1.7 = -1.2 → declining
        assert len(insights) == 1
        assert insights[0].entity == "declining"


# =============================================================================
# Integration: wired into generate_insights()
# =============================================================================


class TestTemporalMoodInsightsIntegration:
    """Verify the new correlators are wired into the generate_insights() loop."""

    @pytest.mark.asyncio
    async def test_temporal_insights_appear_in_generate_insights(self, db):
        """generate_insights() returns temporal_pattern insights when data present."""
        ums = _make_ums(db)
        # Provide a clear early-bird pattern with enough samples
        hours = {str(h): 20 for h in range(5, 11)}
        hours.update({str(h): 3 for h in range(18, 23)})
        hours.update({str(h): 8 for h in range(11, 18)})  # daytime for peak_hour
        # Make hour 9 the peak
        hours["9"] = 40
        days = {
            "monday": 30, "tuesday": 80, "wednesday": 25,
            "thursday": 22, "friday": 20, "saturday": 10, "sunday": 8,
        }
        _set_temporal_profile(ums, activity_by_hour=hours, activity_by_day=days,
                              samples_count=300)
        engine = InsightEngine(db=db, ums=ums)
        insights = await engine.generate_insights()
        types = {i.type for i in insights}
        assert "temporal_pattern" in types, (
            "temporal_pattern insights should appear in generate_insights() output"
        )

    @pytest.mark.asyncio
    async def test_mood_trend_appears_in_generate_insights(self, db):
        """generate_insights() returns mood_trend insight when improving trend present."""
        engine = _make_engine(db)
        # Baseline first (older), recent last (newer)
        for _ in range(9):
            _insert_mood_row(db, energy=0.3, stress=0.8, valence=0.3)
        for _ in range(3):
            _insert_mood_row(db, energy=0.9, stress=0.1, valence=0.9)
        insights = await engine.generate_insights()
        types = {i.type for i in insights}
        assert "mood_trend" in types, (
            "mood_trend insight should appear in generate_insights() output"
        )

    @pytest.mark.asyncio
    async def test_no_crash_when_all_profiles_missing(self, db):
        """generate_insights() runs without error even with no signal data."""
        engine = _make_engine(db)
        insights = await engine.generate_insights()
        # Should return an empty list or only other correlator insights, no exception
        assert isinstance(insights, list)
