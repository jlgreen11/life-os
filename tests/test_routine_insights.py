"""
Tests for the InsightEngine ``_routine_insights`` correlator and the
``_routine_trigger_label`` helper.

The correlator reads the ``routines`` table (populated by RoutineDetector)
and surfaces one ``behavioral_pattern`` insight per qualifying routine.
A routine qualifies when ``times_observed >= 3`` AND
``consistency_score >= 0.70``.

This test suite validates:

- Returns empty list when no routines are stored
- Returns empty list when all routines are below the observation threshold
- Returns empty list when all routines are below the consistency threshold
- Generates one insight per qualifying routine
- Skips routines that fail either threshold independently
- Confidence scales with consistency_score (min 0.85 cap)
- Insight type is always "behavioral_pattern"
- Insight category is always "routine_pattern"
- Entity is set to the routine name (for dedup stability)
- Evidence contains consistency_score, times_observed, steps_count, trigger
- Summary includes step count, times observed, consistency percent, and duration
- Summary omits duration when typical_duration_minutes == 0
- Dedup key is stable across runs (same name → same key)
- Staleness TTL is 168 hours (7 days)
- _routine_trigger_label: "morning" → "morning routine"
- _routine_trigger_label: "midday" → "midday routine"
- _routine_trigger_label: "afternoon" → "afternoon routine"
- _routine_trigger_label: "evening" → "evening routine"
- _routine_trigger_label: "night" → "night routine"
- _routine_trigger_label: "arrive_home" → "arrival routine at home"
- _routine_trigger_label: "arrive_coffee_shop" → "arrival routine at coffee shop"
- _routine_trigger_label: "after_meeting" → "post-meeting routine"
- _routine_trigger_label: "after_long_call" → "post-long call routine"
- _routine_trigger_label: unknown trigger → "<trigger> routine"
- routine correlator is wired into generate_insights()
- "routine_pattern" category is handled by _apply_source_weights
"""

from __future__ import annotations

import pytest

from services.insight_engine.engine import InsightEngine, _routine_trigger_label
from storage.user_model_store import UserModelStore


# =============================================================================
# Helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    """Return an InsightEngine wired to the temp DatabaseManager."""
    ums = UserModelStore(db)
    return InsightEngine(db=db, ums=ums)


def _store_routine(ums: UserModelStore, **kwargs) -> None:
    """Write a routine into the store with sensible defaults.

    Args:
        ums: UserModelStore to write into.
        **kwargs: Override any routine field.
    """
    routine = {
        "name": "Morning routine",
        "trigger": "morning",
        "steps": [
            {"action": "check_email", "duration_minutes": 10.0},
            {"action": "review_calendar", "duration_minutes": 5.0},
        ],
        "typical_duration_minutes": 30.0,
        "consistency_score": 0.80,
        "times_observed": 10,
        "variations": [],
    }
    routine.update(kwargs)
    ums.store_routine(routine)


# =============================================================================
# _routine_trigger_label helper
# =============================================================================


class TestRoutineTriggerLabel:
    """Unit tests for the ``_routine_trigger_label`` module-level helper."""

    def test_morning(self):
        assert _routine_trigger_label("morning") == "morning routine"

    def test_midday(self):
        assert _routine_trigger_label("midday") == "midday routine"

    def test_afternoon(self):
        assert _routine_trigger_label("afternoon") == "afternoon routine"

    def test_evening(self):
        assert _routine_trigger_label("evening") == "evening routine"

    def test_night(self):
        assert _routine_trigger_label("night") == "night routine"

    def test_arrive_home(self):
        assert _routine_trigger_label("arrive_home") == "arrival routine at home"

    def test_arrive_coffee_shop(self):
        assert _routine_trigger_label("arrive_coffee_shop") == "arrival routine at coffee shop"

    def test_after_meeting(self):
        assert _routine_trigger_label("after_meeting") == "post-meeting routine"

    def test_after_long_call(self):
        assert _routine_trigger_label("after_long_call") == "post-long call routine"

    def test_unknown_trigger_falls_back(self):
        """Unrecognised triggers become '<trigger> routine' with underscores replaced."""
        assert _routine_trigger_label("custom_label") == "custom label routine"

    def test_empty_string(self):
        """Empty trigger string should not crash."""
        label = _routine_trigger_label("")
        assert isinstance(label, str)


# =============================================================================
# _routine_insights correlator
# =============================================================================


class TestRoutineInsights:
    """Tests for InsightEngine._routine_insights()."""

    def test_no_routines_returns_empty(self, db):
        """When no routines exist the correlator returns an empty list."""
        engine = _make_engine(db)
        results = engine._routine_insights()
        assert results == []

    def test_below_observation_threshold_skipped(self, db):
        """Routines with times_observed < 3 are not surfaced."""
        ums = UserModelStore(db)
        _store_routine(ums, times_observed=2, consistency_score=0.90)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results == []

    def test_below_consistency_threshold_skipped(self, db):
        """Routines with consistency_score < 0.70 are not surfaced."""
        ums = UserModelStore(db)
        _store_routine(ums, times_observed=15, consistency_score=0.65)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results == []

    def test_qualifying_routine_generates_insight(self, db):
        """A routine that meets both thresholds produces exactly one insight."""
        ums = UserModelStore(db)
        _store_routine(ums, times_observed=10, consistency_score=0.80)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert len(results) == 1

    def test_multiple_routines_each_get_insight(self, db):
        """Two qualifying routines produce two insights."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Morning routine", trigger="morning",
                       times_observed=10, consistency_score=0.80)
        _store_routine(ums, name="Evening routine", trigger="evening",
                       times_observed=7, consistency_score=0.75)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert len(results) == 2

    def test_only_qualifying_routines_surface(self, db):
        """Mix of qualifying and non-qualifying: only qualifying ones surface."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Good routine", trigger="morning",
                       times_observed=10, consistency_score=0.80)
        # Below consistency
        _store_routine(ums, name="Weak routine", trigger="evening",
                       times_observed=10, consistency_score=0.50)
        # Below observation count
        _store_routine(ums, name="Rare routine", trigger="midday",
                       times_observed=1, consistency_score=0.90)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert len(results) == 1
        assert results[0].entity == "Good routine"

    def test_insight_type_is_behavioral_pattern(self, db):
        """Routine insights always have type='behavioral_pattern'."""
        ums = UserModelStore(db)
        _store_routine(ums)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results[0].type == "behavioral_pattern"

    def test_insight_category_is_routine_pattern(self, db):
        """Routine insights always have category='routine_pattern'."""
        ums = UserModelStore(db)
        _store_routine(ums)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results[0].category == "routine_pattern"

    def test_entity_is_routine_name(self, db):
        """Entity is set to the routine name for stable dedup keying."""
        ums = UserModelStore(db)
        _store_routine(ums, name="My Morning Routine")
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results[0].entity == "My Morning Routine"

    def test_staleness_ttl_is_seven_days(self, db):
        """Staleness TTL is 168 hours (7 days) — routines shift slowly."""
        ums = UserModelStore(db)
        _store_routine(ums)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results[0].staleness_ttl_hours == 168

    def test_confidence_scales_with_consistency(self, db):
        """Confidence = min(0.85, 0.50 + consistency * 0.40)."""
        ums = UserModelStore(db)
        _store_routine(ums, name="High consistency", consistency_score=0.90)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        expected = min(0.85, 0.50 + 0.90 * 0.40)
        assert abs(results[0].confidence - expected) < 0.001

    def test_confidence_capped_at_0_85(self, db):
        """Perfect consistency (1.0) caps confidence at 0.85."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Perfect routine", consistency_score=1.0)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results[0].confidence <= 0.85

    def test_evidence_contains_key_fields(self, db):
        """Evidence list includes consistency_score, times_observed, steps_count, trigger."""
        ums = UserModelStore(db)
        _store_routine(ums, trigger="morning", consistency_score=0.80, times_observed=10)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        evidence = results[0].evidence
        assert any("consistency_score=0.80" in e for e in evidence)
        assert any("times_observed=10" in e for e in evidence)
        assert any("steps_count=" in e for e in evidence)
        assert any("trigger=morning" in e for e in evidence)

    def test_summary_contains_step_count(self, db):
        """Summary mentions the number of steps."""
        ums = UserModelStore(db)
        _store_routine(ums, steps=[
            {"action": "a"}, {"action": "b"}, {"action": "c"}
        ])
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "3 steps" in results[0].summary

    def test_summary_contains_times_observed(self, db):
        """Summary mentions how many times the routine was observed."""
        ums = UserModelStore(db)
        _store_routine(ums, times_observed=14)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "14 times" in results[0].summary

    def test_summary_contains_consistency_percent(self, db):
        """Summary shows consistency as an integer percentage."""
        ums = UserModelStore(db)
        _store_routine(ums, consistency_score=0.87)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "87%" in results[0].summary

    def test_summary_contains_duration(self, db):
        """Summary includes approximate duration when typical_duration_minutes > 0."""
        ums = UserModelStore(db)
        _store_routine(ums, typical_duration_minutes=28.0)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "28 min" in results[0].summary

    def test_summary_omits_duration_when_zero(self, db):
        """Summary omits duration when typical_duration_minutes is 0."""
        ums = UserModelStore(db)
        _store_routine(ums, typical_duration_minutes=0.0)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "min" not in results[0].summary

    def test_summary_contains_trigger_label(self, db):
        """Summary uses the human-readable trigger label (e.g. 'morning routine')."""
        ums = UserModelStore(db)
        _store_routine(ums, trigger="morning")
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert "morning routine" in results[0].summary

    def test_dedup_key_is_stable(self, db):
        """Same routine name → same dedup_key across two engine instances."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Stable routine")
        engine1 = InsightEngine(db=db, ums=ums)
        engine2 = InsightEngine(db=db, ums=ums)
        key1 = engine1._routine_insights()[0].dedup_key
        key2 = engine2._routine_insights()[0].dedup_key
        assert key1 == key2

    def test_dedup_key_differs_across_routines(self, db):
        """Different routine names produce different dedup keys."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Routine A", trigger="morning")
        _store_routine(ums, name="Routine B", trigger="evening")
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        keys = [r.dedup_key for r in results]
        assert len(set(keys)) == 2

    async def test_wired_into_generate_insights(self, db):
        """Routine insights are returned by the public generate_insights() API."""
        ums = UserModelStore(db)
        _store_routine(ums, name="Morning routine", consistency_score=0.80, times_observed=10)
        engine = InsightEngine(db=db, ums=ums)
        results = await engine.generate_insights()
        categories = [r.category for r in results]
        assert "routine_pattern" in categories

    def test_routine_pattern_in_source_weight_map(self, db):
        """'routine_pattern' category is handled by _apply_source_weights."""
        from services.insight_engine.source_weights import SourceWeightManager
        ums = UserModelStore(db)
        swm = SourceWeightManager(db)
        engine = InsightEngine(db=db, ums=ums, source_weight_manager=swm)

        # The engine reads category_to_source in _apply_source_weights; we verify
        # that a routine_pattern insight survives source-weight application with
        # non-zero confidence (i.e., it is not silently dropped for being unmapped).
        _store_routine(ums, name="Morning routine", consistency_score=0.80, times_observed=10)
        raw_insights = engine._routine_insights()
        assert len(raw_insights) == 1

        weighted = engine._apply_source_weights(raw_insights)
        # Should still be present (default weight ≥ 0.1 threshold)
        assert len(weighted) == 1
        # Confidence should still be positive
        assert weighted[0].confidence > 0.0

    def test_exact_threshold_consistency_qualifies(self, db):
        """Routine with consistency_score == 0.70 exactly should qualify."""
        ums = UserModelStore(db)
        _store_routine(ums, consistency_score=0.70, times_observed=5)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert len(results) == 1

    def test_exact_threshold_observations_qualifies(self, db):
        """Routine with times_observed == 3 exactly should qualify."""
        ums = UserModelStore(db)
        _store_routine(ums, times_observed=3, consistency_score=0.75)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert len(results) == 1

    def test_empty_name_routine_skipped(self, db):
        """Routines with empty name should be skipped (entity used for dedup)."""
        ums = UserModelStore(db)
        _store_routine(ums, name="", times_observed=10, consistency_score=0.80)
        engine = InsightEngine(db=db, ums=ums)
        results = engine._routine_insights()
        assert results == []
