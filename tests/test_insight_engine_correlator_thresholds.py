"""
Tests that InsightEngine correlator thresholds match between the diagnostic
report (``get_data_sufficiency_report``) and the actual inner MIN_SAMPLES
constants used by each correlator function.

Validates:

- _topic_interest_insights() returns insights when samples_count=25 (above new 20 threshold)
- _topic_interest_insights() returns empty when samples_count=15 (below new 20 threshold)
- _temporal_pattern_insights() returns empty when samples_count=10 (below 50 threshold)
- Diagnostic report min_required values match inner MIN_SAMPLES for key correlators
"""

from __future__ import annotations

import asyncio

import pytest

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore


# =============================================================================
# Helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    """Return an InsightEngine wired to the temp DatabaseManager."""
    ums = UserModelStore(db)
    return InsightEngine(db=db, ums=ums)


def _set_topics_profile(ums: UserModelStore, topic_counts: dict,
                        recent_topics: list | None = None,
                        samples_count: int = 25) -> None:
    """Write a topics signal profile with the given data.

    Calls update_signal_profile() ``samples_count`` times so the stored
    samples_count column matches the requested value.
    """
    data = {
        "topic_counts": topic_counts,
        "recent_topics": recent_topics if recent_topics is not None else [],
    }
    for _ in range(samples_count):
        ums.update_signal_profile("topics", data)


def _set_temporal_profile(ums: UserModelStore, data: dict,
                          samples_count: int = 10) -> None:
    """Write a temporal signal profile with the given data.

    Calls update_signal_profile() ``samples_count`` times so the stored
    samples_count column matches the requested value.
    """
    for _ in range(samples_count):
        ums.update_signal_profile("temporal", data)


# =============================================================================
# Tests: topic_interest threshold alignment
# =============================================================================


def test_topic_interest_returns_insights_at_25_samples(db):
    """_topic_interest_insights() produces insights with 25 samples (above MIN_SAMPLES=20)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {
        "work": 200, "project": 150, "team": 100, "email": 80, "meeting": 60,
    }, samples_count=25)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1, "Expected top_interests insight with 25 samples (above MIN_SAMPLES=20)"


def test_topic_interest_returns_empty_at_15_samples(db):
    """_topic_interest_insights() returns empty with 15 samples (below MIN_SAMPLES=20)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {
        "work": 200, "project": 150, "team": 100, "email": 80, "meeting": 60,
    }, samples_count=15)

    insights = engine._topic_interest_insights()
    assert insights == [], "Expected no insights with 15 samples (below MIN_SAMPLES=20)"


# =============================================================================
# Tests: temporal_pattern threshold alignment
# =============================================================================


def test_temporal_pattern_returns_empty_at_10_samples(db):
    """_temporal_pattern_insights() returns empty with 10 samples (below MIN_SAMPLES=50)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_temporal_profile(ums, {
        "activity_by_hour": {str(h): 5 for h in range(24)},
        "activity_by_day": {"Monday": 20, "Tuesday": 15, "Wednesday": 10},
    }, samples_count=10)

    insights = engine._temporal_pattern_insights()
    assert insights == [], "Expected no insights with 10 samples (below MIN_SAMPLES=50)"


# =============================================================================
# Tests: diagnostic min_required matches inner MIN_SAMPLES
# =============================================================================


def test_diagnostic_thresholds_match_inner_constants(db):
    """Diagnostic report min_required values must match the actual inner MIN_SAMPLES.

    This test prevents threshold mismatches where the diagnostic reports a
    correlator as 'ready' but the correlator's inner guard blocks execution.
    """
    engine = _make_engine(db)
    report = asyncio.run(engine.get_data_sufficiency_report())

    # Expected thresholds: (correlator_name, expected_min_required)
    expected = {
        "_topic_interest_insights": 20,
        "_temporal_pattern_insights": 50,
        "_decision_pattern_insights": 5,
    }

    for correlator_name, expected_min in expected.items():
        assert correlator_name in report, f"{correlator_name} missing from diagnostic report"
        actual_min = report[correlator_name]["min_required"]
        assert actual_min == expected_min, (
            f"{correlator_name}: diagnostic min_required={actual_min} "
            f"but inner MIN_SAMPLES={expected_min}"
        )
