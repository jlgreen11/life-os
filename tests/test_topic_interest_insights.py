"""
Tests for the InsightEngine ``_topic_interest_insights`` correlator.

The correlator reads the ``topics`` signal profile (built by TopicExtractor
from email/message events) and surfaces up to two insight sub-types:

    top_interests   -- User's five most-mentioned topics by all-time frequency
    trending_topic  -- A topic whose recent-window rate is ≥ 2× the historical rate

This test suite validates:

- Returns empty list when topics profile does not exist
- Returns empty list when samples_count is below MIN_SAMPLES (20)
- Returns empty list when topic_counts is empty despite sufficient samples
- top_interests fires with correct summary when profile has enough data
- top_interests includes only topics with count >= MIN_TOP_COUNT (5)
- top_interests entity fingerprint uses top-3 topic names (re-surfaces on shift)
- top_interests confidence is capped at 0.80
- No trending_topic insight when recent_topics has fewer than MIN_RECENT (20) entries
- No trending_topic when no topic meets the TRENDING_RATIO (2.0×) threshold
- No trending_topic when candidate topic appears fewer than MIN_TRENDING_COUNT (3) times
- trending_topic fires when one topic is clearly trending (ratio >= 2.0×, count >= 3)
- trending_topic entity is the trending topic name (per-topic dedup lifecycle)
- Both sub-types can fire together in one call
- New correlator is wired into generate_insights()
- top_interests and trending_topic categories are handled by _apply_source_weights
"""

from __future__ import annotations

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.source_weights import SourceWeightManager
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
                        samples_count: int = 60) -> None:
    """Write a topics signal profile with the given data.

    Calls update_signal_profile() ``samples_count`` times so the stored
    samples_count column matches the requested value.

    Args:
        ums: UserModelStore to write into.
        topic_counts: All-time keyword → frequency mapping.
        recent_topics: Ring-buffer entries.  Defaults to empty list if None.
        samples_count: Number of synthetic update calls to issue.
    """
    data = {
        "topic_counts": topic_counts,
        "recent_topics": recent_topics if recent_topics is not None else [],
    }
    for _ in range(samples_count):
        ums.update_signal_profile("topics", data)


def _make_recent_entries(topics_per_entry: list[str], count: int) -> list[dict]:
    """Return a list of ring-buffer entries each containing the same topics list.

    Args:
        topics_per_entry: List of topic strings for each entry.
        count: Number of entries to generate.
    """
    return [{"topics": topics_per_entry, "timestamp": "2026-01-01T10:00:00Z"}
            for _ in range(count)]


# =============================================================================
# Tests: No profile / insufficient data
# =============================================================================


def test_no_insight_when_profile_absent(db):
    """No insights when the topics profile has never been written."""
    engine = _make_engine(db)
    insights = engine._topic_interest_insights()
    assert insights == []


def test_no_insight_when_below_min_samples(db):
    """No insights when profile exists but has fewer than MIN_SAMPLES (20) updates."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {"work": 100, "email": 80}, samples_count=15)
    insights = engine._topic_interest_insights()
    assert insights == []


def test_no_insight_when_topic_counts_empty(db):
    """No insights when samples_count >= 20 but topic_counts dict is empty."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {}, samples_count=25)
    insights = engine._topic_interest_insights()
    assert insights == []


# =============================================================================
# Tests: top_interests sub-insight
# =============================================================================


def test_top_interests_fires_with_sufficient_data(db):
    """top_interests insight fires when profile has >= 20 samples and >= 5 counts."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {
        "work": 200, "project": 150, "team": 100, "email": 80, "meeting": 60,
        "finance": 40,
    }, samples_count=60)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1

    summary = top[0].summary
    # Top-5 topics should be mentioned in the summary
    assert "work" in summary
    assert "project" in summary
    assert "team" in summary
    assert "email" in summary
    assert "meeting" in summary
    # Total observation count should appear
    assert "630" in summary.replace(",", "")


def test_top_interests_excludes_topics_below_min_count(db):
    """Topics with < 5 occurrences are not included in the top-interests summary."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {
        "work": 50, "project": 30, "team": 20, "email": 10, "meeting": 8,
        # These should be excluded (count < 5)
        "rare": 2, "single": 1,
    }, samples_count=60)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1
    summary = top[0].summary
    assert "rare" not in summary
    assert "single" not in summary


def test_top_interests_entity_fingerprint_uses_top3(db):
    """top_interests entity is the concatenation of the top-3 topic names."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {
        "alpha": 300, "beta": 200, "gamma": 100, "delta": 50,
    }, samples_count=60)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1
    # Entity must be derived from top-3 names in order
    assert top[0].entity == "alpha_beta_gamma"


def test_top_interests_confidence_capped_at_0_80(db):
    """top_interests confidence never exceeds 0.80 regardless of sample count."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # 500 samples would push 0.50 + 500*0.001 = 1.0, but cap should apply.
    _set_topics_profile(ums, {"work": 1000, "email": 800, "team": 600},
                        samples_count=500)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1
    assert top[0].confidence <= 0.80


def test_top_interests_no_insight_when_all_below_min_count(db):
    """No top_interests insight when all topics have count < MIN_TOP_COUNT (5)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {"work": 3, "email": 2, "team": 1}, samples_count=60)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 0


# =============================================================================
# Tests: trending_topic sub-insight
# =============================================================================


def test_no_trending_when_recent_topics_below_min(db):
    """No trending_topic when fewer than MIN_RECENT (20) ring-buffer entries."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(
        ums,
        {"work": 200, "budget": 10},
        # Only 15 recent entries — below MIN_RECENT (20)
        recent_topics=_make_recent_entries(["budget"], 15),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 0


def test_no_trending_when_ratio_below_threshold(db):
    """No trending_topic when the best-candidate ratio is below TRENDING_RATIO (2.0)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # budget appears 4 times in 20 recent entries (rate=0.20).
    # All-time: budget=1000, total=5000 → historical_rate=0.20.  ratio=1.0 < 2.0.
    _set_topics_profile(
        ums,
        {"work": 4000, "budget": 1000},
        recent_topics=(
            _make_recent_entries(["work"], 16) +
            _make_recent_entries(["budget"], 4)
        ),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 0


def test_no_trending_when_count_below_min_trending_count(db):
    """No trending_topic when the candidate appears < MIN_TRENDING_COUNT (3) times."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # budget: historical_rate = 10/1010 ≈ 0.0099.
    # recent: 2 appearances in last 20 entries → rate=0.10.  ratio ≈ 10.1 >= 2.0.
    # But count=2 < MIN_TRENDING_COUNT=3, so it should be skipped.
    _set_topics_profile(
        ums,
        {"work": 1000, "budget": 10},
        recent_topics=(
            _make_recent_entries(["work"], 18) +
            _make_recent_entries(["budget"], 2)
        ),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 0


def test_trending_topic_fires_when_ratio_exceeds_threshold(db):
    """trending_topic fires when a topic is >= 2× more frequent in the recent window."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # budget: historical = 50/2050 ≈ 0.0244.
    # recent window (last 20): 10 budget + 10 work → rate = 0.50.
    # ratio ≈ 20.5 >= 2.0.  count=10 >= 3.
    _set_topics_profile(
        ums,
        {"work": 2000, "budget": 50},
        recent_topics=(
            _make_recent_entries(["work"], 5) +
            _make_recent_entries(["budget"], 10) +
            _make_recent_entries(["work"], 10)
        ),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 1
    assert trending[0].entity == "budget"
    assert "budget" in trending[0].summary
    assert "above your usual rate" in trending[0].summary


def test_trending_topic_entity_is_topic_name(db):
    """trending_topic entity equals the trending topic name (per-topic dedup key)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(
        ums,
        {"work": 1000, "taxes": 20},
        recent_topics=(
            _make_recent_entries(["work"], 5) +
            _make_recent_entries(["taxes"], 10) +
            _make_recent_entries(["work"], 10)
        ),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 1
    assert trending[0].entity == "taxes"


def test_trending_topic_staleness_ttl_48h(db):
    """trending_topic uses a 48-hour staleness TTL (trends shift quickly)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(
        ums,
        {"work": 1000, "ai": 20},
        recent_topics=(
            _make_recent_entries(["work"], 5) +
            _make_recent_entries(["ai"], 10) +
            _make_recent_entries(["work"], 10)
        ),
        samples_count=25,
    )

    insights = engine._topic_interest_insights()
    trending = [i for i in insights if i.category == "trending_topic"]
    assert len(trending) == 1
    assert trending[0].staleness_ttl_hours == 48


def test_top_interests_staleness_ttl_168h(db):
    """top_interests uses a 168-hour (7-day) staleness TTL."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {"work": 100, "email": 80, "team": 60},
                        samples_count=60)

    insights = engine._topic_interest_insights()
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) == 1
    assert top[0].staleness_ttl_hours == 168


# =============================================================================
# Tests: Both sub-types together
# =============================================================================


def test_both_subtypes_can_fire_together(db):
    """top_interests and trending_topic can both be returned in one call."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(
        ums,
        {"work": 2000, "email": 1000, "team": 500, "project": 400, "meeting": 300,
         "budget": 30},
        recent_topics=(
            _make_recent_entries(["work"], 5) +
            _make_recent_entries(["budget"], 10) +
            _make_recent_entries(["work"], 10)
        ),
        samples_count=100,
    )

    insights = engine._topic_interest_insights()
    categories = {i.category for i in insights}
    assert "top_interests" in categories
    assert "trending_topic" in categories


# =============================================================================
# Tests: Integration — wired into generate_insights()
# =============================================================================


def test_correlator_wired_into_generate_insights(db):
    """_topic_interest_insights is called as part of generate_insights()."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    _set_topics_profile(ums, {"work": 200, "email": 150, "team": 100,
                               "project": 80, "meeting": 70},
                        samples_count=60)

    import asyncio
    insights = asyncio.run(engine.generate_insights())
    top = [i for i in insights if i.category == "top_interests"]
    assert len(top) >= 1


# =============================================================================
# Tests: Source weight mapping
# =============================================================================


def test_source_weight_mapping_top_interests(db):
    """top_interests and trending_topic categories are handled by _apply_source_weights."""
    from services.insight_engine.source_weights import SourceWeightManager

    ums = UserModelStore(db)
    swm = SourceWeightManager(db)
    engine = InsightEngine(db=db, ums=ums, source_weight_manager=swm)

    _set_topics_profile(
        ums,
        {"work": 2000, "email": 1000, "team": 500, "project": 400, "meeting": 300,
         "budget": 30},
        recent_topics=(
            _make_recent_entries(["work"], 5) +
            _make_recent_entries(["budget"], 10) +
            _make_recent_entries(["work"], 10)
        ),
        samples_count=100,
    )

    raw = engine._topic_interest_insights()
    # _apply_source_weights should handle these categories without dropping insights
    # (default weights are 1.0, so confidence should be unchanged at this level)
    weighted = engine._apply_source_weights(raw)
    # All insights should survive with default weights (confidence still >= 0.1)
    assert len(weighted) == len(raw)
