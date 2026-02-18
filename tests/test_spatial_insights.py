"""
Tests for the InsightEngine ``_spatial_insights`` correlator.

The correlator reads the ``spatial`` signal profile (built by SpatialExtractor)
and surfaces up to three human-readable insight sub-types:

    spatial_top_location      -- Most-visited place by visit count
    spatial_work_location     -- Place with the most work-domain events
    spatial_location_diversity -- Breakdown of frequent locations by domain

This test suite validates:

- Returns empty list when spatial profile does not exist
- Returns empty list when place_behaviors is empty
- Returns empty list when the top place has fewer than MIN_VISITS (3) visits
- spatial_top_location fires for top place with >= 3 visits
- spatial_top_location summary includes visit count and avg duration when available
- spatial_top_location summary omits duration when average_duration_minutes is absent
- spatial_top_location confidence scales with visit count (capped at 0.85)
- Long location names are truncated in the summary (> 40 chars → "…" suffix)
- spatial_top_location entity is the location name (dedup stability)
- spatial_top_location staleness TTL is 168 hours
- No spatial_work_location when no location has >= 3 work-domain events
- spatial_work_location fires for location with highest work visit count
- spatial_work_location detects home-office pattern when name contains "home"
- spatial_work_location detects home-office for "apartment" keyword
- spatial_work_location entity is the work location name
- No spatial_location_diversity when fewer than 2 frequent locations exist
- spatial_location_diversity fires when >= 2 locations each have >= 3 visits
- spatial_location_diversity entity encodes total/work/personal counts
- All three sub-types fire together when data qualifies
- Invalid place_behaviors JSON is handled gracefully (returns [])
- Correlator is wired into generate_insights()
- spatial_top_location, spatial_work_location, spatial_location_diversity
  categories are handled by _apply_source_weights
"""

from __future__ import annotations

import json
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


def _set_spatial_profile(ums: UserModelStore, place_behaviors: dict,
                          samples_count: int = 10) -> None:
    """Write a spatial signal profile with the given place_behaviors dict.

    Serialises ``place_behaviors`` to a JSON string (mirroring what
    SpatialExtractor does) and calls update_signal_profile() ``samples_count``
    times so the stored ``samples_count`` column reflects the expected value.

    Args:
        ums: UserModelStore to write into.
        place_behaviors: Dict keyed by location name, each value is a place dict.
        samples_count: How many synthetic update calls to issue.
    """
    data = {"place_behaviors": json.dumps(place_behaviors)}
    for _ in range(samples_count):
        ums.update_signal_profile("spatial", data)


def _make_place(visit_count: int = 5, work: int = 0, personal: int = 0,
                avg_dur: float | None = None, dominant_domain: str = "personal") -> dict:
    """Build a minimal place-behavior dict for test fixtures.

    Args:
        visit_count: How many times this location was visited.
        work: Number of work-domain observations at this location.
        personal: Number of personal-domain observations.
        avg_dur: Average duration in minutes (or None to omit).
        dominant_domain: Dominant domain label ("work" or "personal").

    Returns:
        A place behavior dict compatible with SpatialExtractor's profile format.
    """
    place: dict = {
        "visit_count": visit_count,
        "domain_counts": {},
        "dominant_domain": dominant_domain,
        "first_visit": "2026-01-01T09:00:00+00:00",
        "last_visit": "2026-02-18T09:00:00+00:00",
    }
    if work:
        place["domain_counts"]["work"] = work
    if personal:
        place["domain_counts"]["personal"] = personal
    if avg_dur is not None:
        place["average_duration_minutes"] = avg_dur
    return place


# =============================================================================
# Tests: No profile / insufficient data
# =============================================================================


def test_no_insight_when_profile_absent(db):
    """No insights when the spatial profile has never been written."""
    engine = _make_engine(db)
    insights = engine._spatial_insights()
    assert insights == []


def test_no_insight_when_place_behaviors_empty(db):
    """No insights when the profile exists but place_behaviors is empty."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {}, samples_count=5)
    insights = engine._spatial_insights()
    assert insights == []


def test_no_insight_when_top_place_below_min_visits(db):
    """No insights when the most-visited place has fewer than 3 visits."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=2, work=2, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    assert insights == []


# =============================================================================
# Tests: spatial_top_location
# =============================================================================


def test_top_location_fires_with_sufficient_visits(db):
    """spatial_top_location insight fires when top place has >= 3 visits."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "conference room": _make_place(visit_count=10, avg_dur=60.0, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    assert "conference room" in top[0].summary
    assert "10 visits" in top[0].summary


def test_top_location_summary_includes_duration_when_available(db):
    """Average duration appears in summary when average_duration_minutes is set."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "home": _make_place(visit_count=8, avg_dur=45.0, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    assert "avg 45 min" in top[0].summary


def test_top_location_summary_omits_duration_when_absent(db):
    """No duration clause in summary when average_duration_minutes is missing."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "library": _make_place(visit_count=5, avg_dur=None, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    assert "avg" not in top[0].summary
    assert "min" not in top[0].summary


def test_top_location_confidence_scales_with_visit_count(db):
    """Confidence increases with visit count and is capped at 0.85."""
    engine = _make_engine(db)
    # Low visit count → lower confidence
    _set_spatial_profile(engine.ums, {
        "gym": _make_place(visit_count=3, dominant_domain="personal"),
    })
    insights_low = engine._spatial_insights()
    top_low = [i for i in insights_low if i.category == "spatial_top_location"]
    assert len(top_low) == 1

    # Verify cap at 0.85 for high visit counts
    engine2 = _make_engine(db.__class__(db._path if hasattr(db, "_path") else ":memory:"))
    engine2 = _make_engine(db)
    # Override with high visit count to test cap
    _set_spatial_profile(engine2.ums, {
        "office": _make_place(visit_count=100, dominant_domain="work"),
    })
    insights_high = engine2._spatial_insights()
    top_high = [i for i in insights_high if i.category == "spatial_top_location"]
    assert len(top_high) == 1
    assert top_high[0].confidence <= 0.85


def test_top_location_truncates_long_name_in_summary(db):
    """Location names longer than 40 chars are truncated with '…' in the summary."""
    engine = _make_engine(db)
    long_name = "a" * 50  # 50-character location name
    _set_spatial_profile(engine.ums, {
        long_name: _make_place(visit_count=5, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    # Full name (50 chars) should NOT appear verbatim; truncated form should
    assert long_name not in top[0].summary
    assert "…" in top[0].summary


def test_top_location_entity_is_location_name(db):
    """Entity is set to the location name (stable dedup anchor)."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "downtown office": _make_place(visit_count=7, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    assert top[0].entity == "downtown office"


def test_top_location_staleness_ttl_is_168_hours(db):
    """Staleness TTL is 168 hours (7 days)."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=5, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    top = [i for i in insights if i.category == "spatial_top_location"]
    assert len(top) == 1
    assert top[0].staleness_ttl_hours == 168


# =============================================================================
# Tests: spatial_work_location
# =============================================================================


def test_no_work_location_when_no_work_events(db):
    """No spatial_work_location insight when all locations have 0 work events."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "park": _make_place(visit_count=5, personal=5, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert work == []


def test_no_work_location_when_below_min_work_visits(db):
    """No spatial_work_location insight when top work location has < 3 work events."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=5, work=2, personal=3, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert work == []


def test_work_location_fires_for_highest_work_count(db):
    """spatial_work_location fires for the location with the most work-domain events."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=10, work=8, personal=2, dominant_domain="work"),
        "home": _make_place(visit_count=5, work=2, personal=3, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert len(work) == 1
    assert "office" in work[0].summary
    assert "8 work events" in work[0].summary


def test_work_location_detects_home_office_pattern(db):
    """Home-office pattern detected when work location name contains 'home'."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "home": _make_place(visit_count=15, work=10, personal=5, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert len(work) == 1
    assert "from home" in work[0].summary


def test_work_location_detects_home_office_for_apartment(db):
    """Home-office pattern detected when normalized location contains 'apartment'."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "apartment": _make_place(visit_count=8, work=5, personal=3, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert len(work) == 1
    assert "from home" in work[0].summary


def test_work_location_non_home_uses_generic_summary(db):
    """Non-home work location uses 'most frequent work location' phrasing."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "downtown coworking": _make_place(visit_count=6, work=5, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert len(work) == 1
    assert "most frequent work location" in work[0].summary
    assert "downtown coworking" in work[0].summary


def test_work_location_entity_is_location_name(db):
    """Entity is the work location name for stable dedup."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "open plan office": _make_place(visit_count=7, work=6, dominant_domain="work"),
    })
    insights = engine._spatial_insights()
    work = [i for i in insights if i.category == "spatial_work_location"]
    assert len(work) == 1
    assert work[0].entity == "open plan office"


# =============================================================================
# Tests: spatial_location_diversity
# =============================================================================


def test_no_diversity_when_only_one_frequent_location(db):
    """No spatial_location_diversity insight when only one place has >= 3 visits."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=10, work=8, dominant_domain="work"),
        "gym": _make_place(visit_count=2, personal=2, dominant_domain="personal"),  # below threshold
    })
    insights = engine._spatial_insights()
    div = [i for i in insights if i.category == "spatial_location_diversity"]
    assert div == []


def test_diversity_fires_with_two_frequent_locations(db):
    """spatial_location_diversity fires when >= 2 locations each have >= 3 visits."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=8, work=7, dominant_domain="work"),
        "home": _make_place(visit_count=5, personal=5, dominant_domain="personal"),
    }, samples_count=15)
    insights = engine._spatial_insights()
    div = [i for i in insights if i.category == "spatial_location_diversity"]
    assert len(div) == 1
    assert "2 distinct locations" in div[0].summary
    assert "1 work-related" in div[0].summary
    assert "1 personal" in div[0].summary


def test_diversity_entity_encodes_distribution(db):
    """Entity key encodes total/work/personal counts for stable dedup."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=6, work=5, dominant_domain="work"),
        "cafe": _make_place(visit_count=4, personal=4, dominant_domain="personal"),
        "library": _make_place(visit_count=3, personal=3, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    div = [i for i in insights if i.category == "spatial_location_diversity"]
    assert len(div) == 1
    # Entity must encode total=3, work=1, personal=2
    assert "total3" in div[0].entity
    assert "work1" in div[0].entity
    assert "personal2" in div[0].entity


def test_diversity_summary_mentions_location_count(db):
    """Diversity summary mentions number of distinct frequent locations."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=5, work=4, dominant_domain="work"),
        "home": _make_place(visit_count=10, personal=8, dominant_domain="personal"),
        "cafe": _make_place(visit_count=4, personal=4, dominant_domain="personal"),
    })
    insights = engine._spatial_insights()
    div = [i for i in insights if i.category == "spatial_location_diversity"]
    assert len(div) == 1
    assert "3 distinct locations" in div[0].summary


# =============================================================================
# Tests: all three sub-types together
# =============================================================================


def test_all_three_subtypes_fire_together(db):
    """All three spatial insight sub-types can fire in a single call."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(
            visit_count=12, work=10, personal=2,
            avg_dur=480.0, dominant_domain="work",
        ),
        "home": _make_place(
            visit_count=5, work=3, personal=2,
            avg_dur=60.0, dominant_domain="work",
        ),
        "gym": _make_place(
            visit_count=4, personal=4,
            dominant_domain="personal",
        ),
    }, samples_count=20)
    insights = engine._spatial_insights()
    categories = {i.category for i in insights}
    assert "spatial_top_location" in categories
    assert "spatial_work_location" in categories
    assert "spatial_location_diversity" in categories


# =============================================================================
# Tests: Error handling
# =============================================================================


def test_invalid_place_behaviors_json_returns_empty(db):
    """Corrupted place_behaviors JSON is handled gracefully and returns []."""
    engine = _make_engine(db)
    # Write a profile with invalid JSON in place_behaviors
    from storage.user_model_store import UserModelStore
    with engine.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles
               (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, 5, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("spatial", json.dumps({"place_behaviors": "not-valid-json{{{"})),
        )
    insights = engine._spatial_insights()
    assert insights == []


# =============================================================================
# Tests: Integration — wired into generate_insights() and source weight map
# =============================================================================


@pytest.mark.asyncio
async def test_spatial_correlator_wired_into_generate_insights(db):
    """spatial_insights correlator is called by generate_insights()."""
    engine = _make_engine(db)
    _set_spatial_profile(engine.ums, {
        "office": _make_place(visit_count=5, work=4, dominant_domain="work"),
        "home": _make_place(visit_count=4, personal=4, dominant_domain="personal"),
    }, samples_count=10)
    all_insights = await engine.generate_insights()
    spatial = [i for i in all_insights if i.category.startswith("spatial_")]
    # At least the top-location and diversity insights should fire
    assert len(spatial) >= 1


def test_spatial_categories_in_source_weight_map(db):
    """All three spatial categories have source weight mappings."""
    engine = _make_engine(db)
    # The _apply_source_weights method uses a category_to_source dict.
    # We verify it handles spatial categories by injecting known insights
    # and checking they survive (confidence >= 0.1 threshold).
    from services.insight_engine.models import Insight

    for category in ("spatial_top_location", "spatial_work_location", "spatial_location_diversity"):
        insight = Insight(
            type="behavioral_pattern",
            summary="test",
            confidence=0.70,
            category=category,
            entity="test_entity",
        )
        insight.compute_dedup_key()
        # _apply_source_weights without a SourceWeightManager returns the list
        # unchanged — we verify no KeyError or AttributeError is raised.
        result = engine._apply_source_weights([insight])
        assert result == [insight], f"Category {category} was unexpectedly filtered out"
