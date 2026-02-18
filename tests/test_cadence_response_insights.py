"""
Tests for the InsightEngine ``_cadence_response_insights`` correlator.

The correlator reads the ``cadence`` signal profile (built by CadenceExtractor)
and surfaces up to four insight sub-types:

    response_time_baseline    -- Overall average reply latency across all contacts
    fastest_contacts          -- Contacts the user consistently replies to fastest
    communication_peak_hours  -- Top-3 most active hours of the day
    channel_cadence           -- Fastest vs slowest communication channel

This test suite validates:

- Returns empty list when cadence profile does not exist
- Returns empty list when cadence profile has no data
- response_time_baseline fires with correct summary (hours format)
- response_time_baseline fires with correct summary (minutes format for <1h avg)
- response_time_baseline requires MIN_RT_SAMPLES (10) observations
- response_time_baseline confidence scales with sample count, capped at 0.85
- response_time_baseline entity is "global_avg" (fixed, one slot)
- fastest_contacts fires for a contact with avg < 50% of global avg
- fastest_contacts respects MIN_CT_SAMPLES (3) per contact
- fastest_contacts filters marketing/no-reply addresses
- fastest_contacts surfaces at most MAX_CONTACTS (3) insights
- fastest_contacts formats duration correctly (hours vs minutes)
- communication_peak_hours fires and lists top-3 hours
- communication_peak_hours requires MIN_HOURLY (30) total counts
- communication_peak_hours entity encodes the top-hour strings
- channel_cadence fires when fastest < 50% of slowest
- channel_cadence does NOT fire when gap is less than 2×
- channel_cadence requires ≥2 channels with MIN_CT_SAMPLES each
- cadence_response correlator is wired into generate_insights()
- cadence categories are handled by _apply_source_weights
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


def _set_cadence_profile(ums: UserModelStore, data: dict, n_updates: int = 1) -> None:
    """Write a cadence signal profile with the given data.

    Args:
        ums: UserModelStore to write into.
        data: Raw profile data dict (response_times, per_contact_response_times, etc.).
        n_updates: Number of synthetic update calls (sets samples_count).
    """
    for _ in range(n_updates):
        ums.update_signal_profile("cadence", data)


def _hours(h: float) -> float:
    """Convert hours to seconds for readability in test data."""
    return h * 3600.0


# =============================================================================
# Tests: No profile / empty data
# =============================================================================


def test_no_insight_when_profile_absent(db):
    """No insights when the cadence profile has never been written."""
    engine = _make_engine(db)
    insights = engine._cadence_response_insights()
    assert insights == []


def test_no_insight_when_data_empty(db):
    """No insights when cadence profile exists but has empty data dict."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    ums.update_signal_profile("cadence", {})
    insights = engine._cadence_response_insights()
    assert insights == []


# =============================================================================
# Tests: response_time_baseline
# =============================================================================


def test_response_time_baseline_fires_hours_format(db):
    """response_time_baseline surfaces with hours format when avg >= 1h."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # 12 responses averaging 3 hours
    rts = [_hours(3.0)] * 12
    _set_cadence_profile(ums, {"response_times": rts})
    insights = engine._cadence_response_insights()
    baseline = [i for i in insights if i.category == "response_time_baseline"]
    assert len(baseline) == 1
    assert "3.0 hours" in baseline[0].summary
    assert "12" in baseline[0].summary


def test_response_time_baseline_fires_minutes_format(db):
    """response_time_baseline uses minutes format when avg < 1h."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # 10 responses averaging 30 minutes
    rts = [_hours(0.5)] * 10
    _set_cadence_profile(ums, {"response_times": rts})
    insights = engine._cadence_response_insights()
    baseline = [i for i in insights if i.category == "response_time_baseline"]
    assert len(baseline) == 1
    assert "30 minutes" in baseline[0].summary


def test_response_time_baseline_requires_min_rt_samples(db):
    """response_time_baseline does not fire with fewer than 10 response-time samples."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts = [_hours(2.0)] * 9  # one short of MIN_RT_SAMPLES
    _set_cadence_profile(ums, {"response_times": rts})
    insights = engine._cadence_response_insights()
    baseline = [i for i in insights if i.category == "response_time_baseline"]
    assert len(baseline) == 0


def test_response_time_baseline_confidence_capped(db):
    """response_time_baseline confidence is capped at 0.85."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Enough samples that naive formula would exceed cap
    rts = [_hours(2.0)] * 500
    _set_cadence_profile(ums, {"response_times": rts})
    insights = engine._cadence_response_insights()
    baseline = [i for i in insights if i.category == "response_time_baseline"]
    assert len(baseline) == 1
    assert baseline[0].confidence <= 0.85


def test_response_time_baseline_entity_is_global_avg(db):
    """response_time_baseline uses entity='global_avg' for single-slot deduplication."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts = [_hours(1.5)] * 15
    _set_cadence_profile(ums, {"response_times": rts})
    insights = engine._cadence_response_insights()
    baseline = [i for i in insights if i.category == "response_time_baseline"]
    assert len(baseline) == 1
    assert baseline[0].entity == "global_avg"


# =============================================================================
# Tests: fastest_contacts
# =============================================================================


def test_fastest_contacts_fires_for_fast_contact(db):
    """fastest_contacts fires for a contact whose avg is < 50% of global avg."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Global avg = 4h; alice avg = 1h (25% of global — well below 50% threshold)
    rts_global = [_hours(4.0)] * 15
    per_contact = {"alice@example.com": [_hours(1.0)] * 5}
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) == 1
    assert "alice@example.com" in fast[0].summary
    assert fast[0].entity == "alice@example.com"


def test_fastest_contacts_not_fire_when_ratio_above_threshold(db):
    """fastest_contacts does NOT fire when contact avg is > 50% of global avg."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Global avg = 4h; bob avg = 3h (75% of global — above 50% threshold)
    rts_global = [_hours(4.0)] * 15
    per_contact = {"bob@example.com": [_hours(3.0)] * 5}
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) == 0


def test_fastest_contacts_requires_min_ct_samples(db):
    """fastest_contacts ignores contacts with fewer than 3 response-time samples."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts_global = [_hours(4.0)] * 15
    # Only 2 samples for this contact — below MIN_CT_SAMPLES
    per_contact = {"alice@example.com": [_hours(0.5), _hours(0.5)]}
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) == 0


def test_fastest_contacts_filters_marketing_addresses(db):
    """fastest_contacts excludes no-reply and marketing addresses."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts_global = [_hours(4.0)] * 15
    per_contact = {
        "noreply@newsletter.com": [_hours(0.1)] * 5,
        "updates@service.com": [_hours(0.2)] * 5,
    }
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) == 0


def test_fastest_contacts_capped_at_max_contacts(db):
    """fastest_contacts surfaces at most 3 contacts."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts_global = [_hours(8.0)] * 20
    per_contact = {
        f"contact{i}@example.com": [_hours(0.5)] * 5
        for i in range(6)  # 6 qualifying contacts
    }
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) <= 3


def test_fastest_contacts_minutes_format_for_sub_hour(db):
    """fastest_contacts formats contact avg in minutes when < 1h."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    rts_global = [_hours(4.0)] * 15
    # 15-minute avg for alice
    per_contact = {"alice@example.com": [_hours(0.25)] * 5}
    _set_cadence_profile(ums, {
        "response_times": rts_global,
        "per_contact_response_times": per_contact,
    })
    insights = engine._cadence_response_insights()
    fast = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast) == 1
    assert "15m" in fast[0].summary


# =============================================================================
# Tests: communication_peak_hours
# =============================================================================


def test_communication_peak_hours_fires_with_top_3(db):
    """communication_peak_hours fires and lists the top-3 active hours."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Hours 9, 10, 14 are most active; total counts > MIN_HOURLY (30)
    hourly = {"9": 20, "10": 15, "14": 12, "16": 5, "8": 3}
    _set_cadence_profile(ums, {"hourly_activity": hourly})
    insights = engine._cadence_response_insights()
    peak = [i for i in insights if i.category == "communication_peak_hours"]
    assert len(peak) == 1
    assert "9:00" in peak[0].summary
    assert "10:00" in peak[0].summary
    assert "14:00" in peak[0].summary


def test_communication_peak_hours_requires_min_hourly(db):
    """communication_peak_hours does not fire when total activity < MIN_HOURLY (30)."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Only 10 total counts — below threshold
    hourly = {"9": 5, "10": 3, "14": 2}
    _set_cadence_profile(ums, {"hourly_activity": hourly})
    insights = engine._cadence_response_insights()
    peak = [i for i in insights if i.category == "communication_peak_hours"]
    assert len(peak) == 0


def test_communication_peak_hours_entity_encodes_hours(db):
    """communication_peak_hours entity encodes the top-hour strings for dedup."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    hourly = {"9": 20, "10": 15, "14": 12, "16": 5}
    _set_cadence_profile(ums, {"hourly_activity": hourly})
    insights = engine._cadence_response_insights()
    peak = [i for i in insights if i.category == "communication_peak_hours"]
    assert len(peak) == 1
    # Entity should contain the top-3 hour keys
    entity = peak[0].entity
    assert "9" in entity
    assert "10" in entity
    assert "14" in entity


# =============================================================================
# Tests: channel_cadence
# =============================================================================


def test_channel_cadence_fires_when_gap_substantial(db):
    """channel_cadence fires when fastest channel is < 50% of slowest."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    per_channel = {
        "imessage": [_hours(0.25)] * 5,      # 15-minute avg
        "proton_mail": [_hours(4.0)] * 5,    # 4-hour avg (16× faster)
    }
    _set_cadence_profile(ums, {"per_channel_response_times": per_channel})
    insights = engine._cadence_response_insights()
    ch = [i for i in insights if i.category == "channel_cadence"]
    assert len(ch) == 1
    assert "imessage" in ch[0].summary
    assert "proton_mail" in ch[0].summary


def test_channel_cadence_not_fire_when_gap_small(db):
    """channel_cadence does NOT fire when fastest is > 50% of slowest."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # imessage is 60% of email avg — gap not substantial enough
    per_channel = {
        "imessage": [_hours(1.8)] * 5,
        "proton_mail": [_hours(3.0)] * 5,
    }
    _set_cadence_profile(ums, {"per_channel_response_times": per_channel})
    insights = engine._cadence_response_insights()
    ch = [i for i in insights if i.category == "channel_cadence"]
    assert len(ch) == 0


def test_channel_cadence_requires_two_channels(db):
    """channel_cadence requires at least two channels with sufficient samples."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Only one channel
    per_channel = {"imessage": [_hours(0.5)] * 5}
    _set_cadence_profile(ums, {"per_channel_response_times": per_channel})
    insights = engine._cadence_response_insights()
    ch = [i for i in insights if i.category == "channel_cadence"]
    assert len(ch) == 0


def test_channel_cadence_requires_min_ct_samples_per_channel(db):
    """channel_cadence ignores channels with fewer than 3 response-time samples."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    per_channel = {
        "imessage": [_hours(0.25)] * 2,      # Only 2 samples — below threshold
        "proton_mail": [_hours(4.0)] * 5,
    }
    _set_cadence_profile(ums, {"per_channel_response_times": per_channel})
    insights = engine._cadence_response_insights()
    ch = [i for i in insights if i.category == "channel_cadence"]
    assert len(ch) == 0


def test_channel_cadence_entity_encodes_both_channels(db):
    """channel_cadence entity encodes 'fastest:slowest' for dedup."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    per_channel = {
        "imessage": [_hours(0.25)] * 5,
        "proton_mail": [_hours(4.0)] * 5,
    }
    _set_cadence_profile(ums, {"per_channel_response_times": per_channel})
    insights = engine._cadence_response_insights()
    ch = [i for i in insights if i.category == "channel_cadence"]
    assert len(ch) == 1
    assert "imessage" in ch[0].entity
    assert "proton_mail" in ch[0].entity


# =============================================================================
# Tests: all four sub-types together
# =============================================================================


def test_all_four_subtypes_can_fire_together(db):
    """All four sub-types fire when sufficient data is present for each."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    data = {
        "response_times": [_hours(4.0)] * 20,
        "per_contact_response_times": {
            "alice@example.com": [_hours(1.0)] * 5,  # fast contact
        },
        "hourly_activity": {"9": 15, "10": 12, "14": 10, "16": 5, "8": 3},
        "per_channel_response_times": {
            "imessage": [_hours(0.5)] * 5,
            "proton_mail": [_hours(5.0)] * 5,
        },
    }
    _set_cadence_profile(ums, data)
    insights = engine._cadence_response_insights()
    categories = {i.category for i in insights}
    assert "response_time_baseline" in categories
    assert "fastest_contacts" in categories
    assert "communication_peak_hours" in categories
    assert "channel_cadence" in categories


# =============================================================================
# Tests: integration with generate_insights() and source weight mapping
# =============================================================================


async def test_cadence_correlator_wired_into_generate_insights(db):
    """_cadence_response_insights is called by generate_insights()."""
    ums = UserModelStore(db)
    engine = InsightEngine(db=db, ums=ums)
    # Plant a cadence profile that should produce a baseline insight
    data = {
        "response_times": [_hours(3.0)] * 15,
        "hourly_activity": {"9": 20, "10": 15, "14": 10},
    }
    ums.update_signal_profile("cadence", data)

    insights = await engine.generate_insights()
    cadence_cats = {i.category for i in insights}
    # At least one cadence category must be present
    assert cadence_cats & {"response_time_baseline", "communication_peak_hours"}


def test_cadence_categories_in_source_weight_map(db):
    """All four cadence categories appear in _apply_source_weights category map."""
    ums = UserModelStore(db)
    swm = SourceWeightManager(db=db)
    engine = InsightEngine(db=db, ums=ums, source_weight_manager=swm)

    # The category_to_source dict is inline inside _apply_source_weights.
    # We verify it by checking that a cadence insight is NOT stripped when
    # source weights are applied (weight defaults to 1.0 so confidence is unchanged).
    from services.insight_engine.models import Insight

    cadence_insights = [
        Insight(type="behavioral_pattern", summary="avg reply 3h", confidence=0.7,
                category="response_time_baseline", entity="global_avg"),
        Insight(type="behavioral_pattern", summary="fast alice", confidence=0.6,
                category="fastest_contacts", entity="alice@example.com"),
        Insight(type="behavioral_pattern", summary="peak 9:00", confidence=0.7,
                category="communication_peak_hours", entity="9_10_14"),
        Insight(type="behavioral_pattern", summary="imessage fastest", confidence=0.6,
                category="channel_cadence", entity="imessage:email"),
    ]
    weighted = engine._apply_source_weights(cadence_insights)
    # All four should pass the 0.1 confidence threshold with default weights
    assert len(weighted) == 4
