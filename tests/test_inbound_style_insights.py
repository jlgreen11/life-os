"""
Tests for the InsightEngine ``_inbound_style_insights`` correlator.

The correlator reads the ``linguistic_inbound`` signal profile, which stores
per-contact averages for every inbound message the system has received
(100K+ real samples).  It compares each contact's inbound formality against
the user's outbound formality baseline and surfaces a ``communication_style``
/ ``style_mismatch`` insight when the gap exceeds 0.3.

This test suite validates:
- Returns empty list when ``linguistic_inbound`` profile does not exist
- Skips contacts with fewer than 5 samples (insufficient data)
- Skips marketing/automated senders (unfulfillable insights)
- No insight when formality gap <= 0.3 (within noise threshold)
- Correct "casually" direction when contact formality < user baseline
- Correct "formally" direction when contact formality > user baseline
- Insight confidence scales correctly with gap size
- Caps output at 10 insights sorted by gap descending
- Uses 0.5 as fallback when ``linguistic`` (outbound) profile is absent
- New correlator is wired into ``generate_insights()``
- ``style_mismatch`` category is handled by ``_apply_source_weights``
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


def _set_inbound_profile(ums: UserModelStore,
                         per_contact_averages: dict) -> None:
    """Write a ``linguistic_inbound`` profile with given per-contact averages.

    Args:
        ums: UserModelStore to write into.
        per_contact_averages: Dict mapping contact email → averages dict.
            Each averages dict should contain ``formality`` and
            ``samples_count`` at minimum.
    """
    data = {
        "per_contact": {},
        "per_contact_averages": per_contact_averages,
    }
    # update_signal_profile needs samples_count to be passed correctly;
    # set it to the number of contacts as a rough proxy.
    for _ in range(max(1, len(per_contact_averages))):
        ums.update_signal_profile("linguistic_inbound", data)


def _set_outbound_profile(ums: UserModelStore, formality: float) -> None:
    """Write a ``linguistic`` (outbound) profile with given formality average.

    Args:
        ums: UserModelStore to write into.
        formality: Average outbound formality score (0.0 = casual, 1.0 = formal).
    """
    data = {
        "samples": [{"formality": formality, "avg_sentence_length": 10,
                     "hedge_rate": 0.1, "assertion_rate": 0.1,
                     "exclamation_rate": 0.0, "emoji_count": 0,
                     "word_count": 20, "emoji_rate": 0.0,
                     "profanity_rate": 0.0}],
        "averages": {
            "formality": formality,
            "avg_sentence_length": 10.0,
            "hedge_rate": 0.1,
            "assertion_rate": 0.1,
            "exclamation_rate": 0.0,
            "emoji_rate": 0.0,
            "profanity_rate": 0.0,
        },
        "per_contact": {},
        "common_greetings": [],
        "common_closings": [],
    }
    ums.update_signal_profile("linguistic", data)


# =============================================================================
# Tests: empty/missing profile
# =============================================================================


def test_no_inbound_profile_returns_empty(db):
    """Should return [] when ``linguistic_inbound`` profile does not exist."""
    engine = _make_engine(db)
    assert engine._inbound_style_insights() == []


def test_empty_per_contact_averages_returns_empty(db):
    """Should return [] when inbound profile exists but has no contacts."""
    engine = _make_engine(db)
    _set_inbound_profile(engine.ums, {})
    assert engine._inbound_style_insights() == []


# =============================================================================
# Tests: sample-count filtering
# =============================================================================


def test_skips_contact_with_fewer_than_5_samples(db):
    """Contacts with fewer than 5 inbound samples should be skipped."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    # 4 samples, gap of 0.6 — should still be skipped
    _set_inbound_profile(engine.ums, {
        "sparse@example.com": {"formality": 0.2, "samples_count": 4},
    })
    assert engine._inbound_style_insights() == []


def test_includes_contact_with_5_samples(db):
    """Contacts with exactly 5 inbound samples should be included."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "good@example.com": {"formality": 0.2, "samples_count": 5},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 1


# =============================================================================
# Tests: marketing filter
# =============================================================================


def test_skips_noreply_sender(db):
    """Marketing/noreply addresses should be excluded."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "noreply@newsletter.com": {"formality": 0.1, "samples_count": 20},
    })
    assert engine._inbound_style_insights() == []


def test_skips_newsletter_sender(db):
    """Addresses containing 'newsletter' should be excluded."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "updates@newsletter.co": {"formality": 0.1, "samples_count": 30},
    })
    assert engine._inbound_style_insights() == []


def test_includes_real_contact(db):
    """Real human contacts should not be filtered by the marketing filter."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "alice@gmail.com": {"formality": 0.1, "samples_count": 10},
    })
    assert len(engine._inbound_style_insights()) == 1


# =============================================================================
# Tests: gap threshold
# =============================================================================


def test_no_insight_when_gap_below_threshold(db):
    """Gap <= 0.3 should not produce an insight (within noise threshold)."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.5)
    _set_inbound_profile(engine.ums, {
        "close@example.com": {"formality": 0.79, "samples_count": 10},
    })
    # gap = |0.5 - 0.79| = 0.29 < 0.3 — below threshold, should NOT fire
    assert engine._inbound_style_insights() == []


def test_insight_fires_when_gap_just_above_threshold(db):
    """Gap > 0.3 should produce an insight."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.5)
    _set_inbound_profile(engine.ums, {
        "borderline@example.com": {"formality": 0.81, "samples_count": 10},
    })
    # gap = |0.5 - 0.81| = 0.31 > 0.3
    insights = engine._inbound_style_insights()
    assert len(insights) == 1


# =============================================================================
# Tests: direction and content
# =============================================================================


def test_casual_contact_direction(db):
    """Contact writing more casually than the user → 'casually' in summary."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "bob@example.com": {"formality": 0.2, "samples_count": 15},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert "casually" in insights[0].summary
    assert "bob@example.com" in insights[0].summary
    assert "warmer" in insights[0].summary.lower()


def test_formal_contact_direction(db):
    """Contact writing more formally than the user → 'formally' in summary."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.3)
    _set_inbound_profile(engine.ums, {
        "professional@corp.com": {"formality": 0.9, "samples_count": 8},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert "formally" in insights[0].summary
    assert "professional@corp.com" in insights[0].summary
    assert "professional" in insights[0].summary.lower()


def test_insight_metadata(db):
    """Insight should have correct type, category, and entity."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "alice@example.com": {"formality": 0.1, "samples_count": 20},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    ins = insights[0]
    assert ins.type == "communication_style"
    assert ins.category == "style_mismatch"
    assert ins.entity == "alice@example.com"
    assert ins.dedup_key  # should have a computed dedup key


def test_insight_confidence_scales_with_gap(db):
    """Higher gap → higher confidence, capped at 0.80."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.5)

    # Small gap (0.31)
    _set_inbound_profile(engine.ums, {
        "small@example.com": {"formality": 0.19, "samples_count": 10},
    })
    small_insights = engine._inbound_style_insights()
    small_conf = small_insights[0].confidence if small_insights else 0.0

    # Reset and use large gap (0.75)
    ums2 = UserModelStore(db)
    engine2 = InsightEngine(db=db, ums=ums2)
    _set_outbound_profile(engine2.ums, formality=0.9)
    _set_inbound_profile(engine2.ums, {
        "large@example.com": {"formality": 0.15, "samples_count": 10},
    })
    large_insights = engine2._inbound_style_insights()
    large_conf = large_insights[0].confidence if large_insights else 0.0

    assert large_conf > small_conf
    assert large_conf <= 0.80


def test_confidence_capped_at_0_80(db):
    """Confidence should never exceed 0.80 regardless of gap size."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=1.0)
    _set_inbound_profile(engine.ums, {
        "extreme@example.com": {"formality": 0.0, "samples_count": 50},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert insights[0].confidence <= 0.80


# =============================================================================
# Tests: cap at 10
# =============================================================================


def test_caps_at_10_insights(db):
    """Should return at most 10 insights even when many contacts qualify."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)

    # Create 15 qualifying contacts
    per_contact_avgs = {
        f"contact{i}@example.com": {"formality": 0.1, "samples_count": 10}
        for i in range(15)
    }
    _set_inbound_profile(engine.ums, per_contact_avgs)

    insights = engine._inbound_style_insights()
    assert len(insights) <= 10


def test_largest_gap_contacts_first(db):
    """Insights should be ordered by gap size descending."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.5)

    _set_inbound_profile(engine.ums, {
        "big_gap@example.com": {"formality": 0.0, "samples_count": 10},
        "small_gap@example.com": {"formality": 0.15, "samples_count": 10},
    })
    insights = engine._inbound_style_insights()
    assert len(insights) == 2
    # big_gap (0.5) should come before small_gap (0.35)
    assert "big_gap@example.com" in insights[0].entity
    assert "small_gap@example.com" in insights[1].entity


# =============================================================================
# Tests: fallback when outbound profile is absent
# =============================================================================


def test_fallback_user_formality_when_outbound_absent(db):
    """Should use 0.5 as user_formality when ``linguistic`` profile is absent."""
    engine = _make_engine(db)
    # No outbound profile set — only inbound
    _set_inbound_profile(engine.ums, {
        "formal@corp.com": {"formality": 0.9, "samples_count": 10},
    })
    # gap = |0.5 - 0.9| = 0.4 > 0.3 → should fire
    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert "formally" in insights[0].summary


# =============================================================================
# Tests: wiring
# =============================================================================


def test_inbound_style_wired_into_generate_insights(db):
    """``_inbound_style_insights`` should be called by ``generate_insights``."""
    engine = _make_engine(db)
    _set_outbound_profile(engine.ums, formality=0.8)
    _set_inbound_profile(engine.ums, {
        "wire@example.com": {"formality": 0.1, "samples_count": 10},
    })

    import asyncio
    insights = asyncio.run(engine.generate_insights())
    # At least one style_mismatch insight should appear in the output
    mismatch_insights = [i for i in insights if i.category == "style_mismatch"]
    assert len(mismatch_insights) >= 1


def test_style_mismatch_in_source_weight_map(db):
    """``style_mismatch`` category must appear in _apply_source_weights map."""
    engine = _make_engine(db)
    from services.insight_engine.engine import InsightEngine
    import inspect
    # Read the source code of _apply_source_weights to verify the mapping
    source = inspect.getsource(InsightEngine._apply_source_weights)
    assert "style_mismatch" in source
