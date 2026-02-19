"""
Tests for the InsightEngine ``_relationship_intelligence_insights`` correlator.

The correlator reads the ``relationships`` signal profile and surfaces two
sub-categories of ``relationship_intelligence`` insights:

    reciprocity_imbalance  -- Contacts with strong directional asymmetry
                              (user always initiates ≥ 85%, or contact always
                              initiates ≥ 85% while user has outbound > 0)

    fast_responder         -- Contacts the user responds to under 30 minutes
                              on average (based on ≥ 5 measured response times)

This test suite validates:

- Returns empty list when relationships profile is absent
- Returns empty list when profile has no contacts
- reciprocity_imbalance fires when outbound_ratio ≥ 0.85
- reciprocity_imbalance fires when outbound_ratio ≤ 0.15 (and outbound > 0)
- reciprocity_imbalance does NOT fire when ratio is between 0.15 and 0.85
- reciprocity_imbalance requires total_interactions ≥ 10
- reciprocity_imbalance skips marketing/automated senders
- reciprocity_imbalance summary uses display name when contact record exists
- reciprocity_imbalance category is "reciprocity_imbalance"
- reciprocity_imbalance type is "relationship_intelligence"
- fast_responder fires when avg_response_time_seconds < 1800 with ≥ 5 samples
- fast_responder does NOT fire when avg_response_time_seconds ≥ 1800
- fast_responder requires at least 5 response_time samples
- fast_responder skips marketing/automated senders
- fast_responder category is "fast_responder"
- fast_responder type is "relationship_intelligence"
- Both sub-types can fire for the same contact
- reciprocity_imbalance and fast_responder are wired into generate_insights()
- Both categories are handled by _apply_source_weights (no KeyError)
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


def _set_rel_profile(ums: UserModelStore, contacts: dict) -> None:
    """Write a relationships signal profile with the given contacts dict.

    Args:
        ums: UserModelStore to write into.
        contacts: Dict mapping email address → per-contact profile dict.
    """
    ums.update_signal_profile("relationships", {"contacts": contacts})


def _contact(
    *,
    inbound: int = 5,
    outbound: int = 5,
    response_times: list[float] | None = None,
    interaction_count: int | None = None,
) -> dict:
    """Build a minimal per-contact relationship profile dict.

    Args:
        inbound: Number of inbound messages received from the contact.
        outbound: Number of outbound messages sent to the contact.
        response_times: List of response time samples in seconds.
        interaction_count: Total interactions (defaults to inbound + outbound).
    """
    total = inbound + outbound if interaction_count is None else interaction_count
    return {
        "inbound_count": inbound,
        "outbound_count": outbound,
        "interaction_count": total,
        "response_times_seconds": response_times or [],
        "avg_response_time_seconds": (
            sum(response_times) / len(response_times) if response_times else None
        ),
        "channels_used": ["email"],
        "avg_message_length": 100,
        "last_interaction": "2026-02-10T10:00:00Z",
    }


# =============================================================================
# Tests: No profile / empty data
# =============================================================================


def test_no_insight_when_profile_absent(db):
    """No insights when the relationships profile has never been written."""
    engine = _make_engine(db)
    result = engine._relationship_intelligence_insights()
    assert result == []


def test_no_insight_when_contacts_empty(db):
    """No insights when the relationships profile has an empty contacts dict."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {})
    engine = _make_engine(db)
    result = engine._relationship_intelligence_insights()
    assert result == []


# =============================================================================
# Tests: reciprocity_imbalance — high outbound ratio (user always initiates)
# =============================================================================


def test_reciprocity_fires_when_user_always_initiates(db):
    """reciprocity_imbalance fires when outbound/(inbound+outbound) ≥ 0.85."""
    ums = UserModelStore(db)
    # 9 outbound, 1 inbound → 90% outbound ratio, 10 total
    _set_rel_profile(ums, {"alice@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = [i.category for i in results]
    assert "reciprocity_imbalance" in categories


def test_reciprocity_high_outbound_summary_mentions_contact_name(db):
    """High-outbound reciprocity summary says 'You initiate X%'."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"bob@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    ri = next(i for i in results if i.category == "reciprocity_imbalance")
    assert "90%" in ri.summary
    assert "bob@example.com" in ri.summary


def test_reciprocity_high_outbound_uses_display_name(db):
    """Display name replaces raw email when a contact record exists."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"carol@example.com": _contact(outbound=9, inbound=1)})
    # Insert a contact record so the name map has an entry.  The contacts table
    # uses a separate contact_identifiers join table for email addresses rather
    # than a plain 'email' column.
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO contacts (id, name) VALUES (?, ?)",
            ("c1", "Carol Chen"),
        )
        conn.execute(
            "INSERT INTO contact_identifiers (contact_id, identifier, identifier_type)"
            " VALUES (?, ?, ?)",
            ("c1", "carol@example.com", "email"),
        )
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    ri = next(i for i in results if i.category == "reciprocity_imbalance")
    assert "Carol Chen" in ri.summary
    assert "carol@example.com" not in ri.summary


def test_reciprocity_high_outbound_type_is_relationship_intelligence(db):
    """High-outbound reciprocity insight type is 'relationship_intelligence'."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"dave@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    ri = next(i for i in results if i.category == "reciprocity_imbalance")
    assert ri.type == "relationship_intelligence"


# =============================================================================
# Tests: reciprocity_imbalance — low outbound ratio (contact always initiates)
# =============================================================================


def test_reciprocity_fires_when_contact_always_initiates(db):
    """reciprocity_imbalance fires when outbound/(inbound+outbound) ≤ 0.15 and outbound > 0."""
    ums = UserModelStore(db)
    # 1 outbound, 9 inbound → 10% outbound ratio, 10 total
    _set_rel_profile(ums, {"eve@example.com": _contact(outbound=1, inbound=9)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = [i.category for i in results]
    assert "reciprocity_imbalance" in categories


def test_reciprocity_low_outbound_summary_mentions_contact_initiates(db):
    """Low-outbound reciprocity summary says the contact initiates X%."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"frank@example.com": _contact(outbound=1, inbound=9)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    ri = next(i for i in results if i.category == "reciprocity_imbalance")
    # Summary should indicate the contact (not the user) initiates most messages
    assert "frank@example.com" in ri.summary
    # 90% of messages come from the contact
    assert "90%" in ri.summary


def test_reciprocity_low_outbound_skipped_when_outbound_is_zero(db):
    """Low-outbound reciprocity is NOT fired when outbound_count is 0.

    A pure inbound-only contact (no outbound messages from user) is not a
    real bidirectional relationship — likely a mailing list or newsletter
    that slipped through the marketing filter.
    """
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"newsletter@example.com": _contact(outbound=0, inbound=20)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.entity != "newsletter@example.com" for i in results)


# =============================================================================
# Tests: reciprocity_imbalance — no-fire conditions
# =============================================================================


def test_reciprocity_does_not_fire_for_balanced_ratio(db):
    """reciprocity_imbalance does NOT fire when ratio is between 0.15 and 0.85."""
    ums = UserModelStore(db)
    # 50% outbound — perfectly balanced
    _set_rel_profile(ums, {"grace@example.com": _contact(outbound=5, inbound=5)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "reciprocity_imbalance" for i in results)


def test_reciprocity_does_not_fire_below_min_interactions(db):
    """reciprocity_imbalance is skipped when total interactions < 10."""
    ums = UserModelStore(db)
    # 9 outbound, 0 inbound = 9 total — below threshold even though ratio is 1.0
    _set_rel_profile(ums, {"henry@example.com": _contact(outbound=9, inbound=0)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "reciprocity_imbalance" for i in results)


def test_reciprocity_skips_marketing_senders(db):
    """Marketing/automated senders are excluded from reciprocity insights."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "noreply@newsletter.com": _contact(outbound=10, inbound=0),
        "updates@service.com": _contact(outbound=10, inbound=0),
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    # Marketing senders should not appear in any insight
    entities = {i.entity for i in results}
    assert "noreply@newsletter.com" not in entities
    assert "updates@service.com" not in entities


def test_reciprocity_boundary_exactly_85_percent(db):
    """Boundary: exactly 85% outbound (17/20) fires the imbalance insight."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"iris@example.com": _contact(outbound=17, inbound=3)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = [i.category for i in results]
    assert "reciprocity_imbalance" in categories


def test_reciprocity_boundary_84_percent_does_not_fire(db):
    """Boundary: 84% outbound (16/19 ≈ 84.2%) does NOT fire the imbalance insight."""
    ums = UserModelStore(db)
    # 16 outbound, 3 inbound = 19 total → 84.2% outbound (just under threshold)
    _set_rel_profile(ums, {"jake@example.com": _contact(outbound=16, inbound=3)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "reciprocity_imbalance" for i in results)


# =============================================================================
# Tests: fast_responder
# =============================================================================


def test_fast_responder_fires_when_avg_under_30_min(db):
    """fast_responder fires when avg_response_time_seconds < 1800 with ≥ 5 samples."""
    ums = UserModelStore(db)
    # 5 samples averaging 600 seconds (10 minutes) — well under threshold
    _set_rel_profile(ums, {
        "kate@example.com": _contact(
            outbound=10, inbound=10,
            response_times=[600.0, 500.0, 700.0, 450.0, 650.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = [i.category for i in results]
    assert "fast_responder" in categories


def test_fast_responder_summary_mentions_contact(db):
    """fast_responder summary mentions the contact name/address."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "liam@example.com": _contact(
            outbound=10, inbound=10,
            response_times=[300.0, 400.0, 350.0, 420.0, 380.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    fr = next(i for i in results if i.category == "fast_responder")
    assert "liam@example.com" in fr.summary
    assert "minutes" in fr.summary


def test_fast_responder_type_is_relationship_intelligence(db):
    """fast_responder insight type is 'relationship_intelligence'."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "mia@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[200.0, 300.0, 250.0, 180.0, 220.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    fr = next(i for i in results if i.category == "fast_responder")
    assert fr.type == "relationship_intelligence"


def test_fast_responder_does_not_fire_when_avg_over_30_min(db):
    """fast_responder does NOT fire when avg_response_time_seconds ≥ 1800."""
    ums = UserModelStore(db)
    # Average 3600 seconds (1 hour) — above threshold
    _set_rel_profile(ums, {
        "noah@example.com": _contact(
            outbound=10, inbound=10,
            response_times=[3600.0, 3000.0, 4200.0, 3300.0, 3900.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "fast_responder" for i in results)


def test_fast_responder_requires_5_samples(db):
    """fast_responder is skipped when fewer than 5 response time samples exist."""
    ums = UserModelStore(db)
    # Only 4 samples — below minimum
    _set_rel_profile(ums, {
        "olivia@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[100.0, 200.0, 150.0, 120.0],  # 4 samples
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "fast_responder" for i in results)


def test_fast_responder_skips_marketing_senders(db):
    """Marketing/automated senders are excluded from fast_responder insights."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "alerts@bank.com": _contact(
            outbound=5, inbound=20,
            response_times=[60.0, 90.0, 75.0, 80.0, 70.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.entity != "alerts@bank.com" for i in results)


def test_fast_responder_boundary_exactly_30_min_does_not_fire(db):
    """Boundary: avg exactly 1800 seconds (30 min) does NOT fire fast_responder."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "pete@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[1800.0, 1800.0, 1800.0, 1800.0, 1800.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    assert all(i.category != "fast_responder" for i in results)


def test_fast_responder_boundary_1799_seconds_fires(db):
    """Boundary: avg 1799 seconds (just under 30 min) DOES fire fast_responder."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "quinn@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[1799.0, 1799.0, 1799.0, 1799.0, 1799.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = [i.category for i in results]
    assert "fast_responder" in categories


# =============================================================================
# Tests: Both sub-types can fire for the same contact
# =============================================================================


def test_both_subtypes_can_fire_for_same_contact(db):
    """A contact can generate both reciprocity_imbalance and fast_responder."""
    ums = UserModelStore(db)
    # User always initiates (90% outbound) AND responds very quickly
    _set_rel_profile(ums, {
        "rachel@example.com": _contact(
            outbound=9, inbound=1,
            response_times=[300.0, 250.0, 400.0, 320.0, 280.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    categories = {i.category for i in results}
    assert "reciprocity_imbalance" in categories
    assert "fast_responder" in categories


# =============================================================================
# Tests: Integration — wired into generate_insights()
# =============================================================================


async def test_relationship_intelligence_wired_into_generate_insights(db):
    """_relationship_intelligence_insights is called from generate_insights()."""
    ums = UserModelStore(db)
    # 9 outbound, 1 inbound → should fire reciprocity_imbalance
    _set_rel_profile(ums, {"sam@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    all_insights = await engine.generate_insights()
    rel_insights = [i for i in all_insights if i.type == "relationship_intelligence"
                    and i.category == "reciprocity_imbalance"]
    assert len(rel_insights) >= 1


async def test_fast_responder_wired_into_generate_insights(db):
    """fast_responder is reachable via generate_insights()."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "tina@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[200.0, 300.0, 250.0, 180.0, 220.0],
        )
    })
    engine = _make_engine(db)
    all_insights = await engine.generate_insights()
    fr = [i for i in all_insights if i.category == "fast_responder"]
    assert len(fr) >= 1


# =============================================================================
# Tests: Source weight map coverage (no KeyError for new categories)
# =============================================================================


def test_reciprocity_imbalance_handled_by_source_weights(db):
    """reciprocity_imbalance category does not raise KeyError in _apply_source_weights."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"uma@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    # Should not raise even though source weights may be at default 1.0
    results = engine._relationship_intelligence_insights()
    weighted = engine._apply_source_weights(results)
    # All insights should still pass the confidence threshold
    assert len(weighted) >= 1


def test_fast_responder_handled_by_source_weights(db):
    """fast_responder category does not raise KeyError in _apply_source_weights."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "victor@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[100.0, 150.0, 120.0, 130.0, 110.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    weighted = engine._apply_source_weights(results)
    assert len(weighted) >= 1


# =============================================================================
# Tests: Evidence fields
# =============================================================================


def test_reciprocity_evidence_contains_counts(db):
    """reciprocity_imbalance evidence includes outbound/inbound counts and ratio."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {"wendy@example.com": _contact(outbound=9, inbound=1)})
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    ri = next(i for i in results if i.category == "reciprocity_imbalance")
    evidence_str = " ".join(ri.evidence)
    assert "outbound_count=9" in evidence_str
    assert "inbound_count=1" in evidence_str
    assert "outbound_ratio=" in evidence_str


def test_fast_responder_evidence_contains_timing(db):
    """fast_responder evidence includes avg_response_time and sample count."""
    ums = UserModelStore(db)
    _set_rel_profile(ums, {
        "xena@example.com": _contact(
            outbound=5, inbound=5,
            response_times=[600.0, 600.0, 600.0, 600.0, 600.0],
        )
    })
    engine = _make_engine(db)
    results = engine._relationship_intelligence_insights()
    fr = next(i for i in results if i.category == "fast_responder")
    evidence_str = " ".join(fr.evidence)
    assert "avg_response_time_seconds=600" in evidence_str
    assert "response_time_samples=5" in evidence_str
