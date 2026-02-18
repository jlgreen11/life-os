"""
Tests for the inbound-only contact filter in relationship maintenance predictions.

The relationship maintenance predictor generates "reach out to X" opportunity
predictions for contacts the user hasn't spoken to in a while.  However, if
the user has *never* sent anything to a contact (outbound_count == 0), there
is no bidirectional relationship to maintain — the contact is purely inbound
(e.g. a low-volume mailing list, a SaaS alert, or a personal-looking automated
system) and any "opportunity" prediction for them would be a false positive.

This test suite verifies:
  1. Contacts with outbound_count == 0 are skipped even when overdue.
  2. Contacts with outbound_count >= 1 continue to generate predictions normally.
  3. The inbound_only_filtered count is correctly logged in the diagnostic summary.
  4. Edge cases around missing outbound_count fields are handled gracefully.
"""

import asyncio
import io
import sys
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(now: datetime, num: int, gap_days: float = 10) -> list[str]:
    """Return a list of ISO timestamps evenly spaced ``gap_days`` apart, ending at ``now``."""
    return [
        (now - timedelta(days=gap_days * (num - 1 - i))).isoformat()
        for i in range(num)
    ]


def _overdue_contact(
    now: datetime,
    *,
    inbound_count: int,
    outbound_count: int,
    gap_days: float = 10,
    overdue_days: float = 25,
) -> dict:
    """Return a contact profile that is overdue.

    The contact's last interaction was ``overdue_days`` days ago.  The
    interaction timestamps are spread at ``gap_days`` intervals ending at
    ``overdue_days`` days in the past, so:

        days_since          = overdue_days
        avg_gap             ≈ gap_days
        overdue threshold   = gap_days * 1.5

    For the default values (gap=10, overdue=25):
        25 > 10 * 1.5 (= 15)  ✓  AND  25 > 7  ✓  → triggers prediction.

    Args:
        now: Current time reference.
        inbound_count: Number of inbound messages.
        outbound_count: Number of outbound messages.
        gap_days: Average gap between consecutive interactions in days.
        overdue_days: How many days ago the last interaction occurred.  Must
            be greater than gap_days * 1.5 + some margin AND > 7 to trigger
            the overdue check in _check_relationship_maintenance.

    Returns:
        A relationship profile dict compatible with the ``relationships`` signal profile.
    """
    total = inbound_count + outbound_count
    # Build timestamps ending exactly at now - overdue_days, spread by gap_days
    last_interaction_time = now - timedelta(days=overdue_days)
    timestamps = _make_timestamps(last_interaction_time, total, gap_days)
    return {
        "interaction_count": total,
        "inbound_count": inbound_count,
        "outbound_count": outbound_count,
        "channels_used": ["google"],
        "avg_message_length": 200,
        "last_interaction": timestamps[-1],
        "interaction_timestamps": timestamps,
        "last_inbound_timestamp": timestamps[-1],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def user_model_store(db):
    """UserModelStore wired to the temporary DatabaseManager."""
    return UserModelStore(db)


@pytest.fixture()
def engine(db, user_model_store):
    """PredictionEngine wired to the temporary stores."""
    return PredictionEngine(db, user_model_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inbound_only_contact_is_excluded(db, user_model_store, engine):
    """Contacts with outbound_count == 0 must NOT generate opportunity predictions.

    Rationale: if the user has never sent anything to the address, there is no
    bidirectional relationship to maintain — the contact is inbound-only (e.g.
    an automated notification or low-volume list that slipped through the
    marketing filter).
    """
    now = datetime.now(timezone.utc)

    profile = {
        "contacts": {
            # Purely inbound — user has never replied; should be skipped.
            "inbound-only@example.com": _overdue_contact(
                now, inbound_count=8, outbound_count=0
            ),
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})

    emails = [p.relevant_contacts[0] for p in predictions if p.relevant_contacts]
    assert "inbound-only@example.com" not in emails, (
        "inbound-only contact must not generate an opportunity prediction"
    )
    assert predictions == [], "Expected zero predictions for a purely inbound contact"


@pytest.mark.asyncio
async def test_bidirectional_contact_is_included(db, user_model_store, engine):
    """Contacts with at least one outbound message SHOULD generate predictions when overdue."""
    now = datetime.now(timezone.utc)

    profile = {
        "contacts": {
            # Bidirectional — user has replied; should be eligible.
            "friend@example.com": _overdue_contact(
                now, inbound_count=7, outbound_count=3
            ),
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})

    emails = [p.relevant_contacts[0] for p in predictions if p.relevant_contacts]
    assert "friend@example.com" in emails, (
        "bidirectional contact must generate an opportunity prediction when overdue"
    )


@pytest.mark.asyncio
async def test_mixed_contacts_only_bidirectional_predicted(db, user_model_store, engine):
    """When the profile contains both inbound-only and bidirectional contacts,
    only the bidirectional ones should produce predictions (assuming both are overdue).
    """
    now = datetime.now(timezone.utc)

    profile = {
        "contacts": {
            # Inbound-only — should be suppressed
            "newsletter@example.com": _overdue_contact(
                now, inbound_count=10, outbound_count=0
            ),
            # Bidirectional — should generate prediction
            "colleague@example.com": _overdue_contact(
                now, inbound_count=6, outbound_count=4
            ),
            # Bidirectional with minimum 1 outbound — edge case; should generate
            "acquaintance@example.com": _overdue_contact(
                now, inbound_count=9, outbound_count=1
            ),
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})
    emails = {p.relevant_contacts[0] for p in predictions if p.relevant_contacts}

    assert "newsletter@example.com" not in emails, (
        "inbound-only contact must not appear in predictions"
    )
    assert "colleague@example.com" in emails, (
        "bidirectional contact must appear in predictions"
    )
    assert "acquaintance@example.com" in emails, (
        "contact with outbound_count==1 is bidirectional and must appear"
    )


@pytest.mark.asyncio
async def test_missing_outbound_count_field_treated_as_zero(db, user_model_store, engine):
    """Contacts missing the outbound_count key entirely should be treated as
    outbound_count == 0 (i.e. skipped), to be conservative with old profiles
    that pre-date the field.
    """
    now = datetime.now(timezone.utc)
    timestamps = _make_timestamps(now - timedelta(days=20), 5, gap_days=7)

    profile = {
        "contacts": {
            "legacy@example.com": {
                "interaction_count": 5,
                "inbound_count": 5,
                # outbound_count key deliberately absent — simulate old profile format
                "channels_used": ["google"],
                "avg_message_length": 200,
                "last_interaction": timestamps[-1],
                "interaction_timestamps": timestamps,
                "last_inbound_timestamp": timestamps[-1],
            },
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})

    emails = [p.relevant_contacts[0] for p in predictions if p.relevant_contacts]
    assert "legacy@example.com" not in emails, (
        "contact with missing outbound_count should be treated as inbound-only and skipped"
    )


@pytest.mark.asyncio
async def test_inbound_only_filter_does_not_affect_marketing_contacts(db, user_model_store, engine):
    """Marketing senders already filtered by _is_marketing_or_noreply must stay
    filtered regardless of outbound_count — the inbound-only check adds a second,
    independent layer of defence but does not interact with the marketing filter.
    """
    now = datetime.now(timezone.utc)
    timestamps = _make_timestamps(now - timedelta(days=30), 5, gap_days=10)

    profile = {
        "contacts": {
            # Marketing domain — blocked by the marketing filter first
            "offers@email.brand.com": {
                "interaction_count": 5,
                "inbound_count": 3,
                "outbound_count": 2,  # Has outbound; would pass inbound-only check
                "channels_used": ["google"],
                "avg_message_length": 5000,
                "last_interaction": timestamps[-1],
                "interaction_timestamps": timestamps,
                "last_inbound_timestamp": timestamps[-1],
            },
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})

    emails = [p.relevant_contacts[0] for p in predictions if p.relevant_contacts]
    assert "offers@email.brand.com" not in emails, (
        "marketing domain contact must remain filtered even with outbound messages"
    )


@pytest.mark.asyncio
async def test_diagnostic_log_includes_inbound_only_filtered_count(db, user_model_store, engine, caplog):
    """The diagnostic summary logged by _check_relationship_maintenance must
    include the ``inbound_only_filtered`` count so operators can see how many
    contacts are being suppressed by this new filter.

    Uses caplog (not capsys) because the engine logs via Python's logging module,
    not via print() to stdout.
    """
    import logging
    now = datetime.now(timezone.utc)

    profile = {
        "contacts": {
            # One inbound-only (should appear in inbound_only_filtered count)
            "inbound-only@example.com": _overdue_contact(
                now, inbound_count=6, outbound_count=0
            ),
            # One bidirectional (should not appear in inbound_only_filtered count)
            "friend@example.com": _overdue_contact(
                now, inbound_count=5, outbound_count=5
            ),
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    with caplog.at_level(logging.DEBUG):
        await engine._check_relationship_maintenance({})

    assert "inbound_only_filtered=1" in caplog.text, (
        "Diagnostic log must include 'inbound_only_filtered=1' when one contact is filtered"
    )


@pytest.mark.asyncio
async def test_outbound_count_zero_but_not_enough_interactions(db, user_model_store, engine):
    """Contacts with outbound_count == 0 AND fewer than 5 interactions should be
    skipped by the interaction-count guard *before* the inbound-only guard — both
    filters must be independently correct.
    """
    now = datetime.now(timezone.utc)
    timestamps = _make_timestamps(now - timedelta(days=60), 3, gap_days=20)

    profile = {
        "contacts": {
            "low-volume@example.com": {
                "interaction_count": 3,
                "inbound_count": 3,
                "outbound_count": 0,
                "channels_used": ["google"],
                "avg_message_length": 200,
                "last_interaction": timestamps[-1],
                "interaction_timestamps": timestamps,
                "last_inbound_timestamp": timestamps[-1],
            },
        }
    }
    user_model_store.update_signal_profile("relationships", profile)

    predictions = await engine._check_relationship_maintenance({})

    # No predictions for any reason — graceful skip regardless of which guard fires first
    assert predictions == [], "Contact with < 5 interactions and outbound_count=0 must be skipped"
