"""
Tests for the InsightEngine marketing filter and fractional gap fix.

Validates two bugs fixed in iteration 190:

1. Marketing filter gap (PR #185-equivalent for InsightEngine):
   ``_contact_gap_insights`` had no marketing/automated-sender filter,
   so newsletter, noreply, and brokerage addresses produced
   ``relationship_intelligence`` insights the user can never act on.
   The fix applies ``is_marketing_or_noreply()`` from the shared
   marketing filter module (introduced in PR #202).

2. Integer gap truncation bug (PR #166-equivalent for InsightEngine):
   Gap computation used ``.days`` (integer truncation), which rounds
   sub-24-hour gaps to 0.  Contacts who interact daily end up with
   ``avg_gap = 0``, making the threshold condition trivially true for
   any contact unseen for >7 days — a false-positive generator.
   The fix uses ``.total_seconds() / 86400`` (fractional days).

3. Inbound-only filter (PR #204-equivalent for InsightEngine):
   Contacts with ``outbound_count == 0`` are skipped.  If the user
   has never messaged someone, there is no relationship to maintain.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ums(contacts: dict) -> MagicMock:
    """Return a mock UserModelStore whose relationships profile contains *contacts*."""
    ums = MagicMock()
    ums.get_signal_profile.return_value = {
        "data": {"contacts": contacts}
    }
    return ums


def _make_contact(
    *,
    days_since: float,
    avg_gap_days: float,
    count: int = 10,
    outbound_count: int = 3,
    extra_timestamps: int = 0,
) -> dict:
    """Build a synthetic contact data dict.

    Args:
        days_since: How many days ago the last interaction was.
        avg_gap_days: Desired average gap between interactions (days).
        count: Total interaction_count field.
        outbound_count: Number of outbound messages sent to this contact.
        extra_timestamps: Extra timestamps to pad the list with.
    """
    now = datetime.now(timezone.utc)
    last_dt = now - timedelta(days=days_since)

    # Build timestamps spaced avg_gap_days apart, ending at last_dt
    timestamps = []
    for i in range(9, -1, -1):
        ts = last_dt - timedelta(days=avg_gap_days * i)
        timestamps.append(ts.isoformat().replace("+00:00", "Z"))

    return {
        "last_interaction": last_dt.isoformat().replace("+00:00", "Z"),
        "interaction_count": count,
        "outbound_count": outbound_count,
        "interaction_timestamps": timestamps,
    }


def _engine_with_contacts(contacts: dict) -> InsightEngine:
    """Return an InsightEngine backed by a mock UMS with *contacts*."""
    db = MagicMock()
    ums = _make_ums(contacts)
    return InsightEngine(db=db, ums=ums)


# ---------------------------------------------------------------------------
# Marketing filter tests
# ---------------------------------------------------------------------------


class TestMarketingFilterApplied:
    """_contact_gap_insights must not surface automated-sender contacts."""

    def test_noreply_address_excluded(self):
        """noreply@ addresses must be filtered out."""
        contacts = {
            "noreply@company.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], (
            "noreply@ is an automated sender and must not generate an insight"
        )

    def test_newsletter_address_excluded(self):
        """newsletter@ addresses must be filtered out."""
        contacts = {
            "newsletter@somesite.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "newsletter@ is an automated sender"

    def test_no_reply_hyphen_excluded(self):
        """no-reply@ variant must be filtered out."""
        contacts = {
            "no-reply@service.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "no-reply@ is an automated sender"

    def test_mailer_daemon_excluded(self):
        """mailer-daemon@ must be filtered out."""
        contacts = {
            "mailer-daemon@example.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "mailer-daemon@ is an automated sender"

    def test_updates_address_excluded(self):
        """updates@ pattern must be filtered out (bulk marketing)."""
        contacts = {
            "updates@app.io": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "updates@ is a bulk-marketing sender"

    def test_marketing_email_subdomain_excluded(self):
        """Addresses on marketing subdomains (@email., @comms.) must be excluded."""
        contacts = {
            "info@email.company.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "@email. subdomain is a marketing sender pattern"

    def test_human_contact_included(self):
        """A genuine human contact overdue for a message must produce an insight."""
        contacts = {
            "alice@gmail.com": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        assert insights[0].entity == "alice@gmail.com"
        assert insights[0].type == "relationship_intelligence"

    def test_mixed_contacts_only_human_surfaces(self):
        """When both human and marketing contacts are present, only human surfaces."""
        contacts = {
            "alice@gmail.com": _make_contact(days_since=30, avg_gap_days=10),
            "noreply@company.com": _make_contact(days_since=30, avg_gap_days=10),
            "newsletter@news.io": _make_contact(days_since=30, avg_gap_days=10),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        assert insights[0].entity == "alice@gmail.com"


# ---------------------------------------------------------------------------
# Inbound-only filter tests
# ---------------------------------------------------------------------------


class TestInboundOnlyFilter:
    """Contacts with outbound_count == 0 should not generate insights."""

    def test_inbound_only_contact_excluded(self):
        """A contact the user has never messaged must be excluded."""
        contacts = {
            "stranger@example.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=0
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], (
            "Inbound-only contact: user never replied, so no relationship to maintain"
        )

    def test_bidirectional_contact_included(self):
        """A contact the user has messaged at least once must be eligible."""
        contacts = {
            "colleague@example.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=1
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1

    def test_high_outbound_contact_included(self):
        """A frequently messaged contact must pass the inbound-only filter."""
        contacts = {
            "bestfriend@example.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=100
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1


# ---------------------------------------------------------------------------
# Fractional gap calculation tests
# ---------------------------------------------------------------------------


class TestFractionalGapCalculation:
    """avg_gap must use total_seconds()/86400 to handle high-frequency contacts."""

    def test_high_frequency_contact_uses_fractional_gap(self):
        """A contact with daily interaction must NOT produce a false-positive insight.

        With the old integer ``.days`` gap calculation:
        - Gaps of <24 h round to 0
        - avg_gap = 0
        - Threshold = 0 * 1.5 = 0
        - Any days_since > 0 triggers an insight (false positive)

        With the fractional fix:
        - Gaps ≈ 1.0 fractional day
        - avg_gap ≈ 1.0
        - Threshold = 1.5 days
        - 10 days since last contact → 10 > 7 (min) AND 10 > 1.5 → fires correctly
        """
        # Build a contact with daily interactions; 10 days since last contact.
        # avg_gap ≈ 1 day, so threshold = 1.5 days. 10 days > 1.5 → SHOULD fire.
        contacts = {
            "daily_colleague@example.com": _make_contact(
                days_since=10, avg_gap_days=1, outbound_count=5
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        # 10 days since last contact, 10 > 1.5 threshold AND 10 > 7-day minimum
        assert len(insights) == 1, (
            "Contact with avg_gap=1d, 10 days stale should surface (10 > 1.5 * 1)"
        )

    def test_contact_not_stale_enough_does_not_fire(self):
        """A contact with 5-day gap and avg_gap=5 days must NOT generate an insight.

        5 days since last contact vs threshold of 5 * 1.5 = 7.5 days: not stale.
        Also, the 7-day minimum is not met (5 < 7).
        """
        contacts = {
            "weekly_contact@example.com": _make_contact(
                days_since=5, avg_gap_days=5, outbound_count=3
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], (
            "Contact 5 days stale with 5-day avg gap does not meet threshold"
        )

    def test_contact_well_below_threshold_does_not_fire(self):
        """A contact at 1.2× the average gap (below 1.5×) must NOT fire.

        avg_gap=10 days → threshold=15 days.  Contact stale 12 days → 12 < 15 → no insight.
        """
        contacts = {
            "threshold_contact@example.com": _make_contact(
                days_since=12, avg_gap_days=10, outbound_count=2
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "At 1.2x threshold (12 days < 15 day threshold) should not fire"

    def test_contact_just_over_threshold_fires(self):
        """A contact at 1.6× the average gap and >7 days must fire."""
        contacts = {
            "overdue_contact@example.com": _make_contact(
                days_since=16, avg_gap_days=10, outbound_count=2
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1, "At 1.6x threshold (16 > 15) should fire"

    def test_insight_days_displayed_as_integer(self):
        """The insight summary should display whole-day counts for readability."""
        contacts = {
            "bob@example.com": _make_contact(
                days_since=20, avg_gap_days=7, outbound_count=5
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        # Summary must contain integer day count (e.g. "20 days"), not "20.0 days"
        assert "20 days" in insights[0].summary, (
            f"Expected integer in summary, got: {insights[0].summary}"
        )


# ---------------------------------------------------------------------------
# Combined / integration tests
# ---------------------------------------------------------------------------


class TestCombinedFilters:
    """All three filters (marketing, inbound-only, threshold) work together."""

    def test_three_contacts_only_one_surfaces(self):
        """Marketing sender, inbound-only, and genuine contact — only genuine fires."""
        contacts = {
            "noreply@corp.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=5
            ),  # marketing → blocked
            "stranger@example.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=0
            ),  # inbound-only → blocked
            "friend@example.com": _make_contact(
                days_since=30, avg_gap_days=10, outbound_count=3
            ),  # genuine → surfaces
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        assert insights[0].entity == "friend@example.com"

    def test_no_profile_returns_empty(self):
        """When the relationships profile is missing, return an empty list gracefully."""
        db = MagicMock()
        ums = MagicMock()
        ums.get_signal_profile.return_value = None
        engine = InsightEngine(db=db, ums=ums)
        insights = engine._contact_gap_insights()
        assert insights == []

    def test_empty_contacts_returns_empty(self):
        """When the profile has no contacts, return empty list."""
        engine = _engine_with_contacts({})
        insights = engine._contact_gap_insights()
        assert insights == []

    def test_contact_below_5_interactions_excluded(self):
        """Contacts with fewer than 5 interactions lack sufficient data."""
        contacts = {
            "new_contact@example.com": {
                **_make_contact(days_since=30, avg_gap_days=10, outbound_count=3),
                "interaction_count": 4,
            },
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert insights == [], "< 5 interactions means insufficient history"

    def test_insight_type_is_relationship_intelligence(self):
        """Surfaced insights must be typed 'relationship_intelligence'."""
        contacts = {
            "colleague@example.com": _make_contact(
                days_since=20, avg_gap_days=7, outbound_count=4
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        assert insights[0].type == "relationship_intelligence"
        assert insights[0].category == "contact_gap"

    def test_confidence_bounded_between_0_and_0_8(self):
        """Confidence must never exceed 0.8 regardless of how stale the contact is."""
        contacts = {
            "very_stale@example.com": _make_contact(
                days_since=365, avg_gap_days=7, outbound_count=5
            ),
        }
        engine = _engine_with_contacts(contacts)
        insights = engine._contact_gap_insights()
        assert len(insights) == 1
        assert insights[0].confidence <= 0.8, (
            f"Confidence capped at 0.8, got {insights[0].confidence}"
        )
        assert insights[0].confidence > 0.0
