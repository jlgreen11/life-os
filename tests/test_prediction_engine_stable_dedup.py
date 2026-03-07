"""
Tests for stable prediction descriptions that enable effective deduplication.

The prediction engine generates descriptions with dynamic day counts that change
daily, bypassing both the in-memory pre-filter and DB-level dedup. These tests
verify that _humanize_duration() produces coarse, stable buckets so descriptions
remain identical across consecutive days.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine, _humanize_duration


# -------------------------------------------------------------------------
# _humanize_duration unit tests
# -------------------------------------------------------------------------


class TestHumanizeDuration:
    """Unit tests for the _humanize_duration helper."""

    def test_short_days(self):
        """Days below 7 should return exact integer days."""
        assert _humanize_duration(0) == "0 days"
        assert _humanize_duration(3) == "3 days"
        assert _humanize_duration(6) == "6 days"
        assert _humanize_duration(6.9) == "6 days"

    def test_about_a_week(self):
        """7-13 days should return 'about a week'."""
        assert _humanize_duration(7) == "about a week"
        assert _humanize_duration(10) == "about a week"
        assert _humanize_duration(13) == "about a week"

    def test_about_two_weeks(self):
        """14-20 days should return 'about 2 weeks'."""
        assert _humanize_duration(14) == "about 2 weeks"
        assert _humanize_duration(17) == "about 2 weeks"
        assert _humanize_duration(18) == "about 2 weeks"
        assert _humanize_duration(20) == "about 2 weeks"

    def test_about_a_month(self):
        """21-34 days should return 'about a month'."""
        assert _humanize_duration(21) == "about a month"
        assert _humanize_duration(30) == "about a month"
        assert _humanize_duration(34) == "about a month"

    def test_about_six_weeks(self):
        """35-59 days should return 'about 6 weeks'."""
        assert _humanize_duration(35) == "about 6 weeks"
        assert _humanize_duration(45) == "about 6 weeks"
        assert _humanize_duration(59) == "about 6 weeks"

    def test_about_two_months(self):
        """60-89 days should return 'about 2 months'."""
        assert _humanize_duration(60) == "about 2 months"
        assert _humanize_duration(75) == "about 2 months"
        assert _humanize_duration(89) == "about 2 months"

    def test_many_months(self):
        """90+ days should return 'about N months'."""
        assert _humanize_duration(90) == "about 3 months"
        assert _humanize_duration(120) == "about 4 months"
        assert _humanize_duration(365) == "about 12 months"

    def test_boundary_values(self):
        """Verify exact boundary transitions."""
        # 6.99 -> "6 days", 7.0 -> "about a week"
        assert _humanize_duration(6.99) == "6 days"
        assert _humanize_duration(7.0) == "about a week"
        # 13.99 -> "about a week", 14.0 -> "about 2 weeks"
        assert _humanize_duration(13.99) == "about a week"
        assert _humanize_duration(14.0) == "about 2 weeks"


# -------------------------------------------------------------------------
# Relationship prediction description stability
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relationship_description_stable_across_days(db, user_model_store):
    """Relationship maintenance predictions for day 17 and day 18 should have identical descriptions.

    Before this fix, the description contained the exact day count ('17 days', '18 days'),
    causing daily dedup bypass. With _humanize_duration, both map to 'about 2 weeks'.
    """
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    # Build a contact with 5+ interactions and avg_gap ~14 days
    # so that days_since=17 and days_since=18 both trigger (> avg_gap * 1.5 = 21? no)
    # Actually need days_since > avg_gap * 1.5 AND days_since > 7
    # With avg_gap=10, threshold = 15. days_since=17 and 18 both exceed it.
    timestamps = []
    for i in range(6):
        timestamps.append((now - timedelta(days=17 + i * 10)).isoformat())

    contacts = {
        "alice@example.com": {
            "interaction_count": 6,
            "last_interaction": (now - timedelta(days=17)).isoformat(),
            "outbound_count": 3,
            "interaction_timestamps": timestamps,
        }
    }

    # Store a relationships signal profile
    user_model_store.update_signal_profile(
        "relationships",
        {"contacts": contacts},
    )

    # Generate predictions for "day 17"
    preds_day17 = await engine._check_relationship_maintenance({})

    # Now simulate day 18 by adjusting the last_interaction
    contacts["alice@example.com"]["last_interaction"] = (now - timedelta(days=18)).isoformat()
    timestamps_18 = []
    for i in range(6):
        timestamps_18.append((now - timedelta(days=18 + i * 10)).isoformat())
    contacts["alice@example.com"]["interaction_timestamps"] = timestamps_18

    user_model_store.update_signal_profile(
        "relationships",
        {"contacts": contacts},
    )

    preds_day18 = await engine._check_relationship_maintenance({})

    # Both should generate a prediction
    assert len(preds_day17) >= 1, "Should generate prediction for day 17"
    assert len(preds_day18) >= 1, "Should generate prediction for day 18"

    # The descriptions should be identical (both use "about 2 weeks")
    desc_17 = preds_day17[0].description
    desc_18 = preds_day18[0].description
    assert desc_17 == desc_18, (
        f"Descriptions should be stable across days but differ:\n"
        f"  day 17: {desc_17}\n"
        f"  day 18: {desc_18}"
    )
    assert "about 2 weeks" in desc_17, f"Expected 'about 2 weeks' in description: {desc_17}"


@pytest.mark.asyncio
async def test_relationship_description_uses_humanized_avg_gap(db, user_model_store):
    """The avg_gap portion of the description should also use humanized duration."""
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    # Build contact with avg_gap ~14 days, days_since=25
    timestamps = []
    for i in range(6):
        timestamps.append((now - timedelta(days=25 + i * 14)).isoformat())

    contacts = {
        "bob@example.com": {
            "interaction_count": 6,
            "last_interaction": (now - timedelta(days=25)).isoformat(),
            "outbound_count": 2,
            "interaction_timestamps": timestamps,
        }
    }

    user_model_store.update_signal_profile(
        "relationships",
        {"contacts": contacts},
    )

    preds = await engine._check_relationship_maintenance({})
    assert len(preds) >= 1, "Should generate a prediction"

    desc = preds[0].description
    # Should use humanized form, not raw int
    assert "every ~about" in desc or "every ~" in desc, f"Unexpected description: {desc}"
    # Should NOT contain raw day count like "every ~14 days)"
    assert "every ~14 days)" not in desc, f"Description still uses raw day count: {desc}"


# -------------------------------------------------------------------------
# Connector health prediction description stability
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connector_health_description_stable_across_days(db, user_model_store):
    """Connector health descriptions for day 10 and day 11 should be identical.

    Both fall in the 'about a week' bucket (7-13 days), producing stable descriptions.
    """
    engine = PredictionEngine(db=db, ums=user_model_store)
    now = datetime.now(timezone.utc)

    # Insert a broken connector into state.db
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO connector_state
               (connector_id, status, enabled, last_sync, error_count, last_error, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test_connector",
                "error",
                1,
                (now - timedelta(days=10)).isoformat(),
                5,
                "Connection timeout",
                now.isoformat(),
            ),
        )
        conn.commit()

    preds_day10 = await engine._check_connector_health({})

    # Update staleness to 11 days
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE connector_state SET last_sync = ? WHERE connector_id = ?",
            ((now - timedelta(days=11)).isoformat(), "test_connector"),
        )
        conn.commit()

    # Clear any dedup state by clearing previously predicted connectors
    preds_day11 = await engine._check_connector_health({})

    # Both should produce predictions
    assert len(preds_day10) >= 1, "Should produce prediction for day 10"
    assert len(preds_day11) >= 1, "Should produce prediction for day 11"

    # Descriptions should be identical — both map to "about a week"
    desc_10 = preds_day10[0].description
    desc_11 = preds_day11[0].description
    assert desc_10 == desc_11, (
        f"Connector health descriptions should be stable:\n"
        f"  day 10: {desc_10}\n"
        f"  day 11: {desc_11}"
    )
    assert "about a week" in desc_10, f"Expected 'about a week' in: {desc_10}"
