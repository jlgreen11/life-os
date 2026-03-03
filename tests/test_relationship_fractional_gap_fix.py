"""
Test relationship maintenance fractional day gap calculation fix.

Iteration 166: Fix relationship maintenance predictions to use fractional days
instead of integer days for gap calculation. The original implementation used
.days which returns integers, causing avg_gap=0 for frequently-contacted people,
making it impossible to generate predictions (316 eligible contacts → 0 predictions).
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


@pytest.fixture
def setup_relationship_test(db: DatabaseManager):
    """
    Create relationship profile with contacts at various interaction frequencies.

    We test:
    - Daily interactions (gaps < 24 hours) - these broke with integer .days
    - Weekly interactions (7 day gaps)
    - Monthly interactions (30 day gaps)
    """
    ums = UserModelStore(db)
    now = datetime.now(timezone.utc)

    # Contact 1: Daily interactions (gaps of ~12 hours)
    # With integer .days: all gaps become 0, avg_gap=0, threshold=0, NO predictions
    # With fractional days: avg_gap=0.5d, threshold=0.75d, SHOULD predict after 1+ days
    daily_timestamps = [
        (now - timedelta(hours=12 * i)).isoformat()
        for i in range(10, 0, -1)
    ]

    # Contact 2: Weekly interactions (gaps of 7 days)
    weekly_timestamps = [
        (now - timedelta(days=7 * i)).isoformat()
        for i in range(5, 0, -1)
    ]

    # Contact 3: Monthly interactions (gaps of 30 days)
    monthly_timestamps = [
        (now - timedelta(days=30 * i)).isoformat()
        for i in range(3, 0, -1)
    ]

    # Contact 4: Variable frequency (mix of 1d, 2d, 3d gaps)
    variable_timestamps = []
    days_ago = 0
    for gap in [1, 2, 3, 1, 2]:
        days_ago += gap
        variable_timestamps.append((now - timedelta(days=days_ago)).isoformat())
    variable_timestamps.reverse()

    # Store relationships profile
    rel_data = {
        "contacts": {
            "daily@example.com": {
                "interaction_count": 10,
                "last_interaction": (now - timedelta(days=8)).isoformat(),  # 8 days (exceeds 7-day minimum)
                "interaction_timestamps": daily_timestamps,
                "inbound_count": 5,
                "outbound_count": 5,
            },
            "weekly@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=12)).isoformat(),  # 5 days overdue
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            },
            "monthly@example.com": {
                "interaction_count": 5,  # Need ≥5 interactions to be eligible
                "last_interaction": (now - timedelta(days=50)).isoformat(),  # 5 days overdue
                "interaction_timestamps": monthly_timestamps + [
                    (now - timedelta(days=60)).isoformat(),
                    (now - timedelta(days=90)).isoformat()
                ],
                "inbound_count": 3,
                "outbound_count": 2,
            },
            "variable@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=8)).isoformat(),  # 8 days (exceeds 7-day minimum)
                "interaction_timestamps": variable_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            },
            # Marketing contact - should be filtered even with correct gaps
            "noreply@marketing.com": {
                "interaction_count": 100,
                "last_interaction": (now - timedelta(days=60)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=i)).isoformat()
                    for i in range(100, 0, -1)
                ],
                "inbound_count": 100,
                "outbound_count": 0,
            },
        }
    }

    ums.update_signal_profile("relationships", rel_data)

    return ums


@pytest.mark.asyncio
async def test_fractional_day_gap_calculation(db: DatabaseManager, setup_relationship_test):
    """
    Test that fractional day gaps enable predictions for frequently-contacted people.

    Before fix: Contacts with < 24h gaps had avg_gap=0, generated 0 predictions.
    After fix: Contacts with any frequency generate predictions when overdue.
    """
    ums = setup_relationship_test
    engine = PredictionEngine(db, ums)

    # Force time-based prediction run
    engine._last_time_based_run = None

    predictions = await engine._check_relationship_maintenance({})

    # Should generate predictions for overdue contacts
    # daily@example.com: avg_gap ~0.5d, threshold ~0.75d, days_since=8d → SHOULD PREDICT (exceeds threshold AND 7-day min)
    # weekly@example.com: avg_gap ~7d, threshold ~10.5d, days_since=12d → SHOULD PREDICT
    # monthly@example.com: avg_gap ~30d, threshold ~45d, days_since=50d → SHOULD PREDICT
    # variable@example.com: avg_gap ~1.8d, threshold ~2.7d, days_since=8d → SHOULD PREDICT (exceeds threshold AND 7-day min)
    # noreply@marketing.com: should be filtered out by marketing detection

    assert len(predictions) >= 4, f"Expected ≥4 predictions, got {len(predictions)}"

    # Verify prediction types
    assert all(p.prediction_type == "opportunity" for p in predictions), \
        "All relationship maintenance predictions should be type 'opportunity'"

    # Check that daily contact generates prediction (this was broken before)
    # Description now uses resolved contact name (email prefix as fallback)
    daily_preds = [p for p in predictions if "daily" in p.description]
    assert len(daily_preds) > 0, \
        "Daily contact with 12h gaps should generate prediction after 8 days (was broken with integer .days)"

    # Verify confidence gates are reasonable (should be SUGGEST level)
    for pred in predictions:
        assert pred.confidence >= 0.3, \
            f"Relationship predictions should meet SUGGEST threshold (0.3+), got {pred.confidence}"
        assert pred.confidence <= 0.6, \
            f"Relationship predictions capped at 0.6, got {pred.confidence}"


@pytest.mark.asyncio
async def test_integer_vs_fractional_gap_difference(db: DatabaseManager):
    """
    Demonstrate the exact bug: integer .days vs fractional .total_seconds()/86400.

    For contacts with interactions < 24h apart, integer .days returns 0 for all gaps,
    making avg_gap=0 and threshold=0, preventing any predictions.
    """
    ums = UserModelStore(db)
    now = datetime.now(timezone.utc)

    # Create contact with 8-hour gaps (3 interactions per day)
    # These are high-frequency business contacts or family members
    timestamps = [
        (now - timedelta(hours=8 * i)).isoformat()
        for i in range(10, 0, -1)
    ]

    rel_data = {
        "contacts": {
            "frequent@example.com": {
                "interaction_count": 10,
                "last_interaction": (now - timedelta(days=3)).isoformat(),  # 3 days since last contact
                "interaction_timestamps": timestamps,
                "inbound_count": 5,
                "outbound_count": 5,
            }
        }
    }

    ums.update_signal_profile("relationships", rel_data)
    engine = PredictionEngine(db, ums)

    # With the fix, this should generate a prediction
    # avg_gap = 8h / 24 = 0.333 days
    # threshold = 0.333 * 1.5 = 0.5 days
    # days_since = 3 days
    # 3 > 0.5 AND 3 > 7 → second condition fails, but demonstrates calculation works

    # Let's make it 8 days since last contact to exceed both thresholds
    rel_data["contacts"]["frequent@example.com"]["last_interaction"] = \
        (now - timedelta(days=8)).isoformat()
    ums.update_signal_profile("relationships", rel_data)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    # Should generate prediction now
    # Description now uses resolved contact name (email prefix as fallback)
    frequent_preds = [p for p in predictions if "frequent" in p.description]
    assert len(frequent_preds) > 0, \
        "Contact with 8-hour gaps should generate prediction after 8 days (broken with integer .days)"

    # Verify the prediction mentions the contact (using resolved name)
    pred = frequent_preds[0]
    assert "frequent" in pred.description


@pytest.mark.asyncio
async def test_marketing_filter_still_works(db: DatabaseManager, setup_relationship_test):
    """
    Verify that the fractional day fix doesn't break marketing email filtering.

    Marketing contacts should still be filtered out even if their gap calculation
    now works correctly.
    """
    ums = setup_relationship_test
    engine = PredictionEngine(db, ums)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    # noreply@marketing.com should be filtered out
    marketing_preds = [p for p in predictions if p.supporting_signals.get("contact_email") == "noreply@marketing.com"]
    assert len(marketing_preds) == 0, \
        "Marketing/noreply addresses should still be filtered from relationship predictions"


@pytest.mark.asyncio
async def test_weekly_contact_threshold_logic(db: DatabaseManager):
    """
    Test the threshold logic for a typical weekly contact (7 day gaps).

    Threshold = avg_gap * 1.5 = 7 * 1.5 = 10.5 days
    Should predict after 11+ days since last contact.
    """
    ums = UserModelStore(db)
    now = datetime.now(timezone.utc)

    weekly_timestamps = [
        (now - timedelta(days=7 * i)).isoformat()
        for i in range(5, 0, -1)
    ]

    rel_data = {
        "contacts": {
            "weekly-on-time@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=7)).isoformat(),  # Exactly on schedule
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            },
            "weekly-overdue@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=15)).isoformat(),  # 4 days overdue
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            }
        }
    }

    ums.update_signal_profile("relationships", rel_data)
    engine = PredictionEngine(db, ums)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    # Should NOT predict for on-time contact (7 days < 10.5 day threshold)
    # Description now uses resolved contact name (email prefix as fallback)
    on_time_preds = [p for p in predictions if "weekly-on-time" in p.description]
    assert len(on_time_preds) == 0, \
        "Should not predict for weekly contact that's on schedule (7d < 10.5d threshold)"

    # SHOULD predict for overdue contact (15 days > 10.5 day threshold AND > 7 days minimum)
    overdue_preds = [p for p in predictions if "weekly-overdue" in p.description]
    assert len(overdue_preds) > 0, \
        "Should predict for weekly contact that's overdue (15d > 10.5d threshold)"


@pytest.mark.asyncio
async def test_minimum_7_day_threshold(db: DatabaseManager):
    """
    Test that the 7-day minimum prevents nagging about very frequent contacts.

    Even if a contact is technically "overdue" (days_since > avg_gap * 1.5),
    we don't nag until at least 7 days have passed. This prevents annoying
    predictions for daily/hourly contacts who are a bit late.
    """
    ums = UserModelStore(db)
    now = datetime.now(timezone.utc)

    # Contact with 2-hour gaps (very frequent)
    hourly_timestamps = [
        (now - timedelta(hours=2 * i)).isoformat()
        for i in range(20, 0, -1)
    ]

    rel_data = {
        "contacts": {
            "hourly@example.com": {
                "interaction_count": 20,
                "last_interaction": (now - timedelta(days=1)).isoformat(),  # 1 day late
                "interaction_timestamps": hourly_timestamps,
                "inbound_count": 10,
                "outbound_count": 10,
            }
        }
    }

    ums.update_signal_profile("relationships", rel_data)
    engine = PredictionEngine(db, ums)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    # Should NOT predict (days_since=1 < 7 day minimum, even though overdue by threshold)
    # Description now uses resolved contact name (email prefix as fallback)
    hourly_preds = [p for p in predictions if "hourly" in p.description]
    assert len(hourly_preds) == 0, \
        "Should not predict for very frequent contact until 7+ days pass (prevents nagging)"

    # Now make it 8 days since last contact - should predict
    rel_data["contacts"]["hourly@example.com"]["last_interaction"] = \
        (now - timedelta(days=8)).isoformat()
    ums.update_signal_profile("relationships", rel_data)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    hourly_preds = [p for p in predictions if "hourly" in p.description]
    assert len(hourly_preds) > 0, \
        "Should predict after 8 days (exceeds both threshold and 7-day minimum)"


@pytest.mark.asyncio
async def test_confidence_scaling(db: DatabaseManager):
    """
    Test that confidence scales based on how far past threshold the contact is.

    Formula: confidence = min(0.6, 0.3 + (days_since / avg_gap - 1.5) * 0.2)

    At threshold (days_since = avg_gap * 1.5): confidence = 0.3
    At 2x avg_gap: confidence = 0.3 + (2.0 - 1.5) * 0.2 = 0.4
    At 3x avg_gap: confidence = 0.3 + (3.0 - 1.5) * 0.2 = 0.6 (capped)
    """
    ums = UserModelStore(db)
    now = datetime.now(timezone.utc)

    weekly_timestamps = [
        (now - timedelta(days=7 * i)).isoformat()
        for i in range(5, 0, -1)
    ]

    rel_data = {
        "contacts": {
            "barely-overdue@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=11)).isoformat(),  # 1.57x avg_gap
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            },
            "moderately-overdue@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=14)).isoformat(),  # 2x avg_gap
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            },
            "very-overdue@example.com": {
                "interaction_count": 5,
                "last_interaction": (now - timedelta(days=21)).isoformat(),  # 3x avg_gap
                "interaction_timestamps": weekly_timestamps,
                "inbound_count": 3,
                "outbound_count": 2,
            }
        }
    }

    ums.update_signal_profile("relationships", rel_data)
    engine = PredictionEngine(db, ums)

    engine._last_time_based_run = None
    predictions = await engine._check_relationship_maintenance({})

    # Find each prediction (descriptions now use resolved contact name, email prefix as fallback)
    barely = [p for p in predictions if "barely-overdue" in p.description][0]
    moderately = [p for p in predictions if "moderately-overdue" in p.description][0]
    very = [p for p in predictions if "very-overdue" in p.description][0]

    # Verify confidence increases with how overdue they are
    assert barely.confidence < moderately.confidence < very.confidence, \
        "Confidence should increase as contact becomes more overdue"

    # Verify barely is around 0.3 (threshold)
    assert 0.29 <= barely.confidence <= 0.35, \
        f"Barely overdue should have confidence ~0.3, got {barely.confidence}"

    # Verify moderately is around 0.4
    assert 0.38 <= moderately.confidence <= 0.45, \
        f"Moderately overdue should have confidence ~0.4, got {moderately.confidence}"

    # Verify very is capped at 0.6
    assert very.confidence == 0.6, \
        f"Very overdue should cap at 0.6, got {very.confidence}"
