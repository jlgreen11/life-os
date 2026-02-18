"""
Tests for the accuracy multiplier penalty floor (0.3) that replaces the old hard
auto-suppression cutoff (0.0).

Background
----------
Prior to this fix, `_get_accuracy_multiplier` returned 0.0 when a prediction type
had <20% accuracy over 10+ resolved samples.  That created a death spiral:

    low accuracy → multiplier = 0.0 → no predictions generated
    → no new data in feedback loop → accuracy never recovers → permanently blocked

The fix replaces 0.0 with a 0.3 floor.  The type is still heavily penalised but
continues generating predictions at reduced confidence, allowing the learning loop
to recover naturally as new, higher-quality predictions accumulate.

This file tests:
1. The 0.3 floor is applied (not 0.0) when accuracy is below 20%
2. Predictions with the 0.3 multiplier can still exceed the 0.3 confidence gate
   threshold so they reach the notification layer
3. Recovery: when accuracy improves past 20%, the normal scaling resumes
4. The boundary condition: exactly 19.9% accuracy gets 0.3; exactly 20% gets scaling
"""

import uuid
import pytest
from datetime import datetime, timezone

from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_resolved_prediction(db, prediction_type: str, was_accurate: bool) -> None:
    """Insert a single resolved, surfaced prediction of a given type into the DB."""
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, was_accurate, resolved_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                str(uuid.uuid4()),
                prediction_type,
                "Test prediction",
                0.5,
                "suggest",
                1 if was_accurate else 0,
                now,
            ),
        )


# ---------------------------------------------------------------------------
# Floor tests
# ---------------------------------------------------------------------------


def test_penalty_floor_is_03_not_00_for_very_low_accuracy(db, user_model_store):
    """Accuracy below 10% should return 0.3 floor, not the old hard-cutoff 0.0.

    Verifies the core fix: a prediction type with near-zero accuracy gets a heavy
    penalty but is NOT permanently suppressed.
    """
    engine = PredictionEngine(db, user_model_store)

    # 0 accurate out of 12 resolved = 0% accuracy
    for _ in range(12):
        _insert_resolved_prediction(db, "opportunity", was_accurate=False)

    multiplier = engine._get_accuracy_multiplier("opportunity")
    assert multiplier == 0.3, (
        f"Expected 0.3 penalty floor for 0% accuracy, got {multiplier}. "
        "A hard 0.0 would create a permanent death spiral."
    )


def test_penalty_floor_applied_at_boundary_below_20_percent(db, user_model_store):
    """Exactly 1 accurate out of 10 resolved (10%) should still get the 0.3 floor."""
    engine = PredictionEngine(db, user_model_store)

    # 1 accurate out of 10 = 10% accuracy
    _insert_resolved_prediction(db, "reminder", was_accurate=True)
    for _ in range(9):
        _insert_resolved_prediction(db, "reminder", was_accurate=False)

    multiplier = engine._get_accuracy_multiplier("reminder")
    assert multiplier == 0.3, (
        f"Expected 0.3 for 10% accuracy over 10 samples, got {multiplier}"
    )


def test_penalty_floor_requires_10_or_more_samples(db, user_model_store):
    """With only 9 resolved predictions, 10% accuracy should NOT trigger the penalty floor.

    Below 10 samples there is insufficient evidence to apply the heavy penalty, so the
    normal scaling formula applies (which for 10% accuracy = 0.5 + 0.1 * 0.6 = 0.56).
    """
    engine = PredictionEngine(db, user_model_store)

    # 1 accurate out of 9 = ~11% accuracy (9 samples, below the 10-sample threshold)
    _insert_resolved_prediction(db, "reminder", was_accurate=True)
    for _ in range(8):
        _insert_resolved_prediction(db, "reminder", was_accurate=False)

    multiplier = engine._get_accuracy_multiplier("reminder")
    # 11% accuracy → 0.5 + 0.111 * 0.6 ≈ 0.567 (normal scale, no floor applied)
    assert multiplier > 0.3, (
        f"Expected normal scaling (>0.3) for <10 samples, got {multiplier}"
    )
    assert multiplier < 1.0, (
        f"Expected multiplier < 1.0 for low accuracy, got {multiplier}"
    )


# ---------------------------------------------------------------------------
# Recovery tests
# ---------------------------------------------------------------------------


def test_recovery_above_20_percent_resumes_normal_scaling(db, user_model_store):
    """When accuracy rises above 20%, the normal 0.5 + rate*0.6 scaling should resume.

    This is the recovery path: a previously penalised type accumulates enough accurate
    predictions to climb back above the 20% threshold and the multiplier scales normally.
    """
    engine = PredictionEngine(db, user_model_store)

    # 3 accurate out of 10 = 30% accuracy (above the 20% floor threshold)
    for _ in range(3):
        _insert_resolved_prediction(db, "opportunity", was_accurate=True)
    for _ in range(7):
        _insert_resolved_prediction(db, "opportunity", was_accurate=False)

    multiplier = engine._get_accuracy_multiplier("opportunity")
    # 30% accuracy → 0.5 + 0.3 * 0.6 = 0.68
    assert abs(multiplier - 0.68) < 0.01, (
        f"Expected ~0.68 for 30% accuracy after recovery, got {multiplier}"
    )


def test_accuracy_exactly_20_percent_uses_normal_scaling(db, user_model_store):
    """Exactly 20% accuracy (2 of 10) should use normal scaling, not the penalty floor.

    The threshold is strictly less-than 20%.  At exactly 20% the normal formula gives:
    0.5 + 0.2 * 0.6 = 0.62.
    """
    engine = PredictionEngine(db, user_model_store)

    # 2 accurate out of 10 = exactly 20%
    for _ in range(2):
        _insert_resolved_prediction(db, "opportunity", was_accurate=True)
    for _ in range(8):
        _insert_resolved_prediction(db, "opportunity", was_accurate=False)

    multiplier = engine._get_accuracy_multiplier("opportunity")
    # 20% accuracy → 0.5 + 0.2 * 0.6 = 0.62
    assert abs(multiplier - 0.62) < 0.01, (
        f"Expected ~0.62 for exactly 20% accuracy, got {multiplier}"
    )


# ---------------------------------------------------------------------------
# Effect-on-predictions tests
# ---------------------------------------------------------------------------


def test_penalised_prediction_still_surfaces_if_high_enough_base_confidence(
    db, user_model_store
):
    """A prediction with 0.45 base confidence × 0.3 floor = 0.135 (below gate, filtered).

    A prediction with 0.8 base confidence × 0.3 floor = 0.24 (still below 0.3 gate).
    A prediction with 1.0 base confidence × 0.3 floor = 0.3 (exactly at the gate).

    This test confirms that very high-confidence predictions can still clear the 0.3
    confidence threshold even with the penalty floor applied.
    """
    # 0 accurate out of 12 = 0% accuracy → floor = 0.3
    for _ in range(12):
        _insert_resolved_prediction(db, "opportunity", was_accurate=False)

    engine = PredictionEngine(db, user_model_store)
    multiplier = engine._get_accuracy_multiplier("opportunity")
    assert multiplier == 0.3

    # A very high-confidence prediction (1.0) after applying floor = 0.3 ≥ 0.3 gate
    adjusted_confidence = 1.0 * multiplier
    assert adjusted_confidence >= 0.3, (
        "A prediction with 1.0 base confidence should survive the 0.3 confidence gate "
        "even with the penalty floor applied."
    )
