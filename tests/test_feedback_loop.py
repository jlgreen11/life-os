"""
Tests for the prediction feedback loop.

Verifies that prediction accuracy tracking works correctly:
- Dismissed predictions reduce future confidence for that prediction type
- Accurate predictions maintain or boost future confidence
- Insufficient data returns a neutral multiplier (1.0)
- Very low accuracy (<20% after 10+ samples) gets the 0.3 heavy-penalty floor
  (NOT a hard 0.0, which would create a permanent death spiral)
"""

import pytest
import json
import uuid
from datetime import datetime, timezone
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_prediction_accuracy_decay(db, user_model_store):
    """Dismissed predictions should reduce future confidence for that type."""
    engine = PredictionEngine(db, user_model_store)

    # Store 10 predictions of type "reminder", all marked was_accurate=False
    for i in range(10):
        pred_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at)
                   VALUES (?, ?, ?, ?, ?, 1, 0, ?)""",
                (pred_id, "reminder", f"Test prediction {i}", 0.7, "suggest",
                 datetime.now(timezone.utc).isoformat()),
            )

    # Engine should compute a reduced multiplier for "reminder" type
    multiplier = engine._get_accuracy_multiplier("reminder")
    assert multiplier < 1.0, f"Expected multiplier < 1.0 for inaccurate predictions, got {multiplier}"


@pytest.mark.asyncio
async def test_accurate_predictions_maintain_multiplier(db, user_model_store):
    """Accurate predictions should maintain or boost future confidence."""
    engine = PredictionEngine(db, user_model_store)

    # Store 10 predictions of type "conflict", all marked was_accurate=True
    for i in range(10):
        pred_id = str(uuid.uuid4())
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at)
                   VALUES (?, ?, ?, ?, ?, 1, 1, ?)""",
                (pred_id, "conflict", f"Test prediction {i}", 0.9, "default",
                 datetime.now(timezone.utc).isoformat()),
            )

    multiplier = engine._get_accuracy_multiplier("conflict")
    assert multiplier >= 1.0, f"Expected multiplier >= 1.0 for accurate predictions, got {multiplier}"


@pytest.mark.asyncio
async def test_insufficient_data_returns_neutral_multiplier(db, user_model_store):
    """With fewer than 5 resolved predictions, return 1.0 (no adjustment)."""
    engine = PredictionEngine(db, user_model_store)
    multiplier = engine._get_accuracy_multiplier("nonexistent_type")
    assert multiplier == 1.0


@pytest.mark.asyncio
async def test_heavy_penalty_floor_below_20_percent_accuracy(db, user_model_store):
    """Prediction types with <20% accuracy after 10+ samples get the 0.3 heavy-penalty floor.

    The floor is 0.3, not 0.0.  A hard 0.0 multiplier creates a permanent death spiral:
    the type is blocked, no new predictions are generated, no new data enters the feedback
    loop, and accuracy can never recover.  With 0.3 the type is heavily penalised but
    remains active, allowing the learning loop to rehabilitate it naturally.
    """
    engine = PredictionEngine(db, user_model_store)

    # 10 inaccurate + 1 accurate = ~9% accuracy
    for i in range(10):
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at)
                   VALUES (?, ?, ?, ?, ?, 1, 0, ?)""",
                (str(uuid.uuid4()), "reminder", f"bad {i}", 0.5, "suggest",
                 datetime.now(timezone.utc).isoformat()),
            )
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, was_accurate, resolved_at)
               VALUES (?, ?, ?, ?, ?, 1, 1, ?)""",
            (str(uuid.uuid4()), "reminder", "good one", 0.5, "suggest",
             datetime.now(timezone.utc).isoformat()),
        )

    multiplier = engine._get_accuracy_multiplier("reminder")
    # Heavy penalty but NOT a hard cutoff — 0.3 floor keeps the learning loop alive
    assert multiplier == 0.3, f"Expected 0.3 penalty floor for <20% accuracy, got {multiplier}"
