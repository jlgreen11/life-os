"""
Test that routine deviation predictions use the correct prediction_type.

CONTEXT:
The prediction engine generates 6 types of predictions:
- reminder (follow-up needs)
- conflict (calendar conflicts)
- opportunity (relationship maintenance)
- need (preparation needs)
- risk (spending patterns)
- routine_deviation (missed routines)

The diagnostics endpoint (/api/admin/predictions/diagnostics) queries the database
for each prediction_type to determine if that type is "active", "limited", or "blocked".

CRITICAL BUG:
Before this fix, routine deviation predictions were stored with prediction_type="opportunity",
but the diagnostics queried for prediction_type="routine_deviation". This mismatch meant:
- Routine predictions would appear under "opportunity" diagnostics
- "routine_deviation" diagnostics would always show 0 predictions (broken observability)
- Impossible to distinguish relationship maintenance from routine deviations

FIX:
Store routine deviation predictions with prediction_type="routine_deviation" so they
match what the diagnostics query for, enabling proper observability and categorization.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_routine_deviation_uses_correct_prediction_type(db, user_model_store):
    """
    Verify that routine deviation predictions are stored with
    prediction_type='routine_deviation', not 'opportunity'.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a high-consistency routine in the database
    # (consistency_score > 0.6 makes it eligible for deviation detection)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([
                    {"order": 0, "action": "email_received", "typical_duration_minutes": 5.0},
                    {"order": 1, "action": "task_created", "typical_duration_minutes": 5.0},
                    {"order": 2, "action": "email_sent", "typical_duration_minutes": 5.0},
                ]),
                15.0,
                0.85,  # High consistency — reliable routine
                50,
            ),
        )
        conn.commit()

    # Generate predictions (routine hasn't been completed today)
    predictions = await engine._check_routine_deviations({})

    # Should generate exactly 1 prediction
    assert len(predictions) == 1, f"Expected 1 routine deviation prediction, got {len(predictions)}"

    pred = predictions[0]

    # CRITICAL ASSERTION: prediction_type must be "routine_deviation"
    assert pred.prediction_type == "routine_deviation", (
        f"Routine deviation predictions must use prediction_type='routine_deviation', "
        f"got '{pred.prediction_type}'. This breaks diagnostics observability."
    )

    # Verify other fields are correct
    assert "Morning routine" in pred.description
    assert pred.confidence > 0.3  # Should meet SUGGEST threshold
    assert pred.time_horizon == "today"
    assert pred.suggested_action == "Start Morning routine"
    assert pred.supporting_signals["routine_name"] == "Morning routine"


@pytest.mark.asyncio
async def test_routine_deviation_stored_in_database_with_correct_type(db, user_model_store):
    """
    Verify that routine deviation predictions are stored in the database
    with the correct prediction_type so diagnostics can find them.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Evening routine",
                "evening",
                json.dumps([{"order": 0, "action": "task_completed"}]),
                10.0,
                0.75,
                30,
            ),
        )
        conn.commit()

    # Generate and store predictions
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 1

    # Store the prediction in the database (convert to dict)
    user_model_store.store_prediction(predictions[0].model_dump())

    # Query the database to verify prediction_type
    with db.get_connection("user_model") as conn:
        stored = conn.execute(
            "SELECT prediction_type, description FROM predictions WHERE prediction_type = 'routine_deviation'"
        ).fetchall()

    assert len(stored) == 1, "Prediction should be findable by prediction_type='routine_deviation'"
    assert "Evening routine" in stored[0]["description"]


@pytest.mark.asyncio
async def test_opportunity_predictions_do_not_include_routine_deviations(db, user_model_store):
    """
    Verify that routine deviations are NOT stored as 'opportunity' predictions,
    ensuring they don't get mixed with relationship maintenance predictions.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Workout routine",
                "health",
                json.dumps([{"order": 0, "action": "calendar_event_created"}]),
                60.0,
                0.90,
                100,
            ),
        )
        conn.commit()

    # Generate and store predictions
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 1
    user_model_store.store_prediction(predictions[0].model_dump())

    # Query for 'opportunity' predictions — should find NONE from routines
    with db.get_connection("user_model") as conn:
        opportunity_preds = conn.execute(
            "SELECT * FROM predictions WHERE prediction_type = 'opportunity'"
        ).fetchall()

    # CRITICAL: Routine deviations should NOT appear in opportunity predictions
    for pred in opportunity_preds:
        description = pred["description"]
        assert "routine" not in description.lower(), (
            f"Routine deviation found in 'opportunity' predictions: {description}. "
            f"This breaks categorization — routines should use prediction_type='routine_deviation'."
        )


@pytest.mark.asyncio
async def test_diagnostics_can_find_routine_deviation_predictions(db, user_model_store):
    """
    Verify that the diagnostics endpoint can correctly count routine_deviation
    predictions when they're stored in the database.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create 2 routines
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine", "morning", json.dumps([{"order": 0, "action": "email_received"}]),
                15.0, 0.85, 50,
                "Evening routine", "evening", json.dumps([{"order": 0, "action": "task_completed"}]),
                10.0, 0.75, 30,
            ),
        )
        conn.commit()

    # Generate and store predictions
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 2
    for pred in predictions:
        user_model_store.store_prediction(pred.model_dump())

    # Get diagnostics (simulating /api/admin/predictions/diagnostics)
    diagnostics = await engine.get_diagnostics()

    # Verify routine_deviation diagnostics include the stored predictions
    routine_diag = diagnostics["prediction_types"]["routine_deviation"]
    assert routine_diag["generated_last_7d"] == 2, (
        f"Diagnostics should find 2 routine_deviation predictions, "
        f"got {routine_diag['generated_last_7d']}"
    )
    assert routine_diag["status"] == "active", (
        "Routine deviation type should be 'active' when predictions are being generated"
    )


@pytest.mark.asyncio
async def test_routine_deviation_prediction_fields(db, user_model_store):
    """
    Verify that routine deviation predictions contain all expected fields
    for proper notification and feedback loop integration.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Reading routine",
                "learning",
                json.dumps([
                    {"order": 0, "action": "task_created", "typical_duration_minutes": 30.0},
                ]),
                30.0,
                0.70,
                40,
            ),
        )
        conn.commit()

    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 1

    pred = predictions[0]

    # Verify prediction type
    assert pred.prediction_type == "routine_deviation"

    # Verify supporting_signals contains routine metadata
    assert "routine_name" in pred.supporting_signals
    assert pred.supporting_signals["routine_name"] == "Reading routine"
    assert "consistency_score" in pred.supporting_signals
    assert pred.supporting_signals["consistency_score"] == 0.70
    assert "expected_actions" in pred.supporting_signals

    # Verify confidence calculation (consistency_score * 0.5, capped at 0.65)
    expected_confidence = min(0.70 * 0.5, 0.65)
    assert pred.confidence == expected_confidence

    # Verify confidence gate matches confidence level
    assert pred.confidence_gate.value == "suggest"  # 0.35 is in SUGGEST range (0.3-0.6)


@pytest.mark.asyncio
async def test_low_consistency_routines_filtered_out(db, user_model_store):
    """
    Verify that routines with low consistency scores don't generate predictions,
    preventing noise from unreliable patterns.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a low-consistency routine (below 0.6 threshold)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Unreliable routine",
                "misc",
                json.dumps([{"order": 0, "action": "email_sent"}]),
                5.0,
                0.4,  # Below 0.6 threshold
                10,
            ),
        )
        conn.commit()

    # Should generate 0 predictions (filtered by consistency_score check)
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 0, "Low-consistency routines should not generate predictions"


@pytest.mark.asyncio
async def test_completed_routines_do_not_generate_predictions(db, user_model_store):
    """
    Verify that routines that have already been completed today don't
    generate deviation predictions.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a routine that looks for email_received events
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "Email routine",
                "work",
                json.dumps([{"order": 0, "action": "email_received"}]),
                5.0,
                0.80,
                60,
            ),
        )
        conn.commit()

    # Add an email_received event today (completes the routine)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "test-email-1",
                "email.received",
                "google",
                datetime.now(timezone.utc).isoformat(),
                json.dumps({"from_address": "test@example.com", "subject": "Test"}),
                json.dumps({}),
            ),
        )
        conn.commit()

    # Should generate 0 predictions (routine completed today)
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 0, "Completed routines should not generate predictions"
