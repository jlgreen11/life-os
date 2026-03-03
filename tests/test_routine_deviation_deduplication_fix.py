"""
Test suite for routine deviation deduplication bug fix.

This test validates the fix for a critical bug in _check_routine_deviations()
where the deduplication query incorrectly looked for prediction_type = 'opportunity'
instead of prediction_type = 'routine_deviation', causing:

1. Deduplication to fail (query returns wrong prediction type)
2. Infinite duplicate predictions every 15 minutes
3. 0 routine_deviation predictions generated (all filtered due to event checks)

The fix changes line 696 from:
    WHERE prediction_type = 'opportunity'
to:
    WHERE prediction_type = 'routine_deviation'

Impact: Enables routine deviation predictions (0 → active predictions).
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine


@pytest.fixture
def prediction_engine(db, user_model_store):
    """Create a PredictionEngine instance with test database."""
    return PredictionEngine(db=db, ums=user_model_store)


@pytest.fixture
def sample_routine(db):
    """
    Create a sample routine in the database.

    This routine has:
    - High consistency (1.0) to ensure predictions are generated
    - Morning pattern (not time-specific to avoid filtering)
    - Steps that check for email_received events
    """
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "Morning email check",
                "morning",
                json.dumps([
                    {"order": 0, "action": "email_received", "typical_duration_minutes": 5.0},
                    {"order": 1, "action": "email_received", "typical_duration_minutes": 5.0},
                ]),
                1.0,  # Perfect consistency
                50,   # Well-established routine
            ),
        )
        conn.commit()


def test_deduplication_query_uses_correct_prediction_type(db, user_model_store, prediction_engine, sample_routine):
    """
    Test that deduplication query looks for 'routine_deviation' not 'opportunity'.

    This is the core bug fix validation. Before the fix, the query at line 696
    looked for prediction_type = 'opportunity', which would never match routine
    deviation predictions, causing deduplication to fail.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Use today_start + 1 hour as the creation time so the existing prediction
    # is always within "today" regardless of when the test runs.
    # Using (now - timedelta(hours=2)) causes a failure when the test runs in
    # the first 2 UTC hours of the day (the prediction lands on the previous UTC
    # day and the deduplication query misses it).
    created_today = today_start + timedelta(hours=1)

    # Create a routine_deviation prediction from 1 hour into today (UTC)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (
                id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-routine-deviation-1",
                "routine_deviation",  # Critical: this is the prediction type to match
                "You usually do your 'Morning email check' routine by now",
                0.5,
                "SUGGEST",
                "today",
                "Start Morning email check",
                json.dumps({"routine_name": "Morning email check"}),
                0,
                created_today.isoformat(),
            ),
        )
        conn.commit()

    # Also create an 'opportunity' prediction to verify it's NOT matched.
    # Both predictions share the same routine_name in supporting_signals; only
    # the routine_deviation one should block deduplication.
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (
                id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-opportunity-1",
                "opportunity",  # Different prediction type — must NOT block dedup
                "Good time to contact someone",
                0.6,
                "DEFAULT",
                json.dumps({"routine_name": "Morning email check"}),  # Same routine_name
                (created_today + timedelta(minutes=30)).isoformat(),
            ),
        )
        conn.commit()

    # Run prediction generation
    # The deduplication check should find the routine_deviation prediction
    # and skip creating a duplicate, even though no events occurred today
    predictions = asyncio.run(prediction_engine._check_routine_deviations({}))

    # Verify deduplication worked:
    # - Should NOT create a new prediction for "Morning email check"
    # - The existing routine_deviation prediction should block it
    routine_names = [
        pred.supporting_signals.get("routine_name")
        for pred in predictions
    ]

    assert "Morning email check" not in routine_names, (
        "Deduplication failed: created duplicate prediction for routine that "
        "already has a routine_deviation prediction today"
    )


def test_deduplication_ignores_opportunity_predictions(db, user_model_store, prediction_engine, sample_routine):
    """
    Test that opportunity predictions don't prevent routine_deviation predictions.

    Before the fix, the query looked for 'opportunity' predictions, which meant
    if an opportunity prediction existed with the same routine_name, it would
    incorrectly block routine_deviation predictions.
    """
    now = datetime.now(timezone.utc)

    # Create ONLY an opportunity prediction (not routine_deviation)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (
                id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-opportunity-2",
                "opportunity",
                "Good time for morning routine",
                0.6,
                "DEFAULT",
                json.dumps({"routine_name": "Morning email check"}),
                (now - timedelta(hours=1)).isoformat(),
            ),
        )
        conn.commit()

    # Run prediction generation
    # Should create a routine_deviation prediction because the opportunity
    # prediction should NOT be matched by the deduplication query
    predictions = asyncio.run(prediction_engine._check_routine_deviations({}))

    # Verify a routine_deviation prediction was created
    routine_predictions = [
        pred for pred in predictions
        if pred.supporting_signals.get("routine_name") == "Morning email check"
    ]

    assert len(routine_predictions) == 1, (
        "Deduplication incorrectly blocked routine_deviation prediction "
        "due to matching opportunity prediction with same routine_name"
    )
    assert routine_predictions[0].prediction_type == "routine_deviation"


def test_routine_deviation_prediction_generated_when_routine_skipped(
    db, user_model_store, prediction_engine, sample_routine
):
    """
    Test that routine_deviation predictions are generated when routine is skipped.

    This validates the happy path: when a routine hasn't been completed today
    and no prior routine_deviation prediction exists, a new one should be created.
    """
    # No events today, no prior predictions
    predictions = asyncio.run(prediction_engine._check_routine_deviations({}))

    # Should create a routine_deviation prediction
    assert len(predictions) == 1
    pred = predictions[0]

    assert pred.prediction_type == "routine_deviation"
    assert pred.supporting_signals["routine_name"] == "Morning email check"
    assert pred.confidence > 0.3  # Should meet SUGGEST threshold
    # time_horizon should be a valid ISO datetime (end of today), not the literal "today"
    parsed_horizon = datetime.fromisoformat(pred.time_horizon)
    assert parsed_horizon.hour == 23
    assert parsed_horizon.minute == 59
    assert "Morning email check" in pred.description


def test_routine_deviation_not_generated_when_routine_completed(
    db, user_model_store, prediction_engine, sample_routine
):
    """
    Test that routine_deviation predictions are NOT generated when routine completed.

    If the expected event types (email_received in this case) occurred today,
    no deviation prediction should be created.
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Create email.received events today (matching the routine steps)
    with db.get_connection("events") as conn:
        for i in range(2):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"test-email-{i}",
                    "email.received",
                    "test",
                    (today_start + timedelta(hours=8, minutes=i*5)).isoformat(),
                    "medium",
                    json.dumps({"subject": f"Test email {i}"}),
                    json.dumps({}),
                ),
            )
        conn.commit()

    # Run prediction generation
    predictions = asyncio.run(prediction_engine._check_routine_deviations({}))

    # Should NOT create a routine_deviation prediction
    routine_predictions = [
        pred for pred in predictions
        if pred.supporting_signals.get("routine_name") == "Morning email check"
    ]

    assert len(routine_predictions) == 0, (
        "Routine deviation prediction created even though routine was completed today"
    )


def test_deduplication_prevents_duplicate_predictions_across_multiple_runs(
    db, user_model_store, prediction_engine, sample_routine
):
    """
    Test that deduplication prevents duplicate predictions across multiple runs.

    This is the critical use case: the prediction loop runs every 15 minutes,
    and without proper deduplication, it would create a new routine_deviation
    prediction every 15 minutes for the same routine.
    """
    # First run: should create a prediction
    predictions_run1 = asyncio.run(prediction_engine._check_routine_deviations({}))
    assert len(predictions_run1) == 1

    # Store the prediction (simulating normal flow)
    pred = predictions_run1[0]
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (
                id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"pred-{pred.supporting_signals['routine_name']}-1",
                pred.prediction_type,
                pred.description,
                pred.confidence,
                pred.confidence_gate,
                pred.time_horizon,
                pred.suggested_action,
                json.dumps(pred.supporting_signals),
                0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    # Second run (15 minutes later): should NOT create a duplicate
    predictions_run2 = asyncio.run(prediction_engine._check_routine_deviations({}))
    assert len(predictions_run2) == 0, (
        "Deduplication failed: created duplicate prediction on second run"
    )

    # Third run (another 15 minutes later): still no duplicate
    predictions_run3 = asyncio.run(prediction_engine._check_routine_deviations({}))
    assert len(predictions_run3) == 0, (
        "Deduplication failed: created duplicate prediction on third run"
    )


def test_multiple_routines_with_independent_deduplication(db, user_model_store, prediction_engine):
    """
    Test that deduplication works independently for multiple routines.

    Each routine should be tracked separately — creating a prediction for
    routine A shouldn't affect routine B.
    """
    # Create two different routines
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                1.0,
                50,
                "Evening routine",
                "evening",
                json.dumps([{"order": 0, "action": "task_completed"}]),
                0.9,
                40,
            ),
        )
        conn.commit()

    # First run: should create predictions for both routines
    predictions_run1 = asyncio.run(prediction_engine._check_routine_deviations({}))
    assert len(predictions_run1) == 2

    # Store both predictions
    for i, pred in enumerate(predictions_run1):
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO predictions (
                    id, prediction_type, description, confidence, confidence_gate,
                    supporting_signals, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"pred-{i}",
                    pred.prediction_type,
                    pred.description,
                    pred.confidence,
                    pred.confidence_gate,
                    json.dumps(pred.supporting_signals),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    # Second run: should not create duplicates for either routine
    predictions_run2 = asyncio.run(prediction_engine._check_routine_deviations({}))
    assert len(predictions_run2) == 0


def test_deduplication_resets_next_day(db, user_model_store, prediction_engine, sample_routine):
    """
    Test that deduplication resets at day boundary.

    Predictions created yesterday should not block predictions today —
    the query filters by created_at > today_start.
    """
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    # Create a routine_deviation prediction from yesterday
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (
                id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-yesterday-pred",
                "routine_deviation",
                "You usually do your 'Morning email check' routine by now",
                0.5,
                "SUGGEST",
                json.dumps({"routine_name": "Morning email check"}),
                yesterday.isoformat(),
            ),
        )
        conn.commit()

    # Run prediction generation today
    # Should create a new prediction because yesterday's shouldn't count
    predictions = asyncio.run(prediction_engine._check_routine_deviations({}))

    assert len(predictions) == 1, (
        "Deduplication incorrectly blocked prediction using yesterday's data"
    )
    assert predictions[0].supporting_signals["routine_name"] == "Morning email check"
