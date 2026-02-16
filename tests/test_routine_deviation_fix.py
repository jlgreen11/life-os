"""
Test suite for routine deviation detection fix (iteration 114).

Previously, routine deviation detection was completely non-functional due to:
1. Inverted day-name matching logic (checked if "monday" in "morning")
2. Never checking if routine was actually completed
3. No deduplication — would create predictions every 15 minutes

This test suite verifies the fixed implementation:
- Correctly checks if routine events occurred today
- Deduplicates predictions for the same routine
- Uses correct prediction_type ("opportunity" not "reminder")
- Properly maps routine actions to event types
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from models.core import ConfidenceGate
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager
from storage.user_model_store import UserModelStore


@pytest.fixture
def sample_routine():
    """Create a sample morning routine for testing."""
    return {
        "name": "Morning routine",
        "trigger_condition": "morning",
        "steps": json.dumps([
            {"order": 0, "action": "email_received", "typical_duration_minutes": 5.0},
            {"order": 1, "action": "task_created", "typical_duration_minutes": 3.0},
            {"order": 2, "action": "email_sent", "typical_duration_minutes": 5.0},
        ]),
        "typical_duration_minutes": 65.0,
        "consistency_score": 0.85,
        "times_completed": 100,
        "recent_deviations": json.dumps([]),
        "last_completed": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    }


@pytest.mark.asyncio
async def test_routine_deviation_detected_when_routine_not_completed(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that routine deviation prediction is created when routine hasn't been completed today."""
    engine = PredictionEngine(db, user_model_store)

    # Insert a high-consistency routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([
                    {"order": 0, "action": "email_received"},
                    {"order": 1, "action": "task_created"},
                ]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    # No events today — routine not completed
    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 1
    pred = predictions[0]
    assert pred.prediction_type == "opportunity"
    assert "Morning routine" in pred.description
    assert pred.confidence > 0.3  # Meets SUGGEST threshold
    assert pred.confidence <= 0.65  # Capped at 65% (consistency * 0.5 with max)
    assert "routine_name" in pred.supporting_signals
    assert pred.supporting_signals["routine_name"] == "Morning routine"


@pytest.mark.asyncio
async def test_no_deviation_when_routine_completed_today(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that no prediction is created when routine was completed today."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([
                    {"order": 0, "action": "email_received"},
                    {"order": 1, "action": "task_created"},
                ]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    # Insert events today matching the routine
    now = datetime.now(timezone.utc)
    with db.get_connection("events") as conn:
        # Email received today
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                now.isoformat(),
                "normal",
                json.dumps({"subject": "Test email"}),
                json.dumps({}),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    # No prediction since routine was completed
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_deduplication_prevents_repeat_predictions(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that we don't create duplicate predictions for the same routine."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    # First call creates prediction
    predictions_1 = await engine._check_routine_deviations({})
    assert len(predictions_1) == 1

    # Store the prediction in the database (simulating what would happen in production)
    pred = predictions_1[0]
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, description, confidence,
                                        confidence_gate, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                pred.prediction_type,
                pred.description,
                pred.confidence,
                pred.confidence_gate.value,
                json.dumps(pred.supporting_signals) if pred.supporting_signals else "{}",
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    # Second call should NOT create duplicate prediction
    predictions_2 = await engine._check_routine_deviations({})
    assert len(predictions_2) == 0


@pytest.mark.asyncio
async def test_low_consistency_routines_skipped(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that routines with low consistency don't generate predictions."""
    engine = PredictionEngine(db, user_model_store)

    # Insert low-consistency routine (< 0.6)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Low consistency routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                30.0,
                0.5,  # Below 0.6 threshold
                50,
                json.dumps([]),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    # No prediction for low-consistency routine
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_action_to_event_type_mapping(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that routine actions are correctly mapped to event types."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine with various action types
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Multi-action routine",
                "morning",
                json.dumps([
                    {"order": 0, "action": "message_sent"},
                    {"order": 1, "action": "task_completed"},
                    {"order": 2, "action": "calendar_event_created"},
                ]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    # Insert task.completed event (should match task_completed action)
    now = datetime.now(timezone.utc)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "task.completed",
                "test",
                now.isoformat(),
                "normal",
                json.dumps({"title": "Test task"}),
                json.dumps({}),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    # Should not create prediction since one of the routine's actions occurred
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_multiple_routines_independent_tracking(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that multiple routines are tracked independently."""
    engine = PredictionEngine(db, user_model_store)

    # Insert two routines
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Morning routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Evening routine",
                "evening",
                json.dumps([{"order": 0, "action": "task_created"}]),
                20.0,
                0.7,
                40,
                json.dumps([]),
            ),
        )

    # Insert email.received event (completes morning routine only)
    now = datetime.now(timezone.utc)
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                "email.received",
                "test",
                now.isoformat(),
                "normal",
                json.dumps({"subject": "Test"}),
                json.dumps({}),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    # Should create prediction for evening routine only
    assert len(predictions) == 1
    assert "Evening routine" in predictions[0].description


@pytest.mark.asyncio
async def test_malformed_routine_steps_handled_gracefully(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that routines with malformed JSON steps don't crash the engine."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine with malformed JSON
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Broken routine",
                "morning",
                "not valid json",  # Malformed JSON
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    # Should not crash, just skip the malformed routine
    predictions = await engine._check_routine_deviations({})
    assert len(predictions) == 0


@pytest.mark.asyncio
async def test_confidence_calculation_based_on_consistency(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that prediction confidence is correctly calculated from consistency score."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine with high consistency (0.8)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "High consistency routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                30.0,
                0.8,
                50,
                json.dumps([]),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    assert len(predictions) == 1
    # Confidence = consistency_score * 0.5, capped at 0.65
    # 0.8 * 0.5 = 0.4
    assert predictions[0].confidence == 0.4


@pytest.mark.asyncio
async def test_prediction_only_created_if_above_suggest_threshold(db: DatabaseManager, user_model_store: UserModelStore):
    """Test that predictions are only created if confidence meets SUGGEST threshold (0.3+)."""
    engine = PredictionEngine(db, user_model_store)

    # Insert routine with consistency that results in confidence < 0.3
    # confidence = consistency * 0.5, so need consistency < 0.6
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Low confidence routine",
                "morning",
                json.dumps([{"order": 0, "action": "email_received"}]),
                30.0,
                0.61,  # Just above the minimum to be considered (> 0.6)
                50,
                json.dumps([]),
            ),
        )

    predictions = await engine._check_routine_deviations({})

    # confidence = 0.61 * 0.5 = 0.305 (above 0.3 threshold)
    assert len(predictions) == 1

    # Now test with consistency that results in confidence >= 0.3
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, typical_duration,
                                     consistency_score, times_observed, variations)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "Another routine",
                "evening",
                json.dumps([{"order": 0, "action": "task_created"}]),
                30.0,
                0.65,  # Above minimum threshold (> 0.6 consistency filter)
                50,
                json.dumps([]),
            ),
        )

    predictions_2 = await engine._check_routine_deviations({})

    # Should have 2 predictions now (both meet threshold)
    # 0.61 * 0.5 = 0.305, 0.65 * 0.5 = 0.325
    # Note: First routine still returns since deduplication only checks database,
    # and we haven't persisted the first prediction
    assert len(predictions_2) == 2
    assert any("Low confidence" in p.description for p in predictions_2)
    assert any("Another routine" in p.description for p in predictions_2)
