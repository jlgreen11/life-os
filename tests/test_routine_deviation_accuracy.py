"""
Tests for behavioral accuracy tracking of routine_deviation predictions.

This test suite verifies that BehavioralAccuracyTracker correctly infers
prediction accuracy for the 'routine_deviation' prediction type, which was
previously unhandled (always returned None, preventing the learning loop
from ever resolving these predictions).

Routine deviation predictions are generated when the user deviates from a
detected routine pattern (e.g., "You usually do your 'morning_email_review'
routine by now"). The accuracy is inferred by checking whether the expected
routine events occurred within the observation window.

Before this fix (iteration 172):
    routine_deviation predictions always returned None from _infer_accuracy,
    so they were never resolved as accurate or inaccurate. The data quality
    analysis confirmed: 4 total, 0 accurate, 0 inaccurate, 4 unresolved.

After this fix:
    routine_deviation predictions are resolved within a 4-hour observation window:
    - Accurate: routine events occurred within 2 hours of the prediction
    - Inaccurate: no routine events within 4 hours (legitimate skip day)
    - None: still within the observation window
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ============================================================================
# Helper: insert a routine_deviation prediction
# ============================================================================


def _insert_routine_prediction(
    db,
    prediction_id: str,
    routine_name: str,
    expected_actions: list[str],
    consistency_score: float = 0.85,
    age: timedelta = timedelta(hours=0),
) -> dict:
    """Insert a routine_deviation prediction into the test database.

    Args:
        db: DatabaseManager fixture (from conftest).
        prediction_id: Unique ID for the prediction.
        routine_name: Name of the routine (e.g., "morning_email_review").
        expected_actions: List of action names the routine expects
            (e.g., ["email_received", "task_created"]).
        consistency_score: How consistently this routine occurs (0-1).
        age: How long ago the prediction was created.

    Returns:
        The prediction dict as it would be returned from the DB.
    """
    created_at = datetime.now(timezone.utc) - age
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction_id,
                "routine_deviation",
                f"You usually do your '{routine_name}' routine by now",
                min(consistency_score * 0.5, 0.65),
                "suggest",
                "today",
                f"Start {routine_name}",
                json.dumps({
                    "routine_name": routine_name,
                    "consistency_score": consistency_score,
                    "expected_actions": expected_actions,
                }),
                1,
                created_at.isoformat(),
            ),
        )
    return {
        "id": prediction_id,
        "prediction_type": "routine_deviation",
        "description": f"You usually do your '{routine_name}' routine by now",
        "suggested_action": f"Start {routine_name}",
        "supporting_signals": json.dumps({
            "routine_name": routine_name,
            "consistency_score": consistency_score,
            "expected_actions": expected_actions,
        }),
        "created_at": created_at.isoformat(),
    }


def _insert_event(db, event_type: str, age: timedelta = timedelta(minutes=30)):
    """Insert a synthetic event into events.db.

    Args:
        db: DatabaseManager fixture.
        event_type: The event type string (e.g., "email.received").
        age: How long ago the event occurred relative to now.
    """
    ts = datetime.now(timezone.utc) - age
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                event_type,
                "test",
                ts.isoformat(),
                "normal",
                json.dumps({}),
            ),
        )


# ============================================================================
# Core accuracy inference tests
# ============================================================================


@pytest.mark.asyncio
async def test_routine_deviation_accurate_within_two_hours(db):
    """Prediction is ACCURATE when routine events occur within 2 hours."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_email_review",
        expected_actions=["email_received"],
        age=timedelta(hours=1),  # prediction created 1h ago
    )

    # email.received occurred 30 minutes after the prediction (well within 2h window)
    _insert_event(db, "email.received", age=timedelta(minutes=30))

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is True, "Should be ACCURATE: routine events occurred within 2h window"


@pytest.mark.asyncio
async def test_routine_deviation_accurate_within_four_hours(db):
    """Prediction is ACCURATE even when routine events occur between 2-4 hours.

    A late start is still a confirmation that the deviation was detected correctly.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="task_management_review",
        expected_actions=["task_created"],
        age=timedelta(hours=3),  # prediction created 3h ago
    )

    # task.created occurred 30 minutes after the prediction (within 4h window)
    _insert_event(db, "task.created", age=timedelta(hours=2, minutes=30))

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is True, "Should be ACCURATE: routine events occurred within 4h window"


@pytest.mark.asyncio
async def test_routine_deviation_inaccurate_after_four_hours(db):
    """Prediction is INACCURATE when no routine events occur within 4 hours.

    This represents a legitimate skip day, not a detectable deviation.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_email_review",
        expected_actions=["email_received"],
        age=timedelta(hours=5),  # prediction created 5h ago (past the 4h window)
    )

    # No email.received events inserted → routine was skipped

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is False, "Should be INACCURATE: no events after 4h window elapsed"


@pytest.mark.asyncio
async def test_routine_deviation_pending_within_window(db):
    """Returns None when still within the observation window with no events yet."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_email_review",
        expected_actions=["email_received"],
        age=timedelta(minutes=30),  # prediction only 30 minutes old
    )

    # No events inserted — too early to determine

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is None, "Should return None: still within observation window"


@pytest.mark.asyncio
async def test_routine_deviation_multiple_expected_actions(db):
    """Accurate when ANY of the expected event types occurs (OR logic)."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_work_routine",
        expected_actions=["email_received", "task_created", "message_received"],
        age=timedelta(hours=1),
    )

    # Only task.created occurred — that's enough (OR logic across all expected types)
    _insert_event(db, "task.created", age=timedelta(minutes=45))

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is True, "Should be ACCURATE: at least one expected event occurred"


@pytest.mark.asyncio
async def test_routine_deviation_ignores_events_before_prediction(db):
    """Events that occurred BEFORE the prediction are not counted as evidence."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_email_review",
        expected_actions=["email_received"],
        age=timedelta(hours=5),  # prediction created 5h ago
    )

    # email.received occurred BEFORE the prediction (6h ago) — should not count
    _insert_event(db, "email.received", age=timedelta(hours=6))

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is False, (
        "Should be INACCURATE: only events before prediction window exist, "
        "4h observation window has elapsed"
    )


@pytest.mark.asyncio
async def test_routine_deviation_no_expected_actions_times_out(db):
    """Falls back to False after 24h when no expected_actions are available."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=25)
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, suggested_action, supporting_signals,
                was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "routine_deviation",
                "You usually do your 'some_routine' routine by now",
                0.45,
                "suggest",
                "today",
                "Start some_routine",
                json.dumps({
                    "routine_name": "some_routine",
                    "consistency_score": 0.9,
                    "expected_actions": [],  # empty — no event types to check
                }),
                1,
                created_at.isoformat(),
            ),
        )

    prediction = {
        "id": pred_id,
        "prediction_type": "routine_deviation",
        "description": "You usually do your 'some_routine' routine by now",
        "suggested_action": "Start some_routine",
        "supporting_signals": json.dumps({
            "routine_name": "some_routine",
            "consistency_score": 0.9,
            "expected_actions": [],
        }),
        "created_at": created_at.isoformat(),
    }

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is False, "Should resolve as False after 24h with no expected actions"


@pytest.mark.asyncio
async def test_routine_deviation_no_expected_actions_pending(db):
    """Returns None within 24h when no expected_actions are available."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc) - timedelta(hours=1)

    prediction = {
        "id": pred_id,
        "prediction_type": "routine_deviation",
        "description": "You usually do your 'some_routine' routine by now",
        "suggested_action": "Start some_routine",
        "supporting_signals": json.dumps({
            "routine_name": "some_routine",
            "consistency_score": 0.9,
            "expected_actions": [],
        }),
        "created_at": created_at.isoformat(),
    }

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is None, "Should return None within 24h with no expected actions"


# ============================================================================
# Integration: run_inference_cycle dispatch test
# ============================================================================


@pytest.mark.asyncio
async def test_run_inference_cycle_resolves_routine_deviation(db):
    """run_inference_cycle dispatches routine_deviation predictions correctly.

    Verifies that the full inference cycle (not just the isolated method)
    resolves routine_deviation predictions that are past their window.
    """
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    # Created 6h ago with expected actions → well past the 4h inaccurate window
    _insert_routine_prediction(
        db,
        pred_id,
        routine_name="morning_email_review",
        expected_actions=["email_received"],
        age=timedelta(hours=6),
    )

    # Mark as surfaced (run_inference_cycle only processes surfaced predictions)
    with db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE predictions SET was_surfaced = 1 WHERE id = ?",
            (pred_id,),
        )

    stats = await tracker.run_inference_cycle()

    # Should have resolved at least one prediction
    total_resolved = stats["marked_accurate"] + stats["marked_inaccurate"]
    assert total_resolved >= 1, (
        "run_inference_cycle should resolve at least one routine_deviation prediction"
    )

    # Verify the prediction was actually updated in the database
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT was_accurate FROM predictions WHERE id = ?",
            (pred_id,),
        ).fetchone()

    assert row is not None, "Prediction should still exist"
    assert row["was_accurate"] is not None, (
        "was_accurate should be set after inference cycle"
    )


@pytest.mark.asyncio
async def test_routine_deviation_underscore_to_dot_mapping(db):
    """Verifies action-to-event-type mapping converts underscores to dots correctly."""
    tracker = BehavioralAccuracyTracker(db)

    pred_id = str(uuid.uuid4())
    prediction = _insert_routine_prediction(
        db,
        pred_id,
        routine_name="end_of_day_review",
        # Mix of known actions and unknown actions with underscores
        expected_actions=["email_sent", "task_completed"],
        age=timedelta(hours=1),
    )

    # email.sent occurred 45 minutes after the prediction
    _insert_event(db, "email.sent", age=timedelta(minutes=15))

    result = await tracker._infer_routine_deviation_accuracy(
        prediction,
        json.loads(prediction["supporting_signals"]),
        datetime.fromisoformat(prediction["created_at"]),
    )

    assert result is True, (
        "Should correctly map 'email_sent' → 'email.sent' and find the event"
    )
