"""
Tests for cold-start diagnostics in BehavioralAccuracyTracker.

Verifies that:
- get_pipeline_health() correctly detects cold-start (all cycles found 0 predictions)
- Cycle stats (predictions_queried, last_cycle_timestamp) are updated after run_inference_cycle()
- INFO log fires on the 10th empty cycle (and not on every cycle)
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_surfaced_prediction(db, pred_id: str | None = None, *, hours_ago: float = 2.0) -> str:
    """Insert a surfaced, unresolved reminder prediction into user_model.db.

    Returns the inserted prediction id.
    """
    if pred_id is None:
        pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(hours=hours_ago)).isoformat()

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                suggested_action, supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id,
                "reminder",
                "Reply to test contact",
                0.75,
                "DEFAULT",
                "Send message",
                json.dumps({"contact_name": "Test", "contact_id": "test@example.com"}),
                1,
                created_at,
            ),
        )
    return pred_id


# ---------------------------------------------------------------------------
# get_pipeline_health() — cold-start detection
# ---------------------------------------------------------------------------

def test_pipeline_health_initial_state(db):
    """Before any cycles run, health shows zeros and cold_start_detected=False."""
    tracker = BehavioralAccuracyTracker(db)
    health = tracker.get_pipeline_health()

    assert health["total_cycles"] == 0
    assert health["cycles_with_no_predictions"] == 0
    assert health["last_cycle_stats"] is None
    assert health["last_cycle_timestamp"] is None
    # cold_start_detected requires total_cycles > 0, so False before first cycle
    assert health["cold_start_detected"] is False


@pytest.mark.asyncio
async def test_pipeline_health_cold_start_detected_when_no_predictions(db):
    """cold_start_detected=True when every cycle has found 0 surfaced predictions."""
    tracker = BehavioralAccuracyTracker(db)

    # Run one cycle with an empty predictions table
    await tracker.run_inference_cycle()

    health = tracker.get_pipeline_health()

    assert health["total_cycles"] == 1
    assert health["cycles_with_no_predictions"] == 1
    assert health["cold_start_detected"] is True
    assert health["predictions_table_count"] == 0


@pytest.mark.asyncio
async def test_pipeline_health_cold_start_false_when_predictions_exist(db):
    """cold_start_detected=False once any cycle processes surfaced predictions.

    Even if the tracker cannot RESOLVE those predictions (no matching behavior),
    it counted them as 'found', so the system is not cold-starting.
    """
    tracker = BehavioralAccuracyTracker(db)

    # Insert a prediction before running the cycle
    _insert_surfaced_prediction(db)

    await tracker.run_inference_cycle()

    health = tracker.get_pipeline_health()

    assert health["total_cycles"] == 1
    # The prediction was found, so this cycle is NOT an empty cycle
    assert health["cycles_with_no_predictions"] == 0
    assert health["cold_start_detected"] is False
    # The predictions table has at least 1 row
    assert health["predictions_table_count"] >= 1


@pytest.mark.asyncio
async def test_pipeline_health_partial_cold_start(db):
    """When some cycles found predictions and some did not, cold_start_detected=False."""
    tracker = BehavioralAccuracyTracker(db)

    # First cycle: no predictions (empty DB)
    await tracker.run_inference_cycle()

    # Second cycle: prediction present
    _insert_surfaced_prediction(db)
    await tracker.run_inference_cycle()

    health = tracker.get_pipeline_health()

    assert health["total_cycles"] == 2
    assert health["cycles_with_no_predictions"] == 1
    # Not ALL cycles were empty, so cold_start_detected must be False
    assert health["cold_start_detected"] is False


# ---------------------------------------------------------------------------
# Cycle stats updated after run_inference_cycle()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cycle_stats_updated_after_empty_cycle(db):
    """After an empty cycle, last_cycle_stats includes predictions_queried=0."""
    tracker = BehavioralAccuracyTracker(db)

    await tracker.run_inference_cycle()

    health = tracker.get_pipeline_health()
    assert health["last_cycle_stats"] is not None
    assert health["last_cycle_stats"]["predictions_queried"] == 0
    assert health["last_cycle_stats"]["marked_accurate"] == 0
    assert health["last_cycle_stats"]["marked_inaccurate"] == 0


@pytest.mark.asyncio
async def test_cycle_stats_include_predictions_queried_count(db):
    """predictions_queried in last_cycle_stats equals the number of surfaced predictions found."""
    tracker = BehavioralAccuracyTracker(db)

    # Insert 3 surfaced predictions
    for _ in range(3):
        _insert_surfaced_prediction(db)

    await tracker.run_inference_cycle()

    health = tracker.get_pipeline_health()
    assert health["last_cycle_stats"]["predictions_queried"] == 3


@pytest.mark.asyncio
async def test_last_cycle_timestamp_set_after_run(db):
    """last_cycle_timestamp is populated as an ISO-8601 UTC string after a cycle runs."""
    tracker = BehavioralAccuracyTracker(db)

    before = datetime.now(timezone.utc)
    await tracker.run_inference_cycle()
    after = datetime.now(timezone.utc)

    health = tracker.get_pipeline_health()
    assert health["last_cycle_timestamp"] is not None

    ts = datetime.fromisoformat(health["last_cycle_timestamp"])
    # Timestamps should bracket the cycle execution
    assert before <= ts <= after


@pytest.mark.asyncio
async def test_total_cycles_increments_per_run(db):
    """total_cycles increments by exactly 1 for each completed run_inference_cycle()."""
    tracker = BehavioralAccuracyTracker(db)

    for expected in range(1, 4):
        await tracker.run_inference_cycle()
        assert tracker.get_pipeline_health()["total_cycles"] == expected


# ---------------------------------------------------------------------------
# INFO log fires on 10th empty cycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_info_log_fires_on_10th_empty_cycle(db, caplog):
    """INFO log about idle learning loop fires on the 10th cycle with no predictions."""
    tracker = BehavioralAccuracyTracker(db)

    # Advance internal counter to simulate 9 previous empty cycles without
    # running them (they would be identical and slow the test).
    tracker._total_cycles = 9
    tracker._cycles_with_no_predictions = 9

    with caplog.at_level(logging.INFO, logger="services.behavioral_accuracy_tracker.tracker"):
        await tracker.run_inference_cycle()

    assert any(
        "0 surfaced predictions" in record.message
        and "accuracy learning loop idle" in record.message
        for record in caplog.records
    ), "Expected cold-start INFO log on cycle 10, but it was not emitted"


@pytest.mark.asyncio
async def test_info_log_does_not_fire_on_first_9_empty_cycles(db, caplog):
    """INFO cold-start log must NOT fire on cycles 1-9 to avoid log spam."""
    tracker = BehavioralAccuracyTracker(db)

    with caplog.at_level(logging.INFO, logger="services.behavioral_accuracy_tracker.tracker"):
        for _ in range(9):
            await tracker.run_inference_cycle()

    cold_start_logs = [
        r for r in caplog.records
        if "0 surfaced predictions" in r.message and "accuracy learning loop idle" in r.message
    ]
    assert len(cold_start_logs) == 0, (
        f"Cold-start log should not fire in cycles 1-9, but got {len(cold_start_logs)} log(s)"
    )


@pytest.mark.asyncio
async def test_info_log_fires_on_20th_empty_cycle(db, caplog):
    """INFO log fires again on cycle 20 (every 10th cycle)."""
    tracker = BehavioralAccuracyTracker(db)

    # Simulate 19 previous empty cycles
    tracker._total_cycles = 19
    tracker._cycles_with_no_predictions = 19

    with caplog.at_level(logging.INFO, logger="services.behavioral_accuracy_tracker.tracker"):
        await tracker.run_inference_cycle()

    assert any(
        "0 surfaced predictions" in record.message
        and "accuracy learning loop idle" in record.message
        for record in caplog.records
    ), "Expected cold-start INFO log on cycle 20, but it was not emitted"


# ---------------------------------------------------------------------------
# predictions_table_count in health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predictions_table_count_reflects_resolved_predictions(db):
    """predictions_table_count includes resolved predictions (total row count)."""
    tracker = BehavioralAccuracyTracker(db)

    # Insert and manually resolve a prediction
    pred_id = _insert_surfaced_prediction(db)
    with db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE predictions SET resolved_at = ?, was_accurate = 1 WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), pred_id),
        )

    health = tracker.get_pipeline_health()
    assert health["predictions_table_count"] == 1
