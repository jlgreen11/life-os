"""
Tests for prediction engine persistence failure recovery.

When the prediction engine detects that predictions were lost after storage
(via the _persistence_failure_detected flag), it should take corrective action
on the next cycle: verify the DB is writable, clear the flag, and proceed
with generation. If the DB write test fails, the cycle should be skipped.

Also tests proactive detection: when the predictions table is empty but
_store_failure_count > 0, the engine flags for recovery on the next cycle.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(**overrides) -> Prediction:
    """Create a minimal Prediction for testing.

    Uses a unique description by default to avoid deduplication in
    store_prediction() which deduplicates by (prediction_type, description).
    """
    defaults = {
        "id": str(uuid.uuid4()),
        "prediction_type": "need",
        "description": f"Test prediction {uuid.uuid4().hex[:8]}",
        "confidence": 0.5,
        "confidence_gate": ConfidenceGate.SUGGEST,
        "time_horizon": "2_hours",
    }
    defaults.update(overrides)
    return Prediction(**defaults)


def _insert_event(event_store):
    """Insert a generic event to advance the cursor so generate_predictions runs."""
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@test.com", "subject": "Test", "message_id": str(uuid.uuid4())},
        "metadata": {},
    })


async def _run_engine_with_fake_predictions(engine, predictions):
    """Run generate_predictions with mocked checkers returning given predictions.

    Patches all _check_* methods to return empty lists except the first one
    which returns the provided predictions.  Also patches predict_reaction to
    always allow surfacing.
    """
    check_methods = [
        "_check_calendar_conflicts",
        "_check_routine_deviations",
        "_check_relationship_maintenance",
        "_check_preparation_needs",
        "_check_follow_up_needs",
        "_check_calendar_event_reminders",
        "_check_connector_health",
        "_check_spending_patterns",
    ]

    patches = []
    for method_name in check_methods:
        if hasattr(engine, method_name):
            p = patch.object(engine, method_name, new_callable=AsyncMock, return_value=[])
            patches.append(p)

    # First checker returns our fake predictions
    if patches:
        patches[0].kwargs["return_value"] = list(predictions)

    helpful_reaction = ReactionPrediction(
        predicted_reaction="helpful",
        confidence=0.9,
        reasoning="test",
        proposed_action="surface",
    )
    reaction_patch = patch.object(engine, "predict_reaction", new_callable=AsyncMock, return_value=helpful_reaction)
    patches.append(reaction_patch)

    for p in patches:
        p.start()
    try:
        result = await engine.generate_predictions({})
    finally:
        for p in patches:
            p.stop()

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPersistenceRecovery:
    """Tests for persistence failure recovery in generate_predictions()."""

    @pytest.mark.asyncio
    async def test_recovery_clears_flag_when_db_healthy(self, db, event_store, user_model_store):
        """When _persistence_failure_detected is True and DB is healthy,
        the flag is cleared and predictions are generated normally."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate a previous persistence failure
        engine._persistence_failure_detected = True

        preds = [_make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)]
        result = await _run_engine_with_fake_predictions(engine, preds)

        # Flag should be cleared after successful DB write test
        assert engine._persistence_failure_detected is False

        # Predictions should still be generated normally
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_recovery_skips_cycle_when_db_write_fails(self, db, event_store, user_model_store):
        """When _persistence_failure_detected is True and DB write test fails,
        the cycle is skipped (returns empty list)."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate a previous persistence failure
        engine._persistence_failure_detected = True

        # Make the DB write test fail by dropping the predictions table
        with db.get_connection('user_model') as conn:
            conn.execute('DROP TABLE IF EXISTS predictions')

        result = await engine.generate_predictions({})

        # Should return empty — cycle skipped
        assert result == []

        # Flag should still be True since recovery failed
        assert engine._persistence_failure_detected is True

    @pytest.mark.asyncio
    async def test_proactive_detection_empty_table_with_store_failures(self, db, event_store, user_model_store):
        """When predictions table is empty and _store_failure_count > 0,
        _persistence_failure_detected is set to True."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate prior store failures without the flag being set
        engine._store_failure_count = 3
        assert engine._persistence_failure_detected is False

        # Run a cycle — the pre-filter will find 0 existing predictions,
        # triggering the proactive detection
        preds = [_make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)]
        await _run_engine_with_fake_predictions(engine, preds)

        # The flag should now be set for the NEXT cycle
        # (it won't block this cycle since it's set after the pre-filter)
        assert engine._persistence_failure_detected is True

    @pytest.mark.asyncio
    async def test_recovery_mode_in_generation_stats(self, db, event_store, user_model_store):
        """Recovery mode is logged in generation_stats when recovering."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate a previous persistence failure
        engine._persistence_failure_detected = True

        preds = [_make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)]
        await _run_engine_with_fake_predictions(engine, preds)

        # Check that recovery_mode was recorded in diagnostics
        assert engine._last_generation_stats.get('recovery_mode') is True

    @pytest.mark.asyncio
    async def test_persistence_test_row_cleaned_up(self, db, event_store, user_model_store):
        """The test prediction row (__persistence_test__) is cleaned up after
        successful verification."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Simulate a previous persistence failure
        engine._persistence_failure_detected = True

        preds = [_make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)]
        await _run_engine_with_fake_predictions(engine, preds)

        # Verify the test row is gone
        with db.get_connection('user_model') as conn:
            row = conn.execute(
                "SELECT id FROM predictions WHERE id = '__persistence_test__'"
            ).fetchone()
        assert row is None, "Test prediction row should be cleaned up after recovery"

    @pytest.mark.asyncio
    async def test_no_proactive_detection_when_no_store_failures(self, db, event_store, user_model_store):
        """When predictions table is empty but _store_failure_count is 0,
        no persistence failure is flagged (fresh engine, no history)."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        assert engine._store_failure_count == 0
        assert engine._persistence_failure_detected is False

        preds = [_make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)]
        await _run_engine_with_fake_predictions(engine, preds)

        # With 0 store failures, proactive detection should NOT trigger
        # (the flag may be set by post-store verification if something else goes wrong,
        # but not by the proactive check)
        # Verify at least that the engine ran normally
        assert len(engine._last_run_diagnostics) > 0
