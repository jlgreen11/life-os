"""
Tests for post-store persistence verification in PredictionEngine.

Verifies that the engine detects when predictions appear to store successfully
but are not actually persisted (e.g. WAL checkpoint issues, DB recovery).

The verification step runs after the storage loop in generate_predictions()
and sets _persistence_failure_detected = True if stored_count > 0 but the
predictions table has 0 unresolved rows.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

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


class TestPersistenceVerification:
    """Tests for post-store verification that detects silent persistence failures."""

    def test_persistence_failure_flag_initially_false(self, db, user_model_store):
        """Fresh engine should have _persistence_failure_detected = False."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._persistence_failure_detected is False

    @pytest.mark.asyncio
    async def test_stored_count_matches_db_on_success(self, db, event_store, user_model_store):
        """When predictions store successfully, stored_count matches actual DB rows."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        preds = [_make_prediction(), _make_prediction(), _make_prediction()]
        await _run_engine_with_fake_predictions(engine, preds)

        # Verify predictions actually exist in the database
        with db.get_connection("user_model") as conn:
            actual_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert actual_count == 3, f"Expected 3 predictions in DB, got {actual_count}"

        # stored_count should be in diagnostics
        assert engine._last_run_diagnostics["stored_count"] == 3

        # Persistence failure should NOT be detected
        assert engine._persistence_failure_detected is False

    @pytest.mark.asyncio
    async def test_detects_persistence_failure_when_rows_vanish(self, db, event_store, user_model_store):
        """Detects persistence failure when predictions disappear after storage.

        Simulates the anomaly where store_prediction() succeeds but data is
        lost (e.g. due to WAL checkpoint failure or DB recovery).  We achieve
        this by wrapping store_prediction to delete rows immediately after
        the INSERT, so the post-store verification finds 0 unresolved rows
        despite stored_count > 0.
        """
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        original_store = user_model_store.store_prediction

        def store_then_delete(prediction):
            """Store the prediction, then delete all unresolved rows to simulate data loss."""
            original_store(prediction)
            with db.get_connection("user_model") as conn:
                conn.execute("DELETE FROM predictions WHERE resolved_at IS NULL")

        with patch.object(user_model_store, "store_prediction", side_effect=store_then_delete):
            pred = _make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)
            await _run_engine_with_fake_predictions(engine, [pred])

        # stored_count=1 (store didn't raise) but verification finds 0 unresolved rows
        assert engine._persistence_failure_detected is True
        assert engine._last_run_diagnostics["stored_count"] == 1

    @pytest.mark.asyncio
    async def test_persistence_failure_exposed_in_runtime_diagnostics(self, db, event_store, user_model_store):
        """_persistence_failure_detected should appear in get_runtime_diagnostics()."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        diag = engine.get_runtime_diagnostics()
        assert "persistence_failure_detected" in diag
        assert diag["persistence_failure_detected"] is False

        # Trigger the flag manually to verify exposure
        engine._persistence_failure_detected = True
        diag = engine.get_runtime_diagnostics()
        assert diag["persistence_failure_detected"] is True

    @pytest.mark.asyncio
    async def test_persistence_failure_exposed_in_full_diagnostics(self, db, event_store, user_model_store):
        """_persistence_failure_detected should appear in get_diagnostics() overall section."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        diag = await engine.get_diagnostics()
        assert "persistence_failure_detected" in diag["overall"]
        assert diag["overall"]["persistence_failure_detected"] is False

        engine._persistence_failure_detected = True
        diag = await engine.get_diagnostics()
        assert diag["overall"]["persistence_failure_detected"] is True

    @pytest.mark.asyncio
    async def test_no_false_positive_when_unresolved_predictions_exist(self, db, event_store, user_model_store):
        """Verification should NOT flag when unresolved predictions exist in DB."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        # Store a high-confidence prediction that gets surfaced (not filtered)
        # so resolved_at stays NULL.
        pred = _make_prediction(confidence=0.7, confidence_gate=ConfidenceGate.DEFAULT)
        await _run_engine_with_fake_predictions(engine, [pred])

        assert engine._persistence_failure_detected is False

        # Verify the prediction is indeed unresolved in DB
        with db.get_connection("user_model") as conn:
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NULL"
            ).fetchone()[0]
        assert unresolved >= 1
