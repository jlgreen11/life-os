"""
Tests for prediction storage failure tracking in PredictionEngine.

Verifies that when store_prediction() raises exceptions, the engine:
- Increments failure counters (per-run and lifetime)
- Captures error details in the _last_store_errors ring buffer
- Caps the error buffer at 10 entries
- Exposes failure metrics via get_runtime_diagnostics()
- Resets per-run counters between generate_predictions() calls
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction(**overrides) -> Prediction:
    """Create a minimal Prediction for testing.

    Each call generates a unique description by default to prevent intra-batch
    deduplication from silently discarding predictions when multiple identical
    predictions are created for testing storage failure scenarios.
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


async def _run_engine_with_fake_predictions(engine, user_model_store, predictions, store_side_effect=None):
    """Run generate_predictions with mocked checkers that return given predictions.

    Patches all _check_* methods to return empty lists except the first one
    which returns the provided predictions. Also patches predict_reaction to
    always allow surfacing, and optionally patches store_prediction to raise.
    """
    # All _check_* methods return empty by default
    check_methods = [
        "_check_calendar_conflicts",
        "_check_routine_deviations",
        "_check_relationship_maintenance",
        "_check_preparation_needs",
        "_check_follow_up_needs",
        "_check_financial_patterns",
        "_check_health_wellness",
        "_check_opportunity_windows",
    ]

    patches = []
    for method_name in check_methods:
        if hasattr(engine, method_name):
            p = patch.object(engine, method_name, new_callable=AsyncMock, return_value=[])
            patches.append(p)

    # First checker returns our fake predictions
    if patches:
        patches[0].kwargs["return_value"] = list(predictions)

    # predict_reaction always returns "helpful"
    from models.user_model import ReactionPrediction
    helpful_reaction = ReactionPrediction(
        predicted_reaction="helpful",
        confidence=0.9,
        reasoning="test",
        proposed_action="surface",
    )
    reaction_patch = patch.object(engine, "predict_reaction", new_callable=AsyncMock, return_value=helpful_reaction)
    patches.append(reaction_patch)

    # Optionally break store_prediction
    if store_side_effect:
        store_patch = patch.object(user_model_store, "store_prediction", side_effect=store_side_effect)
        patches.append(store_patch)

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


class TestStoreFailureTracking:
    """Tests for prediction store failure counters and error capture."""

    def test_initial_counters_are_zero(self, db, user_model_store):
        """Fresh engine should have zero store failure counters."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        assert engine._store_failure_count == 0
        assert engine._last_store_errors == []

    @pytest.mark.asyncio
    async def test_store_failure_increments_counter(self, db, event_store, user_model_store):
        """A failing store_prediction() should increment the failure counter."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        preds = [_make_prediction(), _make_prediction()]
        await _run_engine_with_fake_predictions(
            engine, user_model_store, preds,
            store_side_effect=RuntimeError("DB locked"),
        )

        assert engine._store_failure_count == 2, "Should count both failed stores"

    @pytest.mark.asyncio
    async def test_store_failure_captures_error_details(self, db, event_store, user_model_store):
        """Error details should be captured with prediction_id, error_type, etc."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        pred = _make_prediction(prediction_type="conflict")
        await _run_engine_with_fake_predictions(
            engine, user_model_store, [pred],
            store_side_effect=ValueError("bad column"),
        )

        assert len(engine._last_store_errors) == 1
        err = engine._last_store_errors[0]
        assert "timestamp" in err
        assert err["prediction_id"] == pred.id
        assert err["prediction_type"] == "conflict"
        assert err["error_type"] == "ValueError"
        assert "bad column" in err["error_message"]

    @pytest.mark.asyncio
    async def test_store_errors_capped_at_10(self, db, event_store, user_model_store):
        """The _last_store_errors ring buffer should never exceed 10 entries."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # Pre-populate with 9 fake errors
        engine._last_store_errors = [
            {"timestamp": "t", "prediction_id": f"old-{i}", "prediction_type": "test",
             "error_message": "old", "error_type": "RuntimeError"}
            for i in range(9)
        ]

        _insert_event(event_store)
        preds = [_make_prediction() for _ in range(5)]
        await _run_engine_with_fake_predictions(
            engine, user_model_store, preds,
            store_side_effect=RuntimeError("fail"),
        )

        # 9 pre-existing + 5 new = 14, capped to 10
        assert len(engine._last_store_errors) == 10, (
            f"Error buffer should be capped at 10, got {len(engine._last_store_errors)}"
        )
        # Most recent errors should be the new ones
        assert engine._last_store_errors[-1]["error_message"] == "fail"

    @pytest.mark.asyncio
    async def test_diagnostics_include_store_failures(self, db, event_store, user_model_store):
        """get_runtime_diagnostics() should include store failure metrics."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        preds = [_make_prediction()]
        await _run_engine_with_fake_predictions(
            engine, user_model_store, preds,
            store_side_effect=RuntimeError("oops"),
        )

        diag = engine.get_runtime_diagnostics()

        # Runtime diagnostics has store_failures section
        assert "store_failures" in diag
        assert diag["store_failures"]["total"] == 1
        assert len(diag["store_failures"]["recent_errors"]) == 1

        # Run-level diagnostics also have the counters
        run_stats = diag["run_statistics"]
        assert run_stats["store_failures_this_run"] == 1
        assert run_stats["store_failures_total"] == 1

    @pytest.mark.asyncio
    async def test_per_run_counter_resets_between_calls(self, db, event_store, user_model_store):
        """The per-run store failure counter should reset at the start of each cycle."""
        engine = PredictionEngine(db=db, ums=user_model_store)

        # First run: trigger store failures
        _insert_event(event_store)
        preds = [_make_prediction(), _make_prediction()]
        await _run_engine_with_fake_predictions(
            engine, user_model_store, preds,
            store_side_effect=RuntimeError("fail"),
        )

        assert engine._last_run_diagnostics["store_failures_this_run"] == 2
        assert engine._store_failure_count == 2

        # Second run: store succeeds
        _insert_event(event_store)
        await _run_engine_with_fake_predictions(
            engine, user_model_store, [_make_prediction()],
            store_side_effect=None,
        )

        assert engine._last_run_diagnostics["store_failures_this_run"] == 0, (
            "Per-run counter should reset to 0 on successful run"
        )
        # Lifetime counter should NOT reset
        assert engine._store_failure_count == 2, "Lifetime counter should persist across runs"

    @pytest.mark.asyncio
    async def test_no_failures_when_store_succeeds(self, db, event_store, user_model_store):
        """When store_prediction succeeds, failure counters should stay at zero."""
        engine = PredictionEngine(db=db, ums=user_model_store)
        _insert_event(event_store)

        preds = [_make_prediction()]
        await _run_engine_with_fake_predictions(
            engine, user_model_store, preds,
            store_side_effect=None,
        )

        assert engine._store_failure_count == 0
        assert engine._last_store_errors == []
        assert engine._last_run_diagnostics["store_failures_this_run"] == 0
