"""
Tests for prediction confidence edge cases.

Verifies that _gate_from_confidence handles NaN, inf, and negative values
safely (mapping to OBSERVE), and that confidence clamping after multiplier
application keeps values within [0.0, 1.0].

These guards prevent a dangerous latent bug where corrupted confidence
scores (e.g. from division-by-zero in accuracy tracking) could silently
escalate predictions to AUTONOMOUS gate.
"""

import math
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import PredictionEngine


class TestGateFromConfidenceEdgeCases:
    """Test _gate_from_confidence with invalid and boundary inputs."""

    def test_nan_returns_observe(self):
        """NaN confidence must map to OBSERVE, never AUTONOMOUS."""
        result = PredictionEngine._gate_from_confidence(float("nan"))
        assert result == ConfidenceGate.OBSERVE

    def test_positive_inf_returns_observe(self):
        """Positive infinity confidence must map to OBSERVE, never AUTONOMOUS."""
        result = PredictionEngine._gate_from_confidence(float("inf"))
        assert result == ConfidenceGate.OBSERVE

    def test_negative_inf_returns_observe(self):
        """Negative infinity confidence must map to OBSERVE."""
        result = PredictionEngine._gate_from_confidence(float("-inf"))
        assert result == ConfidenceGate.OBSERVE

    def test_negative_value_returns_observe(self):
        """Negative confidence values must map to OBSERVE."""
        result = PredictionEngine._gate_from_confidence(-0.5)
        assert result == ConfidenceGate.OBSERVE

    def test_negative_zero_returns_observe(self):
        """Negative zero should be treated as zero (OBSERVE)."""
        result = PredictionEngine._gate_from_confidence(-0.0)
        assert result == ConfidenceGate.OBSERVE

    def test_value_above_one_returns_autonomous(self):
        """Values > 1.0 are valid high confidence — should return AUTONOMOUS."""
        result = PredictionEngine._gate_from_confidence(1.5)
        assert result == ConfidenceGate.AUTONOMOUS

    def test_exactly_zero_returns_observe(self):
        """Zero confidence should return OBSERVE."""
        result = PredictionEngine._gate_from_confidence(0.0)
        assert result == ConfidenceGate.OBSERVE

    def test_boundary_0_3_returns_suggest(self):
        """Exactly 0.3 should return SUGGEST (not OBSERVE)."""
        result = PredictionEngine._gate_from_confidence(0.3)
        assert result == ConfidenceGate.SUGGEST

    def test_boundary_0_6_returns_default(self):
        """Exactly 0.6 should return DEFAULT (not SUGGEST)."""
        result = PredictionEngine._gate_from_confidence(0.6)
        assert result == ConfidenceGate.DEFAULT

    def test_boundary_0_8_returns_autonomous(self):
        """Exactly 0.8 should return AUTONOMOUS (not DEFAULT)."""
        result = PredictionEngine._gate_from_confidence(0.8)
        assert result == ConfidenceGate.AUTONOMOUS


@pytest.mark.asyncio
async def test_confidence_clamped_after_multiplier(db, user_model_store):
    """Confidence is clamped to [0, 1] after accuracy multiplier application.

    When the accuracy multiplier produces a value > 1.0 or < 0.0,
    the engine should clamp the result rather than allowing unbounded values.
    """
    engine = PredictionEngine(db, user_model_store)

    # Create a prediction with high confidence
    pred = Prediction(
        id=f"pred-clamp-{uuid.uuid4().hex[:8]}",
        prediction_type="reminder",
        description="Test clamping",
        confidence=0.9,
        confidence_gate=ConfidenceGate.AUTONOMOUS,
        time_horizon="1d",
        suggested_action="Test action",
        supporting_signals={},
        was_surfaced=False,
        user_response=None,
        was_accurate=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        resolved_at=None,
        filter_reason=None,
    )

    # Mock to return our prediction
    engine._check_follow_up_needs = AsyncMock(return_value=[pred])
    engine._check_calendar_conflicts = AsyncMock(return_value=[])
    engine._check_routine_deviations = AsyncMock(return_value=[])
    engine._check_relationship_maintenance = AsyncMock(return_value=[])
    engine._check_preparation_needs = AsyncMock(return_value=[])
    engine._check_spending_patterns = AsyncMock(return_value=[])

    # Multiplier of 2.0 would push 0.9 to 1.8 without clamping
    engine._get_accuracy_multiplier = MagicMock(return_value=2.0)

    # Approve via reaction gating
    engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test",
        )
    )

    # Force new events to trigger prediction pipeline
    engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"evt-clamp-{uuid.uuid4().hex[:8]}",
                "email.received",
                "test",
                datetime.now(timezone.utc).isoformat(),
                "medium",
                "{}",
            ),
        )

    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await engine.generate_predictions(context)

    assert len(surfaced) == 1
    # After clamping, 0.9 * 2.0 = 1.8 -> clamped to 1.0
    assert surfaced[0].confidence == 1.0, (
        f"Confidence should be clamped to 1.0, got {surfaced[0].confidence}"
    )


@pytest.mark.asyncio
async def test_confidence_clamped_negative_multiplier(db, user_model_store):
    """Confidence is clamped to 0.0 when multiplier produces a negative value."""
    engine = PredictionEngine(db, user_model_store)

    pred = Prediction(
        id=f"pred-neg-{uuid.uuid4().hex[:8]}",
        prediction_type="reminder",
        description="Test negative clamping",
        confidence=0.5,
        confidence_gate=ConfidenceGate.SUGGEST,
        time_horizon="1d",
        suggested_action="Test action",
        supporting_signals={},
        was_surfaced=False,
        user_response=None,
        was_accurate=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        resolved_at=None,
        filter_reason=None,
    )

    engine._check_follow_up_needs = AsyncMock(return_value=[pred])
    engine._check_calendar_conflicts = AsyncMock(return_value=[])
    engine._check_routine_deviations = AsyncMock(return_value=[])
    engine._check_relationship_maintenance = AsyncMock(return_value=[])
    engine._check_preparation_needs = AsyncMock(return_value=[])
    engine._check_spending_patterns = AsyncMock(return_value=[])

    # Negative multiplier would push 0.5 to -0.5 without clamping
    engine._get_accuracy_multiplier = MagicMock(return_value=-1.0)

    engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test",
        )
    )

    engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"evt-neg-{uuid.uuid4().hex[:8]}",
                "email.received",
                "test",
                datetime.now(timezone.utc).isoformat(),
                "medium",
                "{}",
            ),
        )

    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await engine.generate_predictions(context)

    # Clamped to 0.0, which is < 0.3 OBSERVE threshold, so it gets filtered
    # by the confidence gate and not surfaced
    assert len(surfaced) == 0, (
        f"Prediction with negative-clamped confidence should be filtered, got {len(surfaced)}"
    )
