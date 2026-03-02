"""Tests for prediction notification priority logic.

Verifies that _prediction_priority() assigns the correct notification priority
based on prediction_type and confidence, ensuring that high-confidence
predictions reach users in minimal notification mode.
"""

import pytest

from main import _prediction_priority
from models.core import ConfidenceGate
from models.user_model import Prediction


def _make_prediction(prediction_type: str, confidence: float) -> Prediction:
    """Create a minimal Prediction object for testing priority logic."""
    # Map confidence to the appropriate gate
    if confidence >= 0.8:
        gate = ConfidenceGate.AUTONOMOUS
    elif confidence >= 0.6:
        gate = ConfidenceGate.DEFAULT
    elif confidence >= 0.3:
        gate = ConfidenceGate.SUGGEST
    else:
        gate = ConfidenceGate.OBSERVE

    return Prediction(
        prediction_type=prediction_type,
        description=f"Test {prediction_type} prediction",
        confidence=confidence,
        confidence_gate=gate,
        time_horizon="24_hours",
    )


class TestPredictionPriority:
    """Tests for the _prediction_priority helper function."""

    # --- Conflict predictions always get 'high' ---

    def test_conflict_high_confidence_is_high(self):
        """Conflict predictions with high confidence get priority='high'."""
        pred = _make_prediction("conflict", 0.85)
        assert _prediction_priority(pred) == "high"

    def test_conflict_low_confidence_is_high(self):
        """Conflict predictions with low confidence still get priority='high'."""
        pred = _make_prediction("conflict", 0.25)
        assert _prediction_priority(pred) == "high"

    def test_conflict_medium_confidence_is_high(self):
        """Conflict predictions with medium confidence get priority='high'."""
        pred = _make_prediction("conflict", 0.5)
        assert _prediction_priority(pred) == "high"

    # --- Risk predictions always get 'high' ---

    def test_risk_high_confidence_is_high(self):
        """Risk predictions with high confidence get priority='high'."""
        pred = _make_prediction("risk", 0.9)
        assert _prediction_priority(pred) == "high"

    def test_risk_low_confidence_is_high(self):
        """Risk predictions with low confidence still get priority='high'."""
        pred = _make_prediction("risk", 0.2)
        assert _prediction_priority(pred) == "high"

    def test_risk_medium_confidence_is_high(self):
        """Risk predictions with medium confidence get priority='high'."""
        pred = _make_prediction("risk", 0.45)
        assert _prediction_priority(pred) == "high"

    # --- Reminder predictions: confidence >= 0.6 -> 'high', else 'normal' ---

    def test_reminder_high_confidence_is_high(self):
        """Reminder predictions in DEFAULT gate (>=0.6) get priority='high'."""
        pred = _make_prediction("reminder", 0.7)
        assert _prediction_priority(pred) == "high"

    def test_reminder_at_threshold_is_high(self):
        """Reminder predictions exactly at 0.6 boundary get priority='high'."""
        pred = _make_prediction("reminder", 0.6)
        assert _prediction_priority(pred) == "high"

    def test_reminder_below_threshold_is_normal(self):
        """Reminder predictions below 0.6 (SUGGEST gate) get priority='normal'."""
        pred = _make_prediction("reminder", 0.5)
        assert _prediction_priority(pred) == "normal"

    def test_reminder_low_confidence_is_normal(self):
        """Reminder predictions with low confidence get priority='normal'."""
        pred = _make_prediction("reminder", 0.35)
        assert _prediction_priority(pred) == "normal"

    # --- Opportunity predictions: same confidence logic ---

    def test_opportunity_high_confidence_is_high(self):
        """Opportunity predictions in DEFAULT gate (>=0.6) get priority='high'."""
        pred = _make_prediction("opportunity", 0.75)
        assert _prediction_priority(pred) == "high"

    def test_opportunity_suggest_gate_is_normal(self):
        """Opportunity predictions in SUGGEST gate (0.3-0.59) get priority='normal'."""
        pred = _make_prediction("opportunity", 0.45)
        assert _prediction_priority(pred) == "normal"

    def test_opportunity_at_threshold_is_high(self):
        """Opportunity predictions exactly at 0.6 get priority='high'."""
        pred = _make_prediction("opportunity", 0.6)
        assert _prediction_priority(pred) == "high"

    def test_opportunity_just_below_threshold_is_normal(self):
        """Opportunity predictions at 0.59 get priority='normal'."""
        pred = _make_prediction("opportunity", 0.59)
        assert _prediction_priority(pred) == "normal"

    # --- Need and routine_deviation follow the same pattern ---

    def test_need_high_confidence_is_high(self):
        """Need predictions with confidence >= 0.6 get priority='high'."""
        pred = _make_prediction("need", 0.65)
        assert _prediction_priority(pred) == "high"

    def test_need_low_confidence_is_normal(self):
        """Need predictions with confidence < 0.6 get priority='normal'."""
        pred = _make_prediction("need", 0.4)
        assert _prediction_priority(pred) == "normal"

    def test_routine_deviation_high_confidence_is_high(self):
        """Routine deviation predictions with confidence >= 0.6 get priority='high'."""
        pred = _make_prediction("routine_deviation", 0.8)
        assert _prediction_priority(pred) == "high"

    def test_routine_deviation_low_confidence_is_normal(self):
        """Routine deviation predictions with confidence < 0.6 get priority='normal'."""
        pred = _make_prediction("routine_deviation", 0.3)
        assert _prediction_priority(pred) == "normal"

    # --- Autonomous gate (>= 0.8) also gets 'high' ---

    def test_autonomous_gate_reminder_is_high(self):
        """Predictions in the AUTONOMOUS gate always get priority='high'."""
        pred = _make_prediction("reminder", 0.9)
        assert _prediction_priority(pred) == "high"
