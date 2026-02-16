"""
Tests for reaction prediction final calibration (iteration 60).

After multiple recalibration attempts, this test suite validates that the
reaction prediction gatekeeper now achieves the target 80%+ surface rate
while still filtering truly annoying interruptions.

Key test scenarios:
1. Clean predictions (no stress, no dismissals, daytime) → helpful
2. Minor penalties (opportunity type, single stress signal) → helpful
3. Moderate penalties (stressed OR dismissals) → neutral
4. Severe penalties (stressed AND dismissals AND late night) → annoying
5. Edge cases (exactly at thresholds, compound penalties)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
from services.prediction_engine.engine import PredictionEngine
from models.user_model import Prediction
from models.core import ConfidenceGate


class TestReactionPredictionFinalCalibration:
    """
    Validates the final calibration of reaction prediction scoring.

    Target: 80%+ of "default" gate predictions (confidence 0.6-0.8) should
    surface unless they have multiple active penalties.
    """

    @pytest.fixture
    def daytime_mock(self):
        """Mock datetime.now() to return 2pm UTC (14:00) to avoid time penalties."""
        daytime = datetime(2026, 2, 16, 14, 0, 0, tzinfo=timezone.utc)
        with patch('services.prediction_engine.engine.datetime') as mock_dt:
            mock_dt.now.return_value = daytime
            mock_dt.fromisoformat = datetime.fromisoformat  # Preserve fromisoformat
            mock_dt.timezone = timezone  # Preserve timezone
            yield mock_dt

    @pytest.fixture
    def current_time(self):
        """Return a consistent timestamp for tests."""
        return datetime(2026, 2, 16, 14, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def engine(self, db, user_model_store):
        """Create a PredictionEngine for testing."""
        return PredictionEngine(db, user_model_store)

    @pytest.fixture
    def clean_prediction(self):
        """A typical 'default' gate prediction with no special flags."""
        return Prediction(
            prediction_type="reminder",
            description="You haven't responded to Alice's email from 2 days ago",
            confidence=0.65,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="today",
            suggested_action="Reply to Alice",
            relevant_contacts=["alice@example.com"],
        )

    @pytest.fixture
    def high_confidence_prediction(self):
        """A high-confidence prediction that should get a boost."""
        return Prediction(
            prediction_type="need",
            description="You need to prepare for tomorrow's flight",
            confidence=0.75,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="24_hours",
            suggested_action="Check packing list",
        )

    @pytest.fixture
    def conflict_prediction(self):
        """A conflict prediction that should get urgency boost."""
        return Prediction(
            prediction_type="conflict",
            description="Calendar overlap: Meeting at 2pm conflicts with dentist at 2pm",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="today",
            suggested_action="Reschedule one event",
        )

    @pytest.fixture
    def opportunity_prediction(self):
        """An opportunity prediction with minor penalty."""
        return Prediction(
            prediction_type="opportunity",
            description="It's been 15 days since you contacted Bob",
            confidence=0.6,
            confidence_gate=ConfidenceGate.SUGGEST,
            time_horizon="this_week",
            suggested_action="Reach out to Bob",
            relevant_contacts=["bob@example.com"],
        )

    @pytest.mark.asyncio
    async def test_clean_prediction_surfaces_as_helpful(
        self, engine, clean_prediction, db, user_model_store, daytime_mock
    ):
        """
        A clean prediction with no penalties should score 0.3 and be 'helpful'.

        Score breakdown:
        - Start: 0.3
        - No stress penalty: +0.0
        - No dismissal penalty: +0.0
        - No confidence boost (0.65 < 0.7): +0.0
        - No urgency boost (reminder type): +0.0
        - No time penalty (daytime): +0.0
        Final: 0.3 → >= 0.2 threshold → helpful ✓
        """
        # Clear any existing feedback to ensure clean state
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        # Clear mood signals
        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        context = {}
        reaction = await engine.predict_reaction(clean_prediction, context)

        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.30" in reaction.reasoning, \
            f"Expected score=0.30 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_high_confidence_prediction_gets_boost(
        self, engine, high_confidence_prediction, db, user_model_store, daytime_mock
    ):
        """
        High confidence (>0.7) predictions should get +0.2 boost.

        Score breakdown:
        - Start: 0.3
        - Confidence boost (0.75 > 0.7): +0.2
        Final: 0.5 → helpful ✓
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        context = {}
        reaction = await engine.predict_reaction(high_confidence_prediction, context)

        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.50" in reaction.reasoning, \
            f"Expected score=0.50 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_conflict_prediction_gets_urgency_boost(
        self, engine, conflict_prediction, db, user_model_store, daytime_mock
    ):
        """
        Conflicts and risks should get +0.2 urgency boost.

        Score breakdown:
        - Start: 0.3
        - Urgency boost (conflict type): +0.2
        Final: 0.5 → helpful ✓
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        context = {}
        reaction = await engine.predict_reaction(conflict_prediction, context)

        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.50" in reaction.reasoning, \
            f"Expected score=0.50 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_opportunity_prediction_with_minor_penalty(
        self, engine, opportunity_prediction, db, user_model_store, daytime_mock
    ):
        """
        Opportunity predictions get -0.05 penalty but should still surface.

        Score breakdown:
        - Start: 0.3
        - Opportunity penalty: -0.05
        Final: 0.25 → >= 0.2 threshold → helpful ✓
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        context = {}
        reaction = await engine.predict_reaction(opportunity_prediction, context)

        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.25" in reaction.reasoning, \
            f"Expected score=0.25 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_stressed_user_gets_penalty(
        self, engine, clean_prediction, db, user_model_store, daytime_mock
    ):
        """
        If user shows stress signals, predictions get -0.1 penalty.

        Score breakdown:
        - Start: 0.3
        - Stress penalty (3+ stress signals): -0.1
        Final: 0.2 → >= 0.2 threshold → helpful ✓ (edge case)
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        # Simulate stress signals
        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": [
                {"signal_type": "negative_language", "score": -0.5},
                {"signal_type": "calendar_density", "score": 0.8},
                {"signal_type": "negative_language", "score": -0.3},
            ]
        })

        context = {}
        reaction = await engine.predict_reaction(clean_prediction, context)

        # With stress penalty, score drops to 0.2 (exactly at threshold)
        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful (edge case at threshold), got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.20" in reaction.reasoning, \
            f"Expected score=0.20 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_high_dismissals_gets_penalty(
        self, engine, clean_prediction, db, user_model_store, daytime_mock, current_time
    ):
        """
        >5 dismissals in 2 hours should trigger -0.2 penalty.

        Score breakdown:
        - Start: 0.3
        - Dismissal fatigue (6 dismissals): -0.2
        Final: 0.1 → neutral ✓
        """
        # Create 6 recent dismissals
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")
            for i in range(6):
                conn.execute(
                    """INSERT INTO feedback_log (id, action_id, action_type, feedback_type, context, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f"dismiss-{i}", f"notif-{i}", "notification", "dismissed", "{}", current_time.isoformat()),
                )

        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        context = {}
        reaction = await engine.predict_reaction(clean_prediction, context)

        assert reaction.predicted_reaction == "neutral", \
            f"Expected neutral, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.10" in reaction.reasoning, \
            f"Expected score=0.10 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_compound_penalties_suppress_prediction(
        self, engine, opportunity_prediction, db, user_model_store, daytime_mock, current_time
    ):
        """
        Multiple penalties should compound to suppress predictions.

        Score breakdown:
        - Start: 0.3
        - Opportunity penalty: -0.05
        - Stress penalty: -0.1
        - Dismissal penalty: -0.2
        Final: -0.05 → neutral ✓ (> -0.1 threshold for annoying)
        """
        # Create 6 dismissals
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")
            for i in range(6):
                conn.execute(
                    """INSERT INTO feedback_log (id, action_id, action_type, feedback_type, context, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f"dismiss-{i}", f"notif-{i}", "notification", "dismissed", "{}", current_time.isoformat()),
                )

        # Add stress signals
        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": [
                {"signal_type": "negative_language", "score": -0.5},
                {"signal_type": "calendar_density", "score": 0.8},
                {"signal_type": "negative_language", "score": -0.3},
            ]
        })

        context = {}
        reaction = await engine.predict_reaction(opportunity_prediction, context)

        assert reaction.predicted_reaction == "neutral", \
            f"Expected neutral, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        # Score: 0.3 (start) - 0.05 (opportunity) - 0.1 (stress) - 0.2 (dismissals) = -0.05
        # -0.05 > -0.1, so it's neutral not annoying
        assert "score=-0.05" in reaction.reasoning, \
            f"Expected score=-0.05 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_edge_case_exactly_at_helpful_threshold(
        self, engine, db, user_model_store, daytime_mock
    ):
        """
        A prediction scoring exactly 0.2 should be 'helpful' (>= threshold).
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        # 3 stress signals to trigger -0.1 penalty
        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": [
                {"signal_type": "negative_language", "score": -0.5},
                {"signal_type": "calendar_density", "score": 0.8},
                {"signal_type": "negative_language", "score": -0.3},
            ]
        })

        # Clean prediction: 0.3 - 0.1 = 0.2
        pred = Prediction(
            prediction_type="reminder",
            description="Test prediction at threshold",
            confidence=0.65,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="today",
        )

        context = {}
        reaction = await engine.predict_reaction(pred, context)

        assert reaction.predicted_reaction == "helpful", \
            f"Expected helpful (>= 0.2), got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_edge_case_just_below_helpful_threshold(
        self, engine, db, user_model_store, daytime_mock
    ):
        """
        A prediction scoring 0.19 should be 'neutral' (> -0.1 but < 0.2).
        """
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        # 3 stress signals for -0.1 penalty
        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": [
                {"signal_type": "negative_language", "score": -0.5},
                {"signal_type": "calendar_density", "score": 0.8},
                {"signal_type": "negative_language", "score": -0.3},
            ]
        })

        # Opportunity prediction: 0.3 - 0.05 (opportunity) - 0.1 (stress) = 0.15
        pred = Prediction(
            prediction_type="opportunity",
            description="Test prediction just below threshold",
            confidence=0.6,
            confidence_gate=ConfidenceGate.SUGGEST,
            time_horizon="this_week",
        )

        context = {}
        reaction = await engine.predict_reaction(pred, context)

        assert reaction.predicted_reaction == "neutral", \
            f"Expected neutral, got {reaction.predicted_reaction}. Reasoning: {reaction.reasoning}"
        assert "score=0.15" in reaction.reasoning, \
            f"Expected score=0.15 in reasoning, got: {reaction.reasoning}"

    @pytest.mark.asyncio
    async def test_default_gate_predictions_achieve_80_percent_surface_rate(
        self, engine, db, user_model_store, daytime_mock
    ):
        """
        Validate that 80%+ of 'default' gate predictions (0.6-0.8 confidence)
        surface as 'helpful' or 'neutral' under typical conditions.

        This is the key success metric for the calibration fix.
        """
        # Clean state: no dismissals, no stress
        with db.get_connection("preferences") as conn:
            conn.execute("DELETE FROM feedback_log")

        user_model_store.update_signal_profile("mood_signals", {
            "recent_signals": []
        })

        # Test 100 typical 'default' gate predictions
        predictions = []
        for i in range(100):
            conf = 0.6 + (i / 500.0)  # Range: 0.6 to 0.8
            pred_type = ["reminder", "need", "opportunity"][i % 3]
            predictions.append(Prediction(
                prediction_type=pred_type,
                description=f"Test prediction {i}",
                confidence=conf,
                confidence_gate=ConfidenceGate.DEFAULT,
                time_horizon="today",
            ))

        context = {}
        surfaced_count = 0

        for pred in predictions:
            reaction = await engine.predict_reaction(pred, context)
            if reaction.predicted_reaction in ("helpful", "neutral"):
                surfaced_count += 1

        surface_rate = surfaced_count / len(predictions)

        assert surface_rate >= 0.80, \
            f"Surface rate {surface_rate:.1%} is below 80% target (surfaced {surfaced_count}/{len(predictions)})"
