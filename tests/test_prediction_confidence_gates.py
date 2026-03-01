"""
Comprehensive unit tests for prediction engine confidence gates and reaction scoring.

Tests the most safety-critical logic in the system — the confidence gates that
determine automation behavior (OBSERVE / SUGGEST / DEFAULT / AUTONOMOUS) and
the reaction scoring that decides whether to surface predictions to the user.

These thresholds have been calibrated across multiple PRs:
- stress penalty: -0.2 -> -0.1
- dismissal threshold: >3 -> >5
- dismissal penalty: -0.3 -> -0.2
- reaction thresholds: helpful >= 0.4/0.1 -> 0.2/-0.1
- start score: 0.5 -> 0.3
- opportunity penalty: -0.1 -> -0.05

A regression in any threshold could cause the system to either spam the user
or go completely silent.
"""

import json
import uuid
from datetime import datetime, time, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Group 1: _gate_from_confidence() — static method, no setup needed
# ---------------------------------------------------------------------------


class TestGateFromConfidence:
    """Test the confidence-to-gate mapping at all boundary conditions.

    The gate thresholds are:
        < 0.3  -> OBSERVE
        0.3-0.6 -> SUGGEST
        0.6-0.8 -> DEFAULT
        >= 0.8  -> AUTONOMOUS
    """

    def test_zero_confidence_returns_observe(self):
        """confidence=0.0 should map to OBSERVE (lowest gate)."""
        assert PredictionEngine._gate_from_confidence(0.0) == ConfidenceGate.OBSERVE

    def test_below_observe_boundary_returns_observe(self):
        """confidence=0.29 should still be OBSERVE (just below 0.3 boundary)."""
        assert PredictionEngine._gate_from_confidence(0.29) == ConfidenceGate.OBSERVE

    def test_at_suggest_boundary_returns_suggest(self):
        """confidence=0.3 should map to SUGGEST (exact boundary)."""
        assert PredictionEngine._gate_from_confidence(0.3) == ConfidenceGate.SUGGEST

    def test_mid_suggest_range_returns_suggest(self):
        """confidence=0.5 should map to SUGGEST (middle of range)."""
        assert PredictionEngine._gate_from_confidence(0.5) == ConfidenceGate.SUGGEST

    def test_just_below_default_boundary_returns_suggest(self):
        """confidence=0.59 should still be SUGGEST (just below 0.6 boundary)."""
        assert PredictionEngine._gate_from_confidence(0.59) == ConfidenceGate.SUGGEST

    def test_at_default_boundary_returns_default(self):
        """confidence=0.6 should map to DEFAULT (exact boundary)."""
        assert PredictionEngine._gate_from_confidence(0.6) == ConfidenceGate.DEFAULT

    def test_just_below_autonomous_boundary_returns_default(self):
        """confidence=0.79 should still be DEFAULT (just below 0.8 boundary)."""
        assert PredictionEngine._gate_from_confidence(0.79) == ConfidenceGate.DEFAULT

    def test_at_autonomous_boundary_returns_autonomous(self):
        """confidence=0.8 should map to AUTONOMOUS (exact boundary)."""
        assert PredictionEngine._gate_from_confidence(0.8) == ConfidenceGate.AUTONOMOUS

    def test_perfect_confidence_returns_autonomous(self):
        """confidence=1.0 should map to AUTONOMOUS."""
        assert PredictionEngine._gate_from_confidence(1.0) == ConfidenceGate.AUTONOMOUS

    def test_negative_confidence_returns_observe(self):
        """confidence=-0.1 (edge case) should still map to OBSERVE."""
        assert PredictionEngine._gate_from_confidence(-0.1) == ConfidenceGate.OBSERVE

    def test_over_one_confidence_returns_autonomous(self):
        """confidence=1.5 (edge case) should still map to AUTONOMOUS."""
        assert PredictionEngine._gate_from_confidence(1.5) == ConfidenceGate.AUTONOMOUS


# ---------------------------------------------------------------------------
# Group 2: predict_reaction() — scoring components
# ---------------------------------------------------------------------------


def _make_prediction(**kwargs) -> Prediction:
    """Helper to create a Prediction with sensible defaults for testing."""
    defaults = {
        "prediction_type": "reminder",
        "description": "Test prediction",
        "confidence": 0.5,
        "confidence_gate": ConfidenceGate.SUGGEST,
        "time_horizon": "24_hours",
    }
    defaults.update(kwargs)
    return Prediction(**defaults)


class TestPredictReaction:
    """Test the reaction scoring logic in predict_reaction().

    The scoring formula starts at 0.3 and adds/subtracts based on:
    - Stress signals (>2 negative in last 5): -0.1
    - Dismissal fatigue (>5 recent): -0.2
    - High confidence (>0.7): +0.2
    - Urgency (conflict/risk): +0.2
    - Opportunity type: -0.05
    - Quiet hours (non-urgent): -0.2

    Classification thresholds:
    - score >= 0.2 -> "helpful"
    - score > -0.1 -> "neutral"
    - score <= -0.1 -> "annoying"
    """

    @pytest.mark.asyncio
    async def test_baseline_score_no_modifiers(self, db, user_model_store):
        """With no penalties or bonuses, the base score of 0.3 yields 'helpful' (>= 0.2)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        assert reaction.predicted_reaction == "helpful"
        assert "score=0.30" in reaction.reasoning

    @pytest.mark.asyncio
    async def test_stress_penalty_applied(self, db, user_model_store):
        """When >2 of last 5 mood signals are stress-related, score decreases by 0.1.

        Both 'negative_language' and 'calendar_density' count as stress signals.
        """
        # Populate mood_signals profile with 3 stress signals (negative_language + calendar_density)
        stress_signals = [
            {"signal_type": "negative_language", "ts": "2026-01-01T00:00:00Z"},
            {"signal_type": "negative_language", "ts": "2026-01-01T01:00:00Z"},
            {"signal_type": "calendar_density", "ts": "2026-01-01T02:00:00Z"},
            {"signal_type": "positive_language", "ts": "2026-01-01T03:00:00Z"},
            {"signal_type": "positive_language", "ts": "2026-01-01T04:00:00Z"},
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": stress_signals})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        # 3 stress signals (2 negative_language + 1 calendar_density) > threshold of 2
        # Base 0.3 - 0.1 stress = 0.2 → still helpful (>= 0.2)
        assert "stress_signals=3" in reaction.reasoning
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_stress_penalty_not_applied_below_threshold(self, db, user_model_store):
        """When <=2 of last 5 mood signals are negative, no stress penalty is applied."""
        mild_signals = [
            {"signal_type": "negative_language", "ts": "2026-01-01T00:00:00Z"},
            {"signal_type": "positive_language", "ts": "2026-01-01T01:00:00Z"},
            {"signal_type": "neutral", "ts": "2026-01-01T02:00:00Z"},
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": mild_signals})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3, no penalty → helpful
        assert "stress_signals=1" in reaction.reasoning
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_dismissal_fatigue_penalty(self, db, user_model_store):
        """When >5 recent dismissals exist, score decreases by 0.2."""
        now = datetime.now(timezone.utc)

        # Insert 6 recent dismissals (>5 threshold)
        with db.get_connection("preferences") as conn:
            for i in range(6):
                conn.execute(
                    """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "dismissed",
                        str(uuid.uuid4()),
                        "notification",
                        (now - timedelta(minutes=30 + i * 10)).isoformat(),
                    ),
                )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 - 0.2 dismissal = 0.1 → neutral (> -0.1 but < 0.2)
        assert "dismissals=6" in reaction.reasoning
        assert reaction.predicted_reaction == "neutral"

    @pytest.mark.asyncio
    async def test_dismissal_fatigue_not_applied_at_threshold(self, db, user_model_store):
        """When exactly 5 dismissals exist (<=5), no penalty is applied."""
        now = datetime.now(timezone.utc)

        with db.get_connection("preferences") as conn:
            for i in range(5):
                conn.execute(
                    """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "dismissed",
                        str(uuid.uuid4()),
                        "notification",
                        (now - timedelta(minutes=30 + i * 10)).isoformat(),
                    ),
                )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3, no penalty (5 <= 5) → helpful
        assert "dismissals=5" in reaction.reasoning
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_high_confidence_boost(self, db, user_model_store):
        """Predictions with confidence > 0.7 get a +0.2 boost."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.8, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 + 0.2 high confidence = 0.5 → helpful
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_conflict_urgency_boost(self, db, user_model_store):
        """Conflict predictions get a +0.2 urgency boost."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="conflict",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 + 0.2 conflict = 0.5 → helpful
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_risk_urgency_boost(self, db, user_model_store):
        """Risk predictions get a +0.2 urgency boost."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="risk",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 + 0.2 risk = 0.5 → helpful
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_opportunity_penalty(self, db, user_model_store):
        """Opportunity predictions get a -0.05 penalty (nice-to-have, not need-to-know)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="opportunity",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 - 0.05 opportunity = 0.25 → helpful (>= 0.2)
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_quiet_hours_penalty_for_non_urgent(self, db, user_model_store):
        """During quiet hours, non-urgent predictions get a -0.2 penalty."""
        # Configure quiet hours for every day, all hours (to ensure we're in them)
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "00:00", "end": "23:59", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="opportunity",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 - 0.05 opportunity - 0.2 quiet = 0.05 → neutral (> -0.1 but < 0.2)
        assert "quiet_hours=True" in reaction.reasoning
        assert reaction.predicted_reaction == "neutral"

    @pytest.mark.asyncio
    async def test_quiet_hours_exemption_for_conflicts(self, db, user_model_store):
        """Conflict predictions are exempt from quiet hours penalty — they always get through."""
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "00:00", "end": "23:59", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="conflict",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 + 0.2 conflict, no quiet penalty = 0.5 → helpful
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_quiet_hours_exemption_for_risks(self, db, user_model_store):
        """Risk predictions are exempt from quiet hours penalty."""
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "00:00", "end": "23:59", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.5,
            prediction_type="risk",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # Base 0.3 + 0.2 risk, no quiet penalty = 0.5 → helpful
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_classification_helpful_threshold(self, db, user_model_store):
        """Score >= 0.2 should classify as 'helpful'."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        # Base score 0.3 with no modifiers → 0.3 >= 0.2 → helpful
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})
        assert reaction.predicted_reaction == "helpful"

    @pytest.mark.asyncio
    async def test_classification_neutral_threshold(self, db, user_model_store):
        """Score > -0.1 but < 0.2 should classify as 'neutral'."""
        # Use dismissal fatigue to push score down:
        # Base 0.3 - 0.2 dismissal = 0.1 → neutral
        now = datetime.now(timezone.utc)
        with db.get_connection("preferences") as conn:
            for i in range(6):
                conn.execute(
                    """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "dismissed",
                        str(uuid.uuid4()),
                        "notification",
                        (now - timedelta(minutes=30 + i * 10)).isoformat(),
                    ),
                )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(confidence=0.5, prediction_type="reminder")

        reaction = await engine.predict_reaction(pred, {})
        assert reaction.predicted_reaction == "neutral"

    @pytest.mark.asyncio
    async def test_classification_annoying_threshold(self, db, user_model_store):
        """Score <= -0.1 should classify as 'annoying'."""
        # Stack penalties: stress + dismissals + opportunity + quiet hours
        # Base 0.3 - 0.1 stress - 0.2 dismissals - 0.05 opportunity - 0.2 quiet = -0.25 → annoying
        now = datetime.now(timezone.utc)

        # Add stress signals
        stress_signals = [
            {"signal_type": "negative_language", "ts": "2026-01-01T00:00:00Z"},
            {"signal_type": "negative_language", "ts": "2026-01-01T01:00:00Z"},
            {"signal_type": "calendar_density", "ts": "2026-01-01T02:00:00Z"},
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": stress_signals})

        # Add 6 dismissals
        with db.get_connection("preferences") as conn:
            for i in range(6):
                conn.execute(
                    """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "dismissed",
                        str(uuid.uuid4()),
                        "notification",
                        (now - timedelta(minutes=30 + i * 10)).isoformat(),
                    ),
                )

        # Add quiet hours
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "00:00", "end": "23:59", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.3,
            prediction_type="opportunity",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})
        assert reaction.predicted_reaction == "annoying"

    @pytest.mark.asyncio
    async def test_combined_scoring_stress_dismissals_quiet_opportunity(self, db, user_model_store):
        """Stress + dismissals + quiet hours on a low-confidence opportunity should be 'annoying'.

        This tests the worst-case combined penalty scenario, verifying that
        all negative signals accumulate correctly.
        """
        now = datetime.now(timezone.utc)

        # Stack all penalties
        stress_signals = [
            {"signal_type": "negative_language"},
            {"signal_type": "negative_language"},
            {"signal_type": "calendar_density"},
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": stress_signals})

        with db.get_connection("preferences") as conn:
            for i in range(7):
                conn.execute(
                    """INSERT INTO feedback_log (id, feedback_type, action_id, action_type, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "dismissed",
                        str(uuid.uuid4()),
                        "notification",
                        (now - timedelta(minutes=10 + i * 5)).isoformat(),
                    ),
                )

        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "00:00", "end": "23:59", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        pred = _make_prediction(
            confidence=0.3,
            prediction_type="opportunity",
            confidence_gate=ConfidenceGate.SUGGEST,
        )

        reaction = await engine.predict_reaction(pred, {})

        # 0.3 - 0.1 stress - 0.2 dismissal - 0.05 opportunity - 0.2 quiet = -0.25
        assert reaction.predicted_reaction == "annoying"


# ---------------------------------------------------------------------------
# Group 3: _get_accuracy_multiplier()
# ---------------------------------------------------------------------------


class TestAccuracyMultiplier:
    """Test the calibration curve for the accuracy multiplier.

    Returns:
        1.0  — insufficient data (<5 resolved predictions)
        0.3  — heavy penalty floor (<20% accuracy with >= 10 samples)
        0.5 + (accuracy_rate * 0.6) — scaled by accuracy rate
    """

    def test_insufficient_data_returns_baseline(self, db, user_model_store):
        """With <5 resolved predictions, accuracy multiplier should be 1.0."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        multiplier = engine._get_accuracy_multiplier("reminder")
        assert multiplier == 1.0

    def test_insufficient_data_with_some_predictions(self, db, user_model_store):
        """With exactly 4 resolved predictions (still <5), should return 1.0."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(4):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "reminder", "Test", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 1,
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("reminder")
        assert multiplier == 1.0

    def test_zero_accuracy_returns_floor(self, db, user_model_store):
        """10 predictions with 0% accuracy returns 0.3 (penalty floor, not 0.0)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(10):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "reminder", "Test", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 0,
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("reminder")
        assert multiplier == 0.3, f"Expected 0.3 penalty floor, got {multiplier}"

    def test_fifty_percent_accuracy(self, db, user_model_store):
        """10 predictions with 50% accuracy returns 0.8 (0.5 + 0.5 * 0.6)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(10):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "reminder", "Test", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(),
                        1 if i < 5 else 0,
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("reminder")
        assert abs(multiplier - 0.8) < 0.01

    def test_perfect_accuracy(self, db, user_model_store):
        """10 predictions with 100% accuracy returns 1.1 (0.5 + 1.0 * 0.6)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(10):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "reminder", "Test", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 1,
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("reminder")
        assert abs(multiplier - 1.1) < 0.01

    def test_automated_sender_fast_path_excluded(self, db, user_model_store):
        """Predictions resolved via automated_sender_fast_path should be excluded from accuracy calc.

        These are structural bugs in prediction generation, not real user behavior.
        Without this exclusion, fast-path resolutions inflate the inaccuracy count.
        """
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # 5 real predictions — all accurate
            for i in range(5):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        resolution_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Real prediction", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 1,
                        None,  # No special resolution reason
                    ),
                )

            # 50 fast-path resolutions — all inaccurate (should be excluded)
            for i in range(50):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        resolution_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Marketing sender", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 0,
                        "automated_sender_fast_path",
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("opportunity")
        # Should only count the 5 real predictions (100% accurate)
        # 0.5 + 1.0 * 0.6 = 1.1
        assert abs(multiplier - 1.1) < 0.01, (
            f"Expected ~1.1 (only real predictions counted), got {multiplier}. "
            "Fast-path resolutions may not be excluded correctly."
        )

    def test_penalty_floor_requires_ten_samples(self, db, user_model_store):
        """The 0.3 penalty floor only applies with >= 10 samples; 9 samples uses the formula."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        # 9 predictions, 0% accuracy → should use formula (0.5 + 0 * 0.6 = 0.5), not floor
        with db.get_connection("user_model") as conn:
            for i in range(9):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "reminder", "Test", 0.7, "suggest",
                        "24_hours", now.isoformat(), 1, now.isoformat(), 0,
                    ),
                )

        multiplier = engine._get_accuracy_multiplier("reminder")
        # 0% accuracy with 9 samples → formula: 0.5 + 0.0 * 0.6 = 0.5 (not the 0.3 floor)
        assert abs(multiplier - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Group 4: _get_contact_accuracy_multiplier()
# ---------------------------------------------------------------------------


class TestContactAccuracyMultiplier:
    """Test per-contact calibration in _get_contact_accuracy_multiplier().

    Returns:
        1.0  — insufficient data (<3 resolved predictions for this contact)
        0.5  — poor response rate (< 20% accuracy, 3+ samples)
        0.5 + (accuracy_rate * 0.7) — scaled, capped at 1.2
    """

    def test_insufficient_data_returns_baseline(self, db, user_model_store):
        """With <3 resolved predictions for a contact, returns 1.0."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        multiplier = engine._get_contact_accuracy_multiplier("alice@example.com")
        assert multiplier == 1.0

    def test_insufficient_data_with_two_predictions(self, db, user_model_store):
        """With exactly 2 resolved predictions (still <3), returns 1.0."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(2):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Alice", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 1,
                        json.dumps({"contact_email": "alice@example.com"}),
                    ),
                )

        multiplier = engine._get_contact_accuracy_multiplier("alice@example.com")
        assert multiplier == 1.0

    def test_zero_accuracy_returns_floor(self, db, user_model_store):
        """5 predictions with 0% accuracy for a contact returns 0.5 (floor)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(5):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Bob", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 0,
                        json.dumps({"contact_email": "bob@example.com"}),
                    ),
                )

        multiplier = engine._get_contact_accuracy_multiplier("bob@example.com")
        assert multiplier == 0.5

    def test_perfect_accuracy_returns_cap(self, db, user_model_store):
        """5 predictions with 100% accuracy returns 1.2 (cap)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            for i in range(5):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Carol", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 1,
                        json.dumps({"contact_email": "carol@example.com"}),
                    ),
                )

        multiplier = engine._get_contact_accuracy_multiplier("carol@example.com")
        assert abs(multiplier - 1.2) < 0.01

    def test_contact_isolation(self, db, user_model_store):
        """Predictions for other contacts should not affect this contact's multiplier."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # 5 inaccurate predictions for alice
            for i in range(5):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Alice", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 0,
                        json.dumps({"contact_email": "alice@example.com"}),
                    ),
                )

            # 5 accurate predictions for bob
            for i in range(5):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Bob", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 1,
                        json.dumps({"contact_email": "bob@example.com"}),
                    ),
                )

        # Alice's accuracy should be poor (0%), unaffected by Bob's 100%
        alice_multiplier = engine._get_contact_accuracy_multiplier("alice@example.com")
        assert alice_multiplier == 0.5

        # Bob's accuracy should be high (100%), unaffected by Alice's 0%
        bob_multiplier = engine._get_contact_accuracy_multiplier("bob@example.com")
        assert abs(bob_multiplier - 1.2) < 0.01

    def test_fast_path_excluded_for_contacts(self, db, user_model_store):
        """Fast-path automated sender resolutions should also be excluded per-contact."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        now = datetime.now(timezone.utc)

        with db.get_connection("user_model") as conn:
            # 3 real accurate predictions
            for i in range(3):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals, resolution_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Dave", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 1,
                        json.dumps({"contact_email": "dave@example.com"}),
                        None,
                    ),
                )

            # 10 fast-path inaccurate (should be excluded)
            for i in range(10):
                conn.execute(
                    """INSERT INTO predictions
                       (id, prediction_type, description, confidence, confidence_gate,
                        time_horizon, created_at, was_surfaced, resolved_at, was_accurate,
                        supporting_signals, resolution_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()), "opportunity", "Reach out to Dave", 0.5, "suggest",
                        "this_week", now.isoformat(), 1, now.isoformat(), 0,
                        json.dumps({"contact_email": "dave@example.com"}),
                        "automated_sender_fast_path",
                    ),
                )

        multiplier = engine._get_contact_accuracy_multiplier("dave@example.com")
        # Only 3 real predictions (100% accurate): 0.5 + 1.0 * 0.7 = 1.2
        assert abs(multiplier - 1.2) < 0.01


# ---------------------------------------------------------------------------
# Group 5: _is_quiet_hours()
# ---------------------------------------------------------------------------


class TestIsQuietHours:
    """Test quiet hours detection logic.

    Quiet hours are configured as JSON in the preferences DB:
        [{"start": "22:00", "end": "07:00", "days": ["monday", ...]}]

    The method returns False (fail-open) when no config exists or data is malformed.
    """

    def test_within_same_day_range(self, db, user_model_store):
        """Time within a same-day quiet hours range returns True."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Configure quiet hours for all days, 09:00-17:00
        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "09:00", "end": "17:00", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # Test at 12:00 on a known day — should be within range
        test_time = datetime(2026, 3, 2, 12, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(test_time) is True

    def test_outside_same_day_range(self, db, user_model_store):
        """Time outside a same-day quiet hours range returns False."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "09:00", "end": "17:00", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # 20:00 is outside 09:00-17:00
        test_time = datetime(2026, 3, 2, 20, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(test_time) is False

    def test_overnight_range_before_midnight(self, db, user_model_store):
        """Overnight range (22:00-07:00) with time at 23:00 returns True."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "22:00", "end": "07:00", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # 23:00 is >= 22:00 → within overnight range
        test_time = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(test_time) is True

    def test_overnight_range_after_midnight(self, db, user_model_store):
        """Overnight range (22:00-07:00) with time at 03:00 returns True."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "22:00", "end": "07:00", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # 03:00 is <= 07:00 → within overnight range
        test_time = datetime(2026, 3, 2, 3, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(test_time) is True

    def test_overnight_range_outside(self, db, user_model_store):
        """Overnight range (22:00-07:00) with time at 08:00 returns False."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        all_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        quiet_config = json.dumps([{"start": "22:00", "end": "07:00", "days": all_days}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # 08:00 is > 07:00 and < 22:00 → outside overnight range
        test_time = datetime(2026, 3, 2, 8, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(test_time) is False

    def test_no_quiet_hours_preference_returns_false(self, db, user_model_store):
        """Missing quiet_hours preference returns False (fail-open)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # No quiet hours configured
        test_time = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(test_time) is False

    def test_malformed_json_returns_false(self, db, user_model_store):
        """Malformed JSON in quiet_hours preference returns False (fail-open)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                ("not valid json {{{",),
            )

        test_time = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(test_time) is False

    def test_day_of_week_filtering(self, db, user_model_store):
        """Quiet hours only apply on configured days."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Quiet hours only on weekdays
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        quiet_config = json.dumps([{"start": "22:00", "end": "07:00", "days": weekdays}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # Saturday at 23:00 — not in quiet hours (weekdays only)
        saturday = datetime(2026, 2, 28, 23, 0, 0, tzinfo=ZoneInfo("UTC"))  # Saturday
        assert engine._is_quiet_hours(saturday) is False

        # Monday at 23:00 — in quiet hours (weekday)
        monday = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))  # Monday
        assert engine._is_quiet_hours(monday) is True

    def test_multiple_quiet_hour_ranges(self, db, user_model_store):
        """Multiple quiet hour ranges are supported (e.g., different weekday/weekend times)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        quiet_config = json.dumps([
            {"start": "22:00", "end": "07:00", "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]},
            {"start": "23:00", "end": "09:00", "days": ["saturday", "sunday"]},
        ])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        # Monday 22:30 — weekday range applies
        monday_late = datetime(2026, 3, 2, 22, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(monday_late) is True

        # Saturday 22:30 — not in weekend range yet (starts at 23:00)
        saturday_late = datetime(2026, 2, 28, 22, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(saturday_late) is False

        # Saturday 23:30 — weekend range applies
        saturday_later = datetime(2026, 2, 28, 23, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(saturday_later) is True

    def test_empty_quiet_hours_list_returns_false(self, db, user_model_store):
        """An empty quiet hours list returns False."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                ("[]",),
            )

        test_time = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(test_time) is False

    def test_missing_days_field_skips_range(self, db, user_model_store):
        """A quiet hour range with no 'days' field is skipped (fail-open)."""
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Range with empty days list — current day won't match
        quiet_config = json.dumps([{"start": "22:00", "end": "07:00", "days": []}])
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT INTO user_preferences (key, value) VALUES ('quiet_hours', ?)",
                (quiet_config,),
            )

        test_time = datetime(2026, 3, 2, 23, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert engine._is_quiet_hours(test_time) is False
