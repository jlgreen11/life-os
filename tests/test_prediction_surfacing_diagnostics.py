"""
Tests for the prediction engine's surfacing diagnostics.

Verifies that _surfacing_diagnostics tracks per-prediction filter reasons,
reaction score distributions, penalty frequencies, and sample filtered
reasons — giving operators visibility into WHY predictions are filtered.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import (
    PredictionEngine,
    _parse_penalty_frequency,
    _parse_score_from_reasoning,
)


def _make_prediction(prediction_type="reminder", confidence=0.5):
    """Create a minimal Prediction for testing."""
    return Prediction(
        id=str(uuid.uuid4()),
        prediction_type=prediction_type,
        description=f"Test {prediction_type} prediction",
        confidence=confidence,
        confidence_gate="suggest" if confidence >= 0.3 else "observe",
        time_horizon="24_hours",
        supporting_signals={},
    )


def _make_reaction(score, label=None):
    """Create a ReactionPrediction with a structured reasoning string."""
    if label is None:
        label = "helpful" if score >= 0.2 else ("neutral" if score > -0.1 else "annoying")
    return ReactionPrediction(
        proposed_action="test",
        predicted_reaction=label,
        confidence=abs(score),
        reasoning=f"score={score:.2f}, dismissals=0, stress_signals=0, quiet_hours=False, low_activity=False",
    )


def _patch_prediction_generation(engine, predictions, reaction_fn):
    """Return a context manager that patches the engine to use fixed predictions and reactions.

    Patches all _check_* methods to return nothing, then injects *predictions*
    via _check_follow_up_needs (which always runs).  Also patches predict_reaction
    with *reaction_fn* and forces triggers so the pipeline runs.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        # All time-based check methods return empty
        time_checks = [
            "_check_calendar_conflicts",
            "_check_routine_deviations",
            "_check_relationship_maintenance",
            "_check_preparation_needs",
            "_check_calendar_event_reminders",
            "_check_connector_health",
            "_check_spending_patterns",
        ]
        patches = [patch.object(engine, m, new_callable=AsyncMock, return_value=[]) for m in time_checks]

        # Inject predictions via follow_up_needs
        patches.append(patch.object(engine, "_check_follow_up_needs", new_callable=AsyncMock, return_value=predictions))
        # Force triggers
        patches.append(patch.object(engine, "_has_new_events", return_value=True))
        # Mock predict_reaction
        patches.append(patch.object(engine, "predict_reaction", side_effect=reaction_fn))
        # Suppress not-relevant filter (no suppressed keys)
        patches.append(patch.object(engine, "_get_suppressed_prediction_keys", return_value=set()))
        # Identity accuracy multiplier
        patches.append(patch.object(engine, "_get_accuracy_multiplier", return_value=1.0))

        for p in patches:
            p.start()
        try:
            yield
        finally:
            for p in patches:
                p.stop()

    return _ctx()


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestParseScoreFromReasoning:
    """Tests for _parse_score_from_reasoning()."""

    def test_parses_positive_score(self):
        """Extracts a positive score from a standard reasoning string."""
        reasoning = "score=0.30, dismissals=0, stress_signals=0, quiet_hours=False, low_activity=False"
        assert _parse_score_from_reasoning(reasoning) == 0.30

    def test_parses_negative_score(self):
        """Extracts a negative score."""
        reasoning = "score=-0.10, dismissals=6, stress_signals=3, quiet_hours=True, low_activity=False"
        assert _parse_score_from_reasoning(reasoning) == -0.10

    def test_returns_none_on_malformed(self):
        """Returns None when reasoning has no score= prefix."""
        assert _parse_score_from_reasoning("no score here") is None

    def test_returns_none_on_empty(self):
        """Returns None for empty string."""
        assert _parse_score_from_reasoning("") is None


class TestParsePenaltyFrequency:
    """Tests for _parse_penalty_frequency()."""

    def test_counts_stress_penalty(self):
        """Stress penalty counted when stress_signals > 2."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.20, dismissals=0, stress_signals=3, quiet_hours=False, low_activity=False"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["stress"] == 1

    def test_no_stress_below_threshold(self):
        """Stress penalty NOT counted when stress_signals <= 2."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.30, dismissals=0, stress_signals=2, quiet_hours=False, low_activity=False"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["stress"] == 0

    def test_counts_dismissals_penalty(self):
        """Dismissals penalty counted when dismissals > 5."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.10, dismissals=6, stress_signals=0, quiet_hours=False, low_activity=False"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["dismissals"] == 1

    def test_counts_quiet_hours(self):
        """Quiet hours penalty counted when quiet_hours=True."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.10, dismissals=0, stress_signals=0, quiet_hours=True, low_activity=False"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["quiet_hours"] == 1

    def test_counts_low_activity(self):
        """Low activity penalty counted when low_activity=True."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.10, dismissals=0, stress_signals=0, quiet_hours=False, low_activity=True"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["low_activity"] == 1

    def test_counts_opportunity_type(self):
        """Opportunity type penalty counted for opportunity predictions."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.25, dismissals=0, stress_signals=0, quiet_hours=False, low_activity=False"
        _parse_penalty_frequency(reasoning, "opportunity", freq)
        assert freq["opportunity_type"] == 1

    def test_no_opportunity_for_other_types(self):
        """Opportunity type penalty NOT counted for non-opportunity predictions."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=0.25, dismissals=0, stress_signals=0, quiet_hours=False, low_activity=False"
        _parse_penalty_frequency(reasoning, "reminder", freq)
        assert freq["opportunity_type"] == 0

    def test_multiple_penalties(self):
        """All applicable penalties counted from a single reasoning string."""
        freq = {"stress": 0, "dismissals": 0, "quiet_hours": 0, "low_activity": 0, "opportunity_type": 0}
        reasoning = "score=-0.20, dismissals=8, stress_signals=4, quiet_hours=True, low_activity=False"
        _parse_penalty_frequency(reasoning, "opportunity", freq)
        assert freq["stress"] == 1
        assert freq["dismissals"] == 1
        assert freq["quiet_hours"] == 1
        assert freq["opportunity_type"] == 1


# ---------------------------------------------------------------------------
# Score distribution bucketing
# ---------------------------------------------------------------------------


class TestScoreDistributionBucketing:
    """Tests for PredictionEngine._bucket_reaction_score()."""

    def test_below_neg01(self, prediction_engine):
        """Scores < -0.1 go to 'below_neg0.1' bucket."""
        assert prediction_engine._bucket_reaction_score(-0.2) == "below_neg0.1"
        assert prediction_engine._bucket_reaction_score(-1.0) == "below_neg0.1"

    def test_neg01_to_02(self, prediction_engine):
        """Scores in [-0.1, 0.2) go to 'neg0.1_to_0.2' bucket."""
        assert prediction_engine._bucket_reaction_score(-0.1) == "neg0.1_to_0.2"
        assert prediction_engine._bucket_reaction_score(0.0) == "neg0.1_to_0.2"
        assert prediction_engine._bucket_reaction_score(0.19) == "neg0.1_to_0.2"

    def test_02_to_05(self, prediction_engine):
        """Scores in [0.2, 0.5] go to '0.2_to_0.5' bucket."""
        assert prediction_engine._bucket_reaction_score(0.2) == "0.2_to_0.5"
        assert prediction_engine._bucket_reaction_score(0.35) == "0.2_to_0.5"
        assert prediction_engine._bucket_reaction_score(0.5) == "0.2_to_0.5"

    def test_above_05(self, prediction_engine):
        """Scores > 0.5 go to 'above_0.5' bucket."""
        assert prediction_engine._bucket_reaction_score(0.51) == "above_0.5"
        assert prediction_engine._bucket_reaction_score(1.0) == "above_0.5"


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestSurfacingDiagnosticsStructure:
    """Tests that _surfacing_diagnostics is properly initialized and structured."""

    def test_initial_diagnostics_structure(self, prediction_engine):
        """_surfacing_diagnostics has all required keys after engine creation."""
        diag = prediction_engine._surfacing_diagnostics
        assert "total_generated" in diag
        assert "filtered_by_reaction" in diag
        assert "filtered_by_confidence" in diag
        assert "score_distribution" in diag
        assert "penalty_frequency" in diag
        assert "sample_filtered_reasons" in diag

    def test_initial_values_are_zero(self, prediction_engine):
        """All counters start at zero."""
        diag = prediction_engine._surfacing_diagnostics
        assert diag["total_generated"] == 0
        assert diag["filtered_by_confidence"] == 0
        assert diag["filtered_by_reaction"]["total"] == 0
        assert all(v == 0 for v in diag["score_distribution"].values())
        assert all(v == 0 for v in diag["penalty_frequency"].values())
        assert diag["sample_filtered_reasons"] == []

    def test_runtime_diagnostics_includes_surfacing(self, prediction_engine):
        """get_runtime_diagnostics() includes the 'surfacing' key."""
        runtime = prediction_engine.get_runtime_diagnostics()
        assert "surfacing" in runtime
        assert runtime["surfacing"]["total_generated"] == 0

    async def test_get_diagnostics_includes_surfacing(self, prediction_engine):
        """get_diagnostics() includes the 'surfacing' key."""
        diag = await prediction_engine.get_diagnostics()
        assert "surfacing" in diag
        assert "total_generated" in diag["surfacing"]


# ---------------------------------------------------------------------------
# Integration tests — diagnostics populated after generate_predictions()
# ---------------------------------------------------------------------------


class TestSurfacingDiagnosticsAfterRun:
    """Tests that diagnostics are populated correctly after generate_predictions()."""

    async def test_reaction_filtering_tracked(self, prediction_engine):
        """Predictions filtered by reaction are counted in diagnostics."""
        preds = [_make_prediction() for _ in range(3)]
        reaction = _make_reaction(-0.20, "annoying")

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert diag["total_generated"] == 3
        assert diag["filtered_by_reaction"]["total"] == 3
        assert diag["filtered_by_reaction"]["annoying"] == 3

    async def test_confidence_filtering_tracked(self, prediction_engine):
        """Predictions filtered by confidence floor are counted."""
        preds = [_make_prediction(confidence=0.1) for _ in range(2)]
        reaction = _make_reaction(0.50, "helpful")

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert diag["total_generated"] == 2
        assert diag["filtered_by_confidence"] == 2

    async def test_score_distribution_bucketing(self, prediction_engine):
        """Reaction scores are bucketed correctly in score_distribution."""
        preds = [_make_prediction() for _ in range(4)]
        scores = [-0.20, 0.10, 0.30, 0.60]
        call_idx = 0

        async def react(pred, ctx):
            nonlocal call_idx
            r = _make_reaction(scores[call_idx])
            call_idx += 1
            return r

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert diag["score_distribution"]["below_neg0.1"] == 1   # -0.20
        assert diag["score_distribution"]["neg0.1_to_0.2"] == 1  # 0.10
        assert diag["score_distribution"]["0.2_to_0.5"] == 1     # 0.30
        assert diag["score_distribution"]["above_0.5"] == 1      # 0.60

    async def test_penalty_frequency_tracking(self, prediction_engine):
        """Penalty frequency counters track each penalty type."""
        preds = [_make_prediction(prediction_type="opportunity")]
        reaction = ReactionPrediction(
            proposed_action="test",
            predicted_reaction="annoying",
            confidence=0.5,
            reasoning="score=-0.20, dismissals=8, stress_signals=4, quiet_hours=True, low_activity=False",
        )

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert diag["penalty_frequency"]["stress"] == 1
        assert diag["penalty_frequency"]["dismissals"] == 1
        assert diag["penalty_frequency"]["quiet_hours"] == 1
        assert diag["penalty_frequency"]["opportunity_type"] == 1
        assert diag["penalty_frequency"]["low_activity"] == 0

    async def test_sample_filtered_reasons_contains_details(self, prediction_engine):
        """Sample filtered reasons include prediction type, score, and reason."""
        preds = [_make_prediction(prediction_type="reminder")]
        reaction = _make_reaction(-0.10, "annoying")

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert len(diag["sample_filtered_reasons"]) == 1
        sample = diag["sample_filtered_reasons"][0]
        assert sample["prediction_type"] == "reminder"
        assert sample["score"] == -0.10
        assert sample["reaction"] == "annoying"
        assert "reason" in sample

    async def test_sample_filtered_reasons_capped_at_5(self, prediction_engine):
        """sample_filtered_reasons never exceeds 5 entries."""
        preds = [_make_prediction() for _ in range(8)]
        reaction = _make_reaction(-0.20, "annoying")

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        diag = prediction_engine._surfacing_diagnostics
        assert len(diag["sample_filtered_reasons"]) == 5

    async def test_last_run_diagnostics_includes_surfacing(self, prediction_engine):
        """_last_run_diagnostics includes the 'surfacing' key after a run."""
        preds = [_make_prediction()]
        reaction = _make_reaction(0.50, "helpful")

        async def react(pred, ctx):
            return reaction

        with _patch_prediction_generation(prediction_engine, preds, react):
            await prediction_engine.generate_predictions({})

        assert "surfacing" in prediction_engine._last_run_diagnostics

    async def test_diagnostics_reset_between_runs(self, prediction_engine):
        """Surfacing diagnostics are reset at the start of each run."""
        # First run: 3 predictions, all annoying
        preds1 = [_make_prediction() for _ in range(3)]
        reaction1 = _make_reaction(-0.20, "annoying")

        async def react1(pred, ctx):
            return reaction1

        with _patch_prediction_generation(prediction_engine, preds1, react1):
            await prediction_engine.generate_predictions({})

        assert prediction_engine._surfacing_diagnostics["total_generated"] == 3

        # Second run: 1 prediction, helpful
        preds2 = [_make_prediction()]
        reaction2 = _make_reaction(0.50, "helpful")

        async def react2(pred, ctx):
            return reaction2

        with _patch_prediction_generation(prediction_engine, preds2, react2):
            await prediction_engine.generate_predictions({})

        # Should reflect only the second run
        diag = prediction_engine._surfacing_diagnostics
        assert diag["total_generated"] == 1
        assert diag["filtered_by_reaction"]["total"] == 0
