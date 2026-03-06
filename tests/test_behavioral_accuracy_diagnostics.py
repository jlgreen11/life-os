"""Tests for BehavioralAccuracyTracker.get_diagnostics() method.

Validates that the diagnostics method returns correct per-type resolution
stats, resolution method breakdown, unresolved prediction details, inference
cycle stats, and health assessment.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


@pytest.fixture()
def tracker(db):
    """A BehavioralAccuracyTracker wired to the temporary DatabaseManager."""
    return BehavioralAccuracyTracker(db)


def _insert_prediction(
    db,
    *,
    prediction_type="reminder",
    description="Test prediction",
    confidence=0.5,
    was_surfaced=1,
    was_accurate=None,
    user_response=None,
    resolved_at=None,
    resolution_reason=None,
    supporting_signals=None,
    created_at=None,
):
    """Insert a prediction row for testing."""
    pred_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if supporting_signals is None:
        supporting_signals = "[]"
    elif isinstance(supporting_signals, (dict, list)):
        supporting_signals = json.dumps(supporting_signals)

    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, was_accurate, user_response, resolved_at,
                resolution_reason, supporting_signals, created_at)
               VALUES (?, ?, ?, ?, 'SUGGEST', ?, ?, ?, ?, ?, ?, ?)""",
            (
                pred_id, prediction_type, description, confidence,
                was_surfaced, was_accurate, user_response, resolved_at,
                resolution_reason, supporting_signals, created_at,
            ),
        )
    return pred_id


class TestPerTypeResolutionStats:
    """Tests for per-type resolution stats in get_diagnostics()."""

    def test_returns_correct_per_type_stats(self, db, tracker):
        """get_diagnostics() groups predictions by type with correct counts."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=6)).isoformat()

        # 2 reminder predictions: 1 resolved accurate, 1 unresolved
        _insert_prediction(db, prediction_type="reminder", was_accurate=1,
                           resolved_at=recent, user_response="inferred", created_at=recent)
        _insert_prediction(db, prediction_type="reminder", was_surfaced=1, created_at=recent)

        # 1 routine_deviation: resolved inaccurate
        _insert_prediction(db, prediction_type="routine_deviation", was_accurate=0,
                           resolved_at=recent, user_response="inferred", created_at=recent)

        result = tracker.get_diagnostics()
        stats = result["per_type_stats"]

        assert "reminder" in stats
        assert stats["reminder"]["total"] == 2
        assert stats["reminder"]["resolved"] == 1
        assert stats["reminder"]["accurate"] == 1
        assert stats["reminder"]["unresolved_surfaced"] == 1

        assert "routine_deviation" in stats
        assert stats["routine_deviation"]["total"] == 1
        assert stats["routine_deviation"]["inaccurate"] == 1

    def test_empty_predictions_table(self, db, tracker):
        """get_diagnostics() returns empty stats for no predictions."""
        result = tracker.get_diagnostics()
        assert result["per_type_stats"] == {}
        assert result["resolution_methods"] == {}
        assert result["unresolved_details"] == []
        assert result["health"] == "healthy"


class TestResolutionMethodBreakdown:
    """Tests for resolution method breakdown in get_diagnostics()."""

    def test_resolution_method_counts(self, db, tracker):
        """get_diagnostics() groups resolved predictions by user_response."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()
        resolved = recent

        _insert_prediction(db, user_response="inferred", resolved_at=resolved,
                           was_accurate=1, created_at=recent)
        _insert_prediction(db, user_response="inferred", resolved_at=resolved,
                           was_accurate=0, created_at=recent)
        _insert_prediction(db, user_response="acted_on", resolved_at=resolved,
                           was_accurate=1, created_at=recent)
        _insert_prediction(db, user_response="dismissed", resolved_at=resolved,
                           was_accurate=0, created_at=recent)

        result = tracker.get_diagnostics()
        methods = result["resolution_methods"]
        assert methods["inferred"] == 2
        assert methods["acted_on"] == 1
        assert methods["dismissed"] == 1


class TestUnresolvedDetails:
    """Tests for unresolved prediction details in get_diagnostics()."""

    def test_shows_unresolved_with_signal_keys(self, db, tracker):
        """get_diagnostics() returns unresolved predictions with supporting_signals info."""
        now = datetime.now(timezone.utc)
        # Old enough to be past observation window
        old = (now - timedelta(hours=12)).isoformat()

        _insert_prediction(
            db,
            prediction_type="reminder",
            description="Reply to Alice about dinner plans" + " " * 100,
            was_surfaced=1,
            created_at=old,
            supporting_signals={"expected_actions": ["send_message"], "contact": "Alice"},
        )

        result = tracker.get_diagnostics()
        details = result["unresolved_details"]
        assert len(details) == 1
        detail = details[0]
        assert detail["prediction_type"] == "reminder"
        assert len(detail["description"]) <= 100
        assert detail["age_hours"] is not None
        assert detail["age_hours"] > 10
        assert "expected_actions" in detail["signal_keys"]
        assert "contact" in detail["signal_keys"]
        assert detail["reason"] == "no_matching_behavior_detected"

    def test_within_observation_window(self, db, tracker):
        """Predictions created recently show within_observation_window reason."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=1)).isoformat()

        _insert_prediction(db, was_surfaced=1, created_at=recent,
                           supporting_signals={"expected_actions": ["reply"]})

        result = tracker.get_diagnostics()
        details = result["unresolved_details"]
        assert len(details) == 1
        assert details[0]["reason"] == "within_observation_window"

    def test_missing_supporting_signals(self, db, tracker):
        """Predictions with no supporting_signals show missing_supporting_signals reason."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=12)).isoformat()

        _insert_prediction(db, was_surfaced=1, created_at=old, supporting_signals="[]")

        result = tracker.get_diagnostics()
        details = result["unresolved_details"]
        assert len(details) == 1
        assert details[0]["reason"] == "missing_supporting_signals"

    def test_limits_to_10_results(self, db, tracker):
        """Unresolved details are limited to 10 most recent predictions."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=12)).isoformat()

        for i in range(15):
            _insert_prediction(db, was_surfaced=1, created_at=old,
                               description=f"Prediction {i}")

        result = tracker.get_diagnostics()
        assert len(result["unresolved_details"]) == 10


class TestInferenceCycleStats:
    """Tests for inference cycle stats in get_diagnostics()."""

    def test_returns_initial_cycle_stats(self, db, tracker):
        """Before any cycles run, stats show zero cycles and no last stats."""
        result = tracker.get_diagnostics()
        assert result["inference_cycles"]["total_cycles"] == 0
        assert result["inference_cycles"]["last_cycle_stats"] is None

    @pytest.mark.asyncio
    async def test_returns_last_cycle_stats_after_run(self, db, tracker):
        """After run_inference_cycle(), get_diagnostics() returns cached stats."""
        stats = await tracker.run_inference_cycle()

        result = tracker.get_diagnostics()
        assert result["inference_cycles"]["total_cycles"] == 1
        assert result["inference_cycles"]["last_cycle_stats"] == stats

    @pytest.mark.asyncio
    async def test_increments_cycle_count(self, db, tracker):
        """Each run_inference_cycle() call increments total_cycles."""
        await tracker.run_inference_cycle()
        await tracker.run_inference_cycle()

        result = tracker.get_diagnostics()
        assert result["inference_cycles"]["total_cycles"] == 2


class TestHealthAssessment:
    """Tests for health assessment in get_diagnostics()."""

    def test_stalled_at_zero_resolution(self, db, tracker):
        """Health returns 'stalled' when 0% resolution rate."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()

        # 5 unresolved surfaced predictions
        for _ in range(5):
            _insert_prediction(db, was_surfaced=1, created_at=recent)

        result = tracker.get_diagnostics()
        assert result["health"] == "stalled"
        assert any("0%" in r for r in result["recommendations"])

    def test_healthy_above_50_percent(self, db, tracker):
        """Health returns 'healthy' when resolution rate > 50%."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()
        resolved = recent

        # 3 resolved, 2 unresolved = 60% resolution rate
        for _ in range(3):
            _insert_prediction(db, was_accurate=1, resolved_at=resolved,
                               user_response="inferred", created_at=recent)
        for _ in range(2):
            _insert_prediction(db, was_surfaced=1, created_at=recent)

        result = tracker.get_diagnostics()
        assert result["health"] == "healthy"

    def test_degraded_between_10_and_50_percent(self, db, tracker):
        """Health returns 'degraded' when resolution rate between 10-50%."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()
        resolved = recent

        # 2 resolved, 6 unresolved = 25% resolution rate
        for _ in range(2):
            _insert_prediction(db, was_accurate=1, resolved_at=resolved,
                               user_response="inferred", created_at=recent)
        for _ in range(6):
            _insert_prediction(db, was_surfaced=1, created_at=recent)

        result = tracker.get_diagnostics()
        assert result["health"] == "degraded"

    def test_no_cycles_recommendation(self, db, tracker):
        """Recommends starting inference loop when zero cycles have run."""
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=2)).isoformat()
        _insert_prediction(db, was_surfaced=1, created_at=recent)

        result = tracker.get_diagnostics()
        assert any("inference cycle" in r.lower() for r in result["recommendations"])


class TestFailOpen:
    """Tests that diagnostics fail open when database is unavailable."""

    def test_diagnostics_fail_open_on_db_error(self, db):
        """get_diagnostics() returns safe defaults when user_model.db is unavailable."""
        tracker = BehavioralAccuracyTracker(db)

        # Sabotage the db connection to simulate unavailability
        original_get = db.get_connection

        def broken_get(name):
            if name == "user_model":
                raise Exception("Database unavailable")
            return original_get(name)

        db.get_connection = broken_get

        result = tracker.get_diagnostics()

        # Should return a valid diagnostics dict with safe defaults
        assert result["per_type_stats"] == {}
        assert result["resolution_methods"] == {}
        assert result["unresolved_details"] == []
        assert result["health"] == "healthy"
        assert isinstance(result["inference_cycles"], dict)

        # Restore
        db.get_connection = original_get
