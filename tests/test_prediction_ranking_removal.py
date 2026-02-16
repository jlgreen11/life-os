"""
Test suite verifying removal of arbitrary top-5 ranking cutoff.

Before fix: Prediction engine capped surfaced predictions at 5 per cycle, causing
6.6% of legitimate predictions to be silently discarded purely due to ranking.

After fix: All predictions that pass reaction gating and confidence threshold
surface. Ranking determines priority order, not existence.

Test coverage:
- 10+ predictions with confidence ≥ 0.3 all surface (no top-5 cap)
- Predictions are sorted by confidence (highest first)
- No predictions filtered with "ranking:position_X" filter_reason
- Reaction gating still works (filters "annoying" predictions)
- Confidence gating still works (filters predictions < 0.3)
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.prediction_engine.engine import PredictionEngine
from storage.database import DatabaseManager, UserModelStore


@pytest.fixture
def mock_engine(db, user_model_store):
    """Create a PredictionEngine with mocked prediction methods."""
    engine = PredictionEngine(db, user_model_store)
    return engine


@pytest.mark.asyncio
async def test_no_top_5_cap_all_predictions_surface(mock_engine, db, user_model_store):
    """
    Test that all predictions passing reaction + confidence gates surface.

    Before fix: Only top 5 surfaced, rest marked with ranking:position_X filter.
    After fix: All 10 predictions surface.
    """
    # Create 10 high-confidence predictions (all above 0.3 threshold)
    predictions = []
    for i in range(10):
        pred = Prediction(
            id=f"pred-{i}",
            prediction_type="reminder",
            description=f"Test prediction {i}",
            confidence=0.7 + (i * 0.01),  # 0.7 to 0.79
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="1d",
            suggested_action=f"Action {i}",
            supporting_signals={"signal1": "value1", "signal2": "value2"},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        )
        predictions.append(pred)

    # Mock prediction generation methods to return our test predictions
    mock_engine._check_follow_up_needs = AsyncMock(return_value=predictions)
    mock_engine._check_calendar_conflicts = AsyncMock(return_value=[])
    mock_engine._check_routine_deviations = AsyncMock(return_value=[])
    mock_engine._check_relationship_maintenance = AsyncMock(return_value=[])
    mock_engine._check_preparation_needs = AsyncMock(return_value=[])
    mock_engine._check_spending_patterns = AsyncMock(return_value=[])

    # Mock accuracy multiplier (no adjustment)
    mock_engine._get_accuracy_multiplier = MagicMock(return_value=1.0)

    # Mock reaction prediction to approve all predictions
    mock_engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test action",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test approval"
        )
    )

    # Force new events to trigger event-based predictions
    mock_engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-1", "email.received", "test", datetime.now(timezone.utc).isoformat(), "medium", "{}")
        )

    # Generate predictions
    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await mock_engine.generate_predictions(context)

    # VERIFY: All 10 predictions should surface (no top-5 cap)
    assert len(surfaced) == 10, f"Expected 10 surfaced predictions, got {len(surfaced)}"

    # VERIFY: No predictions filtered with "ranking:" filter_reason
    with db.get_connection("user_model") as conn:
        ranking_filtered = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE filter_reason LIKE 'ranking:%'"
        ).fetchone()
        assert ranking_filtered["count"] == 0, "No predictions should be filtered by ranking"

    # VERIFY: All predictions marked as surfaced
    with db.get_connection("user_model") as conn:
        surfaced_count = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE was_surfaced = 1"
        ).fetchone()
        assert surfaced_count["count"] == 10, "All 10 predictions should be marked as surfaced"


@pytest.mark.asyncio
async def test_predictions_sorted_by_confidence(mock_engine, db, user_model_store):
    """
    Test that surfaced predictions are sorted by confidence (highest first).

    Ranking should determine priority order for delivery, not filter predictions.
    """
    # Create predictions with varying confidence (all above 0.3)
    predictions = [
        Prediction(
            id="pred-low",
            prediction_type="reminder",
            description="Low confidence",
            confidence=0.35,
            confidence_gate=ConfidenceGate.SUGGEST,
            time_horizon="1d",
            suggested_action="Action",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ),
        Prediction(
            id="pred-high",
            prediction_type="reminder",
            description="High confidence",
            confidence=0.85,
            confidence_gate=ConfidenceGate.AUTONOMOUS,
            time_horizon="1d",
            suggested_action="Action",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ),
        Prediction(
            id="pred-med",
            prediction_type="reminder",
            description="Medium confidence",
            confidence=0.65,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="1d",
            suggested_action="Action",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ),
    ]

    # Mock prediction generation
    mock_engine._check_follow_up_needs = AsyncMock(return_value=predictions)
    mock_engine._check_calendar_conflicts = AsyncMock(return_value=[])
    mock_engine._check_routine_deviations = AsyncMock(return_value=[])
    mock_engine._check_relationship_maintenance = AsyncMock(return_value=[])
    mock_engine._check_preparation_needs = AsyncMock(return_value=[])
    mock_engine._check_spending_patterns = AsyncMock(return_value=[])
    mock_engine._get_accuracy_multiplier = MagicMock(return_value=1.0)
    mock_engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test action",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test"
        )
    )

    # Force new events
    mock_engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-2", "email.received", "test", datetime.now(timezone.utc).isoformat(), "medium", "{}")
        )

    # Generate predictions
    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await mock_engine.generate_predictions(context)

    # VERIFY: All 3 surface
    assert len(surfaced) == 3, f"Expected 3 surfaced, got {len(surfaced)}"

    # VERIFY: Sorted by confidence (highest first)
    assert surfaced[0].id == "pred-high", "Highest confidence should be first"
    assert surfaced[1].id == "pred-med", "Medium confidence should be second"
    assert surfaced[2].id == "pred-low", "Lowest confidence should be third"

    # VERIFY: Order matches confidence values
    assert surfaced[0].confidence == 0.85
    assert surfaced[1].confidence == 0.65
    assert surfaced[2].confidence == 0.35


@pytest.mark.asyncio
async def test_reaction_gating_still_filters(mock_engine, db, user_model_store):
    """
    Test that reaction gating still works (filters annoying predictions).

    Only the ranking cap was removed. Reaction and confidence gates remain active.
    """
    predictions = []
    for i in range(5):
        predictions.append(Prediction(
            id=f"pred-{i}",
            prediction_type="reminder",
            description=f"Test {i}",
            confidence=0.7,
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="1d",
            suggested_action=f"Action {i}",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ))

    mock_engine._check_follow_up_needs = AsyncMock(return_value=predictions)
    mock_engine._check_calendar_conflicts = AsyncMock(return_value=[])
    mock_engine._check_routine_deviations = AsyncMock(return_value=[])
    mock_engine._check_relationship_maintenance = AsyncMock(return_value=[])
    mock_engine._check_preparation_needs = AsyncMock(return_value=[])
    mock_engine._check_spending_patterns = AsyncMock(return_value=[])
    mock_engine._get_accuracy_multiplier = MagicMock(return_value=1.0)

    # Mock reaction to filter first 2 predictions (annoying), approve rest
    call_count = 0
    async def mock_reaction(pred, ctx):
        nonlocal call_count
        if call_count < 2:
            call_count += 1
            return ReactionPrediction(
                proposed_action="Test action",
                predicted_reaction="annoying",
                confidence=0.8,
                reasoning="User is stressed"
            )
        call_count += 1
        return ReactionPrediction(
            proposed_action="Test action",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Good timing"
        )

    mock_engine.predict_reaction = mock_reaction

    # Force new events
    mock_engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-3", "email.received", "test", datetime.now(timezone.utc).isoformat(), "medium", "{}")
        )

    # Generate predictions
    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await mock_engine.generate_predictions(context)

    # VERIFY: Only 3 surfaced (2 filtered by reaction)
    assert len(surfaced) == 3, f"Expected 3 surfaced (2 filtered by reaction), got {len(surfaced)}"

    # VERIFY: Filtered predictions have reaction filter_reason
    with db.get_connection("user_model") as conn:
        reaction_filtered = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE filter_reason LIKE 'reaction:%'"
        ).fetchone()
        assert reaction_filtered["count"] == 2, "2 predictions should be filtered by reaction"


@pytest.mark.asyncio
async def test_confidence_gating_still_filters(mock_engine, db, user_model_store):
    """
    Test that confidence gating still works (filters predictions < 0.3).

    Only the ranking cap was removed. Confidence threshold remains at 0.3.
    """
    predictions = [
        Prediction(
            id="pred-below",
            prediction_type="reminder",
            description="Below threshold",
            confidence=0.25,  # Below 0.3 threshold
            confidence_gate=ConfidenceGate.OBSERVE,
            time_horizon="1d",
            suggested_action="Action",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ),
        Prediction(
            id="pred-above",
            prediction_type="reminder",
            description="Above threshold",
            confidence=0.7,  # Above 0.3 threshold
            confidence_gate=ConfidenceGate.DEFAULT,
            time_horizon="1d",
            suggested_action="Action",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ),
    ]

    mock_engine._check_follow_up_needs = AsyncMock(return_value=predictions)
    mock_engine._check_calendar_conflicts = AsyncMock(return_value=[])
    mock_engine._check_routine_deviations = AsyncMock(return_value=[])
    mock_engine._check_relationship_maintenance = AsyncMock(return_value=[])
    mock_engine._check_preparation_needs = AsyncMock(return_value=[])
    mock_engine._check_spending_patterns = AsyncMock(return_value=[])
    mock_engine._get_accuracy_multiplier = MagicMock(return_value=1.0)
    mock_engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test action",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test"
        )
    )

    # Force new events
    mock_engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-4", "email.received", "test", datetime.now(timezone.utc).isoformat(), "medium", "{}")
        )

    # Generate predictions
    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await mock_engine.generate_predictions(context)

    # VERIFY: Only 1 surfaced (1 filtered by confidence)
    assert len(surfaced) == 1, f"Expected 1 surfaced (1 filtered by confidence), got {len(surfaced)}"
    assert surfaced[0].id == "pred-above", "Only above-threshold prediction should surface"

    # VERIFY: Filtered prediction has confidence filter_reason
    with db.get_connection("user_model") as conn:
        confidence_filtered = conn.execute(
            "SELECT filter_reason FROM predictions WHERE id = 'pred-below'"
        ).fetchone()
        assert confidence_filtered is not None
        assert confidence_filtered["filter_reason"].startswith("confidence:"), \
            f"Expected confidence filter_reason, got {confidence_filtered['filter_reason']}"


@pytest.mark.asyncio
async def test_high_volume_predictions_all_surface(mock_engine, db, user_model_store):
    """
    Test that even with 50+ predictions, all that pass gates surface.

    This is the key improvement: no arbitrary cutoff means all legitimate
    predictions surface. The notification manager handles batching.
    """
    # Create 50 predictions with confidence ≥ 0.3
    predictions = []
    for i in range(50):
        predictions.append(Prediction(
            id=f"pred-{i}",
            prediction_type="reminder",
            description=f"Test prediction {i}",
            confidence=0.3 + (i * 0.01),  # 0.3 to 0.79
            confidence_gate=ConfidenceGate.SUGGEST,
            time_horizon="1d",
            suggested_action=f"Action {i}",
            supporting_signals={},
            was_surfaced=False,
            user_response=None,
            was_accurate=None,
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            filter_reason=None
        ))

    mock_engine._check_follow_up_needs = AsyncMock(return_value=predictions)
    mock_engine._check_calendar_conflicts = AsyncMock(return_value=[])
    mock_engine._check_routine_deviations = AsyncMock(return_value=[])
    mock_engine._check_relationship_maintenance = AsyncMock(return_value=[])
    mock_engine._check_preparation_needs = AsyncMock(return_value=[])
    mock_engine._check_spending_patterns = AsyncMock(return_value=[])
    mock_engine._get_accuracy_multiplier = MagicMock(return_value=1.0)
    mock_engine.predict_reaction = AsyncMock(
        return_value=ReactionPrediction(
            proposed_action="Test action",
            predicted_reaction="helpful",
            confidence=0.8,
            reasoning="Test"
        )
    )

    # Force new events
    mock_engine._last_event_cursor = 0
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("evt-5", "email.received", "test", datetime.now(timezone.utc).isoformat(), "medium", "{}")
        )

    # Generate predictions
    context = {"current_time": datetime.now(timezone.utc)}
    surfaced = await mock_engine.generate_predictions(context)

    # VERIFY: All 50 predictions surface (no arbitrary cap)
    assert len(surfaced) == 50, f"Expected all 50 predictions to surface, got {len(surfaced)}"

    # VERIFY: No ranking filter reasons exist
    with db.get_connection("user_model") as conn:
        ranking_filtered = conn.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE filter_reason LIKE 'ranking:%'"
        ).fetchone()
        assert ranking_filtered["count"] == 0, "No predictions should be filtered by ranking"

    # VERIFY: Sorted by confidence (descending)
    confidences = [p.confidence for p in surfaced]
    assert confidences == sorted(confidences, reverse=True), \
        "Predictions should be sorted by confidence (highest first)"
