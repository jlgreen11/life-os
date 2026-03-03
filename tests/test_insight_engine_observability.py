"""
Tests for InsightEngine observability logging.

Validates that the insight pipeline produces diagnostic log messages when
insights are dropped by source-weight filtering or removed by deduplication,
so operators can trace why zero insights reach the user.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


@pytest.fixture()
def insight_engine_with_low_weight(db, user_model_store):
    """InsightEngine with a mock SourceWeightManager returning very low weights."""
    swm = MagicMock()
    swm.get_effective_weight.return_value = 0.01  # Weight that drops most insights
    swm.record_interaction = MagicMock()
    return InsightEngine(db, user_model_store, source_weight_manager=swm)


@pytest.fixture()
def insight_engine_with_high_weight(db, user_model_store):
    """InsightEngine with a mock SourceWeightManager returning full weights."""
    swm = MagicMock()
    swm.get_effective_weight.return_value = 1.0
    swm.record_interaction = MagicMock()
    return InsightEngine(db, user_model_store, source_weight_manager=swm)


@pytest.fixture()
def sample_insights():
    """A list of sample insights with mapped categories for source-weight filtering."""
    return [
        Insight(
            type="behavioral_pattern",
            summary="You visit Cafe X 5x/week",
            confidence=0.7,
            category="place",
            entity="Cafe X",
        ),
        Insight(
            type="relationship_intelligence",
            summary="Contact gap: haven't spoken to Bob in 30 days",
            confidence=0.6,
            category="contact_gap",
            entity="Bob",
        ),
        Insight(
            type="communication_style",
            summary="Your emails are getting longer",
            confidence=0.5,
            category="communication_style",
            entity=None,
        ),
    ]


def test_source_weight_filtering_logs_dropped_insights(
    insight_engine_with_low_weight, sample_insights, caplog
):
    """When insights are dropped by low source weights, debug-level log messages
    should record each drop and an info-level summary should report totals."""
    engine = insight_engine_with_low_weight

    with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
        result = engine._apply_source_weights(sample_insights)

    # All 3 insights have mapped categories and weight=0.01, so:
    #   0.7 * 0.01 = 0.007 < 0.1 → dropped
    #   0.6 * 0.01 = 0.006 < 0.1 → dropped
    #   0.5 * 0.01 = 0.005 < 0.1 → dropped
    assert len(result) == 0

    # Verify per-insight debug messages
    drop_messages = [r for r in caplog.records if "Insight dropped by source weight" in r.message]
    assert len(drop_messages) == 3

    # Verify the summary info message
    summary_messages = [r for r in caplog.records if "Source weight filtering:" in r.message]
    assert len(summary_messages) == 1
    assert "kept 0 insights, dropped 3" in summary_messages[0].message


def test_source_weight_filtering_keeps_high_confidence(
    insight_engine_with_high_weight, sample_insights, caplog
):
    """When source weights are high (1.0), no insights should be dropped
    and no drop log messages should appear."""
    engine = insight_engine_with_high_weight

    with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
        result = engine._apply_source_weights(sample_insights)

    # weight=1.0 → confidence unchanged → all above 0.1
    assert len(result) == 3

    # No drop messages
    drop_messages = [r for r in caplog.records if "Insight dropped by source weight" in r.message]
    assert len(drop_messages) == 0

    # No summary message (nothing was dropped)
    summary_messages = [r for r in caplog.records if "Source weight filtering:" in r.message]
    assert len(summary_messages) == 0


def test_source_weight_drop_log_includes_details(
    insight_engine_with_low_weight, caplog
):
    """Drop log messages should include the insight type, source key,
    original confidence, and weighted confidence."""
    engine = insight_engine_with_low_weight
    insights = [
        Insight(
            type="behavioral_pattern",
            summary="Test insight",
            confidence=0.8,
            category="place",
            entity="TestPlace",
        ),
    ]

    with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
        engine._apply_source_weights(insights)

    drop_messages = [r for r in caplog.records if "Insight dropped by source weight" in r.message]
    assert len(drop_messages) == 1
    msg = drop_messages[0].message
    assert "original_confidence=0.800" in msg
    assert "weighted_confidence=0.008" in msg
    assert "location.visits" in msg  # source_key for "place" category


@pytest.mark.asyncio
async def test_deduplication_logs_removed_count(db, user_model_store, caplog):
    """When deduplication removes insights, a debug message should report
    how many were removed out of the total."""
    swm = MagicMock()
    swm.get_effective_weight.return_value = 1.0
    swm.record_interaction = MagicMock()
    engine = InsightEngine(db, user_model_store, source_weight_manager=swm)

    # First call: all insights are fresh (no prior entries in DB)
    insights_a = [
        Insight(
            type="behavioral_pattern",
            summary="Pattern A",
            confidence=0.7,
            category="place",
            entity="Home",
        ),
        Insight(
            type="relationship_intelligence",
            summary="Pattern B",
            confidence=0.6,
            category="contact_gap",
            entity="Alice",
        ),
    ]

    # Patch the correlators to return our controlled insights
    with patch.object(engine, "_place_frequency_insights", return_value=insights_a[:1]):
        with patch.object(engine, "_contact_gap_insights", return_value=insights_a[1:]):
            # Suppress all other correlators
            for name in [
                "_relationship_intelligence_insights",
                "_email_volume_insights",
                "_communication_style_insights",
                "_inbound_style_insights",
                "_actionable_alert_insights",
                "_temporal_pattern_insights",
                "_mood_trend_insights",
                "_spending_pattern_insights",
                "_decision_pattern_insights",
                "_topic_interest_insights",
                "_cadence_response_insights",
                "_routine_insights",
                "_spatial_insights",
                "_workflow_pattern_insights",
            ]:
                setattr(engine, name, MagicMock(return_value=[]))

            # First generate: both insights are fresh → stored
            with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
                first_result = await engine.generate_insights()
            assert len(first_result) == 2

    # Reset the correlator cache so the second call actually runs correlators
    engine._last_insight_run = 0.0

    # Second call: same dedup_keys → should be deduplicated
    insights_b = [
        Insight(
            type="behavioral_pattern",
            summary="Pattern A again",
            confidence=0.7,
            category="place",
            entity="Home",
        ),
        Insight(
            type="relationship_intelligence",
            summary="Pattern B again",
            confidence=0.6,
            category="contact_gap",
            entity="Alice",
        ),
    ]

    caplog.clear()

    with patch.object(engine, "_place_frequency_insights", return_value=insights_b[:1]):
        with patch.object(engine, "_contact_gap_insights", return_value=insights_b[1:]):
            for name in [
                "_relationship_intelligence_insights",
                "_email_volume_insights",
                "_communication_style_insights",
                "_inbound_style_insights",
                "_actionable_alert_insights",
                "_temporal_pattern_insights",
                "_mood_trend_insights",
                "_spending_pattern_insights",
                "_decision_pattern_insights",
                "_topic_interest_insights",
                "_cadence_response_insights",
                "_routine_insights",
                "_spatial_insights",
                "_workflow_pattern_insights",
            ]:
                setattr(engine, name, MagicMock(return_value=[]))

            with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
                second_result = await engine.generate_insights()

    # Both should be deduplicated (same type+category+entity → same dedup_key)
    assert len(second_result) == 0

    dedup_messages = [r for r in caplog.records if "Deduplication removed" in r.message]
    assert len(dedup_messages) == 1
    assert "removed 2 of 2" in dedup_messages[0].message


def test_unmapped_category_not_dropped_by_source_weight(
    insight_engine_with_low_weight, caplog
):
    """Insights with categories not in the source-weight mapping should
    pass through unmodified (not dropped), since actionable alerts and
    unmapped categories bypass source-weight modulation."""
    engine = insight_engine_with_low_weight
    insights = [
        Insight(
            type="actionable_alert",
            summary="Task overdue: finish report",
            confidence=0.9,
            category="overdue_task",  # Not in category_to_source mapping
            entity=None,
        ),
    ]

    with caplog.at_level(logging.DEBUG, logger="services.insight_engine.engine"):
        result = engine._apply_source_weights(insights)

    # Unmapped category → confidence stays at 0.9 → kept
    assert len(result) == 1
    assert result[0].confidence == 0.9

    # No drop messages
    drop_messages = [r for r in caplog.records if "Insight dropped by source weight" in r.message]
    assert len(drop_messages) == 0
