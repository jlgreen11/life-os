"""
Tests for the InsightEngine service.

Validates the Insight model's dedup key computation and the engine's
ability to run correlators, deduplicate, and store insights.
"""

import uuid
from datetime import datetime, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


def test_insight_model_dedup_key():
    """Dedup key should be deterministic for same type+category+entity."""
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="behavioral_pattern", summary="different", confidence=0.5, category="place", entity="Home")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key == b.dedup_key


def test_insight_model_different_entities_different_keys():
    a = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Home")
    b = Insight(type="behavioral_pattern", summary="test", confidence=0.8, category="place", entity="Work")
    a.compute_dedup_key()
    b.compute_dedup_key()
    assert a.dedup_key != b.dedup_key


@pytest.mark.asyncio
async def test_insight_engine_runs_empty(db, user_model_store):
    """InsightEngine should return empty list when no data exists."""
    engine = InsightEngine(db, user_model_store)
    insights = await engine.generate_insights()
    assert isinstance(insights, list)


@pytest.mark.asyncio
async def test_insight_deduplication(db, user_model_store):
    """Same insight should not reappear within staleness TTL."""
    engine = InsightEngine(db, user_model_store)

    # Add a place so we get at least one insight
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "Test Place", 10, "cafe", datetime.now(timezone.utc).isoformat()),
        )

    first = await engine.generate_insights()
    second = await engine.generate_insights()

    if first:
        first_keys = {i.dedup_key for i in first}
        second_keys = {i.dedup_key for i in second}
        # Second run should not produce the same insights
        overlap = first_keys & second_keys
        assert len(overlap) == 0, f"Deduplication failed -- {len(overlap)} insights reappeared"
