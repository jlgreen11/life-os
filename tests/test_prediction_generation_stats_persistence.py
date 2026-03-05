"""
Tests for PredictionEngine generation_stats persistence.

Verifies that the per-check generation breakdown (generation_stats) is
persisted to prediction_engine_state after each generate_predictions() run,
survives engine restarts, and appears in runtime diagnostics.  This enables
debugging of the 0-prediction anomaly by showing which _check_* methods
returned 0, errored, or were skipped.
"""

import json
from datetime import datetime, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Persistence after generate_predictions()
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_stats_persisted_after_run(db, event_store, user_model_store):
    """Run generate_predictions and verify generation_stats is saved to the DB."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    await engine.generate_predictions(current_context={})

    # Check in-memory state
    assert engine._last_generation_stats != {}
    assert engine._last_generation_timestamp is not None

    # Check persisted state in prediction_engine_state table
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT value FROM prediction_engine_state WHERE key = 'last_generation_stats'"
        ).fetchone()
    assert row is not None
    persisted = json.loads(row["value"])
    assert isinstance(persisted, dict)
    # Should have entries for the check methods (skipped or count)
    assert len(persisted) > 0


@pytest.mark.asyncio
async def test_generation_timestamp_persisted_after_run(db, event_store, user_model_store):
    """Verify last_generation_timestamp is persisted after each run."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    await engine.generate_predictions(current_context={})

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT value FROM prediction_engine_state WHERE key = 'last_generation_timestamp'"
        ).fetchone()
    assert row is not None
    # Should be a valid ISO timestamp
    ts = row["value"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None  # Should be UTC-aware


# -------------------------------------------------------------------------
# Diagnostics inclusion
# -------------------------------------------------------------------------


def test_generation_stats_in_diagnostics_fresh_engine(db, user_model_store):
    """A fresh engine should have null generation breakdown in diagnostics."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    diag = engine.get_runtime_diagnostics()
    assert "last_generation_breakdown" in diag
    assert diag["last_generation_breakdown"] is None
    assert "last_generation_timestamp" in diag
    assert diag["last_generation_timestamp"] is None


@pytest.mark.asyncio
async def test_generation_stats_in_diagnostics_after_run(db, event_store, user_model_store):
    """After generate_predictions, diagnostics should include generation breakdown."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    await engine.generate_predictions(current_context={})

    diag = engine.get_runtime_diagnostics()
    breakdown = diag["last_generation_breakdown"]
    assert breakdown is not None
    assert isinstance(breakdown, dict)

    # The breakdown should contain entries for the check methods
    # (either counts, error strings, or skipped markers)
    expected_keys = {
        "calendar_conflicts",
        "routine_deviations",
        "relationship_maintenance",
        "preparation_needs",
        "calendar_reminders",
        "connector_health",
    }
    # At least the time-based checks should be present (either run or skipped)
    assert expected_keys.issubset(set(breakdown.keys()))

    # Timestamp should also be present
    assert diag["last_generation_timestamp"] is not None


# -------------------------------------------------------------------------
# Survives restart
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_stats_survives_restart(db, event_store, user_model_store):
    """Persist stats, create new PredictionEngine instance, verify stats are loaded."""
    engine1 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    await engine1.generate_predictions(current_context={})

    saved_stats = engine1._last_generation_stats.copy()
    saved_timestamp = engine1._last_generation_timestamp
    assert saved_stats  # Should not be empty after a run
    assert saved_timestamp is not None

    # Create a brand-new engine pointing at the same DB
    engine2 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Should have loaded the persisted stats
    assert engine2._last_generation_stats == saved_stats
    assert engine2._last_generation_timestamp == saved_timestamp


# -------------------------------------------------------------------------
# Timestamp updates each run
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_timestamp_updated_each_run(db, event_store, user_model_store):
    """Verify last_generation_timestamp is updated after each generate_predictions run."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    await engine.generate_predictions(current_context={})
    ts1 = engine._last_generation_timestamp

    await engine.generate_predictions(current_context={})
    ts2 = engine._last_generation_timestamp

    assert ts1 is not None
    assert ts2 is not None
    # Second timestamp should be >= first (could be same if fast enough)
    assert ts2 >= ts1


# -------------------------------------------------------------------------
# Empty / null stats on fresh engine
# -------------------------------------------------------------------------


def test_empty_stats_on_fresh_engine(db, user_model_store):
    """New engine with no persisted state should have empty generation stats."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    assert engine._last_generation_stats == {}
    assert engine._last_generation_timestamp is None

    diag = engine.get_runtime_diagnostics()
    assert diag["last_generation_breakdown"] is None
    assert diag["last_generation_timestamp"] is None
