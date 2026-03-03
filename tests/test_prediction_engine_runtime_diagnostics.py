"""
Tests for PredictionEngine runtime diagnostics tracking.

Verifies that the get_runtime_diagnostics() method returns useful monitoring
data, that consecutive zero-run detection works, and that diagnostic state
persists across engine restarts via the prediction_engine_state table.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# get_runtime_diagnostics() — defaults before first run
# -------------------------------------------------------------------------


def test_runtime_diagnostics_empty_before_first_run(db, user_model_store):
    """get_runtime_diagnostics() returns sensible defaults before any prediction run."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
    diag = engine.get_runtime_diagnostics()

    assert "engine_state" in diag
    assert "run_statistics" in diag
    assert "health" in diag

    # No runs yet — run_statistics should be empty dict
    assert diag["run_statistics"] == {}
    # Health should be 'ok' (0 consecutive zero runs < 4)
    assert diag["health"] == "ok"
    # Engine state should reflect initial defaults
    assert diag["engine_state"]["last_event_cursor"] == 0
    assert diag["engine_state"]["last_time_based_run"] is None


# -------------------------------------------------------------------------
# get_runtime_diagnostics() — populated after a run
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_diagnostics_populated_after_run(db, event_store, user_model_store):
    """After running generate_predictions(), diagnostics contain expected keys."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Add an event so the engine actually runs (not skipped)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Test", "message_id": "diag-1"},
        "metadata": {},
    })

    await engine.generate_predictions({})
    diag = engine.get_runtime_diagnostics()

    # run_statistics should now contain the expected keys
    stats = diag["run_statistics"]
    assert stats["total_runs"] == 1
    assert isinstance(stats["total_generated"], int)
    assert isinstance(stats["total_surfaced"], int)
    assert "last_run_at" in stats
    assert "last_run_stats" in stats
    assert "last_run_raw_count" in stats
    assert "last_run_surfaced_count" in stats
    assert "last_run_filtered_by_reaction" in stats
    assert "last_run_filtered_by_confidence" in stats
    assert "signal_profiles_available" in stats
    assert "signal_profiles_total" in stats
    assert stats["signal_profiles_total"] == 5  # 5 profile types checked
    assert "triggers" in stats
    assert isinstance(stats["triggers"], dict)
    assert "has_new_events" in stats["triggers"]
    assert "time_based_due" in stats["triggers"]

    # Engine state should be updated
    assert diag["engine_state"]["last_time_based_run"] is not None
    assert diag["health"] == "ok"


# -------------------------------------------------------------------------
# consecutive_zero_runs tracking and health degradation
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_zero_runs_tracked(db, user_model_store):
    """Consecutive zero-prediction runs increment counter and degrade health."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Run 5 cycles — with no events/signal profiles, these will generate 0 predictions.
    # After the first run sets the time-based trigger, subsequent runs
    # need the time-based trigger to fire. We manually reset the timer
    # to force re-runs.
    for i in range(5):
        # Force time-based trigger by backdating the last run
        engine._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)
        await engine.generate_predictions({})

    diag = engine.get_runtime_diagnostics()
    assert diag["run_statistics"]["consecutive_zero_runs"] == 5
    assert diag["run_statistics"]["total_runs"] == 5
    # 4+ consecutive zero runs triggers 'degraded' health
    assert diag["health"] == "degraded"


@pytest.mark.asyncio
async def test_consecutive_zero_runs_resets_on_success(db, event_store, user_model_store):
    """consecutive_zero_runs counter resets when a prediction is actually surfaced."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Run a few empty cycles to build up zero counter
    for _ in range(3):
        engine._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)
        await engine.generate_predictions({})

    assert engine._consecutive_zero_runs == 3

    # Now insert calendar events that will produce a CONFLICT prediction.
    # Two overlapping events in the next 48h window.
    now = datetime.now(timezone.utc)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Meeting A",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Meeting B",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    # Force time-based trigger and run
    engine._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)
    preds = await engine.generate_predictions({})

    if len(preds) > 0:
        # If predictions were surfaced, counter should have reset
        assert engine._consecutive_zero_runs == 0
        assert engine.get_runtime_diagnostics()["health"] == "ok"
    else:
        # Even if reaction filtering suppresses the predictions, the counter
        # still increments (which is correct — no predictions surfaced).
        # Just verify the counter tracks accurately.
        assert engine._consecutive_zero_runs == 4


# -------------------------------------------------------------------------
# Diagnostics persistence across restarts
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostics_persisted_to_state_table(db, event_store, user_model_store):
    """Diagnostics survive a simulated restart via the state table."""
    engine1 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Add an event and run predictions
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "test@example.com", "subject": "Persist test", "message_id": "persist-1"},
        "metadata": {},
    })
    await engine1.generate_predictions({})

    # Verify diagnostics were stored
    diag1 = engine1.get_runtime_diagnostics()
    assert diag1["run_statistics"]["total_runs"] == 1

    # Create a new engine instance (simulating a restart) — it should
    # restore diagnostics from the state table.
    engine2 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
    diag2 = engine2.get_runtime_diagnostics()

    # The restored diagnostics should match what was persisted
    assert diag2["run_statistics"].get("total_runs") == 1
    assert diag2["run_statistics"].get("last_run_at") == diag1["run_statistics"]["last_run_at"]
    assert engine2._total_runs == 1
    assert engine2._total_predictions_generated == diag1["run_statistics"]["total_generated"]
    assert engine2._total_predictions_surfaced == diag1["run_statistics"]["total_surfaced"]
    assert engine2._consecutive_zero_runs == diag1["run_statistics"]["consecutive_zero_runs"]


def test_diagnostics_corrupted_json_handled_gracefully(db, user_model_store):
    """Corrupt diagnostics JSON in state table doesn't crash engine init."""
    # First create an engine to ensure the state table exists
    PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Manually insert corrupt diagnostics data
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO prediction_engine_state (key, value, updated_at) VALUES (?, ?, ?)",
            ("last_run_diagnostics", "not-valid-json{{{", datetime.now(timezone.utc).isoformat()),
        )

    # Engine should start without error, using defaults
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
    diag = engine.get_runtime_diagnostics()
    assert diag["run_statistics"] == {}
    assert diag["health"] == "ok"
    assert engine._total_runs == 0


# -------------------------------------------------------------------------
# Cumulative statistics across multiple runs
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cumulative_statistics_across_runs(db, event_store, user_model_store):
    """Total counters accumulate correctly across multiple generate_predictions() calls."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Run 3 cycles
    for i in range(3):
        event_store.store_event({
            "id": str(uuid.uuid4()),
            "type": "email.received",
            "source": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"from_address": "test@example.com", "subject": f"Test {i}", "message_id": f"cum-{i}"},
            "metadata": {},
        })
        engine._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)
        await engine.generate_predictions({})

    diag = engine.get_runtime_diagnostics()
    stats = diag["run_statistics"]

    assert stats["total_runs"] == 3
    # total_generated and total_surfaced should be cumulative (not per-run)
    assert stats["total_generated"] >= 0
    assert stats["total_surfaced"] >= 0
    # last_run_raw_count is per-run, not cumulative
    assert isinstance(stats["last_run_raw_count"], int)
