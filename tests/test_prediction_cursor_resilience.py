"""
Tests for PredictionEngine cursor resilience and degraded health alerting.

Verifies that:
- _has_new_events() updates in-memory cursor but does NOT persist to DB
- generate_predictions() persists cursor only after successful completion
- Cursor is NOT persisted if generate_predictions() raises mid-pipeline
- _consecutive_zero_runs == 4 triggers a degraded WARNING log (once)
- _consecutive_zero_runs == 5 does NOT re-trigger the alert
"""

import logging
import uuid
from datetime import datetime, timezone
from unittest import mock

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _insert_email_event(event_store, subject: str = "Test email") -> None:
    """Insert a simple email event so _has_new_events() returns True."""
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": "test@example.com",
            "subject": subject,
            "message_id": f"msg-{uuid.uuid4().hex[:8]}",
        },
        "metadata": {},
    })


def _read_persisted_cursor(db) -> int | None:
    """Read the persisted last_event_cursor from the DB, or None if absent."""
    try:
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value FROM prediction_engine_state WHERE key = 'last_event_cursor'"
            ).fetchone()
            return int(row["value"]) if row else None
    except Exception:
        return None


# -------------------------------------------------------------------------
# Test 1: _has_new_events does NOT persist cursor to DB
# -------------------------------------------------------------------------


def test_has_new_events_does_not_persist_cursor(db, event_store, user_model_store):
    """_has_new_events() should update the in-memory cursor but must NOT write it to the DB.

    This ensures that if generate_predictions() crashes afterward, the events
    will be reconsidered on the next cycle.
    """
    engine = PredictionEngine(db=db, ums=user_model_store)

    _insert_email_event(event_store)

    # Should detect the new event
    assert engine._has_new_events() is True
    assert engine._last_event_cursor > 0

    # But the DB should NOT have the cursor persisted yet
    persisted = _read_persisted_cursor(db)
    assert persisted is None or persisted == 0, (
        f"Expected cursor not yet persisted to DB, but found {persisted}"
    )


# -------------------------------------------------------------------------
# Test 2: generate_predictions persists cursor after successful completion
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_predictions_persists_cursor_on_success(db, event_store, user_model_store):
    """After generate_predictions() completes successfully, the cursor must be
    persisted to the DB so a new engine instance picks up where we left off.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    _insert_email_event(event_store)

    # Before running, DB cursor should be absent/zero
    assert _read_persisted_cursor(db) in (None, 0)

    # Run the full pipeline
    await engine.generate_predictions({})

    # After successful completion, cursor should be persisted
    persisted = _read_persisted_cursor(db)
    assert persisted is not None and persisted > 0, (
        f"Expected cursor persisted to DB after successful run, but found {persisted}"
    )
    assert persisted == engine._last_event_cursor


# -------------------------------------------------------------------------
# Test 3: Cursor NOT persisted if generate_predictions raises mid-pipeline
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_not_persisted_on_pipeline_crash(db, event_store, user_model_store):
    """If generate_predictions() raises an exception mid-pipeline, the cursor
    should NOT be persisted. Those events must be reconsidered on the next cycle.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    _insert_email_event(event_store)

    # Verify _has_new_events detects the event (this is called inside generate_predictions)
    # We need to verify state AFTER a crash, so we force a crash in the pipeline.

    # Patch the prediction storage step to raise an unhandled exception.
    # The per-prediction try/except won't save us if we blow up the diagnostics
    # persistence step. Instead, we patch _get_suppressed_prediction_keys which
    # is called after cursor advance but before cursor persistence.
    with mock.patch.object(
        engine, "_get_suppressed_prediction_keys",
        side_effect=RuntimeError("Simulated mid-pipeline crash"),
    ):
        with pytest.raises(RuntimeError, match="Simulated mid-pipeline crash"):
            await engine.generate_predictions({})

    # The in-memory cursor was advanced (by _has_new_events inside generate_predictions)
    assert engine._last_event_cursor > 0

    # But the DB should NOT have the new cursor persisted
    persisted = _read_persisted_cursor(db)
    assert persisted is None or persisted == 0, (
        f"Expected cursor NOT persisted after crash, but found {persisted}"
    )

    # A new engine instance should start fresh and reprocess those events
    engine2 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
    assert engine2._last_event_cursor == 0
    assert engine2._has_new_events() is True


# -------------------------------------------------------------------------
# Test 4: _consecutive_zero_runs == 4 triggers degraded warning (once)
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degraded_alert_fires_at_4_consecutive_zeros(db, event_store, user_model_store, caplog):
    """When _consecutive_zero_runs reaches exactly 4, a WARNING log is emitted.

    The log message should mention 'DEGRADED' and fire only once (at == 4,
    not >= 4) to avoid log spam.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Pre-set consecutive_zero_runs to 3 so the next zero run hits exactly 4
    engine._consecutive_zero_runs = 3

    # Insert an event so the pipeline triggers (has_new_events returns True)
    _insert_email_event(event_store)

    with caplog.at_level(logging.WARNING, logger="services.prediction_engine.engine"):
        # Run predictions — all checks will likely produce 0 surfaced predictions
        # since there's no data worth predicting on
        await engine.generate_predictions({})

    assert engine._consecutive_zero_runs == 4

    # Check that the degraded warning was logged
    degraded_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "DEGRADED" in r.message
    ]
    assert len(degraded_warnings) >= 1, (
        f"Expected a DEGRADED warning log at consecutive_zero_runs == 4. "
        f"All warnings: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
    )


# -------------------------------------------------------------------------
# Test 5: _consecutive_zero_runs == 5 does NOT re-trigger the alert
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degraded_alert_does_not_fire_at_5(db, event_store, user_model_store, caplog):
    """When _consecutive_zero_runs goes from 4 to 5, the degraded alert should
    NOT fire again. The == 4 check ensures it fires exactly once.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Pre-set consecutive_zero_runs to 4 so the next zero run hits 5
    engine._consecutive_zero_runs = 4

    # Insert an event so the pipeline triggers
    _insert_email_event(event_store)

    with caplog.at_level(logging.WARNING, logger="services.prediction_engine.engine"):
        await engine.generate_predictions({})

    assert engine._consecutive_zero_runs == 5

    # The DEGRADED warning should NOT appear for run 5
    degraded_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "DEGRADED" in r.message
    ]
    assert len(degraded_warnings) == 0, (
        f"Expected NO DEGRADED warning at consecutive_zero_runs == 5, "
        f"but got: {[r.message for r in degraded_warnings]}"
    )


# -------------------------------------------------------------------------
# Test 6: Consecutive zero counter resets on successful surfaced predictions
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_zeros_resets_on_surfaced_predictions(db, event_store, user_model_store):
    """When predictions are surfaced, _consecutive_zero_runs resets to 0."""
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # Pre-set a high consecutive zero count
    engine._consecutive_zero_runs = 10

    # Insert overlapping calendar events to force at least one conflict prediction
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": now.isoformat(),
        "payload": {
            "title": "Meeting A",
            "start_time": (now + timedelta(hours=3)).isoformat(),
            "end_time": (now + timedelta(hours=4)).isoformat(),
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
            "start_time": (now + timedelta(hours=3, minutes=15)).isoformat(),
            "end_time": (now + timedelta(hours=4, minutes=15)).isoformat(),
        },
        "metadata": {},
    })

    result = await engine.generate_predictions({})

    if len(result) > 0:
        # If predictions were surfaced, counter should reset
        assert engine._consecutive_zero_runs == 0
    else:
        # If no predictions surfaced (e.g., filtered by reaction), counter increments
        assert engine._consecutive_zero_runs == 11


# -------------------------------------------------------------------------
# Test 7: Cursor persisted correctly across multiple successful runs
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cursor_persisted_across_multiple_runs(db, event_store, user_model_store):
    """Multiple successful generate_predictions() calls should each persist the
    latest cursor, and a new engine instance should restore the final value.
    """
    engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

    # First run
    _insert_email_event(event_store, subject="Email 1")
    await engine.generate_predictions({})
    cursor_after_first = engine._last_event_cursor
    assert cursor_after_first > 0

    # Second run with more events
    _insert_email_event(event_store, subject="Email 2")
    await engine.generate_predictions({})
    cursor_after_second = engine._last_event_cursor
    assert cursor_after_second > cursor_after_first

    # A new engine should restore the latest cursor
    engine2 = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
    assert engine2._last_event_cursor == cursor_after_second

    # And should NOT see old events as new
    assert engine2._has_new_events() is False
