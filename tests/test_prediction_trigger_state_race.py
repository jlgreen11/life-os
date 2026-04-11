"""
Tests for the prediction engine time-based trigger state persistence race fix.

Root cause (March 6 – April 11 stall):
    _should_run_time_based_predictions() was persisting last_time_based_run to
    the DB BEFORE generate_predictions() completed.  If an exception occurred
    anywhere between the trigger check and the storage loop, the DB timestamp
    was updated but no predictions were stored.  On the next restart the engine
    loaded the now-current timestamp, decided time_based_due=False, and
    skipped time-based predictions for 15 minutes — generating 0 predictions
    indefinitely.

Fix:
    1. Remove _persist_state() calls from _should_run_time_based_predictions()
       (in-memory update only).
    2. Persist last_time_based_run at the END of generate_predictions() after
       all predictions have been stored.
    3. Persist last_successful_generation when stored_count > 0 so the stale
       recovery check has a reference point.
    4. Stale recovery check: if last_successful_generation is >2h old and
       time_based_due would otherwise be False, force it True.
    5. reset_state() clears last_time_based_run and last_successful_generation
       from the DB so stale values are not loaded after a DB rebuild.

Test strategy:
    - Real SQLite via the db / event_store / user_model_store conftest fixtures.
    - No mocking of the storage layer (project convention).
    - Mocking limited to _should_run_time_based_predictions and individual
      _check_* methods where needed to isolate the race condition logic.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from services.prediction_engine.engine import PredictionEngine
from storage.database import UserModelStore


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prediction_engine(db, user_model_store):
    """PredictionEngine wired to a temporary DatabaseManager."""
    return PredictionEngine(db, user_model_store, timezone="UTC")


def _read_state_key(db, key: str) -> str | None:
    """Read a single key from prediction_engine_state; return None if absent."""
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT value FROM prediction_engine_state WHERE key = ?",
            (key,),
        ).fetchone()
    return row["value"] if row else None


def _write_state_key(db, key: str, value: str) -> None:
    """Insert or replace a key in prediction_engine_state."""
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT OR REPLACE INTO prediction_engine_state (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (key, value, now),
        )


# ---------------------------------------------------------------------------
# Test 1: _should_run_time_based_predictions does NOT call _persist_state
# ---------------------------------------------------------------------------


def test_should_run_time_based_does_not_persist_first_run(prediction_engine, db):
    """On the very first call (last_run=None), _should_run_time_based_predictions()
    must return True without persisting anything to the DB.

    The DB write must happen only AFTER generate_predictions() completes so that
    a crash between the trigger check and the storage loop doesn't 'use up' the
    trigger timestamp without storing any predictions.
    """
    pe = prediction_engine
    pe._last_time_based_run = None  # Simulate first-ever run

    # Clear any existing state that __init__ may have written
    with db.get_connection("user_model") as conn:
        conn.execute("DELETE FROM prediction_engine_state WHERE key = 'last_time_based_run'")

    result = pe._should_run_time_based_predictions()

    assert result is True
    # Nothing should have been written to the DB
    assert _read_state_key(db, "last_time_based_run") is None, (
        "_should_run_time_based_predictions() must NOT persist to DB "
        "(state is written at the end of generate_predictions() instead)"
    )


def test_should_run_time_based_does_not_persist_interval_run(prediction_engine, db):
    """When 15+ minutes have passed, _should_run_time_based_predictions() must
    return True without persisting the new timestamp to the DB."""
    pe = prediction_engine
    # Set last run to 20 minutes ago so the interval check fires
    pe._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)

    # Seed an old value in the DB
    old_ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    _write_state_key(db, "last_time_based_run", old_ts)

    result = pe._should_run_time_based_predictions()

    assert result is True
    # The DB value must still be the OLD timestamp (not the new in-memory one)
    db_val = _read_state_key(db, "last_time_based_run")
    assert db_val == old_ts, (
        "_should_run_time_based_predictions() must not overwrite the DB timestamp; "
        "the new timestamp is written only after generate_predictions() completes"
    )


def test_should_run_time_based_no_persist_via_mock(prediction_engine):
    """Verify via mock that _persist_state is never called inside
    _should_run_time_based_predictions(), regardless of which branch executes."""
    pe = prediction_engine

    # Branch 1: first run
    pe._last_time_based_run = None
    with patch.object(pe, "_persist_state") as mock_persist:
        pe._should_run_time_based_predictions()
    mock_persist.assert_not_called()

    # Branch 2: interval elapsed
    pe._last_time_based_run = datetime.now(timezone.utc) - timedelta(minutes=20)
    with patch.object(pe, "_persist_state") as mock_persist:
        pe._should_run_time_based_predictions()
    mock_persist.assert_not_called()

    # Branch 3: interval NOT elapsed (returns False)
    pe._last_time_based_run = datetime.now(timezone.utc)
    with patch.object(pe, "_persist_state") as mock_persist:
        pe._should_run_time_based_predictions()
    mock_persist.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: last_time_based_run IS persisted after generate_predictions completes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_time_based_run_persisted_after_generate_predictions(prediction_engine, db):
    """After generate_predictions() completes successfully with time_based_due=True,
    the last_time_based_run timestamp must be written to the DB."""
    pe = prediction_engine
    pe._last_time_based_run = None  # Ensure time trigger fires

    # Confirm nothing is in the DB yet
    with db.get_connection("user_model") as conn:
        conn.execute("DELETE FROM prediction_engine_state WHERE key = 'last_time_based_run'")

    await pe.generate_predictions({})

    persisted = _read_state_key(db, "last_time_based_run")
    assert persisted is not None, (
        "last_time_based_run must be persisted to the DB at the end of "
        "generate_predictions() so that restarts load the correct timestamp"
    )
    # The persisted value must be a parseable datetime
    datetime.fromisoformat(persisted)


@pytest.mark.asyncio
async def test_last_time_based_run_not_persisted_when_skipped(prediction_engine, db):
    """When neither trigger fires (pipeline is skipped), last_time_based_run
    must NOT be written to the DB."""
    pe = prediction_engine

    # Force both triggers to False
    pe._last_time_based_run = datetime.now(timezone.utc)  # within 15-min window
    pe._last_event_cursor = 999_999  # ahead of any real events

    # Clear state table
    with db.get_connection("user_model") as conn:
        conn.execute("DELETE FROM prediction_engine_state WHERE key = 'last_time_based_run'")

    result = await pe.generate_predictions({})

    assert result == []
    assert _read_state_key(db, "last_time_based_run") is None, (
        "last_time_based_run must NOT be persisted when generate_predictions() "
        "skips (neither trigger was active)"
    )


# ---------------------------------------------------------------------------
# Test 3: last_successful_generation is only persisted when predictions stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_successful_generation_not_written_when_zero_stored(prediction_engine, db):
    """If generate_predictions() runs but stores 0 predictions (e.g., empty DB,
    all pre-filtered), last_successful_generation must NOT be written.

    This ensures the stale recovery check still fires on the next restart
    when the engine has been generating 0 results for hours.
    """
    pe = prediction_engine
    pe._last_time_based_run = None  # Force time trigger

    # Make sure last_successful_generation is not already in DB
    with db.get_connection("user_model") as conn:
        conn.execute(
            "DELETE FROM prediction_engine_state WHERE key = 'last_successful_generation'"
        )

    # With an empty DB, all _check_* methods return [] → stored_count = 0
    await pe.generate_predictions({})

    assert _read_state_key(db, "last_successful_generation") is None, (
        "last_successful_generation must NOT be persisted when stored_count==0; "
        "it is only written when predictions are actually stored"
    )


# ---------------------------------------------------------------------------
# Test 4: Stale recovery forces time_based_due=True when last_successful_generation is old
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_recovery_forces_time_based_due_when_no_recent_success(prediction_engine, db):
    """If last_successful_generation is >2h old and time_based_due would normally
    be False, the engine must force time_based_due=True to break the stall.

    Scenario: engine crashed after persisting last_time_based_run but before
    storing predictions.  On the next cycle (no restart, in-memory state is
    'recent'), _should_run_time_based_predictions() returns False but the DB
    shows no successful generation in the last 2 hours.
    """
    pe = prediction_engine

    # Set _last_time_based_run to "just now" so the normal 15-min check returns False
    pe._last_time_based_run = datetime.now(timezone.utc)

    # Place a stale last_successful_generation (3 hours ago) in the DB
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    _write_state_key(db, "last_successful_generation", stale_ts)

    # Track whether time-based prediction methods were called (proves time_based_due=True)
    calendar_check_called = []

    async def mock_calendar_check(_ctx):
        calendar_check_called.append(True)
        return []

    with patch.object(pe, "_check_calendar_conflicts", side_effect=mock_calendar_check):
        await pe.generate_predictions({})

    assert len(calendar_check_called) > 0, (
        "Stale recovery should have forced time_based_due=True and triggered "
        "_check_calendar_conflicts() even though _last_time_based_run was 'just now'"
    )


@pytest.mark.asyncio
async def test_stale_recovery_does_not_trigger_when_recent_success(prediction_engine, db):
    """If last_successful_generation is recent (<2h), no stale recovery happens
    and time_based_due remains False when the interval hasn't elapsed.
    """
    pe = prediction_engine

    # Set _last_time_based_run to "just now" so the 15-min check returns False
    pe._last_time_based_run = datetime.now(timezone.utc)
    # No new events either
    pe._last_event_cursor = 999_999

    # Place a RECENT last_successful_generation (30 min ago) in the DB
    recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    _write_state_key(db, "last_successful_generation", recent_ts)

    calendar_check_called = []

    async def mock_calendar_check(_ctx):
        calendar_check_called.append(True)
        return []

    with patch.object(pe, "_check_calendar_conflicts", side_effect=mock_calendar_check):
        result = await pe.generate_predictions({})

    # Pipeline should be skipped (no stale recovery, no new events, time not due)
    assert result == []
    assert len(calendar_check_called) == 0, (
        "Stale recovery must NOT fire when last_successful_generation is <2h old"
    )


# ---------------------------------------------------------------------------
# Test 5: reset_state() clears last_time_based_run from the DB
# ---------------------------------------------------------------------------


def test_reset_state_clears_last_time_based_run_from_db(prediction_engine, db):
    """reset_state() must delete last_time_based_run from prediction_engine_state
    so that a subsequent _load_persisted_state() call (after a DB rebuild) does
    not reload a stale timestamp that would prevent time-based predictions from firing.
    """
    pe = prediction_engine

    # Seed both keys in the DB
    now_str = datetime.now(timezone.utc).isoformat()
    _write_state_key(db, "last_time_based_run", now_str)
    _write_state_key(db, "last_successful_generation", now_str)

    pe.reset_state()

    assert _read_state_key(db, "last_time_based_run") is None, (
        "reset_state() must delete last_time_based_run from the DB"
    )
    assert _read_state_key(db, "last_successful_generation") is None, (
        "reset_state() must delete last_successful_generation from the DB"
    )


def test_reset_state_clears_in_memory_last_time_based_run(prediction_engine):
    """reset_state() must also clear the in-memory _last_time_based_run so that
    the NEXT call to _should_run_time_based_predictions() fires immediately
    (returns True on first-run path, not skipped by a stale in-memory value).
    """
    pe = prediction_engine
    pe._last_time_based_run = datetime.now(timezone.utc)

    pe.reset_state()

    assert pe._last_time_based_run is None


def test_reset_state_clears_db_even_when_table_has_other_keys(prediction_engine, db):
    """reset_state() must only delete the trigger-state keys, not other state entries."""
    pe = prediction_engine

    now_str = datetime.now(timezone.utc).isoformat()
    _write_state_key(db, "last_time_based_run", now_str)
    _write_state_key(db, "last_successful_generation", now_str)
    # Seed another key that should survive
    _write_state_key(db, "last_event_cursor", "42")

    pe.reset_state()

    # Trigger-state keys cleared
    assert _read_state_key(db, "last_time_based_run") is None
    assert _read_state_key(db, "last_successful_generation") is None
    # Other keys preserved
    assert _read_state_key(db, "last_event_cursor") == "42"
