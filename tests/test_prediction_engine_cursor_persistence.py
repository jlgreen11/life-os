"""
Tests for PredictionEngine state persistence across restarts.

Verifies that the event cursor and last-time-based-run timestamp survive
engine re-instantiation, preventing the post-restart dead zone where all
events look 'new' and predictions are drowned out by deduplication.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Event cursor persistence
# -------------------------------------------------------------------------


def test_cursor_persisted_after_has_new_events(db, event_store, user_model_store):
    """After _has_new_events() advances the cursor, a new engine instance restores it."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    # Insert an event so there's a non-zero MAX(rowid)
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "a@b.com", "subject": "hi", "message_id": "m1"},
        "metadata": {},
    })

    # _has_new_events should detect the new event and persist cursor
    assert engine._has_new_events() is True
    saved_cursor = engine._last_event_cursor
    assert saved_cursor > 0

    # Create a brand-new engine instance against the same DB — should restore the cursor
    engine2 = PredictionEngine(db=db, ums=user_model_store)
    assert engine2._last_event_cursor == saved_cursor

    # And it should NOT see the same events as new
    assert engine2._has_new_events() is False


def test_cursor_advances_on_subsequent_events(db, event_store, user_model_store):
    """Cursor should advance each time new events arrive and persist correctly."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    # First event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "a@b.com", "subject": "1", "message_id": "m1"},
        "metadata": {},
    })
    engine._has_new_events()
    cursor_after_first = engine._last_event_cursor

    # Second event
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "email.received",
        "source": "google",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"from_address": "a@b.com", "subject": "2", "message_id": "m2"},
        "metadata": {},
    })
    engine._has_new_events()
    cursor_after_second = engine._last_event_cursor

    assert cursor_after_second > cursor_after_first

    # New engine should see the latest cursor
    engine2 = PredictionEngine(db=db, ums=user_model_store)
    assert engine2._last_event_cursor == cursor_after_second


# -------------------------------------------------------------------------
# Time-based run timestamp persistence
# -------------------------------------------------------------------------


def test_time_based_run_persisted(db, user_model_store):
    """_should_run_time_based_predictions() persists its timestamp and a new engine restores it."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    # First call should run (last_time_based_run is None) and persist
    assert engine._should_run_time_based_predictions() is True
    saved_time = engine._last_time_based_run
    assert saved_time is not None

    # New engine should restore the timestamp
    engine2 = PredictionEngine(db=db, ums=user_model_store)
    assert engine2._last_time_based_run is not None
    # Allow 1-second tolerance for rounding
    assert abs((engine2._last_time_based_run - saved_time).total_seconds()) < 1

    # And the second engine should NOT immediately re-run time-based predictions
    assert engine2._should_run_time_based_predictions() is False


def test_time_based_run_respects_15min_interval_after_restore(db, user_model_store):
    """After restoring state, time-based predictions wait for the 15-min interval."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    # Force a last_time_based_run that's 20 minutes ago
    twenty_min_ago = datetime.now(timezone.utc) - timedelta(minutes=20)
    engine._last_time_based_run = twenty_min_ago
    engine._persist_state("last_time_based_run", twenty_min_ago.isoformat())

    # New engine restores the 20-min-ago timestamp
    engine2 = PredictionEngine(db=db, ums=user_model_store)
    assert engine2._last_time_based_run is not None

    # Should run because 20 min > 15 min interval
    assert engine2._should_run_time_based_predictions() is True


# -------------------------------------------------------------------------
# Fresh database behavior (no persisted state)
# -------------------------------------------------------------------------


def test_fresh_db_starts_with_defaults(db, user_model_store):
    """With no persisted state, engine starts with cursor=0 and last_run=None."""
    engine = PredictionEngine(db=db, ums=user_model_store)

    assert engine._last_event_cursor == 0
    assert engine._last_time_based_run is None


# -------------------------------------------------------------------------
# State table auto-creation
# -------------------------------------------------------------------------


def test_state_table_created_automatically(db, user_model_store):
    """The prediction_engine_state table should be created if it doesn't exist."""
    # Verify no table exists before engine creation
    with db.get_connection("user_model") as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_engine_state'"
        ).fetchall()
    # Table might or might not exist depending on schema initialization,
    # but after creating engine it MUST exist
    engine = PredictionEngine(db=db, ums=user_model_store)

    with db.get_connection("user_model") as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_engine_state'"
        ).fetchall()
    assert len(tables) == 1
    assert tables[0]["name"] == "prediction_engine_state"


def test_state_table_creation_is_idempotent(db, user_model_store):
    """Creating multiple engines against the same DB should not fail."""
    engine1 = PredictionEngine(db=db, ums=user_model_store)
    engine2 = PredictionEngine(db=db, ums=user_model_store)
    engine3 = PredictionEngine(db=db, ums=user_model_store)

    # All should work fine — table creation is IF NOT EXISTS
    assert engine3._last_event_cursor == 0
