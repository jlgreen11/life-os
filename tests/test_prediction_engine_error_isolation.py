"""
Tests for per-method error isolation in PredictionEngine.generate_predictions().

Verifies that when individual _check_* methods throw exceptions (e.g. from
a corrupted user_model.db), the remaining checks still execute and return
whatever predictions they can produce.  This is the fail-open pattern
required by the codebase convention.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_engine_with_triggers(db, user_model_store):
    """Create a PredictionEngine and ensure both trigger conditions are met.

    Sets the engine state so that generate_predictions() will run all
    _check methods (both time-based and event-based triggers active).
    """
    engine = PredictionEngine(db=db, ums=user_model_store)
    # Force both triggers active: no last run (time-based) and cursor at 0 (event-based)
    engine._last_time_based_run = None
    engine._last_event_cursor = 0
    return engine


def _insert_event(event_store, event_type="email.received", hours_ago=6, **payload_extra):
    """Insert a test event into the event store."""
    now = datetime.now(timezone.utc)
    payload = {
        "from_address": "test@example.com",
        "subject": "Test",
        "message_id": f"msg-{uuid.uuid4().hex[:8]}",
        **payload_extra,
    }
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "test",
        "timestamp": (now - timedelta(hours=hours_ago)).isoformat(),
        "payload": payload,
        "metadata": {},
    })


# -------------------------------------------------------------------------
# Test 1: Calendar check succeeds when follow-up fails
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calendar_check_succeeds_when_followup_fails(db, event_store, user_model_store):
    """When _check_follow_up_needs raises a DatabaseError, calendar conflict
    predictions should still be generated and returned."""
    engine = _make_engine_with_triggers(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert overlapping calendar events to trigger a calendar conflict prediction
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
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
        "timestamp": (now + timedelta(hours=2, minutes=30)).isoformat(),
        "payload": {
            "title": "Meeting B",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    # Make _check_follow_up_needs raise a DatabaseError (simulates corrupted DB)
    with patch.object(
        engine, "_check_follow_up_needs",
        new_callable=AsyncMock,
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        predictions = await engine.generate_predictions({})

    # Calendar conflict predictions should still be returned
    calendar_preds = [p for p in predictions if p.prediction_type == "conflict"]
    assert len(calendar_preds) >= 1, (
        "Calendar conflict predictions should still be generated even when follow-up check fails"
    )


# -------------------------------------------------------------------------
# Test 2: All checks run independently
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_checks_run_independently(db, event_store, user_model_store):
    """When one _check method raises, all other _check methods still execute.

    Verified by inspecting generation_stats for entries from every method.
    """
    engine = _make_engine_with_triggers(db, user_model_store)

    # Insert an event so has_new_events is True (spending_patterns runs)
    _insert_event(event_store)

    # Make _check_routine_deviations crash
    with patch.object(
        engine, "_check_routine_deviations",
        new_callable=AsyncMock,
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        predictions = await engine.generate_predictions({})

    # The method doesn't directly return generation_stats, but we can verify
    # the other checks ran by inspecting log output. Instead, let's verify
    # all other methods were called by patching them all with spies.

    # Re-run with all methods as spies, one crashing
    engine2 = _make_engine_with_triggers(db, user_model_store)
    _insert_event(event_store)

    original_calendar = engine2._check_calendar_conflicts
    original_relationship = engine2._check_relationship_maintenance
    original_prep = engine2._check_preparation_needs
    original_followup = engine2._check_follow_up_needs
    original_spending = engine2._check_spending_patterns

    call_tracker = {}

    async def track_calendar(ctx):
        call_tracker['calendar_conflicts'] = True
        return await original_calendar(ctx)

    async def track_relationship(ctx):
        call_tracker['relationship_maintenance'] = True
        return await original_relationship(ctx)

    async def track_prep(ctx):
        call_tracker['preparation_needs'] = True
        return await original_prep(ctx)

    async def track_followup(ctx):
        call_tracker['follow_up_needs'] = True
        return await original_followup(ctx)

    async def track_spending(ctx):
        call_tracker['spending_patterns'] = True
        return await original_spending(ctx)

    with (
        patch.object(engine2, "_check_calendar_conflicts", side_effect=track_calendar),
        patch.object(
            engine2, "_check_routine_deviations",
            new_callable=AsyncMock,
            side_effect=RuntimeError("simulated crash"),
        ),
        patch.object(engine2, "_check_relationship_maintenance", side_effect=track_relationship),
        patch.object(engine2, "_check_preparation_needs", side_effect=track_prep),
        patch.object(engine2, "_check_follow_up_needs", side_effect=track_followup),
        patch.object(engine2, "_check_spending_patterns", side_effect=track_spending),
    ):
        await engine2.generate_predictions({})

    # All methods except the crashing one should have been called
    assert 'calendar_conflicts' in call_tracker, "calendar_conflicts should have run"
    assert 'relationship_maintenance' in call_tracker, "relationship_maintenance should have run"
    assert 'preparation_needs' in call_tracker, "preparation_needs should have run"
    assert 'follow_up_needs' in call_tracker, "follow_up_needs should have run"
    assert 'spending_patterns' in call_tracker, "spending_patterns should have run"


# -------------------------------------------------------------------------
# Test 3: store_prediction failure doesn't crash
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_prediction_failure_doesnt_crash(db, event_store, user_model_store):
    """When ums.store_prediction raises, generate_predictions() should still
    return the predictions list without crashing."""
    engine = _make_engine_with_triggers(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert overlapping calendar events to produce at least one prediction
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Meeting X",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2, minutes=30)).isoformat(),
        "payload": {
            "title": "Meeting Y",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    # Make store_prediction raise (simulates user_model.db write failure)
    with patch.object(
        user_model_store, "store_prediction",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        # Should not raise — predictions are still returned even if storage fails
        predictions = await engine.generate_predictions({})

    # The function should complete and return results
    assert isinstance(predictions, list)


# -------------------------------------------------------------------------
# Test 4: generation_stats records errors
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_stats_records_errors(db, event_store, user_model_store):
    """When a _check method fails, generation_stats should contain an error
    string (not a count) for that method."""
    engine = _make_engine_with_triggers(db, user_model_store)
    _insert_event(event_store)

    # Capture the generation_stats dict via the logger.info call
    logged_messages = []
    with (
        patch.object(
            engine, "_check_relationship_maintenance",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ),
        patch("services.prediction_engine.engine.logger") as mock_logger,
    ):
        # Let error log calls pass through, but capture info calls
        mock_logger.error = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.info = MagicMock(side_effect=lambda msg, *args: logged_messages.append((msg, args)))

        await engine.generate_predictions({})

    # Find the generation stats log line
    stats_log = None
    for msg, args in logged_messages:
        if "Generated predictions by type" in msg:
            # args[0] is generation_stats dict
            stats_log = args[0]
            break

    assert stats_log is not None, "generation_stats should have been logged"
    assert isinstance(stats_log['relationship_maintenance'], str), (
        "Failed check should record error string, not a count"
    )
    assert "error:" in stats_log['relationship_maintenance'], (
        "Error entry should start with 'error:'"
    )
    assert "database disk image is malformed" in stats_log['relationship_maintenance']


# -------------------------------------------------------------------------
# Test 5: Healthy path unchanged
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthy_path_unchanged(db, event_store, user_model_store):
    """With no errors, behavior is identical to before: predictions generated,
    stats recorded as counts, no error strings."""
    engine = _make_engine_with_triggers(db, user_model_store)
    now = datetime.now(timezone.utc)

    # Insert overlapping calendar events for a reliable prediction
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2)).isoformat(),
        "payload": {
            "title": "Standup",
            "start_time": (now + timedelta(hours=2)).isoformat(),
            "end_time": (now + timedelta(hours=3)).isoformat(),
        },
        "metadata": {},
    })
    event_store.store_event({
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": (now + timedelta(hours=2, minutes=30)).isoformat(),
        "payload": {
            "title": "1:1",
            "start_time": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "end_time": (now + timedelta(hours=3, minutes=30)).isoformat(),
        },
        "metadata": {},
    })

    # Capture generation_stats via logger
    logged_stats = []
    with patch("services.prediction_engine.engine.logger") as mock_logger:
        mock_logger.error = MagicMock()
        mock_logger.debug = MagicMock()
        mock_logger.info = MagicMock(
            side_effect=lambda msg, *args: logged_stats.append((msg, args))
        )

        predictions = await engine.generate_predictions({})

    # Find the generation stats log
    stats = None
    for msg, args in logged_stats:
        if "Generated predictions by type" in msg:
            stats = args[0]
            break

    assert stats is not None, "generation_stats should be logged"

    # All stats entries should be integers (counts), not error strings
    for key in ['calendar_conflicts', 'routine_deviations', 'relationship_maintenance',
                'preparation_needs', 'follow_up_needs', 'spending_patterns']:
        val = stats[key]
        assert isinstance(val, int), (
            f"In healthy path, {key} should be an int count, got {type(val).__name__}: {val}"
        )

    # logger.error should NOT have been called for any _check method
    mock_logger.error.assert_not_called()


# -------------------------------------------------------------------------
# Test 6: Signal profile read failure is handled gracefully
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_profile_read_failure_handled(db, event_store, user_model_store):
    """When ums.get_signal_profile raises, generate_predictions() should
    still complete without crashing (available_profiles defaults to [])."""
    engine = _make_engine_with_triggers(db, user_model_store)
    _insert_event(event_store)

    # Make get_signal_profile raise (simulates user_model.db read failure)
    original_get = user_model_store.get_signal_profile

    call_count = 0

    def failing_get(profile_type):
        nonlocal call_count
        call_count += 1
        # Fail on the summary log call (after predictions are generated)
        # but allow earlier calls in _check methods to use the real implementation
        if call_count > 10:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_get(profile_type)

    with patch.object(user_model_store, "get_signal_profile", side_effect=failing_get):
        # Should not raise
        predictions = await engine.generate_predictions({})

    assert isinstance(predictions, list)
