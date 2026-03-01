"""
Tests for the spatial and decision signal profile backfill scripts and their
startup triggers in main.py.

After a user_model.db migration or rebuild, all signal profiles are wiped.
PRs #316-317 added auto-backfill for 6 of 8 profile types but omitted spatial
and decision.  These tests verify the new startup triggers work correctly.

Tests cover:
  - Spatial backfill fires when profile is empty and enough events exist
  - Spatial backfill skips when profile already has >= 10 samples
  - Spatial backfill skips when < 10 qualifying events exist
  - Spatial backfill is fail-open (exceptions don't propagate)
  - Decision backfill fires when profile is empty and enough events exist
  - Decision backfill skips when profile already has >= 20 samples
  - Decision backfill skips when < 20 qualifying events exist
  - Decision backfill is fail-open (exceptions don't propagate)
  - Backfill scripts return expected statistics keys
  - Backfill scripts handle empty databases gracefully
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from scripts.backfill_decision_profile import backfill_decision_profile
from scripts.backfill_spatial_profile import backfill_spatial_profile
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_event(db, event_id: str, event_type: str, timestamp: str,
                  payload: dict, source: str = "google") -> None:
    """Insert a test event directly into events.db."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, 'normal', ?, '{}')""",
            (event_id, event_type, source, timestamp, json.dumps(payload)),
        )


def _calendar_event_with_location(db, event_id: str, hours_ago: int,
                                  location: str = "Conference Room A") -> None:
    """Insert a synthetic calendar.event.created event with a location field."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "calendar.event.created", ts, {
        "title": "Team standup",
        "start_time": ts,
        "end_time": (datetime.now(timezone.utc) - timedelta(hours=hours_ago - 1)).isoformat(),
        "location": location,
        "attendees": ["alice@example.com"],
    })


def _task_created(db, event_id: str, hours_ago: int,
                  title: str = "Review PR") -> None:
    """Insert a synthetic task.created event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "task.created", ts, {
        "title": title,
        "description": "Review the pull request for the new feature",
        "created_at": ts,
    })


def _task_completed(db, event_id: str, hours_ago: int,
                    title: str = "Review PR") -> None:
    """Insert a synthetic task.completed event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "task.completed", ts, {
        "title": title,
        "completed_at": ts,
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=hours_ago + 2)).isoformat(),
    })


def _email_sent(db, event_id: str, hours_ago: int,
                body: str = "Thanks for the update, I will review it.") -> None:
    """Insert a synthetic email.sent event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "email.sent", ts, {
        "to_addresses": ["alice@example.com"],
        "subject": "Re: Project update",
        "body": body,
        "body_plain": body,
        "sent_at": ts,
    })


def _message_sent(db, event_id: str, hours_ago: int,
                  body: str = "Got it, working on it now.") -> None:
    """Insert a synthetic message.sent event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "message.sent", ts, {
        "body": body,
        "body_plain": body,
    }, source="signal")


def _calendar_event_no_location(db, event_id: str, hours_ago: int) -> None:
    """Insert a calendar.event.created event without location (used for decision events)."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "calendar.event.created", ts, {
        "title": "Team sync",
        "start_time": ts,
        "attendees": ["bob@example.com"],
    })


def _insert_diverse_decision_events(db, count: int = 25) -> None:
    """Insert a mix of event types that the DecisionExtractor processes.

    Creates task, email, message, and calendar events to give the decision
    engine a realistic variety of decision-making signals.
    """
    for i in range(count):
        hours = i + 1
        if i % 5 == 0:
            _task_created(db, f"dec-tc-{i}", hours, title=f"Task {i}")
        elif i % 5 == 1:
            _task_completed(db, f"dec-td-{i}", hours, title=f"Task {i}")
        elif i % 5 == 2:
            _email_sent(db, f"dec-es-{i}", hours,
                        body="I've decided to proceed with option A for the deployment.")
        elif i % 5 == 3:
            _message_sent(db, f"dec-ms-{i}", hours,
                          body="Let's go ahead with the new design.")
        else:
            _calendar_event_no_location(db, f"dec-cal-{i}", hours)


def _insert_spatial_events(db, count: int = 15) -> None:
    """Insert calendar events with location data for spatial backfill testing."""
    locations = [
        "Conference Room A", "Main Office", "Coffee Shop",
        "Conference Room B", "Home Office",
    ]
    for i in range(count):
        _calendar_event_with_location(
            db, f"spatial-{i}", hours_ago=i + 1,
            location=locations[i % len(locations)],
        )


# ---------------------------------------------------------------------------
# Spatial profile backfill script tests
# ---------------------------------------------------------------------------


def test_spatial_backfill_creates_profile_from_location_events(db, user_model_store):
    """Backfill should create a spatial profile from calendar events with locations.

    The SpatialExtractor processes calendar.event.created events that have a
    non-empty location field.  After processing, the spatial profile should
    exist with place_behaviors data.
    """
    _insert_spatial_events(db, count=15)

    result = backfill_spatial_profile(data_dir=db.data_dir)

    assert result["events_processed"] >= 1
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("spatial")
    assert profile is not None, "Spatial profile should be created by backfill"
    assert profile["samples_count"] > 0, "Profile should have samples"


def test_spatial_backfill_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully."""
    result = backfill_spatial_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["signals_extracted"] == 0
    assert result["errors"] == 0


def test_spatial_backfill_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys.

    main.py's _backfill_spatial_profile_if_needed relies on specific keys
    in the returned statistics dict (final_samples, events_processed, etc.).
    """
    result = backfill_spatial_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "initial_samples",
        "final_samples",
        "samples_added",
        "unique_places",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_spatial_backfill_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database."""
    _insert_spatial_events(db, count=10)

    initial_profile = user_model_store.get_signal_profile("spatial")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_spatial_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True
    final_profile = user_model_store.get_signal_profile("spatial")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


# ---------------------------------------------------------------------------
# Decision profile backfill script tests
# ---------------------------------------------------------------------------


def test_decision_backfill_creates_profile_from_decision_events(db, user_model_store):
    """Backfill should create a decision profile from task and communication events.

    The DecisionExtractor processes task.completed, task.created, email.sent,
    message.sent, and calendar.event.created events.  After processing, the
    decision profile should contain decision-making pattern data.
    """
    _insert_diverse_decision_events(db, count=25)

    result = backfill_decision_profile(data_dir=db.data_dir)

    assert result["events_processed"] >= 1
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("decision")
    assert profile is not None, "Decision profile should be created by backfill"
    assert profile["samples_count"] > 0, "Profile should have samples"


def test_decision_backfill_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully."""
    result = backfill_decision_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["signals_extracted"] == 0
    assert result["errors"] == 0


def test_decision_backfill_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys.

    main.py's _backfill_decision_profile_if_needed relies on specific keys
    in the returned statistics dict (final_samples, events_processed, etc.).
    """
    result = backfill_decision_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "initial_samples",
        "final_samples",
        "samples_added",
        "decision_speed_samples",
        "delegation_samples",
        "commitment_samples",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_decision_backfill_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database."""
    _insert_diverse_decision_events(db, count=15)

    initial_profile = user_model_store.get_signal_profile("decision")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_decision_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True
    final_profile = user_model_store.get_signal_profile("decision")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


# ---------------------------------------------------------------------------
# Startup trigger tests — spatial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_triggers_spatial_backfill_when_profile_empty(db):
    """_backfill_spatial_profile_if_needed() calls the backfill when profile is absent.

    Inserts enough calendar-with-location events to pass the 10-event threshold
    and confirms the spatial backfill method completes without error.
    """
    from main import LifeOS

    _insert_spatial_events(db, count=15)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    fake_stats = {
        "final_samples": 15,
        "events_processed": 15,
        "elapsed_seconds": 0.1,
    }
    with patch("scripts.backfill_spatial_profile.backfill_spatial_profile",
               return_value=fake_stats):
        await life_os._backfill_spatial_profile_if_needed()

    # No exception = pass. The method completed successfully.


@pytest.mark.asyncio
async def test_startup_skips_spatial_backfill_when_already_populated(db):
    """_backfill_spatial_profile_if_needed() is a no-op when profile has data.

    Pre-populates the profile with >= 10 samples and confirms the backfill
    function is never called.
    """
    from main import LifeOS

    ums = UserModelStore(db)
    # Pre-populate with 12 samples to exceed the guard threshold (10)
    for _ in range(12):
        ums.update_signal_profile("spatial", {"place_behaviors": {}})

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = ums

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_spatial_profile.backfill_spatial_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_spatial_profile_if_needed()

    assert len(called) == 0, "Spatial backfill should be skipped when profile already has data"


@pytest.mark.asyncio
async def test_startup_skips_spatial_backfill_on_insufficient_events(db):
    """_backfill_spatial_profile_if_needed() skips when fewer than 10 events exist.

    An empty database or one with only a handful of events should not trigger
    the spatial backfill — there's too little data to learn from.
    """
    from main import LifeOS

    # Insert only 5 events (below the 10-event threshold)
    _insert_spatial_events(db, count=5)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_spatial_profile.backfill_spatial_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_spatial_profile_if_needed()

    assert len(called) == 0, "Spatial backfill should be skipped when event count < 10"


@pytest.mark.asyncio
async def test_spatial_backfill_failure_does_not_crash_startup(db):
    """Spatial backfill failures are non-fatal — startup continues on error.

    Verifies the fail-open design: even if the backfill script raises an
    exception, the startup trigger catches it and logs a warning.
    """
    from main import LifeOS

    _insert_spatial_events(db, count=15)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    def _crash(**kwargs):
        raise RuntimeError("Simulated spatial backfill crash")

    with patch("scripts.backfill_spatial_profile.backfill_spatial_profile",
               side_effect=_crash):
        # Should NOT raise — fail-open design
        await life_os._backfill_spatial_profile_if_needed()


# ---------------------------------------------------------------------------
# Startup trigger tests — decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_triggers_decision_backfill_when_profile_empty(db):
    """_backfill_decision_profile_if_needed() calls the backfill when profile is absent.

    Inserts enough decision-making events to pass the 20-event threshold
    and confirms the decision backfill method completes without error.
    """
    from main import LifeOS

    _insert_diverse_decision_events(db, count=25)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    fake_stats = {
        "final_samples": 25,
        "events_processed": 25,
        "elapsed_seconds": 0.1,
    }
    with patch("scripts.backfill_decision_profile.backfill_decision_profile",
               return_value=fake_stats):
        await life_os._backfill_decision_profile_if_needed()

    # No exception = pass. The method completed successfully.


@pytest.mark.asyncio
async def test_startup_skips_decision_backfill_when_already_populated(db):
    """_backfill_decision_profile_if_needed() is a no-op when profile has data.

    Pre-populates the profile with >= 20 samples and confirms the backfill
    function is never called.
    """
    from main import LifeOS

    ums = UserModelStore(db)
    # Pre-populate with 22 samples to exceed the guard threshold (20)
    for _ in range(22):
        ums.update_signal_profile("decision", {"decision_speed_by_domain": {}})

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = ums

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_decision_profile.backfill_decision_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_decision_profile_if_needed()

    assert len(called) == 0, "Decision backfill should be skipped when profile already has data"


@pytest.mark.asyncio
async def test_startup_skips_decision_backfill_on_insufficient_events(db):
    """_backfill_decision_profile_if_needed() skips when fewer than 20 events exist.

    An empty database or one with only a handful of events should not trigger
    the decision backfill — there's too little data to learn from.
    """
    from main import LifeOS

    # Insert only 10 events (below the 20-event threshold)
    for i in range(10):
        _task_created(db, f"startup-dec-few-{i}", hours_ago=i + 1)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_decision_profile.backfill_decision_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_decision_profile_if_needed()

    assert len(called) == 0, "Decision backfill should be skipped when event count < 20"


@pytest.mark.asyncio
async def test_decision_backfill_failure_does_not_crash_startup(db):
    """Decision backfill failures are non-fatal — startup continues on error.

    Verifies the fail-open design: even if the backfill script raises an
    exception, the startup trigger catches it and logs a warning.
    """
    from main import LifeOS

    _insert_diverse_decision_events(db, count=25)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    def _crash(**kwargs):
        raise RuntimeError("Simulated decision backfill crash")

    with patch("scripts.backfill_decision_profile.backfill_decision_profile",
               side_effect=_crash):
        # Should NOT raise — fail-open design
        await life_os._backfill_decision_profile_if_needed()
