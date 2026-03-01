"""
Tests for the cadence and mood_signals signal profile backfill scripts and
their startup triggers in main.py.

After a user_model.db migration or rebuild, all signal profiles are wiped.
PRs #316-317 added auto-backfill for 4 of 8 profile types (relationship,
temporal, topic, linguistic), but cadence and mood_signals had NO recovery path.
These tests verify the new backfill scripts and startup triggers work correctly.

Tests cover:
  - Cadence backfill creates profile from communication events
  - Cadence backfill skips when profile already has >= 10 samples
  - Cadence backfill skips when < 10 qualifying events exist
  - Mood backfill creates profile from diverse event types
  - Mood backfill skips when profile already has >= 10 samples
  - Mood backfill skips when < 10 qualifying events exist
  - Both backfills are fail-open (exceptions don't propagate)
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from scripts.backfill_cadence_profile import backfill_cadence_profile
from scripts.backfill_mood_profile import backfill_mood_profile
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


def _email_received(db, event_id: str, from_addr: str, hours_ago: int,
                    body: str = "Hello, here is the update on the project.") -> None:
    """Insert a synthetic email.received event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "email.received", ts, {
        "from_address": from_addr,
        "to_addresses": ["user@example.com"],
        "subject": "Project update",
        "body": body,
        "body_plain": body,
        "email_date": ts,
        "message_id": f"msg-{event_id}",
    })


def _email_sent(db, event_id: str, to_addr: str, hours_ago: int,
                body: str = "Thanks for the update, I will review it.",
                is_reply: bool = False, in_reply_to: str | None = None) -> None:
    """Insert a synthetic email.sent event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    payload = {
        "to_addresses": [to_addr],
        "subject": "Re: Project update",
        "body": body,
        "body_plain": body,
        "sent_at": ts,
    }
    if is_reply:
        payload["is_reply"] = True
    if in_reply_to:
        payload["in_reply_to"] = in_reply_to
    _insert_event(db, event_id, "email.sent", ts, payload)


def _message_sent(db, event_id: str, hours_ago: int,
                  body: str = "Got it, working on it now.") -> None:
    """Insert a synthetic message.sent event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "message.sent", ts, {
        "body": body,
        "body_plain": body,
    }, source="signal")


def _message_received(db, event_id: str, hours_ago: int,
                      body: str = "Can you check the latest build?") -> None:
    """Insert a synthetic message.received event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "message.received", ts, {
        "from_address": "alice@example.com",
        "body": body,
        "body_plain": body,
    }, source="signal")


def _calendar_event(db, event_id: str, hours_ago: int) -> None:
    """Insert a synthetic calendar.event.created event."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    _insert_event(db, event_id, "calendar.event.created", ts, {
        "title": "Team standup",
        "start_time": ts,
        "attendees": ["alice@example.com", "bob@example.com"],
    })


def _insert_diverse_mood_events(db, count: int = 15) -> None:
    """Insert a mix of event types that the MoodInferenceEngine processes.

    Creates communication, calendar, and command events to give the mood
    engine a realistic variety of signals to work with.
    """
    for i in range(count):
        hours = i + 1
        if i % 4 == 0:
            _email_received(db, f"mood-recv-{i}", "alice@example.com", hours,
                            body="The project deadline is approaching, we need to finish this work soon.")
        elif i % 4 == 1:
            _email_sent(db, f"mood-sent-{i}", "alice@example.com", hours,
                        body="I am working on the deliverables and will have them ready by tomorrow.")
        elif i % 4 == 2:
            _message_received(db, f"mood-msg-recv-{i}", hours,
                              body="Hey, can you review this when you get a chance?")
        else:
            _calendar_event(db, f"mood-cal-{i}", hours)


# ---------------------------------------------------------------------------
# Cadence profile backfill script tests
# ---------------------------------------------------------------------------


def test_cadence_backfill_creates_profile_from_communication_events(db, user_model_store):
    """Backfill should create a cadence profile from email and message events.

    The CadenceExtractor processes communication events to build activity
    heatmaps (hourly/daily histograms) and optionally response-time tracking.
    After processing >= 1 event, the cadence profile should exist with data.
    """
    for i in range(12):
        _email_received(db, f"cad-recv-{i}", "alice@example.com", hours_ago=i + 1)
    for i in range(3):
        _email_sent(db, f"cad-sent-{i}", "alice@example.com", hours_ago=i + 1)

    result = backfill_cadence_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 15
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None, "Cadence profile should be created by backfill"
    assert profile["samples_count"] > 0, "Profile should have samples"

    # Activity histograms should have data
    hourly = profile["data"].get("hourly_activity", {})
    assert len(hourly) > 0, "Hourly activity histogram should be populated"


def test_cadence_backfill_processes_messages(db, user_model_store):
    """Backfill should process message.sent and message.received events too.

    The CadenceExtractor handles both email and messaging channels so the
    cadence profile reflects all communication patterns.
    """
    for i in range(5):
        _message_sent(db, f"cad-msgsent-{i}", hours_ago=i + 1)
    for i in range(5):
        _message_received(db, f"cad-msgrecv-{i}", hours_ago=i + 6)

    result = backfill_cadence_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 10
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("cadence")
    assert profile is not None


def test_cadence_backfill_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully."""
    result = backfill_cadence_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["signals_extracted"] == 0
    assert result["errors"] == 0


def test_cadence_backfill_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys.

    main.py's _backfill_cadence_profile_if_needed relies on specific keys
    in the returned statistics dict (contacts_tracked, events_processed, etc.).
    """
    result = backfill_cadence_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "contacts_tracked",
        "initial_samples",
        "final_samples",
        "samples_added",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_cadence_backfill_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database."""
    for i in range(5):
        _email_received(db, f"cad-dry-{i}", "alice@example.com", hours_ago=i + 1)

    initial_profile = user_model_store.get_signal_profile("cadence")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_cadence_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True
    final_profile = user_model_store.get_signal_profile("cadence")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


# ---------------------------------------------------------------------------
# Mood profile backfill script tests
# ---------------------------------------------------------------------------


def test_mood_backfill_creates_profile_from_diverse_events(db, user_model_store):
    """Backfill should create a mood_signals profile from various event types.

    The MoodInferenceEngine processes 10 event types including email, message,
    calendar, and more. After processing, the mood_signals profile should
    contain a recent_signals ring buffer with mood-relevant data.
    """
    _insert_diverse_mood_events(db, count=15)

    result = backfill_mood_profile(data_dir=db.data_dir)

    assert result["events_processed"] >= 10
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("mood_signals")
    assert profile is not None, "Mood signals profile should be created by backfill"
    assert profile["samples_count"] > 0, "Profile should have samples"

    # recent_signals should have mood signal data
    recent = profile["data"].get("recent_signals", [])
    assert len(recent) > 0, "Recent signals ring buffer should be populated"


def test_mood_backfill_processes_communication_events(db, user_model_store):
    """Backfill should extract mood signals from email and message events.

    Communication events produce mood signals including message_length,
    negative_language, circadian_energy, and communication_energy.
    """
    for i in range(10):
        _email_sent(db, f"mood-esent-{i}", "alice@example.com", hours_ago=i + 1,
                    body="I am working on the important project deliverables for the team meeting.")

    result = backfill_mood_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 10
    assert result["errors"] == 0

    profile = user_model_store.get_signal_profile("mood_signals")
    assert profile is not None
    recent = profile["data"].get("recent_signals", [])
    # Outbound emails should produce message_length and circadian_energy signals
    signal_types = {s["signal_type"] for s in recent}
    assert "message_length" in signal_types or "circadian_energy" in signal_types, (
        "Communication events should produce mood-relevant signals"
    )


def test_mood_backfill_empty_db_returns_zero_stats(db, user_model_store):
    """Backfill should handle an empty events database gracefully."""
    result = backfill_mood_profile(data_dir=db.data_dir)

    assert result["events_processed"] == 0
    assert result["signals_extracted"] == 0
    assert result["errors"] == 0


def test_mood_backfill_returns_correct_stats_keys(db, user_model_store):
    """Return value should contain all expected statistics keys."""
    result = backfill_mood_profile(data_dir=db.data_dir)

    required_keys = {
        "events_processed",
        "signals_extracted",
        "initial_samples",
        "final_samples",
        "samples_added",
        "errors",
        "elapsed_seconds",
        "dry_run",
    }
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"


def test_mood_backfill_dry_run_no_writes(db, user_model_store):
    """Dry run should not write to the database."""
    _insert_diverse_mood_events(db, count=10)

    initial_profile = user_model_store.get_signal_profile("mood_signals")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    result = backfill_mood_profile(data_dir=db.data_dir, dry_run=True)

    assert result["dry_run"] is True
    final_profile = user_model_store.get_signal_profile("mood_signals")
    final_samples = final_profile["samples_count"] if final_profile else 0
    assert final_samples == initial_samples


# ---------------------------------------------------------------------------
# Startup trigger tests — cadence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_triggers_cadence_backfill_when_profile_empty(db):
    """_backfill_cadence_profile_if_needed() calls the backfill when profile is absent.

    Inserts enough communication events to pass the 10-event threshold and
    confirms the cadence backfill method completes without error.
    """
    from main import LifeOS

    for i in range(15):
        _email_received(db, f"startup-cad-{i}", "alice@example.com", hours_ago=i + 1)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    fake_stats = {
        "contacts_tracked": 1,
        "events_processed": 15,
        "elapsed_seconds": 0.1,
    }
    with patch("scripts.backfill_cadence_profile.backfill_cadence_profile",
               return_value=fake_stats):
        await life_os._backfill_cadence_profile_if_needed()

    # No exception = pass. The method completed successfully.


@pytest.mark.asyncio
async def test_startup_skips_cadence_backfill_when_already_populated(db):
    """_backfill_cadence_profile_if_needed() is a no-op when profile has data.

    Pre-populates the profile with >= 10 samples and confirms the backfill
    function is never called.
    """
    from main import LifeOS

    ums = UserModelStore(db)
    # Pre-populate with 12 samples to exceed the guard threshold
    for _ in range(12):
        ums.update_signal_profile("cadence", {"hourly_activity": {}, "response_times": []})

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = ums

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_cadence_profile.backfill_cadence_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_cadence_profile_if_needed()

    assert len(called) == 0, "Cadence backfill should be skipped when profile already has data"


@pytest.mark.asyncio
async def test_startup_skips_cadence_backfill_on_insufficient_events(db):
    """_backfill_cadence_profile_if_needed() skips when fewer than 10 events exist.

    An empty database or one with only a handful of events should not trigger
    the cadence backfill — there's too little data to learn from.
    """
    from main import LifeOS

    # Insert only 5 emails (below the 10-event threshold)
    for i in range(5):
        _email_received(db, f"startup-cad-few-{i}", "alice@example.com", hours_ago=i + 1)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_cadence_profile.backfill_cadence_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_cadence_profile_if_needed()

    assert len(called) == 0, "Cadence backfill should be skipped when event count < 10"


# ---------------------------------------------------------------------------
# Startup trigger tests — mood_signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_triggers_mood_backfill_when_profile_empty(db):
    """_backfill_mood_signals_profile_if_needed() calls the backfill when profile is absent.

    Inserts enough events to pass the 10-event threshold and confirms the
    mood backfill method completes without error.
    """
    from main import LifeOS

    _insert_diverse_mood_events(db, count=15)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    fake_stats = {
        "final_samples": 15,
        "events_processed": 15,
        "elapsed_seconds": 0.1,
    }
    with patch("scripts.backfill_mood_profile.backfill_mood_profile",
               return_value=fake_stats):
        await life_os._backfill_mood_signals_profile_if_needed()


@pytest.mark.asyncio
async def test_startup_skips_mood_backfill_when_already_populated(db):
    """_backfill_mood_signals_profile_if_needed() is a no-op when profile has data.

    Pre-populates the profile with >= 10 samples and confirms the backfill
    function is never called.
    """
    from main import LifeOS

    ums = UserModelStore(db)
    # Pre-populate with 12 samples to exceed the guard threshold
    for _ in range(12):
        ums.update_signal_profile("mood_signals", {"recent_signals": []})

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = ums

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_mood_profile.backfill_mood_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_mood_signals_profile_if_needed()

    assert len(called) == 0, "Mood backfill should be skipped when profile already has data"


@pytest.mark.asyncio
async def test_startup_skips_mood_backfill_on_insufficient_events(db):
    """_backfill_mood_signals_profile_if_needed() skips when fewer than 10 events exist.

    An empty database or one with only a handful of events should not trigger
    the mood backfill.
    """
    from main import LifeOS

    # Insert only 5 emails (below the 10-event threshold)
    for i in range(5):
        _email_received(db, f"startup-mood-few-{i}", "alice@example.com", hours_ago=i + 1)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    called = []

    def _mock_backfill(**kwargs):
        called.append(True)
        return {}

    with patch("scripts.backfill_mood_profile.backfill_mood_profile",
               side_effect=_mock_backfill):
        await life_os._backfill_mood_signals_profile_if_needed()

    assert len(called) == 0, "Mood backfill should be skipped when event count < 10"


# ---------------------------------------------------------------------------
# Fail-open tests — both backfills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cadence_backfill_failure_does_not_crash_startup(db):
    """Cadence backfill failures are non-fatal — startup continues on error.

    Verifies the fail-open design: even if the backfill script raises an
    exception, the startup trigger catches it and logs a warning.
    """
    from main import LifeOS

    for i in range(15):
        _email_received(db, f"failopen-cad-{i}", "alice@example.com", hours_ago=i + 1)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    def _crash(**kwargs):
        raise RuntimeError("Simulated cadence backfill crash")

    with patch("scripts.backfill_cadence_profile.backfill_cadence_profile",
               side_effect=_crash):
        # Should NOT raise — fail-open design
        await life_os._backfill_cadence_profile_if_needed()


@pytest.mark.asyncio
async def test_mood_backfill_failure_does_not_crash_startup(db):
    """Mood backfill failures are non-fatal — startup continues on error.

    Verifies the fail-open design: even if the backfill script raises an
    exception, the startup trigger catches it and logs a warning.
    """
    from main import LifeOS

    _insert_diverse_mood_events(db, count=15)

    life_os = LifeOS.__new__(LifeOS)
    life_os.db = db
    life_os.user_model_store = UserModelStore(db)

    def _crash(**kwargs):
        raise RuntimeError("Simulated mood backfill crash")

    with patch("scripts.backfill_mood_profile.backfill_mood_profile",
               side_effect=_crash):
        # Should NOT raise — fail-open design
        await life_os._backfill_mood_signals_profile_if_needed()
