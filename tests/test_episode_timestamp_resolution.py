"""
Tests for episode timestamp resolution in _create_episode().

Verifies that episodes use the actual event date (from email_date, sent_at,
received_at, date, or start_time payload fields) rather than the connector
sync timestamp.

Root cause: The Google connector stores the actual email date as
payload["email_date"] (RFC 2822 Date header), but the original _create_episode()
only checked payload["date"] — so all 55K+ email episodes fell back to
event["timestamp"] (sync time), collapsing to just 3 dates and breaking
routine detection entirely.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def life_os_instance(db, event_store, user_model_store):
    """A minimal LifeOS instance wired with real temp DBs for episode tests.

    Provides only the dependencies needed for _create_episode() — avoids
    starting NATS, Ollama, or any connectors.
    """
    from main import LifeOS

    lo = object.__new__(LifeOS)
    lo.db = db
    lo.event_store = event_store
    lo.user_model_store = user_model_store

    # Stub the signal extractor's mood lookup — mood is optional for episodes
    signal_extractor_stub = MagicMock()
    signal_extractor_stub.get_current_mood = MagicMock(return_value=None)
    lo.signal_extractor = signal_extractor_stub

    return lo


def _get_episodes(user_model_store, limit: int = 10) -> list[dict]:
    """Query episodes directly from the DB since UserModelStore has no get_episodes."""
    import json as json_mod
    with user_model_store.db.get_connection("user_model") as conn:
        rows = conn.execute(
            "SELECT id, timestamp, interaction_type FROM episodes ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def _make_event(event_type: str, payload: dict, sync_ts: str = "2026-02-22T08:00:00+00:00") -> dict:
    """Build a minimal event envelope for testing."""
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "google",
        "timestamp": sync_ts,   # connector sync time — should NOT be used for timestamp
        "priority": "normal",
        "payload": payload,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests: episode timestamp resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_episode_uses_email_date_field(life_os_instance):
    """Episodes for email.received events must use payload['email_date'], not sync ts.

    The Google connector stores the actual send/receive time from the RFC 2822
    Date header as payload['email_date']. This field has highest priority.
    """
    actual_email_date = "2026-01-15T09:30:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"  # sync is 38 days later

    event = _make_event("email.received", {
        "subject": "Test email",
        "from_address": "alice@example.com",
        "email_date": actual_email_date,
        # 'date' field NOT set — email_date should be preferred
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes, "Expected one episode to be stored"
    ep = episodes[0]
    assert ep["timestamp"] == actual_email_date, (
        f"Episode timestamp should be email_date={actual_email_date!r}, "
        f"got {ep['timestamp']!r} (sync_ts={sync_ts!r})"
    )


@pytest.mark.asyncio
async def test_episode_uses_sent_at_field(life_os_instance):
    """Episodes for message.sent events must use payload['sent_at']."""
    sent_at = "2026-01-20T14:45:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("message.sent", {
        "body": "Hey there",
        "to_addresses": ["bob@example.com"],
        "sent_at": sent_at,
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == sent_at


@pytest.mark.asyncio
async def test_episode_uses_received_at_field(life_os_instance):
    """Episodes for message.received events must use payload['received_at']."""
    received_at = "2026-01-22T18:12:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("message.received", {
        "body": "Hi",
        "from_address": "carol@example.com",
        "received_at": received_at,
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == received_at


@pytest.mark.asyncio
async def test_episode_uses_date_field_as_fallback(life_os_instance):
    """payload['date'] is used when email_date/sent_at/received_at are absent."""
    date_val = "2026-01-10T07:00:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("email.received", {
        "subject": "Old style",
        "from_address": "dave@example.com",
        "date": date_val,
        # No email_date, sent_at, received_at
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == date_val


@pytest.mark.asyncio
async def test_episode_uses_start_time_for_calendar(life_os_instance):
    """Calendar episodes use payload['start_time'] — the actual meeting start."""
    start_time = "2026-01-25T10:00:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("calendar.event.created", {
        "title": "Team standup",
        "start_time": start_time,
        "end_time": "2026-01-25T10:30:00+00:00",
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == start_time


@pytest.mark.asyncio
async def test_episode_falls_back_to_sync_ts_when_no_actual_date(life_os_instance):
    """When no actual-date field exists, the sync timestamp is used as last resort."""
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("email.received", {
        "subject": "No date fields",
        "from_address": "eve@example.com",
        # No email_date, sent_at, received_at, date, or start_time
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == sync_ts


@pytest.mark.asyncio
async def test_email_date_takes_priority_over_date(life_os_instance):
    """email_date has higher priority than the generic 'date' field."""
    email_date = "2026-01-05T06:00:00+00:00"
    generic_date = "2026-01-06T06:00:00+00:00"
    sync_ts = "2026-02-22T08:00:00+00:00"

    event = _make_event("email.received", {
        "subject": "Priority test",
        "from_address": "frank@example.com",
        "email_date": email_date,
        "date": generic_date,  # should be ignored
    }, sync_ts=sync_ts)

    await life_os_instance._create_episode(event)

    episodes = _get_episodes(life_os_instance.user_model_store, 1)
    assert episodes
    assert episodes[0]["timestamp"] == email_date, (
        "email_date should take priority over generic 'date' field"
    )


# ---------------------------------------------------------------------------
# Tests: backfill script
# ---------------------------------------------------------------------------


def test_backfill_script_extract_actual_timestamp():
    """_extract_actual_timestamp() returns the best available date from a payload."""
    from scripts.backfill_episode_timestamps import _extract_actual_timestamp

    # email_date wins
    assert _extract_actual_timestamp(
        {"email_date": "2026-01-01", "sent_at": "2026-01-02", "date": "2026-01-03"},
        "2026-02-22",
    ) == "2026-01-01"

    # sent_at used when no email_date
    assert _extract_actual_timestamp(
        {"sent_at": "2026-01-02", "date": "2026-01-03"},
        "2026-02-22",
    ) == "2026-01-02"

    # received_at used when no email_date or sent_at
    assert _extract_actual_timestamp(
        {"received_at": "2026-01-03", "date": "2026-01-04"},
        "2026-02-22",
    ) == "2026-01-03"

    # date as last payload fallback
    assert _extract_actual_timestamp(
        {"date": "2026-01-04"},
        "2026-02-22",
    ) == "2026-01-04"

    # start_time for calendar
    assert _extract_actual_timestamp(
        {"start_time": "2026-01-05T09:00:00"},
        "2026-02-22",
    ) == "2026-01-05T09:00:00"

    # Returns None when nothing better than sync ts is available
    assert _extract_actual_timestamp({}, "2026-02-22") is None


def test_backfill_script_dry_run(tmp_path):
    """Backfill in dry-run mode does not modify the database.

    Creates minimal user_model.db and events.db with one episode whose
    timestamp matches the sync time, then verifies that --dry-run leaves
    the episode timestamp unchanged.
    """
    import sqlite3 as sqlite3_direct

    from scripts.backfill_episode_timestamps import backfill

    # Create minimal events.db
    ev_db = str(tmp_path / "events.db")
    conn = sqlite3_direct.connect(ev_db)
    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT,
            source TEXT,
            timestamp TEXT,
            payload TEXT,
            priority TEXT,
            metadata TEXT
        )
    """)
    sync_ts = "2026-02-22T08:00:00+00:00"
    email_date = "2026-01-15T09:30:00+00:00"
    ev_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ev_id,
            "email.received",
            "google",
            sync_ts,
            json.dumps({"email_date": email_date, "subject": "Test"}),
            "normal",
            "{}",
        ),
    )
    conn.commit()
    conn.close()

    # Create minimal user_model.db
    um_db = str(tmp_path / "user_model.db")
    conn = sqlite3_direct.connect(um_db)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            event_id TEXT
        )
    """)
    ep_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO episodes VALUES (?, ?, ?)",
        (ep_id, sync_ts, ev_id),  # timestamp == sync_ts (the bug)
    )
    conn.commit()
    conn.close()

    # Run dry-run — should report 1 update but not modify the DB
    count = backfill(data_dir=str(tmp_path), dry_run=True)
    assert count == 1

    # Verify episode timestamp was NOT changed
    conn = sqlite3_direct.connect(um_db)
    row = conn.execute("SELECT timestamp FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    conn.close()
    assert row[0] == sync_ts, "Dry-run should not modify episode timestamps"


def test_backfill_script_live_run(tmp_path):
    """Backfill in live mode corrects stale episode timestamps.

    Verifies that episodes whose timestamp equals the event sync time are
    updated to use the actual date from the event payload.
    """
    import sqlite3 as sqlite3_direct

    from scripts.backfill_episode_timestamps import backfill

    ev_db = str(tmp_path / "events.db")
    conn = sqlite3_direct.connect(ev_db)
    conn.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            type TEXT,
            source TEXT,
            timestamp TEXT,
            payload TEXT,
            priority TEXT,
            metadata TEXT
        )
    """)
    sync_ts = "2026-02-22T08:00:00+00:00"
    email_date = "2026-01-15T09:30:00+00:00"
    ev_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ev_id,
            "email.received",
            "google",
            sync_ts,
            json.dumps({"email_date": email_date, "subject": "Test"}),
            "normal",
            "{}",
        ),
    )
    conn.commit()
    conn.close()

    um_db = str(tmp_path / "user_model.db")
    conn = sqlite3_direct.connect(um_db)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            event_id TEXT
        )
    """)
    ep_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO episodes VALUES (?, ?, ?)",
        (ep_id, sync_ts, ev_id),
    )
    conn.commit()
    conn.close()

    count = backfill(data_dir=str(tmp_path), dry_run=False)
    assert count == 1

    conn = sqlite3_direct.connect(um_db)
    row = conn.execute("SELECT timestamp FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    conn.close()
    assert row[0] == email_date, (
        f"Episode timestamp should be updated to {email_date!r}, got {row[0]!r}"
    )
