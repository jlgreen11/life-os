"""
Tests for the iMessage connector.

Uses a fake chat.db (same schema as macOS ~/Library/Messages/chat.db)
created in a temp directory so no Full Disk Access is required.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from connectors.imessage.connector import iMessageConnector, APPLE_EPOCH_OFFSET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fake_chat_db(path: str) -> str:
    """Create a minimal Messages-compatible SQLite database at *path*.

    Returns the full file path to the created database.
    """
    db_file = os.path.join(path, "chat.db")
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS handle (
            ROWID   INTEGER PRIMARY KEY AUTOINCREMENT,
            id      TEXT NOT NULL,
            service TEXT NOT NULL DEFAULT 'iMessage'
        );

        CREATE TABLE IF NOT EXISTS chat (
            ROWID            INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_identifier  TEXT,
            display_name     TEXT
        );

        CREATE TABLE IF NOT EXISTS message (
            ROWID            INTEGER PRIMARY KEY AUTOINCREMENT,
            guid             TEXT UNIQUE NOT NULL,
            text             TEXT,
            handle_id        INTEGER,
            date             INTEGER DEFAULT 0,
            is_from_me       INTEGER DEFAULT 0,
            cache_roomnames  TEXT,
            service          TEXT DEFAULT 'iMessage'
        );

        CREATE TABLE IF NOT EXISTS chat_message_join (
            chat_id    INTEGER,
            message_id INTEGER
        );
    """)
    conn.close()
    return db_file


def _apple_ns_from_unix(unix_ts: float) -> int:
    """Convert a Unix timestamp to Apple's nanosecond epoch (Core Data)."""
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1e9)


def _insert_message(
    chat_db: str,
    text: str,
    sender_id: str = "+15551234567",
    service: str = "iMessage",
    is_from_me: int = 0,
    dt: datetime | None = None,
    group_name: str | None = None,
    chat_identifier: str | None = None,
) -> int:
    """Insert a message into the fake chat.db and return its ROWID."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    apple_ns = _apple_ns_from_unix(dt.timestamp())

    conn = sqlite3.connect(chat_db)
    cur = conn.cursor()

    # Upsert handle
    cur.execute("SELECT ROWID FROM handle WHERE id = ?", (sender_id,))
    row = cur.fetchone()
    if row:
        handle_rowid = row[0]
    else:
        cur.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            (sender_id, service),
        )
        handle_rowid = cur.lastrowid

    # Insert message
    guid = f"msg-{time.time_ns()}"
    cur.execute(
        """INSERT INTO message (guid, text, handle_id, date, is_from_me,
                                cache_roomnames, service)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (guid, text, handle_rowid, apple_ns, is_from_me, group_name, service),
    )
    msg_rowid = cur.lastrowid

    # Insert chat + join
    ci = chat_identifier or sender_id
    cur.execute("SELECT ROWID FROM chat WHERE chat_identifier = ?", (ci,))
    chat_row = cur.fetchone()
    if chat_row:
        chat_rowid = chat_row[0]
    else:
        cur.execute(
            "INSERT INTO chat (chat_identifier, display_name) VALUES (?, ?)",
            (ci, group_name),
        )
        chat_rowid = cur.lastrowid

    cur.execute(
        "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
        (chat_rowid, msg_rowid),
    )

    conn.commit()
    conn.close()
    return msg_rowid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_chat_db(tmp_path):
    """Return the path to a freshly-created fake chat.db."""
    return _create_fake_chat_db(str(tmp_path))


@pytest.fixture()
def mock_event_bus():
    """An AsyncMock standing in for EventBus."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value="evt-id-1")
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture()
def connector(fake_chat_db, mock_event_bus, db):
    """An iMessageConnector wired to the fake chat.db.

    Seeds a connector_state row so that set_sync_cursor (which uses
    UPDATE) has a row to modify -- in production this row is created
    by BaseConnector.start() -> _update_state("active").
    """
    c = iMessageConnector(
        event_bus=mock_event_bus,
        db=db,
        config={"db_path": fake_chat_db, "include_sms": True},
    )
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT OR IGNORE INTO connector_state (connector_id, status) VALUES (?, ?)",
            (c.CONNECTOR_ID, "active"),
        )
    return c


# ---------------------------------------------------------------------------
# TestAuthenticate
# ---------------------------------------------------------------------------

class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_success_db_exists(self, connector):
        result = await connector.authenticate()
        assert result is True

    @pytest.mark.asyncio
    async def test_failure_db_missing(self, mock_event_bus, db, tmp_path):
        missing = os.path.join(str(tmp_path), "nonexistent", "chat.db")
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": missing},
        )
        result = await c.authenticate()
        assert result is False


# ---------------------------------------------------------------------------
# TestSync
# ---------------------------------------------------------------------------

class TestSync:
    @pytest.mark.asyncio
    async def test_no_messages_returns_zero(self, connector):
        count = await connector.sync()
        assert count == 0

    @pytest.mark.asyncio
    async def test_inbound_message_publishes_received(self, connector, fake_chat_db, mock_event_bus):
        _insert_message(fake_chat_db, "Hello from friend", sender_id="+15559876543")
        count = await connector.sync()

        assert count == 1
        mock_event_bus.publish.assert_called_once()

        call_args = mock_event_bus.publish.call_args
        event_type = call_args[0][0]
        payload = call_args[0][1]

        assert event_type == "message.received"
        assert payload["channel"] == "imessage"
        assert payload["direction"] == "inbound"
        assert payload["from_address"] == "+15559876543"
        assert payload["body"] == "Hello from friend"
        assert payload["service_type"] == "iMessage"

    @pytest.mark.asyncio
    async def test_outbound_message_publishes_sent(self, connector, fake_chat_db, mock_event_bus):
        _insert_message(
            fake_chat_db, "My reply", sender_id="+15559876543", is_from_me=1,
        )
        count = await connector.sync()

        assert count == 1
        call_args = mock_event_bus.publish.call_args
        event_type = call_args[0][0]
        payload = call_args[0][1]

        assert event_type == "message.sent"
        assert payload["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_incremental_cursor(self, connector, fake_chat_db, mock_event_bus):
        """After first sync, a second sync with no new messages returns 0."""
        _insert_message(fake_chat_db, "First message")
        count1 = await connector.sync()
        assert count1 == 1

        # Reset mock to track only subsequent calls
        mock_event_bus.publish.reset_mock()

        count2 = await connector.sync()
        assert count2 == 0
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_incremental_cursor_picks_up_new(self, connector, fake_chat_db, mock_event_bus):
        """After first sync, new messages are picked up by a second sync."""
        _insert_message(fake_chat_db, "First message")
        await connector.sync()
        mock_event_bus.publish.reset_mock()

        _insert_message(fake_chat_db, "Second message", sender_id="+15550001111")
        count = await connector.sync()
        assert count == 1

    @pytest.mark.asyncio
    async def test_group_message(self, connector, fake_chat_db, mock_event_bus):
        _insert_message(
            fake_chat_db, "Group hello",
            sender_id="+15559876543",
            group_name="Family Chat",
            chat_identifier="chat123456",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_group"] is True
        assert payload["group_name"] == "Family Chat"

    @pytest.mark.asyncio
    async def test_sms_message(self, connector, fake_chat_db, mock_event_bus):
        _insert_message(
            fake_chat_db, "SMS text",
            sender_id="+15559876543",
            service="SMS",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["service_type"] == "SMS"

    @pytest.mark.asyncio
    async def test_sms_excluded_when_disabled(self, fake_chat_db, mock_event_bus, db):
        """When include_sms is False, SMS messages should be skipped."""
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": fake_chat_db, "include_sms": False},
        )
        with db.get_connection("state") as conn:
            conn.execute(
                "INSERT OR IGNORE INTO connector_state (connector_id, status) VALUES (?, ?)",
                (c.CONNECTOR_ID, "active"),
            )
        _insert_message(fake_chat_db, "SMS text", service="SMS")
        count = await c.sync()
        assert count == 0
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_messages_skipped(self, connector, fake_chat_db, mock_event_bus):
        """Messages with None or empty text should be silently skipped."""
        _insert_message(fake_chat_db, None)
        _insert_message(fake_chat_db, "")
        _insert_message(fake_chat_db, "Real message", sender_id="+15550009999")

        count = await connector.sync()
        assert count == 1
        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["body"] == "Real message"

    @pytest.mark.asyncio
    async def test_snippet_truncated_at_150(self, connector, fake_chat_db, mock_event_bus):
        long_text = "A" * 300
        _insert_message(fake_chat_db, long_text)
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert len(payload["snippet"]) == 150

    @pytest.mark.asyncio
    async def test_timestamp_conversion(self, connector, fake_chat_db, mock_event_bus):
        """Verify the Apple-epoch timestamp round-trips to a valid ISO string."""
        known_dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        _insert_message(fake_chat_db, "Timed msg", dt=known_dt)
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        parsed = datetime.fromisoformat(payload["timestamp"])
        # Allow 1-second tolerance for float rounding
        assert abs((parsed - known_dt).total_seconds()) < 1


# ---------------------------------------------------------------------------
# TestExecute
# ---------------------------------------------------------------------------

class TestExecute:
    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, connector):
        with pytest.raises(ValueError, match="Unknown action"):
            await connector.execute("nonexistent_action", {})

    @pytest.mark.asyncio
    async def test_send_message_calls_osascript(self, connector, mock_event_bus):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Hi there"},
            )

        assert result["status"] == "sent"
        mock_exec.assert_called_once()
        # Verify osascript was the first argument
        assert mock_exec.call_args[0][0] == "osascript"

    @pytest.mark.asyncio
    async def test_send_message_publishes_sent_event(self, connector, mock_event_bus):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Hello"},
            )

        # publish_event calls bus.publish under the hood
        mock_event_bus.publish.assert_called_once()
        event_type = mock_event_bus.publish.call_args[0][0]
        assert event_type == "message.sent"

    @pytest.mark.asyncio
    async def test_send_message_failure_raises(self, connector):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"AppleScript error"))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError):
                await connector.execute(
                    "send_message",
                    {"to": "+15559876543", "message": "Hi"},
                )


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_ok_when_db_exists(self, connector):
        result = await connector.health_check()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_error_when_db_missing(self, mock_event_bus, db, tmp_path):
        missing = os.path.join(str(tmp_path), "nonexistent", "chat.db")
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": missing},
        )
        result = await c.health_check()
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# TestClassifyDomain
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    def test_work_keywords(self, connector):
        assert connector._classify_domain("Work Chat") == "work"
        assert connector._classify_domain("Team Standup") == "work"
        assert connector._classify_domain("Project Alpha") == "work"

    def test_personal_default(self, connector):
        assert connector._classify_domain("Family") == "personal"
        assert connector._classify_domain(None) == "personal"
        assert connector._classify_domain("") == "personal"
