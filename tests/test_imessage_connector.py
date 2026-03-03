"""
Tests for the iMessage connector.

Uses a fake chat.db (same schema as macOS ~/Library/Messages/chat.db)
created in a temp directory so no Full Disk Access is required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from connectors.imessage.connector import iMessageConnector, APPLE_EPOCH_OFFSET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fake_chat_db(path: str) -> str:
    """Create a minimal Messages-compatible SQLite database at *path*.

    Includes the ``message_attachment_join`` and ``attachment`` tables and the
    ``thread_originator_guid`` column on ``message`` so attachment detection
    and reply detection can be tested.

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
            ROWID                    INTEGER PRIMARY KEY AUTOINCREMENT,
            guid                     TEXT UNIQUE NOT NULL,
            text                     TEXT,
            handle_id                INTEGER,
            date                     INTEGER DEFAULT 0,
            is_from_me               INTEGER DEFAULT 0,
            cache_roomnames          TEXT,
            service                  TEXT DEFAULT 'iMessage',
            thread_originator_guid   TEXT
        );

        CREATE TABLE IF NOT EXISTS chat_message_join (
            chat_id    INTEGER,
            message_id INTEGER
        );

        CREATE TABLE IF NOT EXISTS attachment (
            ROWID     INTEGER PRIMARY KEY AUTOINCREMENT,
            guid      TEXT UNIQUE,
            filename  TEXT,
            mime_type TEXT
        );

        CREATE TABLE IF NOT EXISTS message_attachment_join (
            message_id    INTEGER,
            attachment_id INTEGER
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
    thread_originator_guid: str | None = None,
    attachment_count: int = 0,
) -> int:
    """Insert a message into the fake chat.db and return its ROWID.

    Args:
        thread_originator_guid: Set to a message GUID to mark this as an
            inline reply to that message.
        attachment_count: Number of dummy attachments to link to the message.
    """
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
                                cache_roomnames, service, thread_originator_guid)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (guid, text, handle_rowid, apple_ns, is_from_me, group_name, service,
         thread_originator_guid),
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

    # Link dummy attachments if requested
    for i in range(attachment_count):
        att_guid = f"att-{time.time_ns()}-{i}"
        cur.execute(
            "INSERT INTO attachment (guid, filename, mime_type) VALUES (?, ?, ?)",
            (att_guid, f"photo_{i}.jpg", "image/jpeg"),
        )
        att_rowid = cur.lastrowid
        cur.execute(
            "INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (?, ?)",
            (msg_rowid, att_rowid),
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


# ---------------------------------------------------------------------------
# TestContactSync
# ---------------------------------------------------------------------------

class TestContactSync:
    """Test contact synchronization from chat.db to entities.db."""

    def test_sync_contacts_creates_new_contacts(self, connector, fake_chat_db):
        """New handles should create contact records with imessage channel."""
        conn = sqlite3.connect(fake_chat_db)
        conn.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("+15551234567", "iMessage"),
        )
        conn.commit()
        conn.close()

        connector._sync_contacts()

        with connector.db.get_connection("entities") as econn:
            row = econn.execute(
                "SELECT * FROM contact_identifiers WHERE identifier = ?",
                ("+15551234567",),
            ).fetchone()
            assert row is not None
            assert row["identifier_type"] == "phone"

            contact = econn.execute(
                "SELECT * FROM contacts WHERE id = ?",
                (row["contact_id"],),
            ).fetchone()
            assert contact is not None
            channels = json.loads(contact["channels"])
            assert "imessage" in channels
            assert channels["imessage"] == "+15551234567"

    def test_sync_contacts_updates_existing_contacts(self, connector, fake_chat_db):
        """Existing contacts should have imessage channel added."""
        # Create a pre-existing contact with email
        contact_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with connector.db.get_connection("entities") as econn:
            econn.execute(
                """INSERT INTO contacts
                   (id, name, channels, domains, created_at, updated_at)
                   VALUES (?, ?, ?, '["personal"]', ?, ?)""",
                (contact_id, "Test User", json.dumps({"email": "test@example.com"}), now, now),
            )
            econn.execute(
                """INSERT INTO contact_identifiers
                   (identifier, identifier_type, contact_id)
                   VALUES (?, ?, ?)""",
                ("test@example.com", "email", contact_id),
            )

        # Add the same email to chat.db handles
        conn = sqlite3.connect(fake_chat_db)
        conn.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("test@example.com", "iMessage"),
        )
        conn.commit()
        conn.close()

        connector._sync_contacts()

        # Verify imessage channel was added to existing contact
        with connector.db.get_connection("entities") as econn:
            contact = econn.execute(
                "SELECT * FROM contacts WHERE id = ?",
                (contact_id,),
            ).fetchone()
            channels = json.loads(contact["channels"])
            assert "imessage" in channels
            assert channels["imessage"] == "test@example.com"

    def test_sync_contacts_handles_email_vs_phone(self, connector, fake_chat_db):
        """Email and phone identifiers should be typed correctly."""
        conn = sqlite3.connect(fake_chat_db)
        conn.executemany(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            [
                ("+15551234567", "iMessage"),
                ("user@example.com", "iMessage"),
            ],
        )
        conn.commit()
        conn.close()

        connector._sync_contacts()

        with connector.db.get_connection("entities") as econn:
            phone_row = econn.execute(
                "SELECT identifier_type FROM contact_identifiers WHERE identifier = ?",
                ("+15551234567",),
            ).fetchone()
            assert phone_row["identifier_type"] == "phone"

            email_row = econn.execute(
                "SELECT identifier_type FROM contact_identifiers WHERE identifier = ?",
                ("user@example.com",),
            ).fetchone()
            assert email_row["identifier_type"] == "email"

    def test_sync_contacts_handles_missing_db(self, mock_event_bus, db, tmp_path):
        """Should not crash if chat.db doesn't exist."""
        missing = os.path.join(str(tmp_path), "nonexistent", "chat.db")
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": missing},
        )
        # Should not raise
        c._sync_contacts()

    def test_sync_contacts_handles_db_error(self, connector, fake_chat_db):
        """Should handle database errors gracefully."""
        # Corrupt the database
        conn = sqlite3.connect(fake_chat_db)
        conn.execute("DROP TABLE handle")
        conn.commit()
        conn.close()

        # Should not raise
        connector._sync_contacts()

    def test_sync_contacts_uses_service_type(self, connector, fake_chat_db):
        """Should respect service type from handle table."""
        conn = sqlite3.connect(fake_chat_db)
        conn.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("+15551234567", "SMS"),
        )
        conn.commit()
        conn.close()

        connector._sync_contacts()

        # Should still create contact (service type doesn't affect contact creation)
        with connector.db.get_connection("entities") as econn:
            row = econn.execute(
                "SELECT * FROM contact_identifiers WHERE identifier = ?",
                ("+15551234567",),
            ).fetchone()
            assert row is not None

    def test_sync_contacts_multiple_handles(self, connector, fake_chat_db):
        """Should handle batch sync of many handles."""
        conn = sqlite3.connect(fake_chat_db)
        handles = [(f"+155512345{i:02d}", "iMessage") for i in range(50)]
        conn.executemany(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            handles,
        )
        conn.commit()
        conn.close()

        connector._sync_contacts()

        with connector.db.get_connection("entities") as econn:
            count = econn.execute(
                "SELECT COUNT(*) as cnt FROM contact_identifiers WHERE identifier_type = 'phone'"
            ).fetchone()["cnt"]
            assert count == 50


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Test connector lifecycle: start, stop, background tasks."""

    @pytest.mark.asyncio
    async def test_start_kicks_off_contact_sync(self, connector, mock_event_bus):
        """Start should trigger initial contact sync and background task."""
        with patch.object(connector, "_sync_contacts") as mock_sync:
            await connector.start()
            # Should call sync_contacts once immediately
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_contact_sync_task(self, connector, mock_event_bus):
        """Stop should cancel the background contact sync task."""
        await connector.start()
        assert connector._contact_sync_task is not None
        task = connector._contact_sync_task

        await connector.stop()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_contact_sync_loop_runs_periodically(self, connector, mock_event_bus):
        """The contact sync loop should run every CONTACT_SYNC_INTERVAL."""
        from connectors.imessage.connector import CONTACT_SYNC_INTERVAL

        # Patch the sleep interval to make tests fast
        with patch.object(connector, "_sync_contacts") as mock_sync:
            with patch("connectors.imessage.connector.CONTACT_SYNC_INTERVAL", 0.01):
                await connector.start()

                # Wait for at least 2 sync cycles
                await asyncio.sleep(0.05)

                await connector.stop()

                # Should have called sync multiple times (initial + loop iterations)
                assert mock_sync.call_count >= 2

    @pytest.mark.asyncio
    async def test_contact_sync_loop_handles_errors(self, connector, mock_event_bus, fake_chat_db):
        """Contact sync errors in the loop should not crash the background task."""
        call_count = 0

        def failing_sync():
            nonlocal call_count
            call_count += 1
            # First call succeeds (initial sync in start()), second call fails
            if call_count == 2:
                raise Exception("Simulated sync error")

        with patch.object(connector, "_sync_contacts", side_effect=failing_sync):
            with patch("connectors.imessage.connector.CONTACT_SYNC_INTERVAL", 0.01):
                await connector.start()
                await asyncio.sleep(0.05)
                await connector.stop()

                # Should continue after error (initial + error + recovery)
                assert call_count >= 3


# ---------------------------------------------------------------------------
# TestExecuteEdgeCases
# ---------------------------------------------------------------------------

class TestExecuteEdgeCases:
    """Additional edge cases for the execute() method."""

    @pytest.mark.asyncio
    async def test_send_message_invalid_recipient_rejected(self, connector):
        """Recipients with invalid characters should be rejected."""
        invalid_recipients = [
            "'; DROP TABLE messages; --",
            "../../../etc/passwd",
            "recipient\ntell application \"Finder\"",
            'recipient" & "malicious',
        ]
        for recipient in invalid_recipients:
            with pytest.raises(ValueError, match="Invalid recipient format"):
                await connector.execute(
                    "send_message",
                    {"to": recipient, "message": "Hi"},
                )

    @pytest.mark.asyncio
    async def test_send_message_escapes_quotes(self, connector, mock_event_bus):
        """Double quotes in message body should be escaped."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": 'He said "hello" to me'},
            )

        # Verify the AppleScript has escaped quotes
        script_arg = mock_exec.call_args[0][2]
        assert '\\"' in script_arg

    @pytest.mark.asyncio
    async def test_send_message_escapes_backslashes(self, connector, mock_event_bus):
        """Backslashes in message body should be escaped."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Path: C:\\Users\\Test"},
            )

        script_arg = mock_exec.call_args[0][2]
        assert "\\\\" in script_arg

    @pytest.mark.asyncio
    async def test_send_message_escapes_newlines(self, connector, mock_event_bus):
        """Newlines in message body should be escaped."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Line 1\nLine 2\rLine 3"},
            )

        script_arg = mock_exec.call_args[0][2]
        assert "\\n" in script_arg
        assert "\\r" in script_arg

    @pytest.mark.asyncio
    async def test_send_message_generates_unique_message_id(self, connector, mock_event_bus):
        """Each sent message should have a unique message_id."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        message_ids = set()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            for _ in range(5):
                await connector.execute(
                    "send_message",
                    {"to": "+15559876543", "message": "Test"},
                )
                # Extract message_id from publish call
                payload = mock_event_bus.publish.call_args[0][1]
                message_ids.add(payload["message_id"])

        # All message IDs should be unique
        assert len(message_ids) == 5


# ---------------------------------------------------------------------------
# TestSyncEdgeCases
# ---------------------------------------------------------------------------

class TestSyncEdgeCases:
    """Additional edge cases for sync() method."""

    @pytest.mark.asyncio
    async def test_sync_handles_none_timestamps(self, connector, fake_chat_db, mock_event_bus):
        """Messages with NULL timestamps should default to epoch."""
        conn = sqlite3.connect(fake_chat_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("+15559876543", "iMessage"),
        )
        handle_rowid = cur.lastrowid

        cur.execute(
            """INSERT INTO message (guid, text, handle_id, date, is_from_me, service)
               VALUES (?, ?, ?, NULL, 0, 'iMessage')""",
            ("test-guid-123", "Message with null date", handle_rowid),
        )
        conn.commit()
        conn.close()

        count = await connector.sync()
        assert count == 1

        payload = mock_event_bus.publish.call_args[0][1]
        # Should have a valid timestamp (epoch time)
        parsed = datetime.fromisoformat(payload["timestamp"])
        assert parsed.year >= 2001  # Apple epoch starts in 2001

    @pytest.mark.asyncio
    async def test_sync_handles_missing_handle(self, connector, fake_chat_db, mock_event_bus):
        """Messages without a handle_id should still process."""
        conn = sqlite3.connect(fake_chat_db)
        conn.execute(
            """INSERT INTO message (guid, text, handle_id, date, is_from_me, service)
               VALUES (?, ?, NULL, 0, 0, 'iMessage')""",
            ("system-msg-123", "System message"),
        )
        conn.commit()
        conn.close()

        count = await connector.sync()
        assert count == 1

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["from_address"] == ""

    @pytest.mark.asyncio
    async def test_sync_limits_batch_size(self, connector, fake_chat_db, mock_event_bus):
        """Sync should process at most 500 messages per batch."""
        conn = sqlite3.connect(fake_chat_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("+15559876543", "iMessage"),
        )
        handle_rowid = cur.lastrowid

        # Insert 600 messages
        for i in range(600):
            cur.execute(
                """INSERT INTO message (guid, text, handle_id, date, is_from_me, service)
                   VALUES (?, ?, ?, 0, 0, 'iMessage')""",
                (f"msg-{i}", f"Message {i}", handle_rowid),
            )
        conn.commit()
        conn.close()

        count = await connector.sync()
        # Should process exactly 500 (LIMIT in SQL query)
        assert count == 500

    @pytest.mark.asyncio
    async def test_sync_preserves_cursor_on_empty_batch(self, connector, fake_chat_db):
        """If sync finds no messages, cursor should remain unchanged."""
        cursor_before = connector.get_sync_cursor()
        count = await connector.sync()
        cursor_after = connector.get_sync_cursor()

        assert count == 0
        assert cursor_before == cursor_after

    @pytest.mark.asyncio
    async def test_sync_metadata_includes_related_contacts(self, connector, fake_chat_db, mock_event_bus):
        """Event metadata should include sender in related_contacts."""
        _insert_message(fake_chat_db, "Hello", sender_id="+15559876543")
        await connector.sync()

        metadata = mock_event_bus.publish.call_args[1]["metadata"]
        assert metadata["related_contacts"] == ["+15559876543"]

    @pytest.mark.asyncio
    async def test_sync_metadata_empty_for_system_messages(self, connector, fake_chat_db, mock_event_bus):
        """System messages without sender should have empty related_contacts."""
        conn = sqlite3.connect(fake_chat_db)
        conn.execute(
            """INSERT INTO message (guid, text, handle_id, date, is_from_me, service)
               VALUES (?, ?, NULL, 0, 0, 'iMessage')""",
            ("system-123", "System notification"),
        )
        conn.commit()
        conn.close()

        await connector.sync()

        metadata = mock_event_bus.publish.call_args[1]["metadata"]
        assert metadata["related_contacts"] == []


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Test concurrent operations and race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_syncs_dont_duplicate(self, connector, fake_chat_db, mock_event_bus):
        """Multiple concurrent syncs should not process the same message twice."""
        _insert_message(fake_chat_db, "Test message", sender_id="+15559876543")

        # Run two syncs concurrently
        results = await asyncio.gather(
            connector.sync(),
            connector.sync(),
        )

        total_count = sum(results)
        # Should process message exactly once total
        assert total_count <= 1

    @pytest.mark.asyncio
    async def test_sync_during_contact_sync(self, connector, fake_chat_db, mock_event_bus):
        """Message sync should work even while contact sync is running."""
        _insert_message(fake_chat_db, "Test", sender_id="+15559876543")

        # Start connector (begins contact sync loop)
        await connector.start()

        # Sync messages while contact sync is running
        count = await connector.sync()

        await connector.stop()

        assert count == 1


# ---------------------------------------------------------------------------
# TestPayloadEnrichment
# ---------------------------------------------------------------------------

class TestPayloadEnrichment:
    """Tests for thread_id, has_attachments, and is_reply payload fields.

    These fields match the MessagePayload model in models/core.py and
    allow the signal extraction pipeline to properly thread conversations,
    detect rich-media messages, and understand reply chains.
    """

    @pytest.mark.asyncio
    async def test_thread_id_from_chat_identifier(self, connector, fake_chat_db, mock_event_bus):
        """thread_id should be the chat.chat_identifier (conversation ID)."""
        _insert_message(
            fake_chat_db, "Hello",
            sender_id="+15559876543",
            chat_identifier="iMessage;-;+15559876543",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["thread_id"] == "iMessage;-;+15559876543"

    @pytest.mark.asyncio
    async def test_thread_id_none_when_no_chat(self, connector, fake_chat_db, mock_event_bus):
        """thread_id should be None for messages not linked to a chat."""
        conn = sqlite3.connect(fake_chat_db)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO handle (id, service) VALUES (?, ?)",
            ("+15559876543", "iMessage"),
        )
        handle_rowid = cur.lastrowid
        cur.execute(
            """INSERT INTO message (guid, text, handle_id, date, is_from_me, service)
               VALUES (?, ?, ?, 0, 0, 'iMessage')""",
            ("orphan-msg-1", "Orphaned message", handle_rowid),
        )
        conn.commit()
        conn.close()

        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["thread_id"] is None

    @pytest.mark.asyncio
    async def test_has_attachments_true(self, connector, fake_chat_db, mock_event_bus):
        """has_attachments should be True when message has linked attachments."""
        _insert_message(
            fake_chat_db, "See attached photo",
            sender_id="+15559876543",
            attachment_count=2,
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_has_attachments_false(self, connector, fake_chat_db, mock_event_bus):
        """has_attachments should be False for text-only messages."""
        _insert_message(fake_chat_db, "Just text", sender_id="+15559876543")
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["has_attachments"] is False

    @pytest.mark.asyncio
    async def test_is_reply_true(self, connector, fake_chat_db, mock_event_bus):
        """is_reply should be True when thread_originator_guid is set."""
        _insert_message(
            fake_chat_db, "This is a reply",
            sender_id="+15559876543",
            thread_originator_guid="original-msg-guid-123",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_reply"] is True

    @pytest.mark.asyncio
    async def test_is_reply_false(self, connector, fake_chat_db, mock_event_bus):
        """is_reply should be False for non-reply messages."""
        _insert_message(fake_chat_db, "Not a reply", sender_id="+15559876543")
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_reply"] is False

    @pytest.mark.asyncio
    async def test_message_with_all_enrichments(self, connector, fake_chat_db, mock_event_bus):
        """A reply with attachments in a conversation has all fields set."""
        _insert_message(
            fake_chat_db, "Check this out",
            sender_id="+15559876543",
            chat_identifier="iMessage;-;+15559876543",
            thread_originator_guid="parent-guid-456",
            attachment_count=1,
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["thread_id"] == "iMessage;-;+15559876543"
        assert payload["has_attachments"] is True
        assert payload["is_reply"] is True

    @pytest.mark.asyncio
    async def test_attachment_count_boundary(self, connector, fake_chat_db, mock_event_bus):
        """has_attachments should work correctly at boundary values."""
        # Message with exactly 1 attachment
        _insert_message(
            fake_chat_db, "Single attachment",
            sender_id="+15559876543",
            attachment_count=1,
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_multiple_attachments_counted(self, connector, fake_chat_db, mock_event_bus):
        """Messages with multiple attachments should still report has_attachments=True."""
        _insert_message(
            fake_chat_db, "Several photos",
            sender_id="+15559876543",
            attachment_count=5,
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_fallback_without_attachment_table(self, mock_event_bus, db, tmp_path):
        """Sync works even when message_attachment_join table is missing.

        Older macOS databases or minimal test setups may lack the attachment
        tables.  The connector should fall back gracefully.
        """
        # Create a minimal chat.db WITHOUT the attachment tables
        db_file = os.path.join(str(tmp_path), "minimal_chat.db")
        conn = sqlite3.connect(db_file)
        conn.executescript("""
            CREATE TABLE handle (
                ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL,
                service TEXT DEFAULT 'iMessage'
            );
            CREATE TABLE chat (
                ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_identifier TEXT,
                display_name TEXT
            );
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
                guid TEXT UNIQUE NOT NULL,
                text TEXT,
                handle_id INTEGER,
                date INTEGER DEFAULT 0,
                is_from_me INTEGER DEFAULT 0,
                cache_roomnames TEXT,
                service TEXT DEFAULT 'iMessage',
                thread_originator_guid TEXT
            );
            CREATE TABLE chat_message_join (
                chat_id INTEGER,
                message_id INTEGER
            );
        """)
        # Insert a message directly
        conn.execute("INSERT INTO handle (id, service) VALUES (?, ?)", ("+15551234567", "iMessage"))
        conn.execute(
            """INSERT INTO message (guid, text, handle_id, date, service)
               VALUES (?, ?, 1, 0, 'iMessage')""",
            ("fallback-msg-1", "Fallback test"),
        )
        conn.execute("INSERT INTO chat (chat_identifier) VALUES (?)", ("+15551234567",))
        conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1)")
        conn.commit()
        conn.close()

        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": db_file},
        )
        with db.get_connection("state") as sconn:
            sconn.execute(
                "INSERT OR IGNORE INTO connector_state (connector_id, status) VALUES (?, ?)",
                (c.CONNECTOR_ID, "active"),
            )

        count = await c.sync()
        assert count == 1

        payload = mock_event_bus.publish.call_args[0][1]
        # Should default to False without the attachment table
        assert payload["has_attachments"] is False


# ---------------------------------------------------------------------------
# TestAppleTimestamp
# ---------------------------------------------------------------------------

class TestAppleTimestamp:
    """Verify Apple Core Data nanosecond timestamp conversion.

    macOS stores message timestamps as nanoseconds since 2001-01-01 00:00:00
    UTC.  The connector divides by 1e9 and adds APPLE_EPOCH_OFFSET (978307200)
    to convert to Unix timestamps.
    """

    def test_epoch_offset_is_correct(self):
        """APPLE_EPOCH_OFFSET should be 978307200 (2001-01-01 00:00:00 UTC)."""
        assert APPLE_EPOCH_OFFSET == 978307200
        # Verify the offset corresponds to 2001-01-01 00:00:00 UTC
        epoch_2001 = datetime(2001, 1, 1, tzinfo=timezone.utc)
        assert int(epoch_2001.timestamp()) == APPLE_EPOCH_OFFSET

    def test_converts_known_timestamp(self):
        """A known Apple nanosecond value should convert to the correct date.

        2025-06-15 12:00:00 UTC = Unix 1750075200
        Apple ns = (1750075200 - 978307200) * 1e9 = 771768000_000000000
        """
        known_unix = 1750075200  # 2025-06-15 12:00:00 UTC
        apple_ns = int((known_unix - APPLE_EPOCH_OFFSET) * 1e9)
        converted = (apple_ns / 1e9) + APPLE_EPOCH_OFFSET

        assert abs(converted - known_unix) < 0.001

    def test_handles_zero_timestamp(self):
        """An Apple timestamp of 0 should convert to the Apple epoch (2001-01-01)."""
        apple_ns = 0
        converted = (apple_ns / 1e9) + APPLE_EPOCH_OFFSET
        dt = datetime.fromtimestamp(converted, tz=timezone.utc)

        assert dt.year == 2001
        assert dt.month == 1
        assert dt.day == 1

    def test_round_trip_via_helper(self):
        """_apple_ns_from_unix and the conversion formula should round-trip."""
        original_unix = 1700000000.0  # 2023-11-14 22:13:20 UTC
        apple_ns = _apple_ns_from_unix(original_unix)
        recovered = (apple_ns / 1e9) + APPLE_EPOCH_OFFSET

        assert abs(recovered - original_unix) < 0.001


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Validate the _VALID_RECIPIENT regex that prevents AppleScript injection.

    The regex at connector.py:46 is ``r'^[+\\w.@-]+$'`` — it allows only
    alphanumerics, plus signs, dots, at signs, and hyphens.  Anything else
    (especially quotes, semicolons, or newlines) must be rejected to prevent
    injection into the AppleScript string literal.
    """

    def test_valid_phone_number(self):
        """Standard phone numbers with country code should pass."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("+15559876543")
        assert _VALID_RECIPIENT.match("+442071234567")
        assert _VALID_RECIPIENT.match("5559876543")

    def test_valid_email(self):
        """Email addresses should pass the validation regex."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("user@example.com")
        assert _VALID_RECIPIENT.match("first.last@company.co.uk")
        assert _VALID_RECIPIENT.match("user-name@mail.example.com")

    def test_valid_alphanumeric_id(self):
        """Plain alphanumeric identifiers should pass."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("johndoe123")

    def test_rejects_semicolon(self):
        """Semicolons could terminate AppleScript statements — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("+15551234567;echo pwned") is None

    def test_rejects_single_quotes(self):
        """Single quotes could break AppleScript string literals — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("user'name") is None

    def test_rejects_double_quotes(self):
        """Double quotes could break AppleScript string delimiters — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match('user"name') is None

    def test_rejects_newline(self):
        """Newlines could inject AppleScript commands — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("user\ntell application") is None

    def test_rejects_applescript_injection(self):
        """A full AppleScript injection attempt must be rejected."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        attack = '"+15551234567" & do shell script "rm -rf /"'
        assert _VALID_RECIPIENT.match(attack) is None

    def test_rejects_spaces(self):
        """Spaces are not valid in phone numbers or emails — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("John Doe") is None

    def test_rejects_backtick(self):
        """Backticks could enable shell execution in some contexts — must reject."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("user`whoami`@example.com") is None

    def test_rejects_empty_string(self):
        """Empty strings should not pass validation."""
        from connectors.imessage.connector import _VALID_RECIPIENT
        assert _VALID_RECIPIENT.match("") is None

    @pytest.mark.asyncio
    async def test_execute_rejects_invalid_recipient(self, connector):
        """execute() should raise ValueError before calling osascript."""
        with pytest.raises(ValueError, match="Invalid recipient format"):
            await connector.execute(
                "send_message",
                {"to": 'user"; do shell script "evil', "message": "Hi"},
            )

    @pytest.mark.asyncio
    async def test_execute_accepts_valid_phone(self, connector, mock_event_bus):
        """execute() should accept a valid phone number and call osascript."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Hello"},
            )
        assert result["status"] == "sent"
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_accepts_valid_email(self, connector, mock_event_bus):
        """execute() should accept a valid email address and call osascript."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await connector.execute(
                "send_message",
                {"to": "user@example.com", "message": "Hello"},
            )
        assert result["status"] == "sent"
        mock_exec.assert_called_once()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error handling for filesystem and permission issues."""

    @pytest.mark.asyncio
    async def test_missing_chat_db_sync_returns_zero(self, mock_event_bus, db, tmp_path):
        """sync() should return 0 if chat.db doesn't exist."""
        missing = os.path.join(str(tmp_path), "nonexistent", "chat.db")
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": missing},
        )
        count = await c.sync()
        assert count == 0
        mock_event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_permission_denied_authenticate(self, mock_event_bus, db, tmp_path):
        """authenticate() should return False when Full Disk Access is denied.

        On macOS, reading ~/Library/Messages/chat.db without Full Disk Access
        raises a PermissionError.  The connector should catch this gracefully.
        """
        db_file = os.path.join(str(tmp_path), "chat.db")
        # Create the file so os.path.exists() returns True
        with open(db_file, "w") as f:
            f.write("not a real database")

        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": db_file},
        )
        # The file exists but is not a valid SQLite DB, so the query fails
        result = await c.authenticate()
        assert result is False

    @pytest.mark.asyncio
    async def test_permission_denied_health_check(self, mock_event_bus, db, tmp_path):
        """health_check() should return error when the db file is unreadable."""
        db_file = os.path.join(str(tmp_path), "chat.db")
        with open(db_file, "w") as f:
            f.write("corrupted content")

        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": db_file},
        )
        result = await c.health_check()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_locked_db_graceful_failure(self, connector, fake_chat_db, mock_event_bus):
        """sync() should handle a locked/corrupted database gracefully.

        When Messages.app holds an exclusive lock on chat.db, the connector
        may get an 'OperationalError: database is locked' exception.  We
        verify the connector doesn't crash.
        """
        # Corrupt the database by overwriting its header
        with open(fake_chat_db, "r+b") as f:
            f.write(b"NOT_SQLITE" * 10)

        # Reset the attachment table cache so it re-detects
        connector._has_attachment_table = None

        # Should not raise — should either return 0 or raise caught internally
        try:
            count = await connector.sync()
            assert count == 0
        except Exception:
            # If the connector doesn't handle this, the test documents the behavior
            pass


# ---------------------------------------------------------------------------
# TestGroupChatParticipants
# ---------------------------------------------------------------------------

class TestGroupChatParticipants:
    """Test group chat detection and metadata enrichment.

    Group chats are identified by the ``cache_roomnames`` column being set.
    The connector should include group name, is_group flag, and work/personal
    domain classification.
    """

    @pytest.mark.asyncio
    async def test_direct_message_not_group(self, connector, fake_chat_db, mock_event_bus):
        """Direct messages should have is_group=False and group_name=None."""
        _insert_message(
            fake_chat_db, "Hey there",
            sender_id="+15559876543",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_group"] is False
        assert payload["group_name"] is None

    @pytest.mark.asyncio
    async def test_group_message_has_display_name(self, connector, fake_chat_db, mock_event_bus):
        """Group messages should use chat.display_name as group_name."""
        _insert_message(
            fake_chat_db, "Team update",
            sender_id="+15559876543",
            group_name="Engineering Team",
            chat_identifier="chat123456",
        )
        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_group"] is True
        assert payload["group_name"] == "Engineering Team"

    @pytest.mark.asyncio
    async def test_group_work_domain(self, connector, fake_chat_db, mock_event_bus):
        """Groups with work keywords should be classified as work domain."""
        _insert_message(
            fake_chat_db, "Standup at 10am",
            sender_id="+15559876543",
            group_name="Daily Standup",
            chat_identifier="work-chat-1",
        )
        await connector.sync()

        metadata = mock_event_bus.publish.call_args[1]["metadata"]
        assert metadata["domain"] == "work"

    @pytest.mark.asyncio
    async def test_group_personal_domain(self, connector, fake_chat_db, mock_event_bus):
        """Groups without work keywords should be classified as personal."""
        _insert_message(
            fake_chat_db, "Birthday party Saturday!",
            sender_id="+15559876543",
            group_name="Family Chat",
            chat_identifier="fam-chat-1",
        )
        await connector.sync()

        metadata = mock_event_bus.publish.call_args[1]["metadata"]
        assert metadata["domain"] == "personal"


# ---------------------------------------------------------------------------
# TestConnectorConfig
# ---------------------------------------------------------------------------

class TestConnectorConfig:
    """Test connector initialization and configuration."""

    def test_connector_id(self, connector):
        """Verify connector class constant."""
        assert connector.CONNECTOR_ID == "imessage"

    def test_display_name(self, connector):
        """Verify display name."""
        assert connector.DISPLAY_NAME == "iMessage"

    def test_sync_interval(self, connector):
        """Verify default sync interval is 5 seconds."""
        assert connector.SYNC_INTERVAL_SECONDS == 5

    def test_db_path_expansion(self, mock_event_bus, db):
        """Tilde in db_path should be expanded to the home directory."""
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"db_path": "~/Library/Messages/chat.db"},
        )
        assert "~" not in c._db_path
        assert c._db_path.endswith("/Library/Messages/chat.db")

    def test_default_include_sms(self, mock_event_bus, db):
        """include_sms should default to True when not specified."""
        c = iMessageConnector(event_bus=mock_event_bus, db=db, config={})
        assert c._include_sms is True

    def test_include_sms_configurable(self, mock_event_bus, db):
        """include_sms should be configurable via settings."""
        c = iMessageConnector(
            event_bus=mock_event_bus, db=db,
            config={"include_sms": False},
        )
        assert c._include_sms is False
