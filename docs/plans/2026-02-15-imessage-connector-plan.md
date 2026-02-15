# iMessage Connector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bidirectional iMessage connector that reads from macOS `chat.db` and sends via AppleScript.

**Architecture:** Standard `BaseConnector` subclass. Inbound sync polls `~/Library/Messages/chat.db` (read-only SQLite) every 5 seconds using `message.ROWID` as cursor. Outbound sends use `osascript` to invoke AppleScript's Messages.app API.

**Tech Stack:** Python 3.12, SQLite3 (read-only), asyncio subprocess (osascript), pytest + pytest-asyncio

---

### Task 1: Add `IMESSAGE` to SourceType enum

**Files:**
- Modify: `models/core.py:121` (SourceType enum)

**Step 1: Add the enum value**

In `models/core.py`, add `IMESSAGE = "imessage"` to the `SourceType` enum after the existing `WHATSAPP` entry:

```python
    WHATSAPP = "whatsapp"
    IMESSAGE = "imessage"
```

**Step 2: Commit**

```bash
git add models/core.py
git commit -m "feat: add IMESSAGE to SourceType enum"
```

---

### Task 2: Register iMessage connector in registry

**Files:**
- Modify: `connectors/registry.py:89` (after Signal entry in CONNECTOR_REGISTRY)

**Step 1: Add registry entry**

Add after the `"signal"` entry (line 89) and before the `"caldav"` entry:

```python
    "imessage": ConnectorTypeDef(
        connector_id="imessage",
        display_name="iMessage",
        description="macOS iMessage and SMS via chat.db and AppleScript",
        category="api",
        module_path="connectors.imessage.connector",
        class_name="iMessageConnector",
        config_fields=[
            ConnectorFieldDef("db_path", "string",
                              default="~/Library/Messages/chat.db",
                              help_text="Path to Messages database",
                              placeholder="~/Library/Messages/chat.db"),
            ConnectorFieldDef("sync_interval", "integer", default=5,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("include_sms", "boolean", default=True,
                              help_text="Include SMS/MMS messages (not just iMessage)"),
        ],
    ),
```

**Step 2: Commit**

```bash
git add connectors/registry.py
git commit -m "feat: register imessage connector in registry"
```

---

### Task 3: Create connector module with `__init__.py`

**Files:**
- Create: `connectors/imessage/__init__.py`

**Step 1: Create the file**

```python
"""iMessage connector — integrates via macOS chat.db and AppleScript."""
```

**Step 2: Commit**

```bash
git add connectors/imessage/__init__.py
git commit -m "feat: create imessage connector package"
```

---

### Task 4: Write tests for iMessage connector

**Files:**
- Create: `tests/test_imessage_connector.py`

**Context:** Tests create a fake `chat.db` in a temp directory with the same schema as the real Messages database. This avoids needing Full Disk Access in CI. The connector reads from this fake DB instead of `~/Library/Messages/chat.db`.

We mock the EventBus since NATS isn't available in tests. The `db` fixture from `conftest.py` provides a real DatabaseManager with temp SQLite databases.

**Step 1: Write the test file**

```python
"""
Tests for the iMessage connector.

Creates a fake chat.db with the same schema as macOS Messages to test
sync logic without needing Full Disk Access.
"""

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from connectors.imessage.connector import iMessageConnector


# -- Helpers ------------------------------------------------------------------

# Apple epoch offset: seconds between 1970-01-01 and 2001-01-01
APPLE_EPOCH_OFFSET = 978307200


def _apple_timestamp(dt: datetime) -> int:
    """Convert a datetime to Apple's nanosecond-since-2001 timestamp."""
    unix_ts = dt.timestamp()
    return int((unix_ts - APPLE_EPOCH_OFFSET) * 1_000_000_000)


def _create_fake_chat_db(path: str) -> sqlite3.Connection:
    """Create a minimal chat.db with the tables the connector reads."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS handle (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL,
            service TEXT DEFAULT 'iMessage'
        );

        CREATE TABLE IF NOT EXISTS chat (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_identifier TEXT,
            display_name TEXT,
            service_name TEXT DEFAULT 'iMessage'
        );

        CREATE TABLE IF NOT EXISTS message (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT UNIQUE,
            text TEXT,
            handle_id INTEGER DEFAULT 0,
            date INTEGER DEFAULT 0,
            is_from_me INTEGER DEFAULT 0,
            cache_roomnames TEXT,
            service TEXT DEFAULT 'iMessage',
            FOREIGN KEY (handle_id) REFERENCES handle(ROWID)
        );

        CREATE TABLE IF NOT EXISTS chat_message_join (
            chat_id INTEGER,
            message_id INTEGER,
            PRIMARY KEY (chat_id, message_id)
        );
    """)
    conn.commit()
    return conn


def _insert_message(
    chat_db: sqlite3.Connection,
    text: str,
    sender_id: str = "+15551234567",
    service: str = "iMessage",
    is_from_me: int = 0,
    dt: datetime | None = None,
    group_name: str | None = None,
    chat_identifier: str | None = None,
) -> int:
    """Insert a message into the fake chat.db and return its ROWID."""
    dt = dt or datetime.now(timezone.utc)
    apple_ts = _apple_timestamp(dt)
    guid = str(uuid.uuid4())

    # Ensure handle exists
    chat_db.execute(
        "INSERT OR IGNORE INTO handle (id, service) VALUES (?, ?)",
        (sender_id, service),
    )
    handle_row = chat_db.execute(
        "SELECT ROWID FROM handle WHERE id = ?", (sender_id,)
    ).fetchone()
    handle_rowid = handle_row[0]

    # Insert message
    chat_db.execute(
        """INSERT INTO message (guid, text, handle_id, date, is_from_me,
                                cache_roomnames, service)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (guid, text, handle_rowid, apple_ts, is_from_me, group_name, service),
    )
    msg_rowid = chat_db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Create/find chat and link
    ci = chat_identifier or (f"chat{sender_id}" if group_name else sender_id)
    chat_db.execute(
        "INSERT OR IGNORE INTO chat (chat_identifier, display_name, service_name) VALUES (?, ?, ?)",
        (ci, group_name or "", service),
    )
    chat_row = chat_db.execute(
        "SELECT ROWID FROM chat WHERE chat_identifier = ?", (ci,)
    ).fetchone()
    chat_db.execute(
        "INSERT OR IGNORE INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
        (chat_row[0], msg_rowid),
    )

    chat_db.commit()
    return msg_rowid


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture()
def fake_chat_db(tmp_path):
    """Create a fake chat.db and return its path."""
    db_path = str(tmp_path / "chat.db")
    conn = _create_fake_chat_db(db_path)
    yield db_path, conn
    conn.close()


@pytest.fixture()
def mock_event_bus():
    """A mock EventBus that records published events."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value="event-id-123")
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture()
def connector(db, mock_event_bus, fake_chat_db):
    """An iMessageConnector wired to fake chat.db and mock event bus."""
    db_path, _ = fake_chat_db
    config = {"db_path": db_path, "sync_interval": 5, "include_sms": True}
    return iMessageConnector(event_bus=mock_event_bus, db=db, config=config)


# -- Tests --------------------------------------------------------------------

class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_success(self, connector):
        """authenticate() returns True when chat.db exists and is readable."""
        result = await connector.authenticate()
        assert result is True

    @pytest.mark.asyncio
    async def test_authenticate_missing_db(self, db, mock_event_bus):
        """authenticate() returns False when chat.db doesn't exist."""
        config = {"db_path": "/nonexistent/chat.db"}
        c = iMessageConnector(event_bus=mock_event_bus, db=db, config=config)
        result = await c.authenticate()
        assert result is False


class TestSync:
    @pytest.mark.asyncio
    async def test_sync_no_messages(self, connector):
        """sync() returns 0 when chat.db has no messages."""
        count = await connector.sync()
        assert count == 0

    @pytest.mark.asyncio
    async def test_sync_inbound_message(self, connector, fake_chat_db, mock_event_bus):
        """sync() publishes message.received for inbound messages."""
        _, chat_db = fake_chat_db
        _insert_message(chat_db, "Hello from friend", sender_id="+15559876543")

        count = await connector.sync()
        assert count == 1

        # Verify event was published
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "message.received"
        payload = call_args[0][1]
        assert payload["channel"] == "imessage"
        assert payload["direction"] == "inbound"
        assert payload["from_address"] == "+15559876543"
        assert payload["body"] == "Hello from friend"

    @pytest.mark.asyncio
    async def test_sync_outbound_message(self, connector, fake_chat_db, mock_event_bus):
        """sync() publishes message.sent for outbound messages."""
        _, chat_db = fake_chat_db
        _insert_message(chat_db, "My reply", sender_id="+15559876543", is_from_me=1)

        count = await connector.sync()
        assert count == 1

        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == "message.sent"
        payload = call_args[0][1]
        assert payload["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_sync_incremental_cursor(self, connector, fake_chat_db, mock_event_bus):
        """sync() uses cursor to only process new messages on subsequent calls."""
        _, chat_db = fake_chat_db
        _insert_message(chat_db, "First message")

        count1 = await connector.sync()
        assert count1 == 1

        # Second sync should find nothing new
        count2 = await connector.sync()
        assert count2 == 0

        # New message should be picked up
        _insert_message(chat_db, "Second message", sender_id="+15550001111")
        count3 = await connector.sync()
        assert count3 == 1

    @pytest.mark.asyncio
    async def test_sync_group_message(self, connector, fake_chat_db, mock_event_bus):
        """sync() correctly identifies group messages."""
        _, chat_db = fake_chat_db
        _insert_message(
            chat_db, "Group hello", sender_id="+15551112222",
            group_name="Family Chat", chat_identifier="chat123456",
        )

        await connector.sync()

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["is_group"] is True
        assert payload["group_name"] == "Family Chat"

    @pytest.mark.asyncio
    async def test_sync_sms_message(self, connector, fake_chat_db, mock_event_bus):
        """sync() includes SMS messages when include_sms is True."""
        _, chat_db = fake_chat_db
        _insert_message(chat_db, "SMS text", service="SMS")

        count = await connector.sync()
        assert count == 1

        payload = mock_event_bus.publish.call_args[0][1]
        assert payload["service_type"] == "SMS"

    @pytest.mark.asyncio
    async def test_sync_excludes_sms_when_disabled(self, db, mock_event_bus, fake_chat_db):
        """sync() skips SMS messages when include_sms is False."""
        db_path, chat_db = fake_chat_db
        config = {"db_path": db_path, "include_sms": False}
        c = iMessageConnector(event_bus=mock_event_bus, db=db, config=config)

        _insert_message(chat_db, "SMS text", service="SMS")
        _insert_message(chat_db, "iMessage text", service="iMessage")

        count = await c.sync()
        assert count == 1  # Only the iMessage

    @pytest.mark.asyncio
    async def test_sync_skips_empty_messages(self, connector, fake_chat_db, mock_event_bus):
        """sync() skips messages with no text (e.g. reactions, typing indicators)."""
        _, chat_db = fake_chat_db
        _insert_message(chat_db, None)  # No text
        _insert_message(chat_db, "")    # Empty text

        count = await connector.sync()
        assert count == 0


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, connector):
        """execute() raises ValueError for unknown actions."""
        with pytest.raises(ValueError, match="Unknown action"):
            await connector.execute("unknown_action", {})


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_ok(self, connector):
        """health_check() returns ok when db exists."""
        result = await connector.health_check()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_missing_db(self, db, mock_event_bus):
        """health_check() returns error when db missing."""
        config = {"db_path": "/nonexistent/chat.db"}
        c = iMessageConnector(event_bus=mock_event_bus, db=db, config=config)
        result = await c.health_check()
        assert result["status"] == "error"
```

**Step 2: Run the tests to verify they fail**

Run: `cd /Users/jeremygreenwood/life-os && source .venv/bin/activate && python -m pytest tests/test_imessage_connector.py -v 2>&1 | head -30`

Expected: FAIL with `ModuleNotFoundError: No module named 'connectors.imessage'`

**Step 3: Commit**

```bash
git add tests/test_imessage_connector.py
git commit -m "test: add iMessage connector tests"
```

---

### Task 5: Implement iMessage connector

**Files:**
- Create: `connectors/imessage/connector.py`

**Context:** This is the main connector implementation. It reads from the macOS Messages SQLite database for inbound sync, and sends via AppleScript for outbound execution. Key details:
- Apple epoch: timestamps in `chat.db` are nanoseconds since 2001-01-01 (offset 978307200 seconds from Unix epoch)
- Use `ROWID` as the sync cursor (monotonically increasing integer)
- Open the database read-only (`?mode=ro` URI)
- Group chats have `cache_roomnames` set on the message
- `handle.id` contains the phone number or email of the sender
- `handle.service` is `"iMessage"` or `"SMS"`
- `message.is_from_me` flag distinguishes direction

**Step 1: Write the connector**

```python
"""
Life OS -- iMessage Connector

Reads messages from macOS Messages database (chat.db) and sends via AppleScript.

Requirements:
    - macOS with Messages app
    - Full Disk Access granted to Python/Terminal (System Settings > Privacy)
    - For sending: Automation permission for Messages.app (prompted on first use)

Configuration (in settings.yaml):
    connectors:
      imessage:
        db_path: "~/Library/Messages/chat.db"
        sync_interval: 5
        include_sms: true
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

# Seconds between 1970-01-01 (Unix epoch) and 2001-01-01 (Apple epoch)
APPLE_EPOCH_OFFSET = 978307200

# How often to re-sync contacts (seconds)
CONTACT_SYNC_INTERVAL = 3600  # 1 hour


class iMessageConnector(BaseConnector):
    """Connector that reads from macOS Messages database and sends via AppleScript.

    Integration pattern:
        - Inbound: poll ~/Library/Messages/chat.db (read-only SQLite) for new
          messages using message.ROWID as an incremental cursor.
        - Outbound: send messages via osascript calling AppleScript's Messages API.
        - Both iMessage and SMS/MMS are supported (configurable via include_sms).
        - Group chats are detected via cache_roomnames on the message row.
    """

    CONNECTOR_ID = "imessage"
    DISPLAY_NAME = "iMessage"
    SYNC_INTERVAL_SECONDS = 5

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._db_path = os.path.expanduser(
            config.get("db_path", "~/Library/Messages/chat.db")
        )
        self._include_sms = config.get("include_sms", True)
        self._contact_sync_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the connector and kick off initial contact sync."""
        await super().start()
        if self._running:
            await self._sync_contacts()
            self._contact_sync_task = asyncio.create_task(self._contact_sync_loop())

    async def stop(self):
        if self._contact_sync_task:
            self._contact_sync_task.cancel()
            try:
                await self._contact_sync_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def _contact_sync_loop(self):
        """Re-sync contacts every CONTACT_SYNC_INTERVAL seconds."""
        while self._running:
            await asyncio.sleep(CONTACT_SYNC_INTERVAL)
            try:
                await self._sync_contacts()
            except Exception as e:
                print(f"[imessage] Contact sync error: {e}")

    # ------------------------------------------------------------------
    # BaseConnector overrides
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        """Verify that the Messages database exists and is readable."""
        if not os.path.exists(self._db_path):
            print(f"[imessage] Database not found: {self._db_path}")
            return False
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.execute("SELECT COUNT(*) FROM message")
            conn.close()
            return True
        except Exception as e:
            print(f"[imessage] Database access denied: {e}")
            print("[imessage] Grant Full Disk Access to Terminal/Python in System Settings")
            return False

    async def sync(self) -> int:
        """Poll chat.db for new messages since last cursor."""
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except Exception:
            return 0

        try:
            cursor = self.get_sync_cursor()
            last_rowid = int(cursor) if cursor else 0

            rows = conn.execute(
                """SELECT
                        message.ROWID,
                        message.guid,
                        message.text,
                        message.date,
                        message.is_from_me,
                        message.cache_roomnames,
                        message.service,
                        handle.id AS sender_id,
                        handle.service AS handle_service,
                        chat.chat_identifier,
                        chat.display_name
                   FROM message
                   LEFT JOIN handle ON message.handle_id = handle.ROWID
                   LEFT JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
                   LEFT JOIN chat ON chat_message_join.chat_id = chat.ROWID
                   WHERE message.ROWID > ?
                   ORDER BY message.ROWID ASC
                   LIMIT 500""",
                (last_rowid,),
            ).fetchall()

            count = 0
            for row in rows:
                text = row["text"]
                if not text:
                    last_rowid = row["ROWID"]
                    continue

                service_type = row["service"] or row["handle_service"] or "iMessage"
                if not self._include_sms and service_type == "SMS":
                    last_rowid = row["ROWID"]
                    continue

                # Convert Apple nanosecond timestamp to ISO format
                apple_ns = row["date"] or 0
                unix_ts = (apple_ns / 1_000_000_000) + APPLE_EPOCH_OFFSET
                timestamp = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()

                is_from_me = bool(row["is_from_me"])
                direction = "outbound" if is_from_me else "inbound"
                event_type = "message.sent" if is_from_me else "message.received"
                sender_id = row["sender_id"] or ""

                is_group = bool(row["cache_roomnames"])
                group_name = row["cache_roomnames"] if is_group else None

                payload = {
                    "message_id": row["guid"],
                    "channel": "imessage",
                    "direction": direction,
                    "from_address": sender_id if not is_from_me else "me",
                    "to_addresses": ["me"] if not is_from_me else [sender_id],
                    "body": text,
                    "body_plain": text,
                    "snippet": text[:150],
                    "is_group": is_group,
                    "group_name": group_name,
                    "service_type": service_type,
                    "timestamp": timestamp,
                }

                metadata = {
                    "related_contacts": [sender_id] if sender_id else [],
                    "domain": self._classify_domain(group_name),
                }

                await self.publish_event(
                    event_type, payload,
                    priority="normal", metadata=metadata,
                )
                count += 1
                last_rowid = row["ROWID"]

            if last_rowid > (int(cursor) if cursor else 0):
                self.set_sync_cursor(str(last_rowid))

            return count
        finally:
            conn.close()

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a message via AppleScript."""
        if action == "send_message":
            recipient = params["to"]
            message = params["message"]

            # Escape for AppleScript string literals
            message_escaped = message.replace("\\", "\\\\").replace('"', '\\"')
            recipient_escaped = recipient.replace("\\", "\\\\").replace('"', '\\"')

            applescript = (
                'tell application "Messages"\n'
                f'    set targetBuddy to buddy "{recipient_escaped}" '
                f'of (service 1 whose service type is iMessage)\n'
                f'    send "{message_escaped}" to targetBuddy\n'
                'end tell'
            )

            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"AppleScript failed (rc={proc.returncode}): {stderr.decode().strip()}"
                )

            await self.publish_event(
                "message.sent",
                {
                    "channel": "imessage",
                    "direction": "outbound",
                    "from_address": "me",
                    "to_addresses": [recipient],
                    "body": message,
                    "body_plain": message,
                    "snippet": message[:150],
                },
            )

            return {"status": "sent", "to": recipient}

        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
        """Check that the Messages database is accessible."""
        if not os.path.exists(self._db_path):
            return {"status": "error", "details": "Messages database not found"}
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.execute("SELECT COUNT(*) FROM message")
            conn.close()
            return {"status": "ok", "connector": self.CONNECTOR_ID}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    # ------------------------------------------------------------------
    # Contact sync
    # ------------------------------------------------------------------

    async def _sync_contacts(self):
        """Sync iMessage handles to entities.db contacts."""
        try:
            msg_conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            msg_conn.row_factory = sqlite3.Row
        except Exception:
            return

        try:
            handles = msg_conn.execute(
                """SELECT DISTINCT
                       handle.id AS identifier,
                       handle.service AS service_type
                   FROM handle
                   JOIN message ON message.handle_id = handle.ROWID
                   WHERE handle.id IS NOT NULL AND handle.id != ''"""
            ).fetchall()

            now = datetime.now(timezone.utc).isoformat()
            synced = 0

            with self.db.get_connection("entities") as conn:
                for handle in handles:
                    identifier = handle["identifier"]
                    identifier_type = "phone" if identifier.startswith("+") else "email"

                    # Check if we already have this identifier
                    row = conn.execute(
                        "SELECT contact_id FROM contact_identifiers WHERE identifier = ? AND identifier_type = ?",
                        (identifier, identifier_type),
                    ).fetchone()

                    if row:
                        # Update channel info
                        conn.execute(
                            """UPDATE contacts SET
                                channels = json_set(COALESCE(channels, '{}'), '$.imessage', ?),
                                updated_at = ?
                               WHERE id = ?""",
                            (identifier, now, row["contact_id"]),
                        )
                    else:
                        # Create new contact
                        contact_id = str(uuid.uuid4())
                        name = f"Unknown ({identifier})"
                        conn.execute(
                            """INSERT INTO contacts
                                (id, name, channels, domains, created_at, updated_at)
                               VALUES (?, ?, ?, '["personal"]', ?, ?)""",
                            (
                                contact_id, name,
                                json.dumps({"imessage": identifier}),
                                now, now,
                            ),
                        )
                        conn.execute(
                            """INSERT INTO contact_identifiers
                                (identifier, identifier_type, contact_id)
                               VALUES (?, ?, ?)""",
                            (identifier, identifier_type, contact_id),
                        )
                    synced += 1

            print(f"[imessage] Synced {synced} contacts")
        finally:
            msg_conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_domain(self, group_name: Optional[str]) -> str:
        """Classify a message as work or personal based on group name."""
        if group_name:
            name_lower = group_name.lower()
            if any(w in name_lower for w in ["work", "team", "project", "standup"]):
                return "work"
        return "personal"
```

**Step 2: Run the tests**

Run: `cd /Users/jeremygreenwood/life-os && source .venv/bin/activate && python -m pytest tests/test_imessage_connector.py -v`

Expected: All tests PASS

**Step 3: Fix any failing tests and re-run**

Run: `python -m pytest tests/test_imessage_connector.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add connectors/imessage/connector.py
git commit -m "feat: implement iMessage connector with chat.db sync and AppleScript send"
```

---

### Task 6: Run full test suite

**Step 1: Run all tests**

Run: `cd /Users/jeremygreenwood/life-os && source .venv/bin/activate && python -m pytest tests/ -v`

Expected: All tests PASS, no regressions

**Step 2: Verify connector can be imported from registry**

Run: `cd /Users/jeremygreenwood/life-os && source .venv/bin/activate && python -c "from connectors.registry import get_connector_class; cls = get_connector_class('imessage'); print(f'OK: {cls.__name__}')"`

Expected: `OK: iMessageConnector`

**Step 3: Commit if any fixes were needed**

---

### Task 7: Final commit and verify

**Step 1: Verify all files are committed**

Run: `git status`

Expected: Clean working tree

**Step 2: Review the full diff**

Run: `git log --oneline -5`

Expected: 4-5 commits for enum, registry, tests, and implementation
