"""
Life OS -- iMessage Connector

Reads inbound messages from the macOS Messages database
(~/Library/Messages/chat.db) and sends outbound messages via AppleScript.

Supports:
    - iMessage and SMS (configurable via ``include_sms``)
    - 1-on-1 and group conversations
    - Incremental sync using ROWID-based cursor
    - Contact sync into entities.db

Requirements:
    - Full Disk Access for the process reading chat.db (System Settings >
      Privacy & Security > Full Disk Access)
    - Messages.app must be signed in with an Apple ID

Configuration (in settings.yaml):
    connectors:
      imessage:
        db_path: "~/Library/Messages/chat.db"
        include_sms: true
        sync_interval: 5
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Recipient addresses must look like a phone number or email — reject
# anything that could smuggle AppleScript via the ``participant`` string.
_VALID_RECIPIENT = re.compile(r'^[+\w.@-]+$')

# Apple Core Data epoch (2001-01-01 00:00:00 UTC) expressed as a Unix
# timestamp.  macOS stores message timestamps as nanoseconds since this
# epoch; to convert: ``unix_ts = (apple_ns / 1e9) + APPLE_EPOCH_OFFSET``.
APPLE_EPOCH_OFFSET = 978307200

# How often to re-sync contacts from chat.db handles (seconds).
CONTACT_SYNC_INTERVAL = 3600  # 1 hour

# SQL query to pull messages newer than a given ROWID.  The LEFT JOINs
# tolerate messages that lack a handle (system messages) or that haven't
# been linked to a chat yet.
_SYNC_QUERY = """\
SELECT
    message.ROWID,
    message.guid,
    message.text,
    message.date,
    message.is_from_me,
    message.cache_roomnames,
    message.service,
    handle.id          AS sender_id,
    handle.service     AS handle_service,
    chat.chat_identifier,
    chat.display_name
FROM message
LEFT JOIN handle            ON message.handle_id = handle.ROWID
LEFT JOIN chat_message_join ON message.ROWID     = chat_message_join.message_id
LEFT JOIN chat              ON chat_message_join.chat_id = chat.ROWID
WHERE message.ROWID > ?
ORDER BY message.ROWID ASC
LIMIT 500
"""


class iMessageConnector(BaseConnector):
    """Connector that reads from macOS chat.db and sends via AppleScript.

    Integration pattern:
        - **Inbound**: the macOS Messages database (``chat.db``) is opened
          read-only via a ``file:...?mode=ro`` SQLite URI.  Each sync cycle
          queries for messages with a ``ROWID`` greater than the persisted
          cursor, converts Apple-epoch timestamps, and publishes
          ``message.received`` or ``message.sent`` events.
        - **Outbound**: messages are sent via ``osascript`` running a short
          AppleScript that tells the Messages application to send a message
          to a given recipient.
        - Domain classification is heuristic, based on group names containing
          keywords like "work" or "team".
    """

    CONNECTOR_ID = "imessage"
    DISPLAY_NAME = "iMessage"
    SYNC_INTERVAL_SECONDS = 5

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        raw_path = config.get("db_path", "~/Library/Messages/chat.db")
        self._db_path: str = os.path.expanduser(raw_path)
        self._include_sms: bool = config.get("include_sms", True)
        self._contact_sync_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    async def start(self):
        """Start the connector and kick off a periodic contact sync."""
        await super().start()
        if self._running:
            self._sync_contacts()
            self._contact_sync_task = asyncio.create_task(self._contact_sync_loop())

    async def stop(self):
        """Stop the sync loop and the contact sync task."""
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
                self._sync_contacts()
            except Exception as e:
                logger.error("Contact sync error: %s", e)

    # ------------------------------------------------------------------
    # Core lifecycle methods
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        """Verify that chat.db exists and is readable.

        Opens the database in read-only mode and runs a trivial query to
        confirm that the file is a valid SQLite database with the expected
        ``message`` table.
        """
        if not os.path.exists(self._db_path):
            return False
        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("SELECT COUNT(*) FROM message")
            conn.close()
            return True
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    async def sync(self) -> int:
        """Pull new messages from chat.db and publish events.

        Uses a ROWID-based cursor for incremental syncing.  Each message is
        converted into either a ``message.received`` or ``message.sent``
        event depending on the ``is_from_me`` flag.

        Returns the number of events published.
        """
        if not os.path.exists(self._db_path):
            return 0

        uri = f"file:{self._db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row

            cursor_val = self.get_sync_cursor() or "0"
            last_rowid = int(cursor_val)

            rows = conn.execute(_SYNC_QUERY, (last_rowid,)).fetchall()

            count = 0
            new_last_rowid = last_rowid

            for row in rows:
                text = row["text"]
                if not text or text.strip() == "":
                    new_last_rowid = row["ROWID"]
                    continue

                service = row["service"] or "iMessage"
                if not self._include_sms and service.upper() == "SMS":
                    new_last_rowid = row["ROWID"]
                    continue

                # Convert Apple nanosecond timestamp to Unix timestamp
                apple_ns = row["date"] or 0
                unix_ts = (apple_ns / 1e9) + APPLE_EPOCH_OFFSET
                ts_iso = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()

                is_from_me = bool(row["is_from_me"])
                sender_id = row["sender_id"] or ""
                is_group = bool(row["cache_roomnames"])
                group_name = row["display_name"] or row["cache_roomnames"] or None

                direction = "outbound" if is_from_me else "inbound"
                event_type = "message.sent" if is_from_me else "message.received"

                payload = {
                    "message_id": row["guid"],
                    "channel": "imessage",
                    "direction": direction,
                    "from_address": sender_id,
                    "to_addresses": [],
                    "body": text,
                    "body_plain": text,
                    "snippet": text[:150],
                    "is_group": is_group,
                    "group_name": group_name,
                    "service_type": service,
                    "timestamp": ts_iso,
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
                new_last_rowid = row["ROWID"]
        finally:
            conn.close()

        if new_last_rowid > last_rowid:
            self.set_sync_cursor(str(new_last_rowid))

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform an outbound action.

        Supported actions:
            ``send_message`` -- send a message via AppleScript.  Expects
            ``params["to"]`` (phone number or email) and ``params["message"]``.

        Raises ``ValueError`` for unknown actions and ``RuntimeError`` if the
        AppleScript execution fails.
        """
        if action == "send_message":
            recipient = params["to"]
            message = params["message"]

            if not _VALID_RECIPIENT.match(recipient):
                raise ValueError(f"Invalid recipient format: {recipient}")

            # Escape characters that would break AppleScript string literals.
            safe_message = (message
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r"))
            safe_recipient = (recipient
                .replace("\\", "\\\\")
                .replace('"', '\\"'))

            script = (
                'tell application "Messages"\n'
                '  set targetService to 1st account whose service type = iMessage\n'
                f'  set targetBuddy to participant "{safe_recipient}" of targetService\n'
                f'  send "{safe_message}" to targetBuddy\n'
                'end tell'
            )

            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"AppleScript failed (rc={proc.returncode}): {stderr.decode().strip()}"
                )

            # Publish a message.sent event for outbound tracking.
            await self.publish_event(
                "message.sent",
                {
                    "message_id": str(uuid.uuid4()),
                    "channel": "imessage",
                    "direction": "outbound",
                    "from_address": "me",
                    "to_addresses": [recipient],
                    "body": message,
                    "body_plain": message,
                    "snippet": message[:150],
                    "service_type": "iMessage",
                },
            )

            return {"status": "sent", "to": recipient}

        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
        """Check that chat.db is accessible and readable."""
        if not os.path.exists(self._db_path):
            return {"status": "error", "details": "chat.db not found"}
        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("SELECT COUNT(*) FROM message")
            conn.close()
            return {"status": "ok", "connector": self.CONNECTOR_ID}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    # ------------------------------------------------------------------
    # Contact sync
    # ------------------------------------------------------------------

    def _sync_contacts(self):
        """Read handles from chat.db and upsert into entities.db contacts.

        Each unique handle (phone number or email) gets a contact record
        with ``imessage`` added to its channels dict.
        """
        if not os.path.exists(self._db_path):
            return

        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                conn.row_factory = sqlite3.Row
                handles = conn.execute("SELECT id, service FROM handle").fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.error("Could not read handles: %s", e)
            return

        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("entities") as econn:
            for h in handles:
                identifier = h["id"]
                service = h["service"] or "iMessage"
                identifier_type = "email" if "@" in identifier else "phone"

                # Check if this identifier already has a contact
                row = econn.execute(
                    "SELECT contact_id FROM contact_identifiers "
                    "WHERE identifier = ? AND identifier_type = ?",
                    (identifier, identifier_type),
                ).fetchone()

                if row:
                    contact_id = row["contact_id"]
                    # Add imessage channel
                    econn.execute(
                        """UPDATE contacts SET
                            channels = json_set(COALESCE(channels, '{}'), '$.imessage', ?),
                            updated_at = ?
                           WHERE id = ?""",
                        (identifier, now, contact_id),
                    )
                else:
                    contact_id = str(uuid.uuid4())
                    econn.execute(
                        """INSERT INTO contacts
                            (id, name, phones, channels, domains, created_at, updated_at)
                           VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                        (
                            contact_id,
                            f"Unknown ({identifier})",
                            json.dumps([identifier]),
                            json.dumps({"imessage": identifier}),
                            now, now,
                        ),
                    )
                    econn.execute(
                        """INSERT INTO contact_identifiers
                            (identifier, identifier_type, contact_id)
                           VALUES (?, ?, ?)
                           ON CONFLICT(identifier, identifier_type) DO UPDATE SET contact_id = ?""",
                        (identifier, identifier_type, contact_id, contact_id),
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_domain(self, group_name: str | None) -> str:
        """Classify a message as 'work' or 'personal' based on group name.

        Checks the group name for work-related keywords.  Defaults to
        ``"personal"`` when there is no group name or no keyword matches.
        """
        if group_name:
            name_lower = group_name.lower()
            if any(w in name_lower for w in ("work", "team", "project", "standup")):
                return "work"
        return "personal"
