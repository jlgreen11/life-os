"""
Life OS — Signal Messenger Connector

Connects to Signal via signal-cli running in daemon mode with JSON-RPC.

Requirements:
    - signal-cli installed and registered with a phone number
    - Running in daemon mode: signal-cli -u +1XXXXXXXXXX daemon --socket /tmp/signal-cli.sock

Configuration (in settings.yaml):
    connectors:
      signal:
        socket_path: "/tmp/signal-cli.sock"
        phone_number: "+1XXXXXXXXXX"
        sync_interval: 5
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# How often to re-sync contacts/groups (seconds)
CONTACT_SYNC_INTERVAL = 3600  # 1 hour


class SignalConnector(BaseConnector):
    """Connector that communicates with Signal via the signal-cli JSON-RPC daemon.

    Integration pattern:
        - ``signal-cli`` must be running in daemon mode, exposing a Unix domain
          socket (``--socket /tmp/signal-cli.sock``).
        - This connector sends JSON-RPC 2.0 requests over that socket to
          receive new messages (``receive``) and send outbound messages (``send``).
        - Each inbound message is normalised into a Life OS ``message.received``
          event; outbound sends also publish ``message.sent`` so the signal-
          extraction pipeline can analyse what the user communicated.
        - Group messages are detected via the ``groupInfo`` field in the
          Signal envelope and are tagged with group ID/name in the payload.
        - Domain classification (work vs. personal) is heuristic, based on
          group names containing keywords like "work" or "team".
    """

    CONNECTOR_ID = "signal"
    DISPLAY_NAME = "Signal Messenger"
    # Very short polling interval (5 s) because instant messaging is
    # latency-sensitive.
    SYNC_INTERVAL_SECONDS = 5

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        # Path to the Unix domain socket exposed by signal-cli daemon.
        self._socket_path = config.get("socket_path", "/tmp/signal-cli.sock")
        # The phone number registered with signal-cli (E.164 format).
        self._phone = config.get("phone_number", "")
        # Monotonically increasing ID for JSON-RPC request correlation.
        self._request_id = 0
        self._contact_sync_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    async def start(self):
        """Start the connector, then kick off an initial contact sync."""
        await super().start()
        if self._running:
            # Initial sync, then periodic
            await self.sync_contacts()
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
        """Re-sync contacts/groups every CONTACT_SYNC_INTERVAL seconds."""
        while self._running:
            await asyncio.sleep(CONTACT_SYNC_INTERVAL)
            try:
                await self.sync_contacts()
            except Exception as e:
                logger.error("Contact sync error: %s", e)

    # ------------------------------------------------------------------
    # Contact & Group Sync
    # ------------------------------------------------------------------

    async def sync_contacts(self):
        """Pull contacts and groups from signal-cli into entities.db."""
        contacts = await self._rpc_call("listContacts") or []
        groups = await self._rpc_call("listGroups") or []
        now = datetime.now(timezone.utc).isoformat()

        # Build a phone→contact_id map for deduplication
        phone_to_id = self._load_existing_phone_map()

        with self.db.get_connection("entities") as conn:
            # ---- Contacts ----
            synced = 0
            for c in contacts:
                number = c.get("number")
                if not number:
                    continue

                name = c.get("name") or c.get("profileName") or c.get("contactName") or ""
                if not name:
                    # Skip contacts with no name at all
                    continue

                # Skip self
                if number == self._phone:
                    continue

                blocked = c.get("isBlocked", False)

                if number in phone_to_id:
                    # Update existing contact
                    contact_id = phone_to_id[number]
                    conn.execute(
                        """UPDATE contacts SET
                            name = CASE WHEN name LIKE '%Unknown%' OR name = '' THEN ? ELSE name END,
                            phones = ?,
                            channels = json_set(COALESCE(channels, '{}'), '$.signal', ?),
                            updated_at = ?
                           WHERE id = ?""",
                        (name, json.dumps([number]), number, now, contact_id),
                    )
                else:
                    # Try to match by name (fuzzy: first name match)
                    contact_id = self._find_contact_by_name(conn, name)

                    if contact_id:
                        # Enrich existing contact with phone/channel
                        conn.execute(
                            """UPDATE contacts SET
                                phones = ?,
                                channels = json_set(COALESCE(channels, '{}'), '$.signal', ?),
                                updated_at = ?
                               WHERE id = ?""",
                            (json.dumps([number]), number, now, contact_id),
                        )
                    else:
                        # Create new contact
                        contact_id = str(uuid.uuid4())
                        conn.execute(
                            """INSERT INTO contacts
                                (id, name, phones, channels, domains, created_at, updated_at)
                               VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                            (
                                contact_id, name,
                                json.dumps([number]),
                                json.dumps({"signal": number}),
                                now, now,
                            ),
                        )

                    phone_to_id[number] = contact_id

                # Upsert identifier
                conn.execute(
                    """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                       VALUES (?, 'phone', ?)
                       ON CONFLICT(identifier, identifier_type) DO UPDATE SET contact_id = ?""",
                    (number, contact_id, contact_id),
                )
                synced += 1

            # ---- Groups ----
            group_count = 0
            for g in groups:
                group_id = g.get("id", "")
                group_name = g.get("name", "Unknown Group")
                members = g.get("members", [])

                # Store group as an entity relationship between members
                member_ids = []
                for m in members:
                    m_number = m.get("number")
                    if m_number and m_number in phone_to_id:
                        member_ids.append(phone_to_id[m_number])

                # Create pairwise relationships for group members
                for i, mid_a in enumerate(member_ids):
                    for mid_b in member_ids[i + 1:]:
                        rel_id = f"signal-group-{group_id}-{mid_a}-{mid_b}"
                        conn.execute(
                            """INSERT INTO entity_relationships
                                (id, entity_a_type, entity_a_id, relationship,
                                 entity_b_type, entity_b_id, weight, metadata, created_at)
                               VALUES (?, 'contact', ?, 'signal_group_member',
                                       'contact', ?, 1.0, ?, ?)
                               ON CONFLICT(id) DO UPDATE SET metadata = ?, created_at = ?""",
                            (
                                rel_id, mid_a, mid_b,
                                json.dumps({"group_name": group_name, "group_id": group_id}),
                                now,
                                json.dumps({"group_name": group_name, "group_id": group_id}),
                                now,
                            ),
                        )
                group_count += 1

        logger.info("Synced %d contacts, %d groups", synced, group_count)

    def _load_existing_phone_map(self) -> dict[str, str]:
        """Load phone→contact_id map from existing identifiers."""
        with self.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT identifier, contact_id FROM contact_identifiers WHERE identifier_type = 'phone'"
            ).fetchall()
            return {row["identifier"]: row["contact_id"] for row in rows}

    def _find_contact_by_name(self, conn, name: str) -> Optional[str]:
        """Try to match a Signal contact name to an existing contact by first name."""
        first_name = name.split()[0].lower() if name else ""
        if not first_name or len(first_name) < 2:
            return None

        rows = conn.execute(
            "SELECT id, name FROM contacts"
        ).fetchall()
        for row in rows:
            existing_name = row["name"].lower()
            # Match if existing name contains the first name
            if first_name in existing_name:
                return row["id"]
        return None

    # ------------------------------------------------------------------
    # Original lifecycle methods
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        """Verify that the signal-cli daemon is running and the socket is reachable.

        Calls ``listAccounts`` as a lightweight connectivity check.  If the
        daemon is not running or the socket path is wrong, the connection will
        fail and we return False.
        """
        try:
            result = await self._rpc_call("listGroups")
            return isinstance(result, list)
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    async def sync(self) -> int:
        """Receive new messages from Signal via the ``receive`` RPC method.

        signal-cli's ``receive`` command returns all messages that have arrived
        since the last call and marks them as received on the Signal server.
        Each message envelope contains:
            - ``source`` / ``sourceName`` -- sender's phone number and profile name.
            - ``timestamp`` -- millisecond-precision Unix timestamp set by the
              sender's device.
            - ``dataMessage`` -- the actual content (text body, attachments,
              quotes, group info).  Envelopes without a ``dataMessage`` (e.g.,
              read receipts, typing indicators) are skipped.

        Returns the number of inbound message events published.
        """
        try:
            messages = await self._rpc_call("receive", {"account": self._phone})
            if not messages:
                return 0

            count = 0
            for msg in messages:
                # The envelope wraps metadata about the message transport.
                envelope = msg.get("envelope", {})
                # dataMessage contains the user-visible content; skip
                # non-content envelopes (receipts, typing indicators, etc.).
                data_msg = envelope.get("dataMessage")
                if not data_msg:
                    continue

                sender = envelope.get("source", "")
                sender_name = envelope.get("sourceName", sender)
                # Signal timestamps are in milliseconds since Unix epoch.
                timestamp_ms = envelope.get("timestamp", 0)
                body = data_msg.get("message", "")
                # groupInfo is present only for group messages.
                group_info = data_msg.get("groupInfo")

                # Skip messages with no text body (e.g., reaction-only, sticker).
                if not body:
                    continue

                # Convert millisecond timestamp to a UTC-aware datetime.
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

                # ---- Normalised Life OS Message Payload ----
                payload = {
                    # Use the timestamp as a unique message identifier since
                    # Signal does not expose a dedicated message ID.
                    "message_id": str(timestamp_ms),
                    "channel": "signal",
                    "direction": "inbound",
                    "from_address": sender,
                    "from_contact": sender_name,
                    "to_addresses": [self._phone],
                    "body": body,
                    "body_plain": body,
                    # Short preview for notification surfaces.
                    "snippet": body[:150] if len(body) > 150 else body,
                    # A "quote" in Signal means the user replied to a specific
                    # prior message -- analogous to email In-Reply-To.
                    "is_reply": bool(data_msg.get("quote")),
                    "group_id": group_info.get("groupId") if group_info else None,
                    "group_name": group_info.get("groupName") if group_info else None,
                    "has_attachments": bool(data_msg.get("attachments")),
                }

                metadata = {
                    "related_contacts": [sender],
                    # Heuristic domain classification (work vs. personal).
                    "domain": self._classify_domain(sender, group_info),
                }

                await self.publish_event(
                    "message.received", payload,
                    priority="normal", metadata=metadata,
                )
                count += 1

            return count
        except Exception as e:
            logger.error("Sync error: %s", e)
            return 0

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a message through Signal via the ``send`` RPC method.

        After sending, the connector also publishes a ``message.sent`` event
        so that the signal-extraction pipeline can analyse outbound
        communication alongside inbound messages.

        The ``to`` parameter can be a single phone number string or a list of
        phone numbers for multi-recipient sends.
        """
        if action == "send_message":
            recipient = params["to"]
            message = params["message"]

            # Normalise recipient to a list for the signal-cli API.
            await self._rpc_call("send", {
                "account": self._phone,
                "recipients": [recipient] if isinstance(recipient, str) else recipient,
                "message": message,
            })

            # Publish a "message.sent" event so the signal-extraction pipeline
            # and conversation history can track outbound messages too.
            await self.publish_event(
                "message.sent",
                {
                    "channel": "signal",
                    "direction": "outbound",
                    "from_address": self._phone,
                    "to_addresses": [recipient] if isinstance(recipient, str) else recipient,
                    "body": message,
                    "body_plain": message,
                    "snippet": message[:150],
                },
            )

            return {"status": "sent", "to": recipient}

        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
        """Check that the signal-cli daemon is responsive via ``listAccounts``."""
        try:
            result = await self._rpc_call("listGroups")
            return {"status": "ok", "connector": self.CONNECTOR_ID}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _rpc_call(self, method: str, params: Optional[dict] = None) -> Any:
        """Make a JSON-RPC 2.0 call to the signal-cli daemon over a Unix socket.

        Protocol details:
            - Each request is a single JSON object terminated by a newline.
            - The daemon responds with a single JSON line containing either a
              ``result`` (success) or ``error`` (failure) field.
            - A new socket connection is opened per call (short-lived) to avoid
              stale-connection issues.  The 10-second read timeout guards
              against a hung daemon.

        Returns the ``result`` value from the JSON-RPC response.
        Raises on RPC-level errors.
        """
        # Increment the request ID for JSON-RPC correlation.
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
        }
        if params:
            request["params"] = params

        # Open a fresh Unix domain socket connection to signal-cli.
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        try:
            # Newline-delimited JSON is signal-cli's wire format.
            data = json.dumps(request) + "\n"
            writer.write(data.encode())
            await writer.drain()

            # Wait for the response with a timeout to avoid blocking forever
            # if the daemon stalls.
            response_data = await asyncio.wait_for(reader.readline(), timeout=10)
            response = json.loads(response_data.decode())

            # JSON-RPC errors are returned in an "error" field.
            if "error" in response:
                raise Exception(f"Signal RPC error: {response['error']}")

            return response.get("result")
        finally:
            # Always close the socket, even on error.
            writer.close()
            await writer.wait_closed()

    def _classify_domain(self, sender: str, group_info: Optional[dict]) -> str:
        """Heuristically classify a message as 'work' or 'personal'.

        The classification checks the group name for work-related keywords.
        In production, this should also consult the contact store to check the
        sender's domain tag.  For now, any group whose name contains "work",
        "team", "project", or "standup" is treated as work; everything else
        defaults to personal.
        """
        if group_info:
            name = (group_info.get("groupName") or "").lower()
            if any(w in name for w in ["work", "team", "project", "standup"]):
                return "work"
        return "personal"
