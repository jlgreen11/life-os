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
import socket
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class SignalConnector(BaseConnector):

    CONNECTOR_ID = "signal"
    DISPLAY_NAME = "Signal Messenger"
    SYNC_INTERVAL_SECONDS = 5

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._socket_path = config.get("socket_path", "/tmp/signal-cli.sock")
        self._phone = config.get("phone_number", "")
        self._request_id = 0

    async def authenticate(self) -> bool:
        """Verify signal-cli daemon is running and accessible."""
        try:
            result = await self._rpc_call("listAccounts")
            return result is not None
        except Exception as e:
            print(f"[signal] Auth failed: {e}")
            return False

    async def sync(self) -> int:
        """Receive new messages from Signal."""
        try:
            messages = await self._rpc_call("receive", {"account": self._phone})
            if not messages:
                return 0

            count = 0
            for msg in messages:
                envelope = msg.get("envelope", {})
                data_msg = envelope.get("dataMessage")
                if not data_msg:
                    continue

                sender = envelope.get("source", "")
                sender_name = envelope.get("sourceName", sender)
                timestamp_ms = envelope.get("timestamp", 0)
                body = data_msg.get("message", "")
                group_info = data_msg.get("groupInfo")

                if not body:
                    continue

                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

                payload = {
                    "message_id": str(timestamp_ms),
                    "channel": "signal",
                    "direction": "inbound",
                    "from_address": sender,
                    "from_contact": sender_name,
                    "to_addresses": [self._phone],
                    "body": body,
                    "body_plain": body,
                    "snippet": body[:150] if len(body) > 150 else body,
                    "is_reply": bool(data_msg.get("quote")),
                    "group_id": group_info.get("groupId") if group_info else None,
                    "group_name": group_info.get("groupName") if group_info else None,
                    "has_attachments": bool(data_msg.get("attachments")),
                }

                metadata = {
                    "related_contacts": [sender],
                    "domain": self._classify_domain(sender, group_info),
                }

                await self.publish_event(
                    "message.received", payload,
                    priority="normal", metadata=metadata,
                )
                count += 1

            return count
        except Exception as e:
            print(f"[signal] Sync error: {e}")
            return 0

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a message through Signal."""
        if action == "send_message":
            recipient = params["to"]
            message = params["message"]

            await self._rpc_call("send", {
                "account": self._phone,
                "recipients": [recipient] if isinstance(recipient, str) else recipient,
                "message": message,
            })

            # Also publish the sent event for signal extraction
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
        try:
            result = await self._rpc_call("listAccounts")
            return {"status": "ok", "connector": self.CONNECTOR_ID}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    async def _rpc_call(self, method: str, params: Optional[dict] = None) -> Any:
        """Make a JSON-RPC call to signal-cli daemon."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
        }
        if params:
            request["params"] = params

        # Use Unix domain socket
        reader, writer = await asyncio.open_unix_connection(self._socket_path)
        try:
            data = json.dumps(request) + "\n"
            writer.write(data.encode())
            await writer.drain()

            response_data = await asyncio.wait_for(reader.readline(), timeout=10)
            response = json.loads(response_data.decode())

            if "error" in response:
                raise Exception(f"Signal RPC error: {response['error']}")

            return response.get("result")
        finally:
            writer.close()
            await writer.wait_closed()

    def _classify_domain(self, sender: str, group_info: Optional[dict]) -> str:
        """Simple domain classification (work vs personal)."""
        # In production, check the contact's domain field
        if group_info:
            name = (group_info.get("groupName") or "").lower()
            if any(w in name for w in ["work", "team", "project", "standup"]):
                return "work"
        return "personal"
