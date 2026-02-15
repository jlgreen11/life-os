"""
Life OS — Proton Mail Connector

Connects to Proton Mail through Proton Bridge, which exposes a local
IMAP server. All encryption/decryption is handled by Bridge transparently.

Requirements:
    - Proton Bridge running locally (default: localhost:1143 for IMAP)
    - Bridge configured with your Proton account
    - Bridge password (different from your Proton password)

Configuration (in settings.yaml):
    connectors:
      proton_mail:
        imap_host: "127.0.0.1"
        imap_port: 1143
        smtp_host: "127.0.0.1"
        smtp_port: 1025
        username: "your@proton.me"
        password: "bridge-password-here"
        sync_interval: 30
        folders:
          - "INBOX"
          - "Sent"
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class ProtonMailConnector(BaseConnector):

    CONNECTOR_ID = "proton_mail"
    DISPLAY_NAME = "Proton Mail"
    SYNC_INTERVAL_SECONDS = 30

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._imap: Optional[imaplib.IMAP4] = None
        self._folders = config.get("folders", ["INBOX"])

    async def authenticate(self) -> bool:
        """Connect to Proton Bridge IMAP."""
        try:
            host = self.config.get("imap_host", "127.0.0.1")
            port = self.config.get("imap_port", 1143)
            username = self.config["username"]
            password = self.config["password"]

            self._imap = imaplib.IMAP4(host, port)
            # Proton Bridge uses STARTTLS
            self._imap.starttls()
            self._imap.login(username, password)

            return True
        except Exception as e:
            print(f"[proton_mail] Auth failed: {e}")
            return False

    async def sync(self) -> int:
        """Fetch new emails since last sync."""
        if not self._imap:
            return 0

        total_new = 0
        cursor = self.get_sync_cursor()

        for folder in self._folders:
            try:
                self._imap.select(folder, readonly=True)

                # Build search criteria
                if cursor:
                    # Fetch emails since last sync (IMAP SINCE uses date, not datetime)
                    search_criteria = f'(SINCE "{cursor}")'
                else:
                    # First sync — get last 100 emails
                    search_criteria = "ALL"

                _, message_nums = self._imap.search(None, search_criteria)
                nums = message_nums[0].split()

                # Limit to most recent on first sync
                if not cursor and len(nums) > 100:
                    nums = nums[-100:]

                for num in nums:
                    try:
                        event_count = await self._process_email(num, folder)
                        total_new += event_count
                    except Exception as e:
                        print(f"[proton_mail] Error processing message {num}: {e}")

            except Exception as e:
                print(f"[proton_mail] Error syncing folder {folder}: {e}")

        # Update sync cursor
        now = datetime.now(timezone.utc)
        self.set_sync_cursor(now.strftime("%d-%b-%Y"))

        return total_new

    async def _process_email(self, num: bytes, folder: str) -> int:
        """Process a single email and publish events."""
        _, data = self._imap.fetch(num, "(RFC822)")
        if not data or not data[0]:
            return 0

        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email)

        # Extract fields
        from_addr = self._parse_address(msg.get("From", ""))
        to_addrs = self._parse_address_list(msg.get("To", ""))
        cc_addrs = self._parse_address_list(msg.get("Cc", ""))
        subject = msg.get("Subject", "")
        message_id = msg.get("Message-ID", "")
        in_reply_to = msg.get("In-Reply-To", "")
        date_str = msg.get("Date", "")

        # Parse date
        try:
            date_tuple = email.utils.parsedate_tz(date_str)
            if date_tuple:
                timestamp = email.utils.mktime_tz(date_tuple)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)

        # Extract body
        body_plain, body_html = self._extract_body(msg)

        # Detect attachments
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)

        # Determine if this is inbound or outbound
        my_address = self.config["username"].lower()
        is_outbound = from_addr.lower() == my_address
        event_type = "email.sent" if is_outbound else "email.received"

        # Build payload
        payload = {
            "message_id": message_id,
            "thread_id": in_reply_to or message_id,  # Simplified threading
            "channel": "proton_mail",
            "direction": "outbound" if is_outbound else "inbound",
            "from_address": from_addr,
            "to_addresses": to_addrs,
            "cc_addresses": cc_addrs,
            "subject": subject,
            "body": body_html or body_plain,
            "body_plain": body_plain,
            "snippet": (body_plain[:150] + "...") if len(body_plain) > 150 else body_plain,
            "has_attachments": len(attachments) > 0,
            "attachment_names": attachments,
            "is_reply": bool(in_reply_to),
            "in_reply_to": in_reply_to,
            "folder": folder,
        }

        # Build metadata
        all_contacts = list(set(to_addrs + cc_addrs + [from_addr]))
        all_contacts = [c for c in all_contacts if c.lower() != my_address]

        metadata = {
            "related_contacts": all_contacts,
        }

        # Check for urgency signals in subject
        priority = "normal"
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
        if any(kw in subject.lower() for kw in urgent_keywords):
            priority = "high"

        # Publish the event
        await self.publish_event(
            event_type,
            payload,
            priority=priority,
            metadata=metadata,
        )

        return 1

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an outbound action (send email)."""
        if action == "send_email":
            return await self._send_email(params)
        elif action == "reply_email":
            return await self._reply_email(params)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _send_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send an email through Proton Bridge SMTP."""
        host = self.config.get("smtp_host", "127.0.0.1")
        port = self.config.get("smtp_port", 1025)
        username = self.config["username"]
        password = self.config["password"]

        msg = MIMEMultipart("alternative")
        msg["From"] = username
        msg["To"] = ", ".join(params["to"])
        msg["Subject"] = params["subject"]

        if params.get("cc"):
            msg["Cc"] = ", ".join(params["cc"])

        body = params.get("body", "")
        msg.attach(MIMEText(body, "plain"))

        if params.get("body_html"):
            msg.attach(MIMEText(params["body_html"], "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)

        return {"status": "sent", "to": params["to"], "subject": params["subject"]}

    async def _reply_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reply to an existing email."""
        # Build the reply with proper headers
        params["subject"] = f"Re: {params.get('original_subject', '')}"
        return await self._send_email(params)

    async def health_check(self) -> dict[str, Any]:
        """Check IMAP connection health."""
        try:
            if self._imap:
                status, _ = self._imap.noop()
                if status == "OK":
                    return {"status": "ok", "connector": self.CONNECTOR_ID}
            return {"status": "error", "details": "Not connected"}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _parse_address(raw: str) -> str:
        """Extract email address from 'Name <email>' format."""
        _, addr = email.utils.parseaddr(raw)
        return addr

    @staticmethod
    def _parse_address_list(raw: str) -> list[str]:
        """Extract email addresses from a comma-separated list."""
        if not raw:
            return []
        addrs = email.utils.getaddresses([raw])
        return [addr for _, addr in addrs if addr]

    @staticmethod
    def _extract_body(msg: email.message.Message) -> tuple[str, str]:
        """Extract plain text and HTML body from a message."""
        plain = ""
        html = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="replace")
                        if content_type == "text/plain":
                            plain = decoded
                        elif content_type == "text/html":
                            html = decoded
                except Exception:
                    pass
        else:
            content_type = msg.get_content_type()
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    decoded = payload.decode(charset, errors="replace")
                    if content_type == "text/plain":
                        plain = decoded
                    elif content_type == "text/html":
                        html = decoded
            except Exception:
                pass

        return plain, html
