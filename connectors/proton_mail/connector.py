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
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class ProtonMailConnector(BaseConnector):
    """Connector that syncs email via Proton Mail Bridge's local IMAP/SMTP.

    Architecture overview:
        - **Read path (sync):** Opens an IMAP connection to Proton Bridge
          (running on localhost), searches for messages since the last sync
          cursor, parses each RFC 822 message into a normalised Life OS event
          payload, and publishes ``email.received`` or ``email.sent`` events.
        - **Write path (execute):** Sends email through Proton Bridge's local
          SMTP server using STARTTLS.
        - **Thread detection:** Uses the ``In-Reply-To`` header.  If present
          the message's ``thread_id`` is set to that header's value; otherwise
          the message's own ``Message-ID`` is used, establishing a new thread.
        - **Urgency detection:** Subject lines containing keywords such as
          "urgent" or "asap" trigger ``priority="high"`` so the agent can
          surface them immediately.

    All encryption/decryption is transparent -- handled by Proton Bridge
    before data reaches IMAP/SMTP.
    """

    CONNECTOR_ID = "proton_mail"
    DISPLAY_NAME = "Proton Mail"
    # Poll for new mail every 30 seconds for near-real-time awareness.
    SYNC_INTERVAL_SECONDS = 30

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        # Persistent IMAP connection to Proton Bridge; set in authenticate().
        self._imap: Optional[imaplib.IMAP4] = None
        # IMAP folders to poll (e.g., "INBOX", "Sent").
        self._folders = config.get("folders", ["INBOX"])

    async def authenticate(self) -> bool:
        """Connect to Proton Bridge's local IMAP server.

        Proton Bridge exposes a standard IMAP interface on localhost.  The
        connection starts unencrypted (plain IMAP4 on port 1143) and is
        upgraded to TLS via STARTTLS.  The password used here is the
        *Bridge password*, not the user's Proton account password.
        """
        try:
            host = self.config.get("imap_host", "127.0.0.1")
            port = self.config.get("imap_port", 1143)
            username = self.config["username"]
            password = self.config["password"]

            # Open a plain IMAP4 connection to the local Bridge instance.
            self._imap = imaplib.IMAP4(host, port)
            # Upgrade the connection to TLS -- required by Proton Bridge.
            self._imap.starttls()
            # Authenticate with the Bridge-specific password.
            self._imap.login(username, password)

            return True
        except Exception as e:
            logger.error("Auth failed: %s", e)
            return False

    async def sync(self) -> int:
        """Fetch new emails since last sync via IMAP SEARCH.

        Incremental sync strategy:
            - The cursor is a date string in IMAP format (e.g., "15-Feb-2026").
            - On each cycle we ``SEARCH (SINCE "<cursor>")`` in every watched
              folder.  IMAP SINCE is date-granular (not time-granular), so we
              may re-fetch some messages from the cursor day; downstream
              deduplication by ``message_id`` handles this.
            - On the very first sync (no cursor), we fetch ALL messages but
              cap processing at the 100 most recent to avoid overwhelming the
              event bus with historical data.
            - After processing, the cursor is updated to today's date.

        Returns the total number of email events published across all folders.
        """
        if not self._imap:
            return 0

        total_new = 0
        # Retrieve the IMAP-format date string saved from the last sync.
        cursor = self.get_sync_cursor()

        for folder in self._folders:
            try:
                # Open the folder in read-only mode so we never accidentally
                # modify flags or delete messages.
                self._imap.select(folder, readonly=True)

                # ---- Build IMAP SEARCH Criteria ----
                if cursor:
                    # Incremental: fetch emails dated on or after the cursor day.
                    # Note: IMAP SINCE is inclusive and date-only (no time component).
                    search_criteria = f'(SINCE "{cursor}")'
                else:
                    # First sync -- retrieve everything, then trim below.
                    search_criteria = "ALL"

                _, message_nums = self._imap.search(None, search_criteria)
                # message_nums is a list with a single bytes element containing
                # space-separated message sequence numbers.
                nums = message_nums[0].split()

                # On the initial sync, cap at the 50 most recent messages to
                # avoid a massive backfill that could flood the event bus.
                if not cursor and len(nums) > 50:
                    nums = nums[-50:]

                for num in nums:
                    try:
                        event_count = await self._process_email(num, folder)
                        total_new += event_count
                    except Exception as e:
                        # Log and skip individual message failures so that one
                        # malformed email does not block the rest of the sync.
                        logger.warning("Error processing message %s: %s", num, e)

            except Exception as e:
                logger.error("Error syncing folder %s: %s", folder, e)

        # ---- Update Sync Cursor ----
        # Persist today's date so the next cycle only searches from this point.
        now = datetime.now(timezone.utc)
        self.set_sync_cursor(now.strftime("%d-%b-%Y"))

        return total_new

    async def _process_email(self, num: bytes, folder: str) -> int:
        """Parse a single RFC 822 email and publish a normalised event.

        Processing pipeline:
            1. Fetch the full message body (RFC822) from the IMAP server.
            2. Parse it with Python's ``email`` module.
            3. Extract and normalise headers (From, To, Cc, Subject, etc.).
            4. Parse the Date header into a UTC datetime, falling back to
               ``now()`` if the header is missing or malformed.
            5. Walk the MIME tree to extract plain-text and HTML body parts,
               plus a list of attachment filenames.
            6. Detect direction (inbound vs. outbound) by comparing From
               against the configured account address.
            7. Detect thread membership via the In-Reply-To header.
            8. Scan the subject for urgency keywords to set event priority.
            9. Publish the normalised payload to the event bus.

        Returns 1 on success, 0 if the message data is empty.
        """
        # Fetch the complete message using the RFC822 profile.
        _, data = self._imap.fetch(num, "(RFC822)")
        if not data or not data[0]:
            return 0

        raw_email = data[0][1]
        # Parse the raw bytes into a structured email.message.Message object.
        msg = email.message_from_bytes(raw_email)

        # ---- Header Extraction ----
        from_addr = self._parse_address(msg.get("From", ""))
        to_addrs = self._parse_address_list(msg.get("To", ""))
        cc_addrs = self._parse_address_list(msg.get("Cc", ""))
        subject = msg.get("Subject", "")
        message_id = msg.get("Message-ID", "")
        # In-Reply-To links this message to the one it is replying to,
        # forming the basis of our simplified thread detection.
        in_reply_to = msg.get("In-Reply-To", "")
        date_str = msg.get("Date", "")

        # ---- Date Parsing ----
        # Email Date headers follow RFC 2822 format; parsedate_tz handles
        # timezone offsets.  We normalise to UTC for consistent storage.
        try:
            date_tuple = email.utils.parsedate_tz(date_str)
            if date_tuple:
                timestamp = email.utils.mktime_tz(date_tuple)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)

        # ---- Body Extraction ----
        # Walk the MIME tree and extract both plain-text and HTML alternatives.
        body_plain, body_html = self._extract_body(msg)

        # ---- Attachment Detection ----
        # Only collect filenames; actual attachment content is not stored.
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)

        # ---- Direction Detection ----
        # Compare the From address against our own to determine inbound vs.
        # outbound.  This matters for routing (e.g., "Sent" folder emails are
        # outbound even though we fetched them via IMAP).
        my_address = self.config["username"].lower()
        is_outbound = from_addr.lower() == my_address
        event_type = "email.sent" if is_outbound else "email.received"

        # ---- Normalised Payload (Life OS email schema) ----
        payload = {
            "message_id": message_id,
            # Thread detection: if this message is a reply, group it under
            # the original message's ID; otherwise start a new thread.
            "thread_id": in_reply_to or message_id,
            "channel": "proton_mail",
            "direction": "outbound" if is_outbound else "inbound",
            "from_address": from_addr,
            "to_addresses": to_addrs,
            "cc_addresses": cc_addrs,
            "subject": subject,
            # Prefer the HTML body for rich display; fall back to plain text.
            "body": body_html or body_plain,
            "body_plain": body_plain,
            # Short preview for notifications and summaries.
            "snippet": (body_plain[:150] + "...") if len(body_plain) > 150 else body_plain,
            "has_attachments": len(attachments) > 0,
            "attachment_names": attachments,
            "is_reply": bool(in_reply_to),
            "in_reply_to": in_reply_to,
            "folder": folder,
            # CRITICAL: Include the email's actual Date header timestamp so downstream
            # systems (episodic memory, routine detection, temporal analysis, and
            # relationship frequency analysis) can use the true interaction time
            # instead of the sync timestamp. Without this, all episodes collapse to
            # a single day (the sync date), breaking routine detection and causing
            # relationship maintenance predictions to fail (avg gap = 0 days).
            #
            # Field name MUST be "email_date" to match the field name expected by
            # RelationshipExtractor (services/signal_extractor/relationship.py:62-67).
            "email_date": dt.isoformat(),
        }

        # ---- Related Contacts Metadata ----
        # Collect all external participants (excluding self) so the contact
        # graph and CRM features can link this email to known contacts.
        all_contacts = list(set(to_addrs + cc_addrs + [from_addr]))
        all_contacts = [c for c in all_contacts if c.lower() != my_address]

        metadata = {
            "related_contacts": all_contacts,
        }

        # ---- Urgency Detection ----
        # Scan the subject for common urgency keywords and promote the event
        # priority so the agent can surface it immediately.
        priority = "normal"
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
        if any(kw in subject.lower() for kw in urgent_keywords):
            priority = "high"

        # Publish the normalised email event to the bus.
        await self.publish_event(
            event_type,
            payload,
            priority=priority,
            metadata=metadata,
        )

        return 1

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute an outbound action (send or reply to email).

        Supported actions:
            - ``send_email``  -- compose and send a new email.
            - ``reply_email`` -- reply to an existing thread (auto-prefixes
              "Re:" to the original subject).
        """
        if action == "send_email":
            return await self._send_email(params)
        elif action == "reply_email":
            return await self._reply_email(params)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _send_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send an email through Proton Bridge's local SMTP server.

        The message is built as a ``multipart/alternative`` so mail clients can
        choose between the plain-text and HTML renderings.  STARTTLS is used
        to encrypt the connection to Bridge (even though it is localhost,
        Bridge requires it).
        """
        host = self.config.get("smtp_host", "127.0.0.1")
        port = self.config.get("smtp_port", 1025)
        username = self.config["username"]
        password = self.config["password"]

        # Build a multipart/alternative message with plain + optional HTML.
        msg = MIMEMultipart("alternative")
        msg["From"] = username
        msg["To"] = ", ".join(params["to"])
        msg["Subject"] = params["subject"]

        if params.get("cc"):
            msg["Cc"] = ", ".join(params["cc"])

        # Attach the plain-text body first (per RFC 2046, the last part is
        # preferred, so HTML -- if present -- will be displayed by default).
        body = params.get("body", "")
        msg.attach(MIMEText(body, "plain"))

        if params.get("body_html"):
            msg.attach(MIMEText(params["body_html"], "html"))

        # RFC 2822 threading headers — allow replies to thread correctly in
        # recipients' mail clients.  These are set by _reply_email() before
        # delegating here, but _send_email can also be called directly with
        # these params for programmatic threading.
        if params.get("in_reply_to"):
            msg["In-Reply-To"] = params["in_reply_to"]
        if params.get("references"):
            msg["References"] = params["references"]

        # Connect to Bridge SMTP, upgrade to TLS, authenticate, and send.
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)

        return {"status": "sent", "to": params["to"], "subject": params["subject"]}

    async def _reply_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reply to an existing email with proper threading headers.

        Sets the ``In-Reply-To`` and ``References`` RFC 2822 headers so that
        recipients' mail clients correctly thread the conversation.  The
        caller should provide ``in_reply_to`` (the Message-ID of the email
        being replied to) and optionally ``references`` (the full References
        chain).  If ``references`` is not given, ``in_reply_to`` is used as
        the sole reference.
        """
        params["subject"] = f"Re: {params.get('original_subject', '')}"

        # Resolve threading identifiers.  Prefer explicit params; fall back
        # to message_id (the original message's ID) for backwards compat.
        if not params.get("in_reply_to"):
            params["in_reply_to"] = params.get("message_id", "")
        if not params.get("references"):
            params["references"] = params.get("in_reply_to", "")

        return await self._send_email(params)

    async def health_check(self) -> dict[str, Any]:
        """Check IMAP connection health using the NOOP command.

        IMAP NOOP is a lightweight keep-alive that also lets the server report
        any mailbox changes.  If it returns "OK", the connection is alive.
        """
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
        """Extract a bare email address from a 'Display Name <user@example.com>' string.

        Uses ``email.utils.parseaddr`` which handles quoted names, angle
        brackets, and other RFC 2822 edge cases.
        """
        _, addr = email.utils.parseaddr(raw)
        return addr

    @staticmethod
    def _parse_address_list(raw: str) -> list[str]:
        """Extract all email addresses from a comma-separated header value.

        Returns an empty list for missing or empty headers.  Relies on
        ``email.utils.getaddresses`` which correctly handles multiple
        addresses with display names and quoted strings.
        """
        if not raw:
            return []
        addrs = email.utils.getaddresses([raw])
        return [addr for _, addr in addrs if addr]

    @staticmethod
    def _extract_body(msg: email.message.Message) -> tuple[str, str]:
        """Walk the MIME tree and extract the plain-text and HTML body parts.

        For multipart messages, we iterate over all sub-parts and keep the
        first ``text/plain`` and ``text/html`` payloads found.  For simple
        (non-multipart) messages, the top-level payload is used directly.

        Character set detection falls back to UTF-8 with replacement chars
        to avoid decoding failures on exotic encodings.

        Returns:
            A ``(plain_text, html_text)`` tuple; either may be empty.
        """
        plain = ""
        html = ""

        if msg.is_multipart():
            # Walk recursively through all MIME parts (nested multiparts,
            # attachments, alternative views, etc.).
            for part in msg.walk():
                content_type = part.get_content_type()
                try:
                    # decode=True handles Content-Transfer-Encoding (base64,
                    # quoted-printable) and returns raw bytes.
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="replace")
                        if content_type == "text/plain":
                            plain = decoded
                        elif content_type == "text/html":
                            html = decoded
                except Exception:
                    # Skip parts that cannot be decoded (e.g., binary data
                    # without a proper content-type).
                    pass
        else:
            # Simple, non-multipart message -- the whole body is one part.
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
