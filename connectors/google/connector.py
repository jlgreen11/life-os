"""
Life OS — Google Services Connector

Unified connector for Gmail, Google Calendar, and Google Contacts via
Google's OAuth2 APIs. One auth flow, one token, three sync sub-services.

Prerequisites:
    1. Create a project in Google Cloud Console
    2. Enable: Gmail API, Google Calendar API, People API
    3. Create OAuth 2.0 credentials (Desktop app type)
    4. Download credentials.json → place at data/google_credentials.json
    5. Complete OAuth via /api/admin/connectors/google/auth endpoint

Configuration (in settings.yaml or admin UI):
    connectors:
      google:
        email_address: "you@gmail.com"
        credentials_file: "data/google_credentials.json"
        token_file: "data/google_token.json"
        sync_interval: 30
        calendars:
          - "primary"
        gmail_labels:
          - "INBOX"
          - "SENT"
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/contacts.readonly",
]

# How often to re-sync contacts (seconds)
CONTACT_SYNC_INTERVAL = 3600  # 1 hour


class GoogleConnector(BaseConnector):

    CONNECTOR_ID = "google"
    DISPLAY_NAME = "Google (Gmail, Calendar, Contacts)"
    SYNC_INTERVAL_SECONDS = 30

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict[str, Any]):
        super().__init__(event_bus, db, config)
        self._gmail_service = None
        self._calendar_service = None
        self._people_service = None
        self._credentials_file = config.get("credentials_file", "data/google_credentials.json")
        self._token_file = config.get("token_file", "data/google_token.json")
        self._email_address = config.get("email_address", "")
        self._calendars = config.get("calendars", ["primary"])
        self._gmail_labels = config.get("gmail_labels", ["INBOX", "SENT"])
        self._last_contact_sync: float = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Override start to delay first sync so NATS subscribe can complete."""
        if self._running:
            return

        success = await self.authenticate()
        if not success:
            error_detail = getattr(self, "_auth_error", None) or "Authentication failed"
            await self._update_state("error", error_detail)
            return

        self._running = True
        await self._update_state("active")

        # Subscribe to action requests FIRST (before sync loop starts)
        await self.bus.subscribe(
            f"action.{self.CONNECTOR_ID}.*",
            self._handle_action_request,
            consumer_name=f"connector-{self.CONNECTOR_ID}",
        )

        # Start sync loop with a brief delay so NATS settles
        self._task = asyncio.create_task(self._delayed_sync_loop())

    async def _delayed_sync_loop(self):
        """Sync loop with initial delay to avoid flooding NATS at startup."""
        await asyncio.sleep(2)
        await self._sync_loop()

    async def authenticate(self) -> bool:
        """Load stored OAuth token and build API service objects.

        Returns True on success, False on failure. Sets self._auth_error with a
        descriptive message when authentication fails, so start() can pass it
        to _update_state() for display in the admin UI.
        """
        self._auth_error = None
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = self._load_credentials()
            if not creds:
                self._auth_error = "No valid token found — complete OAuth via /admin connector panel"
                logger.warning(self._auth_error)
                return False

            self._gmail_service = build("gmail", "v1", credentials=creds)
            self._calendar_service = build("calendar", "v3", credentials=creds)
            self._people_service = build("people", "v1", credentials=creds)

            # Verify by fetching profile
            profile = self._gmail_service.users().getProfile(userId="me").execute()
            self._email_address = profile.get("emailAddress", self._email_address)
            logger.info("Authenticated as %s", self._email_address)

            return True
        except ValueError as e:
            # Descriptive errors from _load_credentials() — surface directly
            self._auth_error = str(e)
            logger.error("Auth failed: %s", e)
            return False
        except Exception as e:
            self._auth_error = f"Authentication failed: {e}. Try re-authenticating via /admin connector panel."
            logger.error("Auth failed: %s", e)
            return False

    # Retry delays (seconds) for transient network errors during token refresh.
    # creds.refresh() is synchronous, so time.sleep() is appropriate here.
    TOKEN_REFRESH_RETRY_DELAYS = [2, 5, 10]

    def _load_credentials(self):
        """Load and refresh OAuth credentials from token file.

        Returns valid credentials or None. Raises ValueError with a descriptive
        message on known failure modes so authenticate() can surface actionable
        guidance in the admin UI.

        Transient network errors (TransportError) are retried up to 3 times with
        increasing delays (2s, 5s, 10s). This handles DNS/network blips at startup
        when the network stack isn't yet ready. RefreshError (invalid_grant) is
        never retried — it means the token was revoked and requires user action.
        """
        import os

        from google.auth import exceptions as google_exceptions
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not os.path.exists(self._token_file):
            return None

        creds = Credentials.from_authorized_user_file(self._token_file, SCOPES)

        if creds.valid:
            return creds

        if creds.expired and creds.refresh_token:
            retry_delays = self.TOKEN_REFRESH_RETRY_DELAYS
            last_transport_error = None
            for attempt, delay in enumerate(retry_delays):
                try:
                    creds.refresh(Request())
                    last_transport_error = None
                    break
                except google_exceptions.RefreshError as e:
                    # Don't retry revoked/expired tokens — requires user re-auth
                    logger.error(
                        "Token refresh failed (revoked or expired grant): %s — "
                        "re-authenticate via /admin connector panel",
                        e,
                    )
                    raise ValueError(
                        "Token refresh failed (invalid_grant) — re-authenticate via /admin connector panel"
                    ) from e
                except google_exceptions.TransportError as e:
                    last_transport_error = e
                    if attempt < len(retry_delays) - 1:
                        logger.warning(
                            "Token refresh attempt %d/%d failed (network): %s, retrying in %ds...",
                            attempt + 1,
                            len(retry_delays),
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Token refresh failed (network error) after %d attempts: %s",
                            len(retry_delays),
                            e,
                        )
            if last_transport_error:
                raise ValueError(
                    f"Token refresh failed (network error) after {len(retry_delays)} attempts "
                    f"— check network connectivity: {last_transport_error}"
                ) from last_transport_error

            # Save refreshed token
            with open(self._token_file, "w") as f:
                f.write(creds.to_json())
            return creds

        if creds.expired:
            logger.warning(
                "Token expired but no refresh_token present — "
                "re-authenticate via /admin connector panel to get a new refresh token"
            )
            raise ValueError(
                "Token expired with no refresh_token — re-authenticate via /admin connector panel"
            )

        return None

    async def sync(self) -> int:
        """Run all three sub-sync methods."""
        total = 0
        total += await self._sync_gmail()
        total += await self._sync_calendar()

        # Contact sync only runs hourly
        now = time.time()
        if now - self._last_contact_sync >= CONTACT_SYNC_INTERVAL:
            total += await self._sync_contacts()
            self._last_contact_sync = now

        return total

    # ------------------------------------------------------------------
    # Gmail sync
    # ------------------------------------------------------------------

    async def _sync_gmail(self) -> int:
        """Poll for new emails since last sync."""
        if not self._gmail_service:
            return 0

        count = 0
        cursor = self.get_sync_cursor()

        # Build query for incremental sync
        if cursor:
            query = f"after:{cursor}"
        else:
            query = None  # First sync — we'll limit results

        # First sync fetches the 200 most recent messages; subsequent syncs are incremental
        max_messages = 200 if cursor else 200

        try:
            # Fetch message IDs (paginate — API returns max 500 per page)
            messages = []
            page_token = None

            while len(messages) < max_messages:
                kwargs = {
                    "userId": "me",
                    "maxResults": min(500, max_messages - len(messages)),
                }
                if query:
                    kwargs["q"] = query
                if page_token:
                    kwargs["pageToken"] = page_token

                results = self._gmail_service.users().messages().list(**kwargs).execute()
                messages.extend(results.get("messages", []))

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            if not cursor:
                logger.info("Initial sync: fetching %d messages", len(messages))

            for i, msg_stub in enumerate(messages):
                try:
                    msg = self._gmail_service.users().messages().get(
                        userId="me", id=msg_stub["id"], format="full"
                    ).execute()
                    event_count = await self._process_gmail_message(msg)
                    count += event_count

                    # Throttle during large syncs to avoid overwhelming NATS
                    if not cursor and (i + 1) % 10 == 0:
                        await asyncio.sleep(0.05)

                    # Progress logging for large syncs
                    if not cursor and (i + 1) % 500 == 0:
                        logger.info("Processed %d/%d messages", i + 1, len(messages))
                except Exception as e:
                    logger.error("Error processing message %s: %s", msg_stub["id"], e)

        except Exception as e:
            logger.error("Gmail sync error: %s", e)

        # Update sync cursor to current epoch
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        self.set_sync_cursor(str(now_epoch))

        return count

    async def _process_gmail_message(self, msg: dict) -> int:
        """Process a single Gmail message and publish an event."""
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

        from_addr = self._parse_email_address(headers.get("From", ""))
        to_addrs = self._parse_email_list(headers.get("To", ""))
        cc_addrs = self._parse_email_list(headers.get("Cc", ""))
        subject = headers.get("Subject", "")
        message_id = headers.get("Message-ID", msg["id"])
        in_reply_to = headers.get("In-Reply-To", "")
        date_str = headers.get("Date", "")

        # Parse date
        try:
            from email.utils import mktime_tz, parsedate_tz
            date_tuple = parsedate_tz(date_str)
            if date_tuple:
                timestamp = mktime_tz(date_tuple)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)

        # Extract body
        body_plain, body_html = self._extract_gmail_body(msg.get("payload", {}))

        # Detect attachments
        attachments = self._extract_attachment_names(msg.get("payload", {}))

        # Determine direction
        my_address = self._email_address.lower()
        is_outbound = from_addr.lower() == my_address
        event_type = "email.sent" if is_outbound else "email.received"

        # Build payload (matching ProtonMail pattern)
        payload = {
            "message_id": message_id,
            "thread_id": msg.get("threadId", in_reply_to or message_id),
            "channel": "google",
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
            "folder": ", ".join(msg.get("labelIds", [])),
            # CRITICAL: Include actual email timestamp from Date header (not sync time).
            # This enables accurate relationship frequency analysis and routine detection.
            # Without this, all interactions appear to happen at sync time, breaking
            # relationship maintenance predictions which depend on measuring real gaps
            # between communications.
            "email_date": dt.isoformat(),
        }

        # Metadata
        all_contacts = list(set(to_addrs + cc_addrs + [from_addr]))
        all_contacts = [c for c in all_contacts if c.lower() != my_address]

        metadata = {
            "related_contacts": all_contacts,
        }

        # Priority detection
        priority = "normal"
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]
        if any(kw in subject.lower() for kw in urgent_keywords):
            priority = "high"

        await self._publish_with_retry(event_type, payload, priority=priority, metadata=metadata)
        return 1

    async def _publish_with_retry(self, event_type: str, payload: dict,
                                   priority: str = "normal", metadata: dict | None = None,
                                   max_retries: int = 3):
        """Publish event with retry on NATS timeout."""
        for attempt in range(max_retries):
            try:
                await self.publish_event(event_type, payload, priority=priority, metadata=metadata)
                return
            except Exception as e:
                if attempt < max_retries - 1 and "timeout" in str(e).lower():
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    raise

    def _extract_gmail_body(self, payload: dict) -> tuple[str, str]:
        """Extract plain text and HTML body from Gmail message payload."""
        plain = ""
        html = ""

        mime_type = payload.get("mimeType", "")
        parts = payload.get("parts", [])

        if parts:
            for part in parts:
                part_mime = part.get("mimeType", "")
                if part_mime == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        plain = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                elif part_mime == "text/html":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        html = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                elif part_mime.startswith("multipart/"):
                    # Recurse into nested multipart
                    sub_plain, sub_html = self._extract_gmail_body(part)
                    if sub_plain and not plain:
                        plain = sub_plain
                    if sub_html and not html:
                        html = sub_html
        elif mime_type == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                plain = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        elif mime_type == "text/html":
            data = payload.get("body", {}).get("data", "")
            if data:
                html = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        return plain, html

    def _extract_attachment_names(self, payload: dict) -> list[str]:
        """Extract attachment filenames from Gmail message payload."""
        attachments = []
        parts = payload.get("parts", [])
        for part in parts:
            filename = part.get("filename", "")
            if filename:
                attachments.append(filename)
            # Check nested parts
            sub_parts = part.get("parts", [])
            for sub in sub_parts:
                sub_filename = sub.get("filename", "")
                if sub_filename:
                    attachments.append(sub_filename)
        return attachments

    @staticmethod
    def _parse_email_address(raw: str) -> str:
        """Extract email address from 'Name <email>' format."""
        from email.utils import parseaddr
        _, addr = parseaddr(raw)
        return addr

    @staticmethod
    def _parse_email_list(raw: str) -> list[str]:
        """Extract email addresses from a comma-separated list."""
        if not raw:
            return []
        from email.utils import getaddresses
        addrs = getaddresses([raw])
        return [addr for _, addr in addrs if addr]

    # ------------------------------------------------------------------
    # Calendar sync
    # ------------------------------------------------------------------

    async def _sync_calendar(self) -> int:
        """Pull calendar events for the upcoming 14 days."""
        if not self._calendar_service:
            return 0

        count = 0
        now = datetime.now(timezone.utc)
        time_min = (now - timedelta(days=1)).isoformat()
        time_max = (now + timedelta(days=14)).isoformat()

        # Get calendar IDs to sync
        cal_ids = self._calendars
        if not cal_ids or cal_ids == ["all"]:
            # Fetch all calendars
            try:
                cal_list = self._calendar_service.calendarList().list().execute()
                cal_ids = [c["id"] for c in cal_list.get("items", [])]
            except Exception as e:
                logger.error("Error listing calendars: %s", e)
                return 0

        for cal_id in cal_ids:
            try:
                events_result = self._calendar_service.events().list(
                    calendarId=cal_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                ).execute()

                for event in events_result.get("items", []):
                    try:
                        event_count = await self._process_calendar_event(event, cal_id)
                        count += event_count
                    except Exception as e:
                        logger.error("Error processing calendar event: %s", e)

            except Exception as e:
                logger.error("Calendar sync error (%s): %s", cal_id, e)

        return count

    async def _process_calendar_event(self, event: dict, cal_id: str) -> int:
        """Process a single Google Calendar event and publish."""
        start = event.get("start", {})
        end = event.get("end", {})

        # All-day events have 'date', timed events have 'dateTime'
        is_all_day = "date" in start and "dateTime" not in start
        start_time = start.get("dateTime") or start.get("date", "")
        end_time = end.get("dateTime") or end.get("date", "")

        # Extract attendees
        attendees = []
        for a in event.get("attendees", []):
            email = a.get("email", "")
            if email:
                attendees.append(email)

        organizer = event.get("organizer", {}).get("email")

        # Build payload (matching CalDAV pattern)
        payload = {
            "event_id": event.get("id", ""),
            "calendar_id": cal_id,
            "title": event.get("summary", "Untitled"),
            "description": event.get("description"),
            "location": event.get("location"),
            "start_time": start_time,
            "end_time": end_time,
            "is_all_day": is_all_day,
            "attendees": attendees,
            "organizer": organizer,
        }

        metadata = {
            "related_contacts": attendees,
            "location": event.get("location"),
        }

        await self._publish_with_retry("calendar.event.created", payload, metadata=metadata)
        return 1

    # ------------------------------------------------------------------
    # Contacts sync
    # ------------------------------------------------------------------

    async def _sync_contacts(self) -> int:
        """Sync Google Contacts into entities.db."""
        if not self._people_service:
            return 0

        count = 0
        now = datetime.now(timezone.utc).isoformat()

        try:
            # Load existing email→contact_id map for dedup
            email_to_id = self._load_existing_email_map()

            contacts = []
            page_token = None

            # Paginate through all contacts
            while True:
                kwargs = {
                    "resourceName": "people/me",
                    "pageSize": 1000,
                    "personFields": "names,emailAddresses,phoneNumbers",
                }
                if page_token:
                    kwargs["pageToken"] = page_token

                results = self._people_service.people().connections().list(**kwargs).execute()
                connections = results.get("connections", [])
                contacts.extend(connections)

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            with self.db.get_connection("entities") as conn:
                for person in contacts:
                    try:
                        count += self._upsert_contact(conn, person, email_to_id, now)
                    except Exception as e:
                        logger.error("Error upserting contact: %s", e)

            logger.info("Synced %d contacts", count)

        except Exception as e:
            logger.error("Contact sync error: %s", e)

        return count

    def _upsert_contact(self, conn, person: dict, email_to_id: dict, now: str) -> int:
        """Upsert a single Google contact into entities.db."""
        # Extract name
        names = person.get("names", [])
        name = names[0].get("displayName", "") if names else ""
        if not name:
            return 0

        # Extract emails and phones
        emails = [e.get("value", "") for e in person.get("emailAddresses", []) if e.get("value")]
        phones = [p.get("value", "") for p in person.get("phoneNumbers", []) if p.get("value")]

        if not emails and not phones:
            return 0

        # Deduplicate: check if any email matches existing contact
        contact_id = None
        for em in emails:
            if em.lower() in email_to_id:
                contact_id = email_to_id[em.lower()]
                break

        if contact_id:
            # Update existing contact with Google channel
            conn.execute(
                """UPDATE contacts SET
                    name = CASE WHEN name LIKE '%Unknown%' OR name = '' THEN ? ELSE name END,
                    emails = ?,
                    phones = CASE WHEN phones IS NULL OR phones = '[]' THEN ? ELSE phones END,
                    channels = json_set(COALESCE(channels, '{}'), '$.google', ?),
                    updated_at = ?
                   WHERE id = ?""",
                (
                    name,
                    json.dumps(emails),
                    json.dumps(phones),
                    emails[0] if emails else name,
                    now,
                    contact_id,
                ),
            )
        else:
            # Try to match by name (same pattern as Signal connector)
            contact_id = self._find_contact_by_name(conn, name)

            if contact_id:
                # Enrich existing contact
                conn.execute(
                    """UPDATE contacts SET
                        emails = ?,
                        phones = CASE WHEN phones IS NULL OR phones = '[]' THEN ? ELSE phones END,
                        channels = json_set(COALESCE(channels, '{}'), '$.google', ?),
                        updated_at = ?
                       WHERE id = ?""",
                    (
                        json.dumps(emails),
                        json.dumps(phones),
                        emails[0] if emails else name,
                        now,
                        contact_id,
                    ),
                )
            else:
                # Create new contact
                contact_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO contacts
                        (id, name, emails, phones, channels, domains, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, '["personal"]', ?, ?)""",
                    (
                        contact_id, name,
                        json.dumps(emails),
                        json.dumps(phones),
                        json.dumps({"google": emails[0] if emails else name}),
                        now, now,
                    ),
                )

            # Update email map
            for em in emails:
                email_to_id[em.lower()] = contact_id

        # Upsert identifiers for each email
        for em in emails:
            conn.execute(
                """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                   VALUES (?, 'email', ?)
                   ON CONFLICT(identifier, identifier_type) DO UPDATE SET contact_id = ?""",
                (em.lower(), contact_id, contact_id),
            )

        # Upsert identifiers for each phone
        for phone in phones:
            conn.execute(
                """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                   VALUES (?, 'phone', ?)
                   ON CONFLICT(identifier, identifier_type) DO UPDATE SET contact_id = ?""",
                (phone, contact_id, contact_id),
            )

        return 1

    def _load_existing_email_map(self) -> dict[str, str]:
        """Load email→contact_id map from existing identifiers."""
        with self.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT identifier, contact_id FROM contact_identifiers WHERE identifier_type = 'email'"
            ).fetchall()
            return {row["identifier"].lower(): row["contact_id"] for row in rows}

    def _find_contact_by_name(self, conn, name: str) -> Optional[str]:
        """Try to match a Google contact name to an existing contact by first name."""
        first_name = name.split()[0].lower() if name else ""
        if not first_name or len(first_name) < 2:
            return None

        rows = conn.execute("SELECT id, name FROM contacts").fetchall()
        for row in rows:
            existing_name = row["name"].lower()
            if first_name in existing_name:
                return row["id"]
        return None

    # ------------------------------------------------------------------
    # Execute actions
    # ------------------------------------------------------------------

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform outbound actions (send email, create event, reply)."""
        if action == "send_email":
            return await self._send_email(params)
        elif action == "reply_email":
            return await self._reply_email(params)
        elif action == "create_event":
            return await self._create_event(params)
        else:
            raise ValueError(f"Unknown action: {action}")

    async def _send_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send an email via Gmail API."""
        if not self._gmail_service:
            return {"status": "error", "details": "Gmail service not initialized"}

        msg = MIMEMultipart("alternative")
        msg["From"] = self._email_address
        msg["To"] = ", ".join(params["to"])
        msg["Subject"] = params["subject"]

        if params.get("cc"):
            msg["Cc"] = ", ".join(params["cc"])

        body = params.get("body", "")
        msg.attach(MIMEText(body, "plain"))

        if params.get("body_html"):
            msg.attach(MIMEText(params["body_html"], "html"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        result = self._gmail_service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()

        return {"status": "sent", "to": params["to"], "subject": params["subject"],
                "message_id": result.get("id")}

    async def _reply_email(self, params: dict[str, Any]) -> dict[str, Any]:
        """Reply to an existing email with proper threading."""
        if not self._gmail_service:
            return {"status": "error", "details": "Gmail service not initialized"}

        msg = MIMEMultipart("alternative")
        msg["From"] = self._email_address
        msg["To"] = ", ".join(params["to"])
        msg["Subject"] = f"Re: {params.get('original_subject', '')}"

        if params.get("in_reply_to"):
            msg["In-Reply-To"] = params["in_reply_to"]
            msg["References"] = params["in_reply_to"]

        if params.get("cc"):
            msg["Cc"] = ", ".join(params["cc"])

        body = params.get("body", "")
        msg.attach(MIMEText(body, "plain"))

        if params.get("body_html"):
            msg.attach(MIMEText(params["body_html"], "html"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        send_body = {"raw": raw}
        if params.get("thread_id"):
            send_body["threadId"] = params["thread_id"]

        result = self._gmail_service.users().messages().send(
            userId="me", body=send_body
        ).execute()

        return {"status": "sent", "to": params["to"],
                "subject": msg["Subject"], "message_id": result.get("id")}

    async def _create_event(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a Google Calendar event."""
        if not self._calendar_service:
            return {"status": "error", "details": "Calendar service not initialized"}

        event_body = {
            "summary": params["title"],
            "start": {},
            "end": {},
        }

        if params.get("description"):
            event_body["description"] = params["description"]
        if params.get("location"):
            event_body["location"] = params["location"]

        # Handle all-day vs timed events
        if params.get("is_all_day"):
            event_body["start"]["date"] = params["start_time"]
            event_body["end"]["date"] = params.get("end_time", params["start_time"])
        else:
            event_body["start"]["dateTime"] = params["start_time"]
            event_body["start"]["timeZone"] = params.get("timezone", "UTC")
            event_body["end"]["dateTime"] = params.get("end_time", params["start_time"])
            event_body["end"]["timeZone"] = params.get("timezone", "UTC")

        # Add attendees
        if params.get("attendees"):
            event_body["attendees"] = [{"email": a} for a in params["attendees"]]

        cal_id = params.get("calendar_id", "primary")

        try:
            result = self._calendar_service.events().insert(
                calendarId=cal_id, body=event_body
            ).execute()
            return {"status": "created", "title": params["title"],
                    "event_id": result.get("id"), "link": result.get("htmlLink")}
        except Exception as e:
            return {"status": "error", "details": str(e)}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Verify Gmail API connectivity with diagnostics and recovery attempt.

        When the connector is not authenticated, gathers diagnostic info about
        the token file and attempts a token refresh. If refresh succeeds,
        re-authenticates and reports 'recovered' status. On failure, returns
        actionable guidance for the user.
        """
        import os

        try:
            if self._gmail_service:
                profile = self._gmail_service.users().getProfile(userId="me").execute()
                return {
                    "status": "ok",
                    "connector": self.CONNECTOR_ID,
                    "email": profile.get("emailAddress"),
                }
        except Exception as e:
            # Authenticated but API call failed — gather full diagnostics
            error_type = self._classify_api_error(e)
            diagnostics = self._build_health_diagnostics()
            diagnostics["status"] = "error"
            diagnostics["details"] = str(e)
            diagnostics["error_type"] = error_type
            diagnostics["recovery_hint"] = self._recovery_hint_for_error_type(error_type)

            # Clear stale service objects so next attempt triggers re-auth
            self._gmail_service = None
            self._calendar_service = None
            self._people_service = None

            return diagnostics

        # Not authenticated — gather diagnostics and attempt recovery
        diagnostics = self._build_health_diagnostics()

        # Attempt token refresh if token file exists
        if diagnostics["token_file_exists"]:
            try:
                self._load_credentials()
                # Credentials loaded successfully — try full re-authentication
                if await self.authenticate():
                    diagnostics["status"] = "recovered"
                    diagnostics["details"] = "Token refreshed and services restored"
                    diagnostics["recovery_hint"] = "Recovered automatically"
                    return diagnostics
            except Exception as e:
                diagnostics["details"] = f"Token refresh failed: {e}"
                diagnostics["recovery_hint"] = "Re-authenticate via /admin connector panel"

        return diagnostics

    def _build_health_diagnostics(self) -> dict[str, Any]:
        """Build diagnostic fields for health_check when not authenticated.

        Gathers token file state and connector sync history to provide
        actionable information in the admin dashboard.
        """
        import os

        token_file_exists = os.path.exists(self._token_file)

        # Token age in hours
        token_age_hours = None
        if token_file_exists:
            try:
                mtime = os.path.getmtime(self._token_file)
                token_age_hours = round((time.time() - mtime) / 3600, 1)
            except OSError:
                pass

        # Check if stored token has a refresh_token
        has_refresh_token = False
        if token_file_exists:
            try:
                import json as _json

                with open(self._token_file) as f:
                    token_data = _json.load(f)
                has_refresh_token = bool(token_data.get("refresh_token"))
            except Exception:
                pass

        # Last successful sync from connector state
        last_sync = None
        try:
            with self.db.get_connection("state") as conn:
                row = conn.execute(
                    "SELECT last_sync FROM connector_state WHERE connector_id = ?",
                    (self.CONNECTOR_ID,),
                ).fetchone()
                if row and row["last_sync"]:
                    last_sync = row["last_sync"]
        except Exception:
            pass

        # Build recovery hint
        if not token_file_exists:
            recovery_hint = "Token file missing — complete initial OAuth setup via /admin connector panel"
        elif not has_refresh_token:
            recovery_hint = "No refresh token — re-authenticate via /admin connector panel"
        else:
            recovery_hint = "Re-authenticate via /admin connector panel"

        return {
            "status": "error",
            "connector": self.CONNECTOR_ID,
            "details": "Not authenticated",
            "token_file_exists": token_file_exists,
            "has_refresh_token": has_refresh_token,
            "token_age_hours": token_age_hours,
            "recovery_hint": recovery_hint,
            "last_sync": last_sync,
        }

    @staticmethod
    def _classify_api_error(exc: Exception) -> str:
        """Classify a Google API exception into an actionable error type.

        Returns a short string identifying the failure category so health_check
        callers can programmatically react (e.g. trigger re-auth for token errors).
        """
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__

        # googleapiclient.errors.HttpError carries a resp.status attribute
        status = getattr(getattr(exc, "resp", None), "status", None)

        if status == 403 or "403" in exc_str or "forbidden" in exc_str:
            return "scope_revoked"
        if status == 401 or "401" in exc_str or "unauthorized" in exc_str:
            return "token_expired"
        if "connection" in exc_type.lower() or "connection" in exc_str or "timeout" in exc_str:
            return "network_error"
        return "unknown"

    @staticmethod
    def _recovery_hint_for_error_type(error_type: str) -> str:
        """Return user-facing recovery guidance for a given error type.

        Each hint tells the user exactly what to do to restore the connector.
        """
        hints = {
            "scope_revoked": (
                "Re-authenticate via /admin connector panel — "
                "the current token may have been revoked or scopes changed"
            ),
            "token_expired": (
                "Re-authenticate via /admin connector panel — "
                "the access token has expired and could not be refreshed"
            ),
            "network_error": (
                "Check network connectivity — "
                "the Google API is unreachable. The connector will retry on next health check."
            ),
        }
        return hints.get(
            error_type,
            "Re-authenticate via /admin connector panel or check logs for details",
        )
