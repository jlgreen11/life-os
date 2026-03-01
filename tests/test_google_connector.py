"""
Tests for GoogleConnector — Gmail, Google Calendar, and Contacts integration.

The Google connector is a unified integration for three Google services:
    1. Gmail — bidirectional email sync (read and send)
    2. Google Calendar — event sync and creation
    3. Google Contacts — contact sync into entities.db

These tests cover all three sub-services plus authentication, error handling,
action execution, and health checks.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.google.connector import GoogleConnector


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def google_config():
    """Default Google connector configuration."""
    return {
        "email_address": "test@gmail.com",
        "credentials_file": "data/google_credentials.json",
        "token_file": "data/google_token.json",
        "sync_interval": 30,
        "calendars": ["primary"],
        "gmail_labels": ["INBOX", "SENT"],
    }


@pytest.fixture
def connector(db, mock_event_bus, google_config):
    """GoogleConnector instance with mocked dependencies."""
    return GoogleConnector(mock_event_bus, db, google_config)


# ------------------------------------------------------------------
# Authentication tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_success(connector):
    """Test successful authentication with valid token."""
    mock_creds = MagicMock()
    mock_creds.valid = True

    with patch("google.oauth2.credentials.Credentials") as mock_creds_class, \
         patch("googleapiclient.discovery.build") as mock_build, \
         patch.object(connector, "_load_credentials", return_value=mock_creds):

        # Mock Gmail service to return profile
        mock_service = MagicMock()
        mock_profile = {"emailAddress": "test@gmail.com"}
        mock_service.users().getProfile().execute.return_value = mock_profile
        mock_build.return_value = mock_service

        result = await connector.authenticate()

        assert result is True
        assert connector._email_address == "test@gmail.com"


@pytest.mark.asyncio
async def test_authenticate_no_token(connector):
    """Test authentication failure when no token file exists."""
    with patch.object(connector, "_load_credentials", return_value=None):
        result = await connector.authenticate()
        assert result is False


@pytest.mark.asyncio
async def test_authenticate_api_error(connector):
    """Test authentication failure when Gmail API call fails."""
    mock_creds = MagicMock()
    mock_creds.valid = True

    with patch("googleapiclient.discovery.build") as mock_build, \
         patch.object(connector, "_load_credentials", return_value=mock_creds):

        # Mock Gmail service to raise exception
        mock_service = MagicMock()
        mock_service.users().getProfile().execute.side_effect = Exception("API error")
        mock_build.return_value = mock_service

        result = await connector.authenticate()
        assert result is False


@pytest.mark.asyncio
async def test_load_credentials_valid(connector):
    """Test loading valid credentials from token file."""
    mock_creds = MagicMock()
    mock_creds.valid = True

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds):

        result = connector._load_credentials()
        assert result == mock_creds


@pytest.mark.asyncio
async def test_load_credentials_refresh_needed(connector):
    """Test loading expired credentials that need refresh."""
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "refresh_token_123"

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds), \
         patch("builtins.open", MagicMock()), \
         patch("google.auth.transport.requests.Request"):

        result = connector._load_credentials()
        assert result == mock_creds
        mock_creds.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_load_credentials_missing_file(connector):
    """Test loading credentials when token file doesn't exist."""
    with patch("os.path.exists", return_value=False):
        result = connector._load_credentials()
        assert result is None


@pytest.mark.asyncio
async def test_load_credentials_refresh_error(connector):
    """Test _load_credentials raises ValueError with actionable message when refresh fails."""
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "refresh_token_123"

    # Import the exception class to use as side_effect
    from google.auth import exceptions as google_exceptions

    mock_creds.refresh.side_effect = google_exceptions.RefreshError("Token has been revoked")

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds), \
         patch("google.auth.transport.requests.Request"):

        with pytest.raises(ValueError, match="re-authenticate via /admin connector panel"):
            connector._load_credentials()

        mock_creds.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_load_credentials_expired_no_refresh_token(connector):
    """Test _load_credentials raises ValueError when token is expired but has no refresh_token."""
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = None

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds):

        with pytest.raises(ValueError, match="no refresh_token"):
            connector._load_credentials()


@pytest.mark.asyncio
async def test_authenticate_descriptive_error_state(connector, db):
    """Test that start() passes specific error message to _update_state, not generic 'Authentication failed'."""
    error_msg = "Token refresh failed (invalid_grant) — re-authenticate via /admin connector panel"

    with patch.object(connector, "_load_credentials", side_effect=ValueError(error_msg)):

        result = await connector.authenticate()
        assert result is False
        assert connector._auth_error == error_msg

        # Simulate what start() does: verify _update_state gets the descriptive message
        await connector.start()

        # Check the state stored in the database
        with db.get_connection("state") as conn:
            row = conn.execute(
                "SELECT last_error FROM connector_state WHERE connector_id = ?",
                (connector.CONNECTOR_ID,),
            ).fetchone()
            assert row is not None
            assert "re-authenticate" in row["last_error"]
            assert "Authentication failed" != row["last_error"]


# ------------------------------------------------------------------
# Gmail sync tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_gmail_first_sync(connector, db):
    """Test initial Gmail sync fetches recent messages."""
    connector._gmail_service = MagicMock()

    # Initialize connector state in database
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO connector_state (connector_id, status, last_sync, sync_cursor)
               VALUES (?, 'active', datetime('now'), NULL)""",
            (connector.CONNECTOR_ID,)
        )

    # Mock Gmail messages.list response
    mock_messages = [{"id": f"msg{i}"} for i in range(5)]
    connector._gmail_service.users().messages().list().execute.return_value = {
        "messages": mock_messages
    }

    # Mock Gmail messages.get response
    mock_message = {
        "id": "msg1",
        "threadId": "thread1",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "test@gmail.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Message-ID", "value": "msg1"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "VGVzdCBib2R5"},  # base64 "Test body"
        }
    }
    connector._gmail_service.users().messages().get().execute.return_value = mock_message

    count = await connector._sync_gmail()

    assert count == 5
    assert connector.get_sync_cursor() is not None


@pytest.mark.asyncio
async def test_sync_gmail_incremental_sync(connector, db):
    """Test incremental Gmail sync using cursor."""
    connector._gmail_service = MagicMock()

    # Initialize connector state with a cursor
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO connector_state (connector_id, status, last_sync, sync_cursor)
               VALUES (?, 'active', datetime('now'), '1708000000')""",
            (connector.CONNECTOR_ID,)
        )

    # Mock empty response for incremental sync
    connector._gmail_service.users().messages().list().execute.return_value = {
        "messages": []
    }

    count = await connector._sync_gmail()

    assert count == 0
    # Cursor should be updated to current time
    new_cursor = connector.get_sync_cursor()
    assert new_cursor is not None
    assert int(new_cursor) > 1708000000


@pytest.mark.asyncio
async def test_sync_gmail_with_pagination(connector):
    """Test Gmail sync with multiple pages of results."""
    connector._gmail_service = MagicMock()

    # First page with nextPageToken
    first_page = {
        "messages": [{"id": "msg1"}],
        "nextPageToken": "token123"
    }
    # Second page without nextPageToken
    second_page = {
        "messages": [{"id": "msg2"}]
    }

    connector._gmail_service.users().messages().list().execute.side_effect = [
        first_page, second_page
    ]

    # Mock message details
    mock_message = {
        "id": "msg1",
        "threadId": "thread1",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "test@gmail.com"},
                {"name": "Subject", "value": "Test"},
                {"name": "Message-ID", "value": "msg1"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "VGVzdA=="},
        }
    }
    connector._gmail_service.users().messages().get().execute.return_value = mock_message

    count = await connector._sync_gmail()

    assert count == 2


@pytest.mark.asyncio
async def test_process_gmail_message_inbound(connector):
    """Test processing an inbound Gmail message."""
    message = {
        "id": "msg123",
        "threadId": "thread123",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "test@gmail.com"},
                {"name": "Cc", "value": "cc@example.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Message-ID", "value": "<msg123@example.com>"},
                {"name": "In-Reply-To", "value": "<parent@example.com>"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "VGVzdCBib2R5"},  # base64 "Test body"
        }
    }

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock) as mock_publish:
        count = await connector._process_gmail_message(message)

        assert count == 1
        mock_publish.assert_called_once()

        call_args = mock_publish.call_args
        assert call_args[0][0] == "email.received"
        payload = call_args[0][1]
        assert payload["from_address"] == "sender@example.com"
        assert "test@gmail.com" in payload["to_addresses"]
        assert "cc@example.com" in payload["cc_addresses"]
        assert payload["subject"] == "Test Subject"
        assert payload["is_reply"] is True


@pytest.mark.asyncio
async def test_process_gmail_message_outbound(connector):
    """Test processing an outbound Gmail message."""
    connector._email_address = "test@gmail.com"

    message = {
        "id": "msg123",
        "threadId": "thread123",
        "labelIds": ["SENT"],
        "payload": {
            "headers": [
                {"name": "From", "value": "test@gmail.com"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Subject", "value": "Outbound Test"},
                {"name": "Message-ID", "value": "<msg123@gmail.com>"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "T3V0Ym91bmQgYm9keQ=="},  # base64 "Outbound body"
        }
    }

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock) as mock_publish:
        count = await connector._process_gmail_message(message)

        assert count == 1
        call_args = mock_publish.call_args
        assert call_args[0][0] == "email.sent"
        payload = call_args[0][1]
        assert payload["direction"] == "outbound"


@pytest.mark.asyncio
async def test_process_gmail_message_with_attachments(connector):
    """Test processing Gmail message with attachments."""
    message = {
        "id": "msg123",
        "threadId": "thread123",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "test@gmail.com"},
                {"name": "Subject", "value": "With Attachments"},
                {"name": "Message-ID", "value": "<msg123@example.com>"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": "Qm9keQ=="}},
                {"filename": "document.pdf", "mimeType": "application/pdf"},
                {"filename": "image.png", "mimeType": "image/png"},
            ]
        }
    }

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock) as mock_publish:
        await connector._process_gmail_message(message)

        payload = mock_publish.call_args[0][1]
        assert payload["has_attachments"] is True
        assert len(payload["attachment_names"]) == 2
        assert "document.pdf" in payload["attachment_names"]
        assert "image.png" in payload["attachment_names"]


@pytest.mark.asyncio
async def test_process_gmail_message_urgent_priority(connector):
    """Test urgent priority detection from subject line."""
    message = {
        "id": "msg123",
        "threadId": "thread123",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "test@gmail.com"},
                {"name": "Subject", "value": "URGENT: Server Down"},
                {"name": "Message-ID", "value": "<msg123@example.com>"},
                {"name": "Date", "value": "Mon, 15 Feb 2026 10:00:00 +0000"},
            ],
            "mimeType": "text/plain",
            "body": {"data": "Qm9keQ=="},
        }
    }

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock) as mock_publish:
        await connector._process_gmail_message(message)

        # Check priority parameter
        assert mock_publish.call_args[1]["priority"] == "high"


@pytest.mark.asyncio
async def test_extract_gmail_body_plain_text(connector):
    """Test extracting plain text body from Gmail payload."""
    payload = {
        "mimeType": "text/plain",
        "body": {"data": "VGVzdCBib2R5"}  # base64 "Test body"
    }

    plain, html = connector._extract_gmail_body(payload)

    assert plain == "Test body"
    assert html == ""


@pytest.mark.asyncio
async def test_extract_gmail_body_html(connector):
    """Test extracting HTML body from Gmail payload."""
    import base64
    html_content = "<html><body>Test</body></html>"
    encoded = base64.urlsafe_b64encode(html_content.encode()).decode()

    payload = {
        "mimeType": "text/html",
        "body": {"data": encoded}
    }

    plain, html = connector._extract_gmail_body(payload)

    assert plain == ""
    assert html == html_content


@pytest.mark.asyncio
async def test_extract_gmail_body_multipart(connector):
    """Test extracting body from multipart Gmail message."""
    import base64

    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {"data": base64.urlsafe_b64encode(b"Plain text").decode()}
            },
            {
                "mimeType": "text/html",
                "body": {"data": base64.urlsafe_b64encode(b"<html>HTML</html>").decode()}
            }
        ]
    }

    plain, html = connector._extract_gmail_body(payload)

    assert plain == "Plain text"
    assert html == "<html>HTML</html>"


@pytest.mark.asyncio
async def test_parse_email_address(connector):
    """Test parsing email address from 'Name <email>' format."""
    result = connector._parse_email_address("John Doe <john@example.com>")
    assert result == "john@example.com"

    result = connector._parse_email_address("jane@example.com")
    assert result == "jane@example.com"


@pytest.mark.asyncio
async def test_parse_email_list(connector):
    """Test parsing comma-separated email list."""
    result = connector._parse_email_list("John <john@example.com>, jane@example.com")
    assert len(result) == 2
    assert "john@example.com" in result
    assert "jane@example.com" in result

    result = connector._parse_email_list("")
    assert result == []


# ------------------------------------------------------------------
# Calendar sync tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_calendar_success(connector):
    """Test successful calendar event sync."""
    connector._calendar_service = MagicMock()

    # Mock calendar events response
    mock_events = {
        "items": [
            {
                "id": "event1",
                "summary": "Team Meeting",
                "description": "Weekly sync",
                "location": "Conference Room",
                "start": {"dateTime": "2026-02-20T10:00:00Z"},
                "end": {"dateTime": "2026-02-20T11:00:00Z"},
                "attendees": [
                    {"email": "alice@example.com"},
                    {"email": "bob@example.com"}
                ],
                "organizer": {"email": "organizer@example.com"}
            }
        ]
    }

    connector._calendar_service.events().list().execute.return_value = mock_events

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock):
        count = await connector._sync_calendar()

        assert count == 1


@pytest.mark.asyncio
async def test_sync_calendar_all_day_event(connector):
    """Test syncing all-day calendar events."""
    connector._calendar_service = MagicMock()

    mock_events = {
        "items": [
            {
                "id": "event1",
                "summary": "Holiday",
                "start": {"date": "2026-02-20"},
                "end": {"date": "2026-02-21"},
                "attendees": []
            }
        ]
    }

    connector._calendar_service.events().list().execute.return_value = mock_events

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock) as mock_publish:
        await connector._sync_calendar()

        payload = mock_publish.call_args[0][1]
        assert payload["is_all_day"] is True
        assert payload["start_time"] == "2026-02-20"


@pytest.mark.asyncio
async def test_sync_calendar_multiple_calendars(connector):
    """Test syncing multiple calendars."""
    connector._calendar_service = MagicMock()
    connector._calendars = ["primary", "work@example.com"]

    # Mock response for each calendar
    mock_events = {"items": [{"id": "event1", "summary": "Event",
                               "start": {"dateTime": "2026-02-20T10:00:00Z"},
                               "end": {"dateTime": "2026-02-20T11:00:00Z"}}]}

    connector._calendar_service.events().list().execute.return_value = mock_events

    with patch.object(connector, "_publish_with_retry", new_callable=AsyncMock):
        count = await connector._sync_calendar()

        # Should sync both calendars
        assert count == 2


@pytest.mark.asyncio
async def test_sync_calendar_api_error(connector):
    """Test calendar sync handles API errors gracefully."""
    connector._calendar_service = MagicMock()
    connector._calendar_service.events().list().execute.side_effect = Exception("API error")

    count = await connector._sync_calendar()

    # Should return 0 without crashing
    assert count == 0


# ------------------------------------------------------------------
# Contacts sync tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_contacts_success(connector, db):
    """Test successful contact sync from Google."""
    connector._people_service = MagicMock()

    # Mock Google contacts response
    mock_contacts = {
        "connections": [
            {
                "names": [{"displayName": "Alice Smith"}],
                "emailAddresses": [{"value": "alice@example.com"}],
                "phoneNumbers": [{"value": "+15551234567"}]
            },
            {
                "names": [{"displayName": "Bob Jones"}],
                "emailAddresses": [{"value": "bob@example.com"}],
                "phoneNumbers": []
            }
        ]
    }

    connector._people_service.people().connections().list().execute.return_value = mock_contacts

    count = await connector._sync_contacts()

    assert count == 2

    # Verify contacts were inserted into database
    with db.get_connection("entities") as conn:
        contacts = conn.execute("SELECT * FROM contacts").fetchall()
        assert len(contacts) == 2


@pytest.mark.asyncio
async def test_sync_contacts_deduplication(connector, db):
    """Test contact sync deduplicates by email address."""
    connector._people_service = MagicMock()

    # Pre-insert a contact with the same email
    with db.get_connection("entities") as conn:
        contact_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO contacts (id, name, emails, phones, channels, domains, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, '["personal"]', ?, ?)""",
            (contact_id, "Alice Old", json.dumps(["alice@example.com"]),
             json.dumps([]), json.dumps({}), now, now)
        )
        conn.execute(
            """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
               VALUES (?, 'email', ?)""",
            ("alice@example.com", contact_id)
        )

    # Mock Google contact with same email
    mock_contacts = {
        "connections": [
            {
                "names": [{"displayName": "Alice Smith"}],
                "emailAddresses": [{"value": "alice@example.com"}],
                "phoneNumbers": [{"value": "+15551234567"}]
            }
        ]
    }

    connector._people_service.people().connections().list().execute.return_value = mock_contacts

    count = await connector._sync_contacts()

    assert count == 1

    # Should still have only one contact
    with db.get_connection("entities") as conn:
        contacts = conn.execute("SELECT * FROM contacts").fetchall()
        assert len(contacts) == 1
        # Should have updated the existing contact
        assert json.loads(contacts[0]["emails"]) == ["alice@example.com"]


@pytest.mark.asyncio
async def test_sync_contacts_with_pagination(connector, db):
    """Test contact sync handles pagination."""
    connector._people_service = MagicMock()

    # First page
    first_page = {
        "connections": [
            {
                "names": [{"displayName": "Alice"}],
                "emailAddresses": [{"value": "alice@example.com"}],
                "phoneNumbers": []
            }
        ],
        "nextPageToken": "token123"
    }

    # Second page
    second_page = {
        "connections": [
            {
                "names": [{"displayName": "Bob"}],
                "emailAddresses": [{"value": "bob@example.com"}],
                "phoneNumbers": []
            }
        ]
    }

    connector._people_service.people().connections().list().execute.side_effect = [
        first_page, second_page
    ]

    count = await connector._sync_contacts()

    assert count == 2


@pytest.mark.asyncio
async def test_upsert_contact_new(connector, db):
    """Test upserting a new contact."""
    now = datetime.now(timezone.utc).isoformat()

    person = {
        "names": [{"displayName": "Charlie Brown"}],
        "emailAddresses": [{"value": "charlie@example.com"}],
        "phoneNumbers": [{"value": "+15559999999"}]
    }

    with db.get_connection("entities") as conn:
        count = connector._upsert_contact(conn, person, {}, now)

        assert count == 1

        # Verify contact was created
        contact = conn.execute(
            "SELECT * FROM contacts WHERE name = ?", ("Charlie Brown",)
        ).fetchone()
        assert contact is not None
        assert json.loads(contact["emails"]) == ["charlie@example.com"]


@pytest.mark.asyncio
async def test_upsert_contact_skip_no_contact_info(connector, db):
    """Test upsert skips contacts without email or phone."""
    now = datetime.now(timezone.utc).isoformat()

    person = {
        "names": [{"displayName": "No Contact"}],
        "emailAddresses": [],
        "phoneNumbers": []
    }

    with db.get_connection("entities") as conn:
        count = connector._upsert_contact(conn, person, {}, now)

        assert count == 0


# ------------------------------------------------------------------
# Execute actions tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_send_email(connector):
    """Test sending email via Gmail API."""
    connector._gmail_service = MagicMock()
    connector._email_address = "test@gmail.com"

    mock_result = {"id": "sent123"}
    connector._gmail_service.users().messages().send().execute.return_value = mock_result

    params = {
        "to": ["recipient@example.com"],
        "subject": "Test Email",
        "body": "This is a test email body"
    }

    result = await connector.execute("send_email", params)

    assert result["status"] == "sent"
    assert result["to"] == ["recipient@example.com"]
    assert result["message_id"] == "sent123"


@pytest.mark.asyncio
async def test_execute_send_email_with_cc_and_html(connector):
    """Test sending email with CC and HTML body."""
    connector._gmail_service = MagicMock()
    connector._email_address = "test@gmail.com"

    mock_result = {"id": "sent123"}
    connector._gmail_service.users().messages().send().execute.return_value = mock_result

    params = {
        "to": ["recipient@example.com"],
        "cc": ["cc@example.com"],
        "subject": "Test Email",
        "body": "Plain text",
        "body_html": "<html><body>HTML body</body></html>"
    }

    result = await connector.execute("send_email", params)

    assert result["status"] == "sent"


@pytest.mark.asyncio
async def test_execute_reply_email(connector):
    """Test replying to an email with threading."""
    connector._gmail_service = MagicMock()
    connector._email_address = "test@gmail.com"

    mock_result = {"id": "reply123"}
    connector._gmail_service.users().messages().send().execute.return_value = mock_result

    params = {
        "to": ["original@example.com"],
        "original_subject": "Original Subject",
        "in_reply_to": "<msg123@example.com>",
        "thread_id": "thread123",
        "body": "This is my reply"
    }

    result = await connector.execute("reply_email", params)

    assert result["status"] == "sent"
    assert "Re:" in result["subject"]


@pytest.mark.asyncio
async def test_execute_create_event(connector):
    """Test creating a calendar event."""
    connector._calendar_service = MagicMock()

    mock_result = {
        "id": "event123",
        "htmlLink": "https://calendar.google.com/event?eid=event123"
    }
    connector._calendar_service.events().insert().execute.return_value = mock_result

    params = {
        "title": "New Meeting",
        "description": "Discuss project",
        "location": "Office",
        "start_time": "2026-02-20T10:00:00Z",
        "end_time": "2026-02-20T11:00:00Z",
        "is_all_day": False,
        "timezone": "UTC",
        "attendees": ["alice@example.com", "bob@example.com"]
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "created"
    assert result["event_id"] == "event123"
    assert result["link"] == "https://calendar.google.com/event?eid=event123"


@pytest.mark.asyncio
async def test_execute_create_all_day_event(connector):
    """Test creating an all-day calendar event."""
    connector._calendar_service = MagicMock()

    mock_result = {"id": "event123", "htmlLink": "https://calendar.google.com/event?eid=event123"}
    connector._calendar_service.events().insert().execute.return_value = mock_result

    params = {
        "title": "Holiday",
        "start_time": "2026-02-20",
        "is_all_day": True
    }

    result = await connector.execute("create_event", params)

    assert result["status"] == "created"


@pytest.mark.asyncio
async def test_execute_unknown_action(connector):
    """Test execute raises ValueError for unknown actions."""
    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("unknown_action", {})


# ------------------------------------------------------------------
# Health check tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_authenticated(connector):
    """Test health check when connector is authenticated."""
    connector._gmail_service = MagicMock()

    mock_profile = {"emailAddress": "test@gmail.com"}
    connector._gmail_service.users().getProfile().execute.return_value = mock_profile

    result = await connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "google"
    assert result["email"] == "test@gmail.com"


@pytest.mark.asyncio
async def test_health_check_not_authenticated(connector):
    """Test health check when connector is not authenticated."""
    connector._gmail_service = None

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Not authenticated" in result["details"]


@pytest.mark.asyncio
async def test_health_check_api_error(connector):
    """Test health check when Gmail API call fails."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = Exception("Network error")

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Network error" in result["details"]


# ------------------------------------------------------------------
# Delayed sync loop tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_with_delayed_sync(connector):
    """Test start() uses delayed sync loop to avoid NATS flooding."""
    with patch.object(connector, "authenticate", return_value=True), \
         patch.object(connector, "_delayed_sync_loop", new_callable=AsyncMock) as mock_delayed, \
         patch.object(connector.bus, "subscribe", new_callable=AsyncMock):

        await connector.start()

        # Should have called delayed sync loop
        assert connector._running is True


@pytest.mark.asyncio
async def test_publish_with_retry_success(connector):
    """Test publish with retry succeeds on first attempt."""
    with patch.object(connector, "publish_event", new_callable=AsyncMock) as mock_publish:
        await connector._publish_with_retry("test.event", {"data": "test"})

        mock_publish.assert_called_once()


@pytest.mark.asyncio
async def test_publish_with_retry_timeout_recovery(connector):
    """Test publish with retry recovers from timeout errors."""
    with patch.object(connector, "publish_event", new_callable=AsyncMock) as mock_publish:
        # First call times out, second succeeds
        mock_publish.side_effect = [
            Exception("timeout error"),
            None
        ]

        await connector._publish_with_retry("test.event", {"data": "test"}, max_retries=3)

        # Should have retried
        assert mock_publish.call_count == 2
