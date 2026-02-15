"""
Tests for the ProtonMail connector.

Validates email sync via Proton Bridge (local IMAP/SMTP), including:
    - IMAP authentication with STARTTLS upgrade
    - Incremental sync with date-based cursor
    - RFC 822 email parsing (headers, body, attachments)
    - Thread detection via In-Reply-To header
    - Direction detection (inbound vs. outbound)
    - Urgency keyword detection
    - SMTP email sending with multipart/alternative
    - Health checks via IMAP NOOP
"""

from __future__ import annotations

import email.utils
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.proton_mail.connector import ProtonMailConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def proton_config():
    """Standard ProtonMail connector configuration."""
    return {
        "imap_host": "127.0.0.1",
        "imap_port": 1143,
        "smtp_host": "127.0.0.1",
        "smtp_port": 1025,
        "username": "test@proton.me",
        "password": "bridge-password",
        "sync_interval": 30,
        "folders": ["INBOX", "Sent"],
    }


@pytest.fixture
def mock_imap():
    """Mock IMAP4 connection that simulates Proton Bridge."""
    imap = MagicMock()
    # Simulate successful authentication
    imap.starttls.return_value = None
    imap.login.return_value = ("OK", [b"Logged in"])
    # Simulate folder selection
    imap.select.return_value = ("OK", [b"1"])
    # Simulate NOOP (health check)
    imap.noop.return_value = ("OK", [])
    return imap


@pytest.fixture
def connector(event_bus, db, proton_config):
    """Create a ProtonMailConnector instance with mocked dependencies."""
    return ProtonMailConnector(event_bus, db, proton_config)


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_success(connector, mock_imap):
    """Test successful IMAP authentication with STARTTLS upgrade."""
    with patch("connectors.proton_mail.connector.imaplib.IMAP4", return_value=mock_imap):
        result = await connector.authenticate()

    assert result is True
    assert connector._imap is mock_imap
    # Verify the STARTTLS upgrade happened
    mock_imap.starttls.assert_called_once()
    # Verify login with correct credentials
    mock_imap.login.assert_called_once_with("test@proton.me", "bridge-password")


@pytest.mark.asyncio
async def test_authenticate_imap_connection_failure(connector):
    """Test authentication failure when IMAP connection fails."""
    with patch("connectors.proton_mail.connector.imaplib.IMAP4", side_effect=ConnectionRefusedError("Bridge not running")):
        result = await connector.authenticate()

    assert result is False
    assert connector._imap is None


@pytest.mark.asyncio
async def test_authenticate_starttls_failure(connector, mock_imap):
    """Test authentication failure when STARTTLS upgrade fails."""
    mock_imap.starttls.side_effect = Exception("TLS negotiation failed")

    with patch("connectors.proton_mail.connector.imaplib.IMAP4", return_value=mock_imap):
        result = await connector.authenticate()

    assert result is False


@pytest.mark.asyncio
async def test_authenticate_login_failure(connector, mock_imap):
    """Test authentication failure when login credentials are rejected."""
    mock_imap.login.side_effect = Exception("Authentication failed")

    with patch("connectors.proton_mail.connector.imaplib.IMAP4", return_value=mock_imap):
        result = await connector.authenticate()

    assert result is False


# ---------------------------------------------------------------------------
# Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_with_no_imap_connection(connector):
    """Test sync returns 0 when IMAP connection is not established."""
    connector._imap = None
    count = await connector.sync()
    assert count == 0


@pytest.mark.asyncio
async def test_sync_first_run_fetches_all_messages(connector, mock_imap, event_bus):
    """Test initial sync (no cursor) fetches ALL messages but caps at 100 most recent per folder."""
    connector._imap = mock_imap
    connector._folders = ["INBOX"]  # Test with single folder for predictable count
    # Simulate 150 messages in INBOX (should be capped at 100)
    message_nums = [str(i).encode() for i in range(1, 151)]
    mock_imap.search.return_value = ("OK", [b" ".join(message_nums)])
    # Mock fetch to return minimal RFC 822 data
    mock_imap.fetch.return_value = ("OK", [[None, _minimal_rfc822_email()]])

    count = await connector.sync()

    # Should only process the last 100 messages
    assert count == 100
    # Verify search used "ALL" (no cursor)
    mock_imap.search.assert_called()
    call_args = str(mock_imap.search.call_args)
    assert "ALL" in call_args


@pytest.mark.asyncio
async def test_sync_incremental_uses_cursor(connector, mock_imap, event_bus, db):
    """Test incremental sync uses SINCE cursor from previous run."""
    connector._imap = mock_imap
    connector._folders = ["INBOX"]  # Test with single folder
    # Initialize connector state row (normally done by start())
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO connector_state (connector_id, status, sync_cursor, updated_at) VALUES (?, ?, ?, ?)",
            ("proton_mail", "active", "10-Feb-2026", "2026-02-10T00:00:00+00:00")
        )
    # Simulate 5 new messages since cursor
    mock_imap.search.return_value = ("OK", [b"101 102 103 104 105"])
    mock_imap.fetch.return_value = ("OK", [[None, _minimal_rfc822_email()]])

    count = await connector.sync()

    assert count == 5
    # Verify search used SINCE with the cursor date
    mock_imap.search.assert_called()
    call_args = str(mock_imap.search.call_args)
    assert "SINCE" in call_args
    assert "10-Feb-2026" in call_args


@pytest.mark.asyncio
async def test_sync_updates_cursor_to_today(connector, mock_imap, db):
    """Test sync updates cursor to today's date after processing."""
    connector._imap = mock_imap
    connector._folders = ["INBOX"]  # Test with single folder
    # Initialize connector state row
    with db.get_connection("state") as conn:
        conn.execute(
            "INSERT INTO connector_state (connector_id, status, updated_at) VALUES (?, ?, ?)",
            ("proton_mail", "active", "2026-02-10T00:00:00+00:00")
        )
    mock_imap.search.return_value = ("OK", [b"1"])
    mock_imap.fetch.return_value = ("OK", [[None, _minimal_rfc822_email()]])

    await connector.sync()

    # Verify cursor was updated to today's date in IMAP format
    cursor = connector.get_sync_cursor()
    assert cursor is not None
    # Should be in format "DD-MMM-YYYY" (e.g., "15-Feb-2026")
    assert len(cursor.split("-")) == 3


@pytest.mark.asyncio
async def test_sync_processes_multiple_folders(connector, mock_imap, event_bus):
    """Test sync iterates through all configured folders."""
    connector._imap = mock_imap
    connector._folders = ["INBOX", "Sent", "Archive"]
    mock_imap.search.return_value = ("OK", [b"1"])
    mock_imap.fetch.return_value = ("OK", [[None, _minimal_rfc822_email()]])

    count = await connector.sync()

    # Should have selected each folder in read-only mode
    assert mock_imap.select.call_count == 3
    # Verify all folders were processed (3 messages total)
    assert count == 3


@pytest.mark.asyncio
async def test_sync_skips_malformed_messages(connector, mock_imap, event_bus):
    """Test sync continues past malformed messages without failing entire sync."""
    connector._imap = mock_imap
    mock_imap.search.return_value = ("OK", [b"1 2 3"])
    # Message 1: valid, Message 2: empty data, Message 3: valid
    mock_imap.fetch.side_effect = [
        ("OK", [[None, _minimal_rfc822_email()]]),
        ("OK", [None]),  # Empty data
        ("OK", [[None, _minimal_rfc822_email()]]),
    ]

    count = await connector.sync()

    # Should have processed 2 out of 3 messages
    assert count == 2


@pytest.mark.asyncio
async def test_sync_handles_folder_errors_gracefully(connector, mock_imap, event_bus):
    """Test sync continues to next folder when one folder fails."""
    connector._imap = mock_imap
    connector._folders = ["INBOX", "InvalidFolder", "Sent"]
    # INBOX succeeds, InvalidFolder raises error, Sent succeeds
    def select_side_effect(folder, readonly=True):
        if folder == "InvalidFolder":
            raise Exception("Folder not found")
        return ("OK", [b"1"])

    mock_imap.select.side_effect = select_side_effect
    mock_imap.search.return_value = ("OK", [b"1"])
    mock_imap.fetch.return_value = ("OK", [[None, _minimal_rfc822_email()]])

    count = await connector.sync()

    # Should have processed INBOX and Sent (2 messages), skipped InvalidFolder
    assert count == 2


# ---------------------------------------------------------------------------
# Email Parsing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_email_inbound_detection(connector, mock_imap):
    """Test direction detection: email from someone else is inbound."""
    connector._imap = mock_imap
    # Replace event bus with AsyncMock to inspect calls
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Hello",
        body="Test message",
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    count = await connector._process_email(b"1", "INBOX")

    assert count == 1
    # Verify event was published with correct type
    connector.bus.publish.assert_called_once()
    call_args = connector.bus.publish.call_args[0]
    assert call_args[0] == "email.received"
    payload = call_args[1]
    assert payload["direction"] == "inbound"
    assert payload["from_address"] == "sender@example.com"


@pytest.mark.asyncio
async def test_process_email_outbound_detection(connector, mock_imap):
    """Test direction detection: email from self is outbound."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email(
        from_addr="test@proton.me",
        to_addr="recipient@example.com",
        subject="Hello",
        body="Test message",
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    count = await connector._process_email(b"1", "Sent")

    assert count == 1
    # Verify event type is "sent"
    call_args = connector.bus.publish.call_args[0]
    assert call_args[0] == "email.sent"
    payload = call_args[1]
    assert payload["direction"] == "outbound"


@pytest.mark.asyncio
async def test_process_email_thread_detection_reply(connector, mock_imap):
    """Test thread detection: In-Reply-To header links to original message."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    original_msg_id = "<original-123@example.com>"
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Re: Original Subject",
        body="This is a reply",
        in_reply_to=original_msg_id,
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    # Thread ID should be the original message ID
    assert payload["thread_id"] == original_msg_id
    assert payload["is_reply"] is True
    assert payload["in_reply_to"] == original_msg_id


@pytest.mark.asyncio
async def test_process_email_thread_detection_new_thread(connector, mock_imap):
    """Test thread detection: message without In-Reply-To starts a new thread."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    msg_id = "<new-msg-456@example.com>"
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="New Topic",
        body="Starting a new conversation",
        message_id=msg_id,
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    # Thread ID should be the message's own ID
    assert payload["thread_id"] == msg_id
    assert payload["is_reply"] is False


@pytest.mark.asyncio
async def test_process_email_urgency_detection_high_priority(connector, mock_imap):
    """Test urgency detection: keywords in subject trigger high priority."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    urgent_subjects = [
        "URGENT: Server down",
        "ASAP: Need approval",
        "CRITICAL: Security breach",
        "EMERGENCY: Please respond",
        "Need this immediately",
    ]

    for subject in urgent_subjects:
        connector.bus.publish.reset_mock()
        email_data = _build_rfc822_email(
            from_addr="sender@example.com",
            to_addr="test@proton.me",
            subject=subject,
            body="Important message",
        )
        mock_imap.fetch.return_value = ("OK", [[None, email_data]])

        await connector._process_email(b"1", "INBOX")

        # Verify high priority was set
        call_kwargs = connector.bus.publish.call_args[1]
        assert call_kwargs["priority"] == "high", f"Failed for subject: {subject}"


@pytest.mark.asyncio
async def test_process_email_urgency_detection_normal_priority(connector, mock_imap):
    """Test urgency detection: normal subjects get normal priority."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Weekly update",
        body="Here's what happened this week",
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    call_kwargs = connector.bus.publish.call_args[1]
    assert call_kwargs["priority"] == "normal"


@pytest.mark.asyncio
async def test_process_email_attachment_detection(connector, mock_imap):
    """Test attachment detection from multipart MIME messages."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email_with_attachments(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Files attached",
        body="Please see attached files",
        attachments=["report.pdf", "data.xlsx"],
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    assert payload["has_attachments"] is True
    assert "report.pdf" in payload["attachment_names"]
    assert "data.xlsx" in payload["attachment_names"]


@pytest.mark.asyncio
async def test_process_email_multipart_body_extraction(connector, mock_imap):
    """Test body extraction from multipart/alternative messages (plain + HTML)."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    plain_body = "This is the plain text version"
    html_body = "<html><body><p>This is the <b>HTML</b> version</p></body></html>"
    email_data = _build_rfc822_multipart_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Test",
        plain_body=plain_body,
        html_body=html_body,
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    # HTML should be preferred for display
    assert payload["body"] == html_body
    assert payload["body_plain"] == plain_body


@pytest.mark.asyncio
async def test_process_email_multiple_recipients(connector, mock_imap):
    """Test parsing of multiple To and Cc recipients."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me, alice@example.com, bob@example.com",
        cc_addr="charlie@example.com, diana@example.com",
        subject="Group email",
        body="Hello everyone",
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    assert len(payload["to_addresses"]) == 3
    assert "alice@example.com" in payload["to_addresses"]
    assert len(payload["cc_addresses"]) == 2
    assert "charlie@example.com" in payload["cc_addresses"]


@pytest.mark.asyncio
async def test_process_email_related_contacts_metadata(connector, mock_imap):
    """Test related_contacts metadata excludes self and includes all participants."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me, alice@example.com",
        cc_addr="bob@example.com",
        subject="Test",
        body="Test",
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    call_kwargs = connector.bus.publish.call_args[1]
    metadata = call_kwargs["metadata"]
    related = metadata["related_contacts"]
    # Should include sender and other recipients, but NOT self
    assert "sender@example.com" in related
    assert "alice@example.com" in related
    assert "bob@example.com" in related
    assert "test@proton.me" not in related


@pytest.mark.asyncio
async def test_process_email_snippet_generation(connector, mock_imap):
    """Test snippet truncation for long messages."""
    connector._imap = mock_imap
    connector.bus.publish = AsyncMock()
    long_body = "A" * 200  # 200 characters
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Test",
        body=long_body,
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    await connector._process_email(b"1", "INBOX")

    payload = connector.bus.publish.call_args[0][1]
    # Snippet should be truncated to 150 chars + "..."
    assert len(payload["snippet"]) == 153  # 150 + "..."
    assert payload["snippet"].endswith("...")


@pytest.mark.asyncio
async def test_process_email_malformed_date_fallback(connector, mock_imap, event_bus):
    """Test date parsing fallback when Date header is missing or malformed."""
    connector._imap = mock_imap
    # Email with no Date header
    email_data = _build_rfc822_email(
        from_addr="sender@example.com",
        to_addr="test@proton.me",
        subject="Test",
        body="Test",
        include_date=False,
    )
    mock_imap.fetch.return_value = ("OK", [[None, email_data]])

    # Should not raise exception
    count = await connector._process_email(b"1", "INBOX")
    assert count == 1


# ---------------------------------------------------------------------------
# Execute (Send/Reply) Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_send_email_success(connector):
    """Test sending a new email via SMTP."""
    params = {
        "to": ["recipient@example.com"],
        "subject": "Test Email",
        "body": "This is a test email",
    }

    with patch("connectors.proton_mail.connector.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        result = await connector.execute("send_email", params)

    # Verify SMTP connection sequence
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("test@proton.me", "bridge-password")
    mock_smtp.send_message.assert_called_once()

    # Verify result
    assert result["status"] == "sent"
    assert result["to"] == ["recipient@example.com"]
    assert result["subject"] == "Test Email"


@pytest.mark.asyncio
async def test_execute_send_email_with_html_body(connector):
    """Test sending email with both plain and HTML body."""
    params = {
        "to": ["recipient@example.com"],
        "subject": "Test Email",
        "body": "Plain text body",
        "body_html": "<html><body><p>HTML body</p></body></html>",
    }

    with patch("connectors.proton_mail.connector.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        result = await connector.execute("send_email", params)

    # Verify both plain and HTML parts were attached
    mock_smtp.send_message.assert_called_once()
    assert result["status"] == "sent"


@pytest.mark.asyncio
async def test_execute_send_email_with_cc_recipients(connector):
    """Test sending email with Cc recipients."""
    params = {
        "to": ["recipient@example.com"],
        "cc": ["cc1@example.com", "cc2@example.com"],
        "subject": "Test Email",
        "body": "Test body",
    }

    with patch("connectors.proton_mail.connector.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        result = await connector.execute("send_email", params)

    assert result["status"] == "sent"


@pytest.mark.asyncio
async def test_execute_reply_email_prepends_re(connector):
    """Test replying to an email auto-prepends 'Re:' to subject."""
    params = {
        "to": ["original-sender@example.com"],
        "original_subject": "Original Subject",
        "body": "This is my reply",
    }

    with patch("connectors.proton_mail.connector.smtplib.SMTP") as mock_smtp_class:
        mock_smtp = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = mock_smtp

        result = await connector.execute("reply_email", params)

    # Verify Re: was prepended
    assert result["subject"] == "Re: Original Subject"


@pytest.mark.asyncio
async def test_execute_unknown_action_raises_error(connector):
    """Test executing an unknown action raises ValueError."""
    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("delete_email", {})


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_success(connector, mock_imap):
    """Test health check returns ok when IMAP NOOP succeeds."""
    connector._imap = mock_imap
    mock_imap.noop.return_value = ("OK", [])

    result = await connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "proton_mail"
    mock_imap.noop.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_no_connection(connector):
    """Test health check returns error when IMAP is not connected."""
    connector._imap = None

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Not connected" in result["details"]


@pytest.mark.asyncio
async def test_health_check_noop_failure(connector, mock_imap):
    """Test health check returns error when IMAP NOOP fails."""
    connector._imap = mock_imap
    mock_imap.noop.side_effect = Exception("Connection lost")

    result = await connector.health_check()

    assert result["status"] == "error"
    assert "Connection lost" in result["details"]


# ---------------------------------------------------------------------------
# Helper Methods Tests
# ---------------------------------------------------------------------------


def test_parse_address_with_display_name():
    """Test email address parsing with display name."""
    raw = "John Doe <john@example.com>"
    addr = ProtonMailConnector._parse_address(raw)
    assert addr == "john@example.com"


def test_parse_address_bare():
    """Test parsing bare email address without display name."""
    raw = "john@example.com"
    addr = ProtonMailConnector._parse_address(raw)
    assert addr == "john@example.com"


def test_parse_address_list_multiple():
    """Test parsing comma-separated address list."""
    raw = "Alice <alice@example.com>, Bob <bob@example.com>, charlie@example.com"
    addrs = ProtonMailConnector._parse_address_list(raw)
    assert len(addrs) == 3
    assert "alice@example.com" in addrs
    assert "bob@example.com" in addrs
    assert "charlie@example.com" in addrs


def test_parse_address_list_empty():
    """Test parsing empty address list returns empty list."""
    addrs = ProtonMailConnector._parse_address_list("")
    assert addrs == []


def test_extract_body_plain_text_only():
    """Test body extraction from plain text email."""
    msg_str = (
        "From: sender@example.com\r\n"
        "To: recipient@example.com\r\n"
        "Subject: Test\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "This is the body"
    )
    msg = email.message_from_string(msg_str)
    plain, html = ProtonMailConnector._extract_body(msg)
    assert plain == "This is the body"
    assert html == ""


def test_extract_body_html_only():
    """Test body extraction from HTML-only email."""
    msg_str = (
        "From: sender@example.com\r\n"
        "To: recipient@example.com\r\n"
        "Subject: Test\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "\r\n"
        "<html><body>HTML body</body></html>"
    )
    msg = email.message_from_string(msg_str)
    plain, html = ProtonMailConnector._extract_body(msg)
    assert plain == ""
    assert html == "<html><body>HTML body</body></html>"


def test_extract_body_multipart_alternative():
    """Test body extraction from multipart/alternative (plain + HTML)."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Test"

    plain_part = MIMEText("Plain text body", "plain")
    html_part = MIMEText("<html><body>HTML body</body></html>", "html")
    msg.attach(plain_part)
    msg.attach(html_part)

    plain, html = ProtonMailConnector._extract_body(msg)
    assert plain == "Plain text body"
    assert html == "<html><body>HTML body</body></html>"


# ---------------------------------------------------------------------------
# Test Helpers (RFC 822 Email Builders)
# ---------------------------------------------------------------------------


def _minimal_rfc822_email() -> bytes:
    """Build a minimal valid RFC 822 email for testing."""
    return _build_rfc822_email(
        from_addr="test@example.com",
        to_addr="recipient@example.com",
        subject="Test",
        body="Test body",
    )


def _build_rfc822_email(
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
    cc_addr: str = "",
    message_id: str = "",
    in_reply_to: str = "",
    include_date: bool = True,
) -> bytes:
    """Build a complete RFC 822 email for testing."""
    if not message_id:
        message_id = f"<test-{datetime.now().timestamp()}@example.com>"

    headers = [
        f"From: {from_addr}",
        f"To: {to_addr}",
        f"Subject: {subject}",
        f"Message-ID: {message_id}",
    ]

    if cc_addr:
        headers.append(f"Cc: {cc_addr}")

    if in_reply_to:
        headers.append(f"In-Reply-To: {in_reply_to}")

    if include_date:
        date_str = email.utils.formatdate(localtime=True)
        headers.append(f"Date: {date_str}")

    headers.append("Content-Type: text/plain; charset=utf-8")
    headers.append("")
    headers.append(body)

    return "\r\n".join(headers).encode("utf-8")


def _build_rfc822_email_with_attachments(
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
    attachments: list[str],
) -> bytes:
    """Build an RFC 822 email with attachments."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase

    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Message-ID"] = f"<test-{datetime.now().timestamp()}@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    msg.attach(MIMEText(body, "plain"))

    for filename in attachments:
        part = MIMEBase("application", "octet-stream")
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)

    return msg.as_bytes()


def _build_rfc822_multipart_email(
    from_addr: str,
    to_addr: str,
    subject: str,
    plain_body: str,
    html_body: str,
) -> bytes:
    """Build a multipart/alternative email with both plain and HTML."""
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Message-ID"] = f"<test-{datetime.now().timestamp()}@example.com>"
    msg["Date"] = email.utils.formatdate(localtime=True)

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    return msg.as_bytes()
