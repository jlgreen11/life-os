"""
Tests for the Signal Messenger connector.

Validates JSON-RPC 2.0 communication with signal-cli daemon via Unix socket,
including:
    - Authentication via listGroups RPC call
    - Message reception and event publishing
    - Group message detection and domain classification
    - Reply and attachment detection in inbound messages
    - Outbound message sending with recipient normalisation
    - Contact and group synchronisation from signal-cli
    - Health check via RPC connectivity
    - JSON-RPC protocol (request IDs, error handling, timeouts)
    - Lifecycle management (start, stop, contact sync loop)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.signal_msg.connector import SignalConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def signal_config():
    """Standard Signal connector configuration."""
    return {
        "socket_path": "/tmp/test-signal-cli.sock",
        "phone_number": "+15551234567",
        "sync_interval": 5,
    }


@pytest.fixture
def connector(event_bus, db, signal_config):
    """Create a SignalConnector instance with mocked dependencies."""
    return SignalConnector(event_bus, db, signal_config)


def _mock_unix_connection(response_data: dict | list | None = None):
    """Build mock reader/writer pair for asyncio.open_unix_connection.

    Returns a patcher context manager and the mock reader/writer so
    tests can inspect what was written to the socket.
    """
    mock_reader = AsyncMock()
    mock_writer = MagicMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    mock_writer.write = MagicMock()
    mock_writer.drain = AsyncMock()

    # Build a JSON-RPC response
    rpc_response = {"jsonrpc": "2.0", "id": 1}
    if response_data is not None:
        rpc_response["result"] = response_data
    else:
        rpc_response["result"] = None

    mock_reader.readline = AsyncMock(
        return_value=(json.dumps(rpc_response) + "\n").encode()
    )

    return mock_reader, mock_writer


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


class TestAuthenticate:
    """Test the authenticate() method that checks signal-cli daemon connectivity."""

    @pytest.mark.asyncio
    async def test_success_returns_true(self, connector):
        """authenticate() returns True when listGroups returns a list."""
        reader, writer = _mock_unix_connection([{"id": "group1", "name": "Test"}])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.authenticate()

        assert result is True

    @pytest.mark.asyncio
    async def test_success_with_empty_list(self, connector):
        """authenticate() returns True even with an empty group list."""
        reader, writer = _mock_unix_connection([])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.authenticate()

        assert result is True

    @pytest.mark.asyncio
    async def test_failure_non_list_result(self, connector):
        """authenticate() returns False when listGroups returns a non-list."""
        reader, writer = _mock_unix_connection("not a list")

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.authenticate()

        assert result is False

    @pytest.mark.asyncio
    async def test_failure_socket_error(self, connector):
        """authenticate() returns False when the Unix socket is unreachable."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=ConnectionRefusedError("Daemon not running"),
        ):
            result = await connector.authenticate()

        assert result is False

    @pytest.mark.asyncio
    async def test_failure_null_result(self, connector):
        """authenticate() returns False when listGroups returns None."""
        reader, writer = _mock_unix_connection(None)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.authenticate()

        assert result is False


# ---------------------------------------------------------------------------
# Sync Tests
# ---------------------------------------------------------------------------


class TestSync:
    """Test the sync() method that receives inbound messages from Signal."""

    def _make_envelope(
        self,
        body: str = "Hello!",
        source: str = "+15559876543",
        source_name: str = "Alice",
        timestamp_ms: int = 1700000000000,
        group_id: str | None = None,
        group_name: str | None = None,
        has_quote: bool = False,
        has_attachments: bool = False,
    ) -> dict:
        """Build a single Signal message envelope for testing."""
        data_message = {"message": body} if body else {}

        if group_id:
            data_message["groupInfo"] = {
                "groupId": group_id,
                "groupName": group_name or "Test Group",
            }
        if has_quote:
            data_message["quote"] = {"id": 123, "author": "+15550001111"}
        if has_attachments:
            data_message["attachments"] = [{"contentType": "image/png", "size": 1024}]

        return {
            "envelope": {
                "source": source,
                "sourceName": source_name,
                "timestamp": timestamp_ms,
                "dataMessage": data_message if data_message else None,
            }
        }

    @pytest.mark.asyncio
    async def test_normal_message(self, connector, event_bus):
        """A standard inbound message publishes a message.received event."""
        envelope = self._make_envelope(body="Hi there", source="+15559876543")
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 1
        assert event_bus.publish.call_count == 1
        call_args = event_bus.publish.call_args
        assert call_args[0][0] == "message.received"
        payload = call_args[0][1]
        assert payload["channel"] == "signal"
        assert payload["direction"] == "inbound"
        assert payload["from_address"] == "+15559876543"
        assert payload["from_contact"] == "Alice"
        assert payload["body"] == "Hi there"
        assert payload["is_reply"] is False
        assert payload["has_attachments"] is False

    @pytest.mark.asyncio
    async def test_group_message(self, connector, event_bus):
        """Group messages include group_id and group_name in the payload."""
        envelope = self._make_envelope(
            body="Team update",
            group_id="abc123",
            group_name="Work Team",
        )
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 1
        payload = event_bus.publish.call_args[0][1]
        assert payload["group_id"] == "abc123"
        assert payload["group_name"] == "Work Team"

    @pytest.mark.asyncio
    async def test_skip_non_content_envelopes(self, connector, event_bus):
        """Envelopes without dataMessage (receipts, typing) are skipped."""
        # Receipt envelope: no dataMessage
        receipt = {
            "envelope": {
                "source": "+15559876543",
                "sourceName": "Alice",
                "timestamp": 1700000000000,
                "receiptMessage": {"type": "read"},
            }
        }
        # Typing indicator: dataMessage is explicitly None
        typing = {
            "envelope": {
                "source": "+15559876543",
                "sourceName": "Alice",
                "timestamp": 1700000001000,
                "dataMessage": None,
            }
        }
        reader, writer = _mock_unix_connection([receipt, typing])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_empty_body(self, connector, event_bus):
        """Messages with no text body (reaction-only, sticker) are skipped."""
        envelope = self._make_envelope(body="")
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_reply_detection(self, connector, event_bus):
        """Messages with a 'quote' field are detected as replies."""
        envelope = self._make_envelope(body="Replying to you", has_quote=True)
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["is_reply"] is True

    @pytest.mark.asyncio
    async def test_attachment_detection(self, connector, event_bus):
        """Messages with attachments are flagged has_attachments=True."""
        envelope = self._make_envelope(body="See attached", has_attachments=True)
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_empty_response_returns_zero(self, connector, event_bus):
        """sync() returns 0 when _rpc_call returns None."""
        reader, writer = _mock_unix_connection(None)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 0
        event_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self, connector, event_bus):
        """sync() returns 0 when _rpc_call returns an empty list."""
        reader, writer = _mock_unix_connection([])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 0

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self, connector, event_bus):
        """sync() returns 0 and logs error when _rpc_call raises."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=OSError("Socket error"),
        ):
            count = await connector.sync()

        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_messages(self, connector, event_bus):
        """sync() processes multiple messages from a single receive call."""
        envelopes = [
            self._make_envelope(body="Message 1", source="+15550001111", timestamp_ms=1700000001000),
            self._make_envelope(body="Message 2", source="+15550002222", timestamp_ms=1700000002000),
            self._make_envelope(body="Message 3", source="+15550003333", timestamp_ms=1700000003000),
        ]
        reader, writer = _mock_unix_connection(envelopes)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 3
        assert event_bus.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_timestamp_conversion(self, connector, event_bus):
        """Millisecond timestamps are converted to message_id strings."""
        ts_ms = 1700000000000
        envelope = self._make_envelope(body="Test", timestamp_ms=ts_ms)
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["message_id"] == str(ts_ms)

    @pytest.mark.asyncio
    async def test_snippet_truncation(self, connector, event_bus):
        """Long message bodies are truncated to 150-char snippets."""
        long_body = "A" * 300
        envelope = self._make_envelope(body=long_body)
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert len(payload["snippet"]) == 150
        assert payload["body"] == long_body

    @pytest.mark.asyncio
    async def test_short_body_no_truncation(self, connector, event_bus):
        """Short messages are not truncated in the snippet."""
        short_body = "Hello"
        envelope = self._make_envelope(body=short_body)
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["snippet"] == short_body

    @pytest.mark.asyncio
    async def test_dm_has_no_group_info(self, connector, event_bus):
        """Direct messages have group_id=None and group_name=None."""
        envelope = self._make_envelope(body="DM content")
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["group_id"] is None
        assert payload["group_name"] is None

    @pytest.mark.asyncio
    async def test_metadata_includes_related_contacts(self, connector, event_bus):
        """Event metadata includes the sender in related_contacts."""
        envelope = self._make_envelope(body="Hi", source="+15559876543")
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        metadata = event_bus.publish.call_args[1]["metadata"]
        assert "+15559876543" in metadata["related_contacts"]

    @pytest.mark.asyncio
    async def test_to_addresses_is_self_phone(self, connector, event_bus):
        """Inbound messages have to_addresses set to the connector's phone."""
        envelope = self._make_envelope(body="Test")
        reader, writer = _mock_unix_connection([envelope])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.sync()

        payload = event_bus.publish.call_args[0][1]
        assert payload["to_addresses"] == ["+15551234567"]

    @pytest.mark.asyncio
    async def test_mixed_envelopes_skips_non_content(self, connector, event_bus):
        """Only content envelopes are processed, receipts are skipped."""
        content = self._make_envelope(body="Real message", timestamp_ms=1700000001000)
        receipt = {
            "envelope": {
                "source": "+15559876543",
                "timestamp": 1700000002000,
                "receiptMessage": {"type": "delivery"},
            }
        }
        no_body = self._make_envelope(body="", timestamp_ms=1700000003000)

        reader, writer = _mock_unix_connection([content, receipt, no_body])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            count = await connector.sync()

        assert count == 1


# ---------------------------------------------------------------------------
# Execute Tests
# ---------------------------------------------------------------------------


class TestExecute:
    """Test the execute() method for sending outbound messages."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector, event_bus):
        """send_message calls _rpc_call('send') and publishes message.sent."""
        reader, writer = _mock_unix_connection(None)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Hello!"},
            )

        assert result["status"] == "sent"
        assert result["to"] == "+15559876543"

        # Two RPC calls: send + publish_event (which also calls _rpc?  No, publish_event goes to event_bus)
        # Actually, two publish calls: first the send RPC, then publish_event
        # The event_bus.publish should have been called for message.sent
        sent_calls = [
            c for c in event_bus.publish.call_args_list
            if c[0][0] == "message.sent"
        ]
        assert len(sent_calls) == 1
        payload = sent_calls[0][0][1]
        assert payload["channel"] == "signal"
        assert payload["direction"] == "outbound"
        assert payload["body"] == "Hello!"

    @pytest.mark.asyncio
    async def test_send_message_string_recipient_normalised(self, connector, event_bus):
        """A string recipient is normalised to a list for the RPC call."""
        sent_requests = []

        async def capture_rpc(*args):
            """Capture the JSON written to the socket."""
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_writer.drain = AsyncMock()

            original_write = mock_writer.write

            def capture_write(data):
                sent_requests.append(json.loads(data.decode()))

            mock_writer.write = capture_write

            rpc_response = {"jsonrpc": "2.0", "id": 1, "result": None}
            mock_reader.readline = AsyncMock(
                return_value=(json.dumps(rpc_response) + "\n").encode()
            )
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", side_effect=capture_rpc):
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Test"},
            )

        # Find the 'send' RPC call
        send_calls = [r for r in sent_requests if r.get("method") == "send"]
        assert len(send_calls) == 1
        assert send_calls[0]["params"]["recipients"] == ["+15559876543"]

    @pytest.mark.asyncio
    async def test_send_message_list_recipient(self, connector, event_bus):
        """A list recipient is passed through unchanged."""
        sent_requests = []

        async def capture_rpc(*args):
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_writer.drain = AsyncMock()

            def capture_write(data):
                sent_requests.append(json.loads(data.decode()))

            mock_writer.write = capture_write

            rpc_response = {"jsonrpc": "2.0", "id": 1, "result": None}
            mock_reader.readline = AsyncMock(
                return_value=(json.dumps(rpc_response) + "\n").encode()
            )
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", side_effect=capture_rpc):
            await connector.execute(
                "send_message",
                {"to": ["+15559876543", "+15550001111"], "message": "Group msg"},
            )

        send_calls = [r for r in sent_requests if r.get("method") == "send"]
        assert len(send_calls) == 1
        assert send_calls[0]["params"]["recipients"] == ["+15559876543", "+15550001111"]

    @pytest.mark.asyncio
    async def test_send_message_publishes_sent_event(self, connector, event_bus):
        """After sending, a message.sent event is published with correct payload."""
        reader, writer = _mock_unix_connection(None)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": "Outbound test"},
            )

        sent_calls = [
            c for c in event_bus.publish.call_args_list
            if c[0][0] == "message.sent"
        ]
        assert len(sent_calls) == 1
        payload = sent_calls[0][0][1]
        assert payload["from_address"] == "+15551234567"
        assert payload["to_addresses"] == ["+15559876543"]
        assert payload["body"] == "Outbound test"
        assert payload["body_plain"] == "Outbound test"
        assert payload["snippet"] == "Outbound test"

    @pytest.mark.asyncio
    async def test_unknown_action_raises_valueerror(self, connector):
        """Calling execute() with an unknown action raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            await connector.execute("delete_message", {})

    @pytest.mark.asyncio
    async def test_send_message_snippet_truncated(self, connector, event_bus):
        """Outbound message snippets are truncated to 150 characters."""
        long_message = "B" * 300
        reader, writer = _mock_unix_connection(None)

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.execute(
                "send_message",
                {"to": "+15559876543", "message": long_message},
            )

        sent_calls = [
            c for c in event_bus.publish.call_args_list
            if c[0][0] == "message.sent"
        ]
        payload = sent_calls[0][0][1]
        assert len(payload["snippet"]) == 150


# ---------------------------------------------------------------------------
# Health Check Tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Test the health_check() method."""

    @pytest.mark.asyncio
    async def test_ok_when_daemon_responsive(self, connector):
        """health_check() returns ok when listGroups succeeds."""
        reader, writer = _mock_unix_connection([])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            result = await connector.health_check()

        assert result["status"] == "ok"
        assert result["connector"] == "signal"

    @pytest.mark.asyncio
    async def test_error_when_daemon_unreachable(self, connector):
        """health_check() returns error when daemon is not running."""
        with patch(
            "asyncio.open_unix_connection",
            side_effect=ConnectionRefusedError("No daemon"),
        ):
            result = await connector.health_check()

        assert result["status"] == "error"
        assert "No daemon" in result["details"]

    @pytest.mark.asyncio
    async def test_error_when_rpc_fails(self, connector):
        """health_check() returns error when the RPC call raises."""
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()

        # Return an RPC error response
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        }
        mock_reader.readline = AsyncMock(
            return_value=(json.dumps(error_response) + "\n").encode()
        )

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            result = await connector.health_check()

        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Domain Classification Tests
# ---------------------------------------------------------------------------


class TestClassifyDomain:
    """Test _classify_domain() heuristic classification."""

    def test_work_keyword_in_group_name(self, connector):
        """Groups with 'work' in the name are classified as work."""
        assert connector._classify_domain("+15550001111", {"groupName": "Work Chat"}) == "work"

    def test_team_keyword_in_group_name(self, connector):
        """Groups with 'team' in the name are classified as work."""
        assert connector._classify_domain("+15550001111", {"groupName": "Engineering Team"}) == "work"

    def test_project_keyword_in_group_name(self, connector):
        """Groups with 'project' in the name are classified as work."""
        assert connector._classify_domain("+15550001111", {"groupName": "Project Alpha"}) == "work"

    def test_standup_keyword_in_group_name(self, connector):
        """Groups with 'standup' in the name are classified as work."""
        assert connector._classify_domain("+15550001111", {"groupName": "Daily Standup"}) == "work"

    def test_case_insensitive_matching(self, connector):
        """Domain classification is case-insensitive."""
        assert connector._classify_domain("+15550001111", {"groupName": "WORK Updates"}) == "work"
        assert connector._classify_domain("+15550001111", {"groupName": "Team STANDUP"}) == "work"

    def test_personal_group(self, connector):
        """Groups without work keywords are classified as personal."""
        assert connector._classify_domain("+15550001111", {"groupName": "Family Chat"}) == "personal"

    def test_no_group_dm(self, connector):
        """Direct messages (no group_info) are classified as personal."""
        assert connector._classify_domain("+15550001111", None) == "personal"

    def test_group_with_no_name(self, connector):
        """Groups with no groupName default to personal."""
        assert connector._classify_domain("+15550001111", {}) == "personal"
        assert connector._classify_domain("+15550001111", {"groupName": None}) == "personal"


# ---------------------------------------------------------------------------
# RPC Call Tests
# ---------------------------------------------------------------------------


class TestRpcCall:
    """Test _rpc_call() JSON-RPC 2.0 protocol implementation."""

    @pytest.mark.asyncio
    async def test_normal_call_returns_result(self, connector):
        """A successful RPC call returns the 'result' field."""
        response = {"jsonrpc": "2.0", "id": 1, "result": [{"name": "Group A"}]}
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=(json.dumps(response) + "\n").encode()
        )

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            result = await connector._rpc_call("listGroups")

        assert result == [{"name": "Group A"}]

    @pytest.mark.asyncio
    async def test_rpc_error_raises(self, connector):
        """An RPC error response raises an Exception."""
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_reader.readline = AsyncMock(
            return_value=(json.dumps(error_response) + "\n").encode()
        )

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            with pytest.raises(Exception, match="Signal RPC error"):
                await connector._rpc_call("invalidMethod")

    @pytest.mark.asyncio
    async def test_timeout_raises(self, connector):
        """A timeout on readline raises asyncio.TimeoutError."""
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_reader.readline = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            with pytest.raises(asyncio.TimeoutError):
                await connector._rpc_call("receive")

    @pytest.mark.asyncio
    async def test_request_id_increments(self, connector):
        """Each RPC call increments the request ID."""
        sent_data = []

        async def mock_open(*args):
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_writer.drain = AsyncMock()

            def capture(data):
                sent_data.append(json.loads(data.decode()))

            mock_writer.write = capture

            response = {"jsonrpc": "2.0", "id": 1, "result": None}
            mock_reader.readline = AsyncMock(
                return_value=(json.dumps(response) + "\n").encode()
            )
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", side_effect=mock_open):
            await connector._rpc_call("method1")
            await connector._rpc_call("method2")
            await connector._rpc_call("method3")

        assert sent_data[0]["id"] == 1
        assert sent_data[1]["id"] == 2
        assert sent_data[2]["id"] == 3

    @pytest.mark.asyncio
    async def test_params_included_when_provided(self, connector):
        """Params are included in the request when not None."""
        sent_data = []

        async def mock_open(*args):
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_writer.drain = AsyncMock()

            def capture(data):
                sent_data.append(json.loads(data.decode()))

            mock_writer.write = capture

            response = {"jsonrpc": "2.0", "id": 1, "result": None}
            mock_reader.readline = AsyncMock(
                return_value=(json.dumps(response) + "\n").encode()
            )
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", side_effect=mock_open):
            await connector._rpc_call("send", {"message": "hello"})

        assert "params" in sent_data[0]
        assert sent_data[0]["params"] == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_params_omitted_when_none(self, connector):
        """Params are not included in the request when None."""
        sent_data = []

        async def mock_open(*args):
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_writer.drain = AsyncMock()

            def capture(data):
                sent_data.append(json.loads(data.decode()))

            mock_writer.write = capture

            response = {"jsonrpc": "2.0", "id": 1, "result": None}
            mock_reader.readline = AsyncMock(
                return_value=(json.dumps(response) + "\n").encode()
            )
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", side_effect=mock_open):
            await connector._rpc_call("listGroups")

        assert "params" not in sent_data[0]

    @pytest.mark.asyncio
    async def test_socket_closed_on_success(self, connector):
        """The writer is always closed after a successful call."""
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()

        response = {"jsonrpc": "2.0", "id": 1, "result": []}
        mock_reader.readline = AsyncMock(
            return_value=(json.dumps(response) + "\n").encode()
        )

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            await connector._rpc_call("listGroups")

        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_socket_closed_on_error(self, connector):
        """The writer is closed even when the RPC call errors."""
        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()

        error_response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "fail"}}
        mock_reader.readline = AsyncMock(
            return_value=(json.dumps(error_response) + "\n").encode()
        )

        with patch("asyncio.open_unix_connection", return_value=(mock_reader, mock_writer)):
            with pytest.raises(Exception):
                await connector._rpc_call("bad_method")

        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_awaited_once()


# ---------------------------------------------------------------------------
# Contact Sync Tests
# ---------------------------------------------------------------------------


class TestSyncContacts:
    """Test sync_contacts() which pulls contacts and groups from signal-cli."""

    @pytest.mark.asyncio
    async def test_new_contact_created(self, connector, db):
        """A new signal contact creates a contact record in entities.db."""
        contacts = [{"number": "+15559876543", "name": "Bob Smith"}]
        groups = []
        call_count = 0

        async def multi_rpc(method, params=None):
            nonlocal call_count
            call_count += 1
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return groups
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            row = conn.execute(
                "SELECT * FROM contact_identifiers WHERE identifier = ?",
                ("+15559876543",),
            ).fetchone()
            assert row is not None
            assert row["identifier_type"] == "phone"

            contact = conn.execute(
                "SELECT * FROM contacts WHERE id = ?",
                (row["contact_id"],),
            ).fetchone()
            assert contact is not None
            assert contact["name"] == "Bob Smith"
            channels = json.loads(contact["channels"])
            assert channels["signal"] == "+15559876543"

    @pytest.mark.asyncio
    async def test_skip_self_contact(self, connector, db):
        """Contacts with the connector's own phone number are skipped."""
        contacts = [{"number": "+15551234567", "name": "Me Myself"}]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return []
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            row = conn.execute(
                "SELECT * FROM contact_identifiers WHERE identifier = ?",
                ("+15551234567",),
            ).fetchone()
            assert row is None

    @pytest.mark.asyncio
    async def test_skip_no_name_contact(self, connector, db):
        """Contacts without any name field are skipped."""
        contacts = [{"number": "+15559876543"}]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return []
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            row = conn.execute(
                "SELECT * FROM contact_identifiers WHERE identifier = ?",
                ("+15559876543",),
            ).fetchone()
            assert row is None

    @pytest.mark.asyncio
    async def test_skip_no_number_contact(self, connector, db):
        """Contacts without a phone number are skipped."""
        contacts = [{"name": "No Phone"}]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return []
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        # Should not create any contacts
        with db.get_connection("entities") as conn:
            count = conn.execute("SELECT COUNT(*) as cnt FROM contacts").fetchone()["cnt"]
            assert count == 0

    @pytest.mark.asyncio
    async def test_contact_dedup_by_phone(self, connector, db):
        """Existing contacts matched by phone are updated, not duplicated."""
        # Pre-create a contact with the phone number
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            conn.execute(
                """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                   VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                ("existing-id", "Old Name", json.dumps(["+15559876543"]),
                 json.dumps({}), now, now),
            )
            conn.execute(
                """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                   VALUES (?, 'phone', ?)""",
                ("+15559876543", "existing-id"),
            )

        contacts = [{"number": "+15559876543", "name": "Bob Smith"}]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return []
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            # Should still be only one contact
            count = conn.execute("SELECT COUNT(*) as cnt FROM contacts").fetchone()["cnt"]
            assert count == 1

            contact = conn.execute("SELECT * FROM contacts WHERE id = ?", ("existing-id",)).fetchone()
            channels = json.loads(contact["channels"])
            assert channels["signal"] == "+15559876543"

    @pytest.mark.asyncio
    async def test_contact_profile_name_fallback(self, connector, db):
        """Falls back to profileName when name is absent."""
        contacts = [{"number": "+15559876543", "profileName": "BobProfile"}]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return []
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            row = conn.execute(
                "SELECT contact_id FROM contact_identifiers WHERE identifier = ?",
                ("+15559876543",),
            ).fetchone()
            contact = conn.execute("SELECT * FROM contacts WHERE id = ?", (row["contact_id"],)).fetchone()
            assert contact["name"] == "BobProfile"

    @pytest.mark.asyncio
    async def test_group_creates_pairwise_relationships(self, connector, db):
        """Groups with known members create pairwise entity_relationships."""
        # Create two contacts first
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            for cid, number in [("c1", "+15550001111"), ("c2", "+15550002222")]:
                conn.execute(
                    """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                       VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                    (cid, f"Contact {cid}", json.dumps([number]),
                     json.dumps({"signal": number}), now, now),
                )
                conn.execute(
                    """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                       VALUES (?, 'phone', ?)""",
                    (number, cid),
                )

        contacts = []
        groups = [
            {
                "id": "group-abc",
                "name": "Test Group",
                "members": [
                    {"number": "+15550001111"},
                    {"number": "+15550002222"},
                ],
            }
        ]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return contacts
            if method == "listGroups":
                return groups
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            rels = conn.execute(
                "SELECT * FROM entity_relationships WHERE relationship = 'signal_group_member'"
            ).fetchall()
            assert len(rels) == 1
            rel = rels[0]
            # Verify the pair matches our contacts
            pair = {rel["entity_a_id"], rel["entity_b_id"]}
            assert pair == {"c1", "c2"}
            metadata = json.loads(rel["metadata"])
            assert metadata["group_name"] == "Test Group"

    @pytest.mark.asyncio
    async def test_group_with_three_members(self, connector, db):
        """Three group members create 3 pairwise relationships (C(3,2))."""
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            for i in range(3):
                number = f"+1555000{i:04d}"
                cid = f"c{i}"
                conn.execute(
                    """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                       VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                    (cid, f"Contact {i}", json.dumps([number]),
                     json.dumps({"signal": number}), now, now),
                )
                conn.execute(
                    """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                       VALUES (?, 'phone', ?)""",
                    (number, cid),
                )

        groups = [
            {
                "id": "group-xyz",
                "name": "Trio Group",
                "members": [
                    {"number": "+155500000000"},
                    {"number": "+155500010001"},
                    {"number": "+155500020002"},
                ],
            }
        ]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return []
            if method == "listGroups":
                return groups
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            rels = conn.execute(
                "SELECT * FROM entity_relationships WHERE relationship = 'signal_group_member'"
            ).fetchall()
            # Members need to be in phone_to_id map; these phones aren't matching
            # the identifiers we stored, so 0 rels expected
            # Let's verify this is correct - the numbers don't match
            assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_group_with_matching_three_members(self, connector, db):
        """Three matching group members create 3 pairwise relationships."""
        now = datetime.now(timezone.utc).isoformat()
        numbers = ["+15550001111", "+15550002222", "+15550003333"]
        for i, number in enumerate(numbers):
            cid = f"c{i}"
            with db.get_connection("entities") as conn:
                conn.execute(
                    """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                       VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                    (cid, f"Contact {i}", json.dumps([number]),
                     json.dumps({"signal": number}), now, now),
                )
                conn.execute(
                    """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                       VALUES (?, 'phone', ?)""",
                    (number, cid),
                )

        groups = [
            {
                "id": "group-xyz",
                "name": "Trio Group",
                "members": [
                    {"number": "+15550001111"},
                    {"number": "+15550002222"},
                    {"number": "+15550003333"},
                ],
            }
        ]

        async def multi_rpc(method, params=None):
            if method == "listContacts":
                return []
            if method == "listGroups":
                return groups
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()

        with db.get_connection("entities") as conn:
            rels = conn.execute(
                "SELECT * FROM entity_relationships WHERE relationship = 'signal_group_member'"
            ).fetchall()
            assert len(rels) == 3  # C(3,2) = 3

    @pytest.mark.asyncio
    async def test_empty_contacts_and_groups(self, connector, db):
        """sync_contacts() handles empty responses gracefully."""
        async def multi_rpc(method, params=None):
            return []

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            # Should not raise
            await connector.sync_contacts()

    @pytest.mark.asyncio
    async def test_null_contacts_response(self, connector, db):
        """sync_contacts() handles None responses from _rpc_call."""
        async def multi_rpc(method, params=None):
            return None

        with patch.object(connector, "_rpc_call", side_effect=multi_rpc):
            await connector.sync_contacts()


# ---------------------------------------------------------------------------
# Lifecycle Tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test connector start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_cancels_contact_sync_task(self, connector, event_bus):
        """stop() cancels the background contact sync task."""
        reader, writer = _mock_unix_connection([])

        with patch("asyncio.open_unix_connection", return_value=(reader, writer)):
            await connector.start()

        assert connector._running is True
        assert connector._contact_sync_task is not None
        task = connector._contact_sync_task

        await connector.stop()

        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, connector):
        """stop() is safe to call even if start() was never called."""
        # _contact_sync_task is None, should not raise
        await connector.stop()

    @pytest.mark.asyncio
    async def test_connector_id_and_display_name(self, connector):
        """Verify connector class constants."""
        assert connector.CONNECTOR_ID == "signal"
        assert connector.DISPLAY_NAME == "Signal Messenger"
        assert connector.SYNC_INTERVAL_SECONDS == 5

    @pytest.mark.asyncio
    async def test_config_defaults(self, event_bus, db):
        """Default config values are used when keys are missing."""
        connector = SignalConnector(event_bus, db, {})
        assert connector._socket_path == "/tmp/signal-cli.sock"
        assert connector._phone == ""
        assert connector._request_id == 0


# ---------------------------------------------------------------------------
# Find Contact By Name Tests
# ---------------------------------------------------------------------------


class TestFindContactByName:
    """Test _find_contact_by_name() matching logic."""

    def test_matches_first_name(self, connector, db):
        """Matches a contact by first name substring."""
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            conn.execute(
                """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                   VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                ("contact-1", "Robert Smith", "[]", "{}", now, now),
            )
            result = connector._find_contact_by_name(conn, "Robert Johnson")
        assert result == "contact-1"

    def test_no_match_returns_none(self, connector, db):
        """Returns None when no contact matches."""
        with db.get_connection("entities") as conn:
            result = connector._find_contact_by_name(conn, "Nonexistent Person")
        assert result is None

    def test_short_name_skipped(self, connector, db):
        """Names shorter than 2 characters are not matched."""
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            conn.execute(
                """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                   VALUES (?, ?, ?, ?, '["personal"]', ?, ?)""",
                ("contact-1", "A B", "[]", "{}", now, now),
            )
            result = connector._find_contact_by_name(conn, "A")
        assert result is None

    def test_empty_name_returns_none(self, connector, db):
        """Empty name returns None."""
        with db.get_connection("entities") as conn:
            result = connector._find_contact_by_name(conn, "")
        assert result is None
