"""
Life OS — SignalConnector Test Suite

Comprehensive test coverage for Signal Messenger integration via signal-cli.

Tests cover:
- Authentication and health checks
- Contact and group synchronization
- Message receiving (individual and group)
- Message sending
- JSON-RPC communication
- Error handling and edge cases
- Domain classification heuristics
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

from connectors.signal_msg.connector import SignalConnector, CONTACT_SYNC_INTERVAL


@pytest.fixture
def signal_config():
    """Standard Signal connector configuration."""
    return {
        "socket_path": "/tmp/signal-cli.sock",
        "phone_number": "+12025551234",
        "sync_interval": 5,
    }


@pytest.fixture
def signal_connector(event_bus, db, signal_config):
    """SignalConnector instance with temp database and mock event bus."""
    return SignalConnector(event_bus, db, signal_config)


# ============================================================================
# Authentication & Health Checks
# ============================================================================


@pytest.mark.asyncio
async def test_authenticate_success(signal_connector):
    """Test successful authentication when signal-cli daemon is responsive."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = []

        result = await signal_connector.authenticate()

        assert result is True
        mock_rpc.assert_called_once_with("listGroups")


@pytest.mark.asyncio
async def test_authenticate_failure_daemon_down(signal_connector):
    """Test authentication fails gracefully when signal-cli daemon is unreachable."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = Exception("Connection refused")

        result = await signal_connector.authenticate()

        assert result is False


@pytest.mark.asyncio
async def test_authenticate_failure_invalid_response(signal_connector):
    """Test authentication fails when daemon returns unexpected response."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = "invalid"

        result = await signal_connector.authenticate()

        assert result is False


@pytest.mark.asyncio
async def test_health_check_ok(signal_connector):
    """Test health check returns ok status when daemon is responsive."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = []

        result = await signal_connector.health_check()

        assert result["status"] == "ok"
        assert result["connector"] == "signal"


@pytest.mark.asyncio
async def test_health_check_error(signal_connector):
    """Test health check returns error status when daemon fails."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = Exception("Socket timeout")

        result = await signal_connector.health_check()

        assert result["status"] == "error"
        assert "Socket timeout" in result["details"]


# ============================================================================
# Contact Synchronization
# ============================================================================


@pytest.mark.asyncio
async def test_sync_contacts_new_contact(signal_connector):
    """Test syncing a new contact creates entry in entities database."""
    contacts = [
        {
            "number": "+12025559999",
            "name": "Alice Smith",
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]  # contacts, then groups

        await signal_connector.sync_contacts()

        # Verify contact was created
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT name, phones FROM contacts WHERE name = 'Alice Smith'"
            ).fetchall()
            assert len(rows) == 1
            assert json.loads(rows[0]["phones"]) == ["+12025559999"]


@pytest.mark.asyncio
async def test_sync_contacts_update_existing(signal_connector):
    """Test syncing an existing contact updates phone/channel info."""
    # Pre-populate a contact
    contact_id = "test-contact-id"
    with signal_connector.db.get_connection("entities") as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
               VALUES (?, ?, '[]', '{}', '["personal"]', ?, ?)""",
            (contact_id, "Unknown Contact", now, now)
        )
        conn.execute(
            """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
               VALUES (?, 'phone', ?)""",
            ("+12025559999", contact_id)
        )

    contacts = [
        {
            "number": "+12025559999",
            "name": "Alice Smith",
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]

        await signal_connector.sync_contacts()

        # Verify contact was updated
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT name, phones FROM contacts WHERE id = ?", (contact_id,)
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["name"] == "Alice Smith"
            assert json.loads(rows[0]["phones"]) == ["+12025559999"]


@pytest.mark.asyncio
async def test_sync_contacts_skip_self(signal_connector):
    """Test syncing skips the user's own phone number."""
    contacts = [
        {
            "number": "+12025551234",  # Matches signal_connector._phone
            "name": "Me",
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]

        await signal_connector.sync_contacts()

        # Verify no contact was created
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT * FROM contacts WHERE name = 'Me'"
            ).fetchall()
            assert len(rows) == 0


@pytest.mark.asyncio
async def test_sync_contacts_skip_nameless(signal_connector):
    """Test syncing skips contacts without any name."""
    contacts = [
        {
            "number": "+12025559999",
            # No name, profileName, or contactName
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]

        await signal_connector.sync_contacts()

        # Verify no contact was created
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute("SELECT * FROM contacts").fetchall()
            assert len(rows) == 0


@pytest.mark.asyncio
async def test_sync_contacts_use_profile_name(signal_connector):
    """Test syncing uses profileName when name is not available."""
    contacts = [
        {
            "number": "+12025559999",
            "profileName": "Alice Profile",
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]

        await signal_connector.sync_contacts()

        # Verify contact was created with profile name
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT name FROM contacts WHERE name = 'Alice Profile'"
            ).fetchall()
            assert len(rows) == 1


@pytest.mark.asyncio
async def test_sync_groups(signal_connector):
    """Test syncing groups creates entity relationships."""
    # Pre-populate contacts
    contact_ids = []
    with signal_connector.db.get_connection("entities") as conn:
        now = datetime.now(timezone.utc).isoformat()
        for i, phone in enumerate(["+12025550001", "+12025550002", "+12025550003"]):
            contact_id = f"contact-{i}"
            contact_ids.append(contact_id)
            conn.execute(
                """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
                   VALUES (?, ?, ?, '{}', '["personal"]', ?, ?)""",
                (contact_id, f"Contact {i}", json.dumps([phone]), now, now)
            )
            conn.execute(
                """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
                   VALUES (?, 'phone', ?)""",
                (phone, contact_id)
            )

    contacts = []
    groups = [
        {
            "id": "group-abc123",
            "name": "Family Chat",
            "members": [
                {"number": "+12025550001"},
                {"number": "+12025550002"},
                {"number": "+12025550003"},
            ]
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, groups]

        await signal_connector.sync_contacts()

        # Verify pairwise relationships were created
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute(
                "SELECT * FROM entity_relationships WHERE relationship = 'signal_group_member'"
            ).fetchall()
            # 3 members = 3 pairwise relationships (0-1, 0-2, 1-2)
            assert len(rows) == 3

            # Verify metadata contains group info
            metadata = json.loads(rows[0]["metadata"])
            assert metadata["group_name"] == "Family Chat"
            assert metadata["group_id"] == "group-abc123"


@pytest.mark.asyncio
async def test_sync_contacts_name_matching(signal_connector):
    """Test syncing matches contacts by first name when phone is new."""
    # Pre-populate contact with similar name
    contact_id = "existing-contact"
    with signal_connector.db.get_connection("entities") as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO contacts (id, name, phones, channels, domains, created_at, updated_at)
               VALUES (?, ?, '[]', '{}', '["personal"]', ?, ?)""",
            (contact_id, "Alice Johnson", now, now)
        )

    contacts = [
        {
            "number": "+12025559999",
            "name": "Alice",  # Matches first name
            "isBlocked": False,
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = [contacts, []]

        await signal_connector.sync_contacts()

        # Verify existing contact was enriched (not new contact created)
        with signal_connector.db.get_connection("entities") as conn:
            rows = conn.execute("SELECT * FROM contacts").fetchall()
            assert len(rows) == 1  # Only one contact exists
            assert rows[0]["id"] == contact_id
            assert json.loads(rows[0]["phones"]) == ["+12025559999"]


# ============================================================================
# Message Receiving
# ============================================================================


@pytest.mark.asyncio
async def test_sync_receive_single_message(signal_connector, event_bus):
    """Test receiving a single direct message publishes message.received event."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": "Hello, how are you?",
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 1
        mock_rpc.assert_called_once_with("receive", {"account": "+12025551234"})

        # Verify event was published
        assert event_bus.publish.called
        call_args = event_bus.publish.call_args
        assert "message.received" in call_args[0][0]  # subject
        payload = call_args[0][1]
        assert payload["from_address"] == "+12025559999"
        assert payload["body"] == "Hello, how are you?"
        assert payload["channel"] == "signal"
        assert payload["direction"] == "inbound"


@pytest.mark.asyncio
async def test_sync_receive_group_message(signal_connector, event_bus):
    """Test receiving a group message includes group metadata."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": "Team meeting at 2pm",
                    "groupInfo": {
                        "groupId": "group-xyz789",
                        "groupName": "Work Team",
                    }
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 1

        # Verify group info in payload
        call_args = event_bus.publish.call_args
        payload = call_args[0][1]
        assert payload["group_id"] == "group-xyz789"
        assert payload["group_name"] == "Work Team"


@pytest.mark.asyncio
async def test_sync_skip_empty_messages(signal_connector):
    """Test receiving messages without body text are skipped."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": "",  # Empty body
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 0


@pytest.mark.asyncio
async def test_sync_skip_non_data_messages(signal_connector):
    """Test receiving non-data envelopes (receipts, typing) are skipped."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "timestamp": 1700000000000,
                # No dataMessage field (e.g., read receipt)
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 0


@pytest.mark.asyncio
async def test_sync_reply_detection(signal_connector, event_bus):
    """Test messages with quote field are marked as replies."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": "Yes, sounds good!",
                    "quote": {
                        "id": 1699999999000,
                        "text": "Want to grab lunch?",
                    }
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 1

        # Verify is_reply flag
        call_args = event_bus.publish.call_args
        payload = call_args[0][1]
        assert payload["is_reply"] is True


@pytest.mark.asyncio
async def test_sync_attachment_detection(signal_connector, event_bus):
    """Test messages with attachments are flagged."""
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": "Check out this photo",
                    "attachments": [
                        {"contentType": "image/jpeg", "filename": "photo.jpg"}
                    ]
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 1

        # Verify has_attachments flag
        call_args = event_bus.publish.call_args
        payload = call_args[0][1]
        assert payload["has_attachments"] is True


@pytest.mark.asyncio
async def test_sync_snippet_truncation(signal_connector, event_bus):
    """Test long messages are truncated in snippet field."""
    long_message = "A" * 200
    messages = [
        {
            "envelope": {
                "source": "+12025559999",
                "sourceName": "Alice Smith",
                "timestamp": 1700000000000,
                "dataMessage": {
                    "message": long_message,
                }
            }
        }
    ]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = messages

        count = await signal_connector.sync()

        assert count == 1

        # Verify snippet is truncated
        call_args = event_bus.publish.call_args
        payload = call_args[0][1]
        assert len(payload["snippet"]) == 150
        assert payload["body"] == long_message  # Full body preserved


@pytest.mark.asyncio
async def test_sync_error_handling(signal_connector):
    """Test sync gracefully handles RPC errors."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.side_effect = Exception("Network timeout")

        count = await signal_connector.sync()

        assert count == 0  # Returns 0 on error, doesn't crash


# ============================================================================
# Message Sending
# ============================================================================


@pytest.mark.asyncio
async def test_execute_send_single_recipient(signal_connector, event_bus):
    """Test sending a message to a single recipient."""
    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = None

        result = await signal_connector.execute(
            "send_message",
            {"to": "+12025559999", "message": "Hello Alice!"}
        )

        assert result["status"] == "sent"
        assert result["to"] == "+12025559999"

        # Verify RPC call
        mock_rpc.assert_called_once()
        call_args = mock_rpc.call_args
        assert call_args[0][0] == "send"
        params = call_args[0][1]
        assert params["account"] == "+12025551234"
        assert params["recipients"] == ["+12025559999"]
        assert params["message"] == "Hello Alice!"

        # Verify message.sent event was published
        assert event_bus.publish.called
        publish_args = event_bus.publish.call_args
        assert "message.sent" in publish_args[0][0]
        payload = publish_args[0][1]
        assert payload["body"] == "Hello Alice!"
        assert payload["direction"] == "outbound"


@pytest.mark.asyncio
async def test_execute_send_multiple_recipients(signal_connector, event_bus):
    """Test sending a message to multiple recipients."""
    recipients = ["+12025559999", "+12025558888"]

    with patch.object(signal_connector, "_rpc_call", new_callable=AsyncMock) as mock_rpc:
        mock_rpc.return_value = None

        result = await signal_connector.execute(
            "send_message",
            {"to": recipients, "message": "Group announcement"}
        )

        assert result["status"] == "sent"
        assert result["to"] == recipients

        # Verify RPC call with list of recipients
        call_args = mock_rpc.call_args
        params = call_args[0][1]
        assert params["recipients"] == recipients


@pytest.mark.asyncio
async def test_execute_unknown_action(signal_connector):
    """Test executing an unknown action raises ValueError."""
    with pytest.raises(ValueError, match="Unknown action"):
        await signal_connector.execute("unknown_action", {})


# ============================================================================
# JSON-RPC Communication
# ============================================================================


@pytest.mark.asyncio
async def test_rpc_call_success(signal_connector):
    """Test successful JSON-RPC call over Unix socket."""
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    response = {"jsonrpc": "2.0", "id": 1, "result": ["data"]}
    mock_reader.readline.return_value = (json.dumps(response) + "\n").encode()

    with patch("asyncio.open_unix_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (mock_reader, mock_writer)

        result = await signal_connector._rpc_call("testMethod", {"param": "value"})

        assert result == ["data"]

        # Verify socket path
        mock_open.assert_called_once_with("/tmp/signal-cli.sock")

        # Verify request format
        write_call = mock_writer.write.call_args[0][0]
        request = json.loads(write_call.decode().strip())
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "testMethod"
        assert request["params"] == {"param": "value"}


@pytest.mark.asyncio
async def test_rpc_call_error_response(signal_connector):
    """Test JSON-RPC error responses are raised as exceptions."""
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32600, "message": "Invalid Request"}
    }
    mock_reader.readline.return_value = (json.dumps(response) + "\n").encode()

    with patch("asyncio.open_unix_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (mock_reader, mock_writer)

        with pytest.raises(Exception, match="Signal RPC error"):
            await signal_connector._rpc_call("testMethod")


@pytest.mark.asyncio
async def test_rpc_call_timeout(signal_connector):
    """Test JSON-RPC calls timeout after 10 seconds."""
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    # Simulate timeout
    mock_reader.readline.side_effect = asyncio.TimeoutError()

    with patch("asyncio.open_unix_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (mock_reader, mock_writer)

        with pytest.raises(asyncio.TimeoutError):
            await signal_connector._rpc_call("testMethod")


@pytest.mark.asyncio
async def test_rpc_call_socket_cleanup(signal_connector):
    """Test JSON-RPC calls always close the socket, even on error."""
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    mock_reader.readline.side_effect = Exception("Read error")

    with patch("asyncio.open_unix_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (mock_reader, mock_writer)

        with pytest.raises(Exception, match="Read error"):
            await signal_connector._rpc_call("testMethod")

        # Verify socket was closed
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_rpc_call_increments_request_id(signal_connector):
    """Test JSON-RPC request IDs increment with each call."""
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()

    response = {"jsonrpc": "2.0", "id": 1, "result": None}
    mock_reader.readline.return_value = (json.dumps(response) + "\n").encode()

    with patch("asyncio.open_unix_connection", new_callable=AsyncMock) as mock_open:
        mock_open.return_value = (mock_reader, mock_writer)

        # Make multiple calls
        await signal_connector._rpc_call("method1")
        await signal_connector._rpc_call("method2")
        await signal_connector._rpc_call("method3")

        # Verify request IDs incremented
        assert signal_connector._request_id == 3


# ============================================================================
# Domain Classification
# ============================================================================


def test_classify_domain_personal_direct(signal_connector):
    """Test direct messages default to personal domain."""
    domain = signal_connector._classify_domain("+12025559999", None)
    assert domain == "personal"


def test_classify_domain_personal_group(signal_connector):
    """Test personal group chats are classified as personal."""
    group_info = {"groupName": "Family Chat", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "personal"


def test_classify_domain_work_by_keyword_work(signal_connector):
    """Test groups with 'work' keyword are classified as work."""
    group_info = {"groupName": "Work Updates", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "work"


def test_classify_domain_work_by_keyword_team(signal_connector):
    """Test groups with 'team' keyword are classified as work."""
    group_info = {"groupName": "Engineering Team", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "work"


def test_classify_domain_work_by_keyword_project(signal_connector):
    """Test groups with 'project' keyword are classified as work."""
    group_info = {"groupName": "Project Alpha", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "work"


def test_classify_domain_work_by_keyword_standup(signal_connector):
    """Test groups with 'standup' keyword are classified as work."""
    group_info = {"groupName": "Daily Standup", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "work"


def test_classify_domain_case_insensitive(signal_connector):
    """Test domain classification is case-insensitive."""
    group_info = {"groupName": "WORK CHAT", "groupId": "abc123"}
    domain = signal_connector._classify_domain("+12025559999", group_info)
    assert domain == "work"


# ============================================================================
# Lifecycle & Contact Sync Loop
# ============================================================================


@pytest.mark.asyncio
async def test_start_triggers_initial_contact_sync(signal_connector):
    """Test starting the connector triggers initial contact sync."""
    with patch.object(signal_connector, "sync_contacts", new_callable=AsyncMock) as mock_sync:
        with patch("asyncio.create_task") as mock_create_task:
            signal_connector._running = True

            await signal_connector.start()

            # Verify initial sync was called
            mock_sync.assert_called_once()
            # Verify periodic sync task was created
            mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_stop_cancels_contact_sync_task(signal_connector):
    """Test stopping the connector cancels the contact sync loop."""
    # Create a real asyncio task that we can cancel
    async def dummy_task():
        """Dummy coroutine that runs forever until cancelled."""
        while True:
            await asyncio.sleep(1)

    task = asyncio.create_task(dummy_task())
    signal_connector._contact_sync_task = task
    signal_connector._running = True

    await signal_connector.stop()

    # Verify task was cancelled
    assert task.cancelled()


@pytest.mark.asyncio
async def test_contact_sync_loop_interval(signal_connector):
    """Test contact sync loop uses correct interval."""
    # This test verifies the constant is set correctly
    assert CONTACT_SYNC_INTERVAL == 3600  # 1 hour


@pytest.mark.asyncio
async def test_contact_sync_loop_error_handling(signal_connector):
    """Test contact sync loop handles errors gracefully without crashing."""
    with patch.object(signal_connector, "sync_contacts", new_callable=AsyncMock) as mock_sync:
        # First call raises error, second call succeeds and stops loop
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            else:
                signal_connector._running = False

        mock_sync.side_effect = side_effect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            signal_connector._running = True

            # Should not raise, should continue after error
            await signal_connector._contact_sync_loop()

            # Verify we made it past the error
            assert call_count == 2


# ============================================================================
# Configuration & Initialization
# ============================================================================


def test_connector_id(signal_connector):
    """Test connector has correct ID."""
    assert signal_connector.CONNECTOR_ID == "signal"


def test_display_name(signal_connector):
    """Test connector has correct display name."""
    assert signal_connector.DISPLAY_NAME == "Signal Messenger"


def test_sync_interval(signal_connector):
    """Test connector has correct sync interval."""
    assert signal_connector.SYNC_INTERVAL_SECONDS == 5


def test_config_socket_path(signal_connector):
    """Test connector reads socket path from config."""
    assert signal_connector._socket_path == "/tmp/signal-cli.sock"


def test_config_phone_number(signal_connector):
    """Test connector reads phone number from config."""
    assert signal_connector._phone == "+12025551234"


def test_config_defaults(event_bus, db):
    """Test connector uses defaults when config values missing."""
    connector = SignalConnector(event_bus, db, {})
    assert connector._socket_path == "/tmp/signal-cli.sock"
    assert connector._phone == ""
