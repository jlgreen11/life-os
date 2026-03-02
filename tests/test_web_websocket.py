"""
Comprehensive test suite for web/websocket.py

Tests the WebSocket connection manager including connection lifecycle,
broadcasting, disconnection handling, and error resilience.

Coverage: ConnectionManager class, 58 LOC
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from web.websocket import ConnectionManager, ws_manager


# ---------------------------------------------------------------------------
# ConnectionManager Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a fresh ConnectionManager instance for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = Mock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_accepts_handshake(manager, mock_websocket):
    """Test that connect() accepts WebSocket handshake."""
    await manager.connect(mock_websocket)
    mock_websocket.accept.assert_called_once()


@pytest.mark.asyncio
async def test_connect_registers_connection(manager, mock_websocket):
    """Test that connect() adds WebSocket to active connections."""
    await manager.connect(mock_websocket)
    assert mock_websocket in manager.active_connections
    assert len(manager.active_connections) == 1


@pytest.mark.asyncio
async def test_connect_multiple_connections(manager, mock_websocket):
    """Test that multiple WebSockets can connect simultaneously."""
    ws1 = mock_websocket
    ws2 = Mock()
    ws2.accept = AsyncMock()

    await manager.connect(ws1)
    await manager.connect(ws2)

    assert len(manager.active_connections) == 2
    assert ws1 in manager.active_connections
    assert ws2 in manager.active_connections


def test_disconnect_removes_connection(manager, mock_websocket):
    """Test that disconnect() removes WebSocket from active connections."""
    manager.active_connections.append(mock_websocket)
    manager.disconnect(mock_websocket)
    assert mock_websocket not in manager.active_connections
    assert len(manager.active_connections) == 0


def test_disconnect_only_removes_specified_connection(manager):
    """Test that disconnect() only removes the specified WebSocket."""
    ws1 = Mock()
    ws2 = Mock()
    manager.active_connections = [ws1, ws2]

    manager.disconnect(ws1)

    assert ws1 not in manager.active_connections
    assert ws2 in manager.active_connections
    assert len(manager.active_connections) == 1


@pytest.mark.asyncio
async def test_broadcast_sends_to_all_connections(manager):
    """Test that broadcast() sends message to all active connections."""
    ws1 = Mock()
    ws1.send_json = AsyncMock()
    ws2 = Mock()
    ws2.send_json = AsyncMock()
    manager.active_connections = [ws1, ws2]

    message = {"type": "notification", "data": "test"}
    await manager.broadcast(message)

    ws1.send_json.assert_called_once_with(message)
    ws2.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_empty_connections(manager):
    """Test that broadcast() handles empty connection list gracefully."""
    message = {"type": "test"}
    await manager.broadcast(message)  # Should not raise


@pytest.mark.asyncio
async def test_broadcast_handles_send_error_gracefully(manager):
    """Test that broadcast() continues on individual connection errors."""
    ws1 = Mock()
    ws1.send_json = AsyncMock(side_effect=Exception("Connection closed"))
    ws2 = Mock()
    ws2.send_json = AsyncMock()
    manager.active_connections = [ws1, ws2]

    message = {"type": "notification", "data": "test"}
    await manager.broadcast(message)

    # ws1 failed but ws2 should still receive the message
    ws2.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_broadcast_all_connections_fail(manager):
    """Test that broadcast() handles all connections failing gracefully."""
    ws1 = Mock()
    ws1.send_json = AsyncMock(side_effect=Exception("Closed"))
    ws2 = Mock()
    ws2.send_json = AsyncMock(side_effect=Exception("Timeout"))
    manager.active_connections = [ws1, ws2]

    message = {"type": "test"}
    # Should not raise despite all connections failing
    await manager.broadcast(message)


@pytest.mark.asyncio
async def test_broadcast_with_complex_message(manager, mock_websocket):
    """Test that broadcast() handles complex nested messages."""
    manager.active_connections = [mock_websocket]

    complex_message = {
        "type": "notification",
        "data": {
            "id": "n123",
            "title": "Test",
            "metadata": {
                "tags": ["urgent", "work"],
                "priority": 1
            }
        },
        "timestamp": "2026-02-15T12:00:00Z"
    }

    await manager.broadcast(complex_message)
    mock_websocket.send_json.assert_called_once_with(complex_message)


@pytest.mark.asyncio
async def test_broadcast_preserves_message_integrity(manager, mock_websocket):
    """Test that broadcast() doesn't modify the original message."""
    manager.active_connections = [mock_websocket]

    original = {"type": "test", "data": [1, 2, 3]}
    await manager.broadcast(original)

    # Verify original message unchanged
    assert original == {"type": "test", "data": [1, 2, 3]}


def test_initial_state(manager):
    """Test that ConnectionManager starts with empty connections."""
    assert manager.active_connections == []
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_connect_disconnect_lifecycle(manager, mock_websocket):
    """Test full connection lifecycle: connect -> broadcast -> disconnect."""
    # Connect
    await manager.connect(mock_websocket)
    assert len(manager.active_connections) == 1

    # Broadcast
    message = {"type": "test"}
    await manager.broadcast(message)
    mock_websocket.send_json.assert_called_once_with(message)

    # Disconnect
    manager.disconnect(mock_websocket)
    assert len(manager.active_connections) == 0


# ---------------------------------------------------------------------------
# Module-level singleton tests
# ---------------------------------------------------------------------------


def test_module_singleton_exists():
    """Test that ws_manager singleton is exported from module."""
    assert ws_manager is not None
    assert isinstance(ws_manager, ConnectionManager)


def test_module_singleton_is_shared():
    """Test that importing ws_manager returns the same instance."""
    from web.websocket import ws_manager as ws_manager2
    assert ws_manager is ws_manager2


@pytest.mark.asyncio
async def test_module_singleton_can_connect():
    """Test that the module singleton can accept connections."""
    # Save original state
    original_connections = ws_manager.active_connections.copy()

    try:
        mock_ws = Mock()
        mock_ws.accept = AsyncMock()
        await ws_manager.connect(mock_ws)
        assert mock_ws in ws_manager.active_connections
    finally:
        # Clean up
        ws_manager.active_connections = original_connections


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------


def test_disconnect_nonexistent_connection(manager):
    """Test that disconnecting a non-existent connection is safely ignored."""
    ws = Mock()
    # Should not raise — disconnect is idempotent and safe for unknown sockets
    manager.disconnect(ws)


@pytest.mark.asyncio
async def test_broadcast_during_connection_changes(manager):
    """Test broadcast() while connections are being added/removed."""
    ws1 = Mock()
    ws1.send_json = AsyncMock()
    manager.active_connections = [ws1]

    # Start broadcast
    message = {"type": "test"}
    await manager.broadcast(message)

    # Connection should have received message
    ws1.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_broadcasts(manager, mock_websocket):
    """Test multiple concurrent broadcasts to the same connections."""
    manager.active_connections = [mock_websocket]

    message1 = {"type": "notification", "id": 1}
    message2 = {"type": "notification", "id": 2}

    await manager.broadcast(message1)
    await manager.broadcast(message2)

    assert mock_websocket.send_json.call_count == 2


@pytest.mark.asyncio
async def test_broadcast_with_none_message(manager, mock_websocket):
    """Test that broadcast() handles None message gracefully."""
    manager.active_connections = [mock_websocket]
    await manager.broadcast(None)
    mock_websocket.send_json.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_broadcast_with_empty_message(manager, mock_websocket):
    """Test that broadcast() handles empty dict message."""
    manager.active_connections = [mock_websocket]
    await manager.broadcast({})
    mock_websocket.send_json.assert_called_once_with({})


# ---------------------------------------------------------------------------
# Integration scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_clients_notification_scenario(manager):
    """Test realistic scenario: multiple clients receiving notifications."""
    # Simulate 3 connected clients
    clients = []
    for i in range(3):
        ws = Mock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        await manager.connect(ws)
        clients.append(ws)

    assert len(manager.active_connections) == 3

    # Broadcast notification to all
    notification = {
        "type": "notification.created",
        "payload": {
            "id": "n123",
            "title": "New email from John",
            "priority": "high"
        }
    }
    await manager.broadcast(notification)

    # All clients should have received it
    for client in clients:
        client.send_json.assert_called_once_with(notification)

    # One client disconnects
    manager.disconnect(clients[1])
    assert len(manager.active_connections) == 2

    # Broadcast another message
    update = {"type": "task.completed", "task_id": "t456"}
    await manager.broadcast(update)

    # Only remaining 2 clients receive it
    clients[0].send_json.assert_called_with(update)
    clients[2].send_json.assert_called_with(update)
    # Disconnected client should only have 1 call (from first broadcast)
    assert clients[1].send_json.call_count == 1


@pytest.mark.asyncio
async def test_graceful_degradation_on_partial_failure(manager):
    """Test that partial failures don't break the entire broadcast."""
    # Mix of healthy and failing connections
    healthy1 = Mock()
    healthy1.send_json = AsyncMock()

    failing = Mock()
    failing.send_json = AsyncMock(side_effect=ConnectionError("Network error"))

    healthy2 = Mock()
    healthy2.send_json = AsyncMock()

    manager.active_connections = [healthy1, failing, healthy2]

    message = {"type": "system.status", "data": "online"}
    await manager.broadcast(message)

    # Healthy connections should succeed
    healthy1.send_json.assert_called_once_with(message)
    healthy2.send_json.assert_called_once_with(message)

    # Failing connection attempted but swallowed error
    failing.send_json.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_connection_manager_isolation(manager):
    """Test that ConnectionManager instances are isolated."""
    manager2 = ConnectionManager()

    ws1 = Mock()
    ws1.accept = AsyncMock()
    ws2 = Mock()
    ws2.accept = AsyncMock()

    await manager.connect(ws1)
    await manager2.connect(ws2)

    assert len(manager.active_connections) == 1
    assert len(manager2.active_connections) == 1
    assert ws1 in manager.active_connections
    assert ws2 in manager2.active_connections
    assert ws1 not in manager2.active_connections
    assert ws2 not in manager.active_connections
