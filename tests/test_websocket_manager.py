"""Tests for the WebSocket ConnectionManager (web/websocket.py).

Verifies that connect, disconnect, and broadcast behave correctly, including
edge cases like double-disconnect and dead connection cleanup during broadcast.
"""

from unittest.mock import AsyncMock

import pytest

from web.websocket import ConnectionManager


@pytest.fixture()
def manager():
    """A fresh ConnectionManager with no connections."""
    return ConnectionManager()


def _make_ws(*, fail_send: bool = False) -> AsyncMock:
    """Create a mock WebSocket.

    Args:
        fail_send: If True, ``send_json`` raises a ``ConnectionError``
            to simulate a client that has already disconnected.
    """
    ws = AsyncMock()
    ws.accept = AsyncMock()
    if fail_send:
        ws.send_json = AsyncMock(side_effect=ConnectionError("client gone"))
    else:
        ws.send_json = AsyncMock()
    return ws


# ------------------------------------------------------------------
# connect()
# ------------------------------------------------------------------


async def test_connect_adds_to_list(manager):
    """connect() should accept the handshake and add the websocket."""
    ws = _make_ws()
    await manager.connect(ws)

    ws.accept.assert_awaited_once()
    assert ws in manager.active_connections
    assert len(manager.active_connections) == 1


# ------------------------------------------------------------------
# disconnect()
# ------------------------------------------------------------------


async def test_disconnect_removes_from_list(manager):
    """disconnect() should remove a previously connected websocket."""
    ws = _make_ws()
    await manager.connect(ws)
    manager.disconnect(ws)

    assert ws not in manager.active_connections
    assert len(manager.active_connections) == 0


async def test_disconnect_idempotent(manager):
    """Calling disconnect() twice for the same websocket must not raise."""
    ws = _make_ws()
    await manager.connect(ws)

    manager.disconnect(ws)
    manager.disconnect(ws)  # should not raise ValueError

    assert ws not in manager.active_connections


async def test_disconnect_unknown_websocket(manager):
    """disconnect() with a websocket that was never added must not raise."""
    ws = _make_ws()
    manager.disconnect(ws)  # should not raise ValueError


# ------------------------------------------------------------------
# broadcast()
# ------------------------------------------------------------------


async def test_broadcast_sends_to_all(manager):
    """broadcast() should deliver the message to every connected client."""
    ws1 = _make_ws()
    ws2 = _make_ws()
    await manager.connect(ws1)
    await manager.connect(ws2)

    msg = {"type": "test", "data": "hello"}
    await manager.broadcast(msg)

    ws1.send_json.assert_awaited_once_with(msg)
    ws2.send_json.assert_awaited_once_with(msg)


async def test_broadcast_removes_dead_connections(manager):
    """A websocket that fails on send_json should be removed after broadcast."""
    dead_ws = _make_ws(fail_send=True)
    await manager.connect(dead_ws)

    await manager.broadcast({"type": "ping"})

    assert dead_ws not in manager.active_connections
    assert len(manager.active_connections) == 0


async def test_broadcast_continues_after_failure(manager):
    """If the middle connection fails, the first and third should still receive."""
    ws1 = _make_ws()
    ws_bad = _make_ws(fail_send=True)
    ws3 = _make_ws()

    await manager.connect(ws1)
    await manager.connect(ws_bad)
    await manager.connect(ws3)

    msg = {"type": "update", "payload": 42}
    await manager.broadcast(msg)

    # Good connections received the message
    ws1.send_json.assert_awaited_once_with(msg)
    ws3.send_json.assert_awaited_once_with(msg)

    # Dead connection was removed, good ones remain
    assert ws_bad not in manager.active_connections
    assert ws1 in manager.active_connections
    assert ws3 in manager.active_connections
    assert len(manager.active_connections) == 2
