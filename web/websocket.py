"""
Life OS — WebSocket Connection Manager

Manages WebSocket connections for real-time notifications and updates.
Services across Life OS (event bus, notification manager, etc.) can call
``ws_manager.broadcast(...)`` to push messages to every connected client
without needing to know about the web layer's internals.
"""

from __future__ import annotations

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections for real-time updates.

    Singleton pattern: a single module-level instance (``ws_manager``) is
    created below and shared across the entire application.  This avoids
    passing the manager through every layer and ensures all broadcast calls
    reach the same set of connected clients.
    """

    def __init__(self):
        # All currently open WebSocket connections.  Connections are added on
        # handshake and removed on disconnect or error.
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket handshake and register the connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a connection from the active set (called on client disconnect).

        Safe to call multiple times for the same websocket — a second call
        (e.g. from a race between broadcast cleanup and the disconnect handler)
        is silently ignored.
        """
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed or never fully connected

    async def broadcast(self, message: dict):
        """Send a JSON message to every connected client.

        Dead connections (clients that disconnected between broadcasts) are
        automatically removed from the active set to prevent error accumulation.
        Removal happens after the iteration to avoid modifying the list mid-loop.
        """
        dead: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for ws in dead:
            try:
                self.active_connections.remove(ws)
            except ValueError:
                pass  # Already removed by a concurrent disconnect call


# Module-level singleton — imported by routes.py and any service that needs
# to push real-time updates to the browser dashboard.
ws_manager = ConnectionManager()
