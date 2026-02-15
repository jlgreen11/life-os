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
        """Remove a connection from the active set (called on client disconnect)."""
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a JSON message to every connected client.

        Errors for individual connections (e.g. a client that disconnected
        between the iteration start and the send call) are silently caught
        so that one broken connection does not prevent delivery to the others.
        A future improvement could remove the failed connection from the list.
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Swallow send errors for disconnected clients to avoid
                # breaking the broadcast loop for remaining connections.
                pass


# Module-level singleton — imported by routes.py and any service that needs
# to push real-time updates to the browser dashboard.
ws_manager = ConnectionManager()
