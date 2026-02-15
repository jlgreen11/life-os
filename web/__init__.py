"""
Life OS — Web Package

FastAPI-based web interface with REST API, WebSocket, and HTML UI.
"""

from web.app import create_web_app
from web.websocket import ws_manager, ConnectionManager

__all__ = ["create_web_app", "ws_manager", "ConnectionManager"]
