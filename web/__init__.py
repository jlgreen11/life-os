"""Web application — FastAPI REST API, WebSocket real-time updates, and HTML dashboard."""

from web.app import create_web_app
from web.websocket import ws_manager, ConnectionManager

__all__ = ["create_web_app", "ws_manager", "ConnectionManager"]
