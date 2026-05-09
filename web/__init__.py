"""Web application — FastAPI REST API, WebSocket real-time updates, and HTML dashboard."""

from web.app import create_web_app
from web.websocket import ConnectionManager, ws_manager

__all__ = ["ConnectionManager", "create_web_app", "ws_manager"]
