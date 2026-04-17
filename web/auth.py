"""
Life OS — API Key Authentication

Optional API key authentication for the web API. When enabled, all requests
must include a valid API key via one of:

  * ``X-API-Key`` HTTP header (preferred for REST clients)
  * ``api_key`` query parameter (required for WebSocket clients that can't
    set custom headers, e.g. browsers and iOS ``URLSessionWebSocketTask``)
  * ``Authorization: Bearer <key>`` header

Auth is opt-in and off by default so local installs keep working without
any config change. Enable it before exposing Life OS beyond localhost (LAN,
Tailscale, reverse proxy, tunnel).

Exempt paths (no key required): ``/health`` is reachable unauthenticated so
that status dashboards and the iOS connection check can confirm the server
is alive before the user configures a key. The payload contains only
service-up/down booleans, not personal data.
"""

from __future__ import annotations

import hmac
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

EXEMPT_PATHS = frozenset({"/health"})


def _extract_key(request: Request) -> str | None:
    header_key = request.headers.get("x-api-key")
    if header_key:
        return header_key.strip()
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    query_key = request.query_params.get("api_key")
    if query_key:
        return query_key.strip()
    return None


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests lacking a valid API key.

    ``expected_key`` is compared with ``hmac.compare_digest`` to avoid
    timing side channels. WebSocket upgrades are handled by a separate
    check in the ``/ws`` route because Starlette middleware runs before
    the WebSocket handshake completes and can't reject cleanly; REST and
    preflight requests go through this middleware.
    """

    def __init__(self, app, expected_key: str):
        super().__init__(app)
        self._expected_key = expected_key

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        # WebSocket upgrades bypass HTTP middleware; auth is enforced in the
        # /ws route handler which calls verify_api_key() before accepting.
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        provided = _extract_key(request)
        if not provided or not hmac.compare_digest(provided, self._expected_key):
            return JSONResponse(
                {"error": "unauthorized", "detail": "Missing or invalid API key"},
                status_code=401,
            )

        return await call_next(request)


def verify_api_key(provided: str | None, expected: str | None) -> bool:
    """Constant-time API key comparison for WebSocket handshakes.

    Returns True when auth is disabled (``expected`` is falsy) — callers
    decide whether to accept anonymous connections in that case.
    """
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)
