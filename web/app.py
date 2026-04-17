"""
Life OS — Web Application Factory

Creates and configures the FastAPI application.
Routes, schemas, websocket, and templates are in separate modules.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.auth import APIKeyAuthMiddleware
from web.routes import register_routes

logger = logging.getLogger(__name__)


def create_web_app(life_os) -> FastAPI:
    """Create the FastAPI application with all routes.

    Application factory pattern: this function constructs and configures a
    fresh FastAPI instance each time it is called.  This makes testing easier
    (each test can create its own app with a mock ``life_os``) and keeps
    module-level side effects to zero.

    CORS Security
    -------------
    The allowed origins are read from ``life_os.config["cors"]["allowed_origins"]``.
    If the config is missing or malformed, the application defaults to a secure
    localhost-only policy (http://localhost:8080, http://127.0.0.1:8080).

    **Never use the wildcard origin "*" in production.** It allows any website
    to make authenticated requests to your Life OS API, which can leak sensitive
    personal data (emails, calendar, messages, transactions, etc.) to malicious
    third-party sites.

    Examples of secure CORS configurations in ``settings.yaml``:

    .. code-block:: yaml

        # Local development
        cors:
          allowed_origins:
            - "http://localhost:8080"
            - "http://localhost:3000"

        # Production deployment
        cors:
          allowed_origins:
            - "https://mylifeos.example.com"

    Args:
        life_os: The LifeOS orchestrator instance containing all services and
            configuration.

    Returns:
        Configured FastAPI application instance.
    """

    app = FastAPI(
        title="Life OS",
        description="Your Private Command Center",
        version="0.1.0",
    )

    # --- CORS Middleware ---
    # Read allowed origins from config. If missing or if life_os has no config
    # attribute (e.g., in minimal test fixtures), default to localhost-only
    # (secure by default). The user must explicitly configure broader access.
    config = getattr(life_os, "config", {}) or {}
    cors_config = config.get("cors", {})
    allowed_origins = cors_config.get("allowed_origins")

    # Validate and sanitize the allowed_origins list
    if allowed_origins is None or not isinstance(allowed_origins, list):
        # No config or invalid config → default to secure localhost-only policy
        allowed_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    else:
        # Filter out empty strings and non-string values
        allowed_origins = [
            origin.strip()
            for origin in allowed_origins
            if isinstance(origin, str) and origin.strip()
        ]

        # If the list is now empty after filtering, fall back to secure defaults
        if not allowed_origins:
            allowed_origins = [
                "http://localhost:8080",
                "http://127.0.0.1:8080",
            ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Optional API Key Authentication ---
    # When ``auth.api_key`` is set in settings.yaml, require all requests to
    # include it via X-API-Key header, Authorization: Bearer, or ?api_key=.
    # Off by default so local installs keep working. Enable before exposing
    # Life OS beyond localhost (LAN, Tailscale, reverse proxy, tunnel).
    auth_config = config.get("auth", {}) or {}
    api_key = auth_config.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        app.add_middleware(APIKeyAuthMiddleware, expected_key=api_key.strip())
        app.state.api_key = api_key.strip()
        logger.info("API key authentication enabled")
    else:
        app.state.api_key = None
        if api_key:
            logger.warning("auth.api_key is set but not a non-empty string — auth disabled")

    # Attach the Life OS instance to ``app.state`` so that route handlers can
    # access all services (event_store, vector_store, ai_engine, task_manager,
    # etc.) without relying on global variables or dependency injection.
    # Route handlers receive it directly via the ``life_os`` closure captured
    # in ``register_routes``.
    app.state.life_os = life_os

    # Register all REST API routes, the WebSocket endpoint, and the HTML UI.
    register_routes(app, life_os)

    return app
