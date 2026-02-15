"""
Life OS — Web Application Factory

Creates and configures the FastAPI application.
Routes, schemas, websocket, and templates are in separate modules.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.routes import register_routes


def create_web_app(life_os) -> FastAPI:
    """Create the FastAPI application with all routes.

    Application factory pattern: this function constructs and configures a
    fresh FastAPI instance each time it is called.  This makes testing easier
    (each test can create its own app with a mock ``life_os``) and keeps
    module-level side effects to zero.
    """

    app = FastAPI(
        title="Life OS",
        description="Your Private Command Center",
        version="0.1.0",
    )

    # --- CORS Middleware ---
    # Currently configured with wildcard origins ("*"), which allows requests
    # from any domain.  This is acceptable during local development but MUST
    # be restricted to the actual frontend origin(s) before any production or
    # public deployment to prevent cross-site request forgery.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO [FLAGGED]: Lock down to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach the Life OS instance to ``app.state`` so that route handlers can
    # access all services (event_store, vector_store, ai_engine, task_manager,
    # etc.) without relying on global variables or dependency injection.
    # Route handlers receive it directly via the ``life_os`` closure captured
    # in ``register_routes``.
    app.state.life_os = life_os

    # Register all REST API routes, the WebSocket endpoint, and the HTML UI.
    register_routes(app, life_os)

    return app
