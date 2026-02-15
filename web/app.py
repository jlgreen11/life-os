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
    """Create the FastAPI application with all routes."""

    app = FastAPI(
        title="Life OS",
        description="Your Private Command Center",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Lock down in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store reference to Life OS instance
    app.state.life_os = life_os

    # Register all routes
    register_routes(app, life_os)

    return app
