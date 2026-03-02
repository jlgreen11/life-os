"""
Tests: Graceful error handling for API endpoints when backing services fail.

Verifies that /api/search, /api/draft, /api/browser/status, and
/api/browser/vault return structured JSON error responses (not 500s) when
their backing services raise exceptions or are unavailable.

Each test mocks the relevant service to raise RuntimeError and asserts the
response contains an 'error' key with a user-friendly message.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(db, *, browser_orchestrator=MagicMock()) -> TestClient:
    """Create a minimal FastAPI test client with mocked services.

    All services are stubbed out so routes can register without errors.
    Individual tests override specific attributes to inject failures.
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    # Stub services used during route registration / health checks
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.user_model_store = MagicMock()
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    life_os.event_store = MagicMock()
    life_os.signal_extractor = MagicMock()
    life_os.browser_orchestrator = browser_orchestrator

    register_routes(app, life_os)
    return TestClient(app), life_os


# ---------------------------------------------------------------------------
# POST /api/search — error handling
# ---------------------------------------------------------------------------


class TestSearchErrorHandling:
    """Verify /api/search returns a graceful fallback when vector search fails."""

    def test_search_returns_error_on_service_failure(self, db):
        """When vector_store.search() raises, the endpoint returns an error
        object with empty results instead of a 500."""
        client, life_os = _make_app(db)
        life_os.vector_store.search.side_effect = RuntimeError("vector store offline")

        resp = client.post("/api/search", json={"query": "test query"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Search temporarily unavailable"
        assert data["results"] == []
        assert data["count"] == 0
        assert data["query"] == "test query"

    def test_search_works_when_service_healthy(self, db):
        """Normal search behavior is unchanged when the service is available."""
        client, life_os = _make_app(db)
        life_os.vector_store.search.return_value = [{"id": "1", "text": "result"}]

        resp = client.post("/api/search", json={"query": "hello"})

        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["count"] == 1
        assert len(data["results"]) == 1


# ---------------------------------------------------------------------------
# POST /api/draft — error handling
# ---------------------------------------------------------------------------


class TestDraftErrorHandling:
    """Verify /api/draft returns a graceful fallback when draft generation fails."""

    def test_draft_returns_error_on_service_failure(self, db):
        """When ai_engine.draft_reply() raises, the endpoint returns an error
        object with draft=None instead of a 500."""
        client, life_os = _make_app(db)
        life_os.ai_engine.draft_reply = AsyncMock(
            side_effect=RuntimeError("AI engine offline")
        )

        resp = client.post(
            "/api/draft",
            json={"incoming_message": "Hello", "channel": "email"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] == "Draft generation temporarily unavailable"
        assert data["draft"] is None

    def test_draft_works_when_service_healthy(self, db):
        """Normal draft behavior is unchanged when the AI engine is available."""
        client, life_os = _make_app(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Suggested reply text")

        resp = client.post(
            "/api/draft",
            json={"incoming_message": "Hello", "channel": "email"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["draft"] == "Suggested reply text"


# ---------------------------------------------------------------------------
# GET /api/browser/status — error handling
# ---------------------------------------------------------------------------


class TestBrowserStatusErrorHandling:
    """Verify /api/browser/status handles missing or failing orchestrators."""

    def test_browser_status_returns_error_when_orchestrator_is_none(self, db):
        """When browser_orchestrator is None (not configured), the endpoint
        returns a structured unavailable response instead of crashing."""
        client, _life_os = _make_app(db, browser_orchestrator=None)

        resp = client.get("/api/browser/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["error"] == "Browser orchestrator not configured"

    def test_browser_status_returns_error_on_exception(self, db):
        """When get_status() raises, the endpoint returns a structured error."""
        orch = MagicMock()
        orch.get_status.side_effect = RuntimeError("browser crashed")
        client, _life_os = _make_app(db, browser_orchestrator=orch)

        resp = client.get("/api/browser/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unavailable"
        assert "error" in data

    def test_browser_status_works_when_orchestrator_healthy(self, db):
        """Normal behavior is unchanged when orchestrator is available."""
        orch = MagicMock()
        orch.get_status.return_value = {"status": "running", "active_connectors": 2}
        client, _life_os = _make_app(db, browser_orchestrator=orch)

        resp = client.get("/api/browser/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "error" not in data


# ---------------------------------------------------------------------------
# GET /api/browser/vault — error handling
# ---------------------------------------------------------------------------


class TestBrowserVaultErrorHandling:
    """Verify /api/browser/vault handles missing or failing orchestrators."""

    def test_browser_vault_returns_error_when_orchestrator_is_none(self, db):
        """When browser_orchestrator is None (not configured), the endpoint
        returns an empty sites list with an error message."""
        client, _life_os = _make_app(db, browser_orchestrator=None)

        resp = client.get("/api/browser/vault")

        assert resp.status_code == 200
        data = resp.json()
        assert data["sites"] == []
        assert data["error"] == "Browser orchestrator not configured"

    def test_browser_vault_returns_error_on_exception(self, db):
        """When get_vault_sites() raises, the endpoint returns a structured error."""
        orch = MagicMock()
        orch.get_vault_sites.side_effect = RuntimeError("vault locked")
        client, _life_os = _make_app(db, browser_orchestrator=orch)

        resp = client.get("/api/browser/vault")

        assert resp.status_code == 200
        data = resp.json()
        assert data["sites"] == []
        assert "error" in data

    def test_browser_vault_works_when_orchestrator_healthy(self, db):
        """Normal behavior is unchanged when orchestrator is available."""
        orch = MagicMock()
        orch.get_vault_sites.return_value = ["site1.com", "site2.com"]
        client, _life_os = _make_app(db, browser_orchestrator=orch)

        resp = client.get("/api/browser/vault")

        assert resp.status_code == 200
        data = resp.json()
        assert data["sites"] == ["site1.com", "site2.com"]
        assert "error" not in data
