"""
Tests: /api/source-weights endpoints — HTTP-level coverage for the tunable
source weight system.

The underlying SourceWeightManager is well-tested in tests/test_source_weights.py.
These tests verify the HTTP routing layer: request parsing, response format,
status codes, and error handling for all 5 endpoints.

Endpoints covered:
  GET  /api/source-weights                         — list all weights grouped by category
  GET  /api/source-weights/{source_key}            — single source stats
  PUT  /api/source-weights/{source_key}            — update user weight
  POST /api/source-weights/{source_key}/reset-drift — reset AI drift to zero
  POST /api/source-weights                         — create a custom source weight
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.insight_engine.source_weights import DEFAULT_WEIGHTS, SourceWeightManager
from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(db) -> TestClient:
    """Create a minimal FastAPI test client with a real SourceWeightManager."""
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    # Real SourceWeightManager backed by the test database
    swm = SourceWeightManager(db)
    swm.seed_defaults()
    life_os.source_weight_manager = swm

    # Stub unrelated services to prevent AttributeError during route setup
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.user_model_store = MagicMock()
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    life_os.event_store = MagicMock()
    life_os.signal_extractor = MagicMock()

    register_routes(app, life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/source-weights — list all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_source_weights_returns_all(db):
    """GET /api/source-weights returns 200 with weights array matching DEFAULT_WEIGHTS count."""
    client = _make_app(db)
    resp = client.get("/api/source-weights")

    assert resp.status_code == 200
    data = resp.json()
    assert "weights" in data
    assert "by_category" in data
    assert "count" in data
    assert data["count"] == len(DEFAULT_WEIGHTS)
    assert len(data["weights"]) == len(DEFAULT_WEIGHTS)


@pytest.mark.asyncio
async def test_list_source_weights_grouped_by_category(db):
    """GET /api/source-weights groups weights correctly in by_category dict."""
    client = _make_app(db)
    resp = client.get("/api/source-weights")

    assert resp.status_code == 200
    data = resp.json()
    by_category = data["by_category"]

    # Verify expected categories are present
    assert "email" in by_category
    assert "messaging" in by_category
    assert "calendar" in by_category

    # Verify grouping is correct: every weight in a category group has that category
    for cat, weights in by_category.items():
        for w in weights:
            assert w["category"] == cat, f"Weight {w['source_key']} has category {w['category']} but is in group {cat}"

    # Total across all categories should equal count
    total_in_groups = sum(len(ws) for ws in by_category.values())
    assert total_in_groups == data["count"]


# ---------------------------------------------------------------------------
# GET /api/source-weights/{source_key} — single source stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_single_source_weight(db):
    """GET /api/source-weights/email.personal returns 200 with stats dict."""
    client = _make_app(db)
    resp = client.get("/api/source-weights/email.personal")

    assert resp.status_code == 200
    data = resp.json()
    assert data["source_key"] == "email.personal"
    assert "effective_weight" in data
    assert "user_weight" in data
    assert "ai_drift_raw" in data
    assert "engagement_rate" in data
    assert "drift_active" in data


@pytest.mark.asyncio
async def test_get_nonexistent_source_weight_404(db):
    """GET /api/source-weights/nonexistent returns 404."""
    client = _make_app(db)
    resp = client.get("/api/source-weights/nonexistent")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/source-weights/{source_key} — update user weight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_source_weight(db):
    """PUT /api/source-weights/email.personal with {weight: 0.9} returns 200 and persists."""
    client = _make_app(db)
    resp = client.put("/api/source-weights/email.personal", json={"weight": 0.9})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "updated"
    assert data["weight"]["user_weight"] == 0.9

    # Verify persistence via GET
    get_resp = client.get("/api/source-weights/email.personal")
    assert get_resp.status_code == 200
    assert get_resp.json()["user_weight"] == 0.9


@pytest.mark.asyncio
async def test_update_nonexistent_source_weight_404(db):
    """PUT /api/source-weights/nonexistent returns 404."""
    client = _make_app(db)
    resp = client.put("/api/source-weights/nonexistent", json={"weight": 0.5})

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/source-weights/{source_key}/reset-drift — reset AI drift
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_drift(db):
    """POST /api/source-weights/email.personal/reset-drift returns 200."""
    client = _make_app(db)
    resp = client.post("/api/source-weights/email.personal/reset-drift")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reset"
    assert data["weight"]["ai_drift"] == 0.0


@pytest.mark.asyncio
async def test_reset_drift_nonexistent_404(db):
    """POST /api/source-weights/nonexistent/reset-drift returns 404."""
    client = _make_app(db)
    resp = client.post("/api/source-weights/nonexistent/reset-drift")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/source-weights — create custom source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_custom_source(db):
    """POST /api/source-weights with valid body returns 200 with created source."""
    client = _make_app(db)
    resp = client.post(
        "/api/source-weights",
        json={
            "source_key": "email.client_acme",
            "category": "email",
            "label": "ACME Corp Email",
            "description": "All email from acme-corp.com",
            "weight": 0.7,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["weight"]["source_key"] == "email.client_acme"
    assert data["weight"]["user_weight"] == 0.7


@pytest.mark.asyncio
async def test_list_after_create_includes_custom(db):
    """Creating a custom source and then listing all should include it."""
    client = _make_app(db)

    # Create a custom source
    client.post(
        "/api/source-weights",
        json={
            "source_key": "messaging.vip",
            "category": "messaging",
            "label": "VIP Messages",
            "description": "Messages from VIP contacts",
            "weight": 0.95,
        },
    )

    # List all and verify the custom source appears
    resp = client.get("/api/source-weights")
    assert resp.status_code == 200
    data = resp.json()

    source_keys = [w["source_key"] for w in data["weights"]]
    assert "messaging.vip" in source_keys
    assert data["count"] == len(DEFAULT_WEIGHTS) + 1
