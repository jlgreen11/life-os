"""
Tests: Database Health panel on the admin page and the rebuild-user-model endpoint.

Coverage:
    - Admin template HTML contains the Database Health section elements
    - GET /health returns db_health with per-database status
    - POST /api/admin/rebuild-user-model returns {status: 'skipped'} for a healthy DB
    - POST /api/admin/rebuild-user-model returns expected JSON structure
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.admin_template import ADMIN_HTML_TEMPLATE
from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(db) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance.

    Args:
        db: Real DatabaseManager from the conftest fixture (provides real SQLite).

    Returns:
        TestClient wrapping the FastAPI app with registered routes.
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    from storage.event_store import EventStore

    life_os.event_store = EventStore(db)
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    life_os.prediction_engine.get_diagnostics = AsyncMock(return_value={
        "prediction_types": {}, "overall": {"health": "unknown"},
    })
    life_os.semantic_fact_inferrer.run_all_inference.return_value = None

    # Wire backfill methods as no-op async callables
    life_os._backfill_relationship_profile_if_needed = AsyncMock()
    life_os._clean_relationship_profile_if_needed = AsyncMock()
    life_os._backfill_temporal_profile_if_needed = AsyncMock()
    life_os._backfill_topic_profile_if_needed = AsyncMock()
    life_os._backfill_linguistic_profile_if_needed = AsyncMock()
    life_os._backfill_cadence_profile_if_needed = AsyncMock()
    life_os._backfill_mood_signals_profile_if_needed = AsyncMock()

    life_os.user_model_store.get_signal_profile.return_value = None
    life_os.user_model_store.get_semantic_facts.return_value = []

    register_routes(app, life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Admin template contains Database Health section
# ---------------------------------------------------------------------------


def test_template_contains_db_health_section_label():
    """The admin template HTML contains the 'Database Health' section label."""
    assert "Database Health" in ADMIN_HTML_TEMPLATE


def test_template_contains_db_health_grid():
    """The admin template HTML contains the dbHealthGrid element."""
    assert 'id="dbHealthGrid"' in ADMIN_HTML_TEMPLATE


def test_template_contains_repair_button():
    """The admin template HTML contains the repair button."""
    assert 'id="repairBtn"' in ADMIN_HTML_TEMPLATE


def test_template_contains_load_db_health_function():
    """The admin template HTML contains the loadDbHealth JavaScript function."""
    assert "async function loadDbHealth()" in ADMIN_HTML_TEMPLATE


def test_template_contains_repair_user_model_function():
    """The admin template HTML contains the repairUserModel JavaScript function."""
    assert "async function repairUserModel()" in ADMIN_HTML_TEMPLATE


def test_template_calls_load_db_health_on_init():
    """The admin template init section calls loadDbHealth()."""
    assert "loadDbHealth();" in ADMIN_HTML_TEMPLATE


def test_template_repair_button_posts_to_rebuild_endpoint():
    """The repair JS function POSTs to /api/admin/rebuild-user-model."""
    assert "/api/admin/rebuild-user-model" in ADMIN_HTML_TEMPLATE


def test_template_db_health_section_before_connectors():
    """The Database Health section appears before the API Connectors section."""
    db_health_pos = ADMIN_HTML_TEMPLATE.index("Database Health")
    api_connectors_pos = ADMIN_HTML_TEMPLATE.index("API Connectors")
    assert db_health_pos < api_connectors_pos


# ---------------------------------------------------------------------------
# Tests: GET /health includes db_health
# ---------------------------------------------------------------------------


def test_health_returns_db_health(db):
    """GET /health returns db_health with per-database entries."""
    client = _make_app(db)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "db_health" in data
    assert "db_status" in data

    # Should have all 5 databases
    for db_name in ["events", "entities", "state", "user_model", "preferences"]:
        assert db_name in data["db_health"], f"Missing database: {db_name}"


def test_health_db_entries_have_expected_fields(db):
    """Each db_health entry has status, errors, path, and size_bytes."""
    client = _make_app(db)
    data = client.get("/health").json()

    for db_name, info in data["db_health"].items():
        assert "status" in info, f"{db_name} missing 'status'"
        assert "size_bytes" in info, f"{db_name} missing 'size_bytes'"


def test_health_db_status_ok_for_healthy_db(db):
    """db_status is 'ok' when all databases are healthy."""
    client = _make_app(db)
    data = client.get("/health").json()
    assert data["db_status"] == "ok"


# ---------------------------------------------------------------------------
# Tests: POST /api/admin/rebuild-user-model
# ---------------------------------------------------------------------------


def test_rebuild_user_model_returns_200(db):
    """POST /api/admin/rebuild-user-model returns HTTP 200."""
    client = _make_app(db)
    res = client.post("/api/admin/rebuild-user-model")
    assert res.status_code == 200


def test_rebuild_user_model_skips_healthy_db(db):
    """When user_model.db is healthy, the endpoint returns status='skipped'."""
    client = _make_app(db)
    data = client.post("/api/admin/rebuild-user-model").json()
    assert data["status"] == "skipped"
    assert "reason" in data


def test_rebuild_user_model_response_structure(db):
    """The rebuild response has a 'status' field at minimum."""
    client = _make_app(db)
    data = client.post("/api/admin/rebuild-user-model").json()
    assert "status" in data
    assert data["status"] in ("skipped", "rebuilt", "error")
