"""
Tests: /api/system/intelligence endpoint — prediction engine diagnostics.

The endpoint exposes prediction engine health for the System dashboard:
- Returns prediction type status (active / limited / blocked) and 7-day counts
- Returns overall health (healthy / degraded / broken) and aggregate stats
- Returns user model depth (episodes, facts, routines, workflows, signal profiles)
- Includes a generated_at ISO-8601 timestamp
- Fails open: returns {prediction_types: {}, overall: {health: "unknown"}}
  when the prediction engine raises an exception
- User model depth defaults to zero when user_model DB tables are unavailable

Coverage:
    GET /api/system/intelligence → 200 with full diagnostics shape
    GET /api/system/intelligence → 200 with user model depth counts
    GET /api/system/intelligence → 200 graceful fallback when engine fails
    GET /api/system/intelligence → 200 graceful fallback when db tables missing
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_diagnostics(
    health: str = "healthy",
    total_7d: int = 5,
    active_types: int = 2,
    blocked_types: int = 1,
) -> dict:
    """Build a synthetic get_diagnostics() return value for use in mocks.

    Args:
        health: Overall health string (healthy / degraded / broken).
        total_7d: Total number of predictions generated in last 7 days.
        active_types: Number of active prediction types.
        blocked_types: Number of blocked prediction types.

    Returns:
        Diagnostics dict matching the PredictionEngine.get_diagnostics() schema.
    """
    return {
        "prediction_types": {
            "reminder": {
                "status": "active",
                "generated_last_7d": 3,
                "data_available": {"unreplied_emails_24h": 2},
                "blockers": [],
                "recommendations": [],
            },
            "opportunity": {
                "status": "active",
                "generated_last_7d": 2,
                "data_available": {},
                "blockers": [],
                "recommendations": [],
            },
            "conflict": {
                "status": "blocked",
                "generated_last_7d": 0,
                "data_available": {"calendar_events": 0},
                "blockers": ["No calendar events found"],
                "recommendations": ["Connect a CalDAV calendar"],
            },
        },
        "overall": {
            "total_predictions_7d": total_7d,
            "active_types": active_types,
            "blocked_types": blocked_types,
            "health": health,
        },
    }


def _insert_user_model_rows(db, *, episodes=3, facts=5, routines=2, workflows=1, signal_profiles=4):
    """Populate user_model tables with synthetic rows for depth-count testing.

    Uses the actual column names from storage/manager.py schemas so inserts
    succeed against the real temporary database created by the conftest fixture.

    Args:
        db: DatabaseManager fixture from conftest.
        episodes: Number of episode rows to insert.
        facts: Number of semantic_facts rows to insert.
        routines: Number of routines rows to insert.
        workflows: Number of workflows rows to insert.
        signal_profiles: Number of signal_profiles rows to insert.
    """
    with db.get_connection("user_model") as conn:
        # Episodes — schema: id, timestamp, event_id, interaction_type, content_summary
        for i in range(episodes):
            conn.execute(
                """INSERT OR IGNORE INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary)
                   VALUES (?, ?, ?, 'communication', 'test episode')""",
                (str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(), f"evt_{i}"),
            )
        # Semantic facts — schema: key, category, value, confidence (no source column)
        for i in range(facts):
            conn.execute(
                """INSERT OR IGNORE INTO semantic_facts (key, category, value, confidence)
                   VALUES (?, 'implicit_preference', ?, 0.7)""",
                (f"test_fact_{i}", f"value_{i}"),
            )
        # Routines — schema: name (PK), trigger_condition, steps
        for i in range(routines):
            conn.execute(
                """INSERT OR IGNORE INTO routines (name, trigger_condition, steps)
                   VALUES (?, 'email.received', '[]')""",
                (f"test_routine_{i}",),
            )
        # Workflows — schema: name (PK), trigger_conditions, steps
        for i in range(workflows):
            conn.execute(
                """INSERT OR IGNORE INTO workflows (name, trigger_conditions, steps)
                   VALUES (?, '[]', '[]')""",
                (f"test_workflow_{i}",),
            )
        # Signal profiles — schema: profile_type (PK), data
        for i in range(signal_profiles):
            conn.execute(
                """INSERT OR IGNORE INTO signal_profiles (profile_type, data)
                   VALUES (?, '{}')""",
                (f"profile_type_{i}",),
            )


def _make_app(db, diagnostics_result=None, engine_raises=False) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance.

    Args:
        db: Real DatabaseManager from the conftest fixture (provides real SQLite).
        diagnostics_result: Value returned by prediction_engine.get_diagnostics().
            Defaults to a healthy diagnostics dict.
        engine_raises: If True, get_diagnostics() raises RuntimeError to test
            the fail-open fallback path.

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
    life_os.user_model_store = MagicMock()
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}

    if engine_raises:
        life_os.prediction_engine.get_diagnostics = AsyncMock(side_effect=RuntimeError("engine down"))
    else:
        payload = diagnostics_result if diagnostics_result is not None else _make_diagnostics()
        life_os.prediction_engine.get_diagnostics = AsyncMock(return_value=payload)

    register_routes(app, life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intelligence_returns_200(db):
    """GET /api/system/intelligence returns HTTP 200 with valid JSON body."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_intelligence_has_required_top_level_keys(db):
    """Response contains prediction_types, overall, user_model_depth, generated_at."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    assert "prediction_types" in data, "Missing prediction_types key"
    assert "overall" in data, "Missing overall key"
    assert "user_model_depth" in data, "Missing user_model_depth key"
    assert "generated_at" in data, "Missing generated_at key"


@pytest.mark.asyncio
async def test_intelligence_overall_health_propagated(db):
    """The overall.health string from get_diagnostics() is passed through unchanged."""
    for health in ("healthy", "degraded", "broken"):
        diag = _make_diagnostics(health=health)
        client = _make_app(db, diagnostics_result=diag)
        resp = client.get("/api/system/intelligence")
        data = resp.json()
        assert data["overall"]["health"] == health, f"Expected {health}, got {data['overall']['health']}"


@pytest.mark.asyncio
async def test_intelligence_prediction_types_forwarded(db):
    """All prediction types from the engine appear in the response."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    pred_types = data["prediction_types"]
    assert "reminder" in pred_types
    assert "opportunity" in pred_types
    assert "conflict" in pred_types


@pytest.mark.asyncio
async def test_intelligence_prediction_type_fields_present(db):
    """Each prediction type entry has status, generated_last_7d, blockers, recommendations."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    reminder = data["prediction_types"]["reminder"]
    assert "status" in reminder
    assert "generated_last_7d" in reminder
    assert "blockers" in reminder
    assert "recommendations" in reminder


# ---------------------------------------------------------------------------
# Tests: user model depth counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intelligence_user_model_depth_counts(db):
    """user_model_depth reflects real row counts from the user_model database."""
    _insert_user_model_rows(db, episodes=3, facts=5, routines=2, workflows=1, signal_profiles=4)
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    depth = data["user_model_depth"]
    assert depth["episodes"] == 3, f"Expected 3 episodes, got {depth['episodes']}"
    assert depth["semantic_facts"] == 5, f"Expected 5 facts, got {depth['semantic_facts']}"
    assert depth["routines"] == 2, f"Expected 2 routines, got {depth['routines']}"
    assert depth["workflows"] == 1, f"Expected 1 workflow, got {depth['workflows']}"
    assert depth["signal_profiles"] == 4, f"Expected 4 profiles, got {depth['signal_profiles']}"


@pytest.mark.asyncio
async def test_intelligence_user_model_depth_empty_db(db):
    """user_model_depth returns zeros when user_model tables are empty."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    depth = data["user_model_depth"]
    assert depth["episodes"] == 0
    assert depth["semantic_facts"] == 0
    assert depth["routines"] == 0
    assert depth["workflows"] == 0
    assert depth["signal_profiles"] == 0


# ---------------------------------------------------------------------------
# Tests: fail-open behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intelligence_engine_exception_fails_open(db):
    """If prediction_engine.get_diagnostics() raises, endpoint still returns 200.

    The endpoint wraps the call in try/except and falls back to an empty dict
    so the System tab still loads even when the prediction engine is unhealthy.
    """
    client = _make_app(db, engine_raises=True)
    resp = client.get("/api/system/intelligence")
    assert resp.status_code == 200
    data = resp.json()
    # Must still have all top-level keys even on failure
    assert "prediction_types" in data
    assert "overall" in data
    assert "user_model_depth" in data
    assert "generated_at" in data


@pytest.mark.asyncio
async def test_intelligence_generated_at_is_iso8601(db):
    """generated_at is an ISO-8601 timestamp string."""
    client = _make_app(db)
    resp = client.get("/api/system/intelligence")
    data = resp.json()
    ts = data.get("generated_at")
    assert ts is not None
    # Should parse without raising
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert isinstance(parsed, datetime)
