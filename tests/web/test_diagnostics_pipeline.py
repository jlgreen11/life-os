"""
Tests: /api/diagnostics/pipeline endpoint — actionable recommendations.

The diagnostics endpoint checks signal profiles, user model tables, predictions,
notifications, and events, then produces an overall_status.  The recommendations
feature analyses these diagnostics and generates specific, actionable guidance
pointing to admin endpoints or pages that can resolve the issue.

Key behaviors:
- A healthy system returns an empty recommendations list
- Missing signal profiles produce a backfill recommendation
- Corrupted user_model tables produce a rebuild recommendation
- Zero predictions in 24h produce a pipeline-health recommendation
- Zero events in 24h produce a connector-check recommendation
- Multiple issues produce multiple independent recommendations
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(db, *, signal_profiles: dict | None = None) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance.

    Args:
        db: DatabaseManager fixture backed by temporary SQLite databases.
        signal_profiles: Optional dict mapping profile type to return value.
            When provided, ``get_signal_profile`` returns the value for
            each requested type.  When ``None``, every profile is returned
            as ``None`` (i.e. missing).
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    from storage.event_store import EventStore

    life_os.event_store = EventStore(db)

    # Configure signal profile lookups
    from storage.user_model_store import UserModelStore

    real_ums = UserModelStore(db, event_bus=MagicMock())

    if signal_profiles is not None:
        def _get_profile(ptype):
            return signal_profiles.get(ptype)
        life_os.user_model_store.get_signal_profile = MagicMock(side_effect=_get_profile)
    else:
        life_os.user_model_store.get_signal_profile = MagicMock(return_value=None)

    # Stubs for unrelated services
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    register_routes(app, life_os)
    return TestClient(app)


def _insert_event(db, hours_ago: float, source: str = "google") -> str:
    """Insert a synthetic event at a controlled time offset."""
    event_id = str(uuid.uuid4())
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, 'normal', '{}', '{}')""",
            (event_id, "email.received", source, ts),
        )
    return event_id


def _insert_prediction(db, hours_ago: float) -> None:
    """Insert a synthetic prediction at a controlled time offset."""
    pred_id = str(uuid.uuid4())
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, description, confidence,
               confidence_gate, created_at)
               VALUES (?, 'NEED', 'test', 0.5, 'SUGGEST', ?)""",
            (pred_id, ts),
        )


def _insert_notification(db, hours_ago: float) -> None:
    """Insert a synthetic notification at a controlled time offset."""
    notif_id = str(uuid.uuid4())
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, status, created_at)
               VALUES (?, 'test', 'test body', 'normal', 'pending', ?)""",
            (notif_id, ts),
        )


def _all_profiles_present() -> dict:
    """Return a signal_profiles dict where every expected profile exists."""
    names = [
        "relationships", "temporal", "topics", "linguistic",
        "cadence", "mood_signals", "spatial", "decision",
    ]
    return {
        name: {"samples_count": 10, "updated_at": "2026-01-01T00:00:00"}
        for name in names
    }


def _insert_episode(db) -> None:
    """Insert a minimal episode row so episodes_count > 0."""
    ts = datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary)
               VALUES (?, ?, ?, 'message', 'test episode')""",
            (str(uuid.uuid4()), ts, str(uuid.uuid4())),
        )


def _insert_semantic_fact(db) -> None:
    """Insert a minimal semantic fact so semantic_facts_count > 0."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO semantic_facts (key, category, value, confidence)
               VALUES (?, 'preference', 'test_value', 0.5)""",
            (f"test_key_{uuid.uuid4().hex[:8]}",),
        )


def _insert_routine(db) -> None:
    """Insert a minimal routine so routines_count > 0."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps)
               VALUES (?, 'morning', '[]')""",
            (f"routine_{uuid.uuid4().hex[:8]}",),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthy_system_no_recommendations(db):
    """When all sections are healthy, recommendations list is empty."""
    # Populate all signal profiles
    client = _make_app(db, signal_profiles=_all_profiles_present())

    # Seed recent events, predictions, notifications, and user model tables
    _insert_event(db, hours_ago=1)
    _insert_prediction(db, hours_ago=1)
    _insert_notification(db, hours_ago=1)
    _insert_episode(db)
    _insert_semantic_fact(db)
    _insert_routine(db)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    assert "recommendations" in data
    assert data["recommendations"] == []
    assert data["recommendations_count"] == 0
    assert data["overall_status"] == "healthy"


@pytest.mark.asyncio
async def test_missing_profiles_recommendation(db):
    """When signal profiles are missing, a backfill recommendation is included."""
    # Only 2 of 8 profiles present
    partial = {
        "relationships": {"samples_count": 5, "updated_at": "2026-01-01T00:00:00"},
        "temporal": {"samples_count": 5, "updated_at": "2026-01-01T00:00:00"},
    }
    client = _make_app(db, signal_profiles=partial)
    _insert_event(db, hours_ago=1)
    _insert_prediction(db, hours_ago=1)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    profile_recs = [r for r in recs if r["area"] == "signal_profiles"]
    assert len(profile_recs) == 1
    assert profile_recs[0]["severity"] == "high"
    assert "6 signal profile(s) missing" in profile_recs[0]["message"]
    assert "backfills" in profile_recs[0]["action"]


@pytest.mark.asyncio
async def test_corrupted_user_model_recommendation(db):
    """When user model queries return errors, a rebuild recommendation appears."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _insert_event(db, hours_ago=1)
    _insert_prediction(db, hours_ago=1)

    # Simulate a corrupt user_model.db by dropping episodes table
    with db.get_connection("user_model") as conn:
        conn.execute("DROP TABLE IF EXISTS episodes")

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    um_recs = [r for r in recs if r["area"] == "user_model" and r["severity"] == "critical"]
    assert len(um_recs) == 1
    assert "episodes_count" in um_recs[0]["message"]
    assert "rebuild" in um_recs[0]["action"]


@pytest.mark.asyncio
async def test_zero_predictions_recommendation(db):
    """When no predictions in 24h, appropriate recommendation appears."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _insert_event(db, hours_ago=1)
    # No predictions inserted

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    pred_recs = [r for r in recs if r["area"] == "predictions"]
    assert len(pred_recs) == 1
    assert pred_recs[0]["severity"] == "high"
    assert "No predictions" in pred_recs[0]["message"]
    assert "signal profiles" in pred_recs[0]["action"]


@pytest.mark.asyncio
async def test_no_recent_events_recommendation(db):
    """When events_pipeline.last_24h is 0, connector check recommendation appears."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    # Insert an old event (>24h ago) so total > 0 but last_24h == 0
    _insert_event(db, hours_ago=48)
    _insert_prediction(db, hours_ago=1)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    event_recs = [r for r in recs if r["area"] == "events_pipeline"]
    assert len(event_recs) == 1
    assert event_recs[0]["severity"] == "high"
    assert "No events received" in event_recs[0]["message"]
    assert "/admin" in event_recs[0]["action"]


@pytest.mark.asyncio
async def test_multiple_recommendations(db):
    """When multiple issues exist, all relevant recommendations are included."""
    # No signal profiles, no events, no predictions, no notifications
    client = _make_app(db)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    areas = {r["area"] for r in recs}

    # Should have recommendations for multiple areas
    assert "signal_profiles" in areas, "Expected signal_profiles recommendation"
    assert "predictions" in areas, "Expected predictions recommendation"
    assert "events_pipeline" in areas, "Expected events_pipeline recommendation"
    assert "notifications" in areas, "Expected notifications recommendation"
    assert data["recommendations_count"] == len(recs)
    assert data["recommendations_count"] >= 4


@pytest.mark.asyncio
async def test_empty_user_model_tables_recommendation(db):
    """When user model tables are empty (not corrupt), medium-severity recommendations appear."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _insert_event(db, hours_ago=1)
    _insert_prediction(db, hours_ago=1)
    _insert_notification(db, hours_ago=1)
    # Don't insert episodes, facts, or routines — tables are empty

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    um_recs = [r for r in recs if r["area"] == "user_model"]
    assert len(um_recs) == 3, f"Expected 3 user_model recommendations for empty tables, got {len(um_recs)}"
    assert all(r["severity"] == "medium" for r in um_recs)
    table_names = {r["message"].split(" table")[0] for r in um_recs}
    assert "episodes" in table_names
    assert "semantic_facts" in table_names
    assert "routines" in table_names


@pytest.mark.asyncio
async def test_recommendations_fields_structure(db):
    """Each recommendation has the required severity, area, message, and action fields."""
    client = _make_app(db)  # defaults produce multiple recommendations

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    recs = data["recommendations"]
    assert len(recs) > 0, "Expected at least one recommendation"
    for rec in recs:
        assert "severity" in rec, f"Missing severity in {rec}"
        assert "area" in rec, f"Missing area in {rec}"
        assert "message" in rec, f"Missing message in {rec}"
        assert "action" in rec, f"Missing action in {rec}"
        assert rec["severity"] in ("critical", "high", "medium", "low"), f"Bad severity: {rec['severity']}"
