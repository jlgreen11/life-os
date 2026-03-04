"""
Tests: /api/diagnostics/pipeline — completeness of profile types, user model counts,
and Layer 3 procedural memory recommendations.

Validates that:
- linguistic_inbound profile type is included in signal_profiles diagnostics
- workflows_count and communication_templates_count appear in user_model diagnostics
- Missing workflows triggers a specific recommendation
- Missing communication_templates triggers a specific recommendation
- overall_status reflects empty Layer 3 procedural memory as degraded
"""

from __future__ import annotations

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
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    from storage.event_store import EventStore

    life_os.event_store = EventStore(db)

    from storage.user_model_store import UserModelStore

    UserModelStore(db, event_bus=MagicMock())

    if signal_profiles is not None:
        def _get_profile(ptype):
            return signal_profiles.get(ptype)
        life_os.user_model_store.get_signal_profile = MagicMock(side_effect=_get_profile)
    else:
        life_os.user_model_store.get_signal_profile = MagicMock(return_value=None)

    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    register_routes(app, life_os)
    return TestClient(app)


def _all_profiles_present() -> dict:
    """Return a signal_profiles dict where every expected profile exists."""
    names = [
        "relationships", "temporal", "topics", "linguistic",
        "linguistic_inbound", "cadence", "mood_signals", "spatial", "decision",
    ]
    return {
        name: {"samples_count": 10, "updated_at": "2026-01-01T00:00:00"}
        for name in names
    }


def _insert_event(db, hours_ago: float) -> None:
    """Insert a synthetic event at a controlled time offset."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, 'normal', '{}', '{}')""",
            (str(uuid.uuid4()), "email.received", "google", ts),
        )


def _insert_prediction(db, hours_ago: float) -> None:
    """Insert a synthetic prediction at a controlled time offset."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, description, confidence,
               confidence_gate, created_at)
               VALUES (?, 'NEED', 'test', 0.5, 'SUGGEST', ?)""",
            (str(uuid.uuid4()), ts),
        )


def _populate_base_data(db) -> None:
    """Insert the baseline data needed for a non-broken pipeline."""
    _insert_event(db, hours_ago=1)
    _insert_prediction(db, hours_ago=1)
    # Populate core user model tables (episodes, facts, routines)
    ts = datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary)
               VALUES (?, ?, ?, 'message', 'test')""",
            (str(uuid.uuid4()), ts, str(uuid.uuid4())),
        )
        conn.execute(
            """INSERT INTO semantic_facts (key, category, value, confidence)
               VALUES (?, 'preference', 'test', 0.5)""",
            (f"key_{uuid.uuid4().hex[:8]}",),
        )
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps)
               VALUES (?, 'morning', '[]')""",
            (f"routine_{uuid.uuid4().hex[:8]}",),
        )
    # Insert notification so that section is clean
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, status, created_at)
               VALUES (?, 'test', 'body', 'normal', 'pending', ?)""",
            (str(uuid.uuid4()), ts),
        )


# ---------------------------------------------------------------------------
# Tests — linguistic_inbound profile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_linguistic_inbound_in_signal_profiles(db):
    """linguistic_inbound appears in signal_profiles when populated."""
    profiles = _all_profiles_present()
    client = _make_app(db, signal_profiles=profiles)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    sp = data["signal_profiles"]
    assert "linguistic_inbound" in sp, "linguistic_inbound should be in signal_profiles"
    assert sp["linguistic_inbound"]["exists"] is True
    assert sp["linguistic_inbound"]["samples_count"] == 10


@pytest.mark.asyncio
async def test_linguistic_inbound_missing_shows_not_exists(db):
    """linguistic_inbound shows exists=False when not populated."""
    # Only provide a subset of profiles — no linguistic_inbound
    partial = {
        "relationships": {"samples_count": 5, "updated_at": "2026-01-01T00:00:00"},
    }
    client = _make_app(db, signal_profiles=partial)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    sp = data["signal_profiles"]
    assert "linguistic_inbound" in sp
    assert sp["linguistic_inbound"]["exists"] is False


@pytest.mark.asyncio
async def test_expected_profiles_count_is_nine(db):
    """EXPECTED_PROFILES now contains 9 profile types including linguistic_inbound."""
    profiles = _all_profiles_present()
    client = _make_app(db, signal_profiles=profiles)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    sp = data["signal_profiles"]
    # All 9 profiles should be present as keys
    assert len(sp) == 9


# ---------------------------------------------------------------------------
# Tests — workflows_count and communication_templates_count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflows_count_in_user_model(db):
    """workflows_count appears in user_model diagnostics."""
    client = _make_app(db, signal_profiles=_all_profiles_present())

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    um = data["user_model"]
    assert "workflows_count" in um, "workflows_count should appear in user_model"
    assert isinstance(um["workflows_count"], int)
    assert um["workflows_count"] == 0  # No workflows inserted


@pytest.mark.asyncio
async def test_communication_templates_count_in_user_model(db):
    """communication_templates_count appears in user_model diagnostics."""
    client = _make_app(db, signal_profiles=_all_profiles_present())

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    um = data["user_model"]
    assert "communication_templates_count" in um, "communication_templates_count should appear in user_model"
    assert isinstance(um["communication_templates_count"], int)
    assert um["communication_templates_count"] == 0


@pytest.mark.asyncio
async def test_workflows_count_reflects_inserted_data(db):
    """workflows_count increases when workflows are inserted."""
    client = _make_app(db, signal_profiles=_all_profiles_present())

    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow_1",),
        )
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow_2",),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()
    assert data["user_model"]["workflows_count"] == 2


@pytest.mark.asyncio
async def test_templates_count_reflects_inserted_data(db):
    """communication_templates_count increases when templates are inserted."""
    client = _make_app(db, signal_profiles=_all_profiles_present())

    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO communication_templates (id, context) VALUES (?, 'general')",
            (str(uuid.uuid4()),),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()
    assert data["user_model"]["communication_templates_count"] == 1


# ---------------------------------------------------------------------------
# Tests — recommendations for empty workflows/templates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_workflows_triggers_recommendation(db):
    """Empty workflows table triggers a medium-severity recommendation."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    # Insert a communication_template so only workflows is empty
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO communication_templates (id, context) VALUES (?, 'general')",
            (str(uuid.uuid4()),),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    recs = data["recommendations"]
    wf_recs = [r for r in recs if "workflows" in r["message"]]
    assert len(wf_recs) == 1, f"Expected 1 workflow recommendation, got {len(wf_recs)}"
    assert wf_recs[0]["severity"] == "medium"
    assert wf_recs[0]["area"] == "user_model"
    assert "workflow detection" in wf_recs[0]["action"] or "routine_detector" in wf_recs[0]["action"]


@pytest.mark.asyncio
async def test_empty_templates_triggers_recommendation(db):
    """Empty communication_templates table triggers a medium-severity recommendation."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    # Insert a workflow so only templates is empty
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow",),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    recs = data["recommendations"]
    tmpl_recs = [r for r in recs if "communication_templates" in r["message"]]
    assert len(tmpl_recs) == 1, f"Expected 1 template recommendation, got {len(tmpl_recs)}"
    assert tmpl_recs[0]["severity"] == "medium"
    assert tmpl_recs[0]["area"] == "user_model"
    assert "outbound messages" in tmpl_recs[0]["action"] or "message connectors" in tmpl_recs[0]["action"]


@pytest.mark.asyncio
async def test_populated_workflows_no_recommendation(db):
    """When workflows and templates are populated, no Layer 3 recommendations appear."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow",),
        )
        conn.execute(
            "INSERT INTO communication_templates (id, context) VALUES (?, 'general')",
            (str(uuid.uuid4()),),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    recs = data["recommendations"]
    layer3_recs = [r for r in recs if "workflows" in r.get("message", "") or "communication_templates" in r.get("message", "")]
    assert len(layer3_recs) == 0, f"Expected no Layer 3 recommendations, got {layer3_recs}"


# ---------------------------------------------------------------------------
# Tests — overall_status reflects Layer 3 health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overall_status_degraded_when_both_layer3_empty(db):
    """overall_status is degraded when both workflows and templates are empty."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    # No workflows or templates inserted

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()
    assert data["overall_status"] == "degraded"


@pytest.mark.asyncio
async def test_overall_status_healthy_when_layer3_populated(db):
    """overall_status is healthy when Layer 3 has at least some data."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow",),
        )
        conn.execute(
            "INSERT INTO communication_templates (id, context) VALUES (?, 'general')",
            (str(uuid.uuid4()),),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()
    assert data["overall_status"] == "healthy"


@pytest.mark.asyncio
async def test_overall_status_not_degraded_when_one_layer3_populated(db):
    """overall_status is not degraded due to Layer 3 if at least one table has data."""
    client = _make_app(db, signal_profiles=_all_profiles_present())
    _populate_base_data(db)
    # Only insert a workflow, templates stay empty
    with db.get_connection("user_model") as conn:
        conn.execute(
            "INSERT INTO workflows (name, steps, trigger_conditions) VALUES (?, '[]', '[]')",
            ("test_workflow",),
        )

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()
    # Not degraded because the condition is both empty, not either
    assert data["overall_status"] == "healthy"
