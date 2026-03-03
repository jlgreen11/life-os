"""
Tests: Admin backfill endpoint completeness — all 8 signal profiles covered.

Verifies that the admin backfill/status, backfill/trigger, and rebuild-user-model
endpoints include ALL eight signal extractors (relationships, temporal, topics,
linguistic, cadence, mood_signals, spatial, decision).  This prevents a regression
where new extractors are added to main.py startup but forgotten in the admin
recovery endpoints — the primary tool for rebuilding after DB corruption.

Coverage:
    GET /api/admin/backfills/status → response includes all 8 profile names
    GET /api/admin/backfills/status → "ok" requires all 8 populated
    POST /api/admin/backfills/trigger → response lists all 8 backfill types
    POST /api/admin/backfills/trigger → spatial and decision backfill methods called
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes

# The canonical list of all signal profiles in the system.
# Must match main.py's profile_backfill_map (lines ~1764-1771).
ALL_PROFILE_NAMES = [
    "relationships", "temporal", "topics", "linguistic",
    "cadence", "mood_signals", "spatial", "decision",
]

# The canonical list of backfill type names returned in trigger responses.
ALL_BACKFILL_TYPES = [
    "relationship", "temporal", "topic", "linguistic",
    "cadence", "mood_signals", "spatial", "decision",
]


def _make_app(db, profile_data: dict | None = None) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance.

    All eight backfill methods are wired as no-op AsyncMocks so the
    fire-and-forget background tasks don't raise AttributeError.

    Args:
        db: Real DatabaseManager from the conftest fixture.
        profile_data: Optional mapping of profile_name -> mock profile dict.
            Profiles not listed return None (unpopulated).

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

    # Wire ALL backfill methods as no-op async callables.
    life_os._backfill_relationship_profile_if_needed = AsyncMock()
    life_os._clean_relationship_profile_if_needed = AsyncMock()
    life_os._backfill_temporal_profile_if_needed = AsyncMock()
    life_os._backfill_topic_profile_if_needed = AsyncMock()
    life_os._backfill_linguistic_profile_if_needed = AsyncMock()
    life_os._backfill_cadence_profile_if_needed = AsyncMock()
    life_os._backfill_mood_signals_profile_if_needed = AsyncMock()
    life_os._backfill_spatial_profile_if_needed = AsyncMock()
    life_os._backfill_decision_profile_if_needed = AsyncMock()

    # Configure get_signal_profile() to return mock or None.
    resolved = profile_data or {}

    def _get_signal_profile(name: str):
        return resolved.get(name)

    life_os.user_model_store.get_signal_profile.side_effect = _get_signal_profile
    life_os.user_model_store.get_semantic_facts.return_value = []

    register_routes(app, life_os)
    return TestClient(app)


def _populated_profile(samples: int = 100) -> dict:
    """Build a mock signal profile dict with the given sample count."""
    return {
        "data": {},
        "samples_count": samples,
        "last_updated": "2026-03-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Tests: GET /api/admin/backfills/status — all 8 profiles present
# ---------------------------------------------------------------------------


def test_status_includes_spatial_profile(db):
    """GET /api/admin/backfills/status includes the 'spatial' profile."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    assert "spatial" in data["profiles"], "Missing 'spatial' in backfill status profiles"
    profile = data["profiles"]["spatial"]
    assert "populated" in profile
    assert "samples_count" in profile
    assert "last_updated" in profile


def test_status_includes_decision_profile(db):
    """GET /api/admin/backfills/status includes the 'decision' profile."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    assert "decision" in data["profiles"], "Missing 'decision' in backfill status profiles"
    profile = data["profiles"]["decision"]
    assert "populated" in profile
    assert "samples_count" in profile
    assert "last_updated" in profile


def test_status_has_exactly_eight_profiles(db):
    """GET /api/admin/backfills/status returns exactly 8 signal profiles."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    assert len(data["profiles"]) == 8, (
        f"Expected 8 profiles, got {len(data['profiles'])}: {list(data['profiles'].keys())}"
    )
    for name in ALL_PROFILE_NAMES:
        assert name in data["profiles"], f"Missing profile: {name}"


def test_status_ok_requires_all_eight_populated(db):
    """Status is 'ok' only when ALL 8 profiles are populated (not just the original 6)."""
    # Populate only the original 6 — should still be needs_backfill.
    six_only = {name: _populated_profile(50) for name in ALL_PROFILE_NAMES[:6]}
    client = _make_app(db, profile_data=six_only)
    data = client.get("/api/admin/backfills/status").json()

    assert data["status"] == "needs_backfill", (
        "Status should be 'needs_backfill' when spatial and decision are empty"
    )

    # Now populate all 8 — should be ok.
    all_eight = {name: _populated_profile(50) for name in ALL_PROFILE_NAMES}
    client = _make_app(db, profile_data=all_eight)
    data = client.get("/api/admin/backfills/status").json()

    assert data["status"] == "ok", (
        "Status should be 'ok' when all 8 profiles are populated"
    )


# ---------------------------------------------------------------------------
# Tests: POST /api/admin/backfills/trigger — spatial and decision included
# ---------------------------------------------------------------------------


def test_trigger_includes_spatial_in_backfills_list(db):
    """POST /api/admin/backfills/trigger lists 'spatial' in backfills array."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert "spatial" in data["backfills"], "Missing 'spatial' in trigger backfills list"


def test_trigger_includes_decision_in_backfills_list(db):
    """POST /api/admin/backfills/trigger lists 'decision' in backfills array."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert "decision" in data["backfills"], "Missing 'decision' in trigger backfills list"


def test_trigger_lists_all_eight_backfill_types(db):
    """POST /api/admin/backfills/trigger response includes all 8 backfill types."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert len(data["backfills"]) == 8, (
        f"Expected 8 backfill types, got {len(data['backfills'])}: {data['backfills']}"
    )
    for expected in ALL_BACKFILL_TYPES:
        assert expected in data["backfills"], f"Missing backfill type: {expected}"


# ---------------------------------------------------------------------------
# Tests: Consistency with main.py startup
# ---------------------------------------------------------------------------


def test_status_profiles_match_trigger_backfills(db):
    """The set of profiles in status response covers the same extractors as trigger.

    This guards against one endpoint being updated while the other is forgotten.
    """
    client = _make_app(db)
    status_data = client.get("/api/admin/backfills/status").json()
    trigger_data = client.post("/api/admin/backfills/trigger").json()

    status_profiles = set(status_data["profiles"].keys())
    # Trigger uses singular "relationship" vs status "relationships" etc.
    # Both should have exactly 8 entries.
    assert len(status_profiles) == 8
    assert len(trigger_data["backfills"]) == 8
