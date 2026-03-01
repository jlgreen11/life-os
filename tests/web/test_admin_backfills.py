"""
Tests: /api/admin/backfills/status and /api/admin/backfills/trigger endpoints.

These admin endpoints expose on-demand signal profile backfill control so that
operators can rebuild empty profiles (e.g., after a DB migration wipes
signal_profiles) without requiring a full system restart.

Coverage:
    GET /api/admin/backfills/status → 200 with all six profile names present
    GET /api/admin/backfills/status → "needs_backfill" when profiles are empty
    GET /api/admin/backfills/status → "ok" when all profiles are populated
    GET /api/admin/backfills/status → populated=true only when samples_count >= 1
    POST /api/admin/backfills/trigger → 200 with status="started" immediately
    POST /api/admin/backfills/trigger → lists all four backfill names in response
    POST /api/admin/backfills/trigger → calls life_os backfill methods
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILE_NAMES = ["relationships", "temporal", "topics", "linguistic", "cadence", "mood_signals"]


def _make_app(
    db,
    profile_data: dict | None = None,
) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance.

    Args:
        db: Real DatabaseManager from the conftest fixture (provides real SQLite).
        profile_data: Mapping of profile_name → mock get_signal_profile return value.
            Profiles not listed return None (unpopulated).  Use None for the default
            "all profiles empty" case.

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

    # Wire backfill methods as no-op async callables (they are awaited in routes).
    life_os._backfill_relationship_profile_if_needed = AsyncMock()
    life_os._clean_relationship_profile_if_needed = AsyncMock()
    life_os._backfill_temporal_profile_if_needed = AsyncMock()
    life_os._backfill_topic_profile_if_needed = AsyncMock()
    life_os._backfill_linguistic_profile_if_needed = AsyncMock()

    # Configure get_signal_profile() to return either None or a mock profile dict.
    resolved = profile_data or {}

    def _get_signal_profile(name: str):
        """Return mock profile data for the given profile name.

        Returns None if the profile name is not in resolved (simulates an empty
        signal_profiles table), or returns the dict from resolved if present.
        """
        return resolved.get(name)

    life_os.user_model_store.get_signal_profile.side_effect = _get_signal_profile
    life_os.user_model_store.get_semantic_facts.return_value = []

    register_routes(app, life_os)
    return TestClient(app)


def _populated_profile(samples: int = 100) -> dict:
    """Build a mock signal profile dict with the given sample count.

    Args:
        samples: Number of samples to report in the profile.

    Returns:
        Dict mimicking the structure returned by UserModelStore.get_signal_profile().
    """
    return {
        "data": {},
        "samples_count": samples,
        "last_updated": "2026-03-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Tests: GET /api/admin/backfills/status
# ---------------------------------------------------------------------------


def test_backfill_status_returns_200(db):
    """GET /api/admin/backfills/status returns HTTP 200."""
    client = _make_app(db)
    res = client.get("/api/admin/backfills/status")
    assert res.status_code == 200


def test_backfill_status_includes_all_profiles(db):
    """Response includes all six expected profile names."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    assert "profiles" in data
    for name in _PROFILE_NAMES:
        assert name in data["profiles"], f"Missing profile: {name}"


def test_backfill_status_profile_shape(db):
    """Each profile entry has populated, samples_count, and last_updated."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    for name in _PROFILE_NAMES:
        profile = data["profiles"][name]
        assert "populated" in profile, f"{name} missing 'populated'"
        assert "samples_count" in profile, f"{name} missing 'samples_count'"
        assert "last_updated" in profile, f"{name} missing 'last_updated'"


def test_backfill_status_needs_backfill_when_empty(db):
    """Status is 'needs_backfill' when no profiles are populated."""
    client = _make_app(db)  # profile_data=None → all profiles return None
    data = client.get("/api/admin/backfills/status").json()

    assert data["status"] == "needs_backfill"
    for name in _PROFILE_NAMES:
        assert data["profiles"][name]["populated"] is False
        assert data["profiles"][name]["samples_count"] == 0
        assert data["profiles"][name]["last_updated"] is None


def test_backfill_status_ok_when_all_populated(db):
    """Status is 'ok' when all profiles report samples_count >= 1."""
    all_profiles = {name: _populated_profile(50) for name in _PROFILE_NAMES}
    client = _make_app(db, profile_data=all_profiles)
    data = client.get("/api/admin/backfills/status").json()

    assert data["status"] == "ok"
    for name in _PROFILE_NAMES:
        assert data["profiles"][name]["populated"] is True
        assert data["profiles"][name]["samples_count"] == 50


def test_backfill_status_partial_population(db):
    """Status is 'needs_backfill' when only some profiles are populated."""
    # Only relationship and temporal populated; the rest are empty.
    partial = {
        "relationships": _populated_profile(200),
        "temporal": _populated_profile(30),
    }
    client = _make_app(db, profile_data=partial)
    data = client.get("/api/admin/backfills/status").json()

    assert data["status"] == "needs_backfill"
    assert data["profiles"]["relationships"]["populated"] is True
    assert data["profiles"]["temporal"]["populated"] is True
    assert data["profiles"]["topics"]["populated"] is False
    assert data["profiles"]["linguistic"]["populated"] is False


def test_backfill_status_zero_samples_not_populated(db):
    """A profile with samples_count=0 is NOT considered populated."""
    # Simulate a profile that exists in the DB but has no data yet.
    empty_profile = {"data": {}, "samples_count": 0, "last_updated": None}
    client = _make_app(db, profile_data={"relationships": empty_profile})
    data = client.get("/api/admin/backfills/status").json()

    assert data["profiles"]["relationships"]["populated"] is False


def test_backfill_status_includes_generated_at(db):
    """Response includes an ISO-8601 generated_at timestamp."""
    client = _make_app(db)
    data = client.get("/api/admin/backfills/status").json()

    assert "generated_at" in data
    # Should parse as ISO-8601
    from datetime import datetime
    datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Tests: POST /api/admin/backfills/trigger
# ---------------------------------------------------------------------------


def test_trigger_backfills_returns_200(db):
    """POST /api/admin/backfills/trigger returns HTTP 200."""
    client = _make_app(db)
    res = client.post("/api/admin/backfills/trigger")
    assert res.status_code == 200


def test_trigger_backfills_returns_started(db):
    """Response has status='started' immediately (fire-and-forget semantics)."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert data["status"] == "started"


def test_trigger_backfills_lists_all_four(db):
    """Response lists all four backfill types in the backfills array."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert "backfills" in data
    for expected in ["relationship", "temporal", "topic", "linguistic"]:
        assert expected in data["backfills"], f"Missing backfill type: {expected}"


def test_trigger_backfills_includes_message(db):
    """Response includes a human-readable message field."""
    client = _make_app(db)
    data = client.post("/api/admin/backfills/trigger").json()

    assert "message" in data
    assert len(data["message"]) > 0
