"""
Tests: /api/system/sources endpoint — per-source event stats and staleness detection.

The endpoint aggregates the events table by source to surface data-freshness
information for the System health dashboard.  Key behaviors:

- Returns a list of sources with last_event, total_events, events_24h, events_7d
- Marks a source as stale if it hasn't emitted events within the threshold
  (6h for external connectors, 24h for internal pipeline sources)
- Provides stale_count and active_count at the top level for quick status bar display
- Returns an empty sources list (not 500) when the events table is empty
- Handles sources that have events 24h ago but not 7d (edge-case window math)
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


def _insert_event(db, source: str, hours_ago: float, event_type: str = "email.received") -> str:
    """Insert a synthetic event with a controlled timestamp and return its id."""
    event_id = str(uuid.uuid4())
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, 'normal', '{}', '{}')""",
            (event_id, event_type, source, ts),
        )
    return event_id


def _make_app(db) -> TestClient:
    """Create a minimal FastAPI test client with a mocked LifeOS instance."""
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db
    # event_store must delegate to the real EventStore backed by the test db
    from storage.event_store import EventStore

    life_os.event_store = EventStore(db)
    # Stub out unrelated services to prevent AttributeError during route setup
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.user_model_store = MagicMock()
    life_os.connectors = []  # no live connectors in tests
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    register_routes(app, life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_events_returns_empty_sources(db):
    """/api/system/sources returns empty list when no events exist."""
    client = _make_app(db)
    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data
    assert data["sources"] == []
    assert data["stale_count"] == 0
    assert data["active_count"] == 0
    assert "generated_at" in data


@pytest.mark.asyncio
async def test_recent_source_not_stale(db):
    """A source with events in the last hour should NOT be flagged stale."""
    _insert_event(db, "google", hours_ago=0.5)  # 30 min ago
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()
    sources = {s["source"]: s for s in data["sources"]}
    assert "google" in sources
    assert sources["google"]["stale"] is False
    assert data["stale_count"] == 0
    assert data["active_count"] == 1


@pytest.mark.asyncio
async def test_old_external_source_is_stale(db):
    """An external connector with no events for 8+ hours should be flagged stale.

    External sources (e.g. "google") have a 6h staleness threshold — they are
    expected to sync at least hourly, so 8 hours of silence is abnormal.
    """
    _insert_event(db, "google", hours_ago=8)  # 8 hours ago
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()
    sources = {s["source"]: s for s in data["sources"]}
    assert "google" in sources
    assert sources["google"]["stale"] is True
    assert data["stale_count"] == 1


@pytest.mark.asyncio
async def test_internal_source_stale_threshold_is_24h(db):
    """Internal pipeline sources (user_model_store, rules_engine) use a 24h threshold.

    They don't self-initiate; they only run when external events arrive, so
    going silent for a few hours is expected (e.g. overnight with no new emails).
    A 10-hour gap should NOT be stale for internal sources.
    """
    _insert_event(db, "user_model_store", hours_ago=10)
    _insert_event(db, "rules_engine", hours_ago=10)
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()
    sources = {s["source"]: s for s in data["sources"]}

    assert sources["user_model_store"]["stale"] is False, "user_model_store at 10h should not be stale (24h threshold)"
    assert sources["rules_engine"]["stale"] is False, "rules_engine at 10h should not be stale (24h threshold)"
    assert data["stale_count"] == 0


@pytest.mark.asyncio
async def test_event_counts_by_window(db):
    """events_24h and events_7d counts reflect correct time windows."""
    # 3 events in the last 24h
    for _ in range(3):
        _insert_event(db, "google", hours_ago=12)  # within 24h
    # 2 events between 2d and 7d ago
    for _ in range(2):
        _insert_event(db, "google", hours_ago=50)  # within 7d but not 24h
    # 1 old event (>7d) — should not count in either window
    _insert_event(db, "google", hours_ago=200)

    client = _make_app(db)
    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()

    sources = {s["source"]: s for s in data["sources"]}
    g = sources["google"]
    assert g["events_24h"] == 3, f"Expected 3 events in 24h, got {g['events_24h']}"
    assert g["events_7d"] == 5, f"Expected 5 events in 7d, got {g['events_7d']}"
    assert g["total_events"] == 6, f"Expected 6 total events, got {g['total_events']}"


@pytest.mark.asyncio
async def test_multiple_sources_returned(db):
    """Multiple sources appear as separate entries sorted by last_event."""
    _insert_event(db, "google", hours_ago=1)
    _insert_event(db, "rules_engine", hours_ago=2)
    _insert_event(db, "task_manager", hours_ago=3)
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()
    source_names = [s["source"] for s in data["sources"]]
    assert "google" in source_names
    assert "rules_engine" in source_names
    assert "task_manager" in source_names
    # Should be sorted newest-first
    assert source_names.index("google") < source_names.index("rules_engine")


@pytest.mark.asyncio
async def test_hours_since_field_present(db):
    """Each source entry includes a numeric hours_since field."""
    _insert_event(db, "google", hours_ago=3)
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()

    sources = {s["source"]: s for s in data["sources"]}
    assert "hours_since" in sources["google"]
    # hours_since should be approximately 3 (allow ±0.2h for test execution time)
    h = sources["google"]["hours_since"]
    assert h is not None
    assert 2.5 <= h <= 3.5, f"Expected hours_since ~3, got {h}"


@pytest.mark.asyncio
async def test_stale_count_reflects_mixed_sources(db):
    """stale_count and active_count correctly reflect mixed stale/active state."""
    _insert_event(db, "google", hours_ago=10)          # stale (external, >6h)
    _insert_event(db, "user_model_store", hours_ago=1)  # active (internal, <24h)
    _insert_event(db, "rules_engine", hours_ago=1)      # active (internal, <24h)
    client = _make_app(db)

    resp = client.get("/api/system/sources")
    assert resp.status_code == 200
    data = resp.json()

    assert data["stale_count"] == 1
    assert data["active_count"] == 2
