"""
Tests for EventStore.get_event_flow_stats() and its integration with /health.

Covers per-source counts, stale source detection, throughput calculations,
and the /health endpoint ``data_flow`` key.
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from storage.event_store import EventStore
from web.app import create_web_app


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_event(source: str, timestamp: str, event_type: str = "test.event") -> dict:
    """Build a minimal event dict suitable for EventStore.store_event()."""
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": source,
        "timestamp": timestamp,
        "priority": "normal",
        "payload": {},
        "metadata": {},
    }


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _hours_ago_iso(hours: int) -> str:
    """Return an ISO-8601 timestamp ``hours`` hours in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# -------------------------------------------------------------------
# Unit tests — EventStore.get_event_flow_stats
# -------------------------------------------------------------------


class TestGetEventFlowStats:
    """Tests for the get_event_flow_stats method on EventStore."""

    def test_empty_table_returns_sensible_defaults(self, event_store: EventStore):
        """An empty events table should return zeros and empty collections."""
        stats = event_store.get_event_flow_stats()

        assert stats["sources"] == {}
        assert stats["stale_sources"] == []
        assert stats["total_24h"] == 0
        assert stats["events_per_hour"] == 0.0

    def test_per_source_counts_last_24h(self, event_store: EventStore):
        """Events within the last 24 h are counted per-source."""
        now = _now_iso()
        event_store.store_event(_make_event("google", now))
        event_store.store_event(_make_event("google", now))
        event_store.store_event(_make_event("signal", now))

        stats = event_store.get_event_flow_stats()

        assert stats["sources"]["google"]["count_24h"] == 2
        assert stats["sources"]["signal"]["count_24h"] == 1
        assert stats["total_24h"] == 3

    def test_old_events_excluded_from_24h_count(self, event_store: EventStore):
        """Events older than 24 h should have count_24h == 0 but still appear in sources."""
        old_ts = _hours_ago_iso(48)
        event_store.store_event(_make_event("caldav", old_ts))

        stats = event_store.get_event_flow_stats()

        assert "caldav" in stats["sources"]
        assert stats["sources"]["caldav"]["count_24h"] == 0
        assert stats["total_24h"] == 0

    def test_stale_source_detection(self, event_store: EventStore):
        """A source whose last event exceeds the threshold is flagged stale."""
        stale_ts = _hours_ago_iso(12)
        fresh_ts = _now_iso()

        event_store.store_event(_make_event("stale_source", stale_ts))
        event_store.store_event(_make_event("fresh_source", fresh_ts))

        stats = event_store.get_event_flow_stats(stale_threshold_hours=6)

        assert "stale_source" in stats["stale_sources"]
        assert "fresh_source" not in stats["stale_sources"]

    def test_custom_stale_threshold(self, event_store: EventStore):
        """Staleness threshold is configurable."""
        ts = _hours_ago_iso(3)
        event_store.store_event(_make_event("src", ts))

        # With a 2-hour threshold the source is stale
        stats_strict = event_store.get_event_flow_stats(stale_threshold_hours=2)
        assert "src" in stats_strict["stale_sources"]

        # With a 6-hour threshold the same source is fresh
        stats_relaxed = event_store.get_event_flow_stats(stale_threshold_hours=6)
        assert "src" not in stats_relaxed["stale_sources"]

    def test_events_per_hour_calculation(self, event_store: EventStore):
        """events_per_hour should equal total_24h / 24, rounded to 1 decimal."""
        now = _now_iso()
        for _ in range(12):
            event_store.store_event(_make_event("test", now))

        stats = event_store.get_event_flow_stats()

        assert stats["total_24h"] == 12
        assert stats["events_per_hour"] == 0.5

    def test_mixed_fresh_and_stale_sources(self, event_store: EventStore):
        """Multiple sources with different freshness are reported correctly."""
        now = _now_iso()
        old = _hours_ago_iso(10)  # within 24h but stale at 6h threshold
        very_old = _hours_ago_iso(100)  # outside 24h window

        event_store.store_event(_make_event("alpha", now))
        event_store.store_event(_make_event("beta", old))
        event_store.store_event(_make_event("gamma", very_old))

        stats = event_store.get_event_flow_stats(stale_threshold_hours=6)

        assert stats["sources"]["alpha"]["count_24h"] == 1
        # beta is 10h old — still within 24h window so count_24h=1, but stale at 6h threshold
        assert stats["sources"]["beta"]["count_24h"] == 1
        assert stats["sources"]["gamma"]["count_24h"] == 0
        assert "alpha" not in stats["stale_sources"]
        assert "beta" in stats["stale_sources"]
        assert "gamma" in stats["stale_sources"]

    def test_stale_sources_sorted(self, event_store: EventStore):
        """stale_sources list should be sorted alphabetically."""
        old = _hours_ago_iso(100)
        event_store.store_event(_make_event("zebra", old))
        event_store.store_event(_make_event("alpha", old))
        event_store.store_event(_make_event("middle", old))

        stats = event_store.get_event_flow_stats(stale_threshold_hours=6)
        assert stats["stale_sources"] == ["alpha", "middle", "zebra"]


# -------------------------------------------------------------------
# Integration — /health endpoint includes data_flow
# -------------------------------------------------------------------


@pytest.fixture
def mock_life_os_with_flow_stats():
    """Minimal mock LifeOS with get_event_flow_stats wired up."""
    life_os = Mock()
    life_os.db = Mock()
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=42)
    life_os.event_store.get_event_flow_stats = Mock(return_value={
        "sources": {"google": {"count_24h": 10, "last_event": "2026-03-03T12:00:00+00:00"}},
        "stale_sources": [],
        "total_24h": 10,
        "events_per_hour": 0.4,
    })
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 50, "dimensions": 384})
    life_os.connectors = []
    return life_os


def test_health_endpoint_includes_data_flow(mock_life_os_with_flow_stats):
    """The /health response should contain a ``data_flow`` key with flow stats."""
    app = create_web_app(mock_life_os_with_flow_stats)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "data_flow" in data
    assert data["data_flow"]["total_24h"] == 10
    assert "google" in data["data_flow"]["sources"]


def test_health_data_flow_graceful_on_error(mock_life_os_with_flow_stats):
    """If get_event_flow_stats raises, /health should still return with data_flow=None."""
    mock_life_os_with_flow_stats.event_store.get_event_flow_stats = Mock(
        side_effect=RuntimeError("DB locked")
    )
    app = create_web_app(mock_life_os_with_flow_stats)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["data_flow"] is None
