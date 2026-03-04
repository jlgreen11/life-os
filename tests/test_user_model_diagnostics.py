"""
Tests for GET /api/diagnostics/user-model endpoint.

Verifies the aggregated user model diagnostics endpoint correctly reports
DB counts, service diagnostics, signal profile coverage, and overall health.
"""

from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conn(query_results=None):
    """Create a mock SQLite connection with configurable query results.

    Args:
        query_results: dict mapping SQL fragments to (value,) tuples.
    """
    query_results = query_results or {}

    def _execute(sql, params=None):
        mock_cursor = Mock()
        for fragment, result in query_results.items():
            if fragment in sql:
                mock_cursor.fetchone = Mock(return_value=result)
                return mock_cursor
        # Default: return (0,)
        mock_cursor.fetchone = Mock(return_value=(0,))
        return mock_cursor

    conn = Mock()
    conn.execute = _execute
    return conn


def _make_life_os(
    signal_profiles=None,
    user_model_conn=None,
    service_diagnostics=None,
):
    """Build a mock life_os with controllable DB connections and services.

    Args:
        signal_profiles: dict mapping profile_type to profile dict (or None for missing).
        user_model_conn: optional pre-built mock connection for user_model.db.
        service_diagnostics: dict mapping service name to diagnostics return value.
    """
    life_os = Mock()
    life_os.config = {}
    service_diagnostics = service_diagnostics or {}

    # --- user_model_store.get_signal_profile ---
    signal_profiles = signal_profiles or {}

    def _get_signal_profile(ptype):
        return signal_profiles.get(ptype)

    life_os.user_model_store = Mock()
    life_os.user_model_store.get_signal_profile = Mock(side_effect=_get_signal_profile)

    # --- db.get_connection (context manager) ---
    conns = {
        "user_model": user_model_conn or _make_mock_conn(),
    }

    @contextmanager
    def _get_connection(db_name):
        if db_name in conns:
            yield conns[db_name]
        else:
            yield _make_mock_conn()

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # --- event_store ---
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=42)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-123")
    life_os.event_store.get_event_flow_stats = Mock(return_value={
        "sources": {}, "stale_sources": [], "total_24h": 0, "events_per_hour": 0.0,
    })

    # --- event_bus ---
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()

    # --- vector_store ---
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0})
    life_os.vector_store.search = Mock(return_value=[])

    # --- Services with optional get_diagnostics() ---
    def _make_service(name):
        svc = Mock()
        if name in service_diagnostics:
            svc.get_diagnostics = Mock(return_value=service_diagnostics[name])
        else:
            # Remove get_diagnostics so hasattr returns False
            del svc.get_diagnostics
        return svc

    life_os.signal_extractor = _make_service("signal_extractor")
    life_os.signal_extractor.get_user_summary = Mock(return_value={})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.3, social_battery=0.5,
        cognitive_load=0.3, emotional_valence=0.5, confidence=0.5, trend="stable",
    ))
    life_os.prediction_engine = _make_service("prediction_engine")
    life_os.notification_manager = _make_service("notification_manager")
    life_os.notification_manager.get_stats = Mock(return_value={})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.routine_detector = _make_service("routine_detector")
    life_os.workflow_detector = _make_service("workflow_detector")
    life_os.semantic_fact_inferrer = _make_service("semantic_fact_inferrer")

    # --- Other stubs needed by route registration ---
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={})
    life_os.ai_engine = Mock()
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})

    return life_os


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_diagnostics_endpoint_returns_200():
    """GET /api/diagnostics/user-model returns 200 with JSON body."""
    life_os = _make_life_os()
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "health" in data
    assert "issues" in data


def test_diagnostics_includes_db_counts():
    """Response includes db_counts with expected table keys."""
    conn = _make_mock_conn({
        "episodes": (150,),
        "semantic_facts": (10,),
        "routines": (3,),
        "predictions": (50,),
    })
    life_os = _make_life_os(user_model_conn=conn)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    data = resp.json()

    assert "db_counts" in data
    db_counts = data["db_counts"]
    assert db_counts["episodes"] == 150
    assert db_counts["semantic_facts"] == 10
    assert db_counts["routines"] == 3
    assert db_counts["predictions"] == 50


def test_diagnostics_includes_signal_profiles():
    """Response includes signal_profiles with present/missing lists."""
    profiles = {
        "linguistic": {"samples_count": 25, "updated_at": "2026-03-01T12:00:00Z"},
        "topics": {"samples_count": 100, "updated_at": "2026-03-01T10:00:00Z"},
    }
    life_os = _make_life_os(signal_profiles=profiles)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    data = resp.json()

    assert "signal_profiles" in data
    sp = data["signal_profiles"]
    assert "present" in sp
    assert "missing" in sp
    # Two profiles are present
    present_names = [p["name"] for p in sp["present"]]
    assert "linguistic" in present_names
    assert "topics" in present_names
    # The other 7 are missing
    assert len(sp["missing"]) == 7


def test_diagnostics_health_degraded_when_facts_zero():
    """When episodes > 100 but semantic_facts == 0, health is 'degraded'."""
    conn = _make_mock_conn({
        "episodes": (200,),
        "semantic_facts": (0,),
        "routines": (5,),
        "predictions": (10,),
    })
    life_os = _make_life_os(user_model_conn=conn)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    data = resp.json()

    assert data["health"] == "degraded"
    assert any("Semantic facts empty" in issue for issue in data["issues"])


def test_diagnostics_health_healthy():
    """When episodes and facts are populated, health is 'healthy'."""
    conn = _make_mock_conn({
        "episodes": (50,),
        "semantic_facts": (10,),
        "routines": (3,),
        "predictions": (20,),
    })
    # Populate enough signal profiles to avoid the 'missing > 3' issue
    all_profiles = {
        ptype: {"samples_count": 10}
        for ptype in [
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "topics", "temporal", "spatial", "decision",
        ]
    }
    life_os = _make_life_os(user_model_conn=conn, signal_profiles=all_profiles)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    data = resp.json()

    assert data["health"] == "healthy"
    assert data["issues"] == []


def test_diagnostics_includes_service_diagnostics():
    """When services have get_diagnostics(), their output is included."""
    service_diags = {
        "prediction_engine": {"generators": 5, "active": 3},
        "notification_manager": {"total_sent": 42},
    }
    life_os = _make_life_os(service_diagnostics=service_diags)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    data = resp.json()

    assert data["prediction_engine"]["generators"] == 5
    assert data["notification_manager"]["total_sent"] == 42


def test_diagnostics_handles_missing_diagnostics_method():
    """Services without get_diagnostics() are silently skipped."""
    # No service_diagnostics → all services lack get_diagnostics
    life_os = _make_life_os(service_diagnostics={})
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/user-model")
    assert resp.status_code == 200
    data = resp.json()
    # Services without get_diagnostics should not appear in the response
    # (only db_counts, signal_profiles, health, issues are guaranteed)
    assert "db_counts" in data
    assert "health" in data
