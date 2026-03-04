"""
Tests for GET /api/diagnostics/pipeline endpoint.

Verifies the pipeline diagnostics endpoint correctly reports health
across all processing stages: signal profiles, user model, predictions,
notifications, and events.
"""

import json
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conn(query_results=None):
    """Create a mock SQLite connection with configurable query results.

    Args:
        query_results: dict mapping SQL fragments to return values.
            Each value should be a dict representing a sqlite3.Row.
    """
    query_results = query_results or {}

    def _execute(sql, params=None):
        mock_cursor = Mock()
        # Find a matching result by checking if any key is a substring of the SQL
        for fragment, result in query_results.items():
            if fragment in sql:
                mock_cursor.fetchone = Mock(return_value=result)
                mock_cursor.fetchall = Mock(return_value=[result] if result else [])
                return mock_cursor
        # Default: return a row with c=0 and ts=None
        default_row = {"c": 0, "ts": None}
        mock_cursor.fetchone = Mock(return_value=default_row)
        mock_cursor.fetchall = Mock(return_value=[])
        return mock_cursor

    conn = Mock()
    conn.execute = _execute
    return conn


def _make_life_os(signal_profiles=None, user_model_conn=None, state_conn=None, events_conn=None):
    """Build a mock life_os with controllable DB connections.

    Args:
        signal_profiles: dict mapping profile_type to profile dict (or None for missing).
        user_model_conn: optional pre-built mock connection for user_model.db.
        state_conn: optional pre-built mock connection for state.db.
        events_conn: optional pre-built mock connection for events.db.
    """
    life_os = Mock()
    life_os.config = {}

    # --- user_model_store.get_signal_profile ---
    signal_profiles = signal_profiles or {}

    def _get_signal_profile(ptype):
        return signal_profiles.get(ptype)

    life_os.user_model_store = Mock()
    life_os.user_model_store.get_signal_profile = Mock(side_effect=_get_signal_profile)

    # --- event_store ---
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=42)

    # --- db.get_connection (context manager) ---
    conns = {
        "user_model": user_model_conn or _make_mock_conn(),
        "state": state_conn or _make_mock_conn(),
        "events": events_conn or _make_mock_conn(),
    }

    @contextmanager
    def _get_connection(db_name):
        if db_name in conns:
            yield conns[db_name]
        else:
            yield _make_mock_conn()

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection

    # --- stubs for other services used by route registration ---
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0})
    life_os.vector_store.search = Mock(return_value=[])
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.3, social_battery=0.5,
        cognitive_load=0.3, emotional_valence=0.5, confidence=0.5, trend="stable"
    ))
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={})
    life_os.notification_manager.get_pending = Mock(return_value=[])
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

EXPECTED_SECTIONS = [
    "signal_profiles",
    "user_model",
    "predictions",
    "notifications",
    "events_pipeline",
    "overall_status",
]


def test_pipeline_diagnostics_returns_all_sections():
    """The endpoint response contains all six top-level keys."""
    life_os = _make_life_os()
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    for section in EXPECTED_SECTIONS:
        assert section in data, f"Missing section: {section}"


def test_pipeline_diagnostics_reports_empty_profiles():
    """With no signal profiles in DB, all 9 types show exists=False."""
    life_os = _make_life_os(signal_profiles={})
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    profiles = data["signal_profiles"]
    expected_types = [
        "relationships", "temporal", "topics", "linguistic",
        "linguistic_inbound", "cadence", "mood_signals", "spatial", "decision",
    ]
    for ptype in expected_types:
        assert ptype in profiles, f"Missing profile type: {ptype}"
        assert profiles[ptype]["exists"] is False
        assert profiles[ptype]["samples_count"] == 0
        assert profiles[ptype]["updated_at"] is None


def test_pipeline_diagnostics_reports_populated_profiles():
    """When a signal profile exists, it shows exists=True with correct samples_count."""
    profiles = {
        "linguistic": {
            "profile_type": "linguistic",
            "data": {"avg_length": 50},
            "samples_count": 25,
            "updated_at": "2026-03-01T12:00:00Z",
        },
        "topics": {
            "profile_type": "topics",
            "data": {"top_topics": ["work"]},
            "samples_count": 100,
            "updated_at": "2026-03-01T10:00:00Z",
        },
    }
    life_os = _make_life_os(signal_profiles=profiles)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    assert data["signal_profiles"]["linguistic"]["exists"] is True
    assert data["signal_profiles"]["linguistic"]["samples_count"] == 25
    assert data["signal_profiles"]["linguistic"]["updated_at"] == "2026-03-01T12:00:00Z"

    assert data["signal_profiles"]["topics"]["exists"] is True
    assert data["signal_profiles"]["topics"]["samples_count"] == 100

    # Unpopulated profiles should still be False
    assert data["signal_profiles"]["cadence"]["exists"] is False


def test_pipeline_diagnostics_handles_db_error():
    """When a DB query fails, the endpoint still returns 200 with error info."""
    life_os = _make_life_os()

    # Make user_model connection raise on execute
    broken_conn = Mock()
    broken_conn.execute = Mock(side_effect=Exception("disk I/O error"))

    @contextmanager
    def _get_connection(db_name):
        if db_name == "user_model":
            yield broken_conn
        else:
            yield _make_mock_conn()

    life_os.db.get_connection = _get_connection

    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    assert resp.status_code == 200
    data = resp.json()

    # user_model section should report per-query errors (the endpoint
    # isolates each query, so individual keys contain "error: ..." strings
    # rather than crashing the whole section).
    um = data["user_model"]
    for key in [
        "episodes_count", "semantic_facts_count", "routines_count",
        "mood_readings_count", "workflows_count", "communication_templates_count",
    ]:
        assert isinstance(um[key], str) and "error" in um[key], (
            f"Expected error string in user_model[{key!r}], got {um[key]!r}"
        )

    # predictions section should also report an error since it uses
    # the same broken user_model connection
    assert "error" in data["predictions"]

    # Other sections should still be present and functional
    assert "notifications" in data
    assert "events_pipeline" in data
    assert isinstance(data["notifications"], dict)
    assert isinstance(data["events_pipeline"], dict)

    # Overall status should be "error" when any section fails
    assert data["overall_status"] == "error"


def test_pipeline_diagnostics_overall_status_broken():
    """With 0 signal profiles, overall_status is 'broken'."""
    life_os = _make_life_os(signal_profiles={})
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    assert data["overall_status"] == "broken"


def test_pipeline_diagnostics_overall_status_healthy():
    """With all profiles, recent predictions, and Layer 3 data, overall_status is 'healthy'."""
    # Build all 9 profiles (including linguistic_inbound)
    all_types = [
        "relationships", "temporal", "topics", "linguistic",
        "linguistic_inbound", "cadence", "mood_signals", "spatial", "decision",
    ]
    profiles = {
        ptype: {
            "profile_type": ptype,
            "data": {},
            "samples_count": 10,
            "updated_at": "2026-03-01T12:00:00Z",
        }
        for ptype in all_types
    }

    # Make predictions and user_model queries return non-zero counts.
    # workflows and communication_templates must have non-zero counts
    # to avoid the "degraded" status from empty Layer 3 tables.
    pred_conn = _make_mock_conn({
        "COUNT(*) as c FROM predictions WHERE": {"c": 5},
        "COUNT(*) as c FROM predictions": {"c": 50},
        "MAX(created_at)": {"ts": "2026-03-01T12:00:00Z"},
        "COUNT(*) as c FROM workflows": {"c": 2},
        "COUNT(*) as c FROM communication_templates": {"c": 3},
    })

    life_os = _make_life_os(signal_profiles=profiles, user_model_conn=pred_conn)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    assert data["overall_status"] == "healthy"


def test_pipeline_diagnostics_overall_status_degraded():
    """With some profiles but 0 predictions in last 24h, status is 'degraded'."""
    # Only 3 out of 8 profiles present
    profiles = {
        ptype: {
            "profile_type": ptype,
            "data": {},
            "samples_count": 5,
            "updated_at": "2026-03-01T12:00:00Z",
        }
        for ptype in ["linguistic", "topics", "cadence"]
    }

    life_os = _make_life_os(signal_profiles=profiles)
    app = create_web_app(life_os)
    client = TestClient(app)

    resp = client.get("/api/diagnostics/pipeline")
    data = resp.json()

    assert data["overall_status"] == "degraded"
