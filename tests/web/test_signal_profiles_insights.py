"""
Tests for the /api/user-model/signal-profiles, /api/user-model/routines,
/api/user-model/workflows, and /api/insights/summary endpoints.

These are the backend data sources consumed by the enhanced Insights tab
(loadInsightsFeed in web/template.py) that surfaces signal profiles, routines,
and workflows alongside AI-generated behavioral insights.
"""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for signal-profile and insights tests."""
    life_os = Mock()

    # Database (used by some endpoints for raw queries)
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []
    mock_conn.execute.return_value = mock_cursor
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / store (required by create_web_app)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable"
        )
    )

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    # User model store — returns no profiles by default
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.get_routines = Mock(return_value=[])
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector stubs
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient against the FastAPI app with a mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/user-model/signal-profiles — response shape
# ---------------------------------------------------------------------------


def test_signal_profiles_endpoint_exists(client):
    """GET /api/user-model/signal-profiles returns 200."""
    response = client.get("/api/user-model/signal-profiles")
    assert response.status_code == 200


def test_signal_profiles_empty_state(client):
    """When no profiles have been recorded, returns empty profiles and types_with_data."""
    response = client.get("/api/user-model/signal-profiles")
    data = response.json()
    assert "profiles" in data
    assert "types_with_data" in data
    assert "generated_at" in data
    assert data["profiles"] == {}
    assert data["types_with_data"] == []


def test_signal_profiles_with_temporal_data(mock_life_os, client):
    """When the temporal profile exists, it is returned with data and samples_count."""
    now_iso = datetime.now(timezone.utc).isoformat()
    temporal_row = {
        "data": {"chronotype": "early_bird", "peak_hours": [8, 9, 10]},
        "samples_count": 500,
        "updated_at": now_iso,
    }
    # Return temporal row when queried, None for other types
    def _get_profile(ptype):
        return temporal_row if ptype == "temporal" else None
    mock_life_os.user_model_store.get_signal_profile = Mock(side_effect=_get_profile)

    response = client.get("/api/user-model/signal-profiles")
    data = response.json()
    assert "temporal" in data["profiles"]
    profile = data["profiles"]["temporal"]
    assert profile["samples_count"] == 500
    assert profile["data"]["chronotype"] == "early_bird"
    assert "temporal" in data["types_with_data"]


def test_signal_profiles_single_type_filter(mock_life_os, client):
    """?profile_type=temporal returns only the temporal profile."""
    row = {"data": {"chronotype": "night_owl"}, "samples_count": 100, "updated_at": "2026-01-01T00:00:00Z"}
    mock_life_os.user_model_store.get_signal_profile = Mock(return_value=row)

    response = client.get("/api/user-model/signal-profiles?profile_type=temporal")
    data = response.json()
    assert "temporal" in data["profiles"]
    # Should only contain the requested type
    assert list(data["profiles"].keys()) == ["temporal"]


def test_signal_profiles_unknown_type_returns_404(client):
    """Requesting an unknown profile type returns HTTP 404."""
    response = client.get("/api/user-model/signal-profiles?profile_type=nonexistent")
    assert response.status_code == 404


def test_signal_profiles_all_known_types_queried(mock_life_os, client):
    """All nine known profile types are queried when no filter is specified."""
    queried = []

    def _get_profile(ptype):
        queried.append(ptype)
        return None

    mock_life_os.user_model_store.get_signal_profile = Mock(side_effect=_get_profile)
    client.get("/api/user-model/signal-profiles")

    expected = {
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    }
    assert expected == set(queried)


def test_signal_profiles_broken_profile_does_not_abort(mock_life_os, client):
    """A failure fetching one profile type does not prevent other types from returning."""
    call_count = [0]

    def _get_profile(ptype):
        call_count[0] += 1
        if ptype == "temporal":
            raise RuntimeError("Simulated DB error")
        if ptype == "linguistic":
            return {"data": {"formality": 0.7}, "samples_count": 50, "updated_at": "2026-01-01T00:00:00Z"}
        return None

    mock_life_os.user_model_store.get_signal_profile = Mock(side_effect=_get_profile)
    response = client.get("/api/user-model/signal-profiles")
    # Should still return 200 (fail-open per architecture convention)
    assert response.status_code == 200
    data = response.json()
    # linguistic should be present; temporal should be absent (errored)
    assert "linguistic" in data["profiles"]
    assert "temporal" not in data["profiles"]


# ---------------------------------------------------------------------------
# /api/user-model/routines
# ---------------------------------------------------------------------------


def test_routines_endpoint_exists(client):
    """GET /api/user-model/routines returns 200."""
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200


def test_routines_empty_state(client):
    """When no routines have been detected, returns an empty list."""
    response = client.get("/api/user-model/routines")
    data = response.json()
    assert "routines" in data
    assert data["routines"] == []
    assert data["count"] == 0


def test_routines_returns_detected_routines(mock_life_os, client):
    """Detected routines are returned with name, trigger, and consistency_score."""
    # The routines store returns dicts with consistency_score and times_observed
    # (the route filters on these keys post-query).
    mock_life_os.user_model_store.get_routines = Mock(return_value=[
        {"name": "Morning standup", "trigger": "weekday_morning", "consistency_score": 0.85, "times_observed": 22},
        {"name": "Evening review", "trigger": "weekday_evening", "consistency_score": 0.7, "times_observed": 15},
    ])
    response = client.get("/api/user-model/routines")
    data = response.json()
    assert data["count"] == 2
    names = [r["name"] for r in data["routines"]]
    assert "Morning standup" in names
    assert "Evening review" in names


def test_routines_min_consistency_filter(mock_life_os, client):
    """?min_consistency filters out routines below the consistency threshold."""
    all_routines = [
        {"name": "High routine", "trigger": "weekday_morning", "consistency_score": 0.9, "times_observed": 30},
        {"name": "Low routine", "trigger": "weekday_evening", "consistency_score": 0.2, "times_observed": 5},
    ]

    def _get_routines(trigger=None):
        return all_routines

    mock_life_os.user_model_store.get_routines = Mock(side_effect=_get_routines)

    response = client.get("/api/user-model/routines?min_consistency=0.5")
    data = response.json()
    # min_consistency=0.5 should exclude the low routine
    names = [r["name"] for r in data["routines"]]
    assert "High routine" in names
    assert "Low routine" not in names


# ---------------------------------------------------------------------------
# /api/user-model/workflows
# ---------------------------------------------------------------------------


def test_workflows_endpoint_exists(client):
    """GET /api/user-model/workflows returns 200."""
    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200


def test_workflows_empty_state(client):
    """When no workflows have been detected, returns an empty list."""
    response = client.get("/api/user-model/workflows")
    data = response.json()
    assert "workflows" in data
    assert data["workflows"] == []
    assert data["count"] == 0
