"""
Tests for the /api/admin/semantic-facts/diagnostics endpoint.

Validates the response structure, profile readiness reporting, health
classification, and graceful error handling for the semantic fact
inference diagnostics API.
"""

from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXPECTED_PROFILES = [
    "linguistic", "linguistic_inbound", "relationships", "topics",
    "cadence", "mood_signals", "temporal", "spatial", "decision",
]


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for semantic-fact diagnostics tests."""
    life_os = Mock()

    # Database mock — returns 0 rows by default
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []
    mock_conn = Mock()
    mock_conn.execute.return_value = mock_cursor
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / event store (required by create_web_app)
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

    # User model store — no signal profiles by default
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector / browser stubs
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

    # Semantic fact inferrer — no get_diagnostics by default
    life_os.semantic_fact_inferrer = Mock(spec=["run_all_inference"])

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Response structure tests
# ---------------------------------------------------------------------------


def test_diagnostics_endpoint_returns_200(client):
    """GET /api/admin/semantic-facts/diagnostics returns 200."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    assert response.status_code == 200


def test_diagnostics_has_expected_top_level_keys(client):
    """Response contains all expected top-level keys."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    for key in ("profile_status", "facts", "inferrer", "health", "health_reason", "generated_at"):
        assert key in data, f"Missing top-level key: {key}"


def test_diagnostics_profile_status_has_all_9_profiles(client):
    """profile_status must contain all 9 expected signal profile types."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    profile_status = response.json()["profile_status"]
    for ptype in EXPECTED_PROFILES:
        assert ptype in profile_status, f"Missing profile type: {ptype}"
    assert len(profile_status) == 9


def test_diagnostics_generated_at_is_iso_timestamp(client):
    """generated_at should be a valid ISO timestamp string."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    generated_at = response.json()["generated_at"]
    assert isinstance(generated_at, str)
    # Basic check: contains 'T' separator and ends with timezone info
    assert "T" in generated_at


# ---------------------------------------------------------------------------
# Health classification tests
# ---------------------------------------------------------------------------


def test_health_blocked_when_no_profiles(client):
    """Health is 'blocked' when no signal profiles exist (default mock)."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    assert data["health"] == "blocked"
    assert "No signal profiles" in data["health_reason"]


def test_health_degraded_when_profiles_ready_but_no_facts(mock_life_os, client):
    """Health is 'degraded' when profiles meet thresholds but 0 facts exist."""
    # Make all profiles return with enough samples
    mock_life_os.user_model_store.get_signal_profile = Mock(
        return_value={"samples_count": 10, "updated_at": "2026-01-01T00:00:00"}
    )
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    assert data["health"] == "degraded"
    assert "0 facts" in data["health_reason"]


def test_health_ok_when_profiles_and_facts_exist(mock_life_os, client):
    """Health is 'ok' when profiles meet thresholds and facts exist."""
    # Make profiles return with samples
    mock_life_os.user_model_store.get_signal_profile = Mock(
        return_value={"samples_count": 10, "updated_at": "2026-01-01T00:00:00"}
    )
    # Make DB return fact counts
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (5,)
    mock_cursor.fetchall.return_value = [
        {"category": "preference", "cnt": 3},
        {"category": "expertise", "cnt": 2},
    ]
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    assert data["health"] == "ok"
    assert data["facts"]["total"] == 5


# ---------------------------------------------------------------------------
# Profile detail tests
# ---------------------------------------------------------------------------


def test_profile_not_exists_shows_threshold(client):
    """Missing profiles still report their required threshold."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    profile = response.json()["profile_status"]["relationships"]
    assert profile["exists"] is False
    assert profile["samples"] == 0
    assert profile["meets_threshold"] is False
    assert profile["threshold"] == 5


def test_profile_exists_but_below_threshold(mock_life_os, client):
    """A profile with samples below threshold reports meets_threshold=False."""
    def side_effect(ptype):
        if ptype == "relationships":
            return {"samples_count": 2, "updated_at": "2026-01-01"}
        return None
    mock_life_os.user_model_store.get_signal_profile = Mock(side_effect=side_effect)

    response = client.get("/api/admin/semantic-facts/diagnostics")
    profile = response.json()["profile_status"]["relationships"]
    assert profile["exists"] is True
    assert profile["samples"] == 2
    assert profile["meets_threshold"] is False


def test_profile_meets_threshold(mock_life_os, client):
    """A profile with enough samples reports meets_threshold=True."""
    def side_effect(ptype):
        if ptype == "topics":
            return {"samples_count": 5, "updated_at": "2026-01-01"}
        return None
    mock_life_os.user_model_store.get_signal_profile = Mock(side_effect=side_effect)

    response = client.get("/api/admin/semantic-facts/diagnostics")
    profile = response.json()["profile_status"]["topics"]
    assert profile["exists"] is True
    assert profile["meets_threshold"] is True


# ---------------------------------------------------------------------------
# Inferrer diagnostics
# ---------------------------------------------------------------------------


def test_inferrer_note_when_no_get_diagnostics(client):
    """When semantic_fact_inferrer lacks get_diagnostics(), report a note."""
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    assert "note" in data["inferrer"]


def test_inferrer_returns_diagnostics_when_available(mock_life_os, client):
    """When get_diagnostics() exists, its return value is included."""
    mock_life_os.semantic_fact_inferrer = Mock()
    mock_life_os.semantic_fact_inferrer.get_diagnostics = Mock(
        return_value={"last_run": "2026-01-01T00:00:00", "runs": 5}
    )
    response = client.get("/api/admin/semantic-facts/diagnostics")
    data = response.json()
    assert data["inferrer"]["last_run"] == "2026-01-01T00:00:00"
    assert data["inferrer"]["runs"] == 5


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_diagnostics_handles_db_error(mock_life_os, client):
    """Endpoint returns 200 even if the DB query fails."""
    mock_life_os.db.get_connection = Mock(
        side_effect=RuntimeError("DB unavailable")
    )
    response = client.get("/api/admin/semantic-facts/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data["facts"]


def test_diagnostics_handles_profile_error(mock_life_os, client):
    """Endpoint returns 200 even if get_signal_profile raises."""
    mock_life_os.user_model_store.get_signal_profile = Mock(
        side_effect=RuntimeError("Profile DB error")
    )
    response = client.get("/api/admin/semantic-facts/diagnostics")
    assert response.status_code == 200
    data = response.json()
    # Each profile should have an error entry
    for ptype in EXPECTED_PROFILES:
        assert "error" in data["profile_status"][ptype]


# ---------------------------------------------------------------------------
# Infer endpoint enhancement tests
# ---------------------------------------------------------------------------


def test_infer_endpoint_returns_facts_before_and_created(mock_life_os, client):
    """POST /api/admin/semantic-facts/infer returns facts_before and facts_created."""
    # First call (before): 3 facts; second call (after): 5 facts
    mock_life_os.user_model_store.get_semantic_facts = Mock(
        side_effect=[
            [{"key": f"f{i}"} for i in range(3)],  # before
            [{"key": f"f{i}", "category": "pref"} for i in range(5)],  # after
        ]
    )
    mock_life_os.semantic_fact_inferrer.run_all_inference = Mock()

    response = client.post("/api/admin/semantic-facts/infer")
    assert response.status_code == 200
    data = response.json()
    assert data["facts_before"] == 3
    assert data["facts_created"] == 2
    assert data["total_facts"] == 5
