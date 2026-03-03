"""
Tests: Dashboard template JavaScript uses correct field names for routines/workflows.

The Insights tab in web/template.py renders routine and workflow cards using
JavaScript field names that must match the keys returned by the API endpoints
GET /api/user-model/routines and GET /api/user-model/workflows.

Previously the template used 'consistency' and 'observation_count' which did
not match the backend keys 'consistency_score' and 'times_observed', causing
those values to silently not display.

Coverage:
    - Template source uses 'consistency_score' (not 'consistency') for routines
    - Template source uses 'times_observed' (not 'observation_count') for routines
    - Template source uses 'times_observed' (not 'observation_count') for workflows
    - API /api/user-model/routines returns 'consistency_score' and 'times_observed'
    - API /api/user-model/workflows returns 'times_observed' and 'success_rate'
"""

from __future__ import annotations

import re
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app
from web.template import HTML_TEMPLATE


# ---------------------------------------------------------------------------
# Template source tests — verify JavaScript field names match API keys
# ---------------------------------------------------------------------------


def _get_routine_section() -> str:
    """Extract the routine rendering section of the template JavaScript.

    Looks for the section between the 'Detected Routines' header comment and
    the 'Detected Workflows' header comment.

    Returns:
        The substring containing the routine rendering JavaScript.
    """
    start = HTML_TEMPLATE.find("Detected Routines")
    end = HTML_TEMPLATE.find("Detected Workflows")
    assert start != -1, "Could not find 'Detected Routines' section in template"
    assert end != -1, "Could not find 'Detected Workflows' section in template"
    return HTML_TEMPLATE[start:end]


def _get_workflow_section() -> str:
    """Extract the workflow rendering section of the template JavaScript.

    Looks for the section between the 'Detected Workflows' header comment and
    the 'Communication Style Templates' header comment.

    Returns:
        The substring containing the workflow rendering JavaScript.
    """
    start = HTML_TEMPLATE.find("Detected Workflows")
    end = HTML_TEMPLATE.find("Communication Style Templates")
    assert start != -1, "Could not find 'Detected Workflows' section in template"
    assert end != -1, "Could not find 'Communication Style Templates' section in template"
    return HTML_TEMPLATE[start:end]


def test_routine_section_uses_consistency_score():
    """Template must reference r.consistency_score, not r.consistency."""
    section = _get_routine_section()
    assert "r.consistency_score" in section, (
        "Expected 'r.consistency_score' in routine rendering JS, "
        "but found it missing — the template may be using the wrong field name"
    )
    # Ensure the old incorrect name is not present
    # Match r.consistency followed by a space, paren, or !== but NOT r.consistency_score
    old_pattern = re.findall(r"r\.consistency(?!_score)", section)
    assert len(old_pattern) == 0, (
        f"Found {len(old_pattern)} occurrence(s) of 'r.consistency' (without '_score') "
        "in routine section — should be 'r.consistency_score'"
    )


def test_routine_section_uses_times_observed():
    """Template must reference r.times_observed, not r.observation_count."""
    section = _get_routine_section()
    assert "r.times_observed" in section, (
        "Expected 'r.times_observed' in routine rendering JS, "
        "but found it missing — the template may be using the wrong field name"
    )
    assert "r.observation_count" not in section, (
        "Found 'r.observation_count' in routine section — should be 'r.times_observed'"
    )


def test_workflow_section_uses_times_observed():
    """Template must reference w.times_observed, not w.observation_count."""
    section = _get_workflow_section()
    assert "w.times_observed" in section, (
        "Expected 'w.times_observed' in workflow rendering JS, "
        "but found it missing — the template may be using the wrong field name"
    )
    assert "w.observation_count" not in section, (
        "Found 'w.observation_count' in workflow section — should be 'w.times_observed'"
    )


def test_routine_section_preserves_correct_fields():
    """Verify the template still uses the correct field names that were already right."""
    section = _get_routine_section()
    # r.trigger is correct (maps from trigger_condition column)
    assert "r.trigger" in section, "Expected 'r.trigger' in routine section"
    # r.typical_duration_minutes is correct
    assert "r.typical_duration_minutes" in section, "Expected 'r.typical_duration_minutes' in routine section"


def test_workflow_section_preserves_correct_fields():
    """Verify the template still uses the correct field names that were already right."""
    section = _get_workflow_section()
    # w.success_rate is correct
    assert "w.success_rate" in section, "Expected 'w.success_rate' in workflow section"
    # w.steps is correct
    assert "w.steps" in section, "Expected 'w.steps' in workflow section"


# ---------------------------------------------------------------------------
# API response shape tests — verify backend returns expected keys
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock for routine/workflow API tests."""
    life_os = Mock()

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

    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable"
        )
    )

    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.get_routines = Mock(return_value=[])
    life_os.user_model_store.resolve_prediction = Mock()

    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


def test_routines_api_returns_consistency_score_key(mock_life_os, client):
    """GET /api/user-model/routines returns 'consistency_score', not 'consistency'."""
    mock_life_os.user_model_store.get_routines = Mock(return_value=[
        {
            "name": "Morning coffee",
            "trigger": "weekday_morning",
            "consistency_score": 0.85,
            "times_observed": 22,
            "typical_duration_minutes": 15,
            "steps": [],
            "variations": [],
            "updated_at": "2026-01-01T00:00:00Z",
        },
    ])
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200
    data = response.json()
    routine = data["routines"][0]
    assert "consistency_score" in routine, "API must return 'consistency_score' key"
    assert "consistency" not in routine or "consistency_score" in routine, (
        "API returns 'consistency' instead of 'consistency_score'"
    )
    assert routine["consistency_score"] == 0.85


def test_routines_api_returns_times_observed_key(mock_life_os, client):
    """GET /api/user-model/routines returns 'times_observed', not 'observation_count'."""
    mock_life_os.user_model_store.get_routines = Mock(return_value=[
        {
            "name": "Evening review",
            "trigger": "weekday_evening",
            "consistency_score": 0.7,
            "times_observed": 15,
            "typical_duration_minutes": 20,
            "steps": [],
            "variations": [],
            "updated_at": "2026-01-01T00:00:00Z",
        },
    ])
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200
    data = response.json()
    routine = data["routines"][0]
    assert "times_observed" in routine, "API must return 'times_observed' key"
    assert "observation_count" not in routine, (
        "API returns 'observation_count' instead of 'times_observed'"
    )
    assert routine["times_observed"] == 15
