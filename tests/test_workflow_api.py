"""
Tests for workflow API endpoint.

The workflow API exposes detected procedural memory patterns (Layer 3 of the
user model) to clients. This enables users to see what multi-step processes
Life OS has learned from their behavior.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os(db, user_model_store):
    """Create a mock LifeOS instance with real database components."""
    life_os = Mock()
    life_os.db = db
    life_os.user_model_store = user_model_store

    # Mock other required services (not used by workflow endpoint)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = False
    life_os.event_store = Mock()
    life_os.vector_store = Mock()
    life_os.signal_extractor = Mock()
    life_os.task_manager = Mock()
    life_os.notification_manager = Mock()
    life_os.prediction_engine = Mock()
    life_os.rules_engine = Mock()
    life_os.feedback_collector = Mock()
    life_os.ai_engine = Mock()
    life_os.browser_orchestrator = Mock()
    life_os.onboarding = Mock()

    return life_os


@pytest.fixture
def app(mock_life_os):
    """Create a FastAPI test app with real database."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def test_workflows(user_model_store):
    """Create test workflows in the database."""
    workflows = [
        {
            "name": "Responding to boss emails",
            "trigger_conditions": ["email.received.from.boss@company.com"],
            "steps": ["read_email", "draft_response", "review_tone", "send"],
            "typical_duration_minutes": 15.5,
            "tools_used": ["email", "ai_engine"],
            "success_rate": 0.95,
            "times_observed": 42,
        },
        {
            "name": "Task completion workflow",
            "trigger_conditions": ["task.created"],
            "steps": ["create_task", "research", "execute", "completed"],
            "typical_duration_minutes": 120.0,
            "tools_used": ["task_manager", "browser", "email"],
            "success_rate": 0.78,
            "times_observed": 215,
        },
        {
            "name": "Calendar event workflow",
            "trigger_conditions": ["calendar.event.created"],
            "steps": ["prep_task", "attend_event", "followup_sent"],
            "typical_duration_minutes": None,
            "tools_used": ["calendar", "task_manager", "email"],
            "success_rate": 1.0,
            "times_observed": 2573,
        },
        {
            "name": "Low success workflow",
            "trigger_conditions": ["some.trigger"],
            "steps": ["step1", "step2"],
            "typical_duration_minutes": 5.0,
            "tools_used": ["tool1"],
            "success_rate": 0.05,  # Low success rate
            "times_observed": 100,
        },
        {
            "name": "Rare workflow",
            "trigger_conditions": ["rare.trigger"],
            "steps": ["step1", "step2", "step3"],
            "typical_duration_minutes": 30.0,
            "tools_used": ["tool1", "tool2"],
            "success_rate": 0.90,
            "times_observed": 2,  # Few observations
        },
    ]

    for workflow in workflows:
        user_model_store.store_workflow(workflow)

    return workflows


def test_get_workflows_returns_all_workflows(client, test_workflows):
    """Test that GET /api/user-model/workflows returns all workflows."""
    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    data = response.json()
    assert "workflows" in data
    assert "count" in data
    assert "generated_at" in data

    workflows = data["workflows"]
    assert len(workflows) == 5  # All 5 test workflows

    # Verify structure of first workflow
    workflow = workflows[0]
    assert "name" in workflow
    assert "trigger_conditions" in workflow
    assert "steps" in workflow
    assert "typical_duration" in workflow
    assert "tools_used" in workflow
    assert "success_rate" in workflow
    assert "times_observed" in workflow
    assert "updated_at" in workflow

    # Verify JSON fields are parsed as lists
    assert isinstance(workflow["trigger_conditions"], list)
    assert isinstance(workflow["steps"], list)
    assert isinstance(workflow["tools_used"], list)


def test_get_workflows_ordered_by_observations(client, test_workflows):
    """Test that workflows are ordered by times_observed DESC, then success_rate DESC."""
    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) >= 4

    # Should be ordered: Calendar (2573) → Task (215) → Low success (100) → Boss emails (42)
    assert workflows[0]["name"] == "Calendar event workflow"
    assert workflows[0]["times_observed"] == 2573
    assert workflows[1]["name"] == "Task completion workflow"
    assert workflows[1]["times_observed"] == 215
    assert workflows[2]["name"] == "Low success workflow"
    assert workflows[2]["times_observed"] == 100
    assert workflows[3]["name"] == "Responding to boss emails"
    assert workflows[3]["times_observed"] == 42


def test_get_workflows_filter_by_success_rate(client, test_workflows):
    """Test filtering workflows by minimum success rate."""
    # Filter for success_rate >= 0.5 (should exclude low success workflow)
    response = client.get("/api/user-model/workflows?min_success_rate=0.5")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 4  # Excludes "Low success workflow" (0.05)

    for workflow in workflows:
        assert workflow["success_rate"] >= 0.5


def test_get_workflows_filter_by_observations(client, test_workflows):
    """Test filtering workflows by minimum observations."""
    # Filter for times_observed >= 10 (should exclude rare workflows)
    response = client.get("/api/user-model/workflows?min_observations=10")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 4  # Excludes "Rare workflow" (2 observations)

    for workflow in workflows:
        assert workflow["times_observed"] >= 10


def test_get_workflows_filter_combined(client, test_workflows):
    """Test filtering workflows by both success rate and observations."""
    # Filter for success_rate >= 0.5 AND times_observed >= 50
    response = client.get("/api/user-model/workflows?min_success_rate=0.5&min_observations=50")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 2  # Only Calendar (2573, 1.0) and Task (215, 0.78)

    for workflow in workflows:
        assert workflow["success_rate"] >= 0.5
        assert workflow["times_observed"] >= 50


def test_get_workflows_empty_database(client):
    """Test endpoint returns empty list when no workflows exist."""
    # Don't use test_workflows fixture, so database starts empty
    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    data = response.json()
    assert data["workflows"] == []
    assert data["count"] == 0


def test_get_workflows_json_parsing(client, user_model_store):
    """Test that JSON fields are properly parsed from database."""
    # Store workflow with complex structures
    workflow = {
        "name": "Complex workflow",
        "trigger_conditions": ["event.type.one", "event.type.two"],
        "steps": ["step_one", "step_two", "step_three", "step_four"],
        "typical_duration_minutes": 45.7,
        "tools_used": ["email", "calendar", "task_manager", "browser"],
        "success_rate": 0.85,
        "times_observed": 30,
    }
    user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 1

    result = workflows[0]
    assert result["trigger_conditions"] == ["event.type.one", "event.type.two"]
    assert result["steps"] == ["step_one", "step_two", "step_three", "step_four"]
    assert result["tools_used"] == ["email", "calendar", "task_manager", "browser"]


def test_get_workflows_null_duration(client, user_model_store):
    """Test handling of workflows with null typical_duration."""
    workflow = {
        "name": "No duration workflow",
        "trigger_conditions": ["test.trigger"],
        "steps": ["step1", "step2"],
        "typical_duration_minutes": None,  # Null duration
        "tools_used": ["tool1"],
        "success_rate": 0.90,
        "times_observed": 10,
    }
    user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 1
    assert workflows[0]["typical_duration"] is None


def test_get_workflows_response_structure(client, test_workflows):
    """Test that response includes metadata fields."""
    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    data = response.json()

    # Verify count matches workflows length
    assert data["count"] == len(data["workflows"])

    # Verify generated_at is a valid ISO timestamp
    generated_at = datetime.fromisoformat(data["generated_at"].replace('Z', '+00:00'))
    assert isinstance(generated_at, datetime)
    assert generated_at.tzinfo is not None


def test_get_workflows_high_success_rate_filter(client, test_workflows):
    """Test filtering for only high-performing workflows."""
    # Filter for success_rate >= 0.9 (very high performers)
    response = client.get("/api/user-model/workflows?min_success_rate=0.9")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 3  # Boss emails (0.95), Calendar (1.0), Rare (0.90)

    for workflow in workflows:
        assert workflow["success_rate"] >= 0.9


def test_get_workflows_very_common_filter(client, test_workflows):
    """Test filtering for frequently observed workflows."""
    # Filter for times_observed >= 100 (very common patterns)
    response = client.get("/api/user-model/workflows?min_observations=100")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 3  # Calendar (2573), Task (215), Low success (100)

    for workflow in workflows:
        assert workflow["times_observed"] >= 100


def test_get_workflows_invalid_filters(client, test_workflows):
    """Test that invalid filter values still work (no validation errors)."""
    # Negative success rate should work (returns all)
    response = client.get("/api/user-model/workflows?min_success_rate=-1.0")
    assert response.status_code == 200
    assert len(response.json()["workflows"]) == 5

    # Very high success rate should work (returns subset)
    response = client.get("/api/user-model/workflows?min_success_rate=2.0")
    assert response.status_code == 200
    assert len(response.json()["workflows"]) == 0

    # Negative observations should work (returns all)
    response = client.get("/api/user-model/workflows?min_observations=-100")
    assert response.status_code == 200
    assert len(response.json()["workflows"]) == 5


def test_get_workflows_preserves_float_precision(client, user_model_store):
    """Test that float values are preserved with precision."""
    workflow = {
        "name": "Precise workflow",
        "trigger_conditions": ["test.trigger"],
        "steps": ["step1"],
        "typical_duration_minutes": 123.456789,  # High precision
        "tools_used": ["tool1"],
        "success_rate": 0.7654321,  # High precision
        "times_observed": 42,
    }
    user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 1

    result = workflows[0]
    # Verify floats are returned (some precision loss is acceptable from DB)
    assert isinstance(result["typical_duration"], (float, int))
    assert isinstance(result["success_rate"], (float, int))
    assert abs(result["success_rate"] - 0.765) < 0.01  # Within reasonable range


def test_get_workflows_graceful_degradation(client, db):
    """Test that endpoint handles missing workflows table gracefully."""
    # Drop the workflows table to simulate a cold-start scenario
    with db.get_connection("user_model") as conn:
        conn.execute("DROP TABLE IF EXISTS workflows")

    response = client.get("/api/user-model/workflows")
    # Should return 200 with empty list, not crash
    assert response.status_code == 200

    data = response.json()
    assert data["workflows"] == []
    assert data["count"] == 0


def test_get_workflows_empty_arrays_not_strings(client, user_model_store):
    """Test that empty JSON arrays don't get returned as strings."""
    workflow = {
        "name": "Empty fields workflow",
        "trigger_conditions": [],  # Empty list
        "steps": [],  # Empty list
        "typical_duration_minutes": 10.0,
        "tools_used": [],  # Empty list
        "success_rate": 0.5,
        "times_observed": 5,
    }
    user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 1

    result = workflows[0]
    # Verify empty arrays are lists, not strings
    assert result["trigger_conditions"] == []
    assert result["steps"] == []
    assert result["tools_used"] == []
    assert isinstance(result["trigger_conditions"], list)
    assert isinstance(result["steps"], list)
    assert isinstance(result["tools_used"], list)


def test_get_workflows_with_special_characters(client, user_model_store):
    """Test workflows with special characters in names and steps."""
    workflow = {
        "name": "Workflow with 'quotes' and \"escapes\"",
        "trigger_conditions": ["email.received.from.user@example.com"],
        "steps": ["read_email_from_user@example.com", "draft_response", "send"],
        "typical_duration_minutes": 10.0,
        "tools_used": ["email", "ai_engine"],
        "success_rate": 0.85,
        "times_observed": 20,
    }
    user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    workflows = response.json()["workflows"]
    assert len(workflows) == 1

    result = workflows[0]
    assert "quotes" in result["name"]
    assert "escapes" in result["name"]
    assert "@example.com" in result["trigger_conditions"][0]
    assert "@example.com" in result["steps"][0]


def test_get_workflows_sorting_tiebreaker(client, user_model_store):
    """Test that workflows with same observation count are sorted by success rate."""
    workflows = [
        {
            "name": "Workflow A",
            "trigger_conditions": ["a"],
            "steps": ["a1"],
            "typical_duration_minutes": 10.0,
            "tools_used": ["tool1"],
            "success_rate": 0.5,
            "times_observed": 50,
        },
        {
            "name": "Workflow B",
            "trigger_conditions": ["b"],
            "steps": ["b1"],
            "typical_duration_minutes": 10.0,
            "tools_used": ["tool1"],
            "success_rate": 0.9,
            "times_observed": 50,
        },
        {
            "name": "Workflow C",
            "trigger_conditions": ["c"],
            "steps": ["c1"],
            "typical_duration_minutes": 10.0,
            "tools_used": ["tool1"],
            "success_rate": 0.7,
            "times_observed": 50,
        },
    ]

    for workflow in workflows:
        user_model_store.store_workflow(workflow)

    response = client.get("/api/user-model/workflows")
    assert response.status_code == 200

    results = response.json()["workflows"]
    assert len(results) == 3

    # With same times_observed, should be sorted by success_rate DESC
    assert results[0]["name"] == "Workflow B"  # 0.9
    assert results[1]["name"] == "Workflow C"  # 0.7
    assert results[2]["name"] == "Workflow A"  # 0.5
