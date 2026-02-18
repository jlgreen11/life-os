"""
Tests for the /api/user-model/routines endpoint.

The routines API exposes Layer 3 procedural memory — habitual, time- or
location-triggered behavioral patterns detected by the RoutineDetector.
These are distinct from workflows (goal-driven multi-step processes): routines
fire automatically at predictable times or places, while workflows are initiated
in response to a specific goal.

The system continuously accumulates routines via the _routine_detection_loop
background task (usermodel.routine.updated events), but without this endpoint
there is no way to retrieve them via the API.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os(db, user_model_store):
    """Create a mock LifeOS instance with real database components.

    Uses real db and user_model_store fixtures (backed by temporary SQLite)
    so that store_routine / get_routines round-trips are exercised against
    real SQL rather than mocks.
    """
    life_os = Mock()
    life_os.db = db
    life_os.user_model_store = user_model_store

    # Mock all services not touched by the routines endpoint.
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
    life_os.config = {}

    return life_os


@pytest.fixture
def app(mock_life_os):
    """Create a FastAPI test app."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Create a synchronous TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def test_routines(user_model_store):
    """Populate the database with a representative set of test routines.

    Covers the full range of trigger types and consistency scores so we can
    verify filtering, ordering, and deserialization in a single fixture.
    """
    routines = [
        {
            "name": "morning_routine",
            "trigger": "morning",
            "steps": ["wake_up", "check_email", "review_calendar", "coffee"],
            "typical_duration_minutes": 45.0,
            "consistency_score": 0.92,
            "times_observed": 87,
            "variations": [],
        },
        {
            "name": "arrive_home",
            "trigger": "location_home",
            "steps": ["unlock_door", "check_messages", "start_dinner"],
            "typical_duration_minutes": 20.5,
            "consistency_score": 0.78,
            "times_observed": 63,
            "variations": [{"step": "check_messages", "alt": "watch_tv", "frequency": 0.2}],
        },
        {
            "name": "weekly_review",
            "trigger": "weekly_sunday_evening",
            "steps": ["review_tasks", "plan_next_week", "send_status_update"],
            "typical_duration_minutes": 60.0,
            "consistency_score": 0.65,
            "times_observed": 12,
            "variations": [],
        },
        {
            "name": "low_consistency_habit",
            "trigger": "evening",
            "steps": ["read_news", "check_social"],
            "typical_duration_minutes": 15.0,
            "consistency_score": 0.3,
            "times_observed": 8,
            "variations": [],
        },
        {
            "name": "rare_routine",
            "trigger": "monthly",
            "steps": ["pay_bills", "review_budget"],
            "typical_duration_minutes": 30.0,
            "consistency_score": 0.5,
            "times_observed": 2,
            "variations": [],
        },
    ]
    for routine in routines:
        user_model_store.store_routine(routine)
    return routines


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_routines_returns_all(client, test_routines):
    """GET /api/user-model/routines returns all stored routines."""
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    data = response.json()
    assert "routines" in data
    assert "count" in data
    assert "generated_at" in data

    # Verify all 5 test routines are returned
    assert data["count"] == 5
    assert len(data["routines"]) == 5


def test_get_routines_response_structure(client, test_routines):
    """Every returned routine has the expected fields."""
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) > 0

    # Check the first routine has all required fields
    r = routines[0]
    assert "name" in r
    assert "trigger" in r
    assert "steps" in r
    assert "typical_duration_minutes" in r
    assert "consistency_score" in r
    assert "times_observed" in r
    assert "variations" in r
    assert "updated_at" in r

    # JSON list fields must be deserialized (not raw JSON strings)
    assert isinstance(r["steps"], list)
    assert isinstance(r["variations"], list)


def test_get_routines_ordered_by_consistency(client, test_routines):
    """Routines are returned ordered by consistency_score DESC, times_observed DESC."""
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    routines = response.json()["routines"]

    # morning_routine has the highest consistency (0.92)
    assert routines[0]["name"] == "morning_routine"
    assert routines[0]["consistency_score"] == pytest.approx(0.92, abs=0.01)

    # arrive_home is second (0.78)
    assert routines[1]["name"] == "arrive_home"

    # weekly_review is third (0.65)
    assert routines[2]["name"] == "weekly_review"


def test_get_routines_filter_by_trigger(client, test_routines):
    """The `trigger` query param filters routines to a single trigger type."""
    response = client.get("/api/user-model/routines?trigger=morning")
    assert response.status_code == 200

    data = response.json()
    routines = data["routines"]

    # Only the morning_routine should be returned
    assert data["count"] == 1
    assert len(routines) == 1
    assert routines[0]["name"] == "morning_routine"
    assert routines[0]["trigger"] == "morning"


def test_get_routines_filter_by_trigger_location(client, test_routines):
    """Trigger filter works for location-based triggers."""
    response = client.get("/api/user-model/routines?trigger=location_home")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 1
    assert routines[0]["name"] == "arrive_home"


def test_get_routines_filter_by_trigger_no_match(client, test_routines):
    """A trigger filter with no matching routines returns an empty list."""
    response = client.get("/api/user-model/routines?trigger=nonexistent_trigger")
    assert response.status_code == 200

    data = response.json()
    assert data["routines"] == []
    assert data["count"] == 0


def test_get_routines_filter_by_min_consistency(client, test_routines):
    """min_consistency param excludes routines below the threshold."""
    # Threshold 0.6 should keep morning (0.92), arrive_home (0.78), weekly (0.65)
    # and exclude low_consistency (0.3) and rare_routine (0.5)
    response = client.get("/api/user-model/routines?min_consistency=0.6")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 3
    for r in routines:
        assert r["consistency_score"] >= 0.6


def test_get_routines_filter_by_min_observations(client, test_routines):
    """min_observations param excludes rarely-observed routines."""
    # Threshold 10 should keep morning (87), arrive_home (63), weekly (12)
    # and exclude low_consistency (8) and rare_routine (2)
    response = client.get("/api/user-model/routines?min_observations=10")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 3
    for r in routines:
        assert r["times_observed"] >= 10


def test_get_routines_filter_combined(client, test_routines):
    """Combining min_consistency and min_observations applies both filters."""
    # consistency >= 0.7 AND observations >= 20 → only morning (0.92, 87) and arrive_home (0.78, 63)
    response = client.get("/api/user-model/routines?min_consistency=0.7&min_observations=20")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 2
    names = {r["name"] for r in routines}
    assert names == {"morning_routine", "arrive_home"}


def test_get_routines_empty_database(client):
    """Endpoint returns empty list and count=0 when no routines exist."""
    # Do not inject test_routines so the database stays empty
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    data = response.json()
    assert data["routines"] == []
    assert data["count"] == 0
    assert "generated_at" in data


def test_get_routines_steps_deserialized(client, user_model_store):
    """Steps list is returned as a parsed Python list, not a JSON string."""
    user_model_store.store_routine({
        "name": "test_steps",
        "trigger": "morning",
        "steps": ["step_a", "step_b", "step_c"],
        "typical_duration_minutes": 10.0,
        "consistency_score": 0.8,
        "times_observed": 5,
        "variations": [],
    })

    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 1
    assert routines[0]["steps"] == ["step_a", "step_b", "step_c"]


def test_get_routines_variations_deserialized(client, user_model_store):
    """Variations list with nested dicts is returned as a parsed Python list."""
    user_model_store.store_routine({
        "name": "test_variations",
        "trigger": "evening",
        "steps": ["step_x"],
        "typical_duration_minutes": 5.0,
        "consistency_score": 0.6,
        "times_observed": 10,
        "variations": [
            {"step": "step_x", "alt": "step_y", "frequency": 0.3},
            {"step": "step_x", "alt": "step_z", "frequency": 0.1},
        ],
    })

    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    routines = response.json()["routines"]
    assert len(routines) == 1
    variations = routines[0]["variations"]
    assert isinstance(variations, list)
    assert len(variations) == 2
    assert variations[0]["step"] == "step_x"
    assert variations[0]["alt"] == "step_y"


def test_get_routines_generated_at_is_iso_timestamp(client, test_routines):
    """generated_at in the response is a valid ISO-8601 timestamp."""
    response = client.get("/api/user-model/routines")
    assert response.status_code == 200

    generated_at = response.json()["generated_at"]
    # Must parse without error
    dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    # Must be recent (within the last minute)
    now = datetime.now(timezone.utc)
    assert abs((now - dt).total_seconds()) < 60
