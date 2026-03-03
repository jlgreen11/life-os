"""
Tests for user_model.db corruption resilience in web route endpoints.

Verifies that endpoints querying user_model.db return HTTP 503 with
descriptive error messages instead of raw 500 errors when the database
is corrupted or otherwise unavailable.

Covers five endpoints:
  - PATCH /api/user-model/facts/{key}  (correct_fact)
  - POST  /api/user-model/facts/{key}/confirm  (confirm_fact)
  - POST  /api/insights/{insight_id}/feedback  (insight_feedback)
  - GET   /api/predictions  (list_predictions)
  - POST  /api/predictions/{prediction_id}/feedback  (prediction_feedback)
"""

import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Create a minimal mock LifeOS with a healthy db by default."""
    life_os = Mock()

    # Default: db is healthy (not degraded)
    life_os.db = Mock()
    life_os.db.user_model_degraded = False
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # Mock event bus
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # Mock event store
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])

    # Mock vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})

    # Mock signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable",
    ))

    # Mock notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])

    # Mock feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})

    # Mock AI engine
    life_os.ai_engine = Mock()

    # Mock task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])

    # Mock rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])

    # Mock user model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Mock connectors
    life_os.connectors = []

    # Mock browser orchestrator
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Mock onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})

    # Mock connector management methods
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})

    return life_os


def _make_corrupted_connection(mock_life_os):
    """Configure mock_life_os.db.get_connection('user_model') to raise DatabaseError."""
    original_get_connection = mock_life_os.db.get_connection

    class CorruptedContextManager:
        """Context manager that raises sqlite3.DatabaseError on __enter__."""
        def __enter__(self):
            raise sqlite3.DatabaseError("database disk image is malformed")

        def __exit__(self, *args):
            pass

    def side_effect(db_name):
        if db_name == "user_model":
            return CorruptedContextManager()
        return original_get_connection(db_name)

    mock_life_os.db.get_connection = Mock(side_effect=side_effect)


@pytest.fixture
def client(mock_life_os):
    """Create a test client with a healthy mock LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


@pytest.fixture
def corrupted_client(mock_life_os):
    """Create a test client where user_model.db raises DatabaseError."""
    _make_corrupted_connection(mock_life_os)
    app = create_web_app(mock_life_os)
    return TestClient(app)


@pytest.fixture
def degraded_client(mock_life_os):
    """Create a test client where user_model_degraded flag is True."""
    mock_life_os.db.user_model_degraded = True
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# PATCH /api/user-model/facts/{key} — correct_fact
# ---------------------------------------------------------------------------

class TestCorrectFactResilience:
    """Tests for corruption resilience in the correct_fact endpoint."""

    def test_correct_fact_returns_503_on_database_error(self, corrupted_client):
        """When user_model.db is corrupted, correct_fact returns 503 not 500."""
        response = corrupted_client.patch(
            "/api/user-model/facts/test_key",
            json={"corrected_value": "new_value"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]

    def test_correct_fact_returns_503_on_degraded_flag(self, degraded_client):
        """When user_model_degraded is True, correct_fact short-circuits to 503."""
        response = degraded_client.patch(
            "/api/user-model/facts/test_key",
            json={"corrected_value": "new_value"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]
        assert "detail" in data


# ---------------------------------------------------------------------------
# POST /api/user-model/facts/{key}/confirm — confirm_fact
# ---------------------------------------------------------------------------

class TestConfirmFactResilience:
    """Tests for corruption resilience in the confirm_fact endpoint."""

    def test_confirm_fact_returns_503_on_database_error(self, corrupted_client):
        """When user_model.db is corrupted, confirm_fact returns 503 not 500."""
        response = corrupted_client.post(
            "/api/user-model/facts/test_key/confirm",
            json={},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]

    def test_confirm_fact_returns_503_on_degraded_flag(self, degraded_client):
        """When user_model_degraded is True, confirm_fact short-circuits to 503."""
        response = degraded_client.post(
            "/api/user-model/facts/test_key/confirm",
            json={},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]


# ---------------------------------------------------------------------------
# POST /api/insights/{insight_id}/feedback — insight_feedback
# ---------------------------------------------------------------------------

class TestInsightFeedbackResilience:
    """Tests for corruption resilience in the insight_feedback endpoint."""

    def test_insight_feedback_returns_503_on_database_error(self, corrupted_client):
        """When user_model.db is corrupted, insight_feedback returns 503 not 500."""
        response = corrupted_client.post(
            "/api/insights/test-insight-id/feedback",
            params={"feedback": "useful"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]

    def test_insight_feedback_returns_503_on_degraded_flag(self, degraded_client):
        """When user_model_degraded is True, insight_feedback short-circuits to 503."""
        response = degraded_client.post(
            "/api/insights/test-insight-id/feedback",
            params={"feedback": "dismissed"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]


# ---------------------------------------------------------------------------
# GET /api/predictions — list_predictions
# ---------------------------------------------------------------------------

class TestListPredictionsResilience:
    """Tests for corruption resilience in the list_predictions endpoint."""

    def test_list_predictions_returns_503_on_database_error(self, corrupted_client):
        """When user_model.db is corrupted, list_predictions returns 503 with valid shape."""
        response = corrupted_client.get("/api/predictions")
        assert response.status_code == 503
        data = response.json()
        # Must preserve the expected response shape so the frontend doesn't break
        assert data["predictions"] == []
        assert data["count"] == 0
        assert "error" in data
        assert "temporarily unavailable" in data["error"]

    def test_list_predictions_returns_503_on_degraded_flag(self, degraded_client):
        """When user_model_degraded is True, list_predictions short-circuits to 503."""
        response = degraded_client.get("/api/predictions")
        assert response.status_code == 503
        data = response.json()
        assert data["predictions"] == []
        assert data["count"] == 0
        assert "error" in data

    def test_list_predictions_shape_preserved_on_error(self, corrupted_client):
        """Verify the error response includes predictions and count keys for frontend compat."""
        response = corrupted_client.get("/api/predictions")
        data = response.json()
        assert isinstance(data["predictions"], list)
        assert isinstance(data["count"], int)


# ---------------------------------------------------------------------------
# POST /api/predictions/{prediction_id}/feedback — prediction_feedback
# ---------------------------------------------------------------------------

class TestPredictionFeedbackResilience:
    """Tests for corruption resilience in the prediction_feedback endpoint."""

    def test_prediction_feedback_returns_503_on_database_error(self, corrupted_client):
        """When user_model.db is corrupted, prediction_feedback returns 503 not 500."""
        response = corrupted_client.post(
            "/api/predictions/test-pred-id/feedback",
            params={"was_accurate": "true"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]

    def test_prediction_feedback_returns_503_on_degraded_flag(self, degraded_client):
        """When user_model_degraded is True, prediction_feedback short-circuits to 503."""
        response = degraded_client.post(
            "/api/predictions/test-pred-id/feedback",
            params={"was_accurate": "false"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "temporarily unavailable" in data["error"]
