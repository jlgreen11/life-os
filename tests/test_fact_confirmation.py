"""
Tests for POST /api/user-model/facts/{key}/confirm endpoint.

Verifies that explicit user confirmation of semantic facts correctly
bumps confidence by +0.05 (the architectural standard), increments
times_confirmed, caps confidence at 1.0, and publishes telemetry.
"""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


def _make_life_os(db):
    """Build a minimal mock LifeOS wired to a real DatabaseManager.

    Uses a real database for the user_model connection so SQL queries
    exercise the actual schema, while other services are mocked.
    """
    life_os = Mock()
    life_os.db = db
    life_os.config = {}

    # Event bus mock — tracks published events
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # Feedback collector mock
    life_os.feedback_collector = Mock()
    life_os.feedback_collector._store_feedback = AsyncMock()

    # Stubs required by other routes so the app can boot
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.5, social_battery=0.5,
        cognitive_load=0.5, emotional_valence=0.5, confidence=0.5, trend="stable",
    ))
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.ai_engine = Mock()
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})

    return life_os


def _insert_fact(db, key, category="preference", value="test", confidence=0.5, times_confirmed=1):
    """Insert a semantic fact directly into the database for testing."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO semantic_facts (key, category, value, confidence, source_episodes, times_confirmed)
               VALUES (?, ?, ?, ?, '[]', ?)""",
            (key, category, json.dumps(value), confidence, times_confirmed),
        )


def _get_fact(db, key):
    """Retrieve a fact from the database by key."""
    with db.get_connection("user_model") as conn:
        row = conn.execute("SELECT * FROM semantic_facts WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_confirm_existing_fact(db):
    """POST confirm bumps confidence by +0.05 and increments times_confirmed."""
    _insert_fact(db, "fav_color", confidence=0.5, times_confirmed=1)
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    response = client.post("/api/user-model/facts/fav_color/confirm")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "confirmed"
    assert data["old_confidence"] == 0.5
    assert data["new_confidence"] == 0.55
    assert data["times_confirmed"] == 2

    # Verify the database was updated
    fact = _get_fact(db, "fav_color")
    assert fact["confidence"] == 0.55
    assert fact["times_confirmed"] == 2


def test_confirm_caps_at_1(db):
    """Confidence should cap at 1.0, never exceed it."""
    _insert_fact(db, "high_fact", confidence=0.98, times_confirmed=5)
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    response = client.post("/api/user-model/facts/high_fact/confirm")
    assert response.status_code == 200

    data = response.json()
    assert data["new_confidence"] == 1.0
    assert data["old_confidence"] == 0.98

    fact = _get_fact(db, "high_fact")
    assert fact["confidence"] == 1.0


def test_confirm_nonexistent_returns_404(db):
    """Confirming a fact that doesn't exist should return 404."""
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    response = client.post("/api/user-model/facts/does_not_exist/confirm")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_confirm_with_reason(db):
    """POST with a reason should include it in feedback context."""
    _insert_fact(db, "job_title", confidence=0.6, times_confirmed=2)
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    response = client.post(
        "/api/user-model/facts/job_title/confirm",
        json={"reason": "Verified manually"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"

    # Verify reason was passed to the feedback collector
    life_os.feedback_collector._store_feedback.assert_called_once()
    call_args = life_os.feedback_collector._store_feedback.call_args[0][0]
    assert call_args["context"]["reason"] == "Verified manually"
    assert call_args["notes"] == "Verified manually"


def test_confirm_increments_times_confirmed(db):
    """Confirming the same fact 3 times should increment times_confirmed from 1 to 4."""
    _insert_fact(db, "location", confidence=0.5, times_confirmed=1)
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    for i in range(3):
        response = client.post("/api/user-model/facts/location/confirm")
        assert response.status_code == 200

    fact = _get_fact(db, "location")
    assert fact["times_confirmed"] == 4
    # 0.5 + 3 * 0.05 = 0.65
    assert fact["confidence"] == 0.65


def test_confirm_publishes_telemetry(db):
    """Confirming a fact should publish 'usermodel.fact.confirmed' to the event bus."""
    _insert_fact(db, "hobby", confidence=0.7, times_confirmed=3)
    life_os = _make_life_os(db)
    client = TestClient(create_web_app(life_os))

    response = client.post("/api/user-model/facts/hobby/confirm")
    assert response.status_code == 200

    life_os.event_bus.publish.assert_called_once()
    call_args = life_os.event_bus.publish.call_args
    assert call_args[0][0] == "usermodel.fact.confirmed"
    payload = call_args[0][1]
    assert payload["key"] == "hobby"
    assert payload["old_confidence"] == 0.7
    assert payload["new_confidence"] == 0.75
    assert payload["times_confirmed"] == 4
    assert call_args[1]["source"] == "web_api"
