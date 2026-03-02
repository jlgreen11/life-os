"""
Tests for warning logging in iOS context event endpoints.

Verifies that:
- Silent error swallowing is replaced with logger.warning() calls
- Fail-open behavior is preserved (endpoints still return 200 on bus failures)
- Batch endpoint tracks and returns publish_failures count
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Minimal mock LifeOS for context endpoint testing."""
    life_os = Mock()

    # Database manager (needed for app startup)
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock())
    )
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # Event bus — default: working
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # Event store
    life_os.event_store = Mock()
    life_os.event_store.store_event = Mock(return_value="evt-test-123")
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])

    # Other services (may be accessed during app creation)
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.5, social_battery=0.5,
        cognitive_load=0.5, emotional_valence=0.5, confidence=0.5, trend="stable"
    ))
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.feedback_collector = Mock()
    life_os.ai_engine = Mock()
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Single context event: fail-open with logging
# ---------------------------------------------------------------------------


def test_submit_context_event_returns_200_when_event_bus_fails(client, mock_life_os):
    """Event bus publish failure should not prevent a 200 response."""
    mock_life_os.event_bus.publish = AsyncMock(side_effect=Exception("NATS down"))

    response = client.post("/api/context/event", json={
        "type": "context.location",
        "source": "ios_app",
        "payload": {"latitude": 37.7749, "longitude": -122.4194},
    })

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert "event_id" in data


def test_submit_context_event_logs_warning_on_bus_failure(client, mock_life_os, caplog):
    """Event bus publish failure should emit a warning log."""
    mock_life_os.event_bus.publish = AsyncMock(side_effect=Exception("NATS connection lost"))

    with caplog.at_level(logging.WARNING, logger="web.routes"):
        client.post("/api/context/event", json={
            "type": "context.location",
            "source": "ios_app",
            "payload": {"latitude": 37.7749, "longitude": -122.4194},
        })

    assert any("Context event bus publish failed" in msg for msg in caplog.messages)
    assert any("context.location" in msg for msg in caplog.messages)


def test_submit_context_event_logs_warning_on_place_update_failure(mock_life_os, caplog):
    """Place update failure should emit a warning log, not crash."""
    # Make the db.get_connection("entities") context manager raise on __enter__
    # to simulate a DB failure during place update.
    original_get_connection = mock_life_os.db.get_connection

    def failing_get_connection(db_name, *args, **kwargs):
        """Raise only for 'entities' db to trigger the place update except branch."""
        if db_name == "entities":
            raise Exception("DB write failed")
        return original_get_connection(db_name, *args, **kwargs)

    mock_life_os.db.get_connection = Mock(side_effect=failing_get_connection)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    with caplog.at_level(logging.WARNING, logger="web.routes"):
        response = client.post("/api/context/event", json={
            "type": "context.location",
            "source": "ios_app",
            "payload": {"latitude": 37.7749, "longitude": -122.4194},
        })

    assert response.status_code == 200
    assert any("Place update from context event failed" in msg for msg in caplog.messages)


def test_submit_context_event_logs_warning_on_device_correlation_failure(mock_life_os, caplog):
    """Device-contact correlation failure should emit a warning log, not crash."""
    original_get_connection = mock_life_os.db.get_connection

    def failing_get_connection(db_name, *args, **kwargs):
        """Raise only for 'entities' db to trigger the correlation except branch."""
        if db_name == "entities":
            raise Exception("Correlation error")
        return original_get_connection(db_name, *args, **kwargs)

    mock_life_os.db.get_connection = Mock(side_effect=failing_get_connection)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    with caplog.at_level(logging.WARNING, logger="web.routes"):
        response = client.post("/api/context/event", json={
            "type": "context.device_nearby",
            "source": "ios_app",
            "payload": {"device_name": "iPhone", "signal_strength": -45},
        })

    assert response.status_code == 200
    assert any("Device-contact correlation failed" in msg for msg in caplog.messages)
    assert any("iPhone" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Batch context events: publish_failures tracking
# ---------------------------------------------------------------------------


def test_submit_context_batch_returns_zero_publish_failures_on_success(client, mock_life_os):
    """Batch endpoint should return publish_failures: 0 when all publishes succeed."""
    response = client.post("/api/context/batch", json={
        "events": [
            {"type": "context.location", "source": "ios_app", "payload": {"latitude": 1.0, "longitude": 2.0}},
            {"type": "context.time", "source": "ios_app", "payload": {}},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["count"] == 2
    assert data["publish_failures"] == 0


def test_submit_context_batch_tracks_publish_failures(client, mock_life_os):
    """Batch endpoint should count and return the number of failed publishes."""
    mock_life_os.event_bus.publish = AsyncMock(side_effect=Exception("Bus unavailable"))

    response = client.post("/api/context/batch", json={
        "events": [
            {"type": "context.location", "source": "ios_app", "payload": {"latitude": 1.0, "longitude": 2.0}},
            {"type": "context.device_nearby", "source": "ios_app", "payload": {"device_name": "Watch"}},
            {"type": "context.time", "source": "ios_app", "payload": {}},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["count"] == 3
    assert len(data["event_ids"]) == 3
    assert data["publish_failures"] == 3


def test_submit_context_batch_partial_publish_failures(client, mock_life_os):
    """Batch endpoint should correctly count when only some publishes fail."""
    call_count = 0

    async def fail_on_second(*args, **kwargs):
        """Fail only on the second call."""
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("Intermittent failure")

    mock_life_os.event_bus.publish = AsyncMock(side_effect=fail_on_second)

    response = client.post("/api/context/batch", json={
        "events": [
            {"type": "context.location", "source": "ios_app", "payload": {"latitude": 1.0, "longitude": 2.0}},
            {"type": "context.time", "source": "ios_app", "payload": {}},
            {"type": "context.location", "source": "ios_app", "payload": {"latitude": 3.0, "longitude": 4.0}},
        ],
    })

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert data["publish_failures"] == 1


def test_submit_context_batch_logs_warning_on_publish_failure(client, mock_life_os, caplog):
    """Batch endpoint should log a warning for each failed publish."""
    mock_life_os.event_bus.publish = AsyncMock(side_effect=Exception("Bus down"))

    with caplog.at_level(logging.WARNING, logger="web.routes"):
        client.post("/api/context/batch", json={
            "events": [
                {"type": "context.location", "source": "ios_app", "payload": {"latitude": 1.0, "longitude": 2.0}},
            ],
        })

    assert any("Context batch event bus publish failed" in msg for msg in caplog.messages)
    assert any("context.location" in msg for msg in caplog.messages)
