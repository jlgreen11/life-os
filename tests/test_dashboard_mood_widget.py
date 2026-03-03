"""
Tests: Dashboard mood widget shows correct state for no-data vs real data.

Verifies that the GET /api/user-model/mood endpoint returns the correct
response structure so the dashboard JavaScript can distinguish between
"no mood data yet" (confidence=0) and real tracked mood values.

Previously the dashboard JS used fallback values like ``|| 0.5`` which made
empty/zero mood data appear as real readings (50% energy, 30% stress, etc.).
The fix checks ``confidence > 0`` before treating data as real, and the API
contract tested here ensures confidence is always present in the response.
"""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.user_model import MoodState
from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mood_return_value) -> tuple:
    """Create a minimal FastAPI test client with a mocked mood endpoint.

    Args:
        mood_return_value: The object that signal_extractor.get_current_mood()
            will return.

    Returns:
        Tuple of (TestClient, mock_life_os).
    """
    app = FastAPI()
    life_os = Mock()

    # Stub services required for route registration
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[]), fetchone=Mock(return_value=None)))
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
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.get_event_flow_stats = Mock(return_value={
        "sources": {}, "stale_sources": [], "total_24h": 0, "events_per_hour": 0.0,
    })
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.user_model_store = Mock()
    life_os.connectors = []
    life_os.feedback_collector = Mock()

    # The key mock: signal_extractor.get_current_mood()
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_current_mood = Mock(return_value=mood_return_value)
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})

    register_routes(app, life_os)
    return TestClient(app), life_os


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMoodEndpointNoData:
    """Verify the mood endpoint returns confidence=0 when no mood data exists."""

    def test_mood_endpoint_returns_zero_confidence_with_defaults(self):
        """When no mood signals have been processed, the endpoint should return
        the MoodState defaults including confidence=0, so the dashboard JS can
        detect the 'no data' case and show an appropriate empty state instead
        of fake bar values.
        """
        # MoodState() with no arguments simulates "no mood data yet"
        default_mood = MoodState()
        client, _ = _make_client(default_mood)

        response = client.get("/api/user-model/mood")

        assert response.status_code == 200
        data = response.json()
        assert "mood" in data
        mood = data["mood"]
        # confidence must be present and zero so the JS can detect no-data
        assert "confidence" in mood
        assert mood["confidence"] == 0.0
        # The default float values exist but should NOT be treated as real data
        assert "energy_level" in mood
        assert "stress_level" in mood
        assert "social_battery" in mood

    def test_mood_endpoint_null_fields_manual_serialization(self):
        """When the mood object uses manual attribute access (non-Pydantic),
        the response still includes confidence for no-data detection.
        """
        class ManualMood:
            """Simulates a mood object without model_dump() (non-Pydantic)."""
            energy_level = 0.5
            stress_level = 0.3
            social_battery = 0.5
            cognitive_load = 0.3
            emotional_valence = 0.5
            confidence = 0.0
            trend = "stable"

        client, _ = _make_client(ManualMood())

        response = client.get("/api/user-model/mood")

        assert response.status_code == 200
        data = response.json()
        mood = data["mood"]
        assert mood["confidence"] == 0.0
        # All dimension fields are present for the JS to check
        assert mood["energy_level"] == 0.5
        assert mood["stress_level"] == 0.3


class TestMoodEndpointWithValidData:
    """Verify that when real mood data exists, the endpoint returns proper values."""

    def test_mood_endpoint_returns_valid_data_with_confidence(self):
        """When mood signals have been processed, the endpoint returns real
        float values with confidence > 0, which the dashboard JS will display
        as actual mood bars.
        """
        real_mood = MoodState(
            energy_level=0.8,
            stress_level=0.4,
            social_battery=0.6,
            cognitive_load=0.5,
            emotional_valence=0.7,
            confidence=0.65,
            trend="improving",
        )
        client, _ = _make_client(real_mood)

        response = client.get("/api/user-model/mood")

        assert response.status_code == 200
        data = response.json()
        mood = data["mood"]
        assert mood["confidence"] == 0.65
        assert mood["energy_level"] == 0.8
        assert mood["stress_level"] == 0.4
        assert mood["social_battery"] == 0.6
        assert mood["trend"] == "improving"

    def test_mood_endpoint_returns_low_confidence_as_real(self):
        """Even low-confidence mood data (e.g. 0.1) should be returned as-is.
        The dashboard JS uses confidence > 0 to distinguish from no-data.
        """
        low_confidence_mood = MoodState(
            energy_level=0.3,
            stress_level=0.6,
            social_battery=0.2,
            confidence=0.1,
            trend="declining",
        )
        client, _ = _make_client(low_confidence_mood)

        response = client.get("/api/user-model/mood")

        assert response.status_code == 200
        mood = response.json()["mood"]
        # Low but non-zero confidence means we have some data
        assert mood["confidence"] == 0.1
        assert mood["energy_level"] == 0.3


class TestMoodEndpointErrorHandling:
    """Verify the endpoint handles errors gracefully."""

    def test_mood_endpoint_handles_exception_in_get_current_mood(self):
        """When signal_extractor.get_current_mood() raises an exception,
        the endpoint should return a 500 rather than crash the server.
        """
        client, life_os = _make_client(MoodState())
        # Override to raise an exception
        life_os.signal_extractor.get_current_mood = Mock(
            side_effect=RuntimeError("user_model.db corrupted")
        )

        # Use raise_server_exceptions=False so TestClient returns the 500
        # response instead of re-raising the exception in the test process.
        client_no_raise = TestClient(client.app, raise_server_exceptions=False)
        response = client_no_raise.get("/api/user-model/mood")

        # FastAPI returns 500 on unhandled exceptions
        assert response.status_code == 500

    def test_mood_response_structure_always_has_mood_key(self):
        """The response must always wrap mood data under a 'mood' key
        so the JS can reliably access data.mood.
        """
        client, _ = _make_client(MoodState())

        response = client.get("/api/user-model/mood")

        data = response.json()
        assert "mood" in data
        assert isinstance(data["mood"], dict)
