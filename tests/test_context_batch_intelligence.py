"""
Tests for context batch event type mapping and intelligence endpoint resilience.

Bug 1: /api/context/batch was hardcoding event type to "system.user.command"
instead of using the event_type_map like the single-event endpoint does.

Bug 2: /api/system/intelligence was wrapping all 5 table counts in a single
try/except, so one missing table would zero out all subsequent counts.
"""

import sqlite3
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Shared fixture: minimal mock LifeOS for route testing
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with enough services for our tests."""
    life_os = Mock()

    # Database manager
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchone=Mock(return_value=(0,))))
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )
    life_os.db.get_database_health = Mock(return_value={
        "events":      {"status": "ok", "errors": [], "path": "/tmp/events.db",      "size_bytes": 1024},
        "entities":    {"status": "ok", "errors": [], "path": "/tmp/entities.db",    "size_bytes": 1024},
        "state":       {"status": "ok", "errors": [], "path": "/tmp/state.db",       "size_bytes": 1024},
        "user_model":  {"status": "ok", "errors": [], "path": "/tmp/user_model.db",  "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # Event bus
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # Event store — capture stored events for assertion
    life_os.event_store = Mock()
    life_os.event_store.stored_events = []

    def _store_event(event):
        life_os.event_store.stored_events.append(event)
        return f"evt-{len(life_os.event_store.stored_events)}"

    life_os.event_store.store_event = Mock(side_effect=_store_event)
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.3, social_battery=0.7,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.5, trend="stable"
    ))

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})

    # AI engine
    life_os.ai_engine = Mock()

    # Task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])

    # User model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)

    # Prediction engine
    life_os.prediction_engine = Mock()
    life_os.prediction_engine.get_diagnostics = AsyncMock(return_value={
        "prediction_types": {},
        "overall": {"health": "unknown"},
    })

    # Connectors
    life_os.connectors = []

    # Browser orchestrator
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})

    return life_os


@pytest.fixture
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug 1: /api/context/batch event type mapping
# ---------------------------------------------------------------------------

class TestContextBatchEventTypeMapping:
    """Verify that the batch endpoint maps context types to internal types."""

    def test_batch_maps_location_type(self, client, mock_life_os):
        """context.location events should be stored as location.changed."""
        response = client.post("/api/context/batch", json={
            "events": [{
                "type": "context.location",
                "source": "ios_app",
                "payload": {"latitude": 40.7128, "longitude": -74.0060},
            }]
        })

        assert response.status_code == 200
        assert response.json()["count"] == 1

        stored = mock_life_os.event_store.stored_events
        assert len(stored) == 1
        assert stored[0]["type"] == "location.changed"
        # Original type preserved in metadata for backward compatibility
        assert stored[0]["metadata"]["mobile_event_type"] == "context.location"

    def test_batch_maps_device_nearby_type(self, client, mock_life_os):
        """context.device_nearby events should be stored as home.device.state_changed."""
        response = client.post("/api/context/batch", json={
            "events": [{
                "type": "context.device_nearby",
                "source": "ios_app",
                "payload": {"device_name": "iPhone"},
            }]
        })

        assert response.status_code == 200
        stored = mock_life_os.event_store.stored_events
        assert len(stored) == 1
        assert stored[0]["type"] == "home.device.state_changed"
        assert stored[0]["metadata"]["mobile_event_type"] == "context.device_nearby"

    def test_batch_maps_background_refresh_type(self, client, mock_life_os):
        """context.background_refresh should map to system.connector.sync_complete."""
        response = client.post("/api/context/batch", json={
            "events": [{
                "type": "context.background_refresh",
                "source": "ios_app",
                "payload": {},
            }]
        })

        assert response.status_code == 200
        stored = mock_life_os.event_store.stored_events
        assert stored[0]["type"] == "system.connector.sync_complete"

    def test_batch_unknown_type_falls_back(self, client, mock_life_os):
        """Unknown context types should fall back to system.user.command."""
        response = client.post("/api/context/batch", json={
            "events": [{
                "type": "context.unknown_sensor",
                "source": "ios_app",
                "payload": {},
            }]
        })

        assert response.status_code == 200
        stored = mock_life_os.event_store.stored_events
        assert stored[0]["type"] == "system.user.command"
        assert stored[0]["metadata"]["mobile_event_type"] == "context.unknown_sensor"

    def test_batch_multiple_events_mapped_correctly(self, client, mock_life_os):
        """Each event in a batch should be independently type-mapped."""
        response = client.post("/api/context/batch", json={
            "events": [
                {"type": "context.location", "source": "ios_app",
                 "payload": {"latitude": 1.0, "longitude": 2.0}},
                {"type": "context.device_nearby", "source": "ios_app",
                 "payload": {"device_name": "Watch"}},
                {"type": "context.time", "source": "ios_app",
                 "payload": {}},
            ]
        })

        assert response.status_code == 200
        assert response.json()["count"] == 3

        stored = mock_life_os.event_store.stored_events
        assert stored[0]["type"] == "location.changed"
        assert stored[1]["type"] == "home.device.state_changed"
        assert stored[2]["type"] == "system.user.command"


# ---------------------------------------------------------------------------
# Bug 2: /api/system/intelligence partial failure resilience
# ---------------------------------------------------------------------------

class TestIntelligenceEndpointResilience:
    """Verify that the intelligence endpoint returns partial results on failure."""

    def test_intelligence_returns_counts_when_all_tables_exist(self, client, mock_life_os):
        """All table counts should be returned when queries succeed."""
        mock_conn = Mock()
        # Return different counts for each table to prove each is queried
        counts = {
            "signal_profiles": 8,
            "routines": 3,
            "workflows": 1,
            "semantic_facts": 23,
            "episodes": 5000,
        }

        def execute_side_effect(query):
            """Return the correct count based on which table is queried."""
            for table, count in counts.items():
                if table in query:
                    return Mock(fetchone=Mock(return_value=(count,)))
            return Mock(fetchone=Mock(return_value=(0,)))

        mock_conn.execute = Mock(side_effect=execute_side_effect)
        mock_life_os.db.get_connection = Mock(
            return_value=Mock(
                __enter__=Mock(return_value=mock_conn),
                __exit__=Mock(return_value=False),
            )
        )

        response = client.get("/api/system/intelligence")
        assert response.status_code == 200
        data = response.json()

        depth = data["user_model_depth"]
        assert depth["signal_profiles"] == 8
        assert depth["routines"] == 3
        assert depth["workflows"] == 1
        assert depth["semantic_facts"] == 23
        assert depth["episodes"] == 5000

    def test_intelligence_partial_results_on_table_failure(self, client, mock_life_os):
        """When one table query fails, other counts should still be populated."""
        mock_conn = Mock()

        def execute_side_effect(query):
            """Fail on workflows table, succeed on others."""
            if "workflows" in query:
                raise sqlite3.OperationalError("no such table: workflows")
            if "signal_profiles" in query:
                return Mock(fetchone=Mock(return_value=(8,)))
            if "routines" in query:
                return Mock(fetchone=Mock(return_value=(3,)))
            if "semantic_facts" in query:
                return Mock(fetchone=Mock(return_value=(23,)))
            if "episodes" in query:
                return Mock(fetchone=Mock(return_value=(5000,)))
            return Mock(fetchone=Mock(return_value=(0,)))

        mock_conn.execute = Mock(side_effect=execute_side_effect)
        mock_life_os.db.get_connection = Mock(
            return_value=Mock(
                __enter__=Mock(return_value=mock_conn),
                __exit__=Mock(return_value=False),
            )
        )

        response = client.get("/api/system/intelligence")
        assert response.status_code == 200
        data = response.json()

        depth = data["user_model_depth"]
        # Workflows failed — should remain 0
        assert depth["workflows"] == 0
        # Other tables should have their real counts
        assert depth["signal_profiles"] == 8
        assert depth["routines"] == 3
        assert depth["semantic_facts"] == 23
        assert depth["episodes"] == 5000

    def test_intelligence_logs_warning_on_table_failure(self, client, mock_life_os):
        """A warning should be logged when a table query fails."""
        mock_conn = Mock()

        def execute_side_effect(query):
            if "routines" in query:
                raise sqlite3.OperationalError("no such table: routines")
            return Mock(fetchone=Mock(return_value=(0,)))

        mock_conn.execute = Mock(side_effect=execute_side_effect)
        mock_life_os.db.get_connection = Mock(
            return_value=Mock(
                __enter__=Mock(return_value=mock_conn),
                __exit__=Mock(return_value=False),
            )
        )

        with patch("web.routes.logger") as mock_logger:
            response = client.get("/api/system/intelligence")
            assert response.status_code == 200

            # Verify that a warning was logged for the failed table
            mock_logger.warning.assert_called()
            warning_args = mock_logger.warning.call_args_list
            found_routines_warning = any(
                "routines" in str(call) for call in warning_args
            )
            assert found_routines_warning, (
                f"Expected warning about 'routines' table, got: {warning_args}"
            )

    def test_intelligence_all_tables_fail_returns_zeros(self, client, mock_life_os):
        """When all table queries fail, all counts should be 0 (not crash)."""
        mock_conn = Mock()
        mock_conn.execute = Mock(
            side_effect=sqlite3.OperationalError("database is locked")
        )
        mock_life_os.db.get_connection = Mock(
            return_value=Mock(
                __enter__=Mock(return_value=mock_conn),
                __exit__=Mock(return_value=False),
            )
        )

        response = client.get("/api/system/intelligence")
        assert response.status_code == 200
        data = response.json()

        depth = data["user_model_depth"]
        for table in ("signal_profiles", "routines", "workflows",
                      "semantic_facts", "episodes"):
            assert depth[table] == 0
