"""
Tests for GET /api/admin/data-quality — real-time system diagnostics.

Verifies that the endpoint returns all seven diagnostic sections, handles
database corruption gracefully, and produces correct counts after events
are stored.
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from storage.event_store import EventStore
from storage.manager import DatabaseManager
from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers — build a mock life_os with real DB connections
# ---------------------------------------------------------------------------

def _make_life_os(db: DatabaseManager):
    """Build a mock life_os wired to a real DatabaseManager.

    Non-DB attributes are stubbed so ``register_routes`` can initialise
    without errors.
    """
    life_os = Mock()
    life_os.config = {}
    life_os.db = db
    life_os.event_store = EventStore(db)
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0})
    life_os.vector_store.search = Mock(return_value=[])
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.5, stress_level=0.3, social_battery=0.5,
        cognitive_load=0.3, emotional_valence=0.5, confidence=0.5, trend="stable",
    ))
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={})
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={})
    life_os.ai_engine = Mock()
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    return life_os


def _make_client(db: DatabaseManager) -> tuple[TestClient, Mock]:
    """Create a TestClient and return the life_os mock alongside it."""
    life_os = _make_life_os(db)
    app = create_web_app(life_os)
    return TestClient(app), life_os


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataQualityEndpoint:
    """Tests for GET /api/admin/data-quality."""

    def test_returns_200(self, db):
        """Endpoint exists and responds with 200."""
        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        assert resp.status_code == 200

    def test_returns_all_sections(self, db):
        """Response contains all seven diagnostic sections plus generated_at."""
        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        expected_keys = [
            "generated_at",
            "event_stats",
            "signal_profiles",
            "prediction_pipeline",
            "source_weight_staleness",
            "connector_health",
            "task_summary",
            "notification_summary",
        ]
        for key in expected_keys:
            assert key in data, f"Missing section: {key}"

    def test_generated_at_is_iso_format(self, db):
        """The generated_at field is a valid ISO 8601 timestamp."""
        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()
        # Should parse without error
        datetime.fromisoformat(data["generated_at"])

    def test_event_stats_counts(self, db):
        """Event stats reflect events actually stored in the database."""
        # Store a few test events
        event_store = EventStore(db)
        for i in range(3):
            event_store.store_event({
                "id": f"dq-test-event-{i}",
                "type": "email_received",
                "source": "email",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "priority": "normal",
                "payload": {"subject": f"Test {i}"},
                "metadata": {},
            })
        # Store one more with a different type
        event_store.store_event({
            "id": "dq-test-event-other",
            "type": "calendar_event_created",
            "source": "calendar",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "priority": "normal",
            "payload": {"title": "Meeting"},
            "metadata": {},
        })

        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        stats = data["event_stats"]
        assert stats["total"] >= 4
        assert stats["last_24h"] >= 4
        assert len(stats["top_types"]) >= 2
        assert len(stats["sources"]) >= 2

        # Verify top_types includes our event types
        type_names = [t["type"] for t in stats["top_types"]]
        assert "email_received" in type_names
        assert "calendar_event_created" in type_names

    def test_handles_corrupted_user_model(self, db):
        """When user_model.db queries fail, other sections still return data."""
        life_os = _make_life_os(db)

        # Monkey-patch get_connection to fail for user_model
        original_get_connection = db.get_connection

        @contextmanager
        def _patched_get_connection(db_name):
            if db_name == "user_model":
                raise Exception("database disk image is malformed")
            with original_get_connection(db_name) as conn:
                yield conn

        life_os.db.get_connection = _patched_get_connection

        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/data-quality")
        assert resp.status_code == 200

        data = resp.json()

        # user_model sections should have errors
        assert "error" in data["signal_profiles"]
        assert "error" in data["prediction_pipeline"]

        # Other sections should still work
        assert "error" not in data["event_stats"]
        assert "error" not in data["task_summary"]
        assert "error" not in data["notification_summary"]

    def test_source_weights_included(self, db):
        """Source weight data appears in the response (empty list for fresh DB)."""
        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        # Fresh DB has no source weights — should be an empty list (not an error)
        weights = data["source_weight_staleness"]
        assert isinstance(weights, list)

    def test_source_weights_with_data(self, db):
        """Source weight entries are returned with never_updated flag."""
        # Insert a source weight directly (category and label are NOT NULL)
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO source_weights "
                "(source_key, category, label, user_weight, ai_drift, ai_updated_at, "
                "interactions, engagements, dismissals) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("email:inbox", "messaging", "Email Inbox", 1.0, 0.1, None, 10, 5, 2),
            )

        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        weights = data["source_weight_staleness"]
        assert len(weights) >= 1

        email_weight = next((w for w in weights if w["source_key"] == "email:inbox"), None)
        assert email_weight is not None
        assert email_weight["never_updated"] is True
        assert email_weight["interactions"] == 10

    def test_task_summary_counts(self, db):
        """Task summary reflects tasks in the database."""
        import uuid

        with db.get_connection("state") as conn:
            for status in ["pending", "pending", "completed"]:
                conn.execute(
                    "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), f"Task {status}", status, datetime.now(timezone.utc).isoformat()),
                )

        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        summary = data["task_summary"]
        assert summary.get("pending", 0) >= 2
        assert summary.get("completed", 0) >= 1

    def test_connector_health_empty(self, db):
        """With no connectors, connector_health is an empty dict."""
        client, _ = _make_client(db)
        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        assert data["connector_health"] == {}

    def test_connector_health_with_connectors(self, db):
        """Connector health reports status for each registered connector."""
        life_os = _make_life_os(db)

        # Add a mock connector
        mock_connector = Mock()
        mock_connector.CONNECTOR_ID = "test_connector"
        mock_connector.health_check = AsyncMock(return_value={"status": "ok", "connector": "test_connector"})
        life_os.connectors = [mock_connector]

        app = create_web_app(life_os)
        client = TestClient(app)

        resp = client.get("/api/admin/data-quality")
        data = resp.json()

        assert "test_connector" in data["connector_health"]
        assert data["connector_health"]["test_connector"]["status"] == "ok"
