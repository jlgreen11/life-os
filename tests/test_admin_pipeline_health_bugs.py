"""
Tests for two admin endpoint bug fixes:

1. /admin/pipeline-health — pending_notifications query used non-existent
   columns (``read``, ``dismissed``). The fix queries ``status IN ('pending',
   'delivered')`` to match the actual notifications schema.

2. /api/admin/semantic-facts/infer — called the synchronous
   ``run_all_inference()`` directly in an async handler, blocking the event
   loop. The fix wraps it in ``asyncio.to_thread()``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.app import create_web_app
from web.routes import register_routes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_notification(db, *, status: str = "pending") -> str:
    """Insert a notification with the given status and return its ID."""
    notif_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, status, created_at)
               VALUES (?, 'test', 'body', 'normal', ?, ?)""",
            (notif_id, status, ts),
        )
    return notif_id


def _make_app_with_real_db(db) -> TestClient:
    """Create a TestClient backed by a real DatabaseManager.

    Uses a MagicMock for LifeOS but wires ``db`` through so that SQL
    queries in the pipeline-health endpoint hit real SQLite tables.
    """
    app = FastAPI()
    life_os = MagicMock()
    life_os.db = db

    from storage.event_store import EventStore

    life_os.event_store = EventStore(db)
    life_os.notification_manager.get_pending.return_value = []
    life_os.task_manager.get_pending_tasks.return_value = []
    life_os.connectors = []
    life_os.event_bus.is_connected = True
    life_os.vector_store.get_stats.return_value = {"document_count": 0}
    life_os.user_model_store.get_signal_profile.return_value = None
    life_os.user_model_store.get_semantic_facts.return_value = []

    register_routes(app, life_os)
    return TestClient(app)


def _make_mock_life_os():
    """Build a minimal mock LifeOS for the semantic-facts endpoint test."""
    life_os = Mock()
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
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable",
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
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[
        {"category": "preference", "value": "coffee"},
        {"category": "expertise", "value": "python"},
    ])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
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
    life_os.semantic_fact_inferrer = Mock()
    life_os.semantic_fact_inferrer.run_all_inference = Mock()
    life_os.db = Mock()

    # Default mock connection returns (0,) for any probe
    default_cursor = MagicMock()
    default_cursor.fetchone.return_value = (0,)
    default_conn = MagicMock()
    default_conn.execute.return_value = default_cursor

    @contextmanager
    def _get_connection(db_name):
        yield default_conn

    life_os.db.get_connection = _get_connection

    return life_os


# ---------------------------------------------------------------------------
# Bug 1: pending_notifications query uses correct schema columns
# ---------------------------------------------------------------------------


class TestPendingNotificationsQuery:
    """Verify the pipeline-health endpoint counts notifications correctly.

    Before the fix, the query referenced ``read`` and ``dismissed`` columns
    that do not exist in the notifications table.  The schema uses a single
    ``status TEXT`` column with values like 'pending', 'delivered', 'read',
    'dismissed', 'suppressed', 'expired'.
    """

    def test_pending_notifications_no_error(self, db):
        """The endpoint returns an integer (not error dict) for pending_notifications."""
        client = _make_app_with_real_db(db)

        resp = client.get("/admin/pipeline-health")
        assert resp.status_code == 200
        data = resp.json()

        # Should be an integer, not an error dict
        assert isinstance(data["pipeline"]["pending_notifications"], int)

    def test_counts_pending_and_delivered(self, db):
        """Only 'pending' and 'delivered' statuses are counted as pending notifications."""
        # Insert notifications with various statuses
        _insert_notification(db, status="pending")
        _insert_notification(db, status="pending")
        _insert_notification(db, status="delivered")

        client = _make_app_with_real_db(db)

        resp = client.get("/admin/pipeline-health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["pipeline"]["pending_notifications"] == 3

    def test_excludes_read_and_dismissed(self, db):
        """Notifications with 'read', 'dismissed', 'suppressed', 'expired' are excluded."""
        # Insert 2 that should be counted
        _insert_notification(db, status="pending")
        _insert_notification(db, status="delivered")
        # Insert 4 that should NOT be counted
        _insert_notification(db, status="read")
        _insert_notification(db, status="dismissed")
        _insert_notification(db, status="suppressed")
        _insert_notification(db, status="expired")

        client = _make_app_with_real_db(db)

        resp = client.get("/admin/pipeline-health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["pipeline"]["pending_notifications"] == 2

    def test_zero_when_all_resolved(self, db):
        """Returns 0 when all notifications are in terminal states."""
        _insert_notification(db, status="read")
        _insert_notification(db, status="dismissed")
        _insert_notification(db, status="expired")

        client = _make_app_with_real_db(db)

        resp = client.get("/admin/pipeline-health")
        assert resp.status_code == 200
        data = resp.json()

        assert data["pipeline"]["pending_notifications"] == 0


# ---------------------------------------------------------------------------
# Bug 2: semantic-facts/infer endpoint uses asyncio.to_thread
# ---------------------------------------------------------------------------


class TestSemanticFactsInferEndpoint:
    """Verify the /api/admin/semantic-facts/infer endpoint works correctly.

    Before the fix, the synchronous ``run_all_inference()`` was called
    directly inside the async handler, blocking the FastAPI event loop.
    The fix wraps it in ``asyncio.to_thread()``.
    """

    def test_endpoint_returns_200(self):
        """The endpoint returns 200 and a success response."""
        mock_life_os = _make_mock_life_os()
        app = create_web_app(mock_life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/semantic-facts/infer")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "success"
        assert data["message"] == "Semantic fact inference completed"

    def test_run_all_inference_is_called(self):
        """The endpoint calls run_all_inference exactly once."""
        mock_life_os = _make_mock_life_os()
        app = create_web_app(mock_life_os)
        client = TestClient(app)

        client.post("/api/admin/semantic-facts/infer")

        mock_life_os.semantic_fact_inferrer.run_all_inference.assert_called_once()

    def test_returns_facts_by_category(self):
        """The response includes correct fact counts grouped by category."""
        mock_life_os = _make_mock_life_os()
        app = create_web_app(mock_life_os)
        client = TestClient(app)

        resp = client.post("/api/admin/semantic-facts/infer")
        data = resp.json()

        assert data["total_facts"] == 2
        assert data["facts_by_category"]["preference"] == 1
        assert data["facts_by_category"]["expertise"] == 1
