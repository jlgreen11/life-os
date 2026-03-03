"""
Tests for GET /api/insights error handling and _classify_notification_source async behavior.

Bug 1: GET /api/insights must return {"insights": [], "error": "..."} when
user_model.db is corrupted, instead of raising an unhandled 500.

Bug 2: _classify_notification_source must not block the async event loop
when performing synchronous SQLite queries.

Test patterns follow tests/web/test_pipeline_health.py (mock LifeOS, TestClient).
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, MagicMock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_cursor(*return_values):
    """Create a mock cursor that returns the given values sequentially."""
    cursor = MagicMock()
    if len(return_values) == 1:
        cursor.fetchone.return_value = return_values[0]
    else:
        cursor.fetchone.side_effect = list(return_values)
    return cursor


def _make_mock_connection(cursor):
    """Wrap a mock cursor in a mock connection whose execute() returns it."""
    conn = MagicMock()
    conn.execute.return_value = cursor
    return conn


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for insights and notification tests."""
    life_os = Mock()

    # Default mock cursor returns (0,) for probes
    default_cursor = _make_mock_cursor((0,))
    default_conn = _make_mock_connection(default_cursor)

    @contextmanager
    def _get_connection(db_name):
        yield default_conn

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection

    # Services required by create_web_app / other routes
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")
    life_os.notification_manager = Mock()
    life_os.notification_manager.dismiss = AsyncMock()
    life_os.notification_manager.mark_acted_on = AsyncMock()
    life_os.notification_manager.mark_read = AsyncMock()
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
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
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
    life_os.source_weight_manager = Mock()
    life_os.source_weight_manager.classify_event = Mock(return_value=None)
    life_os.source_weight_manager.record_dismissal = Mock()
    life_os.source_weight_manager.record_engagement = Mock()

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Bug 1: GET /api/insights error handling
# ---------------------------------------------------------------------------


def test_insights_returns_empty_with_error_on_corrupted_db(mock_life_os):
    """GET /api/insights returns 200 with error field when user_model.db is corrupted."""
    default_cursor = _make_mock_cursor((0,))
    default_conn = _make_mock_connection(default_cursor)

    @contextmanager
    def _get_connection(db_name):
        if db_name == "user_model":
            raise RuntimeError("database disk image is malformed")
        yield default_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights")
    assert response.status_code == 200

    data = response.json()
    assert data["insights"] == []
    assert "error" in data
    assert "malformed" in data["error"]


def test_insights_returns_data_when_db_healthy(mock_life_os):
    """GET /api/insights returns insight rows when user_model.db is healthy."""
    # Create a mock row that behaves like a sqlite3.Row (dict-like)
    mock_row = MagicMock()
    mock_row.__iter__ = Mock(return_value=iter([]))
    row_dict = {
        "id": "insight-1",
        "type": "relationship_intelligence",
        "summary": "You haven't contacted Alice in 2 weeks",
        "confidence": 0.85,
        "category": "contact_gap",
        "entity": "Alice",
        "evidence": '["last contact: 14 days ago"]',
        "feedback": None,
        "created_at": "2026-03-01T10:00:00Z",
    }

    def _dict_side_effect():
        return dict(row_dict)

    mock_row.__getitem__ = lambda self, key: row_dict[key]
    mock_row.keys = lambda: row_dict.keys()

    # Create a cursor that returns our mock row for the insights query
    def _make_cursor_for_query(sql, *args):
        cursor = MagicMock()
        if "insights" in sql and "ORDER BY" in sql:
            # This is the list_insights query — return rows via fetchall
            row_obj = MagicMock()
            row_obj.__iter__ = Mock(return_value=iter(row_dict.items()))
            row_obj.keys = Mock(return_value=row_dict.keys())
            # Make dict(row_obj) work by implementing the Mapping protocol
            dict_copy = dict(row_dict)

            class FakeRow:
                """Fake sqlite3.Row that supports dict() conversion."""

                def __init__(self, data):
                    self._data = data

                def keys(self):
                    return self._data.keys()

                def __getitem__(self, key):
                    return self._data[key]

                def __iter__(self):
                    return iter(self._data)

                def __len__(self):
                    return len(self._data)

            cursor.fetchall.return_value = [FakeRow(dict_copy)]
        else:
            cursor.fetchone.return_value = (0,)
            cursor.fetchall.return_value = []
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = _make_cursor_for_query

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights")
    assert response.status_code == 200

    data = response.json()
    assert "error" not in data
    assert len(data["insights"]) == 1
    insight = data["insights"][0]
    assert insight["id"] == "insight-1"
    assert insight["summary"] == "You haven't contacted Alice in 2 weeks"
    # Evidence should be parsed from JSON string to list
    assert isinstance(insight["evidence"], list)
    assert insight["evidence"] == ["last contact: 14 days ago"]


def test_insights_respects_limit_param(mock_life_os):
    """GET /api/insights passes the limit param to the SQL query."""
    captured_queries = []

    def _capture_execute(sql, *args):
        captured_queries.append((sql, args))
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = (0,)
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = _capture_execute

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights?limit=5")
    assert response.status_code == 200

    # Find the insights query in captured queries
    insights_queries = [(sql, args) for sql, args in captured_queries if "insights" in sql and "ORDER BY" in sql]
    assert len(insights_queries) >= 1
    # The limit param should be (5,) in the args
    sql, args = insights_queries[0]
    assert args == ((5,),) or (len(args) > 0 and args[0] == (5,))


# ---------------------------------------------------------------------------
# Bug 2: _classify_notification_source is now async (non-blocking)
# ---------------------------------------------------------------------------


def test_dismiss_notification_calls_classify_without_blocking(mock_life_os):
    """POST /api/notifications/{id}/dismiss uses async classify (no event loop block).

    This test verifies the dismiss endpoint works end-to-end with the async
    _classify_notification_source. If the function were still synchronous,
    it would still work in TestClient, but the refactor ensures it won't
    block the event loop in production.
    """
    # Set up mock to return a notification with a domain
    notif_row = MagicMock()
    notif_row.__getitem__ = lambda self, key: {
        "source_event_id": None,
        "domain": "email",
    }[key]

    def _make_cursor_for_query(sql, *args):
        cursor = MagicMock()
        if "notifications" in sql and "SELECT" in sql:
            cursor.fetchone.return_value = notif_row
        else:
            cursor.fetchone.return_value = (0,)
            cursor.fetchall.return_value = []
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = _make_cursor_for_query

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.post("/api/notifications/notif-1/dismiss")
    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"

    # Verify source weight was updated for the email domain
    mock_life_os.source_weight_manager.record_dismissal.assert_called_once_with("email.work")


def test_act_on_notification_calls_classify_without_blocking(mock_life_os):
    """POST /api/notifications/{id}/act uses async classify (no event loop block)."""
    notif_row = MagicMock()
    notif_row.__getitem__ = lambda self, key: {
        "source_event_id": None,
        "domain": "messaging",
    }[key]

    def _make_cursor_for_query(sql, *args):
        cursor = MagicMock()
        if "notifications" in sql and "SELECT" in sql:
            cursor.fetchone.return_value = notif_row
        else:
            cursor.fetchone.return_value = (0,)
            cursor.fetchall.return_value = []
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = _make_cursor_for_query

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.post("/api/notifications/notif-2/act")
    assert response.status_code == 200
    assert response.json()["status"] == "acted_on"

    # Verify source weight was updated for the messaging domain
    mock_life_os.source_weight_manager.record_engagement.assert_called_once_with("messaging.direct")
