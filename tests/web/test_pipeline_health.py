"""
Tests for the /admin/pipeline-health diagnostic endpoint.

The pipeline-health endpoint returns a comprehensive snapshot of all database
and pipeline component statuses.  Each database probe is isolated so that a
corrupted or unavailable database never blocks healthy ones from reporting.

Test patterns follow tests/web/test_badge_counts.py (mock LifeOS, TestClient).
"""

from contextlib import contextmanager
from unittest.mock import Mock, MagicMock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_cursor(*return_values):
    """Create a mock cursor that returns the given values sequentially.

    Each call to ``fetchone()`` pops the next value; when only one value is
    provided it is reused for every call.
    """
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
    """Minimal LifeOS mock wired for pipeline-health tests.

    The ``db.get_connection`` context manager returns a mock connection whose
    ``execute().fetchone()`` returns ``(0,)`` by default (zero-count rows).
    """
    life_os = Mock()

    # Default mock cursor returns (0,) for every probe
    default_cursor = _make_mock_cursor((0,))
    default_conn = _make_mock_connection(default_cursor)

    @contextmanager
    def _get_connection(db_name):
        yield default_conn

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection

    # Services required by create_web_app / other routes (not under test)
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

    return life_os


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Endpoint existence and basic shape
# ---------------------------------------------------------------------------


def test_pipeline_health_returns_200(client):
    """GET /admin/pipeline-health returns 200."""
    response = client.get("/admin/pipeline-health")
    assert response.status_code == 200


def test_pipeline_health_has_all_sections(client):
    """Response contains databases, pipeline, and source_weights sections."""
    data = client.get("/admin/pipeline-health").json()
    assert "databases" in data
    assert "pipeline" in data
    assert "source_weights" in data


def test_pipeline_health_databases_all_ok(client):
    """All 5 databases report 'ok' when probes succeed."""
    data = client.get("/admin/pipeline-health").json()
    for db_name in ("events", "user_model", "state", "preferences", "entities"):
        assert db_name in data["databases"], f"Missing database '{db_name}'"
        assert data["databases"][db_name]["status"] == "ok"
        assert "probe_count" in data["databases"][db_name]


# ---------------------------------------------------------------------------
# Corrupted / unavailable database handling
# ---------------------------------------------------------------------------


def test_corrupted_db_shows_error_not_500(mock_life_os):
    """When user_model DB is corrupted, the endpoint still returns 200.

    The user_model entry in databases should show 'error' status, while
    all other databases remain 'ok'.
    """
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

    response = client.get("/admin/pipeline-health")
    assert response.status_code == 200

    data = response.json()
    # user_model should show error
    assert data["databases"]["user_model"]["status"] == "error"
    assert "malformed" in data["databases"]["user_model"]["error"]

    # Other databases should still be ok
    for db_name in ("events", "state", "preferences", "entities"):
        assert data["databases"][db_name]["status"] == "ok"


def test_corrupted_db_pipeline_metrics_show_errors(mock_life_os):
    """Pipeline metrics that depend on a corrupted DB show error dicts."""
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

    data = client.get("/admin/pipeline-health").json()
    pipeline = data["pipeline"]

    # All user_model-backed pipeline metrics should show errors
    for metric in ("signal_profiles", "episodes", "predictions"):
        assert isinstance(pipeline[metric], dict), f"Expected error dict for {metric}"
        assert "error" in pipeline[metric]

    # state-backed metrics should still be integers (healthy)
    assert isinstance(pipeline["pending_notifications"], int)
    assert isinstance(pipeline["pending_tasks"], int)


# ---------------------------------------------------------------------------
# Pipeline section reports counts correctly
# ---------------------------------------------------------------------------


def test_pipeline_section_reports_counts(mock_life_os):
    """Pipeline section returns integer counts from each DB when healthy."""
    counts_by_query = {
        "signal_profiles": 5,
        "episodes": 12,
        "predictions": 3,
        "notifications": 7,
        "tasks": 2,
        "events": 42,
    }

    # Map SQL fragment to return value to differentiate queries
    def _make_cursor_for_query(sql):
        cursor = MagicMock()
        if "signal_profiles" in sql:
            cursor.fetchone.return_value = (counts_by_query["signal_profiles"],)
        elif "episodes" in sql:
            cursor.fetchone.return_value = (counts_by_query["episodes"],)
        elif "predictions" in sql:
            cursor.fetchone.return_value = (counts_by_query["predictions"],)
        elif "notifications" in sql:
            cursor.fetchone.return_value = (counts_by_query["notifications"],)
        elif "tasks" in sql and "completed_at" in sql:
            cursor.fetchone.return_value = (counts_by_query["tasks"],)
        elif "events" in sql and "timestamp" in sql:
            cursor.fetchone.return_value = (counts_by_query["events"],)
        elif "source_weights" in sql:
            cursor.fetchone.return_value = (10, 4, 2)
        else:
            cursor.fetchone.return_value = (0,)
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = lambda sql, *args: _make_cursor_for_query(sql)

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    data = client.get("/admin/pipeline-health").json()
    pipeline = data["pipeline"]

    assert pipeline["signal_profiles"] == 5
    assert pipeline["episodes"] == 12
    assert pipeline["predictions"] == 3
    assert pipeline["pending_notifications"] == 7
    assert pipeline["pending_tasks"] == 2
    assert pipeline["events_last_24h"] == 42


# ---------------------------------------------------------------------------
# Source weights section
# ---------------------------------------------------------------------------


def test_source_weights_populated(mock_life_os):
    """source_weights section reports total, with_user_set, and with_drift."""
    def _make_cursor_for_query(sql):
        cursor = MagicMock()
        if "source_weights" in sql:
            cursor.fetchone.return_value = (15, 8, 3)
        else:
            cursor.fetchone.return_value = (0,)
        return cursor

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = lambda sql, *args: _make_cursor_for_query(sql)

    @contextmanager
    def _get_connection(db_name):
        yield mock_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    data = client.get("/admin/pipeline-health").json()
    sw = data["source_weights"]
    assert sw["total"] == 15
    assert sw["with_user_set"] == 8
    assert sw["with_drift"] == 3


def test_source_weights_error_on_corrupted_preferences(mock_life_os):
    """source_weights shows error dict when preferences DB is corrupted."""
    default_cursor = _make_mock_cursor((0,))
    default_conn = _make_mock_connection(default_cursor)

    @contextmanager
    def _get_connection(db_name):
        if db_name == "preferences":
            raise RuntimeError("preferences DB corrupted")
        yield default_conn

    mock_life_os.db.get_connection = _get_connection

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    data = client.get("/admin/pipeline-health").json()
    assert "error" in data["source_weights"]
    assert "corrupted" in data["source_weights"]["error"]
