"""
Tests for insight endpoint fallback behavior when user_model.db is corrupted.

When user_model.db is unreadable, both /api/insights/summary and /api/insights
should fall back to in-memory insights produced by generate_insights() rather
than returning empty results.

Test patterns follow tests/web/test_insights_error_handling.py.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from services.insight_engine.models import Insight
from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_insights() -> list[Insight]:
    """Create sample Insight objects mimicking generate_insights() output."""
    return [
        Insight(
            id="ins-1",
            type="relationship_intelligence",
            summary="You haven't contacted Alice in 2 weeks",
            confidence=0.85,
            category="contact_gap",
            entity="Alice",
            evidence=["last contact: 14 days ago"],
            feedback=None,
        ),
        Insight(
            id="ins-2",
            type="actionable_alert",
            summary="Unusual spending spike detected this week",
            confidence=0.72,
            category="spending",
            entity=None,
            evidence=["$350 above weekly average"],
            feedback=None,
        ),
    ]


def _make_mock_connection():
    """Create a default mock connection for non-user_model DB queries."""
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    cursor.fetchall.return_value = []
    conn = MagicMock()
    conn.execute.return_value = cursor
    return conn


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for insight degraded-mode tests."""
    life_os = Mock()

    default_conn = _make_mock_connection()

    @contextmanager
    def _get_connection(db_name):
        yield default_conn

    life_os.db = Mock()
    life_os.db.get_connection = _get_connection

    # Insight engine — returns sample insights by default
    life_os.insight_engine = Mock()
    life_os.insight_engine.generate_insights = AsyncMock(
        return_value=_make_sample_insights()
    )

    # Other services required by create_web_app
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
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6,
            trend="stable",
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


def _corrupt_user_model_db(mock_life_os):
    """Configure mock so user_model DB raises on access, others work normally."""
    default_conn = _make_mock_connection()

    @contextmanager
    def _get_connection(db_name):
        if db_name == "user_model":
            raise RuntimeError("database disk image is malformed")
        yield default_conn

    mock_life_os.db.get_connection = _get_connection


# ---------------------------------------------------------------------------
# /api/insights/summary — degraded mode fallback
# ---------------------------------------------------------------------------


def test_insights_summary_falls_back_to_generated_when_db_corrupted(mock_life_os):
    """/api/insights/summary returns in-memory insights when user_model.db read fails."""
    _corrupt_user_model_db(mock_life_os)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    data = response.json()
    assert len(data["insights"]) == 2
    assert "generated_at" in data

    # Verify generate_insights was called
    mock_life_os.insight_engine.generate_insights.assert_called_once()


def test_insights_summary_fallback_has_correct_structure(mock_life_os):
    """Fallback insights from /api/insights/summary have the expected JSON fields."""
    _corrupt_user_model_db(mock_life_os)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights/summary")
    data = response.json()

    expected_fields = {"id", "type", "summary", "confidence", "category",
                       "entity", "evidence", "feedback", "created_at"}

    for insight in data["insights"]:
        assert expected_fields.issubset(set(insight.keys())), (
            f"Missing fields: {expected_fields - set(insight.keys())}"
        )

    # Check specific values from the first sample insight
    ins = data["insights"][0]
    assert ins["id"] == "ins-1"
    assert ins["type"] == "relationship_intelligence"
    assert ins["summary"] == "You haven't contacted Alice in 2 weeks"
    assert ins["confidence"] == 0.85
    assert ins["category"] == "contact_gap"
    assert ins["entity"] == "Alice"
    assert isinstance(ins["evidence"], list)
    assert ins["evidence"] == ["last contact: 14 days ago"]


def test_insights_summary_returns_empty_when_both_db_and_generate_fail(mock_life_os):
    """/api/insights/summary returns empty list if both DB and generate_insights fail."""
    _corrupt_user_model_db(mock_life_os)
    mock_life_os.insight_engine.generate_insights = AsyncMock(
        side_effect=RuntimeError("correlator crashed")
    )

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights/summary")
    assert response.status_code == 200

    data = response.json()
    assert data["insights"] == []
    assert "generated_at" in data


# ---------------------------------------------------------------------------
# /api/insights — degraded mode fallback
# ---------------------------------------------------------------------------


def test_insights_list_falls_back_to_generated_when_db_corrupted(mock_life_os):
    """/api/insights returns in-memory insights when user_model.db is corrupted."""
    _corrupt_user_model_db(mock_life_os)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights")
    assert response.status_code == 200

    data = response.json()
    # Should have fallback insights AND the error field
    assert len(data["insights"]) == 2
    assert "error" in data
    assert "malformed" in data["error"]


def test_insights_list_fallback_has_correct_structure(mock_life_os):
    """Fallback insights from /api/insights have the expected JSON fields."""
    _corrupt_user_model_db(mock_life_os)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights")
    data = response.json()

    expected_fields = {"id", "type", "summary", "confidence", "category",
                       "entity", "evidence", "feedback", "created_at"}

    for insight in data["insights"]:
        assert expected_fields.issubset(set(insight.keys())), (
            f"Missing fields: {expected_fields - set(insight.keys())}"
        )


def test_insights_list_respects_limit_in_fallback(mock_life_os):
    """/api/insights respects limit param when falling back to generate_insights()."""
    _corrupt_user_model_db(mock_life_os)

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights?limit=1")
    assert response.status_code == 200

    data = response.json()
    # Only 1 insight despite 2 being generated
    assert len(data["insights"]) == 1


def test_insights_list_returns_empty_when_both_db_and_generate_fail(mock_life_os):
    """/api/insights returns empty list with error if both DB and generate_insights fail."""
    _corrupt_user_model_db(mock_life_os)
    mock_life_os.insight_engine.generate_insights = AsyncMock(
        side_effect=RuntimeError("correlator crashed")
    )

    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/insights")
    assert response.status_code == 200

    data = response.json()
    assert data["insights"] == []
    assert "error" in data
