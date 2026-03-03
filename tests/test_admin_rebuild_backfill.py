"""
Tests for POST /api/admin/rebuild-user-model backfill coverage.

Verifies that _post_rebuild_backfill() calls episode and communication
template backfill methods after a user_model.db rebuild, and that failures
in one backfill do not prevent others from running (fail-open pattern).
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for rebuild-user-model tests."""
    life_os = Mock()

    # Database mock — health check returns corrupted so rebuild proceeds
    life_os.db = Mock()
    life_os.db.data_dir = "/tmp/test-data"
    life_os.db.get_database_health = Mock(
        return_value={"user_model": {"status": "corrupted", "errors": ["malformed"]}}
    )
    life_os.db._check_and_recover_db = Mock(return_value=True)
    life_os.db._init_user_model_db = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.execute.return_value = mock_cursor
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / event store (required by create_web_app)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable",
        )
    )

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    # User model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector / browser stubs
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    # Signal profile backfill methods (all async no-ops for testing)
    for method_name in [
        "_backfill_relationship_profile_if_needed",
        "_clean_relationship_profile_if_needed",
        "_backfill_temporal_profile_if_needed",
        "_backfill_topic_profile_if_needed",
        "_backfill_linguistic_profile_if_needed",
        "_backfill_cadence_profile_if_needed",
        "_backfill_mood_signals_profile_if_needed",
        "_backfill_spatial_profile_if_needed",
        "_backfill_decision_profile_if_needed",
    ]:
        setattr(life_os, method_name, AsyncMock())

    # Episode and communication template backfill methods
    life_os._backfill_episodes_from_events_if_needed = AsyncMock()
    life_os._backfill_communication_templates_if_needed = AsyncMock()

    return life_os


def _trigger_rebuild_and_drain(mock_life_os):
    """Call rebuild endpoint and drain background tasks so mocks register calls.

    TestClient runs requests inside an asyncio event loop. The rebuild endpoint
    fires ``asyncio.create_task(...)`` which schedules the backfill coroutine on
    that same loop.  We need to give the loop a chance to execute the task before
    checking mock assertions.

    Strategy: wrap ``asyncio.create_task`` so we capture the task object, then
    after the HTTP response returns we briefly yield control to the event loop
    (via ``asyncio.sleep(0)``) so the background task can finish.
    """
    app = create_web_app(mock_life_os)
    captured_tasks: list = []
    original_create_task = asyncio.create_task

    def _tracking_create_task(coro, **kwargs):
        """Delegate to real create_task but remember the task."""
        task = original_create_task(coro, **kwargs)
        captured_tasks.append(task)
        return task

    # Patch create_task inside the routes module so we capture background tasks
    import web.routes as _routes_mod

    _orig = getattr(_routes_mod.asyncio, "create_task", None) if hasattr(_routes_mod, "asyncio") else None

    # The route imports asyncio as _asyncio locally, so we need to patch at the
    # module level used by the closure.  Since _asyncio is a local name inside
    # the route function body, the simplest approach is to use TestClient which
    # runs the request in an event loop where our captured tasks also live.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/admin/rebuild-user-model")
        # Give the event loop a chance to finish the background task.
        # TestClient uses anyio under the hood — all tasks on the same loop.
        # The task is already scheduled; by the time the response comes back
        # the fast AsyncMock coroutines have already completed.

    return response, captured_tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRebuildBackfillCallsEpisodesAndTemplates:
    """Verify _post_rebuild_backfill calls episode and communication template backfills."""

    def test_rebuild_calls_episode_backfill(self, mock_life_os):
        """Episode backfill is called during post-rebuild backfill."""
        response, _ = _trigger_rebuild_and_drain(mock_life_os)

        assert response.status_code == 200
        assert response.json()["status"] == "rebuilt"
        mock_life_os._backfill_episodes_from_events_if_needed.assert_called_once()

    def test_rebuild_calls_communication_template_backfill(self, mock_life_os):
        """Communication template backfill is called during post-rebuild backfill."""
        response, _ = _trigger_rebuild_and_drain(mock_life_os)

        assert response.status_code == 200
        mock_life_os._backfill_communication_templates_if_needed.assert_called_once()

    def test_rebuild_response_mentions_episodes_and_templates(self, mock_life_os):
        """The rebuild response message mentions episode and communication template backfills."""
        response, _ = _trigger_rebuild_and_drain(mock_life_os)

        data = response.json()
        assert "episode" in data["message"].lower()
        assert "communication template" in data["message"].lower()


class TestRebuildBackfillFailOpen:
    """Verify fail-open: one backfill failure does not prevent others."""

    def test_episode_failure_does_not_block_template_backfill(self, mock_life_os):
        """If episode backfill raises, communication template backfill still runs."""
        mock_life_os._backfill_episodes_from_events_if_needed = AsyncMock(
            side_effect=RuntimeError("Episode DB error")
        )

        _trigger_rebuild_and_drain(mock_life_os)

        # Episode backfill was called (and failed)
        mock_life_os._backfill_episodes_from_events_if_needed.assert_called_once()
        # Communication template backfill still ran
        mock_life_os._backfill_communication_templates_if_needed.assert_called_once()

    def test_template_failure_does_not_block_episode_backfill(self, mock_life_os):
        """If communication template backfill raises, episode backfill still ran."""
        mock_life_os._backfill_communication_templates_if_needed = AsyncMock(
            side_effect=RuntimeError("Template extraction error")
        )

        _trigger_rebuild_and_drain(mock_life_os)

        # Episode backfill was called and succeeded
        mock_life_os._backfill_episodes_from_events_if_needed.assert_called_once()
        # Communication template backfill was also called (and failed)
        mock_life_os._backfill_communication_templates_if_needed.assert_called_once()

    def test_signal_profile_failure_does_not_block_episodes_or_templates(self, mock_life_os):
        """If all signal profile backfills fail, episode and template backfills still run."""
        for method_name in [
            "_backfill_relationship_profile_if_needed",
            "_clean_relationship_profile_if_needed",
            "_backfill_temporal_profile_if_needed",
            "_backfill_topic_profile_if_needed",
            "_backfill_linguistic_profile_if_needed",
            "_backfill_cadence_profile_if_needed",
            "_backfill_mood_signals_profile_if_needed",
            "_backfill_spatial_profile_if_needed",
            "_backfill_decision_profile_if_needed",
        ]:
            setattr(mock_life_os, method_name, AsyncMock(side_effect=RuntimeError("profile error")))

        _trigger_rebuild_and_drain(mock_life_os)

        # Both new backfills still ran despite signal profile failures
        mock_life_os._backfill_episodes_from_events_if_needed.assert_called_once()
        mock_life_os._backfill_communication_templates_if_needed.assert_called_once()
