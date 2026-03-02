"""
Tests for async-safety fixes in web/routes.py.

Verifies that /api/status and /api/admin/connectors/google/auth do not
block the FastAPI event loop by using asyncio.to_thread for synchronous
calls.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Minimal mock LifeOS with the services needed by /api/status and Google OAuth."""
    life_os = Mock()

    # Event store
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=42)

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 10, "dimensions": 384})

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": ["likes coffee"]})

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 3, "delivered": 50})

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 7})

    # Connector management (needed by Google OAuth)
    life_os.get_connector_config = Mock(return_value={})

    # Event bus, connectors, and other services the app factory may reference
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()
    life_os.connectors = []
    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=Mock()), __exit__=Mock())
    )
    life_os.db.get_database_health = Mock(return_value={})
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="")
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})

    return life_os


@pytest.fixture
def app(mock_life_os):
    """Create a FastAPI app with mocked dependencies."""
    return create_web_app(mock_life_os)


# ---------------------------------------------------------------------------
# /api/status — async-safe with asyncio.gather + to_thread
# ---------------------------------------------------------------------------


async def test_status_returns_all_keys(app, mock_life_os):
    """Verify /api/status returns all 5 expected keys with correct values."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/status")

    assert response.status_code == 200
    data = response.json()
    assert data["event_count"] == 42
    assert data["vector_store"] == {"total": 10, "dimensions": 384}
    assert data["user_model"] == {"facts": ["likes coffee"]}
    assert data["notification_stats"] == {"pending": 3, "delivered": 50}
    assert data["feedback_summary"] == {"total": 7}


async def test_status_calls_all_services(app, mock_life_os):
    """Verify /api/status invokes every underlying service call exactly once."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/api/status")

    mock_life_os.event_store.get_event_count.assert_called_once()
    mock_life_os.vector_store.get_stats.assert_called_once()
    mock_life_os.signal_extractor.get_user_summary.assert_called_once()
    mock_life_os.notification_manager.get_stats.assert_called_once()
    mock_life_os.feedback_collector.get_feedback_summary.assert_called_once()


async def test_status_does_not_block_event_loop(app, mock_life_os):
    """Verify /api/status uses asyncio.to_thread (doesn't block the loop).

    We patch asyncio.to_thread to confirm it's being called for each
    synchronous service method, proving the calls are offloaded to threads.
    """
    original_to_thread = asyncio.to_thread

    calls = []

    async def tracking_to_thread(func, *args, **kwargs):
        """Track what functions are passed to to_thread."""
        calls.append(func)
        return await original_to_thread(func, *args, **kwargs)

    transport = httpx.ASGITransport(app=app)
    with patch("asyncio.to_thread", side_effect=tracking_to_thread):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/status")

    assert response.status_code == 200
    # All 5 sync calls should have gone through to_thread
    assert len(calls) >= 5
    expected_funcs = {
        mock_life_os.event_store.get_event_count,
        mock_life_os.vector_store.get_stats,
        mock_life_os.signal_extractor.get_user_summary,
        mock_life_os.notification_manager.get_stats,
        mock_life_os.feedback_collector.get_feedback_summary,
    }
    assert expected_funcs.issubset(set(calls))


# ---------------------------------------------------------------------------
# /api/admin/connectors/google/auth — missing credentials returns 400
# ---------------------------------------------------------------------------


async def test_google_auth_missing_credentials(app, mock_life_os):
    """Verify Google OAuth returns 400 when credentials file doesn't exist."""
    mock_life_os.get_connector_config.return_value = {
        "credentials_file": "/nonexistent/path/credentials.json",
        "token_file": "/tmp/test_token.json",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/admin/connectors/google/auth")

    assert response.status_code == 400
    data = response.json()
    assert "Credentials file not found" in data["detail"]


async def test_google_auth_default_paths_missing(app, mock_life_os):
    """Verify Google OAuth returns 400 when using default config paths that don't exist.

    Patches os.path.exists so the test never accidentally finds a real
    credentials file on disk.
    """
    mock_life_os.get_connector_config.return_value = {}

    transport = httpx.ASGITransport(app=app)
    with patch("os.path.exists", return_value=False):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/admin/connectors/google/auth")

    assert response.status_code == 400
    data = response.json()
    assert "Credentials file not found" in data["detail"]
