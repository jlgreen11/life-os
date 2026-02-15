"""
Life OS — BrowserOrchestrator Test Suite

Comprehensive tests for the browser automation orchestrator, which manages:
    1. Shared browser engine and credential vault initialization
    2. Browser connector lifecycle (start, stop, health)
    3. Global rate limiting across all browser connectors
    4. Connector registration and status reporting
    5. API-to-browser fallback wrapper

Coverage: 42 tests for 288 LOC critical browser infrastructure
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from connectors.browser.orchestrator import APIFallbackWrapper, BrowserOrchestrator
from connectors.browser.engine import BrowserEngine, CredentialVault


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def browser_config():
    """Standard browser configuration for testing."""
    return {
        "browser": {
            "enabled": True,
            "headless": True,
            "data_dir": "./data/browser",
            "credential_source": "manual",
            "manual_vault": "",
            "human_speed_factor": 1.0,
            "global_rate_limit": 0.1,  # Fast for tests
            "max_concurrent_contexts": 3,
            "session_refresh_hours": 168,
            "connectors": {},
        }
    }


@pytest.fixture
def disabled_browser_config():
    """Browser configuration with browser disabled."""
    return {"browser": {"enabled": False}}


@pytest.fixture
def browser_config_with_proton(tmp_path):
    """Browser config with Proton Pass export."""
    export_file = tmp_path / "proton_export.json"
    export_file.write_text(
        json.dumps(
            {
                "vaults": [
                    {
                        "items": [
                            {
                                "data": {
                                    "metadata": {"name": "reddit.com"},
                                    "content": {
                                        "username": "testuser",
                                        "password": "testpass123",
                                        "urls": ["https://reddit.com"],
                                    },
                                }
                            }
                        ]
                    }
                ]
            }
        )
    )

    return {
        "browser": {
            "enabled": True,
            "headless": True,
            "data_dir": "./data/browser",
            "credential_source": "proton_pass",
            "proton_pass_export": str(export_file),
            "manual_vault": "",
            "global_rate_limit": 0.1,
            "max_concurrent_contexts": 3,
            "connectors": {},
        }
    }


@pytest.fixture
def mock_browser_engine():
    """Mock BrowserEngine to avoid launching real Chromium."""
    engine = AsyncMock(spec=BrowserEngine)
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.create_context = AsyncMock()
    engine.new_page = AsyncMock()
    engine.save_session = AsyncMock()
    return engine


@pytest.fixture
def mock_credential_vault():
    """Mock CredentialVault with test credentials."""
    vault = MagicMock(spec=CredentialVault)
    vault.list_sites = MagicMock(return_value=["reddit.com", "youtube.com"])
    vault.load_proton_pass_export = MagicMock()
    vault.load_manual_vault = MagicMock()
    vault.get_credential = MagicMock(
        return_value={"username": "testuser", "password": "testpass"}
    )
    return vault


# ===========================================================================
# Initialization Tests
# ===========================================================================


def test_orchestrator_init_disabled(event_bus, db, disabled_browser_config):
    """BrowserOrchestrator initializes in disabled state."""
    orch = BrowserOrchestrator(event_bus, db, disabled_browser_config)

    assert not orch.is_enabled
    assert orch.connectors == []
    assert orch._engine is None
    assert orch._vault is None


def test_orchestrator_init_enabled(event_bus, db, browser_config):
    """BrowserOrchestrator initializes with enabled config."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    assert orch.is_enabled
    assert orch.connectors == []
    assert orch._global_rate_limit == 0.1
    assert isinstance(orch._semaphore, asyncio.Semaphore)


def test_orchestrator_extracts_browser_config(event_bus, db):
    """BrowserOrchestrator extracts 'browser' sub-key from config."""
    config = {
        "browser": {
            "enabled": True,
            "data_dir": "/custom/path",
            "global_rate_limit": 5.0,
        },
        "other_key": "ignored",
    }

    orch = BrowserOrchestrator(event_bus, db, config)

    assert orch.is_enabled
    assert orch._data_dir == "/custom/path"
    assert orch._global_rate_limit == 5.0


def test_orchestrator_defaults_for_missing_config(event_bus, db):
    """BrowserOrchestrator uses defaults when config keys are missing."""
    config = {"browser": {"enabled": True}}

    orch = BrowserOrchestrator(event_bus, db, config)

    assert orch._data_dir == "./data/browser"
    assert orch._global_rate_limit == 2.0  # Default
    assert orch._semaphore._value == 3  # Default max_concurrent_contexts


# ===========================================================================
# Start/Stop Lifecycle Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_start_when_disabled_does_nothing(event_bus, db, disabled_browser_config):
    """start() is a no-op when browser is disabled."""
    orch = BrowserOrchestrator(event_bus, db, disabled_browser_config)

    await orch.start()

    assert orch._engine is None
    assert orch._vault is None
    assert len(orch.connectors) == 0


@pytest.mark.asyncio
async def test_start_initializes_engine_and_vault(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() initializes BrowserEngine and CredentialVault."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ):
        await orch.start()

    mock_browser_engine.start.assert_called_once()
    assert orch._engine is mock_browser_engine
    assert orch._vault is mock_credential_vault


@pytest.mark.asyncio
async def test_start_loads_proton_pass_export(
    event_bus, db, browser_config_with_proton, mock_browser_engine, mock_credential_vault
):
    """start() loads Proton Pass export when configured."""
    orch = BrowserOrchestrator(event_bus, db, browser_config_with_proton)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ):
        await orch.start()

    # Verify Proton Pass export was loaded
    mock_credential_vault.load_proton_pass_export.assert_called_once()
    export_path = mock_credential_vault.load_proton_pass_export.call_args[0][0]
    assert "proton_export.json" in export_path


@pytest.mark.asyncio
async def test_start_loads_manual_vault(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault, tmp_path
):
    """start() loads manual vault when configured."""
    manual_vault = tmp_path / "manual.json"
    manual_vault.write_text("{}")
    browser_config["browser"]["manual_vault"] = str(manual_vault)

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ):
        await orch.start()

    mock_credential_vault.load_manual_vault.assert_called_once_with(str(manual_vault))


@pytest.mark.asyncio
async def test_start_creates_whatsapp_connector(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() creates WhatsAppConnector when configured."""
    browser_config["browser"]["connectors"] = {
        "whatsapp": {"enabled": True, "sync_interval": 300}
    }

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ), patch(
        "connectors.browser.whatsapp.WhatsAppConnector"
    ) as mock_whatsapp:
        mock_whatsapp_instance = AsyncMock()
        mock_whatsapp.return_value = mock_whatsapp_instance

        await orch.start()

    mock_whatsapp.assert_called_once()
    assert mock_whatsapp_instance in orch.connectors


@pytest.mark.asyncio
async def test_start_creates_youtube_connector(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() creates YouTubeConnector when configured."""
    browser_config["browser"]["connectors"] = {
        "youtube": {"enabled": True, "sync_interval": 600}
    }

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ), patch("connectors.browser.youtube.YouTubeConnector") as mock_youtube:
        mock_youtube_instance = AsyncMock()
        mock_youtube.return_value = mock_youtube_instance

        await orch.start()

    mock_youtube.assert_called_once()
    assert mock_youtube_instance in orch.connectors


@pytest.mark.asyncio
async def test_start_creates_reddit_connector(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() creates RedditConnector when configured."""
    browser_config["browser"]["connectors"] = {
        "reddit": {"enabled": True, "sync_interval": 300}
    }

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ), patch("connectors.browser.reddit.RedditConnector") as mock_reddit:
        mock_reddit_instance = AsyncMock()
        mock_reddit.return_value = mock_reddit_instance

        await orch.start()

    mock_reddit.assert_called_once()
    assert mock_reddit_instance in orch.connectors


@pytest.mark.asyncio
async def test_start_creates_generic_connectors(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() creates GenericBrowserConnectors for generic_sources."""
    browser_config["browser"]["connectors"] = {
        "generic_sources": [
            {
                "id": "hacker_news",
                "name": "Hacker News",
                "url": "https://news.ycombinator.com",
                "selectors": {"title": ".storylink"},
            }
        ]
    }

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ), patch(
        "connectors.browser.orchestrator.create_browser_connectors"
    ) as mock_factory:
        mock_generic = AsyncMock()
        mock_factory.return_value = [mock_generic]

        await orch.start()

    mock_factory.assert_called_once()
    assert mock_generic in orch.connectors


@pytest.mark.asyncio
async def test_start_creates_multiple_connectors(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start() creates multiple connectors when all are configured."""
    browser_config["browser"]["connectors"] = {
        "whatsapp": {"enabled": True},
        "youtube": {"enabled": True},
        "reddit": {"enabled": True},
    }

    orch = BrowserOrchestrator(event_bus, db, browser_config)

    with patch(
        "connectors.browser.orchestrator.BrowserEngine", return_value=mock_browser_engine
    ), patch(
        "connectors.browser.orchestrator.CredentialVault",
        return_value=mock_credential_vault,
    ), patch("connectors.browser.whatsapp.WhatsAppConnector") as mock_wa, patch(
        "connectors.browser.youtube.YouTubeConnector"
    ) as mock_yt, patch(
        "connectors.browser.reddit.RedditConnector"
    ) as mock_reddit:
        mock_wa.return_value = AsyncMock()
        mock_yt.return_value = AsyncMock()
        mock_reddit.return_value = AsyncMock()

        await orch.start()

    assert len(orch.connectors) == 3


@pytest.mark.asyncio
async def test_start_connectors_authenticates_all(
    event_bus, db, browser_config, mock_browser_engine, mock_credential_vault
):
    """start_connectors() calls start() on all registered connectors."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    mock_connector_1 = AsyncMock()
    mock_connector_1.DISPLAY_NAME = "Test Connector 1"
    mock_connector_1.start = AsyncMock()

    mock_connector_2 = AsyncMock()
    mock_connector_2.DISPLAY_NAME = "Test Connector 2"
    mock_connector_2.start = AsyncMock()

    orch._connectors = [mock_connector_1, mock_connector_2]

    await orch.start_connectors()

    mock_connector_1.start.assert_called_once()
    mock_connector_2.start.assert_called_once()


@pytest.mark.asyncio
async def test_start_connectors_when_disabled(event_bus, db, disabled_browser_config):
    """start_connectors() is a no-op when browser is disabled."""
    orch = BrowserOrchestrator(event_bus, db, disabled_browser_config)

    # Should not raise even though no connectors exist
    await orch.start_connectors()


@pytest.mark.asyncio
async def test_start_connectors_handles_failures_gracefully(
    event_bus, db, browser_config
):
    """start_connectors() continues if one connector fails to start."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    mock_connector_1 = AsyncMock()
    mock_connector_1.DISPLAY_NAME = "Failing Connector"
    mock_connector_1.start = AsyncMock(side_effect=Exception("Auth failed"))

    mock_connector_2 = AsyncMock()
    mock_connector_2.DISPLAY_NAME = "Working Connector"
    mock_connector_2.start = AsyncMock()

    orch._connectors = [mock_connector_1, mock_connector_2]

    # Should not raise
    await orch.start_connectors()

    # Second connector should still start despite first failure
    mock_connector_2.start.assert_called_once()


@pytest.mark.asyncio
async def test_stop_calls_stop_on_all_connectors(event_bus, db, browser_config):
    """stop() calls stop() on all connectors."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    mock_connector_1 = AsyncMock()
    mock_connector_1.stop = AsyncMock()

    mock_connector_2 = AsyncMock()
    mock_connector_2.stop = AsyncMock()

    orch._connectors = [mock_connector_1, mock_connector_2]
    orch._engine = AsyncMock()
    orch._engine.stop = AsyncMock()

    await orch.stop()

    mock_connector_1.stop.assert_called_once()
    mock_connector_2.stop.assert_called_once()


@pytest.mark.asyncio
async def test_stop_shuts_down_engine(event_bus, db, browser_config):
    """stop() shuts down the browser engine."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)
    orch._engine = AsyncMock()
    orch._engine.stop = AsyncMock()

    await orch.stop()

    orch._engine.stop.assert_called_once()


@pytest.mark.asyncio
async def test_stop_handles_connector_failures_gracefully(event_bus, db, browser_config):
    """stop() continues if a connector fails to stop."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    mock_connector_1 = AsyncMock()
    mock_connector_1.stop = AsyncMock(side_effect=Exception("Stop failed"))

    mock_connector_2 = AsyncMock()
    mock_connector_2.stop = AsyncMock()

    orch._connectors = [mock_connector_1, mock_connector_2]
    orch._engine = AsyncMock()
    orch._engine.stop = AsyncMock()

    # Should not raise
    await orch.stop()

    # Engine should still stop despite connector failure
    orch._engine.stop.assert_called_once()
    mock_connector_2.stop.assert_called_once()


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_global_rate_limit_enforces_delay():
    """global_rate_limit() enforces minimum delay between requests."""
    config = {"browser": {"enabled": True, "global_rate_limit": 0.2}}
    orch = BrowserOrchestrator(AsyncMock(), MagicMock(), config)

    start = time.time()

    # First request should be immediate
    await orch.global_rate_limit()
    first_elapsed = time.time() - start
    assert first_elapsed < 0.05

    # Second request should wait for rate limit
    await orch.global_rate_limit()
    second_elapsed = time.time() - start
    assert second_elapsed >= 0.2


@pytest.mark.asyncio
async def test_global_rate_limit_respects_semaphore():
    """global_rate_limit() limits concurrent requests via semaphore."""
    config = {
        "browser": {
            "enabled": True,
            "global_rate_limit": 0.01,
            "max_concurrent_contexts": 2,
        }
    }
    orch = BrowserOrchestrator(AsyncMock(), MagicMock(), config)

    # Track how many tasks are running concurrently
    concurrent = 0
    max_concurrent = 0

    async def rate_limited_task():
        nonlocal concurrent, max_concurrent
        async with orch._semaphore:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1

    # Launch 5 tasks, but only 2 should run at once
    await asyncio.gather(*[rate_limited_task() for _ in range(5)])

    assert max_concurrent == 2


@pytest.mark.asyncio
async def test_global_rate_limit_updates_last_request_time():
    """global_rate_limit() updates last request timestamp."""
    config = {"browser": {"enabled": True, "global_rate_limit": 0.01}}
    orch = BrowserOrchestrator(AsyncMock(), MagicMock(), config)

    assert orch._last_global_request == 0

    await orch.global_rate_limit()

    assert orch._last_global_request > 0


# ===========================================================================
# Status Reporting Tests
# ===========================================================================


def test_get_status_when_disabled(event_bus, db, disabled_browser_config):
    """get_status() returns disabled state."""
    orch = BrowserOrchestrator(event_bus, db, disabled_browser_config)

    status = orch.get_status()

    assert status["enabled"] is False
    assert status["engine_running"] is False
    assert status["credential_sites"] == 0
    assert status["connectors"] == []


def test_get_status_when_enabled_no_engine(event_bus, db, browser_config):
    """get_status() reports engine not running before start()."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    status = orch.get_status()

    assert status["enabled"] is True
    assert status["engine_running"] is False
    assert status["credential_sites"] == 0


def test_get_status_with_connectors(event_bus, db, browser_config):
    """get_status() reports connector states correctly."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)
    orch._engine = AsyncMock()
    orch._vault = MagicMock()
    orch._vault.list_sites.return_value = ["reddit.com", "youtube.com"]

    # Create mock connectors with different failure states
    mock_connector_1 = AsyncMock()
    mock_connector_1.CONNECTOR_ID = "whatsapp"
    mock_connector_1.DISPLAY_NAME = "WhatsApp"
    mock_connector_1._api_failures = 0
    mock_connector_1._api_failure_threshold = 3
    mock_connector_1._api_mode = True

    mock_connector_2 = AsyncMock()
    mock_connector_2.CONNECTOR_ID = "reddit"
    mock_connector_2.DISPLAY_NAME = "Reddit"
    mock_connector_2._api_failures = 5
    mock_connector_2._api_failure_threshold = 3
    mock_connector_2._api_mode = True

    orch._connectors = [mock_connector_1, mock_connector_2]

    status = orch.get_status()

    assert status["enabled"] is True
    assert status["engine_running"] is True
    assert status["credential_sites"] == 2
    assert len(status["connectors"]) == 2

    # First connector should be in API mode (no failures)
    assert status["connectors"][0]["id"] == "whatsapp"
    assert status["connectors"][0]["mode"] == "api"
    assert status["connectors"][0]["api_failures"] == 0

    # Second connector should be in browser mode (exceeded threshold)
    assert status["connectors"][1]["id"] == "reddit"
    assert status["connectors"][1]["mode"] == "browser"
    assert status["connectors"][1]["api_failures"] == 5


def test_get_vault_sites_when_no_vault(event_bus, db, browser_config):
    """get_vault_sites() returns empty list when vault not initialized."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    sites = orch.get_vault_sites()

    assert sites == []


def test_get_vault_sites_returns_vault_contents(event_bus, db, browser_config):
    """get_vault_sites() returns sites from the credential vault."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)
    orch._vault = MagicMock()
    orch._vault.list_sites.return_value = ["reddit.com", "youtube.com", "whatsapp.com"]

    sites = orch.get_vault_sites()

    assert sites == ["reddit.com", "youtube.com", "whatsapp.com"]


# ===========================================================================
# APIFallbackWrapper Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_api_fallback_wrapper_tries_api_first():
    """APIFallbackWrapper tries API connector before browser."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    mock_api.sync = AsyncMock(return_value=10)

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 3},
    )

    count = await wrapper.sync()

    assert count == 10
    mock_api.sync.assert_called_once()


@pytest.mark.asyncio
async def test_api_fallback_wrapper_tracks_failures():
    """APIFallbackWrapper increments failure count on API errors."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    mock_api.sync = AsyncMock(side_effect=Exception("API error"))

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 3},
    )

    assert wrapper._api_failures == 0

    # First failure
    await wrapper.sync()
    assert wrapper._api_failures == 1

    # Second failure
    await wrapper.sync()
    assert wrapper._api_failures == 2


@pytest.mark.asyncio
async def test_api_fallback_wrapper_resets_on_success():
    """APIFallbackWrapper resets failure count on successful API call."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    call_count = 0

    async def sync_with_failure():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("API error")
        return 5

    mock_api.sync = AsyncMock(side_effect=sync_with_failure)

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 3},
    )

    # Fail twice
    await wrapper.sync()
    await wrapper.sync()
    assert wrapper._api_failures == 2

    # Succeed on third try
    count = await wrapper.sync()
    assert count == 5
    assert wrapper._api_failures == 0


@pytest.mark.asyncio
async def test_api_fallback_wrapper_switches_to_browser_after_threshold():
    """APIFallbackWrapper switches to browser mode after threshold failures."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    mock_api.sync = AsyncMock(side_effect=Exception("API error"))

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 2},
    )

    # First two calls should try API
    await wrapper.sync()
    await wrapper.sync()
    assert mock_api.sync.call_count == 2

    # Third call should skip API and go straight to browser
    await wrapper.sync()
    assert mock_api.sync.call_count == 2  # No additional API call


@pytest.mark.asyncio
async def test_api_fallback_wrapper_reset_api():
    """reset_api() resets the failure counter."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    mock_api.sync = AsyncMock(side_effect=Exception("API error"))

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 2},
    )

    # Accumulate failures
    await wrapper.sync()
    await wrapper.sync()
    assert wrapper._api_failures == 2

    # Reset
    wrapper.reset_api()
    assert wrapper._api_failures == 0


@pytest.mark.asyncio
async def test_api_fallback_wrapper_browser_fallback_not_implemented():
    """_browser_fallback_sync() returns 0 by default (override in subclass)."""
    mock_api = AsyncMock()
    mock_api.CONNECTOR_ID = "test_connector"
    mock_api.sync = AsyncMock(side_effect=Exception("API error"))

    wrapper = APIFallbackWrapper(
        api_connector=mock_api,
        browser_engine=AsyncMock(),
        credential_vault=MagicMock(),
        fallback_config={"failure_threshold": 1},
    )

    # Force browser mode
    await wrapper.sync()
    count = await wrapper.sync()

    assert count == 0  # Default implementation returns 0


# ===========================================================================
# Property Tests
# ===========================================================================


def test_is_enabled_property(event_bus, db, browser_config):
    """is_enabled property returns correct value."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)
    assert orch.is_enabled is True

    orch_disabled = BrowserOrchestrator(
        event_bus, db, {"browser": {"enabled": False}}
    )
    assert orch_disabled.is_enabled is False


def test_connectors_property(event_bus, db, browser_config):
    """connectors property returns list of registered connectors."""
    orch = BrowserOrchestrator(event_bus, db, browser_config)

    assert orch.connectors == []

    mock_connector = AsyncMock()
    orch._connectors = [mock_connector]

    assert orch.connectors == [mock_connector]
