"""
Tests for connectors/browser/base_connector.py — BrowserBaseConnector.

BrowserBaseConnector extends BaseConnector with browser automation capabilities:
API-to-browser failover, session management, rate limiting, and login flows.
It is the base class for all 4 browser connectors (WhatsApp, YouTube, Reddit,
generic), so correctness here is critical.

Coverage:
    - Initialization defaults and config overrides
    - authenticate() with API-first and browser fallback
    - sync() with API-to-browser failover logic
    - stop() with session saving and context cleanup
    - Rate limiting enforcement
    - Browser sync wrapper (session expiry detection, login gating)
    - Default is_logged_in() URL-based heuristic
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


# ---------------------------------------------------------------------------
# Concrete test subclass of BrowserBaseConnector
# ---------------------------------------------------------------------------


def _make_test_connector_class():
    """Create a concrete BrowserBaseConnector subclass for testing.

    We import inside this factory so that module-level imports of
    connectors.browser.engine (which depends on Playwright) are satisfied
    by the patches applied in each test.
    """
    from connectors.browser.base_connector import BrowserBaseConnector

    class _TestBrowserConnector(BrowserBaseConnector):
        """Minimal concrete subclass that stubs all abstract methods."""

        CONNECTOR_ID = "test_browser"
        CONNECTOR_TYPE = "browser_test"
        DISPLAY_NAME = "Test Browser Connector"
        SITE_ID = "test_site"
        LOGIN_URL = "https://example.com/login"
        SYNC_INTERVAL_SECONDS = 60

        def __init__(self, event_bus, db, config, browser_engine=None, credential_vault=None):
            super().__init__(event_bus, db, config,
                             browser_engine=browser_engine,
                             credential_vault=credential_vault)
            # Track calls for assertions
            self.api_authenticate_mock = AsyncMock(return_value=True)
            self.api_sync_mock = AsyncMock(return_value=5)
            self.browser_sync_mock = AsyncMock(return_value=3)

        async def api_authenticate(self) -> bool:
            """Delegate to mock for test control."""
            return await self.api_authenticate_mock()

        async def api_sync(self) -> int:
            """Delegate to mock for test control."""
            return await self.api_sync_mock()

        async def browser_sync(self, page, human, interactor) -> int:
            """Delegate to mock for test control."""
            return await self.browser_sync_mock(page, human, interactor)

        async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
            """Stub — not exercised in these tests."""
            return {"status": "ok"}

        async def health_check(self) -> dict[str, Any]:
            """Stub — not exercised in these tests."""
            return {"status": "ok"}

    return _TestBrowserConnector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_browser_engine():
    """A MagicMock standing in for BrowserEngine."""
    engine = MagicMock()
    engine.start = AsyncMock()
    engine.create_context = AsyncMock(return_value=MagicMock(name="context"))
    engine.new_page = AsyncMock(return_value=MagicMock(name="page"))
    engine.save_session = AsyncMock()
    engine.session_manager = MagicMock()
    engine.session_manager.has_session.return_value = False
    engine.session_manager.clear_session = MagicMock()
    return engine


@pytest.fixture
def mock_credential_vault():
    """A MagicMock standing in for CredentialVault."""
    vault = MagicMock()
    vault.get_credential.return_value = {"username": "user", "password": "pass"}
    vault.get_totp.return_value = None
    return vault


@pytest.fixture
def browser_connector(event_bus, db, mock_browser_engine, mock_credential_vault):
    """Create a TestBrowserConnector with mocked browser dependencies."""
    cls = _make_test_connector_class()
    config = {"prefer_api": True}
    connector = cls(event_bus, db, config,
                    browser_engine=mock_browser_engine,
                    credential_vault=mock_credential_vault)
    return connector


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestInit:
    """Verify __init__ sets correct defaults and honours config overrides."""

    @patch("connectors.browser.base_connector.BrowserEngine")
    @patch("connectors.browser.base_connector.CredentialVault")
    @patch("connectors.browser.base_connector.HumanEmulator")
    @patch("connectors.browser.base_connector.PageInteractor")
    def test_init_defaults(self, _pi, _he, _cv, _be, event_bus, db):
        """Verify __init__ sets correct defaults when no config overrides."""
        cls = _make_test_connector_class()
        connector = cls(event_bus, db, {})

        assert connector._api_mode is True
        assert connector._api_failures == 0
        assert connector._api_failure_threshold == 3
        assert connector.MIN_REQUEST_INTERVAL == 2.0
        assert connector.MAX_PAGES_PER_SYNC == 20
        assert connector._last_request_time == 0

    @patch("connectors.browser.base_connector.BrowserEngine")
    @patch("connectors.browser.base_connector.CredentialVault")
    @patch("connectors.browser.base_connector.HumanEmulator")
    @patch("connectors.browser.base_connector.PageInteractor")
    def test_init_custom_config(self, _pi, mock_he, _cv, _be, event_bus, db):
        """Verify config overrides work for prefer_api, api_failure_threshold,
        and human_speed_factor."""
        cls = _make_test_connector_class()
        config = {
            "prefer_api": False,
            "api_failure_threshold": 5,
            "human_speed_factor": 2.0,
        }
        connector = cls(event_bus, db, config)

        assert connector._api_mode is False
        assert connector._api_failure_threshold == 5
        # HumanEmulator should have been called with speed_factor=2.0
        mock_he.assert_called_once_with(speed_factor=2.0)


# ---------------------------------------------------------------------------
# authenticate() Tests
# ---------------------------------------------------------------------------


class TestAuthenticate:
    """Verify API-first authentication with browser fallback."""

    @pytest.mark.asyncio
    async def test_authenticate_api_success(self, browser_connector):
        """When api_mode=True and api_authenticate() returns True,
        authenticate() returns True without calling _browser_login()."""
        browser_connector.api_authenticate_mock.return_value = True
        browser_connector._browser_login = AsyncMock(return_value=False)

        result = await browser_connector.authenticate()

        assert result is True
        browser_connector.api_authenticate_mock.assert_awaited_once()
        browser_connector._browser_login.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_authenticate_api_failure_falls_back_to_browser(self, browser_connector):
        """When api_authenticate() raises an exception, authenticate()
        falls back to _browser_login()."""
        browser_connector.api_authenticate_mock.side_effect = Exception("API down")
        browser_connector._browser_login = AsyncMock(return_value=True)

        result = await browser_connector.authenticate()

        assert result is True
        browser_connector.api_authenticate_mock.assert_awaited_once()
        browser_connector._browser_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_browser_mode_direct(self, browser_connector):
        """When api_mode=False, authenticate() skips api_authenticate()
        and goes straight to _browser_login()."""
        browser_connector._api_mode = False
        browser_connector._browser_login = AsyncMock(return_value=True)

        result = await browser_connector.authenticate()

        assert result is True
        browser_connector.api_authenticate_mock.assert_not_awaited()
        browser_connector._browser_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_api_returns_false_falls_back(self, browser_connector):
        """When api_authenticate() returns False (not an exception),
        authenticate() falls back to _browser_login()."""
        browser_connector.api_authenticate_mock.return_value = False
        browser_connector._browser_login = AsyncMock(return_value=True)

        result = await browser_connector.authenticate()

        assert result is True
        browser_connector._browser_login.assert_awaited_once()


# ---------------------------------------------------------------------------
# sync() Tests
# ---------------------------------------------------------------------------


class TestSync:
    """Verify API-to-browser failover logic in sync()."""

    @pytest.mark.asyncio
    async def test_sync_api_success_resets_failures(self, browser_connector):
        """When api_sync() succeeds, api_failures resets to 0 and the
        browser path is not called."""
        browser_connector._api_failures = 2
        browser_connector.api_sync_mock.return_value = 10

        count = await browser_connector.sync()

        assert count == 10
        assert browser_connector._api_failures == 0
        browser_connector.browser_sync_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sync_api_failure_increments_counter(self, browser_connector):
        """When api_sync() raises, api_failures increments by 1."""
        browser_connector.api_sync_mock.side_effect = Exception("API error")
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=3)

        count = await browser_connector.sync()

        assert browser_connector._api_failures == 1
        assert count == 3

    @pytest.mark.asyncio
    async def test_sync_switches_to_browser_after_threshold(self, browser_connector):
        """After api_failure_threshold consecutive API failures, sync()
        goes directly to browser mode, skipping the API call entirely."""
        browser_connector._api_failures = 3  # At threshold
        browser_connector._api_failure_threshold = 3
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=7)

        count = await browser_connector.sync()

        assert count == 7
        # API should not have been called at all
        browser_connector.api_sync_mock.assert_not_awaited()
        browser_connector._browser_sync_wrapper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_browser_fallback_on_api_error(self, browser_connector):
        """When api_sync() fails but below threshold, _browser_sync_wrapper
        is called as fallback."""
        browser_connector.api_sync_mock.side_effect = Exception("API error")
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=4)

        count = await browser_connector.sync()

        assert count == 4
        browser_connector._browser_sync_wrapper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_browser_error_returns_zero(self, browser_connector):
        """When both API and browser fail, sync() returns 0."""
        browser_connector.api_sync_mock.side_effect = Exception("API error")
        browser_connector._browser_sync_wrapper = AsyncMock(
            side_effect=Exception("Browser error too")
        )

        count = await browser_connector.sync()

        assert count == 0

    @pytest.mark.asyncio
    async def test_sync_api_failure_logs_switch_at_threshold(self, browser_connector):
        """When api_failures reaches the threshold, the connector should
        mark that it's switching to browser mode."""
        browser_connector._api_failure_threshold = 2
        browser_connector._api_failures = 1  # One failure already
        browser_connector.api_sync_mock.side_effect = Exception("API error")
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=0)

        await browser_connector.sync()

        # After this failure, _api_failures should be at threshold
        assert browser_connector._api_failures == 2
        assert browser_connector._api_failures >= browser_connector._api_failure_threshold

    @pytest.mark.asyncio
    async def test_sync_api_mode_false_skips_api(self, browser_connector):
        """When api_mode=False, sync() goes straight to browser."""
        browser_connector._api_mode = False
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=6)

        count = await browser_connector.sync()

        assert count == 6
        browser_connector.api_sync_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# stop() Tests
# ---------------------------------------------------------------------------


class TestStop:
    """Verify stop() saves session and cleans up browser context."""

    @pytest.mark.asyncio
    async def test_stop_saves_session_and_closes_context(
        self, browser_connector, mock_browser_engine
    ):
        """Verify stop() calls save_session and context.close()."""
        mock_context = AsyncMock()
        browser_connector._context = mock_context

        await browser_connector.stop()

        mock_browser_engine.save_session.assert_awaited_once_with(
            mock_context, browser_connector.SITE_ID
        )
        mock_context.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_handles_save_session_error(
        self, browser_connector, mock_browser_engine
    ):
        """Verify stop() continues even if save_session raises."""
        mock_context = AsyncMock()
        browser_connector._context = mock_context
        mock_browser_engine.save_session.side_effect = Exception("Save failed")

        # Should not raise
        await browser_connector.stop()

        # context.close() should still be called despite save_session failure
        mock_context.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_handles_context_close_error(
        self, browser_connector, mock_browser_engine
    ):
        """Verify stop() continues even if context.close() raises."""
        mock_context = AsyncMock()
        mock_context.close.side_effect = Exception("Close failed")
        browser_connector._context = mock_context

        # Should not raise
        await browser_connector.stop()

        mock_browser_engine.save_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_no_context_skips_cleanup(self, browser_connector, mock_browser_engine):
        """When _context is None, stop() skips session save and close."""
        browser_connector._context = None

        await browser_connector.stop()

        mock_browser_engine.save_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_calls_super_stop(self, browser_connector):
        """stop() should delegate to BaseConnector.stop()."""
        browser_connector._context = None

        await browser_connector.stop()

        # BaseConnector.stop() sets _running = False
        assert browser_connector._running is False


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Verify rate_limit_wait() enforces the minimum request interval."""

    @pytest.mark.asyncio
    async def test_rate_limit_wait_enforces_interval(self, browser_connector):
        """When called rapidly, rate_limit_wait() sleeps for the remaining interval."""
        loop = asyncio.get_event_loop()
        # Set last request time to "just now"
        browser_connector._last_request_time = loop.time()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await browser_connector.rate_limit_wait()

            # Should have slept (approximately MIN_REQUEST_INTERVAL seconds)
            mock_sleep.assert_awaited_once()
            sleep_duration = mock_sleep.call_args[0][0]
            # Sleep time should be positive and roughly close to the interval
            # (minus tiny elapsed + jitter)
            assert sleep_duration >= 0

    @pytest.mark.asyncio
    async def test_rate_limit_wait_no_sleep_when_enough_time_passed(self, browser_connector):
        """When enough time has passed, rate_limit_wait() does not sleep."""
        loop = asyncio.get_event_loop()
        # Set last request time to far in the past
        browser_connector._last_request_time = loop.time() - 100

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await browser_connector.rate_limit_wait()

            mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rate_limit_wait_updates_last_request_time(self, browser_connector):
        """After rate_limit_wait(), _last_request_time is updated."""
        browser_connector._last_request_time = 0

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await browser_connector.rate_limit_wait()

        assert browser_connector._last_request_time > 0


# ---------------------------------------------------------------------------
# _browser_sync_wrapper() Tests
# ---------------------------------------------------------------------------


class TestBrowserSyncWrapper:
    """Verify the browser sync wrapper handles login gating and session recovery."""

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_triggers_login_if_no_page(self, browser_connector):
        """When _page is None, _browser_sync_wrapper() calls _browser_login() first."""
        browser_connector._page = None
        browser_connector._browser_login = AsyncMock(return_value=True)
        # After login, _page and _context should be set; simulate that
        browser_connector._browser_login.side_effect = self._set_page(browser_connector)
        browser_connector.browser_sync_mock.return_value = 5

        count = await browser_connector._browser_sync_wrapper()

        assert count == 5
        browser_connector._browser_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_returns_zero_if_login_fails(self, browser_connector):
        """When _page is None and _browser_login() fails, returns 0."""
        browser_connector._page = None
        browser_connector._browser_login = AsyncMock(return_value=False)

        count = await browser_connector._browser_sync_wrapper()

        assert count == 0

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_session_expiry_detection(
        self, browser_connector, mock_browser_engine
    ):
        """When browser_sync raises an error containing 'session', the wrapper
        clears session and retries after re-login."""
        browser_connector._page = MagicMock(name="page")
        browser_connector._context = MagicMock(name="context")

        # First call raises session error, second call succeeds
        browser_connector.browser_sync_mock.side_effect = [
            Exception("session expired"),
            8,
        ]
        browser_connector._browser_login = AsyncMock(return_value=True)

        count = await browser_connector._browser_sync_wrapper()

        assert count == 8
        mock_browser_engine.session_manager.clear_session.assert_called_once_with(
            browser_connector.SITE_ID
        )
        browser_connector._browser_login.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_login_keyword_detection(
        self, browser_connector, mock_browser_engine
    ):
        """When browser_sync raises an error containing 'login', the wrapper
        also treats it as session expiry."""
        browser_connector._page = MagicMock(name="page")
        browser_connector._context = MagicMock(name="context")

        browser_connector.browser_sync_mock.side_effect = [
            Exception("redirected to login page"),
            4,
        ]
        browser_connector._browser_login = AsyncMock(return_value=True)

        count = await browser_connector._browser_sync_wrapper()

        assert count == 4
        mock_browser_engine.session_manager.clear_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_non_session_error_raises(self, browser_connector):
        """When browser_sync raises an error NOT related to session/login,
        the wrapper re-raises it."""
        browser_connector._page = MagicMock(name="page")
        browser_connector._context = MagicMock(name="context")

        browser_connector.browser_sync_mock.side_effect = Exception("network timeout")

        with pytest.raises(Exception, match="network timeout"):
            await browser_connector._browser_sync_wrapper()

    @pytest.mark.asyncio
    async def test_browser_sync_wrapper_saves_session_on_success(
        self, browser_connector, mock_browser_engine
    ):
        """After a successful browser_sync, session is saved."""
        browser_connector._page = MagicMock(name="page")
        browser_connector._context = MagicMock(name="context")
        browser_connector.browser_sync_mock.return_value = 10

        count = await browser_connector._browser_sync_wrapper()

        assert count == 10
        mock_browser_engine.save_session.assert_awaited_once_with(
            browser_connector._context, browser_connector.SITE_ID
        )

    @staticmethod
    def _set_page(connector):
        """Helper that simulates _browser_login setting _page and _context."""
        async def _side_effect():
            connector._page = MagicMock(name="page")
            connector._context = MagicMock(name="context")
            return True
        return _side_effect


# ---------------------------------------------------------------------------
# is_logged_in() Tests
# ---------------------------------------------------------------------------


class TestIsLoggedIn:
    """Verify the default URL-based login detection heuristic."""

    @pytest.mark.asyncio
    async def test_is_logged_in_returns_false_for_login_urls(self, browser_connector):
        """Default is_logged_in() returns False for URLs containing
        'login', 'signin', 'auth', or 'sso'."""
        login_urls = [
            "https://example.com/login",
            "https://example.com/signin",
            "https://auth.example.com/",
            "https://example.com/sso/callback",
        ]
        for url in login_urls:
            page = MagicMock()
            page.url = url
            result = await browser_connector.is_logged_in(page)
            assert result is False, f"Expected False for URL: {url}"

    @pytest.mark.asyncio
    async def test_is_logged_in_returns_true_for_non_login_urls(self, browser_connector):
        """Default is_logged_in() returns True for normal page URLs."""
        normal_urls = [
            "https://example.com/dashboard",
            "https://example.com/feed",
            "https://example.com/home",
            "https://example.com/messages",
        ]
        for url in normal_urls:
            page = MagicMock()
            page.url = url
            result = await browser_connector.is_logged_in(page)
            assert result is True, f"Expected True for URL: {url}"

    @pytest.mark.asyncio
    async def test_is_logged_in_case_insensitive(self, browser_connector):
        """URL matching is case-insensitive (uses .lower())."""
        page = MagicMock()
        page.url = "https://example.com/LOGIN"
        result = await browser_connector.is_logged_in(page)
        assert result is False


# ---------------------------------------------------------------------------
# _browser_login() Tests
# ---------------------------------------------------------------------------


class TestBrowserLogin:
    """Verify the full browser login flow: credential lookup, session reuse,
    fresh login with form filling, 2FA handling, and error recovery."""

    @pytest.mark.asyncio
    async def test_browser_login_no_credentials_returns_false(
        self, browser_connector, mock_credential_vault
    ):
        """When no credentials exist for the site, _browser_login() returns False."""
        mock_credential_vault.get_credential.return_value = None

        result = await browser_connector._browser_login()

        assert result is False
        # Should not have started the browser engine
        browser_connector._browser_engine.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_login_reuses_valid_session(
        self, browser_connector, mock_browser_engine
    ):
        """When a saved session is still valid, _browser_login() returns True
        without performing a fresh login."""
        mock_browser_engine.session_manager.has_session.return_value = True

        # Create a mock page that reports being logged in
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_browser_engine.new_page.return_value = mock_page

        # Override is_logged_in to return True (session is still valid)
        browser_connector.is_logged_in = AsyncMock(return_value=True)

        result = await browser_connector._browser_login()

        assert result is True
        mock_browser_engine.start.assert_awaited_once()
        mock_browser_engine.create_context.assert_awaited_once()
        # Should have navigated to check session validity
        mock_page.goto.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_login_fresh_login_success(
        self, browser_connector, mock_browser_engine
    ):
        """When no saved session exists, _browser_login() performs a fresh
        login flow and saves the session on success."""
        mock_browser_engine.session_manager.has_session.return_value = False

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_browser_engine.new_page.return_value = mock_page

        # is_logged_in returns True after the login form is submitted
        browser_connector.is_logged_in = AsyncMock(return_value=True)

        with patch.object(browser_connector, "_human") as mock_human, \
             patch.object(browser_connector, "_interactor") as mock_interactor:
            mock_human.wait_human = AsyncMock()
            mock_interactor.login = AsyncMock()

            result = await browser_connector._browser_login()

        assert result is True
        # Session should be saved after successful login
        mock_browser_engine.save_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_browser_login_fresh_login_failure(
        self, browser_connector, mock_browser_engine
    ):
        """When login form submission doesn't result in a logged-in state,
        _browser_login() returns False."""
        mock_browser_engine.session_manager.has_session.return_value = False

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_browser_engine.new_page.return_value = mock_page

        # is_logged_in returns False (login failed)
        browser_connector.is_logged_in = AsyncMock(return_value=False)

        with patch.object(browser_connector, "_human") as mock_human, \
             patch.object(browser_connector, "_interactor") as mock_interactor:
            mock_human.wait_human = AsyncMock()
            mock_interactor.login = AsyncMock()

            result = await browser_connector._browser_login()

        assert result is False
        # Session should NOT be saved when login fails
        mock_browser_engine.save_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_login_2fa_flow(
        self, browser_connector, mock_browser_engine, mock_credential_vault
    ):
        """When REQUIRES_2FA is True and TOTP is available, _browser_login()
        enters the 2FA code after password submission."""
        browser_connector.REQUIRES_2FA = True
        mock_credential_vault.get_totp.return_value = "123456"
        mock_browser_engine.session_manager.has_session.return_value = False

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_browser_engine.new_page.return_value = mock_page

        browser_connector.is_logged_in = AsyncMock(return_value=True)

        with patch.object(browser_connector, "_human") as mock_human, \
             patch.object(browser_connector, "_interactor") as mock_interactor:
            mock_human.wait_human = AsyncMock()
            mock_interactor.login = AsyncMock()
            mock_interactor.handle_2fa = AsyncMock()

            result = await browser_connector._browser_login()

        assert result is True
        # 2FA handler should have been called with the TOTP code
        mock_interactor.handle_2fa.assert_awaited_once()
        call_args = mock_interactor.handle_2fa.call_args
        assert call_args[0][1] == "123456"  # totp_code argument

    @pytest.mark.asyncio
    async def test_browser_login_2fa_no_totp_returns_false(
        self, browser_connector, mock_browser_engine, mock_credential_vault
    ):
        """When REQUIRES_2FA is True but no TOTP URI is configured,
        _browser_login() returns False."""
        browser_connector.REQUIRES_2FA = True
        mock_credential_vault.get_totp.return_value = None
        mock_browser_engine.session_manager.has_session.return_value = False

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_browser_engine.new_page.return_value = mock_page

        with patch.object(browser_connector, "_human") as mock_human, \
             patch.object(browser_connector, "_interactor") as mock_interactor:
            mock_human.wait_human = AsyncMock()
            mock_interactor.login = AsyncMock()

            result = await browser_connector._browser_login()

        assert result is False

    @pytest.mark.asyncio
    async def test_browser_login_exception_returns_false(
        self, browser_connector, mock_browser_engine
    ):
        """When the browser login flow raises an exception,
        _browser_login() returns False (fail-open)."""
        mock_browser_engine.start.side_effect = Exception("Browser crashed")

        result = await browser_connector._browser_login()

        assert result is False

    @pytest.mark.asyncio
    async def test_browser_login_sets_page_and_context(
        self, browser_connector, mock_browser_engine
    ):
        """After successful login, _page and _context are set on the connector."""
        mock_browser_engine.session_manager.has_session.return_value = False

        mock_context = MagicMock(name="context")
        mock_page = AsyncMock(name="page")
        mock_page.goto = AsyncMock()
        mock_browser_engine.create_context.return_value = mock_context
        mock_browser_engine.new_page.return_value = mock_page

        browser_connector.is_logged_in = AsyncMock(return_value=True)

        with patch.object(browser_connector, "_human") as mock_human, \
             patch.object(browser_connector, "_interactor") as mock_interactor:
            mock_human.wait_human = AsyncMock()
            mock_interactor.login = AsyncMock()

            await browser_connector._browser_login()

        assert browser_connector._context is mock_context
        assert browser_connector._page is mock_page


# ---------------------------------------------------------------------------
# navigate_with_rate_limit() Tests
# ---------------------------------------------------------------------------


class TestNavigateWithRateLimit:
    """Verify navigate_with_rate_limit() delegates to rate_limit_wait and page.goto."""

    @pytest.mark.asyncio
    async def test_navigate_calls_rate_limit_then_goto(self, browser_connector):
        """navigate_with_rate_limit() enforces rate limiting before navigation."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        browser_connector.rate_limit_wait = AsyncMock()

        await browser_connector.navigate_with_rate_limit(
            mock_page, "https://example.com/page"
        )

        browser_connector.rate_limit_wait.assert_awaited_once()
        mock_page.goto.assert_awaited_once_with(
            "https://example.com/page", wait_until="networkidle"
        )

    @pytest.mark.asyncio
    async def test_navigate_passes_url_to_page(self, browser_connector):
        """navigate_with_rate_limit() passes the correct URL to page.goto."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        browser_connector.rate_limit_wait = AsyncMock()

        url = "https://example.com/specific/page?q=test"
        await browser_connector.navigate_with_rate_limit(mock_page, url)

        mock_page.goto.assert_awaited_once_with(url, wait_until="networkidle")


# ---------------------------------------------------------------------------
# Default Subclass Interface Tests
# ---------------------------------------------------------------------------


class TestDefaultSubclassInterface:
    """Verify the default implementations of the subclass interface methods."""

    def test_get_login_url_returns_class_attribute(self, browser_connector):
        """Default get_login_url() returns the LOGIN_URL class attribute."""
        assert browser_connector.get_login_url() == "https://example.com/login"

    def test_get_login_selectors_returns_defaults(self, browser_connector):
        """Default get_login_selectors() returns standard CSS selectors."""
        selectors = browser_connector.get_login_selectors()
        assert "username" in selectors
        assert "password" in selectors
        assert "submit" in selectors

    @pytest.mark.asyncio
    async def test_api_authenticate_default_returns_false(self, event_bus, db):
        """The base api_authenticate() returns False by default (for
        connectors without an API)."""
        from connectors.browser.base_connector import BrowserBaseConnector

        # Access the base class default directly (not the test subclass override)
        result = await BrowserBaseConnector.api_authenticate(
            MagicMock(spec=BrowserBaseConnector)
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_api_sync_default_raises(self, event_bus, db):
        """The base api_sync() raises NotImplementedError by default."""
        from connectors.browser.base_connector import BrowserBaseConnector

        with pytest.raises(NotImplementedError):
            await BrowserBaseConnector.api_sync(
                MagicMock(spec=BrowserBaseConnector)
            )

    @pytest.mark.asyncio
    async def test_browser_sync_default_raises(self, event_bus, db):
        """The base browser_sync() raises NotImplementedError by default."""
        from connectors.browser.base_connector import BrowserBaseConnector

        with pytest.raises(NotImplementedError):
            await BrowserBaseConnector.browser_sync(
                MagicMock(spec=BrowserBaseConnector),
                MagicMock(), MagicMock(), MagicMock(),
            )


# ---------------------------------------------------------------------------
# Consecutive Failure Progression Tests
# ---------------------------------------------------------------------------


class TestFailureProgression:
    """Verify the full failure progression from API to browser mode."""

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_switches_permanently(
        self, browser_connector
    ):
        """After exactly api_failure_threshold consecutive API failures,
        subsequent sync calls skip the API entirely."""
        browser_connector._api_failure_threshold = 3
        browser_connector.api_sync_mock.side_effect = Exception("API down")
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=2)

        # Three consecutive failures
        for i in range(3):
            await browser_connector.sync()

        assert browser_connector._api_failures == 3

        # Fourth call should skip API entirely
        browser_connector.api_sync_mock.reset_mock()
        await browser_connector.sync()

        browser_connector.api_sync_mock.assert_not_awaited()
        assert browser_connector._browser_sync_wrapper.await_count == 4

    @pytest.mark.asyncio
    async def test_api_success_mid_failures_resets_counter(self, browser_connector):
        """An API success in the middle of a failure sequence resets the
        failure counter, preventing premature switch to browser mode."""
        browser_connector._api_failure_threshold = 3

        # Two failures
        browser_connector.api_sync_mock.side_effect = Exception("API down")
        browser_connector._browser_sync_wrapper = AsyncMock(return_value=1)
        await browser_connector.sync()
        await browser_connector.sync()
        assert browser_connector._api_failures == 2

        # Success resets counter
        browser_connector.api_sync_mock.side_effect = None
        browser_connector.api_sync_mock.return_value = 5
        count = await browser_connector.sync()
        assert count == 5
        assert browser_connector._api_failures == 0

        # Two more failures — should NOT switch because counter was reset
        browser_connector.api_sync_mock.side_effect = Exception("API down again")
        await browser_connector.sync()
        await browser_connector.sync()
        assert browser_connector._api_failures == 2
        # Still below threshold, so API is still attempted
        assert browser_connector._api_mode is True
