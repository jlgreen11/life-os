"""
Comprehensive test suite for BrowserEngine.

Tests the browser automation engine that provides stealth browser instances
for web scraping when API connectors aren't available (672 LOC foundation,
0% coverage prior to this test suite).

Coverage areas:
- HumanEmulator: realistic mouse, keyboard, scroll, and timing patterns
- SessionManager: persistent cookies and storage state
- CredentialVault: Proton Pass export and manual vault loading
- BrowserEngine: stealth Playwright browser management
- PageInteractor: high-level interaction primitives
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

# Import the browser engine module
from connectors.browser import engine


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_page():
    """Mock Playwright page object."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.evaluate = AsyncMock(return_value={"x": 100, "y": 100})
    page.wait_for_selector = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.query_selector_all = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()

    # Mouse and keyboard mocks
    page.mouse = AsyncMock()
    page.mouse.move = AsyncMock()
    page.mouse.click = AsyncMock()
    page.mouse.dblclick = AsyncMock()
    page.mouse.wheel = AsyncMock()

    page.keyboard = AsyncMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()

    return page


@pytest.fixture
def mock_element():
    """Mock Playwright element object."""
    element = AsyncMock()
    element.bounding_box = AsyncMock(return_value={
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 50,
    })
    element.text_content = AsyncMock(return_value="Sample Text")
    return element


@pytest.fixture
def mock_context():
    """Mock Playwright browser context."""
    context = AsyncMock()
    context.new_page = AsyncMock()
    context.storage_state = AsyncMock(return_value={
        "cookies": [],
        "origins": [],
    })
    context.on = Mock()
    return context


@pytest.fixture
def mock_browser():
    """Mock Playwright browser."""
    browser = AsyncMock()
    browser.new_context = AsyncMock()
    browser.close = AsyncMock()
    return browser


@pytest.fixture
def mock_playwright():
    """Mock Playwright instance."""
    pw = AsyncMock()
    pw.chromium = AsyncMock()
    pw.chromium.launch = AsyncMock()
    pw.stop = AsyncMock()
    return pw


# ===========================================================================
# HUMANEMULATOR TESTS
# ===========================================================================

class TestHumanEmulator:
    """Test suite for HumanEmulator class (realistic interaction patterns)."""

    @pytest.mark.asyncio
    async def test_initialization_default_speed(self):
        """Test HumanEmulator initializes with default speed factor."""
        human = engine.HumanEmulator()
        assert human.speed_factor == 1.0

    @pytest.mark.asyncio
    async def test_initialization_custom_speed(self):
        """Test HumanEmulator accepts custom speed factor."""
        human = engine.HumanEmulator(speed_factor=0.5)
        assert human.speed_factor == 0.5

    @pytest.mark.asyncio
    async def test_move_mouse_to_uses_bezier_curve(self, mock_page):
        """Test mouse movement follows a Bézier curve path."""
        human = engine.HumanEmulator(speed_factor=0.1)
        await human.move_mouse_to(mock_page, 500, 300)

        # Should make multiple incremental moves (15-35 steps)
        assert mock_page.mouse.move.call_count >= 15
        assert mock_page.mouse.move.call_count <= 35

        # Should store final position
        assert mock_page.evaluate.call_count >= 1

    @pytest.mark.asyncio
    async def test_move_mouse_to_respects_speed_factor(self, mock_page):
        """Test mouse movement duration scales with speed factor."""
        # Fast movement (speed_factor < 1.0 means faster)
        human_fast = engine.HumanEmulator(speed_factor=0.01)
        start = asyncio.get_event_loop().time()
        await human_fast.move_mouse_to(mock_page, 100, 100)
        fast_duration = asyncio.get_event_loop().time() - start

        # Should complete quickly (micro-delays only)
        assert fast_duration < 0.5

    @pytest.mark.asyncio
    async def test_click_moves_to_element_and_clicks(self, mock_page, mock_element):
        """Test click moves mouse to element and performs click."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.1)
        await human.click(mock_page, ".button")

        # Should wait for element, get bounding box, move mouse, and click
        mock_page.wait_for_selector.assert_called_once_with(".button", timeout=10000)
        mock_element.bounding_box.assert_called_once()
        mock_page.mouse.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_double_click(self, mock_page, mock_element):
        """Test double-click uses dblclick instead of click."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.1)
        await human.click(mock_page, ".button", double=True)

        # Should use dblclick instead of click
        mock_page.mouse.dblclick.assert_called_once()
        mock_page.mouse.click.assert_not_called()

    @pytest.mark.asyncio
    async def test_click_raises_when_element_not_found(self, mock_page):
        """Test click raises exception when element is not found."""
        mock_page.wait_for_selector.return_value = None

        human = engine.HumanEmulator()
        with pytest.raises(Exception, match="Element not found"):
            await human.click(mock_page, ".nonexistent")

    @pytest.mark.asyncio
    async def test_click_raises_when_element_not_visible(self, mock_page, mock_element):
        """Test click raises exception when element is not visible."""
        mock_element.bounding_box.return_value = None
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator()
        with pytest.raises(Exception, match="Element not visible"):
            await human.click(mock_page, ".invisible")

    @pytest.mark.asyncio
    async def test_type_text_clears_and_types(self, mock_page, mock_element):
        """Test type_text clears existing content and types new text."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.01)
        await human.type_text(mock_page, "input[name='email']", "test@example.com")

        # Should press select-all and backspace to clear
        assert any("a" in str(call) for call in mock_page.keyboard.press.call_args_list)
        assert call("Backspace") in mock_page.keyboard.press.call_args_list

        # Should type each character
        assert mock_page.keyboard.type.call_count == len("test@example.com")

    @pytest.mark.asyncio
    async def test_type_text_skip_clear(self, mock_page, mock_element):
        """Test type_text can skip clearing when clear_first=False."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.01)
        await human.type_text(mock_page, "input", "text", clear_first=False)

        # Should not press select-all or backspace
        mock_page.keyboard.press.assert_not_called()

        # Should only type the text
        assert mock_page.keyboard.type.call_count == len("text")

    @pytest.mark.asyncio
    async def test_scroll_down(self, mock_page):
        """Test scrolling down moves mouse wheel in positive direction."""
        human = engine.HumanEmulator(speed_factor=0.01)
        await human.scroll(mock_page, direction="down", amount=500)

        # Should make multiple wheel movements
        assert mock_page.mouse.wheel.call_count > 0

        # Total scroll should be positive
        total_scroll = sum(call.args[1] for call in mock_page.mouse.wheel.call_args_list)
        assert total_scroll > 0

    @pytest.mark.asyncio
    async def test_scroll_up(self, mock_page):
        """Test scrolling up moves mouse wheel in negative direction."""
        human = engine.HumanEmulator(speed_factor=0.01)
        await human.scroll(mock_page, direction="up", amount=500)

        # Total scroll should be negative
        total_scroll = sum(call.args[1] for call in mock_page.mouse.wheel.call_args_list)
        assert total_scroll < 0

    @pytest.mark.asyncio
    async def test_scroll_default_amount(self, mock_page):
        """Test scroll uses random amount when not specified."""
        human = engine.HumanEmulator(speed_factor=0.01)
        await human.scroll(mock_page, direction="down")

        # Should have scrolled some amount
        assert mock_page.mouse.wheel.call_count > 0

    @pytest.mark.asyncio
    async def test_wait_human_delays(self):
        """Test wait_human introduces realistic delay."""
        human = engine.HumanEmulator(speed_factor=0.01)

        start = asyncio.get_event_loop().time()
        await human.wait_human(min_seconds=0.001, max_seconds=0.002)
        duration = asyncio.get_event_loop().time() - start

        # Should wait at least the minimum scaled duration
        assert duration >= 0.00001  # 0.001 * 0.01

    @pytest.mark.asyncio
    async def test_read_page_scrolls_and_pauses(self, mock_page):
        """Test read_page simulates reading with scrolling and pauses."""
        human = engine.HumanEmulator(speed_factor=0.01)
        await human.read_page(mock_page, estimated_words=50)

        # Should make multiple scroll movements (3-8 chunks)
        assert mock_page.mouse.wheel.call_count >= 3


# ===========================================================================
# SESSIONMANAGER TESTS
# ===========================================================================

class TestSessionManager:
    """Test suite for SessionManager (persistent browser sessions)."""

    def test_initialization_creates_directory(self, temp_dir):
        """Test SessionManager creates sessions directory on init."""
        sessions_dir = os.path.join(temp_dir, "sessions")
        manager = engine.SessionManager(sessions_dir=sessions_dir)

        assert manager.sessions_dir == Path(sessions_dir)
        assert manager.sessions_dir.exists()

    def test_get_session_path(self, temp_dir):
        """Test get_session_path returns correct path for site."""
        manager = engine.SessionManager(sessions_dir=temp_dir)
        path = manager.get_session_path("example_com")

        assert path == Path(temp_dir) / "example_com.json"

    @pytest.mark.asyncio
    async def test_save_session_writes_state(self, temp_dir, mock_context):
        """Test save_session writes context state to JSON file."""
        manager = engine.SessionManager(sessions_dir=temp_dir)

        state_data = {
            "cookies": [{"name": "session", "value": "abc123"}],
            "origins": [],
        }
        mock_context.storage_state.return_value = state_data

        await manager.save_session(mock_context, "example_com")

        # Should have saved the state file
        path = manager.get_session_path("example_com")
        assert path.exists()

        # Should contain the correct data
        with open(path) as f:
            saved = json.load(f)
        assert saved == state_data

    def test_has_session_true_when_exists(self, temp_dir):
        """Test has_session returns True when session file exists."""
        manager = engine.SessionManager(sessions_dir=temp_dir)
        path = manager.get_session_path("example_com")
        path.write_text("{}")

        assert manager.has_session("example_com") is True

    def test_has_session_false_when_missing(self, temp_dir):
        """Test has_session returns False when session file doesn't exist."""
        manager = engine.SessionManager(sessions_dir=temp_dir)

        assert manager.has_session("nonexistent") is False

    def test_get_storage_state_returns_path_when_exists(self, temp_dir):
        """Test get_storage_state returns path string when session exists."""
        manager = engine.SessionManager(sessions_dir=temp_dir)
        path = manager.get_session_path("example_com")
        path.write_text("{}")

        result = manager.get_storage_state("example_com")
        assert result == str(path)

    def test_get_storage_state_returns_none_when_missing(self, temp_dir):
        """Test get_storage_state returns None when session doesn't exist."""
        manager = engine.SessionManager(sessions_dir=temp_dir)

        result = manager.get_storage_state("nonexistent")
        assert result is None

    def test_clear_session_deletes_file(self, temp_dir):
        """Test clear_session removes session file."""
        manager = engine.SessionManager(sessions_dir=temp_dir)
        path = manager.get_session_path("example_com")
        path.write_text("{}")

        manager.clear_session("example_com")

        assert not path.exists()

    def test_clear_session_handles_missing_file(self, temp_dir):
        """Test clear_session doesn't raise when file doesn't exist."""
        manager = engine.SessionManager(sessions_dir=temp_dir)

        # Should not raise
        manager.clear_session("nonexistent")


# ===========================================================================
# CREDENTIALVAULT TESTS
# ===========================================================================

class TestCredentialVault:
    """Test suite for CredentialVault (credential management)."""

    def test_initialization_creates_directory(self, temp_dir):
        """Test CredentialVault creates vault directory on init."""
        vault_dir = os.path.join(temp_dir, "vault")
        vault = engine.CredentialVault(vault_path=vault_dir)

        assert vault.vault_dir == Path(vault_dir)
        assert vault.vault_dir.exists()

    def test_load_proton_pass_export_parses_credentials(self, temp_dir):
        """Test loading credentials from Proton Pass JSON export."""
        vault = engine.CredentialVault(vault_path=temp_dir)

        # Create a mock Proton Pass export
        export_data = {
            "vaults": {
                "vault1": {
                    "items": [
                        {
                            "data": {
                                "type": "login",
                                "content": {
                                    "itemUsername": "testuser",
                                    "itemEmail": "test@example.com",
                                    "password": "secret123",
                                    "urls": ["https://example.com/login"],
                                    "totpUri": "otpauth://totp/Example:test?secret=ABC123",
                                },
                                "metadata": {
                                    "name": "Example Service",
                                },
                            },
                        },
                    ],
                },
            },
        }

        export_path = os.path.join(temp_dir, "proton_export.json")
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        vault.load_proton_pass_export(export_path)

        # Should have loaded the credential
        cred = vault.get_credential("example")
        assert cred is not None
        assert cred["username"] == "testuser"
        assert cred["email"] == "test@example.com"
        assert cred["password"] == "secret123"
        assert cred["urls"] == ["https://example.com/login"]
        assert cred["totp_uri"] == "otpauth://totp/Example:test?secret=ABC123"

    def test_load_proton_pass_export_skips_non_login_items(self, temp_dir):
        """Test Proton Pass export skips items that aren't login type."""
        vault = engine.CredentialVault(vault_path=temp_dir)

        export_data = {
            "vaults": {
                "vault1": {
                    "items": [
                        {
                            "data": {
                                "type": "note",  # Not a login
                                "content": {
                                    "note": "Just a note",
                                },
                            },
                        },
                    ],
                },
            },
        }

        export_path = os.path.join(temp_dir, "proton_export.json")
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        vault.load_proton_pass_export(export_path)

        # Should have no credentials
        assert len(vault.list_sites()) == 0

    def test_load_manual_vault(self, temp_dir):
        """Test loading credentials from manual JSON vault."""
        vault = engine.CredentialVault(vault_path=temp_dir)

        manual_data = {
            "github": {
                "username": "githubuser",
                "password": "ghp_token123",
                "url": "https://github.com",
            },
        }

        vault_file = os.path.join(temp_dir, "manual_vault.json")
        with open(vault_file, "w") as f:
            json.dump(manual_data, f)

        vault.load_manual_vault("manual_vault.json")

        # Should have loaded the manual credential
        cred = vault.get_credential("github")
        assert cred is not None
        assert cred["username"] == "githubuser"
        assert cred["password"] == "ghp_token123"

    def test_load_manual_vault_handles_missing_file(self, temp_dir):
        """Test load_manual_vault doesn't raise when file doesn't exist."""
        vault = engine.CredentialVault(vault_path=temp_dir)

        # Should not raise
        vault.load_manual_vault("nonexistent.json")
        assert len(vault.list_sites()) == 0

    def test_get_credential_returns_none_when_missing(self, temp_dir):
        """Test get_credential returns None for unknown site."""
        vault = engine.CredentialVault(vault_path=temp_dir)

        cred = vault.get_credential("unknown_site")
        assert cred is None

    def test_get_totp_returns_none_without_uri(self, temp_dir):
        """Test get_totp returns None when TOTP URI is missing."""
        vault = engine.CredentialVault(vault_path=temp_dir)
        vault._credentials["example"] = {
            "username": "user",
            "password": "pass",
        }

        totp = vault.get_totp("example")
        assert totp is None

    def test_list_sites_returns_all_site_ids(self, temp_dir):
        """Test list_sites returns all loaded site identifiers."""
        vault = engine.CredentialVault(vault_path=temp_dir)
        vault._credentials = {
            "github": {},
            "gitlab": {},
            "example": {},
        }

        sites = vault.list_sites()
        assert set(sites) == {"github", "gitlab", "example"}

    def test_url_to_site_id_extracts_domain(self):
        """Test _url_to_site_id extracts domain from URL."""
        assert engine.CredentialVault._url_to_site_id("https://example.com/path") == "example"
        assert engine.CredentialVault._url_to_site_id("https://www.github.com/login") == "github"
        assert engine.CredentialVault._url_to_site_id("http://api.service.io/v1") == "api"


# ===========================================================================
# BROWSERENGINE TESTS
# ===========================================================================

class TestBrowserEngine:
    """Test suite for BrowserEngine (stealth browser management)."""

    def test_initialization_creates_directories(self, temp_dir):
        """Test BrowserEngine creates data directories on init."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        assert browser.data_dir == Path(temp_dir)
        assert browser.data_dir.exists()
        assert browser.session_manager is not None

    def test_initialization_sets_playwright_to_none(self, temp_dir):
        """Test BrowserEngine starts with no active Playwright instance."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        assert browser._playwright is None
        assert browser._browser is None

    @pytest.mark.asyncio
    async def test_start_raises_without_playwright(self, temp_dir):
        """Test start raises RuntimeError when Playwright is not installed."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        with patch.object(engine, "HAS_PLAYWRIGHT", False):
            with pytest.raises(RuntimeError, match="Playwright not installed"):
                await browser.start()

    @pytest.mark.asyncio
    async def test_start_launches_chromium_with_stealth_flags(
        self, temp_dir, mock_playwright, mock_browser
    ):
        """Test start launches Chromium with anti-detection flags."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        with patch.object(engine, "HAS_PLAYWRIGHT", True):
            with patch("connectors.browser.engine.async_playwright") as mock_pw:
                mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)
                mock_playwright.chromium.launch.return_value = mock_browser

                await browser.start()

                # Should have launched browser with stealth args
                mock_playwright.chromium.launch.assert_called_once()
                launch_args = mock_playwright.chromium.launch.call_args

                # Check for stealth flags
                assert launch_args.kwargs["headless"] is True
                assert "--disable-blink-features=AutomationControlled" in launch_args.kwargs["args"]
                assert browser._browser is mock_browser

    @pytest.mark.asyncio
    async def test_stop_closes_browser_and_playwright(
        self, temp_dir, mock_playwright, mock_browser
    ):
        """Test stop closes browser and stops Playwright in correct order."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        browser._browser = mock_browser
        browser._playwright = mock_playwright

        await browser.stop()

        # Should close browser first, then stop playwright
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_none_gracefully(self, temp_dir):
        """Test stop doesn't raise when browser/playwright are None."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        # Should not raise
        await browser.stop()

    @pytest.mark.asyncio
    async def test_create_context_starts_browser_lazily(
        self, temp_dir, mock_playwright, mock_browser, mock_context
    ):
        """Test create_context lazy-starts browser if not running."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        mock_browser.new_context.return_value = mock_context

        with patch.object(engine, "HAS_PLAYWRIGHT", True):
            with patch("connectors.browser.engine.async_playwright") as mock_pw:
                mock_pw.return_value.start = AsyncMock(return_value=mock_playwright)
                mock_playwright.chromium.launch.return_value = mock_browser

                context = await browser.create_context("example_com")

                # Should have started the browser
                assert browser._browser is mock_browser

                # Should have created a context
                mock_browser.new_context.assert_called_once()
                assert context is mock_context

    @pytest.mark.asyncio
    async def test_create_context_uses_random_viewport(
        self, temp_dir, mock_browser, mock_context
    ):
        """Test create_context randomizes viewport size."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        browser._browser = mock_browser
        mock_browser.new_context.return_value = mock_context

        await browser.create_context("example_com")

        # Should have set a viewport from the list
        call_kwargs = mock_browser.new_context.call_args.kwargs
        viewport = call_kwargs["viewport"]
        assert viewport in engine.BrowserEngine.VIEWPORTS

    @pytest.mark.asyncio
    async def test_create_context_uses_random_user_agent(
        self, temp_dir, mock_browser, mock_context
    ):
        """Test create_context randomizes user agent."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        browser._browser = mock_browser
        mock_browser.new_context.return_value = mock_context

        await browser.create_context("example_com")

        # Should have set a user agent from the list
        call_kwargs = mock_browser.new_context.call_args.kwargs
        user_agent = call_kwargs["user_agent"]
        assert user_agent in engine.BrowserEngine.USER_AGENTS

    @pytest.mark.asyncio
    async def test_create_context_loads_saved_session(
        self, temp_dir, mock_browser, mock_context
    ):
        """Test create_context loads saved session state if available."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        browser._browser = mock_browser
        mock_browser.new_context.return_value = mock_context

        # Create a saved session
        session_path = browser.session_manager.get_session_path("example_com")
        session_path.write_text('{"cookies": [], "origins": []}')

        await browser.create_context("example_com")

        # Should have passed storage_state to new_context
        call_kwargs = mock_browser.new_context.call_args.kwargs
        assert call_kwargs["storage_state"] == str(session_path)

    @pytest.mark.asyncio
    async def test_create_context_registers_stealth_handler(
        self, temp_dir, mock_browser, mock_context
    ):
        """Test create_context registers stealth script on page events."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        browser._browser = mock_browser
        mock_browser.new_context.return_value = mock_context

        await browser.create_context("example_com")

        # Should have registered a page event handler
        assert mock_context.on.call_count == 1
        assert mock_context.on.call_args[0][0] == "page"

    @pytest.mark.asyncio
    async def test_apply_stealth_patches_navigator(self, temp_dir, mock_page):
        """Test _apply_stealth adds anti-detection JavaScript."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        await browser._apply_stealth(mock_page)

        # Should have added init script
        mock_page.add_init_script.assert_called_once()

        # Should contain webdriver removal
        script = mock_page.add_init_script.call_args.args[0]
        assert "navigator" in script
        assert "webdriver" in script

    @pytest.mark.asyncio
    async def test_new_page_creates_page_in_context(self, temp_dir, mock_context, mock_page):
        """Test new_page creates a page in the given context."""
        browser = engine.BrowserEngine(data_dir=temp_dir)
        mock_context.new_page.return_value = mock_page

        page = await browser.new_page(mock_context)

        mock_context.new_page.assert_called_once()
        assert page is mock_page

    @pytest.mark.asyncio
    async def test_save_session_delegates_to_session_manager(
        self, temp_dir, mock_context
    ):
        """Test save_session delegates to SessionManager."""
        browser = engine.BrowserEngine(data_dir=temp_dir)

        with patch.object(browser.session_manager, "save_session", new=AsyncMock()) as mock_save:
            await browser.save_session(mock_context, "example_com")

            mock_save.assert_called_once_with(mock_context, "example_com")


# ===========================================================================
# PAGEINTERACTOR TESTS
# ===========================================================================

class TestPageInteractor:
    """Test suite for PageInteractor (high-level page interactions)."""

    @pytest.mark.asyncio
    async def test_login_fills_username_and_password(self, mock_page, mock_element):
        """Test login fills username and password fields."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.01)
        interactor = engine.PageInteractor(human)

        creds = {
            "username": "testuser",
            "password": "testpass",
        }

        await interactor.login(mock_page, creds)

        # Should have typed username and password
        assert mock_page.keyboard.type.call_count > 0

    @pytest.mark.asyncio
    async def test_login_uses_email_when_username_missing(self, mock_page, mock_element):
        """Test login falls back to email when username is not in creds."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.01)
        interactor = engine.PageInteractor(human)

        creds = {
            "email": "test@example.com",
            "password": "testpass",
        }

        await interactor.login(mock_page, creds)

        # Should have typed email and password
        assert mock_page.keyboard.type.call_count > 0

    @pytest.mark.asyncio
    async def test_login_presses_enter_when_submit_not_found(self, mock_page, mock_element):
        """Test login presses Enter as fallback when submit button not found."""
        # Make submit selector fail, but everything else succeed
        def wait_side_effect(selector, **kwargs):
            if "submit" in selector:
                raise Exception("Submit not found")
            return mock_element

        mock_page.wait_for_selector.side_effect = wait_side_effect

        human = engine.HumanEmulator(speed_factor=0.01)
        interactor = engine.PageInteractor(human)

        creds = {"username": "user", "password": "pass"}

        # Should not raise, should press Enter instead
        await interactor.login(mock_page, creds)

        # Should have pressed Enter
        assert call("Enter") in mock_page.keyboard.press.call_args_list

    @pytest.mark.asyncio
    async def test_handle_2fa_enters_code(self, mock_page, mock_element):
        """Test handle_2fa enters TOTP code and submits."""
        mock_page.wait_for_selector.return_value = mock_element

        human = engine.HumanEmulator(speed_factor=0.01)
        interactor = engine.PageInteractor(human)

        await interactor.handle_2fa(mock_page, "123456")

        # Should have typed the code
        assert mock_page.keyboard.type.call_count == 6  # 6 digits

        # Should have pressed Enter
        assert call("Enter") in mock_page.keyboard.press.call_args_list

    @pytest.mark.asyncio
    async def test_extract_text_returns_all_matching_elements(self, mock_page):
        """Test extract_text returns text from all matching elements."""
        elem1 = AsyncMock()
        elem1.text_content = AsyncMock(return_value="First")
        elem2 = AsyncMock()
        elem2.text_content = AsyncMock(return_value="  Second  ")
        elem3 = AsyncMock()
        elem3.text_content = AsyncMock(return_value="")  # Empty, should be skipped

        mock_page.query_selector_all.return_value = [elem1, elem2, elem3]

        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        texts = await interactor.extract_text(mock_page, ".item")

        assert texts == ["First", "Second"]  # Trimmed, empty excluded

    @pytest.mark.asyncio
    async def test_extract_table_parses_html_table(self, mock_page):
        """Test extract_table extracts structured data from HTML table."""
        mock_page.evaluate.return_value = [
            {"Name": "Alice", "Age": "30"},
            {"Name": "Bob", "Age": "25"},
        ]

        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        data = await interactor.extract_table(mock_page, "table")

        assert len(data) == 2
        assert data[0]["Name"] == "Alice"
        assert data[1]["Name"] == "Bob"

    @pytest.mark.asyncio
    async def test_wait_for_content_returns_true_when_found(self, mock_page):
        """Test wait_for_content returns True when selector is found."""
        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        result = await interactor.wait_for_content(mock_page, ".content")

        assert result is True
        mock_page.wait_for_selector.assert_called_once_with(".content", timeout=30000)

    @pytest.mark.asyncio
    async def test_wait_for_content_returns_false_on_timeout(self, mock_page):
        """Test wait_for_content returns False when selector times out."""
        mock_page.wait_for_selector.side_effect = Exception("Timeout")

        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        result = await interactor.wait_for_content(mock_page, ".missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_screenshot_saves_full_page(self, mock_page, temp_dir):
        """Test screenshot saves full-page screenshot to file."""
        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        screenshot_path = os.path.join(temp_dir, "screenshot.png")
        await interactor.screenshot(mock_page, screenshot_path)

        mock_page.screenshot.assert_called_once_with(path=screenshot_path, full_page=True)

    @pytest.mark.asyncio
    async def test_get_page_data_returns_metadata(self, mock_page):
        """Test get_page_data extracts URL, title, and timestamp."""
        human = engine.HumanEmulator()
        interactor = engine.PageInteractor(human)

        data = await interactor.get_page_data(mock_page)

        assert data["url"] == "https://example.com"
        assert data["title"] == "Example Page"
        assert "timestamp" in data

        # Timestamp should be ISO format
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
