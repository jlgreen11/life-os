"""
Tests: Logging migration for browser connectors.

Verifies that print() has been replaced with logging calls in:
  - connectors/browser/orchestrator.py   (BrowserOrchestrator, APIFallbackWrapper)
  - connectors/browser/base_connector.py (BrowserBaseConnector)
  - connectors/browser/engine.py         (CredentialVault.get_totp)
  - connectors/browser/whatsapp.py       (WhatsAppConnector)

Each test verifies that:
  1. The module-level ``logger`` attribute is properly defined.
  2. Key error/warning paths emit structured log records instead of writing to stdout.
  3. No ``print(`` calls remain in the relevant source code.
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_print_in_source(source: str, context: str) -> None:
    """Assert that there are no bare print() calls in *source*."""
    assert "print(" not in source, (
        f"{context} still contains print() calls — should use logger instead"
    )


# ---------------------------------------------------------------------------
# BrowserOrchestrator
# ---------------------------------------------------------------------------

class TestBrowserOrchestratorLogging:
    """BrowserOrchestrator emits log records instead of print()."""

    def test_logger_defined(self):
        """The orchestrator module must define a module-level logger."""
        import connectors.browser.orchestrator as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.browser.orchestrator"

    def test_no_print_in_start(self):
        """BrowserOrchestrator.start() source must not contain print()."""
        from connectors.browser.orchestrator import BrowserOrchestrator
        src = inspect.getsource(BrowserOrchestrator.start)
        _no_print_in_source(src, "BrowserOrchestrator.start")

    def test_no_print_in_start_connectors(self):
        """BrowserOrchestrator.start_connectors() source must not contain print()."""
        from connectors.browser.orchestrator import BrowserOrchestrator
        src = inspect.getsource(BrowserOrchestrator.start_connectors)
        _no_print_in_source(src, "BrowserOrchestrator.start_connectors")

    def test_no_print_in_api_fallback_wrapper_sync(self):
        """APIFallbackWrapper.sync() source must not contain print()."""
        from connectors.browser.orchestrator import APIFallbackWrapper
        src = inspect.getsource(APIFallbackWrapper.sync)
        _no_print_in_source(src, "APIFallbackWrapper.sync")

    def test_no_print_in_api_fallback_browser_sync(self):
        """APIFallbackWrapper._browser_fallback_sync() source must not contain print()."""
        from connectors.browser.orchestrator import APIFallbackWrapper
        src = inspect.getsource(APIFallbackWrapper._browser_fallback_sync)
        _no_print_in_source(src, "APIFallbackWrapper._browser_fallback_sync")

    @pytest.mark.asyncio
    async def test_start_connectors_failure_logs_error(self, caplog):
        """start_connectors() logs ERROR when a connector fails to start."""
        from connectors.browser.orchestrator import BrowserOrchestrator

        bus = MagicMock()
        db = MagicMock()
        config = {"browser": {"enabled": True}}
        orchestrator = BrowserOrchestrator(bus, db, config)

        # Inject a fake connector that raises on start()
        fake_connector = MagicMock()
        fake_connector.DISPLAY_NAME = "FakeConnector"
        fake_connector.start = AsyncMock(side_effect=RuntimeError("browser crash"))
        orchestrator._connectors = [fake_connector]

        with caplog.at_level(logging.ERROR, logger="connectors.browser.orchestrator"):
            await orchestrator.start_connectors()

        assert any("browser crash" in r.message for r in caplog.records), (
            "Expected error message not found in log records"
        )

    @pytest.mark.asyncio
    async def test_api_fallback_wrapper_api_failure_logs_warning(self, caplog):
        """APIFallbackWrapper.sync() logs WARNING when the API call fails."""
        from connectors.browser.orchestrator import APIFallbackWrapper

        api_connector = MagicMock()
        api_connector.CONNECTOR_ID = "test_connector"
        api_connector.sync = AsyncMock(side_effect=RuntimeError("rate limited"))

        engine = MagicMock()
        vault = MagicMock()
        wrapper = APIFallbackWrapper(
            api_connector=api_connector,
            browser_engine=engine,
            credential_vault=vault,
            fallback_config={"failure_threshold": 3},
        )

        with caplog.at_level(logging.WARNING, logger="connectors.browser.orchestrator"):
            await wrapper.sync()

        assert any("rate limited" in r.message for r in caplog.records), (
            "Expected warning message not found in log records"
        )

    @pytest.mark.asyncio
    async def test_api_fallback_wrapper_no_fallback_logs_warning(self, caplog):
        """_browser_fallback_sync() logs WARNING about missing implementation."""
        from connectors.browser.orchestrator import APIFallbackWrapper

        api_connector = MagicMock()
        api_connector.CONNECTOR_ID = "test_connector"
        api_connector.sync = AsyncMock(side_effect=RuntimeError("api down"))

        wrapper = APIFallbackWrapper(
            api_connector=api_connector,
            browser_engine=MagicMock(),
            credential_vault=MagicMock(),
            fallback_config={"failure_threshold": 1},
        )
        # Exhaust API threshold on the first call
        with caplog.at_level(logging.WARNING, logger="connectors.browser.orchestrator"):
            result = await wrapper.sync()

        # Should return 0 and log a warning about missing fallback
        assert result == 0
        assert any("No browser fallback implemented" in r.message for r in caplog.records), (
            "Expected 'No browser fallback implemented' warning not found"
        )


# ---------------------------------------------------------------------------
# BrowserBaseConnector
# ---------------------------------------------------------------------------

class TestBrowserBaseConnectorLogging:
    """BrowserBaseConnector error paths emit log records instead of print()."""

    def _make_connector(self):
        """Build a minimal concrete BrowserBaseConnector subclass."""
        from connectors.browser.base_connector import BrowserBaseConnector

        class DummyBrowserConnector(BrowserBaseConnector):
            CONNECTOR_ID = "dummy_browser"
            DISPLAY_NAME = "Dummy Browser"
            SITE_ID = "dummy"
            LOGIN_URL = "https://example.com/login"

            async def browser_sync(self, page, human, interactor):
                return 0

            async def execute(self, action, params):
                return {}

            async def health_check(self):
                return {"status": "ok"}

        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()

        # Inject stub engine and vault to avoid real browser startup
        engine = MagicMock()
        engine.start = AsyncMock()
        engine.session_manager = MagicMock()
        engine.session_manager.has_session.return_value = False
        vault = MagicMock()
        vault.get_credential.return_value = None  # Force "no credentials" path

        connector = DummyBrowserConnector(
            event_bus=bus, db=db, config={},
            browser_engine=engine, credential_vault=vault,
        )
        return connector

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.browser.base_connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.browser.base_connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.browser.base_connector import BrowserBaseConnector
        src = inspect.getsource(BrowserBaseConnector.authenticate)
        _no_print_in_source(src, "BrowserBaseConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.browser.base_connector import BrowserBaseConnector
        src = inspect.getsource(BrowserBaseConnector.sync)
        _no_print_in_source(src, "BrowserBaseConnector.sync")

    def test_no_print_in_browser_login(self):
        """_browser_login() source must not contain print()."""
        from connectors.browser.base_connector import BrowserBaseConnector
        src = inspect.getsource(BrowserBaseConnector._browser_login)
        _no_print_in_source(src, "BrowserBaseConnector._browser_login")

    @pytest.mark.asyncio
    async def test_no_credentials_logs_error(self, caplog):
        """_browser_login() logs ERROR when no credentials are found in the vault."""
        connector = self._make_connector()
        # vault already returns None from get_credential (set in _make_connector)

        with caplog.at_level(logging.ERROR, logger="connectors.browser.base_connector"):
            result = await connector._browser_login()

        assert result is False
        assert any("No credentials found" in r.message for r in caplog.records), (
            "Expected 'No credentials found' error log not found"
        )

    @pytest.mark.asyncio
    async def test_api_sync_failure_logs_warning(self, caplog):
        """sync() logs WARNING when the API call fails."""
        from connectors.browser.base_connector import BrowserBaseConnector

        class APIBrowserConnector(BrowserBaseConnector):
            CONNECTOR_ID = "api_test"
            DISPLAY_NAME = "API Test"
            SITE_ID = "api_test"
            LOGIN_URL = "https://example.com/login"

            async def api_sync(self):
                raise RuntimeError("quota exceeded")

            async def browser_sync(self, page, human, interactor):
                return 5

            async def execute(self, action, params):
                return {}

            async def health_check(self):
                return {"status": "ok"}

        engine = MagicMock()
        engine.start = AsyncMock()
        engine.session_manager = MagicMock()
        engine.session_manager.has_session.return_value = False
        vault = MagicMock()
        vault.get_credential.return_value = {"username": "u", "password": "p"}

        connector = APIBrowserConnector(
            event_bus=MagicMock(), db=MagicMock(),
            config={"prefer_api": True, "api_failure_threshold": 3},
            browser_engine=engine, credential_vault=vault,
        )

        with caplog.at_level(logging.WARNING, logger="connectors.browser.base_connector"):
            # Patch _browser_sync_wrapper to avoid actual browser calls
            connector._browser_sync_wrapper = AsyncMock(return_value=0)
            await connector.sync()

        assert any("quota exceeded" in r.message for r in caplog.records), (
            "Expected API failure warning not found in log records"
        )


# ---------------------------------------------------------------------------
# CredentialVault (engine.py)
# ---------------------------------------------------------------------------

class TestBrowserEngineLogging:
    """CredentialVault.get_totp() emits a log record instead of print()."""

    def test_logger_defined(self):
        """The engine module must define a module-level logger."""
        import connectors.browser.engine as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.browser.engine"

    def test_no_print_in_get_totp(self):
        """CredentialVault.get_totp() source must not contain print()."""
        from connectors.browser.engine import CredentialVault
        src = inspect.getsource(CredentialVault.get_totp)
        _no_print_in_source(src, "CredentialVault.get_totp")

    def test_get_totp_missing_pyotp_logs_warning(self, caplog):
        """get_totp() logs WARNING when pyotp is not installed."""
        from connectors.browser.engine import CredentialVault

        vault = CredentialVault(vault_path="/tmp/test_vault")
        # Inject a credential entry with a totp_uri so we reach the import
        vault._credentials = {"example": {"totp_uri": "otpauth://totp/test?secret=BASE32SECRET"}}

        with patch("builtins.__import__", side_effect=ImportError("No module named 'pyotp'")):
            with caplog.at_level(logging.WARNING, logger="connectors.browser.engine"):
                result = vault.get_totp("example")

        assert result is None
        assert any("pyotp" in r.message for r in caplog.records), (
            "Expected pyotp warning not found in log records"
        )


# ---------------------------------------------------------------------------
# WhatsAppConnector
# ---------------------------------------------------------------------------

class TestWhatsAppConnectorLogging:
    """WhatsAppConnector emits log records instead of print()."""

    def _make_connector(self):
        from connectors.browser.whatsapp import WhatsAppConnector

        engine = MagicMock()
        engine.start = AsyncMock()
        engine.session_manager = MagicMock()
        engine.session_manager.has_session.return_value = False
        engine.create_context = AsyncMock(return_value=MagicMock())
        engine.new_page = AsyncMock(return_value=MagicMock())
        engine.save_session = AsyncMock()

        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()

        connector = WhatsAppConnector(
            event_bus=bus, db=db, config={},
            browser_engine=engine, credential_vault=MagicMock(),
        )
        return connector

    def test_logger_defined(self):
        """The whatsapp module must define a module-level logger."""
        import connectors.browser.whatsapp as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.browser.whatsapp"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.browser.whatsapp import WhatsAppConnector
        src = inspect.getsource(WhatsAppConnector.authenticate)
        _no_print_in_source(src, "WhatsAppConnector.authenticate")

    def test_no_print_in_browser_sync(self):
        """browser_sync() source must not contain print()."""
        from connectors.browser.whatsapp import WhatsAppConnector
        src = inspect.getsource(WhatsAppConnector.browser_sync)
        _no_print_in_source(src, "WhatsAppConnector.browser_sync")

    @pytest.mark.asyncio
    async def test_authenticate_qr_timeout_logs_warning(self, caplog):
        """authenticate() logs WARNING when QR code scan times out."""
        connector = self._make_connector()

        # Page always reports not logged in (simulates QR timeout)
        fake_page = MagicMock()
        connector._browser_engine.new_page = AsyncMock(return_value=fake_page)
        fake_page.goto = AsyncMock()

        with patch.object(connector, "is_logged_in", AsyncMock(return_value=False)):
            with patch.object(connector._human, "wait_human", AsyncMock()):
                with patch.object(connector._interactor, "screenshot", AsyncMock()):
                    with caplog.at_level(logging.WARNING, logger="connectors.browser.whatsapp"):
                        # Patch the polling range to 1 iteration for speed
                        with patch("builtins.range", return_value=range(1)):
                            result = await connector.authenticate()

        assert result is False
        assert any("timed out" in r.message for r in caplog.records), (
            "Expected QR timeout warning not found in log records"
        )

    @pytest.mark.asyncio
    async def test_authenticate_exception_logs_error(self, caplog):
        """authenticate() logs ERROR when an unexpected exception occurs."""
        connector = self._make_connector()

        # Simulate startup failure
        connector._browser_engine.start = AsyncMock(side_effect=RuntimeError("playwright not installed"))

        with caplog.at_level(logging.ERROR, logger="connectors.browser.whatsapp"):
            result = await connector.authenticate()

        assert result is False
        assert any("playwright not installed" in r.message for r in caplog.records), (
            "Expected auth error log not found in log records"
        )

    @pytest.mark.asyncio
    async def test_browser_sync_chat_error_logs_warning(self, caplog):
        """browser_sync() logs WARNING when reading an individual chat fails."""
        from connectors.browser.engine import HumanEmulator, PageInteractor
        connector = self._make_connector()

        # Return one unread chat
        fake_page = MagicMock()
        fake_page.evaluate = AsyncMock(return_value=[{"name": "Alice", "unread_count": 1}])
        fake_human = MagicMock()
        fake_human.click = AsyncMock(side_effect=RuntimeError("element not found"))
        fake_human.wait_human = AsyncMock()
        fake_interactor = MagicMock()

        with caplog.at_level(logging.WARNING, logger="connectors.browser.whatsapp"):
            await connector.browser_sync(fake_page, fake_human, fake_interactor)

        assert any("element not found" in r.message for r in caplog.records), (
            "Expected chat-read warning not found in log records"
        )
