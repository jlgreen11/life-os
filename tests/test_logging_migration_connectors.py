"""
Tests: Logging migration for non-browser connectors.

Verifies that print() has been replaced with logging calls in:
  - connectors/base/connector.py     (BaseConnector._handle_sync_error)
  - connectors/home_assistant/connector.py  (HomeAssistantConnector)
  - connectors/caldav/connector.py   (CalDAVConnector)
  - connectors/proton_mail/connector.py    (ProtonMailConnector)
  - connectors/finance/connector.py  (FinanceConnector)
  - connectors/imessage/connector.py (iMessageConnector)
  - connectors/signal_msg/connector.py     (SignalConnector)

Each test verifies that:
  1. The module-level ``logger`` attribute is properly defined.
  2. Error/warning paths emit structured log records instead of writing to stdout.
  3. No ``print(`` calls remain in the relevant source code.
"""

from __future__ import annotations

import inspect
import logging
import tempfile
from pathlib import Path
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
# BaseConnector
# ---------------------------------------------------------------------------

class TestBaseConnectorLogging:
    """BaseConnector._handle_sync_error() emits a log record instead of print()."""

    def test_logger_defined(self):
        """The connectors.base.connector module must define a module-level logger."""
        import connectors.base.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.base.connector"

    def test_no_print_in_handle_sync_error(self):
        """_handle_sync_error source must not contain print()."""
        from connectors.base.connector import BaseConnector
        src = inspect.getsource(BaseConnector._handle_sync_error)
        _no_print_in_source(src, "BaseConnector._handle_sync_error")

    @pytest.mark.asyncio
    async def test_handle_sync_error_logs_error(self, caplog):
        """_handle_sync_error() emits an ERROR log record for the error message."""
        from connectors.base.connector import BaseConnector

        # Build a minimal concrete subclass so we can instantiate BaseConnector.
        class DummyConnector(BaseConnector):
            CONNECTOR_ID = "dummy"
            DISPLAY_NAME = "Dummy"

            async def authenticate(self): return True
            async def sync(self): return 0
            async def execute(self, action, params): return {}
            async def health_check(self): return {"status": "ok"}

        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        connector = DummyConnector(event_bus=bus, db=db, config={})

        error = RuntimeError("network timeout")
        with caplog.at_level(logging.ERROR, logger="connectors.base.connector"):
            await connector._handle_sync_error(error)

        # The error message should appear in a log record, not in stdout.
        assert any("network timeout" in r.message for r in caplog.records), (
            "Expected error message not found in log records"
        )


# ---------------------------------------------------------------------------
# HomeAssistantConnector
# ---------------------------------------------------------------------------

class TestHomeAssistantConnectorLogging:
    """HomeAssistantConnector error paths emit log records instead of print()."""

    def _make_connector(self):
        from connectors.home_assistant.connector import HomeAssistantConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "url": "http://localhost:8123",
            "token": "fake-token",
            "watched_entities": ["person.user"],
        }
        return HomeAssistantConnector(event_bus=bus, db=db, config=config)

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.home_assistant.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.home_assistant.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.home_assistant.connector import HomeAssistantConnector
        src = inspect.getsource(HomeAssistantConnector.authenticate)
        _no_print_in_source(src, "HomeAssistantConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.home_assistant.connector import HomeAssistantConnector
        src = inspect.getsource(HomeAssistantConnector.sync)
        _no_print_in_source(src, "HomeAssistantConnector.sync")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, caplog):
        """authenticate() logs ERROR when the HTTP request fails."""
        connector = self._make_connector()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.side_effect = Exception("connection refused")
            mock_client_cls.return_value = mock_client

            with caplog.at_level(logging.ERROR, logger="connectors.home_assistant.connector"):
                result = await connector.authenticate()

        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sync_entity_error_logs_warning(self, caplog):
        """sync() logs WARNING when an individual entity read fails."""
        connector = self._make_connector()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.side_effect = Exception("entity not found")
            mock_client_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING, logger="connectors.home_assistant.connector"):
                count = await connector.sync()

        # sync() swallows per-entity errors and returns 0 events published.
        assert count == 0
        assert any("Error reading entity" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CalDAVConnector
# ---------------------------------------------------------------------------

class TestCalDAVConnectorLogging:
    """CalDAVConnector error paths emit log records instead of print()."""

    def _make_connector(self):
        from connectors.caldav.connector import CalDAVConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "url": "https://cal.example.com",
            "username": "user@example.com",
            "password": "secret",
        }
        return CalDAVConnector(event_bus=bus, db=db, config=config)

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.caldav.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.caldav.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.caldav.connector import CalDAVConnector
        src = inspect.getsource(CalDAVConnector.authenticate)
        _no_print_in_source(src, "CalDAVConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.caldav.connector import CalDAVConnector
        src = inspect.getsource(CalDAVConnector.sync)
        _no_print_in_source(src, "CalDAVConnector.sync")

    def test_no_print_in_detect_conflicts(self):
        """_detect_conflicts() source must not contain print()."""
        from connectors.caldav.connector import CalDAVConnector
        src = inspect.getsource(CalDAVConnector._detect_conflicts)
        _no_print_in_source(src, "CalDAVConnector._detect_conflicts")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, caplog):
        """authenticate() logs ERROR when the caldav import or auth fails."""
        connector = self._make_connector()

        with patch("builtins.__import__", side_effect=ImportError("No module named 'caldav'")):
            with caplog.at_level(logging.ERROR, logger="connectors.caldav.connector"):
                result = await connector.authenticate()

        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_conflict_detection_error_logs_error(self, caplog):
        """_detect_conflicts() logs ERROR when an unexpected exception occurs."""
        connector = self._make_connector()

        # db.get_connection raises so conflict detection hits the outer except.
        connector.db = MagicMock()
        connector.db.get_connection.side_effect = RuntimeError("db locked")

        with caplog.at_level(logging.ERROR, logger="connectors.caldav.connector"):
            # Should not raise — fail-open design.
            await connector._detect_conflicts()

        assert any("Conflict detection error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ProtonMailConnector
# ---------------------------------------------------------------------------

class TestProtonMailConnectorLogging:
    """ProtonMailConnector error paths emit log records instead of print()."""

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.proton_mail.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.proton_mail.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.proton_mail.connector import ProtonMailConnector
        src = inspect.getsource(ProtonMailConnector.authenticate)
        _no_print_in_source(src, "ProtonMailConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.proton_mail.connector import ProtonMailConnector
        src = inspect.getsource(ProtonMailConnector.sync)
        _no_print_in_source(src, "ProtonMailConnector.sync")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, caplog):
        """authenticate() logs ERROR when IMAP connection fails."""
        from connectors.proton_mail.connector import ProtonMailConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "imap_host": "127.0.0.1",
            "imap_port": 1143,
            "username": "user@proton.me",
            "password": "bad-password",
        }
        connector = ProtonMailConnector(event_bus=bus, db=db, config=config)

        with patch("imaplib.IMAP4", side_effect=ConnectionRefusedError("Connection refused")):
            with caplog.at_level(logging.ERROR, logger="connectors.proton_mail.connector"):
                result = await connector.authenticate()

        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FinanceConnector
# ---------------------------------------------------------------------------

class TestFinanceConnectorLogging:
    """FinanceConnector error paths emit log records instead of print()."""

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.finance.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.finance.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.finance.connector import FinanceConnector
        src = inspect.getsource(FinanceConnector.authenticate)
        _no_print_in_source(src, "FinanceConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.finance.connector import FinanceConnector
        src = inspect.getsource(FinanceConnector.sync)
        _no_print_in_source(src, "FinanceConnector.sync")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, caplog):
        """authenticate() logs ERROR when Plaid init fails."""
        from connectors.finance.connector import FinanceConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "provider": "plaid",
            "client_id": "fake-id",
            "secret": "fake-secret",
            "access_tokens": ["token1"],
        }
        connector = FinanceConnector(event_bus=bus, db=db, config=config)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'plaid'")):
            with caplog.at_level(logging.ERROR, logger="connectors.finance.connector"):
                result = await connector.authenticate()

        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# iMessageConnector
# ---------------------------------------------------------------------------

class TestiMessageConnectorLogging:
    """iMessageConnector error paths emit log records instead of print()."""

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.imessage.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.imessage.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.imessage.connector import iMessageConnector
        src = inspect.getsource(iMessageConnector.authenticate)
        _no_print_in_source(src, "iMessageConnector.authenticate")

    def test_no_print_in_contact_sync_loop(self):
        """_contact_sync_loop() source must not contain print()."""
        from connectors.imessage.connector import iMessageConnector
        src = inspect.getsource(iMessageConnector._contact_sync_loop)
        _no_print_in_source(src, "iMessageConnector._contact_sync_loop")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, tmp_path, caplog):
        """authenticate() logs ERROR when the chat.db path doesn't exist."""
        from connectors.imessage.connector import iMessageConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "db_path": str(tmp_path / "nonexistent_chat.db"),
        }
        connector = iMessageConnector(event_bus=bus, db=db, config=config)

        # authenticate() returns False (no log) when file is missing.
        # To test the logged path, provide an invalid SQLite file.
        invalid_db = tmp_path / "chat.db"
        invalid_db.write_bytes(b"not a sqlite database")
        connector._db_path = str(invalid_db)

        with caplog.at_level(logging.ERROR, logger="connectors.imessage.connector"):
            result = await connector.authenticate()

        # authenticate() should fail gracefully and log the error.
        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SignalConnector
# ---------------------------------------------------------------------------

class TestSignalConnectorLogging:
    """SignalConnector error paths emit log records instead of print()."""

    def _make_connector(self):
        from connectors.signal_msg.connector import SignalConnector
        bus = MagicMock()
        bus.publish = AsyncMock()
        db = MagicMock()
        config = {
            "socket_path": "/tmp/signal-cli.sock",
            "phone_number": "+15550000000",
        }
        return SignalConnector(event_bus=bus, db=db, config=config)

    def test_logger_defined(self):
        """Module-level logger must exist."""
        import connectors.signal_msg.connector as mod
        assert hasattr(mod, "logger"), "Missing module-level logger"
        assert mod.logger.name == "connectors.signal_msg.connector"

    def test_no_print_in_authenticate(self):
        """authenticate() source must not contain print()."""
        from connectors.signal_msg.connector import SignalConnector
        src = inspect.getsource(SignalConnector.authenticate)
        _no_print_in_source(src, "SignalConnector.authenticate")

    def test_no_print_in_sync(self):
        """sync() source must not contain print()."""
        from connectors.signal_msg.connector import SignalConnector
        src = inspect.getsource(SignalConnector.sync)
        _no_print_in_source(src, "SignalConnector.sync")

    def test_no_print_in_sync_contacts(self):
        """sync_contacts() source must not contain print()."""
        from connectors.signal_msg.connector import SignalConnector
        src = inspect.getsource(SignalConnector.sync_contacts)
        _no_print_in_source(src, "SignalConnector.sync_contacts")

    @pytest.mark.asyncio
    async def test_authenticate_failure_logs_error(self, caplog):
        """authenticate() logs ERROR when the RPC call fails."""
        connector = self._make_connector()

        with patch.object(connector, "_rpc_call", side_effect=ConnectionError("socket not found")):
            with caplog.at_level(logging.ERROR, logger="connectors.signal_msg.connector"):
                result = await connector.authenticate()

        assert result is False
        assert any("Auth failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sync_error_logs_error(self, caplog):
        """sync() logs ERROR when the receive RPC call fails."""
        connector = self._make_connector()

        with patch.object(connector, "_rpc_call", side_effect=ConnectionError("broken pipe")):
            with caplog.at_level(logging.ERROR, logger="connectors.signal_msg.connector"):
                count = await connector.sync()

        assert count == 0
        assert any("Sync error" in r.message for r in caplog.records)
