"""
Tests for Google connector health_check diagnostics and error classification.

Verifies that health_check() returns rich, actionable diagnostics when the
connector is authenticated but the Google API call fails mid-session (e.g.
token revoked, scopes changed, network outage). Also verifies that
authenticate() preserves actionable error messages from _load_credentials().
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.google.connector import GoogleConnector


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def google_config():
    """Default Google connector configuration."""
    return {
        "email_address": "test@gmail.com",
        "credentials_file": "data/google_credentials.json",
        "token_file": "data/google_token.json",
        "sync_interval": 30,
        "calendars": ["primary"],
        "gmail_labels": ["INBOX", "SENT"],
    }


@pytest.fixture
def connector(db, mock_event_bus, google_config):
    """GoogleConnector instance with mocked dependencies."""
    return GoogleConnector(mock_event_bus, db, google_config)


# ------------------------------------------------------------------
# health_check — authenticated-but-broken path
# ------------------------------------------------------------------


def _make_http_error(status_code: int, reason: str = "error"):
    """Create a mock that behaves like googleapiclient.errors.HttpError.

    The mock has a resp attribute with a status field, matching the real
    HttpError interface so _classify_api_error can inspect it.
    """
    exc = Exception(f"<HttpError {status_code} \"{reason}\">")
    exc.resp = MagicMock()
    exc.resp.status = status_code
    return exc


@pytest.mark.asyncio
async def test_health_check_returns_recovery_hint_on_403(connector):
    """health_check returns recovery_hint with scope_revoked when getProfile throws 403."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = _make_http_error(403, "Forbidden")

    with patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=time.time() - 3600), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"refresh_token": "tok123"}):
        result = await connector.health_check()

    assert result["status"] == "error"
    assert result["error_type"] == "scope_revoked"
    assert "recovery_hint" in result
    assert "revoked or scopes changed" in result["recovery_hint"]


@pytest.mark.asyncio
async def test_health_check_returns_recovery_hint_on_401(connector):
    """health_check returns recovery_hint with token_expired when getProfile throws 401."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = _make_http_error(401, "Unauthorized")

    with patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=time.time() - 3600), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"refresh_token": "tok123"}):
        result = await connector.health_check()

    assert result["status"] == "error"
    assert result["error_type"] == "token_expired"
    assert "recovery_hint" in result
    assert "expired" in result["recovery_hint"]


@pytest.mark.asyncio
async def test_health_check_returns_recovery_hint_on_connection_error(connector):
    """health_check returns recovery_hint with network_error when getProfile throws ConnectionError."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = ConnectionError(
        "Connection refused"
    )

    with patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=time.time() - 3600), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"refresh_token": "tok123"}):
        result = await connector.health_check()

    assert result["status"] == "error"
    assert result["error_type"] == "network_error"
    assert "recovery_hint" in result
    assert "network" in result["recovery_hint"].lower()


@pytest.mark.asyncio
async def test_health_check_returns_full_diagnostics_on_mid_session_failure(connector):
    """health_check returns token_file_exists, has_refresh_token, etc. on mid-session API failure."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = _make_http_error(403, "Forbidden")

    with patch("os.path.exists", return_value=True), \
         patch("os.path.getmtime", return_value=time.time() - 7200), \
         patch("builtins.open", MagicMock()), \
         patch("json.load", return_value={"refresh_token": "tok123"}):
        result = await connector.health_check()

    # Full diagnostics should be present (from _build_health_diagnostics)
    assert "token_file_exists" in result
    assert result["token_file_exists"] is True
    assert "has_refresh_token" in result
    assert result["has_refresh_token"] is True
    assert "token_age_hours" in result
    assert result["token_age_hours"] is not None
    assert "last_sync" in result
    assert "error_type" in result
    assert "recovery_hint" in result
    assert result["connector"] == "google"


@pytest.mark.asyncio
async def test_health_check_clears_stale_services_on_failure(connector):
    """health_check clears _gmail_service, _calendar_service, _people_service on API failure.

    This ensures the connector knows it needs re-auth on the next attempt,
    falling through to the unauthenticated recovery path.
    """
    connector._gmail_service = MagicMock()
    connector._calendar_service = MagicMock()
    connector._people_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = _make_http_error(401, "Unauthorized")

    with patch("os.path.exists", return_value=False):
        await connector.health_check()

    assert connector._gmail_service is None
    assert connector._calendar_service is None
    assert connector._people_service is None


@pytest.mark.asyncio
async def test_health_check_unknown_error_type(connector):
    """health_check classifies unrecognized exceptions as 'unknown' with a generic hint."""
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = RuntimeError("Something unexpected")

    with patch("os.path.exists", return_value=False):
        result = await connector.health_check()

    assert result["status"] == "error"
    assert result["error_type"] == "unknown"
    assert "recovery_hint" in result
    assert "/admin" in result["recovery_hint"]


# ------------------------------------------------------------------
# authenticate — error message preservation
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_preserves_value_error_messages(connector):
    """authenticate() surfaces the exact ValueError message from _load_credentials()."""
    specific_msg = "Token refresh failed (invalid_grant) — re-authenticate via /admin connector panel"

    with patch.object(connector, "_load_credentials", side_effect=ValueError(specific_msg)):
        result = await connector.authenticate()

    assert result is False
    # The auth error should be the exact ValueError message, not wrapped
    assert connector._auth_error == specific_msg


@pytest.mark.asyncio
async def test_authenticate_adds_guidance_on_generic_exception(connector):
    """authenticate() adds actionable guidance when a non-ValueError exception occurs."""
    with patch.object(connector, "_load_credentials", side_effect=OSError("disk read error")):
        result = await connector.authenticate()

    assert result is False
    assert "disk read error" in connector._auth_error
    assert "/admin connector panel" in connector._auth_error


# ------------------------------------------------------------------
# _classify_api_error unit tests
# ------------------------------------------------------------------


def test_classify_api_error_403():
    """_classify_api_error returns 'scope_revoked' for 403 errors."""
    exc = _make_http_error(403, "Forbidden")
    assert GoogleConnector._classify_api_error(exc) == "scope_revoked"


def test_classify_api_error_401():
    """_classify_api_error returns 'token_expired' for 401 errors."""
    exc = _make_http_error(401, "Unauthorized")
    assert GoogleConnector._classify_api_error(exc) == "token_expired"


def test_classify_api_error_connection_error():
    """_classify_api_error returns 'network_error' for ConnectionError."""
    exc = ConnectionError("Connection refused")
    assert GoogleConnector._classify_api_error(exc) == "network_error"


def test_classify_api_error_timeout_in_message():
    """_classify_api_error returns 'network_error' when 'timeout' appears in message."""
    exc = Exception("Request timed out: timeout waiting for response")
    assert GoogleConnector._classify_api_error(exc) == "network_error"


def test_classify_api_error_unknown():
    """_classify_api_error returns 'unknown' for unrecognized exceptions."""
    exc = RuntimeError("Something completely unexpected")
    assert GoogleConnector._classify_api_error(exc) == "unknown"


# ------------------------------------------------------------------
# _recovery_hint_for_error_type unit tests
# ------------------------------------------------------------------


def test_recovery_hint_scope_revoked():
    """Recovery hint for scope_revoked mentions re-authentication and scopes."""
    hint = GoogleConnector._recovery_hint_for_error_type("scope_revoked")
    assert "Re-authenticate" in hint
    assert "scopes" in hint


def test_recovery_hint_token_expired():
    """Recovery hint for token_expired mentions re-authentication and expiry."""
    hint = GoogleConnector._recovery_hint_for_error_type("token_expired")
    assert "Re-authenticate" in hint
    assert "expired" in hint


def test_recovery_hint_network_error():
    """Recovery hint for network_error mentions network connectivity."""
    hint = GoogleConnector._recovery_hint_for_error_type("network_error")
    assert "network" in hint.lower()
    assert "retry" in hint.lower()


def test_recovery_hint_unknown():
    """Recovery hint for unknown error type includes /admin guidance."""
    hint = GoogleConnector._recovery_hint_for_error_type("unknown")
    assert "/admin" in hint
