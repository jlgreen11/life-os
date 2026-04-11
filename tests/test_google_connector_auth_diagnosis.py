"""
Tests for Google connector structured auth diagnosis.

Verifies that _compute_auth_diagnosis() correctly identifies the root cause
of each auth-failure mode and that health_check() / start() surface
auth_diagnosis in their responses and published events.

Test cases mirror the six failure modes the admin dashboard must distinguish:
  - No credentials file (OAuth app not set up yet)
  - No token file (OAuth flow never completed)
  - Expired token with refresh token (auto-recoverable)
  - Expired token without refresh token (manual re-auth required)
  - Corrupt/unreadable token file (manual delete + re-auth required)
  - Valid token (no root_cause — auth should be working)

All tests mock file I/O and the Credentials class. No real OAuth calls are
made, matching the patterns in test_google_health_diagnostics.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from connectors.google.connector import GoogleConnector


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


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
# _compute_auth_diagnosis() — root-cause classification
# ------------------------------------------------------------------


def test_auth_diagnosis_missing_credentials_file(connector):
    """No credentials file → root_cause='missing_credentials'.

    This is the very first setup step: the user hasn't downloaded the
    OAuth credentials from Google Cloud Console yet.
    """
    with patch("os.path.exists", return_value=False):
        result = connector._compute_auth_diagnosis()

    assert result["credentials_file_exists"] is False
    assert result["token_file_exists"] is False
    assert result["root_cause"] == "missing_credentials"
    assert "action" in result
    # Action must name the expected file path so the user knows where to put it.
    assert connector._credentials_file in result["action"]


def test_auth_diagnosis_missing_token_file(connector):
    """Credentials file exists but no token file → root_cause='oauth_not_completed'.

    The credentials JSON was downloaded but the user never opened the
    OAuth consent screen to generate a token.
    """
    def exists_side_effect(path):
        # Only the credentials file exists; token file does not.
        return path == connector._credentials_file

    with patch("os.path.exists", side_effect=exists_side_effect):
        result = connector._compute_auth_diagnosis()

    assert result["credentials_file_exists"] is True
    assert result["token_file_exists"] is False
    assert result["root_cause"] == "oauth_not_completed"
    assert "action" in result
    # Action must point to the OAuth auth endpoint.
    assert "/api/admin/connectors/google/auth" in result["action"]


def test_auth_diagnosis_expired_token_with_refresh_token(connector):
    """Both files exist, token expired but has refresh token → root_cause='token_expired_refresh_available'.

    The access token expired, but the refresh token is present so the
    connector can obtain a new access token automatically.
    """
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "valid_refresh_token"

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds):
        result = connector._compute_auth_diagnosis()

    assert result["token_file_exists"] is True
    assert result["credentials_file_exists"] is True
    assert result["token_expired"] is True
    assert result["has_refresh_token"] is True
    assert result["root_cause"] == "token_expired_refresh_available"
    assert "action" in result
    # This is auto-recoverable — action should not demand manual steps.
    assert "automatic" in result["action"].lower() or "refresh" in result["action"].lower()


def test_auth_diagnosis_expired_token_without_refresh_token(connector):
    """Both files exist, token expired with no refresh token → root_cause='no_refresh_token'.

    The access token expired and no refresh token is stored.  The user
    must re-authenticate from scratch.
    """
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = None

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds):
        result = connector._compute_auth_diagnosis()

    assert result["token_file_exists"] is True
    assert result["token_expired"] is True
    assert result["has_refresh_token"] is False
    assert result["root_cause"] == "no_refresh_token"
    assert "action" in result
    assert "/api/admin/connectors/google/auth" in result["action"]


def test_auth_diagnosis_corrupt_token_file(connector):
    """Both files exist but Credentials loading raises an exception → root_cause='token_corrupt'.

    The token file was created but its content is not valid Credentials
    JSON (e.g. truncated write, manual edit, schema mismatch).
    """
    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               side_effect=Exception("Invalid JSON at line 3")):
        result = connector._compute_auth_diagnosis()

    assert result["token_file_exists"] is True
    assert result["root_cause"] == "token_corrupt"
    assert "action" in result
    # Action must name the file to delete and the re-auth endpoint.
    assert connector._token_file in result["action"]
    assert "/api/admin/connectors/google/auth" in result["action"]
    # The original exception message should appear so the user can debug.
    assert "Invalid JSON at line 3" in result["action"]


def test_auth_diagnosis_valid_token_no_root_cause(connector):
    """Both files exist and token is valid → no root_cause in result.

    When credentials are fully intact, _compute_auth_diagnosis should
    return token state fields but NOT set root_cause (auth should work).
    """
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.expired = False
    mock_creds.refresh_token = "valid_refresh_token"

    with patch("os.path.exists", return_value=True), \
         patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
               return_value=mock_creds):
        result = connector._compute_auth_diagnosis()

    assert result["token_file_exists"] is True
    assert result["credentials_file_exists"] is True
    assert result["token_valid"] is True
    assert result["token_expired"] is False
    assert result["has_refresh_token"] is True
    # No root_cause when auth is healthy.
    assert "root_cause" not in result


# ------------------------------------------------------------------
# health_check() surfaces auth_diagnosis in its response
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_includes_auth_diagnosis_when_not_authenticated(connector):
    """health_check() response contains auth_diagnosis when connector is not authenticated.

    The admin UI reads auth_diagnosis to display the specific fix rather
    than a generic "Authentication failed" message.
    """
    # Connector has no active services (never authenticated).
    connector._gmail_service = None

    with patch("os.path.exists", return_value=False):
        result = await connector.health_check()

    assert "auth_diagnosis" in result
    assert result["auth_diagnosis"]["root_cause"] == "missing_credentials"


@pytest.mark.asyncio
async def test_health_check_includes_auth_diagnosis_no_token_file(connector):
    """health_check returns auth_diagnosis with oauth_not_completed when token is absent."""
    connector._gmail_service = None

    def exists_side_effect(path):
        return path == connector._credentials_file

    with patch("os.path.exists", side_effect=exists_side_effect):
        result = await connector.health_check()

    assert "auth_diagnosis" in result
    assert result["auth_diagnosis"]["root_cause"] == "oauth_not_completed"


@pytest.mark.asyncio
async def test_health_check_includes_auth_diagnosis_on_mid_session_api_failure(connector):
    """health_check includes auth_diagnosis when an authenticated API call fails mid-session.

    When the Gmail service is set but getProfile() raises (e.g. token revoked
    after startup), the response should still include auth_diagnosis so the
    caller can distinguish root causes programmatically.
    """
    connector._gmail_service = MagicMock()
    connector._gmail_service.users().getProfile().execute.side_effect = RuntimeError(
        "mid-session network failure"
    )

    with patch("os.path.exists", return_value=False):
        result = await connector.health_check()

    assert result["status"] == "error"
    assert "auth_diagnosis" in result
    # Files don't exist in this test scenario.
    assert result["auth_diagnosis"]["root_cause"] == "missing_credentials"


# ------------------------------------------------------------------
# start() publishes auth_diagnosis in the error event
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_publishes_auth_diagnosis_in_error_event(connector):
    """start() publishes system.connector.error with auth_diagnosis when auth fails.

    The orchestrator and health monitor subscribe to this event to surface
    actionable diagnostics without waiting for the next health_check cycle.
    """
    expected_diagnosis = {
        "root_cause": "oauth_not_completed",
        "action": "Complete initial OAuth setup: visit /api/admin/connectors/google/auth",
        "token_file_exists": False,
        "credentials_file_exists": True,
    }

    with patch.object(connector, "authenticate", return_value=False), \
         patch.object(connector, "_compute_auth_diagnosis", return_value=expected_diagnosis):
        await connector.start()

    # Find the system.connector.error event published to the bus.
    publish_calls = connector.bus.publish.call_args_list
    error_events = [
        call for call in publish_calls
        if call.args[0] == "system.connector.error"
    ]
    assert len(error_events) >= 1, "Expected at least one system.connector.error publish"

    payload = error_events[0].args[1]
    assert "auth_diagnosis" in payload, "auth_diagnosis must be in the error event payload"
    assert payload["auth_diagnosis"]["root_cause"] == "oauth_not_completed"
    assert payload["error_type"] == "authentication"
    assert payload["connector_id"] == "google"


@pytest.mark.asyncio
async def test_start_does_not_publish_error_when_auth_succeeds(connector):
    """start() does NOT publish system.connector.error when authentication succeeds."""
    mock_gmail = MagicMock()
    mock_calendar = MagicMock()
    mock_people = MagicMock()

    async def fake_authenticate():
        """Simulate successful authentication."""
        connector._gmail_service = mock_gmail
        connector._calendar_service = mock_calendar
        connector._people_service = mock_people
        return True

    with patch.object(connector, "authenticate", side_effect=fake_authenticate), \
         patch.object(connector, "_sync_loop", new_callable=AsyncMock):
        await connector.start()

    publish_calls = connector.bus.publish.call_args_list
    error_events = [
        call for call in publish_calls
        if call.args[0] == "system.connector.error"
    ]
    assert len(error_events) == 0, "No error event should be published on successful auth"


# ------------------------------------------------------------------
# authenticate() error message includes token file path
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_authenticate_error_includes_token_file_path(connector):
    """authenticate() sets _auth_error with the token file path when token is missing.

    The explicit path tells the user (and the agent) exactly where to
    put the token file rather than requiring them to guess the location.
    """
    with patch.object(connector, "_load_credentials", return_value=None):
        result = await connector.authenticate()

    assert result is False
    assert connector._token_file in connector._auth_error
    assert "initial OAuth setup" in connector._auth_error
