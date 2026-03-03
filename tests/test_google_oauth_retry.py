"""
Tests for Google OAuth token refresh retry logic.

The Google connector's _load_credentials() method retries transient network
errors (TransportError) with increasing delays [2s, 5s, 10s] to handle DNS/
network blips at startup. RefreshError (invalid_grant) is never retried since
it means the token was revoked and requires user re-authentication.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from google.auth import exceptions as google_exceptions

from connectors.google.connector import GoogleConnector


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing."""
    bus = MagicMock()
    bus.publish = MagicMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def token_dir():
    """Create a temporary directory with a minimal token file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        token_path = os.path.join(tmpdir, "google_token.json")
        token_data = {
            "token": "expired_access_token",
            "refresh_token": "valid_refresh_token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with open(token_path, "w") as f:
            json.dump(token_data, f)
        yield tmpdir, token_path


@pytest.fixture
def connector(db, mock_event_bus, token_dir):
    """GoogleConnector instance with token file pointing to temp directory."""
    tmpdir, token_path = token_dir
    config = {
        "email_address": "test@gmail.com",
        "credentials_file": os.path.join(tmpdir, "google_credentials.json"),
        "token_file": token_path,
        "sync_interval": 30,
        "calendars": ["primary"],
        "gmail_labels": ["INBOX", "SENT"],
    }
    return GoogleConnector(mock_event_bus, db, config)


def _make_expired_creds():
    """Create a mock Credentials object that needs refresh."""
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "valid_refresh_token"
    mock_creds.to_json.return_value = '{"token": "refreshed"}'
    return mock_creds


class TestTransientFailureRetry:
    """Tests for retry behavior on transient network failures during token refresh."""

    def test_transient_failure_retries_then_succeeds(self, connector):
        """TransportError on first two attempts, then success on third.

        Verifies that _load_credentials() retries on TransportError and
        returns valid credentials once refresh succeeds.
        """
        mock_creds = _make_expired_creds()

        # Raise TransportError twice, then succeed
        transport_error = google_exceptions.TransportError("Connection reset by peer")
        mock_creds.refresh.side_effect = [transport_error, transport_error, None]

        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_creds), \
             patch("google.auth.transport.requests.Request"), \
             patch("time.sleep") as mock_sleep, \
             patch("builtins.open", MagicMock()):

            result = connector._load_credentials()

            assert result is mock_creds
            assert mock_creds.refresh.call_count == 3
            # Should have slept twice (before retries 2 and 3)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(2)
            mock_sleep.assert_any_call(5)

    def test_permanent_failure_raises_after_retries(self, connector):
        """TransportError on all attempts raises ValueError after exhausting retries.

        Verifies that the error message includes 'after 3 attempts' and that
        all retry delays were used.
        """
        mock_creds = _make_expired_creds()

        transport_error = google_exceptions.TransportError("DNS resolution failed")
        mock_creds.refresh.side_effect = transport_error

        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_creds), \
             patch("google.auth.transport.requests.Request"), \
             patch("time.sleep") as mock_sleep:

            with pytest.raises(ValueError, match="after 3 attempts"):
                connector._load_credentials()

            assert mock_creds.refresh.call_count == 3
            # Should have slept twice (after attempts 1 and 2, not after final attempt)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(2)
            mock_sleep.assert_any_call(5)

    def test_refresh_error_not_retried(self, connector):
        """RefreshError (invalid_grant) raises ValueError immediately without retry.

        Verifies that revoked token errors short-circuit the retry loop and
        that no sleep calls are made.
        """
        mock_creds = _make_expired_creds()

        refresh_error = google_exceptions.RefreshError("Token has been revoked")
        mock_creds.refresh.side_effect = refresh_error

        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_creds), \
             patch("google.auth.transport.requests.Request"), \
             patch("time.sleep") as mock_sleep:

            with pytest.raises(ValueError, match="re-authenticate via /admin connector panel"):
                connector._load_credentials()

            # Should have called refresh exactly once — no retries
            mock_creds.refresh.assert_called_once()
            # No sleep calls since we don't retry RefreshError
            mock_sleep.assert_not_called()

    def test_success_on_first_attempt_no_delay(self, connector):
        """Successful refresh on first attempt incurs no sleep delay.

        Verifies the happy path: refresh works immediately, no retries needed.
        """
        mock_creds = _make_expired_creds()

        # Refresh succeeds immediately
        mock_creds.refresh.return_value = None

        with patch("google.oauth2.credentials.Credentials.from_authorized_user_file", return_value=mock_creds), \
             patch("google.auth.transport.requests.Request"), \
             patch("time.sleep") as mock_sleep, \
             patch("builtins.open", MagicMock()):

            result = connector._load_credentials()

            assert result is mock_creds
            mock_creds.refresh.assert_called_once()
            mock_sleep.assert_not_called()
