"""
Tests for messaging endpoints: POST /api/messages/send and POST /api/draft.

Covers scenarios not already exercised by test_send_reply.py (which tests
no_connector, 422 validation, iMessage routing, generic-channel preference,
and error-result forwarding).

New send-endpoint coverage:
  - Connector raises an exception during execute() (as opposed to returning
    an error dict) — verifies graceful error response, not a 500.
  - Explicit channel='signal' routes to Signal connector, not iMessage.

New draft-endpoint coverage (zero prior HTTP-level tests):
  - Successful draft generation via mocked ai_engine.draft_reply().
  - AI engine unavailable (raises Exception) — returns error, not 500.
  - contact_id is forwarded to ai_engine.draft_reply() for template lookup.
  - Empty incoming_message is handled gracefully.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_life_os(db, connector_map=None):
    """Build a minimal mock LifeOS with the attributes create_web_app needs.

    Args:
        db: DatabaseManager fixture (from conftest).
        connector_map: Optional dict mapping connector_id -> mock connector.
    """
    life_os = MagicMock()
    life_os.db = db
    life_os.connector_map = connector_map or {}
    life_os.connectors = []
    life_os.notification_manager = MagicMock()
    life_os.task_manager = MagicMock()
    life_os.event_store = MagicMock()
    life_os.vector_store = MagicMock()
    life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    life_os.rules_engine = MagicMock()
    life_os.prediction_engine = MagicMock()
    life_os.insight_engine = MagicMock()
    life_os.ai_engine = MagicMock()
    life_os.ai_engine.draft_reply = AsyncMock(return_value="Here is a draft reply.")
    life_os.source_weight_manager = MagicMock()
    life_os.feedback_collector = MagicMock()
    life_os.event_bus = MagicMock()
    life_os.event_bus.is_connected = False
    return life_os


def _make_connector_mock(connector_id: str, execute_return=None, execute_side_effect=None):
    """Create a mock connector with configurable execute() behaviour.

    Args:
        connector_id: The CONNECTOR_ID for routing (e.g. 'imessage', 'signal').
        execute_return: Value returned by execute() on success.
        execute_side_effect: Exception raised by execute() (overrides return).
    """
    c = MagicMock()
    c.CONNECTOR_ID = connector_id
    if execute_side_effect:
        c.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        c.execute = AsyncMock(return_value=execute_return or {"status": "sent"})
    return c


# ===========================================================================
# POST /api/messages/send — additional coverage
# ===========================================================================


class TestSendMessageConnectorException:
    """Connector raising an exception (not returning error dict) during execute()."""

    @pytest.mark.asyncio
    async def test_send_message_connector_exception_returns_error_status(self, db):
        """When execute() raises, the endpoint catches it and returns status='error'.

        This differs from test_send_connector_error_forwarded in test_send_reply.py
        which tests a connector *returning* an error dict.  Here we test the
        try/except path where execute() itself throws.
        """
        boom_connector = _make_connector_mock(
            "imessage",
            execute_side_effect=RuntimeError("AppleScript process crashed"),
        )
        life_os = _make_mock_life_os(db, connector_map={"imessage": boom_connector})
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/messages/send",
                json={"recipient": "+15555550100", "message": "Hi!", "channel": "imessage"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "AppleScript process crashed" in data["details"]

    @pytest.mark.asyncio
    async def test_send_message_connector_exception_does_not_500(self, db):
        """Ensure the endpoint never returns a 5xx even if the connector explodes."""
        boom_connector = _make_connector_mock(
            "signal",
            execute_side_effect=Exception("unexpected failure"),
        )
        life_os = _make_mock_life_os(db, connector_map={"signal": boom_connector})
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/messages/send",
                json={"recipient": "+1555", "message": "Test", "channel": "signal"},
            )

        assert resp.status_code < 500


class TestSendMessageSignalChannelRouting:
    """Explicit channel='signal' routes to Signal connector, skipping iMessage."""

    @pytest.mark.asyncio
    async def test_signal_channel_routes_to_signal_connector(self, db):
        """When channel='signal', only the Signal connector is used."""
        imessage = _make_connector_mock("imessage")
        signal = _make_connector_mock("signal", execute_return={"status": "sent", "recipient": "+1555"})
        life_os = _make_mock_life_os(
            db,
            connector_map={"imessage": imessage, "signal": signal},
        )
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/messages/send",
                json={"recipient": "+1555", "message": "Hey!", "channel": "signal"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sent"
        assert data["connector"] == "signal"
        signal.execute.assert_called_once_with(
            "send_message",
            {"recipient": "+1555", "message": "Hey!"},
        )
        imessage.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_signal_channel_no_connector_returns_no_connector(self, db):
        """channel='signal' with only iMessage active returns no_connector."""
        imessage = _make_connector_mock("imessage")
        life_os = _make_mock_life_os(db, connector_map={"imessage": imessage})
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/messages/send",
                json={"recipient": "+1555", "message": "Hi", "channel": "signal"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no_connector"
        imessage.execute.assert_not_called()


# ===========================================================================
# POST /api/draft — full HTTP-level coverage (none existed before)
# ===========================================================================


class TestDraftReplySuccess:
    """Happy-path draft generation via the /api/draft endpoint."""

    @pytest.mark.asyncio
    async def test_draft_reply_returns_generated_draft(self, db):
        """POST /api/draft returns the AI-generated draft string."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Sure, I'll be there at 3!")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/draft",
                json={"incoming_message": "Can we meet at 3?", "channel": "imessage"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["draft"] == "Sure, I'll be there at 3!"

    @pytest.mark.asyncio
    async def test_draft_reply_calls_ai_engine_with_correct_params(self, db):
        """Verify the endpoint passes channel, incoming_message, and contact_id to ai_engine."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Draft text")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/draft",
                json={
                    "incoming_message": "Hello",
                    "channel": "signal",
                    "contact_id": "contact-abc",
                },
            )

        life_os.ai_engine.draft_reply.assert_called_once_with(
            contact_id="contact-abc",
            channel="signal",
            incoming_message="Hello",
        )


class TestDraftReplyAIUnavailable:
    """AI engine raises an exception — endpoint must degrade gracefully."""

    @pytest.mark.asyncio
    async def test_draft_reply_ai_error_returns_null_draft(self, db):
        """When ai_engine.draft_reply() raises, the response has draft=None and an error message."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(side_effect=RuntimeError("Ollama connection refused"))
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/draft",
                json={"incoming_message": "Hey there"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["draft"] is None
        assert "error" in data
        assert isinstance(data["error"], str)

    @pytest.mark.asyncio
    async def test_draft_reply_ai_error_does_not_500(self, db):
        """Endpoint never returns 5xx even if the AI engine explodes."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(side_effect=Exception("boom"))
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/draft",
                json={"incoming_message": "Test"},
            )

        assert resp.status_code < 500


class TestDraftReplyContactId:
    """Verify contact_id is properly forwarded for per-contact template lookup."""

    @pytest.mark.asyncio
    async def test_draft_reply_with_contact_id(self, db):
        """contact_id in the request body reaches ai_engine.draft_reply()."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Personalized draft")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/draft",
                json={
                    "incoming_message": "Are you free tomorrow?",
                    "contact_id": "contact-xyz",
                    "channel": "email",
                },
            )

        assert resp.status_code == 200
        call_kwargs = life_os.ai_engine.draft_reply.call_args
        assert call_kwargs.kwargs["contact_id"] == "contact-xyz"

    @pytest.mark.asyncio
    async def test_draft_reply_without_contact_id_passes_none(self, db):
        """When contact_id is omitted, ai_engine.draft_reply() receives None."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Generic draft")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/draft",
                json={"incoming_message": "Hello"},
            )

        call_kwargs = life_os.ai_engine.draft_reply.call_args
        assert call_kwargs.kwargs["contact_id"] is None


class TestDraftReplyMissingMessage:
    """Edge case: empty or missing incoming_message."""

    @pytest.mark.asyncio
    async def test_draft_reply_empty_message_still_succeeds(self, db):
        """Empty incoming_message is valid (defaults to '') — endpoint should not 422."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/draft",
                json={"incoming_message": ""},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "draft" in data

    @pytest.mark.asyncio
    async def test_draft_reply_omitted_message_defaults_to_empty(self, db):
        """incoming_message defaults to '' in the schema — omitting it should work."""
        life_os = _make_mock_life_os(db)
        life_os.ai_engine.draft_reply = AsyncMock(return_value="Draft without context")
        app = create_web_app(life_os)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/draft", json={})

        assert resp.status_code == 200
        call_kwargs = life_os.ai_engine.draft_reply.call_args
        assert call_kwargs.kwargs["incoming_message"] == ""
