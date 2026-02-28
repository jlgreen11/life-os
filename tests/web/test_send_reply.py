"""
Tests for POST /api/messages/send — quick-reply send endpoint.

Verifies that the endpoint correctly routes outbound messages to the
appropriate messaging connector and handles all error cases gracefully:
  - no_connector when no messaging connector is active
  - validation errors (422) for missing required fields
  - error forwarding when the connector returns an error result
  - sent status and connector attribution on success
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from web.app import create_web_app


@pytest.fixture
def mock_life_os(db):
    """Minimal LifeOS stand-in with an empty connector_map."""
    app = MagicMock()
    app.connector_map = {}
    app.db = db
    return app


@pytest.fixture
def fake_imessage_connector():
    """Mock iMessage connector whose execute() returns a sent result."""
    c = MagicMock()
    c.CONNECTOR_ID = "imessage"
    c.execute = AsyncMock(return_value={"status": "sent", "title": "Hi!"})
    return c


@pytest.fixture
def fake_signal_connector():
    """Mock Signal connector whose execute() returns a sent result."""
    c = MagicMock()
    c.CONNECTOR_ID = "signal"
    c.execute = AsyncMock(return_value={"status": "sent", "recipient": "+1555"})
    return c


@pytest.fixture
def error_connector():
    """Mock connector whose execute() returns an error result."""
    c = MagicMock()
    c.CONNECTOR_ID = "imessage"
    c.execute = AsyncMock(return_value={"status": "error", "details": "AppleScript denied"})
    return c


# ---------------------------------------------------------------------------
# Import-level smoke test
# ---------------------------------------------------------------------------

def test_send_message_request_schema_importable():
    """SendMessageRequest is importable and validates required fields."""
    from web.schemas import SendMessageRequest

    req = SendMessageRequest(recipient="+15555550100", message="Hello!", channel="imessage")
    assert req.recipient == "+15555550100"
    assert req.message == "Hello!"
    assert req.channel == "imessage"


def test_send_message_request_default_channel():
    """Channel defaults to 'message' when not specified."""
    from web.schemas import SendMessageRequest

    req = SendMessageRequest(recipient="+1555", message="Hi")
    assert req.channel == "message"


# ---------------------------------------------------------------------------
# Endpoint routing tests (via create_app factory)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_no_connector_returns_no_connector_status(db):
    """POST /api/messages/send returns no_connector when connector_map is empty."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    mock_life_os.connector_map = {}
    mock_life_os.db = db
    # Other attributes needed by create_app
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/messages/send",
            json={"recipient": "+15555550100", "message": "Hello!", "channel": "imessage"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_connector"
    assert "imessage" in data["details"].lower() or "connector" in data["details"].lower()


@pytest.mark.asyncio
async def test_send_missing_required_fields_returns_422(db):
    """POST /api/messages/send returns 422 when recipient or message is missing."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    mock_life_os.connector_map = {}
    mock_life_os.db = db
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing 'message'
        r1 = await client.post("/api/messages/send", json={"recipient": "+1555"})
        # Missing 'recipient'
        r2 = await client.post("/api/messages/send", json={"message": "Hi"})

    assert r1.status_code == 422
    assert r2.status_code == 422


@pytest.mark.asyncio
async def test_send_with_imessage_connector_routes_correctly(db, fake_imessage_connector):
    """When an iMessage connector is active, the message is sent via it."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    mock_life_os.connector_map = {"imessage": fake_imessage_connector}
    mock_life_os.db = db
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/messages/send",
            json={"recipient": "+15555550100", "message": "Hi!", "channel": "imessage"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sent"
    assert data["connector"] == "imessage"
    # Verify the connector's execute() was called with correct args
    fake_imessage_connector.execute.assert_called_once_with(
        "send_message",
        {"recipient": "+15555550100", "message": "Hi!"},
    )


@pytest.mark.asyncio
async def test_send_generic_channel_prefers_imessage_over_signal(db, fake_imessage_connector, fake_signal_connector):
    """Generic 'message' channel uses iMessage when both connectors are active."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    # iMessage listed first — generic channel should prefer it
    mock_life_os.connector_map = {
        "imessage": fake_imessage_connector,
        "signal": fake_signal_connector,
    }
    mock_life_os.db = db
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/messages/send",
            json={"recipient": "+15555550100", "message": "Hey!", "channel": "message"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sent"
    assert data["connector"] == "imessage"
    fake_imessage_connector.execute.assert_called_once()
    fake_signal_connector.execute.assert_not_called()


@pytest.mark.asyncio
async def test_send_falls_back_to_signal_when_no_imessage(db, fake_signal_connector):
    """Generic 'message' channel falls back to Signal when iMessage is absent."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    mock_life_os.connector_map = {"signal": fake_signal_connector}
    mock_life_os.db = db
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/messages/send",
            json={"recipient": "+1555", "message": "Hi!", "channel": "message"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sent"
    assert data["connector"] == "signal"


@pytest.mark.asyncio
async def test_send_connector_error_forwarded(db, error_connector):
    """Connector errors are forwarded as status='error' with a details field."""
    from web.app import create_web_app
    from httpx import AsyncClient, ASGITransport

    mock_life_os = MagicMock()
    mock_life_os.connector_map = {"imessage": error_connector}
    mock_life_os.db = db
    mock_life_os.notification_manager = MagicMock()
    mock_life_os.task_manager = MagicMock()
    mock_life_os.event_store = MagicMock()
    mock_life_os.vector_store = MagicMock()
    mock_life_os.vector_store.get_stats = MagicMock(return_value={"document_count": 0})
    mock_life_os.rules_engine = MagicMock()
    mock_life_os.prediction_engine = MagicMock()
    mock_life_os.insight_engine = MagicMock()
    mock_life_os.ai_engine = MagicMock()
    mock_life_os.source_weight_manager = MagicMock()
    mock_life_os.feedback_collector = MagicMock()
    mock_life_os.event_bus = MagicMock()
    mock_life_os.event_bus.is_connected = False
    mock_life_os.connectors = []

    app = create_web_app(mock_life_os)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/messages/send",
            json={"recipient": "+1555", "message": "Hi!", "channel": "imessage"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "AppleScript denied" in data["details"]


# ---------------------------------------------------------------------------
# JS template smoke test
# ---------------------------------------------------------------------------

def test_send_quick_reply_js_in_template():
    """sendQuickReply function and quick-reply CSS are present in HTML_TEMPLATE."""
    from web.template import HTML_TEMPLATE

    assert "sendQuickReply" in HTML_TEMPLATE, "sendQuickReply() function missing from template"
    assert "/api/messages/send" in HTML_TEMPLATE, "/api/messages/send URL missing from template"
    assert "quick-reply-area" in HTML_TEMPLATE, ".quick-reply-area CSS class missing from template"
    assert "quick-reply-input" in HTML_TEMPLATE, ".quick-reply-input CSS class missing from template"
    assert "quick-reply-send" in HTML_TEMPLATE, ".quick-reply-send CSS class missing from template"
    assert "qr-" in HTML_TEMPLATE, "quick-reply textarea ID prefix 'qr-' missing from template"
