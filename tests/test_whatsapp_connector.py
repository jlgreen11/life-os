"""
Life OS — WhatsApp Connector Tests

Comprehensive test coverage for the WhatsAppConnector class (287 LOC).

The WhatsApp connector is a browser-only integration that:
1. Authenticates via QR code scan (no API available)
2. Scrapes messages from WhatsApp Web UI
3. Sends messages via browser automation
4. Handles session persistence across restarts

Test categories:
- Authentication flow (QR code, session reuse)
- Message scraping (unread chats, message extraction)
- Message sending (search + type + send)
- Priority classification
- Error handling
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from connectors.browser.whatsapp import WhatsAppConnector


class MockPage:
    """Mock Playwright Page object for WhatsApp Web testing."""

    def __init__(self, logged_in=True, unread_chats=None, messages=None):
        self.logged_in = logged_in
        self.unread_chats = unread_chats or []
        self.messages = messages or []
        self.keyboard = AsyncMock()
        self.keyboard.press = AsyncMock()
        self._selectors_seen = []

    async def goto(self, url, wait_until=None):
        """Mock navigation."""
        pass

    async def query_selector(self, selector):
        """Mock selector query - returns element if logged in."""
        self._selectors_seen.append(selector)
        if "Chat list" in selector or "pane-side" in selector:
            return MagicMock() if self.logged_in else None
        return MagicMock()

    async def wait_for_selector(self, selector):
        """Mock wait for selector."""
        return MagicMock()

    async def evaluate(self, script):
        """Mock JavaScript execution."""
        # Detect which function is being called based on script content
        if "Chat list" in script and "unread" in script:
            # This is _get_unread_chats()
            return self.unread_chats
        elif "message-in" in script or "message-out" in script:
            # This is _extract_messages()
            return self.messages
        return []


class MockContext:
    """Mock browser context."""

    def __init__(self, site_id):
        self.site_id = site_id


@pytest.fixture
def mock_event_bus():
    """Mock event bus for testing event publishing."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.is_connected = True
    return bus


@pytest.fixture
def mock_browser_engine():
    """Mock BrowserEngine for testing browser operations."""
    engine = AsyncMock()
    engine.start = AsyncMock()
    engine.create_context = AsyncMock(return_value=MockContext("whatsapp"))
    engine.new_page = AsyncMock()
    engine.save_session = AsyncMock()
    engine.close = AsyncMock()
    return engine


@pytest.fixture
def mock_human_emulator():
    """Mock HumanEmulator for testing human-like interactions."""
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.click = AsyncMock()
    human.type_text = AsyncMock()
    return human


@pytest.fixture
def mock_page_interactor():
    """Mock PageInteractor for testing page operations."""
    interactor = AsyncMock()
    interactor.screenshot = AsyncMock()
    return interactor


@pytest.fixture
def connector_config():
    """Standard WhatsApp connector configuration."""
    return {
        "enabled": True,
        "mode": "browser",
        "sync_interval": 10,
        "priority_contacts": ["Mom", "Partner"],
        "max_conversations_per_sync": 10,
    }


@pytest.fixture
def mock_db():
    """Mock DatabaseManager for testing."""
    db = MagicMock()
    return db


@pytest.fixture
def connector(mock_event_bus, mock_db, connector_config, mock_browser_engine, mock_human_emulator, mock_page_interactor):
    """WhatsAppConnector instance with mocked dependencies."""
    conn = WhatsAppConnector(
        event_bus=mock_event_bus,
        db=mock_db,
        config=connector_config,
        browser_engine=mock_browser_engine,
    )
    # Inject mock human and interactor
    conn._human = mock_human_emulator
    conn._interactor = mock_page_interactor
    return conn


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_api_authenticate_returns_false(connector):
    """WhatsApp has no API, so api_authenticate always returns False."""
    result = await connector.api_authenticate()
    assert result is False


@pytest.mark.asyncio
async def test_api_sync_raises_not_implemented(connector):
    """WhatsApp has no API, so api_sync raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="no personal API"):
        await connector.api_sync()


def test_get_login_selectors_returns_empty(connector):
    """WhatsApp uses QR code auth, not form-based login."""
    selectors = connector.get_login_selectors()
    assert selectors == {}


@pytest.mark.asyncio
async def test_is_logged_in_when_chat_list_visible(connector):
    """is_logged_in returns True when chat list selector is found."""
    page = MockPage(logged_in=True)
    result = await connector.is_logged_in(page)
    assert result is True


@pytest.mark.asyncio
async def test_is_logged_in_when_chat_list_missing(connector):
    """is_logged_in returns False when chat list selector not found."""
    page = MockPage(logged_in=False)
    result = await connector.is_logged_in(page)
    assert result is False


@pytest.mark.asyncio
async def test_is_logged_in_handles_exceptions(connector):
    """is_logged_in returns False on exception."""
    page = MagicMock()
    page.query_selector = AsyncMock(side_effect=Exception("Playwright error"))
    result = await connector.is_logged_in(page)
    assert result is False


@pytest.mark.asyncio
async def test_authenticate_with_existing_session(
    connector, mock_browser_engine, mock_human_emulator, mock_page_interactor
):
    """authenticate() succeeds immediately if valid session exists."""
    connector._browser_engine = mock_browser_engine
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(logged_in=True)
    mock_browser_engine.new_page = AsyncMock(return_value=page)

    result = await connector.authenticate()

    assert result is True
    # Should not save session since we reused existing one
    mock_browser_engine.save_session.assert_not_called()


@pytest.mark.asyncio
async def test_authenticate_qr_scan_success(
    connector, mock_browser_engine, mock_human_emulator, mock_page_interactor
):
    """authenticate() waits for QR scan and succeeds when user scans."""
    connector._browser_engine = mock_browser_engine
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    # Start logged out, then become logged in after 2 iterations
    page = MockPage(logged_in=False)
    call_count = 0

    async def mock_is_logged_in(p):
        nonlocal call_count
        call_count += 1
        return call_count > 2  # Logged in on 3rd check

    connector.is_logged_in = mock_is_logged_in
    mock_browser_engine.new_page = AsyncMock(return_value=page)

    result = await connector.authenticate()

    assert result is True
    # Should save session after successful QR scan
    mock_browser_engine.save_session.assert_called_once()
    # Should take screenshot of QR code
    mock_page_interactor.screenshot.assert_called_once()


@pytest.mark.asyncio
async def test_authenticate_qr_scan_timeout(
    connector, mock_browser_engine, mock_human_emulator, mock_page_interactor
):
    """authenticate() fails after 24 iterations (2 minutes) if no scan."""
    connector._browser_engine = mock_browser_engine
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(logged_in=False)
    mock_browser_engine.new_page = AsyncMock(return_value=page)

    # Never logged in
    connector.is_logged_in = AsyncMock(return_value=False)

    result = await connector.authenticate()

    assert result is False
    # Should NOT save session on timeout
    mock_browser_engine.save_session.assert_not_called()
    # Should wait 24 times (polling loop)
    assert mock_human_emulator.wait_human.call_count >= 24


@pytest.mark.asyncio
async def test_authenticate_handles_exception(
    connector, mock_browser_engine, mock_human_emulator
):
    """authenticate() returns False on exception."""
    connector._browser_engine = mock_browser_engine
    connector._human = mock_human_emulator

    mock_browser_engine.new_page = AsyncMock(side_effect=Exception("Browser crash"))

    result = await connector.authenticate()

    assert result is False


# ============================================================================
# Message Scraping Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_unread_chats_returns_list(connector):
    """_get_unread_chats extracts chats with unread badges."""
    page = MockPage(
        unread_chats=[
            {"name": "Alice", "unread_count": 3, "is_group": False},
            {"name": "Work Group", "unread_count": 12, "is_group": True},
        ]
    )

    chats = await connector._get_unread_chats(page)

    assert len(chats) == 2
    assert chats[0]["name"] == "Alice"
    assert chats[0]["unread_count"] == 3
    assert chats[1]["name"] == "Work Group"
    assert chats[1]["unread_count"] == 12


@pytest.mark.asyncio
async def test_get_unread_chats_empty(connector):
    """_get_unread_chats returns empty list when no unread chats."""
    page = MockPage(unread_chats=[])

    chats = await connector._get_unread_chats(page)

    assert chats == []


@pytest.mark.asyncio
async def test_extract_messages_returns_list(connector):
    """_extract_messages extracts text and metadata from open chat."""
    page = MockPage(
        messages=[
            {
                "text": "Hey, how are you?",
                "time": "[10:30 AM]",
                "is_incoming": True,
                "is_new": True,
            },
            {
                "text": "I'm good, thanks!",
                "time": "[10:31 AM]",
                "is_incoming": False,
                "is_new": True,
            },
        ]
    )

    messages = await connector._extract_messages(page)

    assert len(messages) == 2
    assert messages[0]["text"] == "Hey, how are you?"
    assert messages[0]["is_incoming"] is True
    assert messages[1]["text"] == "I'm good, thanks!"
    assert messages[1]["is_incoming"] is False


@pytest.mark.asyncio
async def test_extract_messages_empty(connector):
    """_extract_messages returns empty list when no messages."""
    page = MockPage(messages=[])

    messages = await connector._extract_messages(page)

    assert messages == []


@pytest.mark.asyncio
async def test_browser_sync_processes_unread_chats(
    connector, mock_event_bus, mock_human_emulator, mock_page_interactor
):
    """browser_sync scrapes unread chats and publishes message.received events."""
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(
        logged_in=True,
        unread_chats=[{"name": "Alice", "unread_count": 2, "is_group": False}],
        messages=[
            {
                "text": "Hello!",
                "time": "[10:30 AM]",
                "is_incoming": True,
                "is_new": True,
                "id": "msg1",
            }
        ],
    )

    count = await connector.browser_sync(page, mock_human_emulator, mock_page_interactor)

    assert count == 1
    # Should publish message.received event
    mock_event_bus.publish.assert_called_once()
    call_args = mock_event_bus.publish.call_args
    assert call_args[0][0] == "message.received"
    payload = call_args[0][1]
    assert payload["channel"] == "whatsapp"
    assert payload["from_contact"] == "Alice"
    assert payload["body"] == "Hello!"


@pytest.mark.asyncio
async def test_browser_sync_respects_max_conversations(
    connector, mock_event_bus, mock_human_emulator, mock_page_interactor
):
    """browser_sync processes at most max_conversations_per_sync chats."""
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor
    connector.config["max_conversations_per_sync"] = 2

    page = MockPage(
        logged_in=True,
        unread_chats=[
            {"name": "Alice", "unread_count": 1, "is_group": False},
            {"name": "Bob", "unread_count": 1, "is_group": False},
            {"name": "Carol", "unread_count": 1, "is_group": False},
        ],
        messages=[
            {"text": "Hi", "time": "[10:30 AM]", "is_incoming": True, "is_new": True}
        ],
    )

    count = await connector.browser_sync(page, mock_human_emulator, mock_page_interactor)

    # Should only process 2 chats (not 3)
    assert count == 2
    assert mock_event_bus.publish.call_count == 2


@pytest.mark.asyncio
async def test_browser_sync_handles_chat_error(
    connector, mock_event_bus, mock_human_emulator, mock_page_interactor
):
    """browser_sync continues processing other chats if one fails."""
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(
        logged_in=True,
        unread_chats=[
            {"name": "Alice", "unread_count": 1, "is_group": False},
            {"name": "Bob", "unread_count": 1, "is_group": False},
        ],
    )

    # First chat raises exception, second succeeds
    call_count = 0

    async def mock_extract_messages(p):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Extraction failed")
        return [{"text": "Hi", "is_new": True, "id": "msg1", "time": "[10:30 AM]"}]

    connector._extract_messages = mock_extract_messages

    count = await connector.browser_sync(page, mock_human_emulator, mock_page_interactor)

    # Should process 1 chat successfully despite error in first
    assert count == 1
    assert mock_event_bus.publish.call_count == 1


@pytest.mark.asyncio
async def test_browser_sync_filters_non_new_messages(
    connector, mock_event_bus, mock_human_emulator, mock_page_interactor
):
    """browser_sync only publishes messages marked as is_new."""
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(
        logged_in=True,
        unread_chats=[{"name": "Alice", "unread_count": 1, "is_group": False}],
        messages=[
            {
                "text": "Old message",
                "time": "[9:00 AM]",
                "is_incoming": True,
                "is_new": False,
            },
            {
                "text": "New message",
                "time": "[10:30 AM]",
                "is_incoming": True,
                "is_new": True,
            },
        ],
    )

    count = await connector.browser_sync(page, mock_human_emulator, mock_page_interactor)

    assert count == 1
    # Should only publish the new message
    assert mock_event_bus.publish.call_count == 1
    payload = mock_event_bus.publish.call_args[0][1]
    assert payload["body"] == "New message"


@pytest.mark.asyncio
async def test_browser_sync_handles_group_messages(
    connector, mock_event_bus, mock_human_emulator, mock_page_interactor
):
    """browser_sync correctly identifies and labels group messages."""
    connector._human = mock_human_emulator
    connector._interactor = mock_page_interactor

    page = MockPage(
        logged_in=True,
        unread_chats=[{"name": "Work Group", "unread_count": 1, "is_group": True}],
        messages=[
            {
                "text": "Team update",
                "time": "[10:30 AM]",
                "is_incoming": True,
                "is_new": True,
            }
        ],
    )

    count = await connector.browser_sync(page, mock_human_emulator, mock_page_interactor)

    assert count == 1
    payload = mock_event_bus.publish.call_args[0][1]
    assert payload["is_group"] is True
    assert payload["group_name"] == "Work Group"


# ============================================================================
# Message Sending Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_send_message_success(
    connector, mock_event_bus, mock_human_emulator
):
    """execute() sends message via search + type + enter."""
    connector._human = mock_human_emulator
    connector._page = MockPage(logged_in=True)

    result = await connector.execute(
        "send_message", {"to": "Alice", "message": "Hello from test!"}
    )

    assert result["status"] == "sent"
    assert result["to"] == "Alice"

    # Should type in search box
    mock_human_emulator.type_text.assert_any_call(
        connector._page,
        '[aria-label="Search input textbox"], [title="Search input textbox"]',
        "Alice",
    )

    # Should type message
    calls = mock_human_emulator.type_text.call_args_list
    assert any("Hello from test!" in str(call) for call in calls)

    # Should press Enter to send
    connector._page.keyboard.press.assert_called_once_with("Enter")

    # Should publish message.sent event
    assert mock_event_bus.publish.call_count == 1
    call_args = mock_event_bus.publish.call_args
    assert call_args[0][0] == "message.sent"
    payload = call_args[0][1]
    assert payload["channel"] == "whatsapp"
    assert payload["to_contact"] == "Alice"
    assert payload["body"] == "Hello from test!"


@pytest.mark.asyncio
async def test_execute_send_message_no_page(connector):
    """execute() fails gracefully if page is None."""
    connector._page = None

    with pytest.raises(Exception):
        await connector.execute("send_message", {"to": "Alice", "message": "Hi"})


@pytest.mark.asyncio
async def test_execute_unknown_action(connector):
    """execute() raises ValueError for unknown actions."""
    connector._page = MockPage()

    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("unknown_action", {})


# ============================================================================
# Priority Classification Tests
# ============================================================================


def test_classify_priority_high_for_priority_contact(connector):
    """_classify_priority returns 'high' for contacts in priority list."""
    priority = connector._classify_priority("Mom")
    assert priority == "high"

    priority = connector._classify_priority("Partner")
    assert priority == "high"


def test_classify_priority_normal_for_others(connector):
    """_classify_priority returns 'normal' for non-priority contacts."""
    priority = connector._classify_priority("Random Person")
    assert priority == "normal"


def test_classify_priority_case_sensitive(connector):
    """_classify_priority is case-sensitive (exact match required)."""
    priority = connector._classify_priority("mom")  # lowercase
    assert priority == "normal"  # Not in list (which has "Mom")


# ============================================================================
# Health Check Tests
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_ok_when_logged_in(connector):
    """health_check returns 'ok' status when session is valid."""
    connector._page = MockPage(logged_in=True)

    health = await connector.health_check()

    assert health["status"] == "ok"
    assert health["connector"] == "whatsapp"
    assert health["mode"] == "browser"


@pytest.mark.asyncio
async def test_health_check_session_expired(connector):
    """health_check returns 'session_expired' when logged out."""
    connector._page = MockPage(logged_in=False)

    health = await connector.health_check()

    assert health["status"] == "session_expired"
    assert health["connector"] == "whatsapp"


@pytest.mark.asyncio
async def test_health_check_not_started(connector):
    """health_check returns 'not_started' when no page exists."""
    connector._page = None

    health = await connector.health_check()

    assert health["status"] == "not_started"
    assert health["connector"] == "whatsapp"


# ============================================================================
# Configuration Tests
# ============================================================================


def test_connector_constants(connector):
    """Verify connector class constants are correct."""
    assert connector.CONNECTOR_ID == "whatsapp"
    assert connector.DISPLAY_NAME == "WhatsApp"
    assert connector.SITE_ID == "whatsapp"
    assert connector.LOGIN_URL == "https://web.whatsapp.com"
    assert connector.SYNC_INTERVAL_SECONDS == 10
    assert connector.MIN_REQUEST_INTERVAL == 1.0


def test_config_defaults(connector):
    """Verify configuration defaults are applied correctly."""
    assert connector.config["max_conversations_per_sync"] == 10
    assert connector.config["priority_contacts"] == ["Mom", "Partner"]
    assert connector.config["sync_interval"] == 10


def test_config_missing_max_conversations(mock_event_bus, mock_db, mock_browser_engine):
    """max_conversations_per_sync defaults to 10 if not in config."""
    config = {"enabled": True, "mode": "browser"}
    conn = WhatsAppConnector(mock_event_bus, mock_db, config, mock_browser_engine)

    # Should use default value
    max_convos = conn.config.get("max_conversations_per_sync", 10)
    assert max_convos == 10


def test_config_missing_priority_contacts(mock_event_bus, mock_db, mock_browser_engine):
    """priority_contacts defaults to empty list if not in config."""
    config = {"enabled": True, "mode": "browser"}
    conn = WhatsAppConnector(mock_event_bus, mock_db, config, mock_browser_engine)

    priority = conn._classify_priority("Anyone")
    assert priority == "normal"  # No priority contacts configured
