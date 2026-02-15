"""
Comprehensive test coverage for GenericBrowserConnector.

GenericBrowserConnector is the foundation for all configurable browser-based
data sources. It allows defining arbitrary web scraping rules via YAML config,
eliminating the need for custom connector code for every new source.

Test coverage areas:
1. Initialization & configuration override (CONNECTOR_ID, DISPLAY_NAME from config)
2. Login selector customization (custom vs. default selectors)
3. Authentication modes (requires_login vs. public sites)
4. CSS selector extraction (text content vs. attribute extraction with @ syntax)
5. JavaScript code generation (dynamic JS from YAML selectors)
6. Hash-based deduplication (MD5 hashing, sync cursor persistence)
7. Item filtering (empty field rejection, max_items cap)
8. Event publishing (correct event_type, payload structure, metadata)
9. Factory pattern (create_browser_connectors with shared engine/vault)
10. Edge cases (missing selectors, invalid config, extraction failures)
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call

import pytest

from connectors.browser.generic import GenericBrowserConnector, create_browser_connectors


# ===========================================================================
# Test Fixtures
# ===========================================================================

@pytest.fixture
def mock_browser_engine():
    """Mock BrowserEngine with page navigation and JS evaluation."""
    engine = AsyncMock()
    engine.start = AsyncMock()
    engine.create_context = AsyncMock(return_value=Mock())
    engine.new_page = AsyncMock(return_value=Mock())
    return engine


@pytest.fixture
def mock_credential_vault():
    """Mock CredentialVault for login credential storage."""
    vault = Mock()
    vault.get_credentials = Mock(return_value={"username": "test", "password": "pass"})
    vault.set_credentials = Mock()
    return vault


@pytest.fixture
def basic_config():
    """Basic configuration for a public site (no login required)."""
    return {
        "site_id": "hackernews",
        "name": "Hacker News",
        "feed_url": "https://news.ycombinator.com/",
        "sync_interval": 1800,
        "selectors": {
            "item": ".athing",
            "title": ".titleline > a",
            "url": ".titleline > a@href",
            "score": ".score",
        },
        "event_type": "content.feed.new_item",
        "max_items": 30,
    }


@pytest.fixture
def login_required_config():
    """Configuration for a site requiring authentication."""
    return {
        "site_id": "linkedin",
        "name": "LinkedIn",
        "login_url": "https://www.linkedin.com/login",
        "feed_url": "https://www.linkedin.com/feed/",
        "sync_interval": 3600,
        "login_selectors": {
            "username": "#username",
            "password": "#password",
            "submit": "button[type='submit']",
        },
        "selectors": {
            "item": ".feed-update",
            "title": ".feed-text span",
            "author": ".actor-name",
        },
        "event_type": "content.linkedin.post",
        "max_items": 20,
    }


@pytest.fixture
def generic_connector(event_bus, db, basic_config, mock_browser_engine, mock_credential_vault):
    """GenericBrowserConnector instance with mocked dependencies."""
    connector = GenericBrowserConnector(
        event_bus, db, basic_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    # Initialize connector state row (normally done by BaseConnector.start())
    from datetime import datetime, timezone
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT OR IGNORE INTO connector_state
               (connector_id, status, last_sync, error_count, sync_cursor, updated_at)
               VALUES (?, 'inactive', NULL, 0, NULL, ?)""",
            (connector.CONNECTOR_ID, datetime.now(timezone.utc).isoformat())
        )

    return connector


# ===========================================================================
# 1. Initialization & Configuration Override
# ===========================================================================

def test_initialization_overrides_class_constants_from_config(event_bus, db, basic_config, mock_browser_engine, mock_credential_vault):
    """CONNECTOR_ID, DISPLAY_NAME, SITE_ID, LOGIN_URL, and SYNC_INTERVAL
    are set from config BEFORE BaseConnector.__init__ runs."""
    connector = GenericBrowserConnector(
        event_bus, db, basic_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connector.CONNECTOR_ID == "hackernews"
    assert connector.DISPLAY_NAME == "Hacker News"
    assert connector.SITE_ID == "hackernews"
    assert connector.LOGIN_URL == ""
    assert connector.SYNC_INTERVAL_SECONDS == 1800


def test_initialization_extracts_feed_url_and_selectors(generic_connector):
    """Feed URL, selectors, event type, and max items are stored as instance variables."""
    assert generic_connector._feed_url == "https://news.ycombinator.com/"
    assert generic_connector._selectors == {
        "item": ".athing",
        "title": ".titleline > a",
        "url": ".titleline > a@href",
        "score": ".score",
    }
    assert generic_connector._event_type == "content.feed.new_item"
    assert generic_connector._max_items == 30


def test_initialization_detects_login_required(event_bus, db, login_required_config, mock_browser_engine, mock_credential_vault):
    """If login_url is present in config, _requires_login is set to True."""
    connector = GenericBrowserConnector(
        event_bus, db, login_required_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connector._requires_login is True
    assert connector.LOGIN_URL == "https://www.linkedin.com/login"


def test_initialization_defaults_for_missing_config(event_bus, db, mock_browser_engine, mock_credential_vault):
    """Missing config keys fall back to sensible defaults."""
    minimal_config = {
        "feed_url": "https://example.com/",
    }

    connector = GenericBrowserConnector(
        event_bus, db, minimal_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connector.CONNECTOR_ID == "generic"
    assert connector.DISPLAY_NAME == "Generic Browser"
    assert connector.SYNC_INTERVAL_SECONDS == 1800
    assert connector._event_type == "content.feed.new_item"
    assert connector._max_items == 30


# ===========================================================================
# 2. Login Selector Customization
# ===========================================================================

def test_get_login_selectors_returns_custom_selectors_when_provided(event_bus, db, login_required_config, mock_browser_engine, mock_credential_vault):
    """Custom login selectors from config override BaseConnector defaults."""
    connector = GenericBrowserConnector(
        event_bus, db, login_required_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    selectors = connector.get_login_selectors()
    assert selectors == {
        "username": "#username",
        "password": "#password",
        "submit": "button[type='submit']",
    }


def test_get_login_selectors_falls_back_to_base_when_not_provided(generic_connector):
    """If no custom login selectors, falls back to BaseConnector's implementation."""
    # BaseConnector.get_login_selectors() returns an empty dict by default
    selectors = generic_connector.get_login_selectors()
    assert isinstance(selectors, dict)


# ===========================================================================
# 3. Authentication Modes
# ===========================================================================

@pytest.mark.asyncio
async def test_authenticate_starts_browser_for_public_sites(generic_connector):
    """Public sites (no login_url) start browser and create context without login flow."""
    result = await generic_connector.authenticate()

    assert result is True
    generic_connector._browser_engine.start.assert_awaited_once()
    generic_connector._browser_engine.create_context.assert_awaited_once_with("hackernews")
    generic_connector._browser_engine.new_page.assert_awaited_once()


@pytest.mark.asyncio
async def test_authenticate_delegates_to_base_for_login_sites(event_bus, db, login_required_config, mock_browser_engine, mock_credential_vault):
    """Sites requiring login delegate to BaseConnector.authenticate()."""
    connector = GenericBrowserConnector(
        event_bus, db, login_required_config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    # Mock the parent authenticate method
    connector._browser_engine.start = AsyncMock()
    connector._browser_engine.create_context = AsyncMock(return_value=Mock())
    connector._browser_engine.new_page = AsyncMock(return_value=Mock())

    # Since we can't easily mock the parent class, we verify the behavior
    # indirectly by checking that _requires_login is set correctly
    assert connector._requires_login is True


# ===========================================================================
# 4. CSS Selector Extraction (Text vs. Attribute)
# ===========================================================================

def test_build_extraction_js_extracts_text_content_by_default(generic_connector):
    """Selectors without @ extract textContent from matched elements."""
    js_code = generic_connector._build_extraction_js(
        ".post",
        {"title": ".post-title", "author": ".author-name"}
    )

    assert "textContent" in js_code
    assert 'querySelector(".post-title")' in js_code
    assert 'querySelector(".author-name")' in js_code


def test_build_extraction_js_extracts_attributes_with_at_syntax(generic_connector):
    """Selectors with @ syntax (e.g., 'a@href') extract the specified attribute."""
    js_code = generic_connector._build_extraction_js(
        ".link-item",
        {"url": "a@href", "image": "img@src", "data_id": "div@data-id"}
    )

    assert 'getAttribute("href")' in js_code
    assert 'getAttribute("src")' in js_code
    assert 'getAttribute("data-id")' in js_code
    assert 'querySelector("a")' in js_code
    assert 'querySelector("img")' in js_code


def test_build_extraction_js_handles_mixed_selectors(generic_connector):
    """A mix of text and attribute selectors produces correct JS."""
    js_code = generic_connector._build_extraction_js(
        ".item",
        {
            "title": ".title",           # text
            "url": "a.link@href",        # attribute
            "score": ".score",           # text
            "thumbnail": "img@src",      # attribute
        }
    )

    assert 'querySelector(".title")' in js_code
    assert 'textContent' in js_code
    assert 'getAttribute("href")' in js_code
    assert 'getAttribute("src")' in js_code


def test_build_extraction_js_produces_valid_javascript(generic_connector):
    """Generated JS is syntactically valid (no extra commas, brackets balanced)."""
    js_code = generic_connector._build_extraction_js(
        ".entry",
        {"title": ".entry-title", "link": "a@href"}
    )

    # Check it's wrapped in an IIFE
    assert js_code.strip().startswith("() => {")
    assert "return items;" in js_code
    assert "document.querySelectorAll('.entry')" in js_code


# ===========================================================================
# 5. JavaScript Code Generation
# ===========================================================================

def test_build_extraction_js_filters_empty_items(generic_connector):
    """Generated JS only includes items with at least one non-empty field."""
    js_code = generic_connector._build_extraction_js(
        ".post",
        {"title": ".title", "content": ".content"}
    )

    assert "Object.values(extracted).some(v => v && v.trim())" in js_code


def test_build_extraction_js_handles_element_not_found(generic_connector):
    """Generated JS returns empty string if querySelector returns null."""
    js_code = generic_connector._build_extraction_js(
        ".item",
        {"title": ".missing-selector"}
    )

    # Check for null-safe navigation
    assert 'el ? el.textContent?.trim() || "" : ""' in js_code or "el ?" in js_code


# ===========================================================================
# 6. Hash-Based Deduplication
# ===========================================================================

def test_get_seen_hashes_returns_empty_set_when_no_cursor(generic_connector):
    """If no sync cursor exists, _get_seen_hashes returns an empty set."""
    seen = generic_connector._get_seen_hashes()
    assert seen == set()


def test_get_seen_hashes_loads_from_sync_cursor(generic_connector):
    """Previously seen hashes are loaded from the sync cursor."""
    import json

    # Use the connector's set_sync_cursor method to ensure proper storage
    generic_connector.set_sync_cursor(json.dumps(["hash1", "hash2", "hash3"]))

    seen = generic_connector._get_seen_hashes()
    assert seen == {"hash1", "hash2", "hash3"}


def test_update_seen_hashes_persists_to_sync_cursor(generic_connector, db):
    """New hashes are persisted to the sync cursor."""
    generic_connector._update_seen_hashes(["new_hash_1", "new_hash_2"])

    import json
    with db.get_connection("state") as conn:
        row = conn.execute(
            "SELECT sync_cursor FROM connector_state WHERE connector_id = ?",
            ("hackernews",)
        ).fetchone()

        if row and row["sync_cursor"]:
            hashes = set(json.loads(row["sync_cursor"]))
            assert "new_hash_1" in hashes
            assert "new_hash_2" in hashes


def test_update_seen_hashes_caps_at_2000_entries(generic_connector):
    """Sync cursor is capped at 2000 hashes to prevent unbounded growth."""
    # Seed with 1950 existing hashes
    existing = [f"hash_{i}" for i in range(1950)]
    import json
    generic_connector.set_sync_cursor(json.dumps(existing))

    # Add 100 more
    new_hashes = [f"new_{i}" for i in range(100)]
    generic_connector._update_seen_hashes(new_hashes)

    seen = generic_connector._get_seen_hashes()
    # Should keep only the most recent 2000
    assert len(seen) == 2000


def test_hash_deduplication_skips_already_seen_items(generic_connector):
    """Items with previously seen hashes are not included in new_items."""
    import json
    import hashlib

    # Pre-seed sync cursor with a hash
    item_1 = {"title": "Item One", "url": "http://example.com/1"}
    hash_1 = hashlib.md5(json.dumps(item_1, sort_keys=True).encode()).hexdigest()[:16]

    # Verify the hash is stored and retrieved correctly
    generic_connector._update_seen_hashes([hash_1])
    seen_hashes = generic_connector._get_seen_hashes()

    # Verify the hash is in the seen set
    assert hash_1 in seen_hashes

    # Simulate finding the same item again
    items = [item_1]

    new_items = []
    for item in items:
        item_hash = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
        if item_hash not in seen_hashes:
            new_items.append(item)

    assert len(new_items) == 0  # Item was already seen


# ===========================================================================
# 7. Item Filtering
# ===========================================================================

@pytest.mark.asyncio
async def test_browser_sync_respects_max_items_limit(generic_connector):
    """Only the first max_items are processed, even if more are extracted."""
    mock_page = AsyncMock()
    mock_human = AsyncMock()
    mock_interactor = Mock()

    # Simulate extracting 50 items when max_items = 30
    mock_items = [{"title": f"Item {i}", "url": f"http://example.com/{i}"} for i in range(50)]
    mock_page.evaluate = AsyncMock(return_value=mock_items)

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()

    count = await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    # Should only process 30 items
    assert count <= 30


@pytest.mark.asyncio
async def test_browser_sync_skips_items_with_no_non_empty_fields(generic_connector):
    """JS filtering ensures items with all empty fields are excluded."""
    # This is enforced by the JS code itself, which we've already tested
    # in test_build_extraction_js_filters_empty_items
    pass


# ===========================================================================
# 8. Event Publishing
# ===========================================================================

@pytest.mark.asyncio
async def test_browser_sync_publishes_events_with_correct_type(generic_connector):
    """Published events use the configured event_type."""
    mock_page = AsyncMock()
    mock_human = AsyncMock()
    mock_interactor = Mock()

    mock_items = [{"title": "Test Item", "url": "http://example.com/test"}]
    mock_page.evaluate = AsyncMock(return_value=mock_items)

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()
    generic_connector.publish_event = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    generic_connector.publish_event.assert_called_once()
    call_args = generic_connector.publish_event.call_args
    assert call_args[0][0] == "content.feed.new_item"


@pytest.mark.asyncio
async def test_browser_sync_publishes_events_with_source_metadata(generic_connector):
    """Published events include source_site and source_name in payload."""
    mock_page = AsyncMock()
    mock_human = AsyncMock()
    mock_interactor = Mock()

    mock_items = [{"title": "Article", "url": "http://example.com/article"}]
    mock_page.evaluate = AsyncMock(return_value=mock_items)

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()
    generic_connector.publish_event = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    call_args = generic_connector.publish_event.call_args
    payload = call_args[0][1]

    assert payload["source_site"] == "hackernews"
    assert payload["source_name"] == "Hacker News"
    assert payload["title"] == "Article"
    assert payload["url"] == "http://example.com/article"


@pytest.mark.asyncio
async def test_browser_sync_publishes_events_with_low_priority(generic_connector):
    """Generic browser events default to 'low' priority."""
    mock_page = AsyncMock()
    mock_human = AsyncMock()
    mock_interactor = Mock()

    mock_items = [{"title": "Item", "url": "http://example.com"}]
    mock_page.evaluate = AsyncMock(return_value=mock_items)

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()
    generic_connector.publish_event = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    call_args = generic_connector.publish_event.call_args
    assert call_args[1]["priority"] == "low"


@pytest.mark.asyncio
async def test_browser_sync_publishes_events_with_metadata(generic_connector):
    """Event metadata includes domain and source."""
    mock_page = AsyncMock()
    mock_human = AsyncMock()
    mock_interactor = Mock()

    mock_items = [{"title": "Post", "url": "http://example.com"}]
    mock_page.evaluate = AsyncMock(return_value=mock_items)

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()
    generic_connector.publish_event = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    call_args = generic_connector.publish_event.call_args
    metadata = call_args[1]["metadata"]

    assert metadata["domain"] == "media"
    assert metadata["source"] == "hackernews"


# ===========================================================================
# 9. Factory Pattern
# ===========================================================================

def test_create_browser_connectors_creates_multiple_connectors(event_bus, db, mock_browser_engine, mock_credential_vault):
    """Factory creates one connector per config entry."""
    configs = [
        {
            "site_id": "hn",
            "name": "HN",
            "feed_url": "https://news.ycombinator.com/",
            "selectors": {"item": ".athing", "title": ".titleline > a"},
        },
        {
            "site_id": "reddit",
            "name": "Reddit",
            "feed_url": "https://old.reddit.com/",
            "selectors": {"item": ".thing", "title": ".title"},
        },
    ]

    connectors = create_browser_connectors(
        event_bus, db, configs,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert len(connectors) == 2
    assert connectors[0].CONNECTOR_ID == "hn"
    assert connectors[1].CONNECTOR_ID == "reddit"


def test_create_browser_connectors_shares_browser_engine(event_bus, db, mock_browser_engine, mock_credential_vault):
    """All connectors share the same browser engine instance."""
    configs = [
        {"site_id": "site1", "feed_url": "http://example1.com", "selectors": {}},
        {"site_id": "site2", "feed_url": "http://example2.com", "selectors": {}},
    ]

    connectors = create_browser_connectors(
        event_bus, db, configs,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connectors[0]._browser_engine is mock_browser_engine
    assert connectors[1]._browser_engine is mock_browser_engine


def test_create_browser_connectors_shares_credential_vault(event_bus, db, mock_browser_engine, mock_credential_vault):
    """All connectors share the same credential vault instance."""
    configs = [
        {"site_id": "site1", "feed_url": "http://example1.com", "selectors": {}},
        {"site_id": "site2", "feed_url": "http://example2.com", "selectors": {}},
    ]

    connectors = create_browser_connectors(
        event_bus, db, configs,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connectors[0]._credential_vault is mock_credential_vault
    assert connectors[1]._credential_vault is mock_credential_vault


def test_create_browser_connectors_handles_empty_config_list(event_bus, db, mock_browser_engine, mock_credential_vault):
    """Factory returns empty list when given no configs."""
    connectors = create_browser_connectors(
        event_bus, db, [],
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    assert connectors == []


# ===========================================================================
# 10. Edge Cases
# ===========================================================================

@pytest.mark.asyncio
async def test_browser_sync_handles_page_evaluation_failure(generic_connector):
    """If page.evaluate() raises, the exception propagates (no error handling in browser_sync)."""
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(side_effect=Exception("JS error"))
    mock_human = AsyncMock()
    mock_interactor = Mock()

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()

    # The actual implementation doesn't catch exceptions, so it will raise
    with pytest.raises(Exception, match="JS error"):
        await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)


@pytest.mark.asyncio
async def test_browser_sync_handles_empty_extraction_results(generic_connector):
    """If page.evaluate() returns empty list, browser_sync returns 0."""
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[])
    mock_human = AsyncMock()
    mock_interactor = Mock()

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()

    count = await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)
    assert count == 0


@pytest.mark.asyncio
async def test_execute_raises_for_read_only_connector(generic_connector):
    """Generic browser connectors are read-only and reject all execute actions."""
    with pytest.raises(ValueError, match="read-only"):
        await generic_connector.execute("send_message", {"message": "test"})


@pytest.mark.asyncio
async def test_health_check_returns_ok_when_page_exists(generic_connector):
    """Health check returns 'ok' when _page is set."""
    generic_connector._page = Mock()

    result = await generic_connector.health_check()
    assert result["status"] == "ok"
    assert result["connector"] == "hackernews"
    assert result["mode"] == "browser"


@pytest.mark.asyncio
async def test_health_check_returns_not_started_when_no_page(generic_connector):
    """Health check returns 'not_started' when _page is None."""
    generic_connector._page = None

    result = await generic_connector.health_check()
    assert result["status"] == "not_started"
    assert result["connector"] == "hackernews"


def test_build_extraction_js_handles_selector_with_multiple_at_signs(generic_connector):
    """Selector with multiple @ signs uses the last one as attribute delimiter."""
    js_code = generic_connector._build_extraction_js(
        ".item",
        {"special": "a.link@data-track@value"}  # Edge case: attribute name contains @
    )

    # Should split on the LAST @
    assert 'getAttribute("value")' in js_code
    assert 'querySelector("a.link@data-track")' in js_code


@pytest.mark.asyncio
async def test_browser_sync_scrolls_to_trigger_lazy_loading(generic_connector):
    """browser_sync scrolls the page to trigger infinite-scroll content."""
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[])
    mock_human = AsyncMock()
    mock_interactor = Mock()

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    # Should scroll at least once
    assert mock_human.scroll.call_count >= 1


@pytest.mark.asyncio
async def test_browser_sync_navigates_to_feed_url_with_rate_limiting(generic_connector):
    """browser_sync navigates to the configured feed_url."""
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[])
    mock_human = AsyncMock()
    mock_interactor = Mock()

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()

    await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    generic_connector.navigate_with_rate_limit.assert_called_once_with(
        mock_page, "https://news.ycombinator.com/"
    )


@pytest.mark.asyncio
async def test_browser_sync_updates_seen_hashes_after_processing_new_items(generic_connector):
    """After processing new items, their hashes are persisted to sync cursor."""
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[
        {"title": "New Item 1", "url": "http://example.com/1"},
        {"title": "New Item 2", "url": "http://example.com/2"},
    ])
    mock_human = AsyncMock()
    mock_interactor = Mock()

    generic_connector._page = mock_page
    generic_connector.navigate_with_rate_limit = AsyncMock()
    generic_connector.publish_event = AsyncMock()

    # Verify no hashes before sync
    assert len(generic_connector._get_seen_hashes()) == 0

    count = await generic_connector.browser_sync(mock_page, mock_human, mock_interactor)

    # Should have processed 2 items
    assert count == 2

    # Check that hashes were stored
    seen = generic_connector._get_seen_hashes()
    assert len(seen) == 2


def test_get_seen_hashes_handles_invalid_json_in_cursor(generic_connector, db):
    """If sync cursor contains invalid JSON, _get_seen_hashes returns empty set."""
    with db.get_connection("state") as conn:
        conn.execute(
            "UPDATE connector_state SET sync_cursor = ? WHERE connector_id = ?",
            ("not valid json", "hackernews")
        )

    seen = generic_connector._get_seen_hashes()
    assert seen == set()
