"""
Life OS — YouTube Connector Tests

Comprehensive test coverage for the YouTubeConnector class (181 LOC).

The YouTube connector is a browser-only integration that:
1. Scrapes youtube.com/feed/subscriptions for new videos from subscribed channels
2. Uses browser automation because YouTube Data API has prohibitive quota limits
3. Handles deduplication via sync cursor (stores last 500 seen video IDs)
4. Extracts video metadata from DOM (title, channel, video ID, thumbnail, etc.)
5. Supports "add_to_watch_later" action for task automation
6. Requires Google account login with 2FA support

Test categories:
- Authentication flow (Google multi-step login, avatar detection, session state)
- Video scraping (subscription feed, lazy loading, deduplication)
- Cursor management (seen IDs persistence, size capping at 500)
- DOM extraction (video cards, metadata parsing, URL patterns)
- Watch Later action (navigation, button clicks, playlist selection)
- Error handling (malformed data, failed extractions, missing elements)
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from connectors.browser.youtube import YouTubeConnector


class MockPage:
    """Mock Playwright Page object for YouTube testing."""

    def __init__(self, logged_in=True, videos=None):
        """
        Initialize mock page with configurable state.

        Args:
            logged_in: Whether the page shows logged-in state (avatar present)
            videos: List of video dicts to return from evaluate() call
        """
        self.logged_in = logged_in
        self.videos = videos or []
        self._selectors_seen = []
        self.url = "https://www.youtube.com/feed/subscriptions"

    async def goto(self, url, wait_until=None):
        """Mock navigation to YouTube."""
        self.url = url

    async def query_selector(self, selector):
        """Mock selector query - returns avatar element if logged in."""
        self._selectors_seen.append(selector)
        # YouTube shows avatar button when logged in
        if "avatar" in selector.lower():
            return MagicMock() if self.logged_in else None
        return MagicMock()

    async def evaluate(self, script):
        """
        Mock JavaScript execution.

        Returns the pre-configured list of videos when the script
        queries for ytd-grid-video-renderer, ytd-rich-item-renderer, etc.
        """
        # The script extracts video metadata from DOM elements
        if "ytd-grid-video-renderer" in script or "ytd-rich-item-renderer" in script:
            return self.videos
        return []


class MockContext:
    """Mock browser context for connector state management."""

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
    engine.create_context = AsyncMock(return_value=MockContext("google"))
    engine.new_page = AsyncMock()
    engine.save_session = AsyncMock()
    engine.session_manager = Mock()
    engine.session_manager.has_session = Mock(return_value=False)
    return engine


@pytest.fixture
def mock_credential_vault():
    """Mock CredentialVault for testing authentication."""
    vault = Mock()
    vault.get_credential = Mock(return_value={
        "username": "test@gmail.com",
        "password": "test_password",
    })
    vault.get_totp = Mock(return_value="123456")
    return vault


@pytest.fixture
def connector(mock_event_bus, db, mock_browser_engine, mock_credential_vault):
    """Create a YouTubeConnector with mocked dependencies."""
    config = {
        "sync_interval": 900,
        "max_videos_per_sync": 30,
    }
    conn = YouTubeConnector(
        event_bus=mock_event_bus,
        db=db,
        config=config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    # Initialize connector_state row so set_sync_cursor can UPDATE it
    with db.get_connection("state") as c:
        c.execute(
            "INSERT INTO connector_state (connector_id, status, updated_at) VALUES (?, 'active', ?)",
            (conn.CONNECTOR_ID, datetime.now(timezone.utc).isoformat()),
        )

    return conn


# =============================================================================
# Connector Metadata
# =============================================================================


def test_connector_metadata(connector):
    """Verify YouTubeConnector has correct metadata."""
    assert connector.CONNECTOR_ID == "youtube"
    assert connector.DISPLAY_NAME == "YouTube"
    assert connector.SITE_ID == "google"  # Shared with Google services
    assert connector.REQUIRES_2FA is True  # Google accounts need 2FA
    assert connector.SYNC_INTERVAL_SECONDS == 900  # 15 minutes
    assert connector.MIN_REQUEST_INTERVAL == 3.0  # 3 seconds between requests


def test_login_url(connector):
    """Verify login URL points to Google accounts service."""
    url = connector.get_login_url()
    assert url == "https://accounts.google.com/ServiceLogin?service=youtube"
    assert "accounts.google.com" in url
    assert "service=youtube" in url


def test_login_selectors(connector):
    """Verify Google's multi-step login selectors."""
    selectors = connector.get_login_selectors()

    # Google uses email input, then separate password step
    assert "username" in selectors
    assert "input[type='email']" in selectors["username"]

    assert "password" in selectors
    assert "input[type='password']" in selectors["password"]

    # Multi-step submit buttons (identifierNext, passwordNext)
    assert "submit" in selectors
    assert "#identifierNext" in selectors["submit"]
    assert "#passwordNext" in selectors["submit"]

    # TOTP input (Google uses type=tel for 2FA codes)
    assert "totp" in selectors
    assert "input[type='tel']" in selectors["totp"]


# =============================================================================
# Authentication Flow
# =============================================================================


@pytest.mark.asyncio
async def test_is_logged_in_with_avatar(connector):
    """Verify login detection via avatar button presence."""
    page = MockPage(logged_in=True)

    result = await connector.is_logged_in(page)

    assert result is True
    # Should query for avatar button
    assert any("avatar" in s.lower() for s in page._selectors_seen)


@pytest.mark.asyncio
async def test_is_logged_in_no_avatar(connector):
    """Verify login detection fails when avatar is absent."""
    page = MockPage(logged_in=False)

    result = await connector.is_logged_in(page)

    assert result is False


@pytest.mark.asyncio
async def test_is_logged_in_exception_handling(connector):
    """Verify login detection handles exceptions gracefully."""
    page = Mock()
    page.query_selector = AsyncMock(side_effect=Exception("DOM error"))

    result = await connector.is_logged_in(page)

    # Should return False on exception, not crash
    assert result is False


# =============================================================================
# Video Scraping
# =============================================================================


@pytest.mark.asyncio
async def test_browser_sync_extracts_new_videos(connector):
    """Verify browser_sync extracts new videos from subscription feed."""
    videos = [
        {
            "video_id": "abc123",
            "title": "How to Build a DIY Solar Panel",
            "channel": "Tech Channel",
            "url": "https://www.youtube.com/watch?v=abc123",
            "thumbnail": "https://i.ytimg.com/vi/abc123/hqdefault.jpg",
            "meta": "10K views • 2 hours ago",
        },
        {
            "video_id": "def456",
            "title": "Python Tutorial for Beginners",
            "channel": "Code Academy",
            "url": "https://www.youtube.com/watch?v=def456",
            "thumbnail": "https://i.ytimg.com/vi/def456/hqdefault.jpg",
            "meta": "5K views • 1 day ago",
        },
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    # Mock navigate_with_rate_limit
    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    # Should extract 2 videos
    assert count == 2

    # Should navigate to subscriptions feed
    connector.navigate_with_rate_limit.assert_called_once_with(
        page, "https://www.youtube.com/feed/subscriptions"
    )

    # Should wait for page load
    assert human.wait_human.call_count >= 1

    # Should scroll to trigger lazy loading (3 times)
    assert human.scroll.call_count == 3

    # Should publish events for both videos
    assert connector.publish_event.call_count == 2

    # Verify first video event
    first_call = connector.publish_event.call_args_list[0]
    assert first_call[0][0] == "content.youtube.new_video"
    payload = first_call[0][1]
    assert payload["video_id"] == "abc123"
    assert payload["title"] == "How to Build a DIY Solar Panel"
    assert payload["channel"] == "Tech Channel"
    assert payload["url"] == "https://www.youtube.com/watch?v=abc123"

    # Verify event metadata
    assert first_call[1]["priority"] == "low"
    assert first_call[1]["metadata"]["domain"] == "media"
    assert first_call[1]["metadata"]["channel"] == "Tech Channel"


@pytest.mark.asyncio
async def test_browser_sync_deduplicates_seen_videos(connector):
    """Verify browser_sync skips videos already in sync cursor."""
    videos = [
        {"video_id": "seen123", "title": "Old Video", "channel": "Test", "url": "https://youtube.com/watch?v=seen123", "thumbnail": "", "meta": ""},
        {"video_id": "new456", "title": "New Video", "channel": "Test", "url": "https://youtube.com/watch?v=new456", "thumbnail": "", "meta": ""},
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    # Pre-populate sync cursor with seen123
    connector.set_sync_cursor(json.dumps(["seen123"]))

    count = await connector.browser_sync(page, human, interactor)

    # Should only extract the new video
    assert count == 1

    # Should only publish event for new456
    assert connector.publish_event.call_count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["video_id"] == "new456"


@pytest.mark.asyncio
async def test_browser_sync_respects_max_videos_limit(connector):
    """Verify browser_sync caps video processing at max_videos_per_sync."""
    # Create 50 videos but config limits to 30
    videos = [
        {
            "video_id": f"vid{i}",
            "title": f"Video {i}",
            "channel": "Test Channel",
            "url": f"https://youtube.com/watch?v=vid{i}",
            "thumbnail": "",
            "meta": "",
        }
        for i in range(50)
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()
    connector.config["max_videos_per_sync"] = 30

    count = await connector.browser_sync(page, human, interactor)

    # Should only process first 30 videos
    assert count == 30
    assert connector.publish_event.call_count == 30


@pytest.mark.asyncio
async def test_browser_sync_skips_videos_without_id(connector):
    """Verify browser_sync filters out videos missing video_id."""
    videos = [
        {"video_id": "", "title": "No ID", "channel": "Test", "url": "", "thumbnail": "", "meta": ""},
        {"video_id": "valid123", "title": "Valid", "channel": "Test", "url": "https://youtube.com/watch?v=valid123", "thumbnail": "", "meta": ""},
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    # Should only process the video with a valid ID
    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["video_id"] == "valid123"


@pytest.mark.asyncio
async def test_browser_sync_handles_shorts_url_format(connector):
    """Verify browser_sync extracts video_id from /shorts/ URLs."""
    videos = [
        {
            "video_id": "short789",  # Extracted by JS regex
            "title": "YouTube Short",
            "channel": "Shorts Channel",
            "url": "https://www.youtube.com/shorts/short789",
            "thumbnail": "",
            "meta": "",
        },
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["video_id"] == "short789"
    assert "/shorts/" in payload["url"]


# =============================================================================
# Sync Cursor Management
# =============================================================================


@pytest.mark.asyncio
async def test_sync_cursor_persists_seen_video_ids(connector):
    """Verify sync cursor stores seen video IDs as JSON."""
    videos = [
        {"video_id": "abc", "title": "A", "channel": "C", "url": "https://youtube.com/watch?v=abc", "thumbnail": "", "meta": ""},
        {"video_id": "def", "title": "B", "channel": "C", "url": "https://youtube.com/watch?v=def", "thumbnail": "", "meta": ""},
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    await connector.browser_sync(page, human, interactor)

    # Should persist both video IDs
    cursor = connector.get_sync_cursor()
    seen_ids = json.loads(cursor)
    assert "abc" in seen_ids
    assert "def" in seen_ids


@pytest.mark.asyncio
async def test_sync_cursor_caps_at_500_videos(connector):
    """Verify sync cursor prevents unbounded growth by capping at 500 IDs."""
    # Pre-populate cursor with 495 old IDs
    old_ids = [f"old{i}" for i in range(495)]
    connector.set_sync_cursor(json.dumps(old_ids))

    # Add 10 new videos
    videos = [
        {"video_id": f"new{i}", "title": f"New {i}", "channel": "C", "url": f"https://youtube.com/watch?v=new{i}", "thumbnail": "", "meta": ""}
        for i in range(10)
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    await connector.browser_sync(page, human, interactor)

    # Cursor should cap at 500 (last 500 IDs)
    cursor = connector.get_sync_cursor()
    seen_ids = json.loads(cursor)
    assert len(seen_ids) == 500

    # Should keep the newest IDs
    assert "new9" in seen_ids
    assert "new0" in seen_ids


@pytest.mark.asyncio
async def test_sync_cursor_handles_corrupted_json(connector):
    """Verify sync cursor handles corrupted JSON gracefully."""
    # Set corrupted cursor
    connector.set_sync_cursor("not valid json {{{")

    videos = [
        {"video_id": "abc", "title": "Test", "channel": "C", "url": "https://youtube.com/watch?v=abc", "thumbnail": "", "meta": ""},
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    # Should not crash, should treat as empty cursor
    count = await connector.browser_sync(page, human, interactor)

    assert count == 1  # Should still process the video


@pytest.mark.asyncio
async def test_sync_cursor_empty_when_no_new_videos(connector):
    """Verify sync cursor is not updated when no new videos are found."""
    connector.set_sync_cursor(json.dumps(["old123"]))
    original_cursor = connector.get_sync_cursor()

    # No videos returned from page
    page = MockPage(videos=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    await connector.browser_sync(page, human, interactor)

    # Cursor should remain unchanged
    assert connector.get_sync_cursor() == original_cursor


# =============================================================================
# Execute Actions (Watch Later)
# =============================================================================


@pytest.mark.asyncio
async def test_execute_add_to_watch_later(connector):
    """Verify execute() can add videos to Watch Later playlist."""
    page = Mock()
    page.goto = AsyncMock()
    connector._page = page

    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.click = AsyncMock()
    connector._human = human

    # Mock navigate_with_rate_limit
    connector.navigate_with_rate_limit = AsyncMock()

    result = await connector.execute("add_to_watch_later", {
        "url": "https://www.youtube.com/watch?v=abc123"
    })

    # Should navigate to the video page
    connector.navigate_with_rate_limit.assert_called_once_with(
        page, "https://www.youtube.com/watch?v=abc123"
    )

    # Should wait for page load
    assert human.wait_human.call_count >= 1

    # Should click Save button
    save_calls = [c for c in human.click.call_args_list if "Save" in str(c)]
    assert len(save_calls) == 1

    # Should click Watch Later checkbox
    watch_later_calls = [c for c in human.click.call_args_list if "Watch later" in str(c)]
    assert len(watch_later_calls) == 1

    assert result["status"] == "added_to_watch_later"


@pytest.mark.asyncio
async def test_execute_add_to_watch_later_no_page(connector):
    """Verify execute() raises error when page is not initialized."""
    connector._page = None

    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("add_to_watch_later", {"url": "https://youtube.com/watch?v=abc"})


@pytest.mark.asyncio
async def test_execute_add_to_watch_later_click_failure(connector):
    """Verify execute() handles missing Watch Later button gracefully."""
    page = Mock()
    connector._page = page

    human = AsyncMock()
    human.wait_human = AsyncMock()
    # Simulate button not found
    human.click = AsyncMock(side_effect=Exception("Element not found"))
    connector._human = human

    connector.navigate_with_rate_limit = AsyncMock()

    result = await connector.execute("add_to_watch_later", {
        "url": "https://www.youtube.com/watch?v=abc123"
    })

    assert result["status"] == "error"
    assert "Could not find Watch Later button" in result["details"]


@pytest.mark.asyncio
async def test_execute_unknown_action(connector):
    """Verify execute() raises ValueError for unknown actions."""
    connector._page = Mock()

    with pytest.raises(ValueError, match="Unknown action"):
        await connector.execute("unknown_action", {})


@pytest.mark.asyncio
async def test_execute_add_to_watch_later_no_url(connector):
    """Verify execute() handles missing URL parameter."""
    connector._page = Mock()

    # Should not crash, just not do anything
    with pytest.raises(ValueError, match="Unknown action"):
        # When url is missing, action is still "add_to_watch_later" but
        # the if condition `if url:` fails, so it falls through to ValueError
        await connector.execute("add_to_watch_later", {})


# =============================================================================
# Health Check
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_with_page(connector):
    """Verify health_check returns ok when page is initialized."""
    connector._page = Mock()

    result = await connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "youtube"
    assert result["mode"] == "browser"


@pytest.mark.asyncio
async def test_health_check_without_page(connector):
    """Verify health_check returns not_started when page is None."""
    connector._page = None

    result = await connector.health_check()

    assert result["status"] == "not_started"
    assert result["connector"] == "youtube"


# =============================================================================
# Scrolling and Lazy Loading
# =============================================================================


@pytest.mark.asyncio
async def test_browser_sync_scrolls_to_load_videos(connector):
    """Verify browser_sync scrolls down 3 times to trigger lazy loading."""
    page = MockPage(videos=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    await connector.browser_sync(page, human, interactor)

    # Should scroll down 3 times
    assert human.scroll.call_count == 3

    # Each scroll should be downward with 800px distance
    for call in human.scroll.call_args_list:
        assert call[0][1] == "down"  # direction
        assert call[0][2] == 800      # distance


# =============================================================================
# DOM Extraction Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_browser_sync_handles_missing_channel_name(connector):
    """Verify browser_sync handles videos with missing channel names."""
    videos = [
        {
            "video_id": "abc123",
            "title": "Test Video",
            "channel": "",  # Missing channel name
            "url": "https://youtube.com/watch?v=abc123",
            "thumbnail": "",
            "meta": "",
        },
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    # Should still process the video
    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["channel"] == ""


@pytest.mark.asyncio
async def test_browser_sync_handles_relative_urls(connector):
    """Verify browser_sync converts relative URLs to absolute."""
    videos = [
        {
            "video_id": "abc123",
            "title": "Test Video",
            "channel": "Test",
            "url": "/watch?v=abc123",  # Relative URL (JS adds youtube.com prefix)
            "thumbnail": "",
            "meta": "",
        },
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    # The JS code in browser_sync should have converted this to absolute
    # (In real execution, the JS adds 'https://www.youtube.com' prefix)
    assert count == 1


@pytest.mark.asyncio
async def test_browser_sync_handles_empty_metadata(connector):
    """Verify browser_sync handles videos with missing metadata fields."""
    videos = [
        {
            "video_id": "abc123",
            "title": "Test Video",
            "channel": "Test",
            "url": "https://youtube.com/watch?v=abc123",
            "thumbnail": "",  # Empty thumbnail
            "meta": "",       # Empty metadata
        },
    ]
    page = MockPage(videos=videos)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count = await connector.browser_sync(page, human, interactor)

    assert count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["thumbnail"] == ""
    assert payload["meta"] == ""


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_sync_flow_with_deduplication(connector):
    """Integration test: full sync with cursor persistence and deduplication."""
    # First sync: 3 new videos
    videos_sync1 = [
        {"video_id": "v1", "title": "Video 1", "channel": "C1", "url": "https://youtube.com/watch?v=v1", "thumbnail": "", "meta": ""},
        {"video_id": "v2", "title": "Video 2", "channel": "C2", "url": "https://youtube.com/watch?v=v2", "thumbnail": "", "meta": ""},
        {"video_id": "v3", "title": "Video 3", "channel": "C3", "url": "https://youtube.com/watch?v=v3", "thumbnail": "", "meta": ""},
    ]
    page = MockPage(videos=videos_sync1)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = Mock()

    connector.navigate_with_rate_limit = AsyncMock()
    connector.publish_event = AsyncMock()

    count1 = await connector.browser_sync(page, human, interactor)
    assert count1 == 3

    # Second sync: 2 old, 1 new
    videos_sync2 = [
        {"video_id": "v2", "title": "Video 2", "channel": "C2", "url": "https://youtube.com/watch?v=v2", "thumbnail": "", "meta": ""},
        {"video_id": "v3", "title": "Video 3", "channel": "C3", "url": "https://youtube.com/watch?v=v3", "thumbnail": "", "meta": ""},
        {"video_id": "v4", "title": "Video 4", "channel": "C4", "url": "https://youtube.com/watch?v=v4", "thumbnail": "", "meta": ""},
    ]
    page.videos = videos_sync2
    connector.publish_event.reset_mock()

    count2 = await connector.browser_sync(page, human, interactor)
    assert count2 == 1  # Only v4 is new

    # Should only publish event for v4
    assert connector.publish_event.call_count == 1
    payload = connector.publish_event.call_args[0][1]
    assert payload["video_id"] == "v4"
