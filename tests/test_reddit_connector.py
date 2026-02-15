"""
Life OS — Reddit Connector Tests

Comprehensive test coverage for the RedditConnector class (247 LOC).

The Reddit connector is a browser-only integration that:
1. Scrapes old.reddit.com for subscribed subreddit posts
2. Monitors inbox for unread messages and replies
3. Handles deduplication via sync cursor
4. Prioritizes posts from configured subreddits
5. Extracts topics from post titles

Test categories:
- Authentication flow (login detection, session state)
- Post scraping (front page, deduplication, priority classification)
- Inbox scraping (unread messages, comment replies)
- Topic extraction (keyword filtering, stop words)
- Cursor management (seen IDs persistence, size capping)
- Error handling (failed extractions, malformed data)
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from connectors.browser.reddit import RedditConnector


class MockPage:
    """Mock Playwright Page object for old.reddit.com testing."""

    def __init__(self, logged_in=True, posts=None, messages=None):
        """
        Initialize mock page with configurable state.

        Args:
            logged_in: Whether the page shows logged-in state
            posts: List of post dicts to return from _extract_posts()
            messages: List of message dicts to return from _extract_messages()
        """
        self.logged_in = logged_in
        self.posts = posts or []
        self.messages = messages or []
        self._selectors_seen = []

    async def goto(self, url, wait_until=None):
        """Mock navigation to old.reddit.com."""
        pass

    async def query_selector(self, selector):
        """Mock selector query - returns element if logged in."""
        self._selectors_seen.append(selector)
        # old.reddit.com shows username link when logged in
        if "login-required" in selector or "/user/" in selector:
            return MagicMock() if self.logged_in else None
        return MagicMock()

    async def evaluate(self, script):
        """
        Mock JavaScript execution.

        Routes to appropriate data based on script content:
        - Scripts looking for '.thing.link' return posts
        - Scripts looking for '.thing.message.unread' return messages
        """
        if ".thing.link" in script:
            # This is _extract_posts()
            return self.posts
        elif ".thing.message.unread" in script:
            # This is _extract_messages()
            return self.messages
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
    engine.create_context = AsyncMock(return_value=MockContext("reddit"))
    engine.new_page = AsyncMock()
    engine.save_session = AsyncMock()
    engine.close = AsyncMock()
    return engine


@pytest.fixture
def mock_credential_vault():
    """Mock CredentialVault for testing credential storage."""
    vault = MagicMock()
    vault.get_credentials = MagicMock(return_value={"username": "testuser", "password": "testpass"})
    vault.store_credentials = MagicMock()
    return vault


@pytest.fixture
def mock_db():
    """Mock database for connector state persistence with cursor storage."""
    db = MagicMock()

    # Shared cursor storage across all connections (keyed by connector_id)
    cursor_store = {}

    # Mock connection context manager
    class MockConnection:
        def __init__(self, db_name, shared_store):
            self.db_name = db_name
            self.cursor_data = shared_store

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=()):
            """Mock execute that handles cursor reads and writes."""
            # Handle SELECT for get_sync_cursor()
            if "SELECT sync_cursor" in query:
                connector_id = params[0] if params else None
                if connector_id and connector_id in self.cursor_data:
                    # Return a row-like object with dict-like access
                    cursor_value = self.cursor_data[connector_id]

                    class Row:
                        def __getitem__(self, key):
                            if key == "sync_cursor":
                                return cursor_value
                            raise KeyError(key)

                    return MockFetchResult(Row())
                return MockFetchResult(None)

            # Handle UPDATE for set_sync_cursor()
            elif "UPDATE connector_state SET sync_cursor" in query:
                cursor_value, _, connector_id = params
                self.cursor_data[connector_id] = cursor_value

            # Handle other queries generically
            return MockFetchResult(None)

    class MockFetchResult:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    db.get_connection = MagicMock(side_effect=lambda name: MockConnection(name, cursor_store))

    return db


@pytest.fixture
def reddit_connector(mock_event_bus, mock_db, mock_browser_engine, mock_credential_vault):
    """Create a RedditConnector instance with mocked dependencies."""
    config = {
        "enabled": True,
        "mode": "browser",
        "sync_interval": 600,
        "use_old_reddit": True,
        "max_posts_per_sync": 25,
        "priority_subreddits": ["homelab", "selfhosted", "privacy"],
    }

    connector = RedditConnector(
        event_bus=mock_event_bus,
        db=mock_db,
        config=config,
        browser_engine=mock_browser_engine,
        credential_vault=mock_credential_vault,
    )

    # Initialize internal state that would normally be set by start()
    connector._page = None
    connector._human = AsyncMock()
    connector._interactor = AsyncMock()

    return connector


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_login_selectors_returns_correct_fields(reddit_connector):
    """Verify get_login_selectors returns old.reddit.com form fields."""
    selectors = reddit_connector.get_login_selectors()

    assert "username" in selectors
    assert "password" in selectors
    assert "submit" in selectors
    assert "#user_login" in selectors["username"]
    assert "name='user'" in selectors["username"]
    assert "name='passwd'" in selectors["password"]


@pytest.mark.asyncio
async def test_is_logged_in_when_authenticated(reddit_connector):
    """Verify login detection when user link is present in header."""
    page = MockPage(logged_in=True)

    result = await reddit_connector.is_logged_in(page)

    assert result is True


@pytest.mark.asyncio
async def test_is_logged_in_when_not_authenticated(reddit_connector):
    """Verify login detection when user link is missing."""
    page = MockPage(logged_in=False)

    result = await reddit_connector.is_logged_in(page)

    assert result is False


@pytest.mark.asyncio
async def test_is_logged_in_handles_exceptions(reddit_connector):
    """Verify is_logged_in returns False on page query errors."""
    page = MagicMock()
    page.query_selector = AsyncMock(side_effect=Exception("Page error"))

    result = await reddit_connector.is_logged_in(page)

    assert result is False


# ============================================================================
# Post Scraping Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_posts_parses_old_reddit_dom(reddit_connector):
    """Verify _extract_posts correctly parses old.reddit.com post structure."""
    mock_posts = [
        {
            "id": "t3_abc123",
            "title": "Great homelab setup guide",
            "url": "https://example.com/guide",
            "subreddit": "homelab",
            "author": "techuser",
            "score": 42,
            "permalink": "/r/homelab/comments/abc123/great_homelab_setup_guide/",
            "comments": 15,
            "is_self": False,
        },
        {
            "id": "t3_def456",
            "title": "Self-hosted password manager recommendations",
            "url": "",
            "subreddit": "selfhosted",
            "author": "privacyfan",
            "score": 128,
            "permalink": "/r/selfhosted/comments/def456/",
            "comments": 42,
            "is_self": True,
        },
    ]

    page = MockPage(posts=mock_posts)

    result = await reddit_connector._extract_posts(page)

    assert len(result) == 2
    assert result[0]["id"] == "t3_abc123"
    assert result[0]["title"] == "Great homelab setup guide"
    assert result[1]["subreddit"] == "selfhosted"
    assert result[1]["is_self"] is True


@pytest.mark.asyncio
async def test_extract_posts_handles_malformed_data(reddit_connector):
    """Verify _extract_posts gracefully handles incomplete post data."""
    # JavaScript extraction returns posts with missing fields
    page = MockPage(posts=[{"id": "t3_xyz"}, {"title": "No ID"}])

    result = await reddit_connector._extract_posts(page)

    # Should return whatever the JS gave us (fault tolerance in JS layer)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_browser_sync_publishes_new_posts(reddit_connector, mock_event_bus):
    """Verify browser_sync publishes events for new posts not in sync cursor."""
    mock_posts = [
        {
            "id": "t3_new1",
            "title": "New homelab post",
            "url": "https://example.com/1",
            "subreddit": "homelab",
            "author": "user1",
            "score": 10,
            "permalink": "/r/homelab/comments/new1/",
            "comments": 2,
            "is_self": False,
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    # Mock navigation and rate limiting
    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    count = await reddit_connector.browser_sync(page, human, interactor)

    # Should publish 1 post event
    assert count == 1
    assert mock_event_bus.publish.call_count == 1

    # Verify event structure
    call_args = mock_event_bus.publish.call_args
    assert call_args[0][0] == "content.reddit.new_post"
    payload = call_args[0][1]
    assert payload["post_id"] == "t3_new1"
    assert payload["title"] == "New homelab post"
    assert payload["subreddit"] == "homelab"


@pytest.mark.asyncio
async def test_browser_sync_skips_seen_posts(reddit_connector, mock_event_bus, mock_db):
    """Verify browser_sync deduplicates posts using sync cursor."""
    # Pre-populate sync cursor with seen post ID
    with mock_db.get_connection("state") as conn:
        conn.cursor_data["reddit"] = json.dumps(["t3_old"])

    mock_posts = [
        {
            "id": "t3_old",
            "title": "Old post",
            "subreddit": "homelab",
            "author": "user1",
            "score": 5,
            "permalink": "/r/homelab/comments/old/",
            "comments": 1,
            "is_self": False,
        },
        {
            "id": "t3_new",
            "title": "New post",
            "subreddit": "homelab",
            "author": "user2",
            "score": 10,
            "permalink": "/r/homelab/comments/new/",
            "comments": 2,
            "is_self": False,
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()

    count = await reddit_connector.browser_sync(page, human, interactor)

    # Should only publish the new post (1 post + 0 messages = 1)
    assert count == 1

    # Verify only new post was published
    call_args = mock_event_bus.publish.call_args_list
    post_events = [c for c in call_args if c[0][0] == "content.reddit.new_post"]
    assert len(post_events) == 1
    payload = post_events[0][0][1]
    assert payload["post_id"] == "t3_new"


@pytest.mark.asyncio
async def test_browser_sync_priority_subreddits_get_normal_priority(
    reddit_connector, mock_event_bus
):
    """Verify posts from priority_subreddits get 'normal' priority."""
    mock_posts = [
        {
            "id": "t3_priority",
            "title": "Homelab setup",
            "subreddit": "homelab",  # In priority_subreddits
            "author": "user1",
            "score": 10,
            "permalink": "/r/homelab/comments/priority/",
            "comments": 2,
            "is_self": False,
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    await reddit_connector.browser_sync(page, human, interactor)

    # Check priority in publish call
    call_args = mock_event_bus.publish.call_args
    assert call_args[1]["priority"] == "normal"


@pytest.mark.asyncio
async def test_browser_sync_non_priority_subreddits_get_low_priority(
    reddit_connector, mock_event_bus
):
    """Verify posts from non-priority subreddits get 'low' priority."""
    mock_posts = [
        {
            "id": "t3_low",
            "title": "Random post",
            "subreddit": "random",  # NOT in priority_subreddits
            "author": "user1",
            "score": 5,
            "permalink": "/r/random/comments/low/",
            "comments": 1,
            "is_self": True,
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    await reddit_connector.browser_sync(page, human, interactor)

    # Check priority in publish call
    call_args = mock_event_bus.publish.call_args
    assert call_args[1]["priority"] == "low"


@pytest.mark.asyncio
async def test_browser_sync_respects_max_posts_per_sync(
    reddit_connector, mock_event_bus
):
    """Verify browser_sync limits posts to max_posts_per_sync config."""
    # Create 50 new posts
    mock_posts = [
        {
            "id": f"t3_post{i}",
            "title": f"Post {i}",
            "subreddit": "homelab",
            "author": "user1",
            "score": i,
            "permalink": f"/r/homelab/comments/post{i}/",
            "comments": 0,
            "is_self": False,
        }
        for i in range(50)
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    count = await reddit_connector.browser_sync(page, human, interactor)

    # Should cap at 25 (max_posts_per_sync default)
    assert count == 25


@pytest.mark.asyncio
async def test_browser_sync_waits_for_human_delay(reddit_connector):
    """Verify browser_sync adds human-like delays during navigation."""
    page = MockPage(logged_in=True, posts=[], messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()

    await reddit_connector.browser_sync(page, human, interactor)

    # Should wait after navigating to front page and inbox
    # (old.reddit.com doesn't need scrolling like YouTube)
    assert human.wait_human.call_count >= 2


# ============================================================================
# Inbox Scraping Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_messages_parses_unread_inbox(reddit_connector):
    """Verify _extract_messages correctly parses unread message structure."""
    mock_messages = [
        {
            "id": "t4_msg1",
            "author": "friend",
            "subject": "Re: Your post",
            "body": "Great write-up!",
            "context": "/r/homelab/comments/abc123/context/3/",
        },
        {
            "id": "t4_msg2",
            "author": "moderator",
            "subject": "Mod notice",
            "body": "Please follow subreddit rules.",
            "context": "/r/selfhosted/comments/def456/context/3/",
        },
    ]

    page = MockPage(messages=mock_messages)

    result = await reddit_connector._extract_messages(page)

    assert len(result) == 2
    assert result[0]["id"] == "t4_msg1"
    assert result[0]["author"] == "friend"
    assert result[0]["subject"] == "Re: Your post"
    assert result[1]["body"] == "Please follow subreddit rules."


@pytest.mark.asyncio
async def test_browser_sync_publishes_inbox_messages(
    reddit_connector, mock_event_bus
):
    """Verify browser_sync publishes message.received events for inbox."""
    mock_messages = [
        {
            "id": "t4_msg1",
            "author": "sender",
            "subject": "Hello",
            "body": "Test message",
            "context": "/r/homelab/comments/abc/context/3/",
        },
    ]

    page = MockPage(logged_in=True, posts=[], messages=mock_messages)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    count = await reddit_connector.browser_sync(page, human, interactor)

    # Should publish 1 message event
    assert count == 1

    # Find message.received event
    message_calls = [
        c for c in mock_event_bus.publish.call_args_list
        if c[0][0] == "message.received"
    ]
    assert len(message_calls) == 1

    payload = message_calls[0][0][1]
    assert payload["channel"] == "reddit"
    assert payload["direction"] == "inbound"
    assert payload["from_contact"] == "sender"
    assert payload["subject"] == "Hello"
    assert payload["body"] == "Test message"
    assert payload["context_url"] == "/r/homelab/comments/abc/context/3/"


@pytest.mark.asyncio
async def test_browser_sync_navigates_to_inbox(reddit_connector):
    """Verify browser_sync navigates to /message/unread/ for inbox check."""
    page = MockPage(logged_in=True, posts=[], messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    await reddit_connector.browser_sync(page, human, interactor)

    # Should navigate to front page and inbox
    nav_calls = reddit_connector.navigate_with_rate_limit.call_args_list
    assert len(nav_calls) == 2
    assert "old.reddit.com/" in str(nav_calls[0])
    assert "message/unread" in str(nav_calls[1])


@pytest.mark.asyncio
async def test_browser_sync_rate_limits_between_phases(reddit_connector):
    """Verify browser_sync waits between front page and inbox scraping."""
    page = MockPage(logged_in=True, posts=[], messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    await reddit_connector.browser_sync(page, human, interactor)

    # Should call rate_limit_wait between phases
    reddit_connector.rate_limit_wait.assert_called_once()


# ============================================================================
# Topic Extraction Tests
# ============================================================================


def test_extract_topics_filters_stop_words(reddit_connector):
    """Verify _extract_topics removes common stop words."""
    title = "The best guide to setting up a homelab for privacy"

    topics = reddit_connector._extract_topics(title)

    # Should exclude: the, a, to, for
    # Should include: best, guide, setting, homelab, privacy
    assert "the" not in topics
    assert "to" not in topics
    assert "for" not in topics
    assert "best" in topics or "guide" in topics or "homelab" in topics


def test_extract_topics_filters_short_words(reddit_connector):
    """Verify _extract_topics removes words <= 3 characters."""
    title = "AI ML tips for NLP dev in Go"

    topics = reddit_connector._extract_topics(title)

    # Should exclude: AI (2), ML (2), for (3), NLP (3), in (2), Go (2)
    # Should include: tips (4), dev (3) - wait, dev is exactly 3, should be excluded
    # Actually, the filter is len(w) > 3, so 3-char words ARE excluded
    for topic in topics:
        assert len(topic) > 3


def test_extract_topics_limits_to_five_keywords(reddit_connector):
    """Verify _extract_topics returns maximum 5 keywords."""
    title = "comprehensive detailed extensive thorough complete guide tutorial walkthrough documentation reference"

    topics = reddit_connector._extract_topics(title)

    assert len(topics) <= 5


def test_extract_topics_lowercases_words(reddit_connector):
    """Verify _extract_topics converts to lowercase for consistency."""
    title = "HomeServer Docker Kubernetes Proxmox Setup"

    topics = reddit_connector._extract_topics(title)

    for topic in topics:
        assert topic.islower()


def test_extract_topics_handles_empty_title(reddit_connector):
    """Verify _extract_topics handles empty input gracefully."""
    topics = reddit_connector._extract_topics("")

    assert topics == []


def test_extract_topics_handles_all_stop_words(reddit_connector):
    """Verify _extract_topics handles titles with only stop words."""
    title = "the and for in of it is"

    topics = reddit_connector._extract_topics(title)

    # Should return empty list (all filtered out)
    assert topics == []


# ============================================================================
# Cursor Management Tests
# ============================================================================


def test_get_seen_ids_loads_from_cursor(reddit_connector, mock_db):
    """Verify _get_seen_ids deserializes sync cursor."""
    # Manually set cursor data in the mock DB's storage
    with mock_db.get_connection("state") as conn:
        conn.cursor_data["reddit"] = json.dumps(["t3_a", "t3_b", "t3_c"])

    seen = reddit_connector._get_seen_ids()

    assert seen == {"t3_a", "t3_b", "t3_c"}


def test_get_seen_ids_handles_missing_cursor(reddit_connector, mock_db):
    """Verify _get_seen_ids returns empty set when cursor is None."""
    # No cursor set
    seen = reddit_connector._get_seen_ids()

    assert seen == set()


def test_get_seen_ids_handles_invalid_json(reddit_connector, mock_db):
    """Verify _get_seen_ids handles malformed cursor gracefully."""
    mock_db.set_sync_cursor("reddit", "invalid json{{{")

    seen = reddit_connector._get_seen_ids()

    # Should return empty set on parse error
    assert seen == set()


def test_update_seen_ids_persists_new_ids(reddit_connector, mock_db):
    """Verify _update_seen_ids adds new post IDs to cursor."""
    # Start with some existing IDs
    with mock_db.get_connection("state") as conn:
        conn.cursor_data["reddit"] = json.dumps(["t3_old1", "t3_old2"])

    reddit_connector._update_seen_ids(["t3_new1", "t3_new2"])

    # Should merge old and new
    with mock_db.get_connection("state") as conn:
        cursor = conn.cursor_data.get("reddit")

    ids = set(json.loads(cursor))
    assert "t3_old1" in ids
    assert "t3_old2" in ids
    assert "t3_new1" in ids
    assert "t3_new2" in ids


def test_update_seen_ids_caps_at_1000_entries(reddit_connector, mock_db):
    """Verify _update_seen_ids limits cursor size to last 1000 IDs."""
    # Start with 999 existing IDs
    existing = [f"t3_old{i}" for i in range(999)]
    with mock_db.get_connection("state") as conn:
        conn.cursor_data["reddit"] = json.dumps(existing)

    # Add 10 new IDs (total would be 1009)
    new_ids = [f"t3_new{i}" for i in range(10)]
    reddit_connector._update_seen_ids(new_ids)

    # Should cap at 1000
    with mock_db.get_connection("state") as conn:
        cursor = conn.cursor_data.get("reddit")

    ids = json.loads(cursor)
    assert len(ids) == 1000

    # Should keep the most recent (last 1000)
    # New IDs should all be present
    for new_id in new_ids:
        assert new_id in ids


def test_update_seen_ids_handles_empty_list(reddit_connector, mock_db):
    """Verify _update_seen_ids handles empty new_ids list."""
    # Pre-populate with existing data
    with mock_db.get_connection("state") as conn:
        conn.cursor_data["reddit"] = json.dumps(["t3_existing"])

    reddit_connector._update_seen_ids([])

    # Should keep existing IDs
    with mock_db.get_connection("state") as conn:
        cursor = conn.cursor_data.get("reddit")

    ids = set(json.loads(cursor))
    assert "t3_existing" in ids


# ============================================================================
# Execute Action Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_raises_for_any_action(reddit_connector):
    """Verify execute() raises ValueError for all actions (read-only connector)."""
    with pytest.raises(ValueError, match="read-only"):
        await reddit_connector.execute("send_message", {"content": "test"})

    with pytest.raises(ValueError, match="read-only"):
        await reddit_connector.execute("unknown_action", {})


# ============================================================================
# Health Check Tests
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_when_page_exists_and_logged_in(reddit_connector):
    """Verify health_check returns 'ok' when page is active and logged in."""
    page = MockPage(logged_in=True)
    reddit_connector._page = page

    result = await reddit_connector.health_check()

    assert result["status"] == "ok"
    assert result["connector"] == "reddit"
    assert result["mode"] == "browser"


@pytest.mark.asyncio
async def test_health_check_when_page_exists_but_not_logged_in(reddit_connector):
    """Verify health_check returns 'session_expired' when logged out."""
    page = MockPage(logged_in=False)
    reddit_connector._page = page

    result = await reddit_connector.health_check()

    assert result["status"] == "session_expired"
    assert result["connector"] == "reddit"


@pytest.mark.asyncio
async def test_health_check_when_page_not_started(reddit_connector):
    """Verify health_check returns 'not_started' when page is None."""
    reddit_connector._page = None

    result = await reddit_connector.health_check()

    assert result["status"] == "not_started"
    assert result["connector"] == "reddit"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_sync_cycle_posts_and_messages(reddit_connector, mock_event_bus):
    """End-to-end test: sync both posts and messages in one cycle."""
    mock_posts = [
        {
            "id": "t3_post1",
            "title": "Homelab tips",
            "subreddit": "homelab",
            "author": "techuser",
            "score": 50,
            "permalink": "/r/homelab/comments/post1/",
            "comments": 10,
            "is_self": False,
        },
    ]

    mock_messages = [
        {
            "id": "t4_msg1",
            "author": "friend",
            "subject": "Question",
            "body": "Can you help?",
            "context": "/context/",
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=mock_messages)
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    count = await reddit_connector.browser_sync(page, human, interactor)

    # Should publish 2 events (1 post + 1 message)
    assert count == 2
    assert mock_event_bus.publish.call_count == 2

    # Verify both event types were published
    event_types = [call[0][0] for call in mock_event_bus.publish.call_args_list]
    assert "content.reddit.new_post" in event_types
    assert "message.received" in event_types


@pytest.mark.asyncio
async def test_metadata_includes_topics_and_subreddit(reddit_connector, mock_event_bus):
    """Verify published events include metadata with topics and subreddit."""
    mock_posts = [
        {
            "id": "t3_test",
            "title": "Docker Kubernetes Proxmox homelab setup guide",
            "subreddit": "homelab",
            "author": "user",
            "score": 10,
            "permalink": "/r/homelab/comments/test/",
            "comments": 2,
            "is_self": False,
        },
    ]

    page = MockPage(logged_in=True, posts=mock_posts, messages=[])
    human = AsyncMock()
    human.wait_human = AsyncMock()
    human.scroll = AsyncMock()
    interactor = AsyncMock()

    reddit_connector.navigate_with_rate_limit = AsyncMock()
    reddit_connector.rate_limit_wait = AsyncMock()
    reddit_connector._human = human

    await reddit_connector.browser_sync(page, human, interactor)

    # Check metadata
    call_args = mock_event_bus.publish.call_args
    metadata = call_args[1]["metadata"]
    assert metadata["domain"] == "media"
    assert metadata["subreddit"] == "homelab"
    assert "topics" in metadata
    assert isinstance(metadata["topics"], list)


@pytest.mark.asyncio
async def test_connector_constants(reddit_connector):
    """Verify RedditConnector class constants are correctly set."""
    assert RedditConnector.CONNECTOR_ID == "reddit"
    assert RedditConnector.DISPLAY_NAME == "Reddit"
    assert RedditConnector.SITE_ID == "reddit"
    assert RedditConnector.LOGIN_URL == "https://old.reddit.com/login"
    assert RedditConnector.SYNC_INTERVAL_SECONDS == 600
    assert RedditConnector.MIN_REQUEST_INTERVAL == 3.0
    assert RedditConnector.MAX_PAGES_PER_SYNC == 5
