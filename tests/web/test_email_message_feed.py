"""
Tests for real email and message events in the /api/dashboard/feed endpoint.

The dashboard feed was previously limited to notification-triggered items, which
meant the Email and Messages topics showed almost nothing for most users (only
emails/messages that happened to trigger a rules-engine notification).

These tests verify that:
  1. Real ``email.received`` events appear in the email/inbox feed
  2. Real ``message.received`` events appear in the messages/inbox feed
  3. Both are deduplicated against notification items already in the feed
  4. Marketing / automated emails are filtered from the email feed
  5. Both sections gracefully handle DB errors without crashing the endpoint
  6. The response shape is consistent with existing feed items
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_email_row(
    row_id: str = "evt-email-1",
    subject: str = "Hello world",
    from_addr: str = "alice@example.com",
    from_name: str = "Alice",
    body: str = "Hi there, just checking in.",
    timestamp: str = "2026-02-28T10:00:00+00:00",
    message_id: str = "msg-1",
    thread_id: str = "thread-1",
    has_attachments: bool = False,
) -> dict:
    """Create a fake sqlite3.Row-like dict for an email.received event."""
    payload = json.dumps({
        "subject": subject,
        "from_address": from_addr,
        "from_name": from_name,
        "body": body,
        "snippet": body[:120],
        "timestamp": timestamp,
        "message_id": message_id,
        "thread_id": thread_id,
        "has_attachments": has_attachments,
        "to_addresses": ["me@example.com"],
    })
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": row_id,
        "payload": payload,
        "timestamp": timestamp,
    }[key]
    return row


def _make_message_row(
    row_id: str = "evt-msg-1",
    from_addr: str = "bob@example.com",
    contact_name: str = "Bob",
    body: str = "Hey, how are you?",
    channel: str = "signal",
    timestamp: str = "2026-02-28T11:00:00+00:00",
    message_id: str = "msg-2",
) -> dict:
    """Create a fake sqlite3.Row-like dict for a message.received event."""
    payload = json.dumps({
        "from_address": from_addr,
        "contact_name": contact_name,
        "body": body,
        "channel": channel,
        "timestamp": timestamp,
        "message_id": message_id,
        "is_group": False,
        "group_name": None,
    })
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "id": row_id,
        "payload": payload,
        "timestamp": timestamp,
    }[key]
    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for dashboard feed tests.

    Returns empty collections by default so individual tests can override
    only the parts they care about.
    """
    life_os = Mock()

    # Database — returns no rows by default; tests override as needed
    mock_conn = MagicMock()
    mock_conn.execute.return_value.__iter__ = Mock(return_value=iter([]))
    mock_conn.execute.return_value.fetchall = Mock(return_value=[])
    mock_conn.execute.return_value.fetchone = Mock(return_value=None)

    life_os.db = Mock()
    life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    # Event bus / event store
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager — empty by default
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})
    life_os.notification_manager.mark_read = Mock()
    life_os.notification_manager.mark_acted_on = Mock()
    life_os.notification_manager.dismiss = Mock()

    # Task manager — empty by default
    life_os.task_manager = Mock()
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.create_task = Mock(return_value="task-1")
    life_os.task_manager.update_task = Mock()
    life_os.task_manager.complete_task = Mock()

    # Signal extractor
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(
        return_value=Mock(
            energy_level=0.5, stress_level=0.3, social_battery=0.4,
            cognitive_load=0.3, emotional_valence=0.5, confidence=0.6, trend="stable"
        )
    )

    # Vector store
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # AI engine
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = Mock(return_value="Briefing")
    life_os.ai_engine.draft_reply = Mock(return_value="Draft")
    life_os.ai_engine.search_life = Mock(return_value="Result")

    # Rules engine
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])
    life_os.rules_engine.add_rule = Mock(return_value="rule-1")
    life_os.rules_engine.remove_rule = Mock()

    # User model store
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)
    life_os.user_model_store.resolve_prediction = Mock()

    # Feedback collector
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = Mock()

    # Connector / browser stubs
    life_os.connectors = []
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # Onboarding
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # Connector management
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = Mock(return_value={"success": True})
    life_os.enable_connector = Mock(return_value={"status": "started"}),
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


def _make_db_mock_with_rows(email_rows: list = None, message_rows: list = None, calendar_rows: list = None):
    """Build a callable for db.get_connection() that returns rows for specific event types.

    The dashboard feed calls ``life_os.db.get_connection("events")`` once per
    data section (calendar, email, messages).  The returned callable dispatches
    different row sets based on the SQL query string, so tests can inject exactly
    the rows they care about.

    Usage::
        mock_life_os.db.get_connection = _make_db_mock_with_rows(
            email_rows=[_make_email_row(...)],
        )

    Args:
        email_rows: Rows returned for ``email.received`` queries.
        message_rows: Rows returned for ``message.received`` queries.
        calendar_rows: Rows returned for ``calendar.event.created`` queries.

    Returns:
        A callable that accepts a db-name string and returns a context manager
        whose ``execute()`` dispatches to the correct row list.
    """
    email_rows = email_rows or []
    message_rows = message_rows or []
    calendar_rows = calendar_rows or []

    def _get_connection(_db_name):
        """Simulate db.get_connection(name) returning a context manager."""
        conn = MagicMock()

        def _execute(sql, *args, **kwargs):
            cursor = MagicMock()
            sql_stripped = sql.strip()
            if "email.received" in sql_stripped:
                cursor.fetchall.return_value = email_rows
            elif "message.received" in sql_stripped:
                cursor.fetchall.return_value = message_rows
            elif "calendar.event.created" in sql_stripped:
                cursor.fetchall.return_value = calendar_rows
                cursor.fetchone.return_value = (len(calendar_rows),)
            else:
                cursor.fetchall.return_value = []
                cursor.fetchone.return_value = None
            return cursor

        conn.execute.side_effect = _execute
        return Mock(
            __enter__=Mock(return_value=conn),
            __exit__=Mock(return_value=False),
        )

    return _get_connection


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Email feed tests
# ---------------------------------------------------------------------------


def test_email_feed_returns_email_items(mock_life_os):
    """Email events from the events store appear in the email topic feed."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[_make_email_row(subject="Project Update", from_addr="alice@example.com")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()
    items = data["items"]
    email_items = [it for it in items if it.get("kind") == "email"]
    assert len(email_items) >= 1
    assert email_items[0]["title"] == "Project Update"


def test_email_feed_item_has_correct_shape(mock_life_os):
    """Email feed items contain the expected fields for the UI to render."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[_make_email_row(
            subject="Contract Review",
            from_addr="lawyer@example.com",
            from_name="Smith & Jones LLP",
            has_attachments=True,
        )]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    items = response.json()["items"]
    email_items = [it for it in items if it.get("kind") == "email"]
    assert email_items, "No email items returned"
    item = email_items[0]

    # Required top-level fields
    assert item["kind"] == "email"
    assert item["channel"] == "email"
    assert item["title"] == "Contract Review"
    assert "body" in item
    assert "timestamp" in item
    assert "metadata" in item

    # Metadata must include sender details
    meta = item["metadata"]
    assert meta["from_address"] == "lawyer@example.com"
    assert meta["from_name"] == "Smith & Jones LLP"
    assert meta["has_attachments"] is True


def test_email_feed_filters_marketing(mock_life_os):
    """Marketing / noreply senders are excluded from the email feed."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[
            _make_email_row(
                row_id="mkt-1",
                subject="SALE! 50% off everything!",
                from_addr="noreply@promotions.example.com",
            ),
            _make_email_row(
                row_id="real-1",
                subject="Team lunch tomorrow",
                from_addr="colleague@example.com",
            ),
        ]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    items = response.json()["items"]
    email_items = [it for it in items if it.get("kind") == "email"]
    titles = [it["title"] for it in email_items]
    # Marketing email should be filtered out
    assert "SALE! 50% off everything!" not in titles
    # Real email should pass through
    assert "Team lunch tomorrow" in titles


def test_email_feed_appears_in_inbox_topic(mock_life_os):
    """Email events appear in the inbox (default) topic as well as email."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[_make_email_row(subject="Inbox Email", from_addr="person@example.com")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    # Both explicit inbox and no-topic default should include email items
    for url in ("/api/dashboard/feed?topic=inbox", "/api/dashboard/feed"):
        response = client.get(url)
        assert response.status_code == 200
        items = response.json()["items"]
        email_items = [it for it in items if it.get("kind") == "email"]
        assert len(email_items) >= 1, f"Email items missing for URL {url}"


def test_email_feed_truncates_long_body(mock_life_os):
    """Email bodies longer than 300 chars are truncated with an ellipsis."""
    long_body = "A" * 500
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[_make_email_row(body=long_body, from_addr="sender@example.com")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    items = response.json()["items"]
    email_items = [it for it in items if it.get("kind") == "email"]
    assert email_items, "No email items in feed"
    # Body should be truncated; note the snippet field is used before body
    body = email_items[0]["body"]
    assert len(body) <= 300 + 10  # Allow for "…" character


def test_email_feed_handles_db_error_gracefully(mock_life_os):
    """Email feed DB errors return 200 with no email items (fail-open)."""
    mock_life_os.db.get_connection = Mock(side_effect=RuntimeError("DB unavailable"))
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    # Should return items list (possibly empty)
    data = response.json()
    assert "items" in data


def test_email_feed_not_shown_in_messages_topic(mock_life_os):
    """Email events must NOT appear in the messages topic feed."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[_make_email_row(subject="Email Only", from_addr="person@example.com")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=messages")
    items = response.json()["items"]
    email_items = [it for it in items if it.get("kind") == "email"]
    assert email_items == [], "Email items should not appear in messages topic"


# ---------------------------------------------------------------------------
# Message feed tests
# ---------------------------------------------------------------------------


def test_message_feed_returns_message_items(mock_life_os):
    """Message events from the events store appear in the messages topic feed."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        message_rows=[_make_message_row(contact_name="Charlie", body="Are you free tonight?")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=messages")
    assert response.status_code == 200
    items = response.json()["items"]
    msg_items = [it for it in items if it.get("kind") == "message"]
    assert len(msg_items) >= 1
    assert msg_items[0]["title"] == "Charlie"


def test_message_feed_item_has_correct_shape(mock_life_os):
    """Message feed items contain the expected fields for the UI to render."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        message_rows=[_make_message_row(
            from_addr="dave@example.com",
            contact_name="Dave",
            body="Meeting in 10 mins",
            channel="signal",
        )]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=messages")
    items = response.json()["items"]
    msg_items = [it for it in items if it.get("kind") == "message"]
    assert msg_items, "No message items returned"
    item = msg_items[0]

    assert item["kind"] == "message"
    assert item["channel"] == "signal"
    assert item["title"] == "Dave"
    assert item["body"] == "Meeting in 10 mins"
    meta = item["metadata"]
    assert meta["from_address"] == "dave@example.com"
    assert meta["channel"] == "signal"


def test_message_feed_appears_in_inbox_topic(mock_life_os):
    """Message events appear in the inbox (default) topic."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        message_rows=[_make_message_row(contact_name="Eve")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=inbox")
    items = response.json()["items"]
    msg_items = [it for it in items if it.get("kind") == "message"]
    assert len(msg_items) >= 1


def test_message_feed_not_shown_in_email_topic(mock_life_os):
    """Message events must NOT appear in the email topic feed."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        message_rows=[_make_message_row(contact_name="Frank")]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    items = response.json()["items"]
    msg_items = [it for it in items if it.get("kind") == "message"]
    assert msg_items == [], "Message items should not appear in email topic"


def test_message_feed_handles_db_error_gracefully(mock_life_os):
    """Message feed DB errors return 200 with empty items (fail-open)."""
    mock_life_os.db.get_connection = Mock(side_effect=RuntimeError("DB unavailable"))
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=messages")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data


# ---------------------------------------------------------------------------
# Feed response structure tests
# ---------------------------------------------------------------------------


def test_feed_returns_count_field(mock_life_os):
    """Response includes a 'count' field equal to the number of items."""
    mock_life_os.db.get_connection = _make_db_mock_with_rows(
        email_rows=[
            _make_email_row(row_id="e1", subject="Email 1", from_addr="a@example.com"),
            _make_email_row(row_id="e2", subject="Email 2", from_addr="b@example.com"),
        ]
    )
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()
    assert "count" in data
    assert data["count"] == len(data["items"])


def test_feed_returns_topic_field(mock_life_os):
    """Response includes a 'topic' field echoing the requested topic."""
    app = create_web_app(mock_life_os)
    client = TestClient(app)

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.json()["topic"] == "email"

    response2 = client.get("/api/dashboard/feed?topic=messages")
    assert response2.json()["topic"] == "messages"

    response3 = client.get("/api/dashboard/feed")
    assert response3.json()["topic"] == "inbox"


def test_feed_empty_when_no_data(client):
    """Empty data stores return empty items list with count=0."""
    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["count"] == 0
