"""
Tests for the dashboard feed email/message event fallback.

The dashboard feed endpoint (/api/dashboard/feed) fetches email.received and
message.received events directly from events.db so that the Email and Messages
tabs always have content, even when no rules-engine notifications exist.

These tests verify:
- Email events appear in the feed with kind='email' when no notifications exist
- Existing notifications are not duplicated when email events are also fetched
- Email event items include correct metadata (from, to, subject)
- Malformed email payloads are skipped gracefully
- Messages topic falls back to message.received events
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_email_row(row_id, subject="Test Email", from_addr="alice@example.com",
                    to_addr="user@example.com", snippet="Hello world",
                    has_attachments=False, timestamp="2026-03-03T10:00:00Z"):
    """Create a fake events DB row for an email.received event."""
    payload = json.dumps({
        "subject": subject,
        "from_address": from_addr,
        "from_name": from_addr.split("@")[0].title(),
        "to_addresses": [to_addr],
        "snippet": snippet,
        "body": snippet,
        "has_attachments": has_attachments,
        "thread_id": f"thread-{row_id}",
        "message_id": f"msg-{row_id}",
        "timestamp": timestamp,
    })
    return {"id": row_id, "payload": payload, "timestamp": timestamp}


def _make_message_row(row_id, contact="Bob", body="Hey there",
                      channel="imessage", from_addr="bob@example.com",
                      timestamp="2026-03-03T11:00:00Z"):
    """Create a fake events DB row for a message.received event."""
    payload = json.dumps({
        "contact_name": contact,
        "from_address": from_addr,
        "body": body,
        "channel": channel,
        "is_group": False,
        "message_id": f"msg-{row_id}",
        "timestamp": timestamp,
    })
    return {"id": row_id, "payload": payload, "timestamp": timestamp}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for dashboard feed tests."""
    life_os = Mock()

    # Database mock — default returns empty results for all queries
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)
    mock_conn.execute.return_value = mock_cursor
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
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager — empty by default
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

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
    life_os.enable_connector = Mock(return_value={"status": "started"})
    life_os.disable_connector = Mock(return_value={"status": "stopped"})

    return life_os


def _setup_db_returning_rows(mock_life_os, rows):
    """Configure the mock DB to return the given rows from fetchall.

    Uses a MagicMock connection that supports context-manager protocol and
    returns `rows` for all execute().fetchall() calls, plus (0,) for
    fetchone() calls (used by calendar/insights count queries).
    """
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows
    mock_result.fetchone.return_value = (0,)
    mock_conn.execute.return_value = mock_result

    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Email event fallback tests
# ---------------------------------------------------------------------------


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_email_events_returned_when_no_notifications(mock_marketing, mock_life_os, client):
    """With 0 notifications and email events in DB, feed returns email items."""
    email_rows = [
        _make_email_row(f"evt-{i}", subject=f"Email {i}", from_addr=f"sender{i}@example.com")
        for i in range(5)
    ]
    _setup_db_returning_rows(mock_life_os, email_rows)

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(email_items) == 5
    for item in email_items:
        assert item["channel"] == "email"


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_email_events_metadata_includes_from_and_subject(mock_marketing, mock_life_os, client):
    """Email event items include correct metadata (from, to, subject)."""
    rows = [_make_email_row(
        "evt-1",
        subject="Important Meeting",
        from_addr="boss@company.com",
        to_addr="me@company.com",
        has_attachments=True,
    )]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(email_items) == 1
    item = email_items[0]
    assert item["title"] == "Important Meeting"
    assert item["metadata"]["from_address"] == "boss@company.com"
    assert "me@company.com" in item["metadata"]["to_addresses"]
    assert item["metadata"]["has_attachments"] is True


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_notifications_not_duplicated_by_email_events(mock_marketing, mock_life_os, client):
    """When email notifications exist, matching events are deduplicated."""
    # Set up 2 email notifications
    mock_life_os.notification_manager.get_pending = Mock(return_value=[
        {
            "id": "notif-1",
            "domain": "email",
            "title": "Email from Alice",
            "body": "Hello",
            "priority": "normal",
            "created_at": "2026-03-03T10:00:00Z",
            "source_event_id": "evt-1",
        },
        {
            "id": "notif-2",
            "domain": "email",
            "title": "Email from Bob",
            "body": "Hey",
            "priority": "normal",
            "created_at": "2026-03-03T09:00:00Z",
            "source_event_id": "evt-2",
        },
    ])

    # Email events in DB include the same events that triggered notifications
    email_rows = [
        _make_email_row("evt-1", subject="Email from Alice"),
        _make_email_row("evt-2", subject="Email from Bob"),
        _make_email_row("evt-3", subject="Email from Carol"),
    ]
    _setup_db_returning_rows(mock_life_os, email_rows)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()

    # evt-1 and evt-2 should be skipped since they already exist as notifications
    notification_items = [item for item in data["items"] if item["kind"] == "notification"]
    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(notification_items) == 2
    # evt-3 should appear as an email event (not duplicated)
    assert len(email_items) == 1
    assert email_items[0]["title"] == "Email from Carol"


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_malformed_email_payloads_skipped(mock_marketing, mock_life_os, client):
    """Malformed email payloads (invalid JSON) are skipped gracefully."""
    rows = [
        {"id": "evt-bad-1", "payload": "not valid json{{{", "timestamp": "2026-03-03T10:00:00Z"},
        _make_email_row("evt-good-1", subject="Valid Email"),
        {"id": "evt-bad-2", "payload": "", "timestamp": "2026-03-03T09:00:00Z"},
    ]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    # Only the valid row should appear
    assert len(email_items) == 1
    assert email_items[0]["title"] == "Valid Email"


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_email_body_truncated_to_300_chars(mock_marketing, mock_life_os, client):
    """Long email snippets are truncated to 300 characters."""
    long_body = "A" * 500
    rows = [_make_email_row("evt-1", snippet=long_body)]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(email_items) == 1
    # Body should be truncated: 297 chars + ellipsis
    assert len(email_items[0]["body"]) <= 300


# ---------------------------------------------------------------------------
# Message event fallback tests
# ---------------------------------------------------------------------------


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_message_events_returned_when_no_notifications(mock_marketing, mock_life_os, client):
    """With 0 message notifications and message events in DB, feed returns message items."""
    msg_rows = [
        _make_message_row(f"msg-{i}", contact=f"Contact {i}")
        for i in range(3)
    ]
    _setup_db_returning_rows(mock_life_os, msg_rows)

    response = client.get("/api/dashboard/feed?topic=messages")
    assert response.status_code == 200
    data = response.json()

    msg_items = [item for item in data["items"] if item["kind"] == "message"]
    assert len(msg_items) == 3


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_message_events_include_channel_metadata(mock_marketing, mock_life_os, client):
    """Message event items include correct channel and contact metadata."""
    rows = [_make_message_row(
        "msg-1",
        contact="Alice",
        body="Hey, want to grab lunch?",
        channel="signal",
        from_addr="alice@signal.org",
    )]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=messages")
    data = response.json()

    msg_items = [item for item in data["items"] if item["kind"] == "message"]
    assert len(msg_items) == 1
    item = msg_items[0]
    assert item["title"] == "Alice"
    assert item["body"] == "Hey, want to grab lunch?"
    assert item["metadata"]["channel"] == "signal"
    assert item["metadata"]["from_address"] == "alice@signal.org"


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_marketing_emails_filtered_out(mock_marketing, mock_life_os, client):
    """Marketing/noreply emails are filtered from the feed."""
    mock_marketing.return_value = True  # Everything is marketing

    rows = [
        _make_email_row("evt-1", subject="50% OFF SALE", from_addr="noreply@store.com"),
        _make_email_row("evt-2", subject="Your receipt", from_addr="no-reply@company.com"),
    ]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(email_items) == 0


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_email_feed_db_error_handled_gracefully(mock_marketing, mock_life_os, client):
    """If the events DB query raises, the endpoint still returns 200."""
    mock_life_os.db.get_connection = Mock(
        side_effect=RuntimeError("events DB unavailable")
    )

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    # Should report the failure in sections_failed
    assert any(s.get("section") in ("email", "calendar") for s in data.get("sections_failed", []))


@patch("services.signal_extractor.marketing_filter.is_marketing_or_noreply", return_value=False)
def test_email_no_subject_fallback(mock_marketing, mock_life_os, client):
    """Emails with no subject field display a fallback title."""
    payload = json.dumps({
        "from_address": "alice@example.com",
        "from_name": "Alice",
        "to_addresses": ["user@example.com"],
        "snippet": "Check this out",
        "timestamp": "2026-03-03T10:00:00Z",
    })
    rows = [{"id": "evt-no-subj", "payload": payload, "timestamp": "2026-03-03T10:00:00Z"}]
    _setup_db_returning_rows(mock_life_os, rows)

    response = client.get("/api/dashboard/feed?topic=email")
    data = response.json()

    email_items = [item for item in data["items"] if item["kind"] == "email"]
    assert len(email_items) == 1
    # Should have a fallback subject
    assert email_items[0]["title"] == "(no subject)"
