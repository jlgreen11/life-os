"""
Tests for dashboard feed and badge notification filtering by domain.

Validates that:
1. /api/dashboard/feed?topic=email includes only notifications with domain="email"
2. /api/dashboard/feed?topic=messages includes only notifications with domain="message"
3. /api/dashboard/badges returns correct email_count and msg_count
4. Notifications with unrelated domains (e.g. "prediction") are excluded from email/messages topics
5. Feed items have the correct `channel` field based on domain
6. Feed item metadata includes source_event_id for deduplication
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


def _make_notification(id, domain, title="Test", source_event_id=None):
    """Helper to create a notification dict matching the notifications table schema."""
    return {
        "id": id,
        "title": title,
        "body": f"Body for {id}",
        "priority": "normal",
        "source_event_id": source_event_id or f"evt-{id}",
        "domain": domain,
        "status": "pending",
        "created_at": "2026-02-15T12:00:00Z",
    }


@pytest.fixture()
def mock_life_os():
    """Create a mock LifeOS instance with all services returning valid data."""
    life_os = Mock()

    # --- Database ---
    life_os.db = Mock()
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[])))
    life_os.db.get_connection = Mock(
        return_value=Mock(__enter__=Mock(return_value=mock_conn), __exit__=Mock(return_value=False))
    )
    life_os.db.get_database_health = Mock(return_value={
        "events": {"status": "ok", "errors": [], "path": "/tmp/events.db", "size_bytes": 1024},
        "entities": {"status": "ok", "errors": [], "path": "/tmp/entities.db", "size_bytes": 1024},
        "state": {"status": "ok", "errors": [], "path": "/tmp/state.db", "size_bytes": 1024},
        "user_model": {"status": "ok", "errors": [], "path": "/tmp/user_model.db", "size_bytes": 1024},
        "preferences": {"status": "ok", "errors": [], "path": "/tmp/preferences.db", "size_bytes": 1024},
    })

    # --- Event bus ---
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = AsyncMock()

    # --- Event store ---
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])

    # --- Vector store ---
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.vector_store.search = Mock(return_value=[])

    # --- Signal extractor ---
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable",
    ))

    # --- Notification manager ---
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0, "delivered": 0})
    life_os.notification_manager.get_pending = Mock(return_value=[])

    # --- Task manager ---
    life_os.task_manager = Mock()
    life_os.task_manager.get_tasks = Mock(return_value=[])
    life_os.task_manager.get_pending_tasks = Mock(return_value=[])

    # --- Feedback collector ---
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})

    # --- AI engine ---
    life_os.ai_engine = Mock()
    life_os.ai_engine.generate_briefing = AsyncMock(return_value="")
    life_os.ai_engine.search_life = AsyncMock(return_value="")

    # --- Rules engine ---
    life_os.rules_engine = Mock()
    life_os.rules_engine.get_all_rules = Mock(return_value=[])

    # --- User model store ---
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_semantic_facts = Mock(return_value=[])
    life_os.user_model_store.get_signal_profile = Mock(return_value=None)

    # --- Connectors ---
    life_os.connectors = []

    # --- Browser orchestrator ---
    life_os.browser_orchestrator = Mock()
    life_os.browser_orchestrator.get_status = Mock(return_value={"active": False})
    life_os.browser_orchestrator.get_vault_sites = Mock(return_value=[])

    # --- Onboarding ---
    life_os.onboarding = Mock()
    life_os.onboarding.get_answers = Mock(return_value={})
    life_os.onboarding.submit_answer = Mock()
    life_os.onboarding.finalize = Mock(return_value={})

    # --- Connector management ---
    life_os.get_connector_status = Mock(return_value={"enabled": False})
    life_os.get_connector_config = Mock(return_value={})
    life_os.save_connector_config = Mock()
    life_os.test_connector = AsyncMock(return_value={"success": True})
    life_os.enable_connector = AsyncMock(return_value={"status": "started"})
    life_os.disable_connector = AsyncMock(return_value={"status": "stopped"})

    return life_os


@pytest.fixture()
def client(mock_life_os):
    """Create a test client for the FastAPI app."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# --- Sample notification sets used across tests ---

MIXED_NOTIFICATIONS = [
    _make_notification("n1", "email", "New email from Alice", "evt-email-1"),
    _make_notification("n2", "email", "New email from Bob", "evt-email-2"),
    _make_notification("n3", "message", "Signal message from Carol", "evt-msg-1"),
    _make_notification("n4", "prediction", "Upcoming meeting prediction"),
    _make_notification("n5", "system", "System alert"),
    _make_notification("n6", "calendar", "Calendar reminder"),
]


# ---------------------------------------------------------------------------
# Test: email topic includes only email-domain notifications
# ---------------------------------------------------------------------------

def test_email_topic_includes_email_notifications(client, mock_life_os):
    """GET /api/dashboard/feed?topic=email includes notifications with domain='email'."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    notif_items = [i for i in data["items"] if i["kind"] == "notification"]
    assert len(notif_items) == 2
    assert {i["id"] for i in notif_items} == {"n1", "n2"}


# ---------------------------------------------------------------------------
# Test: messages topic includes only message-domain notifications
# ---------------------------------------------------------------------------

def test_messages_topic_includes_message_notifications(client, mock_life_os):
    """GET /api/dashboard/feed?topic=messages includes notifications with domain='message'."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/feed?topic=messages")
    assert response.status_code == 200
    data = response.json()

    notif_items = [i for i in data["items"] if i["kind"] == "notification"]
    assert len(notif_items) == 1
    assert notif_items[0]["id"] == "n3"


# ---------------------------------------------------------------------------
# Test: prediction/system/calendar notifications excluded from email topic
# ---------------------------------------------------------------------------

def test_email_topic_excludes_non_email_notifications(client, mock_life_os):
    """Notifications with domain='prediction', 'system', or 'calendar' are excluded
    from the email topic."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    notif_items = [i for i in data["items"] if i["kind"] == "notification"]
    notif_ids = {i["id"] for i in notif_items}
    # n4 (prediction), n5 (system), n6 (calendar) should not appear
    assert "n4" not in notif_ids
    assert "n5" not in notif_ids
    assert "n6" not in notif_ids


# ---------------------------------------------------------------------------
# Test: badges endpoint returns correct counts
# ---------------------------------------------------------------------------

def test_badges_returns_correct_email_and_msg_counts(client, mock_life_os):
    """GET /api/dashboard/badges returns accurate email_count and msg_count
    based on the domain column."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/badges")
    assert response.status_code == 200
    data = response.json()

    badges = data["badges"]
    assert badges["email"] == 2     # n1, n2 have domain="email"
    assert badges["messages"] == 1  # n3 has domain="message"


# ---------------------------------------------------------------------------
# Test: feed items have correct channel field
# ---------------------------------------------------------------------------

def test_feed_items_have_correct_channel(client, mock_life_os):
    """Notification feed items have channel='email' for email domain,
    'message' for message domain, and 'system' for other domains."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/feed")
    assert response.status_code == 200
    data = response.json()

    notif_items = {i["id"]: i for i in data["items"] if i["kind"] == "notification"}

    assert notif_items["n1"]["channel"] == "email"
    assert notif_items["n2"]["channel"] == "email"
    assert notif_items["n3"]["channel"] == "message"
    assert notif_items["n4"]["channel"] == "system"   # prediction -> system
    assert notif_items["n5"]["channel"] == "system"
    assert notif_items["n6"]["channel"] == "system"   # calendar -> system


# ---------------------------------------------------------------------------
# Test: feed item metadata includes source_event_id for deduplication
# ---------------------------------------------------------------------------

def test_feed_item_metadata_includes_source_event_id(client, mock_life_os):
    """Notification feed items include metadata.source_event_id so the
    deduplication logic on email events can detect overlapping items."""
    mock_life_os.notification_manager.get_pending.return_value = [
        _make_notification("n1", "email", "Email notification", "evt-abc-123"),
    ]

    response = client.get("/api/dashboard/feed?topic=email")
    assert response.status_code == 200
    data = response.json()

    notif_items = [i for i in data["items"] if i["kind"] == "notification"]
    assert len(notif_items) == 1
    assert notif_items[0]["metadata"]["source_event_id"] == "evt-abc-123"


# ---------------------------------------------------------------------------
# Test: inbox topic includes all notification domains
# ---------------------------------------------------------------------------

def test_inbox_topic_includes_all_notifications(client, mock_life_os):
    """GET /api/dashboard/feed?topic=inbox includes notifications from all domains."""
    mock_life_os.notification_manager.get_pending.return_value = MIXED_NOTIFICATIONS

    response = client.get("/api/dashboard/feed?topic=inbox")
    assert response.status_code == 200
    data = response.json()

    notif_items = [i for i in data["items"] if i["kind"] == "notification"]
    # All 6 notifications should be present (inbox doesn't filter by domain)
    assert len(notif_items) == 6
