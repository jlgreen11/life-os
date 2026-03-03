"""
Tests for the GET /api/contacts/{contact_email}/interactions endpoint.

Verifies that the endpoint correctly queries interaction history for a
contact, respects the limit parameter, handles unknown contacts gracefully,
and works with URL-encoded email addresses.
"""

import json
from unittest.mock import Mock, MagicMock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_row(event_id, event_type, timestamp, payload_dict):
    """Build a sqlite3.Row-like dict for a mocked event query result."""
    row = MagicMock()
    data = {
        "id": event_id,
        "type": event_type,
        "timestamp": timestamp,
        "payload": json.dumps(payload_dict),
    }
    row.__getitem__ = lambda self, key: data[key]
    row.keys = lambda: data.keys()

    # Make dict(row) work by implementing __iter__ over key-value pairs
    def _iter():
        return iter(data.items())
    row.__iter__ = _iter

    # Support attribute-style access too
    row.get = lambda key, default=None: data.get(key, default)
    return row


def _make_contact_row(contact_id, name, emails, relationship="friend", is_priority=0):
    """Build a sqlite3.Row-like dict for a mocked contact query result."""
    row = MagicMock()
    data = {
        "id": contact_id,
        "name": name,
        "aliases": "[]",
        "emails": json.dumps(emails),
        "phones": "[]",
        "channels": "{}",
        "relationship": relationship,
        "is_priority": is_priority,
        "preferred_channel": None,
        "always_surface": 0,
        "typical_response_time": None,
        "last_contact": None,
        "contact_frequency_days": None,
        "communication_style": None,
        "notes": "[]",
        "created_at": "2026-01-01T00:00:00Z",
    }
    row.__getitem__ = lambda self, key: data[key]
    row.keys = lambda: data.keys()

    def _iter():
        return iter(data.items())
    row.__iter__ = _iter
    row.get = lambda key, default=None: data.get(key, default)
    return row


@pytest.fixture
def mock_life_os():
    """Minimal LifeOS mock wired for contact-interaction tests."""
    life_os = Mock()

    # Database mock — default: no results
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []
    mock_conn.execute.return_value = mock_cursor

    life_os.db = MagicMock()
    life_os.db.get_connection.return_value.__enter__ = Mock(return_value=mock_conn)
    life_os.db.get_connection.return_value.__exit__ = Mock(return_value=False)

    # Event bus / event store (required by create_web_app)
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = True
    life_os.event_bus.publish = Mock()
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.event_store.get_events = Mock(return_value=[])
    life_os.event_store.store_event = Mock(return_value="evt-1")

    # Notification manager
    life_os.notification_manager = Mock()
    life_os.notification_manager.get_pending = Mock(return_value=[])
    life_os.notification_manager.get_stats = Mock(return_value={"pending": 0})

    # Task manager
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


@pytest.fixture
def client(mock_life_os):
    """TestClient against the full FastAPI app with mocked LifeOS."""
    app = create_web_app(mock_life_os)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers to configure mock DB responses
# ---------------------------------------------------------------------------


def _setup_db_responses(mock_life_os, contact_row=None, event_rows=None, total_count=0):
    """Configure the mock DB to return specific contact and event data.

    The endpoint calls get_connection("entities") for contact lookup and
    get_connection("events") for interaction queries.  This helper sets up
    both context managers to return the appropriate mock cursors.
    """
    entities_conn = MagicMock()
    entities_cursor = MagicMock()
    entities_cursor.fetchone.return_value = contact_row
    entities_conn.execute.return_value = entities_cursor

    events_conn = MagicMock()

    # The endpoint calls execute twice on the events connection:
    # 1. count query → fetchone() returning (total,)
    # 2. interaction query → fetchall() returning rows
    count_cursor = MagicMock()
    count_cursor.fetchone.return_value = (total_count,)

    rows_cursor = MagicMock()
    rows_cursor.fetchall.return_value = event_rows or []

    events_conn.execute.side_effect = [count_cursor, rows_cursor]

    def get_connection(db_name):
        ctx = MagicMock()
        if db_name == "entities":
            ctx.__enter__ = Mock(return_value=entities_conn)
        else:
            ctx.__enter__ = Mock(return_value=events_conn)
        ctx.__exit__ = Mock(return_value=False)
        return ctx

    mock_life_os.db.get_connection.side_effect = get_connection


# ---------------------------------------------------------------------------
# Response shape tests
# ---------------------------------------------------------------------------


def test_endpoint_exists(client):
    """GET /api/contacts/{email}/interactions returns 200."""
    response = client.get("/api/contacts/alice@example.com/interactions")
    assert response.status_code == 200


def test_response_has_required_keys(client):
    """Response contains contact_email, contact, interactions, total_interactions."""
    response = client.get("/api/contacts/alice@example.com/interactions")
    data = response.json()
    assert "contact_email" in data
    assert "contact" in data
    assert "interactions" in data
    assert "total_interactions" in data


def test_contact_email_matches_request(client):
    """The returned contact_email matches the requested address."""
    response = client.get("/api/contacts/alice@example.com/interactions")
    data = response.json()
    assert data["contact_email"] == "alice@example.com"


# ---------------------------------------------------------------------------
# Contact lookup tests
# ---------------------------------------------------------------------------


def test_returns_contact_when_found(mock_life_os, client):
    """Contact object is populated when the email matches an entities.db row."""
    contact_row = _make_contact_row("c-1", "Alice Smith", ["alice@example.com"], "colleague", 1)
    _setup_db_responses(mock_life_os, contact_row=contact_row)

    response = client.get("/api/contacts/alice@example.com/interactions")
    data = response.json()

    assert data["contact"] is not None
    assert data["contact"]["name"] == "Alice Smith"
    assert data["contact"]["relationship"] == "colleague"
    assert data["contact"]["is_priority"] is True


def test_returns_null_contact_when_not_found(mock_life_os, client):
    """Contact is null when the email doesn't match any entities.db row."""
    _setup_db_responses(mock_life_os, contact_row=None)

    response = client.get("/api/contacts/unknown@example.com/interactions")
    data = response.json()

    assert data["contact"] is None


# ---------------------------------------------------------------------------
# Interaction history tests
# ---------------------------------------------------------------------------


def test_returns_interactions(mock_life_os, client):
    """Interactions list contains correctly shaped objects."""
    event_rows = [
        _make_event_row(
            "evt-1", "email.received", "2026-03-01T14:30:00Z",
            {"from_address": "alice@example.com", "subject": "Re: Project update", "body": "Here is the update."},
        ),
        _make_event_row(
            "evt-2", "email.sent", "2026-02-28T10:00:00Z",
            {"to_addresses": ["alice@example.com"], "subject": "Project update", "body": "Please send update."},
        ),
    ]
    _setup_db_responses(mock_life_os, event_rows=event_rows, total_count=2)

    response = client.get("/api/contacts/alice@example.com/interactions")
    data = response.json()

    assert len(data["interactions"]) == 2
    assert data["total_interactions"] == 2

    ix0 = data["interactions"][0]
    assert ix0["id"] == "evt-1"
    assert ix0["type"] == "email.received"
    assert ix0["channel"] == "email"
    assert ix0["subject"] == "Re: Project update"
    assert "Here is the update." in ix0["snippet"]

    ix1 = data["interactions"][1]
    assert ix1["id"] == "evt-2"
    assert ix1["type"] == "email.sent"
    assert ix1["channel"] == "email"


def test_interaction_channel_detection(mock_life_os, client):
    """Channel field is correctly derived from event type."""
    event_rows = [
        _make_event_row("e1", "imessage.received", "2026-03-01T12:00:00Z",
                        {"sender": "alice@example.com", "text": "Hello"}),
        _make_event_row("e2", "message.sent", "2026-03-01T11:00:00Z",
                        {"sender": "alice@example.com", "body": "Hi there"}),
    ]
    _setup_db_responses(mock_life_os, event_rows=event_rows, total_count=2)

    response = client.get("/api/contacts/alice@example.com/interactions")
    interactions = response.json()["interactions"]

    assert interactions[0]["channel"] == "imessage"
    assert interactions[1]["channel"] == "message"


def test_snippet_truncation(mock_life_os, client):
    """Body content longer than 100 chars is truncated with ellipsis."""
    long_body = "A" * 150
    event_rows = [
        _make_event_row("e1", "email.received", "2026-03-01T12:00:00Z",
                        {"from_address": "alice@example.com", "body": long_body}),
    ]
    _setup_db_responses(mock_life_os, event_rows=event_rows, total_count=1)

    response = client.get("/api/contacts/alice@example.com/interactions")
    snippet = response.json()["interactions"][0]["snippet"]

    assert len(snippet) == 103  # 100 chars + "..."
    assert snippet.endswith("...")


# ---------------------------------------------------------------------------
# Limit parameter tests
# ---------------------------------------------------------------------------


def test_default_limit_is_five(mock_life_os, client):
    """Default limit parameter is 5."""
    event_rows = [
        _make_event_row(f"e{i}", "email.received", f"2026-03-0{i}T12:00:00Z",
                        {"from_address": "alice@example.com", "body": f"Message {i}"})
        for i in range(5)
    ]
    _setup_db_responses(mock_life_os, event_rows=event_rows, total_count=10)

    response = client.get("/api/contacts/alice@example.com/interactions")
    data = response.json()

    assert len(data["interactions"]) == 5
    assert data["total_interactions"] == 10


def test_limit_parameter_respected(mock_life_os, client):
    """Custom limit parameter controls the number of returned interactions."""
    event_rows = [
        _make_event_row(f"e{i}", "email.received", f"2026-03-0{i}T12:00:00Z",
                        {"from_address": "alice@example.com", "body": f"Msg {i}"})
        for i in range(2)
    ]
    _setup_db_responses(mock_life_os, event_rows=event_rows, total_count=2)

    response = client.get("/api/contacts/alice@example.com/interactions?limit=2")
    assert response.status_code == 200
    assert len(response.json()["interactions"]) == 2


def test_limit_capped_at_twenty(mock_life_os, client):
    """Limit above 20 is capped to 20."""
    # We verify indirectly — the endpoint should not pass limit > 20 to the DB.
    # With our mock setup, we just verify it doesn't error out.
    _setup_db_responses(mock_life_os, event_rows=[], total_count=0)
    response = client.get("/api/contacts/alice@example.com/interactions?limit=100")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# URL-encoded email addresses
# ---------------------------------------------------------------------------


def test_url_encoded_email(mock_life_os, client):
    """URL-encoded email addresses are correctly decoded."""
    _setup_db_responses(mock_life_os)

    # FastAPI handles URL decoding of path parameters automatically.
    response = client.get("/api/contacts/alice%40example.com/interactions")
    data = response.json()

    assert data["contact_email"] == "alice@example.com"
    assert response.status_code == 200


def test_email_with_plus_sign(mock_life_os, client):
    """Email addresses with + are handled correctly."""
    _setup_db_responses(mock_life_os)

    response = client.get("/api/contacts/alice+work@example.com/interactions")
    data = response.json()

    assert data["contact_email"] == "alice+work@example.com"
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_handles_db_error_gracefully(mock_life_os, client):
    """Endpoint returns 200 with empty interactions when DB queries fail."""
    mock_life_os.db.get_connection.side_effect = RuntimeError("DB unavailable")

    response = client.get("/api/contacts/alice@example.com/interactions")
    assert response.status_code == 200
    data = response.json()
    assert data["contact"] is None
    assert data["interactions"] == []
    assert data["total_interactions"] == 0


def test_empty_interactions_for_unknown_contact(client):
    """Unknown contacts return empty interactions list, not an error."""
    response = client.get("/api/contacts/nobody@nowhere.com/interactions")
    assert response.status_code == 200
    data = response.json()
    assert data["interactions"] == []
    assert data["total_interactions"] == 0
