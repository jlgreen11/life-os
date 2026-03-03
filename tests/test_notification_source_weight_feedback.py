"""
Tests for notification → source weight feedback integration.

Verifies that acting on or dismissing a notification correctly updates
the source weight learning system (record_engagement / record_dismissal),
enabling the AI drift to learn from the primary user interaction surface.
"""

import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

from services.insight_engine.source_weights import SourceWeightManager
from web.app import create_web_app


@pytest.fixture()
def source_weight_manager(db):
    """A SourceWeightManager with default weights seeded."""
    mgr = SourceWeightManager(db)
    mgr.seed_defaults()
    return mgr


@pytest.fixture()
def life_os_with_weights(db, notification_manager, source_weight_manager):
    """A mock LifeOS with real DB, notification manager, and source weight manager."""
    life_os = Mock()
    life_os.db = db
    life_os.notification_manager = notification_manager
    life_os.source_weight_manager = source_weight_manager

    # Minimal mocks for services needed by route registration but not under test
    life_os.event_bus = Mock()
    life_os.event_bus.is_connected = False
    life_os.event_store = Mock()
    life_os.event_store.get_event_count = Mock(return_value=0)
    life_os.vector_store = Mock()
    life_os.vector_store.get_stats = Mock(return_value={"total": 0, "dimensions": 384})
    life_os.signal_extractor = Mock()
    life_os.signal_extractor.get_user_summary = Mock(return_value={"facts": []})
    life_os.signal_extractor.get_current_mood = Mock(return_value=Mock(
        energy_level=0.7, stress_level=0.3, social_battery=0.8,
        cognitive_load=0.4, emotional_valence=0.6, confidence=0.75, trend="stable"
    ))
    life_os.feedback_collector = Mock()
    life_os.feedback_collector.get_feedback_summary = Mock(return_value={"total": 0})
    life_os.feedback_collector.process_explicit_feedback = AsyncMock()
    life_os.ai_engine = Mock()
    life_os.ai_engine.process_command = AsyncMock(return_value={"response": ""})
    life_os.task_manager = Mock()
    life_os.task_manager.get_task_stats = Mock(return_value={})
    life_os.user_model_store = Mock()
    life_os.user_model_store.get_profile_completeness = Mock(return_value={"overall": 0.5})
    life_os.user_model_store.get_all_facts = Mock(return_value=[])
    life_os.prediction_engine = Mock()
    life_os.prediction_engine.get_accuracy_stats = Mock(return_value={})
    life_os.insight_engine = Mock()
    life_os.connector_manager = Mock()
    life_os.connector_manager.get_all_status = Mock(return_value={})
    life_os.config = {}

    return life_os


@pytest.fixture()
def client(life_os_with_weights):
    """A FastAPI TestClient with real DB-backed notification and source weight services."""
    app = create_web_app(life_os_with_weights)
    return TestClient(app)


def _insert_event(db, event_id, event_type, payload=None, metadata=None):
    """Helper: insert a synthetic event into the events DB."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, 'test', datetime('now'), 'normal', ?, ?)""",
            (event_id, event_type, json.dumps(payload or {}), json.dumps(metadata or {})),
        )


def _insert_notification(db, notif_id, source_event_id=None, domain=None):
    """Helper: insert a synthetic notification into the state DB."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, source_event_id, domain, status)
               VALUES (?, 'Test notification', 'body', 'normal', ?, ?, 'delivered')""",
            (notif_id, source_event_id, domain),
        )


def _get_weight_row(db, source_key):
    """Helper: fetch a source weight row."""
    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT * FROM source_weights WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        return dict(row) if row else None


# -------------------------------------------------------------------
# Test: Acting on a notification records engagement
# -------------------------------------------------------------------

def test_act_on_notification_records_engagement(client, life_os_with_weights):
    """Acting on a notification with a source event should call record_engagement."""
    db = life_os_with_weights.db

    # Create source event (an email from a personal domain)
    _insert_event(db, "evt-001", "email.received", payload={"from": "friend@gmail.com"})
    # Create notification linked to the event
    _insert_notification(db, "notif-001", source_event_id="evt-001", domain="email")

    before = _get_weight_row(db, "email.personal")
    assert before["engagements"] == 0

    response = client.post("/api/notifications/notif-001/act")
    assert response.status_code == 200
    assert response.json()["status"] == "acted_on"

    after = _get_weight_row(db, "email.personal")
    assert after["engagements"] == 1


# -------------------------------------------------------------------
# Test: Dismissing a notification records dismissal
# -------------------------------------------------------------------

def test_dismiss_notification_records_dismissal(client, life_os_with_weights):
    """Dismissing a notification with a source event should call record_dismissal."""
    db = life_os_with_weights.db

    _insert_event(db, "evt-002", "message.received", payload={"is_group": True})
    _insert_notification(db, "notif-002", source_event_id="evt-002", domain="messaging")

    before = _get_weight_row(db, "messaging.group")
    assert before["dismissals"] == 0

    response = client.post("/api/notifications/notif-002/dismiss")
    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"

    after = _get_weight_row(db, "messaging.group")
    assert after["dismissals"] == 1


# -------------------------------------------------------------------
# Test: Notification without source_event_id uses domain fallback
# -------------------------------------------------------------------

def test_notification_without_source_event_uses_domain_fallback(client, life_os_with_weights):
    """Notifications without a source_event_id should fall back to domain-based classification."""
    db = life_os_with_weights.db

    # Create notification with domain but no source event
    _insert_notification(db, "notif-003", source_event_id=None, domain="finance")

    before = _get_weight_row(db, "finance.transactions")
    assert before["engagements"] == 0

    response = client.post("/api/notifications/notif-003/act")
    assert response.status_code == 200

    after = _get_weight_row(db, "finance.transactions")
    assert after["engagements"] == 1


# -------------------------------------------------------------------
# Test: Notification without source_event_id or domain doesn't crash
# -------------------------------------------------------------------

def test_notification_without_source_or_domain_graceful(client, life_os_with_weights):
    """Notifications with neither source_event_id nor domain should not crash."""
    db = life_os_with_weights.db

    _insert_notification(db, "notif-004", source_event_id=None, domain=None)

    # Should complete without error — no source weight update happens
    response = client.post("/api/notifications/notif-004/dismiss")
    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"


# -------------------------------------------------------------------
# Test: Prediction-domain notification uses domain-based classification
# -------------------------------------------------------------------

def test_prediction_notification_skips_source_weight_update(client, life_os_with_weights):
    """Prediction notifications are cross-domain and should NOT update any source weight.

    Previously the prediction domain was incorrectly hardcoded to map to 'email.work',
    which silently biased the email.work weight based on prediction interactions rather
    than actual email quality. Now prediction-domain notifications return None from
    _classify_notification_source, so weight updates are skipped entirely.
    """
    db = life_os_with_weights.db

    # Prediction notifications have domain='prediction' and a source_event_id that
    # points to a prediction row (not an event), so the events DB lookup returns nothing
    _insert_notification(db, "notif-005", source_event_id="pred-123", domain="prediction")

    before = _get_weight_row(db, "email.work")
    assert before["dismissals"] == 0

    response = client.post("/api/notifications/notif-005/dismiss")
    assert response.status_code == 200

    # Prediction domain should NOT update email.work — the weight must remain unchanged
    after = _get_weight_row(db, "email.work")
    assert after["dismissals"] == 0


# -------------------------------------------------------------------
# Test: Nonexistent notification doesn't crash
# -------------------------------------------------------------------

def test_nonexistent_notification_graceful(client, life_os_with_weights):
    """Attempting to act on a nonexistent notification should not crash the handler."""
    response = client.post("/api/notifications/nonexistent-id/act")
    assert response.status_code == 200
    assert response.json()["status"] == "acted_on"


# -------------------------------------------------------------------
# Test: Source weight AI drift actually nudges after enough interactions
# -------------------------------------------------------------------

def test_engagement_nudges_drift_after_threshold(client, life_os_with_weights):
    """After MIN_INTERACTIONS, repeated engagements should nudge AI drift upward."""
    db = life_os_with_weights.db
    swm = life_os_with_weights.source_weight_manager

    # Set up the source with enough interactions to activate drift
    with db.get_connection("preferences") as conn:
        conn.execute(
            "UPDATE source_weights SET interactions = 10 WHERE source_key = 'messaging.direct'",
        )

    _insert_event(db, "evt-drift", "message.received", payload={"is_group": False})
    _insert_notification(db, "notif-drift", source_event_id="evt-drift", domain="messaging")

    before = _get_weight_row(db, "messaging.direct")
    assert before["ai_drift"] == 0.0

    response = client.post("/api/notifications/notif-drift/act")
    assert response.status_code == 200

    after = _get_weight_row(db, "messaging.direct")
    # Drift should have nudged up by DRIFT_STEP (0.02)
    assert after["ai_drift"] == pytest.approx(0.02, abs=0.001)
    assert after["engagements"] == 1


# -------------------------------------------------------------------
# Test: Dismissal nudges drift downward after threshold
# -------------------------------------------------------------------

def test_dismissal_nudges_drift_after_threshold(client, life_os_with_weights):
    """After MIN_INTERACTIONS, repeated dismissals should nudge AI drift downward."""
    db = life_os_with_weights.db

    with db.get_connection("preferences") as conn:
        conn.execute(
            "UPDATE source_weights SET interactions = 10 WHERE source_key = 'email.marketing'",
        )

    _insert_event(db, "evt-drift2", "email.received", payload={"from": "noreply@mailchimp.com"})
    _insert_notification(db, "notif-drift2", source_event_id="evt-drift2", domain="email")

    before = _get_weight_row(db, "email.marketing")
    assert before["ai_drift"] == 0.0

    response = client.post("/api/notifications/notif-drift2/dismiss")
    assert response.status_code == 200

    after = _get_weight_row(db, "email.marketing")
    assert after["ai_drift"] == pytest.approx(-0.02, abs=0.001)
    assert after["dismissals"] == 1
