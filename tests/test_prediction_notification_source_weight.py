"""Tests for prediction notification source weight tracking.

Verifies that prediction-domain notifications correctly resolve to a
source_key so that dismissals update source weights.  This was previously
broken because prediction notifications use ``domain='prediction'`` with a
``source_event_id`` pointing to the predictions table (not events.db),
causing the lookup to fail silently.
"""

import uuid

import pytest

from main import LifeOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_notification(db, *, notif_id, source_event_id=None, domain="prediction"):
    """Insert a minimal notification row into state.db."""
    with db.get_connection("state") as conn:
        conn.execute(
            """INSERT INTO notifications (id, title, body, priority, domain, source_event_id, created_at)
               VALUES (?, 'test', 'test body', 'low', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            (notif_id, domain, source_event_id),
        )


def _insert_prediction(db, *, prediction_id, prediction_type="REMINDER"):
    """Insert a minimal prediction row into user_model.db."""
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate, created_at)
               VALUES (?, ?, 'test prediction', 0.5, 'SUGGEST',
                       strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            (prediction_id, prediction_type),
        )


@pytest.fixture()
def lifeos_stub(db):
    """A minimal LifeOS-like object with just enough wiring for source key resolution.

    We attach only the ``db`` and ``source_weight_manager`` attributes that the
    methods under test require, avoiding a full LifeOS initialization.
    """
    stub = object.__new__(LifeOS)
    stub.db = db

    # Provide a minimal source_weight_manager with classify_event
    class _StubSWM:
        def classify_event(self, event_dict):
            return None

    stub.source_weight_manager = _StubSWM()
    return stub


# ---------------------------------------------------------------------------
# _classify_prediction_source tests
# ---------------------------------------------------------------------------


class TestClassifyPredictionSource:
    """Tests for LifeOS._classify_prediction_source()."""

    def test_returns_calendar_reminders_for_reminder_type(self, lifeos_stub, db):
        """REMINDER predictions should map to 'calendar.reminders'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="REMINDER")
        assert lifeos_stub._classify_prediction_source(pid) == "calendar.reminders"

    def test_returns_calendar_meetings_for_conflict_type(self, lifeos_stub, db):
        """CONFLICT predictions should map to 'calendar.meetings'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="CONFLICT")
        assert lifeos_stub._classify_prediction_source(pid) == "calendar.meetings"

    def test_returns_email_work_for_need_type(self, lifeos_stub, db):
        """NEED predictions should map to 'email.work'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="NEED")
        assert lifeos_stub._classify_prediction_source(pid) == "email.work"

    def test_returns_email_work_for_opportunity_type(self, lifeos_stub, db):
        """OPPORTUNITY predictions should map to 'email.work'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="OPPORTUNITY")
        assert lifeos_stub._classify_prediction_source(pid) == "email.work"

    def test_returns_email_work_for_risk_type(self, lifeos_stub, db):
        """RISK predictions should map to 'email.work'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="RISK")
        assert lifeos_stub._classify_prediction_source(pid) == "email.work"

    def test_handles_lowercase_prediction_type(self, lifeos_stub, db):
        """Should handle lowercase prediction_type values via .upper()."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="reminder")
        assert lifeos_stub._classify_prediction_source(pid) == "calendar.reminders"

    def test_returns_none_for_missing_prediction(self, lifeos_stub):
        """Should return None when prediction ID doesn't exist in DB."""
        assert lifeos_stub._classify_prediction_source(str(uuid.uuid4())) is None

    def test_falls_back_to_email_work_for_unknown_type(self, lifeos_stub, db):
        """Unknown prediction types should fall back to 'email.work'."""
        pid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="UNKNOWN_TYPE")
        assert lifeos_stub._classify_prediction_source(pid) == "email.work"


# ---------------------------------------------------------------------------
# _resolve_notification_source_key tests (prediction path)
# ---------------------------------------------------------------------------


class TestResolveNotificationSourceKeyPrediction:
    """Tests for the prediction path in _resolve_notification_source_key()."""

    def test_prediction_notification_with_valid_prediction(self, lifeos_stub, db):
        """A prediction notification whose source_event_id points to a real
        prediction should resolve to the correct source_key."""
        pid = str(uuid.uuid4())
        nid = str(uuid.uuid4())
        _insert_prediction(db, prediction_id=pid, prediction_type="REMINDER")
        _insert_notification(db, notif_id=nid, source_event_id=pid, domain="prediction")
        assert lifeos_stub._resolve_notification_source_key(nid) == "calendar.reminders"

    def test_prediction_notification_missing_prediction_falls_back(self, lifeos_stub, db):
        """When the prediction row doesn't exist, the domain fallback should
        return 'email.work' (the _DOMAIN_TO_SOURCE entry for 'prediction')."""
        nid = str(uuid.uuid4())
        _insert_notification(
            db, notif_id=nid, source_event_id=str(uuid.uuid4()), domain="prediction"
        )
        result = lifeos_stub._resolve_notification_source_key(nid)
        assert result == "email.work"

    def test_prediction_notification_without_source_event_id(self, lifeos_stub, db):
        """A prediction notification with no source_event_id should still
        return 'email.work' via the domain fallback."""
        nid = str(uuid.uuid4())
        _insert_notification(db, notif_id=nid, source_event_id=None, domain="prediction")
        result = lifeos_stub._resolve_notification_source_key(nid)
        assert result == "email.work"

    def test_non_prediction_notification_unchanged(self, lifeos_stub, db):
        """Email-domain notifications should still resolve via the domain map."""
        nid = str(uuid.uuid4())
        _insert_notification(db, notif_id=nid, source_event_id=None, domain="email")
        result = lifeos_stub._resolve_notification_source_key(nid)
        assert result == "email.work"

    def test_returns_none_for_unknown_notification(self, lifeos_stub):
        """Should return None when the notification ID doesn't exist."""
        assert lifeos_stub._resolve_notification_source_key(str(uuid.uuid4())) is None
