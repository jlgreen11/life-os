"""Tests for FeedbackCollector -> SourceWeightManager integration.

Verifies that notification feedback flowing through the FeedbackCollector
(event-bus path) correctly updates source weights, closing the integration
gap where only web-route-driven feedback updated weights.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.feedback_collector.collector import FeedbackCollector


@pytest.fixture()
def mock_swm():
    """A mock SourceWeightManager with the methods we care about."""
    swm = MagicMock()
    swm.classify_event.return_value = "email.work"
    swm.record_dismissal = MagicMock()
    swm.record_engagement = MagicMock()
    return swm


@pytest.fixture()
def collector_with_swm(db, user_model_store, mock_swm):
    """FeedbackCollector wired with a mock SourceWeightManager."""
    return FeedbackCollector(db, user_model_store, source_weight_manager=mock_swm)


@pytest.fixture()
def collector_without_swm(db, user_model_store):
    """FeedbackCollector without SourceWeightManager (backward compat)."""
    return FeedbackCollector(db, user_model_store)


@pytest.fixture()
def sample_notification():
    """A minimal notification dict matching the DB row schema."""
    return {
        "id": "notif-001",
        "domain": "email",
        "priority": "medium",
        "source_event_id": None,
    }


class TestLearnFromDismissalSourceWeights:
    """_learn_from_dismissal should call swm.record_dismissal."""

    def test_dismissal_updates_source_weight(self, collector_with_swm, mock_swm, sample_notification):
        """Quick dismissal triggers source weight dismissal."""
        collector_with_swm._learn_from_dismissal(sample_notification, response_time=1.0)
        mock_swm.record_dismissal.assert_called_once_with("email.work")

    def test_slow_dismissal_updates_source_weight(self, collector_with_swm, mock_swm, sample_notification):
        """Slow dismissal also triggers source weight dismissal."""
        collector_with_swm._learn_from_dismissal(sample_notification, response_time=15.0)
        mock_swm.record_dismissal.assert_called_once_with("email.work")

    def test_mid_range_dismissal_still_updates_source_weight(self, collector_with_swm, mock_swm, sample_notification):
        """Mid-range dismissal (2-10 sec) still updates source weights even though
        it doesn't update semantic facts."""
        collector_with_swm._learn_from_dismissal(sample_notification, response_time=5.0)
        mock_swm.record_dismissal.assert_called_once_with("email.work")


class TestLearnFromEngagementSourceWeights:
    """_learn_from_engagement should call swm.record_engagement."""

    def test_engagement_updates_source_weight(self, collector_with_swm, mock_swm, sample_notification):
        """Quick engagement triggers source weight engagement."""
        collector_with_swm._learn_from_engagement(sample_notification, response_time=10.0)
        mock_swm.record_engagement.assert_called_once_with("email.work")

    def test_slow_engagement_still_updates_source_weight(self, collector_with_swm, mock_swm, sample_notification):
        """Slow engagement (>30s) still updates source weights even though it
        doesn't update semantic facts."""
        collector_with_swm._learn_from_engagement(sample_notification, response_time=60.0)
        mock_swm.record_engagement.assert_called_once_with("email.work")


class TestLearnFromIgnoreSourceWeights:
    """_learn_from_ignore should call swm.record_dismissal (ignored = negative signal)."""

    def test_ignore_updates_source_weight_as_dismissal(self, collector_with_swm, mock_swm, sample_notification):
        """Ignored notification triggers source weight dismissal (strongest negative signal)."""
        collector_with_swm._learn_from_ignore(sample_notification)
        mock_swm.record_dismissal.assert_called_once_with("email.work")


class TestNoSourceWeightManager:
    """When source_weight_manager is None, learning methods still work without crashing."""

    def test_dismissal_without_swm(self, collector_without_swm, sample_notification):
        """Dismissal works fine when source_weight_manager is None."""
        collector_without_swm._learn_from_dismissal(sample_notification, response_time=1.0)
        # No exception raised — success

    def test_engagement_without_swm(self, collector_without_swm, sample_notification):
        """Engagement works fine when source_weight_manager is None."""
        collector_without_swm._learn_from_engagement(sample_notification, response_time=10.0)
        # No exception raised — success

    def test_ignore_without_swm(self, collector_without_swm, sample_notification):
        """Ignore works fine when source_weight_manager is None."""
        collector_without_swm._learn_from_ignore(sample_notification)
        # No exception raised — success


class TestClassifyNotificationSource:
    """Tests for the _classify_notification_source helper method."""

    def test_domain_fallback_email(self, collector_with_swm, mock_swm):
        """Email domain falls back to email.work."""
        notif = {"domain": "email", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "email.work"

    def test_domain_fallback_messaging(self, collector_with_swm, mock_swm):
        """Messaging domain falls back to messaging.direct."""
        notif = {"domain": "messaging", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "messaging.direct"

    def test_domain_fallback_calendar(self, collector_with_swm, mock_swm):
        """Calendar domain falls back to calendar.meetings."""
        notif = {"domain": "calendar", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "calendar.meetings"

    def test_domain_fallback_finance(self, collector_with_swm, mock_swm):
        """Finance domain falls back to finance.transactions."""
        notif = {"domain": "finance", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "finance.transactions"

    def test_domain_fallback_health(self, collector_with_swm, mock_swm):
        """Health domain falls back to health.activity."""
        notif = {"domain": "health", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "health.activity"

    def test_domain_fallback_location(self, collector_with_swm, mock_swm):
        """Location domain falls back to location.visits."""
        notif = {"domain": "location", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "location.visits"

    def test_domain_fallback_home(self, collector_with_swm, mock_swm):
        """Home domain falls back to home.devices."""
        notif = {"domain": "home", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result == "home.devices"

    def test_unknown_domain_returns_none(self, collector_with_swm, mock_swm):
        """Unknown domain returns None (no misattribution)."""
        notif = {"domain": "prediction", "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result is None

    def test_no_domain_no_source_event_returns_none(self, collector_with_swm, mock_swm):
        """No domain and no source_event_id returns None."""
        notif = {"domain": None, "source_event_id": None}
        result = collector_with_swm._classify_notification_source(notif)
        assert result is None

    def test_source_event_classification(self, collector_with_swm, mock_swm, db):
        """When source_event_id points to a real event, classify via SWM."""
        # Insert a source event into the events DB
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("evt-001", "email.received", "proton_mail", "2026-01-01T00:00:00Z",
                 "medium", json.dumps({"from": "alice@company.com"}), json.dumps({})),
            )

        mock_swm.classify_event.return_value = "email.personal"
        notif = {"domain": "email", "source_event_id": "evt-001"}
        result = collector_with_swm._classify_notification_source(notif)

        assert result == "email.personal"
        mock_swm.classify_event.assert_called_once()

    def test_without_swm_returns_none(self, collector_without_swm):
        """Without SWM, classification always returns None."""
        notif = {"domain": "email", "source_event_id": None}
        result = collector_without_swm._classify_notification_source(notif)
        assert result is None

    def test_source_event_db_error_falls_back_to_domain(self, collector_with_swm, mock_swm):
        """If source event lookup fails, falls back to domain-based mapping."""
        # source_event_id points to a non-existent event
        notif = {"domain": "email", "source_event_id": "nonexistent-evt"}
        result = collector_with_swm._classify_notification_source(notif)
        # Falls back to domain mapping since event not found
        assert result == "email.work"


class TestSourceWeightErrorHandling:
    """Source weight updates should never break feedback processing."""

    def test_swm_record_dismissal_exception_is_caught(self, collector_with_swm, mock_swm, sample_notification):
        """Exception from swm.record_dismissal is caught and logged."""
        mock_swm.record_dismissal.side_effect = Exception("DB error")
        # Should not raise
        collector_with_swm._learn_from_dismissal(sample_notification, response_time=1.0)

    def test_swm_record_engagement_exception_is_caught(self, collector_with_swm, mock_swm, sample_notification):
        """Exception from swm.record_engagement is caught and logged."""
        mock_swm.record_engagement.side_effect = Exception("DB error")
        # Should not raise
        collector_with_swm._learn_from_engagement(sample_notification, response_time=10.0)

    def test_swm_classify_exception_is_caught(self, collector_with_swm, mock_swm, sample_notification):
        """Exception from _classify_notification_source is caught and logged."""
        mock_swm.classify_event.side_effect = Exception("classify error")
        # Set source_event_id to trigger classify_event path
        sample_notification["source_event_id"] = "evt-bad"
        # Should not raise — falls back to domain mapping
        collector_with_swm._learn_from_dismissal(sample_notification, response_time=1.0)
