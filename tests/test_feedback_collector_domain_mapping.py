"""
Tests for FeedbackCollector._classify_notification_source domain mapping.

Verifies that all notification domains (including 'message' singular) correctly
map to source_key values, enabling source weight updates on dismissal/engagement.
"""

from unittest.mock import MagicMock

import pytest

from services.feedback_collector.collector import FeedbackCollector


@pytest.fixture
def collector_with_swm(db, user_model_store):
    """Create a FeedbackCollector with a mocked SourceWeightManager."""
    swm = MagicMock()
    swm.classify_event = MagicMock(return_value=None)
    return FeedbackCollector(db, user_model_store, source_weight_manager=swm)


class TestClassifyNotificationSource:
    """Tests for _classify_notification_source domain-based fallback mapping."""

    def test_message_singular_maps_to_messaging_direct(self, collector_with_swm):
        """domain='message' (singular) must map to 'messaging.direct'."""
        result = collector_with_swm._classify_notification_source({"domain": "message"})
        assert result == "messaging.direct"

    def test_messaging_plural_maps_to_messaging_direct(self, collector_with_swm):
        """domain='messaging' (plural) must also map to 'messaging.direct'."""
        result = collector_with_swm._classify_notification_source({"domain": "messaging"})
        assert result == "messaging.direct"

    @pytest.mark.parametrize(
        "domain,expected_source_key",
        [
            ("email", "email.work"),
            ("message", "messaging.direct"),
            ("messaging", "messaging.direct"),
            ("calendar", "calendar.meetings"),
            ("finance", "finance.transactions"),
            ("health", "health.activity"),
            ("location", "location.visits"),
            ("home", "home.devices"),
            ("prediction", "email.work"),
            ("system", "home.devices"),
        ],
    )
    def test_all_domain_mappings(self, collector_with_swm, domain, expected_source_key):
        """Every known domain must resolve to the correct source_key."""
        result = collector_with_swm._classify_notification_source({"domain": domain})
        assert result == expected_source_key

    def test_unknown_domain_returns_none(self, collector_with_swm):
        """An unrecognised domain should return None (no weight update)."""
        result = collector_with_swm._classify_notification_source({"domain": "alien"})
        assert result is None

    def test_missing_domain_returns_none(self, collector_with_swm):
        """A notification with no domain key should return None."""
        result = collector_with_swm._classify_notification_source({})
        assert result is None


class TestSourceWeightDismissalWithDomain:
    """Verify _update_source_weight_dismissal calls record_dismissal for domain='message'."""

    def test_dismissal_called_for_message_domain(self, collector_with_swm):
        """record_dismissal must be called when the notification domain is 'message'."""
        collector_with_swm._update_source_weight_dismissal({"domain": "message"})
        collector_with_swm.swm.record_dismissal.assert_called_once_with("messaging.direct")

    def test_dismissal_not_called_for_unknown_domain(self, collector_with_swm):
        """record_dismissal must NOT be called when the domain is unknown."""
        collector_with_swm._update_source_weight_dismissal({"domain": "unknown"})
        collector_with_swm.swm.record_dismissal.assert_not_called()


class TestSourceWeightEngagementWithDomain:
    """Verify _update_source_weight_engagement calls record_engagement for domain='message'."""

    def test_engagement_called_for_message_domain(self, collector_with_swm):
        """record_engagement must be called when the notification domain is 'message'."""
        collector_with_swm._update_source_weight_engagement({"domain": "message"})
        collector_with_swm.swm.record_engagement.assert_called_once_with("messaging.direct")

    def test_engagement_not_called_for_unknown_domain(self, collector_with_swm):
        """record_engagement must NOT be called when the domain is unknown."""
        collector_with_swm._update_source_weight_engagement({"domain": "unknown"})
        collector_with_swm.swm.record_engagement.assert_not_called()
