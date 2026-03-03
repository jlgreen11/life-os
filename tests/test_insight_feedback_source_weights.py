"""
Tests for insight feedback source weight mapping in web/routes.py.

Verifies that the POST /api/insights/{id}/feedback endpoint correctly maps
insight categories (email_timing, meeting_density) to source weight keys,
so that user feedback on these insight types propagates to the source weight
learning loop.

See also: services/insight_engine/engine.py _apply_source_weights()
"""

from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from web.app import create_web_app


@pytest.fixture
def mock_life_os():
    """Create a mock LifeOS instance with source_weight_manager support."""
    life_os = Mock()

    # Mock database manager — the insight feedback route uses get_connection("user_model")
    life_os.db = Mock()

    # Mock source weight manager with engagement/dismissal tracking
    life_os.source_weight_manager = Mock()
    life_os.source_weight_manager.record_engagement = Mock()
    life_os.source_weight_manager.record_dismissal = Mock()

    # Mock event bus (required by create_web_app)
    life_os.event_bus = Mock()
    life_os.event_bus.publish = Mock()

    return life_os


@pytest.fixture
def app(mock_life_os):
    """Create a FastAPI test app with mocked dependencies."""
    return create_web_app(mock_life_os)


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def _setup_insight_row(mock_life_os, category, entity=None):
    """Configure mock DB to return an insight row with the given category.

    Sets up the mock connection so that the route's SELECT query returns
    a dict-like row with the specified category and entity, and the UPDATE
    query succeeds silently.

    Args:
        mock_life_os: The mock LifeOS instance.
        category: The insight category string (e.g. 'email_timing').
        entity: Optional entity value for the insight row.
    """
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = {
        "category": category,
        "entity": entity,
    }
    mock_life_os.db.get_connection = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )


class TestEmailTimingFeedback:
    """Verify email_timing insights map to 'email.work' source key."""

    def test_useful_feedback_records_engagement(self, client, mock_life_os):
        """Marking an email_timing insight as useful records engagement for email.work."""
        _setup_insight_row(mock_life_os, "email_timing")

        response = client.post("/api/insights/ins-email-1/feedback?feedback=useful")

        assert response.status_code == 200
        mock_life_os.source_weight_manager.record_engagement.assert_called_once_with("email.work")
        mock_life_os.source_weight_manager.record_dismissal.assert_not_called()

    def test_dismiss_feedback_records_dismissal(self, client, mock_life_os):
        """Dismissing an email_timing insight records dismissal for email.work."""
        _setup_insight_row(mock_life_os, "email_timing")

        response = client.post("/api/insights/ins-email-2/feedback?feedback=dismissed")

        assert response.status_code == 200
        mock_life_os.source_weight_manager.record_dismissal.assert_called_once_with("email.work")
        mock_life_os.source_weight_manager.record_engagement.assert_not_called()


class TestMeetingDensityFeedback:
    """Verify meeting_density insights map to 'calendar.meetings' source key."""

    def test_useful_feedback_records_engagement(self, client, mock_life_os):
        """Marking a meeting_density insight as useful records engagement for calendar.meetings."""
        _setup_insight_row(mock_life_os, "meeting_density")

        response = client.post("/api/insights/ins-meet-1/feedback?feedback=useful")

        assert response.status_code == 200
        mock_life_os.source_weight_manager.record_engagement.assert_called_once_with("calendar.meetings")
        mock_life_os.source_weight_manager.record_dismissal.assert_not_called()

    def test_dismiss_feedback_records_dismissal(self, client, mock_life_os):
        """Dismissing a meeting_density insight records dismissal for calendar.meetings."""
        _setup_insight_row(mock_life_os, "meeting_density")

        response = client.post("/api/insights/ins-meet-2/feedback?feedback=dismissed")

        assert response.status_code == 200
        mock_life_os.source_weight_manager.record_dismissal.assert_called_once_with("calendar.meetings")
        mock_life_os.source_weight_manager.record_engagement.assert_not_called()


class TestUnknownCategoryFeedback:
    """Verify that unknown categories don't crash and skip source weight updates."""

    def test_unknown_category_no_crash(self, client, mock_life_os):
        """An unknown insight category resolves source_key to None without crashing."""
        _setup_insight_row(mock_life_os, "completely_unknown_category_xyz")

        response = client.post("/api/insights/ins-unknown/feedback?feedback=useful")

        assert response.status_code == 200
        # No source weight call because the category isn't in the mapping
        mock_life_os.source_weight_manager.record_engagement.assert_not_called()
        mock_life_os.source_weight_manager.record_dismissal.assert_not_called()

    def test_missing_insight_row_no_crash(self, client, mock_life_os):
        """Feedback on a nonexistent insight ID still returns 200 without crashing."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_life_os.db.get_connection = Mock(
            return_value=Mock(
                __enter__=Mock(return_value=mock_conn),
                __exit__=Mock(return_value=False),
            )
        )

        response = client.post("/api/insights/nonexistent/feedback?feedback=useful")

        assert response.status_code == 200
        mock_life_os.source_weight_manager.record_engagement.assert_not_called()
        mock_life_os.source_weight_manager.record_dismissal.assert_not_called()
