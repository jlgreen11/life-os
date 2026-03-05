"""
Tests for workflow detector get_diagnostics() episode interaction_type distribution.

Validates that diagnostics include episode_interaction_types with distribution,
type_diversity, episode_total, and sufficient_for_interaction_workflows fields.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from services.workflow_detector import WorkflowDetector


@pytest.fixture
def workflow_detector(db, user_model_store):
    """Create a WorkflowDetector instance for testing."""
    return WorkflowDetector(db, user_model_store)


def _insert_episodes(db, episodes):
    """Insert episode records into the user_model database.

    Args:
        db: DatabaseManager instance
        episodes: List of (interaction_type, timestamp) tuples
    """
    with db.get_connection("user_model") as conn:
        for interaction_type, timestamp in episodes:
            conn.execute(
                """
                INSERT INTO episodes (id, timestamp, event_id, interaction_type,
                    content_summary, content_full)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    timestamp.isoformat(),
                    str(uuid4()),
                    interaction_type,
                    f"Test {interaction_type} episode",
                    f"Full content for {interaction_type}",
                ),
            )


class TestDiagnosticsEpisodeInteractionTypes:
    """Tests for the episode_interaction_types section in get_diagnostics()."""

    def test_diagnostics_includes_episode_interaction_types(self, workflow_detector, db):
        """get_diagnostics() should include the episode_interaction_types key."""
        result = workflow_detector.get_diagnostics(lookback_days=30)
        assert "episode_interaction_types" in result
        section = result["episode_interaction_types"]
        assert "distribution" in section
        assert "episode_total" in section
        assert "type_diversity" in section
        assert "sufficient_for_interaction_workflows" in section

    def test_empty_episodes_table(self, workflow_detector, db):
        """With no episodes, totals should be zero and sufficiency false."""
        result = workflow_detector.get_diagnostics(lookback_days=30)
        section = result["episode_interaction_types"]
        assert section["episode_total"] == 0
        assert section["type_diversity"] == 0
        assert section["sufficient_for_interaction_workflows"] is False
        assert section["distribution"] == {}

    def test_diverse_interaction_types_sufficient(self, workflow_detector, db):
        """3+ distinct types with 3+ episodes each should be sufficient."""
        now = datetime.now(timezone.utc)
        episodes = []
        for itype in ["email_read", "email_send", "calendar_check", "task_complete"]:
            for i in range(4):
                episodes.append((itype, now - timedelta(days=i + 1)))
        _insert_episodes(db, episodes)

        result = workflow_detector.get_diagnostics(lookback_days=30)
        section = result["episode_interaction_types"]
        assert section["episode_total"] == 16
        assert section["type_diversity"] == 4
        assert section["sufficient_for_interaction_workflows"] is True
        assert section["distribution"]["email_read"] == 4
        assert section["distribution"]["email_send"] == 4

    def test_single_type_insufficient(self, workflow_detector, db):
        """Only 1 interaction type should be insufficient for workflows."""
        now = datetime.now(timezone.utc)
        episodes = [("email_read", now - timedelta(days=i)) for i in range(10)]
        _insert_episodes(db, episodes)

        result = workflow_detector.get_diagnostics(lookback_days=30)
        section = result["episode_interaction_types"]
        assert section["episode_total"] == 10
        assert section["type_diversity"] == 1
        assert section["sufficient_for_interaction_workflows"] is False

    def test_types_below_min_occurrences_not_counted(self, workflow_detector, db):
        """Types with fewer than min_occurrences episodes don't count toward sufficiency."""
        now = datetime.now(timezone.utc)
        episodes = []
        # 3 types with enough episodes
        for itype in ["email_read", "email_send", "calendar_check"]:
            for i in range(3):
                episodes.append((itype, now - timedelta(days=i + 1)))
        # 2 types with only 1 episode each (below min_occurrences=3)
        episodes.append(("browse", now - timedelta(days=1)))
        episodes.append(("chat", now - timedelta(days=1)))
        _insert_episodes(db, episodes)

        result = workflow_detector.get_diagnostics(lookback_days=30)
        section = result["episode_interaction_types"]
        assert section["type_diversity"] == 5  # All types counted in diversity
        assert section["sufficient_for_interaction_workflows"] is True  # 3 types meet threshold

    def test_lookback_window_respected(self, workflow_detector, db):
        """Episodes outside the lookback window should not be included."""
        now = datetime.now(timezone.utc)
        episodes = []
        # Recent episodes (within 7 days)
        for i in range(3):
            episodes.append(("email_read", now - timedelta(days=i + 1)))
        # Old episodes (60 days ago, outside a 30-day window)
        for i in range(5):
            episodes.append(("calendar_check", now - timedelta(days=60 + i)))
        _insert_episodes(db, episodes)

        result = workflow_detector.get_diagnostics(lookback_days=30)
        section = result["episode_interaction_types"]
        assert section["episode_total"] == 3
        assert section["type_diversity"] == 1
        assert "calendar_check" not in section["distribution"]
