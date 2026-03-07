"""Tests for routine detector pipeline diagnostics fields.

Validates that get_diagnostics() exposes usable_episode_count,
interaction_type_counts, candidate_pairs_count, and
pairs_meeting_min_occurrences — enabling operators to see exactly
where the detection pipeline filters episodes out.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.routine_detector.detector import RoutineDetector


@pytest.fixture()
def routine_detector(db, user_model_store):
    """A RoutineDetector using UTC timezone for deterministic tests."""
    return RoutineDetector(db, user_model_store, timezone="UTC")


def _insert_episode(db, *, interaction_type="email_received", hours_ago=1, day_offset=0):
    """Insert a single episode into user_model.db.

    Args:
        db: DatabaseManager instance.
        interaction_type: The interaction_type value to store.
        hours_ago: Hours before now for the timestamp (within day_offset day).
        day_offset: Days before today (0 = today).

    Returns:
        The generated episode id.
    """
    ts = datetime.now(UTC) - timedelta(days=day_offset, hours=hours_ago)
    eid = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type,
               content_summary, topics, contacts_involved, location)
               VALUES (?, ?, ?, ?, '', '[]', '[]', '')""",
            (eid, f"evt-{eid}", ts.isoformat(), interaction_type),
        )
    return eid


class TestUsableEpisodeCount:
    """Verify usable_episode_count only counts episodes surviving internal-type filters."""

    def test_diagnostics_includes_usable_episode_count(self, db, routine_detector):
        """Insert episodes with various interaction_types.
        Assert usable_episode_count only counts the usable ones.
        """
        # Usable types
        _insert_episode(db, interaction_type="email_received")
        _insert_episode(db, interaction_type="calendar_review")
        # Filtered types
        _insert_episode(db, interaction_type="unknown")
        _insert_episode(db, interaction_type="communication")
        _insert_episode(db, interaction_type="usermodel_update")
        _insert_episode(db, interaction_type="system_sync")
        _insert_episode(db, interaction_type="test_episode")

        diag = routine_detector.get_diagnostics(lookback_days=30)

        assert diag["episode_count"] == 7  # all episodes
        assert diag["usable_episode_count"] == 2  # only email_received + calendar_review


class TestInteractionTypeCounts:
    """Verify interaction_type_counts returns correct per-type counts."""

    def test_diagnostics_interaction_type_counts(self, db, routine_detector):
        """Insert episodes with different interaction_types.
        Assert the dict contains correct counts.
        """
        for _ in range(5):
            _insert_episode(db, interaction_type="email_received")
        for _ in range(3):
            _insert_episode(db, interaction_type="unknown")
        _insert_episode(db, interaction_type="calendar_review")

        diag = routine_detector.get_diagnostics(lookback_days=30)
        counts = diag["interaction_type_counts"]

        assert counts["email_received"] == 5
        assert counts["unknown"] == 3
        assert counts["calendar_review"] == 1


class TestCandidatePairs:
    """Verify candidate_pairs_count and pairs_meeting_min_occurrences."""

    def test_diagnostics_candidate_pairs_and_min_occurrences(self, db, routine_detector):
        """Insert episodes across 5 distinct days at the same hour with the same
        interaction_type. Assert candidate_pairs_count >= 1 and
        pairs_meeting_min_occurrences >= 1 (since 5 >= min_occurrences of 3).
        """
        for day in range(5):
            _insert_episode(db, interaction_type="email_received", hours_ago=2, day_offset=day)

        diag = routine_detector.get_diagnostics(lookback_days=30)

        assert diag["candidate_pairs_count"] >= 1
        assert diag["pairs_meeting_min_occurrences"] >= 1

    def test_diagnostics_below_min_occurrences(self, db, routine_detector):
        """Insert episodes on only 2 distinct days. Assert candidate_pairs_count >= 1
        but pairs_meeting_min_occurrences == 0 (since 2 < min_occurrences of 3).
        """
        for day in range(2):
            _insert_episode(db, interaction_type="email_received", hours_ago=2, day_offset=day)

        diag = routine_detector.get_diagnostics(lookback_days=30)

        assert diag["candidate_pairs_count"] >= 1
        assert diag["pairs_meeting_min_occurrences"] == 0
