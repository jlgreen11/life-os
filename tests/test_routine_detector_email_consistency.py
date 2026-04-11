"""
Tests for routine detector consistency calculation with mixed interaction types.

Verifies that the max-day-count consistency fix prevents rare co-occurring
interaction types from dragging down dominant patterns below the detection
threshold.

Background: The old avg_day_count approach computed consistency as the mean
across ALL interaction types in a time bucket.  If a morning bucket had
email_received on 39/39 days AND meeting_scheduled on 3/39 days, the average
was (39+3)/2=21 → consistency=21/39=0.54, below the 0.6 threshold.  The fix
uses max_day_count=39 → consistency=1.0 → routine detected.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.routine_detector.detector import RoutineDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_at(days_ago: int, hour: int = 8, minute: int = 0) -> str:
    """Return an ISO 8601 UTC timestamp for N days ago at the given hour/minute.

    Args:
        days_ago: How many days before now (0 = today, 1 = yesterday, …)
        hour: Hour of day in UTC (default 8)
        minute: Minute within the hour (default 0)

    Returns:
        ISO 8601 string with UTC timezone info
    """
    now = datetime.now(timezone.utc)
    dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0) - timedelta(days=days_ago)
    return dt.isoformat()


def _store_episode(user_model_store, interaction_type: str, ts: str) -> None:
    """Insert a single episode with the given interaction type and timestamp.

    Args:
        user_model_store: UserModelStore fixture instance
        interaction_type: The episode interaction type (e.g. 'email_received')
        ts: ISO 8601 timestamp string
    """
    user_model_store.store_episode(
        {
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "event_id": str(uuid.uuid4()),
            "interaction_type": interaction_type,
            "content_summary": f"Auto-generated {interaction_type} episode",
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRoutineConsistencyWithMixedTypes:
    """Validate that consistency uses max day_count, not average day_count.

    These tests reproduce the production scenario where a dominant pattern
    (email_received every morning) was masked by a rare co-occurring type
    (meeting_scheduled a few mornings), causing 0 routines to be detected.
    """

    def test_dominant_pattern_not_masked_by_rare_type(self, db, user_model_store):
        """Morning routine detected when rare type co-occurs in the same bucket.

        Setup:
          - email_received every morning for 20 days (100% consistent)
          - meeting_scheduled on 3 of those same mornings (15% consistent)

        Old behaviour: avg=(20+3)/2=11.5, consistency=0.575 → FAIL
        New behaviour: max=20, consistency=1.0 → PASS
        """
        detector = RoutineDetector(db, user_model_store)

        num_days = 20

        # email_received every morning for the past num_days days
        for d in range(num_days):
            _store_episode(user_model_store, "email_received", _ts_at(days_ago=d + 1, hour=8, minute=5))

        # meeting_scheduled on only 3 of those mornings
        for d in [2, 7, 14]:
            _store_episode(user_model_store, "meeting_scheduled", _ts_at(days_ago=d + 1, hour=8, minute=30))

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected at least one morning routine; email_received appeared on all 20 days "
            "but meeting_scheduled on only 3 days should not drag the score below threshold."
        )

        routine = morning_routines[0]
        assert routine["consistency_score"] >= 0.6, (
            f"Expected consistency >= 0.6, got {routine['consistency_score']}"
        )
        # The dominant step should be email_received
        step_actions = [s["action"] for s in routine["steps"]]
        assert "email_received" in step_actions, (
            f"email_received should be a step in the routine; got {step_actions}"
        )

    def test_base_case_single_type_detected(self, db, user_model_store):
        """Morning routine detected when only email_received episodes exist.

        Verifies that the fix does not break the simple single-type case.
        Setup: 15 email_received episodes spread across 15 days at 8 AM.
        Expected: morning routine detected with consistency = 1.0.
        """
        detector = RoutineDetector(db, user_model_store)

        for d in range(15):
            _store_episode(user_model_store, "email_received", _ts_at(days_ago=d + 1, hour=8, minute=10))

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, "Expected morning routine from 15 days of email_received"

        routine = morning_routines[0]
        assert routine["consistency_score"] >= 0.6
        assert routine["times_observed"] >= 15

    def test_consistency_uses_max_not_average(self, db, user_model_store):
        """Directly verifies max-based consistency produces correct score.

        Setup:
          - email_received appears 10/10 active days in morning bucket
          - meeting_scheduled appears 3/10 active days in morning bucket
          - task_created appears 1/10 active days in morning bucket

        avg_day_count = (10+3+1)/3 ≈ 4.67, consistency ≈ 0.47 → would FAIL
        max_day_count = 10, consistency = 1.0 → PASS
        """
        detector = RoutineDetector(db, user_model_store)

        # email_received every morning for 10 days
        for d in range(10):
            _store_episode(user_model_store, "email_received", _ts_at(days_ago=d + 1, hour=8, minute=0))

        # meeting_scheduled on 3 of those days
        for d in [1, 4, 8]:
            _store_episode(user_model_store, "meeting_scheduled", _ts_at(days_ago=d + 1, hour=8, minute=20))

        # task_created on 1 day
        _store_episode(user_model_store, "task_created", _ts_at(days_ago=5 + 1, hour=8, minute=40))

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected morning routine; max-based consistency should be 1.0 but "
            "avg-based would have been ~0.47 (below threshold)"
        )

    def test_truly_rare_bucket_not_detected(self, db, user_model_store):
        """A time bucket with genuinely inconsistent presence is NOT a routine.

        Setup:
          - 20 days of afternoon email activity (so active_days = 20)
          - morning task_created on only 3 days

        Even with max_day_count, consistency = 3/20 = 0.15 → below threshold.
        """
        detector = RoutineDetector(db, user_model_store)

        # 20 days of afternoon activity so active_days = 20
        for d in range(20):
            _store_episode(user_model_store, "email_received", _ts_at(days_ago=d + 1, hour=14, minute=0))

        # Only 3 morning appearances of task_created
        for d in [0, 10, 19]:
            _store_episode(user_model_store, "task_created", _ts_at(days_ago=d + 1, hour=8, minute=0))

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        # task_created on 3/20 days → consistency=0.15 → should NOT be a routine
        assert len(morning_routines) == 0, (
            f"Did not expect a morning routine for 3/20 day consistency; got {morning_routines}"
        )

    def test_high_volume_mixed_bucket_real_world_scenario(self, db, user_model_store):
        """Reproduces the production scenario: 860 emails across multiple weeks.

        Scaled-down version: 100 email_received episodes across 20 days (5/day),
        plus 3 meeting_scheduled on 3 of those mornings.

        This is the scenario described in the task description where 0 routines
        were detected in production despite ample data.
        """
        detector = RoutineDetector(db, user_model_store)

        # 5 email episodes per day across 20 days = 100 emails total
        for d in range(20):
            for m in range(5):
                _store_episode(
                    user_model_store,
                    "email_received",
                    _ts_at(days_ago=d + 1, hour=8, minute=m * 3),
                )

        # 3 meeting_scheduled on 3 distinct mornings
        for d in [0, 7, 15]:
            _store_episode(user_model_store, "meeting_scheduled", _ts_at(days_ago=d + 1, hour=9, minute=0))

        routines = detector.detect_routines(lookback_days=30)

        morning_routines = [r for r in routines if r["trigger"] == "morning"]
        assert len(morning_routines) >= 1, (
            "Expected morning routine from 100 email episodes across 20 days; "
            "3 meeting_scheduled episodes should not suppress detection."
        )
