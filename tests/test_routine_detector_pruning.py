"""
Tests for RoutineDetector stale routine pruning and min_occurrences threshold.

Validates that:
- Stale routines (not re-detected and older than 14 days) are pruned from the DB
- Recently-updated routines are preserved even if not re-detected
- Re-detected routines are preserved regardless of age
- DB errors during pruning are handled gracefully (fail-open)
- min_occurrences is set to 3
- Temporal detection with only 2 occurrences does NOT produce a routine
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from services.routine_detector.detector import RoutineDetector


class TestRoutineDetectorPruning:
    """Test suite for stale routine pruning in the RoutineDetector."""

    def test_prune_removes_stale_undetected_routines(self, db, user_model_store):
        """Routines not in the detected list and older than 14 days should be deleted."""
        detector = RoutineDetector(db, user_model_store)

        # Store a routine, then manually backdate its updated_at to 15 days ago
        routine = {
            "name": "Abandoned Morning Routine",
            "trigger": "morning",
            "steps": [{"order": 0, "action": "check_email", "typical_duration_minutes": 5.0, "skip_rate": 0.0}],
            "typical_duration_minutes": 5.0,
            "consistency_score": 0.8,
            "times_observed": 10,
            "variations": [],
        }
        user_model_store.store_routine(routine)

        # Backdate the updated_at to 15 days ago so it's past the 14-day cutoff
        stale_timestamp = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE routines SET updated_at = ? WHERE name = ?",
                (stale_timestamp, "Abandoned Morning Routine"),
            )

        # Verify it's stored
        stored = user_model_store.get_routines()
        assert any(r["name"] == "Abandoned Morning Routine" for r in stored)

        # Prune with empty detected list — the stale routine should be removed
        pruned_count = detector.prune_stale_routines([])
        assert pruned_count == 1

        # Verify it was deleted
        stored_after = user_model_store.get_routines()
        assert not any(r["name"] == "Abandoned Morning Routine" for r in stored_after)

    def test_prune_preserves_detected_routines_even_if_old(self, db, user_model_store):
        """Routines that ARE in the detected list should be preserved even if old."""
        detector = RoutineDetector(db, user_model_store)

        routine = {
            "name": "Still Active Evening Routine",
            "trigger": "evening",
            "steps": [{"order": 0, "action": "inbox_zero", "typical_duration_minutes": 10.0, "skip_rate": 0.1}],
            "typical_duration_minutes": 10.0,
            "consistency_score": 0.9,
            "times_observed": 20,
            "variations": [],
        }
        user_model_store.store_routine(routine)

        # Backdate to 30 days ago — well past the stale threshold
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE routines SET updated_at = ? WHERE name = ?",
                (old_timestamp, "Still Active Evening Routine"),
            )

        # Prune, but include the routine in the detected list
        detected = [{"name": "Still Active Evening Routine"}]
        pruned_count = detector.prune_stale_routines(detected)
        assert pruned_count == 0

        # Verify it still exists
        stored = user_model_store.get_routines()
        assert any(r["name"] == "Still Active Evening Routine" for r in stored)

    def test_prune_preserves_recently_updated_undetected_routines(self, db, user_model_store):
        """Routines NOT in the detected list but updated recently (within 14 days)
        should be preserved — they might just be temporarily absent from detection."""
        detector = RoutineDetector(db, user_model_store)

        routine = {
            "name": "Recent But Absent Routine",
            "trigger": "morning",
            "steps": [{"order": 0, "action": "coffee", "typical_duration_minutes": 5.0, "skip_rate": 0.0}],
            "typical_duration_minutes": 5.0,
            "consistency_score": 0.7,
            "times_observed": 5,
            "variations": [],
        }
        user_model_store.store_routine(routine)

        # updated_at is set to "now" by store_routine, so it's well within 14 days

        # Prune with empty detected list — routine should survive because it's recent
        pruned_count = detector.prune_stale_routines([])
        assert pruned_count == 0

        stored = user_model_store.get_routines()
        assert any(r["name"] == "Recent But Absent Routine" for r in stored)

    def test_prune_handles_db_errors_gracefully(self, db, user_model_store):
        """DB errors during pruning should be caught and logged, not crash the detector."""
        detector = RoutineDetector(db, user_model_store)

        # Mock the DB connection to raise an error
        with patch.object(detector.db, "get_connection") as mock_conn:
            mock_conn.side_effect = Exception("Simulated DB corruption")

            # Should not raise, should return 0
            pruned_count = detector.prune_stale_routines([])
            assert pruned_count == 0

    def test_min_occurrences_is_three(self, db, user_model_store):
        """The min_occurrences threshold should be 3 to avoid false-positive routines."""
        detector = RoutineDetector(db, user_model_store)
        assert detector.min_occurrences == 3

    def test_temporal_detection_with_two_occurrences_produces_no_routine(self, db, user_model_store):
        """With min_occurrences=3, exactly 2 episodes at the same time of day
        should NOT be detected as a routine.

        This tests the core fix: 2 occurrences is too few to distinguish a
        genuine recurring pattern from coincidence.
        """
        detector = RoutineDetector(db, user_model_store)

        # Create exactly 2 episodes at 9am on different days (morning bucket)
        base_date = datetime.now(timezone.utc) - timedelta(days=5)

        for day_offset in range(2):
            day_start = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)

            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "two_day_coffee",
                "content_summary": "Coffee run",
            })

        routines = detector.detect_routines(lookback_days=30)

        # No routine should include "two_day_coffee" — only 2 occurrences < min_occurrences=3
        for routine in routines:
            for step in routine["steps"]:
                assert step["action"] != "two_day_coffee", (
                    "two_day_coffee appeared in a routine with only 2 occurrences; "
                    "min_occurrences=3 should have filtered it out"
                )

    def test_prune_mixed_stale_and_fresh_routines(self, db, user_model_store):
        """When both stale and fresh routines exist, only stale undetected ones are pruned."""
        detector = RoutineDetector(db, user_model_store)

        # Store two routines
        stale_routine = {
            "name": "Old Routine",
            "trigger": "morning",
            "steps": [],
            "typical_duration_minutes": 10.0,
            "consistency_score": 0.7,
            "times_observed": 5,
            "variations": [],
        }
        fresh_routine = {
            "name": "Fresh Routine",
            "trigger": "evening",
            "steps": [],
            "typical_duration_minutes": 15.0,
            "consistency_score": 0.8,
            "times_observed": 8,
            "variations": [],
        }
        user_model_store.store_routine(stale_routine)
        user_model_store.store_routine(fresh_routine)

        # Backdate only the stale one
        stale_timestamp = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE routines SET updated_at = ? WHERE name = ?",
                (stale_timestamp, "Old Routine"),
            )

        # Prune with empty detected list — only Old Routine should go
        pruned_count = detector.prune_stale_routines([])
        assert pruned_count == 1

        stored = user_model_store.get_routines()
        names = {r["name"] for r in stored}
        assert "Old Routine" not in names
        assert "Fresh Routine" in names

    def test_detect_routines_calls_prune(self, db, user_model_store):
        """detect_routines() should automatically call prune_stale_routines()."""
        detector = RoutineDetector(db, user_model_store)

        with patch.object(detector, "prune_stale_routines", return_value=0) as mock_prune:
            detector.detect_routines(lookback_days=30)
            mock_prune.assert_called_once()
            # The argument should be the list of detected routines (possibly empty)
            call_args = mock_prune.call_args
            assert isinstance(call_args[0][0], list)
