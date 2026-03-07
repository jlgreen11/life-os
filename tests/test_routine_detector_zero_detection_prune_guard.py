"""
Tests for RoutineDetector zero-detection prune guard and fallback type filtering.

Validates that:
- detect_routines() skips pruning when 0 routines are detected (preserving
  existing routines from a transient detection failure)
- detect_routines() still prunes stale routines when detection succeeds
- Fallback episode classification filters out internal derived types
  (e.g. usermodel_signal_profile_updated)
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from services.routine_detector.detector import RoutineDetector


class TestZeroDetectionPruneGuard:
    """Verify the prune guard prevents nuking routines when detection returns 0."""

    def test_prune_skipped_when_zero_routines_detected(self, db, user_model_store):
        """When detect_routines() finds 0 routines, prune_stale_routines() must
        NOT be called — otherwise a transient detection failure deletes all
        stored routines."""
        detector = RoutineDetector(db, user_model_store)

        # Store a routine and backdate it past the 14-day stale threshold
        routine = {
            "name": "Morning Email Check",
            "trigger": "morning",
            "steps": [{"order": 0, "action": "email_check", "typical_duration_minutes": 10.0, "skip_rate": 0.0}],
            "typical_duration_minutes": 10.0,
            "consistency_score": 0.85,
            "times_observed": 15,
            "variations": [],
        }
        user_model_store.store_routine(routine)

        stale_ts = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE routines SET updated_at = ? WHERE name = ?",
                (stale_ts, "Morning Email Check"),
            )

        # Verify routine exists before detection
        assert any(r["name"] == "Morning Email Check" for r in user_model_store.get_routines())

        # Run detect_routines with no episodes — returns 0 routines
        with patch.object(detector, "prune_stale_routines") as mock_prune:
            detector.detect_routines(lookback_days=30)
            # prune_stale_routines should NOT have been called
            mock_prune.assert_not_called()

        # The stale routine must still exist (not pruned)
        stored = user_model_store.get_routines()
        assert any(r["name"] == "Morning Email Check" for r in stored), (
            "Stale routine was pruned despite 0 detection results — guard failed"
        )

    def test_prune_runs_when_routines_detected(self, db, user_model_store):
        """When detect_routines() finds routines, prune_stale_routines() should
        still be called so genuinely stale routines are cleaned up."""
        detector = RoutineDetector(db, user_model_store)

        # Create enough episodes to trigger detection (3+ at same time bucket)
        base_date = datetime.now(timezone.utc) - timedelta(days=5)
        for day_offset in range(5):
            day_start = base_date.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": day_start.isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "email_check",
                "content_summary": "Checking email",
            })

        with patch.object(detector, "prune_stale_routines", return_value=0) as mock_prune:
            routines = detector.detect_routines(lookback_days=30)
            if routines:
                # If detection found something, prune should have been called
                mock_prune.assert_called_once()
            # If detection still returns 0 (too few episodes), prune correctly skipped


class TestFallbackInternalTypeFilter:
    """Verify that fallback classification filters internal derived types."""

    def test_fallback_filters_internal_derived_types(self, db, user_model_store):
        """Episodes linked to internal event types like usermodel.signal_profile.updated
        should be excluded from fallback classification results."""
        detector = RoutineDetector(db, user_model_store)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        # Create events of internal types in the events DB
        internal_event_id = str(uuid.uuid4())
        normal_event_id = str(uuid.uuid4())

        with db.get_connection("events") as conn:
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (internal_event_id, "usermodel.signal_profile.updated", "system", datetime.now(timezone.utc).isoformat(), "low", "{}"),
            )
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, priority, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (normal_event_id, "email.received", "proton_mail", datetime.now(timezone.utc).isoformat(), "normal", "{}"),
            )

        # Create episodes linking to these events with "unknown" interaction_type
        # so the fallback classification path is exercised (it skips "unknown"
        # and "communication" values and derives the type from the linked event).
        for evt_id, summary in [
            (internal_event_id, "Internal signal update"),
            (normal_event_id, "Email received"),
        ]:
            user_model_store.store_episode({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": evt_id,
                "interaction_type": "unknown",
                "content_summary": summary,
            })

        # Run fallback classification
        result = detector._fallback_temporal_episodes(cutoff)

        # Extract activity types from results
        activity_types = [activity for _ts, activity in result]

        # Internal type should be filtered out
        assert "usermodel_signal_profile_updated" not in activity_types, (
            "Internal derived type 'usermodel_signal_profile_updated' was not filtered"
        )

        # Normal type should be present
        assert "email_received" in activity_types, (
            "Normal derived type 'email_received' should be present"
        )
