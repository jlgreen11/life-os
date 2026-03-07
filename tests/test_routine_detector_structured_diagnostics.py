"""
Tests for RoutineDetector.get_diagnostics() structured observability method.

Validates that the diagnostics method returns all expected fields, correctly
reflects episode data, and is resilient to database errors.
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from services.routine_detector.detector import RoutineDetector


class TestGetDiagnosticsReturnsAllFields:
    """get_diagnostics() should return a dict with all expected keys."""

    def test_get_diagnostics_returns_all_fields(self, db, user_model_store):
        """All expected diagnostic keys must be present in the result."""
        detector = RoutineDetector(db, user_model_store)
        diag = detector.get_diagnostics()

        expected_keys = {
            "episode_count",
            "active_days",
            "effective_consistency_threshold",
            "distinct_interaction_types",
            "episodes_per_day",
            "time_bucket_distribution",
            "usable_episode_count",
            "interaction_type_counts",
            "candidate_pairs_count",
            "pairs_meeting_min_occurrences",
            "stored_routines_count",
            "last_detection_count",
            "last_detection_time",
            "health",
        }
        assert expected_keys == set(diag.keys()), (
            f"Missing keys: {expected_keys - set(diag.keys())}, "
            f"Extra keys: {set(diag.keys()) - expected_keys}"
        )


class TestDiagnosticsEpisodeCount:
    """episode_count should reflect actual episodes in the lookback window."""

    def test_diagnostics_episode_count_matches_data(self, db, user_model_store):
        """Insert 10 episodes and verify episode_count=10."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=5)

        for i in range(10):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test episode",
                }
            )

        diag = detector.get_diagnostics(lookback_days=30)
        assert diag["episode_count"] == 10

    def test_diagnostics_with_no_episodes(self, db, user_model_store):
        """Empty DB returns episode_count=0 and active_days minimum of 1."""
        detector = RoutineDetector(db, user_model_store)
        diag = detector.get_diagnostics()

        assert diag["episode_count"] == 0
        # _count_active_days returns max(1, ...) to avoid division by zero
        assert diag["active_days"] == 1
        assert diag["stored_routines_count"] == 0
        assert diag["last_detection_count"] is None
        assert diag["last_detection_time"] is None
        assert diag["health"] == "no_data"


class TestDiagnosticsActiveDays:
    """active_days should count distinct calendar days with episode data."""

    def test_diagnostics_active_days_calculation(self, db, user_model_store):
        """Insert episodes across 5 distinct days, verify active_days=5."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=10)

        for day_offset in range(5):
            day = base_date + timedelta(days=day_offset * 2)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "calendar_reviewed",
                    "content_summary": "test",
                }
            )

        diag = detector.get_diagnostics(lookback_days=30)
        assert diag["active_days"] == 5


class TestDiagnosticsTimeBucketDistribution:
    """time_bucket_distribution should correctly bucket episodes by time of day."""

    def test_diagnostics_time_bucket_distribution(self, db, user_model_store):
        """Insert episodes at specific hours and verify correct bucket counts."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=3)

        # 3 morning episodes (hour 8)
        for i in range(3):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": base_date.replace(hour=8).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "morning",
                }
            )
            base_date += timedelta(days=1)

        # Reset for evening episodes
        base_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=3)

        # 2 evening episodes (hour 19)
        for i in range(2):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": base_date.replace(hour=19).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "reading",
                    "content_summary": "evening",
                }
            )
            base_date += timedelta(days=1)

        diag = detector.get_diagnostics(lookback_days=30)
        buckets = diag["time_bucket_distribution"]

        assert isinstance(buckets, dict)
        assert buckets["morning"] == 3
        assert buckets["evening"] == 2
        assert buckets["midday"] == 0
        assert buckets["afternoon"] == 0
        assert buckets["night"] == 0


class TestDiagnosticsAfterDetection:
    """After running detect_routines(), diagnostics should reflect the cached result."""

    def test_diagnostics_after_detection_run(self, db, user_model_store):
        """Run detect_routines(), then verify last_detection_count and last_detection_time are populated."""
        detector = RoutineDetector(db, user_model_store)

        # Before detection, values should be None
        diag_before = detector.get_diagnostics()
        assert diag_before["last_detection_count"] is None
        assert diag_before["last_detection_time"] is None

        # Run detection (no episodes, so 0 routines found)
        detector.detect_routines(lookback_days=30)

        diag_after = detector.get_diagnostics()
        assert diag_after["last_detection_count"] == 0
        assert diag_after["last_detection_time"] is not None
        # Verify the timestamp is a valid ISO format string
        datetime.fromisoformat(diag_after["last_detection_time"])

    def test_diagnostics_detection_count_reflects_found_routines(self, db, user_model_store):
        """When routines ARE detected, last_detection_count should match."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=10)

        # Create a clear morning pattern: email_received at 8am on 10 days
        for day_offset in range(10):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "Email received",
                }
            )

        routines = detector.detect_routines(lookback_days=30)
        diag = detector.get_diagnostics()

        assert diag["last_detection_count"] == len(routines)
        assert diag["last_detection_count"] >= 1
        assert diag["health"] == "ok"


class TestDiagnosticsDistinctInteractionTypes:
    """distinct_interaction_types should list all unique types in the window."""

    def test_distinct_interaction_types(self, db, user_model_store):
        """Insert episodes with different types and verify they all appear."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=3)
        types = ["email_received", "calendar_reviewed", "task_created"]

        for i, itype in enumerate(types):
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": (base_date + timedelta(hours=i)).isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": itype,
                    "content_summary": "test",
                }
            )

        diag = detector.get_diagnostics(lookback_days=30)
        assert set(diag["distinct_interaction_types"]) == set(types)


class TestDiagnosticsEpisodesPerDay:
    """episodes_per_day should be the average across active days."""

    def test_episodes_per_day_calculation(self, db, user_model_store):
        """10 episodes across 5 days should give episodes_per_day=2.0."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=10)

        for day_offset in range(5):
            day = base_date + timedelta(days=day_offset * 2)
            for hour_offset in range(2):
                user_model_store.store_episode(
                    {
                        "id": str(uuid.uuid4()),
                        "timestamp": (day + timedelta(hours=hour_offset)).isoformat(),
                        "event_id": str(uuid.uuid4()),
                        "interaction_type": "email_received",
                        "content_summary": "test",
                    }
                )

        diag = detector.get_diagnostics(lookback_days=30)
        assert diag["episodes_per_day"] == 2.0


class TestDiagnosticsDbErrorResilience:
    """get_diagnostics() should not crash if user_model.db is unavailable."""

    def test_diagnostics_db_error_resilience(self, db, user_model_store):
        """Verify get_diagnostics() returns gracefully when DB queries fail."""
        detector = RoutineDetector(db, user_model_store)

        # Replace db with a mock that raises on get_connection
        broken_db = MagicMock()
        broken_db.get_connection.side_effect = Exception("DB unavailable")
        detector.db = broken_db

        # Should not raise — each field wraps queries in try/except
        diag = detector.get_diagnostics()

        assert isinstance(diag, dict)
        # Fields that depend on DB queries should have error dicts
        assert isinstance(diag["episode_count"], dict)
        assert "error" in diag["episode_count"]
        assert isinstance(diag["time_bucket_distribution"], dict)
        assert "error" in diag["time_bucket_distribution"]
        assert isinstance(diag["stored_routines_count"], dict)
        assert "error" in diag["stored_routines_count"]
        # Cached fields should still work (they don't touch DB)
        assert diag["last_detection_count"] is None
        assert diag["last_detection_time"] is None


class TestDiagnosticsHealthIndicator:
    """Health indicator should reflect pipeline state."""

    def test_health_no_data_when_never_run(self, db, user_model_store):
        """Health should be no_data when detect_routines() was never called."""
        detector = RoutineDetector(db, user_model_store)
        diag = detector.get_diagnostics()
        assert diag["health"] == "no_data"

    def test_health_degraded_when_episodes_but_no_routines(self, db, user_model_store):
        """Health should be degraded when there are episodes but 0 routines detected."""
        detector = RoutineDetector(db, user_model_store)

        # Insert some episodes that won't form routines (too few days)
        user_model_store.store_episode(
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(UTC).isoformat(),
                "event_id": str(uuid.uuid4()),
                "interaction_type": "email_received",
                "content_summary": "test",
            }
        )

        # Run detection — expect 0 routines from 1 episode
        detector.detect_routines(lookback_days=30)

        diag = detector.get_diagnostics()
        assert diag["health"] == "degraded"

    def test_health_ok_when_routines_detected(self, db, user_model_store):
        """Health should be ok when routines have been successfully detected."""
        detector = RoutineDetector(db, user_model_store)
        base_date = datetime.now(UTC) - timedelta(days=10)

        for day_offset in range(10):
            day = base_date.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
            user_model_store.store_episode(
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": day.isoformat(),
                    "event_id": str(uuid.uuid4()),
                    "interaction_type": "email_received",
                    "content_summary": "test",
                }
            )

        detector.detect_routines(lookback_days=30)
        diag = detector.get_diagnostics()
        assert diag["health"] == "ok"
