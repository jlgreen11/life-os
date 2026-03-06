"""
Tests for routine detector telemetry filtering.

Verifies that internal telemetry episode types (usermodel_*, system_*, test*)
are excluded from routine detection queries so that they don't inflate
active_days counts or drown out real user activity patterns.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from services.routine_detector.detector import RoutineDetector


def _insert_episode(db, *, interaction_type, timestamp, episode_id=None, event_id=None):
    """Insert a single episode into user_model.db for testing."""
    eid = episode_id or str(uuid.uuid4())
    evid = event_id or str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary, active_domain)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (eid, timestamp.isoformat(), evid, interaction_type, "test summary", "test"),
        )


class TestTelemetryFilterConstants:
    """Verify the class-level filter constants are defined correctly."""

    def test_internal_type_prefixes_defined(self):
        """INTERNAL_TYPE_PREFIXES should list known telemetry prefixes."""
        assert "usermodel_" in RoutineDetector.INTERNAL_TYPE_PREFIXES
        assert "system_" in RoutineDetector.INTERNAL_TYPE_PREFIXES
        assert "test" in RoutineDetector.INTERNAL_TYPE_PREFIXES

    def test_sql_filter_excludes_all_prefixes(self):
        """INTERNAL_TYPE_SQL_FILTER should contain LIKE clauses for each prefix."""
        sql = RoutineDetector.INTERNAL_TYPE_SQL_FILTER
        assert "usermodel_%" in sql
        assert "system_%" in sql
        assert "test%" in sql


class TestCountActiveDays:
    """Verify _count_active_days excludes internal telemetry types."""

    def test_excludes_usermodel_types(self, db, user_model_store):
        """Days with only usermodel_* episodes should not count as active."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=10)

        # Insert 5 days of usermodel telemetry (should be excluded)
        for i in range(5):
            ts = now - timedelta(days=i, hours=1)
            _insert_episode(db, interaction_type="usermodel_signal_profile_updated", timestamp=ts)

        result = detector._count_active_days(cutoff)
        assert result == 1, "Days with only usermodel_ episodes should not count (returns 1 = safe default)"

    def test_excludes_system_types(self, db, user_model_store):
        """Days with only system_* episodes should not count as active."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=10)

        for i in range(5):
            ts = now - timedelta(days=i, hours=1)
            _insert_episode(db, interaction_type="system_rule_triggered", timestamp=ts)

        result = detector._count_active_days(cutoff)
        assert result == 1

    def test_counts_real_user_activity(self, db, user_model_store):
        """Days with real user activity (email_received) should be counted."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=10)

        # Insert 3 days of real user activity
        for i in range(3):
            ts = now - timedelta(days=i, hours=2)
            _insert_episode(db, interaction_type="email_received", timestamp=ts)

        result = detector._count_active_days(cutoff)
        assert result == 3

    def test_mixed_internal_and_real_counts_only_real(self, db, user_model_store):
        """Active days count should only reflect days with real user activity."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)
        cutoff = now - timedelta(days=10)

        # 5 days of system telemetry
        for i in range(5):
            ts = now - timedelta(days=i, hours=1)
            _insert_episode(db, interaction_type="system_rule_triggered", timestamp=ts)

        # 2 days of real activity (day 0 and day 1)
        for i in range(2):
            ts = now - timedelta(days=i, hours=3)
            _insert_episode(db, interaction_type="email_received", timestamp=ts)

        result = detector._count_active_days(cutoff)
        assert result == 2, "Only days with real user activity should be counted"


class TestTemporalDetectionFilter:
    """Verify _detect_temporal_routines excludes internal telemetry."""

    def test_excludes_usermodel_episodes(self, db, user_model_store):
        """Telemetry episodes should not appear in temporal detection results."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)

        # Insert 10 days of usermodel telemetry at 8am — would be a "routine"
        # if not filtered out
        for i in range(10):
            ts = now.replace(hour=8, minute=0, second=0) - timedelta(days=i)
            _insert_episode(db, interaction_type="usermodel_signal_profile_updated", timestamp=ts)

        routines = detector._detect_temporal_routines(lookback_days=30)

        # No routine should be detected from telemetry-only data
        for r in routines:
            steps = r.get("steps", [])
            for step in steps:
                assert not step["action"].startswith("usermodel_"), (
                    f"Telemetry type should be excluded: {step['action']}"
                )

    def test_includes_email_received(self, db, user_model_store):
        """Real user activity like email_received should be detected as routines."""
        detector = RoutineDetector(db, user_model_store)
        detector.min_occurrences = 3
        detector.consistency_threshold = 0.3  # Lenient for test
        now = datetime.now(UTC)

        # Insert 10 days of email_received at 9am
        for i in range(10):
            ts = now.replace(hour=9, minute=0, second=0) - timedelta(days=i)
            _insert_episode(db, interaction_type="email_received", timestamp=ts)

        routines = detector._detect_temporal_routines(lookback_days=30)

        # Should find at least one routine involving email_received
        email_actions = []
        for r in routines:
            for step in r.get("steps", []):
                if step["action"] == "email_received":
                    email_actions.append(step)

        assert len(email_actions) > 0, "email_received should be detected as a routine"

    def test_mixed_telemetry_and_real_detects_only_real(self, db, user_model_store):
        """With a mix of internal + real episodes, only real types form routines.

        Scenario: 30 days of system_rule_triggered + 10 days of email_received
        in the morning bucket. Without filtering, active_days=30 and consistency
        for email = 10/30 = 0.33 < 0.6 threshold. With filtering, active_days=10
        and consistency = 10/10 = 1.0 — routine detected.
        """
        detector = RoutineDetector(db, user_model_store)
        detector.min_occurrences = 3
        now = datetime.now(UTC)

        # 30 days of system telemetry at noon
        for i in range(30):
            ts = now.replace(hour=12, minute=0, second=0) - timedelta(days=i)
            _insert_episode(db, interaction_type="system_rule_triggered", timestamp=ts)

        # 10 days of email_received in the morning
        for i in range(10):
            ts = now.replace(hour=8, minute=30, second=0) - timedelta(days=i)
            _insert_episode(db, interaction_type="email_received", timestamp=ts)

        routines = detector._detect_temporal_routines(lookback_days=45)

        # Should detect an email routine since active_days is now 10 (not 30)
        routine_actions = set()
        for r in routines:
            for step in r.get("steps", []):
                routine_actions.add(step["action"])

        assert "email_received" in routine_actions, (
            "email_received should be detected as a morning routine when telemetry is excluded"
        )
        assert "system_rule_triggered" not in routine_actions, (
            "system_rule_triggered should be excluded from detected routines"
        )


class TestEventTriggeredFilter:
    """Verify _detect_event_triggered_routines excludes internal telemetry."""

    def test_excludes_system_trigger_types(self, db, user_model_store):
        """system_* types should not appear as trigger candidates."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)

        # Insert 10 days of system telemetry
        for i in range(10):
            ts = now - timedelta(days=i, hours=1)
            _insert_episode(db, interaction_type="system_rule_triggered", timestamp=ts)

        routines = detector._detect_event_triggered_routines(lookback_days=30)

        for r in routines:
            assert "system_" not in r.get("trigger", ""), (
                "system_ types should not be routine triggers"
            )


class TestLocationRoutineFilter:
    """Verify _detect_location_routines excludes internal telemetry."""

    def test_excludes_usermodel_at_location(self, db, user_model_store):
        """usermodel_* episodes at a location should not form location routines."""
        detector = RoutineDetector(db, user_model_store)
        now = datetime.now(UTC)

        # Insert episodes with location but telemetry type
        for i in range(10):
            ts = now - timedelta(days=i, hours=1)
            eid = str(uuid.uuid4())
            evid = str(uuid.uuid4())
            with db.get_connection("user_model") as conn:
                conn.execute(
                    """INSERT INTO episodes
                       (id, timestamp, event_id, interaction_type, content_summary,
                        active_domain, location)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (eid, ts.isoformat(), evid, "usermodel_signal_profile_updated",
                     "test", "test", "home"),
                )

        routines = detector._detect_location_routines(lookback_days=30)

        for r in routines:
            for step in r.get("steps", []):
                assert not step["action"].startswith("usermodel_"), (
                    "usermodel_ types should be excluded from location routines"
                )
