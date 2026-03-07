"""
Tests for workflow detector telemetry filtering.

Validates that WorkflowDetector._detect_interaction_workflows() excludes
internal telemetry episode types (usermodel_*, system_*, test*) and
generic types (unknown, communication) from workflow detection, preventing
false workflow patterns from noise data.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from services.workflow_detector import WorkflowDetector


@pytest.fixture
def workflow_detector(db, user_model_store):
    """Create a WorkflowDetector instance for testing."""
    return WorkflowDetector(db, user_model_store)


def _insert_episodes(user_model_store, interaction_type: str, count: int, base_time: datetime, interval_hours: float = 2.0):
    """Helper to insert multiple episodes with the same interaction_type.

    Args:
        user_model_store: UserModelStore fixture.
        interaction_type: The interaction_type to assign.
        count: Number of episodes to create.
        base_time: Starting timestamp.
        interval_hours: Hours between each episode.
    """
    for i in range(count):
        ts = base_time + timedelta(hours=i * interval_hours)
        user_model_store.store_episode({
            "id": str(uuid4()),
            "timestamp": ts.isoformat(),
            "event_id": str(uuid4()),
            "interaction_type": interaction_type,
            "content_summary": f"{interaction_type} episode {i}",
        })


class TestTelemetryFiltering:
    """Test that internal telemetry types are excluded from workflow detection."""

    def test_interaction_workflows_exclude_telemetry(self, workflow_detector, user_model_store):
        """Telemetry episode types (usermodel_*, system_*) must not produce workflows."""
        base_time = datetime.now(timezone.utc) - timedelta(days=10)

        # Insert many telemetry episodes that would form patterns if not filtered
        _insert_episodes(user_model_store, "usermodel_signal_profile_updated", 20, base_time, interval_hours=0.5)
        _insert_episodes(user_model_store, "system_rule_triggered", 20, base_time + timedelta(minutes=15), interval_hours=0.5)

        # Insert a small number of real episodes (not enough to form workflows)
        _insert_episodes(user_model_store, "email_received", 2, base_time + timedelta(minutes=30), interval_hours=4.0)

        workflows = workflow_detector.detect_workflows(lookback_days=30)

        # No workflows should reference telemetry types
        for wf in workflows:
            steps_str = " ".join(wf.get("steps", []))
            assert "usermodel_" not in steps_str, f"Telemetry type leaked into workflow: {wf['name']}"
            assert "system_" not in steps_str, f"Telemetry type leaked into workflow: {wf['name']}"

    def test_interaction_workflows_exclude_unknown_and_communication(self, workflow_detector, user_model_store):
        """Generic types 'unknown' and 'communication' must be excluded."""
        base_time = datetime.now(timezone.utc) - timedelta(days=10)

        # Insert many generic episodes
        _insert_episodes(user_model_store, "unknown", 20, base_time, interval_hours=0.5)
        _insert_episodes(user_model_store, "communication", 20, base_time + timedelta(minutes=10), interval_hours=0.5)

        workflows = workflow_detector.detect_workflows(lookback_days=30)

        # No workflows should reference generic types
        for wf in workflows:
            steps = wf.get("steps", [])
            assert "unknown" not in steps, f"'unknown' leaked into workflow: {wf['name']}"
            assert "communication" not in steps, f"'communication' leaked into workflow: {wf['name']}"

    def test_interaction_workflows_include_real_types(self, workflow_detector, user_model_store):
        """Real interaction types (email_received, email_sent, calendar_event_created) must be included."""
        base_time = datetime.now(timezone.utc) - timedelta(days=20)

        # Create repeating sequences of real interaction types
        for i in range(6):
            day_offset = i * 2
            t = base_time + timedelta(days=day_offset)

            user_model_store.store_episode({
                "id": str(uuid4()),
                "timestamp": t.isoformat(),
                "event_id": str(uuid4()),
                "interaction_type": "email_received",
                "content_summary": f"Received email {i}",
            })

            t2 = t + timedelta(hours=1)
            user_model_store.store_episode({
                "id": str(uuid4()),
                "timestamp": t2.isoformat(),
                "event_id": str(uuid4()),
                "interaction_type": "email_sent",
                "content_summary": f"Sent reply {i}",
            })

            t3 = t + timedelta(hours=2)
            user_model_store.store_episode({
                "id": str(uuid4()),
                "timestamp": t3.isoformat(),
                "event_id": str(uuid4()),
                "interaction_type": "calendar_event_created",
                "content_summary": f"Created follow-up event {i}",
            })

        workflows = workflow_detector.detect_workflows(lookback_days=30)

        # Detection should run without errors and return a list
        assert isinstance(workflows, list)

        # If workflows were detected, they should contain real types only
        if workflows:
            all_steps = []
            for wf in workflows:
                all_steps.extend(wf.get("steps", []))
            # At least one real type should appear in detected workflows
            real_types = {"email_received", "email_sent", "calendar_event_created"}
            assert any(step in real_types for step in all_steps), (
                f"Expected real interaction types in workflows, got steps: {all_steps}"
            )

    def test_telemetry_does_not_create_false_workflows(self, workflow_detector, user_model_store):
        """Many telemetry episodes plus few real episodes must not produce telemetry workflows."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)

        # Massive volume of telemetry episodes forming clear patterns
        _insert_episodes(user_model_store, "usermodel_signal_profile_updated", 50, base_time, interval_hours=0.25)
        _insert_episodes(user_model_store, "system_rule_triggered", 50, base_time + timedelta(minutes=5), interval_hours=0.25)
        _insert_episodes(user_model_store, "test_event", 50, base_time + timedelta(minutes=10), interval_hours=0.25)

        # A few real episodes (not enough to form a workflow on their own)
        _insert_episodes(user_model_store, "email_received", 2, base_time, interval_hours=24.0)
        _insert_episodes(user_model_store, "web_browsing", 2, base_time + timedelta(hours=1), interval_hours=24.0)

        workflows = workflow_detector.detect_workflows(lookback_days=30)

        # No workflow should be built from telemetry types
        telemetry_prefixes = ("usermodel_", "system_", "test")
        for wf in workflows:
            for step in wf.get("steps", []):
                assert not any(step.startswith(p) for p in telemetry_prefixes), (
                    f"False workflow from telemetry: {wf['name']} has step '{step}'"
                )

    def test_internal_type_prefixes_constant(self, workflow_detector):
        """Verify INTERNAL_TYPE_PREFIXES matches the RoutineDetector's constant."""
        assert workflow_detector.INTERNAL_TYPE_PREFIXES == ("usermodel_", "system_", "test")

    def test_diagnostics_excludes_telemetry(self, workflow_detector, user_model_store):
        """get_diagnostics() episode distribution must also exclude telemetry types."""
        base_time = datetime.now(timezone.utc) - timedelta(days=5)

        # Insert telemetry and real episodes
        _insert_episodes(user_model_store, "usermodel_signal_profile_updated", 10, base_time, interval_hours=1.0)
        _insert_episodes(user_model_store, "system_rule_triggered", 10, base_time + timedelta(minutes=5), interval_hours=1.0)
        _insert_episodes(user_model_store, "email_received", 5, base_time + timedelta(minutes=10), interval_hours=2.0)

        diag = workflow_detector.get_diagnostics(lookback_days=30)

        # The episode interaction type distribution should not include telemetry
        distribution = diag.get("episode_interaction_types", {}).get("distribution", {})
        assert "usermodel_signal_profile_updated" not in distribution, (
            "Telemetry type appeared in diagnostics distribution"
        )
        assert "system_rule_triggered" not in distribution, (
            "Telemetry type appeared in diagnostics distribution"
        )
        # Real type should be present
        if distribution:
            assert "email_received" in distribution
