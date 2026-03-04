"""
Tests for workflow detector absolute completion count guard.

Validates that the workflow detector uses absolute completion counts
(min_completions=2) instead of rate-based thresholds to decide whether
a workflow is real.  This fixes the fundamental problem where a user
who replied to their boss 15 times but received 5000 total emails had
a 0.3% rate, below any reasonable percentage threshold.

The absolute count approach correctly identifies workflows based on
whether the user has actually performed the action enough times,
regardless of the total volume of triggering events.
"""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from services.workflow_detector.detector import WorkflowDetector


@pytest.fixture
def detector(db, user_model_store):
    """Create a WorkflowDetector instance for testing."""
    return WorkflowDetector(db, user_model_store)


class TestEmailWorkflowAbsoluteCount:
    """Test email workflow detection with absolute completion count guard."""

    def test_3_replies_out_of_1000_detected(self, detector, db):
        """Email workflow with 3 replies out of 1000 received is detected.

        completion_count=3 >= min_completions=2, so the workflow should
        be detected even though the success rate is only 0.3%.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=20)
        sender = "important@example.com"

        with db.get_connection("events") as conn:
            # 1000 emails received from sender
            for i in range(1000):
                event_time = base_time + timedelta(minutes=i * 20)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        event_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # 3 replies within time window
            for i in range(3):
                response_time = base_time + timedelta(minutes=i * 20 + 30)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.sent", "gmail",
                        response_time.isoformat(), 3,
                        json.dumps({"to": sender}),
                        json.dumps({}),
                        None, sender,
                    ),
                )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) >= 1, (
            f"Should detect workflow with 3 completions out of 1000 received "
            f"(completion_count=3 >= min_completions=2), got {len(sender_workflows)}"
        )

        # Success rate is retained for informational purposes
        workflow = sender_workflows[0]
        assert "success_rate" in workflow
        assert workflow["success_rate"] < 0.01  # 0.3% rate

    def test_1_reply_out_of_10_not_detected(self, detector, db):
        """Email workflow with 1 reply out of 10 received is NOT detected.

        completion_count=1 < min_completions=2, so the workflow should NOT
        be detected even though the success rate is 10%.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        sender = "frequent@example.com"

        with db.get_connection("events") as conn:
            # 10 emails received
            for i in range(10):
                event_time = base_time + timedelta(days=i)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        event_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # Only 1 reply (below min_completions=2)
            for i in range(1):
                response_time = base_time + timedelta(days=i, hours=1)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.sent", "gmail",
                        response_time.isoformat(), 3,
                        json.dumps({"to": sender}),
                        json.dumps({}),
                        None, sender,
                    ),
                )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) == 0, (
            "Should NOT detect workflow with only 1 completion "
            "(completion_count=1 < min_completions=2), even though rate is 10%"
        )

    def test_5_replies_out_of_500_detected(self, detector, db):
        """Email workflow with 5 replies out of 500 received is detected.

        completion_count=5 >= min_completions=2, so the workflow should
        be detected even though the success rate is only 1%.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=20)
        sender = "manager@example.com"

        with db.get_connection("events") as conn:
            # 500 emails received
            for i in range(500):
                event_time = base_time + timedelta(minutes=i * 40)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        event_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # 5 replies within time window
            for i in range(5):
                response_time = base_time + timedelta(minutes=i * 40 + 30)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.sent", "gmail",
                        response_time.isoformat(), 3,
                        json.dumps({"to": sender}),
                        json.dumps({}),
                        None, sender,
                    ),
                )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) >= 1, (
            f"Should detect workflow with 5 completions out of 500 received "
            f"(completion_count=5 >= min_completions=2), got {len(sender_workflows)}"
        )


class TestTaskWorkflowAbsoluteCount:
    """Test task workflow detection with absolute completion count guard."""

    def test_task_workflow_with_3_completions_detected(self, detector, db):
        """Task workflow with 3+ completions is detected regardless of rate.

        Creates 20 tasks, completes 3 — completion_count=3 >= min_completions=2.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=15)

        with db.get_connection("events") as conn:
            for i in range(20):
                task_id = str(uuid4())
                create_time = base_time + timedelta(hours=i * 6)

                # Create task
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "task.created", "task_manager",
                        create_time.isoformat(), 3,
                        json.dumps({"task_id": task_id, "title": f"Task {i}"}),
                        json.dumps({}),
                    ),
                )

                # Send follow-up email after task creation (needed for min_steps)
                email_time = create_time + timedelta(hours=1)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.sent", "gmail",
                        email_time.isoformat(), 3,
                        json.dumps({"subject": f"Re: Task {i}"}),
                        json.dumps({}),
                    ),
                )

                # Complete only first 3 tasks
                if i < 3:
                    complete_time = create_time + timedelta(hours=2)
                    conn.execute(
                        """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid4()), "task.completed", "task_manager",
                            complete_time.isoformat(), 3,
                            json.dumps({"task_id": task_id}),
                            json.dumps({}),
                        ),
                    )

        workflows = detector.detect_workflows(lookback_days=30)

        task_workflows = [w for w in workflows if "task" in w["name"].lower()]
        assert len(task_workflows) >= 1, (
            "Should detect task workflow with 3 completions >= min_completions=2"
        )

    def test_task_workflow_with_1_completion_not_detected(self, detector, db):
        """Task workflow with only 1 completion is NOT detected.

        Creates 5 tasks, completes 1 — completion_count=1 < min_completions=2.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=10)

        with db.get_connection("events") as conn:
            for i in range(5):
                task_id = str(uuid4())
                create_time = base_time + timedelta(days=i)

                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "task.created", "task_manager",
                        create_time.isoformat(), 3,
                        json.dumps({"task_id": task_id, "title": f"Task {i}"}),
                        json.dumps({}),
                    ),
                )

                # Send follow-up email
                email_time = create_time + timedelta(hours=1)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.sent", "gmail",
                        email_time.isoformat(), 3,
                        json.dumps({"subject": f"Re: Task {i}"}),
                        json.dumps({}),
                    ),
                )

                # Only complete first 1
                if i < 1:
                    complete_time = create_time + timedelta(hours=2)
                    conn.execute(
                        """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid4()), "task.completed", "task_manager",
                            complete_time.isoformat(), 3,
                            json.dumps({"task_id": task_id}),
                            json.dumps({}),
                        ),
                    )

        workflows = detector.detect_workflows(lookback_days=30)

        task_workflows = [w for w in workflows if "task" in w["name"].lower()]
        assert len(task_workflows) == 0, (
            "Should NOT detect task workflow with 1 completion < min_completions=2"
        )


class TestMinCompletionsAttribute:
    """Test the min_completions attribute replaces success_threshold."""

    def test_min_completions_attribute_exists(self, detector):
        """Verify min_completions attribute exists and success_threshold is removed."""
        assert hasattr(detector, "min_completions"), "Should have min_completions attribute"
        assert detector.min_completions == 2, "min_completions should default to 2"
        assert not hasattr(detector, "success_threshold"), (
            "success_threshold should be removed (replaced by min_completions)"
        )

    def test_diagnostics_reports_min_completions(self, detector):
        """Verify diagnostics reports min_completions instead of success_threshold."""
        diagnostics = detector.get_diagnostics(lookback_days=1)
        thresholds = diagnostics["thresholds"]

        assert "min_completions" in thresholds, "Diagnostics should report min_completions"
        assert thresholds["min_completions"] == 2
        assert "success_threshold" not in thresholds, (
            "Diagnostics should NOT report success_threshold (removed)"
        )

    def test_success_rate_still_in_workflow_dict(self, detector, db):
        """Verify success_rate is kept for informational purposes in workflow dicts.

        Even though we no longer gate on success_rate, it's still useful
        metadata to include in the workflow dict.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        sender = "info-rate@example.com"

        with db.get_connection("events") as conn:
            # 10 emails with 5 replies
            for i in range(10):
                event_time = base_time + timedelta(days=i)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        event_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

                if i < 5:
                    response_time = event_time + timedelta(hours=1)
                    conn.execute(
                        """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                              email_from, email_to)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid4()), "email.sent", "gmail",
                            response_time.isoformat(), 3,
                            json.dumps({"to": sender}),
                            json.dumps({}),
                            None, sender,
                        ),
                    )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) >= 1, "Should detect workflow with 5 completions"

        workflow = sender_workflows[0]
        assert "success_rate" in workflow, "success_rate should still be in workflow dict"
        assert 0.0 < workflow["success_rate"] <= 1.0, "success_rate should be valid"
