"""
Tests for workflow detector extended time gap and lowered min_completions.

Validates that the workflow detector with max_step_gap_hours=12 correctly
detects multi-session workflows that span a workday (e.g. email at 9am,
response at 4pm = 7h gap), and that min_completions=2 allows less-frequent
but real patterns to surface.

Background:
- With max_step_gap_hours=4, any workflow spanning more than 4 hours between
  steps was invisible. Real workflows often span a full workday or overnight.
- With min_completions=3, contacts emailed only twice were filtered out even
  though two completions represent a real pattern worth surfacing.
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


class TestExtendedGapDetection:
    """Test that the 12h gap window catches workday-spanning workflows."""

    def test_email_workflow_detected_with_8h_gap(self, detector, db):
        """Email received at T, response at T+8h should detect workflow.

        This simulates a real scenario: receive email at 9am, respond at 5pm.
        With the old 4h window this was missed; with 12h it's captured.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "workday-sender@company.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                # Email received in the morning
                receive_time = base_time + timedelta(days=i)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        receive_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

                # Response sent 8 hours later (end of workday)
                response_time = receive_time + timedelta(hours=8)
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
            f"Should detect workflow with 8h gap (within 12h window), got {len(sender_workflows)}"
        )

        workflow = sender_workflows[0]
        assert any("sent" in step for step in workflow["steps"]), "Should include email sending step"
        assert workflow["times_observed"] == 5

    def test_email_workflow_missed_with_13h_gap(self, detector, db):
        """Email received at T, response at T+13h should NOT detect workflow.

        13 hours exceeds the 12h max_step_gap_hours, so these events should
        not be linked into a workflow.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "overnight-sender@company.com"

        with db.get_connection("events") as conn:
            for i in range(5):
                # Email received
                receive_time = base_time + timedelta(days=i)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        receive_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

                # Response sent 13 hours later (beyond 12h window)
                response_time = receive_time + timedelta(hours=13)
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
            "Should NOT detect workflow with 13h gap (exceeds 12h window)"
        )


class TestMinCompletionsTwo:
    """Test that min_completions=2 captures less-frequent but real patterns."""

    def test_min_completions_two_suffices(self, detector, db):
        """Two sent emails that produce enough sliding-window matches to detect.

        With min_completions=2, a sender who has been replied to twice
        represents a real workflow pattern worth surfacing. The sliding
        window matches each sent email against multiple close-together
        received emails, producing enough delay entries to pass the
        min_occurrences filter on action types.
        """
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        sender = "twice-replied@company.com"

        with db.get_connection("events") as conn:
            # 6 emails received close together (2h apart, all within 12h window)
            # This ensures each sent email matches multiple received emails
            for i in range(6):
                receive_time = base_time + timedelta(hours=i * 2)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        receive_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # 2 responses — each matches multiple received emails in window
            # Sent at hour 11 matches received at hours 0,2,4,6,8,10 (6 matches)
            response_time_1 = base_time + timedelta(hours=11)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                      email_from, email_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.sent", "gmail",
                    response_time_1.isoformat(), 3,
                    json.dumps({"to": sender}),
                    json.dumps({}),
                    None, sender,
                ),
            )

            # Add more received emails for second batch
            for i in range(4):
                receive_time = base_time + timedelta(days=1, hours=i * 2)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        receive_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # Second response
            response_time_2 = base_time + timedelta(days=1, hours=7)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                      email_from, email_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.sent", "gmail",
                    response_time_2.isoformat(), 3,
                    json.dumps({"to": sender}),
                    json.dumps({}),
                    None, sender,
                ),
            )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) >= 1, (
            f"Should detect workflow with 2 sent emails (>= min_completions=2), got {len(sender_workflows)}"
        )

    def test_one_completion_below_min(self, detector, db):
        """One completion should still be below min_completions=2."""
        base_time = datetime.now(timezone.utc) - timedelta(days=10)
        sender = "once-replied@company.com"

        with db.get_connection("events") as conn:
            # 5 emails received
            for i in range(5):
                receive_time = base_time + timedelta(days=i)
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                          email_from, email_to)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()), "email.received", "gmail",
                        receive_time.isoformat(), 3,
                        json.dumps({"from_address": sender}),
                        json.dumps({}),
                        sender, None,
                    ),
                )

            # Only 1 response (below min_completions=2)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata,
                                      email_from, email_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.sent", "gmail",
                    (base_time + timedelta(hours=1)).isoformat(), 3,
                    json.dumps({"to": sender}),
                    json.dumps({}),
                    None, sender,
                ),
            )

        workflows = detector.detect_workflows(lookback_days=30)

        sender_workflows = [w for w in workflows if sender in w["name"]]
        assert len(sender_workflows) == 0, (
            "Should NOT detect workflow with only 1 completion (< min_completions=2)"
        )


class TestDiagnosticsThresholds:
    """Test that diagnostics correctly reports the updated thresholds."""

    def test_diagnostics_returns_thresholds(self, detector):
        """Verify get_diagnostics() reports the updated threshold values."""
        diagnostics = detector.get_diagnostics(lookback_days=1)

        assert "thresholds" in diagnostics
        thresholds = diagnostics["thresholds"]

        assert thresholds["max_step_gap_hours"] == 12, (
            f"max_step_gap_hours should be 12, got {thresholds['max_step_gap_hours']}"
        )
        assert thresholds["min_completions"] == 2, (
            f"min_completions should be 2, got {thresholds['min_completions']}"
        )
        assert thresholds["min_occurrences"] == 3, (
            f"min_occurrences should remain 3, got {thresholds['min_occurrences']}"
        )
        assert thresholds["min_steps"] == 2, (
            f"min_steps should remain 2, got {thresholds['min_steps']}"
        )

    def test_diagnostics_has_status_fields(self, detector):
        """Verify diagnostics returns all expected sections."""
        diagnostics = detector.get_diagnostics(lookback_days=1)

        assert "thresholds" in diagnostics
        assert "event_counts" in diagnostics
        assert "detection_results" in diagnostics
        assert "total_detected" in diagnostics
        assert "data_sufficient" in diagnostics
