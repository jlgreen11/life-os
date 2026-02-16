"""
Tests for WorkflowDetector SQL optimization.

Validates that the SQL GROUP BY-based workflow detection:
1. Detects workflows correctly from aggregated data
2. Handles large datasets (78K+ emails) efficiently
3. Matches recipients case-insensitively
4. Filters by minimum occurrences
5. Limits to top 20 senders
6. Calculates success rates and durations accurately
"""

import pytest
from datetime import datetime, timedelta, timezone
from services.workflow_detector.detector import WorkflowDetector


class TestWorkflowDetectorSQLOptimization:
    """Test suite for SQL-optimized workflow detection."""

    def test_detects_simple_email_workflow(self, db, user_model_store, event_store):
        """Test detection of simple receive → send workflow using SQL aggregation."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create 5 emails from boss@company.com with responses within 2 hours
        for i in range(5):
            # Email received
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "boss@company.com",
                    "subject": f"Question {i}",
                    "body_plain": "Quick question..."
                },
                "metadata": {}
            })

            # Response sent 1 hour later
            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": f"Re: Question {i}",
                    "body_plain": "Here's the answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        # Should detect 1 workflow for boss@company.com
        assert len(workflows) >= 1
        boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
        assert boss_workflow is not None
        assert boss_workflow["times_observed"] == 5
        assert "sent" in boss_workflow["steps"]
        assert boss_workflow["success_rate"] >= 0.99  # 5/5 = 100%

    def test_groups_senders_case_insensitively(self, db, user_model_store, event_store):
        """Test that sender grouping is case-insensitive (Boss@company.com == boss@company.com)."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create emails with different case variations
        for i, sender in enumerate([
            "Boss@Company.com",
            "boss@company.com",
            "BOSS@COMPANY.COM"
        ]):
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": sender,
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

            # Response
            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": [sender.lower()],  # Respond using lowercase
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        # Should group all 3 as one sender (lowercase normalized)
        boss_workflows = [w for w in workflows if "boss@company.com" in w["name"].lower()]
        assert len(boss_workflows) == 1
        assert boss_workflows[0]["times_observed"] == 3

    def test_respects_minimum_occurrences_threshold(self, db, user_model_store, event_store):
        """Test that senders with fewer than min_occurrences emails are filtered out."""
        detector = WorkflowDetector(db, user_model_store)
        detector.min_occurrences = 3  # Require at least 3 emails
        base_time = datetime.now(timezone.utc)

        # Sender with only 2 emails (below threshold)
        for i in range(2):
            event_store.store_event({
                "id": f"email-low-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "lowvolume@example.com",
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

        # Sender with 3 emails (meets threshold)
        for i in range(3):
            event_store.store_event({
                "id": f"email-high-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "highvolume@example.com",
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

            event_store.store_event({
                "id": f"email-sent-high-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["highvolume@example.com"],
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        # Should only detect workflow for high-volume sender
        workflow_names = [w["name"] for w in workflows]
        assert any("highvolume@example.com" in name for name in workflow_names)
        assert not any("lowvolume@example.com" in name for name in workflow_names)

    def test_limits_to_top_20_senders(self, db, user_model_store, event_store):
        """Test that only top 20 senders by volume are analyzed (avoids storing 1000s of workflows)."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create 25 senders with varying email volumes
        for sender_idx in range(25):
            # Volume decreases: sender 0 gets 25 emails, sender 1 gets 24, etc.
            volume = 25 - sender_idx
            for email_idx in range(volume):
                event_store.store_event({
                    "id": f"email-{sender_idx}-{email_idx}",
                    "type": "email.received",
                    "source": "protonmail",
                    "timestamp": (base_time + timedelta(days=email_idx)).isoformat(),
                    "priority": "normal",
                    "payload": {
                        "from_address": f"sender{sender_idx}@example.com",
                        "subject": "Question",
                        "body_plain": "Question..."
                    },
                    "metadata": {}
                })

                # Add response for workflow detection
                event_store.store_event({
                    "id": f"email-sent-{sender_idx}-{email_idx}",
                    "type": "email.sent",
                    "source": "protonmail",
                    "timestamp": (base_time + timedelta(days=email_idx, hours=1)).isoformat(),
                    "priority": "normal",
                    "payload": {
                        "to_addresses": [f"sender{sender_idx}@example.com"],
                        "subject": "Re: Question",
                        "body_plain": "Answer..."
                    },
                    "metadata": {}
                })

        workflows = detector.detect_workflows(lookback_days=30)

        # Should have at most 20 workflows (top 20 senders)
        assert len(workflows) <= 20

        # Top senders (0-19) should be included
        top_sender_names = [f"sender{i}@example.com" for i in range(20)]
        workflow_senders = [w["name"].split()[-1] for w in workflows]  # Extract sender from "Responding to X"
        for sender in top_sender_names[:10]:  # Check at least first 10
            assert any(sender in ws for ws in workflow_senders)

        # Bottom senders (20-24) should NOT be included
        bottom_sender_names = [f"sender{i}@example.com" for i in range(20, 25)]
        for sender in bottom_sender_names:
            assert not any(sender in ws for ws in workflow_senders)

    def test_calculates_average_response_time(self, db, user_model_store, event_store):
        """Test that average response time is calculated correctly from SQL aggregation."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create 3 emails with responses at 1h, 2h, 3h (average 2h = 120min)
        for i, hours_to_respond in enumerate([1, 2, 3]):
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "boss@company.com",
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=hours_to_respond)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
        assert boss_workflow is not None

        # Average of 1h, 2h, 3h = 2h = 120min
        expected_avg_minutes = 120.0
        assert boss_workflow["typical_duration_minutes"] == pytest.approx(expected_avg_minutes, rel=0.1)

    def test_filters_responses_outside_time_window(self, db, user_model_store, event_store):
        """Test that responses outside max_step_gap_hours are not counted as workflow steps."""
        detector = WorkflowDetector(db, user_model_store)
        detector.max_step_gap_hours = 4  # Only count responses within 4 hours
        detector.success_threshold = 0.01  # Lower threshold to ensure workflow is detected
        base_time = datetime.now(timezone.utc)

        # Create 5 emails: 3 with responses within window, 2 with response outside window
        for i in range(5):
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "boss@company.com",
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

        # 3 responses within 4h window
        for i in range(3):
            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=2)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        # 2 responses outside 4h window (8 hours later)
        for i in range(3, 5):
            event_store.store_event({
                "id": f"email-sent-late-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=8)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
        assert boss_workflow is not None

        # Success rate should be 3/5 = 60% (only 3 responses within window)
        assert boss_workflow["success_rate"] == pytest.approx(3/5, rel=0.01)

    def test_handles_multiple_action_types(self, db, user_model_store, event_store):
        """Test detection of workflows with multiple action types (email + task + calendar)."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create 3 emails with varied responses
        for i in range(3):
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "boss@company.com",
                    "subject": "Meeting request",
                    "body_plain": "Can we meet?"
                },
                "metadata": {}
            })

            # Create task (1h later)
            event_store.store_event({
                "id": f"task-{i}",
                "type": "task.created",
                "source": "task_manager",
                "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "priority": "normal",
                "payload": {
                    "title": f"Prep for meeting {i}",
                    "description": "Prepare materials"
                },
                "metadata": {}
            })

            # Create calendar event (2h later)
            event_store.store_event({
                "id": f"cal-{i}",
                "type": "calendar.event.created",
                "source": "caldav",
                "timestamp": (base_time + timedelta(days=i, hours=2)).isoformat(),
                "priority": "normal",
                "payload": {
                    "title": f"Meeting {i}",
                    "start_time": (base_time + timedelta(days=i+1)).isoformat()
                },
                "metadata": {}
            })

            # Send response (3h later)
            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=3)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": "Re: Meeting request",
                    "body_plain": "Meeting confirmed"
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)

        boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
        assert boss_workflow is not None

        # Should have multiple steps (task, calendar, email)
        assert len(boss_workflow["steps"]) >= 3

        # Should use multiple tools
        assert len(boss_workflow["tools_used"]) >= 3
        assert "email" in boss_workflow["tools_used"]
        assert "task" in boss_workflow["tools_used"]
        assert "calendar" in boss_workflow["tools_used"]

    def test_performance_with_large_dataset(self, db, user_model_store, event_store):
        """Test that SQL optimization handles 1000+ emails efficiently (< 5 seconds)."""
        import time
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create 1000 emails from 10 different senders
        for sender_idx in range(10):
            for email_idx in range(100):
                event_store.store_event({
                    "id": f"email-{sender_idx}-{email_idx}",
                    "type": "email.received",
                    "source": "protonmail",
                    "timestamp": (base_time + timedelta(days=email_idx, hours=sender_idx)).isoformat(),
                    "priority": "normal",
                    "payload": {
                        "from_address": f"sender{sender_idx}@example.com",
                        "subject": "Question",
                        "body_plain": "Question..."
                    },
                    "metadata": {}
                })

                # 50% get responses
                if email_idx % 2 == 0:
                    event_store.store_event({
                        "id": f"email-sent-{sender_idx}-{email_idx}",
                        "type": "email.sent",
                        "source": "protonmail",
                        "timestamp": (base_time + timedelta(days=email_idx, hours=sender_idx+1)).isoformat(),
                        "priority": "normal",
                        "payload": {
                            "to_addresses": [f"sender{sender_idx}@example.com"],
                            "subject": "Re: Question",
                            "body_plain": "Answer..."
                        },
                        "metadata": {}
                    })

        # Measure detection time
        start = time.time()
        workflows = detector.detect_workflows(lookback_days=30)
        elapsed = time.time() - start

        # Should complete in under 5 seconds (SQL is fast!)
        assert elapsed < 5.0, f"Detection took {elapsed:.2f}s, expected < 5s"

        # Should detect workflows for all 10 senders
        assert len(workflows) == 10

    def test_stores_workflows_to_database(self, db, user_model_store, event_store):
        """Test that detected workflows are persisted correctly to the workflows table."""
        detector = WorkflowDetector(db, user_model_store)
        base_time = datetime.now(timezone.utc)

        # Create workflow pattern
        for i in range(3):
            event_store.store_event({
                "id": f"email-recv-{i}",
                "type": "email.received",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "priority": "normal",
                "payload": {
                    "from_address": "boss@company.com",
                    "subject": "Question",
                    "body_plain": "Question..."
                },
                "metadata": {}
            })

            event_store.store_event({
                "id": f"email-sent-{i}",
                "type": "email.sent",
                "source": "protonmail",
                "timestamp": (base_time + timedelta(days=i, hours=1)).isoformat(),
                "priority": "normal",
                "payload": {
                    "to_addresses": ["boss@company.com"],
                    "subject": "Re: Question",
                    "body_plain": "Answer..."
                },
                "metadata": {}
            })

        workflows = detector.detect_workflows(lookback_days=30)
        stored_count = detector.store_workflows(workflows)

        assert stored_count == len(workflows)

        # Verify workflow is in database
        with db.get_connection("user_model") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM workflows")
            count = cursor.fetchone()[0]
            assert count == stored_count

            cursor.execute("SELECT name, success_rate, times_observed FROM workflows LIMIT 1")
            row = cursor.fetchone()
            assert row is not None
            assert "boss@company.com" in row[0]
            assert row[1] >= 0.99  # 100% success rate
            assert row[2] == 3  # 3 observations
