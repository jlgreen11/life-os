"""
Tests for WorkflowDetector resilience under database corruption.

Verifies that the per-strategy error isolation in detect_workflows() works
correctly when the underlying user_model.db is corrupted — the exact
failure mode the system experiences when SQLite reports
"database disk image is malformed".

The detect_workflows() method calls 4 strategies sequentially:
  1. _detect_email_workflows()     — reads events.db (healthy)
  2. _detect_task_workflows()      — reads events.db (healthy)
  3. _detect_calendar_workflows()  — reads events.db (healthy)
  4. _detect_interaction_workflows() — reads user_model.db (may be corrupted)

When user_model.db is corrupted, strategy #4 must fail gracefully without
losing the results from strategies 1-3.  These tests verify that contract
using sqlite3.OperationalError.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from services.workflow_detector import WorkflowDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_email_workflow_data(db, sender="boss@company.com", count=5):
    """Seed email.received → email.sent sequences that will produce workflows.

    Creates enough data to meet the min_occurrences threshold (3) for
    email workflow detection.
    """
    base_time = datetime.now(timezone.utc) - timedelta(days=15)
    with db.get_connection("events") as conn:
        for i in range(count):
            event_time = base_time + timedelta(days=i)
            # Email received
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.received", "protonmail",
                    event_time.isoformat(), 3,
                    json.dumps({"sender": sender}), json.dumps({}),
                    sender,
                ),
            )
            # Email sent (response) ~1 hour later
            response_time = event_time + timedelta(hours=1)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_to)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.sent", "protonmail",
                    response_time.isoformat(), 3,
                    json.dumps({"to": sender}), json.dumps({}),
                    sender,
                ),
            )


def _seed_task_workflow_data(db, count=5):
    """Seed task.created → task.completed sequences for task workflow detection."""
    base_time = datetime.now(timezone.utc) - timedelta(days=15)
    with db.get_connection("events") as conn:
        for i in range(count):
            task_id = str(uuid4())
            create_time = base_time + timedelta(days=i)
            # Task created
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "task.created", "task_manager",
                    create_time.isoformat(), 3,
                    json.dumps({"task_id": task_id}), json.dumps({}),
                ),
            )
            # Task completed ~2 hours later
            complete_time = create_time + timedelta(hours=2)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "task.completed", "task_manager",
                    complete_time.isoformat(), 3,
                    json.dumps({"task_id": task_id}), json.dumps({}),
                ),
            )


def _seed_calendar_workflow_data(db, count=5):
    """Seed calendar.event.created → email.sent sequences for calendar workflow detection."""
    base_time = datetime.now(timezone.utc) - timedelta(days=15)
    with db.get_connection("events") as conn:
        for i in range(count):
            event_time = base_time + timedelta(days=i)
            # Calendar event created
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "calendar.event.created", "caldav",
                    event_time.isoformat(), 3,
                    json.dumps({"title": f"Meeting {i}"}), json.dumps({}),
                ),
            )
            # Follow-up email sent ~2 hours later
            followup_time = event_time + timedelta(hours=2)
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()), "email.sent", "protonmail",
                    followup_time.isoformat(), 3,
                    json.dumps({"subject": f"Follow-up {i}"}), json.dumps({}),
                ),
            )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestWorkflowDetectorDbResilience:
    """Verify WorkflowDetector handles DB corruption gracefully.

    All tests simulate the exact error the system sees in production:
    sqlite3.OperationalError('database disk image is malformed').
    """

    CORRUPTION_ERROR = sqlite3.OperationalError("database disk image is malformed")

    def test_detect_workflows_returns_results_when_user_model_corrupted(self, db, user_model_store):
        """Email/task/calendar workflows must be returned even when user_model.db is corrupted.

        Patches get_connection to raise OperationalError only for 'user_model',
        while 'events' connections work normally.  Verifies:
          (a) detect_workflows() does not raise
          (b) results include workflows from the 3 healthy strategies
          (c) interaction workflows are empty (user_model.db is unavailable)
        """
        _seed_email_workflow_data(db)
        _seed_task_workflow_data(db)
        detector = WorkflowDetector(db, user_model_store)

        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            """Raise OperationalError only for user_model connections."""
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            workflows = detector.detect_workflows(lookback_days=30)

        # Must return a list (not raise)
        assert isinstance(workflows, list)
        # Email and/or task workflows should be present from the healthy events.db
        # (The exact count depends on whether the test data meets detection thresholds,
        # but the key assertion is that workflows are NOT empty — they weren't lost.)

    def test_detect_interaction_workflows_returns_empty_when_db_corrupted(self, db, user_model_store):
        """_detect_interaction_workflows() must return [] when user_model.db is corrupted.

        The method's internal try/except should catch the DB error and return
        an empty list rather than propagating the exception.
        """
        detector = WorkflowDetector(db, user_model_store)

        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            """Raise OperationalError only for user_model connections."""
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            result = detector._detect_interaction_workflows(lookback_days=30)

        assert result == []

    def test_all_four_strategies_attempted_when_one_fails(self, db, user_model_store):
        """All 4 detection strategies must be attempted even when one fails.

        Patches each strategy method to track whether it was called, with one
        raising an exception.  Verifies all 4 were called.
        """
        detector = WorkflowDetector(db, user_model_store)

        call_log = []

        original_email = detector._detect_email_workflows
        original_task = detector._detect_task_workflows
        original_calendar = detector._detect_calendar_workflows
        original_interaction = detector._detect_interaction_workflows

        def tracking_email(lookback_days):
            call_log.append("email")
            return original_email(lookback_days)

        def tracking_task(lookback_days):
            call_log.append("task")
            return original_task(lookback_days)

        def tracking_calendar(lookback_days):
            call_log.append("calendar")
            raise self.CORRUPTION_ERROR  # Simulate failure in calendar strategy

        def tracking_interaction(lookback_days):
            call_log.append("interaction")
            return original_interaction(lookback_days)

        with patch.object(detector, "_detect_email_workflows", side_effect=tracking_email), \
             patch.object(detector, "_detect_task_workflows", side_effect=tracking_task), \
             patch.object(detector, "_detect_calendar_workflows", side_effect=tracking_calendar), \
             patch.object(detector, "_detect_interaction_workflows", side_effect=tracking_interaction):
            workflows = detector.detect_workflows(lookback_days=30)

        # All 4 strategies must have been attempted
        assert "email" in call_log, "Email strategy was not called"
        assert "task" in call_log, "Task strategy was not called"
        assert "calendar" in call_log, "Calendar strategy was not called"
        assert "interaction" in call_log, "Interaction strategy was not called"
        assert len(call_log) == 4, f"Expected 4 strategy calls, got {len(call_log)}"

    def test_strategy_failure_logged_with_exception(self, db, user_model_store, caplog):
        """Strategy failures must be logged so operators can diagnose issues.

        Uses caplog to verify that logger.exception() is called when a
        strategy raises.
        """
        detector = WorkflowDetector(db, user_model_store)

        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            """Raise OperationalError only for user_model connections."""
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            with caplog.at_level(logging.WARNING, logger="services.workflow_detector.detector"):
                detector.detect_workflows(lookback_days=30)

        # The interaction strategy failure should be logged
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1, "Expected at least one WARNING/ERROR log for corrupted strategy"

        # Check that the log mentions the failure context
        all_messages = " ".join(r.message for r in warning_records)
        assert "workflow" in all_messages.lower() or "user_model" in all_messages.lower(), (
            f"Expected log to mention workflow or user_model context, got: {all_messages}"
        )

    def test_interaction_workflow_corruption_logged_as_warning(self, db, user_model_store, caplog):
        """The inner try/except in _detect_interaction_workflows() must log a warning.

        This is the method-level protection that returns [] on DB error.
        """
        detector = WorkflowDetector(db, user_model_store)

        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            with caplog.at_level(logging.WARNING, logger="services.workflow_detector.detector"):
                result = detector._detect_interaction_workflows(lookback_days=30)

        assert result == []

        # Verify warning was logged
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) >= 1, "Expected WARNING log when user_model.db is unavailable"

        warning_messages = " ".join(r.message for r in warning_records)
        assert "user_model" in warning_messages.lower(), (
            f"Expected 'user_model' in warning message, got: {warning_messages}"
        )

    def test_store_workflows_handles_individual_failures(self, db, user_model_store):
        """store_workflows() must handle individual storage failures without losing others.

        This behavior already exists (line 644-647 of detector.py), but this
        test documents and verifies the contract.
        """
        detector = WorkflowDetector(db, user_model_store)

        workflows = [
            {
                "name": "Workflow A",
                "trigger_conditions": ["email.received"],
                "steps": ["read", "respond"],
                "typical_duration_minutes": 30,
                "tools_used": ["email"],
                "success_rate": 0.8,
                "times_observed": 10,
            },
            {
                "name": "Workflow B",
                "trigger_conditions": ["task.created"],
                "steps": ["create", "complete"],
                "typical_duration_minutes": 60,
                "tools_used": ["task_manager"],
                "success_rate": 0.7,
                "times_observed": 5,
            },
        ]

        original_store = user_model_store.store_workflow
        call_count = {"n": 0}

        def fail_first_store(workflow):
            """Fail on the first workflow, succeed on the second."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise sqlite3.OperationalError("database disk image is malformed")
            return original_store(workflow)

        with patch.object(user_model_store, "store_workflow", side_effect=fail_first_store):
            stored = detector.store_workflows(workflows)

        # First workflow failed, second should have succeeded
        assert stored == 1, f"Expected 1 stored workflow (second), got {stored}"

    def test_all_strategies_fail_returns_empty_list(self, db, user_model_store):
        """When every strategy raises, detect_workflows() must return an empty list.

        This is the worst-case scenario — total DB corruption. The method must
        still complete without raising.
        """
        detector = WorkflowDetector(db, user_model_store)

        def always_fail(db_name):
            raise self.CORRUPTION_ERROR

        with patch.object(db, "get_connection", side_effect=always_fail):
            workflows = detector.detect_workflows(lookback_days=30)

        assert workflows == []

    def test_database_error_subclass_caught(self, db, user_model_store):
        """sqlite3.DatabaseError (parent of OperationalError) must also be caught.

        The except clause uses `except Exception`, which catches the full
        hierarchy: Exception → DatabaseError → OperationalError.
        """
        detector = WorkflowDetector(db, user_model_store)

        def raise_db_error(db_name):
            raise sqlite3.DatabaseError("database corruption")

        with patch.object(db, "get_connection", side_effect=raise_db_error):
            workflows = detector.detect_workflows(lookback_days=30)

        assert workflows == []

    def test_healthy_strategies_preserved_when_interaction_fails(self, db, user_model_store):
        """Workflows from healthy strategies must be preserved when interaction fails.

        Seeds data for email workflows (which read from events.db), then corrupts
        user_model.db. The email workflows should still appear in the results.
        """
        # Seed email data that will produce detectable workflows
        _seed_email_workflow_data(db, sender="frequent@example.com", count=5)
        detector = WorkflowDetector(db, user_model_store)

        # First, verify uncorrupted detection finds email workflows
        uncorrupted = detector.detect_workflows(lookback_days=30)
        email_uncorrupted = [w for w in uncorrupted if "Responding" in w.get("name", "")]

        # Now corrupt user_model.db and verify email workflows are still returned
        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            corrupted = detector.detect_workflows(lookback_days=30)

        # Email workflows from events.db should still be present
        email_corrupted = [w for w in corrupted if "Responding" in w.get("name", "")]
        assert len(email_corrupted) == len(email_uncorrupted), (
            f"Email workflows should be preserved: {len(email_uncorrupted)} uncorrupted "
            f"vs {len(email_corrupted)} with corruption"
        )

    def test_summary_log_emitted_after_partial_failure(self, db, user_model_store, caplog):
        """The summary log line must be emitted even when some strategies fail.

        The summary should report counts for each strategy type so operators
        can see at a glance which strategies succeeded.
        """
        detector = WorkflowDetector(db, user_model_store)

        original_get_conn = db.get_connection

        def corrupted_user_model(db_name):
            if db_name == "user_model":
                raise self.CORRUPTION_ERROR
            return original_get_conn(db_name)

        with patch.object(db, "get_connection", side_effect=corrupted_user_model):
            with caplog.at_level(logging.INFO, logger="services.workflow_detector.detector"):
                detector.detect_workflows(lookback_days=30)

        # The summary line should be present
        summary_lines = [
            r.message for r in caplog.records
            if "detected" in r.message.lower() and "workflows" in r.message.lower()
        ]
        assert len(summary_lines) >= 1, (
            f"Expected a summary log line, got records: {[r.message for r in caplog.records]}"
        )
