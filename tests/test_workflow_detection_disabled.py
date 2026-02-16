"""
Tests for workflow detection performance fix.

This test suite verifies that workflow detection is properly disabled to prevent
continuous improvement loop hangs caused by expensive queries on 800K+ events.
"""

import pytest
from datetime import datetime, timedelta, timezone

from services.workflow_detector.detector import WorkflowDetector


def test_workflow_detection_completes_instantly(db, user_model_store):
    """Verify workflow detection completes in <100ms (was timing out at 30s+)."""
    detector = WorkflowDetector(db, user_model_store)

    import time
    start = time.time()
    workflows = detector.detect_workflows(lookback_days=30)
    elapsed = time.time() - start

    # Should complete instantly since it's disabled
    assert elapsed < 0.1, f"Workflow detection took {elapsed:.3f}s, expected <0.1s"
    assert workflows == [], "Should return empty list when disabled"


def test_workflow_detection_returns_empty_list(db, user_model_store):
    """Verify workflow detection returns empty list when disabled."""
    detector = WorkflowDetector(db, user_model_store)

    workflows = detector.detect_workflows(lookback_days=30)

    assert isinstance(workflows, list), "Should return a list"
    assert len(workflows) == 0, "Should return empty list when disabled"


def test_workflow_detection_with_different_lookback_periods(db, user_model_store):
    """Verify workflow detection is disabled regardless of lookback period."""
    detector = WorkflowDetector(db, user_model_store)

    for lookback_days in [1, 7, 30, 90]:
        workflows = detector.detect_workflows(lookback_days=lookback_days)
        assert workflows == [], f"Should return empty list for {lookback_days} day lookback"


def test_workflow_detection_does_not_crash_system(db, user_model_store):
    """Verify workflow detection doesn't crash when called multiple times."""
    detector = WorkflowDetector(db, user_model_store)

    # Call multiple times to ensure it's safe
    for _ in range(5):
        workflows = detector.detect_workflows(lookback_days=30)
        assert workflows == [], "Should consistently return empty list"


def test_workflow_storage_with_empty_list(db, user_model_store):
    """Verify storing an empty workflow list doesn't cause errors."""
    detector = WorkflowDetector(db, user_model_store)

    workflows = detector.detect_workflows(lookback_days=30)
    stored_count = detector.store_workflows(workflows)

    assert stored_count == 0, "Should store 0 workflows when list is empty"


def test_individual_detection_methods_still_exist(db, user_model_store):
    """Verify individual detection methods still exist for future re-enabling."""
    detector = WorkflowDetector(db, user_model_store)

    # These methods should still exist even though they're not called
    assert hasattr(detector, '_detect_email_workflows'), "Email workflow method should exist"
    assert hasattr(detector, '_detect_task_workflows'), "Task workflow method should exist"
    assert hasattr(detector, '_detect_calendar_workflows'), "Calendar workflow method should exist"
    assert hasattr(detector, '_detect_interaction_workflows'), "Interaction workflow method should exist"


def test_workflow_detector_initialization(db, user_model_store):
    """Verify workflow detector initializes correctly."""
    detector = WorkflowDetector(db, user_model_store)

    assert detector.db is db, "Should store database reference"
    assert detector.user_model_store is user_model_store, "Should store user model store reference"
    assert detector.min_occurrences == 3, "Should have default min_occurrences threshold"
    assert detector.max_step_gap_hours == 4, "Should have default max_step_gap_hours threshold"
    assert detector.min_steps == 2, "Should have default min_steps threshold"
    assert detector.success_threshold == 0.01, "Should have 1% success threshold"


def test_workflow_detection_with_large_event_volume(db, user_model_store, event_store):
    """Verify workflow detection handles large event volumes gracefully."""
    # Insert a large batch of events to simulate production load
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    # Add 1000 test events
    for i in range(1000):
        timestamp = cutoff + timedelta(hours=i)
        event_store.store_event({
            "id": f"test-{i}",
            "type": "email.received",
            "source": "test",
            "timestamp": timestamp.isoformat(),
            "priority": "normal",
            "payload": {"from_address": f"sender{i % 10}@example.com", "subject": f"Test {i}"},
            "metadata": {}
        })

    detector = WorkflowDetector(db, user_model_store)

    import time
    start = time.time()
    workflows = detector.detect_workflows(lookback_days=30)
    elapsed = time.time() - start

    # Should still complete instantly even with extra events
    assert elapsed < 0.1, f"Should complete fast even with 1K extra events, took {elapsed:.3f}s"
    assert workflows == [], "Should return empty list even with events present"


def test_workflow_detection_logging(db, user_model_store, caplog):
    """Verify workflow detection logs appropriate message."""
    import logging
    caplog.set_level(logging.INFO)

    detector = WorkflowDetector(db, user_model_store)
    workflows = detector.detect_workflows(lookback_days=30)

    # Should log that detection is skipped
    assert any("skipped" in record.message.lower() for record in caplog.records), \
        "Should log that workflow detection is skipped"
    assert any("performance" in record.message.lower() for record in caplog.records), \
        "Should mention performance as the reason"
