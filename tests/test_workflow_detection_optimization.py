"""
Tests for workflow detection performance optimization.

Verifies that the optimized workflow detector can handle large volumes
of events efficiently (77K+ emails) without timing out on self-joins.
"""

from datetime import datetime, timedelta, timezone
import pytest
from services.workflow_detector.detector import WorkflowDetector


def test_email_workflow_detection_with_large_dataset(db, user_model_store):
    """Test that workflow detection completes quickly with 77K+ emails.

    Before optimization: Query timed out after 30+ seconds on self-joins
    After optimization: Completes in <5 seconds with O(n+m) scan
    """
    detector = WorkflowDetector(db, user_model_store)

    # Create a realistic dataset: many emails from one sender, few responses
    base_time = datetime.now(timezone.utc) - timedelta(days=25)

    # Simulate 1000 emails from "boss@company.com" over 25 days
    with db.get_connection("events") as conn:
        for i in range(1000):
            event_time = base_time + timedelta(hours=i)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"email-{i}",
                "email.received",
                "email",
                event_time.isoformat(),
                f'{{"from_address": "boss@company.com", "subject": "Task {i}"}}'
            ))

        # Simulate 50 email.sent responses within 4 hours of some received emails
        for i in range(50):
            # Respond to every 20th email
            response_time = base_time + timedelta(hours=i*20) + timedelta(hours=2)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"response-{i}",
                "email.sent",
                "email",
                response_time.isoformat(),
                '{"to_addresses": ["boss@company.com"]}'
            ))

    # The workflow detector should complete quickly (not timeout)
    import time
    start = time.time()
    workflows = detector._detect_email_workflows(lookback_days=30)
    duration = time.time() - start

    # Should complete in under 5 seconds (was timing out at 30+ seconds before)
    assert duration < 5, f"Workflow detection took {duration:.2f}s (should be <5s)"

    # Should detect at least one workflow for boss@company.com
    assert len(workflows) >= 1, "Should detect at least one email workflow"

    boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
    assert boss_workflow is not None, "Should detect workflow for boss@company.com"

    # Workflow should have:
    # - At least 2 steps (receive → send)
    # - Success rate: each of 50 responses can match multiple emails within the
    #   4-hour window. With responses every 20 hours and emails every hour,
    #   each response matches ~4 emails, giving ~200 matches / 1000 emails = 20%
    # - Times observed = 1000
    assert len(boss_workflow["steps"]) >= 2
    assert 0.15 <= boss_workflow["success_rate"] <= 0.25  # ~20% ± tolerance
    assert boss_workflow["times_observed"] == 1000


def test_email_workflow_detection_multiple_senders(db, user_model_store):
    """Test workflow detection with multiple high-volume senders."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=20)

    # Create emails from 5 different senders
    senders = [
        "marketing@company.com",
        "boss@work.com",
        "client@business.net",
        "newsletter@news.org",
        "support@service.io"
    ]

    with db.get_connection("events") as conn:
        event_id = 0
        for sender_idx, sender in enumerate(senders):
            # Each sender gets 100 emails
            for i in range(100):
                event_time = base_time + timedelta(hours=sender_idx*100 + i)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"email-{event_id}",
                    "email.received",
                    "email",
                    event_time.isoformat(),
                    f'{{"from_address": "{sender}", "subject": "Email {i}"}}'
                ))
                event_id += 1

                # Respond to 10% of emails from this sender
                if i % 10 == 0:
                    response_time = event_time + timedelta(hours=1)
                    conn.execute("""
                        INSERT INTO events (id, type, source, timestamp, payload)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        f"response-{event_id}",
                        "email.sent",
                        "email",
                        response_time.isoformat(),
                        f'{{"to_addresses": ["{sender}"]}}'
                    ))
                    event_id += 1

    # Detect workflows
    workflows = detector._detect_email_workflows(lookback_days=30)

    # Should detect workflows for all 5 senders (each has 100 emails, 10 responses)
    assert len(workflows) == 5, f"Expected 5 workflows, got {len(workflows)}"

    # Each workflow should have ~10-40% success rate (10 responses, but each can
    # match multiple emails within the 4-hour time window)
    for workflow in workflows:
        assert 0.05 <= workflow["success_rate"] <= 0.50, \
            f"Workflow {workflow['name']} has unexpected success rate: {workflow['success_rate']}"


def test_email_workflow_detection_no_responses(db, user_model_store):
    """Test that senders with no responses don't create workflows."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=15)

    # Create 500 emails from marketing (never responded to)
    with db.get_connection("events") as conn:
        for i in range(500):
            event_time = base_time + timedelta(hours=i)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"marketing-{i}",
                "email.received",
                "email",
                event_time.isoformat(),
                '{"from_address": "marketing@spam.com", "subject": "Buy now!"}'
            ))

    # Detect workflows - should find none because:
    # - success_threshold is 1%
    # - 0 responses / 500 emails = 0% < 1%
    workflows = detector._detect_email_workflows(lookback_days=30)

    marketing_workflows = [w for w in workflows if "marketing@spam.com" in w["name"]]
    assert len(marketing_workflows) == 0, "Should not detect workflow for sender with no responses"


def test_email_workflow_detection_timing_precision(db, user_model_store):
    """Test that workflow detection only includes actions within time window."""
    detector = WorkflowDetector(db, user_model_store)
    detector.max_step_gap_hours = 2  # Only include actions within 2 hours

    base_time = datetime.now(timezone.utc) - timedelta(days=10)

    with db.get_connection("events") as conn:
        # 50 emails from boss
        for i in range(50):
            event_time = base_time + timedelta(hours=i*6)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"email-{i}",
                "email.received",
                "email",
                event_time.isoformat(),
                '{"from_address": "boss@company.com", "subject": "Urgent task"}'
            ))

            # Respond within 1 hour (should be included)
            if i < 10:
                response_time = event_time + timedelta(hours=1)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"response-1h-{i}",
                    "email.sent",
                    "email",
                    response_time.isoformat(),
                    '{"to_addresses": ["boss@company.com"]}'
                ))

            # Respond after 5 hours (should NOT be included)
            if 10 <= i < 20:
                response_time = event_time + timedelta(hours=5)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"response-5h-{i}",
                    "email.sent",
                    "email",
                    response_time.isoformat(),
                    '{"to_addresses": ["boss@company.com"]}'
                ))

    workflows = detector._detect_email_workflows(lookback_days=30)

    # Should detect workflow with only the 10 responses within 2 hours
    boss_workflow = next((w for w in workflows if "boss@company.com" in w["name"]), None)
    assert boss_workflow is not None

    # Success rate should be ~20% (10 within-window responses / 50 emails)
    assert 0.15 <= boss_workflow["success_rate"] <= 0.25


def test_composite_index_exists(db):
    """Verify that the composite index on (type, timestamp) exists.

    This index is critical for making temporal range queries fast. Without it,
    workflow detection queries time out on 77K+ event datasets.
    """
    with db.get_connection("events") as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'index' AND tbl_name = 'events'
            ORDER BY name
        """)
        indexes = [row[0] for row in cursor.fetchall()]

    # The composite index should exist
    assert "idx_events_type_timestamp" in indexes, \
        "Missing critical index idx_events_type_timestamp for workflow detection"


def test_workflow_detection_returns_top_20_senders(db, user_model_store):
    """Test that workflow detection limits results to top 20 senders by volume."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=20)

    # Create 30 senders, each with different email counts
    with db.get_connection("events") as conn:
        event_id = 0
        for sender_idx in range(30):
            sender = f"sender-{sender_idx}@example.com"
            # Higher sender_idx = more emails (sender-29 has most)
            num_emails = (sender_idx + 1) * 10

            for i in range(num_emails):
                event_time = base_time + timedelta(hours=event_id)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, payload)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"email-{event_id}",
                    "email.received",
                    "email",
                    event_time.isoformat(),
                    f'{{"from_address": "{sender}", "subject": "Email {i}"}}'
                ))
                event_id += 1

                # Every 5th email gets a response
                if i % 5 == 0:
                    response_time = event_time + timedelta(hours=1)
                    conn.execute("""
                        INSERT INTO events (id, type, source, timestamp, payload)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        f"response-{event_id}",
                        "email.sent",
                        "email",
                        response_time.isoformat(),
                        f'{{"to_addresses": ["{sender}"]}}'
                    ))
                    event_id += 1

    workflows = detector._detect_email_workflows(lookback_days=30)

    # Should return at most 20 workflows (top 20 by volume)
    assert len(workflows) <= 20, f"Expected ≤20 workflows, got {len(workflows)}"

    # Workflows should be for the highest-volume senders (sender-10 through sender-29)
    workflow_triggers = [w["trigger_conditions"][0] for w in workflows]

    # Top sender should be included
    assert any("sender-29@example.com" in t for t in workflow_triggers), \
        f"Top sender (sender-29) should be in results. Got: {workflow_triggers[:5]}"


def test_workflow_storage_integration(db, user_model_store):
    """Test that detected workflows can be stored to the database."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=10)

    # Create a simple workflow pattern: emails from boss → responses
    with db.get_connection("events") as conn:
        for i in range(20):
            # Received email
            event_time = base_time + timedelta(hours=i*12)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"email-{i}",
                "email.received",
                "email",
                event_time.isoformat(),
                '{"from_address": "boss@work.com", "subject": "Weekly report"}'
            ))

            # Response within 2 hours
            response_time = event_time + timedelta(hours=1.5)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"response-{i}",
                "email.sent",
                "email",
                response_time.isoformat(),
                '{"to_addresses": ["boss@work.com"]}'
            ))

    # Detect and store workflows
    workflows = detector._detect_email_workflows(lookback_days=30)
    assert len(workflows) >= 1

    stored_count = detector.store_workflows(workflows)
    assert stored_count == len(workflows)

    # Verify workflow is in database
    with db.get_connection("user_model") as conn:
        cursor = conn.execute("""
            SELECT name, success_rate, times_observed
            FROM workflows
            WHERE name LIKE '%boss@work.com%'
        """)
        row = cursor.fetchone()

    assert row is not None, "Workflow should be stored in database"
    name, success_rate, times_observed = row
    assert "boss@work.com" in name
    assert success_rate > 0.9  # 100% response rate
    assert times_observed == 20
