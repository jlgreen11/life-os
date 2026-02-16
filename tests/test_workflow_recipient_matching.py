"""
Tests for workflow detector recipient matching to prevent false positives.

Before this fix, the workflow detector would incorrectly attribute ALL email.sent
events within 4 hours of receiving an email as "responses" to that email, even if
they were sent to completely different recipients. This created 100% false positive
workflows for marketing emails.

After this fix, email.sent events are only counted as responses if their recipient
matches the sender of the original email.received event.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skip(reason="Workflow detection disabled pending algorithmic redesign")

from models.core import Priority
from services.workflow_detector.detector import WorkflowDetector


def test_email_workflow_requires_recipient_match(db, user_model_store):
    """Email workflows should only count email.sent as response if recipient matches sender.

    Regression test for the bug where ALL email.sent events were counted as responses,
    creating false workflows like "Responding to marketing@spam.com" with 100% success rate.
    """
    detector = WorkflowDetector(db, user_model_store)

    # Create test scenario (need 3+ instances for workflow detection):
    # - Receive 3 emails from boss@company.com
    # - Send 3 emails to friend@example.com (NOT responses to boss)
    # - Send 3 emails to boss@company.com (ARE responses to boss)

    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    with db.get_connection("events") as conn:
        for i in range(3):
            # Email received from boss
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-recv-{i}",
                "email.received",
                "proton_mail",
                (base_time + timedelta(hours=i*2)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "from_address": "boss@company.com",
                    "to_addresses": ["me@example.com"],
                    "subject": f"Task {i}",
                    "body": "Please send me the report."
                }),
                "{}"
            ))

            # Email sent to friend (unrelated - should NOT be counted)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-friend-{i}",
                "email.sent",
                "proton_mail",
                (base_time + timedelta(hours=i*2, minutes=15)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "to_addresses": ["friend@example.com"],
                    "subject": f"Chat {i}",
                    "body": "Want to grab lunch?"
                }),
                "{}"
            ))

            # Email sent to boss (actual response - SHOULD be counted)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-boss-{i}",
                "email.sent",
                "proton_mail",
                (base_time + timedelta(hours=i*2, minutes=30)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "to_addresses": ["boss@company.com"],
                    "subject": f"Re: Task {i}",
                    "body": "Here's the report you requested."
                }),
                "{}"
            ))
        conn.commit()

    # Run workflow detection
    workflows = detector.detect_workflows(lookback_days=7)

    # Should detect one workflow for boss@company.com
    boss_workflows = [w for w in workflows if "boss@company.com" in w["name"].lower()]
    assert len(boss_workflows) == 1, f"Expected 1 workflow for boss, got {len(boss_workflows)}"

    boss_wf = boss_workflows[0]
    # Should show 3 emails received, 3 email sent (only the responses to boss, NOT friend emails)
    assert boss_wf["times_observed"] == 3  # 3 emails received from boss
    assert boss_wf["success_rate"] == 1.0  # 100% response rate (3/3)


def test_workflow_prevents_marketing_email_false_positives(db, user_model_store):
    """Marketing emails should not create false-positive workflows from unrelated sent emails.

    This is the PRIMARY bug being fixed: before this patch, receiving 5 marketing emails
    and sending 1 unrelated email would create 5 workflows like "Responding to
    marketing@spam.com" with 20% success rates, all false positives.
    """
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    # Receive 5 marketing emails (each 3+ instances to trigger workflow detection)
    marketing_senders = [
        "store-news@amazon.com",
        "deals@walmart.com",
        "sales@bestbuy.com",
    ]

    with db.get_connection("events") as conn:
        for sender in marketing_senders:
            for i in range(3):
                # Marketing email received
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"evt-{sender}-{i}",
                    "email.received",
                    "proton_mail",
                    (base_time + timedelta(minutes=i*10)).isoformat(),
                    Priority.LOW.value,
                    json.dumps({
                        "from_address": sender,
                        "to_addresses": ["me@example.com"],
                        "subject": "Special offer!",
                        "body": "Buy now and save 50%!"
                    }),
                    "{}"
                ))

        # User sends 3 emails to their actual contact (NOT to any marketing sender)
        for i in range(3):
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-real-email-{i}",
                "email.sent",
                "proton_mail",
                (base_time + timedelta(hours=1, minutes=i*10)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "to_addresses": ["friend@example.com"],
                    "subject": "Let's meet up",
                    "body": "Are you free this weekend?"
                }),
                "{}"
            ))
        conn.commit()

    # Run workflow detection
    workflows = detector.detect_workflows(lookback_days=7)

    # Should NOT detect workflows for any marketing senders
    # (the sent emails were not to any of them)
    marketing_workflows = [
        w for w in workflows
        if any(sender.lower() in w["name"].lower() for sender in marketing_senders)
    ]
    assert len(marketing_workflows) == 0, (
        f"Should not detect marketing workflows (sent emails unrelated), "
        f"got {len(marketing_workflows)}: {[w['name'] for w in marketing_workflows]}"
    )


def test_workflow_without_recipient_data_excluded(db, user_model_store):
    """Email.sent events without recipient data should not be counted as responses."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    with db.get_connection("events") as conn:
        for i in range(3):
            # Email received from client
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-recv-{i}",
                "email.received",
                "proton_mail",
                (base_time + timedelta(hours=i)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "from_address": "client@corp.com",
                    "to_addresses": ["me@example.com"],
                    "subject": "Project update",
                    "body": "How's the project going?"
                }),
                "{}"
            ))

            # Email sent without recipient info (malformed/legacy data)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-sent-{i}",
                "email.sent",
                "proton_mail",
                (base_time + timedelta(hours=i, minutes=30)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "subject": "Re: Project update",
                    "body": "Going great!"
                    # Missing to_addresses field
                }),
                "{}"
            ))
        conn.commit()

    # Run workflow detection
    workflows = detector.detect_workflows(lookback_days=7)

    # Should NOT detect a workflow for client (no valid responses)
    client_workflows = [w for w in workflows if "client@corp.com" in w["name"].lower()]
    assert len(client_workflows) == 0, f"Should not detect workflow without recipient data, got {len(client_workflows)}"


def test_workflow_case_insensitive_email_matching(db, user_model_store):
    """Email matching should be case-insensitive when matching responses to senders."""
    detector = WorkflowDetector(db, user_model_store)

    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    # Use consistent sender case in received emails, but vary response case
    # This tests that response matching is case-insensitive

    with db.get_connection("events") as conn:
        for i in range(3):
            # Received email (all from same lowercase sender)
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-recv-{i}",
                "email.received",
                "proton_mail",
                (base_time + timedelta(hours=i)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "from_address": "boss@company.com",
                    "to_addresses": ["me@example.com"],
                    "subject": f"Task {i}",
                    "body": "Please handle this."
                }),
                "{}"
            ))

            # Sent response with varying case
            response_addresses = ["Boss@Company.com", "boss@COMPANY.com", "BOSS@company.COM"]
            conn.execute("""
                INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"evt-sent-{i}",
                "email.sent",
                "proton_mail",
                (base_time + timedelta(hours=i, minutes=30)).isoformat(),
                Priority.NORMAL.value,
                json.dumps({
                    "to_addresses": [response_addresses[i]],
                    "subject": f"Re: Task {i}",
                    "body": "Done."
                }),
                "{}"
            ))
        conn.commit()

    # Run workflow detection
    workflows = detector.detect_workflows(lookback_days=7)

    # Should detect workflow (responses matched despite case differences)
    boss_workflows = [w for w in workflows if "boss@company.com" in w["name"].lower()]
    assert len(boss_workflows) == 1, f"Expected 1 workflow, got {len(boss_workflows)}"

    boss_wf = boss_workflows[0]
    assert boss_wf["times_observed"] == 3  # 3 emails received
    assert boss_wf["success_rate"] == 1.0  # 100% response rate (all 3 matched despite case differences)
