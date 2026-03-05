"""
Tests for email_to JSON matching fix in workflow detector.

Validates that the WorkflowDetector correctly parses email_to values stored
as JSON arrays (e.g. '["alice@example.com"]') from the denormalized events
column, and performs case-insensitive exact matching against sender addresses.
"""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from services.workflow_detector import WorkflowDetector


@pytest.fixture
def workflow_detector(db, user_model_store):
    """Create a WorkflowDetector with lowered thresholds for testing."""
    detector = WorkflowDetector(db, user_model_store)
    # Lower thresholds so small test datasets can trigger detection
    detector.min_occurrences = 3
    detector.min_completions = 2
    return detector


def _insert_email_pair(conn, sender, email_to_value, base_time, offset_days):
    """Insert an email.received + email.sent pair with explicit email_from/email_to.

    Args:
        conn: SQLite connection to events.db
        sender: The sender email address (used for email_from on received events)
        email_to_value: The value to store in email_to column (JSON array or plain string)
        base_time: Base datetime for the events
        offset_days: Day offset from base_time
    """
    event_time = base_time + timedelta(days=offset_days)

    # email.received from the sender
    conn.execute("""
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid4()), "email.received", "protonmail", event_time.isoformat(), 3,
        json.dumps({"from_address": sender, "subject": f"Message {offset_days}"}),
        json.dumps({}),
        sender.lower(),
    ))

    # email.sent responding to the sender (~1 hour later)
    response_time = event_time + timedelta(hours=1)
    conn.execute("""
        INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid4()), "email.sent", "protonmail", response_time.isoformat(), 3,
        json.dumps({"to_addresses": [sender], "subject": f"Re: Message {offset_days}"}),
        json.dumps({}),
        email_to_value,
    ))


class TestEmailToJsonArrayMatching:
    """Test that email_to stored as JSON array is correctly parsed and matched."""

    def test_json_array_format_matches_sender(self, workflow_detector, db):
        """email_to stored as '["alice@example.com"]' should match sender alice@example.com."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "alice@example.com"
        email_to_json = json.dumps([sender])  # '["alice@example.com"]'

        with db.get_connection("events") as conn:
            for i in range(4):
                _insert_email_pair(conn, sender, email_to_json, base_time, i)

        workflows = workflow_detector._detect_email_workflows(lookback_days=30)

        # Should detect a workflow for alice
        alice_workflows = [w for w in workflows if "alice" in w["name"].lower()]
        assert len(alice_workflows) == 1, f"Expected 1 workflow for alice, got {len(alice_workflows)}: {workflows}"
        assert alice_workflows[0]["times_observed"] == 4

    def test_case_insensitive_matching(self, workflow_detector, db):
        """Sender 'Alice@Example.COM' should match email_to '["alice@example.com"]'."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender_in_received = "Alice@Example.COM"
        email_to_json = json.dumps(["alice@example.com"])

        with db.get_connection("events") as conn:
            for i in range(4):
                event_time = base_time + timedelta(days=i)

                # email.received with mixed-case sender stored as email_from
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid4()), "email.received", "protonmail", event_time.isoformat(), 3,
                    json.dumps({"from_address": sender_in_received}),
                    json.dumps({}),
                    sender_in_received,  # Mixed case in email_from
                ))

                # email.sent with lowercase email_to
                response_time = event_time + timedelta(hours=1)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_to)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid4()), "email.sent", "protonmail", response_time.isoformat(), 3,
                    json.dumps({"to_addresses": ["alice@example.com"]}),
                    json.dumps({}),
                    email_to_json,
                ))

        workflows = workflow_detector._detect_email_workflows(lookback_days=30)

        # Should detect workflow despite case mismatch
        alice_workflows = [w for w in workflows if "alice" in w["name"].lower()]
        assert len(alice_workflows) == 1, f"Expected 1 workflow for alice, got {len(alice_workflows)}: {workflows}"

    def test_plain_string_format_still_works(self, workflow_detector, db):
        """email_to stored as plain string 'bob@example.com' should still match."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "bob@example.com"
        email_to_plain = sender  # Plain string, not JSON array

        with db.get_connection("events") as conn:
            for i in range(4):
                _insert_email_pair(conn, sender, email_to_plain, base_time, i)

        workflows = workflow_detector._detect_email_workflows(lookback_days=30)

        bob_workflows = [w for w in workflows if "bob" in w["name"].lower()]
        assert len(bob_workflows) == 1, f"Expected 1 workflow for bob, got {len(bob_workflows)}: {workflows}"

    def test_partial_address_does_not_match(self, workflow_detector, db):
        """Sender 'alice@ex.com' should NOT match email_to '["alice@example.com"]'."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        real_sender = "alice@ex.com"
        email_to_json = json.dumps(["alice@example.com"])  # Different domain

        with db.get_connection("events") as conn:
            for i in range(4):
                event_time = base_time + timedelta(days=i)

                # email.received from alice@ex.com
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_from)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid4()), "email.received", "protonmail", event_time.isoformat(), 3,
                    json.dumps({"from_address": real_sender}),
                    json.dumps({}),
                    real_sender,
                ))

                # email.sent to alice@example.com (different domain)
                response_time = event_time + timedelta(hours=1)
                conn.execute("""
                    INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, email_to)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid4()), "email.sent", "protonmail", response_time.isoformat(), 3,
                    json.dumps({"to_addresses": ["alice@example.com"]}),
                    json.dumps({}),
                    email_to_json,
                ))

        workflows = workflow_detector._detect_email_workflows(lookback_days=30)

        # alice@ex.com should NOT match alice@example.com
        ex_workflows = [w for w in workflows if "alice@ex.com" in w["name"]]
        assert len(ex_workflows) == 0, (
            f"Partial address alice@ex.com should not match alice@example.com, but got: {ex_workflows}"
        )

    def test_multi_recipient_json_array(self, workflow_detector, db):
        """email_to with multiple recipients should match if sender is among them."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "carol@example.com"
        # JSON array with multiple recipients including the sender
        email_to_json = json.dumps(["other@example.com", "carol@example.com", "third@example.com"])

        with db.get_connection("events") as conn:
            for i in range(4):
                _insert_email_pair(conn, sender, email_to_json, base_time, i)

        workflows = workflow_detector._detect_email_workflows(lookback_days=30)

        carol_workflows = [w for w in workflows if "carol" in w["name"].lower()]
        assert len(carol_workflows) == 1, f"Expected 1 workflow for carol, got {len(carol_workflows)}: {workflows}"

    def test_malformed_json_email_to_does_not_crash(self, workflow_detector, db):
        """Malformed JSON in email_to should not crash detection."""
        base_time = datetime.now(timezone.utc) - timedelta(days=15)
        sender = "dave@example.com"
        email_to_bad = '[invalid json"'  # Malformed JSON

        with db.get_connection("events") as conn:
            for i in range(4):
                _insert_email_pair(conn, sender, email_to_bad, base_time, i)

        # Should not crash
        workflows = workflow_detector._detect_email_workflows(lookback_days=30)
        assert isinstance(workflows, list)
