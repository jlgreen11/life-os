"""
Tests for communication template backfill script.

Verifies that historical communication events are correctly processed to
generate communication templates with proper style extraction (greetings,
closings, formality, emoji usage, etc.).
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

# Add scripts directory to path for importing backfill module
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backfill_communication_templates import backfill_communication_templates
from storage.manager import DatabaseManager


@pytest.fixture
def db_with_communication_events(tmp_path):
    """Fixture providing a database with sample communication events."""
    db = DatabaseManager(str(tmp_path))

    # Initialize all database schemas (events, user_model, etc.)
    db._init_events_db()
    db._init_user_model_db()

    # Create events in the events database
    with db.get_connection("events") as conn:
        events = [
            # Outbound email with greeting and closing
            {
                "id": "email-out-1",
                "type": "email.sent",
                "source": "google",
                "timestamp": "2026-02-10T09:00:00.000Z",
                "priority": 0.5,
                "payload": json.dumps({
                    "to_addresses": ["alice@example.com"],
                    "from_address": "user@example.com",
                    "subject": "Project update",
                    "body": "Hi Alice,\n\nJust wanted to give you a quick update on the project. Everything is on track!\n\nThanks,\nJeremy",
                    "body_plain": "Hi Alice,\n\nJust wanted to give you a quick update on the project. Everything is on track!\n\nThanks,\nJeremy",
                }),
                "metadata": json.dumps({}),
            },
            # Inbound email with formal greeting
            {
                "id": "email-in-1",
                "type": "email.received",
                "source": "google",
                "timestamp": "2026-02-10T10:00:00.000Z",
                "priority": 0.5,
                "payload": json.dumps({
                    "from_address": "bob@company.com",
                    "to_addresses": ["user@example.com"],
                    "subject": "Meeting request",
                    "body": "Dear Mr. Greenwood,\n\nI would like to schedule a meeting to discuss the proposal.\n\nBest regards,\nBob Smith",
                    "body_plain": "Dear Mr. Greenwood,\n\nI would like to schedule a meeting to discuss the proposal.\n\nBest regards,\nBob Smith",
                }),
                "metadata": json.dumps({}),
            },
            # Outbound message with emoji
            {
                "id": "msg-out-1",
                "type": "message.sent",
                "source": "slack",
                "timestamp": "2026-02-10T11:00:00.000Z",
                "priority": 0.5,
                "payload": json.dumps({
                    "to_addresses": ["charlie@example.com"],
                    "from_address": "user@example.com",
                    "body": "Hey! 🎉 Just deployed the new feature. Let me know what you think!",
                    "body_plain": "Hey! 🎉 Just deployed the new feature. Let me know what you think!",
                }),
                "metadata": json.dumps({}),
            },
            # Multiple recipients (should create multiple templates)
            {
                "id": "email-out-2",
                "type": "email.sent",
                "source": "google",
                "timestamp": "2026-02-10T12:00:00.000Z",
                "priority": 0.5,
                "payload": json.dumps({
                    "to_addresses": ["alice@example.com", "dave@example.com"],
                    "from_address": "user@example.com",
                    "subject": "Team update",
                    "body": "Hello team,\n\nHere's the weekly update. Great progress this week!\n\nCheers,\nJeremy",
                    "body_plain": "Hello team,\n\nHere's the weekly update. Great progress this week!\n\nCheers,\nJeremy",
                }),
                "metadata": json.dumps({}),
            },
            # Very short message (should be skipped - less than 10 chars)
            {
                "id": "msg-skip-1",
                "type": "message.sent",
                "source": "slack",
                "timestamp": "2026-02-10T13:00:00.000Z",
                "priority": 0.5,
                "payload": json.dumps({
                    "to_addresses": ["eve@example.com"],
                    "body": "ok",
                    "body_plain": "ok",
                }),
                "metadata": json.dumps({}),
            },
        ]

        for event in events:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event["priority"],
                    event["payload"],
                    event["metadata"],
                ),
            )

    return db


class TestCommunicationTemplateBackfill:
    """Test suite for communication template backfill functionality."""

    def test_backfill_creates_templates_from_historical_events(
        self, db_with_communication_events
    ):
        """Verify backfill processes historical events and creates templates."""
        db = db_with_communication_events

        # Verify no templates exist initially
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]
            assert count == 0, "Should start with no templates"

        # Run backfill
        stats = backfill_communication_templates(
            data_dir=db.data_dir,
            batch_size=10,
        )

        # Verify events were processed
        assert stats["events_processed"] == 4, "Should process 4 valid events (skip <10 char)"
        assert stats["errors"] == 0, "Should have no errors"
        assert stats["templates_created"] > 0, "Should create at least one template"

        # Verify templates were created
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]
            # Expected templates:
            # - alice@example.com:google:out (2 samples merged)
            # - bob@company.com:google:in (1 sample)
            # - charlie@example.com:slack:out (1 sample)
            # - dave@example.com:google:out (1 sample)
            assert count >= 4, f"Should have at least 4 templates, got {count}"

    def test_backfill_extracts_greeting_and_closing(self, db_with_communication_events):
        """Verify templates capture greeting and closing phrases."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        # Check template for alice@example.com (outbound email)
        with db.get_connection("user_model") as conn:
            conn.row_factory = sqlite3.Row
            templates = conn.execute(
                """SELECT * FROM communication_templates
                   WHERE contact_id = 'alice@example.com'
                     AND context = 'user_to_contact'"""
            ).fetchall()

            assert len(templates) > 0, "Should have template for Alice"
            template = templates[0]

            # First email has "Hi Alice", second has "Hello team"
            # Template should capture one of these greetings
            assert template["greeting"] is not None, "Should extract greeting"
            assert template["greeting"].lower() in ["hi", "hello"], \
                f"Greeting should be Hi or Hello, got: {template['greeting']}"

            # Both emails have closings (Thanks/Cheers)
            assert template["closing"] is not None, "Should extract closing"

    def test_backfill_calculates_formality_level(self, db_with_communication_events):
        """Verify formality scoring distinguishes casual from formal messages."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            conn.row_factory = sqlite3.Row

            # Check casual outbound template (alice@example.com - casual style)
            casual = conn.execute(
                """SELECT formality FROM communication_templates
                   WHERE contact_id = 'alice@example.com'
                     AND context = 'user_to_contact'"""
            ).fetchone()

            # Check formal inbound template (bob@company.com - "Dear Mr.", "Best regards")
            formal = conn.execute(
                """SELECT formality FROM communication_templates
                   WHERE contact_id = 'bob@company.com'
                     AND context = 'contact_to_user'"""
            ).fetchone()

            if casual and formal:
                # Formal message should have higher formality score
                assert formal["formality"] > casual["formality"], \
                    f"Formal ({formal['formality']}) should be > casual ({casual['formality']})"

    def test_backfill_detects_emoji_usage(self, db_with_communication_events):
        """Verify emoji detection in messages."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            conn.row_factory = sqlite3.Row

            # Charlie's message has emoji
            template = conn.execute(
                """SELECT uses_emoji FROM communication_templates
                   WHERE contact_id = 'charlie@example.com'
                     AND context = 'user_to_contact'"""
            ).fetchone()

            assert template is not None, "Should have template for Charlie"
            assert template["uses_emoji"] == 1, "Should detect emoji usage"

    def test_backfill_tracks_sample_count(self, db_with_communication_events):
        """Verify sample count tracks number of messages analyzed."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            conn.row_factory = sqlite3.Row

            # Alice received 2 messages (email-out-1 and email-out-2)
            template = conn.execute(
                """SELECT samples_analyzed FROM communication_templates
                   WHERE contact_id = 'alice@example.com'
                     AND context = 'user_to_contact'"""
            ).fetchone()

            assert template is not None, "Should have template for Alice"
            assert template["samples_analyzed"] == 2, \
                f"Should have analyzed 2 samples, got {template['samples_analyzed']}"

    def test_backfill_separates_inbound_and_outbound_templates(
        self, db_with_communication_events
    ):
        """Verify separate templates for user→contact and contact→user."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            conn.row_factory = sqlite3.Row

            # Check that Bob's inbound message creates contact_to_user template
            inbound = conn.execute(
                """SELECT context FROM communication_templates
                   WHERE contact_id = 'bob@company.com'"""
            ).fetchall()

            assert len(inbound) == 1, "Should have exactly one template for Bob"
            assert inbound[0]["context"] == "contact_to_user", \
                "Bob's template should be contact_to_user (inbound)"

            # Check that Alice's outbound messages create user_to_contact template
            outbound = conn.execute(
                """SELECT context FROM communication_templates
                   WHERE contact_id = 'alice@example.com'"""
            ).fetchall()

            assert len(outbound) == 1, "Should have exactly one template for Alice"
            assert outbound[0]["context"] == "user_to_contact", \
                "Alice's template should be user_to_contact (outbound)"

    def test_backfill_respects_limit_parameter(self, db_with_communication_events):
        """Verify limit parameter caps number of events processed."""
        db = db_with_communication_events

        stats = backfill_communication_templates(
            data_dir=db.data_dir,
            limit=2,
        )

        # Should process only first 2 valid events
        assert stats["events_processed"] == 2, \
            f"Should process exactly 2 events, got {stats['events_processed']}"

    def test_backfill_is_idempotent(self, db_with_communication_events):
        """Verify running backfill multiple times produces same result."""
        db = db_with_communication_events

        # Run backfill first time
        stats1 = backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            count_after_first = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]

        # Run backfill again (should update existing templates, not duplicate)
        stats2 = backfill_communication_templates(data_dir=db.data_dir)

        with db.get_connection("user_model") as conn:
            count_after_second = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]

        # Template count should remain the same (no duplicates)
        assert count_after_first == count_after_second, \
            "Backfill should be idempotent - no duplicate templates"

        # Both runs should process the same number of events
        assert stats1["events_processed"] == stats2["events_processed"], \
            "Should process same events on repeat run"

    def test_backfill_dry_run_mode(self, db_with_communication_events):
        """Verify dry-run mode doesn't modify database."""
        db = db_with_communication_events

        stats = backfill_communication_templates(
            data_dir=db.data_dir,
            dry_run=True,
        )

        # Should report events would be processed
        assert stats["events_processed"] == 4, "Should report processing 4 events"

        # But no templates should be created
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]
            assert count == 0, "Dry run should not create templates"

    def test_backfill_handles_multi_recipient_emails(self, db_with_communication_events):
        """Verify emails with multiple recipients create separate templates."""
        db = db_with_communication_events

        backfill_communication_templates(data_dir=db.data_dir)

        # email-out-2 has two recipients: alice and dave
        # Both should get templates (alice's is updated, dave's is new)
        with db.get_connection("user_model") as conn:
            alice_template = conn.execute(
                """SELECT samples_analyzed FROM communication_templates
                   WHERE contact_id = 'alice@example.com'"""
            ).fetchone()

            dave_template = conn.execute(
                """SELECT samples_analyzed FROM communication_templates
                   WHERE contact_id = 'dave@example.com'"""
            ).fetchone()

            assert alice_template is not None, "Should have template for Alice"
            assert dave_template is not None, "Should have template for Dave"

            # Alice gets 2 samples (email-out-1 + email-out-2)
            assert alice_template[0] == 2, "Alice should have 2 samples"

            # Dave gets 1 sample (email-out-2)
            assert dave_template[0] == 1, "Dave should have 1 sample"
