"""
Tests for scripts/backfill_reminder_contacts.py

Verifies the contact extraction and backfill logic for reminder predictions.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.backfill_reminder_contacts import (
    backfill_reminder_contacts,
    extract_contact_info,
)


class TestExtractContactInfo:
    """Test contact extraction from reminder descriptions."""

    def test_extract_simple_email(self):
        """Extract email from standard 'Unreplied message from EMAIL' format."""
        desc = 'Unreplied message from alice@example.com: "Meeting notes" (3 hours ago)'
        result = extract_contact_info(desc)
        assert result == {"contact_email": "alice@example.com"}

    def test_extract_complex_email(self):
        """Extract complex email addresses with dots, dashes, and plus signs."""
        desc = "Unreplied message from john.doe+work@company-name.co.uk: Subject"
        result = extract_contact_info(desc)
        assert result == {"contact_email": "john.doe+work@company-name.co.uk"}

    def test_extract_email_case_insensitive(self):
        """Email extraction should be case-insensitive for trigger words."""
        desc = 'UNREPLIED MESSAGE FROM bob@test.com: "Important"'
        result = extract_contact_info(desc)
        assert result == {"contact_email": "bob@test.com"}

    def test_extract_name_reply_to(self):
        """Extract capitalized name from 'Reply to NAME' format."""
        desc = "Reply to Alice about the project deadline"
        result = extract_contact_info(desc)
        assert result == {"contact_name": "Alice"}

    def test_extract_name_follow_up(self):
        """Extract name from 'Follow up with NAME' format."""
        desc = "Follow up with Bob Smith regarding the meeting"
        result = extract_contact_info(desc)
        assert result == {"contact_name": "Bob Smith"}

    def test_extract_name_message(self):
        """Extract name from 'Message NAME' format."""
        desc = "Message Grace about dinner plans"
        result = extract_contact_info(desc)
        assert result == {"contact_name": "Grace"}

    def test_name_requires_capitalization(self):
        """Name extraction should reject lowercase words (avoid false matches)."""
        desc = "Reply to the office about the documents"
        result = extract_contact_info(desc)
        # 'the' is lowercase, should not match
        assert result == {}

    def test_name_two_words(self):
        """Handle two-word capitalized names."""
        desc = "Follow up with John Smith about the contract"
        result = extract_contact_info(desc)
        assert result == {"contact_name": "John Smith"}

    def test_email_takes_precedence(self):
        """When both email and name patterns exist, email is preferred."""
        desc = 'Unreplied message from alice@example.com: "Reply to Bob about project"'
        result = extract_contact_info(desc)
        # Should extract email, not the name "Bob" from the subject line
        assert result == {"contact_email": "alice@example.com"}

    def test_no_match(self):
        """Return empty dict when no contact info can be extracted."""
        desc = "Task: Prepare slides for tomorrow's presentation"
        result = extract_contact_info(desc)
        assert result == {}

    def test_empty_description(self):
        """Handle empty description gracefully."""
        result = extract_contact_info("")
        assert result == {}

    def test_partial_email_no_match(self):
        """Reject partial email addresses (no TLD)."""
        desc = "Unreplied message from user@domain: test"
        result = extract_contact_info(desc)
        # Missing TLD (.com, .org, etc.) — should not match
        assert result == {}

    def test_real_world_example_1(self):
        """Test with real-world prediction description format."""
        desc = 'Unreplied message from support@company.com: "Your ticket #12345" (5 hours ago)'
        result = extract_contact_info(desc)
        assert result == {"contact_email": "support@company.com"}

    def test_real_world_example_2(self):
        """Test with another common format."""
        desc = "Unreplied message from team-lead@startup.io: Weekly sync (1 day ago)"
        result = extract_contact_info(desc)
        assert result == {"contact_email": "team-lead@startup.io"}


class TestBackfillReminderContacts:
    """Test the backfill function with a temporary database."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory with a test user_model.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "user_model.db"

            # Create minimal schema
            conn = sqlite3.connect(db_path)
            conn.execute(
                """CREATE TABLE predictions (
                    id TEXT PRIMARY KEY,
                    prediction_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_gate TEXT NOT NULL,
                    supporting_signals TEXT DEFAULT '[]',
                    was_surfaced INTEGER DEFAULT 0,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.commit()
            conn.close()

            yield tmpdir

    def test_backfill_updates_empty_signals(self, temp_db_dir):
        """Backfill should update predictions with supporting_signals = '[]'."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        # Insert test prediction with empty signals
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-1",
                "reminder",
                'Unreplied message from alice@test.com: "Hello" (2 hours ago)',
                0.75,
                "suggest",
                "[]",  # Old list format
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        # Run backfill
        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        # Check stats
        assert stats["updated"] == 1
        assert stats["email_only"] == 1
        assert stats["unresolvable"] == 0

        # Verify database update
        conn = sqlite3.connect(db_path)
        result = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?", ("pred-1",)
        ).fetchone()
        conn.close()

        signals = json.loads(result[0])
        assert signals == {"contact_email": "alice@test.com"}

    def test_backfill_updates_null_signals(self, temp_db_dir):
        """Backfill should also update predictions with NULL supporting_signals."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-2",
                "reminder",
                "Follow up with Bob about the contract",
                0.65,
                "suggest",
                None,  # NULL signals
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        assert stats["updated"] == 1
        assert stats["name_only"] == 1

        conn = sqlite3.connect(db_path)
        result = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?", ("pred-2",)
        ).fetchone()
        conn.close()

        signals = json.loads(result[0])
        assert signals == {"contact_name": "Bob"}

    def test_backfill_skips_resolved_predictions(self, temp_db_dir):
        """Backfill should skip predictions that are already resolved."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, resolved_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-3",
                "reminder",
                'Unreplied message from test@test.com: "Test" (1 hour ago)',
                0.70,
                "suggest",
                "[]",
                1,
                datetime.now(timezone.utc).isoformat(),  # Already resolved
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        # Should not update resolved prediction
        assert stats["updated"] == 0

    def test_backfill_skips_non_surfaced_predictions(self, temp_db_dir):
        """Backfill should skip predictions that were never surfaced."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-4",
                "reminder",
                'Unreplied message from hidden@test.com: "Test"',
                0.20,
                "observe",
                "[]",
                0,  # Not surfaced
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        assert stats["updated"] == 0

    def test_backfill_skips_non_reminder_predictions(self, temp_db_dir):
        """Backfill should only process 'reminder' type predictions."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-5",
                "conflict",  # Not a reminder
                "Calendar overlap detected",
                0.85,
                "default",
                "[]",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        assert stats["updated"] == 0

    def test_backfill_handles_unresolvable(self, temp_db_dir):
        """Backfill should track predictions with no extractable contact info."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-6",
                "reminder",
                "Prepare slides for meeting tomorrow",  # No contact info
                0.60,
                "suggest",
                "[]",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        assert stats["updated"] == 0
        assert stats["unresolvable"] == 1

    def test_dry_run_no_changes(self, temp_db_dir):
        """Dry run should not modify the database."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-7",
                "reminder",
                'Unreplied message from dry@run.com: "Test"',
                0.70,
                "suggest",
                "[]",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        # Run in dry-run mode
        stats = backfill_reminder_contacts(temp_db_dir, dry_run=True)

        # Stats should reflect what WOULD be updated
        assert stats["updated"] == 0  # No actual updates in dry-run
        assert stats["email_only"] == 1  # But the email was found

        # Verify database was NOT changed
        conn = sqlite3.connect(db_path)
        result = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?", ("pred-7",)
        ).fetchone()
        conn.close()

        assert result[0] == "[]"  # Still the old value

    def test_backfill_multiple_predictions(self, temp_db_dir):
        """Backfill should handle multiple predictions in one pass."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        # Insert 3 predictions
        for i, (email, name) in enumerate(
            [
                ("first@test.com", None),
                (None, "Second Person"),
                ("third@test.com", None),
            ]
        ):
            if email:
                desc = f'Unreplied message from {email}: "Test {i}"'
            else:
                desc = f"Follow up with {name} about project"

            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    supporting_signals, was_surfaced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"pred-multi-{i}",
                    "reminder",
                    desc,
                    0.70,
                    "suggest",
                    "[]",
                    1,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        assert stats["updated"] == 3
        assert stats["email_only"] == 2
        assert stats["name_only"] == 1

    def test_backfill_preserves_existing_dict_signals(self, temp_db_dir):
        """Backfill should skip predictions that already have dict-formatted signals."""
        db_path = Path(temp_db_dir) / "user_model.db"
        conn = sqlite3.connect(db_path)

        existing_signals = json.dumps({"contact_email": "existing@test.com"})

        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                supporting_signals, was_surfaced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "pred-8",
                "reminder",
                'Unreplied message from new@test.com: "Test"',
                0.70,
                "suggest",
                existing_signals,  # Already in dict format
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        stats = backfill_reminder_contacts(temp_db_dir, dry_run=False)

        # Should skip this prediction (query filters for '[]' or NULL)
        assert stats["updated"] == 0

        # Verify original signals are preserved
        conn = sqlite3.connect(db_path)
        result = conn.execute(
            "SELECT supporting_signals FROM predictions WHERE id = ?", ("pred-8",)
        ).fetchone()
        conn.close()

        assert result[0] == existing_signals
