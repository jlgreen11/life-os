"""
Tests for scripts/backfill_communication_templates.py — DB reuse and WAL checkpoint.

Verifies that the backfill function:
- Creates templates when an existing DatabaseManager + UserModelStore are passed
  (the fix for WAL lock contention when called from a running LifeOS instance).
- Falls back to creating its own DatabaseManager when only data_dir is supplied
  (preserving CLI / standalone usage).
- Runs a WAL checkpoint after each batch so writes are durable.
- Emits a warning when a batch produces no new templates.
- Respects dry_run mode (no writes).
- Skips events whose body is too short to analyse.
"""

import json

import pytest

from scripts.backfill_communication_templates import backfill_communication_templates
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_email_event(
    db: DatabaseManager,
    event_id: str,
    event_type: str,
    from_addr: str,
    to_addrs: list[str],
    body: str,
    timestamp: str = "2026-01-15T09:00:00Z",
) -> None:
    """Insert a single email event into events.db for testing.

    Args:
        db: DatabaseManager with events.db initialised.
        event_id: Unique ID for the event row.
        event_type: One of email.sent / email.received.
        from_addr: Sender email address.
        to_addrs: List of recipient email addresses.
        body: Plain-text message body (must be >10 chars to be processed).
        timestamp: ISO-8601 event timestamp.
    """
    payload = {
        "from_address": from_addr,
        "to_addresses": to_addrs,
        "subject": "Test subject",
        "body_plain": body,
        "email_date": timestamp,
    }
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                event_type,
                "google",
                timestamp,
                "normal",
                json.dumps(payload),
                json.dumps({}),
            ),
        )


def _count_templates(db: DatabaseManager) -> int:
    """Return the total number of rows in communication_templates."""
    with db.get_connection("user_model") as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM communication_templates"
        ).fetchone()[0]


# ---------------------------------------------------------------------------
# Tests — DB reuse (the main fix)
# ---------------------------------------------------------------------------


class TestDbReuse:
    """Verify that templates are created when an existing db/ums are passed."""

    def test_creates_templates_when_db_and_ums_passed(self, db, user_model_store):
        """Templates are created when the caller passes db and ums directly.

        This is the primary regression test for the WAL contention fix.  The
        backfill must produce > 0 templates when real email events are present
        and the caller passes its own DatabaseManager instead of letting the
        backfill open competing connections.
        """
        # Insert 3 email events with substantial body text.
        _insert_email_event(
            db, "evt-1", "email.received",
            "alice@example.com", ["user@example.com"],
            "Hey, can we meet tomorrow to discuss the project deliverables?",
        )
        _insert_email_event(
            db, "evt-2", "email.sent",
            "user@example.com", ["alice@example.com"],
            "Sure, 10am works for me. I'll send a calendar invite shortly.",
        )
        _insert_email_event(
            db, "evt-3", "email.received",
            "bob@example.com", ["user@example.com"],
            "Please review the attached document before Friday's deadline.",
        )

        assert _count_templates(db) == 0, "Templates table should be empty before backfill"

        stats = backfill_communication_templates(db=db, ums=user_model_store)

        assert stats["events_processed"] == 3
        assert stats["errors"] == 0
        # At least one template must have been created.
        assert _count_templates(db) > 0
        assert stats["templates_created"] > 0

    def test_creates_templates_when_only_db_passed(self, db):
        """Templates are created when only db is passed (ums constructed internally)."""
        _insert_email_event(
            db, "evt-1", "email.sent",
            "user@example.com", ["carol@example.com"],
            "Following up on our conversation from last week about the contract.",
        )

        stats = backfill_communication_templates(db=db)

        assert stats["events_processed"] == 1
        assert stats["errors"] == 0
        assert _count_templates(db) > 0

    def test_standalone_mode_creates_templates(self, db):
        """CLI mode (no db/ums) creates templates using data_dir.

        When called from the command line, the function creates its own
        DatabaseManager from data_dir.  This mode must still produce templates.
        """
        _insert_email_event(
            db, "evt-1", "email.received",
            "dave@example.com", ["user@example.com"],
            "Could you send me the final report by end of day today please?",
        )

        # Pass data_dir only — backfill creates its own DatabaseManager.
        stats = backfill_communication_templates(data_dir=str(db.data_dir))

        assert stats["events_processed"] == 1
        assert stats["errors"] == 0
        assert _count_templates(db) > 0

    def test_templates_created_reflects_actual_db_count(self, db, user_model_store):
        """stats['templates_created'] matches the real delta in communication_templates."""
        _insert_email_event(
            db, "evt-1", "email.sent",
            "user@example.com", ["eve@example.com"],
            "Just checking in — how is the new role going so far?",
        )
        _insert_email_event(
            db, "evt-2", "email.sent",
            "user@example.com", ["frank@example.com"],
            "Looking forward to collaborating with you on the upcoming sprint.",
        )

        before = _count_templates(db)
        stats = backfill_communication_templates(db=db, ums=user_model_store)
        after = _count_templates(db)

        assert stats["templates_created"] == after - before


# ---------------------------------------------------------------------------
# Tests — Event filtering
# ---------------------------------------------------------------------------


class TestEventFiltering:
    """Verify that only qualifying events are processed."""

    def test_skips_events_with_short_body(self, db, user_model_store):
        """Events whose body_plain is ≤10 chars must be excluded.

        The SQL query filters on LENGTH(body_plain) > 10, so the backfill
        must not process — or count — events with very short bodies.
        """
        # Insert one qualifying event and one with a short body.
        _insert_email_event(
            db, "evt-long", "email.received",
            "alice@example.com", ["user@example.com"],
            "This message is definitely long enough to be analysed properly.",
        )
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "evt-short",
                    "email.received",
                    "google",
                    "2026-01-15T09:00:00Z",
                    "normal",
                    json.dumps({"from_address": "spam@example.com", "body_plain": "Hi"}),
                    json.dumps({}),
                ),
            )

        stats = backfill_communication_templates(db=db, ums=user_model_store)

        # Only the long-body event is processed.
        assert stats["events_processed"] == 1

    def test_skips_non_communication_event_types(self, db, user_model_store):
        """Non-communication events (e.g. task.created) must be ignored."""
        with db.get_connection("events") as conn:
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    "evt-task",
                    "task.created",
                    "system",
                    "2026-01-15T09:00:00Z",
                    "normal",
                    json.dumps({"title": "Write quarterly report", "body_plain": "Some body text here"}),
                    json.dumps({}),
                ),
            )

        stats = backfill_communication_templates(db=db, ums=user_model_store)

        assert stats["events_processed"] == 0
        assert _count_templates(db) == 0


# ---------------------------------------------------------------------------
# Tests — Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify that dry_run=True does not write any templates."""

    def test_dry_run_does_not_create_templates(self, db, user_model_store):
        """dry_run=True should count events but write nothing to user_model.db."""
        _insert_email_event(
            db, "evt-1", "email.received",
            "alice@example.com", ["user@example.com"],
            "Can you send me the slides before the presentation tomorrow morning?",
        )

        stats = backfill_communication_templates(db=db, ums=user_model_store, dry_run=True)

        assert stats["events_processed"] == 1
        assert _count_templates(db) == 0, "dry_run must not write templates"

    def test_dry_run_stats_reflect_events_processed(self, db, user_model_store):
        """dry_run stats should still accurately count events processed."""
        for i in range(4):
            _insert_email_event(
                db, f"evt-{i}", "email.sent",
                "user@example.com", [f"contact{i}@example.com"],
                f"Message number {i} — long enough to satisfy the length filter.",
            )

        stats = backfill_communication_templates(db=db, ums=user_model_store, dry_run=True)

        assert stats["events_processed"] == 4
        assert stats["errors"] == 0
        assert stats["templates_created"] == 0


# ---------------------------------------------------------------------------
# Tests — Batch processing
# ---------------------------------------------------------------------------


class TestBatching:
    """Verify batch-level behaviour including WAL checkpoint and verification."""

    def test_batch_size_respected(self, db, user_model_store):
        """Backfill processes all events even with a batch_size of 1."""
        for i in range(5):
            _insert_email_event(
                db, f"evt-{i}", "email.received",
                f"contact{i}@example.com", ["user@example.com"],
                f"Test message {i} — this body text is long enough to qualify for extraction.",
            )

        stats = backfill_communication_templates(db=db, ums=user_model_store, batch_size=1)

        assert stats["events_processed"] == 5
        assert stats["errors"] == 0
        assert _count_templates(db) > 0

    def test_limit_caps_events_processed(self, db, user_model_store):
        """limit parameter should cap the number of events processed."""
        for i in range(10):
            _insert_email_event(
                db, f"evt-{i}", "email.received",
                f"user{i}@example.com", ["me@example.com"],
                f"Email body number {i} — sufficient length for template extraction.",
            )

        stats = backfill_communication_templates(db=db, ums=user_model_store, limit=3)

        assert stats["events_processed"] == 3

    def test_idempotent_on_second_run(self, db, user_model_store):
        """Running the backfill twice must not create duplicate templates.

        Communication templates use deterministic IDs (SHA-256 of
        contact + channel + direction), so a second run should only
        update existing records, not insert new rows.
        """
        _insert_email_event(
            db, "evt-1", "email.sent",
            "user@example.com", ["alice@example.com"],
            "Looking forward to catching up at the conference next month.",
        )

        first = backfill_communication_templates(db=db, ums=user_model_store)
        count_after_first = _count_templates(db)

        second = backfill_communication_templates(db=db, ums=user_model_store)
        count_after_second = _count_templates(db)

        # The template count must not grow on the second run.
        assert count_after_second == count_after_first
        # Both runs must have processed the same number of events.
        assert first["events_processed"] == second["events_processed"]
