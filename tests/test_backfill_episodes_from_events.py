"""
Tests for scripts/backfill_episodes_from_events.py

Validates that the episode backfill script correctly creates episodes from
events.db entries, handles idempotency, respects dry-run mode, strips large
payloads, and uses actual event timestamps instead of sync timestamps.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from scripts.backfill_episodes_from_events import (
    EPISODIC_EVENT_TYPES,
    backfill_episodes,
    build_compact_content,
    classify_interaction_type,
    extract_actual_timestamp,
    generate_episode_summary,
)


def _insert_event(db, event_id: str, event_type: str, payload: dict, timestamp: str = "2026-02-20T12:00:00Z", source: str = "test", metadata: dict | None = None):
    """Helper to insert an event into events.db for testing."""
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, embedding_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                event_type,
                source,
                timestamp,
                "normal",
                json.dumps(payload),
                json.dumps(metadata or {}),
                None,
            ),
        )
        conn.commit()


def _count_episodes(db) -> int:
    """Count total episodes in user_model.db."""
    with db.get_connection("user_model") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]


def _get_episode_by_event_id(db, event_id: str) -> dict | None:
    """Fetch an episode by its source event_id."""
    with db.get_connection("user_model") as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM episodes WHERE event_id = ?", (event_id,))
        return cursor.fetchone()


def _dict_factory(cursor, row):
    """SQLite row factory that returns dicts."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class TestCreatesEpisodesFromEmailEvents:
    """Test that email events are correctly converted to episodes."""

    def test_creates_episodes_from_email_events(self, db):
        """Insert 3 email events, run backfill, verify 3 episodes created
        with correct interaction_type, contacts, and summary."""
        # Insert 3 email events
        _insert_event(db, "evt-1", "email.received", {
            "from_address": "alice@example.com",
            "to_addresses": ["user@example.com"],
            "subject": "Meeting tomorrow",
            "body_plain": "Let's meet at 10am.",
        })
        _insert_event(db, "evt-2", "email.sent", {
            "from_address": "user@example.com",
            "to_addresses": ["bob@example.com", "carol@example.com"],
            "subject": "Project update",
            "body_plain": "Here's the latest status.",
        })
        _insert_event(db, "evt-3", "email.received", {
            "from_address": "dave@example.com",
            "to_addresses": ["user@example.com"],
            "subject": "Invoice #123",
        })

        # Run backfill
        stats = backfill_episodes(db)

        # Verify 3 episodes created
        assert stats["episodes_created"] == 3
        assert stats["episodes_skipped_existing"] == 0
        assert stats["errors"] == 0
        assert _count_episodes(db) == 3

        # Verify episode 1 details
        ep1 = _get_episode_by_event_id(db, "evt-1")
        assert ep1 is not None
        assert ep1["interaction_type"] == "email_received"
        contacts = json.loads(ep1["contacts_involved"])
        assert "alice@example.com" in contacts
        assert "Meeting tomorrow" in ep1["content_summary"]

        # Verify episode 2 details
        ep2 = _get_episode_by_event_id(db, "evt-2")
        assert ep2 is not None
        assert ep2["interaction_type"] == "email_sent"
        contacts = json.loads(ep2["contacts_involved"])
        assert "bob@example.com" in contacts
        assert "carol@example.com" in contacts

        # Verify episode 3 details
        ep3 = _get_episode_by_event_id(db, "evt-3")
        assert ep3 is not None
        assert ep3["interaction_type"] == "email_received"


class TestSkipsNonEpisodicEvents:
    """Test that system/internal events don't produce episodes."""

    def test_skips_non_episodic_events(self, db):
        """Insert system events that shouldn't create episodes, verify 0 episodes."""
        _insert_event(db, "evt-sys-1", "system.connector.sync_complete", {
            "connector": "google",
            "events_synced": 100,
        })
        _insert_event(db, "evt-sys-2", "usermodel.signal_profile.updated", {
            "profile_type": "linguistic",
        })
        _insert_event(db, "evt-sys-3", "system.rule.triggered", {
            "rule_id": "auto-tag-work",
        })

        stats = backfill_episodes(db)

        # Non-episodic events are filtered by the SQL WHERE clause,
        # so total_events_scanned should be 0
        assert stats["total_events_scanned"] == 0
        assert stats["episodes_created"] == 0
        assert _count_episodes(db) == 0


class TestIdempotentSkipExisting:
    """Test that running backfill twice doesn't create duplicates."""

    def test_idempotent_skip_existing(self, db):
        """Create an episode for event X, run backfill, verify no duplicate."""
        event_id = "evt-already-has-episode"

        # Insert the event
        _insert_event(db, event_id, "email.received", {
            "from_address": "existing@example.com",
            "subject": "Already processed",
        })

        # Manually create an episode for this event (simulating prior processing)
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary,
                    content_full, contacts_involved, topics, entities,
                    inferred_mood, active_domain)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    "2026-02-20T12:00:00Z",
                    event_id,
                    "email_received",
                    "Already exists",
                    "{}",
                    "[]",
                    "[]",
                    "[]",
                    "{}",
                    "personal",
                ),
            )
            conn.commit()

        assert _count_episodes(db) == 1

        # Run backfill — should skip the existing episode
        stats = backfill_episodes(db)

        assert stats["episodes_created"] == 0
        assert stats["episodes_skipped_existing"] == 1
        assert _count_episodes(db) == 1  # Still only 1 episode


class TestDryRunNoWrites:
    """Test that dry-run mode doesn't write to the database."""

    def test_dry_run_no_writes(self, db):
        """Run with dry_run=True, verify 0 episodes in DB but stats show
        what would be created."""
        _insert_event(db, "evt-dry-1", "email.received", {
            "from_address": "dry@example.com",
            "subject": "Dry run test",
        })
        _insert_event(db, "evt-dry-2", "message.sent", {
            "to_addresses": ["friend@example.com"],
            "body_plain": "Hello!",
        })

        stats = backfill_episodes(db, dry_run=True)

        # Stats should report what would be created
        assert stats["episodes_created"] == 2
        assert stats["total_events_scanned"] == 2

        # But no episodes should actually exist in the database
        assert _count_episodes(db) == 0


class TestContentFullStripsLargeBody:
    """Test that large payload bodies are truncated in content_full."""

    def test_content_full_strips_large_body(self, db):
        """Insert event with 100KB body, verify episode content_full is < 4000 chars."""
        large_body = "A" * 100_000  # 100KB body

        _insert_event(db, "evt-large", "email.received", {
            "from_address": "large@example.com",
            "subject": "Big email",
            "body": large_body,
            "html_body": large_body,
        })

        stats = backfill_episodes(db)

        assert stats["episodes_created"] == 1
        ep = _get_episode_by_event_id(db, "evt-large")
        assert ep is not None
        # content_full should be capped at 4000 chars
        assert len(ep["content_full"]) <= 4000

    def test_build_compact_content_unit(self):
        """Unit test that build_compact_content strips large fields."""
        payload = {
            "from_address": "test@example.com",
            "subject": "Test",
            "body": "X" * 10_000,
            "html_body": "Y" * 10_000,
            "raw": "Z" * 10_000,
        }
        result = build_compact_content(payload)
        assert len(result) <= 4000
        # The small fields should be preserved
        parsed = json.loads(result) if len(result) < 4000 else {}
        if parsed:
            assert parsed["from_address"] == "test@example.com"
            assert parsed["subject"] == "Test"


class TestUsesActualEmailDate:
    """Test that the actual event timestamp is used, not the sync timestamp."""

    def test_uses_actual_email_date_not_sync_timestamp(self, db):
        """Insert email with email_date='2026-01-15T10:00:00', verify
        episode.timestamp matches email_date, not the event.timestamp."""
        actual_date = "2026-01-15T10:00:00Z"
        sync_date = "2026-02-20T03:00:00Z"  # Much later — this is the sync time

        _insert_event(
            db,
            "evt-date-test",
            "email.received",
            {
                "from_address": "date-test@example.com",
                "subject": "Old email",
                "email_date": actual_date,
            },
            timestamp=sync_date,
        )

        stats = backfill_episodes(db)
        assert stats["episodes_created"] == 1

        ep = _get_episode_by_event_id(db, "evt-date-test")
        assert ep is not None
        # The episode timestamp should use email_date, NOT the sync timestamp
        assert ep["timestamp"] == actual_date
        assert ep["timestamp"] != sync_date

    def test_falls_back_to_sync_timestamp(self, db):
        """When no actual date fields exist, the sync timestamp is used."""
        sync_date = "2026-02-20T03:00:00Z"

        _insert_event(
            db,
            "evt-no-date",
            "task.created",
            {"title": "Some task"},
            timestamp=sync_date,
        )

        stats = backfill_episodes(db)
        assert stats["episodes_created"] == 1

        ep = _get_episode_by_event_id(db, "evt-no-date")
        assert ep is not None
        assert ep["timestamp"] == sync_date


class TestExtractActualTimestamp:
    """Unit tests for the extract_actual_timestamp function."""

    def test_email_date_priority(self):
        """email_date takes highest priority."""
        payload = {
            "email_date": "2026-01-01T00:00:00Z",
            "sent_at": "2026-01-02T00:00:00Z",
            "date": "2026-01-03T00:00:00Z",
        }
        assert extract_actual_timestamp(payload, "2026-02-01T00:00:00Z") == "2026-01-01T00:00:00Z"

    def test_sent_at_fallback(self):
        """sent_at is used when email_date is absent."""
        payload = {"sent_at": "2026-01-02T00:00:00Z"}
        assert extract_actual_timestamp(payload, "2026-02-01T00:00:00Z") == "2026-01-02T00:00:00Z"

    def test_sync_timestamp_last_resort(self):
        """Falls back to sync timestamp when no payload date fields exist."""
        assert extract_actual_timestamp({}, "2026-02-01T00:00:00Z") == "2026-02-01T00:00:00Z"


class TestClassifyInteractionType:
    """Unit tests for classify_interaction_type."""

    def test_email_received(self):
        assert classify_interaction_type("email.received", {}) == "email_received"

    def test_email_sent(self):
        assert classify_interaction_type("email.sent", {}) == "email_sent"

    def test_calendar_with_attendees(self):
        """Calendar event with attendees becomes meeting_scheduled."""
        assert classify_interaction_type(
            "calendar.event.created",
            {"attendees": ["a@b.com"]},
        ) == "meeting_scheduled"

    def test_calendar_without_attendees(self):
        """Calendar event without attendees becomes calendar_blocked."""
        assert classify_interaction_type("calendar.event.created", {}) == "calendar_blocked"

    def test_finance_spending(self):
        assert classify_interaction_type("finance.transaction.new", {"amount": -45.23}) == "spending"

    def test_finance_income(self):
        assert classify_interaction_type("finance.transaction.new", {"amount": 1000}) == "income"


class TestGenerateEpisodeSummary:
    """Unit tests for generate_episode_summary."""

    def test_email_received_summary(self):
        summary = generate_episode_summary("email.received", {
            "from_address": "alice@test.com",
            "subject": "Hello",
        })
        assert "Email from alice@test.com" in summary
        assert "Hello" in summary

    def test_task_completed_summary(self):
        summary = generate_episode_summary("task.completed", {
            "title": "Fix the bug",
        })
        assert "Task completed: Fix the bug" in summary

    def test_transaction_summary(self):
        summary = generate_episode_summary("finance.transaction.new", {
            "amount": 45.23,
            "merchant": "Whole Foods",
        })
        assert "Transaction" in summary
        assert "Whole Foods" in summary

    def test_summary_truncated_to_200(self):
        """Summaries should never exceed 200 characters."""
        summary = generate_episode_summary("email.received", {
            "from_address": "x" * 100 + "@example.com",
            "subject": "Y" * 200,
        })
        assert len(summary) <= 200


class TestMixedEventTypes:
    """Integration test with diverse event types."""

    def test_various_event_types_create_correct_episodes(self, db):
        """Test that different episodic event types all create correct episodes."""
        _insert_event(db, "evt-msg-1", "message.received", {
            "from_address": "+1234567890",
            "body_plain": "Hey there!",
        })
        _insert_event(db, "evt-cal-1", "calendar.event.created", {
            "title": "Team standup",
            "start_time": "2026-02-21T09:00:00Z",
            "attendees": ["team@company.com"],
        })
        _insert_event(db, "evt-fin-1", "finance.transaction.new", {
            "amount": -42.50,
            "merchant": "Coffee Shop",
        })
        _insert_event(db, "evt-task-1", "task.completed", {
            "title": "Review PR #42",
        })

        stats = backfill_episodes(db)

        assert stats["episodes_created"] == 4
        assert stats["errors"] == 0

        # Verify interaction types
        msg_ep = _get_episode_by_event_id(db, "evt-msg-1")
        assert msg_ep["interaction_type"] == "message_received"

        cal_ep = _get_episode_by_event_id(db, "evt-cal-1")
        assert cal_ep["interaction_type"] == "meeting_scheduled"

        fin_ep = _get_episode_by_event_id(db, "evt-fin-1")
        assert fin_ep["interaction_type"] == "spending"

        task_ep = _get_episode_by_event_id(db, "evt-task-1")
        assert task_ep["interaction_type"] == "task_completed"


class TestBatchProcessing:
    """Test that batch processing works correctly."""

    def test_batch_commit(self, db):
        """Insert more events than batch_size and verify all get processed."""
        # Create 7 events with batch_size=3
        for i in range(7):
            _insert_event(db, f"evt-batch-{i}", "email.received", {
                "from_address": f"batch{i}@example.com",
                "subject": f"Batch email {i}",
            })

        stats = backfill_episodes(db, batch_size=3)

        assert stats["episodes_created"] == 7
        assert _count_episodes(db) == 7


class TestPostWriteVerification:
    """Test that the post-write verification correctly counts persisted episodes."""

    def test_episodes_verified_key_in_stats(self, db):
        """The returned stats dict must always include episodes_verified."""
        stats = backfill_episodes(db)
        assert "episodes_verified" in stats

    def test_episodes_verified_zero_for_empty_run(self, db):
        """With no events, episodes_verified should be 0."""
        stats = backfill_episodes(db)
        assert stats["episodes_verified"] == 0
        assert stats["episodes_created"] == 0

    def test_episodes_verified_matches_created_on_success(self, db):
        """After a successful backfill, episodes_verified must equal episodes_created.

        This is the core invariant: INSERT OR IGNORE must not silently drop rows.
        If verification and creation counts diverge, the batch write silently
        lost data (e.g. constraint collision or WAL corruption).
        """
        for i in range(5):
            _insert_event(db, f"evt-verify-{i}", "email.received", {
                "from_address": f"verify{i}@example.com",
                "subject": f"Verify email {i}",
            })

        stats = backfill_episodes(db)

        assert stats["episodes_created"] == 5
        assert stats["episodes_verified"] == 5
        # The count in the database must match both stats fields
        assert _count_episodes(db) == 5

    def test_episodes_verified_matches_actual_db_count(self, db):
        """episodes_verified should equal the actual row count in the database."""
        for i in range(4):
            _insert_event(db, f"evt-dbcount-{i}", "task.created", {
                "title": f"Task {i}",
            })

        stats = backfill_episodes(db)

        actual_db_count = _count_episodes(db)
        assert stats["episodes_verified"] == actual_db_count

    def test_episodes_verified_zero_in_dry_run(self, db):
        """In dry-run mode, nothing is written so episodes_verified must be 0."""
        for i in range(3):
            _insert_event(db, f"evt-dry-verify-{i}", "message.received", {
                "from_address": f"+1{i:010d}",
                "body_plain": f"Message {i}",
            })

        stats = backfill_episodes(db, dry_run=True)

        # dry-run reports what would be created but writes nothing
        assert stats["episodes_created"] == 3
        assert stats["episodes_verified"] == 0
        assert _count_episodes(db) == 0

    def test_episodes_verified_across_multiple_batches(self, db):
        """Verification must aggregate correctly across multiple batch commits."""
        # 11 events with batch_size=4 → batches of 4, 4, 3
        for i in range(11):
            _insert_event(db, f"evt-multibatch-{i}", "email.received", {
                "from_address": f"multi{i}@example.com",
                "subject": f"Multi-batch email {i}",
            })

        stats = backfill_episodes(db, batch_size=4)

        assert stats["episodes_created"] == 11
        assert stats["episodes_verified"] == 11
        assert _count_episodes(db) == 11
