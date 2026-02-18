"""
Life OS — Schema Migration Tests (v0 to v2)

Tests the automatic schema migration from version 0 (base schema without
denormalized columns) to version 2 (with denormalized columns for workflow
detection).

This migration adds email_from, email_to, task_id, and calendar_event_id
columns to the events table, creates indexes on them, creates triggers to
auto-populate them on INSERT, and backfills existing events.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from storage.manager import DatabaseManager


class TestSchemaMigrationV0toV2:
    """Test suite for schema migration from v0 to v2."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for test databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def v0_database(self, temp_db_path):
        """Create a v0 schema database (without denormalized columns).

        This simulates the state of the database before the v2 migration,
        allowing us to test the migration logic.
        """
        db_path = Path(temp_db_path) / "events.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Create v0 schema (without denormalized columns)
        conn.executescript("""
            CREATE TABLE events (
                id              TEXT PRIMARY KEY,
                type            TEXT NOT NULL,
                source          TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                priority        TEXT NOT NULL DEFAULT 'normal',
                payload         TEXT NOT NULL DEFAULT '{}',
                metadata        TEXT NOT NULL DEFAULT '{}',
                embedding_id    TEXT,
                created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE INDEX idx_events_type ON events(type);
            CREATE INDEX idx_events_source ON events(source);
            CREATE INDEX idx_events_timestamp ON events(timestamp);
            CREATE INDEX idx_events_priority ON events(priority);
            CREATE INDEX idx_events_type_timestamp ON events(type, timestamp);

            CREATE TABLE event_processing_log (
                event_id        TEXT NOT NULL,
                service         TEXT NOT NULL,
                processed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                result          TEXT,
                PRIMARY KEY (event_id, service)
            );

            CREATE TABLE event_tags (
                event_id    TEXT NOT NULL,
                tag         TEXT NOT NULL,
                rule_id     TEXT,
                created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (event_id, tag)
            );

            CREATE INDEX idx_event_tags_tag ON event_tags(tag);
        """)

        # Insert sample events to test backfill
        now = datetime.now(timezone.utc).isoformat()

        # Email received events
        for i in range(5):
            event_id = str(uuid4())
            payload = json.dumps({
                "from_address": f"sender{i}@example.com",
                "subject": f"Test Email {i}",
                "body": "Test body"
            })
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, "email.received", "proton", now, payload)
            )

        # Email sent events
        for i in range(3):
            event_id = str(uuid4())
            payload = json.dumps({
                "from_address": "user@example.com",
                "to_addresses": f"recipient{i}@example.com",
                "subject": f"Reply {i}",
                "body": "Reply body"
            })
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, "email.sent", "proton", now, payload)
            )

        # Task events
        for i in range(2):
            event_id = str(uuid4())
            task_id = str(uuid4())
            payload = json.dumps({
                "task_id": task_id,
                "title": f"Task {i}",
                "description": "Test task"
            })
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, "task.created", "system", now, payload)
            )

        # Calendar events
        for i in range(2):
            event_id = str(uuid4())
            calendar_event_id = str(uuid4())
            payload = json.dumps({
                "event_id": calendar_event_id,
                "title": f"Meeting {i}",
                "start": now,
                "end": now
            })
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, "calendar.event.created", "caldav", now, payload)
            )

        conn.commit()
        conn.close()
        return temp_db_path

    def test_migration_adds_columns(self, v0_database):
        """Test that migration adds denormalized columns to events table."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check that new columns exist
            cursor = conn.execute("PRAGMA table_info(events)")
            columns = {row[1] for row in cursor.fetchall()}

            assert "email_from" in columns
            assert "email_to" in columns
            assert "task_id" in columns
            assert "calendar_event_id" in columns

    def test_migration_creates_indexes(self, v0_database):
        """Test that migration creates indexes on denormalized columns."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check that indexes exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'")
            indexes = {row[0] for row in cursor.fetchall()}

            assert "idx_events_email_from" in indexes
            assert "idx_events_email_to" in indexes
            assert "idx_events_task_id" in indexes

    def test_migration_creates_triggers(self, v0_database):
        """Test that migration creates triggers for auto-population."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check that triggers exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
            triggers = {row[0] for row in cursor.fetchall()}

            assert "trg_events_email_from" in triggers
            assert "trg_events_email_to" in triggers
            assert "trg_events_task_id" in triggers
            assert "trg_events_calendar_id" in triggers

    def test_migration_backfills_email_from(self, v0_database):
        """Test that migration backfills email_from column from existing events."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check email.received events
            cursor = conn.execute("""
                SELECT email_from, json_extract(payload, '$.from_address') as expected
                FROM events
                WHERE type = 'email.received'
            """)
            for row in cursor.fetchall():
                assert row[0] is not None  # email_from should be populated
                assert row[0].lower() == row[1].lower()  # Should match from_address

            # Check email.sent events
            cursor = conn.execute("""
                SELECT email_from, json_extract(payload, '$.from_address') as expected
                FROM events
                WHERE type = 'email.sent'
            """)
            for row in cursor.fetchall():
                assert row[0] is not None
                assert row[0].lower() == row[1].lower()

    def test_migration_backfills_email_to(self, v0_database):
        """Test that migration backfills email_to column from existing events."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check email.sent events
            cursor = conn.execute("""
                SELECT email_to, json_extract(payload, '$.to_addresses') as expected
                FROM events
                WHERE type = 'email.sent'
            """)
            for row in cursor.fetchall():
                assert row[0] is not None  # email_to should be populated
                assert row[0].lower() == row[1].lower()  # Should match to_addresses

    def test_migration_backfills_task_id(self, v0_database):
        """Test that migration backfills task_id column from existing events."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check task.created events
            cursor = conn.execute("""
                SELECT task_id, json_extract(payload, '$.task_id') as expected
                FROM events
                WHERE type = 'task.created'
            """)
            for row in cursor.fetchall():
                assert row[0] is not None  # task_id should be populated
                assert row[0] == row[1]  # Should match task_id from payload

    def test_migration_backfills_calendar_event_id(self, v0_database):
        """Test that migration backfills calendar_event_id column from existing events."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check calendar.event.created events
            cursor = conn.execute("""
                SELECT calendar_event_id, json_extract(payload, '$.event_id') as expected
                FROM events
                WHERE type = 'calendar.event.created'
            """)
            for row in cursor.fetchall():
                assert row[0] is not None  # calendar_event_id should be populated
                assert row[0] == row[1]  # Should match event_id from payload

    def test_migration_sets_schema_version(self, v0_database):
        """Test that migration updates schema_version table to at least v2.

        The schema may have advanced beyond v2 in subsequent migrations; we only
        assert that the migration brought the version to >= 2.
        """
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Check schema version — use MAX() in case multiple rows exist
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()
            assert version is not None
            assert version[0] >= 2

    def test_triggers_work_on_new_events(self, v0_database):
        """Test that triggers auto-populate denormalized columns for new events."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        now = datetime.now(timezone.utc).isoformat()

        with db.get_connection("events") as conn:
            # Insert a new email.received event
            event_id = str(uuid4())
            payload = json.dumps({
                "from_address": "newemail@example.com",
                "subject": "New Email",
                "body": "New body"
            })
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (event_id, "email.received", "proton", now, payload)
            )

            # Check that email_from was auto-populated by trigger
            cursor = conn.execute("SELECT email_from FROM events WHERE id = ?", (event_id,))
            email_from = cursor.fetchone()[0]
            assert email_from == "newemail@example.com"

    def test_migration_idempotent(self, v0_database):
        """Test that running migration twice doesn't cause errors or downgrade schema."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        # Run initialize_all again (should not fail or reset version)
        db.initialize_all()

        # Verify schema version is still at least v2 after second run
        with db.get_connection("events") as conn:
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()
            assert version[0] >= 2

    def test_migration_preserves_existing_data(self, v0_database):
        """Test that migration doesn't lose or corrupt existing event data."""
        # Count events before migration
        conn = sqlite3.connect(str(Path(v0_database) / "events.db"))
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        count_before = cursor.fetchone()[0]
        conn.close()

        # Run migration
        db = DatabaseManager(v0_database)
        db.initialize_all()

        # Count events after migration
        with db.get_connection("events") as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            count_after = cursor.fetchone()[0]

            # Should have same number of events
            assert count_after == count_before

            # Verify payload integrity (spot check)
            cursor = conn.execute("""
                SELECT type, payload
                FROM events
                WHERE type = 'email.received'
                LIMIT 1
            """)
            row = cursor.fetchone()
            assert row is not None
            payload = json.loads(row[1])
            assert "from_address" in payload
            assert "subject" in payload

    def test_workflow_detection_uses_denormalized_columns(self, v0_database):
        """Test that workflow detection queries work with denormalized columns."""
        db = DatabaseManager(v0_database)
        db.initialize_all()

        with db.get_connection("events") as conn:
            # Test query that workflow detector uses (should be fast now)
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM events
                WHERE type = 'email.received'
                  AND email_from = 'sender0@example.com'
            """)
            count = cursor.fetchone()[0]
            assert count == 1  # Should find the event

            # Test join query (simulating workflow detection pattern)
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM events e1
                JOIN events e2 ON
                    julianday(e2.timestamp) > julianday(e1.timestamp)
                WHERE e1.type = 'email.received'
                  AND e1.email_from = 'sender0@example.com'
                  AND e2.type = 'email.sent'
            """)
            # Should execute without error (actual count depends on data)
            cursor.fetchone()
