"""
Tests for user_model.db schema migration v1 → v2.

Migration 1→2 deletes orphaned communication templates that were created
before PR #130's bidirectional template extraction was implemented. These
templates have context='general' and will never be updated by the new code.

The migration deletes them and lets the system regenerate them with the
correct bidirectional context values (user_to_contact / contact_to_user).
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


def test_migration_deletes_orphaned_general_templates():
    """Test that migration v1→2 deletes templates with context='general'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a v1 database by manually constructing the schema
        db_path = Path(tmpdir) / "user_model.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Create v1 schema (before migration)
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")

        # Create communication_templates table
        conn.execute("""
            CREATE TABLE communication_templates (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                contact_id TEXT,
                channel TEXT,
                greeting TEXT,
                closing TEXT,
                formality REAL DEFAULT 0.5,
                typical_length REAL DEFAULT 50.0,
                uses_emoji INTEGER DEFAULT 0,
                common_phrases TEXT DEFAULT '[]',
                avoids_phrases TEXT DEFAULT '[]',
                tone_notes TEXT DEFAULT '[]',
                example_message_ids TEXT DEFAULT '[]',
                samples_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)

        # Insert old-format templates with context='general'
        conn.executemany(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, greeting, closing, formality, samples_analyzed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("abc123", "general", "alice@example.com", "email", "Hi", "Best", 0.5, 10),
                ("def456", "general", "bob@example.com", "email", "Hey", "Thanks", 0.3, 5),
                ("ghi789", "general", "carol@example.com", "signal_msg", "Yo", "Later", 0.2, 3),
            ]
        )
        conn.commit()

        # Verify templates exist before migration
        count_before = conn.execute(
            "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
        ).fetchone()[0]
        assert count_before == 3
        conn.close()

        # Trigger migration by initializing DatabaseManager
        db = DatabaseManager(data_dir=tmpdir)
        db.initialize_all()  # This runs the migration

        # Verify templates with context='general' are deleted
        with db.get_connection("user_model") as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
            ).fetchone()[0]
            assert count_after == 0

            # Verify schema version was updated to 2
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 2


def test_migration_preserves_new_format_templates():
    """Test that migration v1→2 preserves templates with new context values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a v1 database manually
        db_path = Path(tmpdir) / "user_model.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Create v1 schema
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")

        conn.execute("""
            CREATE TABLE communication_templates (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                contact_id TEXT,
                channel TEXT,
                greeting TEXT,
                closing TEXT,
                formality REAL DEFAULT 0.5,
                typical_length REAL DEFAULT 50.0,
                uses_emoji INTEGER DEFAULT 0,
                common_phrases TEXT DEFAULT '[]',
                avoids_phrases TEXT DEFAULT '[]',
                tone_notes TEXT DEFAULT '[]',
                example_message_ids TEXT DEFAULT '[]',
                samples_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)

        # Insert old-format template with context='general'
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, greeting, closing, formality, samples_analyzed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old123", "general", "old@example.com", "email", "Hi", "Best", 0.5, 10)
        )

        # Insert new-format templates with bidirectional context
        conn.executemany(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, greeting, closing, formality, samples_analyzed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("new123out", "user_to_contact", "alice@example.com", "email", "Hi", "Best", 0.5, 15),
                ("new123in", "contact_to_user", "alice@example.com", "email", "Hey", "Cheers", 0.3, 12),
            ]
        )
        conn.commit()
        conn.close()

        # Trigger migration
        db = DatabaseManager(data_dir=tmpdir)
        db.initialize_all()  # This runs the migration

        # Verify old template is deleted but new templates are preserved
        with db.get_connection("user_model") as conn:
            old_count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
            ).fetchone()[0]
            assert old_count == 0

            new_count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context IN ('user_to_contact', 'contact_to_user')"
            ).fetchone()[0]
            assert new_count == 2

            # Verify new templates have correct data
            outbound = conn.execute(
                "SELECT * FROM communication_templates WHERE id = 'new123out'"
            ).fetchone()
            assert outbound["context"] == "user_to_contact"
            assert outbound["samples_analyzed"] == 15

            inbound = conn.execute(
                "SELECT * FROM communication_templates WHERE id = 'new123in'"
            ).fetchone()
            assert inbound["context"] == "contact_to_user"
            assert inbound["samples_analyzed"] == 12


def test_migration_is_idempotent():
    """Test that running migration v1→2 multiple times is safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First migration run
        db_path = Path(tmpdir) / "user_model.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute("""
            CREATE TABLE communication_templates (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                contact_id TEXT,
                channel TEXT,
                greeting TEXT,
                closing TEXT,
                formality REAL DEFAULT 0.5,
                samples_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, samples_analyzed)
               VALUES (?, ?, ?, ?, ?)""",
            ("abc123", "general", "alice@example.com", "email", 10)
        )
        conn.commit()
        conn.close()

        # Trigger migration once
        db = DatabaseManager(data_dir=tmpdir)
        db.initialize_all()  # This runs the migration

        # Verify templates were deleted
        with db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
            ).fetchone()[0]
            assert count == 0

        # Trigger migration again (should be a no-op)
        db_again = DatabaseManager(data_dir=tmpdir)
        db_again.initialize_all()  # This runs the migration again (should be idempotent)

        # Verify still no orphaned templates
        with db_again.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM communication_templates WHERE context = 'general'"
            ).fetchone()[0]
            assert count == 0

            # Verify schema version is still 2
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 2


def test_migration_handles_empty_database():
    """Test that migration v1→2 works when no templates exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty v1 database
        db_path = Path(tmpdir) / "user_model.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute("""
            CREATE TABLE communication_templates (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                contact_id TEXT,
                channel TEXT,
                samples_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.commit()

        # Verify no templates exist
        count = conn.execute("SELECT COUNT(*) FROM communication_templates").fetchone()[0]
        assert count == 0
        conn.close()

        # Trigger migration (should be a no-op)
        db = DatabaseManager(data_dir=tmpdir)
        db.initialize_all()  # This runs the migration

        # Verify still no templates
        with db.get_connection("user_model") as conn:
            count = conn.execute("SELECT COUNT(*) FROM communication_templates").fetchone()[0]
            assert count == 0

            # Verify schema version was updated to 2
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 2


def test_migration_logs_deletion_count(caplog):
    """Test that migration logs the number of orphaned templates deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create v1 database with orphaned templates
        db_path = Path(tmpdir) / "user_model.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute("""
            CREATE TABLE communication_templates (
                id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                contact_id TEXT,
                channel TEXT,
                samples_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.executemany(
            """INSERT INTO communication_templates
               (id, context, contact_id, channel, samples_analyzed)
               VALUES (?, ?, ?, ?, ?)""",
            [
                ("t1", "general", "a@example.com", "email", 5),
                ("t2", "general", "b@example.com", "email", 10),
            ]
        )
        conn.commit()
        conn.close()

        # Trigger migration with logging enabled
        with caplog.at_level("INFO"):
            db = DatabaseManager(data_dir=tmpdir)
            db.initialize_all()  # This runs the migration

        # Verify log message mentions the count
        assert any("Deleting 2 orphaned communication templates" in record.message
                   for record in caplog.records)


def test_fresh_database_starts_at_v2():
    """Test that a fresh database (no prior schema version) starts at v2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(data_dir=tmpdir)
        db.initialize_all()  # Initialize fresh database

        # Verify schema version is 2 (latest)
        with db.get_connection("user_model") as conn:
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 2

        # Verify communication_templates table exists with correct schema
        with db.get_connection("user_model") as conn:
            cursor = conn.execute("PRAGMA table_info(communication_templates)")
            columns = {row["name"] for row in cursor.fetchall()}

            # Verify all expected columns exist
            expected_columns = {
                "id", "context", "contact_id", "channel", "greeting", "closing",
                "formality", "typical_length", "uses_emoji", "common_phrases",
                "avoids_phrases", "tone_notes", "example_message_ids",
                "samples_analyzed", "updated_at"
            }
            assert expected_columns.issubset(columns)
