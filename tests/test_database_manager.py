"""
Test suite for DatabaseManager — the foundational SQLite orchestrator.

DatabaseManager is the bedrock of Life OS's 5-database architecture. It manages
schema migrations, connection lifecycle, and WAL/foreign-key pragmas for all
five domain-separated databases:
    - events.db       — Append-only event log
    - entities.db     — People, places, subscriptions
    - state.db        — Tasks, notifications, connector state
    - user_model.db   — Cognitive model (episodes, facts, routines, predictions)
    - preferences.db  — User settings, rules, feedback

This test suite verifies:
    1. Database initialization creates all tables and indexes
    2. Connection management applies WAL mode and foreign keys
    3. Transaction rollback on errors preserves consistency
    4. Schema constraints (primary keys, foreign keys, indexes) are enforced
    5. All five databases can be accessed independently
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


class TestDatabaseManagerInitialization:
    """Test database initialization and schema creation."""

    def test_creates_data_directory_on_init(self):
        """DatabaseManager should create the data directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "new_data_dir"
            assert not data_dir.exists()

            db_manager = DatabaseManager(str(data_dir))

            assert data_dir.exists()
            assert data_dir.is_dir()

    def test_initializes_all_five_databases(self):
        """initialize_all() should create all five domain-separated databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Verify all five database files were created
            assert (Path(tmpdir) / "events.db").exists()
            assert (Path(tmpdir) / "entities.db").exists()
            assert (Path(tmpdir) / "state.db").exists()
            assert (Path(tmpdir) / "user_model.db").exists()
            assert (Path(tmpdir) / "preferences.db").exists()

    def test_events_db_schema(self):
        """events.db should have events, event_processing_log, and event_tags tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                # Check events table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
                )
                assert cursor.fetchone() is not None

                # Check event_processing_log table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='event_processing_log'"
                )
                assert cursor.fetchone() is not None

                # Check event_tags table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='event_tags'"
                )
                assert cursor.fetchone() is not None

    def test_events_db_indexes(self):
        """events.db should have indexes on type, source, timestamp, priority, and payload.message_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
                )
                indexes = {row[0] for row in cursor.fetchall()}

                # Verify expected indexes exist
                assert "idx_events_type" in indexes
                assert "idx_events_source" in indexes
                assert "idx_events_timestamp" in indexes
                assert "idx_events_priority" in indexes
                assert "idx_events_payload_message_id" in indexes

    def test_entities_db_schema(self):
        """entities.db should have contacts, contact_identifiers, places, subscriptions, and entity_relationships."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("entities") as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = {row[0] for row in cursor.fetchall()}

                assert "contacts" in tables
                assert "contact_identifiers" in tables
                assert "places" in tables
                assert "subscriptions" in tables
                assert "entity_relationships" in tables

    def test_state_db_schema(self):
        """state.db should have tasks, notifications, connector_state, and kv_store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("state") as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = {row[0] for row in cursor.fetchall()}

                assert "tasks" in tables
                assert "notifications" in tables
                assert "connector_state" in tables
                assert "kv_store" in tables

    def test_user_model_db_schema(self):
        """user_model.db should have episodes, semantic_facts, routines, communication_templates, signal_profiles, mood_history, predictions, and insights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("user_model") as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = {row[0] for row in cursor.fetchall()}

                assert "episodes" in tables
                assert "semantic_facts" in tables
                assert "routines" in tables
                assert "communication_templates" in tables
                assert "signal_profiles" in tables
                assert "mood_history" in tables
                assert "predictions" in tables
                assert "insights" in tables

    def test_preferences_db_schema(self):
        """preferences.db should have user_preferences, rules, feedback_log, and vaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("preferences") as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = {row[0] for row in cursor.fetchall()}

                assert "user_preferences" in tables
                assert "rules" in tables
                assert "feedback_log" in tables
                assert "vaults" in tables


class TestConnectionManagement:
    """Test connection lifecycle, WAL mode, and foreign key enforcement."""

    def test_get_connection_enables_wal_mode(self):
        """All connections should use WAL (Write-Ahead Logging) mode for concurrent read/write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                cursor = conn.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                assert journal_mode.lower() == "wal"

    def test_get_connection_enables_foreign_keys(self):
        """All connections should enforce foreign key constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("entities") as conn:
                cursor = conn.execute("PRAGMA foreign_keys")
                foreign_keys_enabled = cursor.fetchone()[0]
                assert foreign_keys_enabled == 1

    def test_get_connection_sets_row_factory(self):
        """Connections should use sqlite3.Row for dict-like column access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                assert conn.row_factory == sqlite3.Row

    def test_connection_commits_on_success(self):
        """Changes should be committed when the context manager exits normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Insert data in a transaction
            with db_manager.get_connection("state") as conn:
                conn.execute(
                    "INSERT INTO kv_store (key, value) VALUES (?, ?)",
                    ("test_key", "test_value")
                )

            # Verify data persisted in a new connection
            with db_manager.get_connection("state") as conn:
                cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", ("test_key",))
                row = cursor.fetchone()
                assert row is not None
                assert row["value"] == "test_value"

    def test_connection_rolls_back_on_exception(self):
        """Changes should be rolled back when an exception is raised in the context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Insert data then raise exception
            try:
                with db_manager.get_connection("state") as conn:
                    conn.execute(
                        "INSERT INTO kv_store (key, value) VALUES (?, ?)",
                        ("rollback_test", "should_not_persist")
                    )
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Verify data was rolled back
            with db_manager.get_connection("state") as conn:
                cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", ("rollback_test",))
                row = cursor.fetchone()
                assert row is None

    def test_multiple_concurrent_connections_to_same_db(self):
        """WAL mode should allow multiple concurrent readers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Insert test data
            with db_manager.get_connection("events") as conn:
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("test-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

            # Open two concurrent connections and read the same data
            with db_manager.get_connection("events") as conn1:
                with db_manager.get_connection("events") as conn2:
                    cursor1 = conn1.execute("SELECT id FROM events WHERE id = ?", ("test-1",))
                    cursor2 = conn2.execute("SELECT id FROM events WHERE id = ?", ("test-1",))

                    assert cursor1.fetchone()["id"] == "test-1"
                    assert cursor2.fetchone()["id"] == "test-1"

    def test_connection_to_nonexistent_database_raises(self):
        """Attempting to connect to a database that wasn't initialized should raise KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            # Don't call initialize_all()

            with pytest.raises(KeyError):
                with db_manager.get_connection("nonexistent_db"):
                    pass


class TestForeignKeyConstraints:
    """Test that foreign key relationships are enforced."""

    def test_contact_identifiers_enforces_contact_fk(self):
        """contact_identifiers should enforce FOREIGN KEY to contacts(id)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("entities") as conn:
                # Attempting to insert identifier for non-existent contact should fail
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO contact_identifiers (identifier, identifier_type, contact_id) VALUES (?, ?, ?)",
                        ("test@example.com", "email", "nonexistent-contact-id")
                    )

    def test_contact_identifiers_allows_valid_fk(self):
        """contact_identifiers should allow inserts when contact_id references existing contact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("entities") as conn:
                # Create contact
                conn.execute(
                    "INSERT INTO contacts (id, name) VALUES (?, ?)",
                    ("contact-123", "Test Person")
                )

                # Create identifier referencing that contact — should succeed
                conn.execute(
                    "INSERT INTO contact_identifiers (identifier, identifier_type, contact_id) VALUES (?, ?, ?)",
                    ("test@example.com", "email", "contact-123")
                )

                # Verify it was inserted
                cursor = conn.execute(
                    "SELECT contact_id FROM contact_identifiers WHERE identifier = ?",
                    ("test@example.com",)
                )
                row = cursor.fetchone()
                assert row["contact_id"] == "contact-123"


class TestPrimaryKeyConstraints:
    """Test that primary key constraints prevent duplicates."""

    def test_events_pk_prevents_duplicate_ids(self):
        """events table should reject duplicate IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

                # Attempt to insert duplicate ID should fail
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                        ("event-1", "test.event.other", "test", "2026-02-15T12:01:00Z")
                    )

    def test_event_tags_composite_pk_prevents_duplicate_tag_per_event(self):
        """event_tags should enforce composite PRIMARY KEY (event_id, tag)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                # Create event
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

                # Add tag
                conn.execute(
                    "INSERT INTO event_tags (event_id, tag) VALUES (?, ?)",
                    ("event-1", "marketing")
                )

                # Attempt to add same tag again should fail
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO event_tags (event_id, tag) VALUES (?, ?)",
                        ("event-1", "marketing")
                    )

    def test_event_processing_log_composite_pk(self):
        """event_processing_log should enforce composite PRIMARY KEY (event_id, service)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                # Create event
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

                # Log processing
                conn.execute(
                    "INSERT INTO event_processing_log (event_id, service, result) VALUES (?, ?, ?)",
                    ("event-1", "signal_extractor", "success")
                )

                # Attempt to log same service processing same event again should fail
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO event_processing_log (event_id, service, result) VALUES (?, ?, ?)",
                        ("event-1", "signal_extractor", "retry")
                    )


class TestDefaultValues:
    """Test that schema default values are applied correctly."""

    def test_events_default_priority(self):
        """events.priority should default to 'normal'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

                cursor = conn.execute("SELECT priority FROM events WHERE id = ?", ("event-1",))
                row = cursor.fetchone()
                assert row["priority"] == "normal"

    def test_events_default_created_at(self):
        """events.created_at should auto-populate with current timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

                cursor = conn.execute("SELECT created_at FROM events WHERE id = ?", ("event-1",))
                row = cursor.fetchone()
                # Verify created_at is set (should be ISO timestamp)
                assert row["created_at"] is not None
                assert "T" in row["created_at"]
                assert "Z" in row["created_at"]

    def test_tasks_default_status(self):
        """tasks.status should default to 'pending'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("state") as conn:
                conn.execute(
                    "INSERT INTO tasks (id, title) VALUES (?, ?)",
                    ("task-1", "Test task")
                )

                cursor = conn.execute("SELECT status FROM tasks WHERE id = ?", ("task-1",))
                row = cursor.fetchone()
                assert row["status"] == "pending"

    def test_semantic_facts_default_confidence(self):
        """semantic_facts.confidence should default to 0.5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("user_model") as conn:
                conn.execute(
                    "INSERT INTO semantic_facts (key, category, value) VALUES (?, ?, ?)",
                    ("test_fact", "preference", "tea")
                )

                cursor = conn.execute("SELECT confidence FROM semantic_facts WHERE key = ?", ("test_fact",))
                row = cursor.fetchone()
                assert row["confidence"] == 0.5


class TestIndexEffectiveness:
    """Test that indexes are functioning correctly."""

    def test_events_type_index_query_plan(self):
        """Queries filtering by type should use idx_events_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("events") as conn:
                # Check query plan
                cursor = conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM events WHERE type = 'email.received'"
                )
                plan = " ".join(row[3] for row in cursor.fetchall()).lower()

                # Verify the index is being used
                assert "idx_events_type" in plan or "using index" in plan

    def test_contact_identifiers_index_query_plan(self):
        """Queries filtering by identifier should use composite PRIMARY KEY."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            with db_manager.get_connection("entities") as conn:
                cursor = conn.execute(
                    "EXPLAIN QUERY PLAN SELECT contact_id FROM contact_identifiers WHERE identifier = 'test@example.com'"
                )
                plan = " ".join(row[3] for row in cursor.fetchall()).lower()

                # Primary key should be used for lookups
                assert "using index" in plan or "primary key" in plan


class TestDatabaseIndependence:
    """Test that the five databases are independent and can be backed up separately."""

    def test_can_delete_state_db_without_affecting_events_db(self):
        """Deleting state.db should not affect data in events.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Add data to events.db
            with db_manager.get_connection("events") as conn:
                conn.execute(
                    "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                    ("event-1", "test.event", "test", "2026-02-15T12:00:00Z")
                )

            # Add data to state.db
            with db_manager.get_connection("state") as conn:
                conn.execute(
                    "INSERT INTO tasks (id, title) VALUES (?, ?)",
                    ("task-1", "Test task")
                )

            # Delete state.db
            state_db_path = Path(tmpdir) / "state.db"
            state_db_path.unlink()

            # Verify events.db still has data
            with db_manager.get_connection("events") as conn:
                cursor = conn.execute("SELECT id FROM events WHERE id = ?", ("event-1",))
                row = cursor.fetchone()
                assert row["id"] == "event-1"

            # Re-initialize state.db should work
            db_manager._init_state_db()
            with db_manager.get_connection("state") as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM tasks")
                count = cursor.fetchone()[0]
                assert count == 0  # New empty database

    def test_all_databases_have_independent_schemas(self):
        """Each database should have its own distinct set of domain tables.

        Infrastructure/bookkeeping tables that SQLite creates internally
        ("sqlite_sequence") or that each database legitimately owns for its
        own schema-migration tracking ("schema_version") are excluded from
        the domain-table uniqueness check.  Every OTHER table must belong to
        exactly one database — that boundary is what keeps the 5-database
        architecture clean and prevents cross-concern coupling.
        """
        # These tables are infrastructure shared by design across DBs:
        #   - schema_version: per-DB migration tracking (events + user_model use it)
        #   - sqlite_sequence: SQLite internal AUTOINCREMENT bookkeeping
        SHARED_INFRASTRUCTURE_TABLES = {"schema_version", "sqlite_sequence"}

        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = DatabaseManager(tmpdir)
            db_manager.initialize_all()

            # Collect domain tables for each database (excluding infrastructure)
            tables_by_db = {}
            for db_name in ["events", "entities", "state", "user_model", "preferences"]:
                with db_manager.get_connection(db_name) as conn:
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    )
                    all_tables = {row[0] for row in cursor.fetchall()}
                    # Only keep domain tables, not shared infrastructure
                    tables_by_db[db_name] = all_tables - SHARED_INFRASTRUCTURE_TABLES

            # Verify no domain table appears in multiple databases
            domain_tables: list[str] = []
            for tables in tables_by_db.values():
                domain_tables.extend(tables)

            assert len(domain_tables) == len(set(domain_tables)), (
                "Domain tables should not be duplicated across databases — "
                "each table should belong to exactly one database"
            )
