"""
Test schema migration to ensure new tables are created in existing databases.

This test verifies that when the schema is updated with new tables, existing
database files are properly migrated to include the new schema elements.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from storage.manager import DatabaseManager


class TestSchemaMigration:
    """Test that schema changes are applied to existing databases."""

    def test_new_tables_created_in_existing_database(self):
        """
        Verify that when initialize_all() is called on an existing database,
        new tables defined in the schema are created.

        This test simulates the real-world scenario where:
        1. An old version of Life OS creates a database with some tables
        2. Code is updated to add new tables to the schema
        3. initialize_all() is called again
        4. The new tables should be created in the existing database file
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Step 1: Create a "legacy" database with only one table
            legacy_db = str(data_dir / "user_model.db")
            conn = sqlite3.connect(legacy_db)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()

            # Verify the legacy database has only one table
            conn = sqlite3.connect(legacy_db)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            conn.close()
            assert len(tables) == 1
            assert tables[0][0] == "episodes"

            # Step 2: Run initialize_all() which should create all missing tables
            db = DatabaseManager(str(data_dir))
            db.initialize_all()

            # Step 3: Verify ALL tables from the current schema now exist
            with db.get_connection("user_model") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                table_names = {t[0] for t in tables}

            # All tables from the current schema should exist
            required_tables = {
                "episodes",
                "semantic_facts",
                "routines",
                "workflows",
                "communication_templates",
                "signal_profiles",
                "mood_history",
                "predictions",  # CRITICAL: This table is missing in production!
                "insights",
            }

            missing_tables = required_tables - table_names
            assert not missing_tables, f"Missing tables: {missing_tables}"

    def test_predictions_table_has_correct_schema(self):
        """
        Verify the predictions table has all required columns.

        This test ensures that the predictions table schema matches what
        UserModelStore.store_prediction() expects.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(tmpdir)
            db.initialize_all()

            with db.get_connection("user_model") as conn:
                # Get column info for predictions table
                columns = conn.execute("PRAGMA table_info(predictions)").fetchall()
                column_names = {col[1] for col in columns}

            required_columns = {
                "id",
                "prediction_type",
                "description",
                "confidence",
                "confidence_gate",
                "time_horizon",
                "suggested_action",
                "supporting_signals",
                "was_surfaced",
                "user_response",
                "was_accurate",
                "created_at",
                "resolved_at",
            }

            missing_columns = required_columns - column_names
            assert not missing_columns, f"Missing columns in predictions: {missing_columns}"

    def test_all_databases_initialized_correctly(self):
        """
        Verify all 5 databases are created with their full schemas.

        This is a comprehensive check that initialize_all() creates all
        required tables across all five database files.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(tmpdir)
            db.initialize_all()

            # Check events.db
            with db.get_connection("events") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "events" in table_names
                assert "event_tags" in table_names

            # Check entities.db
            with db.get_connection("entities") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "contacts" in table_names
                assert "places" in table_names

            # Check state.db
            with db.get_connection("state") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "tasks" in table_names
                assert "notifications" in table_names

            # Check user_model.db (the critical one!)
            with db.get_connection("user_model") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "predictions" in table_names
                assert "episodes" in table_names
                assert "semantic_facts" in table_names
                assert "signal_profiles" in table_names

            # Check preferences.db
            with db.get_connection("preferences") as conn:
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                table_names = {t[0] for t in tables}
                assert "rules" in table_names
                assert "user_preferences" in table_names

    def test_idempotent_initialization(self):
        """
        Verify that calling initialize_all() multiple times is safe.

        The initialization should be idempotent - running it multiple times
        should not cause errors or data loss.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db = DatabaseManager(tmpdir)

            # Run initialization multiple times
            db.initialize_all()
            db.initialize_all()
            db.initialize_all()

            # Should still have valid databases
            with db.get_connection("user_model") as conn:
                result = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()
                assert result[0] > 0  # Has tables
