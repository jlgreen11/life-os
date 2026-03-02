"""
Life OS — Schema Migration Tests (v2 to v4) for user_model.db

Tests the automatic schema migrations:
  - v2 → v3: Adds `filter_reason TEXT` column to the predictions table
  - v3 → v4: Adds `resolution_reason TEXT` column to the predictions table

These migrations modify the predictions table structure which stores the
system's forward-looking intelligence. A regression in the ALTER TABLE
operations could silently lose prediction data or prevent startup.

The v0→v2 migration path has thorough tests in test_schema_migration_v0_to_v2.py.
This file provides equivalent coverage for the v2→v4 migration path.
"""

import sqlite3
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from storage.manager import DatabaseManager


def _create_user_model_db_at_version(db_path: str, version: int) -> None:
    """Create a user_model.db file with the schema at a specific version.

    This helper builds the database schema as it would have existed at
    a given schema version, allowing migration tests to start from a
    known state.

    Args:
        db_path: Full path to the user_model.db file to create.
        version: Target schema version (2, 3, or 4).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Base tables present at v2 (after v0→v1 rebuilt schema, v1→v2 cleaned templates)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id                  TEXT PRIMARY KEY,
            timestamp           TEXT NOT NULL,
            event_id            TEXT NOT NULL,
            location            TEXT,
            inferred_mood       TEXT,
            active_domain       TEXT,
            energy_level        REAL,
            interaction_type    TEXT NOT NULL,
            content_summary     TEXT NOT NULL,
            content_full        TEXT,
            contacts_involved   TEXT DEFAULT '[]',
            topics              TEXT DEFAULT '[]',
            entities            TEXT DEFAULT '[]',
            outcome             TEXT,
            user_satisfaction   REAL,
            embedding_id        TEXT,
            created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS semantic_facts (
            key                 TEXT PRIMARY KEY,
            category            TEXT NOT NULL,
            value               TEXT NOT NULL,
            confidence          REAL DEFAULT 0.5,
            source_episodes     TEXT DEFAULT '[]',
            first_observed      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            last_confirmed      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            times_confirmed     INTEGER DEFAULT 1,
            is_user_corrected   INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS routines (
            name                TEXT PRIMARY KEY,
            trigger_condition   TEXT NOT NULL,
            steps               TEXT NOT NULL DEFAULT '[]',
            typical_duration    REAL DEFAULT 30.0,
            consistency_score   REAL DEFAULT 0.5,
            times_observed      INTEGER DEFAULT 0,
            variations          TEXT DEFAULT '[]',
            updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS workflows (
            name                TEXT PRIMARY KEY,
            trigger_conditions  TEXT NOT NULL DEFAULT '[]',
            steps               TEXT NOT NULL DEFAULT '[]',
            typical_duration    REAL,
            tools_used          TEXT DEFAULT '[]',
            success_rate        REAL DEFAULT 0.5,
            times_observed      INTEGER DEFAULT 0,
            updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS communication_templates (
            id                  TEXT PRIMARY KEY,
            context             TEXT NOT NULL,
            contact_id          TEXT,
            channel             TEXT,
            greeting            TEXT,
            closing             TEXT,
            formality           REAL DEFAULT 0.5,
            typical_length      REAL DEFAULT 50.0,
            uses_emoji          INTEGER DEFAULT 0,
            common_phrases      TEXT DEFAULT '[]',
            avoids_phrases      TEXT DEFAULT '[]',
            tone_notes          TEXT DEFAULT '[]',
            example_message_ids TEXT DEFAULT '[]',
            samples_analyzed    INTEGER DEFAULT 0,
            updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS signal_profiles (
            profile_type        TEXT PRIMARY KEY,
            data                TEXT NOT NULL DEFAULT '{}',
            samples_count       INTEGER DEFAULT 0,
            updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS mood_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT NOT NULL,
            energy_level        REAL,
            stress_level        REAL,
            social_battery      REAL,
            cognitive_load      REAL,
            emotional_valence   REAL,
            confidence          REAL,
            contributing_signals TEXT DEFAULT '[]',
            trend               TEXT DEFAULT 'stable'
        );

        CREATE TABLE IF NOT EXISTS insights (
            id                  TEXT PRIMARY KEY,
            type                TEXT NOT NULL,
            summary             TEXT NOT NULL,
            confidence          REAL NOT NULL,
            evidence            TEXT DEFAULT '[]',
            category            TEXT DEFAULT '',
            entity              TEXT,
            staleness_ttl_hours INTEGER DEFAULT 168,
            dedup_key           TEXT,
            feedback            TEXT,
            created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
    """)

    # Predictions table — columns depend on version
    if version == 2:
        # v2 schema: no filter_reason, no resolution_reason
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  TEXT PRIMARY KEY,
                prediction_type     TEXT NOT NULL,
                description         TEXT NOT NULL,
                confidence          REAL NOT NULL,
                confidence_gate     TEXT NOT NULL,
                time_horizon        TEXT,
                suggested_action    TEXT,
                supporting_signals  TEXT DEFAULT '[]',
                was_surfaced        INTEGER DEFAULT 0,
                user_response       TEXT,
                was_accurate        INTEGER,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                resolved_at         TEXT
            )
        """)
    elif version == 3:
        # v3 schema: has filter_reason, no resolution_reason
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  TEXT PRIMARY KEY,
                prediction_type     TEXT NOT NULL,
                description         TEXT NOT NULL,
                confidence          REAL NOT NULL,
                confidence_gate     TEXT NOT NULL,
                time_horizon        TEXT,
                suggested_action    TEXT,
                supporting_signals  TEXT DEFAULT '[]',
                was_surfaced        INTEGER DEFAULT 0,
                user_response       TEXT,
                was_accurate        INTEGER,
                filter_reason       TEXT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                resolved_at         TEXT
            )
        """)
    elif version >= 4:
        # v4 schema: has both filter_reason and resolution_reason
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  TEXT PRIMARY KEY,
                prediction_type     TEXT NOT NULL,
                description         TEXT NOT NULL,
                confidence          REAL NOT NULL,
                confidence_gate     TEXT NOT NULL,
                time_horizon        TEXT,
                suggested_action    TEXT,
                supporting_signals  TEXT DEFAULT '[]',
                was_surfaced        INTEGER DEFAULT 0,
                user_response       TEXT,
                was_accurate        INTEGER,
                filter_reason       TEXT,
                resolution_reason   TEXT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                resolved_at         TEXT
            )
        """)

    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    conn.commit()
    conn.close()


def _get_prediction_columns(conn: sqlite3.Connection) -> set[str]:
    """Return the set of column names on the predictions table.

    Args:
        conn: Active connection to user_model.db.

    Returns:
        Set of column name strings.
    """
    rows = conn.execute("PRAGMA table_info(predictions)").fetchall()
    return {row[1] for row in rows}


def _insert_sample_prediction(
    conn: sqlite3.Connection,
    *,
    prediction_id: str | None = None,
    filter_reason: str | None = None,
    resolution_reason: str | None = None,
    columns_present: set[str] | None = None,
) -> str:
    """Insert a sample prediction row, adapting to whichever columns exist.

    Args:
        conn: Active connection to user_model.db.
        prediction_id: Optional explicit ID; auto-generated if omitted.
        filter_reason: Value for filter_reason column (only used if column exists).
        resolution_reason: Value for resolution_reason column (only used if column exists).
        columns_present: Set of column names on the table (queried if omitted).

    Returns:
        The prediction ID.
    """
    if prediction_id is None:
        prediction_id = str(uuid4())

    if columns_present is None:
        columns_present = _get_prediction_columns(conn)

    # Base columns always present at v2+
    cols = ["id", "prediction_type", "description", "confidence", "confidence_gate"]
    vals = [prediction_id, "NEED", "Test prediction", 0.75, "SUGGEST"]

    if "filter_reason" in columns_present and filter_reason is not None:
        cols.append("filter_reason")
        vals.append(filter_reason)

    if "resolution_reason" in columns_present and resolution_reason is not None:
        cols.append("resolution_reason")
        vals.append(resolution_reason)

    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)
    conn.execute(f"INSERT INTO predictions ({col_list}) VALUES ({placeholders})", vals)
    conn.commit()
    return prediction_id


class TestMigrationV2toV3:
    """Test suite for user_model.db schema migration from v2 to v3.

    The v2→v3 migration adds a `filter_reason TEXT` column to the predictions
    table, enabling observability into why predictions are filtered.
    """

    @pytest.fixture
    def v2_db_dir(self):
        """Create a temporary directory with a v2-state user_model.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "user_model.db")
            _create_user_model_db_at_version(db_path, version=2)
            yield tmpdir

    def test_adds_filter_reason_column(self, v2_db_dir):
        """Migration v2→v3 should add filter_reason column to predictions."""
        # Verify column is absent before migration
        conn = sqlite3.connect(str(Path(v2_db_dir) / "user_model.db"))
        columns_before = _get_prediction_columns(conn)
        conn.close()
        assert "filter_reason" not in columns_before

        # Run migration via DatabaseManager initialization
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        # Verify column now exists
        with db.get_connection("user_model") as conn:
            columns_after = _get_prediction_columns(conn)
            assert "filter_reason" in columns_after

    def test_preserves_existing_predictions(self, v2_db_dir):
        """Migration should not lose existing prediction rows."""
        # Insert predictions before migration
        conn = sqlite3.connect(str(Path(v2_db_dir) / "user_model.db"))
        pred_ids = []
        for i in range(5):
            pid = _insert_sample_prediction(conn, prediction_id=f"pred-v2-{i}")
            pred_ids.append(pid)
        conn.close()

        # Run migration
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        # Verify all rows survived and filter_reason defaults to NULL
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, prediction_type, description, confidence, filter_reason "
                "FROM predictions ORDER BY id"
            ).fetchall()

            assert len(rows) == 5
            for row in rows:
                assert row[0] in pred_ids
                assert row[1] == "NEED"
                assert row[2] == "Test prediction"
                assert row[3] == 0.75
                # filter_reason should be NULL for pre-migration rows
                assert row[4] is None

    def test_idempotent_v2_to_v3(self, v2_db_dir):
        """Running migration twice should not error or duplicate columns."""
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        # Run again — should be a no-op
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            columns = _get_prediction_columns(conn)
            assert "filter_reason" in columns

            # Verify no duplicate columns (PRAGMA returns one row per column)
            all_cols = conn.execute("PRAGMA table_info(predictions)").fetchall()
            col_names = [row[1] for row in all_cols]
            assert col_names.count("filter_reason") == 1

    def test_schema_version_updated(self, v2_db_dir):
        """Migration should advance schema_version to at least v3."""
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 3


class TestMigrationV3toV4:
    """Test suite for user_model.db schema migration from v3 to v4.

    The v3→v4 migration adds a `resolution_reason TEXT` column to the
    predictions table, enabling the accuracy multiplier to distinguish
    automated-sender fast-path resolutions from real user-behavior signals.
    """

    @pytest.fixture
    def v3_db_dir(self):
        """Create a temporary directory with a v3-state user_model.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "user_model.db")
            _create_user_model_db_at_version(db_path, version=3)
            yield tmpdir

    def test_adds_resolution_reason_column(self, v3_db_dir):
        """Migration v3→v4 should add resolution_reason column to predictions."""
        # Verify column is absent before migration
        conn = sqlite3.connect(str(Path(v3_db_dir) / "user_model.db"))
        columns_before = _get_prediction_columns(conn)
        conn.close()
        assert "resolution_reason" not in columns_before
        assert "filter_reason" in columns_before  # v3 should already have this

        # Run migration
        db = DatabaseManager(v3_db_dir)
        db.initialize_all()

        # Verify column now exists
        with db.get_connection("user_model") as conn:
            columns_after = _get_prediction_columns(conn)
            assert "resolution_reason" in columns_after
            assert "filter_reason" in columns_after  # should still be there

    def test_preserves_existing_data_with_filter_reason(self, v3_db_dir):
        """Migration should preserve rows including their filter_reason values."""
        # Insert predictions with filter_reason set (v3 feature)
        conn = sqlite3.connect(str(Path(v3_db_dir) / "user_model.db"))
        pred_ids = []
        reasons = ["low_confidence", "duplicate", None, "quiet_hours", "stress_filter"]
        for i, reason in enumerate(reasons):
            pid = _insert_sample_prediction(
                conn,
                prediction_id=f"pred-v3-{i}",
                filter_reason=reason,
            )
            pred_ids.append(pid)
        conn.close()

        # Run migration
        db = DatabaseManager(v3_db_dir)
        db.initialize_all()

        # Verify all rows survived with their filter_reason intact
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, filter_reason, resolution_reason "
                "FROM predictions ORDER BY id"
            ).fetchall()

            assert len(rows) == 5
            for row in rows:
                assert row[0] in pred_ids
                # resolution_reason should be NULL for pre-migration rows
                assert row[2] is None

            # Verify specific filter_reason values survived
            reason_map = {row[0]: row[1] for row in rows}
            assert reason_map["pred-v3-0"] == "low_confidence"
            assert reason_map["pred-v3-1"] == "duplicate"
            assert reason_map["pred-v3-2"] is None
            assert reason_map["pred-v3-3"] == "quiet_hours"
            assert reason_map["pred-v3-4"] == "stress_filter"

    def test_idempotent_v3_to_v4(self, v3_db_dir):
        """Running migration twice should not error or duplicate columns."""
        db = DatabaseManager(v3_db_dir)
        db.initialize_all()

        # Run again
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            columns = _get_prediction_columns(conn)
            assert "resolution_reason" in columns
            assert "filter_reason" in columns

            # Verify no duplicate columns
            all_cols = conn.execute("PRAGMA table_info(predictions)").fetchall()
            col_names = [row[1] for row in all_cols]
            assert col_names.count("resolution_reason") == 1
            assert col_names.count("filter_reason") == 1

    def test_schema_version_updated(self, v3_db_dir):
        """Migration should advance schema_version to v4."""
        db = DatabaseManager(v3_db_dir)
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version >= 4


class TestFullMigrationV2toV4:
    """Test suite for the complete v2→v4 migration path.

    Verifies that both migrations (v2→v3 and v3→v4) are applied sequentially
    in a single DatabaseManager initialization when the database starts at v2.
    """

    @pytest.fixture
    def v2_db_dir(self):
        """Create a temporary directory with a v2-state user_model.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "user_model.db")
            _create_user_model_db_at_version(db_path, version=2)
            yield tmpdir

    def test_v2_to_v4_sequential(self, v2_db_dir):
        """Starting from v2, both columns should be added in one initialization."""
        # Verify neither column exists before migration
        conn = sqlite3.connect(str(Path(v2_db_dir) / "user_model.db"))
        columns_before = _get_prediction_columns(conn)
        conn.close()
        assert "filter_reason" not in columns_before
        assert "resolution_reason" not in columns_before

        # Single initialization should run both v2→v3 and v3→v4
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            columns_after = _get_prediction_columns(conn)
            assert "filter_reason" in columns_after
            assert "resolution_reason" in columns_after

            # Schema version should be at v4
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            assert version == 4

    def test_v2_to_v4_preserves_data(self, v2_db_dir):
        """Predictions inserted at v2 should survive the full v2→v4 migration."""
        # Insert rows at v2 schema
        conn = sqlite3.connect(str(Path(v2_db_dir) / "user_model.db"))
        pred_ids = []
        for i in range(3):
            pid = _insert_sample_prediction(conn, prediction_id=f"pred-full-{i}")
            pred_ids.append(pid)
        conn.close()

        # Migrate v2→v4
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        # Verify data integrity
        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, prediction_type, description, confidence, "
                "filter_reason, resolution_reason FROM predictions ORDER BY id"
            ).fetchall()

            assert len(rows) == 3
            for row in rows:
                assert row[0] in pred_ids
                assert row[1] == "NEED"
                assert row[2] == "Test prediction"
                assert row[3] == 0.75
                # Both new columns should default to NULL
                assert row[4] is None  # filter_reason
                assert row[5] is None  # resolution_reason

    def test_predictions_table_missing_v3_to_v4_gracefully(self):
        """v3→v4 migration should not error if predictions table doesn't exist.

        The v3→v4 migration explicitly checks for table existence before
        ALTER TABLE. If the predictions table is absent (e.g. partial init),
        the migration should skip the ALTER and let CREATE TABLE IF NOT EXISTS
        create it with the full v4 schema afterwards.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "user_model.db")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")

            # Create schema_version at v3 but NO predictions table.
            # The v3→v4 migration guards against missing table, so this
            # should not raise.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version     INTEGER PRIMARY KEY,
                    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)
            conn.execute("INSERT INTO schema_version (version) VALUES (3)")
            conn.commit()
            conn.close()

            # Should not raise — the v3→v4 migration checks table_exists
            db = DatabaseManager(tmpdir)
            db.initialize_all()

            # Predictions table should now exist (created by CREATE TABLE IF NOT EXISTS)
            # with all v4 columns since it's created fresh from the DDL
            with db.get_connection("user_model") as conn:
                columns = _get_prediction_columns(conn)
                assert "filter_reason" in columns
                assert "resolution_reason" in columns

    def test_new_predictions_use_both_columns_after_migration(self, v2_db_dir):
        """After migration, new predictions can use both filter_reason and resolution_reason."""
        db = DatabaseManager(v2_db_dir)
        db.initialize_all()

        with db.get_connection("user_model") as conn:
            # Insert a prediction using the new columns
            pred_id = str(uuid4())
            conn.execute(
                "INSERT INTO predictions "
                "(id, prediction_type, description, confidence, confidence_gate, "
                " filter_reason, resolution_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pred_id, "OPPORTUNITY", "Test post-migration", 0.6, "SUGGEST",
                 "low_confidence", "automated_sender_fast_path"),
            )
            conn.commit()

            row = conn.execute(
                "SELECT filter_reason, resolution_reason FROM predictions WHERE id = ?",
                (pred_id,),
            ).fetchone()
            assert row[0] == "low_confidence"
            assert row[1] == "automated_sender_fast_path"
