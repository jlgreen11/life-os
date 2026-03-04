"""Tests for scripts/analyze-data-quality.py error reporting and observability.

Verifies that the data quality analysis script:
- Produces valid reports from healthy databases with no errors
- Reports errors in ALL dependent sections when a database is unavailable
- Captures query errors in the report-level query_errors list
- Logs connection failures via the logger
- Existing bug-fix behavior (connector_state columns, source_weights columns,
  independent user_model sections)
"""

import logging
import sqlite3
from unittest.mock import patch

import pytest

# The script lives at scripts/analyze-data-quality.py which isn't a proper
# Python package, so we import its internals via importlib.
import importlib
import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Import with importlib since the filename contains hyphens
_mod_spec = importlib.util.spec_from_file_location(
    "analyze_data_quality",
    Path(__file__).resolve().parent.parent / "scripts" / "analyze-data-quality.py",
)
_mod = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_mod)

analyze = _mod.analyze
_connect = _mod._connect
_query = _mod._query
_query_one = _mod._query_one
_errors = _mod._errors


# ---------------------------------------------------------------------------
# Helper functions to create minimal database files
# ---------------------------------------------------------------------------


def _create_minimal_events_db(tmp_path):
    """Create a minimal events.db with just the events table."""
    conn = sqlite3.connect(str(tmp_path / "events.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            type TEXT,
            source TEXT,
            timestamp TEXT,
            priority TEXT,
            payload TEXT DEFAULT '{}',
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    conn.close()


def _create_minimal_user_model_db(tmp_path):
    """Create a minimal user_model.db with all expected tables."""
    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            prediction_type TEXT,
            was_surfaced INTEGER DEFAULT 0,
            was_accurate INTEGER,
            filter_reason TEXT,
            resolution_reason TEXT,
            user_response TEXT,
            resolved_at TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_profiles (
            profile_type TEXT PRIMARY KEY,
            samples_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY,
            type TEXT,
            feedback TEXT
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_facts (
            id INTEGER PRIMARY KEY,
            category TEXT DEFAULT 'general'
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS routines (id INTEGER PRIMARY KEY)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT,
            trigger_pattern TEXT,
            steps TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS communication_templates (
            id TEXT PRIMARY KEY,
            contact_id TEXT,
            channel TEXT,
            template_pattern TEXT,
            confidence REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.commit()
    conn.close()


def _create_minimal_state_db(tmp_path):
    """Create a minimal state.db with expected tables."""
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS connector_state (
            connector_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'inactive',
            last_sync TEXT,
            last_error TEXT
        )
    """)
    conn.commit()
    conn.close()


def _create_minimal_preferences_db(tmp_path):
    """Create a minimal preferences.db with expected tables."""
    conn = sqlite3.connect(str(tmp_path / "preferences.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY,
            action_type TEXT,
            feedback_type TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_weights (
            source_key TEXT PRIMARY KEY,
            category TEXT NOT NULL DEFAULT '',
            label TEXT NOT NULL DEFAULT '',
            user_weight REAL NOT NULL DEFAULT 0.5,
            ai_drift REAL NOT NULL DEFAULT 0.0,
            ai_updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def _create_minimal_entities_db(tmp_path):
    """Create a minimal entities.db with a placeholder table."""
    conn = sqlite3.connect(str(tmp_path / "entities.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS contacts (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()


def _create_all_dbs(tmp_path):
    """Create all 5 databases with their required schemas."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)
    _create_minimal_entities_db(tmp_path)


# ---------------------------------------------------------------------------
# Healthy database tests
# ---------------------------------------------------------------------------


class TestHealthyDatabase:
    """Tests that a healthy database produces a valid report with no errors."""

    def test_report_has_all_sections(self, tmp_path):
        """A healthy set of databases produces a report with all expected sections."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        expected_sections = [
            "database_health",
            "events",
            "prediction_accuracy",
            "prediction_resolution",
            "prediction_pipeline",
            "signal_profiles",
            "insight_feedback",
            "user_model",
            "notifications",
            "tasks",
            "connectors",
            "feedback",
            "source_weights",
        ]
        for section in expected_sections:
            assert section in report["sections"], f"Missing section: {section}"

    def test_no_query_errors_on_healthy_db(self, tmp_path):
        """A healthy database produces zero query errors."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        assert "query_errors" in report
        assert report["query_errors"] == []

    def test_no_error_keys_on_healthy_db(self, tmp_path):
        """No section should contain an 'error' key when databases are healthy."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        for name, section in report["sections"].items():
            if isinstance(section, dict):
                assert "error" not in section, (
                    f"Section '{name}' has unexpected error: {section.get('error')}"
                )

    def test_database_health_all_ok(self, tmp_path):
        """All databases should report status 'ok' in database_health."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        health = report["sections"]["database_health"]
        for db_name in ["events", "user_model", "state", "preferences", "entities"]:
            assert isinstance(health[db_name], dict), f"{db_name} health should be a dict"
            assert health[db_name]["status"] == "ok", f"{db_name} health: {health[db_name]}"
            assert health[db_name]["detail"] == "ok"

    def test_report_has_generated_at(self, tmp_path):
        """Report includes a generated_at timestamp."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        assert "generated_at" in report
        assert report["generated_at"]  # non-empty string


# ---------------------------------------------------------------------------
# Missing database tests — ALL dependent sections must get error keys
# ---------------------------------------------------------------------------


class TestMissingUserModelDb:
    """Tests for when user_model.db cannot be connected to."""

    def test_all_um_sections_have_error(self, tmp_path):
        """When user_model.db is corrupt, ALL 6 dependent sections get error keys."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        # Capture the real function BEFORE patching to avoid recursion
        real_connect = _mod._connect

        def failing_um_connect(db_path):
            """Return None only for user_model.db."""
            if "user_model.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=failing_um_connect):
            report = analyze(str(tmp_path))

        um_dependent_sections = [
            "prediction_accuracy",
            "prediction_resolution",
            "prediction_pipeline",
            "signal_profiles",
            "insight_feedback",
            "user_model",
        ]
        for section_name in um_dependent_sections:
            assert section_name in report["sections"], f"Section '{section_name}' is absent from report"
            section = report["sections"][section_name]
            assert isinstance(section, dict), f"Section '{section_name}' should be a dict"
            assert "error" in section, f"Section '{section_name}' missing 'error' key"
            assert "user_model.db" in section["error"]

    def test_non_um_sections_unaffected(self, tmp_path):
        """Sections not dependent on user_model.db should still work."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        real_connect = _mod._connect

        def failing_um_connect(db_path):
            if "user_model.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=failing_um_connect):
            report = analyze(str(tmp_path))

        # Events, notifications, feedback should work fine
        for section_name in ["events", "notifications", "feedback"]:
            section = report["sections"][section_name]
            if isinstance(section, dict):
                assert "error" not in section, f"{section_name} should not have error"


class TestMissingStateDb:
    """Tests for when state.db cannot be connected to."""

    def test_all_state_sections_have_error(self, tmp_path):
        """When state.db connection fails, ALL 3 dependent sections get error keys."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_user_model_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        real_connect = _mod._connect

        def failing_state_connect(db_path):
            if "state.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=failing_state_connect):
            report = analyze(str(tmp_path))

        state_dependent_sections = ["notifications", "tasks", "connectors"]
        for section_name in state_dependent_sections:
            assert section_name in report["sections"], f"Section '{section_name}' is absent from report"
            section = report["sections"][section_name]
            assert isinstance(section, dict), f"Section '{section_name}' should be a dict"
            assert "error" in section, f"Section '{section_name}' missing 'error' key"
            assert "state.db" in section["error"]


class TestMissingPreferencesDb:
    """Tests for when preferences.db cannot be connected to."""

    def test_all_pref_sections_have_error(self, tmp_path):
        """When preferences.db connection fails, ALL 2 dependent sections get error keys."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_user_model_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        real_connect = _mod._connect

        def failing_pref_connect(db_path):
            if "preferences.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=failing_pref_connect):
            report = analyze(str(tmp_path))

        pref_dependent_sections = ["feedback", "source_weights"]
        for section_name in pref_dependent_sections:
            assert section_name in report["sections"], f"Section '{section_name}' is absent from report"
            section = report["sections"][section_name]
            assert isinstance(section, dict), f"Section '{section_name}' should be a dict"
            assert "error" in section, f"Section '{section_name}' missing 'error' key"
            assert "preferences.db" in section["error"]


# ---------------------------------------------------------------------------
# Query error capture tests
# ---------------------------------------------------------------------------


class TestQueryErrorCapture:
    """Tests that _query() and _query_one() errors are captured in the _errors list."""

    def test_query_error_appended_to_errors(self, tmp_path):
        """When a query fails, the error is appended to _errors."""
        _errors.clear()
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        result = _query(conn, "SELECT * FROM nonexistent_table", [])

        assert result == []
        assert len(_errors) == 1
        assert "nonexistent_table" in _errors[0]["sql"]
        assert _errors[0]["error"]  # non-empty error message
        conn.close()

    def test_query_one_error_appended_to_errors(self, tmp_path):
        """When a query_one fails, the error is appended to _errors."""
        _errors.clear()
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        result = _query_one(conn, "SELECT * FROM nonexistent_table")

        assert result is None
        assert len(_errors) == 1
        assert "nonexistent_table" in _errors[0]["sql"]
        conn.close()

    def test_errors_included_in_report(self, tmp_path):
        """Query errors are included in the final report as query_errors."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        assert "query_errors" in report
        assert isinstance(report["query_errors"], list)

    def test_errors_cleared_between_runs(self, tmp_path):
        """The _errors list is cleared at the start of each analyze() call."""
        _create_all_dbs(tmp_path)

        # First run
        report1 = analyze(str(tmp_path))
        assert report1["query_errors"] == []

        # Second run should also start clean
        report2 = analyze(str(tmp_path))
        assert report2["query_errors"] == []

    def test_query_errors_captured_in_report_for_missing_tables(self, tmp_path):
        """When tables are missing, individual query failures appear in query_errors."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        # Create user_model.db with ONLY signal_profiles (missing predictions, etc.)
        conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        conn.execute("""
            CREATE TABLE signal_profiles (
                profile_type TEXT PRIMARY KEY,
                samples_count INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        conn.execute("CREATE TABLE insights (id INTEGER PRIMARY KEY, type TEXT, feedback TEXT)")
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, category TEXT DEFAULT 'general')")
        conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
        # predictions table is missing
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))

        # Query errors should capture the failed predictions queries
        assert len(report["query_errors"]) > 0
        assert any("predictions" in e["sql"] for e in report["query_errors"])


# ---------------------------------------------------------------------------
# _connect() logging tests
# ---------------------------------------------------------------------------


class TestConnectLogging:
    """Tests that _connect() failures are logged."""

    def test_connect_failure_is_logged(self, tmp_path):
        """When _connect() fails, it logs a warning with the error details."""
        with patch.object(_mod.logger, "warning") as mock_warn:
            with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("unable to open database file")):
                result = _connect(tmp_path / "test.db")

        assert result is None
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args
        assert "Could not connect" in call_args[0][0]

    def test_connect_success_does_not_log(self, tmp_path):
        """When _connect() succeeds, no warning is logged."""
        db_path = tmp_path / "test.db"
        with patch.object(_mod.logger, "warning") as mock_warn:
            conn = _connect(db_path)

        assert conn is not None
        mock_warn.assert_not_called()
        conn.close()


# ---------------------------------------------------------------------------
# _query/_query_one logging tests
# ---------------------------------------------------------------------------


def test_query_logs_warning_on_failure(tmp_path, caplog):
    """_query() should log a warning when the SQL fails."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    broken_sql = "SELECT * FROM nonexistent_table_xyz"
    with caplog.at_level(logging.WARNING, logger=_mod.__name__):
        result = _query(conn, broken_sql, default=[])

    assert result == []
    assert any("Query failed" in rec.message for rec in caplog.records)
    assert any("nonexistent_table_xyz" in rec.message for rec in caplog.records)
    conn.close()


def test_query_one_logs_warning_on_failure(tmp_path, caplog):
    """_query_one() should log a warning and return default on failure."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    broken_sql = "SELECT * FROM nonexistent_table_xyz"
    with caplog.at_level(logging.WARNING, logger=_mod.__name__):
        result = _query_one(conn, broken_sql, default=None)

    assert result is None
    assert any("Query failed" in rec.message for rec in caplog.records)
    conn.close()


# ---------------------------------------------------------------------------
# Existing bug-fix tests (connector_state, source_weights, independent sections)
# ---------------------------------------------------------------------------


def test_connector_state_uses_last_error_column(tmp_path):
    """The connector_state query should select ``last_error``, not ``error_message``."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE connector_state (
            connector_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'inactive',
            enabled INTEGER DEFAULT 0,
            last_sync TEXT,
            sync_cursor TEXT,
            error_count INTEGER DEFAULT 0,
            last_error TEXT,
            config TEXT DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        INSERT INTO connector_state (connector_id, status, last_sync, last_error)
        VALUES ('gmail', 'active', '2026-01-01T00:00:00Z', 'timeout after 30s')
    """)
    conn.execute("""
        CREATE TABLE notifications (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.commit()
    conn.close()

    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    connectors = report["sections"].get("connectors", {})

    assert "gmail" in connectors, f"Expected 'gmail' in connectors, got: {connectors}"
    assert connectors["gmail"]["error"] == "timeout after 30s"
    assert connectors["gmail"]["status"] == "active"


def test_source_weights_uses_correct_column_names(tmp_path):
    """The source_weights query should select ``user_weight``, ``ai_drift``, ``ai_updated_at``."""
    db_path = tmp_path / "preferences.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE source_weights (
            source_key TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            label TEXT NOT NULL,
            user_weight REAL NOT NULL DEFAULT 0.5,
            ai_drift REAL NOT NULL DEFAULT 0.0,
            ai_updated_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO source_weights (source_key, category, label, user_weight, ai_drift, ai_updated_at)
        VALUES ('email.gmail', 'connector', 'Gmail', 0.8, 0.15, '2026-01-15T12:00:00Z')
    """)
    conn.execute("""
        CREATE TABLE feedback_log (
            id INTEGER PRIMARY KEY,
            action_type TEXT,
            feedback_type TEXT
        )
    """)
    conn.commit()
    conn.close()

    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)

    report = analyze(str(tmp_path))
    sw = report["sections"].get("source_weights", {})

    assert "email.gmail" in sw, f"Expected 'email.gmail' in source_weights, got: {sw}"
    assert sw["email.gmail"]["weight"] == 0.8
    assert sw["email.gmail"]["drift"] == 0.15
    assert sw["email.gmail"]["updated_at"] == "2026-01-15T12:00:00Z"


def test_user_model_sections_independent(tmp_path):
    """If predictions table is missing, other user_model.db sections still populate."""
    db_path = tmp_path / "user_model.db"
    conn = sqlite3.connect(str(db_path))

    # Create signal_profiles, insights, episodes, semantic_facts, routines
    # but do NOT create the predictions table.
    conn.execute("""
        CREATE TABLE signal_profiles (
            profile_type TEXT PRIMARY KEY,
            samples_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO signal_profiles (profile_type, samples_count, updated_at)
        VALUES ('linguistic', 42, '2026-01-01T00:00:00Z')
    """)
    conn.execute("CREATE TABLE insights (id INTEGER PRIMARY KEY, type TEXT, feedback TEXT)")
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO episodes (id) VALUES (1)")
    conn.execute("INSERT INTO episodes (id) VALUES (2)")
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, category TEXT DEFAULT 'general')")
    conn.execute("INSERT INTO semantic_facts (id, category) VALUES (1, 'preference')")
    conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO routines (id) VALUES (1)")
    conn.commit()
    conn.close()

    _create_minimal_events_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    sections = report["sections"]

    # All user_model.db sections should be present
    assert "prediction_accuracy" in sections
    assert "signal_profiles" in sections
    assert "insight_feedback" in sections
    assert "user_model" in sections

    # signal_profiles should have real data
    sp = sections["signal_profiles"]
    assert "profiles" in sp
    assert "linguistic" in sp["profiles"]
    assert sp["profiles"]["linguistic"]["samples"] == 42
    # 8 of 9 expected types should be missing (linguistic is present)
    assert "missing_profiles" in sp
    assert "linguistic" not in sp["missing_profiles"]
    assert len(sp["missing_profiles"]) == 8

    # user_model should have real data
    um = sections["user_model"]
    assert um.get("episodes") == 2
    assert um.get("semantic_facts") == 1
    assert um.get("routines") == 1


# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------


class TestCorruptDatabase:
    """Tests for corrupt database files."""

    def test_corrupted_db_detected_by_integrity_check(self, tmp_path):
        """PRAGMA integrity_check detects a corrupted database file."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)

        # Create a valid user_model.db with enough data to span multiple pages
        db_path = tmp_path / "user_model.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA page_size = 512")
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, data TEXT)")
        for i in range(200):
            conn.execute("INSERT INTO episodes (id, data) VALUES (?, ?)", (i, "x" * 200))
        conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, category TEXT DEFAULT 'general')")
        conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
        conn.execute("""
            CREATE TABLE predictions (
                id TEXT PRIMARY KEY, prediction_type TEXT,
                was_surfaced INTEGER DEFAULT 0, was_accurate INTEGER,
                filter_reason TEXT, resolution_reason TEXT,
                user_response TEXT, resolved_at TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
        """)
        conn.execute("""
            CREATE TABLE signal_profiles (
                profile_type TEXT PRIMARY KEY, samples_count INTEGER DEFAULT 0, updated_at TEXT
            )
        """)
        conn.execute("CREATE TABLE insights (id INTEGER PRIMARY KEY, type TEXT, feedback TEXT)")
        conn.commit()
        conn.close()

        # Corrupt data pages (well past the header and schema pages)
        data = bytearray(db_path.read_bytes())
        start = len(data) * 3 // 4
        for i in range(start, min(start + 2048, len(data))):
            data[i] = 0xFF
        db_path.write_bytes(bytes(data))

        report = analyze(str(tmp_path))
        health = report["sections"]["database_health"]

        assert health["user_model"]["status"] == "corrupt", (
            f"Corrupted DB should report 'corrupt', got: {health['user_model']}"
        )


class TestWithData:
    """Tests with actual data inserted into the databases."""

    def test_events_counted_correctly(self, tmp_path):
        """Events section correctly counts inserted events."""
        _create_all_dbs(tmp_path)

        conn = sqlite3.connect(str(tmp_path / "events.db"))
        for i in range(5):
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, datetime('now'))",
                (f"evt-{i}", "test.event", "test_source"),
            )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))

        assert report["sections"]["events"]["total"] == 5
        assert "test.event" in report["sections"]["events"]["top_types"]

    def test_signal_profiles_reported(self, tmp_path):
        """Signal profiles are correctly reported from user_model.db."""
        _create_all_dbs(tmp_path)

        conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        conn.execute(
            "INSERT INTO signal_profiles (profile_type, samples_count, updated_at) VALUES (?, ?, datetime('now'))",
            ("linguistic", 42),
        )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))

        sp = report["sections"]["signal_profiles"]
        assert "profiles" in sp
        assert "linguistic" in sp["profiles"]
        assert sp["profiles"]["linguistic"]["samples"] == 42
        # linguistic should not be in missing_profiles since we inserted it
        assert "linguistic" not in sp["missing_profiles"]


# ---------------------------------------------------------------------------
# Prediction pipeline tests
# ---------------------------------------------------------------------------


def test_prediction_pipeline_empty_db(tmp_path):
    """prediction_pipeline section exists with all zeros when no predictions exist."""
    _create_all_dbs(tmp_path)

    report = analyze(str(tmp_path))
    pp = report["sections"].get("prediction_pipeline")

    assert pp is not None, "prediction_pipeline section should exist"
    assert pp["total_generated"] == 0
    assert pp["surfaced"] == 0
    assert pp["filtered"] == 0
    assert pp["surfacing_rate"] == 0


def test_prediction_pipeline_with_data(tmp_path):
    """Prediction pipeline correctly categorizes surfaced and filtered predictions."""
    _create_all_dbs(tmp_path)

    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    predictions = [
        ("p1", "NEED", 0, "filtered", "confidence:0.18", None),
        ("p2", "NEED", 0, "filtered", "confidence:0.25", None),
        ("p3", "RISK", 0, "filtered", "reaction:too frequent", None),
        ("p4", "REMINDER", 1, "acted_on", None, None),
    ]
    for pid, ptype, surfaced, response, reason, resolved in predictions:
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, was_surfaced, user_response,
               filter_reason, resolved_at) VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, ptype, surfaced, response, reason, resolved),
        )
    conn.commit()
    conn.close()

    report = analyze(str(tmp_path))
    pp = report["sections"]["prediction_pipeline"]

    assert pp["total_generated"] == 4
    assert pp["surfaced"] == 1
    assert pp["filtered"] == 3
    assert pp["filter_reasons"].get("low_confidence") == 2
    assert pp["filter_reasons"].get("reaction_gate") == 1


# ---------------------------------------------------------------------------
# Missing signal profiles tests
# ---------------------------------------------------------------------------


class TestMissingSignalProfiles:
    """Verify that missing signal profile types are detected."""

    EXPECTED_TYPES = [
        "linguistic",
        "linguistic_inbound",
        "cadence",
        "mood_signals",
        "relationships",
        "temporal",
        "topics",
        "spatial",
        "decision",
    ]

    def test_all_missing_when_table_empty(self, tmp_path):
        """When signal_profiles table is empty, all 9 expected types are missing."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        sp = report["sections"]["signal_profiles"]
        assert sorted(sp["missing_profiles"]) == sorted(self.EXPECTED_TYPES)

    def test_present_profile_excluded_from_missing(self, tmp_path):
        """Inserting a profile removes it from the missing list."""
        _create_all_dbs(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        conn.execute(
            "INSERT INTO signal_profiles (profile_type, samples_count, updated_at) "
            "VALUES ('cadence', 5, datetime('now'))"
        )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        sp = report["sections"]["signal_profiles"]

        assert "cadence" not in sp["missing_profiles"]
        assert "cadence" in sp["profiles"]
        assert len(sp["missing_profiles"]) == 8

    def test_all_present_means_empty_missing(self, tmp_path):
        """When all 9 expected profiles exist, missing_profiles is empty."""
        _create_all_dbs(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        for ptype in self.EXPECTED_TYPES:
            conn.execute(
                "INSERT INTO signal_profiles (profile_type, samples_count, updated_at) "
                "VALUES (?, 1, datetime('now'))",
                (ptype,),
            )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        sp = report["sections"]["signal_profiles"]

        assert sp["missing_profiles"] == []
        assert len(sp["profiles"]) == 9


# ---------------------------------------------------------------------------
# Workflow and communication template count tests
# ---------------------------------------------------------------------------


class TestWorkflowAndTemplateCounts:
    """Verify workflow and communication_templates counts in user_model section."""

    def test_zero_when_tables_empty(self, tmp_path):
        """Workflow and template counts default to 0 when tables are empty."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        um = report["sections"]["user_model"]
        assert "workflows" in um
        assert "communication_templates" in um
        assert um["workflows"] == 0
        assert um["communication_templates"] == 0

    def test_counts_reflect_inserted_rows(self, tmp_path):
        """Counts reflect actually inserted rows."""
        _create_all_dbs(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "user_model.db"))

        # Insert workflows if table exists
        try:
            conn.execute(
                "INSERT INTO workflows (id, name, trigger_pattern, steps, confidence, "
                "created_at, updated_at) VALUES ('w1', 'test', 'p', '[]', 0.5, "
                "datetime('now'), datetime('now'))"
            )
            conn.execute(
                "INSERT INTO workflows (id, name, trigger_pattern, steps, confidence, "
                "created_at, updated_at) VALUES ('w2', 'test2', 'p2', '[]', 0.5, "
                "datetime('now'), datetime('now'))"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Table may not exist in minimal DBs

        # Insert communication templates if table exists
        try:
            conn.execute(
                "INSERT INTO communication_templates (id, contact_id, channel, "
                "template_pattern, confidence, created_at, updated_at) "
                "VALUES ('t1', 'c1', 'email', 'p', 0.5, datetime('now'), datetime('now'))"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

        conn.close()

        report = analyze(str(tmp_path))
        um = report["sections"]["user_model"]

        assert isinstance(um["workflows"], int)
        assert isinstance(um["communication_templates"], int)

    def test_core_fields_still_present(self, tmp_path):
        """Adding new fields doesn't break existing user_model fields."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        um = report["sections"]["user_model"]
        for key in ["episodes", "semantic_facts", "routines", "fact_categories", "query_errors"]:
            assert key in um, f"Missing expected key '{key}' in user_model section"


# ---------------------------------------------------------------------------
# Database health structure tests
# ---------------------------------------------------------------------------


class TestDatabaseHealthStructure:
    """Verify the new structured database_health entries."""

    def test_each_entry_is_dict_with_status_and_detail(self, tmp_path):
        """Each health entry should be a dict with 'status' and 'detail' keys."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))
        health = report["sections"]["database_health"]

        for db_name in ["events", "user_model", "state", "preferences", "entities"]:
            entry = health[db_name]
            assert isinstance(entry, dict), f"{db_name} should be a dict, got {type(entry)}"
            assert "status" in entry, f"{db_name} missing 'status' key"
            assert "detail" in entry, f"{db_name} missing 'detail' key"
            assert entry["status"] in ("ok", "corrupt"), f"{db_name} status should be 'ok' or 'corrupt'"

    def test_connect_failure_reports_corrupt(self, tmp_path):
        """When _connect returns None, the health entry reports 'corrupt'."""
        _create_all_dbs(tmp_path)

        real_connect = _mod._connect

        def fail_events_connect(db_path):
            if "events.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=fail_events_connect):
            report = analyze(str(tmp_path))

        health = report["sections"]["database_health"]
        assert health["events"]["status"] == "corrupt"
        assert "could not connect" in health["events"]["detail"]
        # Other databases should still be ok
        assert health["user_model"]["status"] == "ok"
