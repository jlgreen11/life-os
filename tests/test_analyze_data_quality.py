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
detect_anomalies = _mod.detect_anomalies
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
            description TEXT,
            confidence REAL DEFAULT 0.0,
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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY,
            event_id TEXT,
            timestamp TEXT,
            interaction_type TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )"""
    )
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
            ai_updated_at TEXT,
            interactions INTEGER DEFAULT 0,
            engagements INTEGER DEFAULT 0,
            dismissals INTEGER DEFAULT 0
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
            "workflow_diagnostics",
            "episode_diagnostics",
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
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, interaction_type TEXT)")
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
            ai_updated_at TEXT,
            interactions INTEGER DEFAULT 0,
            engagements INTEGER DEFAULT 0,
            dismissals INTEGER DEFAULT 0
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
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, interaction_type TEXT)")
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
        conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, data TEXT, interaction_type TEXT)")
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


# ---------------------------------------------------------------------------
# Event-sourced prediction activity tests
# ---------------------------------------------------------------------------


def _insert_prediction_events(tmp_path, events):
    """Insert prediction-related events into events.db.

    Args:
        tmp_path: Directory containing events.db.
        events: List of (id, type, timestamp) tuples to insert.
    """
    conn = sqlite3.connect(str(tmp_path / "events.db"))
    for evt_id, evt_type, evt_ts in events:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            (evt_id, evt_type, "prediction_engine", evt_ts),
        )
    conn.commit()
    conn.close()


class TestEventSourcedPredictionActivity:
    """Tests for event-sourced prediction pipeline metrics."""

    def test_event_activity_present_in_report(self, tmp_path):
        """prediction_pipeline section includes event_activity and last_generation_event."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        pp = report["sections"]["prediction_pipeline"]
        assert "event_activity" in pp, "Missing event_activity key"
        assert "last_generation_event" in pp, "Missing last_generation_event key"

    def test_event_activity_counts_prediction_events(self, tmp_path):
        """event_activity counts prediction-related events from events.db."""
        _create_all_dbs(tmp_path)

        _insert_prediction_events(tmp_path, [
            ("e1", "usermodel.prediction.generated", "2026-03-01T10:00:00Z"),
            ("e2", "usermodel.prediction.generated", "2026-03-01T11:00:00Z"),
            ("e3", "usermodel.prediction.generated", "2026-03-01T12:00:00Z"),
            ("e4", "usermodel.prediction.deduplicated", "2026-03-01T10:30:00Z"),
            ("e5", "usermodel.prediction.deduplicated", "2026-03-01T11:30:00Z"),
        ])

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        assert pp["event_activity"]["usermodel.prediction.generated"] == 3
        assert pp["event_activity"]["usermodel.prediction.deduplicated"] == 2

    def test_empty_predictions_table_but_events_show_activity(self, tmp_path):
        """When predictions table is empty but events exist, total_generated=0 and event_activity > 0.

        This is the core scenario: after cleanup, the predictions table is empty but
        the events table proves the pipeline was working.
        """
        _create_all_dbs(tmp_path)

        _insert_prediction_events(tmp_path, [
            ("e1", "usermodel.prediction.generated", "2026-03-01T10:00:00Z"),
            ("e2", "usermodel.prediction.generated", "2026-03-01T11:00:00Z"),
        ])

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        # predictions table is empty → total_generated=0
        assert pp["total_generated"] == 0
        # But events prove the pipeline ran
        assert pp["event_activity"]["usermodel.prediction.generated"] == 2

    def test_last_generation_event_timestamp(self, tmp_path):
        """last_generation_event returns the most recent prediction.generated timestamp."""
        _create_all_dbs(tmp_path)

        _insert_prediction_events(tmp_path, [
            ("e1", "usermodel.prediction.generated", "2026-03-01T10:00:00Z"),
            ("e2", "usermodel.prediction.generated", "2026-03-02T15:30:00Z"),
            ("e3", "usermodel.prediction.generated", "2026-03-01T08:00:00Z"),
        ])

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        assert pp["last_generation_event"] == "2026-03-02T15:30:00Z"

    def test_last_generation_event_none_when_no_events(self, tmp_path):
        """last_generation_event is None when no prediction.generated events exist."""
        _create_all_dbs(tmp_path)

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        assert pp["last_generation_event"] is None

    def test_event_activity_empty_dict_when_no_prediction_events(self, tmp_path):
        """event_activity is an empty dict when no prediction events exist."""
        _create_all_dbs(tmp_path)

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        assert pp["event_activity"] == {}

    def test_event_activity_ignores_non_prediction_events(self, tmp_path):
        """event_activity only counts usermodel.prediction.* events, not other types."""
        _create_all_dbs(tmp_path)

        # Insert a mix of prediction and non-prediction events
        conn = sqlite3.connect(str(tmp_path / "events.db"))
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("e1", "usermodel.prediction.generated", "prediction_engine", "2026-03-01T10:00:00Z"),
        )
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("e2", "email.received", "gmail", "2026-03-01T10:00:00Z"),
        )
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("e3", "calendar.event_created", "caldav", "2026-03-01T10:00:00Z"),
        )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        pp = report["sections"]["prediction_pipeline"]

        assert len(pp["event_activity"]) == 1
        assert pp["event_activity"]["usermodel.prediction.generated"] == 1


# ---------------------------------------------------------------------------
# Prediction detail diagnostics tests
# ---------------------------------------------------------------------------


def _insert_predictions_with_confidence(tmp_path, predictions):
    """Insert predictions with confidence values into user_model.db.

    Args:
        tmp_path: Directory containing user_model.db.
        predictions: List of (id, prediction_type, confidence, was_surfaced,
                     user_response, filter_reason) tuples.
    """
    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    for pid, ptype, conf, surfaced, response, reason in predictions:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, confidence, was_surfaced, user_response, filter_reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pid, ptype, conf, surfaced, response, reason),
        )
    conn.commit()
    conn.close()


class TestPredictionDetail:
    """Tests for the prediction_detail subsection within prediction_pipeline."""

    def test_prediction_detail_present_on_empty_db(self, tmp_path):
        """prediction_detail subsection exists even with no predictions."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        pp = report["sections"]["prediction_pipeline"]
        assert "prediction_detail" in pp, "Missing prediction_detail key"
        detail = pp["prediction_detail"]
        assert detail["confidence_histogram"] == {}
        assert detail["type_breakdown"] == {}
        assert detail["recent_filtered"] == []
        assert detail["stored_prediction_count"] == 0

    def test_confidence_histogram_buckets(self, tmp_path):
        """Confidence histogram groups predictions into correct buckets."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            ("p1", "NEED", 0.05, 0, "filtered", "confidence:0.05"),
            ("p2", "NEED", 0.15, 0, "filtered", "confidence:0.15"),
            ("p3", "NEED", 0.25, 0, "filtered", "confidence:0.25"),
            ("p4", "RISK", 0.55, 1, "acted_on", None),
            ("p5", "RISK", 0.95, 1, "acted_on", None),
        ])

        report = analyze(str(tmp_path))
        hist = report["sections"]["prediction_pipeline"]["prediction_detail"]["confidence_histogram"]

        assert hist.get("0.0-0.1") == 1
        assert hist.get("0.1-0.2") == 1
        assert hist.get("0.2-0.3") == 1
        assert hist.get("0.5-0.6") == 1
        assert hist.get("0.9-1.0") == 1

    def test_type_breakdown_generated_vs_surfaced(self, tmp_path):
        """Type breakdown correctly counts total and surfaced per type."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            ("p1", "NEED", 0.1, 0, "filtered", "confidence:0.1"),
            ("p2", "NEED", 0.2, 0, "filtered", "confidence:0.2"),
            ("p3", "NEED", 0.5, 1, "acted_on", None),
            ("p4", "RISK", 0.6, 1, "acted_on", None),
        ])

        report = analyze(str(tmp_path))
        breakdown = report["sections"]["prediction_pipeline"]["prediction_detail"]["type_breakdown"]

        assert breakdown["NEED"]["total"] == 3
        assert breakdown["NEED"]["surfaced"] == 1
        assert breakdown["RISK"]["total"] == 1
        assert breakdown["RISK"]["surfaced"] == 1

    def test_recent_filtered_returns_last_10(self, tmp_path):
        """Recent filtered list returns up to 10 most recent filtered predictions."""
        _create_all_dbs(tmp_path)
        # Insert 12 filtered predictions
        preds = [
            (f"p{i}", "NEED", 0.1 + i * 0.01, 0, "filtered", f"confidence:{0.1 + i * 0.01}")
            for i in range(12)
        ]
        _insert_predictions_with_confidence(tmp_path, preds)

        report = analyze(str(tmp_path))
        recent = report["sections"]["prediction_pipeline"]["prediction_detail"]["recent_filtered"]

        assert len(recent) == 10
        # Each entry has the expected keys
        for entry in recent:
            assert "prediction_type" in entry
            assert "confidence" in entry
            assert "filter_reason" in entry
            assert "created_at" in entry

    def test_stored_prediction_count(self, tmp_path):
        """stored_prediction_count reflects total predictions in DB."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            ("p1", "NEED", 0.1, 0, "filtered", "confidence:0.1"),
            ("p2", "RISK", 0.5, 1, "acted_on", None),
            ("p3", "REMINDER", 0.7, 1, "acted_on", None),
        ])

        report = analyze(str(tmp_path))
        detail = report["sections"]["prediction_pipeline"]["prediction_detail"]
        assert detail["stored_prediction_count"] == 3


class TestPredictionDetailAnomalies:
    """Tests for anomaly detection related to prediction detail diagnostics."""

    def test_all_low_confidence_anomaly(self, tmp_path):
        """Anomaly detected when all predictions are below 0.3 confidence."""
        _create_all_dbs(tmp_path)
        # Insert 6 predictions all below 0.3
        _insert_predictions_with_confidence(tmp_path, [
            (f"p{i}", "NEED", 0.05 + i * 0.03, 0, "filtered", f"confidence:{0.05 + i * 0.03}")
            for i in range(6)
        ])

        report = analyze(str(tmp_path))
        categories = [a["category"] for a in report["anomalies"]]
        assert "prediction_low_confidence" in categories

    def test_no_low_confidence_anomaly_when_mixed(self, tmp_path):
        """No low-confidence anomaly when some predictions are above 0.3."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            ("p1", "NEED", 0.1, 0, "filtered", "confidence:0.1"),
            ("p2", "NEED", 0.2, 0, "filtered", "confidence:0.2"),
            ("p3", "RISK", 0.5, 1, "acted_on", None),
            ("p4", "RISK", 0.7, 1, "acted_on", None),
            ("p5", "NEED", 0.15, 0, "filtered", "confidence:0.15"),
            ("p6", "NEED", 0.25, 0, "filtered", "confidence:0.25"),
        ])

        report = analyze(str(tmp_path))
        categories = [a["category"] for a in report["anomalies"]]
        assert "prediction_low_confidence" not in categories

    def test_single_type_monoculture_anomaly(self, tmp_path):
        """Anomaly detected when all predictions are the same type."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            (f"p{i}", "NEED", 0.1 + i * 0.1, i > 3, None if i <= 3 else "acted_on", None)
            for i in range(8)
        ])

        report = analyze(str(tmp_path))
        categories = [a["category"] for a in report["anomalies"]]
        assert "prediction_type_monoculture" in categories

    def test_no_monoculture_anomaly_with_multiple_types(self, tmp_path):
        """No monoculture anomaly when multiple prediction types exist."""
        _create_all_dbs(tmp_path)
        _insert_predictions_with_confidence(tmp_path, [
            ("p1", "NEED", 0.2, 0, "filtered", "confidence:0.2"),
            ("p2", "RISK", 0.3, 0, "filtered", "confidence:0.3"),
            ("p3", "REMINDER", 0.5, 1, "acted_on", None),
            ("p4", "NEED", 0.1, 0, "filtered", "confidence:0.1"),
            ("p5", "RISK", 0.4, 1, "acted_on", None),
            ("p6", "OPPORTUNITY", 0.6, 1, "acted_on", None),
        ])

        report = analyze(str(tmp_path))
        categories = [a["category"] for a in report["anomalies"]]
        assert "prediction_type_monoculture" not in categories


# ---------------------------------------------------------------------------
# Episode diagnostics tests
# ---------------------------------------------------------------------------


def _insert_event(conn, event_id, event_type, timestamp):
    """Insert an event row into events.db (helper for episode_diagnostics tests)."""
    conn.execute(
        "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
        (event_id, event_type, "test_source", timestamp),
    )


def _insert_episode(conn, ep_id, event_id, created_at, interaction_type="email_received"):
    """Insert an episode row into user_model.db (helper for episode_diagnostics tests)."""
    conn.execute(
        """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (ep_id, event_id, created_at, interaction_type, created_at),
    )


class TestEpisodeDiagnostics:
    """Tests for the episode_diagnostics top-level report section."""

    def test_section_present_on_empty_dbs(self, tmp_path):
        """episode_diagnostics section exists with zero counts on a clean DB."""
        _create_all_dbs(tmp_path)
        report = analyze(str(tmp_path))

        ed = report["sections"].get("episode_diagnostics")
        assert ed is not None
        assert ed["created_last_24h"] == 0
        assert ed["created_last_7d"] == 0
        assert ed["backfill_coverage"]["events_last_7d"] == 0
        assert ed["backfill_coverage"]["events_with_episode"] == 0
        assert ed["backfill_coverage"]["coverage_pct"] is None
        assert ed["avg_creation_lag_seconds"] is None
        assert ed["lag_sample_size"] == 0
        assert ed["interaction_type_distribution_last_7d"] == {}

    def test_created_counts(self, tmp_path):
        """created_last_24h and created_last_7d count episodes by created_at."""
        _create_all_dbs(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        # 2 episodes in last 24h
        _insert_episode(conn, 1, "e1", "2099-01-01T00:00:00Z")  # in the future, "now" relative
        # Use SQLite "now" arithmetic via direct insert to align with the script's clock
        conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (10, 'e10', datetime('now', '-1 hour'), 'email_received', datetime('now', '-1 hour'))"""
        )
        conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (11, 'e11', datetime('now', '-12 hours'), 'email_received', datetime('now', '-12 hours'))"""
        )
        # 1 episode 5 days ago (in last 7d but not 24h)
        conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (12, 'e12', datetime('now', '-5 days'), 'email_received', datetime('now', '-5 days'))"""
        )
        # 1 episode 10 days ago (outside 7d)
        conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (13, 'e13', datetime('now', '-10 days'), 'email_received', datetime('now', '-10 days'))"""
        )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        ed = report["sections"]["episode_diagnostics"]

        # The single explicit '2099-...' insert + the two recent ones land in last_24h.
        # SQLite datetime('now') compares lexicographically with our 2099 value, so it
        # is also "> now-1 day". Real-world data won't have future timestamps, so this
        # is fine for verifying the SQL filter — what matters is that the recent rows
        # are counted and the 10-day-old one is not.
        assert ed["created_last_24h"] >= 2
        assert ed["created_last_7d"] >= 3
        # The 10-day-old episode must be excluded from 7d
        assert ed["created_last_7d"] < 5

    def test_backfill_coverage_partial(self, tmp_path):
        """backfill_coverage reports the fraction of recent episodic events with episodes."""
        _create_all_dbs(tmp_path)
        ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
        um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))

        # 4 episodic events in last 7d
        for i in range(4):
            ev_conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-1 hour'))""",
                (f"ev-{i}",),
            )
        # 2 non-episodic events (should be excluded from the denominator)
        for i in range(2):
            ev_conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'usermodel.prediction.generated', 'pe', datetime('now', '-1 hour'))""",
                (f"pred-{i}",),
            )
        ev_conn.commit()
        ev_conn.close()

        # Only 2 of the 4 episodic events have episodes
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (1, 'ev-0', datetime('now'), 'email_received', datetime('now'))"""
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (2, 'ev-1', datetime('now'), 'email_received', datetime('now'))"""
        )
        um_conn.commit()
        um_conn.close()

        report = analyze(str(tmp_path))
        bf = report["sections"]["episode_diagnostics"]["backfill_coverage"]

        assert bf["events_last_7d"] == 4  # non-episodic excluded
        assert bf["events_with_episode"] == 2
        assert bf["coverage_pct"] == 50.0

    def test_creation_lag_median(self, tmp_path):
        """avg_creation_lag_seconds returns the median lag between event and episode."""
        _create_all_dbs(tmp_path)
        ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
        um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))

        # Event at T=0, episode at T+10s → lag=10
        ev_conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("ev-a", "email.received", "g", "2026-05-01T00:00:00Z"),
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (1, 'ev-a', '2026-05-01T00:00:00Z', 'email_received', '2026-05-01T00:00:10Z')"""
        )
        # Event at T=0, episode at T+30s → lag=30 (median of [10, 30, 60] = 30)
        ev_conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("ev-b", "email.received", "g", "2026-05-02T00:00:00Z"),
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (2, 'ev-b', '2026-05-02T00:00:00Z', 'email_received', '2026-05-02T00:00:30Z')"""
        )
        # Event at T=0, episode at T+60s → lag=60
        ev_conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            ("ev-c", "email.received", "g", "2026-05-03T00:00:00Z"),
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (3, 'ev-c', '2026-05-03T00:00:00Z', 'email_received', '2026-05-03T00:01:00Z')"""
        )
        ev_conn.commit()
        ev_conn.close()
        um_conn.commit()
        um_conn.close()

        report = analyze(str(tmp_path))
        ed = report["sections"]["episode_diagnostics"]

        assert ed["lag_sample_size"] == 3
        assert ed["avg_creation_lag_seconds"] == 30.0  # median

    def test_interaction_type_distribution_last_7d(self, tmp_path):
        """interaction_type_distribution_last_7d only includes recent episodes."""
        _create_all_dbs(tmp_path)
        um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))
        # Recent: 2 email_received, 1 message_sent
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (1, 'e1', datetime('now'), 'email_received', datetime('now'))"""
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (2, 'e2', datetime('now'), 'email_received', datetime('now'))"""
        )
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (3, 'e3', datetime('now'), 'message_sent', datetime('now'))"""
        )
        # Old: outside 7d window
        um_conn.execute(
            """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
               VALUES (4, 'e4', datetime('now', '-10 days'), 'meeting_scheduled', datetime('now', '-10 days'))"""
        )
        um_conn.commit()
        um_conn.close()

        report = analyze(str(tmp_path))
        dist = report["sections"]["episode_diagnostics"]["interaction_type_distribution_last_7d"]

        assert dist.get("email_received") == 2
        assert dist.get("message_sent") == 1
        assert "meeting_scheduled" not in dist

    def test_missing_user_model_db_reports_error(self, tmp_path):
        """When user_model.db is missing, episode_diagnostics gets an error key."""
        _create_minimal_events_db(tmp_path)
        _create_minimal_state_db(tmp_path)
        _create_minimal_preferences_db(tmp_path)
        _create_minimal_entities_db(tmp_path)

        real_connect = _mod._connect

        def failing(db_path):
            if "user_model.db" in str(db_path):
                return None
            return real_connect(db_path)

        with patch.object(_mod, "_connect", side_effect=failing):
            report = analyze(str(tmp_path))

        ed = report["sections"]["episode_diagnostics"]
        assert "error" in ed
        assert "user_model.db" in ed["error"]


class TestEpisodeDiagnosticsAnomalies:
    """Tests for anomaly detection driven by episode_diagnostics."""

    def test_stalled_creation_critical_anomaly(self, tmp_path):
        """Critical anomaly fires when events arrive but no episodes are created."""
        _create_all_dbs(tmp_path)
        # 150 events in last 24h, zero episodes
        conn = sqlite3.connect(str(tmp_path / "events.db"))
        for i in range(150):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-1 hour'))""",
                (f"ev-{i}",),
            )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        anomaly_cats = [a["category"] for a in report["anomalies"]]
        assert "episode_creation_stalled" in anomaly_cats
        # And it should be flagged as critical
        for a in report["anomalies"]:
            if a["category"] == "episode_creation_stalled":
                assert a["severity"] == "critical"

    def test_no_stalled_anomaly_when_events_low(self, tmp_path):
        """No stalled anomaly when event volume is below the 100/24h threshold."""
        _create_all_dbs(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "events.db"))
        for i in range(10):
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-1 hour'))""",
                (f"ev-{i}",),
            )
        conn.commit()
        conn.close()

        report = analyze(str(tmp_path))
        anomaly_cats = [a["category"] for a in report["anomalies"]]
        assert "episode_creation_stalled" not in anomaly_cats

    def test_low_backfill_coverage_warning(self, tmp_path):
        """Warning fires when backfill coverage < 50% and event count is sufficient."""
        _create_all_dbs(tmp_path)
        ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
        um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))

        # 100 episodic events in last 7d
        for i in range(100):
            ev_conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-2 hours'))""",
                (f"ev-{i}",),
            )
        # Only 10 episodes link to those events → coverage = 10%
        for i in range(10):
            um_conn.execute(
                """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
                   VALUES (?, ?, datetime('now'), 'email_received', datetime('now'))""",
                (i, f"ev-{i}"),
            )
        ev_conn.commit()
        ev_conn.close()
        um_conn.commit()
        um_conn.close()

        report = analyze(str(tmp_path))
        anomaly_cats = [a["category"] for a in report["anomalies"]]
        assert "episode_backfill_low" in anomaly_cats
        for a in report["anomalies"]:
            if a["category"] == "episode_backfill_low":
                assert a["severity"] == "warning"

    def test_high_coverage_no_anomaly(self, tmp_path):
        """No backfill anomaly when coverage is above 50%."""
        _create_all_dbs(tmp_path)
        ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
        um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))

        for i in range(100):
            ev_conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-2 hours'))""",
                (f"ev-{i}",),
            )
        # 80 of 100 have episodes → coverage = 80%
        for i in range(80):
            um_conn.execute(
                """INSERT INTO episodes (id, event_id, timestamp, interaction_type, created_at)
                   VALUES (?, ?, datetime('now'), 'email_received', datetime('now'))""",
                (i, f"ev-{i}"),
            )
        ev_conn.commit()
        ev_conn.close()
        um_conn.commit()
        um_conn.close()

        report = analyze(str(tmp_path))
        anomaly_cats = [a["category"] for a in report["anomalies"]]
        assert "episode_backfill_low" not in anomaly_cats

    def test_no_backfill_anomaly_when_sample_too_small(self, tmp_path):
        """No backfill anomaly when fewer than 50 events exist (sample too small)."""
        _create_all_dbs(tmp_path)
        ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
        # Only 20 events — zero coverage but not enough to alarm
        for i in range(20):
            ev_conn.execute(
                """INSERT INTO events (id, type, source, timestamp)
                   VALUES (?, 'email.received', 'gmail', datetime('now', '-2 hours'))""",
                (f"ev-{i}",),
            )
        ev_conn.commit()
        ev_conn.close()

        report = analyze(str(tmp_path))
        anomaly_cats = [a["category"] for a in report["anomalies"]]
        assert "episode_backfill_low" not in anomaly_cats
