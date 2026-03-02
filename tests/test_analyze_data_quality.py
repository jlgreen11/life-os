"""Tests for the data quality analysis script.

Verifies the 4 bug fixes:
1. _query() and _query_one() log warnings on failure (Bug 4)
2. connector_state query uses correct column name ``last_error`` (Bug 2)
3. source_weights query uses correct column names (Bug 3)
4. user_model.db sections are independent — one failure doesn't suppress others (Bug 1)
"""

import logging
import sqlite3

import pytest

# The script lives at scripts/analyze-data-quality.py which isn't a proper
# Python package, so we import its internals by manipulating sys.path.
import importlib
import sys
from pathlib import Path

# Add the scripts directory so we can import the module
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Import with importlib since the filename contains hyphens
_mod = importlib.import_module("analyze-data-quality")
_query = _mod._query
_query_one = _mod._query_one
analyze = _mod.analyze


# ---------------------------------------------------------------------------
# Bug 4: _query / _query_one log warnings on failure
# ---------------------------------------------------------------------------


def test_query_logs_warning_on_failure(tmp_path, caplog):
    """_query() should log a warning when the SQL fails."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    broken_sql = "SELECT * FROM nonexistent_table_xyz"
    with caplog.at_level(logging.WARNING, logger="analyze-data-quality"):
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
    with caplog.at_level(logging.WARNING, logger="analyze-data-quality"):
        result = _query_one(conn, broken_sql, default=None)

    assert result is None
    assert any("Query failed" in rec.message for rec in caplog.records)
    conn.close()


# ---------------------------------------------------------------------------
# Bug 2: connector_state uses correct column name ``last_error``
# ---------------------------------------------------------------------------


def test_connector_state_uses_last_error_column(tmp_path):
    """The connector_state query should select ``last_error``, not ``error_message``."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(str(db_path))
    # Create the real schema
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
    # Also need notifications and tasks tables for the state.db section
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

    # Create minimal other DBs so analyze() doesn't crash
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    connectors = report["sections"].get("connectors", {})

    assert "gmail" in connectors, f"Expected 'gmail' in connectors, got: {connectors}"
    assert connectors["gmail"]["error"] == "timeout after 30s"
    assert connectors["gmail"]["status"] == "active"


# ---------------------------------------------------------------------------
# Bug 3: source_weights uses correct column names
# ---------------------------------------------------------------------------


def test_source_weights_uses_correct_column_names(tmp_path):
    """The source_weights query should select ``user_weight``, ``ai_drift``, ``ai_updated_at``."""
    db_path = tmp_path / "preferences.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE source_weights (
            source_key TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            label TEXT NOT NULL,
            description TEXT DEFAULT '',
            user_weight REAL NOT NULL DEFAULT 0.5,
            ai_drift REAL NOT NULL DEFAULT 0.0,
            drift_reason TEXT DEFAULT '',
            drift_history TEXT DEFAULT '[]',
            user_set_at TEXT,
            ai_updated_at TEXT,
            interactions INTEGER DEFAULT 0,
            engagements INTEGER DEFAULT 0,
            dismissals INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        INSERT INTO source_weights (source_key, category, label, user_weight, ai_drift, ai_updated_at)
        VALUES ('email.gmail', 'connector', 'Gmail', 0.8, 0.15, '2026-01-15T12:00:00Z')
    """)
    # Also need feedback_log table
    conn.execute("""
        CREATE TABLE feedback_log (
            id INTEGER PRIMARY KEY,
            action_type TEXT,
            feedback_type TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Create minimal other DBs
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)

    report = analyze(str(tmp_path))
    sw = report["sections"].get("source_weights", {})

    assert "email.gmail" in sw, f"Expected 'email.gmail' in source_weights, got: {sw}"
    assert sw["email.gmail"]["weight"] == 0.8
    assert sw["email.gmail"]["drift"] == 0.15
    assert sw["email.gmail"]["updated_at"] == "2026-01-15T12:00:00Z"


# ---------------------------------------------------------------------------
# Bug 1: user_model.db sections are independent
# ---------------------------------------------------------------------------


def test_user_model_sections_independent(tmp_path):
    """If predictions table is missing, other user_model.db sections still populate.

    The _query helpers catch missing-table errors internally, so the outer
    try/except won't fire — but the key property being tested is that
    signal_profiles, insight_feedback, and user_model sections are ALL present
    in the output even when the predictions table doesn't exist.  In the old
    monolithic try-block, ANY exception (e.g. from a corrupt DB or a schema
    mismatch that bypasses _query) would have silently suppressed every section
    after the failure point.
    """
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

    conn.execute("""
        CREATE TABLE insights (
            id INTEGER PRIMARY KEY,
            type TEXT,
            feedback TEXT
        )
    """)

    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO episodes (id) VALUES (1)")
    conn.execute("INSERT INTO episodes (id) VALUES (2)")

    conn.execute("""
        CREATE TABLE semantic_facts (
            id INTEGER PRIMARY KEY,
            category TEXT DEFAULT 'general'
        )
    """)
    conn.execute("INSERT INTO semantic_facts (id, category) VALUES (1, 'preference')")

    conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO routines (id) VALUES (1)")

    conn.commit()
    conn.close()

    # Create minimal other DBs
    _create_minimal_events_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    sections = report["sections"]

    # All four user_model.db sections should be present in the output
    assert "prediction_accuracy" in sections, "prediction_accuracy section should exist"
    assert "signal_profiles" in sections, "signal_profiles section should exist"
    assert "insight_feedback" in sections, "insight_feedback section should exist"
    assert "user_model" in sections, "user_model section should exist"

    # signal_profiles should have real data
    sp = sections["signal_profiles"]
    assert "linguistic" in sp, f"signal_profiles should contain 'linguistic', got: {sp}"
    assert sp["linguistic"]["samples"] == 42

    # user_model should have real data
    um = sections["user_model"]
    assert um.get("episodes") == 2, f"user_model.episodes should be 2, got: {um}"
    assert um.get("semantic_facts") == 1
    assert um.get("routines") == 1
    assert um.get("fact_categories") == {"preference": 1}


def test_user_model_sections_independent_reverse(tmp_path):
    """If episodes/routines tables are missing, signal_profiles still populates.

    Verifies independence in the reverse direction: even when the last
    section (user_model depth) has missing tables, earlier sections like
    signal_profiles still have their data.
    """
    db_path = tmp_path / "user_model.db"
    conn = sqlite3.connect(str(db_path))

    # Create predictions and signal_profiles, but NOT episodes/semantic_facts/routines
    conn.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            prediction_type TEXT,
            was_surfaced INTEGER DEFAULT 0,
            was_accurate INTEGER,
            resolution_reason TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE signal_profiles (
            profile_type TEXT PRIMARY KEY,
            samples_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.execute("""
        INSERT INTO signal_profiles (profile_type, samples_count, updated_at)
        VALUES ('mood', 10, '2026-02-01T00:00:00Z')
    """)
    conn.execute("""
        CREATE TABLE insights (
            id INTEGER PRIMARY KEY,
            type TEXT,
            feedback TEXT
        )
    """)
    conn.commit()
    conn.close()

    _create_minimal_events_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    sections = report["sections"]

    # All four user_model.db sections should exist in the output
    assert "signal_profiles" in sections, "signal_profiles section should exist"
    assert "user_model" in sections, "user_model section should exist"

    # signal_profiles should have real data despite missing episodes/routines tables
    sp = sections["signal_profiles"]
    assert "mood" in sp, f"signal_profiles should have 'mood', got: {sp}"
    assert sp["mood"]["samples"] == 10

    # user_model should be populated with zero counts (missing tables return None defaults)
    um = sections["user_model"]
    assert um["episodes"] == 0
    assert um["semantic_facts"] == 0
    assert um["routines"] == 0


# ---------------------------------------------------------------------------
# Prediction pipeline diagnostics
# ---------------------------------------------------------------------------


def test_prediction_pipeline_empty_db(tmp_path):
    """prediction_pipeline section exists with all zeros when no predictions exist."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    report = analyze(str(tmp_path))
    pp = report["sections"].get("prediction_pipeline")

    assert pp is not None, "prediction_pipeline section should exist"
    assert pp["total_generated"] == 0
    assert pp["surfaced"] == 0
    assert pp["filtered"] == 0
    assert pp["surfacing_rate"] == 0
    assert pp["resolved"] == 0
    assert pp["user_acted_on"] == 0
    assert pp["user_dismissed"] == 0
    assert pp["auto_filtered"] == 0
    assert pp["filter_reasons"] == {}


def test_prediction_pipeline_with_filtered_predictions(tmp_path):
    """Filtered predictions are counted and categorized by filter_reason."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    conn.row_factory = sqlite3.Row

    # Insert filtered predictions with various filter_reasons
    predictions = [
        ("p1", "NEED", 0, "filtered", "confidence:0.18 (threshold:0.3)", "2026-01-01T00:00:00Z"),
        ("p2", "NEED", 0, "filtered", "confidence:0.25 (threshold:0.3)", "2026-01-01T00:00:00Z"),
        ("p3", "RISK", 0, "filtered", "reaction:annoying (too frequent)", "2026-01-01T00:00:00Z"),
        ("p4", "OPPORTUNITY", 0, "filtered", "duplicate:similar prediction exists", "2026-01-01T00:00:00Z"),
        ("p5", "REMINDER", 1, "acted_on", None, None),
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

    assert pp["total_generated"] == 5
    assert pp["surfaced"] == 1
    assert pp["filtered"] == 4
    assert pp["auto_filtered"] == 4  # user_response = 'filtered'
    assert pp["user_acted_on"] == 1

    # Check filter_reason categorization
    reasons = pp["filter_reasons"]
    assert reasons.get("low_confidence") == 2, f"Expected 2 low_confidence, got: {reasons}"
    assert reasons.get("reaction_gate") == 1, f"Expected 1 reaction_gate, got: {reasons}"
    assert reasons.get("duplicate") == 1, f"Expected 1 duplicate, got: {reasons}"


def test_prediction_pipeline_surfacing_rate(tmp_path):
    """Surfacing rate is calculated correctly from a mix of surfaced and filtered predictions."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    conn.row_factory = sqlite3.Row

    # Insert 3 surfaced, 7 filtered → 30% surfacing rate
    for i in range(3):
        conn.execute(
            "INSERT INTO predictions (id, prediction_type, was_surfaced) VALUES (?, 'NEED', 1)",
            (f"surfaced_{i}",),
        )
    for i in range(7):
        conn.execute(
            """INSERT INTO predictions (id, prediction_type, was_surfaced, user_response,
               filter_reason, resolved_at) VALUES (?, 'NEED', 0, 'filtered',
               'confidence:0.1 (threshold:0.3)', '2026-01-01T00:00:00Z')""",
            (f"filtered_{i}",),
        )
    conn.commit()
    conn.close()

    report = analyze(str(tmp_path))
    pp = report["sections"]["prediction_pipeline"]

    assert pp["total_generated"] == 10
    assert pp["surfaced"] == 3
    assert pp["filtered"] == 7
    assert pp["surfacing_rate"] == 0.3  # 3/10 = 0.3
    assert pp["auto_filtered"] == 7
    assert pp["filter_reasons"].get("low_confidence") == 7


# ---------------------------------------------------------------------------
# Database health section
# ---------------------------------------------------------------------------


def test_database_health_section_present(tmp_path):
    """Report includes a 'database_health' section with entries for all 5 databases."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)
    # entities.db doesn't exist — should report "could not connect"

    report = analyze(str(tmp_path))
    health = report["sections"].get("database_health")

    assert health is not None, "database_health section should exist"
    expected_dbs = ["events", "user_model", "state", "preferences", "entities"]
    for db_name in expected_dbs:
        assert db_name in health, f"database_health should include '{db_name}', got: {list(health.keys())}"


def test_database_health_reports_ok(tmp_path):
    """Healthy databases all report 'ok' in the database_health section."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)
    _create_minimal_entities_db(tmp_path)

    report = analyze(str(tmp_path))
    health = report["sections"]["database_health"]

    for db_name in ["events", "user_model", "state", "preferences", "entities"]:
        assert health[db_name] == "ok", f"{db_name} should be 'ok', got: {health[db_name]}"


def test_corrupted_db_reports_query_errors(tmp_path, monkeypatch):
    """When _query_one returns None (e.g. due to corruption), query_errors tracks failures."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_user_model_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    # Patch _query_one to simulate corruption: return None for episodes/routines
    original_query_one = _query_one

    def _failing_query_one(conn, sql, default=None):
        """Simulate corruption for episodes and routines tables."""
        if "FROM episodes" in sql or "FROM routines" in sql:
            return None
        return original_query_one(conn, sql, default)

    monkeypatch.setattr(_mod, "_query_one", _failing_query_one)

    report = analyze(str(tmp_path))
    um = report["sections"]["user_model"]

    assert "episodes" in um["query_errors"], f"Expected 'episodes' in query_errors, got: {um['query_errors']}"
    assert "routines" in um["query_errors"], f"Expected 'routines' in query_errors, got: {um['query_errors']}"
    assert "semantic_facts" not in um["query_errors"], "semantic_facts should not be in query_errors"
    # Values should still be 0 (the fallback), but query_errors tells us why
    assert um["episodes"] == 0
    assert um["routines"] == 0


def test_corrupted_db_detected_by_integrity_check(tmp_path):
    """PRAGMA integrity_check detects a corrupted database file."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    # Create a valid user_model.db with enough data to span multiple pages
    db_path = tmp_path / "user_model.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA page_size = 512")  # small pages = more pages for same data
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, data TEXT)")
    # Insert enough rows to create multiple pages
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
    # Corrupt a large section in the latter half of the file
    start = len(data) * 3 // 4
    for i in range(start, min(start + 2048, len(data))):
        data[i] = 0xFF
    db_path.write_bytes(bytes(data))

    report = analyze(str(tmp_path))
    health = report["sections"]["database_health"]

    # The corrupted DB should NOT report 'ok'
    assert health["user_model"] != "ok", f"Corrupted DB should not report 'ok', got: {health['user_model']}"


def test_healthy_db_has_empty_query_errors(tmp_path):
    """A healthy user_model.db with data should have an empty query_errors list."""
    _create_minimal_events_db(tmp_path)
    _create_minimal_state_db(tmp_path)
    _create_minimal_preferences_db(tmp_path)

    db_path = tmp_path / "user_model.db"
    conn = sqlite3.connect(str(db_path))
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
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO episodes (id) VALUES (1)")
    conn.execute("INSERT INTO episodes (id) VALUES (2)")
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, category TEXT DEFAULT 'general')")
    conn.execute("INSERT INTO semantic_facts (id, category) VALUES (1, 'preference')")
    conn.execute("INSERT INTO semantic_facts (id, category) VALUES (2, 'fact')")
    conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO routines (id) VALUES (1)")
    conn.commit()
    conn.close()

    report = analyze(str(tmp_path))
    um = report["sections"]["user_model"]

    assert um["query_errors"] == [], f"Healthy DB should have empty query_errors, got: {um['query_errors']}"
    assert um["episodes"] == 2
    assert um["semantic_facts"] == 2
    assert um["routines"] == 1


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    conn.commit()
    conn.close()
