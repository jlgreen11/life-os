"""Tests for source weight interaction/engagement/dismissal counts in data quality report.

Verifies that:
- The source_weights section includes interaction, engagement, and dismissal counts
- Anomaly detection flags zero interactions when events exist
- Anomaly detection flags zero dismissals when feedback dismissals exist
- No anomaly is generated when learning activity is healthy
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

# Import the hyphenated script via importlib
_mod_spec = importlib.util.spec_from_file_location(
    "analyze_data_quality",
    Path(__file__).resolve().parent.parent / "scripts" / "analyze-data-quality.py",
)
_mod = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_mod)

analyze = _mod.analyze
detect_anomalies = _mod.detect_anomalies


# ---------------------------------------------------------------------------
# Helper: create all minimal databases needed for a full analyze() run
# ---------------------------------------------------------------------------

def _create_events_db(tmp_path, event_count=0):
    """Create events.db, optionally seeding event rows."""
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
    for i in range(event_count):
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
            (f"evt-{i}", "email.received", "gmail", "2026-03-01T10:00:00Z"),
        )
    conn.commit()
    conn.close()


def _create_user_model_db(tmp_path):
    """Create a minimal user_model.db with all expected tables."""
    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
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
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, category TEXT DEFAULT 'general')")
    conn.execute("CREATE TABLE routines (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


def _create_state_db(tmp_path):
    """Create a minimal state.db."""
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("""
        CREATE TABLE notifications (
            id TEXT PRIMARY KEY, status TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE connector_state (
            connector_id TEXT PRIMARY KEY, status TEXT, last_sync TEXT, last_error TEXT
        )
    """)
    conn.commit()
    conn.close()


def _create_preferences_db(tmp_path, source_weights=None, feedback_rows=None):
    """Create preferences.db with source_weights and feedback_log tables.

    Args:
        tmp_path: Directory for the database file.
        source_weights: List of dicts with source_key, user_weight, ai_drift,
            ai_updated_at, interactions, engagements, dismissals.
        feedback_rows: List of (action_type, feedback_type) tuples to insert.
    """
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
            description TEXT DEFAULT '',
            user_weight REAL NOT NULL DEFAULT 0.5,
            ai_drift REAL NOT NULL DEFAULT 0.0,
            drift_reason TEXT,
            drift_history TEXT,
            user_set_at TEXT,
            ai_updated_at TEXT,
            interactions INTEGER DEFAULT 0,
            engagements INTEGER DEFAULT 0,
            dismissals INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    for sw in (source_weights or []):
        conn.execute(
            """INSERT INTO source_weights
               (source_key, user_weight, ai_drift, ai_updated_at,
                interactions, engagements, dismissals)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                sw["source_key"],
                sw.get("user_weight", 0.5),
                sw.get("ai_drift", 0.0),
                sw.get("ai_updated_at"),
                sw.get("interactions", 0),
                sw.get("engagements", 0),
                sw.get("dismissals", 0),
            ),
        )
    for action_type, feedback_type in (feedback_rows or []):
        conn.execute(
            "INSERT INTO feedback_log (action_type, feedback_type) VALUES (?, ?)",
            (action_type, feedback_type),
        )
    conn.commit()
    conn.close()


def _create_entities_db(tmp_path):
    """Create a minimal entities.db."""
    conn = sqlite3.connect(str(tmp_path / "entities.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS contacts (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()


def _create_all_dbs(tmp_path, event_count=0, source_weights=None, feedback_rows=None):
    """Create all databases needed for a full analyze() run."""
    _create_events_db(tmp_path, event_count=event_count)
    _create_user_model_db(tmp_path)
    _create_state_db(tmp_path)
    _create_preferences_db(tmp_path, source_weights=source_weights, feedback_rows=feedback_rows)
    _create_entities_db(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSourceWeightsIncludeInteractionCounts:
    """Verify the report includes interactions, engagements, and dismissals."""

    def test_source_weights_include_interaction_counts(self, tmp_path):
        """Source weights with non-zero interaction counts appear in the report."""
        weights = [
            {
                "source_key": "email.personal",
                "user_weight": 0.8,
                "ai_drift": 0.0,
                "ai_updated_at": None,
                "interactions": 150,
                "engagements": 12,
                "dismissals": 3,
            },
            {
                "source_key": "email.marketing",
                "user_weight": 0.15,
                "ai_drift": -0.05,
                "ai_updated_at": "2026-03-01T00:00:00Z",
                "interactions": 80,
                "engagements": 2,
                "dismissals": 20,
            },
        ]
        _create_all_dbs(tmp_path, source_weights=weights)

        report = analyze(str(tmp_path))
        sw = report["sections"]["source_weights"]

        assert "email.personal" in sw
        assert sw["email.personal"]["interactions"] == 150
        assert sw["email.personal"]["engagements"] == 12
        assert sw["email.personal"]["dismissals"] == 3

        assert "email.marketing" in sw
        assert sw["email.marketing"]["interactions"] == 80
        assert sw["email.marketing"]["engagements"] == 2
        assert sw["email.marketing"]["dismissals"] == 20

    def test_zero_counts_reported(self, tmp_path):
        """Source weights with zero counts still show the fields."""
        weights = [
            {
                "source_key": "calendar.meetings",
                "user_weight": 0.7,
                "ai_drift": 0.0,
                "ai_updated_at": None,
                "interactions": 0,
                "engagements": 0,
                "dismissals": 0,
            },
        ]
        _create_all_dbs(tmp_path, source_weights=weights)

        report = analyze(str(tmp_path))
        sw = report["sections"]["source_weights"]

        assert sw["calendar.meetings"]["interactions"] == 0
        assert sw["calendar.meetings"]["engagements"] == 0
        assert sw["calendar.meetings"]["dismissals"] == 0


class TestAnomalyZeroInteractions:
    """Anomaly: zero interactions despite many events."""

    def test_anomaly_zero_interactions_warning(self, tmp_path):
        """When events > 100 but all source weights have 0 interactions, flag it."""
        weights = [
            {"source_key": "email.personal", "interactions": 0, "engagements": 0, "dismissals": 0},
            {"source_key": "email.work", "interactions": 0, "engagements": 0, "dismissals": 0},
        ]
        _create_all_dbs(tmp_path, event_count=200, source_weights=weights)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        sw_learning = [a for a in anomalies if a["category"] == "source_weight_learning"]
        assert len(sw_learning) == 1
        assert sw_learning[0]["severity"] == "warning"
        assert "0 interactions" in sw_learning[0]["message"]
        assert "200 events" in sw_learning[0]["message"]

    def test_no_anomaly_when_few_events(self, tmp_path):
        """No anomaly when event count is <= 100, even with 0 interactions."""
        weights = [
            {"source_key": "email.personal", "interactions": 0, "engagements": 0, "dismissals": 0},
        ]
        _create_all_dbs(tmp_path, event_count=50, source_weights=weights)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        sw_learning = [a for a in anomalies if a["category"] == "source_weight_learning"]
        assert len(sw_learning) == 0


class TestAnomalyZeroDismissalsWithFeedback:
    """Anomaly: interactions > 0 but 0 dismissals despite feedback dismissals."""

    def test_anomaly_zero_dismissals_with_feedback(self, tmp_path):
        """Flag when interactions recorded but no dismissals despite feedback log entries."""
        weights = [
            {"source_key": "email.personal", "interactions": 100, "engagements": 5, "dismissals": 0},
        ]
        # Create 10 feedback dismissals
        feedback = [("notification", "dismissed")] * 10
        _create_all_dbs(tmp_path, event_count=200, source_weights=weights, feedback_rows=feedback)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        sw_feedback = [a for a in anomalies if a["category"] == "source_weight_feedback"]
        assert len(sw_feedback) == 1
        assert sw_feedback[0]["severity"] == "warning"
        assert "100 interactions" in sw_feedback[0]["message"]
        assert "0 dismissals" in sw_feedback[0]["message"]
        assert "10 notification dismissals" in sw_feedback[0]["message"]

    def test_no_anomaly_when_few_feedback_dismissals(self, tmp_path):
        """No feedback wiring anomaly when there are 5 or fewer feedback dismissals."""
        weights = [
            {"source_key": "email.personal", "interactions": 100, "engagements": 5, "dismissals": 0},
        ]
        feedback = [("notification", "dismissed")] * 3
        _create_all_dbs(tmp_path, event_count=200, source_weights=weights, feedback_rows=feedback)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        sw_feedback = [a for a in anomalies if a["category"] == "source_weight_feedback"]
        assert len(sw_feedback) == 0


class TestNoAnomalyWhenLearningActive:
    """No source_weight anomaly when interaction counts and dismissals are healthy."""

    def test_no_anomaly_when_learning_active(self, tmp_path):
        """Healthy source weights with interactions and dismissals produce no anomaly."""
        weights = [
            {"source_key": "email.personal", "interactions": 200, "engagements": 15, "dismissals": 8},
            {"source_key": "email.marketing", "interactions": 100, "engagements": 2, "dismissals": 25},
        ]
        feedback = [("notification", "dismissed")] * 20
        _create_all_dbs(tmp_path, event_count=500, source_weights=weights, feedback_rows=feedback)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        sw_anomalies = [
            a for a in anomalies
            if a["category"] in ("source_weight_learning", "source_weight_feedback")
        ]
        assert len(sw_anomalies) == 0


class TestDetectAnomaliesDirectly:
    """Test detect_anomalies() with crafted section dicts (no DB needed)."""

    def test_zero_interactions_warning(self):
        """Direct unit test for zero-interactions anomaly."""
        sections = {
            "events": {"total": 500},
            "source_weights": {
                "email.personal": {"interactions": 0, "engagements": 0, "dismissals": 0},
                "email.work": {"interactions": 0, "engagements": 0, "dismissals": 0},
            },
        }
        anomalies = detect_anomalies(sections)
        cats = [a["category"] for a in anomalies]
        assert "source_weight_learning" in cats

    def test_zero_dismissals_feedback_warning(self):
        """Direct unit test for zero-dismissals-with-feedback anomaly."""
        sections = {
            "events": {"total": 500},
            "source_weights": {
                "email.personal": {"interactions": 50, "engagements": 3, "dismissals": 0},
            },
            "feedback": [
                {"action_type": "notification", "feedback_type": "dismissed", "count": 20},
            ],
        }
        anomalies = detect_anomalies(sections)
        cats = [a["category"] for a in anomalies]
        assert "source_weight_feedback" in cats

    def test_no_anomaly_healthy(self):
        """No source weight anomalies when learning is active."""
        sections = {
            "events": {"total": 500},
            "source_weights": {
                "email.personal": {"interactions": 100, "engagements": 10, "dismissals": 5},
            },
            "feedback": [
                {"action_type": "notification", "feedback_type": "dismissed", "count": 20},
            ],
        }
        anomalies = detect_anomalies(sections)
        sw_cats = [a["category"] for a in anomalies if a["category"].startswith("source_weight")]
        assert len(sw_cats) == 0

    def test_skips_when_source_weights_has_error(self):
        """No anomaly when source_weights section has an error key."""
        sections = {
            "events": {"total": 500},
            "source_weights": {"error": "could not connect"},
        }
        anomalies = detect_anomalies(sections)
        sw_cats = [a["category"] for a in anomalies if a["category"].startswith("source_weight")]
        assert len(sw_cats) == 0
