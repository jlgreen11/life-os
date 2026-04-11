"""Tests for connector-aware signal profile analysis in analyze-data-quality.py.

Verifies that:
- Profile missing with qualifying events → severity 'warning'
- Profile missing without qualifying events → severity 'info'
- Health score is higher when missing profiles have no qualifying events
- Connector error state is correctly annotated in related anomalies
- missing_profile_detail is included in the signal_profiles report section
"""

import importlib
import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the script module (hyphenated filename requires importlib)
# ---------------------------------------------------------------------------

_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

_mod_spec = importlib.util.spec_from_file_location(
    "analyze_data_quality",
    Path(__file__).resolve().parent.parent / "scripts" / "analyze-data-quality.py",
)
_mod = importlib.util.module_from_spec(_mod_spec)
_mod_spec.loader.exec_module(_mod)

analyze = _mod.analyze
detect_anomalies = _mod.detect_anomalies
compute_health_score = _mod.compute_health_score
PROFILE_EVENT_TYPES = _mod.PROFILE_EVENT_TYPES


# ---------------------------------------------------------------------------
# Minimal database helpers
# ---------------------------------------------------------------------------


def _create_events_db(tmp_path: Path, event_rows: list[dict] | None = None) -> None:
    """Create events.db with the events table and optional rows.

    Args:
        tmp_path: Directory where events.db will be created.
        event_rows: Optional list of dicts with 'type' and 'source' keys to insert.
    """
    conn = sqlite3.connect(str(tmp_path / "events.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            type TEXT,
            source TEXT,
            timestamp TEXT,
            priority TEXT DEFAULT 'normal',
            payload TEXT DEFAULT '{}',
            metadata TEXT DEFAULT '{}'
        )
    """)
    if event_rows:
        for i, row in enumerate(event_rows):
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp) VALUES (?, ?, ?, ?)",
                (
                    f"evt-{i}",
                    row["type"],
                    row.get("source", "test"),
                    datetime.now(UTC).isoformat(),
                ),
            )
    conn.commit()
    conn.close()


def _create_user_model_db(
    tmp_path: Path,
    signal_profile_rows: list[dict] | None = None,
) -> None:
    """Create user_model.db with required tables and optional signal_profiles rows.

    Args:
        tmp_path: Directory where user_model.db will be created.
        signal_profile_rows: Optional list of dicts with 'profile_type' and 'samples_count'.
    """
    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    # Minimal schema — only tables used by signal_profiles analysis
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            prediction_type TEXT,
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
            data TEXT DEFAULT '{}',
            samples_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS insights (id INTEGER PRIMARY KEY, type TEXT, feedback TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, interaction_type TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS semantic_facts (id INTEGER PRIMARY KEY, category TEXT, key TEXT, value TEXT, confidence REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS routines (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS workflows (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS communication_templates (id INTEGER PRIMARY KEY)")
    if signal_profile_rows:
        for row in signal_profile_rows:
            conn.execute(
                "INSERT INTO signal_profiles (profile_type, samples_count, updated_at) VALUES (?, ?, ?)",
                (row["profile_type"], row.get("samples_count", 10), datetime.now(UTC).isoformat()),
            )
    conn.commit()
    conn.close()


def _create_state_db(tmp_path: Path, connector_rows: list[dict] | None = None) -> None:
    """Create state.db with minimal tables.

    Args:
        tmp_path: Directory where state.db will be created.
        connector_rows: Optional list of dicts with connector_id, status, last_sync, last_error.
    """
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY, status TEXT)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY,
            status TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS connector_state (
            connector_id TEXT PRIMARY KEY,
            status TEXT,
            last_sync TEXT,
            last_error TEXT
        )
    """)
    if connector_rows:
        for row in connector_rows:
            conn.execute(
                "INSERT INTO connector_state (connector_id, status, last_sync, last_error) VALUES (?, ?, ?, ?)",
                (
                    row["connector_id"],
                    row.get("status", "ok"),
                    row.get("last_sync"),
                    row.get("last_error"),
                ),
            )
    conn.commit()
    conn.close()


def _create_preferences_db(tmp_path: Path) -> None:
    """Create preferences.db with minimal tables."""
    conn = sqlite3.connect(str(tmp_path / "preferences.db"))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log (
            id INTEGER PRIMARY KEY,
            action_type TEXT,
            feedback_type TEXT,
            context TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_weights (
            source_key TEXT PRIMARY KEY,
            user_weight REAL DEFAULT 1.0,
            ai_drift REAL DEFAULT 0.0,
            ai_updated_at TEXT,
            interactions INTEGER DEFAULT 0,
            engagements INTEGER DEFAULT 0,
            dismissals INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _create_entities_db(tmp_path: Path) -> None:
    """Create entities.db with minimal tables."""
    conn = sqlite3.connect(str(tmp_path / "entities.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()


def _create_all_dbs(
    tmp_path: Path,
    event_rows: list[dict] | None = None,
    signal_profile_rows: list[dict] | None = None,
    connector_rows: list[dict] | None = None,
) -> None:
    """Create all required databases in tmp_path.

    Args:
        tmp_path: Directory to create databases in.
        event_rows: Event rows to insert (see _create_events_db).
        signal_profile_rows: Signal profile rows to insert (see _create_user_model_db).
        connector_rows: Connector state rows to insert (see _create_state_db).
    """
    _create_events_db(tmp_path, event_rows)
    _create_user_model_db(tmp_path, signal_profile_rows)
    _create_state_db(tmp_path, connector_rows)
    _create_preferences_db(tmp_path)
    _create_entities_db(tmp_path)


# ---------------------------------------------------------------------------
# Tests: PROFILE_EVENT_TYPES constant
# ---------------------------------------------------------------------------


class TestProfileEventTypes:
    """PROFILE_EVENT_TYPES defines qualifying event types for each profile."""

    def test_all_expected_profiles_present(self):
        """All 9 expected signal profile types have entries."""
        expected = {
            "linguistic", "linguistic_inbound", "cadence", "mood_signals",
            "relationships", "temporal", "topics", "spatial", "decision",
        }
        assert expected == set(PROFILE_EVENT_TYPES.keys())

    def test_linguistic_only_outbound(self):
        """'linguistic' only includes outbound event types (email.sent, message.sent)."""
        types = PROFILE_EVENT_TYPES["linguistic"]
        # Must include outbound events
        assert "email.sent" in types
        assert "message.sent" in types
        # Must NOT include inbound-only events
        assert "email.received" not in types
        assert "message.received" not in types

    def test_linguistic_inbound_only_inbound(self):
        """'linguistic_inbound' only includes inbound event types."""
        types = PROFILE_EVENT_TYPES["linguistic_inbound"]
        assert "email.received" in types
        assert "message.received" in types
        assert "email.sent" not in types
        assert "message.sent" not in types

    def test_all_profiles_have_at_least_one_type(self):
        """Every profile has at least one qualifying event type."""
        for profile, types in PROFILE_EVENT_TYPES.items():
            assert len(types) > 0, f"Profile '{profile}' has no qualifying event types"


# ---------------------------------------------------------------------------
# Tests: detect_anomalies — missing profile severity
# ---------------------------------------------------------------------------


class TestMissingProfileSeverity:
    """Anomaly severity depends on whether qualifying events exist."""

    def test_warning_when_qualifying_events_exist(self):
        """Profile missing with qualifying events → severity 'warning'."""
        sections = {
            "signal_profiles": {
                "missing_profiles": ["cadence"],
                "missing_profile_detail": {
                    "cadence": "42_qualifying_events_exist",
                },
            },
        }
        anomalies = detect_anomalies(sections)
        cadence_anomalies = [a for a in anomalies if a["category"] == "missing_profile"]
        assert len(cadence_anomalies) == 1
        assert cadence_anomalies[0]["severity"] == "warning"
        assert "42" in cadence_anomalies[0]["message"]
        assert "cadence" in cadence_anomalies[0]["message"]

    def test_info_when_no_qualifying_events(self):
        """Profile missing without qualifying events → severity 'info'."""
        sections = {
            "signal_profiles": {
                "missing_profiles": ["linguistic"],
                "missing_profile_detail": {
                    "linguistic": "no_qualifying_events",
                },
            },
        }
        anomalies = detect_anomalies(sections)
        linguistic_anomalies = [a for a in anomalies if a["category"] == "missing_profile"]
        assert len(linguistic_anomalies) == 1
        assert linguistic_anomalies[0]["severity"] == "info"
        assert "no qualifying events" in linguistic_anomalies[0]["message"]

    def test_no_anomaly_when_all_profiles_present(self):
        """No missing_profile anomalies when missing_profiles list is empty."""
        sections = {
            "signal_profiles": {
                "missing_profiles": [],
                "missing_profile_detail": {},
            },
        }
        anomalies = detect_anomalies(sections)
        missing_anomalies = [a for a in anomalies if a["category"] == "missing_profile"]
        assert missing_anomalies == []

    def test_mixed_profiles_generate_mixed_severities(self):
        """Multiple missing profiles with different detail values produce correct severities."""
        sections = {
            "signal_profiles": {
                "missing_profiles": ["linguistic", "cadence", "topics"],
                "missing_profile_detail": {
                    "linguistic": "no_qualifying_events",
                    "cadence": "100_qualifying_events_exist",
                    "topics": "no_qualifying_events",
                },
            },
        }
        anomalies = detect_anomalies(sections)
        missing_anomalies = [a for a in anomalies if a["category"] == "missing_profile"]
        assert len(missing_anomalies) == 3

        by_profile = {
            a["message"].split("'")[1]: a["severity"]
            for a in missing_anomalies
        }
        assert by_profile["linguistic"] == "info"
        assert by_profile["cadence"] == "warning"
        assert by_profile["topics"] == "info"

    def test_unknown_detail_falls_back_to_warning(self):
        """Missing profile with unknown detail defaults to 'warning' for safety."""
        sections = {
            "signal_profiles": {
                "missing_profiles": ["spatial"],
                "missing_profile_detail": {
                    "spatial": "unknown",
                },
            },
        }
        anomalies = detect_anomalies(sections)
        spatial_anomalies = [a for a in anomalies if a["category"] == "missing_profile"]
        assert len(spatial_anomalies) == 1
        assert spatial_anomalies[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# Tests: health score reflects profile severity correctly
# ---------------------------------------------------------------------------


class TestHealthScoreWithMissingProfiles:
    """Health score is higher when missing profiles have no qualifying events."""

    def test_info_profiles_cost_less_than_warning_profiles(self):
        """Info-level missing profiles reduce score less than warning-level ones."""
        # All missing profiles have no qualifying events → info severity
        sections_all_info = {
            "signal_profiles": {
                "missing_profiles": ["linguistic", "linguistic_inbound"],
                "missing_profile_detail": {
                    "linguistic": "no_qualifying_events",
                    "linguistic_inbound": "no_qualifying_events",
                },
            },
        }
        # All missing profiles have qualifying events → warning severity
        sections_all_warning = {
            "signal_profiles": {
                "missing_profiles": ["linguistic", "linguistic_inbound"],
                "missing_profile_detail": {
                    "linguistic": "50_qualifying_events_exist",
                    "linguistic_inbound": "30_qualifying_events_exist",
                },
            },
        }

        anomalies_info = detect_anomalies(sections_all_info)
        anomalies_warning = detect_anomalies(sections_all_warning)

        score_info = compute_health_score(anomalies_info)
        score_warning = compute_health_score(anomalies_warning)

        # Warnings (-10 each) should score lower than info (-2 each)
        assert score_info > score_warning

    def _base_sections(self) -> dict:
        """Minimal sections that suppress all anomalies except missing_profile ones.

        Includes a non-empty prediction_accuracy dict (prevents info anomaly (g)),
        and no other triggers. This isolates the health score impact of missing profiles.
        """
        return {
            # Non-empty prediction_accuracy suppresses the "no accuracy data" info anomaly
            "prediction_accuracy": {
                "NEED": {"total": 1, "accurate": 1, "inaccurate": 0},
            },
        }

    def test_two_info_profiles_deduct_4_from_100(self):
        """Two info-level missing profiles deduct 2 each = 4 total from score."""
        sections = self._base_sections()
        sections["signal_profiles"] = {
            "missing_profiles": ["linguistic", "linguistic_inbound"],
            "missing_profile_detail": {
                "linguistic": "no_qualifying_events",
                "linguistic_inbound": "no_qualifying_events",
            },
        }
        anomalies = detect_anomalies(sections)
        score = compute_health_score(anomalies)
        # 2 info anomalies x 2 = 4 deducted; score = 96
        assert score == 96

    def test_two_warning_profiles_deduct_20_from_100(self):
        """Two warning-level missing profiles deduct 10 each = 20 total from score."""
        sections = self._base_sections()
        sections["signal_profiles"] = {
            "missing_profiles": ["cadence", "relationships"],
            "missing_profile_detail": {
                "cadence": "100_qualifying_events_exist",
                "relationships": "200_qualifying_events_exist",
            },
        }
        anomalies = detect_anomalies(sections)
        score = compute_health_score(anomalies)
        # 2 warning anomalies x 10 = 20 deducted; score = 80
        assert score == 80


# ---------------------------------------------------------------------------
# Tests: connector error state annotates related anomalies
# ---------------------------------------------------------------------------


class TestConnectorErrorAnnotation:
    """Anomalies caused by connector downtime receive a root_cause_hint."""

    def _sections_with_google_error(self, days_ago: int = 50) -> dict:
        """Build a sections dict with Google connector in error state."""
        last_sync = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
        return {
            "connectors": {
                "google": {
                    "status": "error",
                    "last_sync": last_sync,
                    "error": "OAuth token expired",
                },
            },
            "user_model": {
                "episodes": 500,
                "routines": 0,
                "workflows": 0,
                "semantic_facts": 10,
            },
            "events": {
                "total": 5000,
                "sources": {},
            },
            "source_weights": {},
            "feedback": [],
        }

    def test_routine_detection_anomaly_gets_root_cause_hint(self):
        """routine_detection anomaly is annotated when connector is in error state."""
        sections = self._sections_with_google_error()
        anomalies = detect_anomalies(sections)

        routine_anomalies = [a for a in anomalies if a["category"] == "routine_detection"]
        assert len(routine_anomalies) == 1
        assert "root_cause_hint" in routine_anomalies[0]
        hint = routine_anomalies[0]["root_cause_hint"]
        assert "google" in hint
        # Hint should mention the connector downtime
        assert "error state" in hint or "down" in hint

    def test_stale_source_anomaly_gets_root_cause_hint(self):
        """stale_source anomaly is annotated when a connector is in error state."""
        now = datetime.now(UTC)
        stale_time = (now - timedelta(days=60)).isoformat()
        sections = {
            "connectors": {
                "google": {
                    "status": "error",
                    "last_sync": (now - timedelta(days=50)).isoformat(),
                    "error": "OAuth token expired",
                },
            },
            "events": {
                "total": 5000,
                "sources": {
                    "google": {"count": 1000, "last_event": stale_time},
                },
            },
            "user_model": {"episodes": 10, "routines": 0, "workflows": 0},
            "source_weights": {},
            "feedback": [],
        }
        anomalies = detect_anomalies(sections)

        stale_anomalies = [a for a in anomalies if a["category"] == "stale_source"]
        assert len(stale_anomalies) == 1
        assert "root_cause_hint" in stale_anomalies[0]

    def test_no_root_cause_hint_when_connectors_ok(self):
        """No root_cause_hint is added when all connectors are healthy."""
        now = datetime.now(UTC)
        recent = (now - timedelta(hours=1)).isoformat()
        sections = {
            "connectors": {
                "google": {"status": "ok", "last_sync": recent, "error": None},
            },
            "user_model": {
                "episodes": 500,
                "routines": 0,
                "workflows": 0,
                "semantic_facts": 10,
            },
            "events": {
                "total": 5000,
                "sources": {
                    "google": {"count": 1000, "last_event": recent},
                },
            },
            "source_weights": {},
            "feedback": [],
        }
        anomalies = detect_anomalies(sections)

        routine_anomalies = [a for a in anomalies if a["category"] == "routine_detection"]
        if routine_anomalies:
            assert "root_cause_hint" not in routine_anomalies[0]

    def test_hint_includes_downtime_duration(self):
        """root_cause_hint includes how long the connector has been down."""
        sections = self._sections_with_google_error(days_ago=50)
        anomalies = detect_anomalies(sections)

        routine_anomalies = [a for a in anomalies if a["category"] == "routine_detection"]
        assert len(routine_anomalies) == 1
        hint = routine_anomalies[0]["root_cause_hint"]
        # Should mention number of days
        assert "50" in hint

    def test_multiple_error_connectors_all_mentioned(self):
        """root_cause_hint lists all errored connectors, not just the first."""
        now = datetime.now(UTC)
        sections = {
            "connectors": {
                "google": {
                    "status": "error",
                    "last_sync": (now - timedelta(days=30)).isoformat(),
                    "error": "OAuth expired",
                },
                "imessage": {
                    "status": "error",
                    "last_sync": (now - timedelta(days=5)).isoformat(),
                    "error": "Permission denied",
                },
            },
            "user_model": {
                "episodes": 500,
                "routines": 0,
                "workflows": 0,
                "semantic_facts": 10,
            },
            "events": {
                "total": 5000,
                "sources": {},
            },
            "source_weights": {},
            "feedback": [],
        }
        anomalies = detect_anomalies(sections)

        routine_anomalies = [a for a in anomalies if a["category"] == "routine_detection"]
        assert len(routine_anomalies) == 1
        hint = routine_anomalies[0]["root_cause_hint"]
        assert "google" in hint
        assert "imessage" in hint


# ---------------------------------------------------------------------------
# Tests: full analyze() integration — missing_profile_detail in report
# ---------------------------------------------------------------------------


class TestAnalyzeMissingProfileDetail:
    """analyze() populates missing_profile_detail in signal_profiles section."""

    def test_missing_profile_with_qualifying_events_flagged(self, tmp_path):
        """Profile absent from signal_profiles table but events exist → qualifying_events_exist."""
        # Insert email.sent events (qualifying for 'linguistic' profile)
        events = [{"type": "email.sent", "source": "proton"}] * 5
        _create_all_dbs(tmp_path, event_rows=events)

        report = analyze(str(tmp_path))
        sp = report["sections"]["signal_profiles"]

        assert "linguistic" in sp.get("missing_profiles", [])
        detail = sp.get("missing_profile_detail", {})
        assert "linguistic" in detail
        assert "qualifying_events_exist" in detail["linguistic"]

    def test_missing_profile_without_qualifying_events_flagged(self, tmp_path):
        """Profile absent and no qualifying events → no_qualifying_events."""
        # No email.sent events — only system events
        events = [{"type": "system.rule.triggered", "source": "rules_engine"}] * 10
        _create_all_dbs(tmp_path, event_rows=events)

        report = analyze(str(tmp_path))
        sp = report["sections"]["signal_profiles"]

        # 'linguistic' requires email.sent / message.sent / system.user.command
        # system.rule.triggered is not a qualifying type
        assert "linguistic" in sp.get("missing_profiles", [])
        detail = sp.get("missing_profile_detail", {})
        assert detail.get("linguistic") == "no_qualifying_events"

    def test_present_profile_not_in_missing_detail(self, tmp_path):
        """A profile that exists in signal_profiles is not in missing_profile_detail."""
        # Insert a signal profile for 'cadence'
        _create_all_dbs(
            tmp_path,
            signal_profile_rows=[{"profile_type": "cadence", "samples_count": 20}],
        )

        report = analyze(str(tmp_path))
        sp = report["sections"]["signal_profiles"]

        assert "cadence" not in sp.get("missing_profiles", [])
        detail = sp.get("missing_profile_detail", {})
        assert "cadence" not in detail

    def test_report_contains_missing_profile_detail_key(self, tmp_path):
        """signal_profiles section always contains missing_profile_detail key."""
        _create_all_dbs(tmp_path)

        report = analyze(str(tmp_path))
        sp = report["sections"].get("signal_profiles", {})

        assert "missing_profile_detail" in sp, (
            "signal_profiles section should always include missing_profile_detail"
        )
