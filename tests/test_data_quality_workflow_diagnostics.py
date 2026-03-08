"""
Tests for workflow detection diagnostics in the data quality analyzer.

Verifies that ``scripts/analyze-data-quality.py`` includes a
``workflow_diagnostics`` section with email/task/calendar event counts,
episode interaction type distribution, and workflow detector thresholds,
plus anomaly detection for data conditions that block workflow discovery.
"""

import importlib.util
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _get_analyze():
    """Import the analyze function from ``scripts/analyze-data-quality.py``.

    The script filename contains a hyphen, so it cannot be imported with a
    standard ``import`` statement.  We use ``importlib`` to load it by file
    path instead.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "analyze-data-quality.py"
    spec = importlib.util.spec_from_file_location("analyze_data_quality", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


def _setup_minimal_dbs(tmp_path: Path) -> None:
    """Create minimal stub databases so analyze() doesn't error on missing DBs."""
    # events.db — with source column for the events section
    ev_conn = sqlite3.connect(str(tmp_path / "events.db"))
    ev_conn.execute(
        "CREATE TABLE events (id TEXT, type TEXT, source TEXT, timestamp TEXT, payload TEXT)"
    )
    ev_conn.commit()
    ev_conn.close()

    # user_model.db — all tables queried by the analyzer
    um_conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    um_conn.execute(
        "CREATE TABLE predictions (id TEXT PRIMARY KEY, prediction_type TEXT, "
        "was_surfaced INTEGER, was_accurate INTEGER, resolution_reason TEXT, "
        "created_at TEXT, resolved_at TEXT, filter_reason TEXT, user_response TEXT)"
    )
    um_conn.execute(
        "CREATE TABLE signal_profiles (profile_type TEXT PRIMARY KEY, "
        "samples_count INTEGER, updated_at TEXT)"
    )
    um_conn.execute("CREATE TABLE insights (id TEXT PRIMARY KEY, type TEXT, feedback TEXT)")
    um_conn.execute(
        "CREATE TABLE episodes (id TEXT PRIMARY KEY, interaction_type TEXT, "
        "timestamp TEXT)"
    )
    um_conn.execute("CREATE TABLE semantic_facts (id TEXT PRIMARY KEY, category TEXT)")
    um_conn.execute("CREATE TABLE routines (id TEXT PRIMARY KEY)")
    um_conn.execute("CREATE TABLE workflows (id TEXT PRIMARY KEY)")
    um_conn.execute("CREATE TABLE communication_templates (id TEXT PRIMARY KEY)")
    um_conn.commit()
    um_conn.close()

    # state.db
    st_conn = sqlite3.connect(str(tmp_path / "state.db"))
    st_conn.execute("CREATE TABLE notifications (id TEXT, status TEXT)")
    st_conn.execute(
        "CREATE TABLE tasks (id TEXT, status TEXT, created_at TEXT)"
    )
    st_conn.execute(
        "CREATE TABLE connector_state (connector_id TEXT, status TEXT, "
        "last_sync TEXT, last_error TEXT)"
    )
    st_conn.commit()
    st_conn.close()

    # preferences.db
    pref_conn = sqlite3.connect(str(tmp_path / "preferences.db"))
    pref_conn.execute(
        "CREATE TABLE feedback_log (id TEXT, action_type TEXT, feedback_type TEXT)"
    )
    pref_conn.execute(
        "CREATE TABLE source_weights (source_key TEXT, user_weight REAL, "
        "ai_drift REAL, ai_updated_at TEXT, interactions INTEGER, "
        "engagements INTEGER, dismissals INTEGER)"
    )
    pref_conn.commit()
    pref_conn.close()

    # entities.db (for database_health section)
    sqlite3.connect(str(tmp_path / "entities.db")).close()


def _insert_events(tmp_path: Path, events: list[tuple]) -> None:
    """Insert events into events.db. Each tuple: (type, timestamp, payload)."""
    conn = sqlite3.connect(str(tmp_path / "events.db"))
    for i, (etype, ts, payload) in enumerate(events):
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            (str(i), etype, "test", ts, payload),
        )
    conn.commit()
    conn.close()


def _insert_episodes(tmp_path: Path, episodes: list[tuple]) -> None:
    """Insert episodes into user_model.db. Each tuple: (interaction_type,)."""
    conn = sqlite3.connect(str(tmp_path / "user_model.db"))
    for i, (itype,) in enumerate(episodes):
        conn.execute(
            "INSERT INTO episodes (id, interaction_type, timestamp) VALUES (?, ?, ?)",
            (str(i), itype, datetime.now(timezone.utc).isoformat()),
        )
    conn.commit()
    conn.close()


class TestWorkflowDiagnosticsSectionPresent:
    """The workflow_diagnostics section must exist in the report."""

    def test_workflow_diagnostics_section_present(self, tmp_path):
        """Run analyze against temp DBs, verify 'workflow_diagnostics' key exists."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        report = analyze(str(tmp_path))

        assert "workflow_diagnostics" in report["sections"], (
            "workflow_diagnostics section must be present in report"
        )
        wf = report["sections"]["workflow_diagnostics"]
        assert "thresholds" in wf
        assert "email" in wf
        assert "tasks" in wf
        assert "calendar" in wf
        assert "episode_interaction_types" in wf

    def test_thresholds_match_workflow_detector_defaults(self, tmp_path):
        """Thresholds should match WorkflowDetector default values."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        report = analyze(str(tmp_path))
        thresholds = report["sections"]["workflow_diagnostics"]["thresholds"]

        assert thresholds["min_occurrences"] == 3
        assert thresholds["min_completions"] == 2
        assert thresholds["min_steps"] == 2
        assert thresholds["max_step_gap_hours"] == 12


class TestWorkflowDiagnosticsEmailCounts:
    """Email counts in workflow diagnostics must reflect seeded data."""

    def test_email_counts_correct(self, tmp_path):
        """Seed events.db with known email counts, verify diagnostics report them."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        now = datetime.now(timezone.utc).isoformat()
        events = []
        # 15 received emails from 3 senders
        for i in range(10):
            events.append(("email.received", now, '{"email_from": "alice@example.com"}'))
        for i in range(3):
            events.append(("email.received", now, '{"email_from": "bob@example.com"}'))
        for i in range(2):
            events.append(("email.received", now, '{"email_from": "carol@example.com"}'))
        # 4 sent emails
        for i in range(4):
            events.append(("email.sent", now, "{}"))
        _insert_events(tmp_path, events)

        report = analyze(str(tmp_path))
        email = report["sections"]["workflow_diagnostics"]["email"]

        assert email["received_30d"] == 15
        assert email["sent_30d"] == 4
        assert email["top_senders"]["alice@example.com"] == 10
        assert email["top_senders"]["bob@example.com"] == 3
        assert email["top_senders"]["carol@example.com"] == 2

    def test_task_and_calendar_counts(self, tmp_path):
        """Verify task and calendar event counts are reported."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        now = datetime.now(timezone.utc).isoformat()
        events = [
            ("task.created", now, "{}"),
            ("task.created", now, "{}"),
            ("task.created", now, "{}"),
            ("calendar.event.created", now, "{}"),
            ("calendar.event.created", now, "{}"),
        ]
        _insert_events(tmp_path, events)

        report = analyze(str(tmp_path))
        wf = report["sections"]["workflow_diagnostics"]

        assert wf["tasks"]["created_30d"] == 3
        assert wf["calendar"]["events_created_30d"] == 2


class TestWorkflowDiagnosticsLowSentAnomaly:
    """Anomaly must fire when email.sent is low relative to email.received."""

    def test_low_sent_anomaly_generated(self, tmp_path):
        """Seed with 200 email.received and 2 email.sent, verify anomaly."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        now = datetime.now(timezone.utc).isoformat()
        events = []
        for _ in range(200):
            events.append(("email.received", now, '{"email_from": "sender@example.com"}'))
        for _ in range(2):
            events.append(("email.sent", now, "{}"))
        _insert_events(tmp_path, events)

        report = analyze(str(tmp_path))
        anomalies = report["anomalies"]

        email_anomalies = [
            a for a in anomalies if a["category"] == "workflow_email_imbalance"
        ]
        assert len(email_anomalies) == 1, (
            f"Expected 1 workflow_email_imbalance anomaly, got {len(email_anomalies)}"
        )
        assert "200 received" in email_anomalies[0]["message"]
        assert "2 sent" in email_anomalies[0]["message"]

    def test_no_anomaly_when_sent_sufficient(self, tmp_path):
        """No anomaly when sent count >= 5."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        now = datetime.now(timezone.utc).isoformat()
        events = []
        for _ in range(200):
            events.append(("email.received", now, '{"email_from": "sender@example.com"}'))
        for _ in range(10):
            events.append(("email.sent", now, "{}"))
        _insert_events(tmp_path, events)

        report = analyze(str(tmp_path))
        email_anomalies = [
            a for a in report["anomalies"] if a["category"] == "workflow_email_imbalance"
        ]
        assert len(email_anomalies) == 0


class TestWorkflowDiagnosticsEpisodeTypes:
    """Episode interaction type diagnostics and anomalies."""

    def test_episode_type_distribution(self, tmp_path):
        """Verify episode interaction type distribution is reported."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        episodes = [
            ("email",), ("email",), ("email",),
            ("task",), ("task",),
            (None,), (None,),
            ("unknown",),
        ]
        _insert_episodes(tmp_path, episodes)

        report = analyze(str(tmp_path))
        ep = report["sections"]["workflow_diagnostics"]["episode_interaction_types"]

        assert ep["total_episodes"] == 8
        assert ep["distribution"]["email"] == 3
        assert ep["distribution"]["task"] == 2
        assert ep["distribution"]["None"] == 2
        assert ep["distribution"]["unknown"] == 1
        assert ep["null_unknown_communication_count"] == 3  # 2 None + 1 unknown

    def test_stale_interaction_type_anomaly(self, tmp_path):
        """Anomaly fires when >50% of episodes have NULL/unknown/communication type."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        # 8 bad types, 2 good types = 80% bad
        episodes = (
            [(None,)] * 4
            + [("unknown",)] * 2
            + [("communication",)] * 2
            + [("email",)] * 2
        )
        _insert_episodes(tmp_path, episodes)

        report = analyze(str(tmp_path))
        type_anomalies = [
            a for a in report["anomalies"]
            if a["category"] == "workflow_stale_interaction_types"
        ]
        assert len(type_anomalies) == 1
        assert "8/10" in type_anomalies[0]["message"]

    def test_no_stale_anomaly_when_types_good(self, tmp_path):
        """No anomaly when <=50% of episodes have bad types."""
        analyze = _get_analyze()
        _setup_minimal_dbs(tmp_path)

        # 2 bad, 8 good = 20% bad
        episodes = [(None,)] * 2 + [("email",)] * 8
        _insert_episodes(tmp_path, episodes)

        report = analyze(str(tmp_path))
        type_anomalies = [
            a for a in report["anomalies"]
            if a["category"] == "workflow_stale_interaction_types"
        ]
        assert len(type_anomalies) == 0
