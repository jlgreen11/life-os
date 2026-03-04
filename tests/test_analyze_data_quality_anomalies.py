"""Tests for anomaly detection and health scoring in analyze-data-quality.py."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# The script lives in scripts/, add it to sys.path for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from importlib import import_module

# Import the module with a hyphenated filename
_mod = import_module("analyze-data-quality")
detect_anomalies = _mod.detect_anomalies
compute_health_score = _mod.compute_health_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _healthy_sections():
    """Return a sections dict that should produce zero anomalies."""
    now = datetime.now(UTC)
    recent = (now - timedelta(hours=1)).isoformat()
    return {
        "prediction_pipeline": {
            "total_generated": 50,
            "surfaced": 30,
            "filtered": 20,
            "event_activity": {
                "usermodel.prediction.generated": 50,
                "usermodel.prediction.deduplicated": 5,
            },
        },
        "prediction_accuracy": {
            "NEED": {"total": 10, "accurate": 8, "inaccurate": 2},
        },
        "user_model": {
            "episodes": 200,
            "routines": 5,
            "workflows": 3,
            "semantic_facts": 42,
        },
        "connectors": {
            "gmail": {"status": "ok", "last_sync": recent, "error": None},
        },
        "events": {
            "total": 1000,
            "sources": {
                "gmail": {"count": 500, "last_event": recent},
                "calendar": {"count": 300, "last_event": recent},
            },
        },
        "notifications": {"pending": 5, "read": 20, "dismissed": 3},
    }


# ---------------------------------------------------------------------------
# Test: healthy sections produce no anomalies
# ---------------------------------------------------------------------------

class TestHealthySections:
    def test_no_anomalies(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert anomalies == []

    def test_health_score_perfect(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert compute_health_score(anomalies) == 100


# ---------------------------------------------------------------------------
# Test: (a) Prediction table empty despite generation events
# ---------------------------------------------------------------------------

class TestPredictionPersistence:
    def test_flags_critical_when_table_empty_but_events_exist(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["total_generated"] = 0
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 42

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "prediction_persistence"]
        assert len(match) == 1
        assert match[0]["severity"] == "critical"
        assert "42 generation events" in match[0]["message"]
        assert "store_prediction()" in match[0]["recommendation"]

    def test_no_flag_when_both_zero(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["total_generated"] = 0
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 0

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "prediction_persistence"]

    def test_no_flag_when_table_has_data(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["total_generated"] = 10
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 10

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "prediction_persistence"]


# ---------------------------------------------------------------------------
# Test: (b) High dedup ratio
# ---------------------------------------------------------------------------

class TestHighDedupRatio:
    def test_flags_warning_when_dedup_exceeds_10x(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 10
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.deduplicated"] = 150

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "prediction_deduplication"]
        assert len(match) == 1
        assert match[0]["severity"] == "warning"
        assert "15.0x" in match[0]["message"]

    def test_no_flag_when_dedup_is_low(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 100
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.deduplicated"] = 50

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "prediction_deduplication"]

    def test_no_flag_when_no_generation_events(self):
        sections = _healthy_sections()
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.generated"] = 0
        sections["prediction_pipeline"]["event_activity"]["usermodel.prediction.deduplicated"] = 100

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "prediction_deduplication"]


# ---------------------------------------------------------------------------
# Test: (c) Zero routines with sufficient episodes
# ---------------------------------------------------------------------------

class TestZeroRoutines:
    def test_flags_warning_when_zero_routines_many_episodes(self):
        sections = _healthy_sections()
        sections["user_model"]["routines"] = 0
        sections["user_model"]["episodes"] = 250

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "routine_detection"]
        assert len(match) == 1
        assert match[0]["severity"] == "warning"
        assert "250 episodes" in match[0]["message"]

    def test_no_flag_when_episodes_low(self):
        sections = _healthy_sections()
        sections["user_model"]["routines"] = 0
        sections["user_model"]["episodes"] = 50

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "routine_detection"]


# ---------------------------------------------------------------------------
# Test: (d) Zero workflows
# ---------------------------------------------------------------------------

class TestZeroWorkflows:
    def test_flags_warning_when_zero_workflows_many_episodes(self):
        sections = _healthy_sections()
        sections["user_model"]["workflows"] = 0
        sections["user_model"]["episodes"] = 150

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "workflow_detection"]
        assert len(match) == 1
        assert match[0]["severity"] == "warning"

    def test_no_flag_when_workflows_exist(self):
        sections = _healthy_sections()
        sections["user_model"]["workflows"] = 2
        sections["user_model"]["episodes"] = 500

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "workflow_detection"]


# ---------------------------------------------------------------------------
# Test: (e) Connector errors
# ---------------------------------------------------------------------------

class TestConnectorErrors:
    def test_flags_critical_for_error_connector(self):
        sections = _healthy_sections()
        sections["connectors"]["signal"] = {
            "status": "error",
            "last_sync": None,
            "error": "authentication failed",
        }

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "connector_error"]
        assert len(match) == 1
        assert match[0]["severity"] == "critical"
        assert "signal" in match[0]["message"]
        assert "authentication failed" in match[0]["message"]

    def test_multiple_error_connectors(self):
        sections = _healthy_sections()
        sections["connectors"]["signal"] = {"status": "error", "last_sync": None, "error": "auth fail"}
        sections["connectors"]["caldav"] = {"status": "error", "last_sync": None, "error": "timeout"}

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "connector_error"]
        assert len(match) == 2

    def test_no_flag_for_ok_connectors(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert not [a for a in anomalies if a["category"] == "connector_error"]

    def test_handles_connectors_section_with_error_key(self):
        """When the connectors section itself has an error, skip detection."""
        sections = _healthy_sections()
        sections["connectors"] = {"error": "could not connect to state.db"}

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "connector_error"]


# ---------------------------------------------------------------------------
# Test: (f) Stale data sources
# ---------------------------------------------------------------------------

class TestStaleSources:
    def test_flags_warning_for_stale_source(self):
        sections = _healthy_sections()
        old_time = (datetime.now(UTC) - timedelta(days=14)).isoformat()
        sections["events"]["sources"]["old_connector"] = {"count": 100, "last_event": old_time}

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "stale_source"]
        assert len(match) == 1
        assert match[0]["severity"] == "warning"
        assert "old_connector" in match[0]["message"]

    def test_no_flag_for_recent_sources(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert not [a for a in anomalies if a["category"] == "stale_source"]

    def test_handles_unparseable_timestamps(self):
        sections = _healthy_sections()
        sections["events"]["sources"]["bad"] = {"count": 10, "last_event": "not-a-date"}

        # Should not crash
        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "stale_source" and "bad" in a["message"]]


# ---------------------------------------------------------------------------
# Test: (g) No prediction accuracy data
# ---------------------------------------------------------------------------

class TestNoPredictionAccuracy:
    def test_flags_info_when_empty(self):
        sections = _healthy_sections()
        sections["prediction_accuracy"] = {}

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "prediction_accuracy"]
        assert len(match) == 1
        assert match[0]["severity"] == "info"

    def test_no_flag_when_data_exists(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert not [a for a in anomalies if a["category"] == "prediction_accuracy"]


# ---------------------------------------------------------------------------
# Test: (h) Pending notification backlog
# ---------------------------------------------------------------------------

class TestNotificationBacklog:
    def test_flags_warning_when_pending_over_50(self):
        sections = _healthy_sections()
        sections["notifications"]["pending"] = 75

        anomalies = detect_anomalies(sections)
        match = [a for a in anomalies if a["category"] == "notification_backlog"]
        assert len(match) == 1
        assert match[0]["severity"] == "warning"
        assert "75" in match[0]["message"]

    def test_no_flag_when_pending_is_low(self):
        anomalies = detect_anomalies(_healthy_sections())
        assert not [a for a in anomalies if a["category"] == "notification_backlog"]

    def test_handles_notifications_with_error(self):
        sections = _healthy_sections()
        sections["notifications"] = {"error": "could not connect"}

        anomalies = detect_anomalies(sections)
        assert not [a for a in anomalies if a["category"] == "notification_backlog"]


# ---------------------------------------------------------------------------
# Test: health score computation
# ---------------------------------------------------------------------------

class TestHealthScore:
    def test_perfect_score(self):
        assert compute_health_score([]) == 100

    def test_critical_deduction(self):
        anomalies = [{"severity": "critical"}]
        assert compute_health_score(anomalies) == 80

    def test_warning_deduction(self):
        anomalies = [{"severity": "warning"}]
        assert compute_health_score(anomalies) == 90

    def test_info_deduction(self):
        anomalies = [{"severity": "info"}]
        assert compute_health_score(anomalies) == 98

    def test_mixed_severities(self):
        anomalies = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        # 100 - 20 - 20 - 10 - 2 = 48
        assert compute_health_score(anomalies) == 48

    def test_floor_at_zero(self):
        anomalies = [{"severity": "critical"}] * 10
        # 100 - 200 = -100 -> clamped to 0
        assert compute_health_score(anomalies) == 0

    def test_all_zeros_sections(self):
        """Sections with all zeros — no episodes means no routine/workflow warning."""
        sections = {
            "prediction_pipeline": {"total_generated": 0, "event_activity": {}},
            "prediction_accuracy": {},
            "user_model": {"episodes": 0, "routines": 0, "workflows": 0},
            "connectors": {},
            "events": {"sources": {}},
            "notifications": {},
        }
        anomalies = detect_anomalies(sections)
        # Only the empty prediction_accuracy info should fire
        assert len(anomalies) == 1
        assert anomalies[0]["category"] == "prediction_accuracy"
        assert compute_health_score(anomalies) == 98


class TestEdgeCases:
    def test_missing_sections(self):
        """detect_anomalies handles completely empty sections dict."""
        anomalies = detect_anomalies({})
        # Only prediction_accuracy empty check should fire (empty dict == {})
        # Actually prediction_accuracy key doesn't exist, so .get returns {}... no.
        # .get("prediction_accuracy", {}) returns {} when key missing, which IS empty
        match = [a for a in anomalies if a["category"] == "prediction_accuracy"]
        assert len(match) == 1

    def test_sections_with_error_values(self):
        """Sections containing error strings should not crash detection."""
        sections = {
            "prediction_pipeline": {"error": "could not connect"},
            "prediction_accuracy": {"error": "could not connect"},
            "user_model": {"error": "could not connect"},
            "connectors": {"error": "could not connect"},
            "events": {"error": "could not connect"},
            "notifications": {"error": "could not connect"},
        }
        # Should not raise
        anomalies = detect_anomalies(sections)
        # The prediction_accuracy section has an error key, so it's not empty
        assert not [a for a in anomalies if a["category"] == "prediction_accuracy"]
