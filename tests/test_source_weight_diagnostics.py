"""
Tests for SourceWeightManager.get_diagnostics().

Validates diagnostic output structure, feedback loop health assessment,
stale source detection, and drift activity reporting.
"""

from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.source_weights import SourceWeightManager


@pytest.fixture()
def swm(db):
    """A SourceWeightManager with seeded defaults."""
    manager = SourceWeightManager(db)
    manager.seed_defaults()
    return manager


def _insert_source(db, source_key, interactions=0, engagements=0, dismissals=0,
                   ai_drift=0.0, user_weight=0.5, ai_updated_at=None):
    """Insert a source weight row directly for test setup."""
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO source_weights
               (source_key, category, label, description, user_weight,
                ai_drift, interactions, engagements, dismissals, ai_updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_key, "test", "Test", "", user_weight,
             ai_drift, interactions, engagements, dismissals, ai_updated_at or now),
        )


class TestDiagnosticsStructure:
    def test_diagnostics_returns_complete_structure(self, swm):
        """Seeded manager should return all expected diagnostic keys."""
        diag = swm.get_diagnostics()

        expected_keys = {
            "total_sources", "total_interactions", "total_engagements",
            "total_dismissals", "sources_with_drift", "feedback_loop_health",
            "per_source", "stale_sources", "drift_active",
        }
        assert expected_keys == set(diag.keys())
        assert diag["total_sources"] > 0
        assert isinstance(diag["per_source"], list)
        assert len(diag["per_source"]) == diag["total_sources"]


class TestFeedbackLoopHealth:
    def test_diagnostics_feedback_loop_broken(self, db):
        """Sources with many interactions but zero feedback should report 'broken'."""
        swm = SourceWeightManager(db)
        _insert_source(db, "test.broken1", interactions=200, engagements=0, dismissals=0)
        _insert_source(db, "test.broken2", interactions=150, engagements=0, dismissals=0)

        diag = swm.get_diagnostics()
        assert diag["feedback_loop_health"] == "broken"

    def test_diagnostics_feedback_loop_healthy(self, db):
        """Sources with both engagements and dismissals should report 'healthy'."""
        swm = SourceWeightManager(db)
        _insert_source(db, "test.healthy", interactions=50, engagements=10, dismissals=5)

        diag = swm.get_diagnostics()
        assert diag["feedback_loop_health"] == "healthy"

    def test_diagnostics_feedback_loop_partial(self, db):
        """Sources with only one type of feedback should report 'partial'."""
        swm = SourceWeightManager(db)
        _insert_source(db, "test.partial", interactions=50, engagements=5, dismissals=0)

        diag = swm.get_diagnostics()
        assert diag["feedback_loop_health"] == "partial"


class TestStaleSources:
    def test_diagnostics_stale_sources(self, db):
        """Sources with interactions but old updated_at should be flagged as stale."""
        swm = SourceWeightManager(db)
        stale_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        _insert_source(db, "test.stale", interactions=10, ai_updated_at=stale_time)
        _insert_source(db, "test.fresh", interactions=10)  # defaults to now

        diag = swm.get_diagnostics()
        assert "test.stale" in diag["stale_sources"]
        assert "test.fresh" not in diag["stale_sources"]


class TestEmptyTable:
    def test_diagnostics_handles_empty_table(self, db):
        """Empty source_weights table should return safely with zero counts."""
        swm = SourceWeightManager(db)
        # Don't seed defaults — table is empty
        diag = swm.get_diagnostics()

        assert diag["total_sources"] == 0
        assert diag["total_interactions"] == 0
        assert diag["total_engagements"] == 0
        assert diag["total_dismissals"] == 0
        assert diag["per_source"] == []
        assert diag["stale_sources"] == []
        assert diag["drift_active"] is False


class TestDriftActive:
    def test_diagnostics_drift_active_true(self, db):
        """Non-zero ai_drift should set drift_active to True."""
        swm = SourceWeightManager(db)
        _insert_source(db, "test.drifted", ai_drift=0.1)

        diag = swm.get_diagnostics()
        assert diag["drift_active"] is True
        assert diag["sources_with_drift"] == 1

    def test_diagnostics_drift_active_false(self, db):
        """All-zero drift should set drift_active to False."""
        swm = SourceWeightManager(db)
        _insert_source(db, "test.nodrift", ai_drift=0.0)

        diag = swm.get_diagnostics()
        assert diag["drift_active"] is False
        assert diag["sources_with_drift"] == 0
