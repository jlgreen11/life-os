"""
Tests for InsightEngine data sufficiency diagnostics.

Validates the get_data_sufficiency_report() method that checks each correlator's
data readiness, reporting 'ready', 'partial', 'no_data', or 'error' status
based on signal profile sample counts and database record availability.
"""

import json
from unittest.mock import patch

import pytest

from services.insight_engine.engine import InsightEngine


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def insight_engine(db, user_model_store):
    """An InsightEngine wired to temporary databases."""
    return InsightEngine(db, user_model_store, timezone="UTC")


# =============================================================================
# get_data_sufficiency_report — structure
# =============================================================================


@pytest.mark.asyncio
async def test_sufficiency_report_returns_dict(insight_engine):
    """Report should return a dict with an entry per correlator."""
    report = await insight_engine.get_data_sufficiency_report()
    assert isinstance(report, dict)
    # Should have entries for all profile-based + episode + routine correlators
    assert len(report) >= 11


@pytest.mark.asyncio
async def test_sufficiency_report_has_expected_correlators(insight_engine):
    """Report should contain every known correlator key."""
    report = await insight_engine.get_data_sufficiency_report()
    expected_keys = {
        "_contact_gap_insights",
        "_communication_style_insights",
        "_inbound_style_insights",
        "_cadence_response_insights",
        "_temporal_pattern_insights",
        "_mood_trend_insights",
        "_topic_interest_insights",
        "_spatial_insights",
        "_decision_pattern_insights",
        "_place_frequency_insights",
        "_routine_insights",
    }
    assert expected_keys.issubset(report.keys())


@pytest.mark.asyncio
async def test_sufficiency_report_entry_structure(insight_engine):
    """Each profile-based entry should have profile, status, samples, min_required."""
    report = await insight_engine.get_data_sufficiency_report()
    entry = report["_contact_gap_insights"]
    assert "profile" in entry
    assert "status" in entry
    assert "samples" in entry
    assert "min_required" in entry
    assert entry["profile"] == "relationships"


@pytest.mark.asyncio
async def test_sufficiency_report_db_entry_structure(insight_engine):
    """Episode and routine entries should have source, status, count, min_required."""
    report = await insight_engine.get_data_sufficiency_report()

    ep_entry = report["_place_frequency_insights"]
    assert ep_entry["source"] == "episodes"
    assert "status" in ep_entry
    assert "count" in ep_entry
    assert "min_required" in ep_entry

    rt_entry = report["_routine_insights"]
    assert rt_entry["source"] == "routines"
    assert "status" in rt_entry
    assert "count" in rt_entry
    assert "min_required" in rt_entry


# =============================================================================
# get_data_sufficiency_report — status values
# =============================================================================


@pytest.mark.asyncio
async def test_sufficiency_reports_no_data_when_empty(insight_engine):
    """All profile-based correlators should report 'no_data' on a fresh database."""
    report = await insight_engine.get_data_sufficiency_report()
    profile_keys = [
        "_contact_gap_insights",
        "_communication_style_insights",
        "_inbound_style_insights",
        "_cadence_response_insights",
        "_temporal_pattern_insights",
        "_mood_trend_insights",
        "_topic_interest_insights",
        "_spatial_insights",
        "_decision_pattern_insights",
    ]
    for key in profile_keys:
        assert report[key]["status"] == "no_data", f"{key} should be no_data, got {report[key]['status']}"
        assert report[key]["samples"] == 0


@pytest.mark.asyncio
async def test_sufficiency_reports_ready_when_enough_samples(insight_engine):
    """Correlator should report 'ready' when samples meet min_required."""
    # Store a signal profile with enough samples to meet threshold
    # _mood_trend_insights needs 5 samples from 'mood_signals' profile
    with insight_engine.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("mood_signals", json.dumps({"valence": 0.7}), 10),
        )

    report = await insight_engine.get_data_sufficiency_report()
    assert report["_mood_trend_insights"]["status"] == "ready"
    assert report["_mood_trend_insights"]["samples"] == 10


@pytest.mark.asyncio
async def test_sufficiency_reports_partial_when_below_threshold(insight_engine):
    """Correlator should report 'partial' when some data exists but below threshold."""
    # _decision_pattern_insights needs 20 samples; insert only 5
    with insight_engine.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
            ("decision", json.dumps({"speed": "moderate"}), 5),
        )

    report = await insight_engine.get_data_sufficiency_report()
    assert report["_decision_pattern_insights"]["status"] == "partial"
    assert report["_decision_pattern_insights"]["samples"] == 5
    assert report["_decision_pattern_insights"]["min_required"] == 20


@pytest.mark.asyncio
async def test_sufficiency_reports_ready_for_episodes(insight_engine):
    """Episode-based correlator should report 'ready' when enough episodes exist."""
    with insight_engine.db.get_connection("user_model") as conn:
        for i in range(10):
            conn.execute(
                """INSERT INTO episodes (id, timestamp, event_id, interaction_type, content_summary)
                   VALUES (?, datetime('now'), ?, 'message', ?)""",
                (f"ep-{i}", f"evt-{i}", f"Episode {i} summary"),
            )

    report = await insight_engine.get_data_sufficiency_report()
    assert report["_place_frequency_insights"]["status"] == "ready"
    assert report["_place_frequency_insights"]["count"] == 10


@pytest.mark.asyncio
async def test_sufficiency_reports_no_data_for_empty_episodes(insight_engine):
    """Episode correlator should report 'no_data' when episodes table is empty."""
    report = await insight_engine.get_data_sufficiency_report()
    assert report["_place_frequency_insights"]["status"] == "no_data"
    assert report["_place_frequency_insights"]["count"] == 0


@pytest.mark.asyncio
async def test_sufficiency_reports_ready_for_routines(insight_engine):
    """Routine correlator should report 'ready' when at least one routine exists."""
    with insight_engine.db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO routines (name, trigger_condition, steps, consistency_score, times_observed)
               VALUES (?, ?, ?, ?, ?)""",
            ("Morning", "morning", "[]", 0.8, 5),
        )

    report = await insight_engine.get_data_sufficiency_report()
    assert report["_routine_insights"]["status"] == "ready"
    assert report["_routine_insights"]["count"] == 1


# =============================================================================
# get_data_sufficiency_report — error handling
# =============================================================================


@pytest.mark.asyncio
async def test_sufficiency_handles_db_error_for_episodes(insight_engine):
    """Episode count should be -1 and status 'error' when DB query fails."""
    original_get_connection = insight_engine.db.get_connection

    def broken_get_connection(db_name):
        if db_name == "user_model":
            raise Exception("database disk image is malformed")
        return original_get_connection(db_name)

    with patch.object(insight_engine.db, "get_connection", side_effect=broken_get_connection):
        report = await insight_engine.get_data_sufficiency_report()

    # All profile-based correlators should show error (get_signal_profile also
    # uses user_model connection), plus episode and routine entries
    assert report["_place_frequency_insights"]["count"] == -1
    assert report["_place_frequency_insights"]["status"] == "error"
    assert report["_routine_insights"]["count"] == -1
    assert report["_routine_insights"]["status"] == "error"


@pytest.mark.asyncio
async def test_sufficiency_handles_profile_read_error(insight_engine):
    """Profile-based correlators should show 'error' when get_signal_profile raises."""
    original_get_signal_profile = insight_engine.ums.get_signal_profile

    def broken_profile(profile_type):
        if profile_type == "relationships":
            raise Exception("disk image malformed")
        return original_get_signal_profile(profile_type)

    with patch.object(insight_engine.ums, "get_signal_profile", side_effect=broken_profile):
        report = await insight_engine.get_data_sufficiency_report()

    assert report["_contact_gap_insights"]["status"] == "error"
    assert report["_contact_gap_insights"]["samples"] == -1
    # Other profile-based correlators should still work (only relationships is broken)
    assert report["_communication_style_insights"]["status"] == "no_data"


# =============================================================================
# generate_insights — sufficiency logging
# =============================================================================


@pytest.mark.asyncio
async def test_generate_insights_logs_sufficiency_when_empty(insight_engine, caplog):
    """generate_insights() should log a sufficiency report when 0 insights produced."""
    import logging

    with caplog.at_level(logging.INFO, logger="services.insight_engine.engine"):
        result = await insight_engine.generate_insights()

    assert result == []
    # Check that the sufficiency report was logged
    sufficiency_messages = [r for r in caplog.records if "correlators have sufficient data" in r.message]
    assert len(sufficiency_messages) == 1
    assert "0/" in sufficiency_messages[0].message  # 0 out of N correlators ready
