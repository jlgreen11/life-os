"""
Tests for prediction engine diagnostics resilience to corrupted user_model.db.

Verifies that get_diagnostics() does not crash when the user model store
raises database errors (e.g. sqlite3.DatabaseError from a corrupted DB),
and that the data_sources section correctly reports the health of each
signal profile the engine depends on.
"""

import sqlite3
from unittest.mock import patch

import pytest

from services.prediction_engine.engine import PredictionEngine


@pytest.mark.asyncio
async def test_diagnostics_survives_corrupted_relationships_profile(db, user_model_store):
    """get_diagnostics() returns a valid result (not crash) when
    ums.get_signal_profile raises sqlite3.DatabaseError."""
    engine = PredictionEngine(db, user_model_store)

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        diagnostics = await engine.get_diagnostics()

    # Must not crash — should return a valid diagnostics dict
    assert "prediction_types" in diagnostics
    assert "overall" in diagnostics
    assert "data_sources" in diagnostics

    # Opportunity section should gracefully degrade to no contacts
    opportunity = diagnostics["prediction_types"]["opportunity"]
    assert opportunity["data_available"]["total_contacts"] == 0
    assert opportunity["data_available"]["eligible_contacts"] == 0


@pytest.mark.asyncio
async def test_data_sources_reports_error_for_inaccessible_profiles(db, user_model_store):
    """data_sources section correctly reports 'error' status for
    profiles that raise exceptions."""
    engine = PredictionEngine(db, user_model_store)

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        diagnostics = await engine.get_diagnostics()

    data_sources = diagnostics["data_sources"]
    for profile_name in ["relationships", "cadence", "mood_signals", "linguistic", "topics"]:
        assert data_sources[profile_name]["status"] == "error"
        assert "malformed" in data_sources[profile_name]["error"]


@pytest.mark.asyncio
async def test_data_sources_reports_available_for_accessible_profiles(db, user_model_store):
    """data_sources section reports 'available' for profiles that
    return data successfully."""
    # Seed a relationships profile so at least one is accessible
    user_model_store.update_signal_profile("relationships", {
        "contacts": {"alice@example.com": {"interaction_count": 5}},
    })

    engine = PredictionEngine(db, user_model_store)
    diagnostics = await engine.get_diagnostics()

    data_sources = diagnostics["data_sources"]
    assert data_sources["relationships"]["status"] == "available"


@pytest.mark.asyncio
async def test_data_sources_reports_empty_for_none_profiles(db, user_model_store):
    """data_sources section reports 'empty' for profiles that return None."""
    engine = PredictionEngine(db, user_model_store)

    # Without seeding any profiles, all should be empty (not error)
    diagnostics = await engine.get_diagnostics()

    data_sources = diagnostics["data_sources"]
    for profile_name in ["relationships", "cadence", "mood_signals", "linguistic", "topics"]:
        assert data_sources[profile_name]["status"] == "empty"
        assert data_sources[profile_name]["samples"] == 0


@pytest.mark.asyncio
async def test_overall_health_degraded_when_data_sources_have_errors(db, user_model_store):
    """Overall health is 'degraded' or 'broken' when data sources have errors."""
    engine = PredictionEngine(db, user_model_store)

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        diagnostics = await engine.get_diagnostics()

    overall = diagnostics["overall"]
    # Health must not be 'healthy' when data sources are erroring
    assert overall["health"] in ("degraded", "broken")
    # Overall blockers should mention corruption
    assert any("corrupted" in b for b in overall["blockers"])


@pytest.mark.asyncio
async def test_overall_blockers_list_failed_profiles(db, user_model_store):
    """Overall blockers mention which specific profiles failed."""
    engine = PredictionEngine(db, user_model_store)

    with patch.object(
        user_model_store,
        "get_signal_profile",
        side_effect=sqlite3.DatabaseError("database disk image is malformed"),
    ):
        diagnostics = await engine.get_diagnostics()

    blockers_text = " ".join(diagnostics["overall"]["blockers"])
    # All profiles should be listed since all raise errors
    assert "relationships" in blockers_text
    assert "cadence" in blockers_text


@pytest.mark.asyncio
async def test_partial_corruption_only_some_profiles_error(db, user_model_store):
    """When only some profiles error, data_sources shows a mix of statuses."""
    engine = PredictionEngine(db, user_model_store)

    original_get = user_model_store.get_signal_profile

    def selective_error(profile_name):
        """Raise only for 'relationships', return normally for others."""
        if profile_name == "relationships":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original_get(profile_name)

    with patch.object(user_model_store, "get_signal_profile", side_effect=selective_error):
        diagnostics = await engine.get_diagnostics()

    data_sources = diagnostics["data_sources"]
    assert data_sources["relationships"]["status"] == "error"
    # Other profiles should be empty (no data seeded), not error
    assert data_sources["cadence"]["status"] == "empty"
    assert data_sources["linguistic"]["status"] == "empty"
