"""Tests for SignalExtractorPipeline.get_diagnostics()."""

import pytest

from services.signal_extractor.pipeline import PROFILE_EVENT_TYPES, SignalExtractorPipeline
from storage.user_model_store import UserModelStore


@pytest.fixture()
def ums(db):
    """A UserModelStore without event bus — sufficient for pipeline tests."""
    return UserModelStore(db)


@pytest.fixture()
def pipeline(db, ums):
    """A SignalExtractorPipeline wired to a fresh temporary database."""
    return SignalExtractorPipeline(db, ums)


def test_diagnostics_returns_expected_keys(pipeline):
    """get_diagnostics() must return all top-level keys."""
    diag = pipeline.get_diagnostics()
    assert "profiles" in diag
    assert "profiles_present" in diag
    assert "profiles_missing" in diag
    assert "extractors" in diag
    assert "extractor_count" in diag
    assert "health" in diag
    assert "available_event_types" in diag
    assert "rebuild_feasibility" in diag


def test_diagnostics_shows_all_profiles_missing_on_fresh_db(pipeline):
    """On a fresh database with no signal profiles, all 9 should be missing."""
    diag = pipeline.get_diagnostics()
    assert diag["profiles_missing"] == 9
    assert diag["profiles_present"] == 0
    assert diag["health"] in ("degraded", "partial")


def test_diagnostics_shows_profile_present_after_update(ums, pipeline):
    """After updating a signal profile, diagnostics should reflect it."""
    ums.update_signal_profile("linguistic_inbound", {
        "per_contact_averages": {},
        "total_samples": 10,
    })
    diag = pipeline.get_diagnostics()
    assert diag["profiles"]["linguistic_inbound"]["status"] == "ok"
    assert diag["profiles_present"] >= 1
    assert diag["profiles_missing"] == 8


def test_diagnostics_health_partial_with_some_profiles(ums, pipeline):
    """Health should be 'partial' when at least 2 profiles exist but some are missing."""
    ums.update_signal_profile("linguistic", {"data": "test"})
    ums.update_signal_profile("cadence", {"data": "test"})
    diag = pipeline.get_diagnostics()
    assert diag["profiles_present"] >= 2
    assert diag["health"] == "partial"


def test_diagnostics_health_ok_with_all_profiles(ums, pipeline):
    """Health should be 'ok' when all 9 profiles exist."""
    expected = [
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    ]
    for pt in expected:
        ums.update_signal_profile(pt, {"data": "test"})
    diag = pipeline.get_diagnostics()
    assert diag["profiles_present"] == 9
    assert diag["profiles_missing"] == 0
    assert diag["health"] == "ok"


def test_diagnostics_rebuild_feasibility_with_events(db, ums, pipeline):
    """Rebuild feasibility should flag rebuildable profiles when matching events exist."""
    # Insert an event whose type matches linguistic_inbound's PROFILE_EVENT_TYPES
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            ("test-diag-1", "email.received", "test", "normal", "{}"),
        )
    diag = pipeline.get_diagnostics()
    feasibility = diag.get("rebuild_feasibility", {})
    # linguistic_inbound needs email.received, so it should be rebuildable
    assert "linguistic_inbound" in feasibility
    assert feasibility["linguistic_inbound"]["can_rebuild"] is True
    assert feasibility["linguistic_inbound"]["available_events"] >= 1


def test_diagnostics_rebuild_feasibility_no_events(pipeline):
    """With no events, no missing profiles should be marked as rebuildable."""
    diag = pipeline.get_diagnostics()
    feasibility = diag.get("rebuild_feasibility", {})
    for pt, info in feasibility.items():
        assert info["can_rebuild"] is False
        assert info["available_events"] == 0


def test_diagnostics_lists_all_extractors(pipeline):
    """Should list all 8 registered extractors."""
    diag = pipeline.get_diagnostics()
    assert diag["extractor_count"] == 8
    assert isinstance(diag["extractors"], list)
    assert len(diag["extractors"]) == 8
    # Verify some known extractor names are present
    assert "LinguisticExtractor" in diag["extractors"]
    assert "CadenceExtractor" in diag["extractors"]
    assert "MoodInferenceEngine" in diag["extractors"]


def test_diagnostics_available_event_types_excludes_test_events(db, pipeline):
    """Events with type starting with 'test' should be excluded from event type counts."""
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            ("test-diag-2", "test.something", "test", "normal", "{}"),
        )
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            ("test-diag-3", "email.sent", "test", "normal", "{}"),
        )
    diag = pipeline.get_diagnostics()
    event_types = diag["available_event_types"]
    assert "test.something" not in event_types
    assert "email.sent" in event_types
