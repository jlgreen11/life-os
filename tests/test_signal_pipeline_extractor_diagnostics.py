"""
Tests for per-extractor hit/error counting and diagnostics in SignalExtractorPipeline.

Covers the additions made to expose extractor-level observability:
- _extractor_hit_counts incremented in process_event()
- _extractor_error_counts incremented on extract() failures
- get_extractor_diagnostics() returning the full diagnostics dict
- get_profile_health() annotating missing/stale profiles with extractor_hits
- rebuild_profiles_from_events() including extractor_event_counts in its result
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import (
    _PROFILE_TO_EXTRACTOR,
    SignalExtractorPipeline,
)
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ums(db):
    """UserModelStore without an event bus — sufficient for pipeline tests."""
    return UserModelStore(db)


@pytest.fixture()
def pipeline(db, ums):
    """A SignalExtractorPipeline wired to a fresh temporary database."""
    return SignalExtractorPipeline(db, ums)


def _make_event(event_type: str, event_id: str = "test-ev-1") -> dict:
    """Build a minimal event dict with required envelope fields."""
    return {
        "id": event_id,
        "type": event_type,
        "source": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority": "normal",
        "payload": {
            "subject": "Test event",
            "body": "Hello world",
            "from": "alice@example.com",
            "to": ["bob@example.com"],
        },
        "metadata": {},
    }


def _insert_event(db, event_id: str, event_type: str) -> None:
    """Insert a minimal event row into events.db for rebuild tests."""
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            (event_id, event_type, "test", "normal", "{}"),
        )


# ---------------------------------------------------------------------------
# Tests: initial state
# ---------------------------------------------------------------------------


def test_extractor_hit_counts_empty_on_init(pipeline):
    """Hit counts should start empty — no events have been processed yet."""
    assert pipeline._extractor_hit_counts == {}


def test_extractor_error_counts_empty_on_init(pipeline):
    """Error counts should start empty."""
    assert pipeline._extractor_error_counts == {}


def test_extractor_last_hit_ts_empty_on_init(pipeline):
    """Timestamp dict should start empty."""
    assert pipeline._extractor_last_hit_ts == {}


# ---------------------------------------------------------------------------
# Tests: process_event() — hit count increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_event_increments_hit_count_for_matching_extractor(pipeline):
    """Sending an email.received event should increment hit counts for
    extractors that handle it (e.g. CadenceExtractor, MoodInferenceEngine,
    RelationshipExtractor, TopicExtractor, LinguisticExtractor)."""
    event = _make_event("email.received")
    await pipeline.process_event(event)

    # At least one extractor must have been hit.
    assert sum(pipeline._extractor_hit_counts.values()) >= 1

    # CadenceExtractor handles email.received — verify it was counted.
    assert pipeline._extractor_hit_counts.get("CadenceExtractor", 0) >= 1


@pytest.mark.asyncio
async def test_process_event_accumulates_hit_count_across_calls(pipeline):
    """Processing multiple events should accumulate counts, not reset them."""
    event_a = _make_event("email.received", "ev-a")
    event_b = _make_event("email.received", "ev-b")
    event_c = _make_event("email.received", "ev-c")

    await pipeline.process_event(event_a)
    await pipeline.process_event(event_b)
    await pipeline.process_event(event_c)

    # CadenceExtractor handles email.received — should have 3 hits.
    assert pipeline._extractor_hit_counts.get("CadenceExtractor", 0) >= 3


@pytest.mark.asyncio
async def test_process_event_no_hit_for_unmatched_event_type(pipeline):
    """An event type that no extractor handles should leave hit counts empty."""
    # system.rule.triggered is not handled by any extractor.
    event = _make_event("system.rule.triggered")
    await pipeline.process_event(event)

    # No extractor should have been hit.
    assert sum(pipeline._extractor_hit_counts.values()) == 0


@pytest.mark.asyncio
async def test_process_event_updates_last_hit_ts_on_success(pipeline):
    """After a successful extract() call the extractor's last hit timestamp
    should be populated."""
    event = _make_event("email.received")
    await pipeline.process_event(event)

    # At least the extractors that handle email.received should have timestamps.
    assert len(pipeline._extractor_last_hit_ts) >= 1
    for ts in pipeline._extractor_last_hit_ts.values():
        assert isinstance(ts, float)
        assert ts > 0


# ---------------------------------------------------------------------------
# Tests: process_event() — error count increments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_event_increments_error_count_on_extract_exception(pipeline):
    """When extract() raises, the error counter for that extractor should
    increment but the pipeline should not crash (fail-open)."""
    event = _make_event("email.received")

    # Patch CadenceExtractor.extract to raise an exception.
    cadence = next(e for e in pipeline.extractors if type(e).__name__ == "CadenceExtractor")
    with patch.object(cadence, "extract", side_effect=RuntimeError("boom")):
        signals = await pipeline.process_event(event)

    # Pipeline returned normally (fail-open).
    assert isinstance(signals, list)

    # Error count for CadenceExtractor should now be 1.
    assert pipeline._extractor_error_counts.get("CadenceExtractor", 0) >= 1


@pytest.mark.asyncio
async def test_process_event_hit_count_incremented_even_when_extract_fails(pipeline):
    """Hit count should be incremented (can_process returned True) even if
    extract() subsequently raises — the extractor did 'receive' the event."""
    event = _make_event("email.received")

    cadence = next(e for e in pipeline.extractors if type(e).__name__ == "CadenceExtractor")
    with patch.object(cadence, "extract", side_effect=RuntimeError("oops")):
        await pipeline.process_event(event)

    # Hit count should have been incremented before extract() was called.
    assert pipeline._extractor_hit_counts.get("CadenceExtractor", 0) >= 1


# ---------------------------------------------------------------------------
# Tests: get_extractor_diagnostics()
# ---------------------------------------------------------------------------


def test_get_extractor_diagnostics_returns_expected_keys(pipeline):
    """get_extractor_diagnostics() must contain all required top-level keys."""
    diag = pipeline.get_extractor_diagnostics()

    assert "extractor_hit_counts" in diag
    assert "extractor_error_counts" in diag
    assert "extractor_last_hit_ts" in diag
    assert "profile_to_extractor" in diag
    assert "registered_extractors" in diag


def test_get_extractor_diagnostics_all_empty_on_fresh_pipeline(pipeline):
    """Before any events are processed the count dicts should be empty."""
    diag = pipeline.get_extractor_diagnostics()

    assert diag["extractor_hit_counts"] == {}
    assert diag["extractor_error_counts"] == {}
    assert diag["extractor_last_hit_ts"] == {}


def test_get_extractor_diagnostics_includes_all_registered_extractors(pipeline):
    """registered_extractors should list all 8 extractors."""
    diag = pipeline.get_extractor_diagnostics()

    assert len(diag["registered_extractors"]) == 8
    for name in [
        "LinguisticExtractor", "CadenceExtractor", "MoodInferenceEngine",
        "RelationshipExtractor", "TopicExtractor", "TemporalExtractor",
        "SpatialExtractor", "DecisionExtractor",
    ]:
        assert name in diag["registered_extractors"]


def test_get_extractor_diagnostics_profile_to_extractor_matches_module_constant(pipeline):
    """profile_to_extractor should be equal to the module-level constant."""
    diag = pipeline.get_extractor_diagnostics()
    assert diag["profile_to_extractor"] == _PROFILE_TO_EXTRACTOR


@pytest.mark.asyncio
async def test_get_extractor_diagnostics_reflects_hit_counts_after_processing(pipeline):
    """After processing events, get_extractor_diagnostics() should reflect the
    accumulated hit counts."""
    for i in range(5):
        await pipeline.process_event(_make_event("email.received", f"ev-{i}"))

    diag = pipeline.get_extractor_diagnostics()
    # CadenceExtractor handles email.received — must appear in hit counts.
    assert diag["extractor_hit_counts"].get("CadenceExtractor", 0) >= 5


@pytest.mark.asyncio
async def test_get_extractor_diagnostics_reflects_error_counts_after_failures(pipeline):
    """After an extract() failure, get_extractor_diagnostics() should show the
    error count for that extractor."""
    cadence = next(e for e in pipeline.extractors if type(e).__name__ == "CadenceExtractor")
    with patch.object(cadence, "extract", side_effect=RuntimeError("test error")):
        await pipeline.process_event(_make_event("email.received"))

    diag = pipeline.get_extractor_diagnostics()
    assert diag["extractor_error_counts"].get("CadenceExtractor", 0) >= 1


# ---------------------------------------------------------------------------
# Tests: get_profile_health() — extractor_hits annotation
# ---------------------------------------------------------------------------


def test_get_profile_health_missing_profile_includes_extractor_hits_field(pipeline):
    """Missing profiles should include an 'extractor_hits' key in their entry."""
    health = pipeline.get_profile_health()

    # All profiles should be missing on a fresh DB.
    for ptype, info in health.items():
        if info["status"] == "missing":
            assert "extractor_hits" in info, (
                f"Missing profile '{ptype}' does not include 'extractor_hits'"
            )


def test_get_profile_health_missing_profile_extractor_hits_zero_without_events(pipeline):
    """With no events processed, extractor_hits for all missing profiles should be 0."""
    health = pipeline.get_profile_health()

    for ptype, info in health.items():
        if info["status"] == "missing":
            assert info["extractor_hits"] == 0, (
                f"Expected 0 extractor_hits for '{ptype}' before any events processed"
            )


@pytest.mark.asyncio
async def test_get_profile_health_missing_profile_shows_nonzero_extractor_hits_after_events(pipeline):
    """After processing events, a missing profile should report the hit count
    of its corresponding extractor(s), revealing the persistence failure."""
    # Process email.received events — CadenceExtractor handles these and
    # should write the 'cadence' profile.
    for i in range(3):
        await pipeline.process_event(_make_event("email.received", f"ev-{i}"))

    health = pipeline.get_profile_health()
    cadence_info = health.get("cadence", {})

    # The cadence profile may be missing (no real persistence in this unit test)
    # or populated; in either case if it's missing, extractor_hits > 0 confirms
    # CadenceExtractor did receive the events.
    if cadence_info.get("status") in ("missing", "stale"):
        assert cadence_info.get("extractor_hits", 0) >= 1


def test_get_profile_health_stale_profile_includes_extractor_hits(db, ums, pipeline):
    """Stale profiles (data present but below quality threshold) should also
    include an 'extractor_hits' key."""
    # Insert a profile row with only 1 sample — will be detected as stale.
    ums.update_signal_profile("cadence", {"averages": {}})
    # Leave samples_count at default (typically 1 after single update call)

    health = pipeline.get_profile_health()
    cadence_info = health.get("cadence", {})

    # If the profile is stale, it must include extractor_hits.
    if cadence_info.get("status") == "stale":
        assert "extractor_hits" in cadence_info


def test_get_profile_health_ok_profile_does_not_include_extractor_hits(db, ums, pipeline):
    """Healthy (ok) profiles should NOT include extractor_hits — it's only
    relevant when diagnosing missing or stale profiles."""
    # Populate a healthy profile.
    ums.update_signal_profile("cadence", {"averages": {"response_time": 3.5}})
    with db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = 10 WHERE profile_type = ?",
            ("cadence",),
        )

    health = pipeline.get_profile_health()
    cadence_info = health.get("cadence", {})

    assert cadence_info.get("status") == "ok"
    # OK profiles don't need the extractor_hits annotation.
    assert "extractor_hits" not in cadence_info


# ---------------------------------------------------------------------------
# Tests: rebuild_profiles_from_events() — extractor_event_counts in result
# ---------------------------------------------------------------------------


def test_rebuild_profiles_includes_extractor_event_counts_key(db, pipeline):
    """rebuild_profiles_from_events() result must contain 'extractor_event_counts'."""
    _insert_event(db, "rb-ev-1", "email.received")

    result = pipeline.rebuild_profiles_from_events(event_limit=100)

    assert "extractor_event_counts" in result


def test_rebuild_profiles_extractor_event_counts_empty_when_no_events(pipeline):
    """With no events in events.db the extractor_event_counts should be empty."""
    result = pipeline.rebuild_profiles_from_events(event_limit=100)

    assert result["extractor_event_counts"] == {}


def test_rebuild_profiles_extractor_event_counts_populated_for_matching_events(db, pipeline):
    """When qualifying events exist, extractor_event_counts should show which
    extractors processed them."""
    for i in range(3):
        _insert_event(db, f"rb-ev-{i}", "email.received")

    result = pipeline.rebuild_profiles_from_events(event_limit=100)

    counts = result["extractor_event_counts"]
    # email.received is handled by CadenceExtractor among others.
    assert counts.get("CadenceExtractor", 0) >= 3


def test_rebuild_profiles_extractor_event_counts_stored_in_last_rebuild_result(db, pipeline):
    """extractor_event_counts should be accessible via get_rebuild_diagnostics()
    after a rebuild, since it is stored in _last_rebuild_result."""
    _insert_event(db, "rb-diag-ev", "email.received")

    pipeline.rebuild_profiles_from_events(event_limit=100)
    diag = pipeline.get_rebuild_diagnostics()

    assert "extractor_event_counts" in diag


def test_rebuild_profiles_extractor_event_counts_type_is_dict(db, pipeline):
    """extractor_event_counts must be a plain dict (not defaultdict or similar)
    so it serialises cleanly to JSON in API responses."""
    _insert_event(db, "rb-type-ev", "email.received")
    result = pipeline.rebuild_profiles_from_events(event_limit=100)
    counts = result["extractor_event_counts"]

    assert isinstance(counts, dict)
    # Values must be plain ints.
    for k, v in counts.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
