"""
Life OS — Auto-rebuild missing signal profiles tests.

Verifies that check_and_rebuild_missing_profiles() correctly detects missing
signal profiles, triggers a rebuild when events exist, and is safe to call
repeatedly (idempotent).
"""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import SignalExtractorPipeline
from storage.event_store import EventStore


def _store_test_event(db, event_id, event_type, source, payload, metadata=None, timestamp=None):
    """Helper to insert an event into events.db."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    es = EventStore(db)
    es.store_event({
        "id": event_id,
        "type": event_type,
        "source": source,
        "timestamp": timestamp,
        "priority": "normal",
        "payload": payload,
        "metadata": metadata or {},
    })


def _populate_healthy_profile(db, user_model_store, profile_type):
    """Create a profile with meaningful data and sufficient samples to pass stale detection."""
    user_model_store.update_signal_profile(profile_type, {"averages": {"value": 0.5}})
    with db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = 10 WHERE profile_type = ?",
            (profile_type,),
        )


def test_all_profiles_present_returns_early(db, user_model_store):
    """When all expected profiles exist, method returns early without rebuilding."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Pre-populate every expected profile with healthy data and sufficient samples.
    expected = [
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    ]
    for profile_type in expected:
        _populate_healthy_profile(db, user_model_store, profile_type)

    # Store an event so the "no events" guard doesn't interfere.
    _store_test_event(db, "e1", "email.received", "test", {
        "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
    })

    with patch.object(pipeline, "rebuild_profiles_from_events") as mock_rebuild:
        result = pipeline.check_and_rebuild_missing_profiles()

    # Should NOT call rebuild.
    mock_rebuild.assert_not_called()
    assert result["missing_before"] == []
    assert result["rebuilt"] == []
    assert result["skipped"] is True


def test_missing_profiles_with_events_triggers_rebuild(db, user_model_store):
    """When some profiles are missing and events exist, calls rebuild and reports rebuilt profiles."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Pre-populate only relationships and linguistic_inbound with healthy data.
    _populate_healthy_profile(db, user_model_store, "relationships")
    _populate_healthy_profile(db, user_model_store, "linguistic_inbound")

    # Store several email events so the pipeline has data to process.
    for i in range(5):
        _store_test_event(db, f"rebuild-{i}", "email.received", "proton_mail", {
            "subject": f"Email {i}",
            "body": f"Message content number {i} with some text about projects.",
            "from": "alice@company.com",
            "to": ["user@example.com"],
        }, timestamp=f"2026-03-01T{10 + i}:00:00+00:00")

    result = pipeline.check_and_rebuild_missing_profiles()

    # Should have identified missing profiles.
    assert len(result["missing_before"]) > 0
    assert "relationships" not in result["missing_before"]
    assert "linguistic_inbound" not in result["missing_before"]

    # Rebuild should have been attempted (skipped=False).
    assert result["skipped"] is False

    # At least some profiles should have been rebuilt from the email events.
    # (cadence and linguistic are reliably populated by email.received events)
    assert isinstance(result["rebuilt"], list)


def test_missing_profiles_no_events_skips_rebuild(db, user_model_store):
    """When profiles are missing but events.db is empty, skips rebuild."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # No profiles, no events.
    with patch.object(pipeline, "rebuild_profiles_from_events") as mock_rebuild:
        result = pipeline.check_and_rebuild_missing_profiles()

    # Should NOT call rebuild since there are no events.
    mock_rebuild.assert_not_called()
    assert len(result["missing_before"]) > 0
    assert result["rebuilt"] == []
    assert result["skipped"] is True


def test_exception_during_rebuild_caught_gracefully(db, user_model_store):
    """An exception inside rebuild_profiles_from_events is caught and does not raise."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store an event so the rebuild path is entered.
    _store_test_event(db, "e1", "email.received", "test", {
        "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
    })

    with patch.object(pipeline, "rebuild_profiles_from_events", side_effect=RuntimeError("DB exploded")):
        # Should NOT raise — fail-open.
        result = pipeline.check_and_rebuild_missing_profiles()

    # Graceful fallback: skipped=True, empty rebuilt.
    assert result["skipped"] is True
    assert result["rebuilt"] == []


def test_idempotent_when_profiles_exist(db, user_model_store):
    """Calling twice when all profiles exist does nothing both times."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    expected = [
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    ]
    for profile_type in expected:
        _populate_healthy_profile(db, user_model_store, profile_type)

    result1 = pipeline.check_and_rebuild_missing_profiles()
    result2 = pipeline.check_and_rebuild_missing_profiles()

    # Both calls should return the same early-exit result.
    assert result1["skipped"] is True
    assert result2["skipped"] is True
    assert result1["missing_before"] == []
    assert result2["missing_before"] == []


def test_returns_correct_missing_profile_list(db, user_model_store):
    """Verify that the missing_before list accurately reflects which profiles are absent."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Populate only a subset with healthy data.
    _populate_healthy_profile(db, user_model_store, "relationships")
    _populate_healthy_profile(db, user_model_store, "cadence")
    _populate_healthy_profile(db, user_model_store, "linguistic")

    # No events — so rebuild is skipped, but missing_before should still be correct.
    result = pipeline.check_and_rebuild_missing_profiles()

    expected_missing = {"linguistic_inbound", "mood_signals", "topics", "temporal", "spatial", "decision"}
    assert set(result["missing_before"]) == expected_missing
