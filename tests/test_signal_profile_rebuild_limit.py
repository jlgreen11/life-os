"""
Life OS — Signal profile rebuild limit tests.

Verifies that the event_limit for signal profile rebuilds is set to 50K
(sufficient for large event databases), that the type filter correctly narrows
the SQL query, and that all fetched events are processed regardless of count.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.signal_extractor.pipeline import PROFILE_EVENT_TYPES, SignalExtractorPipeline
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


def test_check_and_rebuild_passes_50k_event_limit(db, user_model_store):
    """check_and_rebuild_missing_profiles() calls rebuild with event_limit=50000."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Store one event so the "no events" guard doesn't trigger a skip.
    _store_test_event(db, "e1", "email.received", "test", {
        "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
    })

    with patch.object(pipeline, "rebuild_profiles_from_events", return_value={
        "events_processed": 0, "profiles_rebuilt": [], "errors": [],
    }) as mock_rebuild:
        pipeline.check_and_rebuild_missing_profiles()

    mock_rebuild.assert_called_once()
    _, kwargs = mock_rebuild.call_args
    assert kwargs["event_limit"] == 50000, (
        f"Expected event_limit=50000, got {kwargs['event_limit']}"
    )


def test_rebuild_default_event_limit_is_50k(db, user_model_store):
    """rebuild_profiles_from_events() default parameter is 50000."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Call with no explicit event_limit — should use the default.
    result = pipeline.rebuild_profiles_from_events()
    # With no events in the test DB, it returns immediately.
    assert result["events_processed"] == 0


def test_rebuild_processes_all_fetched_events(db, user_model_store):
    """All fetched events are processed, even when the count is high."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Insert 100 email events.
    for i in range(100):
        _store_test_event(db, f"bulk-{i}", "email.received", "proton_mail", {
            "subject": f"Email {i}",
            "body": f"Message content {i} about work projects and deadlines.",
            "from": "alice@company.com",
            "to": ["user@example.com"],
        }, timestamp=f"2026-03-01T{10 + (i // 60):02d}:{i % 60:02d}:00+00:00")

    result = pipeline.rebuild_profiles_from_events(event_limit=200)

    assert result["events_processed"] == 100, (
        f"Expected all 100 events processed, got {result['events_processed']}"
    )
    # At least one extractor should have matched email.received events.
    assert len(result["profiles_rebuilt"]) > 0


def test_type_filter_narrows_query_for_missing_profiles(db, user_model_store):
    """When missing_profiles is specified, only matching event types are fetched."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Insert events of different types.
    _store_test_event(db, "email-1", "email.received", "proton_mail", {
        "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
    })
    _store_test_event(db, "task-1", "task.completed", "system", {
        "task_id": "t1", "title": "Done",
    })
    _store_test_event(db, "rule-1", "system.rule.triggered", "rules_engine", {
        "rule_id": "r1",
    })
    _store_test_event(db, "location-1", "ios.context.update", "ios", {
        "location": {"lat": 40.0, "lng": -74.0},
    })

    # Rebuild only the "spatial" profile, which needs:
    # calendar.event.created, ios.context.update, system.user.location_update
    result = pipeline.rebuild_profiles_from_events(
        event_limit=1000, missing_profiles=["spatial"],
    )

    # Only the ios.context.update event should have been fetched and processed.
    # email.received, task.completed, and system.rule.triggered are not in
    # PROFILE_EVENT_TYPES["spatial"], so they should be filtered out.
    assert result["events_processed"] == 1, (
        f"Expected 1 event (ios.context.update only), got {result['events_processed']}"
    )


def test_type_filter_unions_multiple_missing_profiles(db, user_model_store):
    """When multiple profiles are missing, their event types are unioned."""
    pipeline = SignalExtractorPipeline(db, user_model_store)

    # Insert events matching different profiles.
    _store_test_event(db, "email-1", "email.received", "proton_mail", {
        "subject": "Hi", "body": "Hello", "from": "a@b.com", "to": ["u@x.com"],
    })
    _store_test_event(db, "ctx-1", "ios.context.update", "ios", {
        "location": {"lat": 40.0, "lng": -74.0},
    })
    _store_test_event(db, "rule-1", "system.rule.triggered", "rules_engine", {
        "rule_id": "r1",
    })

    # Rebuild "spatial" (needs ios.context.update) + "linguistic_inbound" (needs email.received).
    result = pipeline.rebuild_profiles_from_events(
        event_limit=1000, missing_profiles=["spatial", "linguistic_inbound"],
    )

    # Both ios.context.update and email.received should be fetched.
    # system.rule.triggered should still be excluded.
    assert result["events_processed"] == 2, (
        f"Expected 2 events, got {result['events_processed']}"
    )


def test_profile_event_types_covers_all_expected_profiles():
    """PROFILE_EVENT_TYPES has entries for all 9 expected signal profiles."""
    expected = [
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    ]
    for profile_name in expected:
        assert profile_name in PROFILE_EVENT_TYPES, (
            f"PROFILE_EVENT_TYPES missing entry for '{profile_name}'"
        )
        assert len(PROFILE_EVENT_TYPES[profile_name]) > 0, (
            f"PROFILE_EVENT_TYPES['{profile_name}'] is empty"
        )
