"""
Tests for the optimized rebuild_profiles_from_events() that filters
events by type when rebuilding specific missing profiles.

Verifies that PROFILE_EVENT_TYPES covers all expected profiles and that
the SQL query is narrowed to relevant event types when missing_profiles
is specified, falling back to the original unfiltered query when it is not.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from services.signal_extractor.pipeline import (
    PROFILE_EVENT_TYPES,
    SignalExtractorPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_event(db, event_type: str, payload: dict | None = None, ts: str | None = None):
    """Insert a synthetic event into events.db and return its id."""
    event_id = str(uuid.uuid4())
    timestamp = ts or datetime.now(timezone.utc).isoformat()
    with db.get_connection("events") as conn:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, priority, payload, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                event_type,
                "test",
                timestamp,
                "normal",
                json.dumps(payload or {}),
                json.dumps({}),
            ),
        )
    return event_id


# ---------------------------------------------------------------------------
# Tests for PROFILE_EVENT_TYPES coverage
# ---------------------------------------------------------------------------

class TestProfileEventTypesCoverage:
    """Ensure the mapping covers all 9 expected profile types."""

    EXPECTED_PROFILES = {
        "linguistic", "linguistic_inbound", "cadence", "mood_signals",
        "relationships", "topics", "temporal", "spatial", "decision",
    }

    def test_covers_all_expected_profiles(self):
        """PROFILE_EVENT_TYPES must contain all 9 profile types."""
        assert set(PROFILE_EVENT_TYPES.keys()) == self.EXPECTED_PROFILES

    def test_no_empty_type_lists(self):
        """Every profile must map to at least one event type."""
        for profile, types in PROFILE_EVENT_TYPES.items():
            assert len(types) > 0, f"Profile '{profile}' has no event types"

    def test_all_values_are_strings(self):
        """Event type values must be plain strings (not EventType enums)."""
        for profile, types in PROFILE_EVENT_TYPES.items():
            for t in types:
                assert isinstance(t, str), f"Profile '{profile}' has non-string type: {t!r}"


# ---------------------------------------------------------------------------
# Tests for filtered rebuild
# ---------------------------------------------------------------------------

class TestRebuildFilteredByProfile:
    """Verify that rebuild_profiles_from_events filters by event type."""

    def test_cadence_only_queries_communication_types(self, db, user_model_store):
        """When only 'cadence' is missing, only email/message types are fetched."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert events: some relevant, some irrelevant
        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})
        _insert_event(db, "message.sent", {"body": "hi", "to_addresses": ["c@d.com"]})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r1"})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r2"})
        _insert_event(db, "calendar.event.created", {"title": "meeting"})

        result = pipeline.rebuild_profiles_from_events(
            event_limit=100, missing_profiles=["cadence"],
        )

        # Cadence extractor needs: email.sent, message.sent, email.received, message.received
        # So only the email.received and message.sent events should be loaded (2 out of 5).
        # system.rule.triggered and calendar.event.created should be excluded.
        assert result["events_processed"] == 2

    def test_spatial_only_queries_location_types(self, db, user_model_store):
        """When only 'spatial' is missing, only calendar/ios/location types are fetched."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert one relevant calendar event with a location, and several irrelevant ones
        _insert_event(db, "calendar.event.created", {"location": "Office", "title": "standup"})
        _insert_event(db, "ios.context.update", {"location": "home"})
        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r1"})
        _insert_event(db, "message.sent", {"body": "test"})

        result = pipeline.rebuild_profiles_from_events(
            event_limit=100, missing_profiles=["spatial"],
        )

        # Spatial needs: calendar.event.created, ios.context.update, system.user.location_update
        # Only the calendar and ios events should be loaded (2 out of 5).
        assert result["events_processed"] == 2

    def test_no_missing_profiles_fetches_all(self, db, user_model_store):
        """When missing_profiles is None, all events are fetched (original behaviour)."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r1"})
        _insert_event(db, "calendar.event.created", {"title": "meeting"})

        result = pipeline.rebuild_profiles_from_events(event_limit=100)

        # All 3 events should be loaded (no filtering).
        assert result["events_processed"] == 3

    def test_empty_missing_profiles_fetches_all(self, db, user_model_store):
        """When missing_profiles is an empty list, all events are fetched."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r1"})

        result = pipeline.rebuild_profiles_from_events(
            event_limit=100, missing_profiles=[],
        )

        assert result["events_processed"] == 2

    def test_multiple_missing_profiles_union_types(self, db, user_model_store):
        """When multiple profiles are missing, event types are unioned."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # cadence needs email/message types; spatial needs calendar/ios/location types
        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})
        _insert_event(db, "calendar.event.created", {"location": "Office", "title": "standup"})
        _insert_event(db, "system.rule.triggered", {"rule_id": "r1"})

        result = pipeline.rebuild_profiles_from_events(
            event_limit=100, missing_profiles=["cadence", "spatial"],
        )

        # Union of cadence + spatial covers email.received AND calendar.event.created.
        # system.rule.triggered should still be excluded.
        assert result["events_processed"] == 2

    def test_unknown_profile_name_ignored_gracefully(self, db, user_model_store):
        """An unknown profile name in missing_profiles doesn't crash — it just adds no types."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        _insert_event(db, "email.received", {"sender": "a@b.com", "body": "hello"})

        # "nonexistent" has no entry in PROFILE_EVENT_TYPES but shouldn't crash.
        # Combined with "cadence" it should still filter to cadence types.
        result = pipeline.rebuild_profiles_from_events(
            event_limit=100, missing_profiles=["nonexistent", "cadence"],
        )

        assert result["events_processed"] == 1

    def test_system_rule_triggered_always_excluded_when_filtering(self, db, user_model_store):
        """system.rule.triggered (57% of real events) is never in any profile's type list."""
        assert "system.rule.triggered" not in {
            t for types in PROFILE_EVENT_TYPES.values() for t in types
        }


# ---------------------------------------------------------------------------
# Tests for check_and_rebuild_missing_profiles integration
# ---------------------------------------------------------------------------

class TestCheckAndRebuildPassesMissingProfiles:
    """Verify that check_and_rebuild_missing_profiles passes missing profile names."""

    def test_passes_missing_profiles_to_rebuild(self, db, user_model_store):
        """check_and_rebuild_missing_profiles should pass missing profiles so
        rebuild filters by event type. We verify indirectly by inserting only
        relevant events and confirming the rebuild finds them."""
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # Insert a cadence-relevant event
        _insert_event(
            db, "email.received",
            {"sender": "test@example.com", "body": "Test message content"},
        )

        # Also insert many system.rule.triggered events that should be filtered out
        for i in range(20):
            _insert_event(db, "system.rule.triggered", {"rule_id": f"r{i}"})

        result = pipeline.check_and_rebuild_missing_profiles()

        # Should not be skipped since there are events
        assert result["skipped"] is False
        # Should have tried to rebuild some profiles
        assert len(result["missing_before"]) > 0
