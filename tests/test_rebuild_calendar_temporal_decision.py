"""
Life OS — Integration tests for signal profile rebuild from calendar events.

Verifies that rebuild_profiles_from_events() and check_and_rebuild_missing_profiles()
correctly produce temporal, decision, spatial, and linguistic profiles when
events.db contains calendar.event.created and email.sent events.

These tests exercise the full rebuild pipeline end-to-end without mocking,
ensuring each extractor processes events and persists non-stale profiles.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.signal_extractor.pipeline import SignalExtractorPipeline
from storage.event_store import EventStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_event(db, event_id, event_type, source, payload, timestamp=None):
    """Insert a single event into events.db via EventStore."""
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
        "metadata": {},
    })


def _insert_calendar_events(db, count=55):
    """Insert realistic calendar.event.created events spread across hours and days.

    Creates events with start_time in the future relative to the event timestamp
    so that the DecisionExtractor's commitment_pattern logic accepts them
    (planning_horizon > 0).

    Args:
        db: DatabaseManager instance.
        count: Number of calendar events to create.
    """
    base = datetime(2026, 2, 1, 8, 0, 0, tzinfo=timezone.utc)
    locations = [
        "Conference Room A",
        "123 Main St, Austin, TX",
        "Zoom Call",
        "",  # some events without location
        "Coffee Shop Downtown",
        "Home Office",
        "",
        "Gym - Central Fitness",
    ]
    summaries = [
        "Team standup", "Design review", "Lunch with Alice",
        "Sprint planning", "Doctor appointment", "Coffee chat",
        "Budget review meeting", "Gym workout", "1:1 with Bob",
        "Project sync call",
    ]

    for i in range(count):
        # Spread events across different hours (8-20) and days
        event_ts = base + timedelta(days=i % 14, hours=(i * 3) % 12)
        # Schedule the event 1-7 days in the future from creation time
        start_offset_hours = 24 + (i % 7) * 24
        start_time = event_ts + timedelta(hours=start_offset_hours)
        end_time = start_time + timedelta(hours=1)

        _store_event(
            db,
            event_id=f"cal-{i:04d}",
            event_type="calendar.event.created",
            source="caldav",
            payload={
                "summary": summaries[i % len(summaries)],
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "location": locations[i % len(locations)],
                "attendees": ["colleague@work.com"] if i % 3 == 0 else [],
            },
            timestamp=event_ts.isoformat(),
        )


def _insert_email_sent_events(db, count=12):
    """Insert email.sent events with realistic body content.

    Args:
        db: DatabaseManager instance.
        count: Number of email events to create.
    """
    base = datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
    bodies = [
        "Hi team, I think we should proceed with option A for the migration.",
        "Thanks for the update. Let me review the proposal and get back to you.",
        "Can you send over the latest financials? I need them for the board meeting.",
        "Great work on the release! Everything looks good on my end.",
        "I'll handle the client call tomorrow morning. No need to reschedule.",
        "What do you think about the new design? Should I go with the blue version?",
        "Please find the attached report. Let me know if you have questions.",
        "I've decided to go with the vendor from Chicago. Their pricing is better.",
        "Could you take care of the deployment? I trust your judgment on the rollback plan.",
        "Meeting confirmed for Thursday. I'll prepare the slides beforehand.",
        "I don't think we should delay the launch. The risks are manageable.",
        "Let's circle back on this next week. I need more time to consider the options.",
    ]

    for i in range(count):
        event_ts = base + timedelta(days=i, hours=(i * 2) % 8)
        _store_event(
            db,
            event_id=f"email-sent-{i:04d}",
            event_type="email.sent",
            source="proton_mail",
            payload={
                "subject": f"Re: Project Update #{i}",
                "body": bodies[i % len(bodies)],
                "from_address": "user@example.com",
                "to": ["colleague@work.com"],
            },
            timestamp=event_ts.isoformat(),
        )


def _fmt_rebuild(result):
    """Format rebuild result dict for assertion messages."""
    return (
        f"events_processed={result.get('events_processed')}, "
        f"profiles_rebuilt={result.get('profiles_rebuilt')}, "
        f"errors={result.get('errors', [])[:5]}, "
        f"extractor_error_counts={result.get('extractor_error_counts', {})}"
    )


def _fmt_health(pipeline):
    """Format profile health for assertion messages."""
    health = pipeline.get_profile_health()
    lines = [f"  {k}: {v}" for k, v in health.items()]
    return "Profile health:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemporalProfileRebuild:
    """Verify that rebuilding from calendar events produces a temporal profile."""

    def test_temporal_rebuild_from_calendar_events(self, db, user_model_store):
        """Temporal profile should be populated with activity_by_hour/day/type after rebuild."""
        _insert_calendar_events(db, count=55)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["temporal"],
        )

        assert "TemporalExtractor" in result["profiles_rebuilt"], (
            f"TemporalExtractor not in profiles_rebuilt. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        profile = user_model_store.get_signal_profile("temporal")
        assert profile is not None, (
            f"Temporal profile is None after rebuild. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )
        assert profile["samples_count"] >= 50, (
            f"Expected >= 50 samples, got {profile['samples_count']}. {_fmt_rebuild(result)}"
        )

        data = profile["data"]
        assert isinstance(data, dict), f"Profile data is not a dict: {type(data)}"
        assert data.get("activity_by_hour"), (
            f"activity_by_hour is empty. data keys: {list(data.keys())}. {_fmt_health(pipeline)}"
        )
        assert data.get("activity_by_day"), (
            f"activity_by_day is empty. data keys: {list(data.keys())}. {_fmt_health(pipeline)}"
        )
        assert data.get("activity_by_type"), (
            f"activity_by_type is empty. data keys: {list(data.keys())}. {_fmt_health(pipeline)}"
        )
        # With 55 events the planning activity type should be present
        assert "planning" in data["activity_by_type"], (
            f"Expected 'planning' in activity_by_type: {data['activity_by_type']}"
        )

    def test_temporal_rebuild_derives_behavioral_fields(self, db, user_model_store):
        """With 55+ events the temporal extractor should derive chronotype, peak_hours, etc."""
        _insert_calendar_events(db, count=55)
        pipeline = SignalExtractorPipeline(db, user_model_store)
        pipeline.rebuild_profiles_from_events(event_limit=1000, missing_profiles=["temporal"])

        profile = user_model_store.get_signal_profile("temporal")
        data = profile["data"]

        # With 55 events spread across hours, we should have at least peak_hours
        total_activity = sum(data.get("activity_by_hour", {}).values())
        assert total_activity >= 50, f"Total activity {total_activity} < 50"

        # Chronotype requires 50+ activities
        assert "chronotype" in data, (
            f"chronotype not derived. total_activity={total_activity}, keys={list(data.keys())}"
        )

    def test_temporal_profile_not_stale_after_rebuild(self, db, user_model_store):
        """After rebuild the temporal profile should pass the stale check."""
        _insert_calendar_events(db, count=55)
        pipeline = SignalExtractorPipeline(db, user_model_store)
        pipeline.rebuild_profiles_from_events(event_limit=1000, missing_profiles=["temporal"])

        health = pipeline.get_profile_health()
        assert health["temporal"]["status"] == "ok", (
            f"Temporal profile is {health['temporal']['status']} after rebuild. "
            f"samples={health['temporal']['samples']}, keys={health['temporal']['data_keys']}"
        )


class TestDecisionProfileRebuild:
    """Verify that rebuilding from calendar events produces a decision profile."""

    def test_decision_rebuild_from_calendar_events(self, db, user_model_store):
        """Decision profile should track commitment patterns from calendar events."""
        _insert_calendar_events(db, count=55)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["decision"],
        )

        assert "DecisionExtractor" in result["profiles_rebuilt"], (
            f"DecisionExtractor not in profiles_rebuilt. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        profile = user_model_store.get_signal_profile("decision")
        assert profile is not None, (
            f"Decision profile is None after rebuild. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        data = profile["data"]
        assert isinstance(data, dict), f"Profile data is not a dict: {type(data)}"

        # Calendar events should produce risk_tolerance_by_domain from commitment patterns
        assert data.get("risk_tolerance_by_domain"), (
            f"risk_tolerance_by_domain is empty. data keys: {list(data.keys())}. "
            f"{_fmt_health(pipeline)}"
        )

    def test_decision_profile_tracks_delegation_from_emails(self, db, user_model_store):
        """Decision profile should detect delegation patterns in email.sent events."""
        _insert_email_sent_events(db, count=12)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["decision"],
        )

        profile = user_model_store.get_signal_profile("decision")
        assert profile is not None, f"Decision profile is None. {_fmt_rebuild(result)}"

        data = profile["data"]
        # Email bodies contain delegation patterns like "What do you think?" and
        # "Could you take care of...?" so delegation_comfort should be set.
        assert "delegation_comfort" in data, (
            f"delegation_comfort missing from decision profile. keys: {list(data.keys())}"
        )

    def test_decision_profile_not_stale_after_rebuild(self, db, user_model_store):
        """After rebuild the decision profile should pass the stale check."""
        _insert_calendar_events(db, count=55)
        _insert_email_sent_events(db, count=12)
        pipeline = SignalExtractorPipeline(db, user_model_store)
        pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["decision"],
        )

        health = pipeline.get_profile_health()
        assert health["decision"]["status"] == "ok", (
            f"Decision profile is {health['decision']['status']} after rebuild. "
            f"samples={health['decision']['samples']}, keys={health['decision']['data_keys']}"
        )


class TestSpatialProfileRebuild:
    """Verify that rebuilding from calendar events with location produces a spatial profile."""

    def test_spatial_rebuild_from_calendar_events(self, db, user_model_store):
        """Spatial profile should capture place_behaviors from events with location field."""
        _insert_calendar_events(db, count=55)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["spatial"],
        )

        assert "SpatialExtractor" in result["profiles_rebuilt"], (
            f"SpatialExtractor not in profiles_rebuilt. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        profile = user_model_store.get_signal_profile("spatial")
        assert profile is not None, (
            f"Spatial profile is None after rebuild. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        data = profile["data"]
        assert isinstance(data, dict), f"Profile data is not a dict: {type(data)}"
        assert data.get("place_behaviors"), (
            f"place_behaviors is empty. data keys: {list(data.keys())}. {_fmt_health(pipeline)}"
        )

        # Deserialize place_behaviors (stored as JSON string by SpatialExtractor)
        place_behaviors = data["place_behaviors"]
        if isinstance(place_behaviors, str):
            place_behaviors = json.loads(place_behaviors)

        # We should have at least a few distinct locations
        assert len(place_behaviors) >= 3, (
            f"Expected >= 3 distinct locations, got {len(place_behaviors)}: "
            f"{list(place_behaviors.keys())}"
        )

    def test_spatial_skips_events_without_location(self, db, user_model_store):
        """SpatialExtractor.can_process returns False for calendar events with empty location."""
        pipeline = SignalExtractorPipeline(db, user_model_store)
        spatial = next(e for e in pipeline.extractors if type(e).__name__ == "SpatialExtractor")

        event_no_location = {
            "type": "calendar.event.created",
            "payload": {"location": "", "summary": "Quick sync"},
        }
        assert not spatial.can_process(event_no_location), (
            "SpatialExtractor should reject calendar events with empty location"
        )

        event_with_location = {
            "type": "calendar.event.created",
            "payload": {"location": "Room 42", "summary": "Planning"},
        }
        assert spatial.can_process(event_with_location), (
            "SpatialExtractor should accept calendar events with non-empty location"
        )


class TestLinguisticOutboundRebuild:
    """Verify that rebuilding from email.sent events produces a linguistic profile."""

    def test_linguistic_rebuild_from_email_sent(self, db, user_model_store):
        """Linguistic (outbound) profile should be populated from email.sent events with body."""
        _insert_email_sent_events(db, count=12)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(
            event_limit=1000, missing_profiles=["linguistic"],
        )

        assert "LinguisticExtractor" in result["profiles_rebuilt"], (
            f"LinguisticExtractor not in profiles_rebuilt. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )

        profile = user_model_store.get_signal_profile("linguistic")
        assert profile is not None, (
            f"Linguistic profile is None after rebuild. {_fmt_rebuild(result)}\n{_fmt_health(pipeline)}"
        )
        assert profile["samples_count"] >= 5, (
            f"Expected >= 5 samples, got {profile['samples_count']}. {_fmt_rebuild(result)}"
        )


class TestFullRebuildWithAllMissingProfiles:
    """Verify the full check_and_rebuild_missing_profiles() startup path."""

    def test_check_and_rebuild_populates_all_four(self, db, user_model_store):
        """With calendar + email events, the startup rebuild should populate temporal,
        decision, spatial, and linguistic profiles."""
        _insert_calendar_events(db, count=55)
        _insert_email_sent_events(db, count=12)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.check_and_rebuild_missing_profiles()

        assert result["skipped"] is False, (
            f"Rebuild was skipped despite events existing. {result}"
        )

        # All four of the originally-missing profiles should now be rebuilt.
        # Note: cadence, mood_signals, relationships, topics, linguistic_inbound
        # may also be rebuilt from the email events — that's fine.
        rebuilt = set(result["rebuilt"])
        health = pipeline.get_profile_health()

        for profile_type in ["temporal", "decision", "spatial", "linguistic"]:
            status = health[profile_type]["status"]
            assert status == "ok", (
                f"Profile '{profile_type}' has status '{status}' after full rebuild. "
                f"rebuilt={result['rebuilt']}, health={health[profile_type]}"
            )

    def test_check_and_rebuild_is_idempotent_for_rebuilt_profiles(self, db, user_model_store):
        """Profiles rebuilt on the first call should not be rebuilt again on the second call.

        Note: linguistic_inbound and relationships require inbound events (email.received,
        message.received) which our test data doesn't include, so they remain missing.
        This test verifies that the profiles we CAN rebuild stay rebuilt.
        """
        _insert_calendar_events(db, count=55)
        _insert_email_sent_events(db, count=12)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        # First call — rebuilds profiles
        result1 = pipeline.check_and_rebuild_missing_profiles()
        assert result1["skipped"] is False
        rebuilt_first = set(result1["rebuilt"])

        # Second call — profiles that were rebuilt should still be present
        result2 = pipeline.check_and_rebuild_missing_profiles()
        still_missing = set(result2.get("missing_before", []))

        # None of the profiles we rebuilt should appear in missing_before
        re_missing = rebuilt_first & still_missing
        assert not re_missing, (
            f"Profiles rebuilt on first call are missing again: {re_missing}. "
            f"{_fmt_health(pipeline)}"
        )

    def test_rebuild_result_includes_error_details(self, db, user_model_store):
        """The rebuild result dict should include structured error reporting fields."""
        _insert_calendar_events(db, count=10)
        pipeline = SignalExtractorPipeline(db, user_model_store)

        result = pipeline.rebuild_profiles_from_events(event_limit=100)

        # Verify the result dict has the expected structure
        assert "events_processed" in result, "Missing events_processed key"
        assert "profiles_rebuilt" in result, "Missing profiles_rebuilt key"
        assert "errors" in result, "Missing errors key"
        assert "extractor_error_counts" in result, "Missing extractor_error_counts key"
        assert isinstance(result["events_processed"], int)
        assert isinstance(result["profiles_rebuilt"], list)
        assert isinstance(result["errors"], list)
        assert isinstance(result["extractor_error_counts"], dict)
