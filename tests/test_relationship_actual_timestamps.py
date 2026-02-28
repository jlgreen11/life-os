"""
Tests for relationship extractor actual timestamp usage.

CRITICAL BUG FIX (iteration 156):
    The relationship extractor was using event sync timestamps instead of
    actual email/message timestamps from the Date header or sent_at field.
    This caused all interactions to appear to happen at sync time, collapsing
    time gaps between interactions to ~0 days and breaking relationship
    maintenance predictions.

    This test suite verifies:
    1. Email Date header timestamps are used (not sync time)
    2. Message sent_at/received_at timestamps are used (not sync time)
    3. Relationship frequency calculations use real gaps
    4. Fallback to sync time works when no actual timestamp available
"""

import json
from datetime import datetime, timedelta, timezone

import pytest


def test_relationship_extractor_uses_email_date_not_sync_time(
    db, user_model_store
):
    """
    Verify relationship extractor uses actual email Date header timestamp.

    Without this fix:
        - All interactions use event.timestamp (sync time: 2026-02-16)
        - Gaps between interactions: ~0 days
        - Relationship maintenance: 0 predictions (everyone contacted "today")

    With this fix:
        - Interactions use payload.email_date (actual Date header)
        - Gaps between interactions: real time spans (7, 14, 30 days)
        - Relationship maintenance: accurate predictions for overdue contacts
    """
    from services.signal_extractor.relationship import RelationshipExtractor

    extractor = RelationshipExtractor(db, user_model_store)

    # Create 3 emails with DIFFERENT actual dates but SAME sync time
    # This simulates a connector catching up on historical emails
    sync_time = "2026-02-16T10:00:00+00:00"

    events = [
        {
            "id": "email-1",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,  # All synced at same time
            "payload": {
                "from_address": "alice@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "First email",
                "body": "Hello",
                "channel": "google",
                # Actual email date: 30 days ago
                "email_date": "2026-01-17T08:00:00+00:00",
            },
        },
        {
            "id": "email-2",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,  # Same sync time
            "payload": {
                "from_address": "alice@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "Second email",
                "body": "Follow up",
                "channel": "google",
                # Actual email date: 14 days ago
                "email_date": "2026-02-02T09:00:00+00:00",
            },
        },
        {
            "id": "email-3",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,  # Same sync time
            "payload": {
                "from_address": "alice@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "Third email",
                "body": "Latest",
                "channel": "google",
                # Actual email date: 7 days ago
                "email_date": "2026-02-09T10:00:00+00:00",
            },
        },
    ]

    # Process all events through the extractor
    for event in events:
        extractor.extract(event)

    # Load the relationship profile
    profile = user_model_store.get_signal_profile("relationships")
    assert profile is not None, "Relationships profile should exist"

    contacts = profile["data"]["contacts"]
    assert "alice@example.com" in contacts, "Alice should be in contacts"

    alice = contacts["alice@example.com"]

    # Verify interaction count
    assert alice["interaction_count"] == 3, "Should have 3 interactions"

    # Verify timestamps are in chronological order (oldest to newest)
    timestamps = alice["interaction_timestamps"]
    assert len(timestamps) == 3, "Should have 3 timestamp entries"

    # Parse timestamps for comparison
    ts_dates = [
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for ts in timestamps
    ]

    # Verify they're in ascending order (oldest first)
    assert ts_dates[0] < ts_dates[1] < ts_dates[2], \
        "Timestamps should be chronologically ordered"

    # Verify actual dates are used (not all collapsed to sync time)
    # Gap 1 → 2: ~16 days
    # Gap 2 → 3: ~7 days
    gap1 = (ts_dates[1] - ts_dates[0]).days
    gap2 = (ts_dates[2] - ts_dates[1]).days

    assert 15 <= gap1 <= 17, \
        f"First gap should be ~16 days, got {gap1}"
    assert 6 <= gap2 <= 8, \
        f"Second gap should be ~7 days, got {gap2}"

    # WITHOUT the fix, all gaps would be ~0 days (same sync time)
    # This would break relationship maintenance predictions


def test_message_connector_uses_sent_at_timestamp(
    db, user_model_store
):
    """
    Verify relationship extractor uses message sent_at/received_at timestamps.

    Message connectors (Slack, SMS, etc.) use different field names than
    email connectors. This test ensures the fallback logic works correctly.
    """
    from services.signal_extractor.relationship import RelationshipExtractor

    extractor = RelationshipExtractor(db, user_model_store)

    sync_time = "2026-02-16T10:00:00+00:00"

    events = [
        {
            "id": "msg-1",
            "type": "message.received",
            "source": "slack",
            "timestamp": sync_time,
            "payload": {
                "from_address": "bob@slack",
                "to_addresses": ["user@slack"],
                "body": "Hey there",
                "channel": "slack",
                # Message timestamp (different from email_date field)
                "received_at": "2026-02-10T14:30:00+00:00",
            },
        },
        {
            "id": "msg-2",
            "type": "message.sent",
            "source": "slack",
            "timestamp": sync_time,
            "payload": {
                "from_address": "user@slack",
                "to_addresses": ["bob@slack"],
                "body": "Hi Bob",
                "channel": "slack",
                # Message timestamp
                "sent_at": "2026-02-14T09:15:00+00:00",
            },
        },
    ]

    for event in events:
        extractor.extract(event)

    profile = user_model_store.get_signal_profile("relationships")
    contacts = profile["data"]["contacts"]

    assert "bob@slack" in contacts, "Bob should be tracked"

    bob = contacts["bob@slack"]
    assert bob["interaction_count"] == 2, "Should have 2 interactions"

    # Verify timestamps use actual message times (not sync time)
    timestamps = bob["interaction_timestamps"]
    ts_dates = [
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for ts in timestamps
    ]

    gap = (ts_dates[1] - ts_dates[0]).days
    assert 3 <= gap <= 5, \
        f"Gap should be ~4 days (Feb 10 → Feb 14), got {gap}"


def test_fallback_to_sync_time_when_no_actual_timestamp(
    db, user_model_store
):
    """
    Verify extractor falls back to sync timestamp when no actual date available.

    Some events (e.g., older connectors, test events) may not include
    email_date or sent_at fields. The system should still work using the
    event sync timestamp as a fallback.
    """
    from services.signal_extractor.relationship import RelationshipExtractor

    extractor = RelationshipExtractor(db, user_model_store)

    event = {
        "id": "legacy-1",
        "type": "email.received",
        "source": "legacy",
        "timestamp": "2026-02-16T10:00:00+00:00",
        "payload": {
            "from_address": "carol@example.com",
            "to_addresses": ["user@example.com"],
            "subject": "Legacy email",
            "body": "No email_date field",
            "channel": "legacy",
            # No email_date, sent_at, or received_at
        },
    }

    extractor.extract(event)

    profile = user_model_store.get_signal_profile("relationships")
    contacts = profile["data"]["contacts"]

    assert "carol@example.com" in contacts, "Carol should be tracked"

    carol = contacts["carol@example.com"]
    timestamps = carol["interaction_timestamps"]

    # Should use event.timestamp as fallback
    assert len(timestamps) == 1
    assert timestamps[0] == "2026-02-16T10:00:00+00:00", \
        "Should fall back to event timestamp"


def test_relationship_maintenance_predictions_after_fix(
    db, user_model_store
):
    """
    Integration test: Verify relationship maintenance predictions work correctly.

    This test creates a realistic scenario where:
    - User has ongoing relationship with regular 14-day frequency
    - Last interaction was 30 days ago (2x overdue)
    - Relationship maintenance prediction should fire with confidence ~0.4

    WITHOUT the timestamp fix:
        - All timestamps collapse to sync time
        - Avg gap: 0 days
        - Last interaction: "today"
        - Prediction: NONE (not overdue)

    WITH the timestamp fix:
        - Timestamps span 6 months
        - Avg gap: 14 days
        - Last interaction: 30 days ago
        - Prediction: GENERATED (confidence ~0.4)
    """
    from services.signal_extractor.relationship import RelationshipExtractor
    from services.prediction_engine.engine import PredictionEngine

    extractor = RelationshipExtractor(db, user_model_store)
    engine = PredictionEngine(db, user_model_store)

    sync_time = "2026-02-16T10:00:00+00:00"
    base_date = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Create 10 interactions over 6 months with regular 14-day frequency
    events = []
    for i in range(10):
        interaction_date = base_date + timedelta(days=i * 14)
        events.append({
            "id": f"email-{i}",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,  # All synced at once
            "payload": {
                "from_address": "dave@example.com",
                "to_addresses": ["user@example.com"],
                "subject": f"Email {i}",
                "body": "Regular check-in",
                "channel": "google",
                "email_date": interaction_date.isoformat(),
            },
        })

    # Last interaction: 30 days before sync_time (2x overdue for 14-day frequency).
    # Use sync_time as the reference point (not datetime.now()) so the test is
    # deterministic regardless of when it runs — datetime.now() drifts daily
    # and causes the average-gap assertion to fail as the real calendar advances.
    sync_time_dt = datetime.fromisoformat(sync_time)
    events.append({
        "id": "email-last",
        "type": "email.received",
        "source": "google",
        "timestamp": sync_time,
        "payload": {
            "from_address": "dave@example.com",
            "to_addresses": ["user@example.com"],
            "subject": "Last email",
            "body": "Long time no talk",
            "channel": "google",
            "email_date": (sync_time_dt - timedelta(days=30)).isoformat(),
        },
    })

    # Extract a single outbound event FIRST so Dave is marked as a bidirectional
    # contact (passes the inbound-only filter added in iteration 187).  We process
    # it before the inbound events so it lands at position 0 in the ring buffer;
    # the last-10 slice used by the avg-gap assertion then contains only the
    # regular 14-day inbound entries, preserving the expected gap pattern.
    extractor.extract({
        "id": "dave-outbound-0",
        "type": "email.sent",
        "source": "google",
        "timestamp": sync_time,
        "payload": {
            "from_address": "user@example.com",
            "to_addresses": ["dave@example.com"],
            "subject": "Re: Earlier check-in",
            "body": "Thanks Dave",
            "channel": "google",
            "email_date": (base_date - timedelta(days=30)).isoformat(),
        },
    })

    # Process all inbound events
    for event in events:
        extractor.extract(event)

    # Verify profile has correct average gap
    profile = user_model_store.get_signal_profile("relationships")
    dave = profile["data"]["contacts"]["dave@example.com"]

    assert dave["interaction_count"] == 12, "Should have 12 interactions (11 inbound + 1 outbound)"

    timestamps = dave["interaction_timestamps"]
    ts_dates = [
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        for ts in timestamps[-10:]  # Last 10 for avg calculation
    ]

    # Calculate average gap across the last-10 slice.
    # With sync_time = 2026-02-16 and last email = 2026-01-17 (30 days before sync):
    #   - 8 gaps of 14 days (inbound events i=1..9)
    #   - 1 gap of 43 days (2025-12-05 → 2026-01-17)
    #   - Average: (8*14 + 43) / 9 ≈ 17.2 days
    gaps = [(ts_dates[i + 1] - ts_dates[i]).days for i in range(len(ts_dates) - 1)]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0

    assert 12 <= avg_gap <= 20, \
        f"Average gap should be ~14-17 days (anchored to sync_time), got {avg_gap:.1f}"

    # Run prediction engine
    import asyncio
    predictions = asyncio.run(engine.generate_predictions({}))

    # Filter to relationship maintenance predictions for Dave
    dave_preds = [
        p for p in predictions
        if p.prediction_type == "opportunity"
        and "dave@example.com" in p.description
    ]

    assert len(dave_preds) > 0, \
        "Should generate relationship maintenance prediction for Dave (30 days overdue)"

    pred = dave_preds[0]
    assert pred.confidence > 0.3, \
        f"Confidence should be > 0.3 for 2x overdue contact, got {pred.confidence:.2f}"

    # The description should mention how many days since last contact.
    # The prediction engine computes days_since_last_contact from datetime.now(),
    # so we can't assert an exact number — instead, verify "days" is in the
    # description and the supporting_signals carry the correct field.
    assert "days" in pred.description, \
        f"Description should mention days since last contact, got: {pred.description}"
    assert pred.supporting_signals.get("days_since_last_contact", 0) > 0, \
        "Supporting signals should include positive days_since_last_contact"


def test_multiple_contacts_different_frequencies(
    db, user_model_store
):
    """
    Verify the fix works correctly with multiple contacts at different frequencies.

    Real-world scenario:
    - Alice: Daily contact (1-day frequency) → last contacted today → NO prediction
    - Bob: Weekly contact (7-day frequency) → last contacted 14 days ago → PREDICTION
    - Carol: Monthly contact (30-day frequency) → last contacted 60 days ago → PREDICTION
    """
    from services.signal_extractor.relationship import RelationshipExtractor
    from services.prediction_engine.engine import PredictionEngine

    extractor = RelationshipExtractor(db, user_model_store)
    engine = PredictionEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    sync_time = now.isoformat()

    # Alice: 10 daily interactions, last one today
    for i in range(10):
        interaction_date = now - timedelta(days=9 - i)
        extractor.extract({
            "id": f"alice-{i}",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,
            "payload": {
                "from_address": "alice@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "Daily check-in",
                "body": "Hi",
                "channel": "google",
                "email_date": interaction_date.isoformat(),
            },
        })

    # Bob: 5 weekly interactions (4 inbound + 1 outbound reply), last one 14 days ago (2x overdue).
    # The outbound event is required so Bob registers as a bidirectional contact and
    # passes the inbound-only filter added in iteration 187.
    for i in range(4):
        interaction_date = now - timedelta(days=14 + (3 - i) * 7)
        extractor.extract({
            "id": f"bob-{i}",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,
            "payload": {
                "from_address": "bob@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "Weekly update",
                "body": "Hello",
                "channel": "google",
                "email_date": interaction_date.isoformat(),
            },
        })
    # Bob outbound — user replied to Bob (establishes bidirectionality)
    extractor.extract({
        "id": "bob-outbound-0",
        "type": "email.sent",
        "source": "google",
        "timestamp": sync_time,
        "payload": {
            "from_address": "user@example.com",
            "to_addresses": ["bob@example.com"],
            "subject": "Re: Weekly update",
            "body": "Thanks Bob",
            "channel": "google",
            "email_date": (now - timedelta(days=35)).isoformat(),
        },
    })

    # Carol: 5 monthly interactions (4 inbound + 1 outbound reply), last one 60 days ago (2x overdue).
    # The outbound event establishes Carol as a bidirectional contact.
    for i in range(4):
        interaction_date = now - timedelta(days=60 + (3 - i) * 30)
        extractor.extract({
            "id": f"carol-{i}",
            "type": "email.received",
            "source": "google",
            "timestamp": sync_time,
            "payload": {
                "from_address": "carol@example.com",
                "to_addresses": ["user@example.com"],
                "subject": "Monthly catch-up",
                "body": "Hi there",
                "channel": "google",
                "email_date": interaction_date.isoformat(),
            },
        })
    # Carol outbound — user replied to Carol (establishes bidirectionality)
    extractor.extract({
        "id": "carol-outbound-0",
        "type": "email.sent",
        "source": "google",
        "timestamp": sync_time,
        "payload": {
            "from_address": "user@example.com",
            "to_addresses": ["carol@example.com"],
            "subject": "Re: Monthly catch-up",
            "body": "Great to hear from you",
            "channel": "google",
            "email_date": (now - timedelta(days=90)).isoformat(),
        },
    })

    # Run predictions
    import asyncio
    predictions = asyncio.run(engine.generate_predictions({}))

    maintenance_preds = [
        p for p in predictions
        if p.prediction_type == "opportunity"
    ]

    # Extract contact emails from predictions
    contacts_with_preds = set()
    for pred in maintenance_preds:
        if "alice@example.com" in pred.description:
            contacts_with_preds.add("alice")
        if "bob@example.com" in pred.description:
            contacts_with_preds.add("bob")
        if "carol@example.com" in pred.description:
            contacts_with_preds.add("carol")

    # Alice should NOT have prediction (contacted today, avg=1 day, threshold=1.5 days)
    assert "alice" not in contacts_with_preds, \
        "Alice should not trigger prediction (contacted today)"

    # Bob SHOULD have prediction (14 days ago, avg=7 days, threshold=10.5 days)
    assert "bob" in contacts_with_preds, \
        "Bob should trigger prediction (14 days > 10.5 day threshold)"

    # Carol SHOULD have prediction (60 days ago, avg=30 days, threshold=45 days)
    assert "carol" in contacts_with_preds, \
        "Carol should trigger prediction (60 days > 45 day threshold)"
