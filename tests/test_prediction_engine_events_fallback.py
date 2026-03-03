"""
Tests for prediction engine events.db fallback paths.

When user_model.db is corrupted or the relationships signal profile is
unavailable, the prediction engine falls back to querying events.db
directly.  These tests verify that the fallback paths produce correct
predictions in degraded mode — they are the system's safety net.

Covers:
    1. _build_contacts_from_events() — reconstructing contact data from raw events
    2. _check_relationship_maintenance() — fallback when signal profile is None or throws
    3. _check_follow_up_needs() — generating predictions without priority boost
    4. _check_calendar_conflicts() — all-day events and overlap detection
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import PropertyMock, patch

import pytest

from services.prediction_engine.engine import PredictionEngine


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _make_email_event(
    from_address: str,
    subject: str = "Test",
    message_id: str | None = None,
    timestamp: str | None = None,
    event_type: str = "email.received",
    to_addresses: list[str] | None = None,
    in_reply_to: str | None = None,
    body_plain: str = "",
) -> dict:
    """Build a well-formed email event dict ready for EventStore.store_event()."""
    payload: dict = {
        "subject": subject,
        "message_id": message_id or f"msg-{uuid.uuid4().hex[:8]}",
    }
    if event_type == "email.received":
        payload["from_address"] = from_address
        payload["snippet"] = subject
        if body_plain:
            payload["body_plain"] = body_plain
    else:
        # email.sent
        payload["to_address"] = from_address
        payload["to_addresses"] = to_addresses or [from_address]
        if in_reply_to:
            payload["in_reply_to"] = in_reply_to
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "source": "google",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "metadata": {},
    }


def _make_calendar_event(
    title: str,
    start_dt: datetime,
    end_dt: datetime,
    is_all_day: bool = False,
    location: str | None = None,
) -> dict:
    """Build a well-formed calendar event dict."""
    payload: dict = {
        "title": title,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "event_id": str(uuid.uuid4()),
    }
    if is_all_day:
        payload["is_all_day"] = True
    if location:
        payload["location"] = location
    return {
        "id": str(uuid.uuid4()),
        "type": "calendar.event.created",
        "source": "caldav",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
        "metadata": {},
    }


# -------------------------------------------------------------------------
# 1. _build_contacts_from_events() tests
# -------------------------------------------------------------------------


class TestBuildContactsFromEvents:
    """Tests for _build_contacts_from_events() which reconstructs contact data
    from raw email events when the relationships signal profile is unavailable."""

    def test_returns_correct_contact_dict_shape(self, db, event_store, user_model_store):
        """Returned contacts should have interaction_count, last_interaction,
        outbound_count, and interaction_timestamps keys."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Seed 6 inbound emails from the same contact (>= 5 threshold)
        for i in range(6):
            event_store.store_event(_make_email_event(
                from_address="alice@example.com",
                subject=f"Email {i}",
                timestamp=(now - timedelta(days=i * 5)).isoformat(),
            ))

        contacts = engine._build_contacts_from_events()

        assert "alice@example.com" in contacts
        alice = contacts["alice@example.com"]
        assert "interaction_count" in alice
        assert "last_interaction" in alice
        assert "outbound_count" in alice
        assert "interaction_timestamps" in alice
        assert alice["interaction_count"] == 6
        assert alice["outbound_count"] == 0
        assert isinstance(alice["interaction_timestamps"], list)
        assert len(alice["interaction_timestamps"]) <= 10

    def test_returns_empty_dict_for_empty_events_db(self, db, user_model_store):
        """Should return empty dict when events.db has no email events."""
        engine = PredictionEngine(db, user_model_store)
        contacts = engine._build_contacts_from_events()
        assert contacts == {}

    def test_merges_inbound_and_outbound_for_same_contact(self, db, event_store, user_model_store):
        """Inbound and outbound interactions for the same contact should be
        correctly merged into a single contact entry."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # 3 inbound emails from bob
        for i in range(3):
            event_store.store_event(_make_email_event(
                from_address="bob@example.com",
                subject=f"Inbound {i}",
                timestamp=(now - timedelta(days=i * 3)).isoformat(),
            ))

        # 3 outbound emails to bob (total = 6, meets threshold)
        for i in range(3):
            event_store.store_event(_make_email_event(
                from_address="bob@example.com",
                subject=f"Outbound {i}",
                event_type="email.sent",
                to_addresses=["bob@example.com"],
                timestamp=(now - timedelta(days=i * 3 + 1)).isoformat(),
            ))

        contacts = engine._build_contacts_from_events()

        assert "bob@example.com" in contacts
        bob = contacts["bob@example.com"]
        assert bob["interaction_count"] == 6  # 3 inbound + 3 outbound
        assert bob["outbound_count"] == 3
        # Timestamps should include both inbound and outbound
        assert len(bob["interaction_timestamps"]) >= 1

    def test_filters_contacts_with_fewer_than_5_interactions(self, db, event_store, user_model_store):
        """Contacts with fewer than 5 total interactions should be filtered out."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # 3 emails from sparse-contact (below threshold)
        for i in range(3):
            event_store.store_event(_make_email_event(
                from_address="sparse@example.com",
                subject=f"Sparse {i}",
                timestamp=(now - timedelta(days=i * 5)).isoformat(),
            ))

        # 6 emails from frequent-contact (above threshold)
        for i in range(6):
            event_store.store_event(_make_email_event(
                from_address="frequent@example.com",
                subject=f"Frequent {i}",
                timestamp=(now - timedelta(days=i * 3)).isoformat(),
            ))

        contacts = engine._build_contacts_from_events()

        assert "sparse@example.com" not in contacts
        assert "frequent@example.com" in contacts

    def test_outbound_only_contact_included_when_threshold_met(self, db, event_store, user_model_store):
        """A contact the user has only sent to (never received from) should
        still appear if the outbound count meets the threshold."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # 5 outbound emails to this contact (no inbound)
        for i in range(5):
            event_store.store_event(_make_email_event(
                from_address="outbound-only@example.com",
                subject=f"Sent {i}",
                event_type="email.sent",
                to_addresses=["outbound-only@example.com"],
                timestamp=(now - timedelta(days=i * 3)).isoformat(),
            ))

        contacts = engine._build_contacts_from_events()

        assert "outbound-only@example.com" in contacts
        entry = contacts["outbound-only@example.com"]
        assert entry["interaction_count"] == 5
        assert entry["outbound_count"] == 5

    def test_handles_db_error_gracefully(self, db, user_model_store):
        """Should return empty dict if the events.db query fails."""
        engine = PredictionEngine(db, user_model_store)

        # Patch get_connection to raise an error
        with patch.object(db, "get_connection", side_effect=sqlite3.DatabaseError("disk I/O error")):
            contacts = engine._build_contacts_from_events()

        assert contacts == {}


# -------------------------------------------------------------------------
# 2. _check_relationship_maintenance() fallback tests
# -------------------------------------------------------------------------


class TestRelationshipMaintenanceFallback:
    """Tests for _check_relationship_maintenance() when the relationships
    signal profile is unavailable or throws, forcing a fallback to events.db."""

    @pytest.mark.asyncio
    async def test_fallback_when_signal_profile_is_none(self, db, event_store, user_model_store):
        """When get_signal_profile('relationships') returns None, the engine
        should fall back to events.db and still produce predictions."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Seed 8 inbound + 2 outbound emails from alice (10 total, well above threshold)
        # Space them out so the average gap is ~7 days
        for i in range(8):
            event_store.store_event(_make_email_event(
                from_address="alice@example.com",
                subject=f"Inbound {i}",
                timestamp=(now - timedelta(days=60 + i * 7)).isoformat(),
            ))
        for i in range(2):
            event_store.store_event(_make_email_event(
                from_address="alice@example.com",
                subject=f"Outbound {i}",
                event_type="email.sent",
                to_addresses=["alice@example.com"],
                timestamp=(now - timedelta(days=65 + i * 7)).isoformat(),
            ))

        # Ensure signal profile returns None
        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_relationship_maintenance({})

        # With 10 interactions over ~60 days, avg gap is ~7 days.
        # Last interaction was 60 days ago => 60 / 7 = 8.5x avg gap > 1.5x threshold.
        # Should generate a prediction.
        assert len(predictions) >= 1
        assert predictions[0].prediction_type == "opportunity"
        assert "alice" in predictions[0].description.lower()

    @pytest.mark.asyncio
    async def test_fallback_when_signal_profile_throws(self, db, event_store, user_model_store):
        """When get_signal_profile raises an exception (e.g. corrupted DB),
        the engine should catch it and fall back to events.db."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Seed sufficient interaction data
        for i in range(8):
            event_store.store_event(_make_email_event(
                from_address="colleague@work.com",
                subject=f"Thread {i}",
                timestamp=(now - timedelta(days=60 + i * 7)).isoformat(),
            ))
        for i in range(2):
            event_store.store_event(_make_email_event(
                from_address="colleague@work.com",
                subject=f"Reply {i}",
                event_type="email.sent",
                to_addresses=["colleague@work.com"],
                timestamp=(now - timedelta(days=62 + i * 7)).isoformat(),
            ))

        # Simulate a corrupted user_model.db
        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            predictions = await engine._check_relationship_maintenance({})

        # Should still produce predictions from events.db
        assert len(predictions) >= 1
        assert predictions[0].prediction_type == "opportunity"

    @pytest.mark.asyncio
    async def test_marketing_contacts_filtered_in_fallback(self, db, event_store, user_model_store):
        """Marketing/no-reply contacts should be filtered even when using
        the events.db fallback path."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Seed noreply marketing contact with many interactions
        for i in range(10):
            event_store.store_event(_make_email_event(
                from_address="noreply@marketing.example.com",
                subject=f"Sale {i}",
                timestamp=(now - timedelta(days=50 + i * 3)).isoformat(),
            ))
        # Add outbound so it won't be filtered by the inbound-only check
        event_store.store_event(_make_email_event(
            from_address="noreply@marketing.example.com",
            event_type="email.sent",
            to_addresses=["noreply@marketing.example.com"],
            timestamp=(now - timedelta(days=55)).isoformat(),
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_relationship_maintenance({})

        # Marketing contacts should not produce predictions
        marketing_preds = [p for p in predictions if "noreply@marketing" in str(p.relevant_contacts)]
        assert len(marketing_preds) == 0

    @pytest.mark.asyncio
    async def test_inbound_only_contacts_skipped_in_fallback(self, db, event_store, user_model_store):
        """Contacts with outbound_count == 0 should be skipped (no bidirectional
        relationship evidence) even in the events.db fallback path."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Seed 8 inbound-only emails from this contact (zero outbound)
        for i in range(8):
            event_store.store_event(_make_email_event(
                from_address="inbound-only@example.com",
                subject=f"Inbound {i}",
                timestamp=(now - timedelta(days=50 + i * 5)).isoformat(),
            ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_relationship_maintenance({})

        # Should not produce prediction for inbound-only contact
        inbound_preds = [p for p in predictions if "inbound-only@example.com" in str(p.relevant_contacts)]
        assert len(inbound_preds) == 0

    @pytest.mark.asyncio
    async def test_only_contacts_exceeding_gap_threshold_generate_predictions(
        self, db, event_store, user_model_store
    ):
        """Only contacts whose current gap exceeds 1.5x the average gap
        should generate predictions."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Contact A: recent interaction (gap NOT exceeded)
        # 6 interactions, average gap ~5 days, last interaction 3 days ago
        for i in range(6):
            event_store.store_event(_make_email_event(
                from_address="recent@example.com",
                subject=f"Recent {i}",
                timestamp=(now - timedelta(days=3 + i * 5)).isoformat(),
            ))
        event_store.store_event(_make_email_event(
            from_address="recent@example.com",
            event_type="email.sent",
            to_addresses=["recent@example.com"],
            timestamp=(now - timedelta(days=5)).isoformat(),
        ))

        # Contact B: stale interaction (gap exceeded)
        # 6 interactions, average gap ~7 days, last interaction 50 days ago
        for i in range(6):
            event_store.store_event(_make_email_event(
                from_address="stale@example.com",
                subject=f"Stale {i}",
                timestamp=(now - timedelta(days=50 + i * 7)).isoformat(),
            ))
        event_store.store_event(_make_email_event(
            from_address="stale@example.com",
            event_type="email.sent",
            to_addresses=["stale@example.com"],
            timestamp=(now - timedelta(days=55)).isoformat(),
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_relationship_maintenance({})

        # Recent contact should NOT trigger prediction (3 days < 7.5 = 5 * 1.5)
        recent_preds = [p for p in predictions if "recent@example.com" in str(p.relevant_contacts)]
        assert len(recent_preds) == 0

        # Stale contact SHOULD trigger prediction (50 days > 10.5 = 7 * 1.5)
        stale_preds = [p for p in predictions if "stale@example.com" in str(p.relevant_contacts)]
        assert len(stale_preds) >= 1

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_when_no_events(self, db, user_model_store):
        """When both signal profile and events.db are empty, should return
        empty predictions list without error."""
        engine = PredictionEngine(db, user_model_store)

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_relationship_maintenance({})

        assert predictions == []


# -------------------------------------------------------------------------
# 3. _check_follow_up_needs() fallback tests
# -------------------------------------------------------------------------


class TestFollowUpNeedsFallback:
    """Tests for _check_follow_up_needs() when the relationships signal profile
    is unavailable (priority contact boost disabled but predictions still work)."""

    @pytest.mark.asyncio
    async def test_detects_unreplied_emails_without_profile(self, db, event_store, user_model_store):
        """Should detect unreplied inbound emails even when relationships
        profile is unavailable — just without the priority boost."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_email_event(
            from_address="boss@company.com",
            subject="Need the report ASAP",
            message_id="msg-boss-1",
            timestamp=(now - timedelta(hours=6)).isoformat(),
        ))

        # Force signal profile to return None (simulating corrupted user_model.db)
        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_follow_up_needs({})

        assert len(predictions) >= 1
        pred = predictions[0]
        assert pred.prediction_type == "reminder"
        assert "boss@company.com" in str(pred.relevant_contacts)
        # Without priority boost, confidence should be at baseline (0.4)
        assert pred.confidence == 0.4

    @pytest.mark.asyncio
    async def test_skips_replied_emails_without_profile(self, db, event_store, user_model_store):
        """Emails that have been replied to should not generate predictions,
        even without the relationships profile."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Inbound email
        event_store.store_event(_make_email_event(
            from_address="colleague@company.com",
            subject="Question about API",
            message_id="msg-q-1",
            timestamp=(now - timedelta(hours=6)).isoformat(),
        ))

        # Our reply
        event_store.store_event(_make_email_event(
            from_address="colleague@company.com",
            subject="Re: Question about API",
            event_type="email.sent",
            to_addresses=["colleague@company.com"],
            in_reply_to="msg-q-1",
            timestamp=(now - timedelta(hours=4)).isoformat(),
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_follow_up_needs({})

        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_skips_marketing_emails_without_profile(self, db, event_store, user_model_store):
        """Marketing/automated emails should be filtered even without the
        relationships profile."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_email_event(
            from_address="no-reply@marketing.example.com",
            subject="Big sale today!",
            message_id="msg-marketing-1",
            timestamp=(now - timedelta(hours=6)).isoformat(),
            body_plain="Click here for deals. Unsubscribe: example.com/unsub",
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_follow_up_needs({})

        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_skips_very_recent_emails(self, db, event_store, user_model_store):
        """Emails received less than 3 hours ago should not trigger predictions
        (grace period for user to respond naturally)."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_email_event(
            from_address="team@company.com",
            subject="Quick question",
            message_id="msg-recent-1",
            timestamp=(now - timedelta(hours=1)).isoformat(),
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_follow_up_needs({})

        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_profile_exception_still_generates_predictions(self, db, event_store, user_model_store):
        """When get_signal_profile throws (corrupted DB), follow-up predictions
        should still be generated without priority contact boosting."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_email_event(
            from_address="partner@company.com",
            subject="Contract review needed",
            message_id="msg-contract-1",
            timestamp=(now - timedelta(hours=8)).isoformat(),
        ))

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database disk image is malformed"),
        ):
            predictions = await engine._check_follow_up_needs({})

        # Should still produce prediction despite the exception
        assert len(predictions) >= 1
        assert predictions[0].prediction_type == "reminder"
        # Without profile, no priority boost — confidence at baseline
        assert predictions[0].confidence == 0.4

    @pytest.mark.asyncio
    async def test_aging_emails_get_confidence_boost(self, db, event_store, user_model_store):
        """Emails older than 24 hours should get a +0.2 confidence boost
        even without the relationships profile."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_email_event(
            from_address="vp@company.com",
            subject="Strategy discussion",
            message_id="msg-old-1",
            timestamp=(now - timedelta(hours=30)).isoformat(),
        ))

        with patch.object(user_model_store, "get_signal_profile", return_value=None):
            predictions = await engine._check_follow_up_needs({})

        assert len(predictions) >= 1
        # Base 0.4 + age boost 0.2 = 0.6
        assert abs(predictions[0].confidence - 0.6) < 0.01


# -------------------------------------------------------------------------
# 4. _check_calendar_conflicts() tests (events.db direct)
# -------------------------------------------------------------------------


class TestCalendarConflictsFromEvents:
    """Tests for _check_calendar_conflicts() which queries events.db directly
    for calendar events and detects overlaps. Does not depend on user_model.db."""

    @pytest.mark.asyncio
    async def test_detects_overlapping_timed_events(self, db, event_store, user_model_store):
        """Two timed events that overlap should produce a conflict prediction."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Event 1: 2:00 PM - 3:00 PM
        event_store.store_event(_make_calendar_event(
            title="Team standup",
            start_dt=now + timedelta(hours=2),
            end_dt=now + timedelta(hours=3),
        ))

        # Event 2: 2:30 PM - 3:30 PM (overlaps by 30 minutes)
        event_store.store_event(_make_calendar_event(
            title="Client call",
            start_dt=now + timedelta(hours=2, minutes=30),
            end_dt=now + timedelta(hours=3, minutes=30),
        ))

        predictions = await engine._check_calendar_conflicts({})

        assert len(predictions) >= 1
        conflict = predictions[0]
        assert conflict.prediction_type == "conflict"
        assert conflict.confidence == 0.95
        assert "overlap" in conflict.description.lower()

    @pytest.mark.asyncio
    async def test_detects_all_day_vs_timed_overlap(self, db, event_store, user_model_store):
        """An all-day event overlapping with a timed event should be detected
        as a conflict (e.g. meeting during a travel day)."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        # All-day event tomorrow
        event_store.store_event(_make_calendar_event(
            title="Conference Day",
            start_dt=tomorrow,
            end_dt=tomorrow + timedelta(days=1),
            is_all_day=True,
        ))

        # Timed meeting during the all-day event
        event_store.store_event(_make_calendar_event(
            title="Internal sync",
            start_dt=tomorrow + timedelta(hours=10),
            end_dt=tomorrow + timedelta(hours=11),
        ))

        predictions = await engine._check_calendar_conflicts({})

        # Should detect the all-day vs timed conflict
        assert len(predictions) >= 1
        conflict = predictions[0]
        assert conflict.prediction_type == "conflict"
        # All-day conflicts get lower confidence (0.8)
        assert conflict.confidence == 0.8

    @pytest.mark.asyncio
    async def test_no_conflict_for_all_day_vs_all_day(self, db, event_store, user_model_store):
        """Two all-day events should NOT be flagged as conflicts (multiple
        all-day markers are fine, e.g. 'Birthday' and 'Travel Day')."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        # Two all-day events on the same day
        event_store.store_event(_make_calendar_event(
            title="Birthday",
            start_dt=tomorrow,
            end_dt=tomorrow + timedelta(days=1),
            is_all_day=True,
        ))
        event_store.store_event(_make_calendar_event(
            title="Travel Day",
            start_dt=tomorrow,
            end_dt=tomorrow + timedelta(days=1),
            is_all_day=True,
        ))

        predictions = await engine._check_calendar_conflicts({})

        # All-day vs all-day pairs are skipped
        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_overlaps(self, db, event_store, user_model_store):
        """Well-spaced events should not produce any conflict predictions."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Event 1: 2:00 PM - 3:00 PM
        event_store.store_event(_make_calendar_event(
            title="Morning meeting",
            start_dt=now + timedelta(hours=2),
            end_dt=now + timedelta(hours=3),
        ))

        # Event 2: 5:00 PM - 6:00 PM (2 hour gap — no conflict)
        event_store.store_event(_make_calendar_event(
            title="Afternoon sync",
            start_dt=now + timedelta(hours=5),
            end_dt=now + timedelta(hours=6),
        ))

        predictions = await engine._check_calendar_conflicts({})
        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_returns_empty_with_single_event(self, db, event_store, user_model_store):
        """A single calendar event cannot have a conflict with anything."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_calendar_event(
            title="Solo meeting",
            start_dt=now + timedelta(hours=2),
            end_dt=now + timedelta(hours=3),
        ))

        predictions = await engine._check_calendar_conflicts({})
        assert len(predictions) == 0

    @pytest.mark.asyncio
    async def test_detects_tight_transitions(self, db, event_store, user_model_store):
        """Events with <15 min gap should be flagged as tight transitions."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        # Event 1: 2:00 PM - 3:00 PM
        event_store.store_event(_make_calendar_event(
            title="Design review",
            start_dt=now + timedelta(hours=2),
            end_dt=now + timedelta(hours=3),
        ))

        # Event 2: 3:05 PM - 4:00 PM (only 5 min gap)
        event_store.store_event(_make_calendar_event(
            title="Sprint planning",
            start_dt=now + timedelta(hours=3, minutes=5),
            end_dt=now + timedelta(hours=4),
        ))

        predictions = await engine._check_calendar_conflicts({})

        assert len(predictions) >= 1
        risk = predictions[0]
        assert risk.prediction_type == "risk"
        assert risk.confidence == 0.7
        assert "5 minutes between" in risk.description

    @pytest.mark.asyncio
    async def test_returns_empty_with_no_events(self, db, user_model_store):
        """No calendar events at all should return empty predictions."""
        engine = PredictionEngine(db, user_model_store)
        predictions = await engine._check_calendar_conflicts({})
        assert predictions == []

    @pytest.mark.asyncio
    async def test_conflict_includes_supporting_signals(self, db, event_store, user_model_store):
        """Conflict predictions should include supporting_signals with
        conflicting_event_ids for the accuracy tracker."""
        engine = PredictionEngine(db, user_model_store)
        now = datetime.now(timezone.utc)

        event_store.store_event(_make_calendar_event(
            title="Meeting A",
            start_dt=now + timedelta(hours=2),
            end_dt=now + timedelta(hours=3),
        ))
        event_store.store_event(_make_calendar_event(
            title="Meeting B",
            start_dt=now + timedelta(hours=2, minutes=30),
            end_dt=now + timedelta(hours=3, minutes=30),
        ))

        predictions = await engine._check_calendar_conflicts({})

        assert len(predictions) >= 1
        signals = predictions[0].supporting_signals
        assert "conflicting_event_ids" in signals
        assert len(signals["conflicting_event_ids"]) == 2
        assert "event_titles" in signals
        assert "overlap_minutes" in signals
