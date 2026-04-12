"""
Tests for MoodInferenceEngine profile persistence.

Verifies that the mood_signals profile is correctly persisted to the
database after processing events, with a focus on email.received events
(the most common qualifying event type).

This regression suite was added to guard against the production issue where
13,726 qualifying events produced 0 mood_signals profile entries.  The root
cause was silent write failures (WAL corruption, JSON serialization errors,
database locked) swallowed by update_signal_profile's try/except.  The
post-write verification added to _update_mood_state() now logs a diagnostic
error when the read-back fails, so operators can investigate.

Test structure:
    - TestMoodProfilePersistenceEmailReceived: email.received-specific tests
    - TestMoodProfilePersistenceMultipleEvents: accumulation across events
    - TestPostWriteVerification: verifies the read-back check works correctly
"""

import logging

import pytest
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.mood import MoodInferenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_email_received(body: str, hour: int = 10, source: str = "gmail") -> dict:
    """Build a synthetic email.received event with the given body text and hour.

    Using a fixed date (2024-01-15) avoids timezone edge cases in CI.  The hour
    parameter lets callers test different parts of the circadian energy curve.
    """
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "source": source,
        "timestamp": datetime(2024, 1, 15, hour, 30, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "body": body,
            "subject": "Test email",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# email.received persistence tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMoodProfilePersistenceEmailReceived:
    """Verify that email.received events with body text persist the mood_signals profile."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return MoodInferenceEngine(db, user_model_store)

    def test_email_received_with_body_produces_signals(self, engine):
        """email.received with body text must produce a non-empty signal list.

        The circadian_energy proxy signal fires for any inbound message with
        body text and a valid ISO timestamp.  This is the minimum signal that
        should be extracted even from a neutral, short email.
        """
        event = _make_email_received(
            "Hello, I wanted to follow up on our meeting from last week.", hour=10
        )

        signals = engine.extract(event)

        assert len(signals) == 1, (
            "email.received with body text must produce signals; "
            "got empty list — circadian_energy proxy not firing"
        )
        assert signals[0]["type"] == "mood_signal"
        mood_signals = signals[0]["signals"]
        assert len(mood_signals) >= 1, (
            "At least circadian_energy signal must be extracted from a "
            "timestamped email.received event with body text"
        )

    def test_email_received_mood_signals_profile_exists_after_extract(self, engine, user_model_store):
        """After processing email.received, mood_signals profile must exist in the DB.

        Regression guard: 13,726 qualifying events produced 0 profile entries.
        This test ensures the basic persistence round-trip works.
        """
        event = _make_email_received(
            "Hello, I wanted to follow up on our meeting from last week.", hour=10
        )

        engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None, (
            "mood_signals profile must exist in the database after "
            "processing an email.received event with body text"
        )
        assert "data" in profile, "profile must have a 'data' key"

    def test_email_received_profile_has_recent_signals_key(self, engine, user_model_store):
        """Persisted mood_signals profile must contain the 'recent_signals' ring buffer."""
        event = _make_email_received(
            "Hello, I wanted to follow up on our meeting from last week.", hour=10
        )

        engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert "recent_signals" in profile["data"], (
            "mood_signals profile data must contain 'recent_signals' key"
        )

    def test_email_received_profile_has_at_least_one_signal(self, engine, user_model_store):
        """Persisted recent_signals list must have at least one entry.

        If recent_signals is empty despite extract() returning signals, the
        _update_mood_state method is not correctly persisting the data.
        """
        event = _make_email_received(
            "Hello, I wanted to follow up on our meeting from last week.", hour=10
        )

        engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        recent_signals = profile["data"]["recent_signals"]
        assert len(recent_signals) >= 1, (
            "Persisted profile must contain at least one signal entry; "
            "recent_signals is empty despite extract() succeeding"
        )

    def test_email_received_circadian_energy_in_persisted_profile(self, engine, user_model_store):
        """Persisted profile must contain a circadian_energy signal for morning email.

        This signal is the minimum expected output for any inbound message
        with body text at a known hour.  Its absence means the circadian
        energy proxy is not running or not persisting.
        """
        event = _make_email_received(
            "Good morning, can we schedule a call this week?", hour=9
        )

        engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        signal_types = [s["signal_type"] for s in profile["data"]["recent_signals"]]
        assert "circadian_energy" in signal_types, (
            "Persisted mood_signals profile must include circadian_energy "
            "for email.received events with body text and a valid timestamp"
        )

    def test_negative_email_produces_incoming_negative_language_signal(self, engine, user_model_store):
        """Emails with negative words persist an incoming_negative_language signal."""
        event = _make_email_received(
            "I'm frustrated with this problem. The situation is difficult and "
            "exhausting, and I'm worried about the deadline.",
            hour=14,
        )

        engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        signal_types = [s["signal_type"] for s in profile["data"]["recent_signals"]]
        assert "incoming_negative_language" in signal_types, (
            "Email with negative words must produce an incoming_negative_language signal "
            "in the persisted profile"
        )

    def test_email_with_missing_body_produces_no_signals(self, engine, user_model_store):
        """email.received with no body text must not create any signals or profile."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "gmail",
            "timestamp": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {},  # No body field at all
        }

        signals = engine.extract(event)

        assert signals == [], "email.received with empty payload must produce no signals"
        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is None, "No profile should be written when no signals are extracted"


# ──────────────────────────────────────────────────────────────────────────────
# Signal accumulation across multiple events
# ──────────────────────────────────────────────────────────────────────────────

class TestMoodProfilePersistenceMultipleEvents:
    """Verify signal accumulation and ring-buffer capping across event sequences."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return MoodInferenceEngine(db, user_model_store)

    def test_multiple_emails_accumulate_signals_in_profile(self, engine, user_model_store):
        """Processing multiple email.received events accumulates signals in the profile."""
        events = [
            _make_email_received(
                f"This is email number {i} with some content for testing.",
                hour=hour,
            )
            for i, hour in enumerate([9, 11, 14], start=1)
        ]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        # Each email produces at least one circadian_energy signal → 3+ total signals
        assert len(profile["data"]["recent_signals"]) >= 3, (
            "Three email.received events must accumulate at least 3 signals "
            "(one circadian_energy per event minimum)"
        )

    def test_profile_samples_count_increments_per_write(self, engine, user_model_store):
        """Each update to the mood_signals profile should increment samples_count."""
        events = [
            _make_email_received("First email body text here.", hour=9),
            _make_email_received("Second email body text here.", hour=11),
        ]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        # Two extract() calls → two update_signal_profile() calls → samples_count >= 2
        assert profile["samples_count"] >= 2, (
            "Each successful write to mood_signals must increment samples_count"
        )

    def test_mixed_event_types_all_persist_to_same_profile(self, engine, user_model_store):
        """Signals from different event types all accumulate in the same mood_signals profile."""
        email_event = _make_email_received("Morning follow-up on the project.", hour=10)
        sleep_event = {
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime(2024, 1, 15, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {"duration_hours": 7.5, "quality_score": 0.8},
        }

        engine.extract(email_event)
        engine.extract(sleep_event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        signal_types = {s["signal_type"] for s in profile["data"]["recent_signals"]}
        # Should contain both circadian_energy (from email) and sleep signals
        assert "circadian_energy" in signal_types, "Email event must contribute circadian_energy"
        assert "sleep_quality" in signal_types, "Sleep event must contribute sleep_quality"


# ──────────────────────────────────────────────────────────────────────────────
# Post-write verification behaviour
# ──────────────────────────────────────────────────────────────────────────────

class TestPostWriteVerification:
    """Verify the post-write read-back in _update_mood_state() works correctly."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return MoodInferenceEngine(db, user_model_store)

    def test_post_write_profile_readable_after_extract(self, engine, user_model_store):
        """Profile written by _update_mood_state must be immediately readable.

        This is the invariant the post-write verification asserts: if we write,
        we must be able to read back.  A failure here means the database is in
        a degraded state and the logger.error in _update_mood_state() would fire.
        """
        event = _make_email_received(
            "Please review the attached document and share your thoughts.", hour=14
        )

        signals = engine.extract(event)

        # Only check read-back when signals were actually produced
        assert len(signals) > 0, "Test event must produce signals to be meaningful"
        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None, (
            "Profile written by _update_mood_state must be immediately readable; "
            "post-write verification would log an error if this fails"
        )

    def test_no_logger_error_for_normal_email_event(self, engine, caplog):
        """Processing a normal email.received event must NOT trigger the post-write error log.

        When persistence works correctly, logger.error must not fire.  If it does,
        the database or JSON serialization is broken.
        """
        event = _make_email_received(
            "Quick question about the project timeline — can we chat tomorrow?", hour=10
        )

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.mood"):
            engine.extract(event)

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        persistence_errors = [m for m in error_messages if "FAILED to persist" in m]
        assert len(persistence_errors) == 0, (
            "No post-write verification error expected for a normal email.received event; "
            f"got: {persistence_errors}"
        )

    def test_signal_data_json_serializable(self, engine):
        """All signal dicts produced by MoodInferenceEngine must be JSON-serializable.

        If any signal field contains a non-serializable type (e.g. an Enum, datetime,
        or set), json.dumps() in update_signal_profile() would raise and the profile
        would silently not be written.
        """
        import json

        events = [
            _make_email_received("Morning email with neutral content.", hour=9),
            _make_email_received(
                "I'm frustrated and stressed about this urgent problem.", hour=14
            ),
            {
                "type": EventType.SLEEP_RECORDED.value,
                "timestamp": datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc).isoformat(),
                "payload": {"duration_hours": 8.0, "quality_score": 0.85},
            },
            {
                "type": EventType.CALENDAR_EVENT_CREATED.value,
                "timestamp": datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc).isoformat(),
                "payload": {
                    "title": "Team standup",
                    "attendees": ["alice@example.com", "bob@example.com"],
                },
            },
        ]

        for event in events:
            result = engine.extract(event)
            for envelope in result:
                for signal in envelope.get("signals", []):
                    try:
                        json.dumps(signal)
                    except (TypeError, ValueError) as exc:
                        pytest.fail(
                            f"Signal is not JSON-serializable: {signal!r}\n"
                            f"Error: {exc}\n"
                            "Non-serializable signals would cause update_signal_profile "
                            "to silently fail and the profile to never persist."
                        )
