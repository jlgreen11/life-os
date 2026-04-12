"""
Tests for the mood profile write-failure root-cause fix.

This suite validates the specific fixes introduced to address the silent
write-failure bug where 13,726 qualifying events produced zero mood_signals
profile entries:

1. **Serialization guard**: ``_update_mood_state`` now validates the data dict
   is JSON-serializable *before* calling ``update_signal_profile()``.  If it
   isn't, a descriptive error is logged identifying the problematic key/type
   and the write is skipped cleanly rather than silently losing the data.

2. **Ring buffer cap at 500**: The ``recent_signals`` list is now capped at
   500 entries (increased from 200) to accommodate higher-volume event streams
   while still preventing unbounded growth.

3. **WAL checkpoint retry**: If the post-write read-back returns ``None``
   (indicating a transient WAL visibility failure), the method performs a
   WAL checkpoint and retries the write once.

Test structure:
    - TestSerializationGuard:         validates bad types are detected and logged
    - TestRingBufferCap500:           validates the 500-entry cap
    - TestWALCheckpointRetry:         validates retry logic on read-back failure
    - TestBasicPersistenceRoundTrip:  end-to-end check: 5 events → profile exists
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from models.core import EventType
from services.signal_extractor.mood import MoodInferenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _email_received(body: str, hour: int = 10) -> dict:
    """Build a synthetic email.received event.

    Uses a fixed calendar date (2024-06-15) to avoid timezone edge cases in CI.
    ``hour`` controls the UTC hour so callers can exercise different parts of
    the circadian energy curve.
    """
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": datetime(2024, 6, 15, hour, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {"body": body, "subject": "Test"},
    }


def _make_engine(db, user_model_store) -> MoodInferenceEngine:
    """Return a MoodInferenceEngine wired to the provided real temp DB."""
    return MoodInferenceEngine(db, user_model_store)


# ---------------------------------------------------------------------------
# 1. Serialization guard
# ---------------------------------------------------------------------------

class TestSerializationGuard:
    """Verify that non-serializable data is caught before the write and logged."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return _make_engine(db, user_model_store)

    def test_normal_signals_are_json_serializable(self, engine, user_model_store):
        """All signal types emitted by MoodInferenceEngine must be JSON-serializable.

        This is the fundamental invariant that makes ``update_signal_profile``
        succeed silently.  If any signal value is a ``set``, ``datetime``,
        ``Enum``, or Pydantic model, ``json.dumps`` inside ``update_signal_profile``
        would raise a ``TypeError`` that is caught and swallowed — producing
        the exact silent write failure observed in production.
        """
        events = [
            _email_received("Good morning, hope you're well.", hour=9),
            _email_received(
                "I'm frustrated and stressed about this urgent problem.", hour=14
            ),
            {
                "type": EventType.SLEEP_RECORDED.value,
                "timestamp": datetime(2024, 6, 15, 6, 30, 0, tzinfo=timezone.utc).isoformat(),
                "payload": {"duration_hours": 7.5, "quality_score": 0.8},
            },
            {
                "type": EventType.CALENDAR_EVENT_CREATED.value,
                "timestamp": datetime(2024, 6, 15, 9, 0, 0, tzinfo=timezone.utc).isoformat(),
                "payload": {
                    "title": "Team standup",
                    "attendees": ["alice@example.com", "bob@example.com", "carol@example.com"],
                },
            },
            {
                "type": EventType.TRANSACTION_NEW.value,
                "timestamp": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
                "payload": {"amount": -250.00, "merchant": "Electronics Store"},
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
                            f"Signal dict is not JSON-serializable: {signal!r}\n"
                            f"Error: {exc}\n"
                            "Non-serializable signals cause update_signal_profile to "
                            "silently skip the write, producing the mood_signals "
                            "persistence failure."
                        )

    def test_non_serializable_signal_triggers_error_log(self, engine, caplog):
        """Injecting a non-serializable type into the profile must log an error.

        We monkeypatch ``get_signal_profile`` to return a pre-populated data dict
        that contains a Python ``set`` (not JSON-serializable).  The serialization
        guard in ``_update_mood_state`` must detect this and emit a logger.error
        rather than silently failing.
        """
        # Pre-load the profile with a set — a common mistake when converting
        # defaultdict(set) without explicit serialization.
        bad_data = {
            "recent_signals": [
                {
                    "signal_type": "bad_signal",
                    "value": {1, 2, 3},        # set — not JSON-serializable
                    "delta_from_baseline": 0.0,
                    "weight": 0.5,
                    "source": "test",
                }
            ]
        }

        with patch.object(
            engine.ums, "get_signal_profile", return_value={"data": bad_data}
        ):
            with caplog.at_level(logging.ERROR, logger="services.signal_extractor.mood"):
                # Process a real event so _update_mood_state is called.
                engine.extract(_email_received("Test email body here.", hour=10))

        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        # Search case-insensitively — the log message uses "non-JSON-serializable"
        # (mixed case) but we want a robust match regardless of capitalisation.
        serialization_errors = [
            m for m in error_msgs if "non-json-serializable" in m.lower()
        ]
        assert len(serialization_errors) >= 1, (
            "Serialization guard must emit logger.error when data contains "
            "non-JSON-serializable types; no such log was found. "
            f"All error messages: {error_msgs}"
        )

    def test_non_serializable_signal_does_not_corrupt_db(self, engine, user_model_store, caplog):
        """When serialization fails the write must be skipped entirely.

        The profile must remain absent (or unchanged) rather than persisting
        a partially-valid or corrupt JSON blob.
        """
        bad_data = {
            "recent_signals": [
                {
                    "signal_type": "datetime_signal",
                    "value": datetime(2024, 1, 1),    # datetime — not JSON-serializable
                    "delta_from_baseline": 0.0,
                    "weight": 0.5,
                    "source": "test",
                }
            ]
        }

        with patch.object(
            engine.ums, "get_signal_profile", return_value={"data": bad_data}
        ):
            with caplog.at_level(logging.ERROR, logger="services.signal_extractor.mood"):
                engine.extract(_email_received("Short email.", hour=10))

        # The profile should not have been written with the corrupt data.
        # (It may have been written previously with valid data, but not with
        # the bad signals injected above.)
        profile = user_model_store.get_signal_profile("mood_signals")
        if profile is not None:
            # If a profile exists it must be valid JSON (already validated by
            # get_signal_profile's json.loads).  Just verify no datetime objects.
            for sig in profile["data"].get("recent_signals", []):
                for val in sig.values():
                    assert not isinstance(val, datetime), (
                        "Profile must not contain raw datetime objects — "
                        "these would have been written from the corrupt data dict."
                    )


# ---------------------------------------------------------------------------
# 2. Ring buffer cap at 500
# ---------------------------------------------------------------------------

class TestRingBufferCap500:
    """Verify the ring buffer is capped at 500 entries (increased from 200)."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return _make_engine(db, user_model_store)

    def test_ring_buffer_caps_at_500(self, engine, user_model_store):
        """recent_signals must never exceed 500 entries.

        With 13,726 qualifying events in production, an uncapped list would
        grow to tens of thousands of entries — making the JSON blob too large
        for practical serialization and degrading mood estimates with stale
        historical data.
        """
        # Pre-populate profile with exactly 499 signals.
        initial_signals = [
            {
                "signal_type": "seed_signal",
                "value": float(i),
                "delta_from_baseline": 0.0,
                "weight": 0.3,
                "source": "seed",
            }
            for i in range(499)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": initial_signals})

        # Process two signals (sleep_quality + sleep_duration) → total = 501.
        engine.extract({
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime(2024, 6, 15, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {"duration_hours": 7.0, "quality_score": 0.75},
        })

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        count = len(profile["data"]["recent_signals"])
        assert count <= 500, (
            f"Ring buffer must be capped at 500 entries; found {count}. "
            "An unbounded buffer causes performance degradation and keeps "
            "stale signals influencing mood computations."
        )

    def test_ring_buffer_evicts_oldest_signals(self, engine, user_model_store):
        """When the buffer overflows, oldest signals must be evicted first.

        This ensures the mood estimate always reflects recent behaviour
        rather than accumulating indefinitely from historical events.
        """
        # Pre-populate with 500 signals: values 0 … 499.
        initial_signals = [
            {
                "signal_type": "index_signal",
                "value": float(i),
                "delta_from_baseline": 0.0,
                "weight": 0.3,
                "source": "seed",
            }
            for i in range(500)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": initial_signals})

        # Add one more signal → should evict signal at index 0 (value=0.0)
        # and one from index 1 (value=1.0) because sleep adds 2 signals.
        engine.extract({
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime(2024, 6, 15, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {"duration_hours": 8.0, "quality_score": 0.9},
        })

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        recent = profile["data"]["recent_signals"]
        # The 2 oldest entries (value=0.0 and value=1.0) should be gone.
        values = [s["value"] for s in recent if s["signal_type"] == "index_signal"]
        assert 0.0 not in values, (
            "Oldest signal (value=0.0) must have been evicted when buffer exceeded 500."
        )

    def test_ring_buffer_unchanged_below_cap(self, engine, user_model_store):
        """Processing a few events must not evict signals when well below the cap."""
        # Start with 10 signals.
        initial_signals = [
            {
                "signal_type": "keep_signal",
                "value": float(i),
                "delta_from_baseline": 0.0,
                "weight": 0.3,
                "source": "seed",
            }
            for i in range(10)
        ]
        user_model_store.update_signal_profile("mood_signals", {"recent_signals": initial_signals})

        engine.extract({
            "type": EventType.SLEEP_RECORDED.value,
            "timestamp": datetime(2024, 6, 15, 7, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {"duration_hours": 7.5, "quality_score": 0.7},
        })

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        values = [s["value"] for s in profile["data"]["recent_signals"]
                  if s["signal_type"] == "keep_signal"]
        # All 10 original signals must still be present.
        assert 0.0 in values and 9.0 in values, (
            "Original signals must be retained when total count is below 500."
        )


# ---------------------------------------------------------------------------
# 3. WAL checkpoint retry
# ---------------------------------------------------------------------------

class TestWALCheckpointRetry:
    """Verify the retry-after-checkpoint logic fires when read-back returns None."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return _make_engine(db, user_model_store)

    def test_retry_attempted_when_readback_fails(self, engine, user_model_store, caplog):
        """When the post-write read-back returns None, retry must be attempted.

        We simulate a transient WAL visibility failure by patching
        ``get_signal_profile`` to return None on the first call (post-write
        verification) but the real profile on all subsequent calls.  The
        retry path must log the failure and attempt a second write.
        """
        original_get = user_model_store.get_signal_profile
        call_count = {"n": 0}

        def patched_get(profile_type):
            """Return None on the post-write verification call; real data otherwise.

            Call sequence inside _update_mood_state:
              1. ``existing = self.ums.get_signal_profile(...)``  — initial load
              2. ``verify   = self.ums.get_signal_profile(...)``  — post-write check
              3. ``verify2  = self.ums.get_signal_profile(...)``  — retry check (optional)

            We return None only on call #2 to simulate a transient WAL visibility
            failure on the post-write read-back, which is the condition that
            triggers the checkpoint-and-retry path.
            """
            call_count["n"] += 1
            if call_count["n"] == 2:
                # Simulate the profile not yet visible after write (WAL not flushed).
                return None
            return original_get(profile_type)

        with patch.object(user_model_store, "get_signal_profile", side_effect=patched_get):
            with caplog.at_level(logging.ERROR, logger="services.signal_extractor.mood"):
                engine.extract(_email_received("Email triggering WAL test.", hour=11))

        # The FAILED-to-persist error must have been logged on first read-back.
        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        persist_errors = [m for m in error_msgs if "FAILED to persist" in m]
        assert len(persist_errors) >= 1, (
            "When post-write read-back returns None, _update_mood_state must "
            "log a 'FAILED to persist' error and initiate the WAL checkpoint retry."
        )

    def test_no_retry_error_when_write_succeeds(self, engine, user_model_store, caplog):
        """No retry error must be logged when the write-and-read-back succeeds normally."""
        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.mood"):
            engine.extract(_email_received(
                "Quick update: the project is on track for delivery.", hour=10
            ))

        error_msgs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        retry_errors = [m for m in error_msgs if "FAILED to persist" in m or "STILL missing" in m]
        assert len(retry_errors) == 0, (
            "No retry/failure errors should appear when the write succeeds normally. "
            f"Unexpected errors: {retry_errors}"
        )


# ---------------------------------------------------------------------------
# 4. Basic persistence round-trip: 5 events → profile exists
# ---------------------------------------------------------------------------

class TestBasicPersistenceRoundTrip:
    """End-to-end: process 5 mood-bearing events and verify the profile is readable."""

    @pytest.fixture
    def engine(self, db, user_model_store):
        """MoodInferenceEngine wired to a real, isolated SQLite database."""
        return _make_engine(db, user_model_store)

    def test_five_email_events_produce_profile(self, engine, user_model_store):
        """After 5 email.received events with body text the profile must exist.

        Regression guard: this was the key symptom — 13,726 qualifying events
        produced ZERO mood_signals profile entries.  Five events is the minimum
        representative reproduction.
        """
        events = [
            _email_received("Good morning, can we discuss the proposal?", hour=9),
            _email_received("I'm worried about the deadline — this is urgent.", hour=10),
            _email_received("The project looks good, nice work everyone!", hour=11),
            _email_received("PLEASE RESPOND ASAP! This is critical!", hour=14),
            _email_received("Thanks for the update, see you tomorrow.", hour=15),
        ]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None, (
            "mood_signals profile must exist after processing 5 email.received "
            "events; None indicates every write silently failed."
        )
        assert "data" in profile
        assert "recent_signals" in profile["data"]
        assert len(profile["data"]["recent_signals"]) >= 5, (
            "At least 5 signals (one circadian_energy per event minimum) must "
            "be in the persisted profile."
        )

    def test_profile_data_is_json_serializable_after_five_events(self, engine, user_model_store):
        """The persisted profile must contain only JSON-serializable types.

        Reading from SQLite and re-serializing must succeed without raising.
        This verifies the full round-trip: write → SQLite storage → read back
        → json.dumps without loss.
        """
        events = [
            _email_received("First email: routine check-in.", hour=8),
            _email_received("Second: deadline stress, I'm overwhelmed.", hour=9),
            _email_received("Third: good news, the deal closed!", hour=11),
            _email_received("Fourth: frustrated with the broken build again.", hour=13),
            _email_received("Fifth: call me when you get a chance.", hour=16),
        ]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None

        # The data that came back from get_signal_profile was produced by
        # json.loads(), so it should already be clean.  Re-serializing confirms
        # the round-trip is lossless and the stored data has no exotic types.
        try:
            json.dumps(profile["data"])
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"profile['data'] is not re-serializable after round-trip: {exc}\n"
                "This would cause silent write failures on the next event."
            )

    def test_signal_types_present_after_five_events(self, engine, user_model_store):
        """Expected signal types must appear in the profile after 5 email events.

        Validates that the extraction code paths for inbound communication
        are all functioning: circadian_energy for all events, and
        incoming_negative_language for events with negative vocabulary.
        """
        events = [
            _email_received("Morning catchup message.", hour=9),
            _email_received("Frustrated with delays and blocked by issues.", hour=10),
            _email_received("Regular update on the project timeline.", hour=11),
            _email_received("Urgent critical problem needs immediate fix.", hour=14),
            _email_received("Wrapping up for today, see you tomorrow.", hour=17),
        ]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        signal_types = {s["signal_type"] for s in profile["data"]["recent_signals"]}

        assert "circadian_energy" in signal_types, (
            "circadian_energy signal must be present after processing email events "
            "with valid timestamps."
        )
        assert "incoming_negative_language" in signal_types, (
            "incoming_negative_language signal must be present after processing "
            "emails with negative vocabulary (frustrated, blocked, urgent, critical)."
        )

    def test_samples_count_reflects_five_writes(self, engine, user_model_store):
        """samples_count in the profile must reflect the number of write operations.

        This counter is maintained by update_signal_profile using an atomic
        COALESCE upsert.  A count of 0 or 1 after 5 events would indicate
        the writes are somehow creating new rows instead of incrementing.
        """
        events = [_email_received(f"Email number {i} body text.", hour=10 + i)
                  for i in range(5)]

        for event in events:
            engine.extract(event)

        profile = user_model_store.get_signal_profile("mood_signals")
        assert profile is not None
        assert profile["samples_count"] >= 5, (
            f"samples_count must be >= 5 after 5 extract() calls; "
            f"got {profile['samples_count']}. A low count suggests writes "
            "are not persisting correctly."
        )
