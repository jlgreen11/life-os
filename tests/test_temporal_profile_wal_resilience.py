"""
Life OS — Temporal Signal Extractor WAL Resilience Tests.

Verifies the three WAL-resilience improvements added to temporal.py:

1. Basic persistence: temporal profile is written and readable after event processing.
2. Retry on missing profile: when post-write verification detects a missing profile,
   ``update_signal_profile`` is called again and a WAL checkpoint is issued.
3. Cross-connection durability: after ``_force_wal_checkpoint()`` is called, the
   profile is readable through a fresh DatabaseManager pointing at the same files
   (simulates WAL frames being flushed into the main DB file after a process restart).
4. Cold-start verification interval: the verification fires every 5 writes for the
   first 20 writes (not every 10) so that persistence failures are caught sooner.
5. WAL checkpoint helper: ``_force_wal_checkpoint()`` does not raise on a healthy DB.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from models.core import EventType
from services.signal_extractor.temporal import TemporalExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _email_event(timestamp: datetime) -> dict:
    """Build a minimal email.received event with the given UTC timestamp.

    Args:
        timestamp: Timezone-aware datetime for the event.

    Returns:
        An event dict suitable for TemporalExtractor.extract().
    """
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "payload": {
            "from_address": "test@example.com",
            "subject": "WAL resilience test",
        },
    }


def _process_n_events(extractor: TemporalExtractor, n: int, base: datetime) -> None:
    """Process *n* email.received events spaced one hour apart from *base*.

    Args:
        extractor: TemporalExtractor instance under test.
        n: Number of events to process.
        base: Starting timestamp (timezone-aware).
    """
    for i in range(n):
        extractor.extract(_email_event(base + timedelta(hours=i)))


# ---------------------------------------------------------------------------
# Test 1: Basic persistence — profile written and readable
# ---------------------------------------------------------------------------


def test_temporal_profile_persists_after_extraction(db, user_model_store):
    """Temporal profile is written to the DB after processing a single qualifying event.

    This is the most fundamental check: if the extractor's persistence call succeeds,
    get_signal_profile must return a non-None result immediately afterward.
    """
    extractor = TemporalExtractor(db, user_model_store)
    base = datetime(2026, 3, 10, 9, 0, 0, tzinfo=UTC)

    extractor.extract(_email_event(base))

    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None, (
        "Temporal profile is None after processing one event. "
        "update_signal_profile() may be silently failing."
    )
    assert profile["samples_count"] == 1
    assert profile["data"]["activity_by_hour"].get("9") == 1
    assert profile["data"]["activity_by_type"].get("email_inbound") == 1


# ---------------------------------------------------------------------------
# Test 2: Retry mechanism — update_signal_profile is called again when profile is missing
# ---------------------------------------------------------------------------


def test_retry_write_called_on_missing_profile(db, user_model_store, caplog):
    """When post-write verification finds a missing profile, a retry write is triggered.

    Simulates the production scenario (WAL corruption causes profile loss) by
    monkey-patching get_signal_profile to return None during the verification
    check.  The test verifies:

    1. A CRITICAL log is emitted naming the missing profile.
    2. update_signal_profile is called *more* times than there are events
       (the extra call is the retry write).
    """
    extractor = TemporalExtractor(db, user_model_store)
    base = datetime(2026, 3, 10, 8, 0, 0, tzinfo=UTC)

    original_update = user_model_store.update_signal_profile
    original_get = user_model_store.get_signal_profile

    update_call_count = {"n": 0}

    def counting_update(profile_type: str, data: dict) -> None:
        """Count calls to update_signal_profile and delegate to the real method."""
        if profile_type == "temporal":
            update_call_count["n"] += 1
        return original_update(profile_type, data)

    def patched_get(profile_type: str):
        """Return None for the verification read (write count >= 5) to simulate loss.

        The load-existing call happens *before* the write-count increment, so
        at the moment of the load-existing call the count is still 4 (for the
        5th event).  The verification fires *after* the increment (count = 5),
        at which point we return None to simulate the profile being missing.
        """
        if profile_type == "temporal" and extractor._profile_write_count >= 5:
            return None
        return original_get(profile_type)

    user_model_store.update_signal_profile = counting_update
    user_model_store.get_signal_profile = patched_get

    try:
        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.temporal"):
            # 5 events → write count reaches 5.  Cold-start interval = 5,
            # so verification fires at write #5 (5 > 1 and 5 % 5 == 0).
            _process_n_events(extractor, 5, base)
    finally:
        # Always restore originals so subsequent tests are unaffected.
        user_model_store.update_signal_profile = original_update
        user_model_store.get_signal_profile = original_get

    # CRITICAL must have been emitted.
    critical_msgs = [
        r.message
        for r in caplog.records
        if r.levelno == logging.CRITICAL and "MISSING" in r.message
    ]
    assert critical_msgs, (
        "Expected a CRITICAL log about the temporal profile being MISSING "
        "after 5 writes, but none was emitted."
    )
    assert "Retrying write" in critical_msgs[0], (
        "CRITICAL message should mention that a retry is being attempted."
    )

    # update_signal_profile should have been called 6 times:
    # 5 normal writes + 1 retry write triggered by the recovery path.
    assert update_call_count["n"] == 6, (
        f"Expected 6 update_signal_profile calls (5 normal + 1 retry), "
        f"got {update_call_count['n']}."
    )


# ---------------------------------------------------------------------------
# Test 3: Cross-connection durability after WAL checkpoint
# ---------------------------------------------------------------------------


def test_profile_survives_separate_db_connection(db, user_model_store, tmp_data_dir):
    """Temporal profile is readable through a fresh DB connection after a WAL checkpoint.

    After ``_force_wal_checkpoint()`` is called, WAL frames are flushed into the
    main database file.  A new DatabaseManager pointing at the same data directory
    must be able to read the profile without seeing the WAL, simulating the
    behaviour after a process restart.
    """
    extractor = TemporalExtractor(db, user_model_store)
    base = datetime(2026, 3, 10, 10, 0, 0, tzinfo=UTC)

    # Write enough events to build a non-trivial profile.
    _process_n_events(extractor, 3, base)

    # Explicitly checkpoint so that all frames are in the main DB file.
    extractor._force_wal_checkpoint()

    # Open a completely independent connection to the same database directory.
    db2 = DatabaseManager(data_dir=tmp_data_dir)
    db2.initialize_all()
    ums2 = UserModelStore(db2)

    profile = ums2.get_signal_profile("temporal")
    assert profile is not None, (
        "Temporal profile is None when read through a fresh DB connection after "
        "_force_wal_checkpoint().  WAL frames may not have been flushed to the "
        "main database file."
    )
    assert profile["samples_count"] == 3, (
        f"Expected samples_count=3 across connections, got {profile['samples_count']}."
    )


# ---------------------------------------------------------------------------
# Test 4: Cold-start verification interval
# ---------------------------------------------------------------------------


def test_cold_start_verification_fires_at_write_5(db, user_model_store, caplog):
    """Verification fires every 5 writes during cold-start (first 20 writes).

    Before the fix, verification only fired every 10 writes, meaning a persistence
    failure after write #1 wouldn't be detected until write #10.  With the cold-start
    interval of 5, the failure is detected at write #5.

    This test verifies the interval by patching get_signal_profile to return None
    during verification calls once the write count reaches 5, then checking that
    the CRITICAL log is emitted before write #10 (i.e. at write #5).
    """
    extractor = TemporalExtractor(db, user_model_store)
    base = datetime(2026, 3, 11, 8, 0, 0, tzinfo=UTC)

    original_get = user_model_store.get_signal_profile
    critical_at_count = {"value": None}

    def patched_get(profile_type: str):
        """Return None for verification reads once cold-start interval is reached."""
        if profile_type == "temporal" and extractor._profile_write_count >= 5:
            return None
        return original_get(profile_type)

    user_model_store.get_signal_profile = patched_get

    try:
        # Process exactly 7 events so we can detect whether CRITICAL fired before #10.
        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.temporal"):
            for i in range(7):
                extractor.extract(_email_event(base + timedelta(hours=i)))
                # Record the write count the first time a CRITICAL is emitted.
                if critical_at_count["value"] is None:
                    critical_msgs = [
                        r for r in caplog.records
                        if r.levelno == logging.CRITICAL and "MISSING" in r.message
                    ]
                    if critical_msgs:
                        critical_at_count["value"] = extractor._profile_write_count
    finally:
        user_model_store.get_signal_profile = original_get

    assert critical_at_count["value"] is not None, (
        "Expected a CRITICAL log before write #10, but none was emitted in 7 events."
    )
    assert critical_at_count["value"] <= 5, (
        f"Expected CRITICAL at write #5 (cold-start interval), "
        f"but it fired at write #{critical_at_count['value']}."
    )


def test_verification_interval_is_10_after_cold_start(db, user_model_store):
    """After 20 writes, verification reverts to every 10 writes (not every 5).

    Processing exactly 25 events, we can detect whether the verification fires at
    write #25 (which would happen if the interval were still 5: 25 % 5 == 0) vs.
    not firing (correct behavior: 25 % 10 = 5 ≠ 0, so interval is 10).

    Key insight: with exactly 25 events, ``get_signal_profile`` is called at most
    once per event *for load-existing* (when ``_profile_write_count`` is still N-1)
    and once for *verification* (when ``_profile_write_count`` is N and the interval
    condition holds).  There is no load-existing call with count=25 unless a 26th
    event is processed.  Therefore, if count=25 appears in ``recorded_counts``,
    it must be a verification call — meaning the interval is still 5.
    """
    extractor = TemporalExtractor(db, user_model_store)
    base = datetime(2026, 3, 12, 6, 0, 0, tzinfo=UTC)

    original_get = user_model_store.get_signal_profile
    recorded_counts: list[int] = []

    def patched_get(profile_type: str):
        """Record the write count at every get_signal_profile call."""
        if profile_type == "temporal":
            recorded_counts.append(extractor._profile_write_count)
        return original_get(profile_type)

    user_model_store.get_signal_profile = patched_get

    try:
        _process_n_events(extractor, 25, base)
    finally:
        user_model_store.get_signal_profile = original_get

    # With exactly 25 events processed, count=25 can ONLY appear in recorded_counts
    # if a verification call fired at write #25.  That happens only if the interval
    # is 5 (wrong) — with interval=10 the verification would not fire at 25.
    assert 25 not in recorded_counts, (
        f"Verification fired at write #25, implying interval is still 5 (not 10) "
        f"after the cold-start period ended at write #20. "
        f"All recorded counts: {recorded_counts}"
    )

    # Also verify that the cold-start interval fired within writes 1-20.
    # With interval=5, verifications occur at write 5, 10, 15, 20.
    # Each appears in recorded_counts alongside load-existing calls.
    for expected in [5, 10, 15, 20]:
        assert expected in recorded_counts, (
            f"Expected a call with count={expected} during the cold-start period "
            f"(verification at write #{expected}, interval=5), "
            f"but {expected} not in recorded_counts={recorded_counts}."
        )


# ---------------------------------------------------------------------------
# Test 5: WAL checkpoint helper — does not raise on healthy DB
# ---------------------------------------------------------------------------


def test_force_wal_checkpoint_does_not_raise(db, user_model_store):
    """``_force_wal_checkpoint()`` completes without raising on a healthy database.

    Calls the helper before and after writing a profile to verify it is safe to
    call at any point in the extractor's lifecycle.
    """
    extractor = TemporalExtractor(db, user_model_store)

    # Should not raise even before any profile data exists.
    extractor._force_wal_checkpoint()

    # Write a profile, then checkpoint — the common recovery scenario.
    base = datetime(2026, 3, 10, 14, 0, 0, tzinfo=UTC)
    extractor.extract(_email_event(base))
    extractor._force_wal_checkpoint()

    # Profile must still be intact after checkpoint.
    profile = user_model_store.get_signal_profile("temporal")
    assert profile is not None, "Profile is None after _force_wal_checkpoint()."
    assert profile["samples_count"] == 1
