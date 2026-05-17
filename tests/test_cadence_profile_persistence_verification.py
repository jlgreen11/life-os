"""
Tests for CadenceExtractor post-write verification and WAL checkpoint.

Verifies that the ``cadence`` signal profile is correctly persisted after
processing communication events, that the post-write read-back surfaces silent
write failures as CRITICAL log entries, and that an opportunistic WAL
checkpoint after each write does not raise even when the underlying
``checkpoint_wal`` call fails.

This guards the same class of silent-persistence bugs that previously hit
``linguistic_inbound``, ``mood_signals``, the episode store, and prediction
persistence: ``update_signal_profile`` swallows exceptions with a warning, so
a JSON serialization error or WAL corruption could erase the cadence signal
(381K+ samples, the second-largest source in the system) with no obvious
trace.  The post-write read-back fires CRITICAL when persistence silently
fails so operators can investigate immediately.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from models.core import EventType
from services.signal_extractor.cadence import CadenceExtractor

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_outbound_email(
    to_address: str = "alice@example.com",
    hour: int = 10,
) -> dict:
    """Build a synthetic email.sent event.

    A simple outbound event without ``is_reply`` so the cadence extractor takes
    the activity-window + initiation tracking path.
    """
    return {
        "type": EventType.EMAIL_SENT.value,
        "source": "proton_mail",
        "timestamp": datetime(2024, 6, 1, hour, 0, 0, tzinfo=UTC).isoformat(),
        "payload": {
            "to_addresses": [to_address],
            "subject": "Test subject",
            "body": "Test body",
        },
    }


def _make_inbound_email(from_address: str = "bob@example.com", hour: int = 11) -> dict:
    """Build a synthetic email.received event for inbound tracking."""
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "proton_mail",
        "timestamp": datetime(2024, 6, 1, hour, 0, 0, tzinfo=UTC).isoformat(),
        "payload": {
            "from_address": from_address,
            "subject": "Re: Test subject",
            "body": "Reply body",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Happy-path: real DatabaseManager fixture, no mocks
# ──────────────────────────────────────────────────────────────────────────────


class TestCadenceProfilePersistenceHappyPath:
    """Verify normal persistence works against the real DatabaseManager fixture.

    Uses the real SQLite-backed ``user_model_store`` fixture from conftest.py
    so the WAL/journal semantics are exercised end-to-end rather than mocked.
    """

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """CadenceExtractor wired to a real, isolated SQLite database."""
        return CadenceExtractor(db=db, user_model_store=user_model_store)

    def test_profile_readable_immediately_after_write(self, extractor, user_model_store):
        """Cadence profile must be immediately readable after one event.

        This is the invariant the post-write verification asserts.  A failure
        here would mean the database is in a degraded state and the CRITICAL
        log in _update_profile() would fire.
        """
        extractor.extract(_make_outbound_email("alice@example.com"))

        profile = user_model_store.get_signal_profile("cadence")
        assert profile is not None, (
            "cadence profile must exist after processing one email.sent; "
            "a None result means _update_profile silently failed to write"
        )
        assert profile.get("data") is not None, (
            "cadence profile data must not be None after a successful write"
        )

    def test_profile_has_expected_keys(self, extractor, user_model_store):
        """Persisted profile must contain the core cadence aggregate keys."""
        extractor.extract(_make_outbound_email("alice@example.com"))

        profile = user_model_store.get_signal_profile("cadence")
        data = profile["data"]
        for key in (
            "response_times",
            "hourly_activity",
            "daily_activity",
            "per_contact_response_times",
            "per_channel_response_times",
            "per_contact_initiations",
            "per_contact_inbound_count",
        ):
            assert key in data, f"cadence profile data must contain {key!r}"

    def test_no_critical_log_for_normal_event(self, extractor, caplog):
        """A normal event must NOT trigger the post-write CRITICAL log.

        When persistence works correctly the post-write verification must not
        fire.  A CRITICAL log here means the read-back returned None, which is
        only expected when the underlying database is broken or json.dumps
        raised inside update_signal_profile.
        """
        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.cadence"):
            extractor.extract(_make_outbound_email("alice@example.com"))

        persistence_failures = [
            r.message for r in caplog.records
            if r.levelno >= logging.CRITICAL and "FAILED to persist" in r.message
        ]
        assert persistence_failures == [], (
            "No post-write CRITICAL expected for a normal email.sent event; "
            f"got: {persistence_failures}"
        )

    def test_no_warning_log_for_normal_event(self, extractor, caplog):
        """A normal event must not log a WAL checkpoint WARNING.

        The opportunistic checkpoint after a successful write must succeed
        against the real DatabaseManager — a WARNING here would indicate
        that ``db.checkpoint_wal`` itself is broken for the user_model DB.
        """
        with caplog.at_level(logging.WARNING, logger="services.signal_extractor.cadence"):
            extractor.extract(_make_outbound_email("alice@example.com"))

        checkpoint_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING and "WAL checkpoint" in r.message
        ]
        assert checkpoint_warnings == [], (
            "No WAL-checkpoint WARNING expected for a normal write; "
            f"got: {checkpoint_warnings}"
        )

    def test_multiple_events_persist_cumulatively(self, extractor, user_model_store):
        """Multiple inbound events must accumulate inbound counts without overwriting."""
        for sender in ("a@example.com", "b@example.com", "a@example.com"):
            extractor.extract(_make_inbound_email(sender))

        profile = user_model_store.get_signal_profile("cadence")
        assert profile is not None
        inbound_counts = profile["data"].get("per_contact_inbound_count", {})
        assert inbound_counts.get("a@example.com") == 2
        assert inbound_counts.get("b@example.com") == 1


# ──────────────────────────────────────────────────────────────────────────────
# Post-write verification: missing read-back logs CRITICAL
# ──────────────────────────────────────────────────────────────────────────────


class TestPostWriteVerificationCriticalLog:
    """Verify that a missing post-write read-back fires the CRITICAL log."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return CadenceExtractor(db=db, user_model_store=user_model_store)

    def test_missing_readback_logs_critical(self, extractor, caplog):
        """If get_signal_profile returns None after a write, log CRITICAL.

        We patch ``get_signal_profile`` to always return None so the
        verification path is exercised even though the write itself may
        have succeeded internally.  This simulates the production failure
        mode where ``update_signal_profile`` silently swallows an exception
        and the row never lands in the table.
        """
        with patch.object(
            extractor.ums, "get_signal_profile", return_value=None
        ), caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.cadence"):
            extractor.extract(_make_outbound_email("alice@example.com"))

        critical_records = [
            r for r in caplog.records
            if r.levelno >= logging.CRITICAL and "FAILED to persist" in r.message
        ]
        assert len(critical_records) >= 1, (
            "Expected at least one CRITICAL log when get_signal_profile "
            f"returns None after write; got: {[r.message for r in caplog.records]}"
        )

    def test_readback_with_none_data_logs_critical(self, extractor, caplog):
        """A read-back returning a dict whose 'data' is None must also log CRITICAL.

        Defensive guard: if the row exists but its payload is None (a
        partial write or JSON decode failure), the post-write verifier must
        still treat that as a persistence failure.
        """
        # _update_profile calls get_signal_profile twice: once at the start
        # to bootstrap ``data`` and once after the write to verify.  We let
        # the bootstrap pass through normally (return None → empty defaults)
        # and only the verify call returns the degraded payload, so the
        # write path itself isn't broken by the patch.
        call_count = {"n": 0}

        def fake_get(profile_type):
            call_count["n"] += 1
            # First call is the bootstrap inside _update_profile; return None
            # so the extractor builds a fresh data dict.  Subsequent calls
            # are the post-write verification — return the degraded payload.
            if call_count["n"] == 1:
                return None
            return {"data": None}

        with patch.object(
            extractor.ums, "get_signal_profile", side_effect=fake_get
        ), caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.cadence"):
            extractor.extract(_make_outbound_email("alice@example.com"))

        critical_records = [
            r for r in caplog.records
            if r.levelno >= logging.CRITICAL and "FAILED to persist" in r.message
        ]
        assert len(critical_records) >= 1, (
            "Expected CRITICAL log when read-back has data=None; "
            f"got: {[r.message for r in caplog.records]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# WAL checkpoint failure: logs WARNING, does not raise
# ──────────────────────────────────────────────────────────────────────────────


class TestWalCheckpointFailureNonFatal:
    """Verify that a checkpoint_wal failure is non-fatal and logs a WARNING."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return CadenceExtractor(db=db, user_model_store=user_model_store)

    def test_checkpoint_failure_logs_warning(self, extractor, caplog):
        """When checkpoint_wal raises, the extractor logs a WARNING.

        The opportunistic post-write checkpoint must never crash the
        extraction pipeline.  A failure here is a quality-of-service issue
        (the WAL file may grow), not a correctness issue, so it must
        downgrade to a WARNING.
        """
        with patch.object(
            extractor.ums.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ), caplog.at_level(logging.WARNING, logger="services.signal_extractor.cadence"):
            # Must not raise even though the checkpoint side-effect throws.
            extractor.extract(_make_outbound_email("alice@example.com"))

        checkpoint_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "WAL checkpoint" in r.message
        ]
        assert len(checkpoint_warnings) >= 1, (
            "Expected a WARNING when checkpoint_wal raises; "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_checkpoint_failure_does_not_raise(self, extractor):
        """checkpoint_wal raising must never propagate out of extract()."""
        with patch.object(
            extractor.ums.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            # Must not raise.
            try:
                extractor.extract(_make_outbound_email("alice@example.com"))
            except RuntimeError:
                pytest.fail(
                    "checkpoint_wal failure must not propagate out of extract(); "
                    "the opportunistic checkpoint is wrapped in try/except"
                )

    def test_checkpoint_failure_does_not_block_persistence(
        self, extractor, user_model_store
    ):
        """The profile must still be persisted when the WAL checkpoint fails.

        Checkpoint runs *after* the write, so a checkpoint failure must not
        roll back the persisted row.  This guards against any future
        refactor that accidentally chains the checkpoint into the write
        critical path.
        """
        with patch.object(
            extractor.ums.db, "checkpoint_wal", side_effect=RuntimeError("disk full")
        ):
            extractor.extract(_make_outbound_email("alice@example.com"))

        profile = user_model_store.get_signal_profile("cadence")
        assert profile is not None, (
            "Profile must still be persisted even when WAL checkpoint fails — "
            "the checkpoint is a post-write optimization, not part of the write"
        )
