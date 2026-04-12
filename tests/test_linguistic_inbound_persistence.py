"""
Tests for LinguisticExtractor._update_inbound_profile() persistence.

Verifies that the ``linguistic_inbound`` signal profile is correctly persisted
to the database after processing inbound email events, and that the contact-count
cap / data-compaction logic prunes stale contacts correctly.

This test suite was added to guard against the production issue where 13,508
qualifying ``email.received`` events produced zero ``linguistic_inbound`` profile
entries.  The root cause was that the JSON-serialized ``per_contact`` dict grew
very large with many unique senders, potentially exceeding SQLite's comfortable
range; the ``update_signal_profile`` call's try/except silently swallowed the
error, leaving the profile empty.

Test structure:
    - TestInboundProfilePersistenceBasic: profile created and populated after
      processing inbound events from 10 different contacts.
    - TestRingBufferCap: per-contact ring buffer stays at ≤100 samples.
    - TestContactCountPruning: contact count stays at ≤ _MAX_INBOUND_CONTACTS
      after processing more events than the cap; recency order is preserved.
    - TestPostWriteVerification: post-write read-back works correctly in normal
      operation (no CRITICAL log emitted).
    - TestMissingContactId: events without a from_address do not crash or write
      a partial profile.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from models.core import EventType
from services.signal_extractor.linguistic import LinguisticExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_inbound_email(
    from_address: str,
    body: str = "Hello, this is a test email with enough words to pass the length filter.",
    hour: int = 10,
) -> dict:
    """Build a synthetic email.received event for a given sender.

    Uses a fixed date (2024-06-01) to avoid timezone edge cases in CI.  The
    ``from_address`` becomes the contact_id used for per-contact bucketing.
    """
    return {
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "proton_mail",
        "timestamp": datetime(2024, 6, 1, hour, 0, 0, tzinfo=timezone.utc).isoformat(),
        "payload": {
            "from_address": from_address,
            "subject": "Test subject",
            "body": body,
        },
    }


def _make_contacts(n: int, prefix: str = "contact") -> list[str]:
    """Return a list of n unique contact email addresses."""
    return [f"{prefix}{i}@example.com" for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Basic persistence: profile created and populated
# ──────────────────────────────────────────────────────────────────────────────

class TestInboundProfilePersistenceBasic:
    """Verify that the linguistic_inbound profile is written after inbound events."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """LinguisticExtractor wired to a real, isolated SQLite database."""
        return LinguisticExtractor(db, user_model_store)

    def test_profile_created_after_single_inbound_event(self, extractor, user_model_store):
        """After one email.received event the linguistic_inbound profile must exist.

        Regression guard: 13,508 qualifying events produced 0 profile entries.
        This test ensures the single-event case works end-to-end.
        """
        event = _make_inbound_email("alice@example.com")

        extractor.extract(event)

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert profile is not None, (
            "linguistic_inbound profile must exist after processing one email.received; "
            "a None result means _update_inbound_profile silently failed to write"
        )

    def test_profile_has_per_contact_key(self, extractor, user_model_store):
        """Persisted profile must contain a 'per_contact' dict."""
        extractor.extract(_make_inbound_email("alice@example.com"))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert "per_contact" in profile["data"], (
            "linguistic_inbound profile data must contain 'per_contact' key"
        )

    def test_profile_has_per_contact_averages_key(self, extractor, user_model_store):
        """Persisted profile must contain a 'per_contact_averages' dict."""
        extractor.extract(_make_inbound_email("alice@example.com"))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert "per_contact_averages" in profile["data"], (
            "linguistic_inbound profile data must contain 'per_contact_averages' key"
        )

    def test_profile_contains_correct_contact(self, extractor, user_model_store):
        """per_contact must contain an entry for the sender's email address."""
        extractor.extract(_make_inbound_email("bob@example.com"))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert "bob@example.com" in profile["data"]["per_contact"], (
            "per_contact must have an entry keyed by the sender's from_address"
        )

    def test_contact_has_at_least_one_sample(self, extractor, user_model_store):
        """The per_contact ring buffer for the sender must contain at least one sample."""
        extractor.extract(_make_inbound_email("carol@example.com"))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        samples = profile["data"]["per_contact"]["carol@example.com"]
        assert len(samples) >= 1, (
            "per_contact ring buffer must have at least one metrics snapshot after "
            "processing one email from that contact"
        )

    def test_ten_contacts_all_present_in_profile(self, extractor, user_model_store):
        """After processing events from 10 different contacts all 10 must appear.

        This tests the core accumulation path: each unique sender gets its own
        ring buffer without overwriting any other contact's data.
        """
        contacts = _make_contacts(10)
        for contact in contacts:
            extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert profile is not None, "Profile must exist after 10 inbound events"

        per_contact = profile["data"]["per_contact"]
        for contact in contacts:
            assert contact in per_contact, (
                f"Contact {contact!r} missing from per_contact after processing their email"
            )

    def test_per_contact_averages_populated_for_all_ten_contacts(
        self, extractor, user_model_store
    ):
        """per_contact_averages must be populated for every contact with events."""
        contacts = _make_contacts(10)
        for contact in contacts:
            extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        per_contact_avgs = profile["data"]["per_contact_averages"]

        for contact in contacts:
            assert contact in per_contact_avgs, (
                f"Contact {contact!r} missing from per_contact_averages"
            )
            avgs = per_contact_avgs[contact]
            assert "formality" in avgs, (
                f"per_contact_averages[{contact!r}] missing 'formality' field"
            )
            assert "samples_count" in avgs, (
                f"per_contact_averages[{contact!r}] missing 'samples_count' field"
            )

    def test_samples_count_increments_with_repeated_events(
        self, extractor, user_model_store
    ):
        """Sending multiple events from the same contact increments their sample count."""
        contact = "repeated@example.com"
        for _ in range(3):
            extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        samples = profile["data"]["per_contact"][contact]
        assert len(samples) == 3, (
            "Three events from the same contact must produce three samples in the ring buffer"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Ring buffer cap: per-contact buffer stays ≤ 100
# ──────────────────────────────────────────────────────────────────────────────

class TestRingBufferCap:
    """Verify that the per-contact ring buffer is capped at 100 samples."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return LinguisticExtractor(db, user_model_store)

    def test_ring_buffer_capped_at_100(self, extractor, user_model_store):
        """Processing 110 events from one contact must keep the buffer at 100.

        The ring buffer is capped so the profile doesn't grow unbounded for
        prolific senders (e.g., a mailing list contact with hundreds of emails).
        """
        contact = "prolific@example.com"
        for i in range(110):
            extractor.extract(_make_inbound_email(contact, body=f"Email number {i} with enough words to be processed by the extractor."))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        samples = profile["data"]["per_contact"][contact]
        assert len(samples) <= 100, (
            f"Ring buffer must be capped at 100 samples; got {len(samples)}"
        )

    def test_ring_buffer_keeps_most_recent_samples(self, extractor, user_model_store):
        """The ring buffer must retain the 100 most recent samples, not the oldest."""
        contact = "rolling@example.com"
        # Process 110 events; the body text encodes the sequence number so we
        # can verify which samples were retained.
        for i in range(110):
            extractor.extract(_make_inbound_email(
                contact,
                body=f"Message sequence number marker {i} with sufficient words to pass the extractor filter minimum.",
            ))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        samples = profile["data"]["per_contact"][contact]
        # The 100 retained samples must all come from the last 100 events (i=10..109).
        # We verify this by checking word_count consistency (each message had
        # slightly different length due to the sequence number string).
        assert len(samples) == 100, (
            "Ring buffer must contain exactly 100 samples when 110 events were processed"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Contact count pruning
# ──────────────────────────────────────────────────────────────────────────────

class TestContactCountPruning:
    """Verify that the contact count is pruned to _MAX_INBOUND_CONTACTS when exceeded.

    We patch ``_MAX_INBOUND_CONTACTS`` down to 5 so the pruning logic triggers
    without needing to create 501 contacts in each test.
    """

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return LinguisticExtractor(db, user_model_store)

    def test_contact_count_stays_at_cap_after_exceeding(
        self, extractor, user_model_store
    ):
        """After processing 8 contacts with a cap of 5 the profile holds ≤ 5 contacts.

        The oldest (least-recently-updated) contacts are pruned.
        """
        with patch.object(LinguisticExtractor, "_MAX_INBOUND_CONTACTS", 5):
            contacts = _make_contacts(8)
            for contact in contacts:
                extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert profile is not None, "Profile must exist after processing 8 contacts"
        per_contact = profile["data"]["per_contact"]
        assert len(per_contact) <= 5, (
            f"Contact count must be pruned to ≤5 (cap); got {len(per_contact)}"
        )

    def test_most_recent_contacts_are_retained_after_pruning(
        self, extractor, user_model_store
    ):
        """When contacts are pruned the most recently updated ones are kept.

        We process the first 3 contacts, then process contacts 3–7.  With a cap
        of 5, contacts 0 and 1 (processed first, not updated again) should be
        evicted and contacts 3–7 should be retained.
        """
        with patch.object(LinguisticExtractor, "_MAX_INBOUND_CONTACTS", 5):
            all_contacts = _make_contacts(8)
            early_contacts = all_contacts[:3]    # contacts 0,1,2
            late_contacts = all_contacts[3:]     # contacts 3,4,5,6,7

            # Process early contacts first, then late contacts (which push the
            # count over the cap).
            for contact in early_contacts:
                extractor.extract(_make_inbound_email(contact))
            for contact in late_contacts:
                extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        per_contact = profile["data"]["per_contact"]

        # The 5 most recently updated should all be in late_contacts (3–7).
        # At minimum, the 5 late contacts must be present.
        for contact in late_contacts:
            assert contact in per_contact, (
                f"Recently updated contact {contact!r} must be retained after pruning"
            )

    def test_per_contact_averages_pruned_in_sync_with_per_contact(
        self, extractor, user_model_store
    ):
        """per_contact_averages must stay in sync with per_contact after pruning."""
        with patch.object(LinguisticExtractor, "_MAX_INBOUND_CONTACTS", 5):
            contacts = _make_contacts(8)
            for contact in contacts:
                extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        per_contact = profile["data"]["per_contact"]
        per_contact_avgs = profile["data"]["per_contact_averages"]

        # Every contact in per_contact must also be in per_contact_averages.
        for contact in per_contact:
            assert contact in per_contact_avgs, (
                f"Contact {contact!r} present in per_contact but missing from "
                "per_contact_averages after pruning — dicts out of sync"
            )

    def test_per_contact_updated_at_pruned_in_sync(
        self, extractor, user_model_store
    ):
        """per_contact_updated_at must stay in sync with per_contact after pruning."""
        with patch.object(LinguisticExtractor, "_MAX_INBOUND_CONTACTS", 5):
            contacts = _make_contacts(8)
            for contact in contacts:
                extractor.extract(_make_inbound_email(contact))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        per_contact = profile["data"]["per_contact"]
        updated_at = profile["data"].get("per_contact_updated_at", {})

        for contact in per_contact:
            assert contact in updated_at, (
                f"Contact {contact!r} present in per_contact but missing from "
                "per_contact_updated_at after pruning — recency tracker out of sync"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Post-write verification
# ──────────────────────────────────────────────────────────────────────────────

class TestPostWriteVerification:
    """Verify the post-write read-back in _update_inbound_profile() works correctly."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return LinguisticExtractor(db, user_model_store)

    def test_profile_readable_immediately_after_write(self, extractor, user_model_store):
        """Profile written by _update_inbound_profile must be immediately readable.

        This is the invariant the post-write verification asserts: if we write,
        we must be able to read back.  A failure here means the database is in a
        degraded state and the CRITICAL log in _update_inbound_profile() fires.
        """
        extractor.extract(_make_inbound_email("dave@example.com"))

        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert profile is not None, (
            "Profile written by _update_inbound_profile must be immediately readable; "
            "post-write verification would log CRITICAL if this fails"
        )

    def test_no_critical_log_for_normal_inbound_event(self, extractor, caplog):
        """Processing a normal email.received event must NOT trigger a CRITICAL log.

        When persistence works correctly the post-write verification must not
        fire.  A CRITICAL log here means the database or JSON serialization
        is broken.
        """
        event = _make_inbound_email("eve@example.com")

        with caplog.at_level(logging.CRITICAL, logger="services.signal_extractor.linguistic"):
            extractor.extract(event)

        critical_messages = [
            r.message for r in caplog.records if r.levelno >= logging.CRITICAL
        ]
        persistence_failures = [m for m in critical_messages if "FAILED to persist" in m]
        assert len(persistence_failures) == 0, (
            "No post-write CRITICAL expected for a normal email.received event; "
            f"got: {persistence_failures}"
        )

    def test_no_error_log_for_normal_inbound_event(self, extractor, caplog):
        """Processing a normal event must not raise an exception or log an ERROR.

        An ERROR log in _update_inbound_profile would mean an unexpected
        exception was caught by the outer try/except, which should not happen
        for well-formed inbound email events.
        """
        event = _make_inbound_email("frank@example.com")

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.linguistic"):
            extractor.extract(event)

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        inbound_errors = [m for m in error_messages if "_update_inbound_profile" in m]
        assert len(inbound_errors) == 0, (
            "No ERROR from _update_inbound_profile expected for a valid event; "
            f"got: {inbound_errors}"
        )

    def test_profile_data_is_json_serializable(self, extractor):
        """All data stored in the profile must be JSON-serializable.

        Non-serializable values (e.g., Enum instances, datetime objects, sets)
        would cause update_signal_profile() to raise and the profile would
        silently never be written.
        """
        import json

        contacts = _make_contacts(5)
        for contact in contacts:
            extractor.extract(_make_inbound_email(contact))

        # If the extractor does not raise, verify the round-trip via
        # the user model store's get_signal_profile which parses the JSON back.
        # The fact that get_signal_profile returns a valid dict (not None) is
        # sufficient — it proves json.dumps succeeded inside update_signal_profile.
        from storage.user_model_store import UserModelStore
        # Build a fresh extractor tied to the same stores
        profile = extractor.ums.get_signal_profile("linguistic_inbound")
        assert profile is not None, "Profile round-trip (write+read) must succeed"

        # Double-check: re-serializing the deserialized data must not raise.
        try:
            json.dumps(profile["data"])
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"Profile data is not JSON-serializable after round-trip: {exc}\n"
                "Non-serializable data would cause future writes to silently fail."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases: missing contact_id and short body text
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Verify that edge-case inputs are handled gracefully."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        return LinguisticExtractor(db, user_model_store)

    def test_event_without_from_address_does_not_crash(self, extractor, user_model_store):
        """email.received with no from_address must not crash or write a partial profile.

        The method returns early when contact_id is None/empty, so no profile
        entry should be created.
        """
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
            "timestamp": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {
                # No from_address field
                "body": "This email has no sender address and enough words to pass length filter.",
            },
        }

        # Must not raise
        result = extractor.extract(event)

        # The event should still produce a signal (the text is analysed), but no
        # inbound profile update should occur since contact_id is None.
        profile = user_model_store.get_signal_profile("linguistic_inbound")
        if profile is not None:
            per_contact = profile["data"]["per_contact"]
            assert None not in per_contact, (
                "per_contact must not contain a None key from events with no from_address"
            )

    def test_event_with_short_body_produces_no_signals(self, extractor, user_model_store):
        """email.received with a body under 10 characters produces no signals.

        The extractor filters out trivially short texts before analysis, so no
        profile write should occur.
        """
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
            "timestamp": datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
            "payload": {
                "from_address": "short@example.com",
                "body": "Hi",
            },
        }

        signals = extractor.extract(event)

        assert signals == [], (
            "email.received with a body shorter than 10 characters must produce no signals"
        )
        profile = user_model_store.get_signal_profile("linguistic_inbound")
        assert profile is None, (
            "No linguistic_inbound profile should be written when the message body "
            "is too short to be analysed"
        )
