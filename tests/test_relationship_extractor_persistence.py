"""Persistence hardening tests for RelationshipExtractor.

Covers the sampled post-write verification path added to
``services/signal_extractor/relationship.py`` to protect the
highest-volume signal profile in the system (~8.8M samples) from silent
write failure without paying the cost of always-on verification.

Three behaviours are exercised:

1. **Pre-write JSON serialization guard** — when ``data`` contains a value
   that ``json.dumps`` cannot handle, the extractor must log an error tagged
   ``relationship_extractor:`` and skip the write (rather than letting
   ``UserModelStore.update_signal_profile`` silently swallow the
   ``TypeError`` and clobber the prior good profile).

2. **Write counter** — ``_profile_write_count`` increments by one for each
   successful call to ``_update_contact_profiles`` (i.e. each call that
   reaches the ``update_signal_profile`` call site).

3. **Verification sampling** — verification only runs once every
   ``PROFILE_VERIFY_SAMPLE_INTERVAL`` writes. In particular, the 99th
   write must NOT trigger a verify call, and the 100th write must.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models.core import EventType
from services.signal_extractor.relationship import (
    PROFILE_VERIFY_SAMPLE_INTERVAL,
    RelationshipExtractor,
)


def _make_event(seq: int = 0) -> dict:
    """Build a minimal inbound email event that the extractor will process.

    ``seq`` lets each test produce a unique address per call so the contact
    table does not collapse signals onto a single row.
    """
    return {
        "id": f"evt-{seq:05d}",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "gmail",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "from_address": f"person{seq}@example.com",
            "from_name": f"Person {seq}",
            "channel": "email",
            "body": "Hello there, this is a test email body.",
            "is_reply": False,
        },
    }


class TestPreWriteJSONGuard:
    """The pre-write guard must catch non-serializable payloads."""

    def test_non_serializable_set_in_contacts_skips_write(self, db, user_model_store, caplog):
        """Injecting a ``set`` into the contacts payload must skip the write."""
        extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

        # Replace get_signal_profile so the initial load returns a payload
        # containing a set — which json.dumps cannot serialize. The set lives
        # inside contacts[<addr>]['channels_used'], a value that the extractor
        # legitimately appends to in normal operation.
        bad_profile = {
            "data": {
                "contacts": {
                    "alice@example.com": {
                        "interaction_count": 1,
                        "inbound_count": 1,
                        "outbound_count": 0,
                        # The poison pill: a set, which json.dumps refuses.
                        "channels_used": {"email"},
                        "avg_message_length": 0,
                        "last_interaction": None,
                        "last_inbound_timestamp": None,
                        "interaction_timestamps": [],
                        "response_times_seconds": [],
                        "avg_response_time_seconds": None,
                    }
                }
            }
        }
        update_mock = MagicMock()
        extractor.ums = MagicMock()
        extractor.ums.get_signal_profile.return_value = bad_profile
        extractor.ums.update_signal_profile = update_mock

        with caplog.at_level("ERROR", logger="services.signal_extractor.relationship"):
            extractor.extract(_make_event(seq=1))

        # Write was skipped — update_signal_profile must NOT have been called.
        assert update_mock.call_count == 0, "write must be skipped when payload is non-serializable"
        # And the counter must NOT have advanced.
        assert extractor._profile_write_count == 0
        # The log must be tagged so operators can grep for it, and must
        # identify the offending top-level key (contacts) by name.
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "relationship_extractor:" in joined
        assert "contacts" in joined


class TestWriteCountIncrement:
    """``_profile_write_count`` must advance once per successful write."""

    def test_counter_starts_at_zero(self, db, user_model_store):
        """The counter is bootstrapped at zero in __init__."""
        extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)
        assert extractor._profile_write_count == 0

    def test_counter_increments_across_many_extractions(self, db, user_model_store):
        """Driving 105 distinct events bumps the counter to 105.

        Uses the real UserModelStore so the read-after-write succeeds and
        the verification branch (when sampled) does not enter the retry
        path — the counter must still reflect every write.
        """
        extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)
        # Each call to extract drives _update_contact_profiles exactly once,
        # so 105 events should produce 105 writes.
        for i in range(105):
            extractor.extract(_make_event(seq=i))
        assert extractor._profile_write_count == 105


class TestSampledVerification:
    """Verification calls must be sampled, not on every write."""

    @pytest.fixture
    def sampling_extractor(self, db, user_model_store):
        """An extractor whose ums is mocked to count read calls precisely.

        ``get_signal_profile`` returns ``None`` for the first call (so the
        extractor bootstraps an empty profile dict and exercises the write
        path) and a non-empty profile for every subsequent call (so verify
        succeeds and never enters the retry branch — we just want to count
        whether verify happened at all on each write).
        """
        extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)
        good_profile = {"data": {"contacts": {}}}
        # First call bootstraps; subsequent calls return a valid profile so
        # the verification branch sees a successful read-back.
        extractor.ums = MagicMock()
        extractor.ums.get_signal_profile.return_value = good_profile
        extractor.ums.update_signal_profile = MagicMock()
        return extractor

    def test_no_verify_before_sample_interval(self, sampling_extractor):
        """Writes 1..99 must not trigger an extra verify read.

        Each call to ``_update_contact_profiles`` reads the prior profile
        exactly once (the bootstrap read at the top of the method). The
        sampled verification path adds one *additional* read on writes that
        land on the sampling interval. So after 99 writes the read count
        equals the write count — no extra verifies have happened yet.
        """
        for i in range(PROFILE_VERIFY_SAMPLE_INTERVAL - 1):  # 99 writes
            sampling_extractor.extract(_make_event(seq=i))

        assert sampling_extractor._profile_write_count == PROFILE_VERIFY_SAMPLE_INTERVAL - 1
        # 99 bootstrap reads, 0 verify reads => 99 total.
        assert sampling_extractor.ums.get_signal_profile.call_count == PROFILE_VERIFY_SAMPLE_INTERVAL - 1

    def test_verify_runs_on_sample_interval(self, sampling_extractor):
        """The 100th write must trigger exactly one additional verify read.

        100 writes = 100 bootstrap reads + 1 verify read = 101 total reads.
        """
        for i in range(PROFILE_VERIFY_SAMPLE_INTERVAL):  # 100 writes
            sampling_extractor.extract(_make_event(seq=i))

        assert sampling_extractor._profile_write_count == PROFILE_VERIFY_SAMPLE_INTERVAL
        # 100 bootstrap reads + 1 verify read on the 100th write.
        assert sampling_extractor.ums.get_signal_profile.call_count == PROFILE_VERIFY_SAMPLE_INTERVAL + 1
