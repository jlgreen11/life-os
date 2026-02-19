"""
Tests for relationship-to-contact metric denormalization.

After every signal-profile update, the RelationshipExtractor must write
three metrics back to the ``contacts`` table in ``entities.db``:

  - ``typical_response_time``   — average reply latency in seconds
  - ``last_contact``            — ISO timestamp of the most recent interaction
  - ``contact_frequency_days``  — average days between interactions

These columns have existed in the schema since the initial migration but
were never populated, leaving contact records incomplete.

Tests also cover the ``_compute_frequency_days`` helper and the
``_sync_contact_metrics`` method directly.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from models.core import EventType
from services.signal_extractor.relationship import (
    RelationshipExtractor,
    _compute_frequency_days,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _insert_contact(db, name: str, email: str) -> str:
    """Insert a contact and its email identifier; return the contact_id."""
    contact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("entities") as conn:
        conn.execute(
            """INSERT INTO contacts
                   (id, name, emails, phones, channels, created_at, updated_at)
               VALUES (?, ?, ?, '[]', '{}', ?, ?)""",
            (contact_id, name, json.dumps([email]), now, now),
        )
        conn.execute(
            """INSERT INTO contact_identifiers
                   (identifier, identifier_type, contact_id)
               VALUES (?, 'email', ?)""",
            (email.lower(), contact_id),
        )
    return contact_id


def _fetch_contact(db, contact_id: str) -> dict:
    """Return a single contact row as a dict."""
    with db.get_connection("entities") as conn:
        row = conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
    return dict(row) if row else {}


def _make_inbound_event(
    from_address: str,
    timestamp: str = "2026-02-10T08:00:00Z",
    body: str = "Hello",
) -> dict:
    """Create a minimal inbound email event."""
    return {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "proton",
        "timestamp": timestamp,
        "payload": {
            "from_address": from_address,
            "subject": "Re: test",
            "body": body,
            "channel": "email",
            "email_date": timestamp,
        },
    }


def _make_outbound_event(
    to_address: str,
    timestamp: str = "2026-02-10T09:00:00Z",
    body: str = "Hi back",
    is_reply: bool = True,
) -> dict:
    """Create a minimal outbound email event."""
    return {
        "id": str(uuid.uuid4()),
        "type": EventType.EMAIL_SENT.value,
        "source": "proton",
        "timestamp": timestamp,
        "payload": {
            "to_addresses": [to_address],
            "subject": "Re: test",
            "body": body,
            "channel": "email",
            "email_date": timestamp,
            "is_reply": is_reply,
        },
    }


# ---------------------------------------------------------------------------
# _compute_frequency_days helper tests
# ---------------------------------------------------------------------------


class TestComputeFrequencyDays:
    """Unit tests for the module-level _compute_frequency_days helper."""

    def test_returns_none_for_empty_list(self):
        """Must return None when no timestamps are available."""
        assert _compute_frequency_days([]) is None

    def test_returns_none_for_single_timestamp(self):
        """Must return None — can't compute a gap from one data point."""
        assert _compute_frequency_days(["2026-02-10T10:00:00Z"]) is None

    def test_two_timestamps_one_week_apart(self):
        """Two timestamps 7 days apart should yield 7.0."""
        ts1 = "2026-02-01T10:00:00Z"
        ts2 = "2026-02-08T10:00:00Z"
        result = _compute_frequency_days([ts1, ts2])
        assert result == pytest.approx(7.0, abs=0.01)

    def test_three_equal_gaps(self):
        """Three timestamps with equal 7-day gaps should average to 7.0."""
        ts = [
            "2026-02-01T10:00:00Z",
            "2026-02-08T10:00:00Z",
            "2026-02-15T10:00:00Z",
        ]
        result = _compute_frequency_days(ts)
        assert result == pytest.approx(7.0, abs=0.01)

    def test_unequal_gaps_averages_correctly(self):
        """Unequal gaps (3 and 11 days) should average to 7.0."""
        ts = [
            "2026-02-01T10:00:00Z",
            "2026-02-04T10:00:00Z",  # +3 days
            "2026-02-15T10:00:00Z",  # +11 days
        ]
        result = _compute_frequency_days(ts)
        assert result == pytest.approx(7.0, abs=0.01)

    def test_handles_z_suffix(self):
        """Timestamps ending in 'Z' must be parsed correctly."""
        ts = ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]
        result = _compute_frequency_days(ts)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_handles_duplicate_timestamps(self):
        """Duplicate timestamps produce zero gap and are filtered out."""
        ts = ["2026-02-01T10:00:00Z", "2026-02-01T10:00:00Z"]
        # Zero gaps are filtered; only 1 valid data point remains → None
        result = _compute_frequency_days(ts)
        # No positive gaps → None
        assert result is None

    def test_out_of_order_timestamps_sorted(self):
        """Ring buffer may contain out-of-order entries; result must still be correct."""
        ts = [
            "2026-02-08T10:00:00Z",  # second (out of order)
            "2026-02-01T10:00:00Z",  # first
        ]
        result = _compute_frequency_days(ts)
        assert result == pytest.approx(7.0, abs=0.01)

    def test_handles_malformed_timestamp_gracefully(self):
        """Malformed timestamps must not raise; should return None."""
        ts = ["not-a-timestamp", "also-bad"]
        result = _compute_frequency_days(ts)
        assert result is None

    def test_sub_day_frequency(self):
        """Interactions 12 hours apart should yield 0.5 days."""
        base = datetime(2026, 2, 10, 8, 0, 0, tzinfo=timezone.utc)
        ts = [
            base.isoformat().replace("+00:00", "Z"),
            (base + timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
        ]
        result = _compute_frequency_days(ts)
        assert result == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# _sync_contact_metrics method tests
# ---------------------------------------------------------------------------


class TestSyncContactMetrics:
    """Tests for RelationshipExtractor._sync_contact_metrics."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a RelationshipExtractor with test dependencies."""
        return RelationshipExtractor(db=db, user_model_store=user_model_store)

    def test_populates_last_contact_for_known_contact(self, db, extractor):
        """last_contact column is written after processing an inbound event."""
        cid = _insert_contact(db, "Alice", "alice@example.com")

        extractor.extract(_make_inbound_event("alice@example.com", "2026-02-10T08:00:00Z"))

        row = _fetch_contact(db, cid)
        assert row["last_contact"] == "2026-02-10T08:00:00Z"

    def test_populates_typical_response_time(self, db, extractor):
        """typical_response_time is written when the user replies to a contact."""
        cid = _insert_contact(db, "Bob", "bob@example.com")

        # Inbound at 08:00, outbound reply at 09:00 → 3600 seconds response time
        extractor.extract(_make_inbound_event("bob@example.com", "2026-02-10T08:00:00Z"))
        extractor.extract(
            _make_outbound_event(
                "bob@example.com",
                timestamp="2026-02-10T09:00:00Z",
                is_reply=True,
            )
        )

        row = _fetch_contact(db, cid)
        assert row["typical_response_time"] is not None
        assert row["typical_response_time"] == pytest.approx(3600.0, rel=0.01)

    def test_populates_contact_frequency_days(self, db, extractor):
        """contact_frequency_days is written with the average interaction gap."""
        cid = _insert_contact(db, "Carol", "carol@example.com")

        extractor.extract(_make_inbound_event("carol@example.com", "2026-02-01T10:00:00Z"))
        extractor.extract(_make_inbound_event("carol@example.com", "2026-02-08T10:00:00Z"))

        row = _fetch_contact(db, cid)
        assert row["contact_frequency_days"] is not None
        assert row["contact_frequency_days"] == pytest.approx(7.0, rel=0.01)

    def test_skips_unknown_addresses(self, db, extractor):
        """Contacts not present in entities.db are skipped silently."""
        # No contact created for unknown@example.com
        # Should not raise, profile should still update
        extractor.extract(_make_inbound_event("unknown@example.com", "2026-02-10T08:00:00Z"))

        with db.get_connection("entities") as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        assert cnt == 0  # No phantom contacts created

    def test_updates_are_idempotent_and_advance_on_new_data(self, db, extractor):
        """Processing additional events correctly updates the stored metrics."""
        cid = _insert_contact(db, "Dave", "dave@example.com")

        extractor.extract(_make_inbound_event("dave@example.com", "2026-02-01T08:00:00Z"))
        row1 = _fetch_contact(db, cid)
        assert row1["last_contact"] == "2026-02-01T08:00:00Z"

        extractor.extract(_make_inbound_event("dave@example.com", "2026-02-10T08:00:00Z"))
        row2 = _fetch_contact(db, cid)
        assert row2["last_contact"] == "2026-02-10T08:00:00Z"

    def test_case_insensitive_email_lookup(self, db, extractor):
        """Email lookup is case-insensitive (contact_identifiers lowercased)."""
        cid = _insert_contact(db, "Eve", "Eve@Example.COM")

        extractor.extract(_make_inbound_event("eve@example.com", "2026-02-10T08:00:00Z"))

        row = _fetch_contact(db, cid)
        assert row["last_contact"] == "2026-02-10T08:00:00Z"

    def test_multiple_contacts_updated_independently(self, db, extractor):
        """Each contact's metrics are independent — one doesn't pollute another."""
        cid_a = _insert_contact(db, "Alice", "alice@example.com")
        cid_b = _insert_contact(db, "Bob", "bob@example.com")

        extractor.extract(_make_inbound_event("alice@example.com", "2026-02-05T08:00:00Z"))
        extractor.extract(_make_inbound_event("bob@example.com", "2026-02-10T08:00:00Z"))

        row_a = _fetch_contact(db, cid_a)
        row_b = _fetch_contact(db, cid_b)

        assert row_a["last_contact"] == "2026-02-05T08:00:00Z"
        assert row_b["last_contact"] == "2026-02-10T08:00:00Z"

    def test_typical_response_time_none_without_reply(self, db, extractor):
        """typical_response_time stays None if the user has not replied."""
        cid = _insert_contact(db, "Frank", "frank@example.com")

        # Inbound only — no outbound reply, so no response time computable
        extractor.extract(_make_inbound_event("frank@example.com", "2026-02-10T08:00:00Z"))

        row = _fetch_contact(db, cid)
        # Response time should remain None (no reply processed)
        assert row["typical_response_time"] is None

    def test_contact_frequency_none_for_single_interaction(self, db, extractor):
        """contact_frequency_days is None after only one interaction (no gap to compute)."""
        cid = _insert_contact(db, "Grace", "grace@example.com")

        extractor.extract(_make_inbound_event("grace@example.com", "2026-02-10T08:00:00Z"))

        row = _fetch_contact(db, cid)
        # One data point → no gap → None
        assert row["contact_frequency_days"] is None

    def test_sync_does_not_overwrite_other_contact_fields(self, db, extractor):
        """Sync only touches the three metrics; name, emails, etc. remain unchanged."""
        cid = _insert_contact(db, "Henry", "henry@example.com")

        extractor.extract(_make_inbound_event("henry@example.com", "2026-02-10T08:00:00Z"))

        row = _fetch_contact(db, cid)
        assert row["name"] == "Henry"
        assert row["last_contact"] is not None  # updated

    def test_marketing_emails_not_written_to_contacts(self, db, extractor):
        """Marketing addresses are filtered before sync; no contact update occurs."""
        cid = _insert_contact(db, "Newsletter", "no-reply@marketing.com")

        extractor.extract(_make_inbound_event("no-reply@marketing.com"))

        row = _fetch_contact(db, cid)
        # Marketing filter blocks extraction → no last_contact update
        assert row["last_contact"] is None
