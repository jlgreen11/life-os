"""
Tests for relationship extractor auto-contact-creation in _sync_contact_metrics.

Verifies that email addresses observed enough times (>= MIN_INTERACTIONS_FOR_AUTO_CREATE)
are automatically inserted into entities.db as minimal contact stubs, ensuring the
People Radar is populated for email-only users who haven't configured a Contacts connector.
"""

import json
from datetime import datetime, timezone, timedelta

import pytest

from services.signal_extractor.relationship import (
    RelationshipExtractor,
    MIN_INTERACTIONS_FOR_AUTO_CREATE,
    _name_from_email,
)
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(interaction_count: int, timestamps: list[str] | None = None) -> dict:
    """Build a minimal relationship profile dict as stored in the signal profile."""
    now = datetime.now(timezone.utc).isoformat()
    ts = timestamps or [
        (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()
        for i in range(interaction_count)
    ]
    return {
        "interaction_count": interaction_count,
        "interaction_timestamps": ts[:interaction_count],
        "last_interaction": ts[0] if ts else now,
        "avg_response_time_seconds": 3600.0,
        "avg_message_length": 50.0,
        "inbound_count": interaction_count // 2,
        "outbound_count": interaction_count - interaction_count // 2,
    }


def _count_contacts(db) -> int:
    """Return the total number of rows in entities.db contacts table."""
    with db.get_connection("entities") as conn:
        return conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]


def _get_contact_by_email(db, email: str) -> dict | None:
    """Fetch a contact record by email address, or None if not found."""
    with db.get_connection("entities") as conn:
        row = conn.execute(
            """SELECT c.* FROM contacts c
               JOIN contact_identifiers ci ON ci.contact_id = c.id
              WHERE ci.identifier_type = 'email'
                AND lower(ci.identifier) = lower(?)""",
            (email,),
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# _name_from_email unit tests
# ---------------------------------------------------------------------------

def test_name_from_email_dotted():
    """Dotted address should produce capitalised first + last name."""
    assert _name_from_email("john.doe@example.com") == "John Doe"


def test_name_from_email_underscore():
    """Underscore-separated address should be capitalised."""
    assert _name_from_email("jane_smith@company.org") == "Jane Smith"


def test_name_from_email_plain():
    """Plain local part with no separators should be capitalised as-is."""
    assert _name_from_email("alice@example.com") == "Alice"


def test_name_from_email_hyphen():
    """Hyphen separator should produce multi-word name."""
    assert _name_from_email("bob-jones@example.com") == "Bob Jones"


# ---------------------------------------------------------------------------
# _sync_contact_metrics auto-creation integration tests
# ---------------------------------------------------------------------------

def test_auto_creates_contact_at_threshold(db):
    """A contact stub is created when interaction count reaches MIN_INTERACTIONS_FOR_AUTO_CREATE."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    addr = "alice@example.com"
    contacts_data = {addr: _make_profile(interaction_count=MIN_INTERACTIONS_FOR_AUTO_CREATE)}

    assert _count_contacts(db) == 0
    extractor._sync_contact_metrics(contacts_data)
    assert _count_contacts(db) == 1

    contact = _get_contact_by_email(db, addr)
    assert contact is not None
    assert contact["name"] == "Alice"
    assert addr in json.loads(contact["emails"])


def test_skips_contact_below_threshold(db):
    """Contacts with fewer than MIN_INTERACTIONS_FOR_AUTO_CREATE interactions are not created."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    addr = "oneoff@example.com"
    contacts_data = {addr: _make_profile(interaction_count=MIN_INTERACTIONS_FOR_AUTO_CREATE - 1)}

    extractor._sync_contact_metrics(contacts_data)
    assert _count_contacts(db) == 0


def test_metrics_written_after_auto_creation(db):
    """Relationship metrics (last_contact, contact_frequency_days) are written on auto-created contacts."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    now = datetime.now(timezone.utc)
    timestamps = [
        (now - timedelta(days=i * 3)).isoformat()
        for i in range(5)
    ]
    addr = "frequent@example.com"
    profile = _make_profile(interaction_count=5, timestamps=timestamps)

    extractor._sync_contact_metrics({addr: profile})

    contact = _get_contact_by_email(db, addr)
    assert contact is not None
    # Frequency should be ~3 days between 5 timestamps spaced 3 days apart
    assert contact["contact_frequency_days"] is not None
    assert 2.5 <= contact["contact_frequency_days"] <= 3.5


def test_idempotent_auto_creation(db):
    """Running _sync_contact_metrics twice for the same address creates only one contact."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    addr = "idempotent@example.com"
    contacts_data = {addr: _make_profile(interaction_count=MIN_INTERACTIONS_FOR_AUTO_CREATE)}

    extractor._sync_contact_metrics(contacts_data)
    extractor._sync_contact_metrics(contacts_data)

    assert _count_contacts(db) == 1


def test_existing_contact_is_updated_not_duplicated(db):
    """When a contact already exists via contact_identifiers, it is updated, not duplicated."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    # Pre-create the contact (simulating a connector having created it)
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO contacts (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("existing-id", "Known Person", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO contact_identifiers (identifier, identifier_type, contact_id) VALUES (?, 'email', ?)",
            ("known@example.com", "existing-id"),
        )

    addr = "known@example.com"
    contacts_data = {addr: _make_profile(interaction_count=10)}

    extractor._sync_contact_metrics(contacts_data)

    # Still only one contact
    assert _count_contacts(db) == 1

    # Metrics were updated on the existing record
    contact = _get_contact_by_email(db, addr)
    assert contact["id"] == "existing-id"
    assert contact["last_contact"] is not None


def test_multiple_addresses_mixed_threshold(db):
    """Only addresses meeting the threshold are created; others are skipped."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    contacts_data = {
        "frequent@example.com": _make_profile(interaction_count=5),
        "oneoff@example.com": _make_profile(interaction_count=1),
        "borderline@example.com": _make_profile(interaction_count=MIN_INTERACTIONS_FOR_AUTO_CREATE),
    }

    extractor._sync_contact_metrics(contacts_data)

    assert _count_contacts(db) == 2  # frequent + borderline; oneoff skipped

    assert _get_contact_by_email(db, "frequent@example.com") is not None
    assert _get_contact_by_email(db, "borderline@example.com") is not None
    assert _get_contact_by_email(db, "oneoff@example.com") is None


def test_deterministic_contact_id(db):
    """Two separate runs produce the same contact_id for the same email address."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    addr = "stable@example.com"
    contacts_data = {addr: _make_profile(interaction_count=MIN_INTERACTIONS_FOR_AUTO_CREATE)}

    extractor._sync_contact_metrics(contacts_data)
    contact1 = _get_contact_by_email(db, addr)

    # Second invocation — INSERT OR IGNORE keeps the original row
    extractor._sync_contact_metrics(contacts_data)
    contact2 = _get_contact_by_email(db, addr)

    assert contact1["id"] == contact2["id"]


def test_name_derived_from_email_in_contact(db):
    """Auto-created contact name is derived from the email local part."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    extractor._sync_contact_metrics(
        {"sarah.jones@company.com": _make_profile(interaction_count=3)}
    )

    contact = _get_contact_by_email(db, "sarah.jones@company.com")
    assert contact is not None
    assert contact["name"] == "Sarah Jones"


def test_empty_contacts_data_does_nothing(db):
    """Passing an empty dict to _sync_contact_metrics is a no-op."""
    ums = UserModelStore(db)
    extractor = RelationshipExtractor(db=db, user_model_store=ums)

    extractor._sync_contact_metrics({})
    assert _count_contacts(db) == 0
