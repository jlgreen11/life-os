"""
Tests for PredictionEngine contact name resolution from entities.db.

Verifies that prediction descriptions and suggested actions display the
real contact name (from the entities.db contacts table) instead of the
crude email-prefix heuristic (e.g. "John Smith" instead of "jsmith").
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_contact(db, name: str, emails: list[str]) -> str:
    """Insert a contact into entities.db and return its ID."""
    contact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("entities") as conn:
        conn.execute(
            """INSERT INTO contacts (id, name, emails, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (contact_id, name, json.dumps(emails), now, now),
        )
    return contact_id


def _insert_email_event(db, event_type: str, from_address: str, to_addresses: list[str],
                        timestamp: datetime, message_id: str | None = None) -> None:
    """Insert an email event directly into events.db."""
    eid = str(uuid.uuid4())
    payload = {
        "message_id": message_id or f"msg-{uuid.uuid4().hex[:8]}",
        "from_address": from_address,
        "to_addresses": to_addresses,
        "subject": "Test subject",
        "snippet": "Test body content",
    }
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (eid, event_type, "google", timestamp.isoformat(), "normal",
             json.dumps(payload), json.dumps({})),
        )


# ---------------------------------------------------------------------------
# Tests: _resolve_contact_name
# ---------------------------------------------------------------------------


class TestResolveContactName:
    """Tests for PredictionEngine._resolve_contact_name()."""

    def test_resolve_contact_name_from_entities_db(self, db, user_model_store):
        """Known contact email should resolve to their stored name."""
        _insert_contact(db, "John Smith", ["jsmith@company.com"])
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        result = engine._resolve_contact_name("jsmith@company.com")
        assert result == "John Smith"

    def test_resolve_contact_name_fallback_on_unknown_email(self, db, user_model_store):
        """Unknown email should fall back to the local part (before @)."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        result = engine._resolve_contact_name("unknown@example.com")
        assert result == "unknown"

    def test_resolve_contact_name_case_insensitive(self, db, user_model_store):
        """Email matching should be case-insensitive."""
        _insert_contact(db, "Jane Doe", ["JDoe@Company.COM"])
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        result = engine._resolve_contact_name("jdoe@company.com")
        assert result == "Jane Doe"

    def test_resolve_contact_name_with_leading_trailing_spaces(self, db, user_model_store):
        """Email lookup should be tolerant of whitespace."""
        _insert_contact(db, "Alice Wonder", ["alice@example.com"])
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        result = engine._resolve_contact_name("  alice@example.com  ")
        assert result == "Alice Wonder"

    def test_resolve_contact_name_multiple_emails_per_contact(self, db, user_model_store):
        """A contact with multiple emails should be findable by any of them."""
        _insert_contact(db, "Bob Builder", ["bob@work.com", "bob@personal.com"])
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        assert engine._resolve_contact_name("bob@work.com") == "Bob Builder"
        assert engine._resolve_contact_name("bob@personal.com") == "Bob Builder"

    def test_resolve_contact_name_no_at_sign(self, db, user_model_store):
        """An address without @ should return the whole string as fallback."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        result = engine._resolve_contact_name("localuser")
        assert result == "localuser"

    def test_resolve_contact_name_cache_refresh(self, db, user_model_store):
        """Cache should refresh after 30 minutes, picking up new contacts."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        # Initially no contact — should fall back
        assert engine._resolve_contact_name("new@example.com") == "new"

        # Add the contact
        _insert_contact(db, "New Person", ["new@example.com"])

        # Cache is still fresh, so it won't see the new contact yet
        assert engine._resolve_contact_name("new@example.com") == "new"

        # Force cache expiry by backdating the loaded_at timestamp
        engine._contact_email_map_loaded_at = datetime.now(timezone.utc) - timedelta(minutes=31)

        # Now it should pick up the new contact
        assert engine._resolve_contact_name("new@example.com") == "New Person"


# ---------------------------------------------------------------------------
# Tests: _load_contact_email_map
# ---------------------------------------------------------------------------


class TestLoadContactEmailMap:
    """Tests for PredictionEngine._load_contact_email_map()."""

    def test_loads_all_contacts(self, db, user_model_store):
        """Should load all contacts and index by email."""
        _insert_contact(db, "Alice", ["alice@example.com"])
        _insert_contact(db, "Bob", ["bob@example.com", "bob@work.com"])

        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        engine._load_contact_email_map()

        assert engine._contact_email_map["alice@example.com"] == "Alice"
        assert engine._contact_email_map["bob@example.com"] == "Bob"
        assert engine._contact_email_map["bob@work.com"] == "Bob"

    def test_handles_empty_contacts_table(self, db, user_model_store):
        """Should produce an empty map when no contacts exist."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        engine._load_contact_email_map()

        assert engine._contact_email_map == {}

    def test_handles_malformed_emails_json(self, db, user_model_store):
        """Should skip contacts with unparseable emails JSON gracefully."""
        now = datetime.now(timezone.utc).isoformat()
        with db.get_connection("entities") as conn:
            conn.execute(
                """INSERT INTO contacts (id, name, emails, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), "Bad Data", "not-valid-json", now, now),
            )
        _insert_contact(db, "Good Contact", ["good@example.com"])

        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        engine._load_contact_email_map()

        # The good contact should still be loaded
        assert engine._contact_email_map["good@example.com"] == "Good Contact"


# ---------------------------------------------------------------------------
# Tests: Follow-up prediction uses resolved name
# ---------------------------------------------------------------------------


class TestFollowUpPredictionContactName:
    """Verify that _check_follow_up_needs() uses resolved contact names."""

    @pytest.mark.asyncio
    async def test_follow_up_prediction_uses_resolved_name(self, db, user_model_store):
        """Follow-up prediction description and action should contain the real name."""
        # Insert a known contact
        _insert_contact(db, "John Smith", ["jsmith@company.com"])

        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        # Insert an inbound email from 6 hours ago (past the 3h minimum)
        msg_time = datetime.now(timezone.utc) - timedelta(hours=6)
        msg_id = "msg-test-followup-001"
        _insert_email_event(
            db, "email.received", "jsmith@company.com", ["me@example.com"],
            msg_time, message_id=msg_id,
        )

        predictions = await engine._check_follow_up_needs({})

        assert len(predictions) >= 1
        pred = predictions[0]

        # Description should use the real name
        assert "John Smith" in pred.description
        assert "jsmith@company.com" not in pred.description

        # Suggested action should include both name and email
        assert "John Smith" in pred.suggested_action
        assert "jsmith@company.com" in pred.suggested_action

        # Supporting signals should have the resolved name
        assert pred.supporting_signals["contact_name"] == "John Smith"
        assert pred.supporting_signals["contact_email"] == "jsmith@company.com"

    @pytest.mark.asyncio
    async def test_follow_up_prediction_falls_back_for_unknown_contact(self, db, user_model_store):
        """Unknown contacts should show the email prefix in predictions."""
        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")

        # Insert an inbound email from an unknown sender
        msg_time = datetime.now(timezone.utc) - timedelta(hours=6)
        _insert_email_event(
            db, "email.received", "stranger@unknown.org", ["me@example.com"],
            msg_time, message_id="msg-test-followup-002",
        )

        predictions = await engine._check_follow_up_needs({})

        assert len(predictions) >= 1
        pred = predictions[0]

        # Should fall back to email prefix
        assert pred.supporting_signals["contact_name"] == "stranger"
        assert "stranger" in pred.description


# ---------------------------------------------------------------------------
# Tests: Relationship maintenance prediction uses resolved name
# ---------------------------------------------------------------------------


class TestRelationshipMaintenanceContactName:
    """Verify that _check_relationship_maintenance() uses resolved contact names."""

    @pytest.mark.asyncio
    async def test_relationship_prediction_uses_resolved_name(self, db, user_model_store):
        """Relationship maintenance predictions should use the real contact name."""
        # Insert a known contact
        _insert_contact(db, "Alice Wonderland", ["alice@example.com"])

        # Set up a signal profile with a contact whose gap exceeds threshold
        now = datetime.now(timezone.utc)
        timestamps = [(now - timedelta(days=60 - i * 5)).isoformat() for i in range(8)]
        signal_data = {
            "contacts": {
                "alice@example.com": {
                    "interaction_count": 8,
                    "last_interaction": (now - timedelta(days=30)).isoformat(),
                    "outbound_count": 4,
                    "interaction_timestamps": timestamps,
                }
            }
        }
        user_model_store.get_signal_profile = MagicMock(return_value={"data": signal_data})

        engine = PredictionEngine(db=db, ums=user_model_store, timezone="UTC")
        predictions = await engine._check_relationship_maintenance({})

        assert len(predictions) >= 1
        pred = predictions[0]

        # Description should use the real name
        assert "Alice Wonderland" in pred.description
        assert "alice@example.com" not in pred.description

        # Suggested action should include both name and email
        assert "Alice Wonderland" in pred.suggested_action
        assert "alice@example.com" in pred.suggested_action

        # Supporting signals should have the resolved name
        assert pred.supporting_signals["contact_name"] == "Alice Wonderland"
        assert pred.supporting_signals["contact_email"] == "alice@example.com"
