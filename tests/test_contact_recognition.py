"""
Tests for improved contact recognition:
  1. Natural language relationship parsing in onboarding
  2. Email domain classification (personal vs business)
  3. Display name capture in connectors
  4. Display name usage in RelationshipExtractor auto-contact creation
"""

import json
from datetime import UTC, datetime, timedelta

from services.onboarding.manager import OnboardingManager
from services.signal_extractor.marketing_filter import (
    FREEMAIL_DOMAINS,
    classify_email_domain,
)
from services.signal_extractor.relationship import (
    MIN_INTERACTIONS_FOR_AUTO_CREATE,
    RelationshipExtractor,
    _name_from_email,
)
from storage.user_model_store import UserModelStore

# ---------------------------------------------------------------------------
# 1. _parse_contacts — natural language relationship parsing
# ---------------------------------------------------------------------------


class TestParseContacts:
    """Tests for OnboardingManager._parse_contacts()."""

    def _parse(self, db, text: str) -> list[dict]:
        manager = OnboardingManager(db)
        return manager._parse_contacts(text)

    def test_explicit_dash_separator(self, db):
        result = self._parse(db, "Tom - coworker")
        assert result == [{"name": "Tom", "relationship": "coworker"}]

    def test_explicit_paren_separator(self, db):
        result = self._parse(db, "Sarah (wife)")
        assert result == [{"name": "Sarah", "relationship": "wife"}]

    def test_natural_language_name_first(self, db):
        """'Nate my brother in law' → name='Nate', relationship='brother-in-law'"""
        result = self._parse(db, "Nate my brother in law")
        assert len(result) == 1
        assert result[0]["name"] == "Nate"
        assert result[0]["relationship"] == "brother-in-law"

    def test_natural_language_relationship_first(self, db):
        """'my wife Sarah' → name='Sarah', relationship='wife'"""
        result = self._parse(db, "my wife Sarah")
        assert len(result) == 1
        assert result[0]["name"] == "Sarah"
        assert result[0]["relationship"] == "wife"

    def test_plain_name_no_relationship(self, db):
        """'Mom' → name='Mom', relationship=None (the word IS the name)"""
        result = self._parse(db, "Mom")
        assert len(result) == 1
        assert result[0]["name"] == "Mom"
        assert result[0]["relationship"] is None

    def test_multi_word_relationship_hyphenated(self, db):
        result = self._parse(db, "Jake my step brother")
        assert result[0]["relationship"] == "step-brother"

    def test_comma_separated_mixed_formats(self, db):
        result = self._parse(db, "Tom - coworker, Nate my brother in law, Mom")
        assert len(result) == 3
        assert result[0] == {"name": "Tom", "relationship": "coworker"}
        assert result[1]["name"] == "Nate"
        assert result[1]["relationship"] == "brother-in-law"
        assert result[2] == {"name": "Mom", "relationship": None}

    def test_empty_input(self, db):
        result = self._parse(db, "")
        assert result == []

    def test_our_possessive(self, db):
        result = self._parse(db, "our neighbor Jim")
        assert result[0]["name"] == "Jim"
        assert result[0]["relationship"] == "neighbor"

    def test_best_friend(self, db):
        result = self._parse(db, "my best friend Alex")
        assert result[0]["name"] == "Alex"
        assert result[0]["relationship"] == "best-friend"


# ---------------------------------------------------------------------------
# 2. classify_email_domain
# ---------------------------------------------------------------------------


class TestClassifyEmailDomain:
    def test_gmail_is_personal(self):
        assert classify_email_domain("nate@gmail.com") == "personal"

    def test_protonmail_is_personal(self):
        assert classify_email_domain("nate@protonmail.com") == "personal"

    def test_icloud_is_personal(self):
        assert classify_email_domain("nate@icloud.com") == "personal"

    def test_corporate_is_business(self):
        assert classify_email_domain("nate@acmecorp.com") == "business"

    def test_no_at_sign_returns_business(self):
        assert classify_email_domain("invalid") == "business"

    def test_case_insensitive(self):
        assert classify_email_domain("Nate@Gmail.COM") == "personal"

    def test_freemail_set_not_empty(self):
        assert len(FREEMAIL_DOMAINS) >= 15


# ---------------------------------------------------------------------------
# 3. Google connector display name capture
# ---------------------------------------------------------------------------


class TestGoogleConnectorDisplayName:
    def test_parse_email_header(self):
        from connectors.google.connector import GoogleConnector

        name, addr = GoogleConnector._parse_email_header(
            '"Nate Smith" <nate@example.com>'
        )
        assert name == "Nate Smith"
        assert addr == "nate@example.com"

    def test_parse_email_header_no_name(self):
        from connectors.google.connector import GoogleConnector

        name, addr = GoogleConnector._parse_email_header("nate@example.com")
        assert name == ""
        assert addr == "nate@example.com"

    def test_parse_email_names(self):
        from connectors.google.connector import GoogleConnector

        names = GoogleConnector._parse_email_names(
            '"Alice Jones" <alice@example.com>, bob@test.com'
        )
        assert names == {"alice@example.com": "Alice Jones"}
        # bob@test.com has no display name, so it's excluded
        assert "bob@test.com" not in names


# ---------------------------------------------------------------------------
# 4. Proton Mail connector display name capture
# ---------------------------------------------------------------------------


class TestProtonMailConnectorDisplayName:
    def test_parse_address_with_name(self):
        from connectors.proton_mail.connector import ProtonMailConnector

        name, addr = ProtonMailConnector._parse_address_with_name(
            '"Nate Smith" <nate@example.com>'
        )
        assert name == "Nate Smith"
        assert addr == "nate@example.com"

    def test_parse_address_names(self):
        from connectors.proton_mail.connector import ProtonMailConnector

        names = ProtonMailConnector._parse_address_names(
            '"Alice Jones" <alice@example.com>, bob@test.com'
        )
        assert names == {"alice@example.com": "Alice Jones"}


# ---------------------------------------------------------------------------
# 5. RelationshipExtractor uses display_name for auto-contacts
# ---------------------------------------------------------------------------


def _make_profile(interaction_count: int, display_name: str = "") -> dict:
    now = datetime.now(UTC).isoformat()
    ts = [
        (datetime.now(UTC) - timedelta(days=i)).isoformat()
        for i in range(interaction_count)
    ]
    profile = {
        "interaction_count": interaction_count,
        "interaction_timestamps": ts[:interaction_count],
        "last_interaction": ts[0] if ts else now,
        "avg_response_time_seconds": 3600.0,
        "avg_message_length": 50.0,
        "inbound_count": interaction_count // 2,
        "outbound_count": interaction_count - interaction_count // 2,
    }
    if display_name:
        profile["display_name"] = display_name
    return profile


def _get_contact_by_email(db, email: str) -> dict | None:
    with db.get_connection("entities") as conn:
        row = conn.execute(
            """SELECT c.* FROM contacts c
               JOIN contact_identifiers ci ON ci.contact_id = c.id
              WHERE ci.identifier_type = 'email'
                AND lower(ci.identifier) = lower(?)""",
            (email,),
        ).fetchone()
        return dict(row) if row else None


class TestAutoContactDisplayName:
    def test_display_name_used_over_email_heuristic(self, db):
        """When display_name is set on profile, auto-contact uses it instead of _name_from_email."""
        ums = UserModelStore(db)
        extractor = RelationshipExtractor(db, ums)

        addr = "n.smith@example.com"
        contacts_data = {
            addr: _make_profile(MIN_INTERACTIONS_FOR_AUTO_CREATE, display_name="Nate Smith"),
        }
        extractor._sync_contact_metrics(contacts_data)

        contact = _get_contact_by_email(db, addr)
        assert contact is not None
        assert contact["name"] == "Nate Smith"

    def test_falls_back_to_email_heuristic_without_display_name(self, db):
        """Without display_name, auto-contact falls back to _name_from_email."""
        ums = UserModelStore(db)
        extractor = RelationshipExtractor(db, ums)

        addr = "john.doe@example.com"
        contacts_data = {
            addr: _make_profile(MIN_INTERACTIONS_FOR_AUTO_CREATE),
        }
        extractor._sync_contact_metrics(contacts_data)

        contact = _get_contact_by_email(db, addr)
        assert contact is not None
        assert contact["name"] == _name_from_email(addr)

    def test_domain_classification_set_on_auto_contact(self, db):
        """Auto-created contacts should have the domains field populated."""
        ums = UserModelStore(db)
        extractor = RelationshipExtractor(db, ums)

        addr = "alice@gmail.com"
        contacts_data = {
            addr: _make_profile(MIN_INTERACTIONS_FOR_AUTO_CREATE),
        }
        extractor._sync_contact_metrics(contacts_data)

        contact = _get_contact_by_email(db, addr)
        assert contact is not None
        domains = json.loads(contact["domains"])
        assert domains == ["personal"]

    def test_business_domain_classification(self, db):
        """Corporate email domains should be classified as 'business'."""
        ums = UserModelStore(db)
        extractor = RelationshipExtractor(db, ums)

        addr = "alice@acmecorp.com"
        contacts_data = {
            addr: _make_profile(MIN_INTERACTIONS_FOR_AUTO_CREATE),
        }
        extractor._sync_contact_metrics(contacts_data)

        contact = _get_contact_by_email(db, addr)
        assert contact is not None
        domains = json.loads(contact["domains"])
        assert domains == ["business"]


class TestExtractCapturesDisplayName:
    """Verify that extract() propagates from_name into the signal profile."""

    def test_inbound_email_captures_from_name(self, db):
        ums = UserModelStore(db)
        extractor = RelationshipExtractor(db, ums)

        event = {
            "type": "email.received",
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": {
                "from_address": "nate@example.com",
                "from_name": "Nate Smith",
                "to_addresses": ["me@example.com"],
                "to_names": {},
                "channel": "google",
                "body": "Hey, just checking in!",
                "email_date": datetime.now(UTC).isoformat(),
            },
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        assert signals[0]["display_name"] == "Nate Smith"

        # Verify display_name was stored in the profile
        profile = ums.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"]["nate@example.com"]
        assert contact_data["display_name"] == "Nate Smith"
