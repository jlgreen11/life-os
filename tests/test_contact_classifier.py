"""
Tests for contact type classification.

Verifies that contacts are correctly classified as "person" or "business"
based on email address patterns, communication channels, relationship fields,
interaction profiles, and name heuristics.
"""

import pytest

from services.contact_classifier import classify_contact_type, classify_email_address


class TestContactTypeClassifier:
    """Test suite for the classify_contact_type function."""

    # ----- Priority contacts -----

    def test_priority_contacts_are_always_people(self):
        """Priority contacts (set during onboarding) are definitively people."""
        assert classify_contact_type(
            email_address="info@company.com",
            is_priority=True,
        ) == "person"

    # ----- Personal relationships -----

    def test_personal_relationships_are_people(self):
        """Contacts with personal relationship labels are always people."""
        relationships = [
            "spouse", "partner", "wife", "husband",
            "parent", "mother", "father", "mom", "dad",
            "sibling", "brother", "sister",
            "friend", "best friend",
            "child", "son", "daughter",
        ]
        for rel in relationships:
            assert classify_contact_type(relationship=rel) == "person", \
                f"Expected relationship '{rel}' to classify as person"

    def test_professional_relationships_lean_person(self):
        """Professional relationships should lean toward person (they're still people)."""
        relationships = ["boss", "coworker", "colleague", "mentor", "client"]
        for rel in relationships:
            result = classify_contact_type(relationship=rel)
            assert result == "person", \
                f"Expected professional relationship '{rel}' to classify as person"

    # ----- Channel signals -----

    def test_phone_contacts_are_people(self):
        """Contacts with phone numbers are almost always people."""
        assert classify_contact_type(phones=["+15551234567"]) == "person"

    def test_imessage_contacts_are_people(self):
        """Contacts reachable via iMessage are people."""
        assert classify_contact_type(channels={"imessage": "+15551234567"}) == "person"

    def test_signal_contacts_are_people(self):
        """Contacts reachable via Signal are people."""
        assert classify_contact_type(channels={"signal": "+15551234567"}) == "person"

    def test_whatsapp_contacts_are_people(self):
        """Contacts reachable via WhatsApp are people."""
        assert classify_contact_type(channels={"whatsapp": "+15551234567"}) == "person"

    # ----- Email-based business classification -----

    def test_generic_localparts_are_business(self):
        """Generic email local-parts indicate business contacts."""
        business_addresses = [
            "info@company.com",
            "support@service.com",
            "help@platform.io",
            "billing@saas.com",
            "sales@enterprise.com",
            "contact@business.org",
            "admin@server.com",
            "feedback@app.com",
        ]
        for addr in business_addresses:
            result = classify_contact_type(email_address=addr)
            assert result == "business", \
                f"Expected {addr} to classify as business, got {result}"

    def test_noreply_is_definitively_business(self):
        """No-reply addresses are definitively business."""
        noreply_addresses = [
            "noreply@company.com",
            "no-reply@service.com",
            "donotreply@platform.com",
            "do-not-reply@notifications.io",
        ]
        for addr in noreply_addresses:
            assert classify_contact_type(email_address=addr) == "business", \
                f"Expected {addr} to classify as business"

    def test_marketing_domain_patterns_are_business(self):
        """ESP/marketing subdomain patterns indicate business."""
        marketing_addresses = [
            "sender@email.company.com",
            "msg@mail.service.com",
            "alert@reply.platform.io",
            "notice@bounce.service.com",
            "info@mg.company.com",
        ]
        for addr in marketing_addresses:
            result = classify_contact_type(email_address=addr)
            assert result == "business", \
                f"Expected {addr} to classify as business, got {result}"

    # ----- Personal email classification -----

    def test_personal_emails_are_people(self):
        """Normal personal email addresses should classify as people."""
        personal_addresses = [
            "john.doe@gmail.com",
            "jane_smith@company.com",
            "alice@protonmail.com",
            "bob123@outlook.com",
        ]
        for addr in personal_addresses:
            result = classify_contact_type(email_address=addr)
            assert result == "person", \
                f"Expected {addr} to classify as person, got {result}"

    # ----- Name-based signals -----

    def test_company_names_are_business(self):
        """Names with company suffixes indicate businesses."""
        company_names = [
            "Acme Inc.",
            "Google LLC",
            "Apple Corp",
            "Johnson & Johnson Ltd",
            "Deloitte Consulting",
            "First National Bank",
            "United Airlines",
        ]
        for name in company_names:
            result = classify_contact_type(name=name)
            assert result == "business", \
                f"Expected name '{name}' to classify as business, got {result}"

    def test_human_names_lean_person(self):
        """Human-looking names (First Last) should lean toward person."""
        human_names = [
            "John Smith",
            "Alice Cooper",
            "Bob Johnson",
        ]
        for name in human_names:
            result = classify_contact_type(name=name)
            assert result == "person", \
                f"Expected name '{name}' to classify as person, got {result}"

    # ----- Interaction profile signals -----

    def test_two_way_communication_is_person(self):
        """Contacts with both inbound and outbound messages are likely people."""
        profile = {
            "interaction_count": 20,
            "inbound_count": 10,
            "outbound_count": 10,
        }
        result = classify_contact_type(
            email_address="sender@example.com",
            interaction_profile=profile,
        )
        assert result == "person"

    def test_one_way_inbound_only_is_business(self):
        """Contacts with only inbound messages (no replies) lean business."""
        profile = {
            "interaction_count": 10,
            "inbound_count": 10,
            "outbound_count": 0,
        }
        result = classify_contact_type(
            email_address="sender@example.com",
            interaction_profile=profile,
        )
        assert result == "business"

    # ----- Combined signals -----

    def test_business_email_with_phone_is_person(self):
        """Even a generic email, if they have a phone number, is likely a person."""
        # Phone overrides the generic local-part signal
        result = classify_contact_type(
            email_address="support@company.com",
            phones=["+15551234567"],
        )
        assert result == "person"

    def test_business_email_with_personal_relationship_is_person(self):
        """A generic email with a personal relationship is a person."""
        result = classify_contact_type(
            email_address="info@company.com",
            relationship="friend",
        )
        assert result == "person"

    # ----- Edge cases -----

    def test_no_signals_defaults_to_person(self):
        """When no signals are available, default to person (fail-safe)."""
        assert classify_contact_type() == "person"

    def test_empty_email_is_person(self):
        """Empty email address defaults to person."""
        assert classify_contact_type(email_address="") == "person"

    def test_none_values_dont_crash(self):
        """All-None inputs should not crash."""
        result = classify_contact_type(
            email_address=None,
            name=None,
            relationship=None,
            channels=None,
            phones=None,
        )
        assert result in ("person", "business")


class TestClassifyEmailAddress:
    """Test the quick email-only classification helper."""

    def test_personal_email(self):
        assert classify_email_address("john@gmail.com") == "person"

    def test_business_email(self):
        assert classify_email_address("info@company.com") == "business"

    def test_noreply_email(self):
        assert classify_email_address("noreply@service.com") == "business"
