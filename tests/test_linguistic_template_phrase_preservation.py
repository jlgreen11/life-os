"""
Tests for linguistic extractor phrase data preservation.

Verifies that the linguistic extractor does not overwrite common_phrases
and avoids_phrases data previously computed by the relationship extractor.
"""

import hashlib
import json

import pytest


def _relationship_template_id(address: str, channel: str, direction: str = "out") -> str:
    """Compute template ID the same way the relationship extractor does."""
    return hashlib.sha256(
        f"{address}:{channel}:{direction}".encode()
    ).hexdigest()[:16]


def _linguistic_template_id(contact_id: str) -> str:
    """Compute template ID the same way the linguistic extractor does."""
    return hashlib.sha256(
        f"linguistic:{contact_id}:outbound".encode()
    ).hexdigest()[:16]


class TestLinguisticTemplatePhrasePreservation:
    """Ensure the linguistic extractor preserves phrase data from relationship templates."""

    def test_linguistic_template_preserves_relationship_phrases(self, user_model_store):
        """Storing a linguistic template after a relationship template preserves phrase data."""
        contact = "alice@example.com"
        channel = "email"

        # Simulate the relationship extractor storing a template with real phrases
        rel_template = {
            "id": _relationship_template_id(contact, channel),
            "context": "user_to_contact",
            "contact_id": contact,
            "channel": channel,
            "greeting": "Hi",
            "closing": "Best",
            "formality": 0.6,
            "typical_length": 120.0,
            "uses_emoji": False,
            "common_phrases": ["thanks", "looking forward", "let me know"],
            "avoids_phrases": ["ASAP", "per my last email"],
            "tone_notes": ["friendly"],
            "example_message_ids": ["msg-1", "msg-2"],
            "samples_analyzed": 10,
        }
        user_model_store.store_communication_template(rel_template)

        # Verify relationship template was stored with phrases
        templates = user_model_store.get_communication_templates(contact_id=contact)
        assert len(templates) >= 1
        rel_stored = next(t for t in templates if t["context"] == "user_to_contact")
        assert rel_stored["common_phrases"] == ["thanks", "looking forward", "let me know"]
        assert rel_stored["avoids_phrases"] == ["ASAP", "per my last email"]

        # Simulate the linguistic extractor storing its template (uses _get_existing_phrase_data)
        # First, get phrase data the way the linguistic extractor would
        existing_phrases = {"common_phrases": [], "avoids_phrases": []}
        existing_templates = user_model_store.get_communication_templates(
            contact_id=contact, limit=10
        )
        for tmpl in existing_templates:
            if tmpl.get("common_phrases") and not existing_phrases["common_phrases"]:
                existing_phrases["common_phrases"] = tmpl["common_phrases"]
            if tmpl.get("avoids_phrases") and not existing_phrases["avoids_phrases"]:
                existing_phrases["avoids_phrases"] = tmpl["avoids_phrases"]

        ling_template = {
            "id": _linguistic_template_id(contact),
            "context": "linguistic_outbound",
            "contact_id": contact,
            "channel": channel,
            "greeting": "Hey",
            "closing": "Cheers",
            "formality": 0.4,
            "typical_length": 85,
            "uses_emoji": True,
            "common_phrases": existing_phrases["common_phrases"],
            "avoids_phrases": existing_phrases["avoids_phrases"],
            "tone_notes": ["tends to hedge", "asks many questions"],
            "example_message_ids": [],
            "samples_analyzed": 15,
        }
        user_model_store.store_communication_template(ling_template)

        # Both templates should exist
        all_templates = user_model_store.get_communication_templates(contact_id=contact)
        assert len(all_templates) == 2

        # The linguistic template should have the relationship extractor's phrases
        ling_stored = next(t for t in all_templates if t["context"] == "linguistic_outbound")
        assert ling_stored["common_phrases"] == ["thanks", "looking forward", "let me know"]
        assert ling_stored["avoids_phrases"] == ["ASAP", "per my last email"]

        # The relationship template should still be intact
        rel_stored = next(t for t in all_templates if t["context"] == "user_to_contact")
        assert rel_stored["common_phrases"] == ["thanks", "looking forward", "let me know"]

    def test_linguistic_template_works_with_no_prior_template(self, user_model_store):
        """Storing a linguistic template when no prior template exists stores empty phrases."""
        contact = "bob@example.com"
        channel = "slack"

        # No prior templates exist — phrase lookup returns empty
        existing_templates = user_model_store.get_communication_templates(
            contact_id=contact, limit=10
        )
        assert len(existing_templates) == 0

        ling_template = {
            "id": _linguistic_template_id(contact),
            "context": "linguistic_outbound",
            "contact_id": contact,
            "channel": channel,
            "greeting": "Hi",
            "closing": None,
            "formality": 0.5,
            "typical_length": 50,
            "uses_emoji": False,
            "common_phrases": [],
            "avoids_phrases": [],
            "tone_notes": ["tends to be assertive"],
            "example_message_ids": [],
            "samples_analyzed": 5,
        }
        user_model_store.store_communication_template(ling_template)

        templates = user_model_store.get_communication_templates(contact_id=contact)
        assert len(templates) == 1
        assert templates[0]["common_phrases"] == []
        assert templates[0]["avoids_phrases"] == []
        assert templates[0]["tone_notes"] == ["tends to be assertive"]

    def test_linguistic_template_preserves_tone_and_greeting(self, user_model_store):
        """Linguistic extractor's tone_notes and greeting/closing are stored correctly."""
        contact = "carol@example.com"
        channel = "email"

        ling_template = {
            "id": _linguistic_template_id(contact),
            "context": "linguistic_outbound",
            "contact_id": contact,
            "channel": channel,
            "greeting": "Dear Carol",
            "closing": "Kind regards",
            "formality": 0.85,
            "typical_length": 200,
            "uses_emoji": False,
            "common_phrases": [],
            "avoids_phrases": [],
            "tone_notes": ["tends to hedge", "uses exclamations frequently"],
            "example_message_ids": [],
            "samples_analyzed": 12,
        }
        user_model_store.store_communication_template(ling_template)

        stored = user_model_store.get_communication_templates(contact_id=contact)
        assert len(stored) == 1
        tmpl = stored[0]
        assert tmpl["greeting"] == "Dear Carol"
        assert tmpl["closing"] == "Kind regards"
        assert tmpl["formality"] == pytest.approx(0.85)
        assert tmpl["typical_length"] == pytest.approx(200.0)
        assert tmpl["tone_notes"] == ["tends to hedge", "uses exclamations frequently"]
        assert not tmpl["uses_emoji"]

    def test_linguistic_template_ids_differ_from_relationship(self):
        """Linguistic and relationship extractors use different ID schemes."""
        contact = "dave@example.com"
        channel = "email"

        ling_id = _linguistic_template_id(contact)
        rel_id = _relationship_template_id(contact, channel)

        # They must differ — different hash inputs
        assert ling_id != rel_id, (
            "Linguistic and relationship template IDs should not collide"
        )

    def test_repeated_linguistic_stores_preserve_phrases(self, user_model_store):
        """Multiple linguistic template stores don't lose phrase data over time."""
        contact = "eve@example.com"
        channel = "email"

        # Relationship extractor stores phrases first
        rel_template = {
            "id": _relationship_template_id(contact, channel),
            "context": "user_to_contact",
            "contact_id": contact,
            "channel": channel,
            "greeting": "Hi",
            "closing": "Thanks",
            "formality": 0.5,
            "typical_length": 100.0,
            "uses_emoji": False,
            "common_phrases": ["sounds good", "thanks"],
            "avoids_phrases": ["urgent"],
            "tone_notes": [],
            "example_message_ids": [],
            "samples_analyzed": 8,
        }
        user_model_store.store_communication_template(rel_template)

        # Simulate multiple rounds of the linguistic extractor storing
        for i in range(3):
            existing_phrases = {"common_phrases": [], "avoids_phrases": []}
            existing_templates = user_model_store.get_communication_templates(
                contact_id=contact, limit=10
            )
            for tmpl in existing_templates:
                if tmpl.get("common_phrases") and not existing_phrases["common_phrases"]:
                    existing_phrases["common_phrases"] = tmpl["common_phrases"]
                if tmpl.get("avoids_phrases") and not existing_phrases["avoids_phrases"]:
                    existing_phrases["avoids_phrases"] = tmpl["avoids_phrases"]

            ling_template = {
                "id": _linguistic_template_id(contact),
                "context": "linguistic_outbound",
                "contact_id": contact,
                "channel": channel,
                "greeting": "Hey",
                "closing": "Cheers",
                "formality": 0.3 + i * 0.05,
                "typical_length": 60 + i * 10,
                "uses_emoji": True,
                "common_phrases": existing_phrases["common_phrases"],
                "avoids_phrases": existing_phrases["avoids_phrases"],
                "tone_notes": ["casual"],
                "example_message_ids": [],
                "samples_analyzed": 10 + i,
            }
            user_model_store.store_communication_template(ling_template)

        # After 3 rounds, phrases should still be preserved
        all_templates = user_model_store.get_communication_templates(contact_id=contact)
        ling_stored = next(t for t in all_templates if t["context"] == "linguistic_outbound")
        assert ling_stored["common_phrases"] == ["sounds good", "thanks"]
        assert ling_stored["avoids_phrases"] == ["urgent"]
