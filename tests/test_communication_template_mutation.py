"""
Tests for communication template mutation methods on UserModelStore.

Validates that delete_communication_template() and
update_communication_template() correctly remove and modify templates
stored via store_communication_template().
"""

import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_template(
    template_id: str,
    context: str = "user_to_contact",
    contact_id: str | None = None,
    channel: str | None = None,
    greeting: str | None = "Hi",
    closing: str | None = "Thanks",
    formality: float = 0.5,
    typical_length: float = 50.0,
    uses_emoji: bool = False,
    common_phrases: list | None = None,
    avoids_phrases: list | None = None,
    tone_notes: list | None = None,
    example_message_ids: list | None = None,
    samples_analyzed: int = 1,
) -> dict:
    """Build a template dict suitable for store_communication_template()."""
    return {
        "id": template_id,
        "context": context,
        "contact_id": contact_id,
        "channel": channel,
        "greeting": greeting,
        "closing": closing,
        "formality": formality,
        "typical_length": typical_length,
        "uses_emoji": uses_emoji,
        "common_phrases": common_phrases or [],
        "avoids_phrases": avoids_phrases or [],
        "tone_notes": tone_notes or [],
        "example_message_ids": example_message_ids or [],
        "samples_analyzed": samples_analyzed,
    }


# ---------------------------------------------------------------------------
# delete_communication_template
# ---------------------------------------------------------------------------


class TestDeleteCommunicationTemplate:
    """Tests for the template deletion method."""

    def test_delete_existing_template(self, user_model_store):
        """Storing a template and deleting by ID removes it from the database."""
        tpl = _make_template("del-1", contact_id="alice@example.com", channel="email")
        user_model_store.store_communication_template(tpl)

        # Verify it exists first
        result = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert result is not None
        assert result["id"] == "del-1"

        # Delete and verify return value
        deleted = user_model_store.delete_communication_template("del-1")
        assert deleted is True

        # Verify it's gone
        result = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert result is None

    def test_delete_nonexistent_returns_false(self, user_model_store):
        """Deleting a non-existent template ID returns False."""
        deleted = user_model_store.delete_communication_template("nonexistent-id")
        assert deleted is False

    def test_delete_only_removes_target(self, user_model_store):
        """Deleting one template does not affect other templates."""
        tpl1 = _make_template("del-keep", contact_id="bob@example.com", channel="email")
        tpl2 = _make_template("del-remove", contact_id="carol@example.com", channel="slack")
        user_model_store.store_communication_template(tpl1)
        user_model_store.store_communication_template(tpl2)

        user_model_store.delete_communication_template("del-remove")

        # The other template should still exist
        remaining = user_model_store.get_communication_templates()
        assert len(remaining) == 1
        assert remaining[0]["id"] == "del-keep"


# ---------------------------------------------------------------------------
# update_communication_template
# ---------------------------------------------------------------------------


class TestUpdateCommunicationTemplate:
    """Tests for the template partial update method."""

    def test_update_formality(self, user_model_store):
        """Updating formality changes the value and preserves other fields."""
        tpl = _make_template(
            "upd-1", contact_id="dave@example.com", channel="email", formality=0.3
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template("upd-1", {"formality": 0.9})
        assert result is not None
        assert result["formality"] == 0.9
        # Other fields should be preserved
        assert result["greeting"] == "Hi"
        assert result["closing"] == "Thanks"
        assert result["contact_id"] == "dave@example.com"

    def test_update_greeting_and_closing(self, user_model_store):
        """Updating multiple fields at once applies all changes."""
        tpl = _make_template(
            "upd-2", contact_id="eve@example.com", channel="email",
            greeting="Hello", closing="Best",
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template(
            "upd-2", {"greeting": "Hey", "closing": "Cheers"}
        )
        assert result is not None
        assert result["greeting"] == "Hey"
        assert result["closing"] == "Cheers"

    def test_update_list_fields(self, user_model_store):
        """Updating common_phrases with a list serializes/deserializes correctly."""
        tpl = _make_template(
            "upd-3", contact_id="frank@example.com", channel="slack",
            common_phrases=["old phrase"],
        )
        user_model_store.store_communication_template(tpl)

        new_phrases = ["sounds good", "will do", "thanks!"]
        result = user_model_store.update_communication_template(
            "upd-3", {"common_phrases": new_phrases}
        )
        assert result is not None
        assert isinstance(result["common_phrases"], list)
        assert result["common_phrases"] == new_phrases

    def test_update_avoids_phrases_and_tone_notes(self, user_model_store):
        """All JSON list fields round-trip correctly through update."""
        tpl = _make_template("upd-lists", contact_id="gina@example.com", channel="email")
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template("upd-lists", {
            "avoids_phrases": ["per my last email", "ASAP"],
            "tone_notes": ["warm", "encouraging"],
        })
        assert result is not None
        assert result["avoids_phrases"] == ["per my last email", "ASAP"]
        assert result["tone_notes"] == ["warm", "encouraging"]

    def test_update_uses_emoji(self, user_model_store):
        """Updating uses_emoji converts boolean to int and back correctly."""
        tpl = _make_template(
            "upd-emoji", contact_id="hank@example.com", channel="slack",
            uses_emoji=False,
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template(
            "upd-emoji", {"uses_emoji": True}
        )
        assert result is not None
        # The raw DB stores 1, but _deserialize_template_row returns it as-is (int 1)
        assert result["uses_emoji"] in (True, 1)

    def test_update_nonexistent_returns_none(self, user_model_store):
        """Updating a non-existent template ID returns None."""
        result = user_model_store.update_communication_template(
            "nonexistent-id", {"formality": 0.8}
        )
        assert result is None

    def test_update_ignores_disallowed_fields(self, user_model_store):
        """Attempting to update immutable fields (id, context, contact_id, channel) is ignored."""
        tpl = _make_template(
            "upd-guard", context="user_to_contact",
            contact_id="ivan@example.com", channel="email",
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template("upd-guard", {
            "id": "hacked-id",
            "context": "contact_to_user",
            "contact_id": "hacker@evil.com",
            "channel": "sms",
        })
        assert result is not None
        # Structural fields must remain unchanged
        assert result["id"] == "upd-guard"
        assert result["context"] == "user_to_contact"
        assert result["contact_id"] == "ivan@example.com"
        assert result["channel"] == "email"

    def test_update_empty_allowed_fields_returns_existing(self, user_model_store):
        """When only disallowed fields are provided, the existing template is returned unchanged."""
        tpl = _make_template(
            "upd-noop", contact_id="joe@example.com", channel="email",
            formality=0.4,
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.update_communication_template(
            "upd-noop", {"id": "should-be-ignored"}
        )
        assert result is not None
        assert result["id"] == "upd-noop"
        assert result["formality"] == 0.4
