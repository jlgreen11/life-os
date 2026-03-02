"""
Tests for communication template retrieval methods on UserModelStore.

Validates that get_communication_template() and get_communication_templates()
correctly query, filter, and deserialize templates stored via
store_communication_template().
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
# get_communication_template — single result
# ---------------------------------------------------------------------------

class TestGetCommunicationTemplate:
    """Tests for the single-template retrieval method."""

    def test_retrieve_by_contact_id(self, user_model_store):
        """Storing a template and retrieving by contact_id returns the template."""
        tpl = _make_template("tpl-1", contact_id="alice@example.com", channel="email")
        user_model_store.store_communication_template(tpl)

        result = user_model_store.get_communication_template(contact_id="alice@example.com")
        assert result is not None
        assert result["id"] == "tpl-1"
        assert result["contact_id"] == "alice@example.com"
        assert result["channel"] == "email"

    def test_retrieve_by_channel(self, user_model_store):
        """Retrieving by channel only returns a matching template."""
        tpl = _make_template("tpl-2", contact_id="bob@example.com", channel="slack")
        user_model_store.store_communication_template(tpl)

        result = user_model_store.get_communication_template(channel="slack")
        assert result is not None
        assert result["id"] == "tpl-2"

    def test_returns_none_when_no_match(self, user_model_store):
        """get_communication_template returns None when nothing matches."""
        result = user_model_store.get_communication_template(
            contact_id="nobody@example.com"
        )
        assert result is None

    def test_prefers_highest_samples_analyzed(self, user_model_store):
        """When multiple templates match, the one with the most samples wins."""
        tpl_low = _make_template(
            "tpl-low", contact_id="carol@example.com", channel="email",
            samples_analyzed=2,
        )
        tpl_high = _make_template(
            "tpl-high", contact_id="carol@example.com", channel="slack",
            samples_analyzed=10,
        )
        user_model_store.store_communication_template(tpl_low)
        user_model_store.store_communication_template(tpl_high)

        result = user_model_store.get_communication_template(
            contact_id="carol@example.com"
        )
        assert result is not None
        assert result["id"] == "tpl-high"
        assert result["samples_analyzed"] == 10

    def test_json_fields_deserialized(self, user_model_store):
        """JSON list fields are returned as native Python lists, not strings."""
        tpl = _make_template(
            "tpl-json",
            contact_id="dave@example.com",
            channel="email",
            common_phrases=["sounds good", "let me know"],
            avoids_phrases=["per my last email"],
            tone_notes=["friendly but professional"],
            example_message_ids=["msg-1", "msg-2", "msg-3"],
        )
        user_model_store.store_communication_template(tpl)

        result = user_model_store.get_communication_template(
            contact_id="dave@example.com"
        )
        assert result is not None
        assert isinstance(result["common_phrases"], list)
        assert result["common_phrases"] == ["sounds good", "let me know"]
        assert isinstance(result["avoids_phrases"], list)
        assert result["avoids_phrases"] == ["per my last email"]
        assert isinstance(result["tone_notes"], list)
        assert result["tone_notes"] == ["friendly but professional"]
        assert isinstance(result["example_message_ids"], list)
        assert result["example_message_ids"] == ["msg-1", "msg-2", "msg-3"]


# ---------------------------------------------------------------------------
# get_communication_templates — multi-result
# ---------------------------------------------------------------------------

class TestGetCommunicationTemplates:
    """Tests for the multi-template retrieval method."""

    def test_returns_multiple_sorted_by_samples(self, user_model_store):
        """Multiple templates are returned sorted by samples_analyzed DESC."""
        for i, samples in enumerate([5, 20, 1, 10]):
            tpl = _make_template(
                f"tpl-multi-{i}",
                contact_id="eve@example.com",
                channel="email",
                samples_analyzed=samples,
            )
            user_model_store.store_communication_template(tpl)

        results = user_model_store.get_communication_templates(
            contact_id="eve@example.com"
        )
        assert len(results) == 4
        # Verify descending order by samples_analyzed
        samples_list = [r["samples_analyzed"] for r in results]
        assert samples_list == sorted(samples_list, reverse=True)

    def test_contact_id_only_filter(self, user_model_store):
        """Filtering by contact_id only returns templates for that contact."""
        tpl_match = _make_template("tpl-match", contact_id="frank@example.com", channel="email")
        tpl_other = _make_template("tpl-other", contact_id="grace@example.com", channel="email")
        user_model_store.store_communication_template(tpl_match)
        user_model_store.store_communication_template(tpl_other)

        results = user_model_store.get_communication_templates(
            contact_id="frank@example.com"
        )
        assert len(results) == 1
        assert results[0]["contact_id"] == "frank@example.com"

    def test_channel_only_filter(self, user_model_store):
        """Filtering by channel only returns templates for that channel."""
        tpl_slack = _make_template("tpl-slack", contact_id="h@ex.com", channel="slack")
        tpl_email = _make_template("tpl-email", contact_id="i@ex.com", channel="email")
        user_model_store.store_communication_template(tpl_slack)
        user_model_store.store_communication_template(tpl_email)

        results = user_model_store.get_communication_templates(channel="slack")
        assert len(results) == 1
        assert results[0]["channel"] == "slack"

    def test_both_filters_use_or(self, user_model_store):
        """Providing both contact_id and channel uses OR logic."""
        tpl_contact = _make_template(
            "tpl-c", contact_id="joe@example.com", channel="email",
        )
        tpl_channel = _make_template(
            "tpl-ch", contact_id="other@example.com", channel="slack",
        )
        tpl_neither = _make_template(
            "tpl-neither", contact_id="nobody@example.com", channel="sms",
        )
        user_model_store.store_communication_template(tpl_contact)
        user_model_store.store_communication_template(tpl_channel)
        user_model_store.store_communication_template(tpl_neither)

        results = user_model_store.get_communication_templates(
            contact_id="joe@example.com", channel="slack"
        )
        assert len(results) == 2
        result_ids = {r["id"] for r in results}
        assert result_ids == {"tpl-c", "tpl-ch"}

    def test_no_filters_returns_all(self, user_model_store):
        """Calling with no filters returns every template in the table."""
        for i in range(5):
            tpl = _make_template(
                f"tpl-all-{i}",
                contact_id=f"contact-{i}@example.com",
                channel="email",
                samples_analyzed=i + 1,
            )
            user_model_store.store_communication_template(tpl)

        results = user_model_store.get_communication_templates()
        assert len(results) == 5

    def test_limit_parameter(self, user_model_store):
        """The limit parameter caps the number of returned templates."""
        for i in range(10):
            tpl = _make_template(
                f"tpl-lim-{i}",
                contact_id=f"lim-{i}@example.com",
                channel="email",
                samples_analyzed=i,
            )
            user_model_store.store_communication_template(tpl)

        results = user_model_store.get_communication_templates(limit=3)
        assert len(results) == 3
        # Should be the 3 with the highest samples_analyzed
        assert results[0]["samples_analyzed"] == 9

    def test_json_fields_deserialized_in_list(self, user_model_store):
        """JSON fields are deserialized for every template in the result list."""
        tpl = _make_template(
            "tpl-json-list",
            contact_id="kate@example.com",
            channel="email",
            common_phrases=["hi there"],
            avoids_phrases=["ASAP"],
            tone_notes=["warm"],
            example_message_ids=["m1"],
        )
        user_model_store.store_communication_template(tpl)

        results = user_model_store.get_communication_templates(
            contact_id="kate@example.com"
        )
        assert len(results) == 1
        r = results[0]
        assert isinstance(r["common_phrases"], list)
        assert r["common_phrases"] == ["hi there"]
        assert isinstance(r["avoids_phrases"], list)
        assert isinstance(r["tone_notes"], list)
        assert isinstance(r["example_message_ids"], list)
