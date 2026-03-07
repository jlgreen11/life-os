"""
Tests for extended linguistic features in ContextAssembler._assemble_draft_context.

Validates that greeting/closing conventions, humor markers, Oxford comma preference,
assertion rate, avg word length, and capitalization style are surfaced to the LLM
when present in the linguistic profile — and omitted when absent or below threshold.
"""

from __future__ import annotations

import pytest

from services.ai_engine.context import ContextAssembler


def _set_linguistic_profile(user_model_store, averages: dict):
    """Helper: seed a linguistic profile with the given averages dict."""
    user_model_store.update_signal_profile(
        profile_type="linguistic",
        data={"averages": averages},
    )


class TestExtendedLinguisticDraftContext:
    """Extended linguistic feature surfacing in draft context."""

    def test_greeting_from_top_greeting(self, db, user_model_store):
        """When top_greeting is present, the context includes the preferred greeting."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "top_greeting": "Hey",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="alice@example.com",
            channel="email",
            incoming_message="Hi there",
        )

        assert 'Preferred greeting: "Hey"' in context

    def test_greeting_from_greeting_detected(self, db, user_model_store):
        """Falls back to greeting_detected when top_greeting is absent."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "greeting_detected": "Hello",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="bob@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert 'Preferred greeting: "Hello"' in context

    def test_closing_from_top_closing(self, db, user_model_store):
        """When top_closing is present, the context includes the preferred closing."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "top_closing": "Cheers",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="carol@example.com",
            channel="email",
            incoming_message="Hey",
        )

        assert 'Preferred closing: "Cheers"' in context

    def test_humor_rate_above_threshold(self, db, user_model_store):
        """When humor_rate > 0.02, the context includes a humor note."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.4,
            "humor_rate": 0.08,
            "top_humor_marker": "lol",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="dave@example.com",
            channel="sms",
            incoming_message="What's up",
        )

        assert "Uses humor frequently (rate=0.08)" in context
        assert '"lol"' in context

    def test_humor_rate_without_marker(self, db, user_model_store):
        """Humor note works without a top_humor_marker."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "humor_rate": 0.05,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="eve@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "Uses humor frequently (rate=0.05)" in context
        assert "common marker" not in context

    def test_humor_rate_below_threshold_omitted(self, db, user_model_store):
        """When humor_rate <= 0.02, no humor note appears."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "humor_rate": 0.01,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="frank@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "humor" not in context.lower()

    def test_oxford_comma_true(self, db, user_model_store):
        """When oxford_comma_preference is True, context includes 'Oxford comma: yes'."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.6,
            "oxford_comma_preference": True,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="grace@example.com",
            channel="email",
            incoming_message="Hey",
        )

        assert "Oxford comma: yes" in context

    def test_oxford_comma_false(self, db, user_model_store):
        """When oxford_comma_preference is False, context includes 'Oxford comma: no'."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "oxford_comma_preference": False,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="hank@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "Oxford comma: no" in context

    def test_oxford_comma_absent_omitted(self, db, user_model_store):
        """When oxford_comma_preference is absent, no Oxford comma line appears."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="ivy@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "Oxford comma" not in context

    def test_assertion_rate_above_threshold(self, db, user_model_store):
        """When assertion_rate > 0.05, it appears in the style summary."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "assertion_rate": 0.12,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="jack@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "assertion_rate=0.12" in context

    def test_assertion_rate_below_threshold_omitted(self, db, user_model_store):
        """When assertion_rate <= 0.05, it does not appear in the context."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "assertion_rate": 0.03,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="kate@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "assertion_rate" not in context

    def test_avg_word_length_surfaced(self, db, user_model_store):
        """When avg_word_length > 0, it appears in the style summary."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "avg_word_length": 5.3,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="leo@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "avg_word_length=5.3" in context

    def test_capitalization_style_surfaced(self, db, user_model_store):
        """When capitalization_style is set and not 'unknown', it appears."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "capitalization_style": "lowercase",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="mia@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "Capitalization style: lowercase" in context

    def test_capitalization_style_unknown_omitted(self, db, user_model_store):
        """When capitalization_style is 'unknown', it is omitted."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
            "capitalization_style": "unknown",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="nick@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "Capitalization style" not in context

    def test_all_extended_features_together(self, db, user_model_store):
        """All extended features appear when all are present and above threshold."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.45,
            "assertion_rate": 0.15,
            "avg_word_length": 4.8,
            "top_greeting": "Yo",
            "top_closing": "Later",
            "humor_rate": 0.10,
            "top_humor_marker": "haha",
            "oxford_comma_preference": True,
            "capitalization_style": "sentence_case",
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="all@example.com",
            channel="sms",
            incoming_message="Hey!",
        )

        assert "assertion_rate=0.15" in context
        assert "avg_word_length=4.8" in context
        assert 'Preferred greeting: "Yo"' in context
        assert 'Preferred closing: "Later"' in context
        assert "Uses humor frequently (rate=0.10)" in context
        assert '"haha"' in context
        assert "Oxford comma: yes" in context
        assert "Capitalization style: sentence_case" in context

    def test_no_extended_features_when_absent(self, db, user_model_store):
        """None of the extended features appear when the profile has only basic data."""
        _set_linguistic_profile(user_model_store, {
            "formality": 0.5,
        })

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="plain@example.com",
            channel="email",
            incoming_message="Hi",
        )

        assert "assertion_rate" not in context
        assert "avg_word_length" not in context
        assert "Preferred greeting" not in context
        assert "Preferred closing" not in context
        assert "humor" not in context.lower()
        assert "Oxford comma" not in context
        assert "Capitalization style" not in context
