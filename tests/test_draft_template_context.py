"""Tests for communication template integration in draft context assembly.

Verifies that the ContextAssembler includes tone_notes and avoids_phrases from
communication templates when assembling draft reply context, and handles missing
or empty fields gracefully.
"""

import json
import uuid

import pytest

from services.ai_engine.context import ContextAssembler


def _insert_template(conn, *, contact_id=None, channel="email", formality=0.5,
                     greeting=None, closing=None, typical_length=100,
                     uses_emoji=0, common_phrases=None, avoids_phrases=None,
                     tone_notes=None, samples_analyzed=10):
    """Insert a communication template with all fields including tone_notes and avoids_phrases."""
    conn.execute(
        """INSERT INTO communication_templates (
            id, context, contact_id, channel, formality, greeting, closing,
            typical_length, uses_emoji, common_phrases, avoids_phrases,
            tone_notes, samples_analyzed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            "general",
            contact_id,
            channel,
            formality,
            greeting,
            closing,
            typical_length,
            uses_emoji,
            json.dumps(common_phrases or []),
            json.dumps(avoids_phrases or []),
            json.dumps(tone_notes or []),
            samples_analyzed,
        ),
    )


class TestDraftTemplateContext:
    """Test that draft context assembly includes all communication template fields."""

    def test_avoids_phrases_included_in_context(self, db, user_model_store):
        """Avoids phrases from the template should appear in the draft context."""
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="alice@example.com",
                avoids_phrases=["Best regards", "Per my last email"],
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="alice@example.com",
            channel="email",
            incoming_message="Hello there",
        )

        assert "Avoids phrases:" in context
        assert "Best regards" in context
        assert "Per my last email" in context

    def test_tone_notes_included_in_context(self, db, user_model_store):
        """Tone notes from the template should appear in the draft context."""
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="bob@work.com",
                tone_notes=["always leads with conclusion", "uses bullet points"],
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="bob@work.com",
            channel="email",
            incoming_message="Quick question",
        )

        assert "Tone notes:" in context
        assert "always leads with conclusion" in context
        assert "uses bullet points" in context

    def test_no_template_skips_gracefully(self, db, user_model_store):
        """When no template exists for the contact, context should still be valid."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="unknown@example.com",
            channel="sms",
            incoming_message="Hey",
        )

        # Should not crash, should still contain the incoming message
        assert "Hey" in context
        # Template-specific sections should be absent
        assert "Communication style for this context:" not in context
        assert "Avoids phrases:" not in context
        assert "Tone notes:" not in context

    def test_empty_avoids_and_tone_notes_omitted(self, db, user_model_store):
        """Empty avoids_phrases and tone_notes should not add noise to context."""
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="carol@example.com",
                avoids_phrases=[],
                tone_notes=[],
                greeting="Hi",
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="carol@example.com",
            channel="email",
            incoming_message="Test",
        )

        # Template should be present (greeting is there)
        assert "Communication style for this context:" in context
        assert "Greeting: Hi" in context
        # But empty lists should not produce output
        assert "Avoids phrases:" not in context
        assert "Tone notes:" not in context

    def test_avoids_phrases_capped_at_five(self, db, user_model_store):
        """Only the first 5 avoids phrases should be included."""
        phrases = [f"avoid_{i}" for i in range(10)]
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="dave@example.com",
                avoids_phrases=phrases,
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="dave@example.com",
            channel="email",
            incoming_message="Test",
        )

        assert "avoid_4" in context
        assert "avoid_5" not in context

    def test_tone_notes_capped_at_five(self, db, user_model_store):
        """Only the first 5 tone notes should be included."""
        notes = [f"note_{i}" for i in range(10)]
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="eve@example.com",
                tone_notes=notes,
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="eve@example.com",
            channel="email",
            incoming_message="Test",
        )

        assert "note_4" in context
        assert "note_5" not in context

    def test_all_template_fields_present(self, db, user_model_store):
        """Full template with all fields should produce complete context."""
        with db.get_connection("user_model") as conn:
            _insert_template(
                conn,
                contact_id="full@example.com",
                channel="slack",
                formality=0.3,
                greeting="Hey",
                closing="Cheers",
                typical_length=50,
                uses_emoji=1,
                common_phrases=["sounds good", "let me check"],
                avoids_phrases=["per my last email"],
                tone_notes=["keeps it brief", "uses humor"],
                samples_analyzed=25,
            )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_draft_context(
            contact_id="full@example.com",
            channel="slack",
            incoming_message="What do you think?",
        )

        assert "Formality: 0.3" in context
        assert "Greeting: Hey" in context
        assert "Closing: Cheers" in context
        assert "Typical length: 50" in context
        assert "Uses emoji: yes" in context
        assert "Common phrases: sounds good, let me check" in context
        assert "Avoids phrases: per my last email" in context
        assert "Tone notes: keeps it brief; uses humor" in context
