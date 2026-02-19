"""
Tests for the enriched assemble_draft_context() in ContextAssembler.

Prior to this PR, assemble_draft_context() exposed only ``formality`` from the
linguistic profile, ignoring 9 other computed metrics (question_rate, hedge_rate,
emoji_rate, unique_word_ratio, avg_sentence_length, etc.) and never consulting the
``linguistic_inbound`` profile for the contact's inbound writing style.

This test module verifies that all new dimensions are correctly surfaced and that the
method remains fail-open when profiles are absent.
"""

from __future__ import annotations

import json
import uuid

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_outbound_profile(user_model_store, averages: dict, common_greetings=None,
                           common_closings=None):
    """Seed the ``linguistic`` (outbound) signal profile."""
    data = {"averages": averages}
    if common_greetings is not None:
        data["common_greetings"] = common_greetings
    if common_closings is not None:
        data["common_closings"] = common_closings
    user_model_store.update_signal_profile("linguistic", data)


def _seed_inbound_profile(user_model_store, contact_id: str, contact_avg: dict):
    """Seed the ``linguistic_inbound`` profile with a single contact's averages."""
    user_model_store.update_signal_profile(
        "linguistic_inbound",
        {"per_contact": {}, "per_contact_averages": {contact_id: contact_avg}},
    )


# ---------------------------------------------------------------------------
# Outbound linguistic averages
# ---------------------------------------------------------------------------


class TestDraftContextLinguisticAverages:
    """Verify that all key outbound linguistic averages appear in draft context."""

    def test_formality_included(self, db, user_model_store):
        """Formality should always appear in User's general style section."""
        _seed_outbound_profile(user_model_store, {"formality": 0.75})

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("alice@example.com", "email", "Hi")

        assert "User's general style:" in ctx
        assert "formality=0.75" in ctx

    def test_question_rate_included_when_above_threshold(self, db, user_model_store):
        """question_rate > 0.05 should appear in the style line."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "question_rate": 0.30,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("bob@example.com", "email", "Hi")

        assert "question_rate=0.30" in ctx

    def test_question_rate_omitted_when_below_threshold(self, db, user_model_store):
        """question_rate <= 0.05 (noise-floor) should be omitted to keep context clean."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "question_rate": 0.02,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("carol@example.com", "email", "Hi")

        assert "question_rate" not in ctx

    def test_hedge_rate_included_when_above_threshold(self, db, user_model_store):
        """hedge_rate > 0.05 should appear in the style line."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "hedge_rate": 0.20,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("dave@example.com", "email", "Hi")

        assert "hedge_rate=0.20" in ctx

    def test_hedge_rate_omitted_when_zero(self, db, user_model_store):
        """Zero hedge_rate should not appear (noise suppression)."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "hedge_rate": 0.0,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("eve@example.com", "email", "Hi")

        assert "hedge_rate" not in ctx

    def test_emoji_rate_included_when_above_threshold(self, db, user_model_store):
        """emoji_rate > 0.01 should appear in the style line."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.3,
            "emoji_rate": 0.05,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("frank@example.com", "email", "Hi")

        assert "emoji_rate=0.050" in ctx

    def test_emoji_rate_omitted_when_below_threshold(self, db, user_model_store):
        """emoji_rate <= 0.01 is noise and should be omitted."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "emoji_rate": 0.005,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("grace@example.com", "email", "Hi")

        assert "emoji_rate" not in ctx

    def test_vocabulary_diversity_included(self, db, user_model_store):
        """unique_word_ratio > 0 should appear as vocabulary_diversity."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.6,
            "unique_word_ratio": 0.72,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("heidi@example.com", "email", "Hi")

        assert "vocabulary_diversity=0.72" in ctx

    def test_avg_sentence_length_included(self, db, user_model_store):
        """avg_sentence_length > 0 should appear in the style line."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.5,
            "avg_sentence_length": 14.5,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("ivan@example.com", "email", "Hi")

        assert "avg_sentence_length=14w" in ctx or "avg_sentence_length=15w" in ctx

    def test_all_metrics_together(self, db, user_model_store):
        """When all metrics are above threshold, they all appear in the style line."""
        _seed_outbound_profile(user_model_store, {
            "formality": 0.65,
            "question_rate": 0.15,
            "hedge_rate": 0.10,
            "emoji_rate": 0.03,
            "unique_word_ratio": 0.68,
            "avg_sentence_length": 12.0,
        })

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("julia@example.com", "email", "Hey")

        style_line = next(
            line for line in ctx.splitlines() if "User's general style:" in line
        )
        assert "formality=" in style_line
        assert "question_rate=" in style_line
        assert "hedge_rate=" in style_line
        assert "emoji_rate=" in style_line
        assert "vocabulary_diversity=" in style_line
        assert "avg_sentence_length=" in style_line

    def test_no_profile_does_not_crash(self, db, user_model_store):
        """assemble_draft_context should succeed even with no linguistic profile."""
        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("karl@example.com", "email", "Hi")

        # Must always include the incoming message
        assert "Incoming message to reply to:" in ctx
        assert "Hi" in ctx


# ---------------------------------------------------------------------------
# Common greetings / closings fallback
# ---------------------------------------------------------------------------


class TestDraftContextGreetingsClosings:
    """Verify that common_greetings and common_closings are surfaced as fallback."""

    def test_common_greetings_surfaced(self, db, user_model_store):
        """When the profile has common_greetings, they should appear in the context."""
        _seed_outbound_profile(
            user_model_store,
            {"formality": 0.5},
            common_greetings=["hey", "hi", "hello"],
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("leo@example.com", "email", "Hi")

        assert "User's typical greetings:" in ctx
        assert "hey" in ctx
        assert "hi" in ctx
        assert "hello" in ctx

    def test_common_closings_surfaced(self, db, user_model_store):
        """When the profile has common_closings, they should appear in the context."""
        _seed_outbound_profile(
            user_model_store,
            {"formality": 0.6},
            common_closings=["best", "cheers"],
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("mia@example.com", "email", "Hi")

        assert "User's typical closings:" in ctx
        assert "best" in ctx
        assert "cheers" in ctx

    def test_no_greetings_when_list_empty(self, db, user_model_store):
        """Empty common_greetings list should not produce an empty greetings line."""
        _seed_outbound_profile(
            user_model_store,
            {"formality": 0.5},
            common_greetings=[],
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("noah@example.com", "email", "Hi")

        assert "User's typical greetings:" not in ctx

    def test_greetings_capped_at_three(self, db, user_model_store):
        """Only up to 3 greetings should be shown (top-N, not all)."""
        _seed_outbound_profile(
            user_model_store,
            {"formality": 0.5},
            common_greetings=["hey", "hi", "hello", "yo", "sup"],
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("olivia@example.com", "email", "Hi")

        greetings_line = next(
            (line for line in ctx.splitlines() if "User's typical greetings:" in line),
            "",
        )
        # Count commas + 1 to get number of items
        items = [g.strip() for g in greetings_line.split(":", 1)[-1].split(",")]
        assert len(items) <= 3


# ---------------------------------------------------------------------------
# Contact inbound style (Layer 4)
# ---------------------------------------------------------------------------


class TestDraftContextInboundStyle:
    """Verify that the contact's inbound writing style is surfaced in draft context."""

    def test_inbound_style_included_when_available(self, db, user_model_store):
        """Contact's inbound formality should appear when per_contact_averages exist."""
        _seed_inbound_profile(
            user_model_store,
            "paula@example.com",
            {"formality": 0.3, "avg_sentence_length": 8.0},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(
            "paula@example.com", "email", "What's up?"
        )

        assert "Contact's writing style (paula@example.com):" in ctx
        assert "formality=0.30" in ctx

    def test_inbound_question_rate_included_above_threshold(self, db, user_model_store):
        """Contact's high question_rate should appear in their style section."""
        _seed_inbound_profile(
            user_model_store,
            "quinn@example.com",
            {"formality": 0.5, "question_rate": 0.4, "avg_sentence_length": 10.0},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(
            "quinn@example.com", "email", "Can you help?"
        )

        contact_line = next(
            line for line in ctx.splitlines()
            if "Contact's writing style" in line
        )
        assert "question_rate=0.40" in contact_line

    def test_inbound_hedge_rate_included_above_threshold(self, db, user_model_store):
        """Contact's high hedge_rate should appear in their style section."""
        _seed_inbound_profile(
            user_model_store,
            "ruby@example.com",
            {"formality": 0.5, "hedge_rate": 0.25},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("ruby@example.com", "email", "Hmm")

        contact_line = next(
            line for line in ctx.splitlines()
            if "Contact's writing style" in line
        )
        assert "hedge_rate=0.25" in contact_line

    def test_inbound_emoji_rate_included_above_threshold(self, db, user_model_store):
        """Contact's high emoji_rate should appear in their style section."""
        _seed_inbound_profile(
            user_model_store,
            "sam@example.com",
            {"formality": 0.3, "emoji_rate": 0.06},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("sam@example.com", "email", "😊")

        contact_line = next(
            line for line in ctx.splitlines()
            if "Contact's writing style" in line
        )
        assert "emoji_rate=0.060" in contact_line

    def test_inbound_avg_sentence_length_included(self, db, user_model_store):
        """Contact's avg_sentence_length should appear in their style section."""
        _seed_inbound_profile(
            user_model_store,
            "tara@example.com",
            {"formality": 0.6, "avg_sentence_length": 20.0},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("tara@example.com", "email", "Hi")

        contact_line = next(
            line for line in ctx.splitlines()
            if "Contact's writing style" in line
        )
        assert "avg_sentence_length=20w" in contact_line

    def test_inbound_style_absent_for_unknown_contact(self, db, user_model_store):
        """No inbound style section when the contact has no inbound data."""
        # Seed data for a different contact
        _seed_inbound_profile(
            user_model_store,
            "uma@example.com",
            {"formality": 0.5},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(
            "victor@example.com", "email", "Hello"
        )

        assert "Contact's writing style (victor@example.com):" not in ctx

    def test_inbound_style_missing_profile_does_not_crash(self, db, user_model_store):
        """Missing linguistic_inbound profile must not crash the draft context."""
        # No inbound profile seeded at all
        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("wendy@example.com", "email", "Hi")

        assert "Incoming message to reply to:" in ctx

    def test_inbound_style_does_not_bleed_across_contacts(self, db, user_model_store):
        """Style for contactA should not appear when drafting a reply to contactB."""
        user_model_store.update_signal_profile(
            "linguistic_inbound",
            {
                "per_contact": {},
                "per_contact_averages": {
                    "xavier@example.com": {"formality": 0.1},
                    "yara@example.com": {"formality": 0.9},
                },
            },
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context("yara@example.com", "email", "Hi")

        # Only yara's line should appear, not xavier's
        assert "Contact's writing style (yara@example.com):" in ctx
        assert "Contact's writing style (xavier@example.com):" not in ctx


# ---------------------------------------------------------------------------
# Ordering and structure invariants
# ---------------------------------------------------------------------------


class TestDraftContextStructure:
    """Verify structural invariants of assemble_draft_context()."""

    def test_incoming_message_always_last(self, db, user_model_store):
        """The incoming message section must be the final section in every draft context."""
        _seed_outbound_profile(
            user_model_store,
            {"formality": 0.5, "question_rate": 0.2},
            common_greetings=["hey"],
            common_closings=["thanks"],
        )
        _seed_inbound_profile(
            user_model_store,
            "zara@example.com",
            {"formality": 0.4},
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(
            "zara@example.com", "email", "Let me know."
        )

        lines = ctx.splitlines()
        # The actual message text should be in the last two lines
        assert "Let me know." in "\n".join(lines[-3:])

    def test_outbound_style_before_inbound_style(self, db, user_model_store):
        """User's outbound style section must precede the contact's inbound style."""
        _seed_outbound_profile(user_model_store, {"formality": 0.7})
        _seed_inbound_profile(
            user_model_store, "alpha@example.com", {"formality": 0.3}
        )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(
            "alpha@example.com", "email", "Hello"
        )

        outbound_pos = ctx.find("User's general style:")
        inbound_pos = ctx.find("Contact's writing style")
        assert outbound_pos != -1
        assert inbound_pos != -1
        assert outbound_pos < inbound_pos, (
            "Outbound style should appear before inbound style in context"
        )
