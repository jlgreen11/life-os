"""
Tests for observability logging in ContextAssembler's fail-open exception handlers.

The context assembler has 4 exception handlers that were previously bare ``pass``
blocks, silently swallowing failures in critical context-building paths.  These
tests verify that each handler now emits a ``logger.warning`` while still preserving
the fail-open behavior (no crash, returns a usable result).
"""

from __future__ import annotations

import logging
import sqlite3
from unittest.mock import patch

from services.ai_engine.context import ContextAssembler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_outbound_profile(user_model_store, averages: dict):
    """Seed a minimal outbound linguistic profile so draft context has content."""
    user_model_store.update_signal_profile("linguistic", {"averages": averages})


# ---------------------------------------------------------------------------
# Search context: semantic facts failure
# ---------------------------------------------------------------------------


class TestSearchContextSemanticFactsObservability:
    """Verify warning is logged when get_semantic_facts raises."""

    def test_search_context_returns_result_despite_facts_failure(self, db, user_model_store, caplog):
        """assemble_search_context must not crash when get_semantic_facts raises."""
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_semantic_facts",
            side_effect=sqlite3.DatabaseError("disk I/O error"),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                result = assembler.assemble_search_context("find my meetings")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_context_logs_warning_on_facts_failure(self, db, user_model_store, caplog):
        """A warning must be logged when get_semantic_facts raises."""
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_semantic_facts",
            side_effect=sqlite3.DatabaseError("disk I/O error"),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                assembler.assemble_search_context("find my meetings")

        assert any("failed to load semantic facts" in rec.message for rec in caplog.records), (
            f"Expected 'failed to load semantic facts' warning, got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Search context: mood profile failure
# ---------------------------------------------------------------------------


class TestSearchContextMoodObservability:
    """Verify warning is logged when get_signal_profile('mood_signals') raises."""

    def test_search_context_returns_result_despite_mood_failure(self, db, user_model_store, caplog):
        """assemble_search_context must not crash when mood profile fetch raises."""
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database is locked"),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                result = assembler.assemble_search_context("what happened today")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_context_logs_warning_on_mood_failure(self, db, user_model_store, caplog):
        """A warning must be logged when mood profile fetch raises."""
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=sqlite3.DatabaseError("database is locked"),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                assembler.assemble_search_context("what happened today")

        assert any("failed to load mood profile" in rec.message for rec in caplog.records), (
            f"Expected 'failed to load mood profile' warning, got: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# Draft context: inbound writing style failure
# ---------------------------------------------------------------------------


class TestDraftContextInboundStyleObservability:
    """Verify warning is logged when inbound style retrieval fails."""

    def _make_failing_get_signal_profile(self, original):
        """Return a wrapper that raises only for 'linguistic_inbound' profile."""

        def _patched(profile_type):
            if profile_type == "linguistic_inbound":
                raise sqlite3.DatabaseError("corrupt database")
            return original(profile_type)

        return _patched

    def test_draft_context_returns_result_despite_inbound_failure(self, db, user_model_store, caplog):
        """assemble_draft_context must not crash when inbound profile fetch raises."""
        _seed_outbound_profile(user_model_store, {"formality": 0.5})
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self._make_failing_get_signal_profile(user_model_store.get_signal_profile),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                result = assembler.assemble_draft_context("alice@example.com", "email", "Hello!")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Incoming message to reply to:" in result

    def test_draft_context_logs_warning_on_inbound_failure(self, db, user_model_store, caplog):
        """A warning must be logged when inbound style retrieval raises."""
        _seed_outbound_profile(user_model_store, {"formality": 0.5})
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self._make_failing_get_signal_profile(user_model_store.get_signal_profile),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                assembler.assemble_draft_context("alice@example.com", "email", "Hello!")

        assert any("failed to load inbound writing style" in rec.message for rec in caplog.records), (
            f"Expected 'failed to load inbound writing style' warning, got: {[r.message for r in caplog.records]}"
        )

    def test_draft_context_warning_includes_contact_id(self, db, user_model_store, caplog):
        """The warning should include the contact_id for diagnosability."""
        _seed_outbound_profile(user_model_store, {"formality": 0.5})
        assembler = ContextAssembler(db, user_model_store)

        with patch.object(
            user_model_store,
            "get_signal_profile",
            side_effect=self._make_failing_get_signal_profile(user_model_store.get_signal_profile),
        ):
            with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
                assembler.assemble_draft_context("bob@example.com", "email", "Hey")

        warning_records = [r for r in caplog.records if "failed to load inbound writing style" in r.message]
        assert len(warning_records) > 0
        assert "bob@example.com" in warning_records[0].message


# ---------------------------------------------------------------------------
# Draft context: conversation history failure
# ---------------------------------------------------------------------------


class TestDraftContextConversationHistoryObservability:
    """Verify warning is logged when conversation history retrieval fails.

    The conversation history block (Layer 5) directly queries the ``episodes``
    table.  We force a failure by dropping that table before calling
    ``assemble_draft_context``, so the SELECT raises an OperationalError.
    Earlier layers (template, relationship, linguistic profiles) use different
    tables and are unaffected.
    """

    def _drop_episodes_table(self, db):
        """Drop the episodes table to make the Layer 5 query fail."""
        with db.get_connection("user_model") as conn:
            conn.execute("DROP TABLE IF EXISTS episodes")

    def test_draft_context_returns_result_despite_history_failure(self, db, user_model_store, caplog):
        """assemble_draft_context must not crash when conversation history query raises."""
        self._drop_episodes_table(db)
        assembler = ContextAssembler(db, user_model_store)

        with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
            result = assembler.assemble_draft_context("carol@example.com", "email", "Hi there")

        assert isinstance(result, str)
        assert "Incoming message to reply to:" in result

    def test_draft_context_logs_warning_on_history_failure(self, db, user_model_store, caplog):
        """A warning must be logged when conversation history retrieval raises."""
        self._drop_episodes_table(db)
        assembler = ContextAssembler(db, user_model_store)

        with caplog.at_level(logging.WARNING, logger="services.ai_engine.context"):
            assembler.assemble_draft_context("carol@example.com", "email", "Hi there")

        assert any("failed to load conversation history" in rec.message for rec in caplog.records), (
            f"Expected 'failed to load conversation history' warning, got: {[r.message for r in caplog.records]}"
        )
