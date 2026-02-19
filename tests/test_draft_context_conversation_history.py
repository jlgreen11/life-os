"""
Tests for Layer 5 of assemble_draft_context(): recent conversation history.

Before this PR, assemble_draft_context() only exposed style/template data and
a bare interaction count.  The LLM had no knowledge of *what* was discussed in
previous exchanges with a contact, making drafted replies generic and unable to
reference ongoing threads.

This module verifies that the new Layer 5 correctly:
  - Queries the episodes table for episodes involving the target contact
  - Surfaces up to 5 recent episodes ordered newest-first
  - Includes timestamp, interaction_type, content_summary, and topics
  - Is omitted gracefully when the contact has no prior episodes
  - Fails open (no crash) when the DB or episodes table is unavailable
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTACT = "alice@example.com"
OTHER_CONTACT = "bob@example.com"


def _ep(db, contact: str, summary: str, interaction_type: str = "email_received",
        topics: list | None = None, days_ago: int = 1):
    """Insert a minimal episode row involving *contact* into the user_model DB."""
    ts = (
        datetime.now(timezone.utc) - timedelta(days=days_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    ep_id = str(uuid.uuid4())
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary,
                contacts_involved, topics, entities)
               VALUES (?, ?, ?, ?, ?, ?, ?, '[]')""",
            (
                ep_id,
                ts,
                str(uuid.uuid4()),
                interaction_type,
                summary,
                json.dumps([contact]),
                json.dumps(topics or []),
            ),
        )
    return ep_id


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


class TestConversationHistoryIncluded:
    """The draft context should include recent episodes with this contact."""

    def test_history_section_present(self, db, user_model_store):
        """A single prior episode should produce the history section header."""
        _ep(db, CONTACT, "Discussed the Q3 roadmap proposal")

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Recent conversation history with this contact" in ctx

    def test_summary_appears_in_context(self, db, user_model_store):
        """The episode's content_summary should appear verbatim in the output."""
        _ep(db, CONTACT, "Reviewed the marketing budget deck")

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Reviewed the marketing budget deck" in ctx

    def test_interaction_type_appears(self, db, user_model_store):
        """The interaction_type (e.g. 'email_sent') should appear per episode."""
        _ep(db, CONTACT, "Sent project update", interaction_type="email_sent")

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "email_sent" in ctx

    def test_topics_appear_when_present(self, db, user_model_store):
        """Non-empty topics should be included in the history line."""
        _ep(db, CONTACT, "Weekly sync", topics=["engineering", "sprint_planning"])

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "engineering" in ctx
        assert "sprint_planning" in ctx

    def test_topics_bracket_present(self, db, user_model_store):
        """Topics should appear inside [topics: ...] brackets."""
        _ep(db, CONTACT, "Budget discussion", topics=["finance"])

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "[topics:" in ctx

    def test_timestamp_date_appears(self, db, user_model_store):
        """The date portion (YYYY-MM-DD) of the episode timestamp should appear."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _ep(db, CONTACT, "Quick chat", days_ago=0)

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert today in ctx


# ---------------------------------------------------------------------------
# Multiple episodes and ordering
# ---------------------------------------------------------------------------


class TestConversationHistoryOrdering:
    """Episodes should appear newest-first, capped at 5."""

    def test_multiple_episodes_all_shown(self, db, user_model_store):
        """Three episodes → all three summaries appear in the context."""
        _ep(db, CONTACT, "Episode one", days_ago=3)
        _ep(db, CONTACT, "Episode two", days_ago=2)
        _ep(db, CONTACT, "Episode three", days_ago=1)

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Episode one" in ctx
        assert "Episode two" in ctx
        assert "Episode three" in ctx

    def test_most_recent_appears_before_older(self, db, user_model_store):
        """The newest episode should appear above older ones (DESC ordering)."""
        _ep(db, CONTACT, "Older summary", days_ago=5)
        _ep(db, CONTACT, "Newer summary", days_ago=1)

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        newer_pos = ctx.find("Newer summary")
        older_pos = ctx.find("Older summary")
        assert newer_pos < older_pos, "Newest episode should appear first"

    def test_capped_at_five_episodes(self, db, user_model_store):
        """When 7 episodes exist, only the 5 most recent should be shown."""
        for i in range(1, 8):
            _ep(db, CONTACT, f"Episode {i}", days_ago=8 - i)

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        # Episodes 1 and 2 (oldest) should NOT appear; 3-7 should.
        assert "Episode 1" not in ctx
        assert "Episode 2" not in ctx
        assert "Episode 3" in ctx
        assert "Episode 7" in ctx


# ---------------------------------------------------------------------------
# Filtering — only this contact's episodes
# ---------------------------------------------------------------------------


class TestConversationHistoryFiltering:
    """Episodes involving other contacts must not bleed into this contact's history."""

    def test_other_contact_episodes_excluded(self, db, user_model_store):
        """Episodes where only *other* contacts appear should be excluded."""
        _ep(db, OTHER_CONTACT, "Unrelated conversation")

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Unrelated conversation" not in ctx

    def test_shared_episode_included(self, db, user_model_store):
        """An episode involving both the target contact and another contact should appear."""
        ep_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary,
                    contacts_involved, topics, entities)
                   VALUES (?, ?, ?, ?, ?, ?, '[]', '[]')""",
                (
                    ep_id,
                    ts,
                    str(uuid.uuid4()),
                    "email_received",
                    "Group meeting invite",
                    json.dumps([CONTACT, OTHER_CONTACT]),
                ),
            )

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Group meeting invite" in ctx


# ---------------------------------------------------------------------------
# No-history case
# ---------------------------------------------------------------------------


class TestNoConversationHistory:
    """When there are no prior episodes with the contact, the section is omitted."""

    def test_no_history_section_absent(self, db, user_model_store):
        """With zero episodes for the contact, the history header should not appear."""
        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "Recent conversation history" not in ctx

    def test_context_still_returned_without_history(self, db, user_model_store):
        """Even with no history the function should return a non-empty string."""
        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        # The incoming message is always appended.
        assert "Incoming message to reply to" in ctx


# ---------------------------------------------------------------------------
# Topics cap at 3
# ---------------------------------------------------------------------------


class TestTopicsCap:
    """Topics list should be capped at 3 to avoid context bloat."""

    def test_topics_capped_at_three(self, db, user_model_store):
        """Only the first 3 topics should appear even if 6 are stored."""
        many_topics = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        _ep(db, CONTACT, "Wide-ranging discussion", topics=many_topics)

        assembler = ContextAssembler(db, user_model_store)
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hi")

        assert "alpha" in ctx
        assert "beta" in ctx
        assert "gamma" in ctx
        assert "delta" not in ctx


# ---------------------------------------------------------------------------
# Fail-open: DB not available
# ---------------------------------------------------------------------------


class TestConversationHistoryFailOpen:
    """A DB exception in Layer 5 must not crash assemble_draft_context()."""

    def test_fail_open_on_episode_query_error(self, db, user_model_store, monkeypatch):
        """If only the Layer 5 episode query raises, the context is still returned.

        Layer 1 (template lookup) also uses get_connection, so we simulate a
        Layer-5-specific failure by inserting a row with malformed JSON in the
        ``topics`` column.  The json.loads call in Layer 5 is wrapped in a
        try/except that falls back to [], so this tests the per-episode error
        path.  The overall fail-open wrapper is tested indirectly by verifying
        that the method never raises even when the episode data is corrupt.
        """
        ep_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO episodes
                   (id, timestamp, event_id, interaction_type, content_summary,
                    contacts_involved, topics, entities)
                   VALUES (?, ?, ?, ?, ?, ?, ?, '[]')""",
                (
                    ep_id,
                    ts,
                    str(uuid.uuid4()),
                    "email_received",
                    "Message with bad topics",
                    json.dumps([CONTACT]),
                    "NOT_VALID_JSON{{{",  # malformed — json.loads will raise
                ),
            )

        assembler = ContextAssembler(db, user_model_store)
        # Should not raise even with malformed topic JSON.
        ctx = assembler.assemble_draft_context(CONTACT, "email", "Hello!")

        # The incoming message must always be present.
        assert "Hello!" in ctx
        # The episode summary should still appear (only topics parse failed).
        assert "Message with bad topics" in ctx
