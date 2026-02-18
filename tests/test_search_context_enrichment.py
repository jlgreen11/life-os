"""
Tests for the enriched ContextAssembler.assemble_search_context() method.

Prior to this improvement, assemble_search_context() returned only a bare
one-liner:

    "User is searching across their entire digital life for: <query>"

The enriched version now includes:
  - Current timestamp (anchors relative time expressions)
  - User preferences (verbosity, preferred name)
  - High-confidence semantic facts (disambiguation)
  - Recent mood signals (tone calibration)

These tests verify that each enrichment section appears in the output and
that the method degrades gracefully when data is absent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_assembler(db, user_model_store) -> ContextAssembler:
    """Return a ContextAssembler wired to real temporary databases."""
    return ContextAssembler(db, user_model_store)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def assembler(db, user_model_store) -> ContextAssembler:
    """A ContextAssembler backed by empty temporary databases."""
    return _make_assembler(db, user_model_store)


@pytest.fixture()
def assembler_with_facts(db, user_model_store) -> ContextAssembler:
    """A ContextAssembler whose user model store has two high-confidence facts.

    Values are stored as JSON strings (via json.dumps) to match the format
    written by UserModelStore.store_semantic_fact().
    """
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO semantic_facts (key, category, value, confidence)
               VALUES (?, ?, ?, ?)""",
            ("employer", "work", json.dumps("Acme Corp"), 0.9),
        )
        conn.execute(
            """INSERT INTO semantic_facts (key, category, value, confidence)
               VALUES (?, ?, ?, ?)""",
            ("preferred_name", "identity", json.dumps("Jeremy"), 0.95),
        )
    return _make_assembler(db, user_model_store)


@pytest.fixture()
def assembler_with_preferences(db, user_model_store) -> ContextAssembler:
    """A ContextAssembler whose preferences database has a verbosity entry."""
    with db.get_connection("preferences") as conn:
        conn.execute(
            "INSERT INTO user_preferences (key, value) VALUES (?, ?)",
            ("verbosity", "concise"),
        )
    return _make_assembler(db, user_model_store)


@pytest.fixture()
def assembler_with_mood(db, user_model_store) -> ContextAssembler:
    """A ContextAssembler whose user model has mood signal history."""
    # Write a mood_signals profile directly into the signal_profiles table.
    mood_data = {
        "recent_signals": [
            {"valence": 0.7, "energy": 0.6, "stress": 0.2, "ts": "2026-02-18T10:00:00Z"},
            {"valence": 0.5, "energy": 0.5, "stress": 0.3, "ts": "2026-02-18T12:00:00Z"},
        ]
    }
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            ("mood_signals", json.dumps(mood_data), 2),
        )
    return _make_assembler(db, user_model_store)


# ---------------------------------------------------------------------------
# Tests: always-present sections
# ---------------------------------------------------------------------------

class TestSearchContextCore:
    """The search intent and timestamp must always appear in the output."""

    def test_contains_search_query(self, assembler):
        """The search query appears verbatim in the returned context."""
        ctx = assembler.assemble_search_context("emails from Mike about Denver")
        assert "emails from Mike about Denver" in ctx

    def test_contains_current_timestamp(self, assembler):
        """The output contains a human-readable current date string."""
        ctx = assembler.assemble_search_context("find recipe mom sent")
        # The timestamp section always includes the year.
        assert "2026" in ctx or str(datetime.now(timezone.utc).year) in ctx

    def test_sections_separated_by_delimiter(self, assembler):
        """Sections are separated by '---' so the LLM can parse them."""
        ctx = assembler.assemble_search_context("project update")
        assert "---" in ctx

    def test_search_intent_is_first_section(self, assembler):
        """The search intent section appears before all other sections."""
        ctx = assembler.assemble_search_context("budget report Q4")
        # The query should appear before the timestamp section.
        query_pos = ctx.find("budget report Q4")
        time_pos = ctx.find("Current time:")
        assert query_pos < time_pos, "Search intent must precede timestamp"

    def test_timestamp_appears_before_preferences(self, assembler_with_preferences):
        """Timestamp section appears before preferences section."""
        ctx = assembler_with_preferences.assemble_search_context("task status")
        time_pos = ctx.find("Current time:")
        pref_pos = ctx.find("User preferences:")
        assert time_pos < pref_pos, "Timestamp must precede preferences"

    def test_empty_query_still_returns_context(self, assembler):
        """An empty query string is handled gracefully."""
        ctx = assembler.assemble_search_context("")
        # Should still return a multi-section string with timestamp.
        assert "Current time:" in ctx


# ---------------------------------------------------------------------------
# Tests: preferences section
# ---------------------------------------------------------------------------

class TestSearchContextPreferences:
    """User preferences are included in the search context."""

    def test_preferences_included_when_present(self, assembler_with_preferences):
        """When preferences exist they appear in the context."""
        ctx = assembler_with_preferences.assemble_search_context("any query")
        assert "User preferences:" in ctx
        assert "verbosity" in ctx

    def test_preferences_fallback_when_empty(self, assembler):
        """When no preferences exist the fallback string appears instead of crashing."""
        ctx = assembler.assemble_search_context("any query")
        # _get_preference_context() returns a fallback string when table is empty.
        assert "User preferences:" in ctx


# ---------------------------------------------------------------------------
# Tests: semantic facts section
# ---------------------------------------------------------------------------

class TestSearchContextSemanticFacts:
    """High-confidence semantic facts are injected for disambiguation."""

    def test_facts_included_when_present(self, assembler_with_facts):
        """Known facts appear in the context when they have confidence >= 0.6."""
        ctx = assembler_with_facts.assemble_search_context("find emails about work")
        assert "Known facts about user" in ctx
        assert "employer" in ctx
        assert "Acme Corp" in ctx

    def test_facts_absent_when_none_stored(self, assembler):
        """No facts section appears when the semantic_facts table is empty."""
        ctx = assembler.assemble_search_context("find emails")
        assert "Known facts" not in ctx

    def test_facts_limited_to_15(self, db, user_model_store):
        """At most 15 facts are injected to control token budget."""
        # Insert 20 high-confidence facts (values JSON-encoded to match store format).
        with db.get_connection("user_model") as conn:
            for i in range(20):
                conn.execute(
                    """INSERT INTO semantic_facts (key, category, value, confidence)
                       VALUES (?, ?, ?, ?)""",
                    (f"fact_key_{i}", "test", json.dumps(f"fact_val_{i}"), 0.9),
                )
        assembler = _make_assembler(db, user_model_store)
        ctx = assembler.assemble_search_context("anything")
        # Count the number of "- fact_key_" occurrences as a proxy for row count.
        fact_rows = [line for line in ctx.splitlines() if line.strip().startswith("- fact_key_")]
        assert len(fact_rows) <= 15, f"Expected at most 15 facts, got {len(fact_rows)}"

    def test_low_confidence_facts_excluded(self, db, user_model_store):
        """Facts with confidence < 0.6 are not included."""
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT INTO semantic_facts (key, category, value, confidence)
                   VALUES (?, ?, ?, ?)""",
                ("speculative_fact", "test", json.dumps("maybe_value"), 0.4),
            )
        assembler = _make_assembler(db, user_model_store)
        ctx = assembler.assemble_search_context("anything")
        assert "speculative_fact" not in ctx


# ---------------------------------------------------------------------------
# Tests: mood context section
# ---------------------------------------------------------------------------

class TestSearchContextMoodSignals:
    """Recent mood signals are appended for tone calibration."""

    def test_mood_included_when_present(self, assembler_with_mood):
        """Recent mood signals appear in the context when available."""
        ctx = assembler_with_mood.assemble_search_context("find task list")
        assert "Recent mood context" in ctx

    def test_mood_absent_when_no_profile(self, assembler):
        """No mood section appears when mood_signals profile does not exist."""
        ctx = assembler.assemble_search_context("find task list")
        assert "Recent mood context" not in ctx

    def test_mood_absent_when_recent_signals_empty(self, db, user_model_store):
        """No mood section appears when recent_signals list is empty."""
        mood_data = {"recent_signals": []}
        with db.get_connection("user_model") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO signal_profiles (profile_type, data, samples_count, updated_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                ("mood_signals", json.dumps(mood_data), 0),
            )
        assembler = _make_assembler(db, user_model_store)
        ctx = assembler.assemble_search_context("find anything")
        assert "Recent mood context" not in ctx


# ---------------------------------------------------------------------------
# Tests: fail-open / resilience
# ---------------------------------------------------------------------------

class TestSearchContextFailOpen:
    """The method must degrade gracefully when data sources fail."""

    def test_facts_exception_does_not_crash(self, db, event_bus):
        """If get_semantic_facts() raises, the method still returns a context string."""
        from unittest.mock import MagicMock
        ums = MagicMock()
        ums.get_semantic_facts.side_effect = RuntimeError("DB unavailable")
        ums.get_signal_profile.return_value = None

        assembler = ContextAssembler(db, ums)
        ctx = assembler.assemble_search_context("test query")
        # Must return a non-empty string without raising.
        assert "test query" in ctx
        assert "Current time:" in ctx

    def test_mood_exception_does_not_crash(self, db, event_bus):
        """If get_signal_profile() raises for mood, the method still returns context."""
        from unittest.mock import MagicMock
        ums = MagicMock()
        ums.get_semantic_facts.return_value = []
        ums.get_signal_profile.side_effect = RuntimeError("profile read failed")

        assembler = ContextAssembler(db, ums)
        ctx = assembler.assemble_search_context("another test")
        assert "another test" in ctx
        assert "Current time:" in ctx

    def test_output_is_richer_than_old_stub(self, assembler_with_facts):
        """The enriched output is longer than the old one-liner stub."""
        query = "project status update"
        old_stub = f"User is searching across their entire digital life for: {query}"
        new_ctx = assembler_with_facts.assemble_search_context(query)
        assert len(new_ctx) > len(old_stub), (
            f"Enriched context ({len(new_ctx)} chars) must exceed "
            f"old stub ({len(old_stub)} chars)"
        )
