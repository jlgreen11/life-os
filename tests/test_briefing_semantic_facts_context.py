"""Tests for _get_semantic_facts_context() — Layer 2 Semantic Memory in briefing.

PR #265: Replace the flat 7-line semantic facts dump with a categorized,
noise-filtered section that groups facts into values, behavioral patterns, and
preferences.  These tests verify:

  1. The new section header appears ("Layer 2 Semantic Memory")
  2. Facts are grouped by category (values, behavioral, preferences)
  3. Noise tokens (relationship_balance_*, relationship_priority_*, generic
     interest_* words ≤ 6 chars, frequent_location_*, location_domain_*) are
     excluded
  4. Low-confidence facts (< 0.6) are excluded
  5. Per-group caps are respected (values ≤ 5, behavioral ≤ 5, prefs ≤ 8)
  6. Section is omitted when no qualifying facts exist
  7. Ordering: episodic section precedes semantic section (Layer 1 before Layer 2)
  8. Values category facts appear under "Values:" sub-header
  9. Behavioral-pattern keys appear under "Behavioral patterns:" sub-header
  10. Confidence scores are included in each fact line
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def assembler(db, user_model_store):
    """Return a ContextAssembler backed by empty test databases."""
    return ContextAssembler(db, user_model_store)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _seed_fact(ums, key: str, category: str, value: object, confidence: float = 0.9) -> None:
    """Convenience wrapper for update_semantic_fact."""
    ums.update_semantic_fact(key=key, category=category, value=value, confidence=confidence)


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------


class TestSectionHeader:
    """The new section header should appear when qualifying facts exist."""

    def test_header_present_when_facts_exist(self, db, user_model_store, assembler):
        """'Layer 2 Semantic Memory' appears in briefing when facts qualify."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "tuesday")
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" in ctx

    def test_header_absent_when_no_qualifying_facts(self, db, user_model_store, assembler):
        """Section is omitted entirely when no qualifying facts exist."""
        # Only seed a low-confidence fact (below 0.6 threshold)
        _seed_fact(user_model_store, "some_key", "implicit_preference", "value", confidence=0.3)
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_header_absent_when_table_empty(self, db, user_model_store, assembler):
        """Section is omitted when semantic_facts table has no rows."""
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx


# ---------------------------------------------------------------------------
# Category grouping
# ---------------------------------------------------------------------------


class TestCategoryGrouping:
    """Facts are grouped into named sub-sections."""

    def test_values_subheader(self, db, user_model_store, assembler):
        """Facts with category='values' appear under 'Values:' sub-header."""
        _seed_fact(user_model_store, "work_life_boundaries", "values", "flexible_boundaries", 0.95)
        ctx = assembler.assemble_briefing_context()
        assert "Values:" in ctx
        assert "flexible_boundaries" in ctx

    def test_behavioral_subheader_for_communication_style(self, db, user_model_store, assembler):
        """Facts with communication_style_* key appear under 'Behavioral patterns:'."""
        _seed_fact(user_model_store, "communication_style_directness", "implicit_preference", "direct")
        ctx = assembler.assemble_briefing_context()
        assert "Behavioral patterns:" in ctx
        assert "direct" in ctx

    def test_behavioral_subheader_for_stress_baseline(self, db, user_model_store, assembler):
        """stress_baseline key appears under 'Behavioral patterns:'."""
        _seed_fact(user_model_store, "stress_baseline", "implicit_preference", "low_stress")
        ctx = assembler.assemble_briefing_context()
        assert "Behavioral patterns:" in ctx
        assert "low_stress" in ctx

    def test_behavioral_subheader_for_most_productive_day(self, db, user_model_store, assembler):
        """most_productive_day key appears under 'Behavioral patterns:'."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "tuesday")
        ctx = assembler.assemble_briefing_context()
        assert "Behavioral patterns:" in ctx
        assert "tuesday" in ctx

    def test_preferences_subheader(self, db, user_model_store, assembler):
        """Non-relationship, non-behavioral implicit preferences appear under 'Preferences:'."""
        _seed_fact(user_model_store, "work_location_type", "implicit_preference", "home_office")
        ctx = assembler.assemble_briefing_context()
        assert "Preferences:" in ctx
        assert "home_office" in ctx

    def test_confidence_score_in_fact_line(self, db, user_model_store, assembler):
        """Each fact line includes its confidence score."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "tuesday", 0.85)
        ctx = assembler.assemble_briefing_context()
        assert "0.85" in ctx


# ---------------------------------------------------------------------------
# Noise filtering
# ---------------------------------------------------------------------------


class TestNoiseFiltering:
    """Noisy fact keys are excluded from the section."""

    def test_relationship_balance_excluded(self, db, user_model_store, assembler):
        """relationship_balance_* keys are excluded (too numerous, covered elsewhere)."""
        _seed_fact(
            user_model_store,
            "relationship_balance_alice@example.com",
            "implicit_preference",
            "mutual",
        )
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx  # No other qualifying facts seeded

    def test_relationship_priority_excluded(self, db, user_model_store, assembler):
        """relationship_priority_* keys are excluded."""
        _seed_fact(
            user_model_store,
            "relationship_priority_alice@example.com",
            "implicit_preference",
            "high_priority",
        )
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_frequent_location_excluded(self, db, user_model_store, assembler):
        """frequent_location_* keys are excluded (duplicate of primary_work_location)."""
        _seed_fact(
            user_model_store,
            "frequent_location_office",
            "implicit_preference",
            "office",
        )
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_location_domain_excluded(self, db, user_model_store, assembler):
        """location_domain_* keys are excluded."""
        _seed_fact(
            user_model_store,
            "location_domain_office",
            "implicit_preference",
            "work",
        )
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_short_interest_token_excluded(self, db, user_model_store, assembler):
        """interest_* facts with value ≤ 6 chars (stop-words) are excluded."""
        for word in ("here", "more", "free", "shop", "line", "valid"):
            _seed_fact(user_model_store, f"interest_{word}", "implicit_preference", word)
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_longer_interest_value_included(self, db, user_model_store, assembler):
        """interest_* facts with value > 6 chars are kept (genuine topics)."""
        _seed_fact(user_model_store, "interest_machine_learning", "implicit_preference", "machine_learning")
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" in ctx
        assert "machine_learning" in ctx

    def test_low_confidence_excluded(self, db, user_model_store, assembler):
        """Facts below the 0.6 confidence threshold are excluded."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "friday", confidence=0.55)
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" not in ctx

    def test_boundary_confidence_included(self, db, user_model_store, assembler):
        """Facts at exactly 0.6 confidence are included."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "friday", confidence=0.6)
        ctx = assembler.assemble_briefing_context()
        assert "Layer 2 Semantic Memory" in ctx
        assert "friday" in ctx


# ---------------------------------------------------------------------------
# Per-group caps
# ---------------------------------------------------------------------------


class TestPerGroupCaps:
    """Per-category fact counts are capped to avoid context-window bloat."""

    def test_values_capped_at_5(self, db, user_model_store, assembler):
        """At most 5 values facts appear in the section."""
        for i in range(10):
            _seed_fact(user_model_store, f"value_key_{i}", "values", f"val_{i}")
        ctx = assembler.assemble_briefing_context()
        # Count lines that start with "  - " after the "Values:" header
        in_values = False
        values_lines = 0
        for line in ctx.splitlines():
            if "Values:" in line:
                in_values = True
            elif in_values and line.strip().startswith("- "):
                values_lines += 1
            elif in_values and line.strip() and not line.strip().startswith("- "):
                in_values = False
        assert values_lines <= 5, f"Expected ≤ 5 values facts, got {values_lines}"

    def test_behavioral_capped_at_5(self, db, user_model_store, assembler):
        """At most 5 behavioral facts appear in the section."""
        behavioral_keys = [
            "communication_style_verbosity",
            "communication_style_directness",
            "peak_communication_hour",
            "stress_baseline",
            "most_productive_day",
            "incoming_pressure_exposure",
            "chronotype",
        ]
        for k in behavioral_keys:
            _seed_fact(user_model_store, k, "implicit_preference", "value_x")
        ctx = assembler.assemble_briefing_context()

        in_behavioral = False
        behavioral_lines = 0
        for line in ctx.splitlines():
            if "Behavioral patterns:" in line:
                in_behavioral = True
            elif in_behavioral and line.strip().startswith("- "):
                behavioral_lines += 1
            elif in_behavioral and line.strip() and not line.strip().startswith("- "):
                in_behavioral = False
        assert behavioral_lines <= 5, f"Expected ≤ 5 behavioral facts, got {behavioral_lines}"

    def test_preferences_capped_at_8(self, db, user_model_store, assembler):
        """At most 8 preference facts appear in the section."""
        for i in range(15):
            _seed_fact(user_model_store, f"pref_key_{i}", "implicit_preference", f"pref_val_{i}")
        ctx = assembler.assemble_briefing_context()

        in_prefs = False
        pref_lines = 0
        for line in ctx.splitlines():
            if "Preferences:" in line:
                in_prefs = True
            elif in_prefs and line.strip().startswith("- "):
                pref_lines += 1
            elif in_prefs and line.strip() and not line.strip().startswith("- "):
                in_prefs = False
        assert pref_lines <= 8, f"Expected ≤ 8 preference facts, got {pref_lines}"


# ---------------------------------------------------------------------------
# Section ordering (Layer 1 before Layer 2)
# ---------------------------------------------------------------------------


class TestSectionOrdering:
    """Layer 2 Semantic Memory section appears after Layer 1 episodes."""

    def test_semantic_section_after_episodes(self, db, user_model_store, assembler):
        """Episodic (Layer 1) section must appear before semantic (Layer 2) section."""
        import uuid as _uuid

        ts = datetime.now(timezone.utc).isoformat()
        with db.get_connection("user_model") as conn:
            # episodes table requires both id and event_id (NOT NULL).
            conn.execute(
                """INSERT INTO episodes
                       (id, event_id, interaction_type, content_summary,
                        contacts_involved, topics, active_domain, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(_uuid.uuid4()),
                    str(_uuid.uuid4()),
                    "email_sent",
                    "Sent a note to Alice",
                    "[]",
                    "[]",
                    "work",
                    ts,
                ),
            )
            conn.commit()

        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", "tuesday")

        ctx = assembler.assemble_briefing_context()

        episodes_pos = ctx.find("Recent activity (last 24h):")
        semantic_pos = ctx.find("Layer 2 Semantic Memory")

        assert episodes_pos != -1, "Episodes section must be present"
        assert semantic_pos != -1, "Semantic Memory section must be present"
        assert episodes_pos < semantic_pos, (
            "Layer 1 (episodic) section must appear before Layer 2 (semantic)"
        )


# ---------------------------------------------------------------------------
# Direct method tests
# ---------------------------------------------------------------------------


class TestGetSemanticFactsContext:
    """Unit tests for the _get_semantic_facts_context() method directly."""

    def test_returns_empty_string_when_no_facts(self, db, user_model_store, assembler):
        """Returns '' when semantic_facts is empty."""
        result = assembler._get_semantic_facts_context()
        assert result == ""

    def test_returns_empty_when_all_noise(self, db, user_model_store, assembler):
        """Returns '' when all qualifying facts are noise tokens."""
        _seed_fact(user_model_store, "relationship_balance_x@y.com", "implicit_preference", "mutual")
        _seed_fact(user_model_store, "frequent_location_office", "implicit_preference", "office")
        _seed_fact(user_model_store, "interest_here", "implicit_preference", "here")
        result = assembler._get_semantic_facts_context()
        assert result == ""

    def test_values_label_appears(self, db, user_model_store, assembler):
        """'Values:' sub-header appears when at least one values fact qualifies."""
        _seed_fact(user_model_store, "work_life_boundaries", "values", "weekday_only")
        result = assembler._get_semantic_facts_context()
        assert "Values:" in result

    def test_behavioral_label_appears(self, db, user_model_store, assembler):
        """'Behavioral patterns:' sub-header appears for behavioral keys."""
        _seed_fact(user_model_store, "stress_baseline", "implicit_preference", "low_stress")
        result = assembler._get_semantic_facts_context()
        assert "Behavioral patterns:" in result

    def test_preferences_label_appears(self, db, user_model_store, assembler):
        """'Preferences:' sub-header appears for non-behavioral implicit preferences."""
        _seed_fact(user_model_store, "primary_work_location", "implicit_preference", "home office")
        result = assembler._get_semantic_facts_context()
        assert "Preferences:" in result

    def test_value_stripped_of_json_quotes(self, db, user_model_store, assembler):
        """String values stored as JSON with quotes are displayed without them."""
        _seed_fact(user_model_store, "most_productive_day", "implicit_preference", '"tuesday"')
        result = assembler._get_semantic_facts_context()
        assert "tuesday" in result
        # The outer JSON quotes should be stripped in the output
        assert '""tuesday""' not in result

    def test_most_confident_facts_appear_first_within_group(self, db, user_model_store, assembler):
        """Within each group, highest-confidence facts are listed first."""
        _seed_fact(user_model_store, "work_life_x", "values", "high", 0.95)
        _seed_fact(user_model_store, "work_life_y", "values", "low", 0.70)
        result = assembler._get_semantic_facts_context()
        pos_high = result.find("high")
        pos_low = result.find("low")
        assert pos_high < pos_low, "Higher-confidence fact should appear before lower-confidence fact"
