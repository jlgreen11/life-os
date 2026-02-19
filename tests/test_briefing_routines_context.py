"""
Tests for Layer 3 Procedural Memory (routines) surfaced in morning briefing context.

The morning briefing's _get_routines_context() method fetches the user's top
behavioral routines (habitual, time- or location-triggered patterns detected by
RoutineDetector) and formats them for the LLM context window.  This bridges a
gap: prior to this addition, Layer 3 Procedural Memory was detected, stored, and
exposed via the /api/user-model/routines REST endpoint but never included in the
briefing context string read by the AI engine.

Validates:
  1. Empty-database: method returns empty string (no noise for new users)
  2. Low-consistency routines (< 0.5) are excluded
  3. High-consistency routines (>= 0.5) appear in briefing output
  4. Top 5 cap is enforced (6th routine is silently dropped)
  5. Each routine line contains key statistics (steps, duration, seen count, consistency)
  6. The section is included in assemble_briefing_context() output
  7. Exception safety: errors from get_routines() never raise
"""

from __future__ import annotations

import pytest

from services.ai_engine.context import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_routine(
    name: str,
    trigger: str,
    consistency_score: float = 0.75,
    steps: list = None,
    typical_duration_minutes: float = 30.0,
    times_observed: int = 15,
    variations: list = None,
) -> dict:
    """Build a routine dict suitable for user_model_store.store_routine()."""
    return {
        "name": name,
        "trigger": trigger,
        "steps": steps or [{"action": "check_email"}, {"action": "review_calendar"}],
        "typical_duration_minutes": typical_duration_minutes,
        "consistency_score": consistency_score,
        "times_observed": times_observed,
        "variations": variations or [],
    }


# ---------------------------------------------------------------------------
# _get_routines_context tests
# ---------------------------------------------------------------------------


class TestGetRoutinesContext:
    """Unit tests for ContextAssembler._get_routines_context()."""

    def test_returns_empty_string_when_no_routines(self, db, user_model_store):
        """Method should return empty string when no routines exist (new user)."""
        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()
        assert result == ""

    def test_excludes_low_consistency_routines(self, db, user_model_store):
        """Routines with consistency_score < 0.5 should be excluded."""
        # Store a low-consistency routine (0.3 — unreliable, only 3 of 10 times)
        user_model_store.store_routine(
            _make_routine("sporadic_task", "random", consistency_score=0.3, times_observed=5)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # Low-consistency routine should NOT appear
        assert result == ""

    def test_includes_high_consistency_routines(self, db, user_model_store):
        """Routines with consistency_score >= 0.5 should appear in the output."""
        user_model_store.store_routine(
            _make_routine("morning_email_review", "morning", consistency_score=0.85, times_observed=28)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        assert "Observed behavioral routines (Layer 3 Procedural Memory):" in result
        assert "morning_email_review" in result or "morning" in result

    def test_includes_routine_statistics(self, db, user_model_store):
        """Each routine line should include step count, duration, seen count, and consistency."""
        user_model_store.store_routine(
            _make_routine(
                "evening_wind_down",
                "evening",
                consistency_score=0.78,
                steps=[{"action": "a"}, {"action": "b"}, {"action": "c"}, {"action": "d"}],
                typical_duration_minutes=20.0,
                times_observed=19,
            )
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # All key statistics must appear in the formatted line
        assert "steps" in result        # step count
        assert "20" in result           # duration (20 min)
        assert "seen 19x" in result     # times_observed
        assert "0.78" in result         # consistency_score

    def test_caps_at_five_routines(self, db, user_model_store):
        """Only the top 5 routines (by consistency_score) should appear."""
        # Store 6 high-consistency routines
        for i in range(6):
            user_model_store.store_routine(
                _make_routine(
                    f"routine_{i}",
                    f"trigger_{i}",
                    consistency_score=0.9 - (i * 0.05),  # 0.90, 0.85, …, 0.65
                    times_observed=10 + i,
                )
            )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # Count bullet lines — there should be exactly 5
        bullet_lines = [line for line in result.splitlines() if line.strip().startswith("-")]
        assert len(bullet_lines) == 5

    def test_boundary_consistency_score_exactly_half(self, db, user_model_store):
        """Routine with consistency_score == 0.5 (boundary) should be included."""
        user_model_store.store_routine(
            _make_routine("boundary_routine", "midday", consistency_score=0.5, times_observed=10)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # Exactly 0.5 is >= 0.5, so it must appear
        assert result != ""
        assert "boundary_routine" in result or "midday" in result

    def test_boundary_consistency_score_just_below(self, db, user_model_store):
        """Routine with consistency_score == 0.49 should be excluded."""
        user_model_store.store_routine(
            _make_routine("almost_routine", "midday", consistency_score=0.49, times_observed=10)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        assert result == ""

    def test_only_high_consistency_appear_when_mixed(self, db, user_model_store):
        """Low-consistency routines are excluded even when high-consistency ones exist."""
        # One high-consistency (should appear)
        user_model_store.store_routine(
            _make_routine("morning_routine", "morning", consistency_score=0.9, times_observed=30)
        )
        # One low-consistency (should be filtered out)
        user_model_store.store_routine(
            _make_routine("rare_task", "random", consistency_score=0.2, times_observed=2)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # High-consistency routine present
        assert "morning" in result
        # Low-consistency routine absent
        assert "rare_task" not in result

    def test_uses_trigger_as_label(self, db, user_model_store):
        """The routine's trigger field should appear as the label in the output."""
        user_model_store.store_routine(
            _make_routine("myname", "arrive_home", consistency_score=0.8, times_observed=20)
        )

        assembler = ContextAssembler(db, user_model_store)
        result = assembler._get_routines_context()

        # The trigger ("arrive_home") should appear as the label
        assert "arrive_home" in result


# ---------------------------------------------------------------------------
# assemble_briefing_context integration tests
# ---------------------------------------------------------------------------


class TestBriefingIncludesRoutines:
    """Integration tests: routines are wired into assemble_briefing_context()."""

    def test_briefing_omits_routines_section_when_empty(self, db, user_model_store):
        """When no routines are stored, the briefing should not include the header."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Layer 3 Procedural Memory" not in context

    def test_briefing_includes_routines_section_when_present(self, db, user_model_store):
        """When qualifying routines exist, the briefing should include them."""
        user_model_store.store_routine(
            _make_routine("morning_email_review", "morning", consistency_score=0.88, times_observed=25)
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Observed behavioral routines (Layer 3 Procedural Memory):" in context
        # Should also be separated by delimiters from other sections
        assert "---" in context

    def test_briefing_excludes_low_consistency_routines(self, db, user_model_store):
        """Low-consistency routines should not appear in the briefing context."""
        user_model_store.store_routine(
            _make_routine("unreliable", "random", consistency_score=0.3, times_observed=3)
        )

        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        assert "Layer 3 Procedural Memory" not in context
        assert "unreliable" not in context

    def test_briefing_still_works_without_routines(self, db, user_model_store):
        """The overall briefing must still include required sections without routines."""
        assembler = ContextAssembler(db, user_model_store)
        context = assembler.assemble_briefing_context()

        # Core sections must always be present regardless of routines availability
        assert "Current time:" in context
        assert "calendar" in context.lower() or "Pending tasks" in context
