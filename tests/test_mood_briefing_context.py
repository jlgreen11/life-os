"""
Tests for _get_mood_context() in ContextAssembler.

Verifies that the morning briefing mood section uses the pre-aggregated
mood_history snapshot instead of raw recent_signals JSON, and that the
output is human-readable and LLM-friendly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_mood_row(db: DatabaseManager, **overrides):
    """Insert a mood_history row with sensible defaults, applying overrides."""
    defaults = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "energy_level": 0.75,
        "stress_level": 0.25,
        "emotional_valence": 0.70,
        "social_battery": 0.60,
        "cognitive_load": 0.30,
        "confidence": 0.80,
        "trend": "stable",
        "contributing_signals": "[]",
    }
    row = {**defaults, **overrides}
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO mood_history
               (timestamp, energy_level, stress_level, emotional_valence,
                social_battery, cognitive_load, confidence, contributing_signals, trend)
               VALUES (:timestamp, :energy_level, :stress_level, :emotional_valence,
                       :social_battery, :cognitive_load, :confidence,
                       :contributing_signals, :trend)""",
            row,
        )


# ---------------------------------------------------------------------------
# Tests: _get_mood_context()
# ---------------------------------------------------------------------------


class TestGetMoodContext:
    """Unit tests for ContextAssembler._get_mood_context()."""

    def test_returns_empty_when_no_history(self, db, user_model_store):
        """Returns empty string when mood_history has no rows."""
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert result == ""

    def test_returns_empty_when_history_too_old(self, db, user_model_store):
        """Returns empty string when most recent mood snapshot is older than 24h."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        _insert_mood_row(db, timestamp=old_ts)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert result == ""

    def test_returns_context_with_recent_history(self, db, user_model_store):
        """Returns a non-empty string when a recent mood snapshot exists."""
        _insert_mood_row(db)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert result.startswith("User mood context:")

    def test_contains_energy_level(self, db, user_model_store):
        """energy_level is formatted with value and label."""
        _insert_mood_row(db, energy_level=0.82)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "energy_level=0.82" in result
        assert "(high)" in result

    def test_contains_stress_level(self, db, user_model_store):
        """stress_level is formatted with value and label."""
        _insert_mood_row(db, stress_level=0.20)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "stress_level=0.20" in result
        assert "(low)" in result

    def test_contains_emotional_valence(self, db, user_model_store):
        """emotional_valence is formatted with positive/neutral/negative label."""
        _insert_mood_row(db, emotional_valence=0.70)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "emotional_valence=0.70" in result
        assert "(positive)" in result

    def test_contains_social_battery(self, db, user_model_store):
        """social_battery is included as a bare numeric value."""
        _insert_mood_row(db, social_battery=0.45)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "social_battery=0.45" in result

    def test_contains_trend(self, db, user_model_store):
        """trend field appears in the output."""
        _insert_mood_row(db, trend="improving")
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "trend=improving" in result

    def test_confidence_included_when_high(self, db, user_model_store):
        """confidence is included when ≥0.3."""
        _insert_mood_row(db, confidence=0.80)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "confidence=0.80" in result

    def test_confidence_omitted_when_low(self, db, user_model_store):
        """confidence is omitted when <0.3 (too uncertain to be useful)."""
        _insert_mood_row(db, confidence=0.20)
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        assert "confidence" not in result

    def test_uses_most_recent_row(self, db, user_model_store):
        """When multiple rows exist, the most recent one is used."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        _insert_mood_row(db, timestamp=old_ts, energy_level=0.20, trend="declining")
        _insert_mood_row(db, energy_level=0.90, trend="improving")  # newer row
        ca = ContextAssembler(db, user_model_store)
        result = ca._get_mood_context()
        # Should reflect the newer high-energy row
        assert "energy_level=0.90" in result
        assert "trend=improving" in result
        assert "energy_level=0.20" not in result

    # -----------------------------------------------------------------------
    # Label boundary tests
    # -----------------------------------------------------------------------

    def test_energy_label_high(self, db, user_model_store):
        """energy_level ≥0.65 → 'high'."""
        _insert_mood_row(db, energy_level=0.65)
        ca = ContextAssembler(db, user_model_store)
        assert "(high)" in ca._get_mood_context()

    def test_energy_label_moderate(self, db, user_model_store):
        """0.35 ≤ energy_level < 0.65 → 'moderate'."""
        _insert_mood_row(db, energy_level=0.50)
        ca = ContextAssembler(db, user_model_store)
        assert "(moderate)" in ca._get_mood_context()

    def test_energy_label_low(self, db, user_model_store):
        """energy_level < 0.35 → 'low'."""
        _insert_mood_row(db, energy_level=0.20)
        ca = ContextAssembler(db, user_model_store)
        assert "(low)" in ca._get_mood_context()

    def test_stress_label_high(self, db, user_model_store):
        """stress_level ≥0.65 → 'high'."""
        _insert_mood_row(db, stress_level=0.70)
        ca = ContextAssembler(db, user_model_store)
        # context will contain something like "stress_level=0.70 (high)"
        result = ca._get_mood_context()
        # check the stress dimension specifically
        stress_part = [p for p in result.split(", ") if "stress_level" in p]
        assert stress_part, f"stress_level not in result: {result}"
        assert "(high)" in stress_part[0]

    def test_valence_label_negative(self, db, user_model_store):
        """emotional_valence < 0.35 → 'negative'."""
        _insert_mood_row(db, emotional_valence=0.20)
        ca = ContextAssembler(db, user_model_store)
        assert "(negative)" in ca._get_mood_context()

    def test_valence_label_neutral(self, db, user_model_store):
        """0.35 ≤ emotional_valence < 0.65 → 'neutral'."""
        _insert_mood_row(db, emotional_valence=0.50)
        ca = ContextAssembler(db, user_model_store)
        assert "(neutral)" in ca._get_mood_context()

    def test_valence_label_positive(self, db, user_model_store):
        """emotional_valence ≥0.65 → 'positive'."""
        _insert_mood_row(db, emotional_valence=0.80)
        ca = ContextAssembler(db, user_model_store)
        assert "(positive)" in ca._get_mood_context()

    # -----------------------------------------------------------------------
    # Integration: briefing includes mood context
    # -----------------------------------------------------------------------

    def test_briefing_context_includes_mood_section(self, db, user_model_store):
        """assemble_briefing_context() includes the mood section when data present."""
        _insert_mood_row(db, energy_level=0.75, stress_level=0.25)
        ca = ContextAssembler(db, user_model_store)
        ctx = ca.assemble_briefing_context()
        assert "User mood context:" in ctx
        assert "energy_level" in ctx
        assert "stress_level" in ctx

    def test_briefing_context_excludes_raw_signals(self, db, user_model_store):
        """assemble_briefing_context() does NOT include raw signal_type JSON blobs."""
        _insert_mood_row(db)
        ca = ContextAssembler(db, user_model_store)
        ctx = ca.assemble_briefing_context()
        # Raw signal dicts look like {"signal_type": "...", "value": ...}
        assert '"signal_type"' not in ctx, (
            "Raw signal_type keys should not appear in briefing context"
        )

    def test_briefing_context_mood_absent_when_no_history(self, db, user_model_store):
        """assemble_briefing_context() omits mood section when mood_history is empty."""
        ca = ContextAssembler(db, user_model_store)
        ctx = ca.assemble_briefing_context()
        assert "User mood context:" not in ctx
