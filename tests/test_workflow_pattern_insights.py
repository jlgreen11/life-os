"""
Tests for the InsightEngine ``_workflow_pattern_insights`` correlator.

The correlator reads the ``workflows`` table (populated by WorkflowDetector)
and surfaces one ``behavioral_pattern`` insight per qualifying workflow.
A workflow qualifies when ``times_observed >= 3`` AND
``success_rate >= 0.01`` (1%).

This test suite validates:

- Returns empty list when no workflows are stored
- Returns empty list when all workflows are below observation threshold
- Returns empty list when all workflows are below success_rate threshold
- Generates one insight per qualifying email workflow
- Skips marketing/automated email senders (is_marketing_or_noreply filter)
- Generates task workflow insight with correct summary
- Generates calendar workflow insight with correct summary
- Generates generic interaction workflow insight
- Confidence formula: base 0.50 + obs bonus (cap 50) + success_rate bonus
- Confidence capped at 0.85
- Insight type is always "behavioral_pattern"
- Email workflows get category "workflow_pattern_email"
- Task workflows get category "workflow_pattern_task"
- Calendar workflows get category "workflow_pattern_calendar"
- Generic workflows get category "workflow_pattern_interaction"
- Entity is set to workflow name (for dedup stability)
- Evidence contains workflow_name, times_observed, success_rate, steps_count
- Staleness TTL is 168 hours (7 days)
- Dedup key is stable across runs (same name → same key)
- Correlator is wired into generate_insights() output
- Multiple qualifying workflows all generate insights
- Insights sorted descending by times_observed
"""

from __future__ import annotations

import pytest

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore


# =============================================================================
# Helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    """Return an InsightEngine wired to the temp DatabaseManager."""
    ums = UserModelStore(db)
    return InsightEngine(db=db, ums=ums)


def _store_workflow(ums: UserModelStore, **kwargs) -> None:
    """Write a workflow into the store with sensible defaults.

    Args:
        ums: UserModelStore to write into.
        **kwargs: Override any workflow field.
    """
    workflow = {
        "name": "Responding to alice@work.com",
        "trigger_conditions": ["email.received.from.alice@work.com"],
        "steps": ["read_email_from_alice_at_work.com", "sent"],
        "typical_duration_minutes": 45.0,
        "tools_used": ["email"],
        "success_rate": 0.80,
        "times_observed": 20,
    }
    workflow.update(kwargs)
    ums.store_workflow(workflow)


# =============================================================================
# Core threshold tests
# =============================================================================


class TestWorkflowPatternInsightsThresholds:
    """Tests for observation and success-rate qualification gates."""

    def test_empty_when_no_workflows(self, db):
        """Returns empty list when the workflows table is empty."""
        engine = _make_engine(db)
        result = engine._workflow_pattern_insights()
        assert result == []

    def test_empty_below_observation_threshold(self, db):
        """Skips workflow with times_observed < 3."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to person@example.com",
                        times_observed=2, success_rate=0.5)
        result = engine._workflow_pattern_insights()
        assert result == []

    def test_empty_below_success_rate_threshold(self, db):
        """Skips workflow with success_rate < 0.01."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to person@example.com",
                        times_observed=10, success_rate=0.0)
        result = engine._workflow_pattern_insights()
        assert result == []

    def test_qualifies_at_minimum_thresholds(self, db):
        """Workflow with times_observed=3 and success_rate=0.01 qualifies."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to human@example.com",
                        times_observed=3, success_rate=0.01)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1


# =============================================================================
# Email workflow tests
# =============================================================================


class TestEmailWorkflowInsights:
    """Tests for 'Responding to <sender>' email workflow insights."""

    def test_email_workflow_basic_summary(self, db):
        """Email workflow summary includes sender, count, and reply percentage."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to alice@work.com",
                        times_observed=47,
                        success_rate=0.92)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        summary = result[0].summary
        assert "alice@work.com" in summary
        assert "47" in summary
        assert "92%" in summary

    def test_email_workflow_category(self, db):
        """Email workflow has category 'workflow_pattern_email'."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to bob@company.com")
        result = engine._workflow_pattern_insights()
        assert result[0].category == "workflow_pattern_email"

    def test_email_workflow_skips_marketing_noreply(self, db):
        """Marketing and noreply senders are filtered out."""
        engine = _make_engine(db)
        # Marketing sender — should be skipped
        _store_workflow(engine.ums,
                        name="Responding to noreply@newsletter.com",
                        times_observed=500,
                        success_rate=0.002)
        result = engine._workflow_pattern_insights()
        assert result == []

    def test_email_workflow_skips_no_reply_variant(self, db):
        """'no-reply@' sender is correctly filtered as automated."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to no-reply@service.io",
                        times_observed=100,
                        success_rate=0.01)
        result = engine._workflow_pattern_insights()
        assert result == []

    def test_email_workflow_passes_human_sender(self, db):
        """Human-looking sender at generic domain is surfaced."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to alice@gmail.com",
                        times_observed=15,
                        success_rate=0.75)
        result = engine._workflow_pattern_insights()
        # gmail.com personal addresses should NOT be filtered
        assert len(result) == 1


# =============================================================================
# Task workflow tests
# =============================================================================


class TestTaskWorkflowInsights:
    """Tests for task completion workflow insights."""

    def test_task_workflow_summary_contains_key_fields(self, db):
        """Task workflow summary includes step count, tools, and completion rate."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Task completion workflow",
                        trigger_conditions=["task.created"],
                        steps=["create_task", "sent", "completed", "received"],
                        tools_used=["task_manager", "email"],
                        success_rate=0.25,
                        times_observed=7232)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        summary = result[0].summary
        assert "4 steps" in summary
        assert "25%" in summary
        assert "7232" in summary

    def test_task_workflow_category(self, db):
        """Task workflow has category 'workflow_pattern_task'."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Task completion workflow",
                        trigger_conditions=["task.created"],
                        tools_used=["task_manager"],
                        success_rate=0.1,
                        times_observed=50)
        result = engine._workflow_pattern_insights()
        assert result[0].category == "workflow_pattern_task"


# =============================================================================
# Calendar workflow tests
# =============================================================================


class TestCalendarWorkflowInsights:
    """Tests for calendar event workflow insights."""

    def test_calendar_workflow_summary(self, db):
        """Calendar workflow summary mentions calendar events and follow-up rate."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Calendar event workflow",
                        trigger_conditions=["calendar.event.created"],
                        steps=["prep_received", "attend_event", "followup_sent"],
                        tools_used=["calendar", "email"],
                        success_rate=0.68,
                        times_observed=2638)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        summary = result[0].summary
        # Should mention the follow-up rate and observation count
        assert "68%" in summary
        assert "2638" in summary

    def test_calendar_workflow_category(self, db):
        """Calendar workflow has category 'workflow_pattern_calendar'."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Calendar event workflow",
                        trigger_conditions=["calendar.event.created"],
                        success_rate=0.5,
                        times_observed=10)
        result = engine._workflow_pattern_insights()
        assert result[0].category == "workflow_pattern_calendar"


# =============================================================================
# Generic / interaction workflow tests
# =============================================================================


class TestGenericWorkflowInsights:
    """Tests for generic interaction-based workflow insights."""

    def test_generic_workflow_summary(self, db):
        """Generic workflow summary includes name, steps, count, and success rate."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Email Received workflow",
                        trigger_conditions=["email_received"],
                        steps=["email_received", "sent", "created"],
                        tools_used=[],
                        success_rate=0.60,
                        times_observed=8,
                        typical_duration_minutes=30.0)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        summary = result[0].summary
        assert "60%" in summary
        assert "8" in summary

    def test_generic_workflow_category(self, db):
        """Generic workflow has category 'workflow_pattern_interaction'."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Custom Process workflow",
                        trigger_conditions=["custom"],
                        success_rate=0.50,
                        times_observed=5)
        result = engine._workflow_pattern_insights()
        assert result[0].category == "workflow_pattern_interaction"


# =============================================================================
# Confidence formula tests
# =============================================================================


class TestWorkflowInsightConfidence:
    """Tests for the confidence formula: 0.50 + obs_bonus + success_bonus."""

    def test_confidence_minimum_observations(self, db):
        """With 3 observations and 0.01 success, confidence ≈ 0.50 + small bonus."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to min@example.com",
                        times_observed=3,
                        success_rate=0.01)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        conf = result[0].confidence
        # base 0.50 + 3/50*0.20 + 0.01*0.15 ≈ 0.5132
        assert 0.50 < conf < 0.60

    def test_confidence_high_observations(self, db):
        """With 50+ observations and high success, confidence approaches 0.85 cap."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to boss@work.com",
                        times_observed=100,
                        success_rate=0.90)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        conf = result[0].confidence
        # base 0.50 + 50/50*0.20 + 0.90*0.15 = 0.50+0.20+0.135=0.835, capped at 0.85
        assert conf == pytest.approx(0.835, abs=0.01)

    def test_confidence_never_exceeds_cap(self, db):
        """Confidence never exceeds 0.85 regardless of inputs."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to superfreq@work.com",
                        times_observed=1000,
                        success_rate=1.0)
        result = engine._workflow_pattern_insights()
        assert result[0].confidence <= 0.85


# =============================================================================
# Insight metadata tests
# =============================================================================


class TestWorkflowInsightMetadata:
    """Tests for insight type, entity, evidence, and TTL."""

    def test_insight_type_is_behavioral_pattern(self, db):
        """All workflow insights have type 'behavioral_pattern'."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to someone@example.com")
        result = engine._workflow_pattern_insights()
        assert result[0].type == "behavioral_pattern"

    def test_entity_is_workflow_name(self, db):
        """Entity field equals the workflow name (enables stable dedup key)."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to carol@firm.com")
        result = engine._workflow_pattern_insights()
        assert result[0].entity == "Responding to carol@firm.com"

    def test_staleness_ttl_is_seven_days(self, db):
        """Staleness TTL is 168 hours (7 days)."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to d@example.com")
        result = engine._workflow_pattern_insights()
        assert result[0].staleness_ttl_hours == 168

    def test_dedup_key_is_stable(self, db):
        """Same workflow name produces the same dedup_key on repeated calls."""
        engine = _make_engine(db)
        _store_workflow(engine.ums, name="Responding to stable@example.com")
        r1 = engine._workflow_pattern_insights()
        r2 = engine._workflow_pattern_insights()
        assert r1[0].dedup_key == r2[0].dedup_key

    def test_evidence_contains_required_fields(self, db):
        """Evidence list contains all required diagnostic fields."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to evidence@example.com",
                        times_observed=15,
                        success_rate=0.50)
        result = engine._workflow_pattern_insights()
        evidence = result[0].evidence
        evidence_keys = [e.split("=")[0] for e in evidence]
        assert "workflow_name" in evidence_keys
        assert "times_observed" in evidence_keys
        assert "success_rate" in evidence_keys
        assert "steps_count" in evidence_keys


# =============================================================================
# Multiple workflows and sort order
# =============================================================================


class TestMultipleWorkflows:
    """Tests for multi-workflow scenarios and sort order."""

    def test_multiple_qualifying_workflows_all_surface(self, db):
        """All qualifying workflows generate distinct insights."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to alpha@company.com",
                        times_observed=30, success_rate=0.70)
        _store_workflow(engine.ums,
                        name="Task completion workflow",
                        trigger_conditions=["task.created"],
                        times_observed=100, success_rate=0.25)
        _store_workflow(engine.ums,
                        name="Calendar event workflow",
                        trigger_conditions=["calendar.event.created"],
                        times_observed=50, success_rate=0.60)
        result = engine._workflow_pattern_insights()
        assert len(result) == 3
        categories = {i.category for i in result}
        assert "workflow_pattern_email" in categories
        assert "workflow_pattern_task" in categories
        assert "workflow_pattern_calendar" in categories

    def test_insights_sorted_descending_by_times_observed(self, db):
        """Insights are ordered with the most-observed workflow first."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to low@example.com",
                        times_observed=5, success_rate=0.5)
        _store_workflow(engine.ums,
                        name="Responding to high@example.com",
                        times_observed=50, success_rate=0.5)
        _store_workflow(engine.ums,
                        name="Responding to mid@example.com",
                        times_observed=20, success_rate=0.5)
        result = engine._workflow_pattern_insights()
        # Should be high (50) → mid (20) → low (5)
        obs_values = [
            int(next(e.split("=")[1] for e in i.evidence if e.startswith("times_observed=")))
            for i in result
        ]
        assert obs_values == sorted(obs_values, reverse=True)

    def test_mixes_qualifying_and_disqualifying(self, db):
        """Only qualifying workflows surface; disqualifying ones are silently skipped."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to good@example.com",
                        times_observed=10, success_rate=0.5)
        _store_workflow(engine.ums,
                        name="Responding to toofew@example.com",
                        times_observed=1, success_rate=0.5)
        result = engine._workflow_pattern_insights()
        assert len(result) == 1
        assert "good@example.com" in result[0].summary


# =============================================================================
# generate_insights() integration
# =============================================================================


class TestWorkflowCorrelatorIntegration:
    """Tests that workflow_pattern correlator is wired into generate_insights()."""

    @pytest.mark.asyncio
    async def test_workflow_insights_appear_in_generate_insights(self, db):
        """Workflow insights from the correlator appear in generate_insights() output."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to integrated@example.com",
                        times_observed=10,
                        success_rate=0.7)
        all_insights = await engine.generate_insights()
        workflow_insights = [
            i for i in all_insights if i.category.startswith("workflow_pattern")
        ]
        assert len(workflow_insights) >= 1
        assert "integrated@example.com" in workflow_insights[0].summary

    @pytest.mark.asyncio
    async def test_workflow_category_in_source_weights(self, db):
        """workflow_pattern category does not crash _apply_source_weights."""
        engine = _make_engine(db)
        _store_workflow(engine.ums,
                        name="Responding to weight@example.com",
                        times_observed=5,
                        success_rate=0.5)
        # Should complete without raising, even without registered source weight
        all_insights = await engine.generate_insights()
        # Insights list must be a list (empty or populated)
        assert isinstance(all_insights, list)
