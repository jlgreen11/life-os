"""
Tests verifying that the routes.py feedback endpoint category_to_source map
stays in sync with InsightEngine._apply_source_weights().

Background:
-----------
InsightEngine._apply_source_weights() maps each insight category to a source
weight key, then multiplies the insight's confidence by that weight.  The
routes.py POST /api/insights/{id}/feedback handler has a *separate* copy of
the same map.  When a new correlator is added to the engine (e.g., the
decision_pattern, topic_interest, cadence_response, routine, or spatial
correlators added in PRs #228-250, or the relationship_intelligence and
workflow_pattern correlators in PRs #278-279), if the feedback map in
routes.py is not updated at the same time, user feedback on those insights
silently drops the source weight update — the AI drift system never learns
from the feedback.

This test suite:
  1. Verifies that every category the engine maps has a corresponding entry
     in the routes.py feedback dict source text (string search, no eval).
  2. Tests that feedback via _apply_source_weights() for the previously
     missing categories correctly calls record_interaction() on the right key.
  3. Confirms actionable_alert sub-types are intentionally absent from both
     maps so they always surface regardless of source-weight tuning.
  4. Tests the 6 new categories from PRs #278-279 (reciprocity_imbalance,
     fast_responder, and all 4 workflow_pattern_* variants).
"""

from __future__ import annotations

import inspect

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_insight(category: str) -> Insight:
    """Return a minimal Insight with the given category."""
    return Insight(
        id=f"test-{category}",
        type="behavioral_pattern",
        category=category,
        summary=f"Test insight for {category}",
        confidence=0.8,
        evidence=[],
    )


def _make_engine_with_mock_swm(db):
    """Return an InsightEngine wired to a mock SourceWeightManager."""
    from unittest.mock import MagicMock
    from storage.user_model_store import UserModelStore
    from services.insight_engine.source_weights import SourceWeightManager
    ums = UserModelStore(db)
    mock_swm = MagicMock(spec=SourceWeightManager)
    mock_swm.get_effective_weight.return_value = 1.0
    engine = InsightEngine(db=db, ums=ums, source_weight_manager=mock_swm)
    return engine, mock_swm


def _get_routes_source() -> str:
    """Return the full source text of web/routes.py."""
    from web import routes as routes_module
    return inspect.getsource(routes_module)


def _category_in_routes_map(category: str, routes_src: str) -> bool:
    """Return True if the category key appears in the routes.py category_to_source dict.

    Searches for  "category": "..."  within the category_to_source block of
    the source text.  This avoids eval() while still being precise enough to
    distinguish a key entry from a comment mention.
    """
    return f'"{category}"' in routes_src


# ---------------------------------------------------------------------------
# The complete expected mapping (mirrors engine._apply_source_weights).
# Update this dict whenever a new correlator is added to the engine.
# ---------------------------------------------------------------------------

ALL_ENGINE_CATEGORIES: dict[str, str] = {
    "place": "location.visits",
    "contact_gap": "messaging.direct",
    "email_volume": "email.work",
    "communication_style": "messaging.direct",
    "style_mismatch": "messaging.direct",
    "chronotype": "email.work",
    "peak_hour": "email.work",
    "busiest_day": "email.work",
    "mood_trajectory": "messaging.direct",
    "top_spending_category": "finance.transactions",
    "spending_increase": "finance.transactions",
    "spending_decrease": "finance.transactions",
    "recurring_subscription": "finance.transactions",
    "decision_speed": "email.work",
    "delegation_tendency": "messaging.direct",
    "decision_fatigue": "messaging.direct",
    "top_interests": "email.work",
    "trending_topic": "email.work",
    "response_time_baseline": "email.work",
    "fastest_contacts": "messaging.direct",
    "communication_peak_hours": "email.work",
    "channel_cadence": "email.work",
    "routine_pattern": "email.work",
    "spatial_top_location": "location.visits",
    "spatial_work_location": "location.visits",
    "spatial_location_diversity": "location.visits",
    # Relationship intelligence correlator (PR #278)
    "reciprocity_imbalance": "messaging.direct",
    "fast_responder": "messaging.direct",
    # Workflow pattern correlator (PR #279)
    "workflow_pattern_email": "email.work",
    "workflow_pattern_task": "email.work",
    "workflow_pattern_calendar": "email.work",
    "workflow_pattern_interaction": "messaging.direct",
}

# Categories intentionally excluded from both maps (always surfaced).
INTENTIONALLY_EXCLUDED = {"overdue_task", "upcoming_calendar"}

# The 14 categories fixed in PR #243 (previously missing).
PREVIOUSLY_MISSING_CATEGORIES = [
    "style_mismatch",
    "decision_speed",
    "delegation_tendency",
    "decision_fatigue",
    "top_interests",
    "trending_topic",
    "response_time_baseline",
    "fastest_contacts",
    "communication_peak_hours",
    "channel_cadence",
    "routine_pattern",
    "spatial_top_location",
    "spatial_work_location",
    "spatial_location_diversity",
]

# The 6 categories from PRs #278-279 that were missing until this fix.
PR_278_279_MISSING_CATEGORIES = [
    "reciprocity_imbalance",
    "fast_responder",
    "workflow_pattern_email",
    "workflow_pattern_task",
    "workflow_pattern_calendar",
    "workflow_pattern_interaction",
]


# ---------------------------------------------------------------------------
# Tests: routes.py map completeness
# ---------------------------------------------------------------------------

class TestRoutesFeedbackMapCompleteness:
    """Verify the routes.py feedback endpoint source map covers all engine categories."""

    def test_all_engine_categories_present_in_routes(self):
        """Every category in ALL_ENGINE_CATEGORIES must appear in the routes.py map.

        Uses source-text search to avoid eval() — looks for the quoted category
        string inside the create_routes() source.
        """
        routes_src = _get_routes_source()
        missing = [
            cat for cat in ALL_ENGINE_CATEGORIES
            if not _category_in_routes_map(cat, routes_src)
        ]
        assert not missing, (
            f"These categories are missing from routes.py category_to_source map "
            f"(feedback for these insights will silently drop source weight updates): "
            f"{missing}"
        )

    def test_previously_missing_categories_now_present(self):
        """The 14 categories added in this fix must appear in the routes.py map."""
        routes_src = _get_routes_source()
        still_missing = [
            c for c in PREVIOUSLY_MISSING_CATEGORIES
            if not _category_in_routes_map(c, routes_src)
        ]
        assert not still_missing, (
            f"These previously-missing categories are still absent from routes.py: "
            f"{still_missing}"
        )

    def test_intentionally_excluded_not_in_routes(self):
        """overdue_task and upcoming_calendar must NOT be in the routes.py map.

        These actionable_alert sub-types should always surface regardless of
        source-weight tuning, so they are excluded from both maps.
        """
        routes_src = _get_routes_source()
        # Look for the key in the dict (not in comments).  A comment mention is
        # acceptable; a dict entry would appear as '"overdue_task": ' with a colon.
        for excluded in INTENTIONALLY_EXCLUDED:
            # Check for dict-entry pattern: '"key": '
            assert f'"{excluded}": ' not in routes_src, (
                f"'{excluded}' should NOT be a dict key in routes.py category_to_source "
                f"(it is intentionally excluded from source-weight tuning)"
            )


# ---------------------------------------------------------------------------
# Tests: engine interaction recording for the 14 new categories
# ---------------------------------------------------------------------------

class TestNewCategoryInteractionRecording:
    """Verify _apply_source_weights records interactions for the 14 new categories.

    This confirms the engine side correctly handles these categories at
    generation time (the routes.py tests above confirm the feedback side).
    """

    @pytest.mark.parametrize("category,expected_source", [
        ("style_mismatch", "messaging.direct"),
        ("decision_speed", "email.work"),
        ("delegation_tendency", "messaging.direct"),
        ("decision_fatigue", "messaging.direct"),
        ("top_interests", "email.work"),
        ("trending_topic", "email.work"),
        ("response_time_baseline", "email.work"),
        ("fastest_contacts", "messaging.direct"),
        ("communication_peak_hours", "email.work"),
        ("channel_cadence", "email.work"),
        ("routine_pattern", "email.work"),
        ("spatial_top_location", "location.visits"),
        ("spatial_work_location", "location.visits"),
        ("spatial_location_diversity", "location.visits"),
    ])
    def test_category_records_correct_interaction(self, db, category, expected_source):
        """Each new category must call record_interaction() with the correct source key."""
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight(category)])
        mock_swm.record_interaction.assert_called_once_with(expected_source), (
            f"Category '{category}' should map to source key '{expected_source}'"
        )

    def test_spending_categories_record_finance_transactions(self, db):
        """Spending insight categories use finance.transactions (already in routes.py)."""
        engine, mock_swm = _make_engine_with_mock_swm(db)
        for cat in [
            "top_spending_category",
            "spending_increase",
            "spending_decrease",
            "recurring_subscription",
        ]:
            mock_swm.reset_mock()
            engine._apply_source_weights([_make_insight(cat)])
            mock_swm.record_interaction.assert_called_once_with("finance.transactions"), (
                f"Category '{cat}' should map to 'finance.transactions'"
            )

    def test_intentionally_excluded_categories_record_no_interaction(self, db):
        """overdue_task and upcoming_calendar produce no interaction call."""
        engine, mock_swm = _make_engine_with_mock_swm(db)
        for excluded in INTENTIONALLY_EXCLUDED:
            mock_swm.reset_mock()
            engine._apply_source_weights([_make_insight(excluded)])
            mock_swm.record_interaction.assert_not_called(), (
                f"'{excluded}' should produce no record_interaction() call"
            )


# ---------------------------------------------------------------------------
# Tests: PR #278-279 new categories (reciprocity_imbalance, fast_responder,
#         and workflow_pattern_* variants)
# ---------------------------------------------------------------------------

class TestPR278279NewCategories:
    """Verify the 6 categories added by PRs #278-279 are in both the engine
    _apply_source_weights map and the routes.py category_to_source map.

    Prior to this fix, user feedback on relationship_intelligence insights
    (reciprocity_imbalance, fast_responder) and all four workflow_pattern_*
    categories was silently discarded — the source weight system never learned
    from user reactions to these insights.
    """

    def test_pr278_279_categories_present_in_routes(self):
        """All 6 new categories must appear in routes.py category_to_source."""
        routes_src = _get_routes_source()
        missing = [
            cat for cat in PR_278_279_MISSING_CATEGORIES
            if not _category_in_routes_map(cat, routes_src)
        ]
        assert not missing, (
            f"These PR #278-279 categories are still absent from routes.py: {missing}. "
            f"User feedback on these insights will be silently discarded."
        )

    @pytest.mark.parametrize("category,expected_source", [
        ("reciprocity_imbalance", "messaging.direct"),
        ("fast_responder", "messaging.direct"),
        ("workflow_pattern_email", "email.work"),
        ("workflow_pattern_task", "email.work"),
        ("workflow_pattern_calendar", "email.work"),
        ("workflow_pattern_interaction", "messaging.direct"),
    ])
    def test_new_category_records_correct_interaction(self, db, category, expected_source):
        """Each PR #278-279 category must call record_interaction() with the right source key.

        Verifies the engine-side _apply_source_weights() handles these categories
        at insight generation time so source weights are consulted when ranking.
        """
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight(category)])
        mock_swm.record_interaction.assert_called_once_with(expected_source), (
            f"Category '{category}' should map to source key '{expected_source}'"
        )

    def test_reciprocity_imbalance_maps_to_messaging_direct(self, db):
        """reciprocity_imbalance (relationship balance) maps to messaging.direct source.

        Relationship balance (who initiates conversations) is fundamentally a
        messaging/cadence signal, so it maps to messaging.direct weight — the
        same source key as contact_gap and other person-to-person relationship
        insights.
        """
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight("reciprocity_imbalance")])
        mock_swm.record_interaction.assert_called_once_with("messaging.direct")

    def test_fast_responder_maps_to_messaging_direct(self, db):
        """fast_responder (sub-30-minute average response time) maps to messaging.direct.

        Response-time patterns are measured from inbound/outbound email cadence,
        but the insight is person-centric, so it maps to messaging.direct.
        """
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight("fast_responder")])
        mock_swm.record_interaction.assert_called_once_with("messaging.direct")

    def test_workflow_pattern_email_maps_to_email_work(self, db):
        """workflow_pattern_email maps to email.work source weight."""
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight("workflow_pattern_email")])
        mock_swm.record_interaction.assert_called_once_with("email.work")

    def test_workflow_pattern_interaction_maps_to_messaging_direct(self, db):
        """workflow_pattern_interaction maps to messaging.direct source weight.

        Interaction workflow patterns (e.g., batching quick replies) reflect
        messaging behavior rather than email-processing behavior.
        """
        engine, mock_swm = _make_engine_with_mock_swm(db)
        engine._apply_source_weights([_make_insight("workflow_pattern_interaction")])
        mock_swm.record_interaction.assert_called_once_with("messaging.direct")
