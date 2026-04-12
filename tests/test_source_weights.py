"""
Tests for the tunable source weight system.

Validates:
    - Default weight seeding
    - Event classification into source_keys
    - User weight updates
    - AI drift learning (engagement/dismissal)
    - Drift decay over time
    - Drift reset
    - Bulk drift recalculation
    - Integration with InsightEngine confidence modulation
    - Custom source creation
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine
from services.insight_engine.models import Insight
from services.insight_engine.source_weights import (
    DECAY_HALF_LIFE_DAYS,
    DEFAULT_WEIGHTS,
    DRIFT_STEP,
    MAX_DRIFT,
    MIN_INTERACTIONS,
    SourceWeightManager,
)


@pytest.fixture()
def swm(db):
    """A SourceWeightManager seeded with defaults."""
    manager = SourceWeightManager(db)
    manager.seed_defaults()
    return manager


# ------------------------------------------------------------------
# Seeding & Initialization
# ------------------------------------------------------------------


class TestSeeding:
    def test_seed_defaults_creates_rows(self, swm):
        """seed_defaults should populate the source_weights table."""
        weights = swm.get_all_weights()
        assert len(weights) == len(DEFAULT_WEIGHTS)

    def test_seed_defaults_idempotent(self, swm):
        """Calling seed_defaults twice should not duplicate rows."""
        swm.seed_defaults()  # Second call
        weights = swm.get_all_weights()
        assert len(weights) == len(DEFAULT_WEIGHTS)

    def test_default_weights_have_zero_drift(self, swm):
        """Freshly seeded weights should have ai_drift = 0."""
        for w in swm.get_all_weights():
            assert w["ai_drift"] == 0.0

    def test_effective_equals_user_weight_initially(self, swm):
        """With zero drift, effective_weight should equal user_weight."""
        for w in swm.get_all_weights():
            assert abs(w["effective_weight"] - w["user_weight"]) < 0.001


# ------------------------------------------------------------------
# Event Classification
# ------------------------------------------------------------------


class TestClassification:
    def test_classify_personal_email(self, swm):
        event = {
            "type": "email.received",
            "payload": {"from": "friend@gmail.com", "subject": "Hey!"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.personal"

    def test_classify_marketing_email_by_sender(self, swm):
        event = {
            "type": "email.received",
            "payload": {"from": "noreply@shop.com", "subject": "Sale!"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.marketing"

    def test_classify_marketing_email_by_header(self, swm):
        event = {
            "type": "email.received",
            "payload": {
                "from": "offers@example.com",
                "subject": "50% off today only!",
                "headers": {"List-Unsubscribe": "<mailto:unsub@example.com>"},
            },
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.marketing"

    def test_classify_transactional_email(self, swm):
        event = {
            "type": "email.received",
            "payload": {"from": "orders@amazon.com", "subject": "Order Confirmation #123"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.transactional"

    def test_classify_newsletter(self, swm):
        event = {
            "type": "email.received",
            "payload": {"from": "newsletter@techcrunch.com", "subject": "Daily digest"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.newsletter"

    def test_classify_work_email(self, swm):
        event = {
            "type": "email.received",
            "payload": {"from": "boss@acme-corp.com", "subject": "Q1 Review"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "email.work"

    def test_classify_direct_message(self, swm):
        event = {
            "type": "message.received",
            "payload": {"from": "friend", "body": "Hey"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "messaging.direct"

    def test_classify_group_message(self, swm):
        event = {
            "type": "message.received",
            "payload": {"from": "alice", "body": "Hey team", "is_group": True},
            "metadata": {},
        }
        assert swm.classify_event(event) == "messaging.group"

    def test_classify_bot_message(self, swm):
        event = {
            "type": "message.received",
            "payload": {"from": "slackbot", "body": "Reminder", "is_bot": True},
            "metadata": {},
        }
        assert swm.classify_event(event) == "messaging.bot"

    def test_classify_calendar_meeting(self, swm):
        event = {
            "type": "calendar.event.created",
            "payload": {"title": "Standup", "attendees": ["alice@co.com"]},
            "metadata": {},
        }
        assert swm.classify_event(event) == "calendar.meetings"

    def test_classify_calendar_reminder(self, swm):
        event = {
            "type": "calendar.event.created",
            "payload": {"title": "Dentist appointment"},
            "metadata": {},
        }
        assert swm.classify_event(event) == "calendar.reminders"

    def test_classify_finance_transaction(self, swm):
        event = {"type": "finance.transaction.new", "payload": {}, "metadata": {}}
        assert swm.classify_event(event) == "finance.transactions"

    def test_classify_sleep(self, swm):
        event = {"type": "sleep.recorded", "payload": {}, "metadata": {}}
        assert swm.classify_event(event) == "health.sleep"

    def test_classify_location(self, swm):
        event = {"type": "location.changed", "payload": {}, "metadata": {}}
        assert swm.classify_event(event) == "location.visits"

    def test_classify_unknown_falls_back(self, swm):
        event = {"type": "custom.thing", "payload": {}, "metadata": {}}
        result = swm.classify_event(event)
        assert result == "custom.general"


# ------------------------------------------------------------------
# User Weight Updates
# ------------------------------------------------------------------


class TestUserWeightUpdates:
    def test_set_user_weight(self, swm):
        """set_user_weight should update the stored value."""
        swm.set_user_weight("email.marketing", 0.1)
        row = swm._get_weight_row("email.marketing")
        assert row["user_weight"] == 0.1

    def test_set_user_weight_clamps_high(self, swm):
        swm.set_user_weight("email.marketing", 1.5)
        row = swm._get_weight_row("email.marketing")
        assert row["user_weight"] == 1.0

    def test_set_user_weight_clamps_low(self, swm):
        swm.set_user_weight("email.marketing", -0.5)
        row = swm._get_weight_row("email.marketing")
        assert row["user_weight"] == 0.0

    def test_set_user_weight_unknown_key_raises(self, swm):
        with pytest.raises(ValueError, match="Unknown source_key"):
            swm.set_user_weight("nonexistent.source", 0.5)

    def test_set_user_weight_records_timestamp(self, swm):
        swm.set_user_weight("email.marketing", 0.1)
        row = swm._get_weight_row("email.marketing")
        assert row["user_set_at"] is not None

    def test_effective_weight_reflects_user_change(self, swm):
        swm.set_user_weight("email.marketing", 0.9)
        eff = swm.get_effective_weight("email.marketing")
        assert abs(eff - 0.9) < 0.01


# ------------------------------------------------------------------
# AI Drift Learning
# ------------------------------------------------------------------


class TestAIDrift:
    def _prime_interactions(self, swm, source_key, count=MIN_INTERACTIONS):
        """Add enough interactions to pass the drift threshold."""
        for _ in range(count):
            swm.record_interaction(source_key)

    def test_drift_does_not_change_below_threshold(self, swm):
        """AI drift should not change until MIN_INTERACTIONS is reached."""
        swm.record_engagement("email.marketing")
        row = swm._get_weight_row("email.marketing")
        assert row["ai_drift"] == 0.0

    def test_engagement_nudges_drift_up(self, swm):
        self._prime_interactions(swm, "email.marketing")
        swm.record_engagement("email.marketing")
        row = swm._get_weight_row("email.marketing")
        assert row["ai_drift"] == pytest.approx(DRIFT_STEP, abs=0.001)

    def test_dismissal_nudges_drift_down(self, swm):
        self._prime_interactions(swm, "email.marketing")
        swm.record_dismissal("email.marketing")
        row = swm._get_weight_row("email.marketing")
        assert row["ai_drift"] == pytest.approx(-DRIFT_STEP, abs=0.001)

    def test_drift_bounded_at_max(self, swm):
        self._prime_interactions(swm, "email.marketing")
        # Push drift past the max
        for _ in range(100):
            swm.record_engagement("email.marketing")
        row = swm._get_weight_row("email.marketing")
        assert row["ai_drift"] <= MAX_DRIFT

    def test_drift_bounded_at_negative_max(self, swm):
        self._prime_interactions(swm, "email.marketing")
        for _ in range(100):
            swm.record_dismissal("email.marketing")
        row = swm._get_weight_row("email.marketing")
        assert row["ai_drift"] >= -MAX_DRIFT

    def test_drift_history_tracked(self, swm):
        self._prime_interactions(swm, "email.marketing")
        swm.record_engagement("email.marketing")
        row = swm._get_weight_row("email.marketing")
        history = json.loads(row["drift_history"])
        assert len(history) == 1
        assert history[0]["reason"] == "engagement"

    def test_drift_history_capped_at_50(self, swm):
        self._prime_interactions(swm, "email.marketing")
        for _ in range(60):
            swm.record_engagement("email.marketing")
        row = swm._get_weight_row("email.marketing")
        history = json.loads(row["drift_history"])
        assert len(history) <= 50

    def test_effective_weight_includes_drift(self, swm):
        """Effective weight should be user_weight + ai_drift."""
        self._prime_interactions(swm, "email.marketing")
        original_weight = swm._get_weight_row("email.marketing")["user_weight"]
        # Apply several engagement signals
        for _ in range(5):
            swm.record_engagement("email.marketing")
        eff = swm.get_effective_weight("email.marketing")
        expected = original_weight + DRIFT_STEP * 5
        assert eff == pytest.approx(min(1.0, expected), abs=0.01)


# ------------------------------------------------------------------
# Drift Reset
# ------------------------------------------------------------------


class TestDriftReset:
    def test_reset_clears_drift(self, swm):
        # First add some drift
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.marketing")
        for _ in range(5):
            swm.record_engagement("email.marketing")

        row_before = swm._get_weight_row("email.marketing")
        assert row_before["ai_drift"] != 0.0

        swm.reset_ai_drift("email.marketing")

        row_after = swm._get_weight_row("email.marketing")
        assert row_after["ai_drift"] == 0.0

    def test_reset_records_in_history(self, swm):
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.marketing")
        swm.record_engagement("email.marketing")
        swm.reset_ai_drift("email.marketing")

        row = swm._get_weight_row("email.marketing")
        history = json.loads(row["drift_history"])
        assert any(h["reason"] == "user_reset" for h in history)

    def test_reset_unknown_key_raises(self, swm):
        with pytest.raises(ValueError, match="Unknown source_key"):
            swm.reset_ai_drift("nonexistent.source")


# ------------------------------------------------------------------
# Drift Decay
# ------------------------------------------------------------------


class TestDriftDecay:
    def test_recent_drift_barely_decays(self, swm):
        """Drift updated moments ago should be nearly unchanged."""
        now = datetime.now(timezone.utc).isoformat()
        decayed = swm._decay_drift(0.2, now)
        assert abs(decayed - 0.2) < 0.01

    def test_drift_halves_after_half_life(self, swm):
        """Drift should be ~half after DECAY_HALF_LIFE_DAYS."""
        past = (datetime.now(timezone.utc) - timedelta(days=DECAY_HALF_LIFE_DAYS)).isoformat()
        decayed = swm._decay_drift(0.2, past)
        assert abs(decayed - 0.1) < 0.02

    def test_drift_nearly_zero_after_long_time(self, swm):
        """Drift should approach zero after several half-lives."""
        past = (datetime.now(timezone.utc) - timedelta(days=DECAY_HALF_LIFE_DAYS * 5)).isoformat()
        decayed = swm._decay_drift(0.3, past)
        assert abs(decayed) < 0.02

    def test_zero_drift_stays_zero(self, swm):
        past = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        decayed = swm._decay_drift(0.0, past)
        assert decayed == 0.0

    def test_no_timestamp_returns_raw(self, swm):
        decayed = swm._decay_drift(0.2, None)
        assert decayed == 0.2


# ------------------------------------------------------------------
# Bulk Drift Recalculation
# ------------------------------------------------------------------


class TestBulkRecalc:
    def test_bulk_recalc_runs_without_error(self, swm):
        """Should not error even with no interactions."""
        swm.bulk_recalculate_drift()

    def test_bulk_recalc_adjusts_outlier_sources(self, swm):
        """A source with very low engagement rate should drift down."""
        # Prime two sources
        for _ in range(MIN_INTERACTIONS * 2):
            swm.record_interaction("email.marketing")
            swm.record_interaction("email.personal")

        # email.personal: lots of engagement
        for _ in range(10):
            swm.record_engagement("email.personal")

        # email.marketing: lots of dismissals
        for _ in range(10):
            swm.record_dismissal("email.marketing")

        swm.bulk_recalculate_drift()

        marketing = swm._get_weight_row("email.marketing")
        personal = swm._get_weight_row("email.personal")

        # Marketing drift should be negative (below-average engagement)
        assert marketing["ai_drift"] < 0
        # Personal drift should be positive (above-average engagement)
        assert personal["ai_drift"] > 0


# ------------------------------------------------------------------
# Custom Sources
# ------------------------------------------------------------------


class TestCustomSources:
    def test_add_custom_source(self, swm):
        result = swm.add_source(
            source_key="email.client_acme",
            category="email",
            label="ACME Corp Email",
            description="All email from acme-corp.com",
            user_weight=0.6,
        )
        assert result is not None
        assert result["source_key"] == "email.client_acme"
        assert result["user_weight"] == 0.6

    def test_custom_source_appears_in_list(self, swm):
        swm.add_source("email.vip", "email", "VIP Contacts", user_weight=0.95)
        all_weights = swm.get_all_weights()
        keys = [w["source_key"] for w in all_weights]
        assert "email.vip" in keys

    def test_get_stats_for_source(self, swm):
        stats = swm.get_source_stats("email.marketing")
        assert stats is not None
        assert stats["source_key"] == "email.marketing"
        assert "effective_weight" in stats
        assert "engagement_rate" in stats
        assert "drift_active" in stats

    def test_get_stats_unknown_returns_none(self, swm):
        assert swm.get_source_stats("nonexistent") is None


# ------------------------------------------------------------------
# InsightEngine Integration
# ------------------------------------------------------------------


class TestInsightEngineIntegration:
    @pytest.mark.asyncio
    async def test_insights_modulated_by_weight(self, db, user_model_store):
        """Insight confidence should be multiplied by source weight."""
        swm = SourceWeightManager(db)
        swm.seed_defaults()

        # Set location weight very low
        swm.set_user_weight("location.visits", 0.1)

        engine = InsightEngine(db, user_model_store, source_weight_manager=swm)

        # Add a place to trigger the place_frequency correlator
        with db.get_connection("entities") as conn:
            conn.execute(
                "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), "Test Cafe", 20, "cafe", datetime.now(timezone.utc).isoformat()),
            )

        insights = await engine.generate_insights()

        # If insights were generated, their confidence should be low
        # because location.visits weight is 0.1
        for i in insights:
            if i.category == "place":
                assert i.confidence < 0.5, (
                    f"Insight confidence {i.confidence} should be reduced by low weight"
                )

    @pytest.mark.asyncio
    async def test_zero_weight_filters_insights(self, db, user_model_store):
        """Insights from a zeroed-out source should be filtered entirely."""
        swm = SourceWeightManager(db)
        swm.seed_defaults()

        # Completely silence location insights
        swm.set_user_weight("location.visits", 0.0)

        engine = InsightEngine(db, user_model_store, source_weight_manager=swm)

        with db.get_connection("entities") as conn:
            conn.execute(
                "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), "Test Cafe", 20, "cafe", datetime.now(timezone.utc).isoformat()),
            )

        insights = await engine.generate_insights()
        place_insights = [i for i in insights if i.category == "place"]
        assert len(place_insights) == 0, "Zeroed-out source should produce no insights"

    @pytest.mark.asyncio
    async def test_engine_works_without_weight_manager(self, db, user_model_store):
        """InsightEngine should work normally when no SourceWeightManager is provided."""
        engine = InsightEngine(db, user_model_store)

        with db.get_connection("entities") as conn:
            conn.execute(
                "INSERT INTO places (id, name, visit_count, place_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), "Cafe", 10, "cafe", datetime.now(timezone.utc).isoformat()),
            )

        insights = await engine.generate_insights()
        assert isinstance(insights, list)


# ------------------------------------------------------------------
# Weights by Category (UI helper)
# ------------------------------------------------------------------


class TestWeightsByCategory:
    def test_returns_grouped_dict(self, swm):
        grouped = swm.get_weights_by_category()
        assert isinstance(grouped, dict)
        assert "email" in grouped
        assert "messaging" in grouped
        assert "calendar" in grouped

    def test_email_category_has_expected_keys(self, swm):
        grouped = swm.get_weights_by_category()
        email_keys = [w["source_key"] for w in grouped["email"]]
        assert "email.personal" in email_keys
        assert "email.marketing" in email_keys
        assert "email.work" in email_keys


# ------------------------------------------------------------------
# Effective Weight for Unknown Source
# ------------------------------------------------------------------


class TestEffectiveWeightEdgeCases:
    def test_unknown_source_returns_neutral(self, swm):
        """Unknown source_key should return 0.5 (neutral)."""
        assert swm.get_effective_weight("unknown.source") == 0.5

    def test_effective_weight_clamped_at_one(self, swm):
        """Even with positive drift, effective weight should not exceed 1.0."""
        swm.set_user_weight("email.personal", 1.0)
        # Manually set drift to max
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.personal")
        for _ in range(100):
            swm.record_engagement("email.personal")

        eff = swm.get_effective_weight("email.personal")
        assert eff <= 1.0

    def test_effective_weight_clamped_at_zero(self, swm):
        """Even with negative drift, effective weight should not go below 0.0."""
        swm.set_user_weight("email.marketing", 0.0)
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.marketing")
        for _ in range(100):
            swm.record_dismissal("email.marketing")

        eff = swm.get_effective_weight("email.marketing")
        assert eff >= 0.0


# ------------------------------------------------------------------
# Drift Saturation Warnings & Diagnostics
# ------------------------------------------------------------------


class TestDriftSaturation:
    """Tests for the drift saturation detection added to record_engagement,
    record_dismissal, bulk_recalculate_drift, and get_diagnostics."""

    def _prime_and_max_drift(self, swm, source_key: str):
        """Prime interactions then engage enough times to saturate drift.

        DRIFT_STEP=0.02, MAX_DRIFT=0.3 → 15 engagements reach saturation.
        """
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction(source_key)
        # 15 engagements * 0.02 = 0.30 == MAX_DRIFT
        for _ in range(15):
            swm.record_engagement(source_key)

    def test_record_engagement_logs_saturation_warning(self, swm, caplog):
        """Engaging with a source until drift hits MAX_DRIFT should log a WARNING."""
        with caplog.at_level(logging.WARNING, logger="services.insight_engine.source_weights"):
            self._prime_and_max_drift(swm, "email.work")

        # At least one saturation warning should have been emitted when drift == MAX_DRIFT
        saturation_warnings = [
            r for r in caplog.records
            if "saturated" in r.message.lower() and "email.work" in r.message
        ]
        assert len(saturation_warnings) >= 1, (
            f"Expected saturation WARNING for email.work; log records: {[r.message for r in caplog.records]}"
        )

    def test_history_entry_includes_saturated_flag(self, swm):
        """The drift history entry written when saturation occurs should contain saturated=True."""
        self._prime_and_max_drift(swm, "email.work")

        row = swm._get_weight_row("email.work")
        history = json.loads(row["drift_history"])

        saturated_entries = [e for e in history if e.get("saturated") is True]
        assert len(saturated_entries) >= 1, (
            f"Expected at least one history entry with saturated=True; history={history}"
        )

    def test_get_diagnostics_returns_saturated_sources(self, swm):
        """get_diagnostics should list sources whose drift is at MAX_DRIFT in saturated_sources."""
        self._prime_and_max_drift(swm, "email.work")

        diag = swm.get_diagnostics()

        assert "saturated_sources" in diag, "get_diagnostics must include 'saturated_sources' key"
        assert "email.work" in diag["saturated_sources"], (
            f"email.work should be in saturated_sources; got {diag['saturated_sources']}"
        )

    def test_get_diagnostics_drift_health_saturated(self, swm):
        """drift_health should be 'saturated' when any source has reached MAX_DRIFT."""
        self._prime_and_max_drift(swm, "email.work")

        diag = swm.get_diagnostics()

        assert diag.get("drift_health") == "saturated", (
            f"Expected drift_health='saturated'; got {diag.get('drift_health')}"
        )

    def test_get_diagnostics_drift_health_active(self, swm):
        """drift_health should be 'active' when there is drift but no saturation."""
        # Apply a small amount of drift (well below MAX_DRIFT)
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.personal")
        swm.record_engagement("email.personal")  # drift = 0.02, far from 0.3

        diag = swm.get_diagnostics()

        assert diag.get("drift_health") == "active", (
            f"Expected drift_health='active'; got {diag.get('drift_health')}"
        )

    def test_get_diagnostics_drift_health_inactive_when_no_drift(self, swm):
        """drift_health should be 'inactive' when no sources have any drift."""
        diag = swm.get_diagnostics()

        # Fresh seeds have zero drift
        assert diag.get("drift_health") == "inactive", (
            f"Expected drift_health='inactive' on fresh seed; got {diag.get('drift_health')}"
        )

    def test_check_drift_saturation_returns_empty_when_not_saturated(self, swm):
        """_check_drift_saturation should return {} when drift is below MAX_DRIFT."""
        result = swm._check_drift_saturation("email.work", user_weight=0.7, new_drift=0.1)
        assert result == {}, f"Expected empty dict for unsaturated drift; got {result}"

    def test_check_drift_saturation_detects_max_drift(self, swm):
        """_check_drift_saturation should detect saturation when new_drift == MAX_DRIFT."""
        result = swm._check_drift_saturation("email.work", user_weight=0.7, new_drift=MAX_DRIFT)
        assert result.get("saturated") is True
        assert result["source_key"] == "email.work"
        assert result["drift"] == MAX_DRIFT

    def test_check_drift_saturation_detects_clamped_effective_weight(self, swm):
        """_check_drift_saturation should detect saturation when effective weight is clamped to 1.0."""
        # user_weight=0.9 + drift=0.2 → unclamped=1.1 → clamped to 1.0
        result = swm._check_drift_saturation("email.personal", user_weight=0.9, new_drift=0.2)
        assert result.get("saturated") is True
        assert result["effective_weight"] == 1.0

    def test_record_dismissal_logs_saturation_warning(self, swm, caplog):
        """Dismissing with a low-weight source to drift floor should log a WARNING."""
        # Set user weight low so negative drift quickly pins effective to 0
        swm.set_user_weight("email.marketing", 0.15)
        for _ in range(MIN_INTERACTIONS):
            swm.record_interaction("email.marketing")

        with caplog.at_level(logging.WARNING, logger="services.insight_engine.source_weights"):
            # 15 dismissals * 0.02 = 0.30, drift = -0.30 → effective = max(0, 0.15-0.30) = 0.0
            for _ in range(15):
                swm.record_dismissal("email.marketing")

        saturation_warnings = [
            r for r in caplog.records
            if "saturated" in r.message.lower() and "email.marketing" in r.message
        ]
        assert len(saturation_warnings) >= 1, (
            f"Expected saturation WARNING for email.marketing; log records: {[r.message for r in caplog.records]}"
        )
