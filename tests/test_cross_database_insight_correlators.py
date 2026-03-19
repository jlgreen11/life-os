"""
Tests for the five new cross-database insight correlators added to InsightEngine:

  1. _mood_finance_correlation_insights  -- stress x spending join
  2. _stress_trigger_insights            -- event-type sequence mining before stress spikes
  3. _weekly_mood_cycle_insights         -- weekly rhythm in mood_history
  4. _prediction_accuracy_insights       -- prediction table accuracy feedback
  5. _episode_satisfaction_insights      -- episode outcome / satisfaction analysis

Each test class focuses on:
  - Data-gate: no insight below minimum row count
  - Happy path: correct insight generated with right category/evidence
  - Suppression: insight not fired when threshold not met
  - Wiring: correlator appears in generate_insights() output
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

from services.insight_engine.engine import InsightEngine
from storage.user_model_store import UserModelStore

# =============================================================================
# Shared helpers
# =============================================================================


def _make_engine(db) -> InsightEngine:
    ums = UserModelStore(db)
    return InsightEngine(db=db, ums=ums, timezone="UTC")


def _insert_mood(
    db,
    stress: float,
    energy: float = 0.6,
    valence: float = 0.6,
    ts: str | None = None,
) -> None:
    timestamp = ts or datetime.now(timezone.utc).isoformat()
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO mood_history
               (timestamp, energy_level, stress_level, emotional_valence,
                social_battery, cognitive_load, confidence, trend)
               VALUES (?, ?, ?, ?, 0.5, 0.5, 0.8, 'stable')""",
            (timestamp, energy, stress, valence),
        )


def _insert_txn(db, amount: float, days_ago: int = 1, category: str = "SHOPPING") -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    payload = json.dumps({"amount": amount, "category": category, "merchant": "TestMerchant"})
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), "finance.transaction.new", "test", ts, 2, payload, "{}"),
        )


def _insert_event(db, event_type: str, ts: str) -> None:
    with db.get_connection("events") as conn:
        conn.execute(
            """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4()), event_type, "test", ts, 2, "{}", "{}"),
        )


def _insert_prediction(
    db,
    prediction_type: str,
    was_accurate: int,
    user_response: str = "acted_on",
) -> None:
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                time_horizon, supporting_signals, was_surfaced, was_accurate,
                user_response, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                prediction_type,
                "test prediction",
                0.6,
                "SUGGEST",
                "24h",
                "{}",
                1,
                was_accurate,
                user_response,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def _insert_episode(
    db,
    satisfaction: float,
    contacts: list[str] | None = None,
    interaction_type: str = "text_input",
    outcome: str = "user_acted",
) -> None:
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO episodes
               (id, timestamp, event_id, interaction_type, content_summary,
                contacts_involved, topics, entities, outcome, user_satisfaction,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(),
                str(uuid.uuid4()),
                interaction_type,
                "test episode",
                json.dumps(contacts or []),
                "[]",
                "[]",
                outcome,
                satisfaction,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


# =============================================================================
# Tests: _mood_finance_correlation_insights
# =============================================================================


class TestMoodFinanceCorrelation:
    def test_no_insight_when_too_few_mood_rows(self, db):
        engine = _make_engine(db)
        # Only 5 mood rows — below the 10-row gate
        for i in range(5):
            _insert_mood(db, stress=0.8)
        _insert_txn(db, amount=200.0)
        insights = engine._mood_finance_correlation_insights()
        assert insights == []

    def test_no_insight_when_too_few_transactions(self, db):
        engine = _make_engine(db)
        for i in range(15):
            _insert_mood(db, stress=0.8 if i % 2 == 0 else 0.3)
        # Only 5 transactions — below the 10-row gate
        for i in range(5):
            _insert_txn(db, amount=100.0)
        insights = engine._mood_finance_correlation_insights()
        assert insights == []

    def test_no_insight_when_difference_below_threshold(self, db):
        """Stressed spend only 10% higher than calm spend — no insight."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)
        # 15 stressed mood readings paired with modest transactions
        for i in range(15):
            ts = (now - timedelta(hours=i * 2)).isoformat()
            _insert_mood(db, stress=0.8, ts=ts)
            _insert_txn(db, amount=110.0, days_ago=i // 12 + 1)  # ~$110 stressed
        # 15 calm mood readings
        for i in range(15):
            ts = (now - timedelta(hours=i * 2 + 1)).isoformat()
            _insert_mood(db, stress=0.3, ts=ts)
            _insert_txn(db, amount=100.0, days_ago=i // 12 + 1)  # $100 calm
        insights = engine._mood_finance_correlation_insights()
        assert insights == []

    def test_insight_fires_when_stressed_spending_much_higher(self, db):
        """High-stress spend 6x calm spend → insight with correct category."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # Use clearly separated time slots so transactions match unambiguously.
        # Stressed slots: even 6-hour offsets (0h, 6h, 12h, 18h, 24h, 30h, 36h, 42h, 48h, 54h)
        # Calm slots: odd 6-hour offsets (3h, 9h, 15h, 21h, 27h, 33h, 39h, 45h, 51h, 57h)
        for i in range(10):
            stressed_ts = (now - timedelta(hours=i * 6)).isoformat()
            _insert_mood(db, stress=0.85, ts=stressed_ts)

        for i in range(10):
            calm_ts = (now - timedelta(hours=i * 6 + 3)).isoformat()
            _insert_mood(db, stress=0.25, ts=calm_ts)

        # Stressed transactions: 30 min after stressed mood (clearly within 3h, closest to stressed)
        for i in range(5):
            ts = (now - timedelta(hours=i * 6, minutes=30)).isoformat()
            with db.get_connection("events") as conn:
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "finance.transaction.new",
                        "test",
                        ts,
                        2,
                        json.dumps({"amount": 300.0, "category": "SHOPPING"}),
                        "{}",
                    ),
                )

        # Calm transactions: 30 min after calm mood (closest to calm mood at 3h offset)
        for i in range(5):
            ts = (now - timedelta(hours=i * 6 + 3, minutes=30)).isoformat()
            with db.get_connection("events") as conn:
                conn.execute(
                    """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        "finance.transaction.new",
                        "test",
                        ts,
                        2,
                        json.dumps({"amount": 50.0, "category": "FOOD"}),
                        "{}",
                    ),
                )

        insights = engine._mood_finance_correlation_insights()
        mood_finance = [i for i in insights if i.category == "mood_finance_correlation"]
        assert len(mood_finance) == 1
        insight = mood_finance[0]
        assert insight.type == "behavioral_pattern"
        assert "stress" in insight.summary.lower() or "stressed" in insight.summary.lower()
        assert insight.confidence > 0.0
        assert any("avg_stressed_spend" in e for e in insight.evidence)


# =============================================================================
# Tests: _stress_trigger_insights
# =============================================================================


class TestStressTriggerInsights:
    def test_no_insight_when_too_few_mood_rows(self, db):
        engine = _make_engine(db)
        _insert_mood(db, stress=0.3)
        _insert_mood(db, stress=0.8)
        insights = engine._stress_trigger_insights()
        assert insights == []

    def test_no_insight_when_too_few_spikes(self, db):
        """Fewer than 5 stress spikes → no insight."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)
        # Only 3 spikes
        for i in range(3):
            ts_calm = (now - timedelta(hours=i * 10 + 5)).isoformat()
            ts_spike = (now - timedelta(hours=i * 10)).isoformat()
            _insert_mood(db, stress=0.3, ts=ts_calm)
            _insert_mood(db, stress=0.8, ts=ts_spike)
        # Pad with stable rows
        for i in range(10):
            _insert_mood(db, stress=0.5, ts=(now - timedelta(hours=100 + i)).isoformat())
        insights = engine._stress_trigger_insights()
        assert insights == []

    def test_insight_fires_for_dominant_trigger_type(self, db):
        """Event type appearing before ≥30% of spikes → insight with correct entity."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # Create 6 stress spikes, each preceded by finance events AND 4 filler events.
        # Each spike window [spike_ts - 2h, spike_ts] will contain 5 events,
        # giving total_events_scanned = 6 * 5 = 30 (above the 20-event gate).
        for i in range(6):
            ts_calm = (now - timedelta(hours=i * 12 + 3)).isoformat()
            ts_spike = (now - timedelta(hours=i * 12)).isoformat()
            _insert_mood(db, stress=0.25, ts=ts_calm)
            _insert_mood(db, stress=0.85, ts=ts_spike)

            # 4 finance trigger events 1h before the spike (dominant type)
            for _ in range(4):
                trigger_ts = (now - timedelta(hours=i * 12 + 1)).isoformat()
                _insert_event(db, "finance.transaction.new", trigger_ts)

            # 1 filler event at 90 min before spike (minority type)
            filler_ts = (now - timedelta(hours=i * 12 + 1, minutes=30)).isoformat()
            _insert_event(db, "email.received", filler_ts)

        insights = engine._stress_trigger_insights()
        trigger_insights = [i for i in insights if i.category == "stress_trigger"]
        assert len(trigger_insights) == 1
        insight = trigger_insights[0]
        assert insight.entity == "finance.transaction.new"
        assert "finance" in insight.summary
        assert any("trigger_event_type=finance.transaction.new" in e for e in insight.evidence)

    def test_no_insight_when_frequency_below_30pct(self, db):
        """Trigger appears before only 2/7 spikes (~28%) → no insight."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        for i in range(7):
            ts_calm = (now - timedelta(hours=i * 12 + 3)).isoformat()
            ts_spike = (now - timedelta(hours=i * 12)).isoformat()
            _insert_mood(db, stress=0.25, ts=ts_calm)
            _insert_mood(db, stress=0.85, ts=ts_spike)

        # Only put the trigger before 2 of the 7 spikes
        for i in range(2):
            trigger_ts = (now - timedelta(hours=i * 12 + 1)).isoformat()
            _insert_event(db, "finance.transaction.new", trigger_ts)

        # Lots of varied noise events to push total_events_scanned above 20
        for i in range(30):
            filler_ts = (now - timedelta(hours=i * 2 + 1, minutes=30)).isoformat()
            _insert_event(db, "email.received", filler_ts)

        insights = engine._stress_trigger_insights()
        trigger_insights = [i for i in insights if i.category == "stress_trigger"]
        assert trigger_insights == []


# =============================================================================
# Tests: _weekly_mood_cycle_insights
# =============================================================================


class TestWeeklyMoodCycleInsights:
    def test_no_insight_when_too_few_rows(self, db):
        engine = _make_engine(db)
        for i in range(10):
            _insert_mood(db, stress=0.5)
        insights = engine._weekly_mood_cycle_insights()
        assert insights == []

    def test_no_insight_when_pattern_flat(self, db):
        """All days equally stressed → no weekly cycle insight."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)
        # 28 days of uniform stress
        for i in range(28):
            ts = (now - timedelta(days=i)).isoformat()
            _insert_mood(db, stress=0.5, ts=ts)
        insights = engine._weekly_mood_cycle_insights()
        cycle_insights = [i for i in insights if i.category == "weekly_mood_cycle"]
        assert cycle_insights == []

    def test_insight_fires_for_clear_weekly_rhythm(self, db):
        """Mondays very stressed, Fridays very calm → weekly cycle insight fires."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)

        # Go back 8 weeks, inserting high stress on Mondays and low stress on Fridays
        for week in range(8):
            # Find Monday of this week relative to now
            days_back = week * 7
            # Monday: stress = 0.9
            monday = now - timedelta(days=days_back + (now.weekday() % 7))
            _insert_mood(db, stress=0.9, ts=monday.isoformat())
            # Also insert Wednesday and Thursday at moderate stress
            _insert_mood(db, stress=0.55, ts=(monday + timedelta(days=2)).isoformat())
            _insert_mood(db, stress=0.52, ts=(monday + timedelta(days=3)).isoformat())
            # Friday: stress = 0.15
            friday = monday + timedelta(days=4)
            _insert_mood(db, stress=0.15, ts=friday.isoformat())

        insights = engine._weekly_mood_cycle_insights()
        cycle_insights = [i for i in insights if i.category == "weekly_mood_cycle"]
        assert len(cycle_insights) == 1
        insight = cycle_insights[0]
        assert insight.type == "behavioral_pattern"
        assert "stress" in insight.summary.lower()
        assert any("most_stressed_day" in e for e in insight.evidence)
        assert insight.confidence > 0.0


# =============================================================================
# Tests: _prediction_accuracy_insights
# =============================================================================


class TestPredictionAccuracyInsights:
    def test_no_insight_when_too_few_predictions(self, db):
        engine = _make_engine(db)
        for _ in range(4):
            _insert_prediction(db, "need", was_accurate=0)
        insights = engine._prediction_accuracy_insights()
        assert insights == []

    def test_low_accuracy_insight_fires(self, db):
        """< 40% accuracy across ≥ 5 predictions → calibration_low insight."""
        engine = _make_engine(db)
        # 1 accurate, 6 inaccurate for 'reminder' type (~14% accuracy)
        _insert_prediction(db, "reminder", was_accurate=1)
        for _ in range(6):
            _insert_prediction(db, "reminder", was_accurate=0, user_response="dismissed")
        insights = engine._prediction_accuracy_insights()
        low = [i for i in insights if i.category == "prediction_calibration_low"]
        assert len(low) == 1
        assert low[0].entity == "reminder"
        assert "reminder" in low[0].summary
        assert any("accuracy_rate" in e for e in low[0].evidence)

    def test_high_accuracy_insight_fires(self, db):
        """≥ 80% accuracy across ≥ 5 predictions → calibration_high insight."""
        engine = _make_engine(db)
        for _ in range(9):
            _insert_prediction(db, "conflict", was_accurate=1)
        _insert_prediction(db, "conflict", was_accurate=0)
        insights = engine._prediction_accuracy_insights()
        high = [i for i in insights if i.category == "prediction_calibration_high"]
        assert len(high) == 1
        assert high[0].entity == "conflict"
        assert "90%" in high[0].summary or "accurate" in high[0].summary.lower()

    def test_mid_range_accuracy_no_insight(self, db):
        """50-60% accuracy → neither high nor low calibration insight."""
        engine = _make_engine(db)
        for _ in range(5):
            _insert_prediction(db, "opportunity", was_accurate=1)
        for _ in range(5):
            _insert_prediction(db, "opportunity", was_accurate=0)
        insights = engine._prediction_accuracy_insights()
        calibration = [i for i in insights if "calibration" in i.category]
        assert calibration == []

    def test_multiple_prediction_types_handled(self, db):
        """Different types can independently trigger high or low calibration."""
        engine = _make_engine(db)
        # 'need' type: high accuracy
        for _ in range(8):
            _insert_prediction(db, "need", was_accurate=1)
        for _ in range(2):
            _insert_prediction(db, "need", was_accurate=0)
        # 'risk' type: low accuracy
        _insert_prediction(db, "risk", was_accurate=1)
        for _ in range(6):
            _insert_prediction(db, "risk", was_accurate=0)
        insights = engine._prediction_accuracy_insights()
        high = [i for i in insights if i.category == "prediction_calibration_high"]
        low = [i for i in insights if i.category == "prediction_calibration_low"]
        assert any(i.entity == "need" for i in high)
        assert any(i.entity == "risk" for i in low)


# =============================================================================
# Tests: _episode_satisfaction_insights
# =============================================================================


class TestEpisodeSatisfactionInsights:
    def test_no_insight_when_too_few_episodes(self, db):
        engine = _make_engine(db)
        for _ in range(10):
            _insert_episode(db, satisfaction=0.8)
        insights = engine._episode_satisfaction_insights()
        assert insights == []

    def test_no_insight_when_satisfaction_uniform(self, db):
        """All contacts near overall average → no contact-level insights."""
        engine = _make_engine(db)
        for i in range(30):
            _insert_episode(db, satisfaction=0.6, contacts=["alice@example.com"])
        insights = engine._episode_satisfaction_insights()
        contact_insights = [
            i for i in insights
            if i.category in ("high_satisfaction_contact", "low_satisfaction_contact")
        ]
        assert contact_insights == []

    def test_high_satisfaction_contact_insight(self, db):
        """Contact with avg satisfaction 0.4 above overall → high_satisfaction_contact insight."""
        engine = _make_engine(db)
        # 20 baseline episodes with no contacts, satisfaction 0.5
        for _ in range(20):
            _insert_episode(db, satisfaction=0.5)
        # 5 high-satisfaction episodes with alice@example.com
        for _ in range(5):
            _insert_episode(db, satisfaction=0.9, contacts=["alice@example.com"])
        insights = engine._episode_satisfaction_insights()
        high = [i for i in insights if i.category == "high_satisfaction_contact"]
        assert len(high) >= 1
        alice_insights = [i for i in high if "alice@example.com" in i.entity]
        assert len(alice_insights) == 1
        assert "alice" in alice_insights[0].summary.lower() or "alice@example.com" in alice_insights[0].summary

    def test_low_satisfaction_contact_insight(self, db):
        """Contact with avg satisfaction 0.4 below overall → low_satisfaction_contact insight."""
        engine = _make_engine(db)
        # 20 baseline episodes with no contacts, satisfaction 0.7
        for _ in range(20):
            _insert_episode(db, satisfaction=0.7)
        # 5 low-satisfaction episodes with draining@example.com
        for _ in range(5):
            _insert_episode(db, satisfaction=0.2, contacts=["draining@example.com"])
        insights = engine._episode_satisfaction_insights()
        low = [i for i in insights if i.category == "low_satisfaction_contact"]
        assert len(low) >= 1
        assert any("draining@example.com" in i.entity for i in low)

    def test_interaction_type_satisfaction_insight(self, db):
        """Interaction type with avg 0.2 above overall → interaction_type_satisfaction insight."""
        engine = _make_engine(db)
        # 20 baseline episodes, type 'text_input', satisfaction 0.5
        for _ in range(20):
            _insert_episode(db, satisfaction=0.5, interaction_type="text_input")
        # 10 high-satisfaction episodes of type 'voice_command'
        for _ in range(10):
            _insert_episode(db, satisfaction=0.9, interaction_type="voice_command")
        insights = engine._episode_satisfaction_insights()
        type_insights = [i for i in insights if i.category == "interaction_type_satisfaction"]
        voice_insights = [i for i in type_insights if i.entity == "voice_command"]
        assert len(voice_insights) == 1
        assert "voice" in voice_insights[0].summary.lower()

    def test_marketing_contacts_excluded(self, db):
        """Marketing/noreply contacts should not generate satisfaction insights."""
        engine = _make_engine(db)
        for _ in range(20):
            _insert_episode(db, satisfaction=0.5)
        for _ in range(5):
            _insert_episode(db, satisfaction=0.95, contacts=["noreply@marketing.com"])
        insights = engine._episode_satisfaction_insights()
        noreply_insights = [
            i for i in insights
            if i.entity and "noreply" in i.entity
        ]
        assert noreply_insights == []


# =============================================================================
# Integration: new correlators appear in generate_insights output
# =============================================================================


class TestNewCorrelatorsWiredIn:
    async def test_prediction_accuracy_wired_in(self, db):
        """prediction_accuracy correlator is wired into generate_insights."""
        engine = _make_engine(db)
        # Create 8 inaccurate 'risk' predictions (should fire low calibration)
        _insert_prediction(db, "risk", was_accurate=1)
        for _ in range(7):
            _insert_prediction(db, "risk", was_accurate=0, user_response="dismissed")
        results = await engine.generate_insights()
        categories = {i.category for i in results}
        assert "prediction_calibration_low" in categories

    async def test_episode_satisfaction_wired_in(self, db):
        """episode_satisfaction correlator is wired into generate_insights."""
        engine = _make_engine(db)
        for _ in range(20):
            _insert_episode(db, satisfaction=0.5)
        for _ in range(5):
            _insert_episode(db, satisfaction=0.95, contacts=["star@example.com"])
        results = await engine.generate_insights()
        categories = {i.category for i in results}
        assert "high_satisfaction_contact" in categories

    async def test_weekly_mood_cycle_wired_in(self, db):
        """weekly_mood_cycle correlator is wired into generate_insights."""
        engine = _make_engine(db)
        now = datetime.now(timezone.utc)
        for week in range(8):
            monday = now - timedelta(days=week * 7 + now.weekday())
            _insert_mood(db, stress=0.9, ts=monday.isoformat())
            _insert_mood(db, stress=0.52, ts=(monday + timedelta(days=2)).isoformat())
            _insert_mood(db, stress=0.50, ts=(monday + timedelta(days=3)).isoformat())
            _insert_mood(db, stress=0.15, ts=(monday + timedelta(days=4)).isoformat())
        results = await engine.generate_insights()
        categories = {i.category for i in results}
        assert "weekly_mood_cycle" in categories
