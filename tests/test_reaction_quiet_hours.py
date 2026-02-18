"""
Tests for predict_reaction() quiet hours integration.

Verifies that the reaction prediction gatekeeper respects:
1. Explicitly configured quiet hours (stored in user_preferences as 'quiet_hours')
2. Naturally observed low-activity hours from the cadence signal profile
3. The _is_quiet_hours() helper method on its own

Prior to this improvement, predict_reaction() used a hardcoded UTC hour check
(before 7 or after 22) which ignored the user's actual configured sleep schedule
and did not adjust for timezone.  The fix queries the same `quiet_hours`
preference that NotificationManager uses, so both gates agree on "do not disturb".
"""

import json
import pytest
from datetime import datetime, timezone

from models.core import ConfidenceGate
from models.user_model import Prediction
from services.prediction_engine.engine import PredictionEngine
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction(prediction_type: str = "reminder", confidence: float = 0.7) -> Prediction:
    """Build a minimal Prediction for testing."""
    return Prediction(
        prediction_type=prediction_type,
        description=f"Test {prediction_type}",
        confidence=confidence,
        confidence_gate=ConfidenceGate.DEFAULT,
        time_horizon="2_hours",
    )


def _store_quiet_hours(db: DatabaseManager, quiet_hours: list[dict]) -> None:
    """Write quiet_hours preference to the preferences DB."""
    with db.get_connection("preferences") as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
               VALUES ('quiet_hours', ?, datetime('now'))""",
            (json.dumps(quiet_hours),),
        )


def _monkeypatch_now(monkeypatch, hour: int, minute: int = 0, weekday_name: str = "monday") -> None:
    """Patch datetime.now() in the prediction engine to return a fixed time.

    Uses a simple wrapper so that calls with ``tz`` keyword or positional
    ``timezone.utc`` argument both work.
    """
    fixed = datetime(2026, 2, 16, hour, minute, 0, tzinfo=timezone.utc)

    class _MockDatetime:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr("services.prediction_engine.engine.datetime", _MockDatetime)


# ---------------------------------------------------------------------------
# Tests for _is_quiet_hours() helper
# ---------------------------------------------------------------------------

class TestIsQuietHours:
    """Unit tests for the _is_quiet_hours() method in isolation."""

    def test_no_quiet_hours_configured_returns_false(
        self, prediction_engine: PredictionEngine
    ):
        """When no quiet hours are configured, _is_quiet_hours() returns False (fail-open)."""
        now = datetime(2026, 2, 16, 23, 0, 0, tzinfo=timezone.utc)
        assert prediction_engine._is_quiet_hours(now) is False

    def test_same_day_range_inside_returns_true(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """A time inside a same-day quiet range should return True."""
        _store_quiet_hours(db, [{"start": "09:00", "end": "11:00", "days": ["monday"]}])
        # Monday at 10:00 — inside range
        now = datetime(2026, 2, 16, 10, 0, 0, tzinfo=timezone.utc)  # 2026-02-16 is Monday
        assert prediction_engine._is_quiet_hours(now) is True

    def test_same_day_range_outside_returns_false(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """A time outside a same-day quiet range should return False."""
        _store_quiet_hours(db, [{"start": "09:00", "end": "11:00", "days": ["monday"]}])
        now = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)  # 12:00, outside 09-11
        assert prediction_engine._is_quiet_hours(now) is False

    def test_overnight_range_before_midnight_returns_true(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """23:00 should be inside a 22:00–07:00 overnight quiet range."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        now = datetime(2026, 2, 16, 23, 30, 0, tzinfo=timezone.utc)
        assert prediction_engine._is_quiet_hours(now) is True

    def test_overnight_range_after_midnight_returns_true(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """03:00 should be inside a 22:00–07:00 overnight quiet range (crosses midnight)."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        now = datetime(2026, 2, 16, 3, 0, 0, tzinfo=timezone.utc)
        assert prediction_engine._is_quiet_hours(now) is True

    def test_overnight_range_middle_of_day_returns_false(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """14:00 should NOT be inside a 22:00–07:00 overnight quiet range."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        now = datetime(2026, 2, 16, 14, 0, 0, tzinfo=timezone.utc)
        assert prediction_engine._is_quiet_hours(now) is False

    def test_wrong_day_of_week_returns_false(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """Quiet hours configured for Saturday only should not fire on Monday."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["saturday"]}])
        now = datetime(2026, 2, 16, 23, 0, 0, tzinfo=timezone.utc)  # Monday
        assert prediction_engine._is_quiet_hours(now) is False

    def test_malformed_data_returns_false(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """Malformed quiet_hours JSON should fail open (return False)."""
        with db.get_connection("preferences") as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES ('quiet_hours', ?, datetime('now'))",
                ("not-valid-json!!!",),
            )
        now = datetime(2026, 2, 16, 23, 0, 0, tzinfo=timezone.utc)
        assert prediction_engine._is_quiet_hours(now) is False

    def test_multiple_ranges_second_matches(
        self, prediction_engine: PredictionEngine, db: DatabaseManager
    ):
        """When multiple ranges are configured, any match returns True."""
        _store_quiet_hours(db, [
            {"start": "22:00", "end": "07:00", "days": ["saturday", "sunday"]},
            {"start": "12:00", "end": "13:00", "days": ["monday"]},  # lunchtime quiet
        ])
        now = datetime(2026, 2, 16, 12, 30, 0, tzinfo=timezone.utc)  # Monday at 12:30
        assert prediction_engine._is_quiet_hours(now) is True


# ---------------------------------------------------------------------------
# Tests for predict_reaction() — configured quiet hours integration
# ---------------------------------------------------------------------------

class TestPredictReactionConfiguredQuietHours:
    """Verify predict_reaction() uses configured quiet hours to penalize predictions."""

    @pytest.mark.asyncio
    async def test_non_urgent_prediction_penalized_in_quiet_hours(
        self,
        prediction_engine: PredictionEngine,
        db: DatabaseManager,
        monkeypatch,
    ):
        """A non-urgent prediction during configured quiet hours should score lower."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        # Patch now to 23:00 Monday (inside quiet hours)
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        score = float(reaction.reasoning.split("score=")[1].split(",")[0])
        # Base 0.3 + high_confidence 0.2 − quiet_hours 0.2 = 0.3, but we
        # just verify it was penalized vs. no quiet hours (daytime score).
        assert score <= 0.3, f"Expected quiet hours to reduce score, got {score}"
        assert "quiet_hours=True" in reaction.reasoning

    @pytest.mark.asyncio
    async def test_urgent_prediction_not_penalized_in_quiet_hours(
        self,
        prediction_engine: PredictionEngine,
        db: DatabaseManager,
        monkeypatch,
    ):
        """Conflict/risk predictions must always surface even during quiet hours."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        _monkeypatch_now(monkeypatch, hour=23)

        for urgent_type in ("conflict", "risk"):
            prediction = _make_prediction(urgent_type, confidence=0.7)
            reaction = await prediction_engine.predict_reaction(prediction, {})

            assert reaction.predicted_reaction in ("helpful", "neutral"), (
                f"{urgent_type} prediction was suppressed during quiet hours"
            )

    @pytest.mark.asyncio
    async def test_prediction_not_penalized_outside_quiet_hours(
        self,
        prediction_engine: PredictionEngine,
        db: DatabaseManager,
        monkeypatch,
    ):
        """During active hours (not quiet), the prediction should not get the quiet penalty."""
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        _monkeypatch_now(monkeypatch, hour=14)  # 14:00 — well outside quiet hours

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        assert "quiet_hours=False" in reaction.reasoning
        assert reaction.predicted_reaction in ("helpful", "neutral")

    @pytest.mark.asyncio
    async def test_no_quiet_hours_config_falls_back_to_no_penalty(
        self,
        prediction_engine: PredictionEngine,
        monkeypatch,
    ):
        """When no quiet hours are configured, no quiet-hours penalty is applied."""
        # 23:00 — old code would have penalized; new code should not (no config)
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_react(prediction, {}) if False else \
                   await prediction_engine.predict_reaction(prediction, {})

        # Without config OR low-activity data, quiet_hours penalty should not fire
        assert "quiet_hours=False" in reaction.reasoning


# ---------------------------------------------------------------------------
# Tests for predict_reaction() — low-activity cadence fallback
# ---------------------------------------------------------------------------

class TestPredictReactionLowActivityFallback:
    """Verify the cadence profile low-activity fallback works when no quiet hours are configured."""

    @pytest.mark.asyncio
    async def test_low_activity_hour_penalizes_non_urgent(
        self,
        prediction_engine: PredictionEngine,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """A hour with < 5% of peak activity should trigger the low-activity penalty."""
        # Set up cadence profile: hour 23 has 1 message, peak is hour 10 with 100 messages.
        user_model_store.update_signal_profile(
            "cadence",
            {
                "hourly_activity": {
                    "10": 100,  # peak
                    "11": 80,
                    "14": 60,
                    "23": 1,    # < 5% of peak → low activity
                }
            },
        )
        _monkeypatch_now(monkeypatch, hour=23)  # No configured quiet hours

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        assert "low_activity=True" in reaction.reasoning
        score = float(reaction.reasoning.split("score=")[1].split(",")[0])
        assert score <= 0.3, f"Expected low-activity penalty to reduce score, got {score}"

    @pytest.mark.asyncio
    async def test_normal_activity_hour_no_penalty(
        self,
        prediction_engine: PredictionEngine,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """A normal-activity hour should not trigger the low-activity penalty."""
        user_model_store.update_signal_profile(
            "cadence",
            {
                "hourly_activity": {
                    "10": 100,
                    "14": 60,   # 60% of peak — not low activity
                    "23": 1,
                }
            },
        )
        _monkeypatch_now(monkeypatch, hour=14)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        assert "low_activity=False" in reaction.reasoning

    @pytest.mark.asyncio
    async def test_low_activity_urgent_prediction_not_penalized(
        self,
        prediction_engine: PredictionEngine,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Conflict/risk predictions bypass the low-activity penalty."""
        user_model_store.update_signal_profile(
            "cadence",
            {"hourly_activity": {"10": 100, "23": 1}},
        )
        _monkeypatch_now(monkeypatch, hour=23)

        for urgent_type in ("conflict", "risk"):
            prediction = _make_prediction(urgent_type, confidence=0.7)
            reaction = await prediction_engine.predict_reaction(prediction, {})
            assert reaction.predicted_reaction in ("helpful", "neutral"), (
                f"{urgent_type} should not be penalized for low-activity hours"
            )

    @pytest.mark.asyncio
    async def test_configured_quiet_hours_takes_precedence_over_low_activity(
        self,
        prediction_engine: PredictionEngine,
        user_model_store: UserModelStore,
        db: DatabaseManager,
        monkeypatch,
    ):
        """When configured quiet hours are active, the low-activity fallback is skipped.

        The reasoning should show quiet_hours=True, low_activity=False (fallback not evaluated).
        """
        _store_quiet_hours(db, [{"start": "22:00", "end": "07:00", "days": ["monday"]}])
        # Also set up cadence profile that would trigger low-activity
        user_model_store.update_signal_profile(
            "cadence",
            {"hourly_activity": {"10": 100, "23": 1}},
        )
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        # quiet_hours takes precedence — low_activity fallback not evaluated
        assert "quiet_hours=True" in reaction.reasoning
        assert "low_activity=False" in reaction.reasoning


# ---------------------------------------------------------------------------
# Reasoning string format regression test
# ---------------------------------------------------------------------------

class TestPredictReactionReasoningFormat:
    """Ensure reasoning string contains all expected fields for observability."""

    @pytest.mark.asyncio
    async def test_reasoning_contains_all_fields(
        self,
        prediction_engine: PredictionEngine,
    ):
        """The reasoning string must include score, dismissals, stress_signals, quiet_hours, and low_activity."""
        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await prediction_engine.predict_reaction(prediction, {})

        assert "score=" in reaction.reasoning
        assert "dismissals=" in reaction.reasoning
        assert "stress_signals=" in reaction.reasoning
        assert "quiet_hours=" in reaction.reasoning
        assert "low_activity=" in reaction.reasoning
