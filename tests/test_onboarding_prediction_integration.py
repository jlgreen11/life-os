"""
Integration tests: Onboarding preferences flow through to PredictionEngine behavior.

Verifies the cross-service contract between OnboardingManager.finalize() and
PredictionEngine.predict_reaction() / _check_follow_up_needs(). This is critical
because both services read/write the same preference keys in preferences.db —
if the key names or value formats drift, the system silently ignores user
configuration set during onboarding.

Tests cover:
1. Quiet hours set during onboarding suppress predictions at night
2. Quiet hours set during onboarding do NOT suppress predictions during the day
3. No quiet hours configured means no suppression at any time
4. Preference key names and value formats match between onboarding and prediction engine
"""

import json

import pytest
from datetime import datetime, timezone

from models.core import ConfidenceGate
from models.user_model import Prediction, ReactionPrediction
from services.onboarding.manager import ONBOARDING_PHASES, OnboardingManager
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


def _complete_onboarding(manager: OnboardingManager, overrides: dict | None = None) -> dict:
    """Submit all required onboarding answers and finalize.

    Fills in sensible defaults for every required question, then applies
    any overrides from the caller (keyed by step id, e.g. "quiet_hours").
    Returns the finalized preferences dict.
    """
    overrides = overrides or {}

    # Default answers for all required (non-info) steps
    defaults = {
        "morning_style": "minimal",
        "tone": "professional",
        "proactivity": "moderate",
        "autonomy": "moderate",
        "drafting": True,
        "domains": "work, personal",
        "priority_people": "none",
        "work_life_boundary": "unified",
        "vault": False,
        "notifications": "batched",
        "quiet_hours": "no quiet hours",
    }
    defaults.update(overrides)

    for phase in ONBOARDING_PHASES:
        if phase["type"] == "info":
            continue
        manager.submit_answer(phase["id"], defaults[phase["id"]])

    return manager.finalize()


def _monkeypatch_now(monkeypatch, hour: int, minute: int = 0) -> None:
    """Patch datetime.now() in the prediction engine to return a fixed Monday time.

    2026-02-16 is a Monday, matching the pattern used in existing quiet hours tests.
    """
    fixed = datetime(2026, 2, 16, hour, minute, 0, tzinfo=timezone.utc)

    class _MockDatetime:
        @staticmethod
        def now(tz=None):
            return fixed

    monkeypatch.setattr("services.prediction_engine.engine.datetime", _MockDatetime)


# ---------------------------------------------------------------------------
# Test 1: Onboarding quiet hours suppress predictions during quiet period
# ---------------------------------------------------------------------------


class TestOnboardingQuietHoursSuppressPredictions:
    """Verify that quiet hours configured during onboarding actually cause
    the PredictionEngine to penalize non-urgent predictions."""

    @pytest.mark.asyncio
    async def test_prediction_penalized_during_onboarding_quiet_hours(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Onboarding sets quiet_hours='10pm to 7am'. At 23:00 (inside quiet hours),
        a non-urgent reminder should receive the quiet-hours penalty."""
        # --- Onboarding phase ---
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        # --- Prediction phase (same db instance) ---
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Simulate 23:00 Monday — inside the 22:00-07:00 quiet window
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        # The quiet hours penalty should fire
        assert "quiet_hours=True" in reaction.reasoning, (
            f"Expected quiet_hours=True in reasoning, got: {reaction.reasoning}"
        )
        # Score should be reduced (base 0.3 + high_conf 0.2 - quiet 0.2 = 0.3)
        score = float(reaction.reasoning.split("score=")[1].split(",")[0])
        assert score <= 0.3, f"Expected quiet hours to reduce score, got {score}"

    @pytest.mark.asyncio
    async def test_prediction_not_penalized_outside_onboarding_quiet_hours(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Onboarding sets quiet_hours='10pm to 7am'. At 10:00 (outside quiet hours),
        a reminder should NOT receive the quiet-hours penalty."""
        # --- Onboarding phase ---
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        # --- Prediction phase ---
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Simulate 10:00 Monday — outside the 22:00-07:00 quiet window
        _monkeypatch_now(monkeypatch, hour=10)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert "quiet_hours=False" in reaction.reasoning, (
            f"Expected quiet_hours=False outside quiet hours, got: {reaction.reasoning}"
        )
        assert reaction.predicted_reaction in ("helpful", "neutral"), (
            f"Non-quiet-hours prediction should not be suppressed, got: {reaction.predicted_reaction}"
        )

    @pytest.mark.asyncio
    async def test_24h_format_quiet_hours_from_onboarding(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Onboarding sets quiet_hours in 24h format ('22:00 - 07:00'). Verify
        the prediction engine reads it correctly."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "22:00 - 07:00"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert "quiet_hours=True" in reaction.reasoning


# ---------------------------------------------------------------------------
# Test 2: Onboarding without quiet hours means no suppression
# ---------------------------------------------------------------------------


class TestOnboardingDefaultsWithoutQuietHours:
    """Verify that when onboarding completes WITHOUT quiet hours, no time-based
    suppression occurs."""

    @pytest.mark.asyncio
    async def test_no_quiet_hours_no_suppression_at_night(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Finalize onboarding with user declining quiet hours. Even at 23:00,
        predict_reaction() should NOT apply the quiet-hours penalty."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "no quiet hours"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert "quiet_hours=False" in reaction.reasoning, (
            f"No quiet hours configured but got: {reaction.reasoning}"
        )

    @pytest.mark.asyncio
    async def test_no_quiet_hours_no_suppression_at_any_hour(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """With no quiet hours configured, predictions should not get the quiet-hours
        penalty at any time of day (tested at 03:00, typically sleep time)."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "none"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        _monkeypatch_now(monkeypatch, hour=3)

        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert "quiet_hours=False" in reaction.reasoning

    @pytest.mark.asyncio
    async def test_vague_quiet_hours_input_gets_default_range(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """When the user gives a vague answer like 'yes' to quiet hours, the
        onboarding parser provides a default 22:00-07:00 range. Verify the
        prediction engine respects this default."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "yes"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # 23:00 should be inside the default 22:00-07:00 range
        _monkeypatch_now(monkeypatch, hour=23)
        prediction = _make_prediction("reminder", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert "quiet_hours=True" in reaction.reasoning, (
            f"Default quiet hours (22:00-07:00) should suppress at 23:00, "
            f"got: {reaction.reasoning}"
        )


# ---------------------------------------------------------------------------
# Test 3: Urgent predictions bypass quiet hours even from onboarding config
# ---------------------------------------------------------------------------


class TestOnboardingQuietHoursUrgentBypass:
    """Verify that urgent prediction types (conflict, risk) bypass quiet hours
    penalty even when onboarding configured quiet hours."""

    @pytest.mark.asyncio
    async def test_conflict_bypasses_onboarding_quiet_hours(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Conflict predictions must surface even during onboarding-configured quiet hours."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("conflict", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        # Conflict should still surface as helpful/neutral even during quiet hours
        assert reaction.predicted_reaction in ("helpful", "neutral"), (
            f"Conflict prediction was suppressed during onboarding quiet hours: "
            f"{reaction.predicted_reaction}"
        )

    @pytest.mark.asyncio
    async def test_risk_bypasses_onboarding_quiet_hours(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Risk predictions must surface even during onboarding-configured quiet hours."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        engine = PredictionEngine(db, user_model_store, timezone="UTC")
        _monkeypatch_now(monkeypatch, hour=23)

        prediction = _make_prediction("risk", confidence=0.7)
        reaction = await engine.predict_reaction(prediction, {})

        assert reaction.predicted_reaction in ("helpful", "neutral"), (
            f"Risk prediction was suppressed during onboarding quiet hours: "
            f"{reaction.predicted_reaction}"
        )


# ---------------------------------------------------------------------------
# Test 4: Schema contract — preference keys and formats match
# ---------------------------------------------------------------------------


class TestPreferenceKeyFormatContract:
    """Schema-contract tests that verify the preference keys and value formats
    written by OnboardingManager.finalize() match what PredictionEngine reads.

    These tests catch drift between the two services — e.g., if onboarding
    renames a key or changes the JSON structure, the prediction engine would
    silently read None instead of the configured value."""

    def test_quiet_hours_key_name_matches(self, db: DatabaseManager):
        """The key name 'quiet_hours' used by finalize() must match the key
        that _is_quiet_hours() queries from user_preferences."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        # Read back exactly what finalize() wrote
        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        assert row is not None, (
            "OnboardingManager.finalize() did not write a 'quiet_hours' key — "
            "PredictionEngine._is_quiet_hours() will never find configured quiet hours"
        )

    def test_quiet_hours_value_is_valid_json_list(self, db: DatabaseManager):
        """The quiet_hours value must be a JSON-parseable list, since
        PredictionEngine._is_quiet_hours() calls json.loads() on it."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        parsed = json.loads(row["value"])
        assert isinstance(parsed, list), (
            f"quiet_hours must be a JSON list, got {type(parsed).__name__}: {parsed}"
        )

    def test_quiet_hours_entries_have_required_fields(self, db: DatabaseManager):
        """Each quiet_hours entry must have 'start', 'end', and 'days' fields,
        matching the schema that _is_quiet_hours() expects."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        quiet_hours = json.loads(row["value"])
        assert len(quiet_hours) > 0, "Expected at least one quiet hours entry"

        for i, qh in enumerate(quiet_hours):
            assert "start" in qh, f"Entry {i} missing 'start' field (required by _is_quiet_hours)"
            assert "end" in qh, f"Entry {i} missing 'end' field (required by _is_quiet_hours)"
            assert "days" in qh, f"Entry {i} missing 'days' field (required by _is_quiet_hours)"
            assert isinstance(qh["days"], list), f"Entry {i} 'days' must be a list"

    def test_quiet_hours_time_format_is_hh_mm(self, db: DatabaseManager):
        """Start/end times must be in HH:MM format, since _is_quiet_hours()
        calls time.fromisoformat() on them."""
        from datetime import time as time_type

        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        quiet_hours = json.loads(row["value"])
        for qh in quiet_hours:
            # These should not raise ValueError
            start = time_type.fromisoformat(qh["start"])
            end = time_type.fromisoformat(qh["end"])
            assert isinstance(start, time_type)
            assert isinstance(end, time_type)

    def test_quiet_hours_days_use_lowercase_names(self, db: DatabaseManager):
        """Day names must be lowercase (e.g., 'monday'), matching the format
        that _is_quiet_hours() compares via now.strftime('%A').lower()."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        quiet_hours = json.loads(row["value"])
        valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        for qh in quiet_hours:
            for day in qh["days"]:
                assert day in valid_days, (
                    f"Day name '{day}' is not a valid lowercase weekday — "
                    f"_is_quiet_hours() uses strftime('%A').lower() which produces: {valid_days}"
                )

    def test_priority_contacts_key_written_by_onboarding(self, db: DatabaseManager):
        """Verify that finalize() writes a 'priority_contacts' key to preferences.db."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(
            onboarding,
            overrides={"priority_people": "Alice - wife, Bob - coworker"},
        )

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'priority_contacts'"
            ).fetchone()

        assert row is not None, (
            "OnboardingManager.finalize() did not write a 'priority_contacts' key"
        )
        parsed = json.loads(row["value"])
        assert isinstance(parsed, list), (
            f"priority_contacts must be a JSON list, got {type(parsed).__name__}"
        )
        assert len(parsed) == 2

    def test_declined_quiet_hours_writes_empty_list(self, db: DatabaseManager):
        """When the user declines quiet hours, finalize() should write an empty
        JSON list [], not omit the key entirely. _is_quiet_hours() handles both
        cases (no row = False, empty list = no ranges to match = False), but
        writing an explicit empty list is cleaner and more observable."""
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "no"})

        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        assert row is not None, "Expected quiet_hours key to exist even when declined"
        parsed = json.loads(row["value"])
        assert parsed == [], f"Declined quiet hours should be [], got: {parsed}"


# ---------------------------------------------------------------------------
# Test 5: End-to-end roundtrip — onboarding -> DB -> prediction engine read
# ---------------------------------------------------------------------------


class TestEndToEndPreferenceRoundtrip:
    """Verify that the full path from onboarding answer -> finalize() -> DB ->
    PredictionEngine read works correctly."""

    def test_quiet_hours_roundtrip_value_integrity(self, db: DatabaseManager):
        """The quiet hours value written by finalize() should be exactly what
        _is_quiet_hours() will read and parse."""
        onboarding = OnboardingManager(db)
        prefs = _complete_onboarding(onboarding, overrides={"quiet_hours": "11pm to 6am"})

        # What finalize() returned in memory
        in_memory_qh = prefs["quiet_hours"]

        # What's in the database
        with db.get_connection("preferences") as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key = 'quiet_hours'"
            ).fetchone()

        db_qh = json.loads(row["value"])

        # They should be identical
        assert in_memory_qh == db_qh, (
            f"Mismatch between finalize() return and DB: {in_memory_qh} != {db_qh}"
        )

        # Verify the parsed times match what we expect
        assert db_qh[0]["start"] == "23:00"
        assert db_qh[0]["end"] == "06:00"

    @pytest.mark.asyncio
    async def test_full_onboarding_to_reaction_pipeline(
        self,
        db: DatabaseManager,
        user_model_store: UserModelStore,
        monkeypatch,
    ):
        """Full integration: onboarding -> finalize -> prediction engine -> reaction.

        This is the most important test: it exercises the complete user journey
        from answering an onboarding question to having that answer actually
        affect prediction behavior."""
        # Step 1: User completes onboarding with quiet hours
        onboarding = OnboardingManager(db)
        _complete_onboarding(onboarding, overrides={"quiet_hours": "10pm to 7am"})

        # Step 2: Create prediction engine (reads from same DB)
        engine = PredictionEngine(db, user_model_store, timezone="UTC")

        # Step 3: At 23:00 (quiet hours) — should be penalized
        _monkeypatch_now(monkeypatch, hour=23)
        prediction = _make_prediction("reminder", confidence=0.7)
        quiet_reaction = await engine.predict_reaction(prediction, {})

        # Step 4: At 10:00 (active hours) — should NOT be penalized
        _monkeypatch_now(monkeypatch, hour=10)
        active_reaction = await engine.predict_reaction(prediction, {})

        # The quiet-hours reaction should have a lower score
        quiet_score = float(quiet_reaction.reasoning.split("score=")[1].split(",")[0])
        active_score = float(active_reaction.reasoning.split("score=")[1].split(",")[0])

        assert quiet_score < active_score, (
            f"Quiet-hours score ({quiet_score}) should be lower than "
            f"active-hours score ({active_score})"
        )
        assert "quiet_hours=True" in quiet_reaction.reasoning
        assert "quiet_hours=False" in active_reaction.reasoning
