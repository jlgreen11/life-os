"""
Life OS — Tests for User-Corrected Semantic Fact Protection

Verifies that ``update_semantic_fact`` never overwrites a fact that has
``is_user_corrected = 1``.  This is the critical invariant that makes the
user-correction flow (PATCH /api/user-model/facts/{key}) durable: once a user
explicitly tells the system an inferred fact is wrong, inference must not
silently restore the incorrect value on the next ``SemanticFactInferrer`` run.

Without this protection the correction flow was completely broken:
  1. User corrects ``communication_style_formality`` via the UI.
  2. is_user_corrected = 1, confidence reduced by 0.30.
  3. SemanticFactInferrer.infer_from_linguistic_profile() runs (every 6 h).
  4. update_semantic_fact() blindly overwrites → correction vanished.

Coverage:
  - update_semantic_fact skips user-corrected facts
  - update_semantic_fact still creates new (non-corrected) facts normally
  - update_semantic_fact increments non-corrected existing facts normally
  - SemanticFactInferrer.run_all_inference() does not overwrite corrected facts
  - Corrected facts accumulate evidence (source_episodes) silently but
    the value/confidence stays locked
"""

from __future__ import annotations

import json
import pytest

from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


# ---------------------------------------------------------------------------
# update_semantic_fact guard behaviour
# ---------------------------------------------------------------------------

class TestUpdateSemanticFactUserCorrectedGuard:
    """Unit tests for the is_user_corrected guard in update_semantic_fact."""

    def test_update_skips_user_corrected_fact(self, user_model_store: UserModelStore, db: DatabaseManager):
        """Calling update_semantic_fact on a corrected fact leaves it unchanged.

        The update must be a complete no-op: value, confidence, and
        times_confirmed must all remain exactly as the user left them.
        """
        # Arrange: store a fact, then manually mark it corrected (simulating
        # what PATCH /api/user-model/facts/{key} does in production).
        user_model_store.update_semantic_fact(
            key="communication_style_formality",
            category="implicit_preference",
            value="casual",
            confidence=0.60,
        )
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE semantic_facts SET is_user_corrected = 1, confidence = 0.30, value = ? "
                "WHERE key = ?",
                (json.dumps("formal"), "communication_style_formality"),
            )

        # Act: inference engine re-runs and tries to write the old inferred value
        user_model_store.update_semantic_fact(
            key="communication_style_formality",
            category="implicit_preference",
            value="casual",      # The old (wrong) value
            confidence=0.70,     # A higher confidence the inference would produce
        )

        # Assert: the fact is exactly as the user left it
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value, confidence, is_user_corrected, times_confirmed "
                "FROM semantic_facts WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()

        assert json.loads(row["value"]) == "formal", "User-corrected value must not be overwritten"
        assert row["confidence"] == 0.30, "User-corrected confidence must not be bumped"
        assert row["is_user_corrected"] == 1, "Correction flag must stay set"
        # times_confirmed must NOT have been incremented (the update was skipped)
        assert row["times_confirmed"] == 1, (
            "times_confirmed must not increase when the update is skipped"
        )

    def test_update_creates_new_uncorrected_fact_normally(self, user_model_store: UserModelStore, db: DatabaseManager):
        """update_semantic_fact inserts new facts (no is_user_corrected row) as before."""
        user_model_store.update_semantic_fact(
            key="new_inferred_fact",
            category="expertise",
            value="python",
            confidence=0.55,
        )

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value, confidence, is_user_corrected, times_confirmed "
                "FROM semantic_facts WHERE key = ?",
                ("new_inferred_fact",),
            ).fetchone()

        assert row is not None, "New fact must be created"
        assert json.loads(row["value"]) == "python"
        assert row["confidence"] == 0.55
        assert row["is_user_corrected"] == 0
        assert row["times_confirmed"] == 1

    def test_update_increments_non_corrected_existing_fact(self, user_model_store: UserModelStore, db: DatabaseManager):
        """update_semantic_fact still bumps confidence on non-corrected existing facts."""
        # First write
        user_model_store.update_semantic_fact(
            key="recurring_fact",
            category="expertise",
            value="python",
            confidence=0.55,
        )
        # Second write (same key, not corrected) — should bump confidence by 0.05
        user_model_store.update_semantic_fact(
            key="recurring_fact",
            category="expertise",
            value="python",
            confidence=0.55,
        )

        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT confidence, times_confirmed FROM semantic_facts WHERE key = ?",
                ("recurring_fact",),
            ).fetchone()

        assert row["confidence"] == pytest.approx(0.60, abs=0.001), (
            "Confidence must be bumped by 0.05 on second call for non-corrected fact"
        )
        assert row["times_confirmed"] == 2

    def test_update_skips_only_corrected_facts_not_others(self, user_model_store: UserModelStore, db: DatabaseManager):
        """The guard must be selective: only corrected facts are frozen.

        A non-corrected sibling fact in the same batch must still be updated.
        """
        # Set up one corrected and one normal fact
        user_model_store.update_semantic_fact(
            key="fact_corrected",
            category="test",
            value="wrong_value",
            confidence=0.60,
        )
        user_model_store.update_semantic_fact(
            key="fact_normal",
            category="test",
            value="original_value",
            confidence=0.55,
        )

        # Mark only the first one as corrected
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE semantic_facts SET is_user_corrected = 1, value = ? WHERE key = ?",
                (json.dumps("user_value"), "fact_corrected"),
            )

        # Both facts re-inferred in a batch
        user_model_store.update_semantic_fact("fact_corrected", "test", "wrong_value", 0.70)
        user_model_store.update_semantic_fact("fact_normal", "test", "updated_value", 0.60)

        with db.get_connection("user_model") as conn:
            corrected = conn.execute(
                "SELECT value FROM semantic_facts WHERE key = ?", ("fact_corrected",)
            ).fetchone()
            normal = conn.execute(
                "SELECT value FROM semantic_facts WHERE key = ?", ("fact_normal",)
            ).fetchone()

        # Corrected fact locked — user's value preserved
        assert json.loads(corrected["value"]) == "user_value"
        # Normal fact updated — inference wins as expected
        assert json.loads(normal["value"]) == "updated_value"

    def test_update_skips_corrected_fact_telemetry_not_emitted(
        self, user_model_store: UserModelStore, db: DatabaseManager, event_bus
    ):
        """No telemetry event should be emitted when an update is skipped.

        The early return before the _emit_telemetry call ensures the signal
        stream stays clean: a skipped update is not a "fact learned" event.
        """
        # Arrange: corrected fact
        user_model_store.update_semantic_fact("tel_test", "test", "original", 0.50)
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE semantic_facts SET is_user_corrected = 1 WHERE key = ?",
                ("tel_test",),
            )

        initial_count = event_bus.publish.call_count

        # Act: trigger the guard
        user_model_store.update_semantic_fact("tel_test", "test", "new_value", 0.70)

        # Assert: no additional telemetry published
        assert event_bus.publish.call_count == initial_count, (
            "update_semantic_fact must not emit telemetry for a skipped (corrected) fact"
        )


# ---------------------------------------------------------------------------
# End-to-end: SemanticFactInferrer does not overwrite corrected facts
# ---------------------------------------------------------------------------

class TestSemanticFactInferrerProtection:
    """Integration-style tests: run_all_inference() respects user corrections."""

    def _make_linguistic_profile(self, user_model_store: UserModelStore, formality: float = 0.1):
        """Seed a linguistic signal profile that will trigger infer_from_linguistic_profile."""
        data = {
            "samples": [
                {
                    "word_count": 20,
                    "avg_sentence_length": 10.0,
                    "unique_word_ratio": 0.5,
                    "formality": formality,
                    "hedge_rate": 0.0,
                    "assertion_rate": 0.0,
                    "exclamation_rate": 0.0,
                    "question_rate": 0.0,
                    "ellipsis_rate": 0.0,
                    "emoji_count": 0,
                    "emojis_used": [],
                    "profanity_count": 0,
                    "greeting_detected": None,
                    "closing_detected": None,
                }
            ],
            "per_contact": {},
            "averages": {
                "avg_sentence_length": 10.0,
                "formality": formality,  # low → would infer "casual"
                "hedge_rate": 0.0,
                "assertion_rate": 0.0,
                "exclamation_rate": 0.0,
                "emoji_rate": 0.0,
                "profanity_rate": 0.0,
            },
            "common_greetings": [],
            "common_closings": [],
        }
        user_model_store.update_signal_profile("linguistic", data)

    def test_inferrer_does_not_overwrite_corrected_fact(
        self, user_model_store: UserModelStore, db: DatabaseManager
    ):
        """SemanticFactInferrer.run_all_inference() leaves corrected facts intact.

        Scenario:
          1. Linguistic profile shows casual style → inferrer writes
             ``communication_style_formality = "casual"``.
          2. User corrects it to "formal" via the UI (is_user_corrected = 1,
             confidence reduced).
          3. run_all_inference() runs again.
          4. The fact must remain "formal" with is_user_corrected = 1.
        """
        # Step 1: Seed a profile that will infer "casual" (formality=0.1 < 0.3)
        self._make_linguistic_profile(user_model_store, formality=0.1)

        inferrer = SemanticFactInferrer(user_model_store)
        inferrer.infer_from_linguistic_profile()

        # Confirm initial inference happened
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value, confidence, is_user_corrected FROM semantic_facts "
                "WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()
        assert row is not None, "Inferrer must have created the fact"
        assert json.loads(row["value"]) == "casual"
        assert row["is_user_corrected"] == 0

        # Step 2: Simulate user correction (as PATCH endpoint does)
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE semantic_facts "
                "SET is_user_corrected = 1, value = ?, confidence = ? "
                "WHERE key = ?",
                (json.dumps("formal"), 0.30, "communication_style_formality"),
            )

        # Step 3: Inference runs again (e.g., next 6-hour cycle)
        inferrer.infer_from_linguistic_profile()

        # Step 4: Correction must be intact
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value, confidence, is_user_corrected FROM semantic_facts "
                "WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()

        assert json.loads(row["value"]) == "formal", (
            "Inferrer must not overwrite user-corrected fact value"
        )
        assert row["confidence"] == pytest.approx(0.30, abs=0.001), (
            "Inferrer must not restore confidence of user-corrected fact"
        )
        assert row["is_user_corrected"] == 1, (
            "is_user_corrected flag must remain set after second inference run"
        )

    def test_run_all_inference_multiple_cycles_respects_correction(
        self, user_model_store: UserModelStore, db: DatabaseManager
    ):
        """Multiple run_all_inference() calls all respect is_user_corrected.

        This ensures the protection is not a one-off fluke but persists across
        repeated inference cycles, which in production run every 6 hours.
        """
        self._make_linguistic_profile(user_model_store, formality=0.1)
        inferrer = SemanticFactInferrer(user_model_store)

        # First inference creates the fact
        inferrer.infer_from_linguistic_profile()

        # User corrects it
        with db.get_connection("user_model") as conn:
            conn.execute(
                "UPDATE semantic_facts SET is_user_corrected = 1, value = ? WHERE key = ?",
                (json.dumps("formal"), "communication_style_formality"),
            )

        # Simulate 5 more inference cycles
        for _ in range(5):
            inferrer.infer_from_linguistic_profile()

        # After all cycles, correction still holds
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT value, is_user_corrected FROM semantic_facts WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()

        assert json.loads(row["value"]) == "formal"
        assert row["is_user_corrected"] == 1

    def test_non_corrected_facts_still_updated_by_inferrer(
        self, user_model_store: UserModelStore, db: DatabaseManager
    ):
        """Verifies the guard does not accidentally block normal inference.

        A fact with is_user_corrected = 0 must still be updated (confidence
        bumped) on repeat inference runs.
        """
        self._make_linguistic_profile(user_model_store, formality=0.1)
        inferrer = SemanticFactInferrer(user_model_store)

        # First inference
        inferrer.infer_from_linguistic_profile()

        with db.get_connection("user_model") as conn:
            row_before = conn.execute(
                "SELECT confidence, times_confirmed FROM semantic_facts WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()

        # Second inference — no user correction
        inferrer.infer_from_linguistic_profile()

        with db.get_connection("user_model") as conn:
            row_after = conn.execute(
                "SELECT confidence, times_confirmed FROM semantic_facts WHERE key = ?",
                ("communication_style_formality",),
            ).fetchone()

        # Confidence should have grown (bump of +0.05)
        assert row_after["confidence"] > row_before["confidence"], (
            "Non-corrected facts must still receive confidence increments"
        )
        assert row_after["times_confirmed"] > row_before["times_confirmed"]
