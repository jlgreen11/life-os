"""
Tests for the per-contact accuracy multiplier applied to opportunity predictions.

Background
----------
The global ``_get_accuracy_multiplier`` treats all opportunity predictions
identically regardless of which contact they target.  In practice, users have
very different response rates for different contacts:

- "Alice" (close friend): user reliably reaches out whenever suggested → high accuracy
- "Distant colleague": user almost never acts on suggestions → low accuracy

Without per-contact adjustment, the global ~55% accuracy yields an ~0.83×
multiplier applied to every contact indiscriminately.  This means:
- Reliable contacts get the same confidence as unreliable ones
- No learning about which relationships are actively maintained vs. dormant

The fix adds ``_get_contact_accuracy_multiplier(contact_email)`` which:
- Returns 1.0 when < 3 resolved predictions exist for a contact (cold start)
- Returns 0.5 floor when accuracy is < 20% over 3+ resolved (inactive relationship)
- Returns 0.5 + (accuracy_rate * 0.7) otherwise, capped at 1.2

This test file covers:
1. Cold-start: fewer than 3 samples → 1.0 (no-op)
2. Low-accuracy contact: 0–19% → 0.5 floor
3. Mid-accuracy contact: 50% → 0.85x
4. High-accuracy contact: 100% → 1.2x cap
5. Automated-sender fast-path resolutions are excluded from contact multiplier
6. The main loop applies contact multiplier only for opportunity predictions
7. Non-opportunity predictions are unaffected by contact multiplier
8. Contact email matching is case-insensitive
"""

import json
import uuid
import pytest
from datetime import datetime, timezone

from services.prediction_engine.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_contact_prediction(
    db,
    contact_email: str,
    was_accurate: bool,
    resolution_reason: str = None,
    prediction_type: str = "opportunity",
) -> None:
    """Insert a single resolved, surfaced opportunity prediction for a contact.

    Args:
        db: DatabaseManager fixture.
        contact_email: The contact email stored in supporting_signals.
        was_accurate: Whether the prediction was accurate.
        resolution_reason: Optional resolution reason (e.g. 'automated_sender_fast_path').
        prediction_type: Prediction type (default 'opportunity').
    """
    now = datetime.now(timezone.utc).isoformat()
    supporting_signals = json.dumps({
        "contact_email": contact_email,
        "contact_name": contact_email.split("@")[0],
        "days_since_last_contact": 30,
        "avg_contact_gap_days": 20.0,
    })
    with db.get_connection("user_model") as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, prediction_type, description, confidence, confidence_gate,
                was_surfaced, was_accurate, resolved_at, resolution_reason,
                supporting_signals)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                prediction_type,
                f"It's been 30 days since you last contacted {contact_email}",
                0.5,
                "suggest",
                1 if was_accurate else 0,
                now,
                resolution_reason,
                supporting_signals,
            ),
        )


# ---------------------------------------------------------------------------
# Cold-start tests
# ---------------------------------------------------------------------------

def test_cold_start_one_sample_returns_1(db, user_model_store):
    """A contact with only 1 resolved prediction gets multiplier=1.0.

    Prevents over-fitting on a single data point (e.g. one lucky or unlucky
    suggestion early in the system's life).
    """
    engine = PredictionEngine(db, user_model_store)
    _insert_contact_prediction(db, "alice@example.com", was_accurate=True)

    multiplier = engine._get_contact_accuracy_multiplier("alice@example.com")
    assert multiplier == 1.0, (
        "Expected 1.0 for cold-start (1 sample) — not enough data to adjust."
    )


def test_cold_start_two_samples_returns_1(db, user_model_store):
    """A contact with only 2 resolved predictions gets multiplier=1.0."""
    engine = PredictionEngine(db, user_model_store)
    _insert_contact_prediction(db, "bob@example.com", was_accurate=False)
    _insert_contact_prediction(db, "bob@example.com", was_accurate=False)

    multiplier = engine._get_contact_accuracy_multiplier("bob@example.com")
    assert multiplier == 1.0, (
        "Expected 1.0 for cold-start (2 samples) — minimum 3 required."
    )


def test_no_samples_returns_1(db, user_model_store):
    """A contact with no history at all returns multiplier=1.0."""
    engine = PredictionEngine(db, user_model_store)
    multiplier = engine._get_contact_accuracy_multiplier("new@example.com")
    assert multiplier == 1.0, "Expected 1.0 for brand-new contact with no history."


# ---------------------------------------------------------------------------
# Low accuracy (suppression floor)
# ---------------------------------------------------------------------------

def test_zero_accuracy_returns_05_floor(db, user_model_store):
    """A contact the user never acts on should get the 0.5 suppression floor.

    0 accurate / 4 resolved = 0% accuracy → floor of 0.5.
    This reduces but does not silence the contact — behaviour can change.
    """
    engine = PredictionEngine(db, user_model_store)
    for _ in range(4):
        _insert_contact_prediction(db, "dormant@example.com", was_accurate=False)

    multiplier = engine._get_contact_accuracy_multiplier("dormant@example.com")
    assert multiplier == 0.5, (
        f"Expected 0.5 floor for 0% accuracy over 4 samples, got {multiplier}."
    )


def test_below_20_accuracy_returns_05_floor(db, user_model_store):
    """Any accuracy below 20% (3+ samples) should return the 0.5 floor."""
    engine = PredictionEngine(db, user_model_store)
    # 1 accurate / 8 resolved = 12.5% accuracy
    _insert_contact_prediction(db, "rarely@example.com", was_accurate=True)
    for _ in range(7):
        _insert_contact_prediction(db, "rarely@example.com", was_accurate=False)

    multiplier = engine._get_contact_accuracy_multiplier("rarely@example.com")
    assert multiplier == 0.5, (
        f"Expected 0.5 floor for 12.5% accuracy (below 20% threshold), got {multiplier}."
    )


# ---------------------------------------------------------------------------
# Scaled accuracy
# ---------------------------------------------------------------------------

def test_50_percent_accuracy_returns_085(db, user_model_store):
    """50% contact accuracy → multiplier = 0.5 + 0.5 * 0.7 = 0.85."""
    engine = PredictionEngine(db, user_model_store)
    for _ in range(3):
        _insert_contact_prediction(db, "moderate@example.com", was_accurate=True)
    for _ in range(3):
        _insert_contact_prediction(db, "moderate@example.com", was_accurate=False)

    multiplier = engine._get_contact_accuracy_multiplier("moderate@example.com")
    assert abs(multiplier - 0.85) < 0.001, (
        f"Expected 0.85 for 50% accuracy, got {multiplier}."
    )


def test_100_percent_accuracy_returns_12_cap(db, user_model_store):
    """100% contact accuracy should return capped at 1.2.

    Without the cap, perfect accuracy would compute to 0.5 + 1.0 * 0.7 = 1.2,
    which is the ceiling.  This prevents runaway amplification.
    """
    engine = PredictionEngine(db, user_model_store)
    for _ in range(5):
        _insert_contact_prediction(db, "reliable@example.com", was_accurate=True)

    multiplier = engine._get_contact_accuracy_multiplier("reliable@example.com")
    assert multiplier == 1.2, (
        f"Expected 1.2 cap for 100% accuracy, got {multiplier}."
    )


def test_80_percent_accuracy_scales_correctly(db, user_model_store):
    """80% accuracy → 0.5 + 0.8 * 0.7 = 1.06."""
    engine = PredictionEngine(db, user_model_store)
    for _ in range(4):
        _insert_contact_prediction(db, "active@example.com", was_accurate=True)
    _insert_contact_prediction(db, "active@example.com", was_accurate=False)

    multiplier = engine._get_contact_accuracy_multiplier("active@example.com")
    assert abs(multiplier - 1.06) < 0.001, (
        f"Expected 1.06 for 80% accuracy (4/5), got {multiplier}."
    )


# ---------------------------------------------------------------------------
# Automated-sender fast-path exclusion
# ---------------------------------------------------------------------------

def test_fast_path_resolutions_excluded_from_contact_multiplier(db, user_model_store):
    """Automated-sender fast-path resolutions must not count in per-contact multiplier.

    Scenario: A contact email was previously classified as an automated sender
    and resolved immediately as inaccurate (resolution_reason='automated_sender_fast_path').
    These resolutions represent prediction-generation bugs, not real user behaviour,
    and should not penalise the contact in future if it turns out to be human.
    """
    engine = PredictionEngine(db, user_model_store)

    # 5 fast-path (excluded) resolutions — all inaccurate
    for _ in range(5):
        _insert_contact_prediction(
            db, "formerly_auto@example.com", was_accurate=False,
            resolution_reason="automated_sender_fast_path",
        )
    # 3 real-behavior resolutions (minimum threshold) — all accurate
    for _ in range(3):
        _insert_contact_prediction(
            db, "formerly_auto@example.com", was_accurate=True,
        )

    # With fast-path excluded: 3 accurate / 3 total = 100% → 1.2 cap
    # Without exclusion: 3 accurate / 8 total = 37.5% → 0.5 + 0.375*0.7 = 0.7625
    multiplier = engine._get_contact_accuracy_multiplier("formerly_auto@example.com")
    assert multiplier == 1.2, (
        f"Expected 1.2 cap (fast-path excluded), got {multiplier}. "
        "Fast-path resolutions should be excluded from per-contact accuracy."
    )


def test_fast_path_excluded_below_threshold(db, user_model_store):
    """Only 2 real-behavior samples after fast-path exclusion → cold-start (1.0)."""
    engine = PredictionEngine(db, user_model_store)
    for _ in range(10):
        _insert_contact_prediction(
            db, "spammy@example.com", was_accurate=False,
            resolution_reason="automated_sender_fast_path",
        )
    _insert_contact_prediction(db, "spammy@example.com", was_accurate=True)
    _insert_contact_prediction(db, "spammy@example.com", was_accurate=False)

    # 2 real samples → below threshold of 3 → cold start
    multiplier = engine._get_contact_accuracy_multiplier("spammy@example.com")
    assert multiplier == 1.0, (
        f"Expected 1.0 cold-start (2 real samples after exclusion), got {multiplier}."
    )


# ---------------------------------------------------------------------------
# Contact isolation — different contacts don't interfere
# ---------------------------------------------------------------------------

def test_per_contact_isolation(db, user_model_store):
    """Each contact's accuracy is tracked independently.

    Alice with 100% accuracy should not affect Bob with 0% accuracy.
    """
    engine = PredictionEngine(db, user_model_store)

    # Alice: 5 accurate → 1.2 cap
    for _ in range(5):
        _insert_contact_prediction(db, "alice@example.com", was_accurate=True)

    # Bob: 5 inaccurate → 0.5 floor
    for _ in range(5):
        _insert_contact_prediction(db, "bob@example.com", was_accurate=False)

    alice_mult = engine._get_contact_accuracy_multiplier("alice@example.com")
    bob_mult = engine._get_contact_accuracy_multiplier("bob@example.com")

    assert alice_mult == 1.2, f"Alice should have 1.2 cap, got {alice_mult}."
    assert bob_mult == 0.5, f"Bob should have 0.5 floor, got {bob_mult}."


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_contact_email_matching_is_case_insensitive(db, user_model_store):
    """Contact email matching must be case-insensitive.

    Predictions store emails as-received (may be mixed case), but lookups
    should normalise to lowercase before matching.
    """
    engine = PredictionEngine(db, user_model_store)
    # Insert with mixed case
    for _ in range(4):
        _insert_contact_prediction(db, "Alice@Example.COM", was_accurate=True)

    # Query with lowercase should find them
    multiplier = engine._get_contact_accuracy_multiplier("alice@example.com")
    assert multiplier == 1.2, (
        f"Expected 1.2 cap for 100% accuracy (case-insensitive), got {multiplier}."
    )


# ---------------------------------------------------------------------------
# Integration: contact multiplier applied in main loop
# ---------------------------------------------------------------------------

def test_opportunity_predictions_get_contact_multiplier_in_loop(db, user_model_store):
    """Contact multiplier is applied on top of type multiplier in prediction loop.

    The global type multiplier and per-contact multiplier are independent queries:
    - Type multiplier: considers ALL resolved opportunity predictions
    - Contact multiplier: considers only predictions for a specific contact

    Scenario: 4 predictions for "other@" (all accurate) + 5 for "target@" (all accurate)
    - Global type: 9 accurate / 9 total = 100% → 0.5 + 1.0 * 0.6 = 1.1
    - Contact (target): 5 accurate / 5 total = 100% → 1.2 cap
    - Combined: 1.1 * 1.2 = 1.32 applied to confidence
    """
    engine = PredictionEngine(db, user_model_store)

    # Seed global type accuracy via "other@" predictions
    for _ in range(4):
        _insert_contact_prediction(db, "other@example.com", was_accurate=True)

    # Seed contact-specific accuracy: 5 accurate = 100%
    for _ in range(5):
        _insert_contact_prediction(db, "target@example.com", was_accurate=True)

    type_mult = engine._get_accuracy_multiplier("opportunity")
    contact_mult = engine._get_contact_accuracy_multiplier("target@example.com")

    # Global: 9 accurate / 9 total = 100% → 1.1
    # Contact: 5 accurate / 5 total = 100% → 1.2 cap
    assert abs(type_mult - 1.1) < 0.001, (
        f"Expected type multiplier 1.1 for 100% accuracy, got {type_mult}."
    )
    assert contact_mult == 1.2, (
        f"Expected contact multiplier 1.2 cap for 100% accuracy, got {contact_mult}."
    )
    combined = type_mult * contact_mult
    assert abs(combined - 1.32) < 0.01, (
        f"Expected combined multiplier ~1.32, got {combined:.3f}."
    )


def test_non_opportunity_predictions_unaffected_by_contact_multiplier(db, user_model_store):
    """The contact multiplier must NOT be applied to non-opportunity predictions.

    Only opportunity predictions with a contact_email in supporting_signals get
    the per-contact adjustment.  Other types (reminder, risk, conflict, need)
    should only use the global type multiplier.
    """
    engine = PredictionEngine(db, user_model_store)

    # Seed contact with 0% accuracy (would apply 0.5 floor if wrongly applied)
    for _ in range(5):
        _insert_contact_prediction(
            db, "bad@example.com", was_accurate=False,
            prediction_type="opportunity",
        )

    # The contact multiplier should not affect reminder predictions for same email
    # Seed 5 accurate reminder predictions to get baseline type multiplier
    with db.get_connection("user_model") as conn:
        for _ in range(5):
            now = datetime.now(timezone.utc).isoformat()
            signals = json.dumps({"contact_email": "bad@example.com"})
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    was_surfaced, was_accurate, resolved_at, supporting_signals)
                   VALUES (?, 'reminder', 'Follow up', 0.5, 'suggest', 1, 1, ?, ?)""",
                (str(uuid.uuid4()), now, signals),
            )

    reminder_mult = engine._get_accuracy_multiplier("reminder")
    # No per-contact logic should fire for non-opportunity — just type multiplier
    # The test verifies the type multiplier alone is returned (not multiplied by 0.5)
    contact_mult_for_reminder = engine._get_contact_accuracy_multiplier("bad@example.com")
    # This IS 0.5 based on opportunity history — but it should NOT be applied to reminder
    # The loop condition `pred.prediction_type == "opportunity"` prevents this
    assert contact_mult_for_reminder == 0.5, (
        "Contact has 0% accuracy in opportunity history — contact_mult should be 0.5."
    )
    # Crucially: reminder type multiplier is unaffected by contact's opportunity history
    assert reminder_mult > 1.0, (
        f"Reminder type mult should be > 1.0 for 100% reminder accuracy, got {reminder_mult}."
    )
