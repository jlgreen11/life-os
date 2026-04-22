"""Tests for :class:`core.moment.feedback_weight.FeedbackWeightStore`.

Covers the four public properties of the EWMA feedback store:

1. unknown insight types return the cold-start default ``(0.5, 0)``,
   and their threshold is the cold-start threshold ``1.1``;
2. a single :meth:`update` step applies the documented EWMA formula
   (``w_new = alpha*signal + (1-alpha)*w_old``) and bumps
   ``decision_count`` by exactly one;
3. a long run of same-signal updates converges monotonically toward
   the signal value (1.0 for ACCEPTED, 0.0 for DISMISSED);
4. ``EXPIRED`` / ``DONE`` / other non-terminal states are no-ops — they
   must not touch either ``weight`` or ``decision_count`` even when a
   row already exists.

Tests run against the real consolidated v2 schema on an in-memory
SQLite, so schema drift (e.g. column rename from ``decision_count`` to
something else) would trip the test before it hit production.
"""

from __future__ import annotations

import math
import sqlite3

import pytest

from core.moment.feedback_weight import (
    ALPHA,
    BASE_THRESHOLD,
    DEFAULT_DECISION_COUNT,
    DEFAULT_WEIGHT,
    FeedbackWeightStore,
)
from core.moment.types import InsightType, MomentState
from storage import schema

REF_NOW = 1_777_204_800


class Clock:
    """Mutable stand-in for :func:`time.time` so tests can advance time."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    _apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(conn, clock):
    return FeedbackWeightStore(conn, now_fn=clock)


# ---------------------------------------------------------------------------
# get() / cold start
# ---------------------------------------------------------------------------


def test_get_unknown_type_returns_defaults(store):
    """Never-seen insight types read as (0.5, 0) without writing a row."""
    assert store.get(InsightType.CADENCE) == (DEFAULT_WEIGHT, DEFAULT_DECISION_COUNT)


def test_get_unknown_type_does_not_create_row(store, conn):
    """``get`` is read-only; must not implicitly materialize a row."""
    store.get(InsightType.RELATIONSHIP)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM feedback_weights WHERE insight_type=?",
        (InsightType.RELATIONSHIP,),
    ).fetchone()
    assert row["n"] == 0


def test_threshold_cold_start_is_base_plus_half(store):
    """Cold-start threshold = BASE + (1 - 0.5) = 1.1."""
    assert math.isclose(
        store.get_threshold_for(InsightType.TEMPORAL),
        BASE_THRESHOLD + (1.0 - DEFAULT_WEIGHT),
    )


# ---------------------------------------------------------------------------
# update() — single step
# ---------------------------------------------------------------------------


def test_update_accepted_applies_ewma_from_default(store):
    """First ACCEPTED update: w = 0.1*1.0 + 0.9*0.5 = 0.55."""
    store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    weight, count = store.get(InsightType.CADENCE)
    assert math.isclose(weight, ALPHA * 1.0 + (1 - ALPHA) * DEFAULT_WEIGHT)
    assert count == 1


def test_update_dismissed_applies_ewma_from_default(store):
    """First DISMISSED update: w = 0.1*0.0 + 0.9*0.5 = 0.45."""
    store.update(InsightType.CADENCE, MomentState.DISMISSED)
    weight, count = store.get(InsightType.CADENCE)
    assert math.isclose(weight, ALPHA * 0.0 + (1 - ALPHA) * DEFAULT_WEIGHT)
    assert count == 1


def test_update_snoozed_applies_half_signal(store):
    """First SNOOZED update: w = 0.1*0.5 + 0.9*0.5 = 0.5 (unchanged)."""
    store.update(InsightType.CADENCE, MomentState.SNOOZED)
    weight, count = store.get(InsightType.CADENCE)
    assert math.isclose(weight, ALPHA * 0.5 + (1 - ALPHA) * DEFAULT_WEIGHT)
    assert count == 1


def test_update_stamps_last_updated(store, conn, clock):
    """``last_updated`` reflects the clock at the time of the write."""
    clock.t = REF_NOW + 100
    store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    row = conn.execute(
        "SELECT last_updated FROM feedback_weights WHERE insight_type=?",
        (InsightType.CADENCE,),
    ).fetchone()
    assert row["last_updated"] == REF_NOW + 100


# ---------------------------------------------------------------------------
# update() — no-op states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [MomentState.EXPIRED, MomentState.DONE, MomentState.SUGGESTED],
)
def test_update_noop_states_do_not_touch_store(store, conn, state):
    """EXPIRED/DONE/SUGGESTED must not create a row or bump counts."""
    store.update(InsightType.CADENCE, state)
    # No row materialized, so get() still reads defaults.
    assert store.get(InsightType.CADENCE) == (DEFAULT_WEIGHT, DEFAULT_DECISION_COUNT)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM feedback_weights WHERE insight_type=?",
        (InsightType.CADENCE,),
    ).fetchone()
    assert row["n"] == 0


def test_noop_state_after_existing_row_preserves_weight(store):
    """An existing row must not shift on EXPIRED/DONE."""
    store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    w_before, count_before = store.get(InsightType.CADENCE)

    store.update(InsightType.CADENCE, MomentState.EXPIRED)
    store.update(InsightType.CADENCE, MomentState.DONE)

    w_after, count_after = store.get(InsightType.CADENCE)
    assert math.isclose(w_after, w_before)
    assert count_after == count_before


# ---------------------------------------------------------------------------
# update() — convergence
# ---------------------------------------------------------------------------


def test_many_accepts_converge_toward_one(store):
    """A long run of ACCEPTED must drive weight monotonically toward 1.0."""
    previous = DEFAULT_WEIGHT
    for _ in range(50):
        store.update(InsightType.CADENCE, MomentState.ACCEPTED)
        current, _ = store.get(InsightType.CADENCE)
        # Strictly increasing — EWMA with signal > w_old always rises.
        assert current > previous
        previous = current
    # After 50 steps with alpha=0.1, weight is within a few percent of 1.0.
    assert previous > 0.99


def test_many_dismisses_converge_toward_zero(store):
    """A long run of DISMISSED must drive weight monotonically toward 0.0."""
    previous = DEFAULT_WEIGHT
    for _ in range(50):
        store.update(InsightType.CADENCE, MomentState.DISMISSED)
        current, _ = store.get(InsightType.CADENCE)
        assert current < previous
        previous = current
    assert previous < 0.01


def test_decision_count_grows_by_one_per_update(store):
    """``decision_count`` = number of signal-bearing updates applied."""
    for i in range(1, 11):
        store.update(InsightType.CADENCE, MomentState.ACCEPTED)
        _, count = store.get(InsightType.CADENCE)
        assert count == i


# ---------------------------------------------------------------------------
# get_threshold_for()
# ---------------------------------------------------------------------------


def test_threshold_shrinks_as_weight_grows(store):
    """High accept rate → threshold drops toward BASE_THRESHOLD."""
    for _ in range(50):
        store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    t = store.get_threshold_for(InsightType.CADENCE)
    # weight ≈ 1.0, so threshold ≈ 0.6 within a small margin.
    assert t == pytest.approx(BASE_THRESHOLD, abs=0.02)


def test_threshold_grows_as_weight_shrinks(store):
    """High dismiss rate → threshold climbs toward 1.6 (silencing)."""
    for _ in range(50):
        store.update(InsightType.CADENCE, MomentState.DISMISSED)
    t = store.get_threshold_for(InsightType.CADENCE)
    assert t == pytest.approx(BASE_THRESHOLD + 1.0, abs=0.02)


def test_threshold_formula_matches_weight(store):
    """Always: threshold == BASE + (1 - weight), regardless of history."""
    store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    store.update(InsightType.CADENCE, MomentState.DISMISSED)
    store.update(InsightType.CADENCE, MomentState.SNOOZED)
    weight, _ = store.get(InsightType.CADENCE)
    assert math.isclose(
        store.get_threshold_for(InsightType.CADENCE),
        BASE_THRESHOLD + (1.0 - weight),
    )


# ---------------------------------------------------------------------------
# Isolation between insight types
# ---------------------------------------------------------------------------


def test_insight_types_are_independent(store):
    """Updating one insight type must not move another."""
    store.update(InsightType.CADENCE, MomentState.ACCEPTED)
    # RELATIONSHIP still at defaults.
    assert store.get(InsightType.RELATIONSHIP) == (
        DEFAULT_WEIGHT,
        DEFAULT_DECISION_COUNT,
    )
    # And its threshold matches cold-start.
    assert math.isclose(
        store.get_threshold_for(InsightType.RELATIONSHIP),
        BASE_THRESHOLD + (1.0 - DEFAULT_WEIGHT),
    )
