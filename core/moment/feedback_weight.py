"""Per-insight-type feedback weight — EWMA of user acceptance.

The :class:`FeedbackWeightStore` owns the ``feedback_weights`` SQLite
table and implements the confidence-threshold feedback loop defined in
the CEO plan (§ "Feedback weights — EWMA") and engineering review §1d.

Contract
--------
For each ``InsightType``, we maintain a weight ``w ∈ [0.0, 1.0]`` that
tracks how often the user *accepts* Moments of that type. The weight
moves only on a terminal user decision (ACCEPTED / DISMISSED / SNOOZED);
lifecycle-only transitions (EXPIRED, DONE, and non-terminal states)
leave both ``weight`` and ``decision_count`` untouched.

Signals (mapped from :class:`MomentState`):

- ``ACCEPTED``  → signal = 1.0
- ``DISMISSED`` → signal = 0.0
- ``SNOOZED``   → signal = 0.5  (tentative — user deferred, not rejected)
- everything else → **no update**

Update rule (standard EWMA, ``alpha = 0.1``, half-life ≈ 7 decisions):

    w_new = alpha * signal + (1 - alpha) * w_old

Cold start: a never-seen ``insight_type`` behaves as ``(weight=0.5,
decision_count=0)``. The first ``update`` starts from that prior and
writes a fresh row.

Threshold rule (higher bar when the user rejects more):

    threshold(t) = BASE_THRESHOLD + (1.0 - weight(t))

Where ``BASE_THRESHOLD = 0.6``. At ``weight=1.0`` (always accepted) the
threshold drops to ``0.6`` — producers only need strong signal to fire.
At ``weight=0.0`` (always rejected) the threshold climbs to ``1.6``,
which no bounded confidence can clear, silencing that insight type
until the user flips a dismissal into an accept.

The ``MomentEngine`` (next task) consults :meth:`get_threshold_for`
before creating a Moment; anything below it is filtered out.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

from core.moment.types import MomentState

ALPHA: float = 0.1
DEFAULT_WEIGHT: float = 0.5
DEFAULT_DECISION_COUNT: int = 0
BASE_THRESHOLD: float = 0.6

_SIGNAL_BY_STATE: dict[MomentState, float] = {
    MomentState.ACCEPTED: 1.0,
    MomentState.DISMISSED: 0.0,
    MomentState.SNOOZED: 0.5,
}


class FeedbackWeightStore:
    """SQLite-backed EWMA of user acceptance per insight type.

    Constructor injection per eng review §1a — the caller owns the
    ``sqlite3.Connection`` and its lifecycle. Writes use
    ``BEGIN IMMEDIATE`` for the same concurrency reason as the outbox
    repo: SQLite serialises on the writer lock, so two concurrent
    :meth:`update` calls on the same row cannot interleave and lose an
    increment.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._conn = conn
        self._conn.isolation_level = None
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time

    def _now(self) -> int:
        return int(self._now_fn())

    def get(self, insight_type: str) -> tuple[float, int]:
        """Return ``(weight, decision_count)`` for ``insight_type``.

        Returns the cold-start default ``(0.5, 0)`` when the insight
        type has no persisted row yet. Callers should treat both values
        as read-only — use :meth:`update` to mutate.
        """
        row = self._conn.execute(
            "SELECT weight, decision_count FROM feedback_weights WHERE insight_type=?",
            (insight_type,),
        ).fetchone()
        if row is None:
            return (DEFAULT_WEIGHT, DEFAULT_DECISION_COUNT)
        return (float(row["weight"]), int(row["decision_count"]))

    def update(self, insight_type: str, moment_state: MomentState) -> None:
        """Apply one EWMA step for ``insight_type`` given a terminal state.

        No-op when ``moment_state`` is not in the signal table (i.e.
        ``EXPIRED``, ``DONE``, ``SUGGESTED``): lifecycle transitions that
        the user did not drive do not count as feedback.

        On an updating state, the row is upserted: cold-start priors
        ``(0.5, 0)`` seed the first write; subsequent writes read
        back the current row inside the same ``BEGIN IMMEDIATE`` txn so
        the read-modify-write is atomic.
        """
        signal = _SIGNAL_BY_STATE.get(moment_state)
        if signal is None:
            return

        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT weight, decision_count FROM feedback_weights WHERE insight_type=?",
                (insight_type,),
            ).fetchone()
            if row is None:
                w_old = DEFAULT_WEIGHT
                count_old = DEFAULT_DECISION_COUNT
            else:
                w_old = float(row["weight"])
                count_old = int(row["decision_count"])

            w_new = ALPHA * signal + (1.0 - ALPHA) * w_old
            count_new = count_old + 1

            self._conn.execute(
                """
                INSERT INTO feedback_weights (insight_type, weight, decision_count, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(insight_type) DO UPDATE SET
                    weight = excluded.weight,
                    decision_count = excluded.decision_count,
                    last_updated = excluded.last_updated
                """,
                (insight_type, w_new, count_new, now),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_threshold_for(self, insight_type: str) -> float:
        """Return the confidence threshold a Moment must meet to fire.

        ``BASE_THRESHOLD + (1.0 - weight)``. Low-accept insight types get
        a high bar (up to 1.6 at weight=0.0, effectively silencing
        them); high-accept types clamp toward the base (0.6 at
        weight=1.0). Cold-start types sit at ``0.6 + 0.5 = 1.1``, so
        producers must emit above-cap confidence to break through until
        the user accepts a few Moments.
        """
        weight, _ = self.get(insight_type)
        return BASE_THRESHOLD + (1.0 - weight)


__all__ = [
    "ALPHA",
    "BASE_THRESHOLD",
    "DEFAULT_DECISION_COUNT",
    "DEFAULT_WEIGHT",
    "FeedbackWeightStore",
]
