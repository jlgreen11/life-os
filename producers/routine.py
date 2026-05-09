"""Routine producer — surfaces a Moment at detected-routine times.

The sixth and final Phase 1 producer. Where :mod:`producers.cadence` and
:mod:`producers.relationship` watch people-shaped signals,
:mod:`producers.temporal` watches clock-shaped signals, and
:mod:`producers.spatial` / :mod:`producers.comm_template` watch
place- and conversation-shaped signals, the routine producer watches
*habit-shaped* signals: "every Sunday at 5pm you plan the week" —
and nudges at that time if the user has not yet begun.

Trigger
-------
Only ``time.tick`` events wake the producer. Every other event
short-circuits to ``[]``. The scheduler (Week 3) pulses ``time.tick``
at least once per hour, so an hourly granularity on routine firing is
the contract the producer can rely on.

Profile shape
-------------
Profiles live in ``signal_profiles`` under ``producer='routine'`` with
one row per detected routine keyed by a stable routine id (e.g.
``"plan_week"``). The routine detector (future work) owns writing
these; the producer only reads. A reasonable profile:

.. code-block:: json

    {
        "description": "plan your week",
        "weekday": 6,
        "hour": 17,
        "tz_offset_hours": -7,
        "last_occurrences": ["evt-1", "evt-2", "evt-3"],
        "consistency": 0.85
    }

- ``description`` is the human-readable routine name used in microcopy
  (``"You usually {description}. Want to start now?"``). A missing or
  empty description silently skips the row — with no verb there is
  nothing to propose.
- ``weekday`` is an integer ``0..6`` in Python's
  :meth:`datetime.datetime.weekday` convention (0 = Monday, 6 = Sunday).
  ``None`` or missing means "daily" (match any weekday).
- ``hour`` is a local-clock hour ``0..23``. Required.
- ``tz_offset_hours`` shifts UTC into the user's local clock. Defaults
  to ``0``.
- ``last_occurrences`` is a list of event ids for recent occurrences of
  the routine (newest first). The producer uses the first three as the
  Moment's evidence list. Fewer than :data:`MIN_OCCURRENCES` (3) means
  "not yet a routine" and the row is skipped — this is the history gate
  that makes routine detections statistically distinguishable from
  noise.
- ``consistency`` is an optional float in ``[0.0, 1.0]`` the detector
  may write to quantify how reliably the routine happens at that time.
  When present, it drives confidence directly; when absent, confidence
  falls back to a count-based curve over ``last_occurrences``.

A missing or malformed profile silently skips rather than raising —
producers are fail-open per the v1 convention preserved in v2.

Emission rule
-------------
A Moment is emitted iff:

- ``len(last_occurrences) >= MIN_OCCURRENCES`` (history gate), AND
- The local-now hour equals ``profile.hour``, AND
- ``profile.weekday`` is ``None`` **or** equals the local-now weekday.

Confidence is either ``min(0.9, 0.4 + consistency * 0.5)`` when the
profile supplies ``consistency``, or
``min(0.9, 0.4 + min(len(last_occurrences), 20) / 40.0)`` otherwise —
both saturate at 0.9 to leave headroom for the Week 6 feedback EWMA to
demote a routine the user keeps dismissing.

Idempotency per (routine, day, hour)
------------------------------------
The ``evidence_hash`` basis is ``(profile_key, local-date, hour)`` so
re-firing within the same local hour collapses via the
``UNIQUE (source_insight_type, evidence_hash)`` constraint on
``moments``. The next hour's pulse generates a new bucket, so a daily
routine at 17:00 can fire at most once per day; a weekly Sunday
routine can fire at most once per week.

The ``evidence`` list stored on the Moment is ``last_occurrences[:3]``;
when absent or empty, it falls back to a synthetic id of the form
``routine:{key}:{YYYY-MM-DD}`` so the UI always has a non-empty
evidence list to render.

References
----------
- CEO plan § "The Moment Primitive (producers)":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 5: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- Sibling producer :mod:`producers.temporal` for the same
  ``time.tick`` trigger, tz-offset handling, and stdlib-clock
  convention.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from core.moment.producer import Event, Producer, register
from core.moment.types import Action, ActionKind, InsightType, Moment

logger = logging.getLogger(__name__)


ROUTINE_PRODUCER_KEY = "routine"
"""``signal_profiles.producer`` value owned by the routine producer."""

MIN_OCCURRENCES = 3
"""A routine needs at least three observed occurrences before firing."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

SECONDS_PER_HOUR = 3600

TRIGGER_EVENT_TYPES: frozenset[str] = frozenset({"time.tick"})
"""Event types this producer cares about. All others short-circuit to ``[]``."""


@register
class RoutineProducer(Producer):
    """Emits a routine-reminder Moment when the clock hits a detected habit.

    Constructor takes a ``sqlite3.Connection`` that points at the v2
    ``lifeos.db``. The producer never writes — only the (future)
    routine detector owns profile maintenance — so the connection's
    transaction settings are left alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.ROUTINE

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so
        column access in :meth:`_read_profiles` is by name. This
        matches the sibling producers' convention; sharing a
        connection across producers is safe because none of them
        write.
        """
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._id_fn: Callable[[], str] = id_fn or (lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Producer contract
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> list[Moment]:
        """Return Moments for any routine profiles matching ``now``.

        Filters non-trigger events first so the common case never
        touches the DB. On a real ``time.tick`` the producer scans
        every routine profile and lets :meth:`_maybe_emit` decide
        per profile.
        """
        if event.get("type") not in TRIGGER_EVENT_TYPES:
            return []
        now = int(self._now_fn())
        moments: list[Moment] = []
        for profile_key, profile in self._read_profiles():
            moment = self._maybe_emit(profile_key, profile, now)
            if moment is not None:
                moments.append(moment)
        return moments

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _read_profiles(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(key, profile_dict)`` pairs, malformed rows skipped."""
        rows = self._conn.execute(
            "SELECT key, profile FROM signal_profiles WHERE producer=?",
            (ROUTINE_PRODUCER_KEY,),
        ).fetchall()
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            raw = row["profile"]
            try:
                profile = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "routine profile for key=%r is not valid JSON; skipping",
                    row["key"],
                )
                continue
            if not isinstance(profile, dict):
                logger.warning(
                    "routine profile for key=%r is not a JSON object; skipping",
                    row["key"],
                )
                continue
            out.append((row["key"], profile))
        return out

    def _maybe_emit(
        self,
        profile_key: str,
        profile: dict[str, Any],
        now: int,
    ) -> Moment | None:
        """Return a Moment if ``now`` matches the routine's (weekday, hour).

        Any missing or wrong-type required field collapses to "skip".
        This matches the v1 fail-open posture: a malformed profile
        must never crash event processing.
        """
        description = str(profile.get("description") or "").strip()
        if not description:
            return None

        try:
            hour = int(profile["hour"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= hour <= 23:
            return None

        raw_weekday = profile.get("weekday")
        if raw_weekday is None:
            weekday: int | None = None
        else:
            try:
                weekday = int(raw_weekday)
            except (TypeError, ValueError):
                return None
            if not 0 <= weekday <= 6:
                return None

        try:
            tz_offset_hours = int(profile.get("tz_offset_hours", 0))
        except (TypeError, ValueError):
            tz_offset_hours = 0

        raw_occ = profile.get("last_occurrences")
        if not isinstance(raw_occ, list):
            return None
        occurrences = [str(e) for e in raw_occ if e]
        if len(occurrences) < MIN_OCCURRENCES:
            return None

        local_now = now + tz_offset_hours * SECONDS_PER_HOUR
        local_dt = datetime.fromtimestamp(local_now, tz=UTC)
        local_hour = local_dt.hour
        local_weekday = local_dt.weekday()
        local_date_str = local_dt.strftime("%Y-%m-%d")

        if local_hour != hour:
            return None
        if weekday is not None and local_weekday != weekday:
            return None

        evidence = occurrences[:3] or [f"routine:{profile_key}:{local_date_str}"]
        confidence = _compute_confidence(profile, occurrences)
        insight = f"You usually {description}. Want to start now?"

        # Hash basis is (key, local-date, local-hour) so re-firing
        # within the same hour collapses to one row. See module
        # docstring "Idempotency per (routine, day, hour)".
        hour_bucket = f"{local_date_str}:{local_hour:02d}"
        hash_basis = [profile_key, hour_bucket]

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            evidence_hash=self.evidence_hash(hash_basis),
            proposed_action=Action(
                kind=ActionKind.SET_REMINDER,
                params={
                    "routine_key": profile_key,
                    "description": description,
                },
            ),
            source_insight_type=InsightType.ROUTINE,
            evidence=evidence,
            confidence=confidence,
        )


def _compute_confidence(profile: dict[str, Any], occurrences: list[str]) -> float:
    """Return the confidence score for a routine firing.

    Prefers ``profile.consistency`` when present and numeric; falls
    back to a count curve over ``occurrences``. Both formulas
    saturate at 0.9, leaving the Week 6 feedback EWMA room to demote.
    """
    raw = profile.get("consistency")
    if raw is not None:
        try:
            consistency = float(raw)
        except (TypeError, ValueError):
            pass
        else:
            clamped = max(0.0, min(1.0, consistency))
            return min(0.9, 0.4 + clamped * 0.5)
    count = min(len(occurrences), 20)
    return min(0.9, 0.4 + count / 40.0)


__all__ = [
    "DEFAULT_EXPIRY_SECONDS",
    "MIN_OCCURRENCES",
    "ROUTINE_PRODUCER_KEY",
    "TRIGGER_EVENT_TYPES",
    "RoutineProducer",
]
