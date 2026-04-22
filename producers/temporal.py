"""Temporal producer — surfaces historical focus windows + open calendar gaps.

The third of the six Phase 1 producers. Where :mod:`producers.cadence` and
:mod:`producers.relationship` watch *people-shaped* signals, the temporal
producer watches *clock-shaped* signals: when in the day the user has
historically done their best work, and when the calendar has just opened
up enough room to use one of those windows.

Trigger
-------
The producer wakes on either a scheduler tick (``time.tick``) or a
calendar mutation (``calendar.event.created`` / ``.updated`` / ``.deleted``).
The first of those is the natural pulse for "we just entered a focus
window"; the second is the natural pulse for "a 60-minute hole just
opened in the calendar". Other event types short-circuit to ``[]``.

Profile shape
-------------
The temporal profile lives in ``signal_profiles`` under
``producer='temporal'``. There is normally exactly one row (``key='self'``)
because chronotype is a property of the user, not of a contact, but the
producer reads *every* row in the namespace so future per-context
chronotypes (e.g. weekend vs weekday) can be added without changing the
producer.

.. code-block:: json

    {
        "data_days": 30,
        "tz_offset_hours": -7,
        "focus_windows": [
            {"start_hour": 9, "end_hour": 11, "description": "deep work block"},
            {"start_hour": 14, "end_hour": 16, "description": "afternoon focus"}
        ],
        "current_calendar_gaps": [
            {"start_ts": 1777238400, "end_ts": 1777245600,
             "evidence_ids": ["evt-cal-1"]}
        ]
    }

- ``data_days`` is the count of days of behavioral history the temporal
  signal extractor (future work) has aggregated. Below
  :data:`MIN_PROFILE_DAYS` the producer refuses to fire — two weeks is
  the smallest window where a chronotype is statistically distinguishable
  from noise.
- ``tz_offset_hours`` shifts UTC into the user's local clock so the
  ``start_hour`` / ``end_hour`` fields read naturally. Defaults to ``0``.
- ``focus_windows`` is a list of ``[start_hour, end_hour)`` half-open
  intervals in **local hours** with a human-readable ``description`` for
  the Moment microcopy.
- ``current_calendar_gaps`` is a list of ``[start_ts, end_ts)`` UTC
  windows of unscheduled calendar time. The calendar signal extractor
  (future work) refreshes these on every calendar mutation.

A missing or malformed profile silently skips rather than raising —
producers are fail-open per the v1 convention preserved in v2.

Emission rule
-------------
A Moment is emitted iff ``data_days >= MIN_PROFILE_DAYS`` and **either**:

- (a) ``now`` falls inside one of the ``focus_windows`` (chronotype
  trigger), **or**
- (b) ``now`` falls inside one of the ``current_calendar_gaps`` whose
  remaining duration is at least :data:`MIN_GAP_SECONDS` (gap trigger).

When both fire for the same hour bucket, the gap evidence wins (it has
real calendar event ids the user can click) but the description is
borrowed from the matching focus window so the microcopy retains its
"why now" framing. When only the focus window fires, ``free_minutes``
is computed from the remaining hours of the window; when only the gap
fires, ``free_minutes`` is the remaining gap duration and the
description falls back to "open calendar block".

Confidence scales with ``data_days`` (more history → louder push):

.. code-block:: text

    confidence = min(0.9, 0.4 + data_days / 60)

At the 14-day boundary that yields ``0.633``; at 30+ days it saturates
at the 0.9 cap left for the Week 6 feedback EWMA to demote.

Idempotency per (day, hour)
---------------------------
The ``evidence_hash`` is computed from the **(date, hour-of-day)** bucket
together with the contact-equivalent profile key, so re-firing within
the same local hour collapses via the
``UNIQUE (source_insight_type, evidence_hash)`` constraint on
``moments``. The next hour's pulse generates a new bucket, so a focus
window that spans 09:00-11:00 can produce two distinct Moments (one
per hour) - but never two for the same hour, regardless of how many
``time.tick`` events the scheduler emits.

The ``evidence`` list stored on the Moment is either the calendar gap's
``evidence_ids`` (when present) or a synthetic id of the form
``temporal:focus:{start:02d}-{end:02d}:{YYYY-MM-DD}`` so the UI always
has a non-empty evidence list to render.

References
----------
- CEO plan § "The Moment Primitive (producers) → Temporal":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 4: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- Sibling producers :mod:`producers.cadence` and :mod:`producers.relationship`
  for the same trigger filter, fail-open, and stdlib-clock conventions.
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


TEMPORAL_PRODUCER_KEY = "temporal"
"""``signal_profiles.producer`` value owned by the temporal producer."""

MIN_PROFILE_DAYS = 14
"""Minimum days of behavioral history required before the producer will fire."""

MIN_GAP_SECONDS = 3600
"""A calendar gap must have at least 60 minutes remaining to surface."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

TRIGGER_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "time.tick",
        "calendar.event.created",
        "calendar.event.updated",
        "calendar.event.deleted",
    }
)
"""Event types this producer cares about. All others short-circuit to ``[]``."""


@register
class TemporalProducer(Producer):
    """Emits chronotype/gap Moments when free time aligns with focus history.

    Constructor takes a ``sqlite3.Connection`` that points at the v2
    ``lifeos.db``. The producer never writes — only the (future)
    temporal signal extractor and calendar signal extractor own profile
    maintenance — so the connection's transaction settings are left
    alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.TEMPORAL

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so column
        access in :meth:`_read_profiles` is by name. This matches the
        cadence and relationship producers' convention; sharing a
        connection across producers is safe because none of them write.
        """
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._id_fn: Callable[[], str] = id_fn or (lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Producer contract
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> list[Moment]:
        """Return Moments for any temporal profiles that match ``now``.

        Filters non-trigger events first so the common case never touches
        the DB. On a real trigger event, scans every temporal profile
        and lets :meth:`_maybe_emit` decide per profile.
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
        """Return ``(key, profile_dict)`` pairs, malformed rows skipped.

        Returning a fully-materialized list (not a generator) keeps the
        SQLite cursor closed before any per-profile work runs — matches
        the sibling producers' shape.
        """
        rows = self._conn.execute(
            "SELECT key, profile FROM signal_profiles WHERE producer=?",
            (TEMPORAL_PRODUCER_KEY,),
        ).fetchall()
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            raw = row["profile"]
            try:
                profile = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "temporal profile for key=%r is not valid JSON; skipping",
                    row["key"],
                )
                continue
            if not isinstance(profile, dict):
                logger.warning(
                    "temporal profile for key=%r is not a JSON object; skipping",
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
        """Return a Moment if ``now`` fits a focus window or open gap.

        The profile is allowed to be partially populated — any missing or
        wrong-type required field collapses to "skip". This matches the
        v1 fail-open posture: a malformed profile must never crash event
        processing.
        """
        try:
            data_days = int(profile["data_days"])
        except (KeyError, TypeError, ValueError):
            return None
        if data_days < MIN_PROFILE_DAYS:
            return None

        try:
            tz_offset_hours = int(profile.get("tz_offset_hours", 0))
        except (TypeError, ValueError):
            tz_offset_hours = 0

        local_now = now + tz_offset_hours * SECONDS_PER_HOUR
        local_dt = datetime.fromtimestamp(local_now, tz=UTC)
        local_hour = local_dt.hour
        local_date_str = local_dt.strftime("%Y-%m-%d")

        matching_window = self._find_matching_focus_window(profile, local_hour)
        matching_gap = self._find_matching_calendar_gap(profile, now)

        if matching_window is None and matching_gap is None:
            return None

        if matching_gap is not None:
            free_minutes = (matching_gap[1] - now) // 60
            gap_ev = matching_gap[2][:3]
            if gap_ev:
                evidence = gap_ev
            elif matching_window is not None:
                start_h, end_h, _ = matching_window
                evidence = [f"temporal:focus:{start_h:02d}-{end_h:02d}:{local_date_str}"]
            else:
                evidence = [f"temporal:gap:{matching_gap[0]}-{matching_gap[1]}"]
        else:
            assert matching_window is not None
            end_h = matching_window[1]
            local_window_end = (local_now // SECONDS_PER_DAY) * SECONDS_PER_DAY + end_h * SECONDS_PER_HOUR
            free_minutes = max(0, (local_window_end - local_now) // 60)
            start_h = matching_window[0]
            evidence = [f"temporal:focus:{start_h:02d}-{end_h:02d}:{local_date_str}"]

        if matching_window is not None:
            description = matching_window[2] or "open calendar block"
        else:
            description = "open calendar block"

        insight = f"You have {free_minutes} min free. Historical focus pattern at this hour: {description}."
        confidence = min(0.9, 0.4 + data_days / 60.0)

        # Hash basis is (key, local-date, local-hour) so re-firing within
        # the same hour collapses to one row. See module docstring
        # "Idempotency per (day, hour)".
        hour_bucket = f"{local_date_str}:{local_hour:02d}"
        hash_basis = [profile_key, hour_bucket]

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            evidence_hash=self.evidence_hash(hash_basis),
            proposed_action=Action(
                kind=ActionKind.SCHEDULE_BLOCK,
                params={
                    "duration_minutes": int(free_minutes),
                    "label": description,
                },
            ),
            source_insight_type=InsightType.TEMPORAL,
            evidence=evidence,
            confidence=confidence,
        )

    @staticmethod
    def _find_matching_focus_window(
        profile: dict[str, Any],
        local_hour: int,
    ) -> tuple[int, int, str] | None:
        """Return ``(start_hour, end_hour, description)`` covering ``local_hour``.

        Returns the first window in profile order; the signal extractor
        is expected to write non-overlapping windows but the producer
        does not enforce that.
        """
        windows = profile.get("focus_windows")
        if not isinstance(windows, list):
            return None
        for w in windows:
            if not isinstance(w, dict):
                continue
            try:
                start_h = int(w["start_hour"])
                end_h = int(w["end_hour"])
            except (KeyError, TypeError, ValueError):
                continue
            description = str(w.get("description") or "")
            if start_h <= local_hour < end_h:
                return (start_h, end_h, description)
        return None

    @staticmethod
    def _find_matching_calendar_gap(
        profile: dict[str, Any],
        now: int,
    ) -> tuple[int, int, list[str]] | None:
        """Return ``(start_ts, end_ts, evidence_ids)`` covering ``now`` with ≥60 min left."""
        gaps = profile.get("current_calendar_gaps")
        if not isinstance(gaps, list):
            return None
        for g in gaps:
            if not isinstance(g, dict):
                continue
            try:
                start_ts = int(g["start_ts"])
                end_ts = int(g["end_ts"])
            except (KeyError, TypeError, ValueError):
                continue
            if start_ts > now or end_ts <= now:
                continue
            if (end_ts - now) < MIN_GAP_SECONDS:
                continue
            raw_ev = g.get("evidence_ids") or []
            if not isinstance(raw_ev, list):
                raw_ev = []
            evidence_ids = [str(e) for e in raw_ev if e]
            return (start_ts, end_ts, evidence_ids)
        return None


__all__ = [
    "DEFAULT_EXPIRY_SECONDS",
    "MIN_GAP_SECONDS",
    "MIN_PROFILE_DAYS",
    "TEMPORAL_PRODUCER_KEY",
    "TRIGGER_EVENT_TYPES",
    "TemporalProducer",
]
