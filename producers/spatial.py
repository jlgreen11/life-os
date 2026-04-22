"""Spatial producer — notes continuity when the user arrives at or leaves a known place.

The fourth of the six Phase 1 producers. Where :mod:`producers.cadence`
and :mod:`producers.relationship` watch people-shaped signals and
:mod:`producers.temporal` watches clock-shaped signals, the spatial
producer watches *place-shaped* signals: the moments the user crosses
the threshold of a place the system has seen before, and what the
history of that place suggests about what is happening now.

Trigger
-------
Only ``context.location.updated`` events from the iOS compat layer wake
the producer. Every other event short-circuits to ``[]``. The payload
shape mirrors the :class:`~core.moment.scheduler.Scheduler`
``arrive:`` / ``depart:`` grammar (``core/moment/scheduler.py``):

- **Arrival**: ``payload.arrival`` is the place name (e.g. ``"Office"``).
- **Departure**: ``payload.departure`` is the place name. For the
  departure microcopy the producer also expects
  ``payload.duration_minutes`` — the iOS side computes visit duration on
  departure, which is the only moment the duration is actually known.

An event that is neither arrival nor departure (some future location
refresh shape) is treated as non-trigger and skipped.

Profile shape
-------------
Profiles live in ``signal_profiles`` under ``producer='spatial'`` with
one row per known place keyed by ``place_name`` (the same token that
appears on ``payload.arrival`` / ``payload.departure``). A reasonable
profile:

.. code-block:: json

    {
        "place_id": "place-office",
        "place_name": "Office",
        "visit_count": 40,
        "avg_duration_minutes": 180,
        "last_topic": "API design review",
        "last_event_ids": ["evt-loc-1", "evt-loc-2", "evt-loc-3"]
    }

A future spatial signal extractor owns *writing* this row; the producer
only reads it. Missing or malformed fields silently skip the profile
rather than raise — producers are fail-open per the v1 convention
preserved in v2.

Emission rules
--------------
A Moment is emitted iff all of:

- ``visit_count >= MIN_VISIT_COUNT`` (3). Too few visits means no
  historical colour to add.
- The event carries either ``arrival`` or ``departure`` for a place
  whose profile key matches.
- On **arrival**, the profile must also carry a non-empty
  ``last_topic`` — otherwise there is nothing interesting to recall.
- On **departure**, both ``payload.duration_minutes`` and
  ``profile.avg_duration_minutes`` must be numeric and ≥ 1.

The microcopy per the CEO plan:

- Arrival: ``"You're at {Place}. Last time here you worked on
  {topic}."``
- Departure: ``"You've been at {Place} {X} min, avg {Y}."``

The proposed action is :class:`ActionKind.NOTE_OBSERVATION` (read-only
in Phase 1, per the Week 5 task body) with ``params`` carrying the
place name and the arrival/departure kind so downstream consumers can
route without reparsing the microcopy.

Confidence scales with visit count:

.. code-block:: text

    confidence = min(0.9, 0.3 + visit_count / 50.0)

Three visits puts us at ``0.36`` (just above the feedback-weight gate),
fifty visits saturates at ``0.9`` and leaves headroom for the Week 6
feedback EWMA to demote a producer the user is dismissing.

Idempotency within a visit
--------------------------
Per the Week 5 task body: *dedup within same location visit*. The
``evidence_hash`` basis is ``(kind, place_name, UTC-date)`` — so two
arrivals to the same place on the same UTC calendar day collapse to one
row via the ``UNIQUE (source_insight_type, evidence_hash)`` constraint.
Two visits to the same place on *different* days yield distinct hashes
and can each surface. The approximation (visit ≈ same UTC day) is
deliberately coarse: the iOS connector may re-emit arrival on GPS
jitter, and we do not want that to spam the NOW tab.

The ``evidence`` list stored on the Moment is either the profile's
``last_event_ids`` (up to 3) or a synthetic id of the form
``spatial:{kind}:{place_name}:{YYYY-MM-DD}`` so the UI always has a
non-empty evidence list to render.

References
----------
- CEO plan § "The Moment Primitive (producers) → Spatial":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 5: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- Scheduler arrival/departure grammar for payload shape:
  ``core/moment/scheduler.py``.
- Sibling producers :mod:`producers.cadence`,
  :mod:`producers.relationship`, and :mod:`producers.temporal` for the
  same trigger-filter, fail-open, and stdlib-clock conventions.
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


SPATIAL_PRODUCER_KEY = "spatial"
"""``signal_profiles.producer`` value owned by the spatial producer."""

MIN_VISIT_COUNT = 3
"""Minimum historical visits required before the producer will fire."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

TRIGGER_EVENT_TYPE = "context.location.updated"
"""The only event type this producer observes."""

ARRIVAL = "arrival"
"""Payload kind token for an arrival event."""

DEPARTURE = "departure"
"""Payload kind token for a departure event."""


@register
class SpatialProducer(Producer):
    """Emits place-continuity Moments on arrival at / departure from known places.

    Constructor takes a ``sqlite3.Connection`` that points at the v2
    ``lifeos.db``. The producer never writes — the (future) spatial
    signal extractor owns profile maintenance — so the connection's
    transaction settings are left alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.SPATIAL

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so column
        access in :meth:`_read_profile` is by name. This matches the
        sibling producers' convention; sharing a connection across
        producers is safe because none of them write.
        """
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._id_fn: Callable[[], str] = id_fn or (lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Producer contract
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> list[Moment]:
        """Return zero or one Moments for an arrival/departure event.

        Non-matching event types short-circuit without touching the DB.
        A matching event with no resolvable ``place_name`` also skips —
        the producer is fail-open and never raises on shape drift.
        """
        if event.get("type") != TRIGGER_EVENT_TYPE:
            return []
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            return []
        kind, place_name = self._extract_place(payload)
        if kind is None or not place_name:
            return []
        profile = self._read_profile(place_name)
        if profile is None:
            return []
        moment = self._maybe_emit(kind, place_name, payload, profile)
        return [moment] if moment is not None else []

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_place(payload: dict[str, Any]) -> tuple[str | None, str | None]:
        """Return ``(kind, place_name)`` or ``(None, None)`` for non-trigger payloads.

        Arrival wins over departure if both are present on the same
        payload — a physically nonsensical combination that should not
        occur in practice, but deterministic resolution is safer than
        raising.
        """
        arrival_raw = payload.get("arrival")
        departure_raw = payload.get("departure")
        if isinstance(arrival_raw, str) and arrival_raw:
            return (ARRIVAL, arrival_raw)
        if isinstance(departure_raw, str) and departure_raw:
            return (DEPARTURE, departure_raw)
        return (None, None)

    def _read_profile(self, place_name: str) -> dict[str, Any] | None:
        """Return the profile dict for ``place_name`` or ``None``.

        A missing row returns ``None`` silently. Malformed JSON or a
        non-object top-level also returns ``None`` with a logged
        warning, matching sibling producers' fail-open posture.
        """
        row = self._conn.execute(
            "SELECT profile FROM signal_profiles WHERE producer=? AND key=?",
            (SPATIAL_PRODUCER_KEY, place_name),
        ).fetchone()
        if row is None:
            return None
        try:
            profile = json.loads(row["profile"])
        except (TypeError, ValueError):
            logger.warning("spatial profile for key=%r is not valid JSON; skipping", place_name)
            return None
        if not isinstance(profile, dict):
            logger.warning("spatial profile for key=%r is not a JSON object; skipping", place_name)
            return None
        return profile

    def _maybe_emit(
        self,
        kind: str,
        place_name: str,
        payload: dict[str, Any],
        profile: dict[str, Any],
    ) -> Moment | None:
        """Return a Moment if the (kind, profile, payload) triple warrants one.

        All gating is folded in here so the :meth:`observe` critical
        path stays flat. Any missing/wrong-type field collapses to
        ``None`` (fail-open), never raises.
        """
        try:
            visit_count = int(profile["visit_count"])
        except (KeyError, TypeError, ValueError):
            return None
        if visit_count < MIN_VISIT_COUNT:
            return None

        display_name = str(profile.get("place_name") or place_name)

        if kind == ARRIVAL:
            last_topic = profile.get("last_topic")
            if not isinstance(last_topic, str) or not last_topic.strip():
                return None
            insight = f"You're at {display_name}. Last time here you worked on {last_topic.strip()}."
        else:
            assert kind == DEPARTURE
            try:
                duration_minutes = int(payload["duration_minutes"])
                avg_minutes = int(profile["avg_duration_minutes"])
            except (KeyError, TypeError, ValueError):
                return None
            if duration_minutes < 1 or avg_minutes < 1:
                return None
            insight = f"You've been at {display_name} {duration_minutes} min, avg {avg_minutes}."

        now = int(self._now_fn())
        local_date_str = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%d")

        evidence = self._build_evidence(profile, kind, place_name, local_date_str)

        confidence = min(0.9, 0.3 + visit_count / 50.0)

        action_params: dict[str, Any] = {"place": display_name, "kind": kind}
        if kind == DEPARTURE:
            action_params["duration_minutes"] = int(payload["duration_minutes"])
            action_params["avg_duration_minutes"] = int(profile["avg_duration_minutes"])

        # Hash basis is (kind, place, date) so re-firing within the
        # same UTC day for the same arrival/departure collapses to one
        # row. See module docstring "Idempotency within a visit".
        hash_basis = [kind, place_name, local_date_str]

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            evidence_hash=self.evidence_hash(hash_basis),
            proposed_action=Action(kind=ActionKind.NOTE_OBSERVATION, params=action_params),
            source_insight_type=InsightType.SPATIAL,
            evidence=evidence,
            confidence=confidence,
        )

    @staticmethod
    def _build_evidence(
        profile: dict[str, Any],
        kind: str,
        place_name: str,
        date_str: str,
    ) -> list[str]:
        """Return up to 3 real event ids, else a synthetic id."""
        raw = profile.get("last_event_ids")
        if isinstance(raw, list):
            evidence = [str(e) for e in raw[:3] if e]
            if evidence:
                return evidence
        return [f"spatial:{kind}:{place_name}:{date_str}"]


__all__ = [
    "ARRIVAL",
    "DEFAULT_EXPIRY_SECONDS",
    "DEPARTURE",
    "MIN_VISIT_COUNT",
    "SPATIAL_PRODUCER_KEY",
    "TRIGGER_EVENT_TYPE",
    "SpatialProducer",
]
