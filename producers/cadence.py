"""Cadence producer — flags contacts whose inbound rhythm has slipped.

The cadence producer is the first of the six Phase 1 producers. It
turns observations of incoming email and message events into
:class:`~core.moment.types.Moment` instances when a known contact has
gone unusually quiet relative to their historical cadence.

Trigger
-------
Every ``email.received`` and ``message.received`` event wakes the
producer. On each trigger the producer reads **all** rows from
``signal_profiles`` where ``producer = 'cadence'`` and asks each one
"is this contact overdue?". Producers are sparse by design — the
common case is that no profile crosses the drift threshold and
:meth:`observe` returns ``[]``.

Why scan all profiles on every inbound event? The thing that should
emit a cadence Moment is the *passage of time*, not a fresh ping. The
cleanest pulse we have without a separate cron is the firehose of
inbound events; one of them per minute (typical) is more than enough
resolution for a daily-scale signal. A future scheduler-tick wakeup
(Week 6's :class:`MomentEngine`) can replace this trigger without
changing the producer logic.

Profile shape
-------------
The profile JSON stored in ``signal_profiles.profile`` (one row per
contact, ``producer='cadence'``, ``key=contact_id``) must contain at
least:

.. code-block:: json

    {
        "expected_cadence_days": 7.0,
        "count": 12,
        "last_inbound_ts": 1777204800,
        "last_inbound_event_ids": ["evt-a", "evt-b", "evt-c"],
        "contact_name": "Alice",
        "channel": "email"
    }

A future cadence signal extractor owns *writing* this row; the
producer only reads it. Missing or malformed fields silently skip the
profile rather than raising — producers are fail-open per the v1
convention preserved in v2.

Threshold + confidence
----------------------
A Moment is emitted iff:

- ``days_since_last_inbound > expected_cadence_days * 1.3`` — 30%
  past the historical mean, calibrated to feel "noticeably late"
  without firing on every tail-end fluctuation.
- ``count >= 5`` — at least five historical inbound observations,
  so we don't fire on a stranger.

Confidence is ``min(0.9, days_since / expected / 2)`` — the more
overdue the contact, the louder the producer pushes. The cap at 0.9
leaves room for the feedback-weight EWMA (Week 6) to demote a
producer the user has been dismissing.

Idempotency
-----------
The ``evidence_hash`` is computed from the **last 3 inbound event
ids** stored on the profile. Two facts follow:

1. The profile owner (signal extractor) refreshes those ids every
   time a new inbound arrives. The next overdue check then yields a
   fresh hash, so the next Moment slots in alongside the prior one
   without UNIQUE-collision.
2. Repeated producer firings between profile updates collapse to one
   row in ``moments`` thanks to the
   ``UNIQUE (source_insight_type, evidence_hash)`` constraint —
   exactly the dedup behavior the task body requires.

Profiles with an empty ``last_inbound_event_ids`` are skipped: an
empty evidence set hashes to a sentinel digest, and emitting against
that would alias *every* evidence-less Moment to the same row.

References
----------
- CEO plan § "The Moment Primitive (producers) → Cadence":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 4: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- v1 cadence signal extractor (kept for behavior parity reference):
  ``services/signal_extractor/cadence.py``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable
from typing import Any

from core.moment.producer import Event, Producer, register
from core.moment.types import Action, ActionKind, InsightType, Moment

logger = logging.getLogger(__name__)


CADENCE_PRODUCER_KEY = "cadence"
"""``signal_profiles.producer`` value owned by the cadence producer."""

CADENCE_DRIFT_FACTOR = 1.3
"""Multiplier applied to expected cadence to get the drift threshold."""

MIN_HISTORY_COUNT = 5
"""Minimum inbound observations before the producer is willing to fire."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

SECONDS_PER_DAY = 86400.0

INBOUND_EVENT_TYPES: frozenset[str] = frozenset({"email.received", "message.received"})
"""Event types this producer cares about. All others short-circuit to ``[]``."""


@register
class CadenceProducer(Producer):
    """Emits cadence-drift Moments when a contact has gone quiet.

    Constructor takes a read-only ``sqlite3.Connection`` that points
    at the v2 ``lifeos.db``. The producer never writes — only the
    cadence signal extractor (future work) updates the underlying
    profile — so the connection's transaction settings are left
    alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.CADENCE

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so
        column access in :meth:`_read_profiles` is by name. This is
        the same convention :class:`storage.repos.MomentRepository`
        uses; sharing a connection between repo and producer is safe.
        """
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._id_fn: Callable[[], str] = id_fn or (lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Producer contract
    # ------------------------------------------------------------------
    async def observe(self, event: Event) -> list[Moment]:
        """Return one Moment per drifted contact, possibly empty.

        Filters non-inbound events first so the common case never
        touches the DB. On a real inbound event, scans every cadence
        profile and lets :meth:`_maybe_emit` decide per profile.
        """
        if event.get("type") not in INBOUND_EVENT_TYPES:
            return []
        now = int(self._now_fn())
        moments: list[Moment] = []
        for contact_id, profile in self._read_profiles():
            moment = self._maybe_emit(contact_id, profile, now)
            if moment is not None:
                moments.append(moment)
        return moments

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _read_profiles(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(contact_id, profile_dict)`` pairs, malformed rows skipped.

        Returning a fully-materialized list (not a generator) keeps the
        SQLite cursor closed before any per-profile work runs — small
        but matters because :meth:`_maybe_emit` does no I/O and we
        want to avoid holding a cursor open across the whole loop.
        """
        rows = self._conn.execute(
            "SELECT key, profile FROM signal_profiles WHERE producer=?",
            (CADENCE_PRODUCER_KEY,),
        ).fetchall()
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            raw = row["profile"]
            try:
                profile = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "cadence profile for key=%r is not valid JSON; skipping",
                    row["key"],
                )
                continue
            if not isinstance(profile, dict):
                logger.warning(
                    "cadence profile for key=%r is not a JSON object; skipping",
                    row["key"],
                )
                continue
            out.append((row["key"], profile))
        return out

    def _maybe_emit(
        self,
        contact_id: str,
        profile: dict[str, Any],
        now: int,
    ) -> Moment | None:
        """Return a Moment if ``contact_id`` is overdue, else ``None``.

        The profile is allowed to be partially populated — any
        missing/wrong-type required field collapses to "skip". This
        matches the v1 fail-open posture: a malformed profile must
        never crash event processing.
        """
        try:
            expected = float(profile["expected_cadence_days"])
            count = int(profile["count"])
            last_ts = int(profile["last_inbound_ts"])
        except (KeyError, TypeError, ValueError):
            return None
        if expected <= 0 or count < MIN_HISTORY_COUNT:
            return None

        days_since = max(0.0, (now - last_ts) / SECONDS_PER_DAY)
        if days_since <= expected * CADENCE_DRIFT_FACTOR:
            return None

        evidence_raw = profile.get("last_inbound_event_ids") or []
        if not isinstance(evidence_raw, list):
            return None
        # Use the most-recent 3 ids for evidence (stable hash key + UI display).
        evidence: list[str] = [str(e) for e in evidence_raw[:3] if e]
        if not evidence:
            # Empty evidence would alias all evidence-less Moments to one row.
            return None

        contact_name = str(profile.get("contact_name") or contact_id)
        channel = str(profile.get("channel") or "")
        days_int = round(days_since)
        expected_int = max(1, round(expected))
        insight = f"{days_int} days since you've heard from {contact_name}. Usual cadence {expected_int} days."
        confidence = min(0.9, days_since / expected / 2.0)

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            evidence_hash=self.evidence_hash(evidence),
            proposed_action=Action(
                kind=ActionKind.NUDGE,
                params={"contact_id": contact_id, "channel": channel},
            ),
            source_insight_type=InsightType.CADENCE,
            evidence=evidence,
            confidence=confidence,
        )


__all__ = [
    "CADENCE_DRIFT_FACTOR",
    "CADENCE_PRODUCER_KEY",
    "DEFAULT_EXPIRY_SECONDS",
    "INBOUND_EVENT_TYPES",
    "MIN_HISTORY_COUNT",
    "CadenceProducer",
]
