"""Relationship producer — flags reciprocity drift with known contacts.

The second of the six Phase 1 producers. Where :mod:`producers.cadence`
watches *when* a contact was last heard from, the relationship producer
watches *how well the user is keeping up their side of the exchange*.

Trigger
-------
Every ``email.received`` and ``message.received`` event wakes the
producer. On each trigger the producer reads **all** rows from
``signal_profiles`` where ``producer = 'relationship'`` and asks each
one "has the user's reply ratio for this contact drifted?". Producers
are sparse by design — the common case is that no contact crosses the
drift threshold and :meth:`observe` returns ``[]``.

Triggering on inbound events (the same pulse the cadence producer
uses) gives us ~daily resolution without a separate cron. The Week 6
scheduler tick could replace the trigger later without changing the
producer's logic.

Profile shape
-------------
The profile JSON stored in ``signal_profiles.profile`` (one row per
contact, ``producer='relationship'``, ``key=contact_id``) must contain
at least:

.. code-block:: json

    {
        "previous_ratio": 0.7,
        "current_ratio": 0.2,
        "total_interactions": 38,
        "contact_name": "Alice",
        "channel": "email",
        "last_event_ids": ["evt-a", "evt-b", "evt-c"]
    }

- ``previous_ratio`` is the historical outbound/inbound ratio the
  producer treats as the baseline - typically the trailing 12-16 week
  mean written by a relationship signal extractor (future work).
- ``current_ratio`` is the ratio over the **last 4 weeks**; it is what
  we compare against to call drift.
- ``total_interactions`` is the sum of inbound + outbound events over
  the same 4-week window and gates against firing on low-sample
  contacts (the "stranger" case).
- ``last_event_ids`` is the most-recent inbound event ids (newest
  first) used purely for UI display as evidence.

A missing or malformed profile silently skips rather than raises —
producers are fail-open, matching v1's convention preserved in v2.

Threshold + confidence
----------------------
A Moment is emitted iff all four conditions hold:

- ``total_interactions >= 20`` — at least twenty observations over
  the 4-week window so the current ratio is statistically meaningful.
- ``previous_ratio > 0.5`` — the contact used to be a genuinely
  reciprocated relationship (baseline over half the time we wrote
  back).
- ``current_ratio < 0.3`` — our reply ratio has fallen below a third.
- ``last_event_ids`` is a non-empty list — required to surface
  evidence to the user.

Confidence is ``min(0.9, (previous_ratio - current_ratio) / previous_ratio)``,
so a modest drop produces a modest confidence and a dramatic collapse
saturates at 0.9 (leaving headroom for the feedback-weight EWMA in
Week 6 to demote a producer the user is dismissing).

Idempotency per week
--------------------
The task body calls for "idempotent per week": one Moment per
contact per calendar week, even as each new inbound event re-triggers
the scan. The producer achieves that by hashing a **week bucket**
(``week:{iso_year}-W{iso_week:02d}``) together with the contact id:

- Re-firing within the same ISO week for the same contact yields the
  same ``evidence_hash`` — the ``UNIQUE (source_insight_type,
  evidence_hash)`` constraint on ``moments`` then collapses the second
  insert to the first id.
- A new ISO week starting Monday UTC yields a fresh hash, so if the
  drift persists the producer can surface a new Moment.

The ``evidence`` field stored on the Moment stays populated with the
most-recent three inbound event ids for the UI; the evidence_hash is
deliberately computed from a different basis (week bucket +
contact id) so the dedup is week-scoped rather than
event-id-scoped. This is the only producer-level deviation from
cadence's shared-basis convention, and it is documented here because
the next reader will wonder.

References
----------
- CEO plan § "The Moment Primitive (producers) → Relationship":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 4: ``docs/plans/2026-04-21-v2-rewrite-plan.md``
- Sibling producer :mod:`producers.cadence` for the same trigger and
  fail-open conventions.
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


RELATIONSHIP_PRODUCER_KEY = "relationship"
"""``signal_profiles.producer`` value owned by the relationship producer."""

MIN_INTERACTION_COUNT = 20
"""Minimum total interactions in the 4-week window before the producer will fire."""

PREVIOUS_RATIO_MIN = 0.5
"""Historical reciprocity baseline must be strictly greater than this."""

CURRENT_RATIO_MAX = 0.3
"""Current 4-week outbound/inbound ratio must be strictly below this."""

DEFAULT_EXPIRY_SECONDS = 72 * 3600
"""Per CEO plan default: Moments expire 72h after creation if untouched."""

INBOUND_EVENT_TYPES: frozenset[str] = frozenset({"email.received", "message.received"})
"""Event types this producer cares about. All others short-circuit to ``[]``."""


@register
class RelationshipProducer(Producer):
    """Emits reciprocity-drift Moments when the user's reply ratio drops.

    Constructor takes a ``sqlite3.Connection`` that points at the v2
    ``lifeos.db``. The producer never writes — only the (future)
    relationship signal extractor owns profile maintenance — so the
    connection's transaction settings are left alone.

    ``now_fn`` and ``id_fn`` are injected for deterministic tests
    (stdlib only, per eng review §1c — no ``freezegun``).
    """

    insight_type = InsightType.RELATIONSHIP

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
        id_fn: Callable[[], str] | None = None,
    ) -> None:
        """Wire the producer to ``conn``; clocks/ids are injectable.

        ``conn.row_factory`` is forced to :class:`sqlite3.Row` so
        column access in :meth:`_read_profiles` is by name. This matches
        the cadence producer's convention; sharing a connection across
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
        """Return one Moment per drifted contact, possibly empty.

        Filters non-inbound events first so the common case never
        touches the DB. On a real inbound event, scans every
        relationship profile and lets :meth:`_maybe_emit` decide per
        profile.
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
        SQLite cursor closed before any per-profile work runs — matches
        the cadence producer's shape.
        """
        rows = self._conn.execute(
            "SELECT key, profile FROM signal_profiles WHERE producer=?",
            (RELATIONSHIP_PRODUCER_KEY,),
        ).fetchall()
        out: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            raw = row["profile"]
            try:
                profile = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "relationship profile for key=%r is not valid JSON; skipping",
                    row["key"],
                )
                continue
            if not isinstance(profile, dict):
                logger.warning(
                    "relationship profile for key=%r is not a JSON object; skipping",
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
        """Return a Moment if ``contact_id`` is drifted, else ``None``.

        The profile is allowed to be partially populated — any missing
        or wrong-type required field collapses to "skip". This matches
        the v1 fail-open posture: a malformed profile must never crash
        event processing.
        """
        try:
            previous_ratio = float(profile["previous_ratio"])
            current_ratio = float(profile["current_ratio"])
            total_interactions = int(profile["total_interactions"])
        except (KeyError, TypeError, ValueError):
            return None
        if total_interactions < MIN_INTERACTION_COUNT:
            return None
        # previously > 0.5 (strict) and currently < 0.3 (strict)
        if previous_ratio <= PREVIOUS_RATIO_MIN:
            return None
        if current_ratio >= CURRENT_RATIO_MAX:
            return None

        evidence_raw = profile.get("last_event_ids") or []
        if not isinstance(evidence_raw, list):
            return None
        # Use the most-recent 3 ids for the UI display block.
        evidence: list[str] = [str(e) for e in evidence_raw[:3] if e]
        if not evidence:
            return None

        contact_name = str(profile.get("contact_name") or contact_id)
        channel = str(profile.get("channel") or "")
        drop_fraction = (previous_ratio - current_ratio) / previous_ratio
        drop_pct = max(0, min(100, round(drop_fraction * 100)))
        insight = f"You've been replying less to {contact_name}. Outbound dropped {drop_pct}%."

        iso_year, iso_week, _ = datetime.fromtimestamp(now, tz=UTC).isocalendar()
        week_bucket = f"week:{iso_year}-W{iso_week:02d}"
        confidence = min(0.9, drop_fraction)

        return Moment(
            id=self._id_fn(),
            created_at=now,
            expires_at=now + DEFAULT_EXPIRY_SECONDS,
            insight=insight,
            # Hash basis is week-bucket + contact_id so dedup is per-week,
            # not per-event-id. See module docstring "Idempotency per week".
            evidence_hash=self.evidence_hash([week_bucket, contact_id]),
            proposed_action=Action(
                kind=ActionKind.NUDGE,
                params={"contact_id": contact_id, "channel": channel},
            ),
            source_insight_type=InsightType.RELATIONSHIP,
            evidence=evidence,
            confidence=confidence,
        )


__all__ = [
    "CURRENT_RATIO_MAX",
    "DEFAULT_EXPIRY_SECONDS",
    "INBOUND_EVENT_TYPES",
    "MIN_INTERACTION_COUNT",
    "PREVIOUS_RATIO_MIN",
    "RELATIONSHIP_PRODUCER_KEY",
    "RelationshipProducer",
]
