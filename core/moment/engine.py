"""Moment engine — orchestrates producers → feedback-gated persistence.

The :class:`MomentEngine` is the single seam that turns a stream of
events into a stream of persisted :class:`~core.moment.types.Moment`
rows. Producers propose candidates; the feedback-weight EWMA decides
which ones clear the bar for this insight type; the moment repository
persists the survivors and dedupes by evidence hash.

Per-iteration flow
------------------
For each event delivered to :meth:`MomentEngine.on_event`:

1. Every registered producer's :meth:`~core.moment.producer.Producer.observe`
   is awaited. A producer that raises is logged and skipped — **fail-open
   per the v1 convention preserved in v2**: one misbehaving producer must
   never stall event ingest.
2. For each candidate Moment, the engine reads the current threshold for
   the Moment's ``source_insight_type`` from
   :class:`~core.moment.feedback_weight.FeedbackWeightStore`. Candidates
   with ``confidence < threshold`` are dropped silently (this is the
   feedback loop: repeatedly-dismissed insight types raise their own
   bar until the user flips the signal). See feedback_weight module
   docstring for the threshold formula.
3. Survivors are handed to :meth:`storage.repos.moments.MomentRepository.create`,
   which dedupes on ``(source_insight_type, evidence_hash)``. A
   duplicate evidence set thus collapses to a single row across
   re-runs of the same event.

The engine itself is stateless between calls — all ordering /
rate-limiting / scheduling concerns live in the scheduler and outbox
layers, not here. Producers remain the only component that *creates*
Moments; the engine only *gates* and *persists*.

Constructor injection
---------------------
Follows eng review §1a — the caller supplies concrete instances of
every collaborator. No module-level registry walk, no implicit
dependency discovery. The caller typically builds the producer list
from :data:`core.moment.producer.PRODUCERS` at startup (one instance
per entry) and wires the repo + weight store against a shared
``sqlite3.Connection``.

References
----------
- CEO plan § "The Moment Primitive (engine)":
  ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan § Week 6:
  ``docs/plans/2026-04-21-v2-rewrite-plan.md``
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from core.moment.feedback_weight import FeedbackWeightStore
from core.moment.producer import Event, Producer
from core.moment.types import Moment
from storage.repos.moments import MomentRepository

logger = logging.getLogger(__name__)


class MomentEngine:
    """Run producers against each event, gate by feedback weight, persist survivors.

    The engine owns no state of its own — it is a thin orchestration
    layer that each :meth:`on_event` call traverses top-to-bottom
    without side effects on the engine instance. That makes it safe to
    invoke from a single event loop without locking; concurrency is the
    repository's problem (which it solves via ``BEGIN IMMEDIATE``).
    """

    def __init__(
        self,
        producers: Iterable[Producer],
        moment_repo: MomentRepository,
        feedback_weight_store: FeedbackWeightStore,
    ) -> None:
        """Wire the engine to its collaborators.

        ``producers`` is materialized into a list at construction so
        mutations to the caller's source don't change engine behavior
        after wiring. Downstream calls iterate the snapshot.
        """
        self._producers: list[Producer] = list(producers)
        self._moment_repo = moment_repo
        self._feedback = feedback_weight_store

    async def on_event(self, event: Event) -> list[str]:
        """Run all producers against ``event``; return persisted Moment ids.

        Returns the list of Moment ids that ended up in the repo as a
        result of this event — whether freshly inserted or an idempotent
        hit on a pre-existing ``(source_insight_type, evidence_hash)``
        row. Candidates that fall below the feedback-gated confidence
        threshold are dropped silently and do not appear in the output.

        Error policy (fail-open, in order):

        - Producer ``observe`` raises → log + skip that producer.
        - Threshold lookup raises → log + skip that candidate (treat as
          unknown / drop conservatively).
        - Repo ``create`` raises → log + skip that candidate.

        Every failure is scoped to a single candidate; the event itself
        always finishes processing, so a bug in one producer cannot
        drain the queue behind it.
        """
        created_ids: list[str] = []
        for producer in self._producers:
            try:
                candidates = await producer.observe(event)
            except Exception:
                logger.exception(
                    "producer %s crashed on event %r; skipping",
                    producer.insight_type,
                    event.get("id"),
                )
                continue
            for candidate in candidates:
                if not self._passes_threshold(candidate):
                    continue
                moment_id = self._persist(candidate)
                if moment_id is not None:
                    created_ids.append(moment_id)
        return created_ids

    def _passes_threshold(self, moment: Moment) -> bool:
        """Return ``True`` iff ``moment.confidence >= threshold(insight_type)``.

        Threshold comes from the feedback weight EWMA per insight type.
        A lookup failure (malformed row, SQLite error) is logged and
        treated as a drop — the engine prefers quiet to wrong.
        """
        try:
            threshold = self._feedback.get_threshold_for(moment.source_insight_type)
        except Exception:
            logger.exception(
                "feedback threshold lookup failed for %s; dropping candidate",
                moment.source_insight_type,
            )
            return False
        return moment.confidence >= threshold

    def _persist(self, moment: Moment) -> str | None:
        """Persist a candidate; return the repo id or ``None`` on failure."""
        try:
            return self._moment_repo.create(moment)
        except Exception:
            logger.exception(
                "moment_repo.create failed for insight=%s evidence_hash=%s; skipping",
                moment.source_insight_type,
                moment.evidence_hash,
            )
            return None


__all__ = ["MomentEngine"]
