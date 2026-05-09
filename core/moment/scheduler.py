"""Scheduler — wall-clock firing loop for the Moment primitive.

The scheduler is the heartbeat that turns time-anchored Moments into
user-visible alerts. It runs every ``tick_seconds`` and selects rows
from the ``moments`` table whose ``scheduled_for`` has come due:

- ``SNOOZED`` rows transition back to ``SUGGESTED`` via
  :meth:`MomentRepository.transition` with annotation
  ``scheduler_fire`` (wake-ups from user-chosen snoozes).
- ``SUGGESTED`` rows stay in state (they are already eligible to be
  surfaced by the Now tab) but the fire is still recorded in the
  outbox so the notification worker can deliver on first surface.

Each firing enqueues a ``moment.fire`` event on the transactional
outbox (:class:`OutboxRepository`). The ``event_id`` includes a
monotonically-increasing per-fire token so re-fires from successive
snoozes each land a fresh outbox row rather than being deduped.

Boot recovery (:meth:`boot_recovery`) handles the shutdown edge case:
if the process died while a Moment was waiting on its ``scheduled_for``
window, the next boot still fires the Moment with annotation
``boot_recovery`` **iff** the TTL (``expires_at``) has not passed.
Past-TTL rows transition to ``EXPIRED`` instead — the CEO plan's
72-hour default TTL bounds the "late fire" surface area.

Clock injection
---------------
Both :func:`time.time` (wall clock, used for ``scheduled_for`` / DB
stamps) and :func:`time.monotonic` (used for fire-latency histograms)
are injectable via ``now_fn=`` and ``monotonic_fn=`` so tests can
advance time without ``freezegun``. Stdlib only, per eng review.

References
----------
CEO plan § "Scheduler & Moment lifecycle" at
``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``.
Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.moment.types import Moment, MomentState

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FireRecord:
    """Telemetry record for one Moment firing.

    Returned by :meth:`Scheduler.tick` and :meth:`Scheduler.boot_recovery`
    so callers can drive /metrics histograms (fire latency, annotation
    breakdown) without inspecting the outbox directly.
    """

    moment_id: str
    annotation: str
    latency_seconds: float
    outbox_id: str


class Scheduler:
    """Wall-clock firing loop over the ``moments.scheduled_for`` column.

    Constructor-injection per eng review §1a: callers pass their own
    repository instances and clock implementations. No globals, no
    module-level state. The ``bus`` parameter is accepted as a future
    extension point (in-process event bus for websocket push) but is
    not exercised by the Phase 1 firing path — the outbox is the
    authoritative sink.
    """

    def __init__(
        self,
        moment_repo: Any,
        outbox_repo: Any,
        bus: Any = None,
        *,
        now_fn: Callable[[], float] | None = None,
        monotonic_fn: Callable[[], float] | None = None,
        on_fire: Callable[[FireRecord], None] | None = None,
        batch_limit: int = 1000,
    ) -> None:
        """Wire up dependencies.

        ``on_fire`` is an optional synchronous callback invoked once per
        firing with the :class:`FireRecord`; it is the hook the
        /metrics exporter uses to record Prometheus histograms without
        this module importing the metrics subsystem. ``batch_limit``
        caps how many rows a single tick/boot-recovery will drain —
        defaults to 1000, which is far above any plausible backlog.
        """
        self._moments = moment_repo
        self._outbox = outbox_repo
        self._bus = bus
        self._now_fn: Callable[[], float] = now_fn or time.time
        self._mono_fn: Callable[[], float] = monotonic_fn or time.monotonic
        self._on_fire = on_fire
        self._batch_limit = batch_limit

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> int:
        return int(self._now_fn())

    def _fire(self, moment: Moment, annotation: str) -> FireRecord:
        """Wake the Moment (if snoozed) and enqueue a fire event.

        Latency is measured monotonically around the full fire cycle
        (state transition + outbox enqueue). A fresh event id is
        generated per invocation so the outbox's
        ``UNIQUE (event_id, subject)`` constraint does not silently
        dedup re-fires from successive snooze cycles.
        """
        start = self._mono_fn()
        if moment.state == MomentState.SNOOZED:
            self._moments.transition(
                moment.id,
                MomentState.SUGGESTED,
                annotation=annotation,
            )
        event_id = f"moment_fire:{moment.id}:{self._now()}:{uuid.uuid4().hex[:8]}"
        outbox_id = self._outbox.enqueue(
            event_id,
            "moment.fire",
            {
                "moment_id": moment.id,
                "annotation": annotation,
                "insight_type": moment.source_insight_type.value,
                "scheduled_for": moment.scheduled_for,
            },
        )
        latency = self._mono_fn() - start
        record = FireRecord(
            moment_id=moment.id,
            annotation=annotation,
            latency_seconds=latency,
            outbox_id=outbox_id,
        )
        log.info(
            "scheduler.fire moment_id=%s annotation=%s latency=%.4fs",
            moment.id,
            annotation,
            latency,
        )
        if self._on_fire is not None:
            self._on_fire(record)
        return record

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def boot_recovery(self, horizon_seconds: int = 7 * 86400) -> list[FireRecord]:
        """Fire or expire Moments whose ``scheduled_for`` is already past.

        Called once at service startup, before :meth:`run_forever` takes
        over. For every past-due row:

        - If ``expires_at > now``: fire with ``annotation='boot_recovery'``.
        - Else: transition to :attr:`MomentState.EXPIRED` with
          ``annotation='boot_recovery_expired'``.

        ``horizon_seconds`` bounds how far into the future
        :meth:`MomentRepository.list_scheduled` looks, but only past-due
        rows are acted on — future scheduled rows are ignored here and
        picked up later by :meth:`tick`.
        """
        records: list[FireRecord] = []
        now_ts = self._now()
        due = self._moments.list_scheduled(
            horizon_seconds=horizon_seconds,
            limit=self._batch_limit,
        )
        for moment in due:
            if moment.scheduled_for is None or moment.scheduled_for > now_ts:
                continue
            if moment.expires_at <= now_ts:
                self._moments.transition(
                    moment.id,
                    MomentState.EXPIRED,
                    annotation="boot_recovery_expired",
                )
                continue
            records.append(self._fire(moment, annotation="boot_recovery"))
        return records

    async def tick(self) -> list[FireRecord]:
        """Run one firing tick. Fire every past-due Moment; return records.

        Unlike :meth:`boot_recovery`, the tick does not expire: in
        steady state the loop fires due Moments long before their
        72-hour TTL elapses, so the only expiry path Phase 1 exercises
        is the boot-recovery one. The state machine itself still rejects
        fires of an already-terminal Moment via
        :class:`core.moment.state.IllegalTransition`.
        """
        records: list[FireRecord] = []
        now_ts = self._now()
        due = self._moments.list_scheduled(
            horizon_seconds=0,
            limit=self._batch_limit,
        )
        for moment in due:
            if moment.scheduled_for is None or moment.scheduled_for > now_ts:
                continue
            records.append(self._fire(moment, annotation="scheduler_fire"))
        return records

    async def run_forever(self, tick_seconds: int = 30) -> None:
        """Infinite firing loop.

        Sleeps ``tick_seconds`` between ticks. Exits cleanly on
        :class:`asyncio.CancelledError`; any other exception from a
        single tick is logged and swallowed so one malformed Moment
        cannot stall the loop.
        """
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler.tick failed")
            await asyncio.sleep(tick_seconds)

    # ------------------------------------------------------------------
    # ContextTrigger matcher (event-driven wake path)
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_trigger(moment: Moment, event: dict[str, Any]) -> bool:
        """Return ``True`` iff ``event`` satisfies ``moment.context_trigger``.

        Grammar (CEO plan § "ContextTrigger grammar (v1 vocabulary)"):

        - ``event_type:{type}`` — matches ``event['type']`` exactly.
        - ``arrive:{place}`` / ``depart:{place}`` — matches
          ``context.location.updated`` events carrying a matching
          ``payload.arrival`` / ``payload.departure``.
        - ``time:{HH:MM}`` — matches events whose ``timestamp`` falls
          on the given wall-clock minute in UTC. A future
          connectors-aware layer can rewrap in the user's timezone;
          the scheduler itself has no tz knowledge.
        - ``weekday:{name}`` — matches events on that weekday name
          in UTC (e.g. ``monday``).
        - ``calendar:...`` and ``after_inactivity:...`` require state
          the scheduler does not yet carry in Phase 1 (calendar gap
          detection lives in the temporal producer); these return
          ``False`` pending the Week 4+ wire-up. Unknown grammar also
          returns ``False`` so a typo fails closed rather than firing
          incorrectly.
        """
        trigger = moment.context_trigger
        if trigger is None:
            return False
        expr = trigger.expression.strip()
        event_type = event.get("type", "")
        event_ts = event.get("timestamp")

        if expr.startswith("event_type:"):
            return bool(event_type == expr.split(":", 1)[1])
        if expr.startswith("arrive:"):
            place = expr.split(":", 1)[1]
            return bool(event_type == "context.location.updated" and event.get("payload", {}).get("arrival") == place)
        if expr.startswith("depart:"):
            place = expr.split(":", 1)[1]
            return bool(event_type == "context.location.updated" and event.get("payload", {}).get("departure") == place)
        if expr.startswith("time:"):
            hhmm = expr.split(":", 1)[1]
            try:
                target = datetime.strptime(hhmm, "%H:%M").time()
            except ValueError:
                return False
            if event_ts is None:
                return False
            dt = datetime.fromtimestamp(float(event_ts), tz=UTC)
            return dt.hour == target.hour and dt.minute == target.minute
        if expr.startswith("weekday:"):
            day = expr.split(":", 1)[1].lower()
            if event_ts is None:
                return False
            return datetime.fromtimestamp(float(event_ts), tz=UTC).strftime("%A").lower() == day
        return False


__all__ = ["FireRecord", "Scheduler"]
