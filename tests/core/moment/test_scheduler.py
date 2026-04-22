"""Tests for :class:`core.moment.scheduler.Scheduler`.

Covers the four behaviors called out in the engineering plan Week 3
task body:

- **past-due fires via boot_recovery** — a Moment whose
  ``scheduled_for`` already slipped past ``now`` but whose TTL
  (``expires_at``) is still in the future is fired with annotation
  ``boot_recovery`` and an outbox ``moment.fire`` row lands.
- **snoozed past wake-time transitions** — a Moment in SNOOZED state
  whose ``scheduled_for`` is past-due is flipped back to SUGGESTED
  with annotation ``scheduler_fire`` on the next tick.
- **expired is terminal** — a past-due Moment whose TTL has also
  elapsed is transitioned to EXPIRED on boot recovery and does not
  fire; rerunning boot recovery is a no-op (EXPIRED is filtered out
  by :meth:`MomentRepository.list_scheduled`).
- **fire latency recorded** — each fire returns a :class:`FireRecord`
  whose ``latency_seconds`` equals the delta of the injected
  ``monotonic_fn`` calls around the fire (no wall-clock dependency,
  no ``freezegun``).

Additional coverage:

- Wraps the firing path around the real :class:`MomentRepository` and
  :class:`OutboxRepository` so integration with the schema + state
  machine is exercised together (no repo mocking, per the project
  testing convention).
- ``run_forever`` cancellability — the loop exits cleanly on
  :class:`asyncio.CancelledError`.
- ``_matches_trigger`` grammar coverage for each supported clause
  plus the "unknown grammar fails closed" guarantee.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from core.moment.scheduler import FireRecord, Scheduler
from core.moment.types import (
    Action,
    ActionKind,
    ContextTrigger,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.moments import MomentRepository
from storage.repos.outbox import OutboxRepository

# Fixed reference epoch: 2026-04-22T12:00:00Z, same as moments repo tests.
REF_NOW = 1_777_204_800


class Clock:
    """Mutable wall-clock stand-in."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class AdvancingMonotonic:
    """Monotonic-clock stand-in that advances a fixed ``step`` per call.

    Two calls per fire (start/end bracket in :meth:`Scheduler._fire`) give
    a ``latency_seconds`` of exactly ``step`` per firing, which lets
    assertions use :func:`pytest.approx` without wall-clock flakiness.
    """

    def __init__(self, step: float = 0.025) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        v = self.t
        self.t += self.step
        return v


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def monotonic():
    return AdvancingMonotonic(step=0.025)


@pytest.fixture
def moment_repo(conn, clock):
    return MomentRepository(conn, now_fn=clock)


@pytest.fixture
def outbox_repo(conn, clock):
    return OutboxRepository(conn, now_fn=clock)


@pytest.fixture
def scheduler(moment_repo, outbox_repo, clock, monotonic):
    return Scheduler(
        moment_repo,
        outbox_repo,
        bus=None,
        now_fn=clock,
        monotonic_fn=monotonic,
    )


def _make_moment(
    *,
    evidence_hash: str = "hash-1",
    state: MomentState = MomentState.SUGGESTED,
    scheduled_for: int | None = None,
    expires_at: int = REF_NOW + 72 * 3600,
    insight_type: InsightType = InsightType.CADENCE,
    created_at: int = REF_NOW,
    context_trigger: ContextTrigger | None = None,
) -> Moment:
    return Moment(
        id=str(uuid.uuid4()),
        created_at=created_at,
        expires_at=expires_at,
        insight="hi",
        evidence_hash=evidence_hash,
        proposed_action=Action(kind=ActionKind.NUDGE, params={"contact_id": "c1"}),
        source_insight_type=insight_type,
        scheduled_for=scheduled_for,
        context_trigger=context_trigger,
        evidence=["evt-1"],
        state=state,
    )


# ---------------------------------------------------------------------------
# boot_recovery
# ---------------------------------------------------------------------------


def test_boot_recovery_fires_past_due_within_ttl(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="past-due",
        scheduled_for=REF_NOW - 60,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.boot_recovery())

    assert len(records) == 1
    assert records[0].moment_id == m.id
    assert records[0].annotation == "boot_recovery"

    # Outbox row was enqueued with subject 'moment.fire'.
    claimed = outbox_repo.claim_batch()
    assert len(claimed) == 1
    assert claimed[0].subject == "moment.fire"
    assert claimed[0].payload["moment_id"] == m.id
    assert claimed[0].payload["annotation"] == "boot_recovery"
    assert claimed[0].payload["insight_type"] == InsightType.CADENCE.value

    # Moment stayed SUGGESTED (was not snoozed going in).
    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.SUGGESTED


def test_boot_recovery_expires_past_ttl_moment(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="past-ttl",
        scheduled_for=REF_NOW - 7200,
        expires_at=REF_NOW - 60,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.boot_recovery())

    assert records == []
    # Outbox was NOT touched — expiry is not a user-visible fire.
    assert outbox_repo.claim_batch() == []
    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.EXPIRED
    # History records the expiry annotation.
    assert loaded.state_history[-1].to_state == MomentState.EXPIRED
    assert loaded.state_history[-1].annotation == "boot_recovery_expired"


def test_boot_recovery_wakes_snoozed_past_scheduled(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="snooze-past",
        scheduled_for=REF_NOW - 60,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)
    moment_repo.transition(m.id, MomentState.SNOOZED, annotation="user_snooze")

    records = asyncio.run(scheduler.boot_recovery())

    assert len(records) == 1
    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.SUGGESTED
    # History: create → snoozed → suggested (annotated 'boot_recovery').
    assert loaded.state_history[-1].from_state == MomentState.SNOOZED
    assert loaded.state_history[-1].to_state == MomentState.SUGGESTED
    assert loaded.state_history[-1].annotation == "boot_recovery"


def test_boot_recovery_skips_future_scheduled(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="future",
        scheduled_for=REF_NOW + 3600,
        expires_at=REF_NOW + 86400,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.boot_recovery())

    assert records == []
    assert outbox_repo.claim_batch() == []
    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.SUGGESTED


def test_boot_recovery_skips_unscheduled_moments(scheduler, moment_repo, outbox_repo):
    m = _make_moment(evidence_hash="no-sched", scheduled_for=None)
    moment_repo.create(m)

    records = asyncio.run(scheduler.boot_recovery())

    assert records == []
    assert outbox_repo.claim_batch() == []


def test_boot_recovery_rerun_after_expiry_is_noop(scheduler, moment_repo):
    """Expired is terminal — a second boot recovery does nothing."""
    m = _make_moment(
        evidence_hash="term",
        scheduled_for=REF_NOW - 7200,
        expires_at=REF_NOW - 60,
    )
    moment_repo.create(m)

    asyncio.run(scheduler.boot_recovery())
    # Second run should be a no-op — list_scheduled filters by
    # state IN ('suggested','snoozed') so EXPIRED rows never reappear.
    records = asyncio.run(scheduler.boot_recovery())
    assert records == []

    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.EXPIRED


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------


def test_tick_fires_past_due_suggested(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="tick-fire",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.tick())

    assert len(records) == 1
    assert records[0].annotation == "scheduler_fire"
    claimed = outbox_repo.claim_batch()
    assert len(claimed) == 1
    assert claimed[0].payload["annotation"] == "scheduler_fire"


def test_tick_wakes_snoozed_past_scheduled(scheduler, moment_repo):
    m = _make_moment(
        evidence_hash="tick-wake",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)
    moment_repo.transition(m.id, MomentState.SNOOZED, annotation="user_snooze")

    records = asyncio.run(scheduler.tick())

    assert len(records) == 1
    loaded = moment_repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.SUGGESTED
    assert loaded.state_history[-1].annotation == "scheduler_fire"


def test_tick_ignores_future_scheduled(scheduler, moment_repo, outbox_repo):
    m = _make_moment(
        evidence_hash="tick-future",
        scheduled_for=REF_NOW + 3600,
        expires_at=REF_NOW + 86400,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.tick())
    assert records == []
    assert outbox_repo.claim_batch() == []


def test_tick_respects_terminal_state_filter(scheduler, moment_repo):
    """Terminal states never appear in list_scheduled, so tick never refires."""
    m = _make_moment(
        evidence_hash="terminal",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)
    moment_repo.transition(m.id, MomentState.DISMISSED, annotation="user_dismiss")

    records = asyncio.run(scheduler.tick())
    assert records == []


def test_tick_fires_multiple_past_due_moments(scheduler, moment_repo, outbox_repo):
    ids: list[str] = []
    for i in range(3):
        m = _make_moment(
            evidence_hash=f"m-{i}",
            scheduled_for=REF_NOW - (60 - i),  # all past-due, distinct times
            expires_at=REF_NOW + 3600,
        )
        moment_repo.create(m)
        ids.append(m.id)

    records = asyncio.run(scheduler.tick())

    assert len(records) == 3
    fired_ids = {r.moment_id for r in records}
    assert fired_ids == set(ids)
    # Three distinct outbox rows.
    claimed = outbox_repo.claim_batch()
    assert len(claimed) == 3
    assert {row.payload["moment_id"] for row in claimed} == set(ids)


def test_tick_refire_after_new_snooze_produces_new_outbox_row(scheduler, moment_repo, outbox_repo, clock):
    """Event ids are per-fire so re-wakes do not dedup at the outbox."""
    m = _make_moment(
        evidence_hash="refire",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 86400,
    )
    moment_repo.create(m)

    # First fire.
    asyncio.run(scheduler.tick())
    claimed1 = outbox_repo.claim_batch()
    outbox_repo.complete(claimed1[0].id)

    # User snoozes it, wake time elapses, tick fires again.
    moment_repo.transition(m.id, MomentState.SNOOZED, annotation="user_snooze")
    clock.t = REF_NOW + 10
    asyncio.run(scheduler.tick())

    claimed2 = outbox_repo.claim_batch()
    assert len(claimed2) == 1
    assert claimed2[0].id != claimed1[0].id
    # Both payloads point at the same Moment.
    assert claimed1[0].payload["moment_id"] == m.id
    assert claimed2[0].payload["moment_id"] == m.id


# ---------------------------------------------------------------------------
# fire latency
# ---------------------------------------------------------------------------


def test_fire_latency_recorded(scheduler, moment_repo):
    m = _make_moment(
        evidence_hash="latency",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)

    records = asyncio.run(scheduler.tick())

    assert len(records) == 1
    # AdvancingMonotonic steps 0.025 per call; _fire takes two calls
    # (start/end), so latency == step exactly.
    assert records[0].latency_seconds == pytest.approx(0.025)


def test_on_fire_callback_invoked_with_fire_record(moment_repo, outbox_repo, clock, monotonic):
    captured: list[FireRecord] = []
    s = Scheduler(
        moment_repo,
        outbox_repo,
        bus=None,
        now_fn=clock,
        monotonic_fn=monotonic,
        on_fire=captured.append,
    )
    m = _make_moment(
        evidence_hash="cb",
        scheduled_for=REF_NOW - 30,
        expires_at=REF_NOW + 3600,
    )
    moment_repo.create(m)

    asyncio.run(s.tick())

    assert len(captured) == 1
    assert captured[0].moment_id == m.id
    assert captured[0].latency_seconds == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# run_forever cancellation
# ---------------------------------------------------------------------------


def test_run_forever_exits_on_cancel(scheduler):
    async def run_and_cancel() -> None:
        task = asyncio.create_task(scheduler.run_forever(tick_seconds=0))
        # Let the event loop schedule the task + let it hit asyncio.sleep.
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())


def test_run_forever_swallows_tick_exceptions(moment_repo, outbox_repo, clock, monotonic):
    """A malformed Moment kills the tick, not the loop."""

    class BrokenMomentRepo:
        """Raises on every list_scheduled call."""

        def __init__(self, inner: MomentRepository) -> None:
            self._inner = inner
            self.calls = 0

        def list_scheduled(self, horizon_seconds: int = 0, limit: int = 1000) -> list[Moment]:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("boom")
            return []

        def transition(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner.transition(*args, **kwargs)

    broken = BrokenMomentRepo(moment_repo)
    s = Scheduler(broken, outbox_repo, bus=None, now_fn=clock, monotonic_fn=monotonic)

    async def run() -> None:
        task = asyncio.create_task(s.run_forever(tick_seconds=0))
        # Three iterations: two fail, one succeeds.
        while broken.calls < 3:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert broken.calls >= 3


# ---------------------------------------------------------------------------
# _matches_trigger grammar
# ---------------------------------------------------------------------------


def _moment_with_trigger(expr: str) -> Moment:
    return _make_moment(
        evidence_hash=f"trig-{expr}",
        context_trigger=ContextTrigger(expression=expr),
    )


def test_matches_trigger_event_type():
    m = _moment_with_trigger("event_type:email.received")
    assert Scheduler._matches_trigger(m, {"type": "email.received"}) is True
    assert Scheduler._matches_trigger(m, {"type": "message.received"}) is False


def test_matches_trigger_arrive_and_depart():
    arr = _moment_with_trigger("arrive:Home")
    dep = _moment_with_trigger("depart:Home")
    arrived_event = {
        "type": "context.location.updated",
        "payload": {"arrival": "Home"},
    }
    departed_event = {
        "type": "context.location.updated",
        "payload": {"departure": "Home"},
    }
    assert Scheduler._matches_trigger(arr, arrived_event) is True
    assert Scheduler._matches_trigger(arr, departed_event) is False
    assert Scheduler._matches_trigger(dep, departed_event) is True
    assert Scheduler._matches_trigger(dep, arrived_event) is False


def test_matches_trigger_time_and_weekday():
    ts = int(datetime(2026, 4, 22, 9, 30, tzinfo=UTC).timestamp())
    m_time = _moment_with_trigger("time:09:30")
    m_weekday = _moment_with_trigger("weekday:wednesday")
    event = {"type": "tick", "timestamp": ts}
    assert Scheduler._matches_trigger(m_time, event) is True
    assert Scheduler._matches_trigger(m_weekday, event) is True
    # Off-by-one minute misses.
    off = {"type": "tick", "timestamp": ts + 60}
    assert Scheduler._matches_trigger(m_time, off) is False


def test_matches_trigger_unknown_grammar_returns_false():
    m = _moment_with_trigger("calendar:gap>60m")
    # Not wired yet — fails closed.
    assert Scheduler._matches_trigger(m, {"type": "anything"}) is False
    m2 = _moment_with_trigger("nonsense:garbage")
    assert Scheduler._matches_trigger(m2, {"type": "anything"}) is False


def test_matches_trigger_no_trigger_returns_false():
    m = _make_moment(evidence_hash="no-trig", context_trigger=None)
    assert Scheduler._matches_trigger(m, {"type": "whatever"}) is False


def test_matches_trigger_malformed_time_returns_false():
    m = _moment_with_trigger("time:not-a-clock")
    assert Scheduler._matches_trigger(m, {"type": "x", "timestamp": REF_NOW}) is False
