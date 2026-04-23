"""Edge-case tests for :class:`core.moment.scheduler.Scheduler`.

Targets coverage gap #9 in the 2026-04-23 audit — the four narrow
branches left uncovered by the main scheduler test suite:

- ``tick()`` skips Moments whose ``scheduled_for`` is ``None`` or in
  the future (defensive continue, reachable via a stub repo).
- ``run_forever`` re-raises :class:`asyncio.CancelledError` caught
  inside the inner ``try`` block (cancel while ``tick()`` is awaiting,
  not during the outer sleep).
- ``_matches_trigger`` on a ``time:`` expression returns ``False`` when
  the event has no ``timestamp``.
- ``_matches_trigger`` on a ``weekday:`` expression returns ``False``
  when the event has no ``timestamp``.

Uses the stdlib only — no freezegun, no asyncio.wait_for wrapping.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from typing import Any

import pytest

from core.moment.scheduler import Scheduler
from core.moment.types import (
    Action,
    ActionKind,
    ContextTrigger,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.outbox import OutboxRepository

REF_NOW = 1_777_204_800


class _Clock:
    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _Monotonic:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        v = self.t
        self.t += 0.001
        return v


def _moment(**kw: Any) -> Moment:
    defaults: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created_at": REF_NOW,
        "expires_at": REF_NOW + 72 * 3600,
        "insight": "hi",
        "evidence_hash": kw.get("evidence_hash", "h1"),
        "proposed_action": Action(kind=ActionKind.NUDGE, params={"contact_id": "c1"}),
        "source_insight_type": InsightType.CADENCE,
        "scheduled_for": None,
        "context_trigger": None,
        "evidence": ["evt-1"],
        "state": MomentState.SUGGESTED,
    }
    defaults.update(kw)
    return Moment(**defaults)


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
    return _Clock()


# ---------------------------------------------------------------------------
# tick: defensive continue for unscheduled / future moments
# ---------------------------------------------------------------------------


class _StubMomentRepo:
    """Repo double whose ``list_scheduled`` lies so the scheduler hits
    the defensive ``continue`` branch at scheduler.tick."""

    def __init__(self, returned: list[Moment]) -> None:
        self._returned = returned
        self.transition_calls: list[tuple[str, MomentState, str | None]] = []

    def list_scheduled(self, horizon_seconds: int = 0, limit: int = 1000) -> list[Moment]:
        return self._returned

    def transition(self, moment_id: str, new_state: MomentState, annotation: str | None = None) -> Moment:
        # Not called in this path; record if we do reach it so the
        # test can assert the continue fired.
        self.transition_calls.append((moment_id, new_state, annotation))
        raise AssertionError("unexpected transition() call — defensive continue should have skipped")


def test_tick_continues_past_unscheduled_moment(conn, clock):
    """A Moment whose ``scheduled_for`` is ``None`` falls through the continue.

    The stub repo fakes a list_scheduled row that the real repo would
    never return (real repo filters ``scheduled_for IS NOT NULL``), so
    we can assert the defensive check inside :meth:`Scheduler.tick`.
    """
    outbox = OutboxRepository(conn, now_fn=clock)
    stub = _StubMomentRepo([_moment(scheduled_for=None, evidence_hash="unsch")])
    scheduler = Scheduler(stub, outbox, bus=None, now_fn=clock, monotonic_fn=_Monotonic())

    records = asyncio.run(scheduler.tick())

    assert records == []
    assert stub.transition_calls == []
    assert outbox.claim_batch() == []


def test_tick_continues_past_future_scheduled_moment(conn, clock):
    """A Moment whose ``scheduled_for`` is in the future is skipped."""
    outbox = OutboxRepository(conn, now_fn=clock)
    stub = _StubMomentRepo([_moment(scheduled_for=REF_NOW + 3600, evidence_hash="fut")])
    scheduler = Scheduler(stub, outbox, bus=None, now_fn=clock, monotonic_fn=_Monotonic())

    records = asyncio.run(scheduler.tick())

    assert records == []
    assert stub.transition_calls == []


# ---------------------------------------------------------------------------
# run_forever: cancel during tick() re-raises CancelledError
# ---------------------------------------------------------------------------


class _SlowMomentRepo:
    """Repo whose ``list_scheduled`` yields the loop (via sleep) so the
    task can be cancelled while inside ``await self.tick()``."""

    def __init__(self) -> None:
        self.in_flight = asyncio.Event()
        self.calls = 0

    def list_scheduled(self, horizon_seconds: int = 0, limit: int = 1000) -> list[Moment]:
        # Sync method but we need the scheduler task to be suspended so
        # cancel can arrive while ``await self.tick()`` is on the stack.
        # Easiest: bump a counter and return []; the scheduler loop will
        # then hit ``await asyncio.sleep(tick_seconds)`` — but we want to
        # catch it inside tick. See the async wrapper below.
        self.calls += 1
        return []

    def transition(self, *a: Any, **kw: Any) -> None:
        raise AssertionError("unused")


class _AwaitingScheduler(Scheduler):
    """Scheduler subclass whose ``tick`` yields to the loop, creating a
    cancellation point inside the ``try`` block so the outer
    ``except asyncio.CancelledError: raise`` is the one that fires.
    """

    async def tick(self):  # type: ignore[override]
        # Yield so the cancel sent via ``task.cancel()`` can be delivered
        # here (inside the ``try`` block of run_forever) rather than
        # during the outer ``asyncio.sleep(tick_seconds)``.
        await asyncio.sleep(0.01)
        return await super().tick()


def test_run_forever_reraises_cancelled_error_from_tick(conn, clock):
    """Cancelling while tick() is awaiting exercises the inner raise."""
    outbox = OutboxRepository(conn, now_fn=clock)
    repo = _SlowMomentRepo()
    scheduler = _AwaitingScheduler(
        repo,
        outbox,
        bus=None,
        now_fn=clock,
        monotonic_fn=_Monotonic(),
    )

    async def go() -> None:
        task = asyncio.create_task(scheduler.run_forever(tick_seconds=10))
        # Wait one event-loop tick so the task enters run_forever and
        # reaches ``await asyncio.sleep(0.01)`` inside tick.
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


# ---------------------------------------------------------------------------
# _matches_trigger: time: / weekday: with missing event timestamp
# ---------------------------------------------------------------------------


def test_matches_trigger_time_returns_false_without_timestamp():
    m = _moment(
        evidence_hash="time-no-ts",
        context_trigger=ContextTrigger(expression="time:09:30"),
    )
    # Event is missing 'timestamp'.
    assert Scheduler._matches_trigger(m, {"type": "tick"}) is False


def test_matches_trigger_weekday_returns_false_without_timestamp():
    m = _moment(
        evidence_hash="weekday-no-ts",
        context_trigger=ContextTrigger(expression="weekday:monday"),
    )
    assert Scheduler._matches_trigger(m, {"type": "tick"}) is False
