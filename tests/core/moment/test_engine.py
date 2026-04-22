"""Tests for :class:`core.moment.engine.MomentEngine`.

Covers the three properties called out in the Week 6 task body:

1. **Dispatch** — ``on_event`` calls ``observe`` on every registered
   producer with the same event, regardless of the insight type they
   own. Producers returning ``[]`` are no-ops; producers returning
   Moments feed into the gating step.
2. **Feedback-weight gating** — candidates with
   ``confidence < threshold_for(insight_type)`` are dropped, survivors
   are persisted. The cold-start threshold is ``0.6 + 0.5 = 1.1`` which
   no bounded confidence can clear, so producers that want to emit on a
   fresh store must cross that bar (producers in the Phase 1 set cap at
   0.9, so the test drives the EWMA up by pre-accepting enough samples).
3. **Dedup** — replaying the same event twice does not create duplicate
   Moments; the ``UNIQUE (source_insight_type, evidence_hash)``
   constraint collapses the second insert and the engine returns the
   original id.

Plus the fail-open invariants the engine promises in its docstring:

- a producer that raises does not stall other producers;
- a repo failure does not stall subsequent candidates;
- a feedback-store failure drops the candidate without propagating.

The 30-day integration test drives a realistic event stream through
two producers wired to a real ``MomentRepository`` + real
``FeedbackWeightStore`` against the consolidated v2 schema. No mocks
for storage — the schema is authoritative.
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from core.moment.engine import MomentEngine
from core.moment.feedback_weight import FeedbackWeightStore
from core.moment.producer import Event, Producer
from core.moment.types import (
    Action,
    ActionKind,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.moments import MomentRepository

REF_NOW = 1_777_204_800  # 2026-04-22T00:00:00Z, matches the fixture date.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()


def _make_moment(
    insight_type: InsightType,
    evidence_ids: list[str],
    *,
    confidence: float = 1.0,
    created_at: int = REF_NOW,
    insight_text: str = "test moment",
) -> Moment:
    return Moment(
        id=str(uuid.uuid4()),
        created_at=created_at,
        expires_at=created_at + 3600,
        insight=insight_text,
        evidence_hash=Producer.evidence_hash(evidence_ids),
        proposed_action=Action(kind=ActionKind.NUDGE),
        source_insight_type=insight_type,
        evidence=evidence_ids,
        confidence=confidence,
    )


class FakeProducer(Producer):
    """Deterministic producer — emits a caller-supplied list on observe."""

    def __init__(
        self,
        insight_type: InsightType,
        emit_factory: Any = None,
    ) -> None:
        self.insight_type = insight_type
        self._emit_factory = emit_factory or (lambda event: [])
        self.observed: list[Event] = []

    async def observe(self, event: Event) -> list[Moment]:
        self.observed.append(event)
        return list(self._emit_factory(event))


class CrashingProducer(Producer):
    """Producer that always raises from ``observe``."""

    def __init__(self, insight_type: InsightType) -> None:
        self.insight_type = insight_type
        self.call_count = 0

    async def observe(self, event: Event) -> list[Moment]:
        self.call_count += 1
        raise RuntimeError("producer crashed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(":memory:")
    _apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def repo(conn: sqlite3.Connection) -> MomentRepository:
    return MomentRepository(conn, now_fn=lambda: REF_NOW)


@pytest.fixture
def weights(conn: sqlite3.Connection) -> FeedbackWeightStore:
    return FeedbackWeightStore(conn, now_fn=lambda: REF_NOW)


def _train_to_base(weights: FeedbackWeightStore, insight_type: InsightType) -> None:
    """Drive the EWMA toward w≈1.0 so threshold drops to ~BASE_THRESHOLD.

    Cold-start threshold is 1.1 which no producer can clear (confidences
    are bounded ≤ 1.0). The tests that want candidates to land through
    the gate pre-accept enough samples to push the threshold under the
    candidate's confidence.
    """
    for _ in range(50):
        weights.update(insight_type, MomentState.ACCEPTED)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_on_event_invokes_every_producer(repo, weights) -> None:
    """Each producer sees the same event, regardless of what it emits."""
    a = FakeProducer(InsightType.CADENCE)
    b = FakeProducer(InsightType.RELATIONSHIP)
    engine = MomentEngine([a, b], repo, weights)

    event = {"id": "e-1", "type": "email.received", "timestamp": REF_NOW}
    asyncio.run(engine.on_event(event))

    assert a.observed == [event]
    assert b.observed == [event]


def test_on_event_with_no_producers_returns_empty(repo, weights) -> None:
    """Zero producers is a valid wiring — on_event is a no-op."""
    engine = MomentEngine([], repo, weights)
    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))
    assert out == []


def test_on_event_returns_empty_when_all_producers_return_empty(repo, weights) -> None:
    """Nothing to persist → empty list, no rows."""
    p = FakeProducer(InsightType.CADENCE)  # default emits []
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))
    assert out == []
    assert repo.list_pending() == []


# ---------------------------------------------------------------------------
# Feedback-weight gating
# ---------------------------------------------------------------------------


def test_cold_start_threshold_blocks_all_candidates(repo, weights) -> None:
    """At cold start, threshold is 1.1 — no bounded confidence clears it."""
    candidate = _make_moment(InsightType.CADENCE, ["e-1"], confidence=0.9)
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [candidate])
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert out == []
    assert repo.list_pending() == []


def test_trained_weight_lets_candidates_through(repo, weights) -> None:
    """After enough accepts, threshold drops below candidate confidence."""
    _train_to_base(weights, InsightType.CADENCE)
    candidate = _make_moment(InsightType.CADENCE, ["e-1"], confidence=0.9)
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [candidate])
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert len(out) == 1
    assert len(repo.list_pending()) == 1


def test_candidate_below_threshold_dropped_silently(repo, weights) -> None:
    """Below-threshold candidates do not persist and do not appear in the output."""
    _train_to_base(weights, InsightType.CADENCE)
    # Threshold is ~0.6; a 0.5-confidence candidate must drop.
    low = _make_moment(InsightType.CADENCE, ["e-low"], confidence=0.5)
    high = _make_moment(InsightType.CADENCE, ["e-high"], confidence=0.9)
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [low, high])
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert len(out) == 1
    pending = repo.list_pending()
    assert len(pending) == 1
    assert pending[0].evidence == ["e-high"]


def test_gating_is_per_insight_type(repo, weights) -> None:
    """Training one insight type must not open the gate for another."""
    _train_to_base(weights, InsightType.CADENCE)
    # RELATIONSHIP remains cold-start (threshold 1.1), so its candidate drops.
    cadence_c = _make_moment(InsightType.CADENCE, ["e-c"], confidence=0.9)
    rel_c = _make_moment(InsightType.RELATIONSHIP, ["e-r"], confidence=0.9)

    pc = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [cadence_c])
    pr = FakeProducer(InsightType.RELATIONSHIP, emit_factory=lambda event: [rel_c])
    engine = MomentEngine([pc, pr], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert len(out) == 1
    pending = repo.list_pending()
    assert len(pending) == 1
    assert pending[0].source_insight_type is InsightType.CADENCE


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_replay_same_event_dedupes_on_evidence_hash(repo, weights) -> None:
    """Two events carrying the same evidence set collapse to one row."""
    _train_to_base(weights, InsightType.CADENCE)
    # Emit identical candidate each call (same evidence → same hash).
    candidate = _make_moment(InsightType.CADENCE, ["e-1", "e-2"], confidence=0.9)
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [candidate])
    engine = MomentEngine([p], repo, weights)

    id1 = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))
    id2 = asyncio.run(engine.on_event({"id": "e-2", "type": "x", "timestamp": REF_NOW + 60}))

    assert id1 == id2  # Same Moment id from idempotent create.
    assert len(repo.list_pending()) == 1


def test_distinct_evidence_sets_create_distinct_moments(repo, weights) -> None:
    """Different evidence = different hash = distinct rows."""
    _train_to_base(weights, InsightType.CADENCE)
    candidates = [
        _make_moment(InsightType.CADENCE, ["e-a"], confidence=0.9, insight_text="m1"),
        _make_moment(InsightType.CADENCE, ["e-b"], confidence=0.9, insight_text="m2"),
    ]
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: candidates)
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert len(out) == 2
    assert len(set(out)) == 2
    assert len(repo.list_pending()) == 2


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------


def test_producer_crash_does_not_stall_other_producers(repo, weights, caplog) -> None:
    """A crashing producer is logged + skipped; the next producer still runs."""
    _train_to_base(weights, InsightType.RELATIONSHIP)
    crash = CrashingProducer(InsightType.CADENCE)
    rel_c = _make_moment(InsightType.RELATIONSHIP, ["e-r"], confidence=0.9)
    good = FakeProducer(InsightType.RELATIONSHIP, emit_factory=lambda event: [rel_c])
    engine = MomentEngine([crash, good], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert crash.call_count == 1  # Was called...
    assert len(out) == 1  # ...and the rest of the pipeline kept going.
    assert any("crashed" in rec.getMessage() and "cadence" in rec.getMessage() for rec in caplog.records)


def test_repo_create_failure_does_not_stall_subsequent_candidates(repo, weights, monkeypatch) -> None:
    """A repo-level error on one candidate does not drop the others."""
    _train_to_base(weights, InsightType.CADENCE)
    bad = _make_moment(InsightType.CADENCE, ["e-bad"], confidence=0.9)
    good = _make_moment(InsightType.CADENCE, ["e-good"], confidence=0.9)

    original_create = repo.create
    calls = {"n": 0}

    def flaky_create(moment: Moment) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("simulated failure")
        return original_create(moment)

    monkeypatch.setattr(repo, "create", flaky_create)

    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [bad, good])
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert len(out) == 1  # Only the good candidate made it.
    pending = repo.list_pending()
    assert len(pending) == 1
    assert pending[0].evidence == ["e-good"]


def test_feedback_lookup_failure_drops_candidate(repo, weights, monkeypatch) -> None:
    """A threshold-lookup error drops the candidate conservatively."""

    def boom(_insight_type: Any) -> float:
        raise RuntimeError("simulated feedback failure")

    monkeypatch.setattr(weights, "get_threshold_for", boom)

    candidate = _make_moment(InsightType.CADENCE, ["e-1"], confidence=0.9)
    p = FakeProducer(InsightType.CADENCE, emit_factory=lambda event: [candidate])
    engine = MomentEngine([p], repo, weights)

    out = asyncio.run(engine.on_event({"id": "e-1", "type": "x", "timestamp": REF_NOW}))

    assert out == []
    assert repo.list_pending() == []


# ---------------------------------------------------------------------------
# 30-day integration — mirrors the Week 6 task body
# ---------------------------------------------------------------------------


def test_30day_event_stream_integration(repo, weights) -> None:
    """Fixture 30-day event stream → Moments per producer land + dedup.

    Drives two producers with a daily event stream over 30 days. Each
    producer has a trained feedback weight so the gate is open. The
    test asserts:

    1. Moments from *both* producers end up in the repo.
    2. Identical evidence on later days collapses (no duplicate rows).
    3. No event is lost to a crash (fail-open holds under load).
    """
    _train_to_base(weights, InsightType.CADENCE)
    _train_to_base(weights, InsightType.RELATIONSHIP)

    # Cadence producer: fires once on event 0 with stable evidence. On
    # every subsequent event it re-emits the *same* candidate — we
    # expect all 30 calls to collapse into one row via evidence_hash.
    stable_cadence = _make_moment(
        InsightType.CADENCE,
        ["cad-alice"],
        confidence=0.9,
        insight_text="Alice overdue",
    )
    cadence_p = FakeProducer(
        InsightType.CADENCE,
        emit_factory=lambda event: [stable_cadence],
    )

    # Relationship producer: fires on weekly boundary (day % 7 == 0),
    # each firing carries a fresh evidence id → fresh hash → fresh row.
    def rel_emit(event: Event) -> list[Moment]:
        day = event["day"]
        if day % 7 != 0:
            return []
        return [
            _make_moment(
                InsightType.RELATIONSHIP,
                [f"rel-wk-{day // 7}"],
                confidence=0.9,
                insight_text=f"reciprocity drift week {day // 7}",
            )
        ]

    rel_p = FakeProducer(InsightType.RELATIONSHIP, emit_factory=rel_emit)

    # A producer that crashes on odd days — asserts fail-open under a
    # sustained bad-actor load.
    class FlakyProducer(Producer):
        insight_type = InsightType.TEMPORAL

        async def observe(self, event: Event) -> list[Moment]:
            if event["day"] % 2 == 1:
                raise RuntimeError("intermittent fault")
            return []

    engine = MomentEngine([cadence_p, rel_p, FlakyProducer()], repo, weights)

    # 30-day stream, one event per day.
    for day in range(30):
        event = {
            "id": f"evt-{day}",
            "type": "email.received",
            "timestamp": REF_NOW + day * 86400,
            "day": day,
        }
        asyncio.run(engine.on_event(event))

    # Every producer was offered every event (fail-open).
    assert len(cadence_p.observed) == 30
    assert len(rel_p.observed) == 30

    # Cadence collapsed to a single row (identical evidence across days).
    cadence_rows = [m for m in repo.list_pending() if m.source_insight_type is InsightType.CADENCE]
    assert len(cadence_rows) == 1
    assert cadence_rows[0].evidence == ["cad-alice"]

    # Relationship fired 5x (days 0, 7, 14, 21, 28 out of 30) with distinct
    # evidence → 5 rows.
    rel_rows = [m for m in repo.list_pending() if m.source_insight_type is InsightType.RELATIONSHIP]
    assert len(rel_rows) == 5
    assert {tuple(m.evidence) for m in rel_rows} == {
        ("rel-wk-0",),
        ("rel-wk-1",),
        ("rel-wk-2",),
        ("rel-wk-3",),
        ("rel-wk-4",),
    }

    # Temporal producer crashed on every odd day but emitted nothing on
    # even days — zero rows, but also no stalling.
    tmp_rows = [m for m in repo.list_pending() if m.source_insight_type is InsightType.TEMPORAL]
    assert tmp_rows == []
