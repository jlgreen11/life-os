"""Tests for :class:`storage.repos.moments.MomentRepository`.

Verifies the six public methods (``create``, ``get``, ``transition``,
``list_pending``, ``list_scheduled``, ``list_done_today``) against a
fresh in-memory SQLite instance with the full v2 schema applied.

Covers:
- Create + get round-trip preserves every field.
- Creation appends an initial ``None → suggested`` history row.
- Create is idempotent on ``(source_insight_type, evidence_hash)`` —
  a duplicate create returns the existing id and does not grow history.
- ``transition`` happy path (SUGGESTED → ACCEPTED → DONE) appends
  history rows with correct ``from_state`` / ``to_state`` / ``annotation``.
- ``transition`` illegal-edge raises :class:`IllegalTransition`, the
  stored state is untouched, and no orphan history row was appended
  (transaction rollback).
- ``transition`` on a missing id raises :class:`KeyError`.
- ``list_pending`` orders by confidence DESC then scheduled_for ASC
  and excludes non-SUGGESTED rows.
- ``list_scheduled`` honors the horizon and excludes rows outside the
  ``(suggested, snoozed)`` gate.
- ``list_done_today`` uses the transition timestamp (not
  ``updated_at``) and filters by UTC-day boundaries.
- Legacy ``source_insight_type='legacy_task'`` rows inserted by the
  migration are transparently filtered from list queries and return
  ``None`` from ``get``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime

import pytest

from core.moment.state import IllegalTransition
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

# Fixed reference epoch: 2026-04-22T12:00:00Z.
REF_NOW = 1_777_204_800


class Clock:
    """Mutable stand-in for :func:`time.time` so tests can advance time."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def conn():
    """In-memory SQLite with the full v2 schema and FKs on."""
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
def repo(conn, clock):
    return MomentRepository(conn, now_fn=clock)


def _make_moment(
    *,
    insight_type: InsightType = InsightType.CADENCE,
    evidence_hash: str = "hash-1",
    state: MomentState = MomentState.SUGGESTED,
    scheduled_for: int | None = None,
    confidence: float = 0.5,
    insight: str = "hello",
    evidence: list[str] | None = None,
    context_trigger: ContextTrigger | None = None,
    moment_id: str | None = None,
    created_at: int = REF_NOW,
    expires_at: int = REF_NOW + 3 * 24 * 3600,
) -> Moment:
    """Helper: produce a default-filled Moment for tests."""
    return Moment(
        id=moment_id or str(uuid.uuid4()),
        created_at=created_at,
        expires_at=expires_at,
        insight=insight,
        evidence_hash=evidence_hash,
        proposed_action=Action(kind=ActionKind.NUDGE, params={"contact_id": "c1"}),
        source_insight_type=insight_type,
        scheduled_for=scheduled_for,
        context_trigger=context_trigger,
        evidence=evidence or ["evt-1", "evt-2"],
        state=state,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# create + get
# ---------------------------------------------------------------------------


def test_create_and_get_roundtrip_preserves_all_fields(repo):
    m = _make_moment(
        scheduled_for=REF_NOW + 3600,
        context_trigger=ContextTrigger(expression="calendar:gap>60m"),
        evidence=["evt-a", "evt-b", "evt-c"],
        confidence=0.77,
    )
    returned_id = repo.create(m)
    assert returned_id == m.id

    loaded = repo.get(m.id)
    assert loaded is not None
    assert loaded.id == m.id
    assert loaded.created_at == m.created_at
    assert loaded.expires_at == m.expires_at
    assert loaded.insight == m.insight
    assert loaded.evidence_hash == m.evidence_hash
    assert loaded.evidence == m.evidence
    assert loaded.source_insight_type == m.source_insight_type
    assert loaded.scheduled_for == m.scheduled_for
    assert loaded.context_trigger is not None
    assert loaded.context_trigger.expression == "calendar:gap>60m"
    assert loaded.state == MomentState.SUGGESTED
    assert loaded.confidence == pytest.approx(0.77)
    assert loaded.feedback_weight == pytest.approx(1.0)
    assert loaded.proposed_action.kind == ActionKind.NUDGE
    assert loaded.proposed_action.params == {"contact_id": "c1"}


def test_create_appends_initial_history_entry(repo):
    m = _make_moment()
    repo.create(m)
    loaded = repo.get(m.id)
    assert loaded is not None
    assert len(loaded.state_history) == 1
    first = loaded.state_history[0]
    assert first.from_state is None
    assert first.to_state == MomentState.SUGGESTED
    assert first.ts == REF_NOW
    assert first.annotation == "create"


def test_get_returns_none_for_missing(repo):
    assert repo.get("does-not-exist") is None


def test_create_generates_uuid_when_id_empty(repo):
    m = _make_moment(moment_id="")
    # Overwrite id after construction so the Moment dataclass validation
    # (none) doesn't trip — _make_moment already allowed empty.
    object.__setattr__(m, "id", "")
    returned_id = repo.create(m)
    assert returned_id != ""
    # UUID4 string form is 36 chars: 8-4-4-4-12 + 4 dashes.
    assert len(returned_id) == 36
    assert repo.get(returned_id) is not None


# ---------------------------------------------------------------------------
# create idempotency
# ---------------------------------------------------------------------------


def test_create_is_idempotent_on_source_type_and_evidence_hash(repo):
    first = _make_moment(evidence_hash="shared-hash")
    second = _make_moment(
        evidence_hash="shared-hash",
        insight="DIFFERENT insight text",
        confidence=0.99,
    )
    assert second.id != first.id

    id_a = repo.create(first)
    id_b = repo.create(second)

    assert id_a == first.id
    assert id_b == first.id  # collision → return existing id

    # Only one row, only one history entry.
    rows = repo._conn.execute("SELECT COUNT(*) FROM moments").fetchone()
    assert rows[0] == 1
    history = repo._conn.execute("SELECT COUNT(*) FROM moment_state_history").fetchone()
    assert history[0] == 1

    # The stored Moment is the first one — second write was a no-op.
    loaded = repo.get(id_a)
    assert loaded is not None
    assert loaded.insight == "hello"
    assert loaded.confidence == pytest.approx(0.5)


def test_create_same_hash_different_insight_type_both_land(repo):
    """Uniqueness is composite — same hash across producers is allowed."""
    a = _make_moment(insight_type=InsightType.CADENCE, evidence_hash="h")
    b = _make_moment(insight_type=InsightType.RELATIONSHIP, evidence_hash="h")

    id_a = repo.create(a)
    id_b = repo.create(b)

    assert id_a != id_b
    assert repo._conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


def test_transition_happy_path_suggested_accepted_done(repo, clock):
    m = _make_moment()
    repo.create(m)

    clock.t = REF_NOW + 10
    accepted = repo.transition(m.id, MomentState.ACCEPTED, annotation="user_accept")
    assert accepted.state == MomentState.ACCEPTED
    assert len(accepted.state_history) == 2
    assert accepted.state_history[-1].from_state == MomentState.SUGGESTED
    assert accepted.state_history[-1].to_state == MomentState.ACCEPTED
    assert accepted.state_history[-1].ts == REF_NOW + 10
    assert accepted.state_history[-1].annotation == "user_accept"

    clock.t = REF_NOW + 20
    done = repo.transition(m.id, MomentState.DONE, annotation="action_delivered")
    assert done.state == MomentState.DONE
    assert len(done.state_history) == 3
    assert done.state_history[-1].from_state == MomentState.ACCEPTED
    assert done.state_history[-1].to_state == MomentState.DONE
    assert done.state_history[-1].ts == REF_NOW + 20
    assert done.state_history[-1].annotation == "action_delivered"


def test_transition_illegal_edge_raises_and_rolls_back(repo):
    m = _make_moment()
    repo.create(m)

    # SUGGESTED → DONE is illegal (must go through ACCEPTED).
    with pytest.raises(IllegalTransition):
        repo.transition(m.id, MomentState.DONE)

    # State unchanged.
    loaded = repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.SUGGESTED
    # History did not grow — transaction was fully rolled back.
    assert len(loaded.state_history) == 1


def test_transition_missing_moment_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.transition("ghost-id", MomentState.ACCEPTED)


def test_transition_dismissed_is_terminal(repo):
    m = _make_moment()
    repo.create(m)
    repo.transition(m.id, MomentState.DISMISSED, annotation="not_relevant")

    # Any onward edge from DISMISSED is illegal.
    for target in (
        MomentState.SUGGESTED,
        MomentState.ACCEPTED,
        MomentState.SNOOZED,
        MomentState.DONE,
        MomentState.EXPIRED,
    ):
        with pytest.raises(IllegalTransition):
            repo.transition(m.id, target)


def test_transition_snooze_then_wake_back_to_suggested(repo):
    m = _make_moment()
    repo.create(m)
    repo.transition(m.id, MomentState.SNOOZED, annotation="user_snoozed_3h")
    woken = repo.transition(m.id, MomentState.SUGGESTED, annotation="scheduler_fire")
    assert woken.state == MomentState.SUGGESTED
    last = woken.state_history[-1]
    assert last.from_state == MomentState.SNOOZED
    assert last.to_state == MomentState.SUGGESTED
    assert last.annotation == "scheduler_fire"


# ---------------------------------------------------------------------------
# list_pending
# ---------------------------------------------------------------------------


def test_list_pending_orders_by_confidence_desc_then_scheduled_for_asc(repo):
    low = _make_moment(evidence_hash="low", confidence=0.2, scheduled_for=REF_NOW + 500)
    mid_early = _make_moment(evidence_hash="mid-e", confidence=0.7, scheduled_for=REF_NOW + 100)
    mid_late = _make_moment(evidence_hash="mid-l", confidence=0.7, scheduled_for=REF_NOW + 900)
    high = _make_moment(evidence_hash="high", confidence=0.95, scheduled_for=REF_NOW + 300)

    for m in (low, mid_late, mid_early, high):
        repo.create(m)

    result = repo.list_pending(limit=10)
    assert [r.id for r in result] == [high.id, mid_early.id, mid_late.id, low.id]


def test_list_pending_only_suggested(repo):
    a = _make_moment(evidence_hash="a", confidence=0.8)
    b = _make_moment(evidence_hash="b", confidence=0.7)
    c = _make_moment(evidence_hash="c", confidence=0.6)
    for m in (a, b, c):
        repo.create(m)

    repo.transition(a.id, MomentState.ACCEPTED)
    repo.transition(c.id, MomentState.DISMISSED)

    result = repo.list_pending()
    assert [r.id for r in result] == [b.id]


def test_list_pending_respects_limit(repo):
    for i in range(5):
        repo.create(_make_moment(evidence_hash=f"h-{i}", confidence=i / 10))

    assert len(repo.list_pending(limit=2)) == 2


# ---------------------------------------------------------------------------
# list_scheduled
# ---------------------------------------------------------------------------


def test_list_scheduled_includes_suggested_and_snoozed_within_horizon(repo):
    inside = _make_moment(
        evidence_hash="inside",
        scheduled_for=REF_NOW + 3600,  # 1h ahead
    )
    far = _make_moment(
        evidence_hash="far",
        scheduled_for=REF_NOW + 90_000,  # ~25h ahead
    )
    past_due = _make_moment(
        evidence_hash="past",
        scheduled_for=REF_NOW - 60,  # already overdue
    )
    no_sched = _make_moment(evidence_hash="nosched", scheduled_for=None)

    for m in (inside, far, past_due, no_sched):
        repo.create(m)

    # Snooze one of the in-window ones to verify both states appear.
    repo.transition(inside.id, MomentState.SNOOZED)

    result = repo.list_scheduled(horizon_seconds=86400)
    ids = {r.id for r in result}
    assert inside.id in ids
    assert past_due.id in ids
    assert far.id not in ids
    assert no_sched.id not in ids


def test_list_scheduled_excludes_terminal_states(repo):
    m = _make_moment(scheduled_for=REF_NOW + 60)
    repo.create(m)
    repo.transition(m.id, MomentState.DISMISSED)

    assert repo.list_scheduled() == []


def test_list_scheduled_orders_by_scheduled_for_asc(repo):
    later = _make_moment(evidence_hash="later", scheduled_for=REF_NOW + 7200)
    sooner = _make_moment(evidence_hash="sooner", scheduled_for=REF_NOW + 300)
    repo.create(later)
    repo.create(sooner)

    result = repo.list_scheduled()
    assert [r.id for r in result] == [sooner.id, later.id]


# ---------------------------------------------------------------------------
# list_done_today
# ---------------------------------------------------------------------------


def _start_of_utc_day(ts: int) -> int:
    dt = datetime.fromtimestamp(ts, tz=UTC)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=UTC).timestamp())


def test_list_done_today_filters_by_transition_ts(repo, clock):
    today_start = _start_of_utc_day(REF_NOW)
    yesterday_midday = today_start - 12 * 3600
    today_midday = today_start + 12 * 3600

    done_yesterday = _make_moment(evidence_hash="yest")
    done_today = _make_moment(evidence_hash="today")

    # Create + fully transition one moment yesterday.
    clock.t = yesterday_midday
    repo.create(done_yesterday)
    repo.transition(done_yesterday.id, MomentState.ACCEPTED)
    repo.transition(done_yesterday.id, MomentState.DONE)

    # Another gets done today.
    clock.t = today_midday
    repo.create(done_today)
    repo.transition(done_today.id, MomentState.ACCEPTED)
    repo.transition(done_today.id, MomentState.DONE)

    result = repo.list_done_today()
    assert [r.id for r in result] == [done_today.id]


def test_list_done_today_excludes_non_done_states(repo):
    m = _make_moment()
    repo.create(m)
    repo.transition(m.id, MomentState.ACCEPTED)  # stops at ACCEPTED, not DONE

    assert repo.list_done_today() == []


def test_list_done_today_orders_by_most_recent_first(repo, clock):
    a = _make_moment(evidence_hash="a")
    b = _make_moment(evidence_hash="b")

    today_start = _start_of_utc_day(REF_NOW)
    clock.t = today_start + 3600
    repo.create(a)
    repo.transition(a.id, MomentState.ACCEPTED)
    repo.transition(a.id, MomentState.DONE)

    clock.t = today_start + 7200
    repo.create(b)
    repo.transition(b.id, MomentState.ACCEPTED)
    repo.transition(b.id, MomentState.DONE)

    result = repo.list_done_today()
    assert [r.id for r in result] == [b.id, a.id]


# ---------------------------------------------------------------------------
# legacy_task filtering
# ---------------------------------------------------------------------------


def _insert_legacy_row(conn, *, moment_id: str, hash_val: str) -> None:
    """Raw insert mimicking the v1→v2 migration path."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            INSERT INTO moments (
                id, created_at, scheduled_for, expires_at, context_trigger,
                insight, evidence, evidence_hash, proposed_action, state,
                snooze_until, confidence, feedback_weight, source_insight_type,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                moment_id,
                REF_NOW,
                None,
                REF_NOW + 86400,
                None,
                "legacy task",
                "[]",
                hash_val,
                json.dumps({"kind": "note_observation", "params": {}}),
                "suggested",
                None,
                0.0,
                1.0,
                "legacy_task",
                REF_NOW,
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def test_legacy_task_rows_are_filtered_from_get_and_list(repo, conn):
    _insert_legacy_row(conn, moment_id="legacy-1", hash_val="leg-1")
    real = _make_moment(evidence_hash="real-1", confidence=0.8)
    repo.create(real)

    assert repo.get("legacy-1") is None
    pending = repo.list_pending()
    assert [m.id for m in pending] == [real.id]


# ---------------------------------------------------------------------------
# snooze
# ---------------------------------------------------------------------------


def test_snooze_transitions_state_and_persists_snooze_until(repo, clock):
    m = _make_moment(expires_at=REF_NOW + 24 * 3600)
    repo.create(m)
    clock.t = REF_NOW + 5

    until = REF_NOW + 3600
    snoozed = repo.snooze(m.id, until, annotation="user_picked_1h")
    assert snoozed.state == MomentState.SNOOZED
    assert snoozed.snooze_until == until
    assert snoozed.state_history[-1].from_state == MomentState.SUGGESTED
    assert snoozed.state_history[-1].to_state == MomentState.SNOOZED
    assert snoozed.state_history[-1].annotation == "user_picked_1h"
    assert snoozed.state_history[-1].ts == REF_NOW + 5


def test_snooze_past_expires_at_coerces_to_expired(repo):
    """Per eng plan § 'Snooze semantics'."""
    m = _make_moment(expires_at=REF_NOW + 60)
    repo.create(m)

    out = repo.snooze(m.id, REF_NOW + 999)
    assert out.state == MomentState.EXPIRED
    # snooze_until is not overwritten — the column stays NULL since the
    # row was expired straight out of SUGGESTED.
    assert out.snooze_until is None


def test_snooze_missing_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.snooze("ghost", REF_NOW + 60)


def test_snooze_illegal_from_terminal_state_raises(repo):
    m = _make_moment()
    repo.create(m)
    repo.transition(m.id, MomentState.DISMISSED)

    with pytest.raises(IllegalTransition):
        repo.snooze(m.id, REF_NOW + 60)

    # Terminal state untouched; no snooze_until side-effect either.
    loaded = repo.get(m.id)
    assert loaded is not None
    assert loaded.state == MomentState.DISMISSED
    assert loaded.snooze_until is None


# ---------------------------------------------------------------------------
# update_action_params
# ---------------------------------------------------------------------------


def test_update_action_params_replaces_params_without_moving_state(repo):
    m = _make_moment()
    repo.create(m)

    updated = repo.update_action_params(m.id, {"contact_id": "c2", "extra": "ok"})
    assert updated.state == MomentState.SUGGESTED
    assert updated.proposed_action.kind == ActionKind.NUDGE  # kind preserved
    assert updated.proposed_action.params == {"contact_id": "c2", "extra": "ok"}
    # No history row appended — edit is not a transition.
    assert len(updated.state_history) == 1


def test_update_action_params_missing_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.update_action_params("ghost", {"x": 1})


# ---------------------------------------------------------------------------
# last_transition — Undo route helper
# ---------------------------------------------------------------------------


def test_last_transition_returns_none_for_unknown_id(repo):
    assert repo.last_transition("nope") is None


def test_last_transition_after_create_is_creation_row(repo):
    m = _make_moment()
    repo.create(m)
    last = repo.last_transition(m.id)
    assert last is not None
    assert last.from_state is None
    assert last.to_state == MomentState.SUGGESTED
    assert last.annotation == "create"


def test_last_transition_returns_newest_row(repo, clock):
    m = _make_moment()
    repo.create(m)
    clock.t = REF_NOW + 10
    repo.transition(m.id, MomentState.ACCEPTED, annotation="user accepted")
    last = repo.last_transition(m.id)
    assert last is not None
    assert last.from_state == MomentState.SUGGESTED
    assert last.to_state == MomentState.ACCEPTED
    assert last.annotation == "user accepted"
    assert last.ts == REF_NOW + 10


def test_last_transition_breaks_ties_by_id(repo):
    """Two transitions written in the same epoch second still return the latest deterministically.

    Two history rows can share ``ts`` (they were written inside the same
    second of wall-clock time); the secondary ``id DESC`` tiebreaker on
    the auto-incrementing ``moment_state_history.id`` column ensures the
    most recently inserted row is returned even when the timestamps
    coincide. This matters for the undo route's "what did the user just
    do?" lookup.
    """
    m = _make_moment()
    repo.create(m)
    # Both transitions land at the same ts because the repo's now_fn
    # is fixed; the row inserted second has the higher id.
    repo.transition(m.id, MomentState.ACCEPTED, annotation="first")
    # Now poke a SUGGESTED-back history entry directly to mimic an undo
    # — repo.transition is the only public path, but the new legal edge
    # ACCEPTED → SUGGESTED supports it.
    repo.transition(m.id, MomentState.SUGGESTED, annotation="undo")
    last = repo.last_transition(m.id)
    assert last is not None
    assert last.to_state == MomentState.SUGGESTED
    assert last.annotation == "undo"
