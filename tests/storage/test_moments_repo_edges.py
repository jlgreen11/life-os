"""Edge-case tests for :class:`storage.repos.moments.MomentRepository`.

Targets coverage gap #7 in the 2026-04-23 audit — narrow lines left
uncovered by the main moments-repo suite:

- ``create`` rolls back when the INSERT raises (lines 232-234).
- ``transition`` invokes ``conn_cb`` **inside** the transaction (line
  329-330 — the outbox-piggyback path used by Undo).
- ``transition`` / ``snooze`` / ``update_action_params`` each raise
  ``RuntimeError('... vanished after commit')`` when the post-commit
  ``get()`` returns ``None`` (lines 340, 403, 453 — guards against
  concurrent deletes).

The RuntimeError guards are unreachable in normal operation (nothing
deletes from ``moments`` under the repo's lock). We drive them via a
monkeypatched ``get`` that returns ``None`` after the commit; this
keeps the test in-process with no real concurrency.
"""

from __future__ import annotations

import sqlite3
import uuid

import pytest

from core.moment.types import (
    Action,
    ActionKind,
    InsightType,
    Moment,
    MomentState,
)
from storage import schema
from storage.repos.moments import MomentRepository
from storage.repos.outbox import OutboxRepository

REF_NOW = 1_777_204_800


class _Clock:
    def __init__(self) -> None:
        self.t = REF_NOW

    def __call__(self) -> float:
        return self.t


def _apply_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    _apply_schema(c)
    yield c
    c.close()


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def repo(conn, clock):
    return MomentRepository(conn, now_fn=clock)


def _make_moment(
    *,
    evidence_hash: str = "hash-1",
    state: MomentState = MomentState.SUGGESTED,
) -> Moment:
    return Moment(
        id=str(uuid.uuid4()),
        created_at=REF_NOW,
        expires_at=REF_NOW + 72 * 3600,
        insight="hi",
        evidence_hash=evidence_hash,
        proposed_action=Action(kind=ActionKind.NUDGE, params={"contact_id": "c1"}),
        source_insight_type=InsightType.CADENCE,
        scheduled_for=None,
        context_trigger=None,
        evidence=["evt-1"],
        state=state,
    )


# ---------------------------------------------------------------------------
# create rollback
# ---------------------------------------------------------------------------


class _BrokenHistoryConn:
    """Wrap a real sqlite3.Connection; raise on the history INSERT only."""

    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def execute(self, sql: str, params=()):  # type: ignore[no-untyped-def]
        if "moment_state_history" in sql.lower() and sql.lstrip().lower().startswith("insert"):
            raise ValueError("history write boom")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def test_create_rollback_on_history_insert_failure(tmp_path, clock):
    """A failure on the history INSERT must roll the Moment row back.

    Before the fix this would have been silently swallowed — the
    Moment existed without a create entry in the audit log.
    """
    db_path = tmp_path / "moments.db"
    raw = sqlite3.connect(db_path)
    _apply_schema(raw)
    patched = _BrokenHistoryConn(raw)
    repo = MomentRepository(patched, now_fn=clock)  # type: ignore[arg-type]

    m = _make_moment(evidence_hash="rolled-back")
    with pytest.raises(ValueError):
        repo.create(m)
    raw.close()

    # Verify rollback: no moments row, no history row.
    check = sqlite3.connect(db_path)
    n_moments = check.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
    n_history = check.execute("SELECT COUNT(*) FROM moment_state_history").fetchone()[0]
    check.close()
    assert n_moments == 0
    assert n_history == 0


# ---------------------------------------------------------------------------
# transition conn_cb — used by the Undo outbox piggyback
# ---------------------------------------------------------------------------


def test_transition_conn_cb_runs_inside_transaction(repo, conn):
    """``conn_cb`` must run **before** COMMIT — if it raises the whole
    transition rolls back.

    Lock the contract the Undo route depends on: the outbox enqueue /
    cancel runs on the same connection, inside the BEGIN IMMEDIATE.
    """
    outbox = OutboxRepository(conn)
    m = _make_moment(evidence_hash="cb-happy")
    moment_id = repo.create(m)

    callbacks: list[str] = []

    def cb(c: sqlite3.Connection) -> None:
        callbacks.append("ran")
        outbox.enqueue("evt-x", "moment.accepted", {"hi": 1}, conn=c)

    repo.transition(moment_id, MomentState.ACCEPTED, annotation="t", conn_cb=cb)
    assert callbacks == ["ran"]
    # Outbox row visible post-commit.
    count = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    assert count == 1


def test_transition_conn_cb_raising_rolls_back(repo, conn):
    """If the callback raises, the transition undoes itself too."""
    m = _make_moment(evidence_hash="cb-boom")
    moment_id = repo.create(m)

    def bad_cb(_c: sqlite3.Connection) -> None:
        raise RuntimeError("cb exploded")

    with pytest.raises(RuntimeError):
        repo.transition(moment_id, MomentState.ACCEPTED, annotation="t", conn_cb=bad_cb)

    # State should still be SUGGESTED — the transition rolled back.
    row = conn.execute("SELECT state FROM moments WHERE id=?", (moment_id,)).fetchone()
    assert row[0] == MomentState.SUGGESTED.value
    # History should only contain the create row.
    hrows = conn.execute(
        "SELECT to_state FROM moment_state_history WHERE moment_id=? ORDER BY ts",
        (moment_id,),
    ).fetchall()
    assert [r[0] for r in hrows] == [MomentState.SUGGESTED.value]


# ---------------------------------------------------------------------------
# "vanished after commit" guards — monkeypatched get returns None
# ---------------------------------------------------------------------------


def test_transition_raises_when_post_commit_get_is_none(repo, monkeypatch):
    m = _make_moment(evidence_hash="vanished-transition")
    moment_id = repo.create(m)

    monkeypatch.setattr(repo, "get", lambda _mid: None)
    with pytest.raises(RuntimeError, match="vanished after commit"):
        repo.transition(moment_id, MomentState.ACCEPTED)


def test_snooze_raises_when_post_commit_get_is_none(repo, monkeypatch):
    m = _make_moment(evidence_hash="vanished-snooze")
    moment_id = repo.create(m)

    monkeypatch.setattr(repo, "get", lambda _mid: None)
    with pytest.raises(RuntimeError, match="vanished after commit"):
        repo.snooze(moment_id, snooze_until=REF_NOW + 60)


def test_update_action_params_raises_when_post_commit_get_is_none(repo, monkeypatch):
    m = _make_moment(evidence_hash="vanished-edit")
    moment_id = repo.create(m)

    monkeypatch.setattr(repo, "get", lambda _mid: None)
    with pytest.raises(RuntimeError, match="vanished after commit"):
        repo.update_action_params(moment_id, {"contact_id": "c2"})
