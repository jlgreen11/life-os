"""Tests for :class:`storage.repos.outbox.OutboxRepository`.

Covers the six public methods against a fresh SQLite instance with the
full v2 schema applied:

- enqueue idempotency on ``(event_id, subject)``
- enqueue inside a caller-supplied transaction stays atomic
- claim_batch moves pending → in_progress with FIFO ordering
- concurrent claim_batch (two threads, two connections) — each row
  claimed exactly once
- complete moves in_progress → done
- complete on a non-in_progress row raises
- fail increments retry_count and requeues, then transitions to ``dead``
  once the retry budget of 5 is exhausted
- requeue_in_progress_on_boot flips stuck rows back to pending
- purge_done_older_than honours the retention cutoff
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from storage import schema
from storage.repos.outbox import MAX_RETRIES, OutboxRepository

REF_NOW = 1_777_204_800


class Clock:
    """Mutable stand-in for :func:`time.time` so tests can advance time."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

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
    return Clock()


@pytest.fixture
def repo(conn, clock):
    return OutboxRepository(conn, now_fn=clock)


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_inserts_pending_row(repo, conn):
    oid = repo.enqueue("evt-1", "moment.accepted", {"hello": "world"})
    row = conn.execute("SELECT * FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row is not None
    assert row["state"] == "pending"
    assert row["retry_count"] == 0
    assert row["last_error"] is None
    assert row["claimed_at"] is None


def test_enqueue_is_idempotent_on_event_and_subject(repo, conn):
    first = repo.enqueue("evt-1", "moment.accepted", {"a": 1})
    second = repo.enqueue("evt-1", "moment.accepted", {"a": 2})
    assert first == second
    count = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    assert count == 1


def test_enqueue_different_subjects_produce_distinct_rows(repo, conn):
    a = repo.enqueue("evt-1", "moment.accepted")
    b = repo.enqueue("evt-1", "send_message.v1")
    assert a != b
    count = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
    assert count == 2


def test_enqueue_with_caller_transaction_is_atomic(repo, conn):
    """``enqueue(conn=c)`` must ride an existing BEGIN IMMEDIATE block.

    Simulates the real call site: the Moment transition opens the
    transaction, inserts into ``moments``, then calls ``enqueue`` on
    the same connection so commit rolls both together.
    """
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        oid = repo.enqueue("evt-1", "moment.accepted", {"x": 1}, conn=conn)
        conn.execute("ROLLBACK")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    # Roll-back should have erased the outbox row alongside the Moment.
    row = conn.execute("SELECT * FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# claim_batch
# ---------------------------------------------------------------------------


def test_claim_batch_moves_pending_to_in_progress(repo, conn, clock):
    repo.enqueue("evt-1", "s1")
    repo.enqueue("evt-2", "s1")
    clock.t = REF_NOW + 10
    claimed = repo.claim_batch(limit=10)
    assert len(claimed) == 2
    for entry in claimed:
        assert entry.state == "in_progress"
        assert entry.claimed_at == REF_NOW + 10
    # Row-level check
    states = [r["state"] for r in conn.execute("SELECT state FROM outbox").fetchall()]
    assert states == ["in_progress", "in_progress"]


def test_claim_batch_respects_limit_and_fifo(repo, clock):
    clock.t = REF_NOW
    a = repo.enqueue("evt-a", "s1")
    clock.t = REF_NOW + 5
    b = repo.enqueue("evt-b", "s1")
    clock.t = REF_NOW + 10
    c = repo.enqueue("evt-c", "s1")

    first_batch = repo.claim_batch(limit=2)
    ids = [e.id for e in first_batch]
    assert ids == [a, b]

    remaining = repo.claim_batch(limit=10)
    assert [e.id for e in remaining] == [c]


def test_claim_batch_empty_returns_empty_list(repo):
    assert repo.claim_batch(limit=10) == []


def test_claim_batch_concurrent_no_double_claim(tmp_path):
    """Two threads on two connections must not claim the same row.

    Writes to a temp file DB (not ``:memory:``) so both connections see
    the same store. SQLite's BEGIN IMMEDIATE single-writer lock
    serialises the claim updates.
    """
    db_path = tmp_path / "outbox.db"
    # Seed: create schema and insert pending rows via a setup connection.
    setup = sqlite3.connect(db_path)
    _apply_schema(setup)
    row_count = 50
    with setup:
        for i in range(row_count):
            setup.execute(
                "INSERT INTO outbox (id, event_id, subject, payload, state, "
                "retry_count, last_error, created_at, updated_at, claimed_at) "
                "VALUES (?, ?, 's1', '{}', 'pending', 0, NULL, ?, ?, NULL)",
                (f"id-{i:03d}", f"evt-{i}", REF_NOW + i, REF_NOW + i),
            )
    setup.close()

    # Workers each open their own connection and claim in small batches
    # until the pending pool is empty.
    claims_by_thread: dict[int, list[str]] = {0: [], 1: []}
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(worker_idx: int) -> None:
        try:
            c = sqlite3.connect(db_path, timeout=30.0)
            c.execute("PRAGMA journal_mode=WAL")
            r = OutboxRepository(c)
            barrier.wait()
            while True:
                batch = r.claim_batch(limit=3)
                if not batch:
                    break
                claims_by_thread[worker_idx].extend(e.id for e in batch)
            c.close()
        except BaseException as exc:  # pragma: no cover — fail the test
            errors.append(exc)

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert not errors, errors

    all_claimed = claims_by_thread[0] + claims_by_thread[1]
    # Exactly-once: every pending id is claimed by exactly one thread.
    # A single worker may consume the full pool if the other blocks long
    # on the BEGIN IMMEDIATE lock — that is still correct behaviour; the
    # invariant we actually care about is "no double-claim".
    assert len(all_claimed) == row_count
    assert len(set(all_claimed)) == row_count


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------


def test_complete_moves_in_progress_to_done(repo, conn, clock):
    oid = repo.enqueue("evt-1", "s1")
    repo.claim_batch()
    clock.t = REF_NOW + 100
    repo.complete(oid)
    row = conn.execute("SELECT state, updated_at FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["state"] == "done"
    assert row["updated_at"] == REF_NOW + 100


def test_complete_on_pending_row_raises(repo):
    oid = repo.enqueue("evt-1", "s1")
    with pytest.raises(RuntimeError):
        repo.complete(oid)


def test_complete_on_missing_row_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.complete("nonexistent")


# ---------------------------------------------------------------------------
# fail — retry + dead-letter progression
# ---------------------------------------------------------------------------


def test_fail_increments_retry_and_requeues_below_threshold(repo, conn):
    oid = repo.enqueue("evt-1", "s1")
    repo.claim_batch()
    repo.fail(oid, "boom-1")
    row = conn.execute("SELECT state, retry_count, last_error FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["state"] == "pending"
    assert row["retry_count"] == 1
    assert row["last_error"] == "boom-1"


def test_fail_transitions_to_dead_after_max_retries(repo, conn):
    oid = repo.enqueue("evt-1", "s1")
    for i in range(MAX_RETRIES):
        # Claim (pending → in_progress), then fail.
        repo.claim_batch()
        repo.fail(oid, f"boom-{i}")
    row = conn.execute("SELECT state, retry_count, last_error FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["state"] == "dead"
    assert row["retry_count"] == MAX_RETRIES
    assert row["last_error"] == f"boom-{MAX_RETRIES - 1}"
    # Dead rows are not re-claimed.
    assert repo.claim_batch() == []


def test_fail_on_missing_row_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.fail("nonexistent", "boom")


# ---------------------------------------------------------------------------
# requeue_in_progress_on_boot
# ---------------------------------------------------------------------------


def test_requeue_in_progress_on_boot_flips_stuck_rows(repo, conn):
    a = repo.enqueue("evt-a", "s1")
    b = repo.enqueue("evt-b", "s1")
    repo.claim_batch()  # a, b → in_progress
    repo.complete(a)  # a → done (untouched by boot recovery)
    count = repo.requeue_in_progress_on_boot()
    assert count == 1
    row_a = conn.execute("SELECT state FROM outbox WHERE id=?", (a,)).fetchone()
    row_b = conn.execute("SELECT state, claimed_at FROM outbox WHERE id=?", (b,)).fetchone()
    assert row_a["state"] == "done"
    assert row_b["state"] == "pending"
    assert row_b["claimed_at"] is None


def test_requeue_in_progress_returns_zero_when_nothing_stuck(repo):
    assert repo.requeue_in_progress_on_boot() == 0


# ---------------------------------------------------------------------------
# purge_done_older_than
# ---------------------------------------------------------------------------


def test_purge_done_older_than_deletes_only_old_done_rows(repo, conn, clock):
    old_done = repo.enqueue("evt-old", "s1")
    recent_done = repo.enqueue("evt-recent", "s1")
    pending = repo.enqueue("evt-pending", "s1")
    # Claim + complete both.
    repo.claim_batch(limit=10)
    # The pending one should be moved back so we don't complete it.
    # Simpler: complete all three that were claimed, but we only want
    # two done. Re-enqueue 'pending' after claim.
    # Actually claim_batch claimed all three — roll back the third.
    repo.fail(pending, "reset")  # goes back to pending (retry_count=1)

    # Complete old_done + recent_done
    repo.complete(old_done)
    repo.complete(recent_done)

    # Age `old_done` by rewinding updated_at directly.
    conn.execute(
        "UPDATE outbox SET updated_at=? WHERE id=?",
        (REF_NOW - 31 * 86400, old_done),
    )
    conn.commit()

    # Clock stays at REF_NOW; purge with 30-day window removes only old_done.
    removed = repo.purge_done_older_than(days=30)
    assert removed == 1
    remaining = {r["id"]: r["state"] for r in conn.execute("SELECT id, state FROM outbox").fetchall()}
    assert old_done not in remaining
    assert remaining[recent_done] == "done"
    assert remaining[pending] == "pending"


def test_purge_done_older_than_rejects_negative_days(repo):
    with pytest.raises(ValueError):
        repo.purge_done_older_than(days=-1)


# ---------------------------------------------------------------------------
# not_before — deferred dispatch (Undo grace window)
# ---------------------------------------------------------------------------


def test_enqueue_with_not_before_persists_column(repo, conn):
    """``enqueue(not_before=X)`` stores the value verbatim."""
    oid = repo.enqueue("evt-1", "s1", not_before=REF_NOW + 3)
    row = conn.execute("SELECT not_before FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["not_before"] == REF_NOW + 3


def test_enqueue_without_not_before_defaults_to_null(repo, conn):
    """Omitting ``not_before`` leaves the column NULL — backwards compatible."""
    oid = repo.enqueue("evt-1", "s1")
    row = conn.execute("SELECT not_before FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["not_before"] is None


def test_claim_batch_skips_deferred_rows(repo, clock):
    """A row with ``not_before > now()`` is not claim-eligible yet."""
    clock.t = REF_NOW
    repo.enqueue("evt-deferred", "s1", not_before=REF_NOW + 60)
    assert repo.claim_batch(limit=10) == []


def test_claim_batch_picks_up_row_once_not_before_elapses(repo, clock):
    """Same deferred row becomes eligible once ``now >= not_before``."""
    clock.t = REF_NOW
    oid = repo.enqueue("evt-deferred", "s1", not_before=REF_NOW + 3)
    assert repo.claim_batch(limit=10) == []

    # At the exact boundary the row is claim-eligible (<= semantics).
    clock.t = REF_NOW + 3
    claimed = repo.claim_batch(limit=10)
    assert [e.id for e in claimed] == [oid]
    assert claimed[0].not_before == REF_NOW + 3


def test_claim_batch_mixed_immediate_and_deferred(repo, clock):
    """NULL not_before still claims immediately; deferred stays behind."""
    clock.t = REF_NOW
    a = repo.enqueue("evt-a", "s1")  # immediate
    clock.t = REF_NOW + 1
    repo.enqueue("evt-b", "s1", not_before=REF_NOW + 60)  # deferred
    clock.t = REF_NOW + 2
    c = repo.enqueue("evt-c", "s1")  # immediate

    claimed = repo.claim_batch(limit=10)
    assert sorted(e.id for e in claimed) == sorted([a, c])


def test_cancel_pending_deletes_matching_row(repo, conn):
    """``cancel_pending`` removes a pending row by (event_id, subject)."""
    oid = repo.enqueue("evt-1", "send_message", not_before=REF_NOW + 3)
    assert repo.cancel_pending("evt-1", "send_message") is True
    row = conn.execute("SELECT id FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row is None


def test_cancel_pending_returns_false_when_missing(repo):
    """No matching row → ``False`` (no error)."""
    assert repo.cancel_pending("evt-nope", "s1") is False


def test_cancel_pending_returns_false_for_in_progress_row(repo, conn):
    """An already-claimed row is not cancellable — returns ``False``.

    Matches the design note's § "Worst case" race: claim fired first,
    state-side grace check still let it through; cancel is a no-op.
    """
    oid = repo.enqueue("evt-1", "s1")
    repo.claim_batch()
    assert repo.cancel_pending("evt-1", "s1") is False
    # Row stayed in_progress.
    state = conn.execute("SELECT state FROM outbox WHERE id=?", (oid,)).fetchone()["state"]
    assert state == "in_progress"


def test_cancel_pending_with_caller_transaction(repo, conn):
    """``cancel_pending(conn=c)`` rides an existing BEGIN IMMEDIATE block."""
    oid = repo.enqueue("evt-1", "s1", not_before=REF_NOW + 3)
    conn.execute("BEGIN IMMEDIATE")
    try:
        assert repo.cancel_pending("evt-1", "s1", conn=conn) is True
        conn.execute("ROLLBACK")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    # Rollback should have restored the row.
    row = conn.execute("SELECT id FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row is not None


def test_requeue_in_progress_on_boot_preserves_not_before(repo, conn, clock):
    """Boot recovery flips in_progress → pending but leaves ``not_before`` alone.

    Scenario: process crashed after claim; the row's original grace
    window is still whatever it was. Preserving the column means the
    next claim tick respects the same not_before.
    """
    clock.t = REF_NOW
    oid = repo.enqueue("evt-1", "s1", not_before=REF_NOW + 3)
    # Force-claim by advancing past the grace window.
    clock.t = REF_NOW + 3
    repo.claim_batch()
    # Simulate crash; not_before stays at REF_NOW + 3.
    repo.requeue_in_progress_on_boot()
    row = conn.execute("SELECT state, not_before FROM outbox WHERE id=?", (oid,)).fetchone()
    assert row["state"] == "pending"
    assert row["not_before"] == REF_NOW + 3


def test_boot_recovery_mid_grace_window_dispatches_on_next_claim(tmp_path):
    """Crash during the 3 s grace: row resumes pending; claim after reboot succeeds.

    Mirrors design note § "Decision 3 — boot recovery during the 3 s
    grace window". Uses a file-backed DB because we simulate a process
    restart by closing + reopening the connection.
    """
    db_path = tmp_path / "outbox.db"

    # "Pre-crash" connection: enqueue with a 3 s grace at t=10.
    pre_clock = Clock(10)
    pre_conn = sqlite3.connect(db_path)
    _apply_schema(pre_conn)
    pre_repo = OutboxRepository(pre_conn, now_fn=pre_clock)
    pre_repo.enqueue("evt-1", "send_message", not_before=13)
    # Simulate crash between enqueue and claim.
    pre_conn.close()

    # "Post-restart" connection: 45 s later the claim loop wakes.
    post_clock = Clock(45)
    post_conn = sqlite3.connect(db_path)
    post_repo = OutboxRepository(post_conn, now_fn=post_clock)
    post_repo.requeue_in_progress_on_boot()  # no-op here; exercises the call
    claimed = post_repo.claim_batch(limit=10)
    assert len(claimed) == 1
    assert claimed[0].event_id == "evt-1"
    post_conn.close()


def test_boot_recovery_within_grace_window_defers(tmp_path):
    """Restart still inside the grace window — claim does not fire yet."""
    db_path = tmp_path / "outbox.db"

    pre_clock = Clock(10)
    pre_conn = sqlite3.connect(db_path)
    _apply_schema(pre_conn)
    pre_repo = OutboxRepository(pre_conn, now_fn=pre_clock)
    pre_repo.enqueue("evt-1", "send_message", not_before=60)
    pre_conn.close()

    post_clock = Clock(11)  # 1 s after crash, still pre-grace
    post_conn = sqlite3.connect(db_path)
    post_repo = OutboxRepository(post_conn, now_fn=post_clock)
    assert post_repo.claim_batch(limit=10) == []
    post_conn.close()


def test_purge_done_older_than_ignores_not_before(repo, conn, clock):
    """Purge predicate is (state='done', updated_at<cutoff). not_before is irrelevant."""
    oid = repo.enqueue("evt-1", "s1", not_before=REF_NOW + 999)
    repo.claim_batch()
    repo.complete(oid)
    # Age it past the cutoff.
    conn.execute(
        "UPDATE outbox SET updated_at=? WHERE id=?",
        (REF_NOW - 60 * 86400, oid),
    )
    conn.commit()
    removed = repo.purge_done_older_than(days=30)
    assert removed == 1
