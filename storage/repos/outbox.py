"""Transactional outbox repository.

The outbox pattern decouples "record a state change" from "publish a
side-effecting event" without the complexity of a distributed
transaction. The same SQLite transaction that updates, e.g., a
``moments`` row also appends a row to ``outbox``; a separate worker
later claims pending rows and drives the real-world effect
(send_message, create_calendar_entry, etc.).

Contract (eng review §1b, engineering plan § "Transactional outbox"):

- :meth:`enqueue` is **idempotent** on ``(event_id, subject)``. Duplicate
  calls return the existing row id and do not insert.
- :meth:`enqueue` can run inside an existing transaction when the caller
  passes ``conn=``; this is how the Moment transition + publish stays
  atomic.
- :meth:`enqueue` accepts ``not_before=`` (epoch seconds) to defer
  dispatch past a grace window. NULL means "claim-eligible immediately".
  The accept path uses this for the 3 s Undo window per design note
  ``docs/plans/2026-04-22-undo-grace.md`` § "Decision 2".
- :meth:`claim_batch` uses ``BEGIN IMMEDIATE`` so two concurrent workers
  cannot claim the same row — SQLite's single-writer lock serialises the
  pending → in_progress update. Rows whose ``not_before > now()`` are
  silently skipped and picked up on a later tick.
- :meth:`complete` moves in_progress → done and stamps ``updated_at``.
- :meth:`fail` increments ``retry_count``, stores ``last_error``, and
  either requeues (state → pending) or dead-letters (state → dead) when
  the retry budget of 5 attempts is exhausted.
- :meth:`cancel_pending` deletes a pending-only row by
  ``(event_id, subject)``. Used by the Undo route to tear down the
  deferred dispatch inside the same transaction as the state reversal.
  Returns ``False`` for ``in_progress`` / terminal rows or missing keys.
- :meth:`requeue_in_progress_on_boot` covers the "worker died mid-delivery"
  case: any ``in_progress`` row is flipped back to ``pending`` at boot.
  ``not_before`` is intentionally preserved — a row mid-grace when the
  process died resumes its grace window on the next boot tick.
- :meth:`purge_done_older_than` is the daily retention job for rows in
  terminal ``done`` state.

Durability assumes SQLite is opened with ``journal_mode=WAL`` and
``synchronous=NORMAL`` (set by :class:`storage.database.DatabaseManager`).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_RETRIES = 5


@dataclass(frozen=True, slots=True)
class OutboxEntry:
    """One row claimed from the outbox, as seen by a worker.

    Returned by :meth:`OutboxRepository.claim_batch`; the worker should
    complete or fail each entry via its ``id``. ``not_before`` carries
    the grace-window stamp so downstream observability can distinguish
    "deferred-then-claimed" rows from immediate claims.
    """

    id: str
    event_id: str
    subject: str
    payload: dict[str, Any]
    state: str
    retry_count: int
    last_error: str | None
    created_at: int
    updated_at: int
    claimed_at: int | None
    not_before: int | None


class OutboxRepository:
    """SQLite-backed transactional outbox.

    Constructor injection per eng review §1a — callers own the
    ``sqlite3.Connection`` and its lifecycle. The repository flips
    ``isolation_level`` to ``None`` so it can issue ``BEGIN IMMEDIATE``
    explicitly.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        """Wrap ``conn`` with the outbox contract.

        ``now_fn`` is injected for determinism in tests (stdlib only,
        per eng review — no ``freezegun``). Defaults to
        :func:`time.time`. Return value is coerced to ``int`` so outbox
        timestamps match the unix-seconds shape used by the schema.

        The repository flips ``conn.isolation_level`` to ``None`` so it
        can issue ``BEGIN IMMEDIATE`` explicitly, and sets
        ``row_factory`` to :class:`sqlite3.Row` so hydrate paths can
        index by column name. Callers should assume the repository owns
        the transaction lifecycle on this connection after construction.
        """
        self._conn = conn
        self._conn.isolation_level = None
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time

    def _now(self) -> int:
        return int(self._now_fn())

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> OutboxEntry:
        return OutboxEntry(
            id=row["id"],
            event_id=row["event_id"],
            subject=row["subject"],
            payload=json.loads(row["payload"]),
            state=row["state"],
            retry_count=row["retry_count"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            claimed_at=row["claimed_at"],
            not_before=row["not_before"],
        )

    def enqueue(
        self,
        event_id: str,
        subject: str,
        payload: dict[str, Any] | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        not_before: int | None = None,
    ) -> str:
        """Insert a pending outbox row; return existing id on duplicate.

        Idempotent on ``(event_id, subject)`` — the caller can safely
        replay the same logical enqueue without duplicating the effect.

        When ``conn`` is passed, the INSERT runs on *that* connection
        without opening a new transaction. This is how a caller can
        atomically bundle "transition the Moment" and "enqueue the
        publish" in a single ``BEGIN IMMEDIATE`` block: the Moment
        transition owns the transaction; ``enqueue(..., conn=c)`` just
        piggybacks on it. When ``conn`` is ``None`` the repository
        manages its own short transaction on ``self._conn``.

        ``not_before`` (epoch seconds) defers dispatch — :meth:`claim_batch`
        skips rows whose ``not_before > now()`` until the window elapses.
        ``None`` (the default) is "claim-eligible immediately", matching
        pre-grace-window producer behaviour.
        """
        target = conn if conn is not None else self._conn
        now = self._now()
        outbox_id = str(uuid.uuid4())
        payload_json = json.dumps(payload or {}, sort_keys=True)

        owns_txn = conn is None
        if owns_txn:
            self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = target.execute(
                "SELECT id FROM outbox WHERE event_id=? AND subject=?",
                (event_id, subject),
            ).fetchone()
            if existing is not None:
                if owns_txn:
                    self._conn.execute("COMMIT")
                return str(existing["id"])

            target.execute(
                """
                INSERT INTO outbox (
                    id, event_id, subject, payload, state, retry_count,
                    last_error, created_at, updated_at, claimed_at, not_before
                ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL, ?)
                """,
                (outbox_id, event_id, subject, payload_json, now, now, not_before),
            )
            if owns_txn:
                self._conn.execute("COMMIT")
            return outbox_id
        except Exception:
            if owns_txn:
                self._conn.execute("ROLLBACK")
            raise

    def claim_batch(self, limit: int = 10) -> list[OutboxEntry]:
        """Atomically move up to ``limit`` claim-eligible rows into in_progress.

        Uses ``BEGIN IMMEDIATE`` so concurrent callers serialise on the
        SQLite writer lock. Ordering is by ``created_at ASC`` (FIFO) —
        older rows are claimed first. ``claimed_at`` is stamped so that
        /health can surface stuck workers.

        A row is claim-eligible iff ``state='pending'`` **and**
        (``not_before IS NULL`` or ``not_before <= now()``). Deferred
        rows (grace-window Undo) silently skip this tick and come back on
        a later one once their ``not_before`` elapses.
        """
        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._conn.execute(
                "SELECT id FROM outbox "
                "WHERE state='pending' AND (not_before IS NULL OR not_before <= ?) "
                "ORDER BY created_at ASC, id ASC LIMIT ?",
                (now, limit),
            ).fetchall()
            if not rows:
                self._conn.execute("COMMIT")
                return []
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"UPDATE outbox SET state='in_progress', claimed_at=?, updated_at=? WHERE id IN ({placeholders})",
                (now, now, *ids),
            )
            claimed = self._conn.execute(
                "SELECT id, event_id, subject, payload, state, retry_count, "
                "last_error, created_at, updated_at, claimed_at, not_before "
                f"FROM outbox WHERE id IN ({placeholders}) "
                "ORDER BY created_at ASC, id ASC",
                ids,
            ).fetchall()
            self._conn.execute("COMMIT")
            return [self._hydrate(r) for r in claimed]
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def complete(self, outbox_id: str) -> None:
        """Mark an in_progress row done; stamp ``updated_at`` as delivered.

        Raises
        ------
        KeyError
            If ``outbox_id`` does not exist.
        RuntimeError
            If the row is not in ``in_progress``; completing a row that
            was never claimed is a programming error, not a transient
            race.
        """
        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT state FROM outbox WHERE id=?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError(f"outbox row {outbox_id!r} not found")
            if row["state"] != "in_progress":
                raise RuntimeError(f"outbox row {outbox_id!r} not in_progress (state={row['state']!r})")
            self._conn.execute(
                "UPDATE outbox SET state='done', updated_at=? WHERE id=?",
                (now, outbox_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def fail(self, outbox_id: str, error_msg: str) -> None:
        """Record a delivery failure; requeue or dead-letter.

        Increments ``retry_count`` and stores ``error_msg``. If the new
        retry count is below :data:`MAX_RETRIES` (5), the row goes back
        to ``pending`` for another worker to pick up. If it reaches 5,
        the row transitions to terminal ``dead`` and surfaces via the
        health endpoint.

        Raises
        ------
        KeyError
            If ``outbox_id`` does not exist.
        """
        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT retry_count FROM outbox WHERE id=?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError(f"outbox row {outbox_id!r} not found")
            new_count = row["retry_count"] + 1
            new_state = "dead" if new_count >= MAX_RETRIES else "pending"
            self._conn.execute(
                "UPDATE outbox SET state=?, retry_count=?, last_error=?, updated_at=? WHERE id=?",
                (new_state, new_count, error_msg, now, outbox_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def cancel_pending(
        self,
        event_id: str,
        subject: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Delete a pending-only outbox row by ``(event_id, subject)``.

        Used by the Undo route to tear down a deferred dispatch row inside
        the same transaction as the state reversal. Returns ``True`` if a
        row was deleted, ``False`` if the row is already ``in_progress`` /
        terminal / missing.

        The ``UNIQUE (event_id, subject)`` constraint guarantees at most
        one match. The delete is restricted to ``state='pending'`` so a
        race with the claim loop is safe: an already-claimed row falls
        through with ``False`` and the route returns 410-equivalent
        behaviour (the state-side grace check in the route layer fires
        first, but this second layer protects against clock skew).

        When ``conn`` is passed, the DELETE rides the caller's existing
        transaction — mirrors :meth:`enqueue` so Undo atomicity is
        preserved.
        """
        target = conn if conn is not None else self._conn
        owns_txn = conn is None
        if owns_txn:
            self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = target.execute(
                "DELETE FROM outbox WHERE event_id=? AND subject=? AND state='pending'",
                (event_id, subject),
            )
            deleted = cursor.rowcount > 0
            if owns_txn:
                self._conn.execute("COMMIT")
            return deleted
        except Exception:
            if owns_txn:
                self._conn.execute("ROLLBACK")
            raise

    def requeue_in_progress_on_boot(self) -> int:
        """Flip any in_progress rows back to pending; return row count.

        Covers the crash-recovery case: if the worker process died
        between ``claim_batch`` and ``complete``/``fail``, the row is
        stuck in ``in_progress`` forever. Called once at service boot,
        before the claim loop starts.
        """
        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "UPDATE outbox SET state='pending', claimed_at=NULL, updated_at=? WHERE state='in_progress'",
                (now,),
            )
            count = cursor.rowcount
            self._conn.execute("COMMIT")
            return count
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def purge_done_older_than(self, days: int = 30) -> int:
        """Delete ``done`` rows whose ``updated_at`` is older than ``days``.

        Bulk delete for the daily maintenance cron. Returns the number
        of rows removed so the cron can record retention telemetry.
        """
        if days < 0:
            raise ValueError("days must be non-negative")
        cutoff = self._now() - days * 86400
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "DELETE FROM outbox WHERE state='done' AND updated_at < ?",
                (cutoff,),
            )
            count = cursor.rowcount
            self._conn.execute("COMMIT")
            return count
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


__all__ = ["MAX_RETRIES", "OutboxEntry", "OutboxRepository"]
