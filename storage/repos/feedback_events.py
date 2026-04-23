"""Append-only repository for ``feedback_events``.

The v2 ``feedback_events`` table preserves the record of every user
decision on a notification / predicted action. Rows arrive from two
sources:

- **v1 migration** (``source='v1_migration'``). The v1 → v2 migration
  script ports every row from ``preferences.db.feedback_log`` verbatim,
  minus the dropped ``mood_at_time`` column (CEO plan § "Killed from
  v1: mood inference").
- **v2 native** (``source='v2'``, the default). A future hook on the
  accept / dismiss / undo paths will write here so v2-native decisions
  accrue alongside the migrated history.

Design rationale: ADR
``docs/adr/2026-04-22-feedback-events-disposition.md``.

The table is **append-only**. Like ``events`` and
``moment_state_history``, a correction is a new row, not an UPDATE. The
repository exposes insert + read helpers only. A future retention purge
can mirror :meth:`storage.repos.outbox.OutboxRepository.purge_done_older_than`
when volume justifies it.

Constructor injection per eng review §1a — callers own the
``sqlite3.Connection`` and its lifecycle.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """One hydrated row from ``feedback_events``.

    All columns from the DDL are represented. ``context`` and ``notes``
    are returned as raw strings — the caller decodes JSON when relevant
    (the table stores JSON blobs opaquely for the v1 → v2 port, matching
    how v1 stored them).
    """

    id: str
    ts: int
    action_id: str | None
    action_type: str | None
    feedback_type: str | None
    response_latency_seconds: float | None
    context: str | None
    notes: str | None
    source: str
    created_at: int


class FeedbackEventsRepository:
    """SQLite-backed append-only log of user feedback decisions.

    Every mutating method runs inside an explicit ``BEGIN IMMEDIATE`` /
    ``COMMIT`` block; readers are not transaction-wrapped. Duplicate
    inserts on the primary key raise :class:`sqlite3.IntegrityError`
    rather than silently upserting — callers that re-run the v1
    migration must drop and recreate the target DB (the migration
    script does this via ``FileExistsError`` on the output path).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        """Wrap ``conn`` with the feedback-events contract.

        ``now_fn`` is injected for determinism in tests (stdlib only,
        per eng review — no ``freezegun``). Defaults to
        :func:`time.time`. The return value is coerced to ``int`` so
        timestamps match the unix-seconds shape used by the schema.

        The repository flips ``conn.isolation_level`` to ``None`` so it
        can issue ``BEGIN IMMEDIATE`` explicitly, and sets
        ``row_factory`` to :class:`sqlite3.Row` so hydrate paths can
        index by column name.
        """
        self._conn = conn
        self._conn.isolation_level = None
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time

    def _now(self) -> int:
        return int(self._now_fn())

    @staticmethod
    def _hydrate(row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent(
            id=row["id"],
            ts=row["ts"],
            action_id=row["action_id"],
            action_type=row["action_type"],
            feedback_type=row["feedback_type"],
            response_latency_seconds=row["response_latency_seconds"],
            context=row["context"],
            notes=row["notes"],
            source=row["source"],
            created_at=row["created_at"],
        )

    def append(
        self,
        *,
        id: str,
        ts: int,
        action_id: str | None = None,
        action_type: str | None = None,
        feedback_type: str | None = None,
        response_latency_seconds: float | None = None,
        context: str | None = None,
        notes: str | None = None,
        source: str = "v2",
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Insert a feedback row; ``source`` defaults to ``'v2'``.

        When ``conn`` is passed, the INSERT runs on *that* connection
        without opening a new transaction. This mirrors
        :meth:`OutboxRepository.enqueue` so a caller can piggyback the
        feedback append on an existing ``BEGIN IMMEDIATE`` block — used
        by the v1 migration script, which writes thousands of rows
        under a single transaction.

        Raises
        ------
        sqlite3.IntegrityError
            If ``id`` already exists (PRIMARY KEY violation) or ``source``
            is not one of ``'v1_migration'`` / ``'v2'`` (CHECK violation).
        """
        target = conn if conn is not None else self._conn
        owns_txn = conn is None
        if owns_txn:
            self._conn.execute("BEGIN IMMEDIATE")
        try:
            target.execute(
                """
                INSERT INTO feedback_events (
                    id, ts, action_id, action_type, feedback_type,
                    response_latency_seconds, context, notes, source,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    ts,
                    action_id,
                    action_type,
                    feedback_type,
                    response_latency_seconds,
                    context,
                    notes,
                    source,
                    self._now(),
                ),
            )
            if owns_txn:
                self._conn.execute("COMMIT")
        except Exception:
            if owns_txn:
                self._conn.execute("ROLLBACK")
            raise

    def count(self) -> int:
        """Return the total number of rows in the table."""
        (n,) = self._conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()
        return int(n)

    def count_by_action_and_type(self, action_id: str, feedback_type: str) -> int:
        """Count rows matching an exact ``(action_id, feedback_type)`` pair.

        This is the dismissal-pattern query the v1 reaction-prediction
        engine used to suppress domains the user had repeatedly
        dismissed. Exposed here so a v2 producer can consult the
        migrated history at prediction time.
        """
        (n,) = self._conn.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE action_id=? AND feedback_type=?",
            (action_id, feedback_type),
        ).fetchone()
        return int(n)

    def list_by_action(
        self,
        action_id: str,
        *,
        limit: int | None = None,
    ) -> list[FeedbackEvent]:
        """Return every row for ``action_id``, most recent first.

        ``limit`` bounds the result set; ``None`` returns every row.
        Ordering is by ``ts DESC`` with ``id`` as a stable tiebreaker
        so the result is deterministic across backends.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        sql = (
            "SELECT id, ts, action_id, action_type, feedback_type, "
            "response_latency_seconds, context, notes, source, created_at "
            "FROM feedback_events WHERE action_id=? "
            "ORDER BY ts DESC, id ASC"
        )
        params: tuple[object, ...] = (action_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (action_id, limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._hydrate(r) for r in rows]

    def recent(self, limit: int = 100) -> list[FeedbackEvent]:
        """Return the ``limit`` most recent rows across all action ids.

        Primary use case: operator inspection via a one-off script or
        upcoming admin view. Ordering by ``ts DESC, id ASC`` matches
        :meth:`list_by_action` so the two share a query shape.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = self._conn.execute(
            "SELECT id, ts, action_id, action_type, feedback_type, "
            "response_latency_seconds, context, notes, source, created_at "
            "FROM feedback_events ORDER BY ts DESC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._hydrate(r) for r in rows]


__all__ = ["FeedbackEvent", "FeedbackEventsRepository"]
