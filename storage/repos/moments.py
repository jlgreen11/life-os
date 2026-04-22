"""SQLite-backed repository for the Moment primitive.

The :class:`MomentRepository` is the single persistence interface for
:class:`core.moment.types.Moment`. All writes run inside a
``BEGIN IMMEDIATE`` transaction so that the ``moments`` row update and
the matching ``moment_state_history`` append happen atomically — the
state machine is authoritative, and an illegal transition (raised by
:func:`core.moment.state.validate_transition`) rolls the whole write
back before either table is touched.

Constructor injection (eng review §1a): every caller passes its own
``sqlite3.Connection``. No module-level connection or globals. The
repository takes ownership of the connection's transaction mode — it
switches ``isolation_level`` to ``None`` so ``BEGIN IMMEDIATE`` can be
issued explicitly. Callers should assume the repository owns the
transaction lifecycle after construction.

Legacy handling
---------------
The v1 → v2 migration inserts ``source_insight_type='legacy_task'``
rows directly via raw SQL. That value is *not* in
:class:`core.moment.types.InsightType` (CEO plan § "The Moment
Primitive → Enums"); the repository transparently filters those rows
out of list results and returns ``None`` from :meth:`get` for them, so
v2 producers and the API never see half-typed Moments.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` § "13-table schema".
- Schema DDL: :mod:`storage.schema`.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from core.moment.state import validate_transition
from core.moment.types import (
    Action,
    ActionKind,
    ContextTrigger,
    InsightType,
    Moment,
    MomentState,
    StateHistoryEntry,
)

_SELECT_COLUMNS = (
    "id, created_at, scheduled_for, expires_at, context_trigger, insight, "
    "evidence, evidence_hash, proposed_action, state, snooze_until, confidence, "
    "feedback_weight, source_insight_type, updated_at"
)


class MomentRepository:
    """Thin persistence façade over the ``moments`` + ``moment_state_history`` tables.

    Every mutating method runs inside an explicit ``BEGIN IMMEDIATE`` /
    ``COMMIT`` block so a concurrent writer cannot observe a partial
    state transition. Readers are not transaction-wrapped — SQLite's
    default snapshot isolation is sufficient for the read-heavy Now/You
    tab queries.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        """Wrap ``conn`` with the repository contract.

        ``now_fn`` is injected for determinism in tests (stdlib only,
        per eng review — no ``freezegun``). Defaults to
        :func:`time.time`. Return value is coerced to ``int`` so
        Moment timestamps match the unix-seconds shape used by the
        schema.
        """
        self._conn = conn
        # Manual transaction management; see module docstring.
        self._conn.isolation_level = None
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> int:
        return int(self._now_fn())

    @staticmethod
    def _serialize_action(action: Action) -> str:
        return json.dumps(
            {"kind": action.kind.value, "params": action.params},
            sort_keys=True,
        )

    @staticmethod
    def _deserialize_action(raw: str) -> Action:
        obj = json.loads(raw)
        return Action(kind=ActionKind(obj["kind"]), params=obj.get("params", {}))

    def _load_history(self, moment_id: str) -> list[StateHistoryEntry]:
        rows = self._conn.execute(
            "SELECT from_state, to_state, ts, annotation FROM moment_state_history "
            "WHERE moment_id=? ORDER BY ts ASC, id ASC",
            (moment_id,),
        ).fetchall()
        history: list[StateHistoryEntry] = []
        for row in rows:
            from_raw = row["from_state"]
            history.append(
                StateHistoryEntry(
                    from_state=MomentState(from_raw) if from_raw else None,
                    to_state=MomentState(row["to_state"]),
                    ts=row["ts"],
                    annotation=row["annotation"],
                )
            )
        return history

    def _hydrate(self, row: sqlite3.Row) -> Moment | None:
        """Return a :class:`Moment` or ``None`` if the row is legacy-only.

        ``source_insight_type='legacy_task'`` rows round-trip through
        raw SQL (the v1 migration) but are not a first-class
        :class:`InsightType`. Returning ``None`` lets list/get callers
        transparently skip them without crashing on the enum coerce.
        """
        try:
            src_type = InsightType(row["source_insight_type"])
        except ValueError:
            return None
        ctx_raw = row["context_trigger"]
        ctx = ContextTrigger(expression=ctx_raw) if ctx_raw is not None else None
        return Moment(
            id=row["id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            insight=row["insight"],
            evidence_hash=row["evidence_hash"],
            proposed_action=self._deserialize_action(row["proposed_action"]),
            source_insight_type=src_type,
            scheduled_for=row["scheduled_for"],
            context_trigger=ctx,
            evidence=json.loads(row["evidence"]),
            state=MomentState(row["state"]),
            state_history=self._load_history(row["id"]),
            snooze_until=row["snooze_until"],
            confidence=row["confidence"],
            feedback_weight=row["feedback_weight"],
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def create(self, moment: Moment) -> str:
        """Insert ``moment``; return existing id on UNIQUE collision.

        Idempotency key is ``(source_insight_type, evidence_hash)`` —
        producers emit the same ``evidence_hash`` for the same
        underlying event set, so replay at-least-once is safe. The
        repository also appends an initial ``None → suggested`` row to
        ``moment_state_history`` so downstream consumers (the /audit
        surface, regression replay) always see creation in the log.

        A fresh ``uuid.uuid4()`` id is generated if the caller left
        ``moment.id`` empty — the producer layer usually supplies a
        stable id, but the migration path deliberately doesn't.
        """
        moment_id = moment.id or str(uuid.uuid4())
        created_at = moment.created_at

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            existing = self._conn.execute(
                "SELECT id FROM moments WHERE source_insight_type=? AND evidence_hash=?",
                (moment.source_insight_type.value, moment.evidence_hash),
            ).fetchone()
            if existing is not None:
                self._conn.execute("COMMIT")
                return existing["id"]

            ctx_raw = moment.context_trigger.expression if moment.context_trigger else None
            self._conn.execute(
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
                    created_at,
                    moment.scheduled_for,
                    moment.expires_at,
                    ctx_raw,
                    moment.insight,
                    json.dumps(moment.evidence),
                    moment.evidence_hash,
                    self._serialize_action(moment.proposed_action),
                    moment.state.value,
                    moment.snooze_until,
                    moment.confidence,
                    moment.feedback_weight,
                    moment.source_insight_type.value,
                    created_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO moment_state_history "
                "(moment_id, from_state, to_state, ts, annotation) "
                "VALUES (?, ?, ?, ?, ?)",
                (moment_id, None, moment.state.value, created_at, "create"),
            )
            self._conn.execute("COMMIT")
            return moment_id
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get(self, moment_id: str) -> Moment | None:
        """Return the Moment, or ``None`` if missing or legacy-only."""
        row = self._conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM moments WHERE id=?",
            (moment_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate(row)

    def transition(
        self,
        moment_id: str,
        new_state: MomentState,
        annotation: str | None = None,
    ) -> Moment:
        """Transition ``moment_id`` to ``new_state`` atomically.

        Runs the whole fetch-validate-update-append sequence inside
        one ``BEGIN IMMEDIATE`` block. On an illegal transition the
        whole transaction rolls back, so the ``moments`` row remains
        in its prior state and no history row is appended.

        Raises
        ------
        KeyError
            If ``moment_id`` does not exist.
        core.moment.state.IllegalTransition
            If the current state cannot transition to ``new_state``.
        """
        now = self._now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT state FROM moments WHERE id=?",
                (moment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"moment {moment_id!r} not found")
            current = MomentState(row["state"])
            validate_transition(current, new_state)
            self._conn.execute(
                "UPDATE moments SET state=?, updated_at=? WHERE id=?",
                (new_state.value, now, moment_id),
            )
            self._conn.execute(
                "INSERT INTO moment_state_history "
                "(moment_id, from_state, to_state, ts, annotation) "
                "VALUES (?, ?, ?, ?, ?)",
                (moment_id, current.value, new_state.value, now, annotation),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

        hydrated = self.get(moment_id)
        if hydrated is None:
            # Should be unreachable — we just updated the row and it's
            # not a legacy_task (transition requires a real InsightType).
            raise RuntimeError(f"moment {moment_id!r} vanished after commit")
        return hydrated

    def list_pending(self, limit: int = 20) -> list[Moment]:
        """Return SUGGESTED Moments ordered by confidence DESC, then scheduled_for ASC.

        Skips legacy-migrated rows (``source_insight_type='legacy_task'``)
        so the Now tab never shows half-typed Moments.
        """
        rows = self._conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM moments WHERE state='suggested' "
            "ORDER BY confidence DESC, scheduled_for ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [m for row in rows if (m := self._hydrate(row)) is not None]

    def list_scheduled(
        self,
        horizon_seconds: int = 86400,
        limit: int = 10,
    ) -> list[Moment]:
        """Return Moments with ``scheduled_for`` inside the horizon window.

        Horizon is measured from the repository's clock (``now_fn``).
        State must be ``suggested`` or ``snoozed`` — these are the two
        states a scheduler would wake up. The upper bound is inclusive;
        there is no lower bound so already-past-due Moments still
        appear (the scheduler's boot recovery catches them).
        """
        upper = self._now() + horizon_seconds
        rows = self._conn.execute(
            f"SELECT {_SELECT_COLUMNS} FROM moments "
            "WHERE state IN ('suggested','snoozed') "
            "AND scheduled_for IS NOT NULL AND scheduled_for <= ? "
            "ORDER BY scheduled_for ASC LIMIT ?",
            (upper, limit),
        ).fetchall()
        return [m for row in rows if (m := self._hydrate(row)) is not None]

    def list_done_today(self, limit: int = 10) -> list[Moment]:
        """Return Moments whose transition into ``done`` happened today (UTC).

        "Today" is the UTC day containing ``now_fn()``. Sources the
        timestamp from ``moment_state_history`` (the authoritative
        transition log), not ``moments.updated_at`` — an accepted
        Moment that gets re-edited would bump updated_at without a
        matching ``done`` transition and would otherwise slip in.
        """
        now_ts = self._now()
        today = datetime.fromtimestamp(now_ts, tz=UTC)
        start_of_today = int(datetime(today.year, today.month, today.day, tzinfo=UTC).timestamp())
        start_of_tomorrow = start_of_today + 86400

        rows = self._conn.execute(
            f"""
            SELECT {", ".join("m." + col for col in _SELECT_COLUMNS.split(", "))}
            FROM moments m
            JOIN moment_state_history msh ON msh.moment_id = m.id
            WHERE m.state = 'done'
              AND msh.to_state = 'done'
              AND msh.ts >= ? AND msh.ts < ?
            ORDER BY msh.ts DESC LIMIT ?
            """,
            (start_of_today, start_of_tomorrow, limit),
        ).fetchall()
        return [m for row in rows if (m := self._hydrate(row)) is not None]


__all__ = ["MomentRepository"]
