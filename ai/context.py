"""Life OS v2 — briefing context assembly.

Fresh port of the 11 useful sections from v1's
``services/ai_engine/context.py``, shaped to the v2 single-database
schema (``storage/schema.py``) and returning a plain ``dict`` instead
of a pre-formatted string.

What changed from v1
--------------------
- **One DB, one connection.** v2 is a single ``lifeos.db`` with all
  thirteen tables co-located. The assembler takes a ``sqlite3.Connection``
  directly; no ``DatabaseManager`` shim, no cross-DB lookups.
- **Dict output.** Callers (briefing synthesis, draft-reply, search)
  compose their own prompts now; this module returns structured data.
- **Moments replace tasks.** ``moments`` is the first-class unit;
  ``completions`` reads from ``moment_state_history`` rather than a
  separate tasks table.
- **Killed insights removed.** Per CEO plan § "Killed Insights", we no
  longer emit mood, decision, expertise, or values sections. They are
  not in the schema and not in the returned dict.

Return shape
------------
:meth:`ContextAssembler.assemble_briefing_context` always returns a
dict with the same 11 keys, regardless of input state::

    {
        "calendar":        list[dict],  # upcoming events, next 7 days
        "moments":         list[dict],  # SUGGESTED Moments
        "unread_messages": list[dict],  # last 12h inbound messages
        "completions":     list[dict],  # DONE Moments today (UTC)
        "predictions":     list[dict],  # scheduled Moments (forward-looking)
        "episodes":        list[dict],  # recent activity, last 24h
        "facts":           list[dict],  # confirmed semantic_facts
        "insights":        list[dict],  # per-insight-type accept rate
        "routines":        list[dict],  # signal_profiles producer=routine
        "habits":          list[dict],  # signal_profiles producer in (cadence, temporal)
        "preferences":     dict,        # preferences key→value
    }

Empty-state contract (per the Week 7 task body): every list/dict
section is present; missing data yields an empty container, never
``None`` and never ``KeyError``.

Fail-open
---------
Each section is computed independently inside a ``try`` block. A query
failure (missing table, corrupt row, clock skew in ``date``) degrades
only that section to empty; the other ten still return. The briefing
pipeline never stalls on context.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "The Moment Primitive", § "Killed Insights".
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` Week 7.
- Schema DDL: :mod:`storage.schema`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from typing import Any

logger = logging.getLogger(__name__)


# Keys the returned dict is contracted to carry — callers may rely on
# the set being exhaustive (len == 11). Kept at module scope so tests
# can assert against it directly.
BRIEFING_SECTIONS: tuple[str, ...] = (
    "calendar",
    "moments",
    "unread_messages",
    "completions",
    "predictions",
    "episodes",
    "facts",
    "insights",
    "routines",
    "habits",
    "preferences",
)


class ContextAssembler:
    """Assemble structured briefing context from the v2 SQLite DB.

    Single-user system: ``user_id`` is accepted for API symmetry with
    the future multi-user scope (Phase 3 deferred) but is not used for
    filtering today. Every row in ``lifeos.db`` belongs to the owner.

    Determinism
    -----------
    ``now_fn`` is injected so tests can pin "today" without touching
    the system clock (stdlib only, matching the v2 convention in
    ``MomentRepository``). Defaults to :func:`datetime.now`.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def assemble_briefing_context(
        self,
        user_id: str,
        date: date_cls,
    ) -> dict[str, Any]:
        """Return the 11-section briefing context for ``user_id`` on ``date``.

        Every key in :data:`BRIEFING_SECTIONS` is guaranteed to be
        present in the returned dict. List sections default to ``[]``
        and ``preferences`` defaults to ``{}`` when the underlying row
        set is empty. Individual section failures degrade to an empty
        value without propagating.
        """
        del user_id  # single-user today; see class docstring

        day_start, day_end = self._utc_day_bounds(date)
        now_ts = int(self._now_fn().timestamp())

        return {
            "calendar": self._safe(self._calendar, day_start),
            "moments": self._safe(self._moments),
            "unread_messages": self._safe(self._unread_messages, now_ts),
            "completions": self._safe(self._completions, day_start, day_end),
            "predictions": self._safe(self._predictions, now_ts),
            "episodes": self._safe(self._episodes, now_ts),
            "facts": self._safe(self._facts),
            "insights": self._safe(self._insights),
            "routines": self._safe(self._routines),
            "habits": self._safe(self._habits),
            "preferences": self._safe_dict(self._preferences),
        }

    # ------------------------------------------------------------------
    # section builders — all fail-open, all return list[dict] | dict
    # ------------------------------------------------------------------
    def _calendar(self, day_start: int) -> list[dict[str, Any]]:
        """Upcoming calendar events, from ``day_start`` through +7 days.

        Events are sourced from the append-only ``events`` table by
        matching the ``calendar.event.*`` type prefix; producers populate
        the ``payload`` JSON with the v1 shape so ``json_extract`` reads
        work without a dedicated calendar table in the v2 schema.
        """
        upper = day_start + 7 * 86400
        rows = self._conn.execute(
            """
            SELECT
                id,
                timestamp,
                json_extract(payload, '$.title')      AS title,
                json_extract(payload, '$.start_time') AS start_time,
                json_extract(payload, '$.end_time')   AS end_time,
                json_extract(payload, '$.location')   AS location,
                json_extract(payload, '$.is_all_day') AS is_all_day
            FROM events
            WHERE type LIKE 'calendar.event.%'
              AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC
            LIMIT 20
            """,
            (day_start, upper),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"] or "",
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "location": r["location"],
                "is_all_day": bool(r["is_all_day"]) if r["is_all_day"] is not None else False,
            }
            for r in rows
        ]

    def _moments(self) -> list[dict[str, Any]]:
        """SUGGESTED Moments ordered by confidence DESC, capped at 20.

        Legacy-migrated rows (``source_insight_type='legacy_task'``)
        are filtered out so the briefing never surfaces half-typed
        Moments, matching ``MomentRepository.list_pending``.
        """
        rows = self._conn.execute(
            """
            SELECT id, insight, proposed_action, source_insight_type,
                   confidence, scheduled_for, expires_at
            FROM moments
            WHERE state = 'suggested'
              AND source_insight_type != 'legacy_task'
            ORDER BY confidence DESC, scheduled_for ASC
            LIMIT 20
            """
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            action_kind = _extract_action_kind(r["proposed_action"])
            out.append(
                {
                    "id": r["id"],
                    "insight": r["insight"],
                    "action_kind": action_kind,
                    "source_insight_type": r["source_insight_type"],
                    "confidence": r["confidence"],
                    "scheduled_for": r["scheduled_for"],
                    "expires_at": r["expires_at"],
                }
            )
        return out

    def _unread_messages(self, now_ts: int) -> list[dict[str, Any]]:
        """Inbound email / chat events from the last 12 hours.

        The v2 schema keeps all ingested messages in ``events`` as
        immutable rows; the 12-hour window matches v1's briefing
        behaviour so callers see the same "recent inbox" horizon.
        """
        cutoff = now_ts - 12 * 3600
        rows = self._conn.execute(
            """
            SELECT
                id,
                type,
                timestamp,
                json_extract(payload, '$.from_address') AS from_address,
                json_extract(payload, '$.subject')      AS subject
            FROM events
            WHERE type IN ('email.received', 'message.received')
              AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 30
            """,
            (cutoff,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["type"],
                "timestamp": r["timestamp"],
                "from_address": r["from_address"],
                "subject": r["subject"] or "",
            }
            for r in rows
        ]

    def _completions(self, day_start: int, day_end: int) -> list[dict[str, Any]]:
        """Moments that transitioned to DONE during the briefing's UTC day.

        Sources the timestamp from ``moment_state_history`` — the
        authoritative transition log — so an accepted Moment that gets
        re-edited does not slip in on ``moments.updated_at`` alone.
        """
        rows = self._conn.execute(
            """
            SELECT m.id, m.insight, m.source_insight_type,
                   msh.ts AS done_at
            FROM moments m
            JOIN moment_state_history msh ON msh.moment_id = m.id
            WHERE m.state = 'done'
              AND msh.to_state = 'done'
              AND msh.ts >= ? AND msh.ts < ?
            ORDER BY msh.ts DESC
            LIMIT 10
            """,
            (day_start, day_end),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "insight": r["insight"],
                "source_insight_type": r["source_insight_type"],
                "done_at": r["done_at"],
            }
            for r in rows
        ]

    def _predictions(self, now_ts: int) -> list[dict[str, Any]]:
        """Scheduled / context-triggered Moments still in play.

        In v2 the "prediction" primitive is subsumed by the Moment;
        forward-looking items are the subset whose ``scheduled_for`` or
        ``context_trigger`` is set and whose state is SUGGESTED /
        SNOOZED. Already-fired Moments (accepted / done / etc.) are
        excluded so the LLM does not re-suggest resolved items.
        """
        horizon = now_ts + 7 * 86400
        rows = self._conn.execute(
            """
            SELECT id, insight, scheduled_for, context_trigger,
                   source_insight_type, confidence
            FROM moments
            WHERE state IN ('suggested', 'snoozed')
              AND source_insight_type != 'legacy_task'
              AND (
                    (scheduled_for IS NOT NULL AND scheduled_for <= ?)
                 OR context_trigger IS NOT NULL
              )
            ORDER BY confidence DESC, scheduled_for ASC
            LIMIT 10
            """,
            (horizon,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "insight": r["insight"],
                "scheduled_for": r["scheduled_for"],
                "context_trigger": r["context_trigger"],
                "source_insight_type": r["source_insight_type"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def _episodes(self, now_ts: int) -> list[dict[str, Any]]:
        """Recent user activity summaries for the last 24 hours.

        v2 does not have a dedicated episodes table (the layered user
        model was cut); we surface recent ``events`` directly so the
        LLM still has narrative material for "yesterday you …" lines.
        Sender / subject fields are optional — ``payload`` is free-form
        JSON and many event types won't have them.
        """
        cutoff = now_ts - 24 * 3600
        rows = self._conn.execute(
            """
            SELECT
                id,
                type,
                source,
                timestamp,
                json_extract(payload, '$.subject')    AS subject,
                json_extract(payload, '$.from_address') AS from_address,
                json_extract(payload, '$.title')      AS title
            FROM events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (cutoff,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "type": r["type"],
                "source": r["source"],
                "timestamp": r["timestamp"],
                "summary": r["subject"] or r["title"] or "",
                "from_address": r["from_address"],
            }
            for r in rows
        ]

    def _facts(self) -> list[dict[str, Any]]:
        """Confirmed semantic facts only.

        The v2 schema keeps ``semantic_facts`` three-state (pending /
        confirmed / denied). Only confirmed facts are safe to surface
        to the LLM — pending facts are hypotheses awaiting user
        confirmation and denied ones were actively rejected.
        """
        rows = self._conn.execute(
            """
            SELECT id, subject, predicate, object, confidence
            FROM semantic_facts
            WHERE status = 'confirmed'
            ORDER BY confidence DESC
            LIMIT 20
            """
        ).fetchall()
        return [
            {
                "id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    def _insights(self) -> list[dict[str, Any]]:
        """Per-insight-type acceptance weights (EWMA).

        ``feedback_weights`` tracks the moving accept-rate per producer;
        surfacing it in the briefing context lets the LLM calibrate
        qualitative language ("your cadence suggestions have been on
        point lately") without needing to re-derive from raw state
        transitions. Only producers with at least one decision
        recorded are included.
        """
        rows = self._conn.execute(
            """
            SELECT insight_type, weight, decision_count
            FROM feedback_weights
            WHERE decision_count > 0
            ORDER BY weight DESC
            """
        ).fetchall()
        return [
            {
                "insight_type": r["insight_type"],
                "weight": r["weight"],
                "decision_count": r["decision_count"],
            }
            for r in rows
        ]

    def _routines(self) -> list[dict[str, Any]]:
        """Detected routines from the routine producer's signal profile.

        The ``signal_profiles`` row is JSON-blob shaped per producer;
        this helper surfaces the raw profile body so the LLM (and tests)
        can see which routines were detected without re-parsing the
        v1 schema.
        """
        return self._signal_profile_rows("routine")

    def _habits(self) -> list[dict[str, Any]]:
        """Repeated behavioural patterns from cadence / temporal producers.

        "Habits" in v2 is the umbrella over the patterns these two
        producers surface — regular reply cadences, chronotype peaks,
        etc. Grouped together here so the briefing has one section for
        "what you tend to do" rather than two overlapping ones.
        """
        return self._signal_profile_rows("cadence") + self._signal_profile_rows("temporal")

    def _preferences(self) -> dict[str, Any]:
        """User preferences as a flat ``key → value`` dict.

        Encrypted values (Fernet creds for connectors) are filtered
        out so they never land in an LLM prompt verbatim — the cloud
        path would otherwise see ciphertext, which is harmless but
        wastes budget. ``value`` is returned as-is (stringly-typed);
        the preferences table is ``TEXT`` for everything.
        """
        rows = self._conn.execute("SELECT key, value, encrypted FROM preferences").fetchall()
        return {r["key"]: r["value"] for r in rows if not r["encrypted"]}

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _signal_profile_rows(self, producer: str) -> list[dict[str, Any]]:
        """Return rows from ``signal_profiles`` for the given producer.

        Used by :meth:`_routines` and :meth:`_habits`. Fails open by
        raising to the calling ``_safe`` wrapper; callers never see
        exceptions from this path.
        """
        rows = self._conn.execute(
            "SELECT key, profile, updated_at FROM signal_profiles WHERE producer = ? ORDER BY updated_at DESC LIMIT 10",
            (producer,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                profile = json.loads(r["profile"])
            except (json.JSONDecodeError, TypeError):
                # Skip malformed rows rather than fail the whole section;
                # the EWMA wiring is resilient to sparse profiles.
                continue
            out.append(
                {
                    "producer": producer,
                    "key": r["key"],
                    "profile": profile,
                    "updated_at": r["updated_at"],
                }
            )
        return out

    @staticmethod
    def _utc_day_bounds(day: date_cls) -> tuple[int, int]:
        """Return ``(start_of_day_utc, start_of_next_day_utc)`` as unix ts."""
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        return int(start.timestamp()), int((start + timedelta(days=1)).timestamp())

    def _safe(self, fn: Callable[..., list[dict[str, Any]]], *args: Any) -> list[dict[str, Any]]:
        """Run ``fn`` and return ``[]`` on any exception (fail-open)."""
        try:
            return fn(*args)
        except Exception as e:
            logger.debug("context: section %s unavailable: %s", fn.__name__, e)
            return []

    def _safe_dict(self, fn: Callable[..., dict[str, Any]]) -> dict[str, Any]:
        """``_safe`` variant for the single dict-shaped section."""
        try:
            return fn()
        except Exception as e:
            logger.debug("context: section %s unavailable: %s", fn.__name__, e)
            return {}


def _extract_action_kind(raw: str | None) -> str | None:
    """Pull the action kind out of the JSON-encoded ``proposed_action``.

    Returns ``None`` when the payload is missing or malformed — the
    briefing LLM can still render the Moment from its ``insight``
    text alone.
    """
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    kind = obj.get("kind") if isinstance(obj, dict) else None
    return kind if isinstance(kind, str) else None


__all__ = ["BRIEFING_SECTIONS", "ContextAssembler"]
