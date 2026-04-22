"""Read-only repository for the You + People API payloads.

The You tab, People tab, and per-contact dossier are all views over the
same underlying signal-profile + entities data. This repository owns the
queries that stitch them together so both ``api.routes.you`` and
``api.routes.people`` can share one thin façade rather than hand-rolling
duplicate SQL.

Data sources (all already populated by producers / signal extractors):

- ``signal_profiles`` — per-producer JSON blobs keyed by ``contact_id``
  (cadence, relationship, comm_template) or a stable routine/temporal
  key. See :mod:`producers.cadence` etc. for the shapes.
- ``entities`` — canonical contact roster (``kind='contact'``).
- ``events`` — raw event log; used for the per-contact sparkline and
  the "interactions observed" counter on the You header.
- ``feedback_weights`` — EWMA per-insight-type; averaged into the You
  header's ``confidence_pct`` so users see "the system's self-reported
  hit rate" without exposing the raw EWMA.

Read-only
---------
No writes. The route layer only ever GETs from this repository; the
producers and signal extractors own the write path. There is therefore
no transaction wrapping — SQLite's default snapshot is sufficient for
the read-heavy list endpoints.

Fail-open
---------
Individual producer sections (drifting / routines / personas / …) are
assembled independently. A malformed profile row is logged and
skipped, matching the v1 convention preserved across v2 producers — a
single bad row must not tank the whole payload.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "The You tab", § "The People tab".
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` Week 8
  § "You/People payloads".
- Sibling façade: :mod:`ai.context` (briefing context assembly).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# Cadence-drift threshold: same 1.3x multiplier the cadence producer uses,
# kept in sync so "drifting" on the You/People tabs matches the set the
# producer would emit Moments for.
DRIFT_FACTOR = 1.3

# Minimum inbound observations before a cadence contact is eligible for the
# "drifting" list. Mirrors ``producers.cadence.MIN_HISTORY_COUNT``.
MIN_CADENCE_COUNT = 5

SECONDS_PER_DAY = 86400

# Default sparkline horizon for the dossier (days).
DEFAULT_SPARKLINE_DAYS = 14

# Default/max page size for ``GET /api/people``. Matches the route handler;
# duplicated here so callers that use the repo directly get the same cap.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Seconds in 30 days — used to convert "observed days" to months on the You
# header. An integer division by this value yields a coarse month count that
# lines up with user expectations ("3 months" rather than "89 days").
SECONDS_PER_MONTH = 30 * SECONDS_PER_DAY


class PeopleRepository:
    """Thin read façade over signal_profiles + entities + events.

    ``now_fn`` is injected so tests can pin "today" without touching
    the system clock — matches the pattern used by
    :class:`storage.repos.MomentRepository`.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._now_fn: Callable[[], float] = now_fn or time.time

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_you(self) -> dict[str, Any]:
        """Return the self-portrait payload for ``GET /api/you``.

        Every list section is guaranteed present (empty when the
        underlying producers have nothing to report). Header counters
        default to 0 so empty installs still round-trip.
        """
        now_ts = int(self._now_fn())
        interactions, observed_months = self._observed_stats(now_ts)
        return {
            "observed_months": observed_months,
            "interactions_count": interactions,
            "confidence_pct": self._confidence_pct(),
            "when_at_best": self._when_at_best(),
            "how_you_write": self._personas(),
            "your_routines": self._routines(),
            "drifting": self._drifting(now_ts=now_ts),
        }

    def list_people(
        self,
        query: str | None,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        """Return the People-tab payload: YOU pinned on top plus two sub-lists.

        ``query`` is a case-insensitive substring match against contact
        name or id. ``page`` is 1-indexed. The two sub-lists
        (``needs_attention`` / ``active_this_week``) are mutually
        exclusive — a drifting contact is never also "active this week",
        and vice versa. ``total`` reflects the full post-filter roster;
        ``you`` is always the current self-portrait.
        """
        now_ts = int(self._now_fn())
        summaries = self._contact_summaries(now_ts=now_ts, query=query)
        total = len(summaries)

        start = max(0, (page - 1) * page_size)
        end = start + page_size
        page_rows = summaries[start:end]

        week_cutoff = now_ts - 7 * SECONDS_PER_DAY
        needs_attention: list[dict[str, Any]] = []
        active_this_week: list[dict[str, Any]] = []
        for summary in page_rows:
            if summary["needs_attention"]:
                needs_attention.append(_strip_internal(summary))
            elif summary["last_contact_ts"] is not None and summary["last_contact_ts"] >= week_cutoff:
                active_this_week.append(_strip_internal(summary))

        return {
            "you": self.get_you(),
            "needs_attention": needs_attention,
            "active_this_week": active_this_week,
            "total": total,
            "query": query,
        }

    def get_dossier(self, contact_id: str) -> dict[str, Any] | None:
        """Return the per-contact dossier or ``None`` if unknown.

        A contact is "known" when it has a row in ``entities`` **or** in
        any of the per-contact signal profile namespaces — the entities
        table is the canonical roster, but a producer may have
        populated a profile before the entity landed (e.g. a fresh
        inbound from a contact the resolver hasn't catalogued yet).
        """
        now_ts = int(self._now_fn())
        names = self._contact_names()
        cadence = self._profile_by_key("cadence").get(contact_id)
        relationship = self._profile_by_key("relationship").get(contact_id)
        template = self._profile_by_key("comm_template").get(contact_id)
        topics = self._profile_by_key("topic").get(contact_id, {})

        if contact_id not in names and not any((cadence, relationship, template)):
            return None

        name = str(
            _first_non_empty(
                (cadence or {}).get("contact_name"),
                (relationship or {}).get("contact_name"),
                (template or {}).get("contact_name"),
                names.get(contact_id),
                contact_id,
            )
        )

        last_ts = _maybe_int((cadence or {}).get("last_inbound_ts"))
        cadence_days = _maybe_positive_int((cadence or {}).get("expected_cadence_days"))

        comm_template = None
        if template:
            raw_template = _first_non_empty(
                template.get("template_body"),
                template.get("draft_template"),
                template.get("template_style"),
            )
            if raw_template is not None:
                comm_template = str(raw_template)

        recent_topics: list[str] = []
        raw_topics = topics.get("recent_topics") if isinstance(topics, dict) else None
        if isinstance(raw_topics, list):
            recent_topics = [str(t) for t in raw_topics if t][:10]

        return {
            "contact_id": contact_id,
            "name": name,
            "last_contact_ts": last_ts,
            "usual_cadence_days": cadence_days,
            "comm_template": comm_template,
            "cadence_sparkline": self._sparkline(contact_id, now_ts=now_ts),
            "recent_topics": recent_topics,
            "predicted_next": _predicted_next(last_ts, cadence_days, now_ts),
        }

    # ------------------------------------------------------------------
    # section builders
    # ------------------------------------------------------------------
    def _observed_stats(self, now_ts: int) -> tuple[int, int]:
        """Return ``(interactions_count, observed_months)`` from ``events``.

        Interactions = total event count. Observed months = whole months
        between the earliest event and now. Empty DB yields ``(0, 0)``.
        """
        try:
            row = self._conn.execute("SELECT COUNT(*) AS n, MIN(timestamp) AS first_ts FROM events").fetchone()
        except sqlite3.DatabaseError as exc:
            logger.debug("people: observed_stats unavailable: %s", exc)
            return (0, 0)
        interactions = int(row["n"] or 0) if row else 0
        first_ts = row["first_ts"] if row else None
        if not first_ts:
            return (interactions, 0)
        months = max(0, (now_ts - int(first_ts)) // SECONDS_PER_MONTH)
        return (interactions, int(months))

    def _confidence_pct(self) -> int:
        """Average EWMA weight across producers that have >=1 decision, as 0-100."""
        try:
            row = self._conn.execute(
                "SELECT AVG(weight) AS avg_w FROM feedback_weights WHERE decision_count > 0"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            logger.debug("people: confidence_pct unavailable: %s", exc)
            return 0
        avg = row["avg_w"] if row else None
        if avg is None:
            return 0
        return max(0, min(100, round(float(avg) * 100)))

    def _when_at_best(self) -> list[str]:
        """Top focus windows from temporal profiles, rendered as labels."""
        out: list[str] = []
        for _, profile, _ in self._profiles("temporal"):
            windows = profile.get("focus_windows") or []
            if not isinstance(windows, list):
                continue
            for win in windows:
                if not isinstance(win, dict):
                    continue
                try:
                    start = int(win["start_hour"])
                    end = int(win["end_hour"])
                except (KeyError, TypeError, ValueError):
                    continue
                desc = str(win.get("description") or "").strip()
                label = f"{start:02d}:00-{end:02d}:00"
                if desc:
                    label = f"{label} - {desc}"
                out.append(label)
        return out[:3]

    def _personas(self) -> list[dict[str, Any]]:
        """Per-audience writing summaries from comm_template profiles."""
        names = self._contact_names()
        out: list[dict[str, Any]] = []
        for key, profile, _ in self._profiles("comm_template"):
            audience = str(_first_non_empty(profile.get("contact_name"), names.get(key), key))
            tone = str(_first_non_empty(profile.get("template_style"), profile.get("tone"), "neutral"))
            try:
                formality = float(profile.get("formality") or 0.0)
            except (TypeError, ValueError):
                formality = 0.0
            sample_size = _maybe_int(profile.get("sample_size")) or 0
            if sample_size == 0:
                evts = profile.get("last_event_ids") or []
                sample_size = len(evts) if isinstance(evts, list) else 0
            out.append(
                {
                    "audience": audience,
                    "tone": tone,
                    "formality": formality,
                    "sample_size": sample_size,
                }
            )
        # Cap at 6 per DESIGN.md "3-6 per-audience summaries" — most
        # recently updated first so the list stays fresh.
        return out[:6]

    def _routines(self) -> list[dict[str, Any]]:
        """Detected routines from the routine producer."""
        out: list[dict[str, Any]] = []
        for key, profile, _ in self._profiles("routine"):
            desc = str(profile.get("description") or "").strip()
            if not desc:
                continue
            occurrences = profile.get("last_occurrences") or []
            try:
                consistency = float(profile.get("consistency") or 0.0)
            except (TypeError, ValueError):
                consistency = 0.0
            sample_size = len(occurrences) if isinstance(occurrences, list) else 0
            out.append(
                {
                    "name": key,
                    "detected": True,
                    "description": desc,
                    "confidence": max(0.0, min(1.0, consistency)),
                    "sample_size": sample_size,
                }
            )
        return out

    def _drifting(self, *, now_ts: int) -> list[dict[str, Any]]:
        """Contacts whose cadence has slipped past the drift threshold."""
        names = self._contact_names()
        drifting: list[dict[str, Any]] = []
        for key, profile, _ in self._profiles("cadence"):
            gap = _drift_gap(profile, now_ts=now_ts)
            if gap is None:
                continue
            days_since, expected = gap
            drifting.append(
                {
                    "contact_id": key,
                    "name": str(_first_non_empty(profile.get("contact_name"), names.get(key), key)),
                    "days_since_last": round(days_since),
                    "usual_cadence_days": max(1, round(expected)),
                }
            )
        drifting.sort(key=lambda d: d["days_since_last"], reverse=True)
        return drifting

    def _contact_summaries(
        self,
        *,
        now_ts: int,
        query: str | None,
    ) -> list[dict[str, Any]]:
        """Roster of every known contact, sorted for deterministic pagination.

        Union of the entities roster and the cadence namespace so a
        contact that has a cadence profile but no entity row still
        appears. Drifting contacts are sorted first, then most-recent
        contact descending, then name ascending — this keeps the
        ``needs_attention`` section at the top of page 1.
        """
        names = self._contact_names()
        cadence_map = self._profile_by_key("cadence")
        roster_ids = set(names.keys()) | set(cadence_map.keys())

        q = (query or "").strip().lower()

        summaries: list[dict[str, Any]] = []
        for cid in roster_ids:
            profile = cadence_map.get(cid, {})
            name = str(_first_non_empty(profile.get("contact_name"), names.get(cid), cid))
            if q and q not in name.lower() and q not in cid.lower():
                continue
            last_ts = _maybe_int(profile.get("last_inbound_ts"))
            expected = _maybe_positive_int(profile.get("expected_cadence_days"))
            deviation: int | None = None
            if last_ts is not None and expected is not None:
                days_since = (now_ts - last_ts) / SECONDS_PER_DAY
                deviation = round(days_since - expected)
            needs_attention = _drift_gap(profile, now_ts=now_ts) is not None
            summaries.append(
                {
                    "contact_id": cid,
                    "name": name,
                    "last_contact_ts": last_ts,
                    "cadence_deviation_days": deviation,
                    "needs_attention": needs_attention,
                    "_sort_last_ts": last_ts or 0,
                }
            )

        summaries.sort(key=lambda s: (not s["needs_attention"], -s["_sort_last_ts"], s["name"]))
        return summaries

    def _sparkline(self, contact_id: str, *, now_ts: int) -> list[int]:
        """Per-day event counts for the last :data:`DEFAULT_SPARKLINE_DAYS`.

        Index 0 is the oldest day in the window, the last index is today.
        Events are attributed to a contact via ``payload.contact_id``
        (the v2 normalisation contract). An event without a resolvable
        contact_id is silently skipped.
        """
        cutoff = now_ts - DEFAULT_SPARKLINE_DAYS * SECONDS_PER_DAY
        try:
            rows = self._conn.execute(
                """
                SELECT timestamp FROM events
                WHERE timestamp >= ?
                  AND json_extract(payload, '$.contact_id') = ?
                """,
                (cutoff, contact_id),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.debug("people: sparkline unavailable for %s: %s", contact_id, exc)
            return [0] * DEFAULT_SPARKLINE_DAYS
        sparkline = [0] * DEFAULT_SPARKLINE_DAYS
        for row in rows:
            try:
                ts = int(row["timestamp"])
            except (TypeError, ValueError):
                continue
            days_ago = (now_ts - ts) // SECONDS_PER_DAY
            if 0 <= days_ago < DEFAULT_SPARKLINE_DAYS:
                idx = DEFAULT_SPARKLINE_DAYS - 1 - days_ago
                sparkline[idx] += 1
        return sparkline

    # ------------------------------------------------------------------
    # low-level readers
    # ------------------------------------------------------------------
    def _profiles(self, producer: str) -> list[tuple[str, dict[str, Any], int]]:
        """Return ``(key, profile_dict, updated_at)`` triples for ``producer``.

        Malformed rows (non-JSON, or JSON that isn't an object) are
        skipped and logged at debug level. Returning a materialised list
        lets the caller close the cursor before per-row work runs.
        """
        try:
            rows = self._conn.execute(
                "SELECT key, profile, updated_at FROM signal_profiles WHERE producer = ? ORDER BY updated_at DESC",
                (producer,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.debug("people: profiles for %s unavailable: %s", producer, exc)
            return []
        out: list[tuple[str, dict[str, Any], int]] = []
        for r in rows:
            try:
                profile = json.loads(r["profile"])
            except (TypeError, ValueError):
                logger.debug("people: profile %s/%r is not valid JSON", producer, r["key"])
                continue
            if not isinstance(profile, dict):
                continue
            out.append((r["key"], profile, int(r["updated_at"] or 0)))
        return out

    def _profile_by_key(self, producer: str) -> dict[str, dict[str, Any]]:
        """Convenience: ``{key: profile_dict}`` for a single producer."""
        return {key: profile for key, profile, _ in self._profiles(producer)}

    def _contact_names(self) -> dict[str, str]:
        """``contact_id → name`` from the entities table (``kind='contact'``)."""
        try:
            rows = self._conn.execute("SELECT id, name FROM entities WHERE kind = 'contact'").fetchall()
        except sqlite3.DatabaseError as exc:
            logger.debug("people: contact_names unavailable: %s", exc)
            return {}
        return {r["id"]: r["name"] for r in rows}


# ---------------------------------------------------------------------------
# helpers (module-level, purely functional)
# ---------------------------------------------------------------------------


def _strip_internal(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop private ``_sort_*`` fields before returning to the API layer."""
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def _first_non_empty(*values: Any) -> Any:
    """First value that is not ``None`` and not an empty string, else ``None``."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _maybe_int(value: Any) -> int | None:
    """Coerce to int, returning ``None`` on any failure (fail-open)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_positive_int(value: Any) -> int | None:
    """Coerce to ``round(float(value))``, returning ``None`` if ≤ 0 or invalid."""
    if value is None:
        return None
    try:
        n = round(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _drift_gap(
    profile: dict[str, Any],
    *,
    now_ts: int,
) -> tuple[float, float] | None:
    """Return ``(days_since, expected)`` if the cadence profile has drifted, else None.

    Mirrors the gating logic in :class:`producers.cadence.CadenceProducer`
    so the You/People "drifting" section stays consistent with the set of
    contacts the producer would surface a Moment for.
    """
    try:
        expected = float(profile["expected_cadence_days"])
        count = int(profile["count"])
        last_ts = int(profile["last_inbound_ts"])
    except (KeyError, TypeError, ValueError):
        return None
    if expected <= 0 or count < MIN_CADENCE_COUNT:
        return None
    days_since = max(0.0, (now_ts - last_ts) / SECONDS_PER_DAY)
    if days_since <= expected * DRIFT_FACTOR:
        return None
    return (days_since, expected)


def _predicted_next(
    last_ts: int | None,
    cadence_days: int | None,
    now_ts: int,
) -> str | None:
    """Render the dossier's "next expected contact" microcopy.

    Returns ``None`` when the cadence signal extractor hasn't written
    enough history to predict — the UI renders a placeholder rather
    than surfacing a made-up number.
    """
    if last_ts is None or cadence_days is None or cadence_days <= 0:
        return None
    predicted_ts = last_ts + cadence_days * SECONDS_PER_DAY
    days_until = round((predicted_ts - now_ts) / SECONDS_PER_DAY)
    if days_until > 1:
        return f"Next contact expected in ~{days_until} days"
    if days_until == 1:
        return "Next contact expected in ~1 day"
    if days_until == 0:
        return "Next contact expected today"
    overdue = -days_until
    if overdue == 1:
        return "Overdue by 1 day"
    return f"Overdue by {overdue} days"


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_SPARKLINE_DAYS",
    "DRIFT_FACTOR",
    "MAX_PAGE_SIZE",
    "MIN_CADENCE_COUNT",
    "PeopleRepository",
]
