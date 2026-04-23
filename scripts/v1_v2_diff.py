#!/usr/bin/env python3
"""Post-migration v1 → v2 data diff.

Implements NEXT_TASKS Category C § "v1/v2 data diff tool". Run **after**
``scripts/migrate_v1_to_v2.py`` has produced the v2 ``lifeos.db`` and the
cutover runbook § 4 has brought v2 up live. This script is the sanity
check the operator runs against the two on-disk artefacts — the
timestamped v1 snapshot and the fresh v2 database — to catch silent data
loss that the migrator's own invariants would not surface (e.g. a v1
row that migrated with a truncated payload).

Three checks
------------
1. **Row counts** — per v1 source table, compare to the v2 target table
   after applying the migrator's known semantics (e.g.
   ``state.db.tasks`` maps to ``moments`` with
   ``source_insight_type='legacy_task'``; ``preferences.db.feedback_log``
   maps to ``feedback_events`` with ``source='v1_migration'``). The
   kept-vs-dropped signal_profile split mirrors the CEO plan and the
   migrator.

2. **Spot checks** — pull ``--sample-size`` (default 10) random events
   from v1 by id using a seeded RNG, fetch each by id from v2, and
   assert every payload field survived verbatim: ``type`` and ``source``
   match exactly; ``payload`` JSON matches exactly; the v1 ISO-8601
   ``timestamp`` coerces to the v2 integer ``timestamp`` (unix
   seconds). Reproducibility is guaranteed by ``--seed``.

3. **FK integrity** — walk v2 tables and confirm every cross-table
   reference resolves:
     * ``moment_state_history.moment_id`` → ``moments.id``
     * ``event_tags.event_id`` → ``events.id``
     * every ``event_id`` appearing inside a moment's ``evidence`` JSON
       resolves to an ``events.id``

The diff never writes to either the v1 snapshot or the v2 database.
Both are opened ``mode=ro``. The report is written to
``docs/cutover-diffs/<YYYY-MM-DD>.md`` (or ``--output``) and exits
``0`` when every check passed, ``1`` on any failure, ``2`` on bad CLI
input (missing v1 snapshot / v2 db file).

Usage
-----

::

    python scripts/v1_v2_diff.py                    # defaults
    python scripts/v1_v2_diff.py --v1-dir data/backup-20260420 \\
        --v2-db data/lifeos.db --output docs/cutover-diffs/2026-04-22.md \\
        --sample-size 20 --seed 42

References
----------
- Task spec: ``NEXT_TASKS.md`` § Category C "v1/v2 data diff tool".
- Migrator:  ``scripts/migrate_v1_to_v2.py``.
- Runbook:   ``docs/cutover-runbook.md`` § 5 "Verify" (post-cutover row-count diff).
- Rehearsal (related but distinct — rehearsal runs the migrator, diff
  inspects an already-migrated DB): ``scripts/cutover_rehearsal.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_V1_DIR = REPO_ROOT / "data"
DEFAULT_V2_DB = REPO_ROOT / "data" / "lifeos.db"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "cutover-diffs"
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_SEED = 0

# Kept / dropped profile-type splits — mirror migrate_v1_to_v2.
KEPT_PROFILE_TYPES: frozenset[str] = frozenset(
    {"cadence", "relationship", "temporal", "spatial", "comm_template", "routine"}
)
DROPPED_PROFILE_TYPES: frozenset[str] = frozenset({"mood", "decision", "expertise", "values"})

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BADINPUT = 2

log = logging.getLogger("v1_v2_diff")


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckResult:
    """One row in the diff report.

    `detail` is the human-readable one-line explanation rendered into the
    markdown table. `expected`/`actual` are populated on row-count rows
    only (``None`` elsewhere) so the markdown renderer can format a
    three-column table without the caller needing to know.
    """

    name: str
    passed: bool
    detail: str
    expected: int | None = None
    actual: int | None = None


@dataclass
class DiffReport:
    """Full diff outcome, serialisable to markdown.

    Collected as three disjoint check lists so the markdown renderer can
    section cleanly. ``all_passed`` is the authoritative pass/fail flag
    used by the CLI exit code.
    """

    v1_dir: Path
    v2_db: Path
    generated_at: int
    seed: int
    sample_size: int
    row_counts: list[CheckResult] = field(default_factory=list)
    spot_checks: list[CheckResult] = field(default_factory=list)
    fk_integrity: list[CheckResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def all_checks(self) -> list[CheckResult]:
        return list(self.row_counts) + list(self.spot_checks) + list(self.fk_integrity)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.all_checks())

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.all_checks() if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.all_checks() if not c.passed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _open_ro(path: Path) -> sqlite3.Connection:
    """Open an SQLite DB read-only (``mode=ro``) via the URI API."""
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _count(conn: sqlite3.Connection, table: str, where: str = "", params: tuple = ()) -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    (n,) = conn.execute(sql, params).fetchone()
    return int(n)


def _iso_to_unix(value: str | None) -> int | None:
    """Convert v1 ISO-8601 to unix seconds. Mirrors migrate_v1_to_v2."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return int(dt.datetime.fromisoformat(text).timestamp())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Row-count checks
# ---------------------------------------------------------------------------
def check_row_counts(
    v1_conns: dict[str, sqlite3.Connection],
    v2: sqlite3.Connection,
) -> list[CheckResult]:
    """Compare v1 source-table row counts against v2 target-table counts.

    Applies migrator semantics: tasks → moments (legacy_task), feedback_log
    → feedback_events (source=v1_migration), kept profile types only, etc.
    A missing v1 DB collapses the affected rows to ``skipped`` rather than
    failing — the rehearsal + scale tests already cover happy-path; this
    tool has to degrade gracefully on a partial snapshot.
    """
    results: list[CheckResult] = []

    def emit(name: str, expected: int, actual: int, note: str = "") -> None:
        ok = expected == actual
        detail = f"expected={expected} actual={actual}"
        if note:
            detail = f"{detail} — {note}"
        results.append(
            CheckResult(
                name=name,
                passed=ok,
                detail=detail,
                expected=expected,
                actual=actual,
            )
        )

    events = v1_conns.get("events")
    if events is not None and _table_exists(events, "events"):
        expected = _count(events, "events")
        actual = _count(v2, "events")
        emit("events", expected, actual)

    entities = v1_conns.get("entities")
    if entities is not None:
        for v1_table, kind in (
            ("contacts", "contact"),
            ("places", "place"),
            ("subscriptions", "subscription"),
        ):
            if _table_exists(entities, v1_table):
                expected = _count(entities, v1_table)
                actual = _count(v2, "entities", "kind=?", (kind,))
                emit(f"entities[kind={kind}]", expected, actual)

    state = v1_conns.get("state")
    if state is not None and _table_exists(state, "tasks"):
        expected = _count(state, "tasks")
        actual = _count(v2, "moments", "source_insight_type=?", ("legacy_task",))
        emit("moments[source=legacy_task]", expected, actual)

    user_model = v1_conns.get("user_model")
    if user_model is not None and _table_exists(user_model, "signal_profiles"):
        kept_placeholders = ",".join("?" for _ in KEPT_PROFILE_TYPES)
        kept_types_tuple = tuple(sorted(KEPT_PROFILE_TYPES))
        expected = _count(
            user_model,
            "signal_profiles",
            f"profile_type IN ({kept_placeholders})",
            kept_types_tuple,
        )
        actual = _count(v2, "signal_profiles")
        emit(
            "signal_profiles (kept types only)",
            expected,
            actual,
            note="dropped types: " + ", ".join(sorted(DROPPED_PROFILE_TYPES)),
        )

    prefs = v1_conns.get("preferences")
    if prefs is not None:
        if _table_exists(prefs, "user_preferences"):
            expected = _count(prefs, "user_preferences")
            actual = _count(v2, "preferences")
            emit("preferences", expected, actual)
        if _table_exists(prefs, "feedback_log"):
            expected = _count(prefs, "feedback_log")
            actual = _count(v2, "feedback_events", "source=?", ("v1_migration",))
            emit("feedback_events[source=v1_migration]", expected, actual)

    return results


# ---------------------------------------------------------------------------
# Spot checks
# ---------------------------------------------------------------------------
def sample_event_ids(
    v1_events: sqlite3.Connection,
    sample_size: int,
    seed: int,
) -> list[str]:
    """Return a seeded-random selection of up to ``sample_size`` event ids.

    Pulled from v1 in id order so the RNG's choice is reproducible across
    runs given the same seed. If v1 has fewer rows than ``sample_size``
    every row is returned.
    """
    ids = [r[0] for r in v1_events.execute("SELECT id FROM events ORDER BY id").fetchall()]
    if not ids:
        return []
    rng = random.Random(seed)
    n = min(sample_size, len(ids))
    return sorted(rng.sample(ids, n))


def check_spot_events(
    v1_events: sqlite3.Connection | None,
    v2: sqlite3.Connection,
    sample_size: int,
    seed: int,
) -> list[CheckResult]:
    """Per-id payload round-trip check for a seeded sample of events.

    For each sampled id: fetch the v1 row and the v2 row. Report one
    ``CheckResult`` per id with a one-line detail describing the first
    mismatch encountered, or ``ok`` when every compared field matches.
    ``type``, ``source``, and ``payload`` must match verbatim; the v1
    ISO-8601 ``timestamp`` is coerced via the same helper the migrator
    uses and compared to v2's integer ``timestamp``.
    """
    results: list[CheckResult] = []
    if v1_events is None or not _table_exists(v1_events, "events"):
        return results

    ids = sample_event_ids(v1_events, sample_size, seed)
    for event_id in ids:
        v1_row = v1_events.execute(
            "SELECT id, type, source, timestamp, payload FROM events WHERE id=?",
            (event_id,),
        ).fetchone()
        v2_row = v2.execute(
            "SELECT id, type, source, timestamp, payload FROM events WHERE id=?",
            (event_id,),
        ).fetchone()
        if v2_row is None:
            results.append(
                CheckResult(
                    name=f"event {event_id}",
                    passed=False,
                    detail="missing from v2 events table",
                )
            )
            continue
        _, v1_type, v1_source, v1_ts_iso, v1_payload = v1_row
        _, v2_type, v2_source, v2_ts_int, v2_payload = v2_row
        mismatches: list[str] = []
        if v1_type != v2_type:
            mismatches.append(f"type: v1={v1_type!r} v2={v2_type!r}")
        if v1_source != v2_source:
            mismatches.append(f"source: v1={v1_source!r} v2={v2_source!r}")
        # Payload compared as decoded JSON so harmless whitespace differences
        # don't mask real value divergence.
        if _json_canonical(v1_payload) != _json_canonical(v2_payload):
            mismatches.append("payload JSON differs")
        v1_ts_unix = _iso_to_unix(v1_ts_iso)
        if v1_ts_unix is None:
            mismatches.append(f"v1 timestamp unparseable: {v1_ts_iso!r}")
        elif v1_ts_unix != v2_ts_int:
            mismatches.append(f"timestamp: v1_iso={v1_ts_iso!r} v1_unix={v1_ts_unix} v2={v2_ts_int}")
        if mismatches:
            results.append(
                CheckResult(
                    name=f"event {event_id}",
                    passed=False,
                    detail="; ".join(mismatches),
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"event {event_id}",
                    passed=True,
                    detail="type/source/payload/timestamp match",
                )
            )
    return results


def _json_canonical(text: str | None) -> str:
    """Canonicalise a JSON string for equality comparison (sort keys)."""
    if text is None:
        return ""
    try:
        return json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # Non-JSON payloads survive as their literal text — the fast-path
        # is still sound, it's just not structure-aware.
        return text


# ---------------------------------------------------------------------------
# FK integrity
# ---------------------------------------------------------------------------
def check_fk_integrity(v2: sqlite3.Connection) -> list[CheckResult]:
    """Return one CheckResult per cross-table reference class.

    Three classes, in order of blast radius:

    * ``moment_state_history.moment_id → moments.id`` — FK is declared
      with ON DELETE CASCADE so violations should be impossible; this
      check catches accidental direct-SQL damage.
    * ``event_tags.event_id → events.id`` — same structural FK with
      ON DELETE CASCADE.
    * ``moments.evidence[*].event_id → events.id`` — evidence is
      free-form JSON, not FK-enforced; this is the interesting check.
    """
    results: list[CheckResult] = []

    orphan_history = v2.execute(
        "SELECT COUNT(*) FROM moment_state_history h LEFT JOIN moments m ON m.id = h.moment_id WHERE m.id IS NULL"
    ).fetchone()[0]
    results.append(
        CheckResult(
            name="moment_state_history.moment_id → moments.id",
            passed=orphan_history == 0,
            detail=f"{orphan_history} orphan rows" if orphan_history else "no orphans",
        )
    )

    orphan_tags = v2.execute(
        "SELECT COUNT(*) FROM event_tags t LEFT JOIN events e ON e.id = t.event_id WHERE e.id IS NULL"
    ).fetchone()[0]
    results.append(
        CheckResult(
            name="event_tags.event_id → events.id",
            passed=orphan_tags == 0,
            detail=f"{orphan_tags} orphan rows" if orphan_tags else "no orphans",
        )
    )

    # Evidence: iterate moments.evidence JSON and collect every event_id
    # referenced. The migrator writes ``evidence='[]'`` so on a freshly-
    # migrated DB this is always 0-of-0, but real-world producers attach
    # evidence over time; this is the check that bites when they drift.
    event_ids_in_evidence, parse_errors = _collect_evidence_event_ids(v2)
    if not event_ids_in_evidence and parse_errors == 0:
        results.append(
            CheckResult(
                name="moments.evidence[*].event_id → events.id",
                passed=True,
                detail="0 evidence references to validate",
            )
        )
    else:
        qs = ",".join("?" for _ in event_ids_in_evidence) or "''"
        found_rows = (
            v2.execute(
                f"SELECT id FROM events WHERE id IN ({qs})",
                tuple(event_ids_in_evidence),
            ).fetchall()
            if event_ids_in_evidence
            else []
        )
        found = {r[0] for r in found_rows}
        missing = [eid for eid in event_ids_in_evidence if eid not in found]
        ok = not missing and parse_errors == 0
        detail_parts: list[str] = [f"{len(event_ids_in_evidence)} refs"]
        if missing:
            preview = ", ".join(missing[:5])
            detail_parts.append(f"{len(missing)} unresolved (first: {preview})")
        if parse_errors:
            detail_parts.append(f"{parse_errors} unparseable evidence JSON blobs")
        results.append(
            CheckResult(
                name="moments.evidence[*].event_id → events.id",
                passed=ok,
                detail=" / ".join(detail_parts),
            )
        )

    return results


def _collect_evidence_event_ids(v2: sqlite3.Connection) -> tuple[list[str], int]:
    """Walk ``moments.evidence`` JSON, return distinct event ids + parse-error count.

    Evidence is documented as a JSON array of objects; any dict with an
    ``event_id`` key contributes its value. Order is preserved so the
    diagnostic `first:` preview stays deterministic.
    """
    rows = v2.execute("SELECT id, evidence FROM moments").fetchall()
    seen: list[str] = []
    seen_set: set[str] = set()
    parse_errors = 0
    for _mid, evidence in rows:
        try:
            parsed = json.loads(evidence) if evidence else []
        except (TypeError, ValueError):
            parse_errors += 1
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if isinstance(item, dict) and "event_id" in item:
                eid = item["event_id"]
                if isinstance(eid, str) and eid not in seen_set:
                    seen_set.add(eid)
                    seen.append(eid)
    return seen, parse_errors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_diff(
    v1_dir: Path,
    v2_db: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> DiffReport:
    """Run all three check classes, returning a populated ``DiffReport``.

    The driver opens every reachable v1 DB read-only, collects check
    results, closes the connections, and hands back the report. Missing
    v1 DBs degrade to a NOTE rather than a failure so a partial snapshot
    still produces a useful report.
    """
    report = DiffReport(
        v1_dir=v1_dir,
        v2_db=v2_db,
        generated_at=int(dt.datetime.now(tz=dt.UTC).timestamp()),
        seed=seed,
        sample_size=sample_size,
    )

    v1_conns: dict[str, sqlite3.Connection] = {}
    try:
        for logical, fname in (
            ("events", "events.db"),
            ("entities", "entities.db"),
            ("state", "state.db"),
            ("user_model", "user_model.db"),
            ("preferences", "preferences.db"),
        ):
            path = v1_dir / fname
            if path.exists():
                v1_conns[logical] = _open_ro(path)
            else:
                report.notes.append(f"v1 source missing: {fname} (checks skipped)")

        with _open_ro(v2_db) as v2:
            report.row_counts = check_row_counts(v1_conns, v2)
            report.spot_checks = check_spot_events(v1_conns.get("events"), v2, sample_size, seed)
            report.fk_integrity = check_fk_integrity(v2)
    finally:
        for conn in v1_conns.values():
            conn.close()

    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(report: DiffReport) -> str:
    """Format a :class:`DiffReport` as the operator-facing markdown report."""
    generated_iso = dt.datetime.fromtimestamp(report.generated_at, tz=dt.UTC).isoformat(timespec="seconds")
    overall = "PASS" if report.all_passed else "FAIL"
    lines: list[str] = []
    lines.append(f"# v1 → v2 cutover diff — {generated_iso[:10]}")
    lines.append("")
    lines.append(f"- v1 snapshot: `{report.v1_dir}`")
    lines.append(f"- v2 database: `{report.v2_db}`")
    lines.append(f"- generated at: `{generated_iso}`")
    lines.append(f"- seed: `{report.seed}`")
    lines.append(f"- sample size: `{report.sample_size}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Result: **{overall}** — {report.pass_count} passed, {report.fail_count} failed.")
    if report.notes:
        lines.append("")
        lines.append("Notes:")
        for n in report.notes:
            lines.append(f"- {n}")
    lines.append("")

    lines.append("## Row counts")
    lines.append("")
    if report.row_counts:
        lines.append("| Table | v1 | v2 | Result |")
        lines.append("| --- | ---: | ---: | --- |")
        for c in report.row_counts:
            lines.append(f"| {c.name} | {c.expected} | {c.actual} | {'PASS' if c.passed else 'FAIL'} |")
    else:
        lines.append("_No row-count checks ran (no reachable v1 source tables)._")
    lines.append("")

    lines.append("## Spot checks")
    lines.append("")
    lines.append(f"{len(report.spot_checks)} events sampled (seed={report.seed}, sample_size={report.sample_size}).")
    if report.spot_checks:
        lines.append("")
        lines.append("| Event | Result | Detail |")
        lines.append("| --- | --- | --- |")
        for c in report.spot_checks:
            lines.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    lines.append("")

    lines.append("## FK integrity")
    lines.append("")
    if report.fk_integrity:
        lines.append("| Reference | Result | Detail |")
        lines.append("| --- | --- | --- |")
        for c in report.fk_integrity:
            lines.append(f"| `{c.name}` | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    else:
        lines.append("_No FK checks ran._")
    lines.append("")

    failures = [c for c in report.all_checks() if not c.passed]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for c in failures:
            lines.append(f"- **{c.name}** — {c.detail}")
        lines.append("")

    return "\n".join(lines)


def write_report(report: DiffReport, output: Path) -> None:
    """Write ``report`` to ``output`` (creating parent dirs as needed)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _default_output_path(today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    return DEFAULT_OUTPUT_DIR / f"{today.isoformat()}.md"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1 → v2 post-migration data diff")
    parser.add_argument(
        "--v1-dir",
        default=str(DEFAULT_V1_DIR),
        help=f"Directory containing the v1 snapshot *.db files (default: {DEFAULT_V1_DIR})",
    )
    parser.add_argument(
        "--v2-db",
        default=str(DEFAULT_V2_DB),
        help=f"Path to the migrated v2 SQLite DB (default: {DEFAULT_V2_DB})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the markdown report (default: docs/cutover-diffs/<YYYY-MM-DD>.md)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of events to spot-check (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for spot-check sampling (default: 0)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    _setup_logging(args.verbose)

    v1_dir = Path(args.v1_dir)
    v2_db = Path(args.v2_db)
    if not v1_dir.exists() or not v1_dir.is_dir():
        log.error("v1 snapshot dir %s does not exist or is not a directory", v1_dir)
        return EXIT_BADINPUT
    if not v2_db.exists() or not v2_db.is_file():
        log.error("v2 DB %s does not exist or is not a file", v2_db)
        return EXIT_BADINPUT
    if args.sample_size < 0:
        log.error("sample-size must be >= 0")
        return EXIT_BADINPUT

    output = Path(args.output) if args.output else _default_output_path()
    report = run_diff(
        v1_dir,
        v2_db,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    write_report(report, output)
    log.info(
        "diff complete: %s (%d passed, %d failed) -> %s",
        "PASS" if report.all_passed else "FAIL",
        report.pass_count,
        report.fail_count,
        output,
    )
    return EXIT_PASS if report.all_passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
