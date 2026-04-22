#!/usr/bin/env python3
"""End-to-end cutover rehearsal for the v1 → v2 migration.

Implements NEXT_TASKS Week 12 § "Cutover rehearsal (full dry-run end-to-end)".
Operates on a v1 backup directory (``./data/backup-YYYYMMDD/`` by default,
newest by lexical sort) and runs the three dry-run checks the CEO plan locks
in for the migration step (``§ "Data migration"``):

    1. Row-count diff: per-table source row count equals
       ``translated + dropped + skipped`` reported by the migrator. PRAGMA
       ``foreign_key_check`` on the v2 output must report zero violations.
    2. Referential integrity: every event id referenced by a Moment's
       ``evidence`` JSON resolves to an event row in the migrated v2
       events table.
    3. Vector-store integrity: every surviving v1 embedding (``lance/``
       directory) has a counterpart event id in the migrated v2 events
       table. If LanceDB is not importable, the check degrades to a
       directory-presence + on-disk size report and is recorded as
       ``N/A`` rather than a failure (the operator can still see
       whether a rehearsal touched the lance corpus).

The rehearsal **never** performs an actual cutover — that is supervised
human work. The script writes a markdown report to
``docs/cutover-rehearsals/<date>.md`` regardless of pass/fail and exits
``0`` only when every executed check passed (or was N/A). Exit ``1`` when
any check FAILed; exit ``0`` with a NOTE message when no v1 backup is
present locally (the autonomous agent runs on a workstation without
production data; the rehearsal is meaningful only on the Mac Mini).

Operator usage::

    python scripts/cutover_rehearsal.py                    # auto-discover
    python scripts/cutover_rehearsal.py --source-dir DIR   # explicit
    python scripts/cutover_rehearsal.py --output-dir docs/cutover-rehearsals/

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "Data migration (expanded — this is the highest-risk operational step)".
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.migrate_v1_to_v2 import MigrationReport, run_migration  # noqa: E402

DEFAULT_BACKUP_GLOB = "backup-*"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "cutover-rehearsals"

EXIT_PASS = 0
EXIT_FAIL = 1


# ---------------------------------------------------------------------------
# Backup discovery
# ---------------------------------------------------------------------------
def discover_backup(data_dir: Path) -> Path | None:
    """Return the newest ``backup-*`` subdir of ``data_dir`` or ``None``.

    "Newest" is lexical sort because backups are named ``backup-YYYYMMDD`` —
    a deterministic ordering that does not need a filesystem stat call.
    """
    if not data_dir.exists():
        return None
    candidates = sorted(p for p in data_dir.glob(DEFAULT_BACKUP_GLOB) if p.is_dir())
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """One of the three CEO-plan checks plus its detail lines."""

    name: str
    status: str  # PASS | FAIL | N/A
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status in {"PASS", "N/A"}


@dataclass
class RehearsalReport:
    """Structured report for one rehearsal run; serialised to markdown."""

    backup_path: Path
    output_db_path: Path
    started_at: dt.datetime
    finished_at: dt.datetime
    output_db_bytes: int
    migration: MigrationReport
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def runtime_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def overall(self) -> str:
        return "PASS" if all(c.passed for c in self.checks) else "FAIL"


# ---------------------------------------------------------------------------
# Check 1 — row-count diff + FK integrity
# ---------------------------------------------------------------------------
def check_row_counts(
    backup_dir: Path,
    output_db: Path,
    report: MigrationReport,
) -> CheckResult:
    """Each translated table's row count must equal the migrator's report.

    Also runs SQLite ``PRAGMA foreign_key_check`` on the output: a single
    violating row fails the whole check.
    """
    details: list[str] = []
    failed = False

    expectations: tuple[tuple[str, str, int], ...] = (
        ("events", "events", report.events.translated),
        ("entities", "entities", report.entities.translated),
        ("moments", "moments", report.moments_from_tasks.translated),
        ("signal_profiles", "signal_profiles", report.signal_profiles.translated),
        ("preferences", "preferences", report.preferences.translated),
    )
    with sqlite3.connect(output_db) as conn:
        for label, table, expected in expectations:
            (actual,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            if actual != expected:
                failed = True
                details.append(f"{label}: expected {expected}, got {actual}")
            else:
                details.append(f"{label}: {actual} rows (matches translated)")

        # Source-vs-translated parity per table (drops + skips accounted for).
        srcs = (
            ("events", report.events.source, report.events.translated, report.events.dropped, 0),
            ("entities", report.entities.source, report.entities.translated, report.entities.dropped, 0),
            (
                "moments_from_tasks",
                report.moments_from_tasks.source,
                report.moments_from_tasks.translated,
                report.moments_from_tasks.dropped,
                0,
            ),
            (
                "signal_profiles",
                report.signal_profiles.source,
                report.signal_profiles.translated,
                report.signal_profiles.dropped,
                0,
            ),
            ("preferences", report.preferences.source, report.preferences.translated, report.preferences.dropped, 0),
            (
                "notification_feedback",
                report.notification_feedback_skipped,
                0,
                0,
                report.notification_feedback_skipped,
            ),
        )
        for label, src, translated, dropped, skipped in srcs:
            if src != translated + dropped + skipped:
                failed = True
                details.append(f"{label}: source={src} != translated={translated}+dropped={dropped}+skipped={skipped}")

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            failed = True
            details.append(f"foreign_key_check: {len(violations)} violation(s) — first: {violations[0]}")
        else:
            details.append("foreign_key_check: 0 violations")

    return CheckResult(
        name="row-count diff",
        status="FAIL" if failed else "PASS",
        details=details,
    )


# ---------------------------------------------------------------------------
# Check 2 — referential integrity (Moment.evidence → event.id)
# ---------------------------------------------------------------------------
def check_moment_evidence(output_db: Path) -> CheckResult:
    """Every event id in ``moments.evidence`` must resolve in ``events``.

    The legacy_task migration writes ``evidence='[]'`` so those rows pass
    trivially; producer-emitted Moments will populate the list. The check
    walks every row to keep the per-call cost predictable.
    """
    details: list[str] = []
    dangling: list[tuple[str, str]] = []  # (moment_id, missing_event_id)

    with sqlite3.connect(output_db) as conn:
        event_ids = {row[0] for row in conn.execute("SELECT id FROM events").fetchall()}
        moment_rows = conn.execute("SELECT id, evidence FROM moments").fetchall()

    inspected = 0
    for moment_id, evidence_json in moment_rows:
        if not evidence_json:
            continue
        try:
            evidence = json.loads(evidence_json)
        except (TypeError, ValueError):
            dangling.append((moment_id, f"<unparseable evidence: {evidence_json!r}>"))
            continue
        if not isinstance(evidence, list):
            dangling.append((moment_id, f"<non-list evidence: {type(evidence).__name__}>"))
            continue
        for ref in evidence:
            inspected += 1
            if not isinstance(ref, str) or ref not in event_ids:
                dangling.append((moment_id, str(ref)))

    details.append(f"inspected {inspected} evidence reference(s) across {len(moment_rows)} moment(s)")
    if dangling:
        details.append(f"dangling references: {len(dangling)} (first 5: {dangling[:5]})")
        return CheckResult(name="moment.evidence integrity", status="FAIL", details=details)
    details.append("all evidence references resolve to events.id")
    return CheckResult(name="moment.evidence integrity", status="PASS", details=details)


# ---------------------------------------------------------------------------
# Check 3 — vector-store integrity
# ---------------------------------------------------------------------------
def check_vector_store(backup_dir: Path, output_db: Path) -> CheckResult:
    """Every surviving v1 embedding must point to a v2-migrated event.

    The v1 layout the production deployment uses is
    ``<data_dir>/lance/documents.lance/`` (cf. ``storage/vector_store.py``
    line 87). When ``lancedb`` is importable we open the table and walk
    every doc_id; otherwise we fall back to a directory-presence + size
    report and mark the check as ``N/A`` because we cannot prove
    reachability without reading the corpus.
    """
    details: list[str] = []
    lance_root = backup_dir / "lance"
    if not lance_root.exists():
        details.append(f"no lance directory at {lance_root} — vector store check N/A")
        return CheckResult(name="vector-store integrity", status="N/A", details=details)

    total_bytes = sum(p.stat().st_size for p in lance_root.rglob("*") if p.is_file())
    details.append(f"lance dir present at {lance_root} ({total_bytes} bytes on disk)")

    try:
        import lancedb  # type: ignore[import-not-found]
    except ImportError:
        details.append("lancedb not importable in this env — reachability check skipped")
        return CheckResult(name="vector-store integrity", status="N/A", details=details)

    try:
        db = lancedb.connect(str(lance_root))
        table_names = list(db.table_names())
    except Exception as exc:  # lancedb raises a plain Exception subclass
        details.append(f"lancedb.connect failed: {exc!r}")
        return CheckResult(name="vector-store integrity", status="FAIL", details=details)

    if "documents" not in table_names:
        details.append(f"lance dir has no 'documents' table (found: {table_names}) — N/A")
        return CheckResult(name="vector-store integrity", status="N/A", details=details)

    try:
        table = db.open_table("documents")
        rows = table.search().select(["doc_id"]).limit(None).to_list()
    except Exception as exc:
        details.append(f"reading documents table failed: {exc!r}")
        return CheckResult(name="vector-store integrity", status="FAIL", details=details)

    details.append(f"documents table has {len(rows)} embedding row(s)")
    with sqlite3.connect(output_db) as conn:
        event_ids = {row[0] for row in conn.execute("SELECT id FROM events").fetchall()}
    dangling: list[str] = []
    for r in rows:
        doc_id = r.get("doc_id")
        if not isinstance(doc_id, str):
            continue
        # v1 doc_id convention is ``<event_id>`` or ``<event_id>_<chunk>`` (see
        # storage/vector_store.py:661 — delete uses ``LIKE doc_id_%``). Strip
        # the chunk suffix before joining.
        event_id = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
        if event_id not in event_ids and doc_id not in event_ids:
            dangling.append(doc_id)
    if dangling:
        details.append(f"dangling embeddings: {len(dangling)} (first 5: {dangling[:5]})")
        return CheckResult(name="vector-store integrity", status="FAIL", details=details)
    details.append("all embeddings reachable from migrated events")
    return CheckResult(name="vector-store integrity", status="PASS", details=details)


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------
def write_report(out_dir: Path, report: RehearsalReport) -> Path:
    """Persist one markdown report per run; filename is the rehearsal date."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.started_at.date().isoformat()}.md"
    lines: list[str] = [
        f"# Cutover rehearsal — {report.started_at.date().isoformat()}",
        "",
        f"- backup: `{report.backup_path}`",
        f"- output db (temp): `{report.output_db_path}`",
        f"- runtime: **{report.runtime_seconds:.2f} s**",
        f"- on-disk size of v2 output: **{report.output_db_bytes} bytes**",
        f"- overall: **{report.overall}**",
        "",
        "> **DRY-RUN ONLY** — this rehearsal does not perform an actual cutover.",
        '> Production cutover is supervised human work (see CEO plan § "Cutover procedure").',
        "",
        "## Migration report",
        "",
        f"- events: {report.migration.events.as_dict()}",
        f"- entities: {report.migration.entities.as_dict()}",
        f"- moments_from_tasks: {report.migration.moments_from_tasks.as_dict()}",
        f"- signal_profiles: {report.migration.signal_profiles.as_dict()}",
        f"- preferences: {report.migration.preferences.as_dict()}",
        f"- notification_feedback skipped: {report.migration.notification_feedback_skipped}",
        "",
        "## Checks",
        "",
    ]
    for check in report.checks:
        lines.append(f"### {check.name} — **{check.status}**")
        lines.append("")
        for detail in check.details:
            lines.append(f"- {detail}")
        lines.append("")
    if report.migration.notes:
        lines.append("## Migration notes")
        lines.append("")
        for note in report.migration.notes:
            lines.append(f"- {note}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_rehearsal(
    backup_dir: Path,
    output_dir: Path,
    *,
    log: logging.Logger | None = None,
) -> RehearsalReport:
    """Run the full rehearsal and persist the markdown report.

    The migration writes to a tmp dir which is deleted on return; the
    on-disk size of the output is captured first so the report retains it.
    """
    log = log or logging.getLogger(__name__)
    started = dt.datetime.now(tz=dt.UTC)
    started_perf = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="lifeos-rehearsal-") as tmpdir:
        out_db = Path(tmpdir) / "lifeos_v2_rehearsal.db"
        log.info("running migration: backup=%s output=%s", backup_dir, out_db)
        migration = run_migration(backup_dir, out_db, log=log)
        out_db_bytes = out_db.stat().st_size

        checks = [
            check_row_counts(backup_dir, out_db, migration),
            check_moment_evidence(out_db),
            check_vector_store(backup_dir, out_db),
        ]

        finished = dt.datetime.now(tz=dt.UTC)
        elapsed = time.perf_counter() - started_perf
        log.info("rehearsal finished in %.2fs", elapsed)

        report = RehearsalReport(
            backup_path=backup_dir,
            output_db_path=out_db,
            started_at=started,
            finished_at=finished,
            output_db_bytes=out_db_bytes,
            migration=migration,
            checks=checks,
        )
        report_path = write_report(output_dir, report)
        log.info("wrote rehearsal report to %s", report_path)
        return report


def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("cutover_rehearsal")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1 → v2 cutover rehearsal (dry-run only)")
    parser.add_argument(
        "--source-dir",
        default=None,
        help=("Explicit path to the v1 backup directory. Default: newest ./data/backup-* subdir."),
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Root data dir to scan for backup-* subdirs (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write the markdown report (default: docs/cutover-rehearsals)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    log = _setup_logging(args.verbose)

    if args.source_dir:
        backup = Path(args.source_dir)
        if not backup.exists():
            log.error("source-dir %s does not exist", backup)
            return EXIT_FAIL
    else:
        backup = discover_backup(Path(args.data_dir))
        if backup is None:
            log.warning(
                "NOTE: no v1 backup found under %s — rehearsal requires a local "
                "v1 backup; run on the Mac Mini that hosts production.",
                args.data_dir,
            )
            # Persist the NOTE so an operator scanning docs/cutover-rehearsals/
            # sees the skip context without needing to re-run.
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stub_path = output_dir / f"{dt.date.today().isoformat()}-skipped.md"
            stub_path.write_text(
                f"# Cutover rehearsal — {dt.date.today().isoformat()} (SKIPPED)\n\n"
                f"No v1 backup found under `{args.data_dir}`. The rehearsal "
                "requires a local v1 backup at `data/backup-YYYYMMDD/` and is "
                "expected to run on the Mac Mini that hosts production.\n\n"
                "> **DRY-RUN ONLY** — this rehearsal does not perform an "
                "actual cutover.\n",
                encoding="utf-8",
            )
            return EXIT_PASS

    output_dir = Path(args.output_dir)
    report = run_rehearsal(backup, output_dir, log=log)
    log.info("overall: %s", report.overall)
    return EXIT_PASS if report.overall == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
