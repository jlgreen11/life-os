#!/usr/bin/env python3
"""Daily SQLite integrity check + outbox alert for Life OS v2.

Runs :func:`core.integrity.check_sqlite_integrity` against
``./data/lifeos.db`` (configurable) and, on failure, enqueues a single
``integrity_alert`` row into the ``outbox`` table so the running Life OS
process surfaces a Moment/notification via the transactional outbox worker
(Week 3 task in ``NEXT_TASKS.md``).

Intended invocation: a launchd/cron job running once per day on the Mac
Mini that hosts the v2 deployment. Exit codes are scripted for that
wrapper:

* ``0`` — integrity_check reported ``ok``; no alert enqueued.
* ``1`` — integrity_check reported one or more errors; alert was enqueued.
* ``2`` — script-level error (file missing, outbox schema absent, etc.);
  no alert could be enqueued, so the wrapper should page the operator.

The alert payload is a compact JSON object the outbox worker can render
without extra lookups::

    {
        "kind": "integrity_failure",
        "db_path": "./data/lifeos.db",
        "errors": ["row 17 missing from index ...", ...],
        "detected_at": 1714000000
    }

The row is inserted with ``state='pending'`` (schema default) and a
unique ``(event_id, subject)`` per-run so the outbox UNIQUE constraint
does not collapse multiple daily alerts into one.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "Reviewer Concerns (resolved)".
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md`` §
  "outbox pattern".
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from core.integrity import IntegrityReport, check_sqlite_integrity

DEFAULT_DB_PATH = "./data/lifeos.db"
ALERT_SUBJECT = "integrity_alert"

EXIT_OK = 0
EXIT_INTEGRITY_FAILED = 1
EXIT_SCRIPT_ERROR = 2


def enqueue_alert(db_path: str, report: IntegrityReport, *, now: int | None = None) -> str:
    """Insert one ``integrity_alert`` row into ``outbox``; return its id.

    The outbox ``UNIQUE(event_id, subject)`` index makes this safe under
    accidental double-invocation within the same wall-clock second only
    if the caller supplies the same ``now`` — otherwise the generated
    ``event_id`` differs and a second row is inserted. That's desired:
    each distinct integrity failure should be an audit-worthy row.
    """
    ts = int(time.time()) if now is None else now
    outbox_id = str(uuid.uuid4())
    event_id = f"integrity-{ts}-{outbox_id[:8]}"
    payload = json.dumps(
        {
            "kind": "integrity_failure",
            "db_path": db_path,
            "errors": report.errors,
            "detected_at": ts,
        },
        sort_keys=True,
    )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO outbox (id, event_id, subject, payload) VALUES (?, ?, ?, ?)",
            (outbox_id, event_id, ALERT_SUBJECT, payload),
        )
        conn.commit()
    finally:
        conn.close()
    return outbox_id


def run(db_path: str) -> int:
    """Execute the check + alert flow; return the process exit code.

    Separated from :func:`main` so tests can drive the logic without
    reaching through :mod:`argparse`.
    """
    path = Path(db_path)
    if not path.exists():
        print(f"error: database not found at {db_path}", file=sys.stderr)
        return EXIT_SCRIPT_ERROR

    report = check_sqlite_integrity(str(path))
    if report.ok:
        print(f"ok: {db_path} passed PRAGMA integrity_check")
        return EXIT_OK

    try:
        outbox_id = enqueue_alert(str(path), report)
    except sqlite3.DatabaseError as exc:
        # Two cases land here: the outbox table is missing (fresh DB or
        # mid-migration -> OperationalError) or the DB is so malformed
        # that the INSERT itself crashes (DatabaseError). Either way we
        # cannot alert via the broken DB, so surface as a script error
        # for the wrapper to page the operator out-of-band.
        print(
            f"error: integrity_check failed but outbox insert raised: {exc}",
            file=sys.stderr,
        )
        return EXIT_SCRIPT_ERROR

    print(
        f"fail: {db_path} integrity_check reported {len(report.errors)} error(s); alert enqueued (outbox={outbox_id})"
    )
    for err in report.errors:
        print(f"  - {err}")
    return EXIT_INTEGRITY_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database to check (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    raise SystemExit(main())
