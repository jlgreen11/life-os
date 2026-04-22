"""SQLite integrity check — critical gap from the v2 engineering review.

Runs ``PRAGMA integrity_check`` against the v2 ``lifeos.db`` file and returns
a structured :class:`IntegrityReport`. A healthy database reports a single
``"ok"`` result row; any other output is surfaced as an error list so the
daily cron (``scripts/daily_integrity_check.py``) can enqueue an outbox alert
without interpreting SQLite's free-form diagnostic strings.

The pragma runs under a short-lived connection opened read-write. The
documented SQLite behaviour is that ``integrity_check`` does not modify the
database; it may allocate temporary pages in the journal but is safe to run
concurrently with a live system. We deliberately avoid a URI ``mode=ro``
because a corrupted header can refuse to open read-only on some platforms
(see SQLite bug history), and the caller owns retry/alerting.

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "Reviewer Concerns (resolved)".
- SQLite docs: https://www.sqlite.org/pragma.html#pragma_integrity_check
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class IntegrityReport:
    """Structured result of a ``PRAGMA integrity_check`` run.

    ``ok`` is ``True`` only when SQLite reports the exact single-row
    ``["ok"]`` result. Any other output — including the empty set, which
    SQLite does not produce but we treat conservatively — is surfaced as
    a non-empty ``errors`` list with the raw pragma strings so operators
    can grep the outbox payload for known signatures.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)


def check_sqlite_integrity(db_path: str) -> IntegrityReport:
    """Run ``PRAGMA integrity_check`` on ``db_path`` and return a report.

    Opens ``db_path`` with a short-lived :class:`sqlite3.Connection`,
    executes the pragma, and closes the connection regardless of
    outcome. The pragma's return is a list of one or more rows; the
    single-row value ``"ok"`` means the database is healthy, anything
    else is an error string describing the corruption.

    If the file is so damaged that either ``connect`` or the pragma
    raises :class:`sqlite3.DatabaseError` (for instance, ``"database
    disk image is malformed"``), we treat that as the strongest possible
    corruption signal and return ``ok=False`` with the exception text
    as the single error entry. The cron wrapper then routes that to the
    outbox alert path — reporters that need to distinguish "pragma ran
    and reported errors" from "pragma itself crashed" can inspect the
    error strings.

    ``sqlite3.connect`` will *create* the file if it does not exist,
    which would mask missing-DB bugs in the cron. Callers that care
    about file existence should check it before invoking this function
    (``scripts/daily_integrity_check.py`` does).
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.DatabaseError as exc:
        return IntegrityReport(ok=False, errors=[f"connect failed: {exc}"])

    try:
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            return IntegrityReport(ok=False, errors=[f"integrity_check raised: {exc}"])
    finally:
        conn.close()

    results = [row[0] for row in rows]
    if len(results) == 1 and results[0] == "ok":
        return IntegrityReport(ok=True, errors=[])
    return IntegrityReport(ok=False, errors=results)


__all__ = ["IntegrityReport", "check_sqlite_integrity"]
