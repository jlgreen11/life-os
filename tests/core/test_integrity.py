"""Tests for :mod:`core.integrity`.

Covers the two cases the task specifies: a healthy DB reports ``ok=True``;
a corrupted DB reports ``ok=False`` with non-empty errors. Corruption is
synthesized by clobbering bytes inside an existing SQLite file — the
standard trick documented in the SQLite integrity-test suite — rather
than mocking, so the test actually exercises the ``PRAGMA`` round-trip.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.integrity import IntegrityReport, check_sqlite_integrity
from storage import schema


@pytest.fixture
def fresh_db(tmp_path):
    """A fresh v2-schema SQLite file on disk (NOT in-memory).

    ``integrity_check`` works on both memory and file DBs, but corruption
    tests need a file we can overwrite, so we match that shape here too.
    """
    db_path = tmp_path / "lifeos.db"
    conn = sqlite3.connect(db_path)
    try:
        for stmt in schema.get_all_ddl():
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_integrity_report_defaults():
    """Dataclass defaults keep ``errors`` as an empty list, not ``None``."""
    report = IntegrityReport(ok=True)
    assert report.ok is True
    assert report.errors == []


def test_integrity_report_errors_isolated_between_instances():
    """Default-factory list is per-instance; appending to one does not leak."""
    a = IntegrityReport(ok=False)
    b = IntegrityReport(ok=False)
    a.errors.append("row 17 missing from index sqlite_autoindex_moments_1")
    assert b.errors == []


def test_healthy_db_returns_ok(fresh_db):
    """A freshly-initialized v2 DB passes ``PRAGMA integrity_check``."""
    report = check_sqlite_integrity(str(fresh_db))
    assert report.ok is True
    assert report.errors == []


def test_healthy_db_with_populated_rows_returns_ok(fresh_db):
    """Integrity stays ok after inserting rows through normal SQL."""
    conn = sqlite3.connect(fresh_db)
    try:
        conn.execute(
            "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
            ("e1", "test.event", "unit-test", 1700000000, "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    report = check_sqlite_integrity(str(fresh_db))
    assert report.ok is True
    assert report.errors == []


def test_corrupted_db_returns_errors(fresh_db):
    """Byte-level corruption surfaces as ``ok=False`` with error strings.

    We insert a few rows first so there is paged content to damage,
    then overwrite a span well past the header (which SQLite would
    refuse to open at all) but inside the first few page bodies.
    """
    conn = sqlite3.connect(fresh_db)
    try:
        for i in range(50):
            conn.execute(
                "INSERT INTO events (id, type, source, timestamp, payload) VALUES (?, ?, ?, ?, ?)",
                (f"e{i}", "test.event", "unit-test", 1700000000 + i, "{}"),
            )
        conn.commit()
    finally:
        conn.close()

    # Clobber a page of content mid-file. The first 100 bytes are the
    # SQLite header; page 1 body starts after that. Damaging page 2+
    # (offset >= 4096 on default page size) is reliably detected by
    # integrity_check without preventing the DB from opening.
    with open(fresh_db, "r+b") as f:
        f.seek(4096)
        f.write(b"\x00" * 512)

    report = check_sqlite_integrity(str(fresh_db))
    assert report.ok is False
    assert len(report.errors) >= 1
    # None of the rows should be the literal string "ok".
    assert "ok" not in report.errors


def test_healthy_inmemory_db_returns_ok(tmp_path):
    """Helper-path sanity: an empty freshly-created file is also ok.

    ``sqlite3.connect`` will create ``db_path`` if missing; that new file
    is a valid (empty) SQLite DB, and should report ok. This is important
    because :mod:`scripts.daily_integrity_check` explicitly guards against
    missing files — ``check_sqlite_integrity`` itself does not.
    """
    db_path = tmp_path / "never_touched.db"
    report = check_sqlite_integrity(str(db_path))
    assert report.ok is True
    assert report.errors == []
