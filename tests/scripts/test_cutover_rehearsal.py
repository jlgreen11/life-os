"""Tests for ``scripts/cutover_rehearsal.py``.

Builds synthetic v1 backups in ``tmp_path`` with the minimal v1 DDL the
migrator reads (events + state.db.tasks). Exercises:

- :func:`discover_backup` — newest-by-name selection and empty-data-dir.
- The no-backup branch of :func:`main` — writes a SKIPPED stub report
  and exits ``0`` so the autonomous orchestrator does not flag it.
- :func:`run_rehearsal` happy path — all three checks pass / N/A on a
  synthetic backup with no Moments populated by producers.
- :func:`check_moment_evidence` failure path — a Moment whose evidence
  references a missing event id is reported as a dangling reference.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import cutover_rehearsal as rehearsal


# ---------------------------------------------------------------------------
# Synthetic v1 backup builders. Minimal DDL — only what the migrator reads.
# ---------------------------------------------------------------------------
def _build_minimal_v1_backup(backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(backup_dir / "events.db") as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                id              TEXT PRIMARY KEY,
                type            TEXT NOT NULL,
                source          TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                priority        TEXT NOT NULL DEFAULT 'normal',
                payload         TEXT NOT NULL DEFAULT '{}',
                metadata        TEXT NOT NULL DEFAULT '{}',
                embedding_id    TEXT,
                created_at      TEXT NOT NULL DEFAULT '2026-04-01T00:00:00.000Z'
            );
            """
        )
        for i in range(5):
            conn.execute(
                """
                INSERT INTO events
                    (id, type, source, timestamp, priority, payload, metadata, embedding_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evt-{i:03d}",
                    "email.received",
                    "proton_mail",
                    f"2026-04-{i + 1:02d}T12:00:00.000Z",
                    "normal",
                    json.dumps({"subject": f"hello {i}"}),
                    "{}",
                    None,
                    "2026-04-01T00:00:00.000Z",
                ),
            )

    with sqlite3.connect(backup_dir / "state.db") as conn:
        conn.executescript(
            """
            CREATE TABLE tasks (
                id                  TEXT PRIMARY KEY,
                title               TEXT NOT NULL,
                description         TEXT,
                source              TEXT,
                source_event_id     TEXT,
                source_context      TEXT,
                domain              TEXT,
                priority            TEXT,
                tags                TEXT,
                due_date            TEXT,
                reminder_at         TEXT,
                estimated_minutes   INTEGER,
                related_contacts    TEXT DEFAULT '[]',
                related_files       TEXT,
                related_events      TEXT,
                depends_on          TEXT,
                status              TEXT DEFAULT 'pending',
                completed_at        TEXT,
                created_at          TEXT,
                updated_at          TEXT
            );
            """
        )
        for i in range(2):
            conn.execute(
                """
                INSERT INTO tasks (id, title, description, source, priority, related_contacts, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"task-{i:03d}",
                    f"Task {i}",
                    "desc",
                    "ai",
                    "normal",
                    json.dumps([]),
                    "2026-04-10T08:00:00.000Z",
                ),
            )


# ---------------------------------------------------------------------------
# discover_backup
# ---------------------------------------------------------------------------
def test_discover_backup_returns_none_when_data_dir_missing(tmp_path: Path) -> None:
    assert rehearsal.discover_backup(tmp_path / "absent") is None


def test_discover_backup_returns_none_when_no_backups(tmp_path: Path) -> None:
    (tmp_path / "v2-runs").mkdir()  # unrelated subdir
    assert rehearsal.discover_backup(tmp_path) is None


def test_discover_backup_returns_newest_by_name(tmp_path: Path) -> None:
    (tmp_path / "backup-20260101").mkdir()
    (tmp_path / "backup-20260420").mkdir()
    (tmp_path / "backup-20260315").mkdir()
    chosen = rehearsal.discover_backup(tmp_path)
    assert chosen is not None
    assert chosen.name == "backup-20260420"


# ---------------------------------------------------------------------------
# main: no-backup path writes a stub report
# ---------------------------------------------------------------------------
def test_main_with_no_backup_writes_stub_and_exits_zero(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "docs" / "cutover-rehearsals"

    rc = rehearsal.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == rehearsal.EXIT_PASS
    stubs = list(output_dir.glob("*-skipped.md"))
    assert len(stubs) == 1
    body = stubs[0].read_text(encoding="utf-8")
    assert "SKIPPED" in body
    assert "DRY-RUN ONLY" in body


def test_main_with_explicit_missing_source_dir_returns_fail(tmp_path: Path) -> None:
    rc = rehearsal.main(
        [
            "--source-dir",
            str(tmp_path / "absent"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == rehearsal.EXIT_FAIL


# ---------------------------------------------------------------------------
# run_rehearsal: happy path
# ---------------------------------------------------------------------------
def test_run_rehearsal_happy_path_writes_report(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup-20260420"
    _build_minimal_v1_backup(backup_dir)
    output_dir = tmp_path / "docs" / "cutover-rehearsals"

    report = rehearsal.run_rehearsal(backup_dir, output_dir)

    assert report.overall == "PASS"
    assert {c.name for c in report.checks} == {
        "row-count diff",
        "moment.evidence integrity",
        "vector-store integrity",
    }
    # No lance dir → vector-store check is N/A, not FAIL.
    vector = next(c for c in report.checks if c.name == "vector-store integrity")
    assert vector.status == "N/A"
    # Markdown report exists.
    md_files = list(output_dir.glob("*.md"))
    assert len(md_files) == 1
    body = md_files[0].read_text(encoding="utf-8")
    assert "DRY-RUN ONLY" in body
    assert "row-count diff" in body
    assert report.runtime_seconds >= 0
    assert report.output_db_bytes > 0


def test_main_happy_path_returns_zero(tmp_path: Path) -> None:
    backup_dir = tmp_path / "data" / "backup-20260420"
    _build_minimal_v1_backup(backup_dir)
    output_dir = tmp_path / "out"

    rc = rehearsal.main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == rehearsal.EXIT_PASS


# ---------------------------------------------------------------------------
# check_moment_evidence: dangling reference fails the check
# ---------------------------------------------------------------------------
def test_check_moment_evidence_passes_when_evidence_resolves(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    rehearsal.run_migration(backup_dir, out_db)
    # Patch a moment evidence to reference a real event.
    with sqlite3.connect(out_db) as conn:
        conn.execute(
            "UPDATE moments SET evidence = ? WHERE id = ?",
            (json.dumps(["evt-000"]), "task-000"),
        )
        conn.commit()
    result = rehearsal.check_moment_evidence(out_db)
    assert result.status == "PASS"


def test_check_moment_evidence_fails_on_dangling_reference(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    rehearsal.run_migration(backup_dir, out_db)
    with sqlite3.connect(out_db) as conn:
        conn.execute(
            "UPDATE moments SET evidence = ? WHERE id = ?",
            (json.dumps(["evt-does-not-exist"]), "task-000"),
        )
        conn.commit()
    result = rehearsal.check_moment_evidence(out_db)
    assert result.status == "FAIL"
    assert any("dangling" in d for d in result.details)


# ---------------------------------------------------------------------------
# check_vector_store: N/A when no lance dir
# ---------------------------------------------------------------------------
def test_check_vector_store_na_when_lance_dir_absent(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    rehearsal.run_migration(backup_dir, out_db)
    result = rehearsal.check_vector_store(backup_dir, out_db)
    assert result.status == "N/A"
    assert any("no lance directory" in d for d in result.details)


def test_check_vector_store_reports_size_when_lance_present(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    rehearsal.run_migration(backup_dir, out_db)
    # Drop a fake lance directory with a non-empty file so the size > 0
    # branch is exercised even when lancedb isn't importable.
    lance_dir = backup_dir / "lance"
    lance_dir.mkdir()
    (lance_dir / "data.lance").write_bytes(b"\x00" * 64)
    result = rehearsal.check_vector_store(backup_dir, out_db)
    # Without lancedb installed in this env we expect N/A; with lancedb, the
    # fake table won't open and we'd get FAIL. Either way the check completes.
    assert result.status in {"N/A", "FAIL"}
    assert any(str(lance_dir) in d for d in result.details)


# ---------------------------------------------------------------------------
# check_row_counts surfaces FK violations and source-vs-translated drift
# ---------------------------------------------------------------------------
def test_check_row_counts_passes_on_clean_migration(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    report = rehearsal.run_migration(backup_dir, out_db)
    result = rehearsal.check_row_counts(backup_dir, out_db, report)
    assert result.status == "PASS"
    assert any("foreign_key_check: 0 violations" in d for d in result.details)


def test_check_row_counts_fails_when_table_count_mismatches(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    out_db = tmp_path / "v2.db"
    report = rehearsal.run_migration(backup_dir, out_db)
    # Introduce drift: delete one event so the COUNT(*) no longer matches
    # report.events.translated. PRAGMA foreign_keys=OFF locally so we don't
    # cascade the deletion through any FK; the row-count mismatch is the
    # signal we're testing.
    with sqlite3.connect(out_db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM events WHERE id = 'evt-000'")
        conn.commit()
    result = rehearsal.check_row_counts(backup_dir, out_db, report)
    assert result.status == "FAIL"


# ---------------------------------------------------------------------------
# Report writer surfaces overall + the DRY-RUN banner
# ---------------------------------------------------------------------------
def test_write_report_includes_dry_run_banner_and_overall(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    _build_minimal_v1_backup(backup_dir)
    output_dir = tmp_path / "out"
    report = rehearsal.run_rehearsal(backup_dir, output_dir)
    body = next(output_dir.glob("*.md")).read_text(encoding="utf-8")
    assert "DRY-RUN ONLY" in body
    assert f"overall: **{report.overall}**" in body
    assert "## Migration report" in body
    assert "## Checks" in body


# ---------------------------------------------------------------------------
# Smoke: rehearsal works even when the v1 backup has only events.db
# ---------------------------------------------------------------------------
def test_rehearsal_with_only_events_db(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup-tiny"
    backup_dir.mkdir()
    with sqlite3.connect(backup_dir / "events.db") as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'normal',
                payload TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding_id TEXT,
                created_at TEXT NOT NULL DEFAULT '2026-04-01T00:00:00.000Z'
            );
            """
        )
        conn.execute(
            """INSERT INTO events
                (id, type, source, timestamp, priority, payload, metadata, embedding_id, created_at)
               VALUES ('e1', 'x', 'y', '2026-04-01T00:00:00.000Z', 'normal', '{}', '{}', NULL,
                       '2026-04-01T00:00:00.000Z')"""
        )
    output_dir = tmp_path / "out"
    report = rehearsal.run_rehearsal(backup_dir, output_dir)
    assert report.overall == "PASS"


@pytest.fixture(autouse=True)
def _no_lancedb_in_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the vector-store check exercises its degraded path.

    The autonomous-agent dev box does not install lancedb, but a developer's
    local env might. Forcing ``import lancedb`` to fail keeps the test
    behaviour stable across machines.
    """
    import builtins

    real_import = builtins.__import__

    def _block_lancedb(name: str, *args: object, **kwargs: object) -> object:
        if name == "lancedb":
            raise ImportError("lancedb blocked by test fixture")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _block_lancedb)
