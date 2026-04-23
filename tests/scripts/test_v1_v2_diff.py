"""Tests for ``scripts/v1_v2_diff.py`` — the post-migration v1/v2 diff tool.

Strategy: build a synthetic v1 fixture via ``tests/fixtures/v1_sample``,
run the real migrator to produce a v2 DB, and then exercise the diff
against that pair. Happy-path asserts every check passes. Negative-path
tests poke a small hole in the v2 DB (delete a row, rename an id, inject
an orphan history row, inject an unresolved evidence reference) and
assert the diff surfaces the right failure in the right section.

All tests are stdlib-only (sqlite3 + pathlib + json). No network, no
pytest plugins needed beyond pytest itself.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import migrate_v1_to_v2 as migrate
from scripts import v1_v2_diff as diff
from tests.fixtures.v1_sample.builder import build_scaled_v1_sample


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _migrate_small(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small v1 fixture + run the migration; return (v1_dir, v2_db)."""
    v1_dir = tmp_path / "v1"
    build_scaled_v1_sample(
        v1_dir,
        events=25,
        contacts=5,
        places=3,
        subscriptions=2,
        tasks=4,
        signal_profiles=8,  # 6 kept + 2 dropped canonical
        preferences=3,
        feedback_log=7,
    )
    v2_db = tmp_path / "lifeos.db"
    migrate.run_migration(v1_dir, v2_db)
    return v1_dir, v2_db


@pytest.fixture
def migrated(tmp_path: Path) -> tuple[Path, Path]:
    return _migrate_small(tmp_path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestHappyPath:
    def test_all_checks_pass_on_clean_migration(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        report = diff.run_diff(v1_dir, v2_db, sample_size=5, seed=1)
        assert report.all_passed, [f"{c.name}: {c.detail}" for c in report.all_checks() if not c.passed]

    def test_row_count_check_names_cover_every_source_table(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        report = diff.run_diff(v1_dir, v2_db)
        names = {c.name for c in report.row_counts}
        assert names == {
            "events",
            "entities[kind=contact]",
            "entities[kind=place]",
            "entities[kind=subscription]",
            "moments[source=legacy_task]",
            "signal_profiles (kept types only)",
            "preferences",
            "feedback_events[source=v1_migration]",
        }

    def test_row_count_expected_and_actual_populated(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        report = diff.run_diff(v1_dir, v2_db)
        for c in report.row_counts:
            assert c.expected is not None, c.name
            assert c.actual is not None, c.name
            assert c.expected == c.actual, c.name

    def test_spot_checks_count_respects_sample_size(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=3, seed=99)
        assert len(r.spot_checks) == 3

    def test_spot_checks_capped_at_row_count(self, tmp_path: Path) -> None:
        v1_dir = tmp_path / "v1"
        build_scaled_v1_sample(
            v1_dir,
            events=2,
            contacts=1,
            places=1,
            subscriptions=1,
            tasks=1,
            signal_profiles=6,
            preferences=1,
            feedback_log=1,
        )
        v2_db = tmp_path / "lifeos.db"
        migrate.run_migration(v1_dir, v2_db)
        r = diff.run_diff(v1_dir, v2_db, sample_size=100, seed=0)
        assert len(r.spot_checks) == 2  # capped at row count

    def test_spot_checks_all_passed_flags_have_clean_detail(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=5, seed=7)
        for c in r.spot_checks:
            assert c.passed, f"{c.name}: {c.detail}"
            assert "match" in c.detail

    def test_fk_integrity_three_classes(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db)
        names = [c.name for c in r.fk_integrity]
        assert names == [
            "moment_state_history.moment_id → moments.id",
            "event_tags.event_id → events.id",
            "moments.evidence[*].event_id → events.id",
        ]
        for c in r.fk_integrity:
            assert c.passed, f"{c.name}: {c.detail}"

    def test_evidence_fk_check_reports_zero_refs_on_legacy_task_moments(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db)
        evidence_check = next(c for c in r.fk_integrity if c.name.startswith("moments.evidence"))
        assert evidence_check.passed
        assert "0 evidence references" in evidence_check.detail


# ---------------------------------------------------------------------------
# Determinism / sampling
# ---------------------------------------------------------------------------
class TestSampling:
    def test_sample_event_ids_is_seeded_and_sorted(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, _ = migrated
        with diff._open_ro(v1_dir / "events.db") as conn:
            a = diff.sample_event_ids(conn, sample_size=5, seed=42)
            b = diff.sample_event_ids(conn, sample_size=5, seed=42)
            c = diff.sample_event_ids(conn, sample_size=5, seed=43)
        assert a == b  # deterministic given seed
        assert a != c  # differing seed → differing sample
        assert a == sorted(a)  # sample is returned sorted

    def test_sample_event_ids_empty_db(self, tmp_path: Path) -> None:
        path = tmp_path / "events.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE events (id TEXT PRIMARY KEY, type TEXT, source TEXT, timestamp TEXT, payload TEXT)"
            )
        with diff._open_ro(path) as conn:
            assert diff.sample_event_ids(conn, sample_size=10, seed=0) == []

    def test_sample_size_zero_emits_no_spot_checks(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=0, seed=0)
        assert r.spot_checks == []


# ---------------------------------------------------------------------------
# Negative — row count mismatches
# ---------------------------------------------------------------------------
class TestRowCountMismatches:
    def test_deleted_event_fails_event_row_count(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute("DELETE FROM events WHERE id='evt-0000000'")
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        failures = [c for c in r.row_counts if not c.passed]
        assert any(c.name == "events" for c in failures)
        evt = next(c for c in r.row_counts if c.name == "events")
        assert evt.expected == 25
        assert evt.actual == 24
        assert not r.all_passed

    def test_deleted_contact_fails_only_contact_kind(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute("DELETE FROM entities WHERE rowid IN (SELECT rowid FROM entities WHERE kind='contact' LIMIT 1)")
        r = diff.run_diff(v1_dir, v2_db)
        failing = {c.name for c in r.row_counts if not c.passed}
        assert failing == {"entities[kind=contact]"}

    def test_deleted_task_fails_moments_legacy_task(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute("PRAGMA foreign_keys=ON")
            # CASCADE DELETEs the state history row too.
            c.execute(
                "DELETE FROM moments WHERE rowid IN "
                "(SELECT rowid FROM moments "
                "WHERE source_insight_type='legacy_task' LIMIT 1)"
            )
        r = diff.run_diff(v1_dir, v2_db)
        assert any(c.name == "moments[source=legacy_task]" and not c.passed for c in r.row_counts)

    def test_deleted_feedback_event_fails_feedback_row_count(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute(
                "DELETE FROM feedback_events WHERE rowid IN "
                "(SELECT rowid FROM feedback_events "
                "WHERE source='v1_migration' LIMIT 1)"
            )
        r = diff.run_diff(v1_dir, v2_db)
        assert any(c.name == "feedback_events[source=v1_migration]" and not c.passed for c in r.row_counts)


# ---------------------------------------------------------------------------
# Negative — spot-check divergence
# ---------------------------------------------------------------------------
class TestSpotCheckMismatches:
    def test_missing_v2_event_fails_spot_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        # Pick the id we know the seeded sample will select, delete from v2,
        # then we expect the corresponding spot-check to fail with
        # "missing from v2".
        with diff._open_ro(v1_dir / "events.db") as c:
            ids = diff.sample_event_ids(c, sample_size=1, seed=0)
        assert len(ids) == 1
        target = ids[0]
        # Delete only the event (ignore row-count-check side effects for now).
        with sqlite3.connect(v2_db) as c:
            c.execute("DELETE FROM events WHERE id=?", (target,))
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        sp = r.spot_checks[0]
        assert sp.name == f"event {target}"
        assert not sp.passed
        assert "missing from v2" in sp.detail

    def test_type_mismatch_fails_spot_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with diff._open_ro(v1_dir / "events.db") as c:
            ids = diff.sample_event_ids(c, sample_size=1, seed=0)
        target = ids[0]
        with sqlite3.connect(v2_db) as c:
            c.execute("UPDATE events SET type='mutated.type' WHERE id=?", (target,))
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        sp = r.spot_checks[0]
        assert not sp.passed
        assert "type: " in sp.detail

    def test_payload_mismatch_fails_spot_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with diff._open_ro(v1_dir / "events.db") as c:
            ids = diff.sample_event_ids(c, sample_size=1, seed=0)
        target = ids[0]
        with sqlite3.connect(v2_db) as c:
            c.execute(
                "UPDATE events SET payload=? WHERE id=?",
                (json.dumps({"mutated": True}), target),
            )
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        sp = r.spot_checks[0]
        assert not sp.passed
        assert "payload JSON differs" in sp.detail

    def test_timestamp_divergence_fails_spot_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with diff._open_ro(v1_dir / "events.db") as c:
            ids = diff.sample_event_ids(c, sample_size=1, seed=0)
        target = ids[0]
        with sqlite3.connect(v2_db) as c:
            c.execute("UPDATE events SET timestamp=? WHERE id=?", (1, target))
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        sp = r.spot_checks[0]
        assert not sp.passed
        assert "timestamp" in sp.detail

    def test_canonical_json_survives_whitespace_differences(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with diff._open_ro(v1_dir / "events.db") as c:
            ids = diff.sample_event_ids(c, sample_size=1, seed=0)
        target = ids[0]
        # Re-encode the payload with different whitespace — canonical
        # JSON comparison should still pass.
        with sqlite3.connect(v2_db) as c:
            row = c.execute("SELECT payload FROM events WHERE id=?", (target,)).fetchone()
            reencoded = json.dumps(json.loads(row[0]), indent=2, separators=(", ", ": "))
            c.execute("UPDATE events SET payload=? WHERE id=?", (reencoded, target))
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        assert r.spot_checks[0].passed


# ---------------------------------------------------------------------------
# Negative — FK integrity
# ---------------------------------------------------------------------------
class TestFkIntegrity:
    def test_orphan_event_tag_fails_fk_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        # Direct-SQL insert into event_tags bypassing the FK — simulates
        # corruption we need to detect. FKs are off by default on fresh
        # connections, so this succeeds.
        with sqlite3.connect(v2_db) as c:
            c.execute(
                "INSERT INTO event_tags (event_id, tag, value) VALUES (?, ?, ?)",
                ("does-not-exist", "tag", "v"),
            )
        r = diff.run_diff(v1_dir, v2_db)
        tag_check = next(c for c in r.fk_integrity if "event_tags" in c.name)
        assert not tag_check.passed
        assert "1 orphan rows" in tag_check.detail

    def test_orphan_moment_state_history_fails_fk_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute(
                "INSERT INTO moment_state_history "
                "(moment_id, from_state, to_state, ts, annotation) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ghost-moment", None, "suggested", 0, "injected for test"),
            )
        r = diff.run_diff(v1_dir, v2_db)
        hist_check = next(c for c in r.fk_integrity if "moment_state_history" in c.name)
        assert not hist_check.passed
        assert "1 orphan rows" in hist_check.detail

    def test_unresolved_evidence_event_id_fails_fk_check(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            # Pick any one moment and point its evidence at a non-existent event.
            row = c.execute("SELECT id FROM moments LIMIT 1").fetchone()
            moment_id = row[0]
            evidence = json.dumps([{"event_id": "evt-does-not-exist", "kind": "email"}])
            c.execute("UPDATE moments SET evidence=? WHERE id=?", (evidence, moment_id))
        r = diff.run_diff(v1_dir, v2_db)
        ev_check = next(c for c in r.fk_integrity if c.name.startswith("moments.evidence"))
        assert not ev_check.passed
        assert "unresolved" in ev_check.detail
        assert "evt-does-not-exist" in ev_check.detail

    def test_resolved_evidence_event_id_passes(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            # Point evidence at a real event id — should resolve cleanly.
            moment_id = c.execute("SELECT id FROM moments LIMIT 1").fetchone()[0]
            event_id = c.execute("SELECT id FROM events LIMIT 1").fetchone()[0]
            c.execute(
                "UPDATE moments SET evidence=? WHERE id=?",
                (json.dumps([{"event_id": event_id}]), moment_id),
            )
        r = diff.run_diff(v1_dir, v2_db)
        ev_check = next(c for c in r.fk_integrity if c.name.startswith("moments.evidence"))
        assert ev_check.passed
        assert "1 refs" in ev_check.detail

    def test_unparseable_evidence_json_reported(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            moment_id = c.execute("SELECT id FROM moments LIMIT 1").fetchone()[0]
            c.execute(
                "UPDATE moments SET evidence=? WHERE id=?",
                ("not-valid-json{", moment_id),
            )
        r = diff.run_diff(v1_dir, v2_db)
        ev_check = next(c for c in r.fk_integrity if c.name.startswith("moments.evidence"))
        assert not ev_check.passed
        assert "unparseable" in ev_check.detail


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------
class TestRender:
    def test_markdown_contains_expected_sections(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=3, seed=0)
        md = diff.render_markdown(r)
        for heading in (
            "# v1 → v2 cutover diff",
            "## Summary",
            "## Row counts",
            "## Spot checks",
            "## FK integrity",
        ):
            assert heading in md, f"missing heading {heading!r}"

    def test_markdown_marks_pass_on_happy_path(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=3, seed=0)
        md = diff.render_markdown(r)
        assert "Result: **PASS**" in md
        assert "## Failures" not in md

    def test_markdown_marks_fail_and_lists_failures(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute("DELETE FROM events WHERE id='evt-0000000'")
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        md = diff.render_markdown(r)
        assert "Result: **FAIL**" in md
        assert "## Failures" in md
        assert "events" in md

    def test_write_report_creates_output_file(self, migrated: tuple[Path, Path], tmp_path: Path) -> None:
        v1_dir, v2_db = migrated
        r = diff.run_diff(v1_dir, v2_db, sample_size=1, seed=0)
        out = tmp_path / "reports" / "today.md"
        diff.write_report(r, out)
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("# v1 → v2 cutover diff")


# ---------------------------------------------------------------------------
# Missing snapshot (partial / degraded runs)
# ---------------------------------------------------------------------------
class TestDegradedInputs:
    def test_missing_v1_db_degrades_to_note(self, migrated: tuple[Path, Path]) -> None:
        v1_dir, v2_db = migrated
        # Remove events.db from the snapshot dir — the diff should skip
        # the events row-count check and the spot-check, and record a
        # NOTE instead of failing.
        (v1_dir / "events.db").unlink()
        r = diff.run_diff(v1_dir, v2_db, sample_size=5, seed=0)
        names = {c.name for c in r.row_counts}
        assert "events" not in names
        assert r.spot_checks == []
        assert any("events.db" in n for n in r.notes)
        # Other checks still run + pass.
        assert all(c.passed for c in r.fk_integrity)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCli:
    def test_main_exit_pass_on_clean(self, migrated: tuple[Path, Path], tmp_path: Path) -> None:
        v1_dir, v2_db = migrated
        out = tmp_path / "diff.md"
        rc = diff.main(
            [
                "--v1-dir",
                str(v1_dir),
                "--v2-db",
                str(v2_db),
                "--output",
                str(out),
                "--sample-size",
                "3",
                "--seed",
                "1",
            ]
        )
        assert rc == diff.EXIT_PASS
        assert out.exists()

    def test_main_exit_fail_on_data_loss(self, migrated: tuple[Path, Path], tmp_path: Path) -> None:
        v1_dir, v2_db = migrated
        with sqlite3.connect(v2_db) as c:
            c.execute("DELETE FROM events WHERE id='evt-0000001'")
        out = tmp_path / "diff.md"
        rc = diff.main(["--v1-dir", str(v1_dir), "--v2-db", str(v2_db), "--output", str(out)])
        assert rc == diff.EXIT_FAIL

    def test_main_exit_badinput_on_missing_v1_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "diff.md"
        # v2 db exists, v1 dir does not
        v2_db = tmp_path / "v2.db"
        with sqlite3.connect(v2_db) as c:
            c.execute("CREATE TABLE events (id TEXT PRIMARY KEY)")
        rc = diff.main(
            [
                "--v1-dir",
                str(tmp_path / "nope"),
                "--v2-db",
                str(v2_db),
                "--output",
                str(out),
            ]
        )
        assert rc == diff.EXIT_BADINPUT

    def test_main_exit_badinput_on_missing_v2_db(self, migrated: tuple[Path, Path], tmp_path: Path) -> None:
        v1_dir, _ = migrated
        out = tmp_path / "diff.md"
        rc = diff.main(
            [
                "--v1-dir",
                str(v1_dir),
                "--v2-db",
                str(tmp_path / "missing.db"),
                "--output",
                str(out),
            ]
        )
        assert rc == diff.EXIT_BADINPUT

    def test_main_exit_badinput_on_negative_sample_size(self, migrated: tuple[Path, Path], tmp_path: Path) -> None:
        v1_dir, v2_db = migrated
        rc = diff.main(
            [
                "--v1-dir",
                str(v1_dir),
                "--v2-db",
                str(v2_db),
                "--output",
                str(tmp_path / "d.md"),
                "--sample-size",
                "-1",
            ]
        )
        assert rc == diff.EXIT_BADINPUT
