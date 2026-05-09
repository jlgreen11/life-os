"""End-to-end dry-run cutover harness.

Stitches together every scripted piece of the cutover runbook into a
single in-process integration test:

1. Fresh v1 fixture built by :mod:`tests.fixtures.v1_sample.builder`.
2. Snapshot of the v1 data dir (mirrors runbook § 1.3 backup step).
3. Migration via :func:`scripts.migrate_v1_to_v2.run_migration`.
4. v2 FastAPI app brought up via :class:`fastapi.testclient.TestClient`.
5. ``GET /api/health`` smoke test against a stub probe wired to the
   freshly migrated SQLite file.
6. A test Moment is created in the SNOOZED state with a past
   ``scheduled_for``; :meth:`core.moment.scheduler.Scheduler.tick`
   wakes it to SUGGESTED, and ``GET /api/now`` confirms the transition.
7. v2 is "shut down" (client + connection closed).
8. :func:`scripts.cutover_rollback.execute_rollback` is driven through a
   scripted :class:`Runner` double that substitutes real filesystem ops
   for the OS-level side effects (no docker, no real HTTP) — proving the
   rollback plan orders steps the way the runbook expects.
9. The restored data dir contains the original v1 SQLite files, byte-for-byte.

Acceptance (NEXT_TASKS § "Dry-run cutover CI harness"):

- completes in < 2 minutes in CI; we assert a wall-clock ceiling on the
  happy path.
- every scripted invariant passes: migration row counts, health ok,
  scheduler transition, rollback step order, restored file hashes.

Why no real docker/HTTP? The task is explicitly "dry-run": every
OS-level side effect (SIGTERM, ``docker compose``, v1 health endpoint)
is gated behind :class:`scripts.cutover_rollback.Runner`, which is how
the rollback unit tests already exercise the plan. A CI harness that
actually tore down containers would be a different test class and
would blow past the 2-minute budget on the first ``docker compose
start`` cold boot.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.moment.feedback_weight import FeedbackWeightStore
from core.moment.scheduler import Scheduler
from core.moment.types import Action, ActionKind, InsightType, Moment, MomentState
from scripts import cutover_rollback as rollback
from scripts import migrate_v1_to_v2 as migrate
from storage.repos.moments import MomentRepository
from storage.repos.outbox import OutboxRepository
from tests.fixtures.v1_sample.builder import SampleCounts, build_scaled_v1_sample

# Fixed reference epoch — 2026-04-22T12:00:00Z. Keeps the scheduler
# transition deterministic across hosts.
REF_NOW = 1_777_204_800

# Scale knob for the harness. Deliberately tiny so the full round-trip
# finishes in well under the 2-minute CI budget. The scale-rehearsal
# test (:mod:`tests.scripts.test_migrate_v1_to_v2_scale`) owns the
# production-size fixture; here we only need enough rows to prove the
# plumbing is wired.
_SMALL_COUNTS = {
    "events": 50,
    "contacts": 10,
    "places": 5,
    "subscriptions": 5,
    "tasks": 5,
    "signal_profiles": 10,
    "preferences": 3,
    "feedback_log": 5,
}

V2_DB_NAME = "lifeos.db"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Clock:
    """Fixed-time clock passed into the repository + scheduler.

    Kept mutable so the scheduler path that fires a SNOOZED moment can
    advance time if a future test needs it. The current tests pin it
    to :data:`REF_NOW` so ``scheduled_for`` comparisons are stable.
    """

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _StubHealthProbe:
    """Minimal health probe — good enough for ``/api/health`` to return 200.

    The real :class:`life_os.health.HealthProbe` (Phase 1 Week 10) queries
    the database, scheduler, and connectors directly; here we only need
    to prove the route is wired and returns the expected multi-key
    payload shape.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ts": REF_NOW,
            "connectors": {"proton_mail": "ok"},
            "db_last_write_ts": REF_NOW - 10,
            "scheduler_heartbeat_ts": REF_NOW - 5,
            "producer_activity": {},
            "pending_moments": 0,
            "notes": [f"db={self._db_path.name}"],
        }


class _LifeOSDouble:
    """In-test ``life_os`` double the v2 routes dereference.

    Mirrors the shape the Now/You/People/Settings/Health routes expect:
    ``moment_repo`` + ``feedback_weight_store`` + ``health_probe`` plus
    a ``config`` dict for CORS resolution. ``outbox_repo`` is also
    attached so the undo/accept grace-window path stays functional if
    this harness grows to exercise it.
    """

    def __init__(
        self,
        *,
        moment_repo: Any,
        feedback_weight_store: Any,
        outbox_repo: Any,
        health_probe: Any,
    ) -> None:
        self.config: dict[str, Any] = {}
        self.moment_repo = moment_repo
        self.feedback_weight_store = feedback_weight_store
        self.outbox_repo = outbox_repo
        self.health_probe = health_probe
        self.metrics = None


class _ScriptedRunner:
    """Scripted :class:`scripts.cutover_rollback.Runner` for the test harness.

    - ``move_file`` / ``copy_db_files`` / ``copy_lance_dirs`` run the
      real filesystem ops so the restored data dir matches the snapshot
      byte-for-byte.
    - ``kill_process`` / ``docker_compose`` / ``http_check_ok`` are
      no-ops that only record the call log — proving the plan issues
      them in the right order without requiring docker or a live v1.

    The recorded ``calls`` list is inspected by the tests to lock the
    step ordering; the real :class:`RealRunner` is exercised by the
    existing ``tests/scripts/test_cutover_rollback.py`` unit suite.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def kill_process(self, pid: int) -> None:
        self.calls.append(("kill", (pid,)))

    def move_file(self, src: Path, dst: Path) -> None:
        self.calls.append(("move", (Path(src), Path(dst))))
        shutil.move(str(src), str(dst))

    def copy_db_files(self, sources: list[Path], dest_dir: Path) -> None:
        self.calls.append(("copy_db", tuple(Path(s) for s in sources)))
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copy2(src, dest_dir / Path(src).name)

    def copy_lance_dirs(self, sources: list[Path], dest_dir: Path) -> None:
        self.calls.append(("copy_lance", tuple(Path(s) for s in sources)))
        # Harness fixture has no LanceDB; intentionally a no-op when empty.

    def docker_compose(self, action: str, service: str) -> None:
        self.calls.append(("compose", (action, service)))

    def http_check_ok(self, url: str, timeout: float) -> bool:
        self.calls.append(("verify", (url,)))
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_data_dir(data_dir: Path, snapshot_dir: Path) -> None:
    """Copy every ``*.db`` from ``data_dir`` into ``snapshot_dir``.

    Mirrors the runbook § 1.3 pre-flight backup (``cp data/*.db
    data/backup-<ts>/``). LanceDB dirs are not part of the harness
    fixture so the copy is SQLite-only.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for db in sorted(data_dir.glob("*.db")):
        shutil.copy2(db, snapshot_dir / db.name)


def _make_snoozed_moment(scheduled_for: int) -> Moment:
    """Build a SNOOZED Moment wired for the scheduler wake-up path.

    ``scheduled_for`` must be ``<= REF_NOW`` so the scheduler's
    ``list_scheduled`` query picks it up on ``tick``. ``expires_at``
    stays well in the future so the wake transitions to SUGGESTED
    (not EXPIRED).
    """
    mid = str(uuid.uuid4())
    return Moment(
        id=mid,
        created_at=REF_NOW - 600,
        scheduled_for=scheduled_for,
        expires_at=REF_NOW + 24 * 3600,
        insight="ping your sister",
        evidence_hash=f"hash-{mid[:8]}",
        proposed_action=Action(
            kind=ActionKind.DRAFT_MESSAGE,
            params={"body": "Hey — been a minute."},
        ),
        source_insight_type=InsightType.CADENCE,
        evidence=["evt-0000001"],
        state=MomentState.SNOOZED,
        snooze_until=scheduled_for,
        confidence=0.9,
    )


def _build_v1_fixture(data_dir: Path) -> SampleCounts:
    data_dir.mkdir(parents=True, exist_ok=True)
    return build_scaled_v1_sample(data_dir, **_SMALL_COUNTS)


# ---------------------------------------------------------------------------
# Shared fixture: the expensive legwork (build v1, snapshot, migrate, serve)
# runs once per module so the per-assertion tests below stay cheap.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the full scripted cutover once and hand the results to tests.

    Module-scoped so the migration (the most expensive step) executes
    a single time. Each assertion below reads pre-computed state off
    this dict instead of rebuilding the world.
    """
    start = time.monotonic()

    root = tmp_path_factory.mktemp("cutover-dryrun")
    data_dir = root / "data"
    snapshot_dir = root / "backup-20260422-120000"

    # 1. Fresh v1 fixture.
    counts = _build_v1_fixture(data_dir)
    v1_hashes_before = {db.name: _sha256(db) for db in sorted(data_dir.glob("*.db"))}

    # 2. Snapshot (runbook § 1.3).
    _snapshot_data_dir(data_dir, snapshot_dir)

    # 3. Migration (runbook § 3).
    v2_db_path = data_dir / V2_DB_NAME
    report = migrate.run_migration(data_dir, v2_db_path)

    # 4. Bring up v2 against the migrated DB + a real repo trio.
    conn = sqlite3.connect(str(v2_db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    clock = _Clock()
    moment_repo = MomentRepository(conn, now_fn=clock)
    feedback_store = FeedbackWeightStore(conn, now_fn=clock)
    outbox_repo = OutboxRepository(conn, now_fn=clock)
    health_probe = _StubHealthProbe(v2_db_path)
    life_os = _LifeOSDouble(
        moment_repo=moment_repo,
        feedback_weight_store=feedback_store,
        outbox_repo=outbox_repo,
        health_probe=health_probe,
    )
    app = create_app(life_os)
    client = TestClient(app)

    # 5. /api/health smoke.
    health_resp = client.get("/api/health")

    # 6. Create + scheduler-wake a SNOOZED Moment.
    moment = _make_snoozed_moment(scheduled_for=REF_NOW - 60)
    moment_repo.create(moment)
    scheduler = Scheduler(moment_repo, outbox_repo, now_fn=clock)
    fires = asyncio.run(scheduler.tick())

    now_resp = client.get("/api/now")

    # 7. Shut down v2.
    client.close()
    conn.close()

    # 8. Rollback via scripted runner (restore v1 snapshot).
    runner = _ScriptedRunner()
    rollback_config = rollback.RollbackConfig(
        snapshot=snapshot_dir,
        data_dir=data_dir,
        v2_db_name=V2_DB_NAME,
        v2_pid=12345,  # pretend-PID; scripted runner records but does not signal.
        v2_pid_file=None,
        v1_service="lifeos",
        nats_service="nats",
        v1_health_url="http://localhost:8080/health",
        timestamp="20260422-120000",
    )
    exit_code, step_results = rollback.execute_rollback(rollback_config, runner)

    v1_hashes_after = {db.name: _sha256(db) for db in sorted(data_dir.glob("*.db"))}

    elapsed_seconds = time.monotonic() - start

    return {
        "counts": counts,
        "report": report,
        "v2_db_path": v2_db_path,
        "health_resp": health_resp,
        "moment": moment,
        "fires": fires,
        "now_resp": now_resp,
        "runner_calls": runner.calls,
        "exit_code": exit_code,
        "step_results": step_results,
        "v1_hashes_before": v1_hashes_before,
        "v1_hashes_after": v1_hashes_after,
        "data_dir": data_dir,
        "snapshot_dir": snapshot_dir,
        "elapsed_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_harness_completes_within_ci_budget(harness: dict[str, Any]) -> None:
    """Acceptance: < 2 min end-to-end. Guards against silent regression."""
    assert harness["elapsed_seconds"] < 120.0, (
        f"harness took {harness['elapsed_seconds']:.1f}s (budget 120s); split the fixture or shrink sample counts."
    )


def test_migration_report_counts_match_sample(harness: dict[str, Any]) -> None:
    """Every source row the migrator was given is either translated or dropped."""
    counts: SampleCounts = harness["counts"]
    report: migrate.MigrationReport = harness["report"]

    assert report.events.source == counts.events
    assert report.events.translated == counts.events
    assert report.entities.translated == counts.total_entities
    assert report.moments_from_tasks.translated == counts.tasks
    assert report.signal_profiles.translated == counts.kept_signal_profiles
    assert report.signal_profiles.dropped == counts.dropped_signal_profiles
    assert report.preferences.translated == counts.preferences
    assert report.notification_feedback.translated == counts.feedback_log
    # No invariant violations surface in the notes.
    assert not any(note.startswith("INVARIANT:") for note in report.notes), report.notes


def test_v2_db_written_on_disk(harness: dict[str, Any]) -> None:
    """The migrator writes a non-empty SQLite file at the expected path."""
    db_path: Path = harness["v2_db_path"]
    assert db_path.exists()
    assert db_path.stat().st_size > 0


def test_health_endpoint_reports_ok(harness: dict[str, Any]) -> None:
    """``/api/health`` round-trips the probe payload for the live app."""
    resp = harness["health_resp"]
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["connectors"] == {"proton_mail": "ok"}


def test_scheduler_wakes_snoozed_moment(harness: dict[str, Any]) -> None:
    """``Scheduler.tick`` wakes a past-scheduled SNOOZED Moment to SUGGESTED.

    The Moment just created is past its ``scheduled_for``, so the tick
    must fire exactly once with annotation ``scheduler_fire``. The
    follow-up ``GET /api/now`` proves the wake is visible through the
    full API stack, not just at the repository level.
    """
    fires = harness["fires"]
    assert len(fires) == 1, fires
    assert fires[0].annotation == "scheduler_fire"
    assert fires[0].moment_id == harness["moment"].id

    resp = harness["now_resp"]
    assert resp.status_code == 200
    data = resp.json()
    pending_ids = [m["id"] for m in data["pending"]]
    assert harness["moment"].id in pending_ids


def test_rollback_plan_orders_every_step(harness: dict[str, Any]) -> None:
    """The scripted rollback drives the 7 runbook steps in order.

    Step names are asserted verbatim so a reorder in
    :func:`scripts.cutover_rollback.plan_rollback` that drifts from the
    runbook breaks this test loudly instead of silently.
    """
    results = harness["step_results"]
    names = [r.step.name for r in results]
    assert names == [
        "validate_snapshot",
        "stop_v2",
        "archive_v2_db",
        "restore_v1_dbs",
        "start_nats",
        "start_v1",
        "verify_v1_health",
    ]
    assert harness["exit_code"] == rollback.EXIT_OK
    assert all(r.ok for r in results), [(r.step.name, r.message) for r in results if not r.ok]


def test_rollback_side_effects_hit_runner_in_order(harness: dict[str, Any]) -> None:
    """The scripted runner sees kill → move → copy_db → compose×2 → verify."""
    kinds = [call[0] for call in harness["runner_calls"]]
    assert kinds == [
        "kill",
        "move",
        "copy_db",
        "copy_lance",
        "compose",
        "compose",
        "verify",
    ]

    kill_call = harness["runner_calls"][0]
    assert kill_call[1] == (12345,)

    compose_calls = [c for c in harness["runner_calls"] if c[0] == "compose"]
    assert [c[1] for c in compose_calls] == [
        ("start", "nats"),
        ("start", "lifeos"),
    ]


def test_restored_v1_dbs_match_original_byte_for_byte(harness: dict[str, Any]) -> None:
    """Restored data dir equals the pre-migration snapshot, byte-for-byte.

    If this ever drifts, something mutated the v1 files during the
    dry-run — which would invalidate the whole rollback contract. The
    v2 db archived to ``lifeos.db.failed-cutover-<ts>`` is allowed to
    remain; we only assert on the v1 files that existed pre-snapshot.
    """
    before = harness["v1_hashes_before"]
    after = harness["v1_hashes_after"]
    for name, digest in before.items():
        assert name in after, f"v1 db {name} missing from restored data dir"
        assert after[name] == digest, f"v1 db {name} content drifted during round-trip"


def test_archived_v2_db_preserved_for_forensics(harness: dict[str, Any]) -> None:
    """The v2 db moves aside to ``.failed-cutover-<ts>`` rather than being deleted."""
    data_dir: Path = harness["data_dir"]
    archived = list(data_dir.glob(f"{V2_DB_NAME}.failed-cutover-*"))
    assert len(archived) == 1, [p.name for p in data_dir.iterdir()]
    assert archived[0].stat().st_size > 0
    # The live v2 db slot is vacated — v1 now owns that directory.
    assert not (data_dir / V2_DB_NAME).exists()


# ---------------------------------------------------------------------------
# Smaller edge-case tests — fresh fixtures so failures don't cascade.
# ---------------------------------------------------------------------------


def test_rollback_fails_when_snapshot_missing(tmp_path: Path) -> None:
    """A bogus snapshot path short-circuits the plan at step 1."""
    config = rollback.RollbackConfig(
        snapshot=tmp_path / "does-not-exist",
        data_dir=tmp_path / "data",
        v2_db_name=V2_DB_NAME,
        v2_pid=1,
        v2_pid_file=None,
        v1_service="lifeos",
        nats_service="nats",
        v1_health_url="http://localhost:8080/health",
        timestamp="20260422-120000",
    )
    runner = _ScriptedRunner()
    exit_code, results = rollback.execute_rollback(config, runner)
    assert exit_code == rollback.EXIT_FAIL
    assert results[0].step.name == "validate_snapshot"
    assert not results[0].ok
    # Every following step is marked skipped.
    assert all(r.message == "skipped (prior step failed)" for r in results[1:])
    # And the runner is never driven.
    assert runner.calls == []


def test_migration_refuses_to_overwrite_existing_target(tmp_path: Path) -> None:
    """``run_migration`` raises ``FileExistsError`` rather than overwrite.

    This matches the CLI's explicit "delete the target first" stance and
    prevents the rehearsal from silently clobbering a previous run.
    """
    data_dir = tmp_path / "data"
    _build_v1_fixture(data_dir)
    target = data_dir / V2_DB_NAME
    target.write_bytes(b"stale")

    with pytest.raises(FileExistsError):
        migrate.run_migration(data_dir, target)
