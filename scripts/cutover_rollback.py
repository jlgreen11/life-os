#!/usr/bin/env python3
"""Cutover rollback automation for the v1 → v2 cutover.

Implements NEXT_TASKS Category C § "Cutover rollback script" and the
scripted-path branch of the cutover runbook § 7.3. Automates the six
manual steps the operator would otherwise perform by hand to restore v1
from the timestamped snapshot produced in runbook § 1.3:

1. Validate the snapshot directory contains the v1 SQLite DBs + LanceDB
   directory.
2. Stop v2 (kill the PID in ``--v2-pid``).
3. Move the v2 SQLite DB aside (keeps it on disk for forensics under
   ``<db>.failed-cutover-<ts>``).
4. Restore the v1 SQLite DBs + LanceDB dir from the snapshot into the
   data directory.
5. Restart NATS, then the v1 service (via ``docker compose start``).
6. Verify v1 is serving ``GET /health`` (legacy endpoint, not the v2
   ``/api/health``).

Budget: **RTO ≤ 30 min** is locked in as a module constant and surfaced
at script start so the operator can flag a rollback run that blows past
the target at the first step rather than waiting until step 6.

Design notes
------------
The step plan is a pure :func:`plan_rollback` that returns a list of
:class:`RollbackStep` — describing the action, the expected target, and
the one-line message printed in dry-run mode. Side effects live behind
the :class:`Runner` protocol; the default :class:`RealRunner` shells out
to ``kill``, ``shutil.move``, ``shutil.copytree``, ``subprocess.run``
and ``urllib.request.urlopen``. The :class:`DryRunRunner` logs each step
without touching the filesystem or the OS, which makes both
``--dry-run`` and the XCTest-style unit tests straightforward.

Exit codes
----------
* ``0`` — every step succeeded (or every step was logged in dry-run).
* ``1`` — a step failed. Remaining steps are skipped; the operator
  should fall back to the manual procedure in runbook § 7.3.

References
----------
- Task spec: ``NEXT_TASKS.md`` § Category C "Cutover rollback script".
- Runbook: ``docs/cutover-runbook.md`` § 7 "Rollback".
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "Data migration (expanded …)".
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_V1_HEALTH_URL = "http://localhost:8080/health"
DEFAULT_V1_SERVICE = "lifeos"
DEFAULT_NATS_SERVICE = "nats"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_V2_DB_NAME = "lifeos.db"

# Runbook § 0 locks the cutover window at 20 min wall-clock; rollback is
# the contingency path with a 30-min RTO (CEO plan § "Data migration").
RTO_MINUTES = 30

EXIT_OK = 0
EXIT_FAIL = 1

log = logging.getLogger("cutover_rollback")


# ---------------------------------------------------------------------------
# Config + plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RollbackConfig:
    """All inputs that parameterise a rollback run.

    Every field is a constructor arg so the tests can instantiate the
    config against a ``tmp_path`` layout without touching real
    production paths.
    """

    snapshot: Path
    data_dir: Path
    v2_db_name: str
    v2_pid: int | None
    v2_pid_file: Path | None
    v1_service: str
    nats_service: str
    v1_health_url: str
    timestamp: str
    dry_run: bool = False
    http_timeout_seconds: float = 5.0

    def failed_db_path(self) -> Path:
        """Where the pre-rollback v2 DB is archived for forensics."""
        return self.data_dir / f"{self.v2_db_name}.failed-cutover-{self.timestamp}"


@dataclass
class RollbackStep:
    """One ordered action in the rollback plan.

    Attributes
    ----------
    name:
        Short machine-readable identifier (``stop_v2``, ``restore_dbs``,
        …). Surfaces in the dry-run log and the summary report.
    description:
        Human-readable sentence for the operator log.
    kind:
        One of ``validate``, ``kill``, ``move``, ``restore``, ``compose``,
        ``verify``. The runner dispatches on ``kind``.
    payload:
        Step-kind-specific arguments. Kept as a plain dict so the plan
        stays introspectable without importing the runner in tests.
    """

    name: str
    description: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


def plan_rollback(config: RollbackConfig) -> list[RollbackStep]:
    """Return the ordered list of steps that ``execute_rollback`` runs.

    Deterministic and side-effect-free so tests can assert on the shape
    of the plan (count, order, payload contents) without touching the
    filesystem or the OS. The runbook locks this sequence as the
    scripted mirror of § 7.3 manual fallback; any reordering here is a
    runbook change and should be reviewed together.
    """
    v2_db = config.data_dir / config.v2_db_name
    return [
        RollbackStep(
            name="validate_snapshot",
            description=f"Validate snapshot directory at {config.snapshot}",
            kind="validate",
            payload={"snapshot": config.snapshot},
        ),
        RollbackStep(
            name="stop_v2",
            description=(
                f"Stop v2 (pid={config.v2_pid})"
                if config.v2_pid is not None
                else f"Stop v2 (pid file={config.v2_pid_file})"
            ),
            kind="kill",
            payload={"pid": config.v2_pid, "pid_file": config.v2_pid_file},
        ),
        RollbackStep(
            name="archive_v2_db",
            description=f"Move {v2_db} → {config.failed_db_path()}",
            kind="move",
            payload={"src": v2_db, "dst": config.failed_db_path()},
        ),
        RollbackStep(
            name="restore_v1_dbs",
            description=f"Restore v1 SQLite DBs + LanceDB dirs from {config.snapshot} into {config.data_dir}",
            kind="restore",
            payload={"snapshot": config.snapshot, "data_dir": config.data_dir},
        ),
        RollbackStep(
            name="start_nats",
            description=f"docker compose start {config.nats_service}",
            kind="compose",
            payload={"service": config.nats_service, "action": "start"},
        ),
        RollbackStep(
            name="start_v1",
            description=f"docker compose start {config.v1_service}",
            kind="compose",
            payload={"service": config.v1_service, "action": "start"},
        ),
        RollbackStep(
            name="verify_v1_health",
            description=f"GET {config.v1_health_url} (must return 200)",
            kind="verify",
            payload={"url": config.v1_health_url, "timeout": config.http_timeout_seconds},
        ),
    ]


# ---------------------------------------------------------------------------
# Snapshot validation
# ---------------------------------------------------------------------------
def validate_snapshot(snapshot: Path) -> list[str]:
    """Return a list of human-readable errors; empty list means valid.

    Pure — no I/O beyond ``os.listdir`` on the snapshot dir. A valid
    snapshot must be an existing directory containing **at least one**
    ``*.db`` file; a LanceDB ``*.lance/`` directory is recommended but
    not required (a v1 without any vector index is still restorable).
    """
    errors: list[str] = []
    if not snapshot.exists():
        errors.append(f"snapshot directory does not exist: {snapshot}")
        return errors
    if not snapshot.is_dir():
        errors.append(f"snapshot is not a directory: {snapshot}")
        return errors
    dbs = sorted(snapshot.glob("*.db"))
    if not dbs:
        errors.append(f"snapshot contains no *.db files: {snapshot}")
    return errors


def snapshot_lance_dirs(snapshot: Path) -> list[Path]:
    """Return LanceDB dirs in the snapshot (``*.lance``), sorted."""
    if not snapshot.is_dir():
        return []
    return sorted(p for p in snapshot.glob("*.lance") if p.is_dir())


def snapshot_db_files(snapshot: Path) -> list[Path]:
    """Return ``*.db`` files in the snapshot, sorted."""
    if not snapshot.is_dir():
        return []
    return sorted(snapshot.glob("*.db"))


# ---------------------------------------------------------------------------
# Runner protocol + implementations
# ---------------------------------------------------------------------------
class Runner(Protocol):
    """Side-effect interface that :func:`execute_rollback` dispatches into.

    The protocol is minimal so the :class:`DryRunRunner` implementation
    (used by ``--dry-run`` and the unit tests) is a handful of logging
    lines. The production :class:`RealRunner` implements each method
    against ``subprocess`` / ``shutil`` / ``urllib``.
    """

    def kill_process(self, pid: int) -> None: ...

    def move_file(self, src: Path, dst: Path) -> None: ...

    def copy_db_files(self, sources: list[Path], dest_dir: Path) -> None: ...

    def copy_lance_dirs(self, sources: list[Path], dest_dir: Path) -> None: ...

    def docker_compose(self, action: str, service: str) -> None: ...

    def http_check_ok(self, url: str, timeout: float) -> bool: ...


class RealRunner:
    """Production runner — talks to the actual host."""

    def kill_process(self, pid: int) -> None:
        """Send SIGTERM to ``pid``; raise :class:`ProcessLookupError` if gone."""
        os.kill(pid, signal.SIGTERM)

    def move_file(self, src: Path, dst: Path) -> None:
        """``shutil.move`` wrapper; fails loudly if ``src`` is missing."""
        if not src.exists():
            raise FileNotFoundError(f"{src} does not exist; nothing to move")
        shutil.move(str(src), str(dst))

    def copy_db_files(self, sources: list[Path], dest_dir: Path) -> None:
        """Copy each ``*.db`` into ``dest_dir`` (preserving metadata)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copy2(src, dest_dir / src.name)

    def copy_lance_dirs(self, sources: list[Path], dest_dir: Path) -> None:
        """Copy each ``*.lance/`` directory tree into ``dest_dir``.

        Existing target directories are removed first; LanceDB corpora
        are append-only-ish but a partial overlay is worse than an
        atomic replace during a rollback.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            target = dest_dir / src.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)

    def docker_compose(self, action: str, service: str) -> None:
        """``docker compose <action> <service>`` with ``check=True``."""
        subprocess.run(
            ["docker", "compose", action, service],
            check=True,
        )

    def http_check_ok(self, url: str, timeout: float) -> bool:
        """Return ``True`` iff ``GET url`` responds 200 within ``timeout``."""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False


class DryRunRunner:
    """No-op runner that logs what would happen.

    Returns successful values where applicable (``http_check_ok`` →
    ``True``) so the plan always completes a dry-run pass. This lets
    the operator audit the full sequence before committing to a real
    rollback.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or log

    def kill_process(self, pid: int) -> None:
        self._log.info("[dry-run] would SIGTERM pid=%d", pid)

    def move_file(self, src: Path, dst: Path) -> None:
        self._log.info("[dry-run] would move %s -> %s", src, dst)

    def copy_db_files(self, sources: list[Path], dest_dir: Path) -> None:
        for src in sources:
            self._log.info("[dry-run] would copy db %s -> %s/", src, dest_dir)

    def copy_lance_dirs(self, sources: list[Path], dest_dir: Path) -> None:
        for src in sources:
            self._log.info("[dry-run] would copy lance %s -> %s/%s", src, dest_dir, src.name)

    def docker_compose(self, action: str, service: str) -> None:
        self._log.info("[dry-run] would run: docker compose %s %s", action, service)

    def http_check_ok(self, url: str, timeout: float) -> bool:
        self._log.info("[dry-run] would GET %s (timeout=%.1fs) and assert 2xx", url, timeout)
        return True


# ---------------------------------------------------------------------------
# PID resolution
# ---------------------------------------------------------------------------
def resolve_pid(config: RollbackConfig) -> int | None:
    """Return the PID to kill for ``stop_v2``, or ``None`` if not derivable.

    Precedence: explicit ``--v2-pid`` > ``--v2-pid-file`` contents. A
    missing / malformed pid-file returns ``None`` rather than raising,
    so the operator sees a clear step-level error instead of a CLI
    crash.
    """
    if config.v2_pid is not None:
        return config.v2_pid
    if config.v2_pid_file is None:
        return None
    try:
        text = config.v2_pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    """Outcome of executing one :class:`RollbackStep`."""

    step: RollbackStep
    ok: bool
    message: str


def execute_rollback(
    config: RollbackConfig,
    runner: Runner,
    logger: logging.Logger | None = None,
) -> tuple[int, list[StepResult]]:
    """Run every step in :func:`plan_rollback`; stop on first failure.

    Returns ``(exit_code, results)`` so callers (CLI + tests) can assert
    on per-step outcomes without scraping log lines. The first failing
    step short-circuits — remaining steps are appended as ``ok=False``
    with message ``"skipped (prior step failed)"`` so the returned list
    still has the full plan length and is trivially renderable as a
    table.
    """
    logger = logger or log
    steps = plan_rollback(config)
    results: list[StepResult] = []
    failed = False

    for step in steps:
        if failed:
            results.append(StepResult(step=step, ok=False, message="skipped (prior step failed)"))
            continue
        logger.info("step=%s kind=%s %s", step.name, step.kind, step.description)
        try:
            _dispatch_step(step, config, runner)
        except Exception as exc:
            logger.error("step=%s FAILED: %s", step.name, exc)
            results.append(StepResult(step=step, ok=False, message=str(exc)))
            failed = True
            continue
        results.append(StepResult(step=step, ok=True, message="ok"))

    exit_code = EXIT_FAIL if failed else EXIT_OK
    return exit_code, results


def _dispatch_step(step: RollbackStep, config: RollbackConfig, runner: Runner) -> None:
    """Route a step to the right :class:`Runner` method.

    Raises any exception the runner raises — the outer loop in
    :func:`execute_rollback` converts them into ``StepResult(ok=False)``.
    """
    if step.kind == "validate":
        snapshot: Path = step.payload["snapshot"]
        errors = validate_snapshot(snapshot)
        if errors:
            raise ValueError("; ".join(errors))
        return

    if step.kind == "kill":
        pid = resolve_pid(config)
        if pid is None:
            raise ValueError("no v2 pid resolved from --v2-pid or --v2-pid-file; refusing to guess")
        runner.kill_process(pid)
        return

    if step.kind == "move":
        src: Path = step.payload["src"]
        dst: Path = step.payload["dst"]
        if not src.exists():
            # Nothing to move — the v2 DB may already be absent (e.g. a
            # rollback right after a failed migration step). Log at
            # INFO and continue rather than failing the whole run.
            log.info("archive_v2_db: source %s absent; nothing to move", src)
            return
        runner.move_file(src, dst)
        return

    if step.kind == "restore":
        snapshot: Path = step.payload["snapshot"]
        data_dir: Path = step.payload["data_dir"]
        dbs = snapshot_db_files(snapshot)
        lances = snapshot_lance_dirs(snapshot)
        runner.copy_db_files(dbs, data_dir)
        runner.copy_lance_dirs(lances, data_dir)
        return

    if step.kind == "compose":
        runner.docker_compose(step.payload["action"], step.payload["service"])
        return

    if step.kind == "verify":
        ok = runner.http_check_ok(step.payload["url"], step.payload["timeout"])
        if not ok:
            raise RuntimeError(f"v1 health check failed: {step.payload['url']}")
        return

    raise ValueError(f"unknown step kind: {step.kind}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_results(results: list[StepResult]) -> str:
    """Human-readable multiline summary for the end-of-run log."""
    lines = [f"{'OK' if r.ok else 'FAIL':4s}  {r.step.name:20s}  {r.message}" for r in results]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(f"Cutover rollback: restore v1 from a snapshot (RTO target ≤ {RTO_MINUTES} min)."),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to ./data/backup-<ts>/ containing the v1 *.db + *.lance/ snapshot.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target data directory where v1 DBs + LanceDB are restored (default: ./data).",
    )
    parser.add_argument(
        "--v2-db-name",
        default=DEFAULT_V2_DB_NAME,
        help=f"Basename of the v2 SQLite DB to archive (default: {DEFAULT_V2_DB_NAME}).",
    )
    pid_group = parser.add_mutually_exclusive_group(required=True)
    pid_group.add_argument("--v2-pid", type=int, help="PID of the running v2 process.")
    pid_group.add_argument(
        "--v2-pid-file",
        type=Path,
        help="Path to a file containing the v2 PID (e.g. ./data/v2.pid).",
    )
    parser.add_argument("--v1-service", default=DEFAULT_V1_SERVICE)
    parser.add_argument("--nats-service", default=DEFAULT_NATS_SERVICE)
    parser.add_argument("--v1-health-url", default=DEFAULT_V1_HEALTH_URL)
    parser.add_argument("--http-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log every step without executing. Use to audit the plan before the real run.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("cutover_rollback")


def _config_from_args(ns: argparse.Namespace) -> RollbackConfig:
    return RollbackConfig(
        snapshot=ns.snapshot,
        data_dir=ns.data_dir,
        v2_db_name=ns.v2_db_name,
        v2_pid=ns.v2_pid,
        v2_pid_file=ns.v2_pid_file,
        v1_service=ns.v1_service,
        nats_service=ns.nats_service,
        v1_health_url=ns.v1_health_url,
        timestamp=dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S"),
        dry_run=ns.dry_run,
        http_timeout_seconds=ns.http_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(sys.argv[1:] if argv is None else argv))
    logger = _setup_logging(ns.verbose)
    config = _config_from_args(ns)
    logger.info(
        "cutover_rollback starting: snapshot=%s dry_run=%s RTO_target=%dm",
        config.snapshot,
        config.dry_run,
        RTO_MINUTES,
    )
    runner: Runner = DryRunRunner(logger) if config.dry_run else RealRunner()
    exit_code, results = execute_rollback(config, runner, logger=logger)
    logger.info("rollback summary:\n%s", render_results(results))
    if exit_code == EXIT_OK:
        logger.info("cutover_rollback complete (dry_run=%s)", config.dry_run)
    else:
        logger.error("cutover_rollback FAILED — fall back to runbook § 7.3 manual procedure")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
