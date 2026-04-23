"""Tests for ``scripts/cutover_rollback.py``.

Covers the pure plan + validation layer, the :class:`DryRunRunner` +
:func:`execute_rollback` orchestrator against fake side effects, and the
CLI argument surface. All FS / HTTP side effects are mocked so the
suite runs in milliseconds without a network or a docker daemon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scripts import cutover_rollback as rb


# ---------------------------------------------------------------------------
# Snapshot + data-dir fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """A minimal valid v1 snapshot: three DBs + one LanceDB dir."""
    snap = tmp_path / "backup-20260422-120000"
    snap.mkdir()
    for name in ("events.db", "entities.db", "state.db"):
        (snap / name).write_bytes(b"SQLite format 3\x00")
    lance = snap / "events.lance"
    lance.mkdir()
    (lance / "data.arrow").write_bytes(b"arrow")
    return snap


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A fresh v2 data dir with a v2 DB + stale pid file."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "lifeos.db").write_bytes(b"SQLite format 3\x00--v2")
    (d / "v2.pid").write_text("4242\n", encoding="utf-8")
    return d


@pytest.fixture
def config(snapshot: Path, data_dir: Path) -> rb.RollbackConfig:
    """A rollback config pointing at the tmp_path fixtures."""
    return rb.RollbackConfig(
        snapshot=snapshot,
        data_dir=data_dir,
        v2_db_name="lifeos.db",
        v2_pid=4242,
        v2_pid_file=None,
        v1_service="lifeos",
        nats_service="nats",
        v1_health_url="http://localhost:8080/health",
        timestamp="20260422-120000",
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# validate_snapshot
# ---------------------------------------------------------------------------
class TestValidateSnapshot:
    def test_valid_snapshot_returns_no_errors(self, snapshot: Path) -> None:
        assert rb.validate_snapshot(snapshot) == []

    def test_missing_directory_reports_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        errors = rb.validate_snapshot(missing)
        assert errors
        assert "does not exist" in errors[0]

    def test_file_instead_of_directory_reports_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "file-not-dir"
        bad.write_text("not a dir")
        errors = rb.validate_snapshot(bad)
        assert errors
        assert "not a directory" in errors[0]

    def test_directory_without_db_files_reports_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "backup-empty"
        empty.mkdir()
        (empty / "notes.txt").write_text("no dbs here")
        errors = rb.validate_snapshot(empty)
        assert errors
        assert "no *.db" in errors[0]

    def test_snapshot_without_lance_dir_is_still_valid(self, tmp_path: Path) -> None:
        """LanceDB is recommended but not required (runbook note)."""
        snap = tmp_path / "backup-no-lance"
        snap.mkdir()
        (snap / "events.db").write_bytes(b"x")
        assert rb.validate_snapshot(snap) == []


# ---------------------------------------------------------------------------
# snapshot_db_files / snapshot_lance_dirs
# ---------------------------------------------------------------------------
class TestSnapshotEnumeration:
    def test_db_files_sorted(self, snapshot: Path) -> None:
        dbs = rb.snapshot_db_files(snapshot)
        assert [p.name for p in dbs] == ["entities.db", "events.db", "state.db"]

    def test_lance_dirs_sorted(self, snapshot: Path) -> None:
        dirs = rb.snapshot_lance_dirs(snapshot)
        assert [p.name for p in dirs] == ["events.lance"]

    def test_lance_files_not_returned(self, tmp_path: Path) -> None:
        """``*.lance`` as a file (edge case) is excluded from dirs."""
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "events.db").write_bytes(b"")
        (snap / "stray.lance").write_bytes(b"not a dir")
        assert rb.snapshot_lance_dirs(snap) == []

    def test_missing_snapshot_returns_empty(self, tmp_path: Path) -> None:
        gone = tmp_path / "gone"
        assert rb.snapshot_db_files(gone) == []
        assert rb.snapshot_lance_dirs(gone) == []


# ---------------------------------------------------------------------------
# plan_rollback — shape + order
# ---------------------------------------------------------------------------
class TestPlanRollback:
    def test_plan_has_seven_steps_in_canonical_order(self, config: rb.RollbackConfig) -> None:
        plan = rb.plan_rollback(config)
        assert [s.name for s in plan] == [
            "validate_snapshot",
            "stop_v2",
            "archive_v2_db",
            "restore_v1_dbs",
            "start_nats",
            "start_v1",
            "verify_v1_health",
        ]

    def test_plan_kinds_are_known(self, config: rb.RollbackConfig) -> None:
        allowed = {"validate", "kill", "move", "restore", "compose", "verify"}
        for step in rb.plan_rollback(config):
            assert step.kind in allowed, f"unknown kind {step.kind} on step {step.name}"

    def test_archive_db_payload_points_at_data_dir(self, config: rb.RollbackConfig) -> None:
        plan = rb.plan_rollback(config)
        step = next(s for s in plan if s.name == "archive_v2_db")
        assert step.payload["src"] == config.data_dir / "lifeos.db"
        assert step.payload["dst"] == config.failed_db_path()
        assert str(step.payload["dst"]).endswith(".failed-cutover-20260422-120000")

    def test_compose_payload_specifies_action_and_service(self, config: rb.RollbackConfig) -> None:
        plan = rb.plan_rollback(config)
        nats_step = next(s for s in plan if s.name == "start_nats")
        v1_step = next(s for s in plan if s.name == "start_v1")
        assert nats_step.payload == {"action": "start", "service": "nats"}
        assert v1_step.payload == {"action": "start", "service": "lifeos"}

    def test_verify_step_uses_v1_health_url(self, config: rb.RollbackConfig) -> None:
        plan = rb.plan_rollback(config)
        step = next(s for s in plan if s.name == "verify_v1_health")
        assert step.payload["url"] == "http://localhost:8080/health"
        assert step.payload["timeout"] == config.http_timeout_seconds

    def test_stop_v2_description_prefers_explicit_pid(self, config: rb.RollbackConfig) -> None:
        plan = rb.plan_rollback(config)
        step = next(s for s in plan if s.name == "stop_v2")
        assert "pid=4242" in step.description


# ---------------------------------------------------------------------------
# resolve_pid
# ---------------------------------------------------------------------------
class TestResolvePid:
    def test_explicit_pid_wins(self, config: rb.RollbackConfig) -> None:
        assert rb.resolve_pid(config) == 4242

    def test_pid_file_used_when_no_explicit(self, data_dir: Path, snapshot: Path) -> None:
        cfg = rb.RollbackConfig(
            snapshot=snapshot,
            data_dir=data_dir,
            v2_db_name="lifeos.db",
            v2_pid=None,
            v2_pid_file=data_dir / "v2.pid",
            v1_service="lifeos",
            nats_service="nats",
            v1_health_url="http://localhost:8080/health",
            timestamp="20260422-120000",
        )
        assert rb.resolve_pid(cfg) == 4242

    def test_missing_pid_file_returns_none(self, tmp_path: Path, snapshot: Path) -> None:
        cfg = rb.RollbackConfig(
            snapshot=snapshot,
            data_dir=tmp_path,
            v2_db_name="lifeos.db",
            v2_pid=None,
            v2_pid_file=tmp_path / "nope.pid",
            v1_service="lifeos",
            nats_service="nats",
            v1_health_url="http://localhost:8080/health",
            timestamp="20260422-120000",
        )
        assert rb.resolve_pid(cfg) is None

    def test_malformed_pid_file_returns_none(self, tmp_path: Path, snapshot: Path) -> None:
        pid_file = tmp_path / "bad.pid"
        pid_file.write_text("not-a-number\n")
        cfg = rb.RollbackConfig(
            snapshot=snapshot,
            data_dir=tmp_path,
            v2_db_name="lifeos.db",
            v2_pid=None,
            v2_pid_file=pid_file,
            v1_service="lifeos",
            nats_service="nats",
            v1_health_url="http://localhost:8080/health",
            timestamp="20260422-120000",
        )
        assert rb.resolve_pid(cfg) is None

    def test_no_pid_source_returns_none(self, tmp_path: Path, snapshot: Path) -> None:
        cfg = rb.RollbackConfig(
            snapshot=snapshot,
            data_dir=tmp_path,
            v2_db_name="lifeos.db",
            v2_pid=None,
            v2_pid_file=None,
            v1_service="lifeos",
            nats_service="nats",
            v1_health_url="http://localhost:8080/health",
            timestamp="20260422-120000",
        )
        assert rb.resolve_pid(cfg) is None


# ---------------------------------------------------------------------------
# Fake runner for execute_rollback
# ---------------------------------------------------------------------------
@dataclass
class FakeRunner:
    """Records calls + per-method controllable outcomes."""

    kills: list[int] = field(default_factory=list)
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    db_copies: list[tuple[tuple[str, ...], Path]] = field(default_factory=list)
    lance_copies: list[tuple[tuple[str, ...], Path]] = field(default_factory=list)
    compose_calls: list[tuple[str, str]] = field(default_factory=list)
    http_calls: list[tuple[str, float]] = field(default_factory=list)
    http_result: bool = True
    fail_step: str | None = None

    def _raise_if_failing(self, step_name: str) -> None:
        if self.fail_step == step_name:
            raise RuntimeError(f"fake failure in {step_name}")

    def kill_process(self, pid: int) -> None:
        self._raise_if_failing("kill")
        self.kills.append(pid)

    def move_file(self, src: Path, dst: Path) -> None:
        self._raise_if_failing("move")
        self.moves.append((src, dst))

    def copy_db_files(self, sources: list[Path], dest_dir: Path) -> None:
        self._raise_if_failing("copy_db")
        self.db_copies.append((tuple(s.name for s in sources), dest_dir))

    def copy_lance_dirs(self, sources: list[Path], dest_dir: Path) -> None:
        self._raise_if_failing("copy_lance")
        self.lance_copies.append((tuple(s.name for s in sources), dest_dir))

    def docker_compose(self, action: str, service: str) -> None:
        self._raise_if_failing(f"compose:{service}")
        self.compose_calls.append((action, service))

    def http_check_ok(self, url: str, timeout: float) -> bool:
        self.http_calls.append((url, timeout))
        return self.http_result


# ---------------------------------------------------------------------------
# execute_rollback — happy path
# ---------------------------------------------------------------------------
class TestExecuteRollbackHappyPath:
    def test_all_steps_ok_returns_exit_zero(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner()
        exit_code, results = rb.execute_rollback(config, runner)
        assert exit_code == rb.EXIT_OK
        assert all(r.ok for r in results)
        assert [r.step.name for r in results] == [s.name for s in rb.plan_rollback(config)]

    def test_kill_receives_resolved_pid(self, config: rb.RollbackConfig) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert runner.kills == [4242]

    def test_move_receives_v2_db_src_and_archive_dst(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert runner.moves == [(config.data_dir / "lifeos.db", config.failed_db_path())]

    def test_db_copies_include_every_snapshot_db_file(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert len(runner.db_copies) == 1
        names, dest = runner.db_copies[0]
        assert set(names) == {"entities.db", "events.db", "state.db"}
        assert dest == config.data_dir

    def test_lance_copies_include_every_snapshot_lance_dir(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert runner.lance_copies == [(("events.lance",), config.data_dir)]

    def test_compose_calls_nats_first_then_v1(self, config: rb.RollbackConfig) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert runner.compose_calls == [("start", "nats"), ("start", "lifeos")]

    def test_http_check_uses_configured_url(self, config: rb.RollbackConfig) -> None:
        runner = FakeRunner()
        rb.execute_rollback(config, runner)
        assert runner.http_calls == [("http://localhost:8080/health", 5.0)]


# ---------------------------------------------------------------------------
# execute_rollback — failure paths
# ---------------------------------------------------------------------------
class TestExecuteRollbackFailures:
    def test_invalid_snapshot_fails_at_validate_and_skips_rest(
        self,
        config: rb.RollbackConfig,
        tmp_path: Path,
    ) -> None:
        bad_cfg = rb.RollbackConfig(
            **{**config.__dict__, "snapshot": tmp_path / "missing"},
        )
        runner = FakeRunner()
        exit_code, results = rb.execute_rollback(bad_cfg, runner)
        assert exit_code == rb.EXIT_FAIL
        assert results[0].step.name == "validate_snapshot"
        assert not results[0].ok
        assert "does not exist" in results[0].message
        # Remaining steps must be marked skipped, not executed.
        assert runner.kills == []
        assert runner.compose_calls == []
        assert runner.http_calls == []
        for r in results[1:]:
            assert not r.ok
            assert "skipped" in r.message

    def test_stop_v2_without_pid_fails_with_clear_message(
        self,
        snapshot: Path,
        data_dir: Path,
    ) -> None:
        cfg = rb.RollbackConfig(
            snapshot=snapshot,
            data_dir=data_dir,
            v2_db_name="lifeos.db",
            v2_pid=None,
            v2_pid_file=None,
            v1_service="lifeos",
            nats_service="nats",
            v1_health_url="http://localhost:8080/health",
            timestamp="20260422-120000",
        )
        runner = FakeRunner()
        exit_code, results = rb.execute_rollback(cfg, runner)
        assert exit_code == rb.EXIT_FAIL
        stop = next(r for r in results if r.step.name == "stop_v2")
        assert not stop.ok
        assert "no v2 pid" in stop.message

    def test_archive_v2_db_absent_source_does_not_fail(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        """If the v2 DB is already gone, archive step logs-and-skips."""
        (config.data_dir / "lifeos.db").unlink()
        runner = FakeRunner()
        exit_code, results = rb.execute_rollback(config, runner)
        assert exit_code == rb.EXIT_OK
        archive = next(r for r in results if r.step.name == "archive_v2_db")
        assert archive.ok
        # move_file was never called on the absent source.
        assert runner.moves == []

    def test_compose_nats_failure_short_circuits(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner(fail_step="compose:nats")
        exit_code, results = rb.execute_rollback(config, runner)
        assert exit_code == rb.EXIT_FAIL
        nats = next(r for r in results if r.step.name == "start_nats")
        assert not nats.ok
        # v1 service compose never attempted.
        assert ("start", "lifeos") not in runner.compose_calls

    def test_verify_v1_health_failure_marks_fail(
        self,
        config: rb.RollbackConfig,
    ) -> None:
        runner = FakeRunner(http_result=False)
        exit_code, results = rb.execute_rollback(config, runner)
        assert exit_code == rb.EXIT_FAIL
        last = results[-1]
        assert last.step.name == "verify_v1_health"
        assert not last.ok
        assert "health check failed" in last.message


# ---------------------------------------------------------------------------
# DryRunRunner — no side effects, always succeeds
# ---------------------------------------------------------------------------
class TestDryRunRunner:
    def test_dry_run_executes_full_plan_with_zero_side_effects(
        self,
        config: rb.RollbackConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.INFO, logger="cutover_rollback")
        runner = rb.DryRunRunner()
        exit_code, results = rb.execute_rollback(config, runner)
        assert exit_code == rb.EXIT_OK
        assert all(r.ok for r in results)
        # Data dir untouched.
        assert (config.data_dir / "lifeos.db").exists()
        assert not config.failed_db_path().exists()
        # Every step logged a "would" message.
        joined = "\n".join(caplog.messages)
        assert "would SIGTERM pid=4242" in joined
        assert "would move" in joined
        assert "would copy db" in joined
        assert "would copy lance" in joined
        assert "would run: docker compose start nats" in joined
        assert "would run: docker compose start lifeos" in joined
        assert "would GET http://localhost:8080/health" in joined

    def test_dry_run_http_check_returns_true(self) -> None:
        runner = rb.DryRunRunner()
        assert runner.http_check_ok("http://example.invalid", timeout=1.0) is True


# ---------------------------------------------------------------------------
# render_results
# ---------------------------------------------------------------------------
class TestRenderResults:
    def test_render_produces_one_line_per_step(self, config: rb.RollbackConfig) -> None:
        runner = FakeRunner()
        _, results = rb.execute_rollback(config, runner)
        rendered = rb.render_results(results)
        lines = rendered.splitlines()
        assert len(lines) == len(results)
        assert all(line.startswith("OK") for line in lines)

    def test_render_marks_failed_step(self, config: rb.RollbackConfig) -> None:
        runner = FakeRunner(http_result=False)
        _, results = rb.execute_rollback(config, runner)
        rendered = rb.render_results(results)
        assert "FAIL  verify_v1_health" in rendered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCLI:
    def test_parse_args_requires_pid_or_pid_file(self, snapshot: Path) -> None:
        with pytest.raises(SystemExit):
            rb._parse_args(["--snapshot", str(snapshot)])

    def test_parse_args_accepts_explicit_pid(self, snapshot: Path) -> None:
        ns = rb._parse_args(
            [
                "--snapshot",
                str(snapshot),
                "--v2-pid",
                "1234",
            ]
        )
        assert ns.v2_pid == 1234
        assert ns.v2_pid_file is None
        assert ns.dry_run is False

    def test_parse_args_pid_group_is_mutually_exclusive(self, snapshot: Path, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            rb._parse_args(
                [
                    "--snapshot",
                    str(snapshot),
                    "--v2-pid",
                    "1",
                    "--v2-pid-file",
                    str(tmp_path / "p"),
                ]
            )

    def test_parse_args_dry_run_flag(self, snapshot: Path) -> None:
        ns = rb._parse_args(["--snapshot", str(snapshot), "--v2-pid", "1", "--dry-run"])
        assert ns.dry_run is True

    def test_rto_constant_is_thirty_minutes(self) -> None:
        """Runbook and CEO plan both lock the RTO at 30 min; protect it."""
        assert rb.RTO_MINUTES == 30
