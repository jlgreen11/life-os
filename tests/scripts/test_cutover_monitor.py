"""Tests for ``scripts/cutover_monitor.py``.

Exercises every alert branch in the pure :func:`evaluate` function and
the orchestrator :func:`run_loop` via a virtual clock + fake fetch.

All alert-threshold tests drive the state at synthetic Unix timestamps
rather than sleeping — a full 30-min simulated watch window runs in
milliseconds. The happy-path test for :func:`fetch_health_snapshot`
is parametrised against a lightweight stdlib ``http.server`` so the
default production path is still exercised without hitting the real
FastAPI app.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from scripts import cutover_monitor as monitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config(tmp_path: Path) -> monitor.MonitorConfig:
    """Config with every threshold defaulted + alerts_log under tmp_path."""
    return monitor.MonitorConfig(alerts_log=tmp_path / "alerts.jsonl")


def _healthy_snapshot(now_ts: int) -> dict[str, Any]:
    """A fully-green ``/api/health`` body at ``now_ts``."""
    return {
        "ok": True,
        "ts": now_ts,
        "connectors": {
            "proton_mail": "ready",
            "imessage": "ready",
            "caldav": "ready",
            "ios_context": "ready",
        },
        "db_last_write_ts": now_ts - 1,
        "scheduler_heartbeat_ts": now_ts - 1,
        "producer_activity": {"cadence": 0, "relationship": 0, "temporal": 0},
        "pending_moments": 0,
        "notes": [],
    }


# ---------------------------------------------------------------------------
# evaluate — healthy path emits no alerts
# ---------------------------------------------------------------------------
def test_evaluate_healthy_snapshot_emits_nothing(config: monitor.MonitorConfig) -> None:
    state = monitor.MonitorState()
    now_ts = 1_700_000_000
    alerts = monitor.evaluate(state, _healthy_snapshot(now_ts), now_ts, config)
    assert alerts == []
    assert state.active_alert_kinds == set()
    assert state.pending_baseline == 0


# ---------------------------------------------------------------------------
# evaluate — ok:false
# ---------------------------------------------------------------------------
def test_evaluate_ok_false_emits_health_not_ok(config: monitor.MonitorConfig) -> None:
    state = monitor.MonitorState()
    now_ts = 1_700_000_000
    snap = _healthy_snapshot(now_ts)
    snap["ok"] = False
    snap["notes"] = ["scheduler not wired"]
    alerts = monitor.evaluate(state, snap, now_ts, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "health_not_ok"
    assert alerts[0].details == {"notes": ["scheduler not wired"]}


def test_evaluate_ok_false_suppressed_on_repeat_until_recovery(
    config: monitor.MonitorConfig,
) -> None:
    """Each scrape rebuilds a fresh snapshot so only the field-under-test varies.

    Reusing a single dict across virtual-clock ticks would drift
    ``db_last_write_ts`` / ``scheduler_heartbeat_ts`` past their
    thresholds and add spurious cross-alerts.
    """
    state = monitor.MonitorState()

    def degraded(now_ts: int) -> dict[str, Any]:
        snap = _healthy_snapshot(now_ts)
        snap["ok"] = False
        return snap

    first = monitor.evaluate(state, degraded(1_700_000_000), 1_700_000_000, config)
    second = monitor.evaluate(state, degraded(1_700_000_010), 1_700_000_010, config)
    assert len(first) == 1
    assert first[0].kind == "health_not_ok"
    assert second == []  # same kind active → suppressed

    # Recover → suppression clears; re-degrade emits again.
    monitor.evaluate(state, _healthy_snapshot(1_700_000_020), 1_700_000_020, config)
    again = monitor.evaluate(state, degraded(1_700_000_030), 1_700_000_030, config)
    assert len(again) == 1
    assert again[0].kind == "health_not_ok"


# ---------------------------------------------------------------------------
# evaluate — connector offline
# ---------------------------------------------------------------------------
def test_evaluate_connector_offline_under_threshold_is_silent(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["connectors"]["proton_mail"] = "error"
    alerts = monitor.evaluate(state, snap, t0, config)
    assert alerts == []
    # State remembers the offline start so a later poll can fire the alert.
    assert state.connector_offline_since == {"proton_mail": t0}


def test_evaluate_connector_offline_over_threshold_alerts(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000

    def degraded(now_ts: int) -> dict[str, Any]:
        snap = _healthy_snapshot(now_ts)
        snap["connectors"]["proton_mail"] = "error"
        return snap

    monitor.evaluate(state, degraded(t0), t0, config)

    # Later poll past the 5-min threshold trips the alert.
    later = t0 + config.connector_offline_alert_seconds + 1
    alerts = monitor.evaluate(state, degraded(later), later, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "connector_offline:proton_mail"
    assert alerts[0].details["connector_id"] == "proton_mail"
    assert alerts[0].details["status"] == "error"
    assert alerts[0].details["offline_seconds"] > config.connector_offline_alert_seconds


def test_evaluate_connector_recovery_clears_offline_since(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["connectors"]["imessage"] = "paused"
    monitor.evaluate(state, snap, t0, config)
    assert "imessage" in state.connector_offline_since

    # Next poll shows recovery.
    recovered = _healthy_snapshot(t0 + 30)
    monitor.evaluate(state, recovered, t0 + 30, config)
    assert "imessage" not in state.connector_offline_since
    assert "connector_offline:imessage" not in state.active_alert_kinds


# ---------------------------------------------------------------------------
# evaluate — DB write lag
# ---------------------------------------------------------------------------
def test_evaluate_db_write_lag_under_threshold_silent(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["db_last_write_ts"] = t0 - config.db_write_lag_alert_seconds + 1
    alerts = monitor.evaluate(state, snap, t0, config)
    assert alerts == []


def test_evaluate_db_write_lag_over_threshold_alerts(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["db_last_write_ts"] = t0 - config.db_write_lag_alert_seconds - 5
    alerts = monitor.evaluate(state, snap, t0, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "db_write_lag"
    assert alerts[0].details["lag_seconds"] == config.db_write_lag_alert_seconds + 5


# ---------------------------------------------------------------------------
# evaluate — scheduler heartbeat
# ---------------------------------------------------------------------------
def test_evaluate_scheduler_heartbeat_missing_alerts(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["scheduler_heartbeat_ts"] = None
    alerts = monitor.evaluate(state, snap, t0, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "scheduler_heartbeat"
    assert "missing" in alerts[0].message


def test_evaluate_scheduler_heartbeat_stale_alerts(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    snap = _healthy_snapshot(t0)
    snap["scheduler_heartbeat_ts"] = t0 - config.scheduler_heartbeat_alert_seconds - 10
    alerts = monitor.evaluate(state, snap, t0, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "scheduler_heartbeat"
    assert alerts[0].details["stale_seconds"] == config.scheduler_heartbeat_alert_seconds + 10


# ---------------------------------------------------------------------------
# evaluate — pending_moments backlog
# ---------------------------------------------------------------------------
def test_evaluate_pending_backlog_requires_growth_over_threshold(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    # Seed baseline: pending=3 at t0.
    snap = _healthy_snapshot(t0)
    snap["pending_moments"] = 3
    monitor.evaluate(state, snap, t0, config)
    assert state.pending_baseline == 3

    # Still inside window — growth to 5 does NOT alert yet.
    t1 = t0 + 60
    snap_t1 = _healthy_snapshot(t1)
    snap_t1["pending_moments"] = 5
    alerts = monitor.evaluate(state, snap_t1, t1, config)
    assert alerts == []

    # Past the threshold with growth → alerts.
    t2 = t0 + config.pending_growth_alert_seconds + 1
    snap_t2 = _healthy_snapshot(t2)
    snap_t2["pending_moments"] = 7
    alerts = monitor.evaluate(state, snap_t2, t2, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "pending_moments_backlog"
    assert alerts[0].details["pending_moments"] == 7
    assert alerts[0].details["baseline"] == 3
    assert alerts[0].details["no_activity_seconds"] > config.pending_growth_alert_seconds


def test_evaluate_pending_decrease_resets_baseline(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    t0 = 1_700_000_000
    # Seed at 5, then grow to 9 over the threshold → would alert …
    snap = _healthy_snapshot(t0)
    snap["pending_moments"] = 5
    monitor.evaluate(state, snap, t0, config)

    snap2 = _healthy_snapshot(t0 + 100)
    snap2["pending_moments"] = 9
    monitor.evaluate(state, snap2, t0 + 100, config)

    # … but an accept/dismiss drops it to 2 → baseline resets to 2.
    snap3 = _healthy_snapshot(t0 + 200)
    snap3["pending_moments"] = 2
    alerts = monitor.evaluate(state, snap3, t0 + 200, config)
    assert alerts == []
    assert state.pending_baseline == 2
    assert state.pending_last_decrease_ts == t0 + 200

    # Growing back past threshold STILL doesn't alert until the new
    # threshold window elapses since the reset.
    t_soon = t0 + 200 + 60
    snap4 = _healthy_snapshot(t_soon)
    snap4["pending_moments"] = 4
    assert monitor.evaluate(state, snap4, t_soon, config) == []

    t_late = t0 + 200 + config.pending_growth_alert_seconds + 1
    snap5 = _healthy_snapshot(t_late)
    snap5["pending_moments"] = 7
    alerts_late = monitor.evaluate(state, snap5, t_late, config)
    assert len(alerts_late) == 1
    assert alerts_late[0].details["baseline"] == 2


# ---------------------------------------------------------------------------
# evaluate — HTTP error
# ---------------------------------------------------------------------------
def test_evaluate_http_error_emits_http_error_alert(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    alerts = monitor.evaluate(state, None, 1_700_000_000, config)
    assert len(alerts) == 1
    assert alerts[0].kind == "http_error"


def test_evaluate_http_error_clears_on_successful_parse(
    config: monitor.MonitorConfig,
) -> None:
    state = monitor.MonitorState()
    monitor.evaluate(state, None, 1_700_000_000, config)
    assert "http_error" in state.active_alert_kinds
    monitor.evaluate(state, _healthy_snapshot(1_700_000_030), 1_700_000_030, config)
    assert "http_error" not in state.active_alert_kinds


# ---------------------------------------------------------------------------
# Alert.as_jsonl_line
# ---------------------------------------------------------------------------
def test_alert_as_jsonl_line_is_single_json_line_newline_terminated() -> None:
    alert = monitor.Alert(kind="db_write_lag", message="stale", ts=42, details={"lag": 99})
    line = alert.as_jsonl_line()
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed == {
        "kind": "db_write_lag",
        "message": "stale",
        "ts": 42,
        "details": {"lag": 99},
    }


# ---------------------------------------------------------------------------
# append_alert writes to JSONL
# ---------------------------------------------------------------------------
def test_append_alert_creates_parent_dir_and_appends(tmp_path: Path) -> None:
    alerts_log = tmp_path / "subdir" / "alerts.jsonl"
    monitor.append_alert(alerts_log, monitor.Alert(kind="a", message="m1", ts=1))
    monitor.append_alert(alerts_log, monitor.Alert(kind="b", message="m2", ts=2))
    lines = alerts_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["kind"] == "a"
    assert json.loads(lines[1])["kind"] == "b"


# ---------------------------------------------------------------------------
# run_loop — happy path exits 0 after healthy_for_minutes
# ---------------------------------------------------------------------------
class _VirtualClock:
    """Monotonically increasing clock the orchestrator can advance by sleep()."""

    def __init__(self, start_ts: int) -> None:
        self.now_ts = start_ts

    def __call__(self) -> int:
        return self.now_ts

    def advance(self, seconds: float) -> None:
        self.now_ts += int(seconds)


def test_run_loop_exits_healthy_after_window(tmp_path: Path) -> None:
    clock = _VirtualClock(1_700_000_000)
    config = monitor.MonitorConfig(
        alerts_log=tmp_path / "alerts.jsonl",
        healthy_for_minutes=1,
        poll_interval_seconds=10,
    )
    snapshots = iter([_healthy_snapshot(1_700_000_000 + i * 10) for i in range(20)])

    def fetch() -> dict[str, Any] | None:
        return next(snapshots)

    rc = monitor.run_loop(
        config,
        fetch=fetch,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        log=logging.getLogger("test"),
        max_iterations=20,
    )
    assert rc == monitor.EXIT_HEALTHY
    # Never wrote an alert line.
    assert not config.alerts_log.exists()


def test_run_loop_exits_alert_on_first_alert_and_writes_jsonl(tmp_path: Path) -> None:
    clock = _VirtualClock(1_700_000_000)
    config = monitor.MonitorConfig(
        alerts_log=tmp_path / "alerts.jsonl",
        healthy_for_minutes=60,
        poll_interval_seconds=10,
    )

    def fetch() -> dict[str, Any] | None:
        # Always reports ok=False so the very first scrape alerts.
        snap = _healthy_snapshot(clock())
        snap["ok"] = False
        return snap

    rc = monitor.run_loop(
        config,
        fetch=fetch,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        log=logging.getLogger("test"),
        max_iterations=5,
    )
    assert rc == monitor.EXIT_ALERT
    lines = config.alerts_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "health_not_ok"


def test_run_loop_exits_alert_on_http_error(tmp_path: Path) -> None:
    clock = _VirtualClock(1_700_000_000)
    config = monitor.MonitorConfig(
        alerts_log=tmp_path / "alerts.jsonl",
        healthy_for_minutes=60,
    )

    rc = monitor.run_loop(
        config,
        fetch=lambda: None,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        log=logging.getLogger("test"),
        max_iterations=3,
    )
    assert rc == monitor.EXIT_ALERT
    lines = config.alerts_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "http_error"


# ---------------------------------------------------------------------------
# fetch_health_snapshot — network-level contract against stdlib http.server
# ---------------------------------------------------------------------------
class _HealthHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler: GET /api/health → 200 with a healthy body."""

    def do_GET(self) -> None:
        if self.path != "/api/health":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(_healthy_snapshot(42)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object, **_kwargs: object) -> None:
        """Silence stdlib access log during tests."""


@pytest.fixture
def local_health_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_fetch_health_snapshot_parses_live_endpoint(local_health_server: str) -> None:
    config = monitor.MonitorConfig(health_url=local_health_server)
    snap = monitor.fetch_health_snapshot(config)
    assert snap is not None
    assert snap["ok"] is True
    assert snap["pending_moments"] == 0


def test_fetch_health_snapshot_returns_none_on_connection_refused(tmp_path: Path) -> None:
    # Port 1 is (on practically every machine) a refused connection.
    config = monitor.MonitorConfig(
        health_url="http://127.0.0.1:1/api/health",
        http_timeout_seconds=0.5,
        alerts_log=tmp_path / "alerts.jsonl",
    )
    assert monitor.fetch_health_snapshot(config) is None


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------
def test_config_from_args_round_trips_every_flag(tmp_path: Path) -> None:
    ns = monitor._parse_args(
        [
            "--health-url",
            "http://example.invalid/api/health",
            "--poll-interval-seconds",
            "15",
            "--healthy-for-minutes",
            "30",
            "--alerts-log",
            str(tmp_path / "a.jsonl"),
            "--connector-offline-alert-seconds",
            "60",
            "--db-write-lag-alert-seconds",
            "90",
            "--scheduler-heartbeat-alert-seconds",
            "45",
            "--pending-growth-alert-seconds",
            "180",
            "--http-timeout-seconds",
            "2.5",
        ]
    )
    cfg = monitor._config_from_args(ns)
    assert cfg.health_url == "http://example.invalid/api/health"
    assert cfg.poll_interval_seconds == 15
    assert cfg.healthy_for_minutes == 30
    assert cfg.alerts_log == tmp_path / "a.jsonl"
    assert cfg.connector_offline_alert_seconds == 60
    assert cfg.db_write_lag_alert_seconds == 90
    assert cfg.scheduler_heartbeat_alert_seconds == 45
    assert cfg.pending_growth_alert_seconds == 180
    assert cfg.http_timeout_seconds == 2.5
