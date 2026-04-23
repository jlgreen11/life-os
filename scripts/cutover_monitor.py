#!/usr/bin/env python3
"""Cutover watch-window monitor for the v1 → v2 cutover.

Implements NEXT_TASKS Category C § "Cutover monitor script" and the
watch-window checks the cutover runbook § 6.1 locks in. Polls
``GET /api/health`` on a fixed cadence (default 10 s), tracks state
across polls, and emits structured alerts (``log.error`` + append to
``data/cutover-alerts.jsonl``) whenever one of the CEO-plan alert
thresholds trips:

* ``ok: false`` at any scrape (CEO plan § "rollback trigger 1")
* connector offline > 5 min (``connectors[id] != 'ready'``)
* DB last-write lag > 30 s (``now - db_last_write_ts``)
* scheduler heartbeat missing / stale > 2 min
* pending-Moment backlog rising without accept/dismiss > 15 min
* HTTP-level failure reaching ``/api/health``

Exit codes (so launchd / an operator wrapper can treat this as a
conventional POSIX daemon):

* ``0`` — healthy for ``--healthy-for-minutes`` of wall-clock with zero
  alerts fired.
* ``1`` — at least one alert fired. The monitor exits on the first
  alert; operators who need continuous tailing relaunch the process
  from the runbook after addressing the trigger.

Design notes
------------
The scrape → alert computation is factored into a pure
:func:`evaluate` function that takes a :class:`MonitorState`, the latest
health snapshot, and a virtual ``now_ts``. This lets the XCTest-style
suite drive the monitor through entire simulated watch windows without
sleeping or running an HTTP server. The loop orchestrator
:func:`run_loop` is the only piece that touches ``time.sleep`` /
``urllib`` / the filesystem.

Alerts of the same ``kind`` are suppressed until the underlying signal
recovers (``active_alert_kinds`` on :class:`MonitorState`). During a
30-min connector outage this keeps the JSONL to one row per outage
rather than one per scrape.

References
----------
- Task spec: ``NEXT_TASKS.md`` § Category C.
- Runbook: ``docs/cutover-runbook.md`` § 6.1 "What to watch".
- Health schema: ``api/schemas.py::HealthOut``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALERTS_LOG = REPO_ROOT / "data" / "cutover-alerts.jsonl"
DEFAULT_HEALTH_URL = "http://localhost:8080/api/health"

EXIT_HEALTHY = 0
EXIT_ALERT = 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonitorConfig:
    """Alert thresholds + loop cadence.

    Defaults match the cutover runbook § 6.1 alert-threshold table.
    Every value is a constructor arg so tests can drive
    :func:`evaluate` at arbitrary virtual clocks without sleeping.
    """

    health_url: str = DEFAULT_HEALTH_URL
    poll_interval_seconds: int = 10
    healthy_for_minutes: int = 10
    alerts_log: Path = DEFAULT_ALERTS_LOG
    connector_offline_alert_seconds: int = 300  # 5 min
    db_write_lag_alert_seconds: int = 30
    scheduler_heartbeat_alert_seconds: int = 120  # 2 min
    pending_growth_alert_seconds: int = 900  # 15 min
    http_timeout_seconds: float = 5.0


# ---------------------------------------------------------------------------
# Alert + state
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """One alert emission. Serialised to JSONL + emitted via ``log.error``."""

    kind: str
    message: str
    ts: int
    details: dict[str, Any] = field(default_factory=dict)

    def as_jsonl_line(self) -> str:
        """Return one newline-terminated JSON line for the append-only log."""
        return (
            json.dumps(
                {
                    "ts": self.ts,
                    "kind": self.kind,
                    "message": self.message,
                    "details": self.details,
                },
                sort_keys=True,
            )
            + "\n"
        )


@dataclass
class MonitorState:
    """Stateful tracking kept across polls.

    :func:`evaluate` reads and writes this in place — the orchestrator
    never mutates it directly. Fresh instances start the watch window
    at ``pending_last_count=None``; the first scrape seeds the baseline.
    """

    connector_offline_since: dict[str, int] = field(default_factory=dict)
    pending_last_count: int | None = None
    pending_baseline: int | None = None
    pending_last_decrease_ts: int | None = None
    alerts_fired: int = 0
    scrapes: int = 0
    # kind → True while the underlying signal is still alerting; prevents
    # duplicate rows in the JSONL during a sustained outage.
    active_alert_kinds: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Pure evaluator
# ---------------------------------------------------------------------------
def evaluate(
    state: MonitorState,
    health: dict[str, Any] | None,
    now_ts: int,
    config: MonitorConfig,
) -> list[Alert]:
    """Compute alerts for one scrape and update ``state`` in place.

    Parameters
    ----------
    state:
        The rolling monitor state. Mutated (connector-offline map,
        pending-moments history, active-alert-kinds set).
    health:
        The parsed ``/api/health`` JSON body, or ``None`` to signal an
        HTTP-level failure (connection refused, timeout, 5xx, bad JSON).
    now_ts:
        Unix-seconds clock reading for this scrape. Tests supply a
        virtual clock so scenarios span hours without sleeping.
    config:
        Thresholds for each alert type.

    Returns
    -------
    list[Alert]
        Zero or more alerts to emit. Same-kind alerts already in
        ``state.active_alert_kinds`` are suppressed; they re-emit once
        the underlying signal recovers and then degrades again.
    """
    if health is None:
        return _emit_if_new(
            state,
            Alert(
                kind="http_error",
                message="failed to reach /api/health",
                ts=now_ts,
            ),
        )

    alerts: list[Alert] = []

    # ok:false (CEO plan rollback trigger 1)
    if not health.get("ok", True):
        alerts.extend(
            _emit_if_new(
                state,
                Alert(
                    kind="health_not_ok",
                    message="/api/health returned ok=False",
                    ts=now_ts,
                    details={"notes": health.get("notes", [])},
                ),
            )
        )
    else:
        state.active_alert_kinds.discard("health_not_ok")

    # http_error clears on a successful parse even when the payload
    # itself is degraded — the "cannot reach endpoint" case is gone.
    state.active_alert_kinds.discard("http_error")

    # Connector offline > threshold
    connectors = health.get("connectors") or {}
    for cid, status in connectors.items():
        kind = f"connector_offline:{cid}"
        if status != "ready":
            state.connector_offline_since.setdefault(cid, now_ts)
            offline_for = now_ts - state.connector_offline_since[cid]
            if offline_for > config.connector_offline_alert_seconds:
                alerts.extend(
                    _emit_if_new(
                        state,
                        Alert(
                            kind=kind,
                            message=(
                                f"connector {cid} status={status} for {offline_for}s "
                                f"(> {config.connector_offline_alert_seconds}s)"
                            ),
                            ts=now_ts,
                            details={
                                "connector_id": cid,
                                "status": status,
                                "offline_seconds": offline_for,
                            },
                        ),
                    )
                )
        else:
            state.connector_offline_since.pop(cid, None)
            state.active_alert_kinds.discard(kind)

    # db_last_write_ts lag
    db_ts = health.get("db_last_write_ts")
    if isinstance(db_ts, int):
        lag = now_ts - db_ts
        if lag > config.db_write_lag_alert_seconds:
            alerts.extend(
                _emit_if_new(
                    state,
                    Alert(
                        kind="db_write_lag",
                        message=(f"db_last_write_ts stale by {lag}s (> {config.db_write_lag_alert_seconds}s)"),
                        ts=now_ts,
                        details={"db_last_write_ts": db_ts, "lag_seconds": lag},
                    ),
                )
            )
        else:
            state.active_alert_kinds.discard("db_write_lag")

    # Scheduler heartbeat missing or stale
    sched_ts = health.get("scheduler_heartbeat_ts")
    if not isinstance(sched_ts, int):
        alerts.extend(
            _emit_if_new(
                state,
                Alert(
                    kind="scheduler_heartbeat",
                    message="scheduler_heartbeat_ts is missing from /api/health",
                    ts=now_ts,
                ),
            )
        )
    else:
        stale = now_ts - sched_ts
        if stale > config.scheduler_heartbeat_alert_seconds:
            alerts.extend(
                _emit_if_new(
                    state,
                    Alert(
                        kind="scheduler_heartbeat",
                        message=(
                            f"scheduler heartbeat stale by {stale}s (> {config.scheduler_heartbeat_alert_seconds}s)"
                        ),
                        ts=now_ts,
                        details={"scheduler_heartbeat_ts": sched_ts, "stale_seconds": stale},
                    ),
                )
            )
        else:
            state.active_alert_kinds.discard("scheduler_heartbeat")

    # pending_moments backlog growing without accept/dismiss.
    # Heuristic: accept/dismiss decrement pending_moments. If the count
    # is monotonically non-decreasing for the threshold window and
    # strictly above the last baseline, the backlog is growing unbounded.
    pending = health.get("pending_moments")
    if isinstance(pending, int):
        if state.pending_last_count is None:
            state.pending_last_count = pending
            state.pending_baseline = pending
            state.pending_last_decrease_ts = now_ts
        else:
            if pending < state.pending_last_count:
                state.pending_last_decrease_ts = now_ts
                state.pending_baseline = pending
            state.pending_last_count = pending

        baseline = state.pending_baseline
        last_dec_ts = state.pending_last_decrease_ts
        if (
            baseline is not None
            and last_dec_ts is not None
            and pending > baseline
            and now_ts - last_dec_ts > config.pending_growth_alert_seconds
        ):
            alerts.extend(
                _emit_if_new(
                    state,
                    Alert(
                        kind="pending_moments_backlog",
                        message=(
                            f"pending_moments grew from {baseline} to {pending} "
                            f"with no decreases for {now_ts - last_dec_ts}s "
                            f"(> {config.pending_growth_alert_seconds}s)"
                        ),
                        ts=now_ts,
                        details={
                            "pending_moments": pending,
                            "baseline": baseline,
                            "no_activity_seconds": now_ts - last_dec_ts,
                        },
                    ),
                )
            )
        else:
            state.active_alert_kinds.discard("pending_moments_backlog")

    return alerts


def _emit_if_new(state: MonitorState, alert: Alert) -> list[Alert]:
    """Return ``[alert]`` the first time its ``kind`` enters the active set.

    Subsequent calls while the signal is still unhealthy return ``[]``
    so the JSONL / logger don't flood. Callers are responsible for
    ``state.active_alert_kinds.discard(kind)`` on recovery.
    """
    if alert.kind in state.active_alert_kinds:
        return []
    state.active_alert_kinds.add(alert.kind)
    return [alert]


# ---------------------------------------------------------------------------
# HTTP + I/O side effects
# ---------------------------------------------------------------------------
def fetch_health_snapshot(config: MonitorConfig) -> dict[str, Any] | None:
    """Fetch and parse ``/api/health`` once.

    Returns ``None`` on any failure mode the monitor treats uniformly
    as "endpoint unreachable": connection refused, timeout, non-200
    status, or JSON parse error. All other exceptions propagate so a
    bug in the script itself isn't silently swallowed.
    """
    try:
        with urllib.request.urlopen(
            config.health_url,
            timeout=config.http_timeout_seconds,
        ) as resp:
            if resp.status != 200:
                return None
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def append_alert(alerts_log: Path, alert: Alert) -> None:
    """Append one alert to the JSONL file, creating the parent dir lazily."""
    alerts_log.parent.mkdir(parents=True, exist_ok=True)
    with alerts_log.open("a", encoding="utf-8") as fh:
        fh.write(alert.as_jsonl_line())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_loop(
    config: MonitorConfig,
    *,
    fetch: Callable[[], dict[str, Any] | None] | None = None,
    clock: Callable[[], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: logging.Logger | None = None,
    max_iterations: int | None = None,
) -> int:
    """Poll ``/api/health`` until healthy-for-N-minutes or first alert.

    Parameters
    ----------
    config:
        Thresholds and cadence (see :class:`MonitorConfig`).
    fetch:
        Injectable snapshot fetcher. Default: :func:`fetch_health_snapshot`
        against the configured URL. Tests pass a deterministic list-
        backed callable.
    clock:
        Injectable clock returning Unix seconds. Default: ``time.time``.
        Tests pass a virtual clock.
    sleep:
        Injectable sleep. Default: :func:`time.sleep`. Tests pass a
        no-op.
    log:
        Logger; defaults to the module logger.
    max_iterations:
        Hard ceiling on poll count. ``None`` means "no ceiling"
        (normal operation). Tests pass a small int as a safety net.

    Returns
    -------
    int
        :data:`EXIT_HEALTHY` (0) after ``healthy_for_minutes`` with no
        alerts; :data:`EXIT_ALERT` (1) on the first alert emission.
    """
    log = log or logging.getLogger("cutover_monitor")
    fetch_fn = fetch or (lambda: fetch_health_snapshot(config))
    now_fn: Callable[[], int] = clock or (lambda: int(time.time()))

    state = MonitorState()
    start_ts = now_fn()
    healthy_for_seconds = config.healthy_for_minutes * 60

    iterations = 0
    while True:
        if max_iterations is not None and iterations >= max_iterations:
            log.info("max_iterations=%d reached without alert; exiting healthy", max_iterations)
            return EXIT_HEALTHY
        iterations += 1

        current = now_fn()
        try:
            health = fetch_fn()
        except Exception:  # pragma: no cover — defence-in-depth
            log.exception("fetch raised unexpectedly; treating as http_error")
            health = None

        alerts = evaluate(state, health, current, config)
        state.scrapes += 1
        _log_scrape(log, current, health)

        for alert in alerts:
            log.error("ALERT %s: %s", alert.kind, alert.message)
            append_alert(config.alerts_log, alert)
            state.alerts_fired += 1

        if alerts:
            return EXIT_ALERT

        elapsed = current - start_ts
        if elapsed >= healthy_for_seconds:
            log.info(
                "healthy_for_minutes=%d reached (scrapes=%d); exiting",
                config.healthy_for_minutes,
                state.scrapes,
            )
            return EXIT_HEALTHY

        sleep(config.poll_interval_seconds)


def _log_scrape(
    log: logging.Logger,
    now_ts: int,
    health: dict[str, Any] | None,
) -> None:
    """Emit one INFO line per scrape so operators can tail the monitor."""
    if health is None:
        log.info("scrape ts=%d http_error", now_ts)
        return
    log.info(
        "scrape ts=%d ok=%s connectors=%s pending=%s",
        now_ts,
        health.get("ok"),
        health.get("connectors"),
        health.get("pending_moments"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cutover watch-window monitor (polls /api/health)")
    parser.add_argument("--health-url", default=DEFAULT_HEALTH_URL)
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument(
        "--healthy-for-minutes",
        type=int,
        default=10,
        help="Exit 0 after this many minutes of clean polling.",
    )
    parser.add_argument(
        "--alerts-log",
        type=Path,
        default=DEFAULT_ALERTS_LOG,
        help="Append-only JSONL file for alerts. Parent dir is created lazily.",
    )
    parser.add_argument("--connector-offline-alert-seconds", type=int, default=300)
    parser.add_argument("--db-write-lag-alert-seconds", type=int, default=30)
    parser.add_argument("--scheduler-heartbeat-alert-seconds", type=int, default=120)
    parser.add_argument("--pending-growth-alert-seconds", type=int, default=900)
    parser.add_argument("--http-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("cutover_monitor")


def _config_from_args(ns: argparse.Namespace) -> MonitorConfig:
    return MonitorConfig(
        health_url=ns.health_url,
        poll_interval_seconds=ns.poll_interval_seconds,
        healthy_for_minutes=ns.healthy_for_minutes,
        alerts_log=ns.alerts_log,
        connector_offline_alert_seconds=ns.connector_offline_alert_seconds,
        db_write_lag_alert_seconds=ns.db_write_lag_alert_seconds,
        scheduler_heartbeat_alert_seconds=ns.scheduler_heartbeat_alert_seconds,
        pending_growth_alert_seconds=ns.pending_growth_alert_seconds,
        http_timeout_seconds=ns.http_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(sys.argv[1:] if argv is None else argv))
    log = _setup_logging(ns.verbose)
    config = _config_from_args(ns)
    log.info(
        "cutover_monitor starting: url=%s poll=%ds healthy_for=%dm alerts_log=%s",
        config.health_url,
        config.poll_interval_seconds,
        config.healthy_for_minutes,
        config.alerts_log,
    )
    log.info("started_at=%s", dt.datetime.now(tz=dt.UTC).isoformat())
    return run_loop(config, log=log)


if __name__ == "__main__":
    sys.exit(main())
