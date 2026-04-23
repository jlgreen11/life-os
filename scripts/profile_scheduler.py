#!/usr/bin/env python3
"""Profile the v2 scheduler's tick latency + throughput at 10K-row scale.

Seeds a synthetic Moment fleet of ``count`` rows (default 10,000) spread
across the next 24 hours, runs ``ticks`` simulated scheduler ticks with
an injected wall clock that advances by ``tick_seconds`` per tick (so
``ticks * tick_seconds`` = simulated window), and emits
``docs/perf-scheduler-{date}.md`` with p50/p95/p99 latency + throughput.

Acceptance (NEXT_TASKS.md § "Scheduler performance profile"):
    p99 tick latency < 500 ms at the 10K-row fleet.

Implementation notes
--------------------
- Stdlib only (``sqlite3``/``statistics``/``random``/``asyncio``). No
  pytest-benchmark; we measure with :func:`time.perf_counter`.
- Seeding goes through raw SQL for speed (``executemany`` inside one
  ``BEGIN IMMEDIATE``). The hot path under test is ``Scheduler.tick``,
  not inserts, so we skip the per-row repo.create round-trip.
- The scheduler is wired to the real :class:`MomentRepository` and
  :class:`OutboxRepository` so fire-path overhead is included — SQL
  transitions, outbox enqueues, and JSON serialization all count.
- Runs against a tempfile DB opened WAL / synchronous=NORMAL (the same
  PRAGMAs the production process uses). The tempfile is cleaned up on
  exit; production DBs at ``data/*`` are never touched (see agent deny
  list in ``.claude/settings.json``).

References
----------
- CEO plan: ``~/.gstack/projects/jlgreen11-life-os/ceo-plans/2026-04-21-life-os-rewrite-mvp.md``
  § "Scheduler & Moment lifecycle".
- Engineering plan: ``docs/plans/2026-04-21-v2-rewrite-plan.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.moment.scheduler import Scheduler
from core.moment.types import InsightType, MomentState
from storage import schema
from storage.repos.moments import MomentRepository
from storage.repos.outbox import OutboxRepository

# Reference epoch — same fixed stamp the scheduler/moment repo tests use so
# the report's "simulated from" timestamp is reproducible run-to-run.
REF_NOW = 1_777_204_800  # 2026-04-22T12:00:00Z

DEFAULT_COUNT = 10_000
DEFAULT_TICKS = 120  # 120 * 30s = 3600s = 1 simulated hour
DEFAULT_TICK_SECONDS = 30
DEFAULT_BATCH_LIMIT = 1000
DEFAULT_SEED = 42

# Acceptance threshold from NEXT_TASKS.md.
P99_LATENCY_BUDGET_MS = 500.0

_INSIGHT_TYPES = tuple(t.value for t in InsightType)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/scripts/test_profile_scheduler.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MomentSeed:
    """One planned synthetic Moment row.

    Held as a plain dataclass so :func:`plan_seeds` stays deterministic
    and unit-testable — the SQL-writing half
    (:func:`insert_seeds`) wraps ``executemany`` around the same list.
    """

    evidence_hash: str
    scheduled_for: int | None
    snooze_until: int | None
    state: str
    source_insight_type: str
    expires_at: int


def plan_seeds(
    count: int,
    *,
    now_ts: int = REF_NOW,
    rng_seed: int = DEFAULT_SEED,
    horizon_seconds: int = 86400,
) -> list[MomentSeed]:
    """Plan ``count`` :class:`MomentSeed` rows; deterministic on ``rng_seed``.

    Distribution (spec: "mix of scheduled + snoozed, time-distributed
    across next 24h"):

    - 70% SUGGESTED with ``scheduled_for`` uniform in
      ``[now_ts, now_ts + horizon_seconds]``.
    - 20% SNOOZED with ``snooze_until`` uniform in the same window and
      the matching ``scheduled_for`` stamp so :meth:`list_scheduled` sees
      it (the scheduler reads ``scheduled_for`` for both states).
    - 10% SUGGESTED with ``scheduled_for=None`` — context-trigger-only
      Moments that :meth:`list_scheduled` filters out, providing a
      realistic "not-every-row-is-time-anchored" baseline.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(rng_seed)
    ttl = 72 * 3600
    out: list[MomentSeed] = []
    for i in range(count):
        bucket = i % 10
        insight = rng.choice(_INSIGHT_TYPES)
        if bucket < 7:
            scheduled = now_ts + rng.randint(0, horizon_seconds)
            state = MomentState.SUGGESTED.value
            snooze = None
        elif bucket < 9:
            scheduled = now_ts + rng.randint(0, horizon_seconds)
            state = MomentState.SNOOZED.value
            snooze = scheduled
        else:
            scheduled = None
            state = MomentState.SUGGESTED.value
            snooze = None
        out.append(
            MomentSeed(
                evidence_hash=f"seed-{i:08d}",
                scheduled_for=scheduled,
                snooze_until=snooze,
                state=state,
                source_insight_type=insight,
                expires_at=now_ts + ttl,
            )
        )
    return out


def insert_seeds(conn: sqlite3.Connection, seeds: Sequence[MomentSeed], *, now_ts: int = REF_NOW) -> None:
    """Bulk-insert ``seeds`` into ``moments`` + ``moment_state_history``.

    One ``BEGIN IMMEDIATE`` / ``executemany`` / ``COMMIT`` per table. The
    scheduler's fire path reads :meth:`MomentRepository.get` after a
    transition and that in turn hydrates state history, so we populate
    the creation rows here — otherwise the first fire would see a Moment
    whose state history is empty.
    """
    action_json = json.dumps({"kind": "nudge", "params": {"contact_id": "c1"}}, sort_keys=True)
    moment_rows = []
    history_rows = []
    for seed in seeds:
        mid = str(uuid.uuid4())
        moment_rows.append(
            (
                mid,
                now_ts,
                seed.scheduled_for,
                seed.expires_at,
                None,
                "synthetic insight",
                "[]",
                seed.evidence_hash,
                action_json,
                seed.state,
                seed.snooze_until,
                0.0,
                1.0,
                seed.source_insight_type,
                now_ts,
            )
        )
        history_rows.append((mid, None, seed.state, now_ts, "create"))

    prev_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            """
            INSERT INTO moments (
                id, created_at, scheduled_for, expires_at, context_trigger,
                insight, evidence, evidence_hash, proposed_action, state,
                snooze_until, confidence, feedback_weight, source_insight_type,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            moment_rows,
        )
        conn.executemany(
            "INSERT INTO moment_state_history (moment_id, from_state, to_state, ts, annotation) "
            "VALUES (?, ?, ?, ?, ?)",
            history_rows,
        )
        conn.execute("COMMIT")
    finally:
        conn.isolation_level = prev_isolation


def percentile(samples: Sequence[float], pct: float) -> float:
    """Return the ``pct`` (0–100) percentile of ``samples`` by linear interpolation.

    Stdlib ``statistics.quantiles`` only exposes fixed quantile buckets;
    we want arbitrary p95/p99 on an ordered float list, so this
    re-implements the classic "index = (n-1) * p" rule. Empty input
    returns ``0.0`` so the summary renderer doesn't need to null-check.
    """
    if not samples:
        return 0.0
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


@dataclass(frozen=True)
class ProfileSummary:
    """Aggregated output of one profile run."""

    count: int
    ticks: int
    tick_seconds: int
    simulated_seconds: int
    total_fires: int
    fires_per_tick_mean: float
    fires_per_simulated_second: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    p99_under_budget: bool


def summarize(
    tick_latencies_seconds: Sequence[float],
    fires_per_tick: Sequence[int],
    *,
    count: int,
    tick_seconds: int,
    budget_ms: float = P99_LATENCY_BUDGET_MS,
) -> ProfileSummary:
    """Fold raw per-tick samples into the :class:`ProfileSummary` shape."""
    ticks = len(tick_latencies_seconds)
    total_fires = sum(fires_per_tick)
    simulated = ticks * tick_seconds
    latencies_ms = [s * 1000.0 for s in tick_latencies_seconds]
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    p99 = percentile(latencies_ms, 99)
    max_ms = max(latencies_ms) if latencies_ms else 0.0
    return ProfileSummary(
        count=count,
        ticks=ticks,
        tick_seconds=tick_seconds,
        simulated_seconds=simulated,
        total_fires=total_fires,
        fires_per_tick_mean=(total_fires / ticks) if ticks else 0.0,
        fires_per_simulated_second=(total_fires / simulated) if simulated else 0.0,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=max_ms,
        p99_under_budget=p99 < budget_ms,
    )


def render_report(
    summary: ProfileSummary,
    *,
    now_ts: int = REF_NOW,
    rng_seed: int = DEFAULT_SEED,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
    budget_ms: float = P99_LATENCY_BUDGET_MS,
    generated_at: datetime | None = None,
) -> str:
    """Render the markdown body for ``docs/perf-scheduler-{date}.md``."""
    ts = generated_at or datetime.now(tz=UTC)
    verdict = "PASS" if summary.p99_under_budget else "FAIL"
    lines: list[str] = [
        f"# Scheduler performance profile — {ts.strftime('%Y-%m-%d')}",
        "",
        f"- Generated: {ts.isoformat(timespec='seconds')}",
        f"- Script: `scripts/profile_scheduler.py` (rng_seed={rng_seed})",
        f"- Simulated reference now: {datetime.fromtimestamp(now_ts, tz=UTC).isoformat()}",
        "",
        "## Setup",
        "",
        f"- Fleet size: **{summary.count:,}** synthetic Moments",
        "  - 70% SUGGESTED scheduled, 20% SNOOZED, 10% context-trigger-only",
        f"  - `scheduled_for` uniform across next 24h from reference now",
        f"- Ticks: **{summary.ticks}** × `tick_seconds={summary.tick_seconds}` = "
        f"{summary.simulated_seconds:,}s of simulated wall time",
        f"- Batch limit per tick: {batch_limit:,}",
        "- Storage: tempfile SQLite, WAL + synchronous=NORMAL (production PRAGMAs)",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| p50 tick latency | {summary.p50_ms:.2f} ms |",
        f"| p95 tick latency | {summary.p95_ms:.2f} ms |",
        f"| p99 tick latency | **{summary.p99_ms:.2f} ms** |",
        f"| max tick latency | {summary.max_ms:.2f} ms |",
        f"| Total fires | {summary.total_fires:,} |",
        f"| Mean fires / tick | {summary.fires_per_tick_mean:.2f} |",
        f"| Throughput | {summary.fires_per_simulated_second:.2f} fires / simulated second |",
        "",
        "## Acceptance",
        "",
        f"- Budget: p99 tick latency < **{budget_ms:.0f} ms** at {summary.count:,}-row fleet",
        f"- Result: **{verdict}** (p99 = {summary.p99_ms:.2f} ms)",
        "",
        "## Reproduce",
        "",
        "```",
        f"python scripts/profile_scheduler.py --count {summary.count} \\",
        f"    --ticks {summary.ticks} --tick-seconds {summary.tick_seconds} \\",
        f"    --batch-limit {batch_limit} --seed {rng_seed}",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _AdvancingClock:
    """Wall-clock stand-in; ``advance(seconds)`` moves the stamp forward."""

    def __init__(self, t: float = REF_NOW) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for stmt in schema.get_all_ddl():
        conn.execute(stmt)
    conn.commit()
    return conn


def run_profile(
    *,
    count: int = DEFAULT_COUNT,
    ticks: int = DEFAULT_TICKS,
    tick_seconds: int = DEFAULT_TICK_SECONDS,
    batch_limit: int = DEFAULT_BATCH_LIMIT,
    rng_seed: int = DEFAULT_SEED,
    now_ts: int = REF_NOW,
    db_path: Path | None = None,
    perf_counter: Callable[[], float] | None = None,
) -> ProfileSummary:
    """End-to-end: seed fleet → run ticks → summarize. Returns the summary.

    ``db_path`` defaults to a fresh tempfile that is cleaned up on exit.
    ``perf_counter`` is injectable so the unit tests can freeze latency.
    """
    perf = perf_counter or time.perf_counter
    tmp_handle: tempfile.TemporaryDirectory[str] | None = None
    if db_path is None:
        tmp_handle = tempfile.TemporaryDirectory(prefix="scheduler-profile-")
        db_path = Path(tmp_handle.name) / "profile.db"
    try:
        conn = _open_db(db_path)
        try:
            seeds = plan_seeds(count, now_ts=now_ts, rng_seed=rng_seed)
            insert_seeds(conn, seeds, now_ts=now_ts)

            clock = _AdvancingClock(now_ts)
            moments_repo = MomentRepository(conn, now_fn=clock)
            outbox_repo = OutboxRepository(conn, now_fn=clock)
            scheduler = Scheduler(
                moments_repo,
                outbox_repo,
                now_fn=clock,
                monotonic_fn=time.monotonic,
                batch_limit=batch_limit,
            )

            latencies: list[float] = []
            fire_counts: list[int] = []
            for _ in range(ticks):
                clock.advance(tick_seconds)
                start = perf()
                records = asyncio.run(scheduler.tick())
                latencies.append(perf() - start)
                fire_counts.append(len(records))
        finally:
            conn.close()
    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()

    return summarize(
        latencies,
        fire_counts,
        count=count,
        tick_seconds=tick_seconds,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _default_report_path(ts: datetime) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "docs" / f"perf-scheduler-{ts.strftime('%Y-%m-%d')}.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Synthetic Moment count (default: %(default)s).")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="Number of simulated ticks (default: %(default)s).")
    parser.add_argument("--tick-seconds", type=int, default=DEFAULT_TICK_SECONDS, help="Simulated wall-clock advance per tick (default: %(default)s).")
    parser.add_argument("--batch-limit", type=int, default=DEFAULT_BATCH_LIMIT, help="Scheduler batch limit (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (default: %(default)s).")
    parser.add_argument("--output", type=Path, default=None, help="Override report output path.")
    parser.add_argument("--no-report", action="store_true", help="Skip writing the markdown report; print summary only.")
    args = parser.parse_args(argv)

    summary = run_profile(
        count=args.count,
        ticks=args.ticks,
        tick_seconds=args.tick_seconds,
        batch_limit=args.batch_limit,
        rng_seed=args.seed,
    )

    ts = datetime.now(tz=UTC)
    print(
        f"count={summary.count} ticks={summary.ticks} fires={summary.total_fires} "
        f"p50={summary.p50_ms:.2f}ms p95={summary.p95_ms:.2f}ms p99={summary.p99_ms:.2f}ms "
        f"max={summary.max_ms:.2f}ms verdict={'PASS' if summary.p99_under_budget else 'FAIL'}"
    )

    if not args.no_report:
        output = args.output or _default_report_path(ts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            render_report(
                summary,
                rng_seed=args.seed,
                batch_limit=args.batch_limit,
                generated_at=ts,
            )
        )
        print(f"wrote {output}")

    return 0 if summary.p99_under_budget else 1


if __name__ == "__main__":
    sys.exit(main())
