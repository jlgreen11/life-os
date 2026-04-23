"""Tests for :mod:`scripts.profile_scheduler`.

Focused on the pure helpers (seed planning, percentile math, summary
folding, report rendering) plus a tiny end-to-end smoke test that runs a
200-row fleet through a handful of simulated ticks so the full pipeline
stays wired together without paying the 10K-row cost in CI.

Stdlib only — matches the project's "no freezegun, no heavy deps" rule.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

from scripts import profile_scheduler as prof
from storage import schema


# ---------------------------------------------------------------------------
# plan_seeds
# ---------------------------------------------------------------------------


def test_plan_seeds_count_and_determinism():
    a = prof.plan_seeds(500, rng_seed=7)
    b = prof.plan_seeds(500, rng_seed=7)
    assert len(a) == 500
    assert a == b  # deterministic on seed


def test_plan_seeds_bucket_ratios():
    seeds = prof.plan_seeds(1000, rng_seed=7)
    scheduled = sum(1 for s in seeds if s.state == "suggested" and s.scheduled_for is not None)
    snoozed = sum(1 for s in seeds if s.state == "snoozed")
    null_scheduled = sum(1 for s in seeds if s.scheduled_for is None)
    assert scheduled == 700
    assert snoozed == 200
    assert null_scheduled == 100


def test_plan_seeds_all_within_horizon():
    now = 1_000_000
    seeds = prof.plan_seeds(200, now_ts=now, horizon_seconds=3600, rng_seed=1)
    for s in seeds:
        if s.scheduled_for is not None:
            assert now <= s.scheduled_for <= now + 3600
        assert s.expires_at == now + 72 * 3600


def test_plan_seeds_snoozed_rows_have_snooze_until_and_scheduled_for():
    seeds = prof.plan_seeds(50, rng_seed=9)
    for s in seeds:
        if s.state == "snoozed":
            assert s.snooze_until is not None
            assert s.scheduled_for == s.snooze_until
        else:
            assert s.snooze_until is None


def test_plan_seeds_zero_is_empty_list():
    assert prof.plan_seeds(0) == []


def test_plan_seeds_negative_count_raises():
    with pytest.raises(ValueError):
        prof.plan_seeds(-1)


def test_plan_seeds_evidence_hashes_are_unique():
    seeds = prof.plan_seeds(250, rng_seed=3)
    assert len({s.evidence_hash for s in seeds}) == 250


# ---------------------------------------------------------------------------
# insert_seeds
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys=ON")
    for stmt in schema.get_all_ddl():
        c.execute(stmt)
    c.commit()
    yield c
    c.close()


def test_insert_seeds_writes_moments_and_history(conn):
    seeds = prof.plan_seeds(30, rng_seed=5)
    prof.insert_seeds(conn, seeds)
    (m_count,) = conn.execute("SELECT COUNT(*) FROM moments").fetchone()
    (h_count,) = conn.execute("SELECT COUNT(*) FROM moment_state_history").fetchone()
    assert m_count == 30
    assert h_count == 30


def test_insert_seeds_preserves_state_distribution(conn):
    seeds = prof.plan_seeds(100, rng_seed=11)
    prof.insert_seeds(conn, seeds)
    (snoozed,) = conn.execute("SELECT COUNT(*) FROM moments WHERE state='snoozed'").fetchone()
    (suggested,) = conn.execute("SELECT COUNT(*) FROM moments WHERE state='suggested'").fetchone()
    assert snoozed == 20
    assert suggested == 80


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_known_values():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert prof.percentile(samples, 50) == pytest.approx(5.5)
    assert prof.percentile(samples, 0) == pytest.approx(1.0)
    assert prof.percentile(samples, 100) == pytest.approx(10.0)


def test_percentile_unsorted_input_is_sorted_internally():
    samples = [10.0, 3.0, 7.0, 1.0, 5.0]
    assert prof.percentile(samples, 50) == pytest.approx(5.0)


def test_percentile_empty_returns_zero():
    assert prof.percentile([], 95) == 0.0


def test_percentile_single_value():
    assert prof.percentile([42.0], 99) == pytest.approx(42.0)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        prof.percentile([1.0], -1)
    with pytest.raises(ValueError):
        prof.percentile([1.0], 101)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_basic():
    # 10 ticks with ascending latencies; fires alternate 0/1.
    latencies = [0.010 * i for i in range(1, 11)]  # 10ms..100ms
    fires = [i % 2 for i in range(10)]
    summary = prof.summarize(latencies, fires, count=100, tick_seconds=30)
    assert summary.count == 100
    assert summary.ticks == 10
    assert summary.tick_seconds == 30
    assert summary.simulated_seconds == 300
    assert summary.total_fires == 5
    assert summary.fires_per_tick_mean == pytest.approx(0.5)
    assert summary.fires_per_simulated_second == pytest.approx(5 / 300)
    assert summary.p50_ms == pytest.approx(55.0)
    assert summary.max_ms == pytest.approx(100.0)
    assert summary.p99_under_budget is True


def test_summarize_flags_budget_miss():
    latencies = [0.6] * 20  # 600 ms everywhere
    summary = prof.summarize(latencies, [0] * 20, count=10, tick_seconds=30)
    assert summary.p99_ms == pytest.approx(600.0)
    assert summary.p99_under_budget is False


def test_summarize_zero_ticks_is_safe():
    summary = prof.summarize([], [], count=0, tick_seconds=30)
    assert summary.ticks == 0
    assert summary.simulated_seconds == 0
    assert summary.total_fires == 0
    assert summary.fires_per_tick_mean == 0.0
    assert summary.fires_per_simulated_second == 0.0
    assert summary.p50_ms == 0.0
    assert summary.p99_under_budget is True


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def _pass_summary() -> prof.ProfileSummary:
    return prof.ProfileSummary(
        count=10000,
        ticks=120,
        tick_seconds=30,
        simulated_seconds=3600,
        total_fires=420,
        fires_per_tick_mean=3.5,
        fires_per_simulated_second=420 / 3600,
        p50_ms=4.2,
        p95_ms=18.0,
        p99_ms=45.0,
        max_ms=80.0,
        p99_under_budget=True,
    )


def test_render_report_contains_verdict_pass():
    body = prof.render_report(_pass_summary())
    assert "# Scheduler performance profile" in body
    assert "10,000" in body
    assert "PASS" in body
    assert "FAIL" not in body
    assert "p99 tick latency" in body


def test_render_report_contains_verdict_fail():
    failing = prof.ProfileSummary(
        count=10000,
        ticks=120,
        tick_seconds=30,
        simulated_seconds=3600,
        total_fires=420,
        fires_per_tick_mean=3.5,
        fires_per_simulated_second=420 / 3600,
        p50_ms=100.0,
        p95_ms=400.0,
        p99_ms=600.0,
        max_ms=900.0,
        p99_under_budget=False,
    )
    body = prof.render_report(failing)
    assert "FAIL" in body
    assert "600.00 ms" in body


def test_render_report_echoes_reproduce_command():
    body = prof.render_report(_pass_summary())
    assert "scripts/profile_scheduler.py --count 10000" in body


# ---------------------------------------------------------------------------
# run_profile end-to-end smoke test
# ---------------------------------------------------------------------------


def test_run_profile_small_fleet_smoke(tmp_path: Path):
    summary = prof.run_profile(
        count=200,
        ticks=6,
        tick_seconds=60,
        batch_limit=100,
        rng_seed=123,
        db_path=tmp_path / "smoke.db",
    )
    assert summary.count == 200
    assert summary.ticks == 6
    assert summary.simulated_seconds == 360
    assert summary.total_fires >= 0
    assert summary.p50_ms >= 0.0
    # Smoke-level: real perf_counter — just assert we didn't explode past
    # an outrageous budget (5s would already be a bug at 200 rows).
    assert summary.max_ms < 5000.0


def test_run_profile_deterministic_fire_count(tmp_path: Path):
    s1 = prof.run_profile(
        count=200,
        ticks=4,
        tick_seconds=3600,
        batch_limit=500,
        rng_seed=99,
        db_path=tmp_path / "a.db",
    )
    s2 = prof.run_profile(
        count=200,
        ticks=4,
        tick_seconds=3600,
        batch_limit=500,
        rng_seed=99,
        db_path=tmp_path / "b.db",
    )
    # Wall-clock advance + plan_seeds are both deterministic on seed,
    # so total fires should match run-to-run.
    assert s1.total_fires == s2.total_fires


def test_main_writes_report(tmp_path: Path, monkeypatch, capsys):
    output = tmp_path / "perf-report.md"
    rc = prof.main(
        [
            "--count",
            "50",
            "--ticks",
            "3",
            "--tick-seconds",
            "600",
            "--batch-limit",
            "20",
            "--seed",
            "1",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert rc in (0, 1)  # depends on timing; we don't assert on it here
    assert output.exists()
    body = output.read_text()
    assert "Scheduler performance profile" in body
    assert "count=50" in captured.out
