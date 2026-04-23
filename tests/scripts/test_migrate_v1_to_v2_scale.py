"""Scale test for ``scripts/migrate_v1_to_v2.py``.

Builds the large fixture defined in :mod:`tests.fixtures.v1_sample`
(10K events, 500 entities, 200 signal_profile rows) and runs the dry-run
migrator against it. Asserts three budgets per NEXT_TASKS:

- (a) peak RSS delta during migration < 512 MB
- (b) wall-clock migration time < 120 s
- (c) every per-table row-count invariant holds (including the dropped-profile
      invariant — no ``mood``/``decision``/``expertise``/``values`` / unknown
      legacy types leak into the v2 ``signal_profiles`` table)

Memory is measured with the stdlib :mod:`resource` module. The NEXT_TASKS
spec allowed a ``psutil`` path with a skip-on-unavailable fallback; we use
:func:`resource.getrusage` instead because it is always available on macOS
and Linux (psutil is not installed in this environment). The delta between
pre- and post-migration peak RSS captures the incremental spike attributable
to the migration run, which is what the 512 MB budget is targeting.
"""

from __future__ import annotations

import resource
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from scripts import migrate_v1_to_v2 as migrate
from tests.fixtures.v1_sample import build_scaled_v1_sample

MEMORY_BUDGET_MB: float = 512.0
TIME_BUDGET_SECONDS: float = 120.0


def _peak_rss_mb() -> float:
    """Return current process peak RSS in megabytes.

    ``ru_maxrss`` is reported in bytes on Darwin and kilobytes on Linux —
    this shim normalizes both to MB so the budget comparison is portable.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


@pytest.fixture(scope="module")
def migration_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build the scaled fixture, run the migration once, capture runtime + memory.

    Module-scoped so all four assertions share a single migration execution;
    each test then probes a different invariant.
    """
    src = tmp_path_factory.mktemp("v1_scaled_src")
    out_dir = tmp_path_factory.mktemp("v2_scaled_out")
    out = out_dir / "dryrun.db"

    counts = build_scaled_v1_sample(src)

    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    report = migrate.run_migration(src, out)
    elapsed = time.perf_counter() - t0
    rss_after = _peak_rss_mb()

    return {
        "src": src,
        "out": out,
        "counts": counts,
        "report": report,
        "elapsed_seconds": elapsed,
        "memory_delta_mb": rss_after - rss_before,
    }


def test_scale_migration_completes_under_time_budget(
    migration_run: dict[str, object],
) -> None:
    elapsed = float(migration_run["elapsed_seconds"])  # type: ignore[arg-type]
    assert elapsed < TIME_BUDGET_SECONDS, f"migration took {elapsed:.2f}s (budget {TIME_BUDGET_SECONDS}s)"


def test_scale_migration_peak_memory_under_budget(
    migration_run: dict[str, object],
) -> None:
    delta = float(migration_run["memory_delta_mb"])  # type: ignore[arg-type]
    assert delta < MEMORY_BUDGET_MB, f"migration added {delta:.1f}MB peak RSS (budget {MEMORY_BUDGET_MB}MB)"


def test_scale_row_count_invariants_hold(
    migration_run: dict[str, object],
) -> None:
    counts = migration_run["counts"]  # type: ignore[assignment]
    report = migration_run["report"]  # type: ignore[assignment]

    assert report.events.source == counts.events
    assert report.events.translated == counts.events
    assert report.events.dropped == 0

    assert report.entities.source == counts.total_entities
    assert report.entities.translated == counts.total_entities
    assert report.entities.dropped == 0

    assert report.moments_from_tasks.source == counts.tasks
    assert report.moments_from_tasks.translated == counts.tasks
    assert report.moments_from_tasks.dropped == 0

    assert report.signal_profiles.source == counts.signal_profiles
    assert report.signal_profiles.translated == counts.kept_signal_profiles
    assert report.signal_profiles.dropped == counts.dropped_signal_profiles

    assert report.preferences.source == counts.preferences
    assert report.preferences.translated == counts.preferences

    assert report.notification_feedback.source == counts.feedback_log
    assert report.notification_feedback.translated == counts.feedback_log
    assert report.notification_feedback.dropped == 0

    assert not [n for n in report.notes if n.startswith("INVARIANT:")]


def test_scale_dropped_profile_types_never_leak_into_output(
    migration_run: dict[str, object],
) -> None:
    out = Path(migration_run["out"])  # type: ignore[arg-type]
    with sqlite3.connect(out) as conn:
        producers = {row[0] for row in conn.execute("SELECT DISTINCT producer FROM signal_profiles").fetchall()}
    for dropped in ("mood", "decision", "expertise", "values"):
        assert dropped not in producers, f"dropped profile type leaked: {dropped}"
    assert not any(p.startswith("unknown-legacy-") for p in producers), (
        "unknown-legacy profile types leaked into the v2 output"
    )
