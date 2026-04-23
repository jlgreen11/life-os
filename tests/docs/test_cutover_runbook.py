"""Structural tests for ``docs/cutover-runbook.md``.

The runbook is a live operator document consumed during the highest-risk step
of the Phase 1 cutover (per CEO plan § "Data migration"). These tests lock
the structural invariants a human operator would rely on:

- All required sections from the NEXT_TASKS spec exist and are ordered:
  pre-flight → stop v1 → run migration → bring up v2 → verify → watch → rollback.
- The ≤ 30-min RTO target is named.
- Each executable command block is fenced bash (so line-paste from a Mac
  terminal is unambiguous).
- Every script referenced by the runbook is either present in the repo today
  or is a Category-C NEXT_TASKS line item (so a reader can see at a glance
  which parts of the runbook depend on not-yet-shipped tooling).

Pure-stdlib on purpose: the runbook is markdown, these checks are string +
path operations, and this module imports nothing from the project so it runs
under the repo's minimum Python (3.12+) without needing FastAPI or any other
runtime dep installed.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNBOOK = REPO_ROOT / "docs" / "cutover-runbook.md"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_exists() -> None:
    assert RUNBOOK.is_file(), f"cutover runbook missing at {RUNBOOK}"


def test_rto_target_named() -> None:
    """CEO plan locks the 30-min RTO; the runbook must say so explicitly."""
    body = _text()
    assert "RTO" in body, "runbook should name the RTO concept"
    # Tolerate '≤ 30 min', '<= 30 minutes', '30 min', etc.
    assert re.search(r"30\s*min", body), "runbook should name the 30-minute RTO target"


def test_required_sections_present_and_ordered() -> None:
    """The seven runbook phases must appear in the order the operator runs them."""
    body = _text()
    required_in_order = [
        "Pre-flight",
        "Stop v1",
        "Run migration",
        "Bring up v2",
        "Verify",
        "24-hour watch window",
        "Rollback",
    ]
    positions: list[int] = []
    for label in required_in_order:
        idx = body.find(label)
        assert idx != -1, f"runbook missing required section: {label!r}"
        positions.append(idx)
    assert positions == sorted(positions), (
        f"runbook sections out of order: {list(zip(required_in_order, positions, strict=True))}"
    )


def test_verify_section_covers_health_now_and_context_shim() -> None:
    """§ 5 (Verify) must exercise /api/health, a sample Moment, and the iOS compat shim."""
    body = _text()
    assert "/api/health" in body, "runbook must reference /api/health"
    assert "/api/now" in body, "runbook must reference /api/now (sample Moment verification)"
    assert "/api/context/event" in body, "runbook must reference the iOS compat shim (/api/context/event)"


def test_watch_window_names_alert_thresholds() -> None:
    """§ 6 must give the operator concrete thresholds, not vibes."""
    body = _text()
    # Spec calls out each of these explicitly.
    assert "connector offline" in body.lower(), "watch-window must cover connector offline"
    assert "db_last_write_ts" in body or "DB last-write" in body or "db last-write" in body.lower(), (
        "watch-window must cover DB last-write lag"
    )
    assert "scheduler heartbeat" in body.lower(), "watch-window must cover scheduler heartbeat"
    assert "pending" in body.lower() and "moment" in body.lower(), "watch-window must cover pending-Moment backlog"


def test_rollback_has_trigger_criteria_and_procedure() -> None:
    body = _text()
    # Section headings for the two sub-pieces the spec demands.
    assert re.search(r"Trigger criteria", body, flags=re.IGNORECASE), "rollback section must list trigger criteria"
    assert re.search(r"Rollback procedure", body, flags=re.IGNORECASE), "rollback section must describe a procedure"
    # The procedure must at minimum invoke the scripted path OR a manual path
    # that restores from backup-${TS}.
    assert "backup-${TS}" in body or "cutover_rollback.py" in body, (
        "rollback procedure must reference the timestamped backup or the rollback script"
    )


def test_migration_command_has_explicit_flags() -> None:
    """§ 3 must specify the exact migrate_v1_to_v2.py invocation."""
    body = _text()
    assert "migrate_v1_to_v2.py" in body, "runbook must invoke the migration script"
    assert "--source-dir" in body, "runbook must pass --source-dir"
    assert "--output" in body, "runbook must pass --output"


def test_bring_up_v2_uses_python_m_life_os() -> None:
    """Startup command is canon (CLAUDE.md + README). Lock it."""
    body = _text()
    assert "python -m life_os" in body, "runbook must bring up v2 via the canonical `python -m life_os` entry point"


def test_every_bash_fence_parses_cleanly() -> None:
    """Defensive: every ``` bash fence is closed, so copy-paste is unambiguous."""
    body = _text()
    fences = re.findall(r"^```", body, flags=re.MULTILINE)
    assert len(fences) % 2 == 0, (
        f"unbalanced triple-backtick fences in runbook (found {len(fences)}); "
        "a half-open fence will mangle operator copy-paste"
    )


def test_referenced_scripts_exist_or_are_queued() -> None:
    """Any `scripts/*.py` named in the runbook must be present OR queued in NEXT_TASKS.

    The Category-C tasks (cutover_monitor.py, cutover_rollback.py, v1_v2_diff.py)
    are legitimately referenced forward; present-today scripts must actually be present.
    """
    body = _text()
    scripts_referenced = set(re.findall(r"scripts/([A-Za-z0-9_]+\.py)", body))
    assert scripts_referenced, "runbook references at least one script; regex should find them"

    next_tasks_body = (REPO_ROOT / "NEXT_TASKS.md").read_text(encoding="utf-8")
    for script_name in scripts_referenced:
        on_disk = (REPO_ROOT / "scripts" / script_name).is_file()
        queued = script_name in next_tasks_body
        assert on_disk or queued, (
            f"runbook references scripts/{script_name} which is neither present on disk nor queued in NEXT_TASKS.md"
        )


def test_timing_estimates_add_up_under_rto() -> None:
    """The phase-by-phase wall-clock estimates must sum to the RTO target (30 min)."""
    body = _text()
    # The § 0 table lists individual phase wall-clock in 'N min' format. We
    # scan the whole table area once.
    table_region = body.split("## 1", 1)[0]
    phase_times = [int(m) for m in re.findall(r"\|\s*(\d+)\s*min\s*\|", table_region)]
    # Two numeric columns per row (wall-clock + cumulative); the cumulative
    # column is the check we care about. Guard we actually saw rows:
    assert phase_times, "could not parse phase timing table in § 0"
    # Sum of the *wall-clock* column (every other value, starting at 0) must
    # be ≤ 30 so the cutover fits inside the RTO budget.
    wall_clocks = phase_times[::2]
    assert sum(wall_clocks) <= 30, (
        f"sum of per-phase wall-clock estimates ({sum(wall_clocks)} min) exceeds the 30-min RTO target: {wall_clocks}"
    )
