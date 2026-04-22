"""Shared fixtures for the golden-dataset regression suite.

The suite needs a *real* v1 snapshot to exercise the v2 pipeline. Production
data is never committed; the snapshot is expected to live at
``data/v1-snapshot/`` on the operator's machine. This conftest centralises
snapshot discovery so the test body can stay focused on assertions.

Snapshot layout (all paths read-only at runtime):

- ``data/v1-snapshot/events.db``
- ``data/v1-snapshot/entities.db``
- ``data/v1-snapshot/state.db``
- ``data/v1-snapshot/user_model.db``
- ``data/v1-snapshot/preferences.db``

The path can be overridden with the ``LIFEOS_V1_SNAPSHOT_DIR`` env var so the
Mac Mini operator can point at e.g. ``./data/backup-20260420/`` without moving
files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "v1-snapshot"

REQUIRED_DBS: tuple[str, ...] = (
    "events.db",
    "user_model.db",
)
"""Minimum set the harness can run with. Other DBs (entities, state,
preferences) are optional — the migration script tolerates them being absent
and the regression assertions only need events + predictions."""


def _resolve_snapshot_dir() -> Path:
    override = os.environ.get("LIFEOS_V1_SNAPSHOT_DIR")
    if override:
        return Path(override).resolve()
    return DEFAULT_SNAPSHOT_DIR


@pytest.fixture(scope="session")
def v1_snapshot_dir() -> Path:
    """Return the snapshot directory or skip cleanly if it's incomplete.

    The skip message names the missing files so an operator dropping a
    snapshot in place can immediately see what the harness wanted.
    """
    snapshot_dir = _resolve_snapshot_dir()
    if not snapshot_dir.exists():
        pytest.skip(
            f"v1 snapshot dir not present at {snapshot_dir}; "
            "create one (or set LIFEOS_V1_SNAPSHOT_DIR) before running the "
            "golden-dataset regression. See tests/regression/__init__.py for "
            "the expected layout."
        )
    missing = [name for name in REQUIRED_DBS if not (snapshot_dir / name).exists()]
    if missing:
        pytest.skip(
            f"v1 snapshot at {snapshot_dir} is missing required DBs: "
            f"{missing}. The harness needs at minimum {list(REQUIRED_DBS)}."
        )
    return snapshot_dir
