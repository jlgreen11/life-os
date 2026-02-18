"""
Tests for BehavioralAccuracyTracker._ensure_resolution_reason_column.

Background
----------
PR #197 added the ``resolution_reason`` column to the predictions table
via a DatabaseManager migration (schema v3 → v4).  However, when the
server is still running with an older DatabaseManager instance that was
initialized before the migration code was merged, the column is absent
from the live database.  Every ``run_inference_cycle`` call then raises::

    sqlite3.OperationalError: no such column: resolution_reason

causing the entire behavioral-accuracy learning loop to silently break.

Fix
---
``BehavioralAccuracyTracker.__init__`` now calls
``_ensure_resolution_reason_column()``, which:

1. Checks whether the column already exists (idempotent, fast path).
2. If missing, applies ``ALTER TABLE predictions ADD COLUMN resolution_reason TEXT``.
3. Updates ``schema_version`` to 4 so the DatabaseManager won't re-run
   the same migration on next restart.
4. Retroactively tags existing inaccurate automated-sender predictions as
   ``'automated_sender_fast_path'`` so ``_get_accuracy_multiplier`` can
   immediately exclude them from the accuracy denominator.

Tests
-----
1. Fresh DB (schema v4): column already present — no-op, no error.
2. Simulated pre-migration DB (column dropped): tracker adds column.
3. Schema_version updated to 4 after inline migration.
4. Existing automated-sender inaccurate predictions retroactively tagged.
5. Existing non-automated-sender inaccurate predictions NOT tagged.
6. Accurate predictions NOT tagged (only inaccurate get fast-path label).
7. run_inference_cycle works end-to-end after inline migration.
8. Calling __init__ twice (re-creation) is idempotent.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from services.behavioral_accuracy_tracker.tracker import BehavioralAccuracyTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drop_resolution_reason_column(db) -> None:
    """Simulate a pre-migration database by recreating predictions without the column.

    SQLite does not support DROP COLUMN on older versions, so we
    recreate the table without the column.

    Args:
        db: DatabaseManager fixture pointing at a temp database.
    """
    with db.get_connection("user_model") as conn:
        conn.execute("ALTER TABLE predictions RENAME TO predictions_old")
        conn.execute("""
            CREATE TABLE predictions (
                id TEXT PRIMARY KEY,
                prediction_type TEXT NOT NULL,
                description TEXT,
                confidence REAL,
                confidence_gate TEXT,
                time_horizon TEXT,
                suggested_action TEXT,
                supporting_signals TEXT,
                was_surfaced INTEGER DEFAULT 0,
                user_response TEXT,
                was_accurate INTEGER,
                created_at TEXT,
                resolved_at TEXT,
                filter_reason TEXT
                -- resolution_reason intentionally omitted to simulate pre-v4 schema
            )
        """)
        conn.execute("INSERT INTO predictions SELECT "
                     "id, prediction_type, description, confidence, confidence_gate, "
                     "time_horizon, suggested_action, supporting_signals, was_surfaced, "
                     "user_response, was_accurate, created_at, resolved_at, filter_reason "
                     "FROM predictions_old")
        conn.execute("DROP TABLE predictions_old")
        # Downgrade schema_version to 3 so the re-migration path is exercised
        conn.execute("DELETE FROM schema_version WHERE version >= 4")


def _column_names(db) -> list[str]:
    """Return column names of the predictions table.

    Args:
        db: DatabaseManager fixture.

    Returns:
        List of column name strings.
    """
    with db.get_connection("user_model") as conn:
        return [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]


def _insert_prediction(db, *, prediction_type="opportunity", was_accurate=None,
                       was_surfaced=True, resolved_at=None, contact_email=None,
                       resolution_reason=None) -> str:
    """Insert a test prediction row and return its ID.

    Args:
        db: DatabaseManager fixture.
        prediction_type: Type string (e.g. 'opportunity', 'reminder').
        was_accurate: True/False/None.
        was_surfaced: Whether the prediction was surfaced to the user.
        resolved_at: ISO timestamp or None for unresolved.
        contact_email: Stored in supporting_signals to drive fast-path detection.
        resolution_reason: Pre-existing reason or None.

    Returns:
        UUID string of the inserted prediction.
    """
    pred_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    signals = json.dumps({"contact_email": contact_email} if contact_email else {})

    with db.get_connection("user_model") as conn:
        columns_present = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        if "resolution_reason" in columns_present:
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    supporting_signals, was_surfaced, was_accurate, resolution_reason,
                    resolved_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id, prediction_type, f"Test {prediction_type}", 0.5, "suggest",
                    signals, 1 if was_surfaced else 0,
                    (1 if was_accurate else 0) if was_accurate is not None else None,
                    resolution_reason,
                    resolved_at or (now if was_accurate is not None else None),
                    now,
                ),
            )
        else:
            # Pre-migration schema: no resolution_reason column
            conn.execute(
                """INSERT INTO predictions
                   (id, prediction_type, description, confidence, confidence_gate,
                    supporting_signals, was_surfaced, was_accurate,
                    resolved_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pred_id, prediction_type, f"Test {prediction_type}", 0.5, "suggest",
                    signals, 1 if was_surfaced else 0,
                    (1 if was_accurate else 0) if was_accurate is not None else None,
                    resolved_at or (now if was_accurate is not None else None),
                    now,
                ),
            )
    return pred_id


# ---------------------------------------------------------------------------
# Test 1: Fresh DB — column already present, no-op
# ---------------------------------------------------------------------------

def test_fresh_db_column_already_present(db):
    """With a fresh DB (schema v4), _ensure_resolution_reason_column is a no-op.

    The tracker should initialize without error and the column should
    remain present.
    """
    # Pre-condition: column exists in fresh DB
    assert "resolution_reason" in _column_names(db)

    # Act: create tracker (calls _ensure_resolution_reason_column internally)
    tracker = BehavioralAccuracyTracker(db)

    # Column should still be present
    assert "resolution_reason" in _column_names(db)


# ---------------------------------------------------------------------------
# Test 2: Pre-migration DB — tracker adds the column
# ---------------------------------------------------------------------------

def test_inline_migration_adds_column(db):
    """When resolution_reason is missing, __init__ adds the column.

    Simulates a server that was not restarted after PR #197 was merged.
    """
    # Simulate pre-v4 schema by dropping the column
    _drop_resolution_reason_column(db)
    assert "resolution_reason" not in _column_names(db)

    # Act: instantiate tracker — should apply migration
    tracker = BehavioralAccuracyTracker(db)

    # Column must now exist
    assert "resolution_reason" in _column_names(db)


# ---------------------------------------------------------------------------
# Test 3: Schema_version updated to 4 after inline migration
# ---------------------------------------------------------------------------

def test_inline_migration_updates_schema_version(db):
    """After inline migration, schema_version table records version 4.

    This prevents the DatabaseManager from re-running the same migration
    on next restart and emitting confusing "column already exists" warnings.
    """
    _drop_resolution_reason_column(db)

    # After dropping the column, schema_version is < 4 (the helper deletes
    # version >= 4 rows, so max is either 3 or NULL if the fresh DB had
    # already recorded version 4 and nothing earlier).
    with db.get_connection("user_model") as conn:
        ver_before = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
    assert ver_before < 4

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver >= 4


# ---------------------------------------------------------------------------
# Test 4: Retroactive tagging of automated-sender inaccurate predictions
# ---------------------------------------------------------------------------

def test_retroactive_tagging_automated_sender(db):
    """Inaccurate automated-sender predictions get resolution_reason tagged.

    After the inline migration, existing inaccurate predictions whose
    contact_email is an automated sender should be tagged as
    'automated_sender_fast_path' so _get_accuracy_multiplier can
    exclude them from the accuracy denominator immediately.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Simulate pre-migration DB with existing predictions
    _drop_resolution_reason_column(db)

    # Insert an inaccurate prediction for an automated sender (pre-migration)
    pred_id = _insert_prediction(
        db, prediction_type="opportunity", was_accurate=False,
        was_surfaced=True, contact_email="noreply@company.com",
        resolved_at=now,
    )

    # Apply migration via tracker init
    BehavioralAccuracyTracker(db)

    # The prediction should now be tagged
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row["resolution_reason"] == "automated_sender_fast_path"


# ---------------------------------------------------------------------------
# Test 5: Non-automated-sender predictions NOT tagged
# ---------------------------------------------------------------------------

def test_retroactive_tagging_skips_real_contacts(db):
    """Inaccurate predictions for real contacts do NOT get a fast-path tag.

    Only automated senders should be labelled — real human contacts who
    the user chose not to message should remain untagged so their
    inaccuracy counts against the accuracy multiplier.
    """
    now = datetime.now(timezone.utc).isoformat()
    _drop_resolution_reason_column(db)

    pred_id = _insert_prediction(
        db, prediction_type="opportunity", was_accurate=False,
        was_surfaced=True, contact_email="alice@gmail.com",
        resolved_at=now,
    )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row["resolution_reason"] is None


# ---------------------------------------------------------------------------
# Test 6: Accurate predictions are NOT tagged
# ---------------------------------------------------------------------------

def test_retroactive_tagging_skips_accurate_predictions(db):
    """Accurate predictions for automated-sender contacts are not tagged.

    An accurate prediction means the user DID reach out — that is always
    a real behavioral signal, regardless of the contact type.  Tagging
    accurate predictions would incorrectly remove them from the accurate
    count, reducing the measured accuracy rate.
    """
    now = datetime.now(timezone.utc).isoformat()
    _drop_resolution_reason_column(db)

    pred_id = _insert_prediction(
        db, prediction_type="opportunity", was_accurate=True,
        was_surfaced=True, contact_email="noreply@company.com",
        resolved_at=now,
    )

    BehavioralAccuracyTracker(db)

    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT resolution_reason FROM predictions WHERE id = ?", (pred_id,)
        ).fetchone()
    assert row["resolution_reason"] is None


# ---------------------------------------------------------------------------
# Test 7: run_inference_cycle works end-to-end after inline migration
# ---------------------------------------------------------------------------

import asyncio

def test_run_inference_cycle_after_inline_migration(db):
    """run_inference_cycle completes without error after the inline migration.

    This is the critical regression test: before the fix, initializing
    BehavioralAccuracyTracker on a pre-v4 database and then calling
    run_inference_cycle would raise::

        sqlite3.OperationalError: no such column: resolution_reason

    After the fix, the migration is applied during __init__ so the cycle
    completes normally.
    """
    _drop_resolution_reason_column(db)
    tracker = BehavioralAccuracyTracker(db)

    # Insert an unresolved surfaced prediction (no supporting_signals → no match)
    _insert_prediction(db, prediction_type="opportunity", was_accurate=None, was_surfaced=True)

    # Must not raise
    stats = asyncio.get_event_loop().run_until_complete(tracker.run_inference_cycle())
    assert isinstance(stats, dict)
    assert "marked_accurate" in stats
    assert "marked_inaccurate" in stats


# ---------------------------------------------------------------------------
# Test 8: Double initialization is idempotent
# ---------------------------------------------------------------------------

def test_double_initialization_idempotent(db):
    """Creating two BehavioralAccuracyTracker instances on the same DB is safe.

    The second __init__ must not fail even though the column already
    exists after the first initialization applied the migration.
    """
    _drop_resolution_reason_column(db)

    # First init: applies migration
    BehavioralAccuracyTracker(db)
    assert "resolution_reason" in _column_names(db)

    # Second init: should be a no-op without error
    BehavioralAccuracyTracker(db)
    assert "resolution_reason" in _column_names(db)
