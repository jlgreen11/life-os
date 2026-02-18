"""
Tests for the data quality analysis script's accuracy calculation.

Verifies that ``scripts/analyze-data-quality.py`` correctly excludes
``automated_sender_fast_path`` resolutions from the accuracy denominator,
matching the same exclusion logic used by
``PredictionEngine._get_accuracy_multiplier()``.

Background
----------
PR #197 introduced ``resolution_reason`` to mark predictions that were
resolved as INACCURATE not because the prediction was wrong, but because
the contact is an automated/marketing sender (e.g. noreply@company.com).
These predictions are structurally unfulfillable — the user will never
"reach out" to a no-reply mailer by definition.

Before this fix the data quality script counted them in the accuracy
denominator, reporting misleadingly low accuracy:

    Before:  opportunity 41/215 = 19.1%  (174 inaccurate in denominator)
    After:   opportunity 41/115 = 35.7%  (74 real misses; 100 excluded)

The prediction engine already excluded these via
``_get_accuracy_multiplier()``; the data quality script now uses the
same exclusion so the reported number matches what the engine acts on.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Import the analyze function directly so we can pass a controlled data dir.
# We need to add the project root to sys.path so the module is importable.
import importlib.util
import sys
import os


def _get_analyze():
    """Import the analyze function from ``scripts/analyze-data-quality.py``.

    The script filename contains a hyphen, so it cannot be imported with a
    standard ``import`` statement.  We use ``importlib`` to load it by file
    path instead.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "analyze-data-quality.py"
    spec = importlib.util.spec_from_file_location("analyze_data_quality", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction_db(tmp_path: Path, predictions: list[dict]) -> str:
    """Create a minimal user_model.db in tmp_path with the given predictions.

    The predictions list should contain dicts with the fields used by
    ``analyze-data-quality.py``::

        {
            "prediction_type": "opportunity",
            "was_surfaced": 1,          # 1 = shown to user, 0 = filtered
            "was_accurate": None|1|0,   # None = unresolved
            "resolution_reason": None|"automated_sender_fast_path",
        }

    Returns the path to the temporary data directory (not the db file).
    """
    db_path = tmp_path / "user_model.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE predictions (
            id TEXT PRIMARY KEY,
            prediction_type TEXT,
            was_surfaced INTEGER,
            was_accurate INTEGER,
            resolution_reason TEXT,
            created_at TEXT,
            resolved_at TEXT
        )
    """)
    # signal_profiles and insights tables are also queried; create stubs.
    conn.execute("""
        CREATE TABLE signal_profiles (
            profile_type TEXT PRIMARY KEY,
            samples_count INTEGER,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE insights (
            id TEXT PRIMARY KEY,
            type TEXT,
            feedback TEXT
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    for i, pred in enumerate(predictions):
        conn.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(i),
                pred["prediction_type"],
                pred.get("was_surfaced", 1),
                pred.get("was_accurate"),  # May be None
                pred.get("resolution_reason"),
                now,
                now if pred.get("was_accurate") is not None else None,
            ),
        )
    conn.commit()
    conn.close()

    # Create stub databases for the other sections so analyze() doesn't error.
    for db_name in ("events.db", "state.db", "preferences.db"):
        stub_path = tmp_path / db_name
        stub_conn = sqlite3.connect(str(stub_path))
        if db_name == "events.db":
            stub_conn.execute("CREATE TABLE events (id TEXT, type TEXT, timestamp TEXT)")
        elif db_name == "state.db":
            stub_conn.execute("CREATE TABLE notifications (id TEXT, status TEXT)")
        elif db_name == "preferences.db":
            stub_conn.execute("""
                CREATE TABLE feedback_log (
                    id TEXT, action_type TEXT, feedback_type TEXT
                )
            """)
        stub_conn.commit()
        stub_conn.close()

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAccuracyExcludesAutomatedSenderFastPath:
    """The accuracy_rate must exclude automated_sender_fast_path resolutions."""

    def test_automated_sender_excluded_from_inaccurate_count(self, tmp_path):
        """automated_sender_fast_path predictions do not appear in inaccurate count."""
        analyze = _get_analyze()
        predictions = [
            # 3 accurate predictions
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": 1},
            # 2 real misses (no resolution_reason)
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": None},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": None},
            # 4 automated-sender fast-path exclusions (should NOT count as inaccurate)
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        # Only real misses counted as inaccurate
        assert stats["inaccurate"] == 2, (
            f"Expected 2 real inaccurate, got {stats['inaccurate']} "
            f"(automated-sender predictions must not count as inaccurate)"
        )

    def test_auto_excluded_field_present_and_correct(self, tmp_path):
        """auto_excluded field counts predictions with automated_sender_fast_path."""
        analyze = _get_analyze()
        predictions = [
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        assert "auto_excluded" in stats, "auto_excluded field must be present in output"
        assert stats["auto_excluded"] == 2

    def test_accuracy_rate_excludes_automated_senders_from_denominator(self, tmp_path):
        """accuracy_rate = accurate / (accurate + real_inaccurate), not / total_resolved."""
        analyze = _get_analyze()
        predictions = [
            # 5 accurate
            *[{"prediction_type": "opportunity", "was_accurate": 1}] * 5,
            # 5 real misses
            *[{"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": None}] * 5,
            # 10 automated-sender exclusions (should NOT count against accuracy)
            *[{"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"}] * 10,
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        # Without exclusion: 5 / 20 = 25%
        # With exclusion:    5 / 10 = 50%
        assert abs(stats["accuracy_rate"] - 0.5) < 0.01, (
            f"Expected accuracy_rate ~50% (5 accurate / 10 real resolved), "
            f"got {stats['accuracy_rate']:.1%} — automated senders must be excluded from denominator"
        )

    def test_zero_real_inaccurate_gives_perfect_accuracy(self, tmp_path):
        """If all inaccurate predictions are auto-excluded, accuracy_rate should be 1.0."""
        analyze = _get_analyze()
        predictions = [
            {"prediction_type": "reminder", "was_accurate": 1},
            {"prediction_type": "reminder", "was_accurate": 1},
            # All inaccurate ones are automated-sender fast-path
            {"prediction_type": "reminder", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
            {"prediction_type": "reminder", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["reminder"]

        assert stats["inaccurate"] == 0
        assert stats["accuracy_rate"] == 1.0, (
            f"Expected 100% accuracy when all inaccurate predictions are auto-excluded, "
            f"got {stats['accuracy_rate']:.1%}"
        )

    def test_no_automated_senders_unchanged_behavior(self, tmp_path):
        """When no automated-sender exclusions exist, accuracy is calculated normally."""
        analyze = _get_analyze()
        predictions = [
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": 0},
            {"prediction_type": "opportunity", "was_accurate": 0},
            {"prediction_type": "opportunity", "was_accurate": None},  # unresolved
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        assert stats["accurate"] == 2
        assert stats["inaccurate"] == 2
        assert stats["unresolved"] == 1
        assert stats["auto_excluded"] == 0
        # accuracy_rate = 2 / (2 + 2) = 50%
        assert abs(stats["accuracy_rate"] - 0.5) < 0.01

    def test_unresolved_not_counted_in_rate(self, tmp_path):
        """Unresolved predictions (was_accurate IS NULL) are excluded from the rate."""
        analyze = _get_analyze()
        predictions = [
            {"prediction_type": "opportunity", "was_accurate": 1},
            {"prediction_type": "opportunity", "was_accurate": None},  # unresolved
            {"prediction_type": "opportunity", "was_accurate": None},  # unresolved
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        # Only 1 accurate, 0 inaccurate → accuracy_rate = 1/1 = 100%
        assert stats["unresolved"] == 2
        assert abs(stats["accuracy_rate"] - 1.0) < 0.01

    def test_total_includes_all_predictions(self, tmp_path):
        """total count includes all predictions: accurate + inaccurate + auto_excluded + unresolved."""
        analyze = _get_analyze()
        predictions = [
            {"prediction_type": "opportunity", "was_accurate": 1},           # accurate
            {"prediction_type": "opportunity", "was_accurate": 0},            # real miss
            {"prediction_type": "opportunity", "was_accurate": 0, "resolution_reason": "automated_sender_fast_path"},  # excluded
            {"prediction_type": "opportunity", "was_accurate": None},         # unresolved
        ]
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        assert stats["total"] == 4, (
            f"Expected total=4 (all predictions regardless of resolution), got {stats['total']}"
        )

    def test_real_world_numbers_match_expected(self, tmp_path):
        """Simulates the actual production data scenario to verify the fix.

        Before fix: opportunity accuracy = 19.1% (41/215 where 174 inaccurate)
        After fix:  opportunity accuracy = 35.7% (41/115 where 74 real misses,
                                                   100 automated-sender excluded)
        """
        analyze = _get_analyze()
        predictions = (
            [{"prediction_type": "opportunity", "was_accurate": 1}] * 41        # accurate
            + [{"prediction_type": "opportunity", "was_accurate": 0}] * 74      # real misses
            + [{"prediction_type": "opportunity", "was_accurate": 0,            # excluded
                "resolution_reason": "automated_sender_fast_path"}] * 100
            + [{"prediction_type": "opportunity", "was_accurate": None}] * 33   # unresolved
        )
        data_dir = _make_prediction_db(tmp_path, predictions)
        report = analyze(data_dir)

        stats = report["sections"]["prediction_accuracy"]["opportunity"]

        assert stats["total"] == 248
        assert stats["accurate"] == 41
        assert stats["inaccurate"] == 74
        assert stats["auto_excluded"] == 100
        assert stats["unresolved"] == 33

        # accuracy_rate = 41 / (41 + 74) = 35.65%
        expected_rate = 41 / 115
        assert abs(stats["accuracy_rate"] - expected_rate) < 0.001, (
            f"Expected accuracy_rate ~{expected_rate:.1%}, got {stats['accuracy_rate']:.1%}"
        )
