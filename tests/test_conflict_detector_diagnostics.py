"""Tests for ConflictDetector.get_diagnostics() observability."""

import pytest

from services.conflict_detector import ConflictDetector


EXPECTED_KEYS = {
    "published_conflicts_count",
    "last_cleanup",
    "calendar_events_in_window",
    "conflict_pairs",
    "state_db_table_exists",
}


class TestGetDiagnostics:
    """Tests for ConflictDetector.get_diagnostics()."""

    def test_get_diagnostics_returns_expected_keys(self, db):
        """A fresh ConflictDetector should return a dict with all expected keys."""
        detector = ConflictDetector(db)
        diag = detector.get_diagnostics()

        assert isinstance(diag, dict)
        assert set(diag.keys()) == EXPECTED_KEYS

    def test_get_diagnostics_default_values(self, db):
        """A fresh detector with no conflicts should report sensible defaults."""
        detector = ConflictDetector(db)
        diag = detector.get_diagnostics()

        assert diag["published_conflicts_count"] == 0
        assert diag["last_cleanup"] is None
        assert diag["calendar_events_in_window"] == 0
        assert diag["conflict_pairs"] == []
        assert diag["state_db_table_exists"] is True

    def test_get_diagnostics_with_published_conflicts(self, db):
        """Published conflict pairs should appear in diagnostics output."""
        detector = ConflictDetector(db)

        # Simulate two published conflicts
        detector._published_conflicts.add(frozenset(["evt-a", "evt-b"]))
        detector._published_conflicts.add(frozenset(["evt-c", "evt-d"]))

        diag = detector.get_diagnostics()

        assert diag["published_conflicts_count"] == 2
        assert len(diag["conflict_pairs"]) == 2
        # Each pair should be a sorted list
        for pair in diag["conflict_pairs"]:
            assert isinstance(pair, list)
            assert pair == sorted(pair)

    def test_get_diagnostics_conflict_pairs_capped_at_20(self, db):
        """Conflict pairs in diagnostics should be capped at 20 for readability."""
        detector = ConflictDetector(db)

        # Add 25 conflict pairs
        for i in range(25):
            detector._published_conflicts.add(frozenset([f"evt-{i}-a", f"evt-{i}-b"]))

        diag = detector.get_diagnostics()

        assert diag["published_conflicts_count"] == 25
        assert len(diag["conflict_pairs"]) == 20

    def test_get_diagnostics_resilient_to_missing_table(self, db):
        """Diagnostics should still return when published_conflicts table is missing.

        Drops the table to simulate a fresh or corrupted state.db, then
        verifies that get_diagnostics() still returns a complete result
        with state_db_table_exists=False.
        """
        # Drop the published_conflicts table
        with db.get_connection("state") as conn:
            conn.execute("DROP TABLE IF EXISTS published_conflicts")

        detector = ConflictDetector(db)
        diag = detector.get_diagnostics()

        assert isinstance(diag, dict)
        assert set(diag.keys()) == EXPECTED_KEYS
        assert diag["state_db_table_exists"] is False
        # Other fields should still be populated
        assert diag["published_conflicts_count"] == 0
        assert diag["last_cleanup"] is None
        assert diag["calendar_events_in_window"] == 0

    def test_get_diagnostics_with_last_cleanup(self, db):
        """After a cleanup runs, last_cleanup should be a non-None ISO string."""
        detector = ConflictDetector(db)
        detector.cleanup_old_conflicts(days=30)

        from datetime import datetime, timezone

        detector._last_cleanup = datetime.now(timezone.utc)
        diag = detector.get_diagnostics()

        assert diag["last_cleanup"] is not None
        # Should be a valid ISO timestamp string
        datetime.fromisoformat(diag["last_cleanup"])
