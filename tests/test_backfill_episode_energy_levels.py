"""
Tests for episode energy_level backfill script.

Verifies that the backfill correctly computes and populates energy_level
for all historical episodes using the circadian rhythm proxy model.
"""

import pytest
from datetime import datetime, timezone
from storage.manager import DatabaseManager
from scripts.backfill_episode_energy_levels import compute_circadian_energy


class TestCircadianEnergyComputation:
    """Test the circadian energy computation function."""

    def test_early_morning_very_low_energy(self):
        """2 AM should have very low energy (0.2)."""
        timestamp = "2026-02-16T02:30:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.2

    def test_morning_ramp_up(self):
        """6 AM should be ramping up (0.4 + 1*0.07 = 0.47)."""
        timestamp = "2026-02-16T06:00:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == pytest.approx(0.47, abs=0.01)

    def test_peak_morning_energy(self):
        """10 AM should be peak morning energy (0.8)."""
        timestamp = "2026-02-16T10:30:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.8

    def test_post_lunch_dip(self):
        """1 PM should be post-lunch dip (0.6)."""
        timestamp = "2026-02-16T13:15:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.6

    def test_afternoon_peak(self):
        """3 PM should be afternoon peak (0.7)."""
        timestamp = "2026-02-16T15:45:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.7

    def test_evening_decline(self):
        """7 PM should be declining (0.5)."""
        timestamp = "2026-02-16T19:20:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.5

    def test_night_very_low_energy(self):
        """11 PM should be very low energy (0.3)."""
        timestamp = "2026-02-16T23:45:00+00:00"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.3

    def test_malformed_timestamp_returns_neutral(self):
        """Malformed timestamp should return neutral energy (0.5)."""
        timestamp = "invalid-timestamp"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.5

    def test_z_suffix_is_handled(self):
        """Timestamps with Z suffix should be parsed correctly."""
        timestamp = "2026-02-16T10:30:00Z"
        energy = compute_circadian_energy(timestamp)
        assert energy == 0.8


class TestEpisodeEnergyBackfill:
    """Test the full backfill process."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary database for testing."""
        db = DatabaseManager(str(tmp_path / "data"))
        db.initialize_all()
        yield db
        # DatabaseManager doesn't need explicit cleanup

    def test_backfill_populates_null_episodes(self, db):
        """Backfill should populate energy_level for episodes with NULL."""
        # Create test episodes with NULL energy_level
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes
                (id, timestamp, event_id, interaction_type, content_summary, energy_level)
                VALUES
                ('ep1', '2026-02-16T10:00:00+00:00', 'evt1', 'email_received', 'Morning email', NULL),
                ('ep2', '2026-02-16T14:00:00+00:00', 'evt2', 'email_sent', 'Afternoon reply', NULL),
                ('ep3', '2026-02-16T22:00:00+00:00', 'evt3', 'email_received', 'Night email', NULL)
            """)

        # Run backfill logic (manually, not via script)
        from scripts.backfill_episode_energy_levels import compute_circadian_energy

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, timestamp FROM episodes WHERE energy_level IS NULL"
            ).fetchall()

            for row in rows:
                energy = compute_circadian_energy(row["timestamp"])
                conn.execute(
                    "UPDATE episodes SET energy_level = ? WHERE id = ?",
                    (energy, row["id"])
                )

        # Verify all episodes now have energy_level
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM episodes WHERE energy_level IS NULL"
            ).fetchone()
            assert row["count"] == 0

            # Verify energy values are correct
            results = conn.execute(
                "SELECT id, energy_level FROM episodes ORDER BY id"
            ).fetchall()

            assert results[0]["id"] == "ep1"
            assert results[0]["energy_level"] == 0.8  # 10 AM = peak morning

            assert results[1]["id"] == "ep2"
            assert results[1]["energy_level"] == 0.7  # 2 PM = afternoon peak (14-17 range)

            assert results[2]["id"] == "ep3"
            assert results[2]["energy_level"] == 0.3  # 10 PM = night

    def test_backfill_preserves_existing_energy_levels(self, db):
        """Backfill should only update NULL energy_level, not existing values."""
        # Create episodes with both NULL and existing energy_level
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes
                (id, timestamp, event_id, interaction_type, content_summary, energy_level)
                VALUES
                ('ep1', '2026-02-16T10:00:00+00:00', 'evt1', 'email_received', 'Morning', NULL),
                ('ep2', '2026-02-16T14:00:00+00:00', 'evt2', 'email_sent', 'Afternoon', 0.9)
            """)

        # Run backfill
        from scripts.backfill_episode_energy_levels import compute_circadian_energy

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, timestamp FROM episodes WHERE energy_level IS NULL"
            ).fetchall()

            for row in rows:
                energy = compute_circadian_energy(row["timestamp"])
                conn.execute(
                    "UPDATE episodes SET energy_level = ? WHERE id = ?",
                    (energy, row["id"])
                )

        # Verify ep1 was updated but ep2 preserved
        with db.get_connection("user_model") as conn:
            results = conn.execute(
                "SELECT id, energy_level FROM episodes ORDER BY id"
            ).fetchall()

            assert results[0]["id"] == "ep1"
            assert results[0]["energy_level"] == 0.8  # Updated

            assert results[1]["id"] == "ep2"
            assert results[1]["energy_level"] == 0.9  # Preserved (not 0.7)

    def test_backfill_handles_large_batch(self, db):
        """Backfill should handle thousands of episodes efficiently."""
        # Create 1000 test episodes
        with db.get_connection("user_model") as conn:
            for i in range(1000):
                hour = i % 24
                conn.execute("""
                    INSERT INTO episodes
                    (id, timestamp, event_id, interaction_type, content_summary, energy_level)
                    VALUES (?, ?, ?, 'email_received', 'Test', NULL)
                """, (f"ep{i}", f"2026-02-16T{hour:02d}:00:00+00:00", f"evt{i}"))

        # Run backfill
        from scripts.backfill_episode_energy_levels import compute_circadian_energy

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, timestamp FROM episodes WHERE energy_level IS NULL"
            ).fetchall()

            for row in rows:
                energy = compute_circadian_energy(row["timestamp"])
                conn.execute(
                    "UPDATE episodes SET energy_level = ? WHERE id = ?",
                    (energy, row["id"])
                )

        # Verify all 1000 episodes have energy_level
        with db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM episodes WHERE energy_level IS NOT NULL"
            ).fetchone()
            assert row["count"] == 1000

            # Verify energy distribution makes sense (should have variety)
            row = conn.execute(
                "SELECT MIN(energy_level) as min_e, MAX(energy_level) as max_e FROM episodes"
            ).fetchone()
            assert row["min_e"] >= 0.2  # Lowest energy (early morning)
            assert row["max_e"] <= 0.8  # Highest energy (mid-morning)

    def test_backfill_handles_edge_case_timestamps(self, db):
        """Backfill should handle edge case timestamps gracefully."""
        with db.get_connection("user_model") as conn:
            conn.execute("""
                INSERT INTO episodes
                (id, timestamp, event_id, interaction_type, content_summary, energy_level)
                VALUES
                ('ep1', '2026-02-16T00:00:00+00:00', 'evt1', 'test', 'Midnight', NULL),
                ('ep2', '2026-02-16T23:59:59+00:00', 'evt2', 'test', 'Last second', NULL),
                ('ep3', '2026-02-16T12:00:00+00:00', 'evt3', 'test', 'Noon', NULL)
            """)

        # Run backfill
        from scripts.backfill_episode_energy_levels import compute_circadian_energy

        with db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT id, timestamp FROM episodes WHERE energy_level IS NULL"
            ).fetchall()

            for row in rows:
                energy = compute_circadian_energy(row["timestamp"])
                conn.execute(
                    "UPDATE episodes SET energy_level = ? WHERE id = ?",
                    (energy, row["id"])
                )

        # Verify all have valid energy values
        with db.get_connection("user_model") as conn:
            results = conn.execute(
                "SELECT id, energy_level FROM episodes ORDER BY id"
            ).fetchall()

            assert all(0.0 <= r["energy_level"] <= 1.0 for r in results)
            assert results[0]["energy_level"] == 0.2  # Midnight
            assert results[1]["energy_level"] == 0.3  # 11 PM
            assert results[2]["energy_level"] == 0.6  # Noon
