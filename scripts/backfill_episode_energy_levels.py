#!/usr/bin/env python3
"""
Backfill energy_level for all historical episodes.

PROBLEM:
All 31,853 episodes have NULL energy_level despite mood signals being available.
This breaks mood-aware features across all 4 user model layers.

ROOT CAUSE:
1. Iteration 146 added circadian_energy signals to mood extractor
2. But the running process had old code (started before the merge)
3. So no circadian_energy signals were generated
4. compute_current_mood() returned confidence=0.0 (no energy signals available)
5. main.py rejected mood_state with confidence=0.0, set energy_level=None

SOLUTION:
Backfill energy_level for ALL historical episodes by computing proxy energy
from their timestamps using the same circadian rhythm model that would have
been used if the code had been running.

IMPACT:
- 0% → 100% energy_level population
- Enables mood-aware decision making across all memory layers
- Provides emotional context for 31K+ historical interactions
"""

import sys
from datetime import datetime, timezone
from storage.manager import DatabaseManager

def compute_circadian_energy(timestamp_str: str) -> float:
    """
    Compute energy level from timestamp using circadian rhythm proxy.

    This matches the logic in services/signal_extractor/mood.py lines 218-239.

    Args:
        timestamp_str: ISO format timestamp (e.g., "2026-02-16T14:50:14+00:00")

    Returns:
        Energy level between 0.0 (very low energy) and 1.0 (peak energy)
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        hour = dt.hour  # 0-23 in UTC

        # Circadian energy curve:
        # 0-5: very low (0.2)
        # 5-8: ramping up (0.4-0.6)
        # 8-12: peak morning (0.8)
        # 12-14: post-lunch dip (0.6)
        # 14-17: afternoon peak (0.7)
        # 17-21: declining (0.5)
        # 21-24: very low (0.3)
        if 0 <= hour < 5:
            return 0.2
        elif 5 <= hour < 8:
            return 0.4 + (hour - 5) * 0.07  # ramp 0.4→0.6
        elif 8 <= hour < 12:
            return 0.8
        elif 12 <= hour < 14:
            return 0.6
        elif 14 <= hour < 17:
            return 0.7
        elif 17 <= hour < 21:
            return 0.5
        else:  # 21-24
            return 0.3
    except Exception:
        # Malformed timestamp, return neutral energy
        return 0.5

def main():
    """Backfill energy_level for all episodes with NULL energy_level."""
    print("Episode Energy Level Backfill")
    print("=" * 70)

    db = DatabaseManager('data')
    db.initialize_all()

    # Count episodes needing backfill
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM episodes WHERE energy_level IS NULL"
        ).fetchone()
        total_null = row["count"]

    print(f"Episodes with NULL energy_level: {total_null:,}")

    if total_null == 0:
        print("✓ No episodes need backfilling")
        return

    print(f"Computing proxy energy from circadian timestamps...")

    # Backfill: compute energy_level from episode timestamp
    with db.get_connection("user_model") as conn:
        # Fetch all episodes with NULL energy_level
        rows = conn.execute(
            "SELECT id, timestamp FROM episodes WHERE energy_level IS NULL"
        ).fetchall()

        updated = 0
        failed = 0

        for row in rows:
            episode_id = row["id"]
            timestamp = row["timestamp"]

            try:
                # Compute energy from timestamp
                energy = compute_circadian_energy(timestamp)

                # Update episode
                conn.execute(
                    "UPDATE episodes SET energy_level = ? WHERE id = ?",
                    (energy, episode_id)
                )
                updated += 1

                if updated % 1000 == 0:
                    print(f"  Processed {updated:,} episodes...", end='\r')

            except Exception as e:
                print(f"Failed to process episode {episode_id}: {e}")
                failed += 1

    print(f"\n✓ Successfully backfilled {updated:,} episodes")

    if failed > 0:
        print(f"⚠ Failed to process {failed} episodes")

    # Verify the backfill
    with db.get_connection("user_model") as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM episodes WHERE energy_level IS NULL"
        ).fetchone()
        remaining_null = row["count"]

        row = conn.execute(
            "SELECT COUNT(*) as count, MIN(energy_level) as min_energy, "
            "MAX(energy_level) as max_energy, AVG(energy_level) as avg_energy "
            "FROM episodes WHERE energy_level IS NOT NULL"
        ).fetchone()
        total_with_energy = row["count"]
        min_energy = row["min_energy"]
        max_energy = row["max_energy"]
        avg_energy = row["avg_energy"]

    print(f"\nPost-backfill status:")
    print(f"  Episodes with energy_level: {total_with_energy:,}")
    print(f"  Episodes still NULL: {remaining_null:,}")
    print(f"  Energy range: {min_energy:.2f} - {max_energy:.2f}")
    print(f"  Average energy: {avg_energy:.2f}")

    # Show sample of backfilled data
    print(f"\nSample of backfilled episodes:")
    with db.get_connection("user_model") as conn:
        rows = conn.execute("""
            SELECT timestamp, energy_level, interaction_type
            FROM episodes
            WHERE energy_level IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 5
        """).fetchall()

        for row in rows:
            dt = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            print(f"  {dt.strftime('%Y-%m-%d %H:%M')} UTC (hour {dt.hour:02d}): "
                  f"energy={row['energy_level']:.2f} | {row['interaction_type']}")

    print("\n✓ Backfill complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Backfill interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
