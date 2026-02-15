#!/usr/bin/env python3
"""
Life OS — One-time Prediction Backlog Cleanup

This script cleans up the massive backlog of unresolved predictions that
accumulated before the auto-resolve system was implemented. It should be run
once after deploying the auto-resolve feature to clear the historical backlog.

Background:
- 271K+ predictions were generated before auto-resolution was implemented
- Only 35 were surfaced to the user via notifications
- The remaining 266K+ sit unresolved, causing database bloat and inaccurate metrics
- The auto_resolve_filtered_predictions() function was added in iteration 7,
  but it only runs on a timer going forward — it doesn't clean up the backlog

This script:
1. Identifies all unsurfaced predictions older than 1 hour
2. Marks them as resolved with user_response='filtered'
3. Preserves the 35 surfaced predictions for user feedback
4. Reports statistics before/after cleanup

Usage:
    python scripts/cleanup-prediction-backlog.py [--data-dir ./data] [--dry-run]

Safety:
- Uses the same logic as auto_resolve_filtered_predictions()
- Only touches predictions with was_surfaced=0 (never shown to user)
- Leaves surfaced predictions alone (they need user feedback)
- Dry-run mode available for verification before applying changes
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager


def analyze_backlog(db: DatabaseManager) -> dict:
    """Analyze the current state of the prediction backlog.

    Returns:
        Dictionary with statistics about predictions needing cleanup.
    """
    with db.get_connection("user_model") as conn:
        # Total predictions
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # Surfaced vs unsurfaced
        surfaced = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 1"
        ).fetchone()[0]
        unsurfaced = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE was_surfaced = 0"
        ).fetchone()[0]

        # Resolved vs unresolved
        resolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE resolved_at IS NULL"
        ).fetchone()[0]

        # Backlog: unsurfaced + unresolved + older than 1 hour
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=1)).isoformat()
        backlog = conn.execute(
            """SELECT COUNT(*) FROM predictions
               WHERE was_surfaced = 0
               AND resolved_at IS NULL
               AND created_at < ?""",
            (cutoff,)
        ).fetchone()[0]

        # Age range of backlog
        age_stats = conn.execute(
            """SELECT MIN(created_at) as oldest, MAX(created_at) as newest
               FROM predictions
               WHERE was_surfaced = 0 AND resolved_at IS NULL"""
        ).fetchone()

        return {
            "total": total,
            "surfaced": surfaced,
            "unsurfaced": unsurfaced,
            "resolved": resolved,
            "unresolved": unresolved,
            "backlog_size": backlog,
            "oldest_backlog": age_stats[0] if age_stats[0] else None,
            "newest_backlog": age_stats[1] if age_stats[1] else None,
        }


def cleanup_backlog(db: DatabaseManager, timeout_hours: int = 1, dry_run: bool = False) -> int:
    """Clean up the prediction backlog.

    Marks all unsurfaced predictions older than timeout_hours as resolved
    with user_response='filtered'. This uses the same logic as
    NotificationManager.auto_resolve_filtered_predictions() but targets
    the historical backlog.

    Args:
        db: DatabaseManager instance
        timeout_hours: Hours after creation to consider stale (default 1)
        dry_run: If True, only count what would be cleaned up

    Returns:
        Number of predictions cleaned up (or that would be cleaned up in dry-run)
    """
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(hours=timeout_hours)).isoformat()

    with db.get_connection("user_model") as conn:
        if dry_run:
            # Just count what would be affected
            result = conn.execute(
                """SELECT COUNT(*) FROM predictions
                   WHERE was_surfaced = 0
                   AND resolved_at IS NULL
                   AND created_at < ?""",
                (cutoff,)
            )
            return result.fetchone()[0]
        else:
            # Actually perform the cleanup
            result = conn.execute(
                """UPDATE predictions SET
                   was_accurate = NULL,
                   resolved_at = ?,
                   user_response = 'filtered'
                   WHERE was_surfaced = 0
                   AND resolved_at IS NULL
                   AND created_at < ?""",
                (now.isoformat(), cutoff),
            )
            return result.rowcount


def main():
    parser = argparse.ArgumentParser(
        description="Clean up the Life OS prediction backlog"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to data directory (default: ./data)"
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=1,
        help="Hours after creation to consider stale (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without making changes"
    )
    args = parser.parse_args()

    # Initialize database
    db = DatabaseManager(args.data_dir)

    print("=" * 70)
    print("Life OS — Prediction Backlog Cleanup")
    print("=" * 70)
    print()

    # Analyze current state
    print("Analyzing current state...")
    stats = analyze_backlog(db)

    print(f"\nCurrent State:")
    print(f"  Total predictions:        {stats['total']:,}")
    print(f"  Surfaced (shown to user): {stats['surfaced']:,}")
    print(f"  Unsurfaced (filtered):    {stats['unsurfaced']:,}")
    print(f"  Resolved:                 {stats['resolved']:,}")
    print(f"  Unresolved:               {stats['unresolved']:,}")
    print(f"  Backlog to clean:         {stats['backlog_size']:,}")

    if stats['oldest_backlog']:
        print(f"\nBacklog age range:")
        print(f"  Oldest: {stats['oldest_backlog']}")
        print(f"  Newest: {stats['newest_backlog']}")

    if stats['backlog_size'] == 0:
        print("\n✓ No backlog to clean up!")
        return 0

    # Perform cleanup
    print()
    if args.dry_run:
        print("DRY RUN MODE — no changes will be made")
        cleaned = cleanup_backlog(db, args.timeout_hours, dry_run=True)
        print(f"\nWould clean up: {cleaned:,} predictions")
    else:
        print(f"Cleaning up {stats['backlog_size']:,} predictions...")
        cleaned = cleanup_backlog(db, args.timeout_hours, dry_run=False)
        print(f"✓ Cleaned up: {cleaned:,} predictions")

        # Show new state
        print("\nAnalyzing final state...")
        final_stats = analyze_backlog(db)
        print(f"\nFinal State:")
        print(f"  Total predictions:        {final_stats['total']:,}")
        print(f"  Resolved:                 {final_stats['resolved']:,}")
        print(f"  Unresolved:               {final_stats['unresolved']:,}")
        print(f"  Backlog remaining:        {final_stats['backlog_size']:,}")

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
