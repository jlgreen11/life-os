"""
Backfill resolution for orphaned predictions.

PROBLEM:
--------
When predictions are created with was_surfaced=1 but no corresponding notification
exists in the notifications table, they become "orphaned" — they'll never be
auto-resolved by the normal auto_resolve_stale_predictions() flow, which only
processes predictions that have associated notifications.

This happens when:
1. The prediction engine was running before the notification system was wired up
2. The notification creation failed silently
3. A code bug prevented notification creation for certain prediction types

As of Feb 16, 2026, there are 2,030 unresolved predictions with was_surfaced=1
but no matching notification. These will sit in the database forever, polluting
accuracy metrics and making the "unresolved predictions" metric meaningless.

SOLUTION:
---------
This script identifies orphaned predictions and resolves them based on age:

1. Surfaced predictions older than 24 hours with no notification → mark as
   'ignored' (user never saw them, treat as implicit dismissal)

2. Surfaced predictions older than 48 hours with no notification → mark as
   'stale' (even more clear they're not actionable)

The resolution logic mirrors the behavior of auto_resolve_stale_predictions()
but operates directly on the predictions table without requiring notifications.

USAGE:
------
python scripts/backfill_orphaned_predictions.py [--dry-run]

    --dry-run: Show what would be resolved without making changes
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def backfill_orphaned_predictions(db_path: Path, dry_run: bool = False) -> dict:
    """
    Resolve orphaned predictions that were surfaced but never got notifications.

    An orphaned prediction is one where:
    - was_surfaced = 1 (prediction was shown to the user in theory)
    - resolved_at IS NULL (prediction was never resolved)
    - No matching notification exists in state.db

    These predictions will never be resolved by the normal auto-resolve flow
    because that flow only looks at predictions with associated notifications.

    Args:
        db_path: Path to the Life OS data directory
        dry_run: If True, report what would be changed without making changes

    Returns:
        Dict with counts of predictions processed
    """
    user_model_db = db_path / "user_model.db"
    state_db = db_path / "state.db"

    if not user_model_db.exists():
        raise FileNotFoundError(f"Database not found: {user_model_db}")
    if not state_db.exists():
        raise FileNotFoundError(f"Database not found: {state_db}")

    # Connect to both databases
    user_model_conn = sqlite3.connect(user_model_db)
    user_model_conn.row_factory = sqlite3.Row
    state_conn = sqlite3.connect(state_db)
    state_conn.row_factory = sqlite3.Row

    now = datetime.now(timezone.utc)
    cutoff_24h = (now - timedelta(hours=24)).isoformat()
    cutoff_48h = (now - timedelta(hours=48)).isoformat()

    stats = {
        "total_unresolved": 0,
        "orphaned_24h": 0,
        "orphaned_48h": 0,
        "resolved": 0,
    }

    try:
        # Step 1: Find all unresolved surfaced predictions
        cursor = user_model_conn.execute(
            """SELECT id, created_at, prediction_type, confidence
               FROM predictions
               WHERE was_surfaced = 1
                 AND resolved_at IS NULL
               ORDER BY created_at ASC"""
        )
        unresolved = cursor.fetchall()
        stats["total_unresolved"] = len(unresolved)

        print(f"Found {stats['total_unresolved']} unresolved surfaced predictions")

        if stats["total_unresolved"] == 0:
            print("✓ No orphaned predictions to resolve")
            return stats

        # Step 2: Check each prediction for a matching notification
        orphaned_predictions = []

        for pred in unresolved:
            pred_id = pred["id"]
            created_at = pred["created_at"]

            # Check if a notification exists for this prediction
            notif_cursor = state_conn.execute(
                "SELECT id FROM notifications WHERE source_event_id = ? LIMIT 1",
                (pred_id,),
            )
            notification = notif_cursor.fetchone()

            if notification is None:
                # This is an orphaned prediction — no notification exists
                orphaned_predictions.append(pred)

                # Categorize by age
                if created_at < cutoff_48h:
                    stats["orphaned_48h"] += 1
                elif created_at < cutoff_24h:
                    stats["orphaned_24h"] += 1

        print(f"Found {len(orphaned_predictions)} orphaned predictions (no notification)")
        print(f"  - {stats['orphaned_24h']} older than 24 hours")
        print(f"  - {stats['orphaned_48h']} older than 48 hours")

        if len(orphaned_predictions) == 0:
            print("✓ All surfaced predictions have notifications")
            return stats

        # Step 3: Resolve orphaned predictions older than 24 hours
        # These are clearly stale — if the user hasn't seen them by now, they never will
        if not dry_run:
            for pred in orphaned_predictions:
                pred_id = pred["id"]
                created_at = pred["created_at"]

                # Only resolve predictions older than 24 hours
                # Recent predictions might still be in the delivery queue
                if created_at < cutoff_24h:
                    user_model_conn.execute(
                        """UPDATE predictions
                           SET was_accurate = 0,
                               resolved_at = ?,
                               user_response = 'orphaned'
                           WHERE id = ?""",
                        (now.isoformat(), pred_id),
                    )
                    stats["resolved"] += 1

            user_model_conn.commit()
            print(f"✓ Resolved {stats['resolved']} orphaned predictions")
        else:
            # Dry run — count what would be resolved
            stats["resolved"] = sum(
                1 for pred in orphaned_predictions if pred["created_at"] < cutoff_24h
            )
            print(f"[DRY RUN] Would resolve {stats['resolved']} orphaned predictions")

        return stats

    finally:
        user_model_conn.close()
        state_conn.close()


def main():
    """CLI entry point for the backfill script."""
    parser = argparse.ArgumentParser(
        description="Resolve orphaned predictions (surfaced but no notification)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be resolved without making changes",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to the Life OS data directory (default: data/)",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Backfill Orphaned Predictions")
    print("=" * 60)
    print()

    try:
        stats = backfill_orphaned_predictions(args.data_dir, dry_run=args.dry_run)
        print()
        print("Summary:")
        print(f"  Total unresolved surfaced predictions: {stats['total_unresolved']}")
        print(f"  Orphaned predictions (24h+): {stats['orphaned_24h']}")
        print(f"  Orphaned predictions (48h+): {stats['orphaned_48h']}")
        print(f"  Resolved: {stats['resolved']}")
        print()

        if args.dry_run:
            print("Run without --dry-run to apply changes")
        else:
            print("✓ Backfill complete")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
