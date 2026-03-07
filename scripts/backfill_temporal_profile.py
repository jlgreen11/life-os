#!/usr/bin/env python3
"""
Backfill temporal profile from historical events.

The TemporalExtractor was added in PR #148 but only processes new events
going forward. This script backfills the temporal profile from all historical
events — both outbound (emails sent, tasks created, calendar events, etc.) and
inbound (emails received, messages received) — so the system immediately has
rich temporal pattern data.

Why backfill matters:
- TemporalProfile is a core Layer 1 (Episodic Memory) feature that's 100% empty
- 15,000+ historical events (12,429 email.received + 2,573 calendar + 371 tasks
  + 329 emails sent + messages) contain rich temporal signals (energy rhythms,
  productive hours, planning horizon, inbound communication patterns)
- Enables time-aware predictions today (best meeting times, energy warnings)
- Without backfill, the profile will take weeks/months to build from new events only

Impact:
- Populates activity_by_hour to detect energy peaks and troughs
- Populates activity_by_day to identify productive days vs. recharge days
- Populates scheduled_hours to understand meeting preferences
- Populates advance_planning_days to detect deadline behavior patterns

Usage:
    python scripts/backfill_temporal_profile.py [--batch-size N] [--limit N]

Example:
    python scripts/backfill_temporal_profile.py --batch-size 1000
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.temporal import TemporalExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_temporal_profile(
    data_dir: str = "data",
    batch_size: int = 1000,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill temporal profile from all historical events (outbound and inbound).

    Processes events through the TemporalExtractor to build activity patterns
    by hour, day of week, and activity type. Includes both user-initiated events
    (emails sent, tasks, calendar) and inbound events (emails received, messages
    received) for complete temporal coverage. Also tracks scheduling preferences
    and advance planning horizons from calendar events.

    Args:
        data_dir: Path to data directory containing SQLite databases
        batch_size: Number of events to process per progress report
        limit: Maximum number of events to process (None = all)
        dry_run: If True, report what would be done without writing to DB

    Returns:
        Dict with processing statistics:
        - events_processed: Total events analyzed
        - signals_extracted: Total temporal signals generated
        - profile_samples: Final sample count in temporal profile
        - errors: Count of events that failed processing
        - elapsed_seconds: Total runtime
    """
    start_time = time.time()

    # Initialize database manager and user model store
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize temporal extractor
    extractor = TemporalExtractor(db, ums)

    # Query all events that trigger temporal extraction (outbound + inbound)
    # Order by timestamp to build the profile chronologically (earliest to latest)
    # This ensures patterns evolve naturally over time
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN (
            'email.sent',
            'email.received',
            'message.sent',
            'message.received',
            'calendar.event.created',
            'calendar.event.updated',
            'task.created',
            'task.completed',
            'task.updated',
            'system.user.command'
        )
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    # Track progress and statistics
    events_processed = 0
    signals_extracted = 0
    errors = 0
    last_report_time = start_time

    with db.get_connection("events") as events_conn:
        events_conn.row_factory = sqlite3.Row

        print(f"[backfill_temporal_profile] Starting backfill from {data_dir}")
        print(f"[backfill_temporal_profile] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")

        # Get initial profile state
        initial_profile = ums.get_signal_profile("temporal")
        initial_samples = initial_profile["samples_count"] if initial_profile else 0
        print(f"[backfill_temporal_profile] Initial temporal profile samples: {initial_samples}")

        # Process events in batches for progress reporting
        cursor = events_conn.execute(query)

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for row in batch:
                # Reconstruct the event dict from database row
                event = {
                    "id": row["id"],
                    "type": row["type"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "priority": row["priority"],
                    "payload": json.loads(row["payload"]) if row["payload"] else {},
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }

                # Verify the extractor will process this event
                if not extractor.can_process(event):
                    continue

                try:
                    # Extract temporal signals
                    # This both returns signals AND updates the temporal profile as a side-effect
                    if not dry_run:
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        # In dry run, just count what would be processed
                        signals_extracted += 1  # Assume one signal per event

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Only print first 10 errors to avoid spam
                        print(f"[backfill_temporal_profile] Error processing event {event['id']}: {e}")

            # Progress reporting every batch
            elapsed = time.time() - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            print(f"[backfill_temporal_profile] Progress: {events_processed} events, "
                  f"{signals_extracted} signals, {errors} errors "
                  f"({rate:.1f} events/sec)")

    # Get final profile state
    final_profile = ums.get_signal_profile("temporal")
    final_samples = final_profile["samples_count"] if final_profile else 0

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_temporal_profile] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_temporal_profile] Events processed: {events_processed}")
    print(f"[backfill_temporal_profile] Signals extracted: {signals_extracted}")
    print(f"[backfill_temporal_profile] Profile samples: {initial_samples} → {final_samples} (+{final_samples - initial_samples})")
    print(f"[backfill_temporal_profile] Errors: {errors}")
    print(f"[backfill_temporal_profile] Elapsed: {elapsed_seconds:.1f}s ({events_processed/elapsed_seconds:.1f} events/sec)")

    if dry_run:
        print(f"[backfill_temporal_profile] DRY RUN - no changes were written to database")

    # Show sample of the temporal profile data for verification
    if final_profile and not dry_run:
        data = final_profile["data"]
        print(f"\n[backfill_temporal_profile] ===== TEMPORAL PROFILE SUMMARY =====")

        # Show top 5 most active hours
        if data.get("activity_by_hour"):
            hourly = sorted(data["activity_by_hour"].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"[backfill_temporal_profile] Top active hours: {hourly}")

        # Show activity by day of week
        if data.get("activity_by_day"):
            print(f"[backfill_temporal_profile] Activity by day: {data['activity_by_day']}")

        # Show activity type breakdown
        if data.get("activity_by_type"):
            print(f"[backfill_temporal_profile] Activity types: {data['activity_by_type']}")

        # Show advance planning statistics
        if data.get("advance_planning_days"):
            planning_days = data["advance_planning_days"]
            if planning_days:
                avg_advance = sum(planning_days) / len(planning_days)
                print(f"[backfill_temporal_profile] Advance planning: avg {avg_advance:.1f} days ({len(planning_days)} events)")

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill temporal profile from historical events"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Events per progress report (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max events to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing to database",
    )

    args = parser.parse_args()

    try:
        result = backfill_temporal_profile(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )

        # Exit with non-zero if there were errors
        if result["errors"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"[backfill_temporal_profile] FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
