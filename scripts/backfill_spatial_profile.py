#!/usr/bin/env python3
"""
Backfill spatial profile from historical events.

The SpatialExtractor was added in PR #150 but only processes new events
going forward. This script backfills the spatial profile from all historical
calendar events with location data so the system immediately has rich spatial
pattern data.

Why backfill matters:
- SpatialProfile is a core Layer 1 (Episodic Memory) feature that's 100% empty
- 642+ historical calendar events contain location data (meeting rooms, offices, etc.)
- Enables location-aware predictions today (notification preferences by place,
  dominant work/personal mode detection, context switching)
- Without backfill, the profile will take weeks/months to build from new events only

Impact:
- Populates place_behaviors to detect where user works, meets, and spends time
- Populates visit_count and average_duration for each location
- Populates dominant_domain (work vs personal) by location
- Populates typical_activities to understand what happens at each place
- Enables location-aware notification preferences and context detection

Usage:
    python scripts/backfill_spatial_profile.py [--batch-size N] [--limit N]

Example:
    python scripts/backfill_spatial_profile.py --batch-size 1000
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.spatial import SpatialExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_spatial_profile(
    data_dir: str = "data",
    batch_size: int = 1000,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill spatial profile from all historical events with location data.

    Processes events through the SpatialExtractor to build place behaviors
    tracking visit patterns, durations, and dominant activities at each location.

    Args:
        data_dir: Path to data directory containing SQLite databases
        batch_size: Number of events to process per progress report
        limit: Maximum number of events to process (None = all)
        dry_run: If True, report what would be done without writing to DB

    Returns:
        Dict with processing statistics:
        - events_processed: Total events analyzed
        - signals_extracted: Total spatial signals generated
        - profile_samples: Final sample count in spatial profile
        - unique_places: Number of distinct locations discovered
        - errors: Count of events that failed processing
        - elapsed_seconds: Total runtime
    """
    start_time = time.time()

    # Initialize database manager and user model store
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize spatial extractor
    extractor = SpatialExtractor(db, ums)

    # Query all events with location data
    # Order by timestamp to build the profile chronologically (earliest to latest)
    # This ensures patterns evolve naturally over time
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE (
            (type = 'calendar.event.created' AND json_extract(payload, '$.location') IS NOT NULL AND json_extract(payload, '$.location') != '')
            OR (type = 'ios.context.update' AND (json_extract(payload, '$.location') IS NOT NULL OR json_extract(payload, '$.device_proximity') IS NOT NULL))
            OR type = 'system.user.location_update'
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

        print(f"[backfill_spatial_profile] Starting backfill from {data_dir}")
        print(f"[backfill_spatial_profile] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")

        # Get initial profile state
        initial_profile = ums.get_signal_profile("spatial")
        initial_samples = initial_profile["samples_count"] if initial_profile else 0
        print(f"[backfill_spatial_profile] Initial spatial profile samples: {initial_samples}")

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
                    # Extract spatial signals
                    # This both returns signals AND updates the spatial profile as a side-effect
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
                        print(f"[backfill_spatial_profile] Error processing event {event['id']}: {e}")

            # Progress reporting every batch
            elapsed = time.time() - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            print(f"[backfill_spatial_profile] Progress: {events_processed} events, "
                  f"{signals_extracted} signals, {errors} errors "
                  f"({rate:.1f} events/sec)")

    # Get final profile state
    final_profile = ums.get_signal_profile("spatial")
    final_samples = final_profile["samples_count"] if final_profile else 0

    # Count unique places
    unique_places = 0
    if final_profile and not dry_run:
        place_behaviors_raw = final_profile.get("data", {}).get("place_behaviors", {})
        if isinstance(place_behaviors_raw, str):
            place_behaviors = json.loads(place_behaviors_raw)
        else:
            place_behaviors = place_behaviors_raw if place_behaviors_raw else {}
        unique_places = len(place_behaviors)

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "unique_places": unique_places,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_spatial_profile] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_spatial_profile] Events processed: {events_processed}")
    print(f"[backfill_spatial_profile] Signals extracted: {signals_extracted}")
    print(f"[backfill_spatial_profile] Profile samples: {initial_samples} → {final_samples} (+{final_samples - initial_samples})")
    print(f"[backfill_spatial_profile] Unique places: {unique_places}")
    print(f"[backfill_spatial_profile] Errors: {errors}")
    print(f"[backfill_spatial_profile] Elapsed: {elapsed_seconds:.1f}s ({events_processed/elapsed_seconds:.1f} events/sec)")

    if dry_run:
        print(f"[backfill_spatial_profile] DRY RUN - no changes were written to database")

    # Show sample of the spatial profile data for verification
    if final_profile and not dry_run:
        data = final_profile["data"]
        print(f"\n[backfill_spatial_profile] ===== SPATIAL PROFILE SUMMARY =====")

        place_behaviors_raw = data.get("place_behaviors", {})
        if isinstance(place_behaviors_raw, str):
            place_behaviors = json.loads(place_behaviors_raw)
        else:
            place_behaviors = place_behaviors_raw if place_behaviors_raw else {}

        if place_behaviors:
            # Show top 5 most visited places
            sorted_places = sorted(
                place_behaviors.items(),
                key=lambda x: x[1].get("visit_count", 0),
                reverse=True
            )[:5]

            print(f"[backfill_spatial_profile] Top 5 visited places:")
            for location, place in sorted_places:
                visits = place.get("visit_count", 0)
                domain = place.get("dominant_domain", "unknown")
                avg_duration = place.get("average_duration_minutes", 0)
                print(f"  - {location[:50]}: {visits} visits, {domain}, avg {avg_duration:.0f}min")

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill spatial profile from historical events"
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
        result = backfill_spatial_profile(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )

        # Exit with non-zero if there were errors
        if result["errors"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"[backfill_spatial_profile] FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
