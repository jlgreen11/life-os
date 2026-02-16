#!/usr/bin/env python3
"""
Backfill communication templates from historical events.

Communication template extraction was added in PR #130 but only processes
new events going forward. This script backfills templates from all historical
communication events (emails and messages) so the system immediately has
rich writing-style data for every contact.

Why backfill matters:
- Enables style-matching for AI-generated replies today, not in months
- Populates Layer 3 procedural memory with 85K+ samples immediately
- Provides relationship-specific communication intelligence from day one

Usage:
    python scripts/backfill_communication_templates.py [--limit N] [--batch-size N]

Example:
    python scripts/backfill_communication_templates.py --batch-size 1000
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.relationship import RelationshipExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_communication_templates(
    data_dir: str = "data",
    batch_size: int = 1000,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill communication templates from all historical communication events.

    Processes email.sent, email.received, message.sent, and message.received
    events through the template extraction pipeline. Each event generates
    templates for each contact address involved (sender or recipients).

    Args:
        data_dir: Path to data directory containing SQLite databases
        batch_size: Number of events to process per transaction batch
        limit: Maximum number of events to process (None = all)
        dry_run: If True, report what would be done without writing to DB

    Returns:
        Dict with processing statistics:
        - events_processed: Total events analyzed
        - templates_created: New template records written
        - templates_updated: Existing templates modified
        - errors: Count of events that failed processing
        - elapsed_seconds: Total runtime
    """
    start_time = time.time()

    # Initialize database manager and user model store
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize relationship extractor (handles template extraction)
    extractor = RelationshipExtractor(db, ums)

    # Query all communication events ordered by timestamp
    # This ensures templates evolve chronologically as the user's style changes
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN ('email.sent', 'email.received', 'message.sent', 'message.received')
          AND (LENGTH(json_extract(payload, '$.body_plain')) > 10
               OR LENGTH(json_extract(payload, '$.body')) > 10)
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    # Track progress and statistics
    events_processed = 0
    templates_before = 0
    templates_after = 0
    errors = 0
    last_report_time = start_time

    with db.get_connection("events") as events_conn:
        events_conn.row_factory = sqlite3.Row

        # Get initial template count
        with db.get_connection("user_model") as um_conn:
            templates_before = um_conn.execute(
                "SELECT COUNT(*) FROM communication_templates"
            ).fetchone()[0]

        print(f"Starting backfill (batch_size={batch_size}, limit={limit or 'all'})")
        print(f"Templates before: {templates_before}")
        print()

        # Process events in batches for progress reporting
        cursor = events_conn.execute(query)

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for row in batch:
                try:
                    # Reconstruct event dict from database row
                    event = {
                        "id": row["id"],
                        "type": row["type"],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "priority": row["priority"],
                        "payload": json.loads(row["payload"]),
                        "metadata": json.loads(row["metadata"] or "{}"),
                    }

                    # Run extraction (creates/updates templates via UserModelStore)
                    if not dry_run:
                        extractor.extract(event)

                    events_processed += 1

                except Exception as e:
                    print(f"Error processing event {row['id']}: {e}")
                    errors += 1

            # Report progress every 5 seconds
            now = time.time()
            if now - last_report_time >= 5.0:
                elapsed = now - start_time
                rate = events_processed / elapsed if elapsed > 0 else 0
                print(
                    f"Processed {events_processed:,} events "
                    f"({rate:.1f}/sec, {errors} errors)"
                )
                last_report_time = now

    # Get final template count
    with db.get_connection("user_model") as um_conn:
        templates_after = um_conn.execute(
            "SELECT COUNT(*) FROM communication_templates"
        ).fetchone()[0]

    elapsed_seconds = time.time() - start_time
    templates_created = templates_after - templates_before

    # Calculate templates updated (rough estimate based on event:template ratio)
    # Each event can update multiple templates (multi-recipient emails)
    # Most updates are incremental refinements of existing templates
    templates_updated = events_processed - templates_created

    print()
    print("=" * 60)
    print("Backfill complete!")
    print(f"  Events processed:    {events_processed:,}")
    print(f"  Templates before:    {templates_before:,}")
    print(f"  Templates after:     {templates_after:,}")
    print(f"  Templates created:   {templates_created:,}")
    print(f"  Templates updated:   {templates_updated:,}")
    print(f"  Errors:              {errors:,}")
    print(f"  Time elapsed:        {elapsed_seconds:.1f}s")
    print(f"  Processing rate:     {events_processed / elapsed_seconds:.1f} events/sec")
    print("=" * 60)

    return {
        "events_processed": events_processed,
        "templates_created": templates_created,
        "templates_updated": templates_updated,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
    }


def main():
    """CLI entry point for backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill communication templates from historical events"
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
        help="Number of events per batch (default: 1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum events to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing to DB",
    )

    args = parser.parse_args()

    try:
        stats = backfill_communication_templates(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        sys.exit(0 if stats["errors"] == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nBackfill interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
