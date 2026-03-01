#!/usr/bin/env python3
"""
Backfill cadence signal profile from historical email and message events.

The CadenceExtractor runs in real-time as events arrive from the NATS bus.
If the user_model.db was reset (via schema migration or rebuild) after the
Google connector synced its emails, the cadence signal profile will be empty
even though thousands of historical communication events are stored in events.db.

This script replays those historical communication events through the
CadenceExtractor to rebuild the cadence profile, enabling:

  1. Response-time priority contact detection in the prediction engine
     (_check_relationship_maintenance uses per-contact response times to
     identify high-priority contacts the user hasn't replied to)
  2. Activity-window heatmaps (peak hours, quiet hours) that drive
     preparation-need predictions and quiet-hours enforcement
  3. Per-domain and per-channel response-time breakdowns shown in the
     Insights tab cadence profile card

Why backfill matters:
  - Without the cadence profile, the prediction engine cannot detect which
    contacts the user prioritises (fast replies) vs. avoids (slow replies).
  - Activity-window detection requires 50+ samples before producing peak/quiet
    hours — live events accumulate too slowly after a DB rebuild.
  - Historical email.received/sent and message.received/sent events sitting in
    events.db were never processed into the cadence profile after the reset.

Performance note:
  - The CadenceExtractor is lightweight (timestamp math, histogram updates).
  - Processing 16K+ events completes in under a minute.

Usage:
    python scripts/backfill_cadence_profile.py [--data-dir ./data]

Example:
    python scripts/backfill_cadence_profile.py --batch-size 500 --dry-run
    python scripts/backfill_cadence_profile.py  # Full backfill, all events
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.cadence import CadenceExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_cadence_profile(
    data_dir: str = "data",
    batch_size: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill the cadence signal profile from all historical communication events.

    Replays historical email and message events through the CadenceExtractor to
    rebuild the cadence profile (response times, activity heatmaps, per-contact
    and per-channel breakdowns).  This profile is the data foundation for:
      - Response-time priority contact detection in the prediction engine
      - Activity-window heatmaps (peak hours, quiet hours)
      - Per-domain and per-channel response-time breakdowns

    Args:
        data_dir: Path to data directory containing SQLite databases.
        batch_size: Number of events to process before reporting progress.
        limit: Maximum number of events to process. None = all events.
            Useful for testing or incremental backfill of recent events.
        dry_run: If True, report what would be done without writing to DB.

    Returns:
        Dict with processing statistics:
          - events_processed: Total communication events analyzed.
          - signals_extracted: Total cadence signals generated.
          - contacts_tracked: Number of unique contacts with response-time data.
          - initial_samples: Profile samples count before backfill.
          - final_samples: Profile samples count after backfill.
          - samples_added: Net new samples written.
          - errors: Count of events that failed processing.
          - elapsed_seconds: Total runtime.
          - dry_run: Whether this was a dry run.

    Example::

        result = backfill_cadence_profile(data_dir="./data")
        print(f"Contacts tracked: {result['contacts_tracked']}")
        print(f"Cadence profile samples: {result['final_samples']}")
    """
    start_time = time.time()

    # Initialize database manager and user model store (no event bus in backfill mode —
    # telemetry events are not needed for a one-time backfill operation).
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize the cadence extractor (same instance used in the live pipeline).
    extractor = CadenceExtractor(db, ums)

    # Get initial profile state for comparison in the summary output.
    initial_profile = ums.get_signal_profile("cadence")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    print(f"[backfill_cadence] Starting cadence profile backfill from {data_dir}/")
    print(f"[backfill_cadence] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    print(f"[backfill_cadence] Initial state: {initial_samples} samples")

    # Query all communication events that the CadenceExtractor processes.
    # Both inbound (email.received, message.received) and outbound (email.sent,
    # message.sent) are needed: inbound anchors response-time calculations,
    # outbound provides the user's actual reply latency.
    # Order chronologically so response-time deltas can be computed correctly.
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN (
            'email.received',
            'email.sent',
            'message.received',
            'message.sent'
        )
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    # Track processing statistics.
    events_processed = 0
    signals_extracted = 0
    errors = 0

    # Count total eligible events for progress display (separate connection).
    with db.get_connection("events") as count_conn:
        total_count_row = count_conn.execute(
            """SELECT COUNT(*) as c FROM events
               WHERE type IN (
                   'email.received', 'email.sent',
                   'message.received', 'message.sent'
               )"""
        ).fetchone()
        total_count = total_count_row["c"] if total_count_row else "?"

    print(f"[backfill_cadence] Total eligible events: {total_count}")

    with db.get_connection("events") as events_conn:
        events_conn.row_factory = sqlite3.Row

        cursor = events_conn.execute(query)

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for row in batch:
                # Reconstruct the event dict in the same format the live pipeline uses.
                # The payload column is stored as a JSON string in events.db; decode it.
                try:
                    payload_raw = row["payload"]
                    payload = json.loads(payload_raw) if payload_raw else {}
                    # Guard against double-encoded payloads (stored as JSON string of string).
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    payload = {}

                try:
                    metadata_raw = row["metadata"]
                    metadata = json.loads(metadata_raw) if metadata_raw else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                event = {
                    "id": row["id"],
                    "type": row["type"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "priority": row["priority"],
                    "payload": payload,
                    "metadata": metadata,
                }

                # Verify the extractor considers this event processable.
                if not extractor.can_process(event):
                    continue

                try:
                    if not dry_run:
                        # extract() returns signals AND writes to signal_profiles as a side-effect.
                        # The CadenceExtractor calls update_signal_profile("cadence", ...)
                        # internally, updating response times and activity histograms.
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        # Dry run: count what would be processed without writing.
                        signals_extracted += 1  # Approximation: 1 signal per event

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Limit error log spam for large backfills.
                        print(f"[backfill_cadence] Error processing event {event['id']}: {e}")

            # Progress report every batch.
            now = time.time()
            elapsed = now - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            pct = f"{events_processed / total_count * 100:.1f}%" if isinstance(total_count, int) else "?%"
            print(
                f"[backfill_cadence] Progress: {events_processed}/{total_count} ({pct}) — "
                f"{signals_extracted} signals, {errors} errors ({rate:.1f} events/sec)"
            )

    # Get final profile state for the summary.
    final_profile = ums.get_signal_profile("cadence") if not dry_run else initial_profile
    final_samples = final_profile["samples_count"] if final_profile else 0
    contacts_tracked = len(final_profile["data"].get("per_contact_response_times", {})) if final_profile else 0

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "contacts_tracked": contacts_tracked,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_cadence] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_cadence] Events processed: {events_processed}")
    print(f"[backfill_cadence] Signals extracted: {signals_extracted}")
    print(
        f"[backfill_cadence] Profile samples: {initial_samples} → {final_samples} "
        f"(+{final_samples - initial_samples})"
    )
    print(f"[backfill_cadence] Contacts tracked: {contacts_tracked}")
    print(f"[backfill_cadence] Errors: {errors}")
    print(
        f"[backfill_cadence] Elapsed: {elapsed_seconds:.1f}s "
        f"({events_processed / max(elapsed_seconds, 0.001):.1f} events/sec)"
    )

    if dry_run:
        print(f"[backfill_cadence] DRY RUN — no changes written to database")

    # Show activity summary for verification.
    if final_profile and not dry_run:
        hourly = final_profile["data"].get("hourly_activity", {})
        if hourly:
            top_hours = sorted(hourly.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print(f"\n[backfill_cadence] ===== TOP 5 ACTIVE HOURS =====")
            for hour, count in top_hours:
                print(f"  Hour {hour}: {count} events")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill cadence signal profile from historical events")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--batch-size", type=int, default=500, help="Events per batch (default: 500)")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing to DB")
    args = parser.parse_args()

    stats = backfill_cadence_profile(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    sys.exit(0 if stats["errors"] == 0 else 1)
