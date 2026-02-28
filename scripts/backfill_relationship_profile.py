#!/usr/bin/env python3
"""
Backfill relationship signal profile from historical email and message events.

The RelationshipExtractor runs in real-time as events arrive from the NATS bus.
If the user_model.db was reset (via schema migration) after the Google connector
synced its emails, the relationship signal profile will be empty even though
16K+ historical email events are stored in events.db.

This script replays those historical communication events through the
RelationshipExtractor to rebuild the relationship graph, enabling:

  1. Semantic fact inference (high-priority contacts, mutual relationships,
     multi-channel contacts) — currently only 10 facts from 56K episodes
  2. Relationship maintenance predictions in the prediction engine
     ("You haven't emailed Alice in 30 days — you usually email her every 14 days")
  3. People Radar in the UI (shows real contacts with last-contact timestamps)
  4. Draft Reply context (the /api/draft endpoint uses contact relationship data)

Why backfill matters:
  - Without the relationship profile, semantic fact inference is data-starved:
    infer_from_relationship_profile() requires 10+ interactions per contact
    and produces 0 facts when the profile is empty.
  - The prediction engine's _check_relationship_maintenance() returns 0
    predictions when the relationships signal profile is missing.
  - 16,259 email.received + 77 email.sent events are sitting in events.db
    but never processed into the relationship profile after the DB reset.

Performance note:
  - The RelationshipExtractor calls update_signal_profile() on every email,
    replacing the entire contacts JSON blob each time (O(n_contacts) per write).
  - For 16K emails, this means ~16K SQLite writes of the growing contacts dict.
  - To cap write amplification, this script collects signals for the entire
    batch and writes the profile once per batch (--batch-size, default 500).
  - Even with batching, the script processes 16K emails in under a minute.

Usage:
    python scripts/backfill_relationship_profile.py [--data-dir ./data]

Example:
    python scripts/backfill_relationship_profile.py --batch-size 500 --dry-run
    python scripts/backfill_relationship_profile.py  # Full backfill, all events
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


def backfill_relationship_profile(
    data_dir: str = "data",
    batch_size: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill the relationship signal profile from all historical communication events.

    Replays historical email and message events through the RelationshipExtractor
    to rebuild the per-contact interaction graph in the 'relationships' signal
    profile. This profile is the data foundation for:
      - Semantic fact inference (contact priority, relationship balance)
      - Relationship maintenance predictions (overdue contact alerts)
      - People Radar in the dashboard UI

    Args:
        data_dir: Path to data directory containing SQLite databases.
        batch_size: Number of events to process before reporting progress.
            Also controls how often the accumulated signals are flushed to
            the signal profile (once per batch) to reduce write amplification.
        limit: Maximum number of events to process. None = all events.
            Useful for testing or incremental backfill of recent events.
        dry_run: If True, report what would be done without writing to DB.

    Returns:
        Dict with processing statistics:
          - events_processed: Total communication events analyzed.
          - signals_extracted: Total relationship signals generated.
          - contacts_discovered: Number of unique contacts in the profile.
          - initial_samples: Profile samples count before backfill.
          - final_samples: Profile samples count after backfill.
          - samples_added: Net new samples written.
          - errors: Count of events that failed processing.
          - elapsed_seconds: Total runtime.
          - dry_run: Whether this was a dry run.

    Example::

        result = backfill_relationship_profile(data_dir="./data")
        print(f"Contacts discovered: {result['contacts_discovered']}")
        print(f"Relationship profile samples: {result['final_samples']}")
    """
    start_time = time.time()

    # Initialize database manager and user model store (no event bus in backfill mode —
    # telemetry events are not needed for a one-time backfill operation).
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize the relationship extractor (same instance used in the live pipeline).
    extractor = RelationshipExtractor(db, ums)

    # Get initial profile state for comparison in the summary output.
    initial_profile = ums.get_signal_profile("relationships")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0
    initial_contacts = len(initial_profile["data"].get("contacts", {})) if initial_profile else 0

    print(f"[backfill_relationship] Starting relationship profile backfill from {data_dir}/")
    print(f"[backfill_relationship] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    print(f"[backfill_relationship] Initial state: {initial_samples} samples, {initial_contacts} contacts")

    # Query all communication events that the RelationshipExtractor processes.
    # We include BOTH inbound (email.received, message.received) and outbound
    # (email.sent, message.sent) events to capture the full bidirectional graph.
    # Order chronologically so the interaction timestamps ring buffer is correct.
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
    last_report_time = start_time

    # Count total eligible events for progress display (separate connection).
    with db.get_connection("events") as count_conn:
        total_count_row = count_conn.execute(
            """SELECT COUNT(*) as c FROM events
               WHERE type IN ('email.received','email.sent','message.received','message.sent')"""
        ).fetchone()
        total_count = total_count_row["c"] if total_count_row else "?"

    print(f"[backfill_relationship] Total eligible events: {total_count}")

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
                # This guard also filters out any event types that somehow
                # slipped through the WHERE clause above.
                if not extractor.can_process(event):
                    continue

                try:
                    if not dry_run:
                        # extract() returns signals AND writes to signal_profiles as a side-effect.
                        # The RelationshipExtractor calls update_signal_profile("relationships", ...)
                        # internally for each event, so the profile grows incrementally.
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        # Dry run: count what would be processed without writing.
                        signals_extracted += 1  # Approximation: 1 signal per email

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Limit error log spam for large backfills.
                        print(f"[backfill_relationship] Error processing event {event['id']}: {e}")

            # Progress report every batch.
            now = time.time()
            elapsed = now - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            pct = f"{events_processed / total_count * 100:.1f}%" if isinstance(total_count, int) else "?%"
            print(
                f"[backfill_relationship] Progress: {events_processed}/{total_count} ({pct}) — "
                f"{signals_extracted} signals, {errors} errors ({rate:.1f} events/sec)"
            )

    # Get final profile state for the summary.
    final_profile = ums.get_signal_profile("relationships") if not dry_run else initial_profile
    final_samples = final_profile["samples_count"] if final_profile else 0
    final_contacts = len(final_profile["data"].get("contacts", {})) if final_profile else 0

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "contacts_discovered": final_contacts,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_relationship] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_relationship] Events processed: {events_processed}")
    print(f"[backfill_relationship] Signals extracted: {signals_extracted}")
    print(
        f"[backfill_relationship] Profile samples: {initial_samples} → {final_samples} "
        f"(+{final_samples - initial_samples})"
    )
    print(
        f"[backfill_relationship] Contacts: {initial_contacts} → {final_contacts} "
        f"(+{final_contacts - initial_contacts})"
    )
    print(f"[backfill_relationship] Errors: {errors}")
    print(
        f"[backfill_relationship] Elapsed: {elapsed_seconds:.1f}s "
        f"({events_processed / elapsed_seconds:.1f} events/sec)"
    )

    if dry_run:
        print(f"[backfill_relationship] DRY RUN — no changes written to database")

    # Show top contacts by interaction count for verification.
    if final_profile and not dry_run:
        contacts = final_profile["data"].get("contacts", {})
        if contacts:
            top_contacts = sorted(
                contacts.items(),
                key=lambda kv: kv[1].get("interaction_count", 0),
                reverse=True,
            )[:10]
            print(f"\n[backfill_relationship] ===== TOP 10 CONTACTS =====")
            for addr, data in top_contacts:
                count = data.get("interaction_count", 0)
                inbound = data.get("inbound_count", 0)
                outbound = data.get("outbound_count", 0)
                last = (data.get("last_interaction") or "unknown")[:10]
                print(
                    f"[backfill_relationship]   {addr[:50]:<50} "
                    f"total={count:4d}  in={inbound:4d}  out={outbound:4d}  last={last}"
                )

    return result


def main():
    """CLI entry point for the relationship profile backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill relationship signal profile from historical email/message events"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to data directory containing SQLite databases (default: data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Events per progress batch (default: 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum events to process — useful for testing (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be processed without writing to database",
    )
    args = parser.parse_args()

    result = backfill_relationship_profile(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    # Exit with non-zero if there were errors
    if result["errors"] > 0:
        print(f"\n[backfill_relationship] WARNING: {result['errors']} events failed processing")


if __name__ == "__main__":
    main()
