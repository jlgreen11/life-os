#!/usr/bin/env python3
"""
Backfill linguistic_inbound signal profile from historical inbound message events.

The LinguisticExtractor builds a ``linguistic_inbound`` signal profile from
incoming messages (email.received, message.received).  This profile stores
per-contact incoming style data — formality levels, hedge rates, question
rates — enabling tone-shift detection and formality-mismatch awareness.

After a DB reset or fresh start, this profile is empty even though thousands
of inbound events are stored in events.db.  Without backfilling, the
SemanticFactInferrer's ``infer_from_inbound_linguistic_profile()`` produces
zero facts because it requires 10+ samples in the ``linguistic_inbound``
profile.

This script replays historical inbound communication events through the
LinguisticExtractor to rebuild the per-contact incoming style profiles.

Performance note:
  - The LinguisticExtractor is lightweight (pure text analysis, no LLM calls).
  - Processing 12K+ inbound emails completes in under 60 seconds.

Usage:
    python scripts/backfill_linguistic_inbound_profile.py [--data-dir ./data]

Example:
    python scripts/backfill_linguistic_inbound_profile.py --batch-size 500 --dry-run
    python scripts/backfill_linguistic_inbound_profile.py  # Full backfill
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.linguistic import LinguisticExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_linguistic_inbound_profile(
    data_dir: str = "data",
    batch_size: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill the linguistic_inbound signal profile from historical inbound events.

    Replays historical email.received and message.received events through the
    LinguisticExtractor to rebuild per-contact incoming style profiles in the
    ``linguistic_inbound`` signal profile.  This profile is the data foundation
    for the SemanticFactInferrer's ``infer_from_inbound_linguistic_profile()``
    which produces facts about the user's communication environment (formal vs.
    casual contacts, high question-rate contacts, etc.).

    Args:
        data_dir: Path to data directory containing SQLite databases.
        batch_size: Number of events to process before reporting progress.
        limit: Maximum number of events to process. None = all events.
        dry_run: If True, report what would be done without writing to DB.

    Returns:
        Dict with processing statistics:
          - events_processed: Total inbound events analyzed.
          - signals_extracted: Total linguistic signals generated.
          - initial_samples: Profile samples count before backfill.
          - final_samples: Profile samples count after backfill.
          - samples_added: Net new samples written.
          - errors: Count of events that failed processing.
          - elapsed_seconds: Total runtime.
          - dry_run: Whether this was a dry run.

    Example::

        result = backfill_linguistic_inbound_profile(data_dir="./data")
        print(f"Inbound profile samples: {result['final_samples']}")
    """
    start_time = time.time()

    # Initialize database manager and user model store (no event bus in backfill mode).
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize the linguistic extractor (same instance used in the live pipeline).
    extractor = LinguisticExtractor(db, ums)

    # Get initial profile state for comparison in the summary output.
    initial_profile = ums.get_signal_profile("linguistic_inbound")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    print(f"[backfill_linguistic_inbound] Starting linguistic_inbound profile backfill from {data_dir}/")
    print(f"[backfill_linguistic_inbound] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    print(f"[backfill_linguistic_inbound] Initial state: {initial_samples} samples")

    # Query ONLY inbound communication events — email.received and message.received.
    # Order chronologically so per-contact rolling averages evolve correctly.
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN ('email.received', 'message.received')
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    # Track processing statistics.
    events_processed = 0
    signals_extracted = 0
    errors = 0

    # Count total eligible events for progress display.
    with db.get_connection("events") as count_conn:
        total_count_row = count_conn.execute(
            "SELECT COUNT(*) as c FROM events WHERE type IN ('email.received', 'message.received')"
        ).fetchone()
        total_count = total_count_row["c"] if total_count_row else "?"

    print(f"[backfill_linguistic_inbound] Total eligible inbound events: {total_count}")

    with db.get_connection("events") as events_conn:
        events_conn.row_factory = sqlite3.Row

        cursor = events_conn.execute(query)

        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            for row in batch:
                # Reconstruct the event dict in the same format the live pipeline uses.
                try:
                    payload_raw = row["payload"]
                    payload = json.loads(payload_raw) if payload_raw else {}
                    # Guard against double-encoded payloads.
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
                        # extract() returns signals AND writes to the linguistic_inbound
                        # profile as a side-effect for inbound events.
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        signals_extracted += 1  # Approximation for dry run

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        print(f"[backfill_linguistic_inbound] Error processing event {event['id']}: {e}")

            # Progress report every batch.
            now = time.time()
            elapsed = now - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            pct = f"{events_processed / total_count * 100:.1f}%" if isinstance(total_count, int) else "?%"
            print(
                f"[backfill_linguistic_inbound] Progress: {events_processed}/{total_count} ({pct}) — "
                f"{signals_extracted} signals, {errors} errors ({rate:.1f} events/sec)"
            )

    # Get final profile state for the summary.
    final_profile = ums.get_signal_profile("linguistic_inbound") if not dry_run else initial_profile
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

    print(f"\n[backfill_linguistic_inbound] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_linguistic_inbound] Events processed: {events_processed}")
    print(f"[backfill_linguistic_inbound] Signals extracted: {signals_extracted}")
    print(
        f"[backfill_linguistic_inbound] Profile samples: {initial_samples} → {final_samples} "
        f"(+{final_samples - initial_samples})"
    )
    print(f"[backfill_linguistic_inbound] Errors: {errors}")
    print(
        f"[backfill_linguistic_inbound] Elapsed: {elapsed_seconds:.1f}s "
        f"({events_processed / max(elapsed_seconds, 0.001):.1f} events/sec)"
    )

    if dry_run:
        print(f"[backfill_linguistic_inbound] DRY RUN — no changes written to database")

    # Show per-contact stats for verification.
    if final_profile and not dry_run:
        data = final_profile["data"]
        per_contact_avgs = data.get("per_contact_averages", {})
        if per_contact_avgs:
            print(f"\n[backfill_linguistic_inbound] ===== CONTACT STYLE SUMMARY =====")
            print(f"  Contacts with style profiles: {len(per_contact_avgs)}")
            for contact, avgs in sorted(per_contact_avgs.items())[:10]:
                formality = avgs.get("formality", "?")
                samples = avgs.get("samples_count", "?")
                print(f"  {contact}: formality={formality:.2f}, samples={samples}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill linguistic_inbound signal profile from historical events")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--batch-size", type=int, default=500, help="Events per batch (default: 500)")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing to DB")
    args = parser.parse_args()

    stats = backfill_linguistic_inbound_profile(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    sys.exit(0 if stats["errors"] == 0 else 1)
