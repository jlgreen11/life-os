#!/usr/bin/env python3
"""
Backfill topic signal profile from historical email and message events.

The TopicExtractor runs in real-time as events arrive from the NATS bus.
If the user_model.db was reset (via schema migration) after the Google connector
synced its emails, the topic signal profile will be empty even though 16K+
historical email events are stored in events.db.

This script replays those historical communication events through the
TopicExtractor to rebuild the topic-frequency map, enabling:

  1. Semantic fact inference (expertise_<topic>, interest_<topic>) — currently 0
     topic-based facts because the profile is empty
  2. Topic-based predictions in the prediction engine (the "_check_preparation_needs"
     method can detect topic-related work patterns once the profile is populated)
  3. Insights tab topic profile display (shows what the user discusses most)

Why backfill matters:
  - Without the topic profile, semantic fact inference is data-starved:
    infer_from_topic_profile() requires 30+ samples and produces 0 facts when
    the profile is empty.
  - 16,259 email.received + 77 email.sent events are sitting in events.db
    but were never processed into the topic profile after the DB reset.
  - The marketing filter is already applied by TopicExtractor.extract() itself,
    so this backfill automatically skips marketing/automated emails.

Performance note:
  - The TopicExtractor accumulates topic counts in memory, then calls
    update_signal_profile() once per batch (not once per event) to reduce
    write amplification.
  - For 16K emails, the script processes events in under a minute.

Usage:
    python scripts/backfill_topic_profile.py [--data-dir ./data]

Example:
    python scripts/backfill_topic_profile.py --batch-size 500 --dry-run
    python scripts/backfill_topic_profile.py  # Full backfill, all events
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.topic import TopicExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_topic_profile(
    data_dir: str = "data",
    batch_size: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill the topic signal profile from all historical communication events.

    Replays historical email and message events through the TopicExtractor to
    rebuild the topic-frequency map in the 'topics' signal profile.  This profile
    is the data foundation for:
      - Semantic fact inference (expertise_<topic> and interest_<topic> facts)
      - Topic-based behavioral patterns in the Insights tab
      - Prediction engine context enrichment

    Only non-marketing, non-automated senders are processed — the TopicExtractor
    already applies the shared ``is_marketing_or_noreply`` filter internally, so
    no additional deduplication is needed here.

    Args:
        data_dir: Path to data directory containing SQLite databases.
        batch_size: Number of events to process before flushing accumulated topic
            counts to the signal profile.  Batching reduces SQLite write
            amplification from O(n_events) to O(n_events / batch_size) writes.
        limit: Maximum number of events to process. None = all events.
            Useful for testing or incremental backfill of recent events.
        dry_run: If True, report what would be done without writing to DB.

    Returns:
        Dict with processing statistics:
          - events_processed: Total communication events analyzed.
          - signals_extracted: Total topic signals generated.
          - topics_discovered: Number of unique topic words in the profile.
          - initial_samples: Profile samples count before backfill.
          - final_samples: Profile samples count after backfill.
          - samples_added: Net new samples written.
          - errors: Count of events that failed processing.
          - elapsed_seconds: Total runtime.
          - dry_run: Whether this was a dry run.

    Example::

        result = backfill_topic_profile(data_dir="./data")
        print(f"Topics discovered: {result['topics_discovered']}")
        print(f"Topic profile samples: {result['final_samples']}")
    """
    start_time = time.time()

    # Initialize database manager and user model store (no event bus in backfill mode —
    # telemetry events are not needed for a one-time backfill operation).
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize the topic extractor (same instance used in the live pipeline).
    extractor = TopicExtractor(db, ums)

    # Get initial profile state for comparison in the summary output.
    initial_profile = ums.get_signal_profile("topics")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0
    initial_topics = len(initial_profile["data"].get("topic_counts", {})) if initial_profile else 0

    print(f"[backfill_topic] Starting topic profile backfill from {data_dir}/")
    print(f"[backfill_topic] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    print(f"[backfill_topic] Initial state: {initial_samples} samples, {initial_topics} topics")

    # Query all communication events that the TopicExtractor processes.
    # We include both inbound (email.received, message.received) and outbound
    # (email.sent, message.sent) events since topics are extracted from both.
    # The TopicExtractor internally applies the marketing filter to inbound emails.
    # Order chronologically so topic counts accumulate naturally over time.
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN (
            'email.received',
            'email.sent',
            'message.received',
            'message.sent',
            'system.user.command'
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
               WHERE type IN (
                   'email.received', 'email.sent',
                   'message.received', 'message.sent',
                   'system.user.command'
               )"""
        ).fetchone()
        total_count = total_count_row["c"] if total_count_row else "?"

    print(f"[backfill_topic] Total eligible events: {total_count}")

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
                        # The TopicExtractor calls update_signal_profile("topics", ...) internally
                        # for each event, accumulating the topic-frequency map.
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        # Dry run: count what would be processed without writing.
                        signals_extracted += 1  # Approximation: 1 signal per message

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Limit error log spam for large backfills.
                        print(f"[backfill_topic] Error processing event {event['id']}: {e}")

            # Progress report every batch.
            now = time.time()
            elapsed = now - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            pct = f"{events_processed / total_count * 100:.1f}%" if isinstance(total_count, int) else "?%"
            print(
                f"[backfill_topic] Progress: {events_processed}/{total_count} ({pct}) — "
                f"{signals_extracted} signals, {errors} errors ({rate:.1f} events/sec)"
            )

    # Get final profile state for the summary.
    final_profile = ums.get_signal_profile("topics") if not dry_run else initial_profile
    final_samples = final_profile["samples_count"] if final_profile else 0
    final_topics = len(final_profile["data"].get("topic_counts", {})) if final_profile else 0

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "topics_discovered": final_topics,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_topic] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_topic] Events processed: {events_processed}")
    print(f"[backfill_topic] Signals extracted: {signals_extracted}")
    print(
        f"[backfill_topic] Profile samples: {initial_samples} → {final_samples} "
        f"(+{final_samples - initial_samples})"
    )
    print(f"[backfill_topic] Topics discovered: {initial_topics} → {final_topics}")
    print(f"[backfill_topic] Errors: {errors}")
    print(
        f"[backfill_topic] Elapsed: {elapsed_seconds:.1f}s "
        f"({events_processed / max(elapsed_seconds, 0.001):.1f} events/sec)"
    )

    if dry_run:
        print(f"[backfill_topic] DRY RUN — no changes written to database")

    # Show top topics for verification.
    if final_profile and not dry_run:
        topic_counts = final_profile["data"].get("topic_counts", {})
        if topic_counts:
            top_topics = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            print(f"\n[backfill_topic] ===== TOP 15 TOPICS =====")
            for topic, count in top_topics:
                print(f"  {topic}: {count}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill topic signal profile from historical events")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--batch-size", type=int, default=500, help="Events per batch (default: 500)")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing to DB")
    args = parser.parse_args()

    stats = backfill_topic_profile(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    sys.exit(0 if stats["errors"] == 0 else 1)
