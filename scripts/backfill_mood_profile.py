#!/usr/bin/env python3
"""
Backfill mood_signals signal profile from historical events.

The MoodInferenceEngine runs in real-time as events arrive from the NATS bus.
If the user_model.db was reset (via schema migration or rebuild) after the
Google connector synced its emails, the mood_signals profile will be empty
even though thousands of historical events are stored in events.db.

This script replays those historical events through the MoodInferenceEngine
to rebuild the mood_signals profile, enabling:

  1. compute_current_mood() for the dashboard mood widget — returns a neutral
     MoodState when no signals are available, so the widget shows no data
  2. Episode energy_level population — episodes reference mood signals to set
     their energy_level field, which drives mood-aware features system-wide
  3. Mood trend computation — requires mood_history rows which are only written
     when mood signals exist

Why backfill matters:
  - Without the mood_signals profile, compute_current_mood() always returns a
    neutral MoodState with 0.0 confidence, making the dashboard mood widget
    permanently empty after a DB rebuild.
  - The mood engine processes 10 event types (email, message, health, sleep,
    calendar, transaction, location, command) — all sitting in events.db
    but never processed after the reset.
  - Proxy energy signals (circadian_energy, communication_energy) require
    historical message data to populate, and without them episode.energy_level
    stays NULL for all episodes.

Performance note:
  - The MoodInferenceEngine is lightweight (text analysis, no LLM calls).
  - The mood_signals ring buffer caps at 200 entries, so even processing 16K+
    events only retains the most recent 200 signals. Processing is fast.

Usage:
    python scripts/backfill_mood_profile.py [--data-dir ./data]

Example:
    python scripts/backfill_mood_profile.py --batch-size 500 --dry-run
    python scripts/backfill_mood_profile.py  # Full backfill, all events
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.mood import MoodInferenceEngine
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


# All event types that MoodInferenceEngine.can_process() accepts.
MOOD_EVENT_TYPES = (
    "email.sent",
    "email.received",
    "message.sent",
    "message.received",
    "health.metric.updated",
    "sleep.recorded",
    "calendar.event.created",
    "transaction.new",
    "location.changed",
    "system.user.command",
)


def backfill_mood_profile(
    data_dir: str = "data",
    batch_size: int = 500,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill the mood_signals signal profile from all historical events.

    Replays historical events through the MoodInferenceEngine to rebuild the
    mood_signals profile (recent_signals ring buffer).  This profile is the data
    foundation for:
      - compute_current_mood() which powers the dashboard mood widget
      - Episode energy_level population via proxy energy signals
      - Mood trend computation from mood_history time-series

    The MoodInferenceEngine processes a wide variety of event types: communication
    (email/message), health metrics, sleep data, calendar events, financial
    transactions, location changes, and user commands.

    Args:
        data_dir: Path to data directory containing SQLite databases.
        batch_size: Number of events to process before reporting progress.
        limit: Maximum number of events to process. None = all events.
            Useful for testing or incremental backfill of recent events.
        dry_run: If True, report what would be done without writing to DB.

    Returns:
        Dict with processing statistics:
          - events_processed: Total events analyzed.
          - signals_extracted: Total mood signals generated.
          - initial_samples: Profile samples count before backfill.
          - final_samples: Profile samples count after backfill.
          - samples_added: Net new samples written.
          - errors: Count of events that failed processing.
          - elapsed_seconds: Total runtime.
          - dry_run: Whether this was a dry run.

    Example::

        result = backfill_mood_profile(data_dir="./data")
        print(f"Mood profile samples: {result['final_samples']}")
    """
    start_time = time.time()

    # Initialize database manager and user model store (no event bus in backfill mode —
    # telemetry events are not needed for a one-time backfill operation).
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize the mood inference engine (same instance used in the live pipeline).
    extractor = MoodInferenceEngine(db, ums)

    # Get initial profile state for comparison in the summary output.
    initial_profile = ums.get_signal_profile("mood_signals")
    initial_samples = initial_profile["samples_count"] if initial_profile else 0

    print(f"[backfill_mood] Starting mood_signals profile backfill from {data_dir}/")
    print(f"[backfill_mood] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")
    print(f"[backfill_mood] Initial state: {initial_samples} samples")

    # Build the WHERE clause from the known event types.
    type_placeholders = ", ".join(f"'{t}'" for t in MOOD_EVENT_TYPES)

    query = f"""
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN ({type_placeholders})
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
            f"SELECT COUNT(*) as c FROM events WHERE type IN ({type_placeholders})"
        ).fetchone()
        total_count = total_count_row["c"] if total_count_row else "?"

    print(f"[backfill_mood] Total eligible events: {total_count}")

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
                        # The MoodInferenceEngine calls _update_mood_state() internally,
                        # appending to the recent_signals ring buffer (capped at 200).
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)
                    else:
                        # Dry run: count what would be processed without writing.
                        signals_extracted += 1  # Approximation

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Limit error log spam for large backfills.
                        print(f"[backfill_mood] Error processing event {event['id']}: {e}")

            # Progress report every batch.
            now = time.time()
            elapsed = now - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            pct = f"{events_processed / total_count * 100:.1f}%" if isinstance(total_count, int) else "?%"
            print(
                f"[backfill_mood] Progress: {events_processed}/{total_count} ({pct}) — "
                f"{signals_extracted} signals, {errors} errors ({rate:.1f} events/sec)"
            )

    # Get final profile state for the summary.
    final_profile = ums.get_signal_profile("mood_signals") if not dry_run else initial_profile
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

    print(f"\n[backfill_mood] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_mood] Events processed: {events_processed}")
    print(f"[backfill_mood] Signals extracted: {signals_extracted}")
    print(
        f"[backfill_mood] Profile samples: {initial_samples} → {final_samples} "
        f"(+{final_samples - initial_samples})"
    )
    print(f"[backfill_mood] Errors: {errors}")
    print(
        f"[backfill_mood] Elapsed: {elapsed_seconds:.1f}s "
        f"({events_processed / max(elapsed_seconds, 0.001):.1f} events/sec)"
    )

    if dry_run:
        print(f"[backfill_mood] DRY RUN — no changes written to database")

    # Show signal type distribution for verification.
    if final_profile and not dry_run:
        recent = final_profile["data"].get("recent_signals", [])
        if recent:
            type_counts: dict[str, int] = {}
            for sig in recent:
                st = sig.get("signal_type", "unknown")
                type_counts[st] = type_counts.get(st, 0) + 1
            top_types = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)
            print(f"\n[backfill_mood] ===== SIGNAL TYPE DISTRIBUTION =====")
            for sig_type, count in top_types:
                print(f"  {sig_type}: {count}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill mood_signals signal profile from historical events")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--batch-size", type=int, default=500, help="Events per batch (default: 500)")
    parser.add_argument("--limit", type=int, default=None, help="Max events to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing to DB")
    args = parser.parse_args()

    stats = backfill_mood_profile(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    sys.exit(0 if stats["errors"] == 0 else 1)
