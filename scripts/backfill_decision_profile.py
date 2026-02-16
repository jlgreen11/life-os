#!/usr/bin/env python3
"""
Backfill decision profile from historical events.

The DecisionExtractor was added in PR #152 but only processes new events
going forward. This script backfills the decision profile from all historical
decision-making events (tasks, calendar commitments, outbound messages) so
the system immediately has rich decision-making pattern data.

Why backfill matters:
- DecisionProfile is a core Layer 1 (Episodic Memory) feature that's 100% empty
- 699+ historical events (404 tasks created, 289 tasks completed, 338 emails sent,
  54 messages sent, 2,573+ calendar events created) contain rich decision signals
- Enables decision-aware predictions today (decision speed by domain, delegation
  patterns, risk tolerance, decision fatigue detection)
- Without backfill, the profile will take weeks/months to build from new events only

Impact:
- Populates decision_speed_by_domain to detect how quickly user acts on different types of tasks
- Populates delegation_comfort to identify when user seeks input vs. decides alone
- Populates risk_tolerance_by_domain from planning horizons (spontaneous vs. cautious)
- Populates fatigue_time_of_day to detect when decision quality degrades
- Enables decision-aware automation (defer complex choices during fatigue hours)

Usage:
    python scripts/backfill_decision_profile.py [--batch-size N] [--limit N]

Example:
    python scripts/backfill_decision_profile.py --batch-size 1000
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.signal_extractor.decision import DecisionExtractor
from storage.manager import DatabaseManager
from storage.user_model_store import UserModelStore


def backfill_decision_profile(
    data_dir: str = "data",
    batch_size: int = 1000,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Backfill decision profile from all historical decision-making events.

    Processes events through the DecisionExtractor to build decision-making
    patterns including speed, delegation, risk tolerance, and fatigue indicators.

    Args:
        data_dir: Path to data directory containing SQLite databases
        batch_size: Number of events to process per progress report
        limit: Maximum number of events to process (None = all)
        dry_run: If True, report what would be done without writing to DB

    Returns:
        Dict with processing statistics:
        - events_processed: Total events analyzed
        - signals_extracted: Total decision signals generated
        - profile_samples: Final sample count in decision profile
        - decision_speed_samples: Number of task completion speed measurements
        - delegation_samples: Number of delegation pattern detections
        - commitment_samples: Number of calendar commitment patterns
        - errors: Count of events that failed processing
        - elapsed_seconds: Total runtime
    """
    start_time = time.time()

    # Initialize database manager and user model store
    db = DatabaseManager(data_dir)
    ums = UserModelStore(db)

    # Initialize decision extractor
    extractor = DecisionExtractor(db, ums)

    # Query all decision-making events that trigger decision extraction
    # Order by timestamp to build the profile chronologically (earliest to latest)
    # This ensures patterns evolve naturally over time
    query = """
        SELECT id, type, source, timestamp, priority, payload, metadata
        FROM events
        WHERE type IN (
            'task.completed',
            'task.created',
            'email.sent',
            'message.sent',
            'calendar.event.created'
        )
        ORDER BY timestamp ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    # Track progress and statistics
    events_processed = 0
    signals_extracted = 0
    decision_speed_samples = 0
    delegation_samples = 0
    commitment_samples = 0
    errors = 0
    last_report_time = start_time

    with db.get_connection("events") as events_conn:
        events_conn.row_factory = sqlite3.Row

        print(f"[backfill_decision_profile] Starting backfill from {data_dir}")
        print(f"[backfill_decision_profile] Batch size: {batch_size}, Limit: {limit or 'all'}, Dry run: {dry_run}")

        # Get initial profile state
        initial_profile = ums.get_signal_profile("decision")
        initial_samples = initial_profile["samples_count"] if initial_profile else 0
        print(f"[backfill_decision_profile] Initial decision profile samples: {initial_samples}")

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
                    # Extract decision signals
                    # This both returns signals AND updates the decision profile as a side-effect
                    if not dry_run:
                        signals = extractor.extract(event)
                        signals_extracted += len(signals)

                        # Count signal types for detailed reporting
                        for signal in signals:
                            signal_type = signal.get("type")
                            if signal_type == "decision_speed":
                                decision_speed_samples += 1
                            elif signal_type == "delegation_pattern":
                                delegation_samples += 1
                            elif signal_type == "commitment_pattern":
                                commitment_samples += 1
                    else:
                        # In dry run, just count what would be processed
                        signals_extracted += 1  # Assume one signal per event

                    events_processed += 1

                except Exception as e:
                    errors += 1
                    if errors <= 10:  # Only print first 10 errors to avoid spam
                        print(f"[backfill_decision_profile] Error processing event {event['id']}: {e}")

            # Progress reporting every batch
            elapsed = time.time() - start_time
            rate = events_processed / elapsed if elapsed > 0 else 0
            print(f"[backfill_decision_profile] Progress: {events_processed} events, "
                  f"{signals_extracted} signals, {errors} errors "
                  f"({rate:.1f} events/sec)")

    # Get final profile state
    final_profile = ums.get_signal_profile("decision")
    final_samples = final_profile["samples_count"] if final_profile else 0

    elapsed_seconds = time.time() - start_time

    result = {
        "events_processed": events_processed,
        "signals_extracted": signals_extracted,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
        "samples_added": final_samples - initial_samples,
        "decision_speed_samples": decision_speed_samples,
        "delegation_samples": delegation_samples,
        "commitment_samples": commitment_samples,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }

    print(f"\n[backfill_decision_profile] ===== BACKFILL COMPLETE =====")
    print(f"[backfill_decision_profile] Events processed: {events_processed}")
    print(f"[backfill_decision_profile] Signals extracted: {signals_extracted}")
    print(f"[backfill_decision_profile]   - Decision speed samples: {decision_speed_samples}")
    print(f"[backfill_decision_profile]   - Delegation patterns: {delegation_samples}")
    print(f"[backfill_decision_profile]   - Commitment patterns: {commitment_samples}")
    print(f"[backfill_decision_profile] Profile samples: {initial_samples} → {final_samples} (+{final_samples - initial_samples})")
    print(f"[backfill_decision_profile] Errors: {errors}")
    print(f"[backfill_decision_profile] Elapsed: {elapsed_seconds:.1f}s ({events_processed/elapsed_seconds:.1f} events/sec)")

    if dry_run:
        print(f"[backfill_decision_profile] DRY RUN - no changes were written to database")

    # Show sample of the decision profile data for verification
    if final_profile and not dry_run:
        data = final_profile["data"]
        print(f"\n[backfill_decision_profile] ===== DECISION PROFILE SUMMARY =====")

        # Show decision speed by domain
        if data.get("decision_speed_by_domain"):
            speeds = data["decision_speed_by_domain"]
            print(f"[backfill_decision_profile] Decision speed by domain (seconds):")
            for domain, avg_seconds in sorted(speeds.items()):
                hours = avg_seconds / 3600
                if hours < 1:
                    print(f"  - {domain}: {avg_seconds/60:.1f} minutes")
                elif hours < 24:
                    print(f"  - {domain}: {hours:.1f} hours")
                else:
                    print(f"  - {domain}: {hours/24:.1f} days")

        # Show risk tolerance by domain
        if data.get("risk_tolerance_by_domain"):
            risk = data["risk_tolerance_by_domain"]
            print(f"[backfill_decision_profile] Risk tolerance by domain (0=cautious, 1=spontaneous):")
            for domain, score in sorted(risk.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {domain}: {score:.2f}")

        # Show decision fatigue indicators
        if data.get("fatigue_time_of_day") is not None:
            fatigue_hour = data["fatigue_time_of_day"]
            print(f"[backfill_decision_profile] Decision fatigue detected after: {fatigue_hour}:00")

        # Show delegation comfort
        if data.get("delegation_comfort") is not None:
            delegation = data["delegation_comfort"]
            print(f"[backfill_decision_profile] Delegation comfort: {delegation:.2f}")

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill decision profile from historical events"
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
        result = backfill_decision_profile(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )

        # Exit with non-zero if there were errors
        if result["errors"] > 0:
            sys.exit(1)

    except Exception as e:
        print(f"[backfill_decision_profile] FATAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
