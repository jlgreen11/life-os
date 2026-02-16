"""
Backfill all denormalized columns in events.db to enable workflow detection.

The schema migration (v0 → v2) added denormalized columns (email_from, email_to,
task_id, calendar_event_id) but only backfilled the 10K most recent email.received
events to minimize startup time. This script completes the backfill for all 83K+
email events, enabling Layer 3 procedural memory (workflow detection) to work
across the full historical dataset.

Usage:
    python scripts/backfill_all_denormalized_columns.py

This script is safe to run multiple times - it uses WHERE clauses to only update
rows where the denormalized column is NULL, so already-backfilled rows are skipped.
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def backfill_denormalized_columns(db_path: str = "data/events.db"):
    """Backfill all denormalized columns from JSON payloads.

    This function completes the partial backfill that was done during schema
    migration. The migration only processed 10K recent email.received events
    to minimize startup time, but workflow detection needs full coverage across
    all 83K+ email events.

    Args:
        db_path: Path to events.db database file

    Returns:
        dict: Statistics about the backfill operation (rows updated per column)
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    stats = {
        "email_from_received": 0,
        "email_from_sent": 0,
        "email_to_received": 0,
        "email_to_sent": 0,
        "task_id": 0,
        "calendar_event_id": 0,
    }

    try:
        # Backfill email_from for ALL email.received events
        # The migration only did the 10K most recent; this completes the rest
        logger.info("Backfilling email_from for all email.received events...")
        cursor = conn.execute("""
            UPDATE events
            SET email_from = LOWER(json_extract(payload, '$.from_address'))
            WHERE type = 'email.received'
              AND email_from IS NULL
              AND json_extract(payload, '$.from_address') IS NOT NULL
        """)
        stats["email_from_received"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} email.received events with email_from")

        # Backfill email_from for ALL email.sent events
        logger.info("Backfilling email_from for all email.sent events...")
        cursor = conn.execute("""
            UPDATE events
            SET email_from = LOWER(json_extract(payload, '$.from_address'))
            WHERE type = 'email.sent'
              AND email_from IS NULL
              AND json_extract(payload, '$.from_address') IS NOT NULL
        """)
        stats["email_from_sent"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} email.sent events with email_from")

        # Backfill email_to for ALL email.received events
        # This enables detection of emails where the user was CC'd or BCC'd
        logger.info("Backfilling email_to for all email.received events...")
        cursor = conn.execute("""
            UPDATE events
            SET email_to = LOWER(json_extract(payload, '$.to_addresses'))
            WHERE type = 'email.received'
              AND email_to IS NULL
              AND json_extract(payload, '$.to_addresses') IS NOT NULL
        """)
        stats["email_to_received"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} email.received events with email_to")

        # Backfill email_to for ALL email.sent events
        logger.info("Backfilling email_to for all email.sent events...")
        cursor = conn.execute("""
            UPDATE events
            SET email_to = LOWER(json_extract(payload, '$.to_addresses'))
            WHERE type = 'email.sent'
              AND email_to IS NULL
              AND json_extract(payload, '$.to_addresses') IS NOT NULL
        """)
        stats["email_to_sent"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} email.sent events with email_to")

        # Backfill task_id for ALL task.* events
        logger.info("Backfilling task_id for all task events...")
        cursor = conn.execute("""
            UPDATE events
            SET task_id = json_extract(payload, '$.task_id')
            WHERE type LIKE 'task.%'
              AND task_id IS NULL
              AND json_extract(payload, '$.task_id') IS NOT NULL
        """)
        stats["task_id"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} task events with task_id")

        # Backfill calendar_event_id for ALL calendar.event.* events
        logger.info("Backfilling calendar_event_id for all calendar events...")
        cursor = conn.execute("""
            UPDATE events
            SET calendar_event_id = json_extract(payload, '$.event_id')
            WHERE type LIKE 'calendar.event.%'
              AND calendar_event_id IS NULL
              AND json_extract(payload, '$.event_id') IS NOT NULL
        """)
        stats["calendar_event_id"] = cursor.rowcount
        logger.info(f"Updated {cursor.rowcount} calendar events with calendar_event_id")

        conn.commit()

        # Report final coverage statistics
        logger.info("\nBackfill complete! Verifying coverage...")

        cursor = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN email_from IS NOT NULL THEN 1 ELSE 0 END) as has_from,
                SUM(CASE WHEN email_to IS NOT NULL THEN 1 ELSE 0 END) as has_to
            FROM events
            WHERE type IN ('email.received', 'email.sent')
        """)
        row = cursor.fetchone()
        total, has_from, has_to = row
        # SUM() returns NULL for empty result sets, so default to 0
        has_from = has_from or 0
        has_to = has_to or 0

        logger.info(f"Email events: {total}")
        if total > 0:
            logger.info(f"  with email_from: {has_from} ({100*has_from/total:.1f}%)")
            logger.info(f"  with email_to: {has_to} ({100*has_to/total:.1f}%)")
        else:
            logger.info("  No email events found")

        return stats

    except Exception as e:
        conn.rollback()
        logger.error(f"Backfill failed: {e}", exc_info=True)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logger.info("Starting denormalized column backfill...")
    stats = backfill_denormalized_columns()

    total_updated = sum(stats.values())
    logger.info(f"\nBackfill complete! Updated {total_updated} total rows:")
    for column, count in stats.items():
        logger.info(f"  {column}: {count}")

    if total_updated == 0:
        logger.info("\nAll denormalized columns are already populated. Nothing to do.")
    else:
        logger.info("\nWorkflow detection is now enabled for the full historical dataset!")
