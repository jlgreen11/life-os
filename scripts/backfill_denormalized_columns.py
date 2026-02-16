#!/usr/bin/env python3
"""
Backfill denormalized workflow columns for ALL historical events.

The schema migration in storage/manager.py only backfilled the most recent 10K
email.received events, but with 82K+ total emails (41K/day volume), this only
covers ~6 hours of history instead of the intended coverage. The workflow
detector requires complete denormalized data to identify patterns.

This script:
1. Backfills email_from for ALL email.received events (~82K events)
2. Backfills email_from for ALL email.sent events (~270 events)
3. Backfills email_to for ALL email.sent events (~270 events)
4. Backfills task_id for ALL task.* events (~180 events)
5. Backfills calendar_event_id for ALL calendar.event.* events (~2.5K events)

Performance: Processes events in batches of 5000 to avoid memory issues.
Estimated time: ~30 seconds for 82K emails.

Why this matters:
- Workflow detection queries use WHERE email_from = ? for efficiency
- NULL denormalized columns mean workflows can't be detected from that data
- With only 10K/82K emails backfilled (12%), workflow detection misses 88% of patterns
- Complete backfill enables Layer 3 procedural memory (0 workflows → N workflows)
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_email_received_events(conn: sqlite3.Connection) -> int:
    """Backfill email_from for all email.received events.

    Only updates events where email_from is NULL and the payload contains
    a from_address field. Uses LOWER() to normalize email addresses.

    Gracefully handles malformed JSON by using json_valid() check.

    Args:
        conn: Database connection to events.db

    Returns:
        Number of events backfilled
    """
    cursor = conn.cursor()

    # Get count of events needing backfill (skip malformed JSON)
    cursor.execute("""
        SELECT COUNT(*)
        FROM events
        WHERE type = 'email.received'
          AND email_from IS NULL
          AND json_valid(payload) = 1
          AND json_extract(payload, '$.from_address') IS NOT NULL
    """)
    total = cursor.fetchone()[0]
    logger.info(f"Found {total} email.received events needing email_from backfill")

    if total == 0:
        return 0

    # Backfill in batches of 5000 to avoid memory issues
    batch_size = 5000
    total_updated = 0

    while True:
        cursor.execute("""
            UPDATE events
            SET email_from = LOWER(json_extract(payload, '$.from_address'))
            WHERE id IN (
                SELECT id FROM events
                WHERE type = 'email.received'
                  AND email_from IS NULL
                  AND json_valid(payload) = 1
                  AND json_extract(payload, '$.from_address') IS NOT NULL
                LIMIT ?
            )
        """, (batch_size,))

        updated = cursor.rowcount
        total_updated += updated
        conn.commit()

        logger.info(f"Backfilled {total_updated}/{total} email.received events...")

        if updated < batch_size:
            break

    return total_updated


def backfill_email_sent_events(conn: sqlite3.Connection) -> tuple[int, int]:
    """Backfill email_from and email_to for all email.sent events.

    Gracefully handles malformed JSON by using json_valid() check.

    Args:
        conn: Database connection to events.db

    Returns:
        Tuple of (email_from_count, email_to_count) backfilled
    """
    cursor = conn.cursor()

    # Backfill email_from
    cursor.execute("""
        UPDATE events
        SET email_from = LOWER(json_extract(payload, '$.from_address'))
        WHERE type = 'email.sent'
          AND email_from IS NULL
          AND json_valid(payload) = 1
          AND json_extract(payload, '$.from_address') IS NOT NULL
    """)
    from_count = cursor.rowcount
    conn.commit()
    logger.info(f"Backfilled email_from for {from_count} email.sent events")

    # Backfill email_to
    cursor.execute("""
        UPDATE events
        SET email_to = LOWER(json_extract(payload, '$.to_addresses'))
        WHERE type = 'email.sent'
          AND email_to IS NULL
          AND json_valid(payload) = 1
          AND json_extract(payload, '$.to_addresses') IS NOT NULL
    """)
    to_count = cursor.rowcount
    conn.commit()
    logger.info(f"Backfilled email_to for {to_count} email.sent events")

    return (from_count, to_count)


def backfill_task_events(conn: sqlite3.Connection) -> int:
    """Backfill task_id for all task.* events.

    Gracefully handles malformed JSON by using json_valid() check.

    Args:
        conn: Database connection to events.db

    Returns:
        Number of events backfilled
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE events
        SET task_id = json_extract(payload, '$.task_id')
        WHERE type LIKE 'task.%'
          AND task_id IS NULL
          AND json_valid(payload) = 1
          AND json_extract(payload, '$.task_id') IS NOT NULL
    """)
    count = cursor.rowcount
    conn.commit()
    logger.info(f"Backfilled task_id for {count} task.* events")

    return count


def backfill_calendar_events(conn: sqlite3.Connection) -> int:
    """Backfill calendar_event_id for all calendar.event.* events.

    Gracefully handles malformed JSON by using json_valid() check.

    Args:
        conn: Database connection to events.db

    Returns:
        Number of events backfilled
    """
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE events
        SET calendar_event_id = json_extract(payload, '$.event_id')
        WHERE type LIKE 'calendar.event.%'
          AND calendar_event_id IS NULL
          AND json_valid(payload) = 1
          AND json_extract(payload, '$.event_id') IS NOT NULL
    """)
    count = cursor.rowcount
    conn.commit()
    logger.info(f"Backfilled calendar_event_id for {count} calendar.event.* events")

    return count


def main():
    """Run the complete backfill process."""
    db_path = Path(__file__).parent.parent / "data" / "events.db"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1

    logger.info(f"Starting denormalized column backfill for {db_path}")

    conn = sqlite3.connect(str(db_path))

    try:
        # Backfill email.received events
        email_received_count = backfill_email_received_events(conn)

        # Backfill email.sent events
        sent_from_count, sent_to_count = backfill_email_sent_events(conn)

        # Backfill task events
        task_count = backfill_task_events(conn)

        # Backfill calendar events
        calendar_count = backfill_calendar_events(conn)

        logger.info("Backfill complete!")
        logger.info(f"Summary:")
        logger.info(f"  email.received email_from: {email_received_count}")
        logger.info(f"  email.sent email_from: {sent_from_count}")
        logger.info(f"  email.sent email_to: {sent_to_count}")
        logger.info(f"  task.* task_id: {task_count}")
        logger.info(f"  calendar.event.* calendar_event_id: {calendar_count}")
        logger.info(f"  Total events backfilled: {email_received_count + sent_from_count + sent_to_count + task_count + calendar_count}")

        # Verify backfill completeness
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE type = 'email.received'
              AND email_from IS NULL
              AND json_valid(payload) = 1
              AND json_extract(payload, '$.from_address') IS NOT NULL
        """)
        remaining = cursor.fetchone()[0]

        if remaining > 0:
            logger.warning(f"WARNING: {remaining} email.received events still have NULL email_from")
        else:
            logger.info("✓ All email.received events successfully backfilled")

        return 0

    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
