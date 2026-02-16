#!/usr/bin/env python3
"""
Life OS — Backfill Episode Classification

Reclassifies all existing episodes with granular interaction types to enable
routine and workflow detection. This script fixes episodes that were created
before the granular classification system was deployed.

Problem:
    All 1,403 existing episodes have interaction_type = "communication" because
    they were created with the old classification logic. The routine detector
    and workflow detector require granular types (email_received, email_sent,
    meeting_scheduled, etc.) to identify behavioral patterns.

Solution:
    Read the event_id from each episode, fetch the original event from events.db,
    and apply the new _classify_interaction_type logic to determine the correct
    granular type. Update the episode in place.

Usage:
    python scripts/backfill_episode_classification.py [--dry-run]

    --dry-run: Show what would be changed without modifying the database
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def classify_interaction_type(event_type: str, payload: dict) -> str:
    """Classify event into a granular interaction type for routine detection.

    This is a copy of main.py:_classify_interaction_type() to avoid importing
    the entire LifeOS class. The logic must stay in sync with the main
    implementation.

    The routine detector relies on seeing specific, granular action types to
    identify recurring behavioral patterns. For example:
    - "email_received" vs "email_sent" reveals inbox-checking vs correspondence routines
    - "meeting_attended" vs "calendar_reviewed" distinguishes participation from planning
    - "task_created" vs "task_completed" shows work initiation vs completion patterns

    Args:
        event_type: The fine-grained event type (e.g., "email.received")
        payload: Event payload containing additional context

    Returns:
        Granular interaction type suitable for routine detection (15+ distinct types)
    """
    # Email interactions — distinguish inbound (inbox checking) from outbound (correspondence)
    if event_type == "email.received":
        return "email_received"
    elif event_type == "email.sent":
        return "email_sent"

    # Messaging interactions — distinguish chat/IM from email
    elif event_type == "message.received":
        return "message_received"
    elif event_type == "message.sent":
        return "message_sent"

    # Call interactions — distinguish answered, missed, initiated
    elif event_type == "call.received":
        return "call_answered"
    elif event_type == "call.missed":
        return "call_missed"

    # Calendar interactions — distinguish meeting participation from calendar management
    elif event_type == "calendar.event.created":
        # If the event has participants, it's a meeting; otherwise it's a personal event
        if payload.get("participants") or payload.get("attendees"):
            return "meeting_scheduled"
        else:
            return "calendar_blocked"
    elif event_type == "calendar.event.updated":
        return "calendar_reviewed"

    # Financial interactions — distinguish spending from income/transfers
    elif event_type == "finance.transaction.new":
        amount = payload.get("amount", 0)
        if amount < 0:
            return "spending"
        else:
            return "income"

    # Task interactions — distinguish creation (work planning) from completion (execution)
    elif event_type == "task.created":
        return "task_created"
    elif event_type == "task.completed":
        return "task_completed"

    # Location interactions — distinguish arrivals (entering contexts) from departures
    elif event_type == "location.arrived":
        return "location_arrived"
    elif event_type == "location.departed":
        return "location_departed"
    elif event_type == "location.changed":
        return "location_changed"

    # Context interactions — device/activity state changes
    elif event_type == "context.location":
        return "context_location"
    elif event_type == "context.activity":
        return "context_activity"

    # User commands — explicit user interactions with the system
    elif event_type == "system.user.command":
        return "user_command"

    # Fallback for any unmapped event types — should be rare
    else:
        # Try to extract a meaningful type from the event_type string
        # e.g., "system.rule.triggered" -> "rule_triggered"
        if "." in event_type:
            return event_type.split(".")[-1]
        return "other"


def backfill_episode_classification(db: DatabaseManager, dry_run: bool = False) -> dict:
    """Reclassify all episodes with granular interaction types.

    Iterates through all episodes, fetches the original event from events.db,
    applies the classification logic, and updates the interaction_type field.

    Args:
        db: DatabaseManager instance with access to events.db and user_model.db
        dry_run: If True, show what would be changed without modifying the database

    Returns:
        Dictionary with statistics:
        - total_episodes: Total episodes processed
        - reclassified: Number of episodes updated
        - unchanged: Number already correct
        - errors: Number of errors
        - type_distribution: Dict mapping interaction_type -> count
    """
    stats = {
        "total_episodes": 0,
        "reclassified": 0,
        "unchanged": 0,
        "errors": 0,
        "type_distribution": {},
    }

    # Fetch all episodes
    with db.get_connection("user_model") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, event_id, interaction_type FROM episodes ORDER BY timestamp")
        episodes = cursor.fetchall()

    stats["total_episodes"] = len(episodes)
    logger.info(f"Processing {len(episodes)} episodes...")

    for episode_id, event_id, current_interaction_type in episodes:
        try:
            # Fetch the original event from events.db
            with db.get_connection("events") as events_conn:
                events_cursor = events_conn.cursor()
                events_cursor.execute(
                    "SELECT type, payload FROM events WHERE id = ?",
                    (event_id,)
                )
                event_row = events_cursor.fetchone()

            if not event_row:
                logger.warning(f"Episode {episode_id}: event {event_id} not found in events.db")
                stats["errors"] += 1
                continue

            event_type, payload_json = event_row

            # Parse payload
            try:
                payload = json.loads(payload_json) if payload_json else {}
            except json.JSONDecodeError:
                logger.warning(f"Episode {episode_id}: invalid JSON payload for event {event_id}")
                stats["errors"] += 1
                continue

            # Apply classification logic
            new_interaction_type = classify_interaction_type(event_type, payload)

            # Track type distribution
            stats["type_distribution"][new_interaction_type] = \
                stats["type_distribution"].get(new_interaction_type, 0) + 1

            # Update if changed
            if new_interaction_type != current_interaction_type:
                if dry_run:
                    logger.debug(
                        f"Episode {episode_id}: {current_interaction_type} -> {new_interaction_type} "
                        f"(event_type={event_type})"
                    )
                else:
                    with db.get_connection("user_model") as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE episodes SET interaction_type = ? WHERE id = ?",
                            (new_interaction_type, episode_id)
                        )
                        conn.commit()
                stats["reclassified"] += 1
            else:
                stats["unchanged"] += 1

        except Exception as e:
            logger.error(f"Episode {episode_id}: error during reclassification: {e}")
            stats["errors"] += 1

    return stats


def main():
    """Run the backfill script with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Backfill episode classification with granular interaction types"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the database"
    )
    args = parser.parse_args()

    # Initialize DatabaseManager
    db = DatabaseManager()

    logger.info("Starting episode classification backfill...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")

    start_time = datetime.now(timezone.utc)

    # Run backfill
    stats = backfill_episode_classification(db, dry_run=args.dry_run)

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Print summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total episodes:     {stats['total_episodes']}")
    logger.info(f"Reclassified:       {stats['reclassified']}")
    logger.info(f"Unchanged:          {stats['unchanged']}")
    logger.info(f"Errors:             {stats['errors']}")
    logger.info(f"Elapsed time:       {elapsed:.2f}s")
    logger.info("")
    logger.info("Interaction type distribution:")
    for interaction_type, count in sorted(
        stats["type_distribution"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        logger.info(f"  {interaction_type:25s} {count:6d}")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN: No changes were made. Run without --dry-run to apply changes.")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
