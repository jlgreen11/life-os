#!/usr/bin/env python3
"""
Life OS — Backfill Episodes from Events

Creates episodic memory entries from events.db for events that don't yet have
corresponding episodes in user_model.db. This is essential when episodes were
lost (e.g., during a user_model.db rebuild) or never created (e.g., events
ingested while NATS was disconnected).

Episodes are the foundation of the cognitive model — they feed routine detection,
semantic fact inference, and all dashboard intelligence metrics. Without episodes,
the system reports "0 episodes, 0 facts, 0 routines".

The existing episode backfill scripts (classification, timestamps, energy levels)
all assume episodes already exist. This script fills the gap by CREATING episodes
from scratch using events.db as the source of truth.

Usage:
    python scripts/backfill_episodes_from_events.py [--dry-run] [--limit N] [--batch-size N]

    --dry-run       Show what would be created without modifying the database
    --limit N       Only process the first N eligible events (default: all)
    --batch-size N  Commit after every N episodes (default: 500)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
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

# Event types that produce episodic memory — mirrors main.py:_create_episode()
EPISODIC_EVENT_TYPES = frozenset({
    "email.received", "email.sent",
    "message.received", "message.sent",
    "call.received", "call.missed",
    "calendar.event.created", "calendar.event.updated",
    "finance.transaction.new",
    "task.created", "task.completed",
    "location.changed", "location.arrived", "location.departed",
    "context.location", "context.activity",
    "system.user.command",
})

# Payload fields that contain large body content — stripped to a short snippet
# to keep episode content_full compact (mirrors main.py lines 1846-1872)
_LARGE_FIELDS = frozenset(
    {"body", "html_body", "raw", "raw_mime", "text_body", "html", "content"}
)
_SNIPPET_CHARS = 500
_MAX_TOTAL_CHARS = 4_000


def classify_interaction_type(event_type: str, payload: dict) -> str:
    """Classify event into a granular interaction type for routine detection.

    This is a standalone copy of main.py:_classify_interaction_type() to avoid
    importing the entire LifeOS class. The logic must stay in sync with the
    main implementation. See also scripts/backfill_episode_classification.py
    which uses the same approach.

    Args:
        event_type: The fine-grained event type (e.g., "email.received")
        payload: Event payload containing additional context

    Returns:
        Granular interaction type suitable for routine detection
    """
    if event_type == "email.received":
        return "email_received"
    elif event_type == "email.sent":
        return "email_sent"
    elif event_type == "message.received":
        return "message_received"
    elif event_type == "message.sent":
        return "message_sent"
    elif event_type == "call.received":
        return "call_answered"
    elif event_type == "call.missed":
        return "call_missed"
    elif event_type == "calendar.event.created":
        if payload.get("participants") or payload.get("attendees"):
            return "meeting_scheduled"
        else:
            return "calendar_blocked"
    elif event_type == "calendar.event.updated":
        return "calendar_reviewed"
    elif event_type == "finance.transaction.new":
        amount = payload.get("amount", 0)
        return "spending" if amount < 0 else "income"
    elif event_type == "task.created":
        return "task_created"
    elif event_type == "task.completed":
        return "task_completed"
    elif event_type == "location.arrived":
        return "location_arrived"
    elif event_type == "location.departed":
        return "location_departed"
    elif event_type == "location.changed":
        return "location_changed"
    elif event_type == "context.location":
        return "context_location"
    elif event_type == "context.activity":
        return "context_activity"
    elif event_type == "system.user.command":
        return "user_command"
    else:
        if "." in event_type:
            return event_type.split(".")[-1]
        return "other"


def generate_episode_summary(event_type: str, payload: dict) -> str:
    """Generate a concise (< 200 char) summary for an episode.

    Standalone copy of main.py:_generate_episode_summary() adapted to accept
    event_type and payload directly instead of an event dict.

    Args:
        event_type: The event type string
        payload: The deserialized event payload

    Returns:
        Human-readable summary suitable for timeline displays
    """
    if event_type == "email.received":
        from_addr = payload.get("from_address", "unknown")
        subject = payload.get("subject", "No subject")
        return f"Email from {from_addr}: {subject}"[:200]
    elif event_type == "email.sent":
        to_addrs = payload.get("to_addresses", [])
        to_str = ", ".join(to_addrs[:2]) if to_addrs else "unknown"
        subject = payload.get("subject", "No subject")
        return f"Email to {to_str}: {subject}"[:200]
    elif event_type == "message.received":
        from_addr = payload.get("from_address", "unknown")
        snippet = payload.get("snippet", payload.get("body_plain", ""))[:50]
        return f"Message from {from_addr}: {snippet}"[:200]
    elif event_type == "message.sent":
        to_addrs = payload.get("to_addresses", [])
        to_str = ", ".join(to_addrs[:2]) if to_addrs else "unknown"
        snippet = payload.get("snippet", payload.get("body_plain", ""))[:50]
        return f"Message to {to_str}: {snippet}"[:200]
    elif "call" in event_type:
        from_addr = payload.get("from_address", "unknown")
        return f"Call from {from_addr}"[:200]
    elif "calendar" in event_type:
        title = payload.get("title", "Untitled event")
        start_time = payload.get("start_time", "")
        return f"Meeting: {title} at {start_time}"[:200]
    elif "task" in event_type:
        title = payload.get("title", "Untitled task")
        status = "completed" if event_type == "task.completed" else "created"
        return f"Task {status}: {title}"[:200]
    elif "finance.transaction" in event_type:
        amount = payload.get("amount", 0)
        merchant = payload.get("merchant", "Unknown")
        return f"Transaction: ${amount:.2f} at {merchant}"[:200]
    elif "location" in event_type:
        location = payload.get("location", "Unknown location")
        action = (
            "arrived at" if "arrived" in event_type
            else "departed from" if "departed" in event_type
            else "changed to"
        )
        return f"Location {action} {location}"[:200]
    elif event_type == "system.user.command":
        command = payload.get("command", "Unknown command")
        return f"Command: {command}"[:200]
    else:
        snippet = payload.get("snippet", payload.get("subject", payload.get("title", "")))
        return f"{event_type}: {snippet}"[:200]


def build_compact_content(payload: dict) -> str:
    """Build a compact JSON representation of the payload for episode storage.

    Strips large body fields and truncates remaining strings to keep the
    episode content_full under 4000 chars. Mirrors main.py lines 1846-1872.

    Args:
        payload: The deserialized event payload

    Returns:
        JSON string capped at _MAX_TOTAL_CHARS characters
    """
    compact: dict = {}
    for k, v in payload.items():
        if k in _LARGE_FIELDS:
            # Replace bulky body fields with a short snippet
            if isinstance(v, str) and v:
                compact[k] = v[:_SNIPPET_CHARS] + ("…" if len(v) > _SNIPPET_CHARS else "")
        elif isinstance(v, str) and len(v) > _SNIPPET_CHARS:
            compact[k] = v[:_SNIPPET_CHARS] + "…"
        else:
            compact[k] = v

    result = json.dumps(compact)
    if len(result) > _MAX_TOTAL_CHARS:
        result = result[:_MAX_TOTAL_CHARS]
    return result


def extract_actual_timestamp(payload: dict, event_timestamp: str) -> str:
    """Return the best available actual-event timestamp from a payload.

    Uses the same priority chain as main.py:_create_episode() (lines 1927-1934)
    so the backfill produces the same timestamps as live episode creation.

    Args:
        payload: The deserialized event payload
        event_timestamp: The event's sync timestamp (last resort fallback)

    Returns:
        The best available timestamp string
    """
    return (
        payload.get("email_date")       # Google/Proton mail — actual Date header
        or payload.get("sent_at")       # iMessage, Signal — message send time
        or payload.get("received_at")   # some connectors — arrival time
        or payload.get("date")          # generic fallback for older connectors
        or payload.get("start_time")    # Calendar: actual event start
        or event_timestamp              # Last resort: sync timestamp
    )


def backfill_episodes(
    db: DatabaseManager,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = 500,
) -> dict:
    """Create episodes from events.db for events that lack corresponding episodes.

    Scans events.db for episodic event types, checks for existing episodes in
    user_model.db, and creates new episodes for any gaps. The operation is
    idempotent — running it multiple times will not create duplicates.

    After each batch write, the actual row count is verified in the database.
    If INSERT OR IGNORE silently drops rows (e.g. due to constraint violations
    or WAL corruption), a CRITICAL warning is logged and the discrepancy is
    captured in the returned ``episodes_verified`` stat.

    Args:
        db: DatabaseManager instance with access to events.db and user_model.db
        dry_run: If True, report what would be created without writing
        limit: Maximum number of events to process (None = all)
        batch_size: Number of episodes to commit per batch

    Returns:
        Dictionary with statistics about the backfill run. Includes:
          - ``episodes_created``: number of episodes passed to the write layer
          - ``episodes_verified``: number confirmed present after each batch commit
            (0 in dry-run mode; a mismatch indicates silent data loss)
    """
    stats = {
        "total_events_scanned": 0,
        "episodes_created": 0,
        "episodes_verified": 0,
        "episodes_skipped_existing": 0,
        "events_skipped_non_episodic": 0,
        "errors": 0,
        "type_distribution": {},
    }

    # Build the SQL query with placeholders for episodic event types
    placeholders = ", ".join("?" for _ in EPISODIC_EVENT_TYPES)
    query = f"SELECT id, type, source, timestamp, payload, metadata FROM events WHERE type IN ({placeholders}) ORDER BY timestamp ASC"
    params = list(EPISODIC_EVENT_TYPES)

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    # Fetch all qualifying events from events.db
    with db.get_connection("events") as events_conn:
        events_conn.row_factory = _dict_factory
        cursor = events_conn.cursor()
        cursor.execute(query, params)
        events = cursor.fetchall()

    stats["total_events_scanned"] = len(events)
    logger.info("Found %d episodic events in events.db", len(events))

    # Load the set of event_ids that already have episodes — used for
    # idempotency checks so we never create duplicate episodes.
    with db.get_connection("user_model") as um_conn:
        cursor = um_conn.cursor()
        cursor.execute("SELECT event_id FROM episodes WHERE event_id IS NOT NULL")
        existing_event_ids = {row[0] for row in cursor.fetchall()}

    logger.info("Found %d existing episodes in user_model.db", len(existing_event_ids))

    # Process events in batches
    pending_episodes = []

    for i, event_row in enumerate(events):
        event_id = event_row["id"]
        event_type = event_row["type"]

        # Skip if an episode already exists for this event (idempotent)
        if event_id in existing_event_ids:
            stats["episodes_skipped_existing"] += 1
            continue

        try:
            # Parse the payload JSON
            payload_raw = event_row["payload"]
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
                # Handle double-encoded payloads (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Event %s: invalid JSON payload, skipping", event_id)
                stats["errors"] += 1
                continue

            # Parse metadata
            metadata_raw = event_row["metadata"]
            try:
                metadata = json.loads(metadata_raw) if metadata_raw else {}
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            # Classify interaction type
            interaction_type = classify_interaction_type(event_type, payload)

            # Track type distribution
            stats["type_distribution"][interaction_type] = (
                stats["type_distribution"].get(interaction_type, 0) + 1
            )

            # Extract contacts involved
            contacts_involved = []
            if payload.get("from_address"):
                contacts_involved.append(payload["from_address"])
            if payload.get("to_addresses"):
                contacts_involved.extend(payload["to_addresses"])

            # Extract topics from payload metadata
            topics = payload.get("topics", [])

            # Generate episode summary
            content_summary = generate_episode_summary(event_type, payload)

            # Build compact content_full
            content_full = build_compact_content(payload)

            # Determine the actual event timestamp (not sync timestamp)
            actual_timestamp = extract_actual_timestamp(
                payload, event_row["timestamp"]
            )

            # Determine active domain from event metadata
            active_domain = metadata.get("domain", "personal")

            # Build the episode dict
            episode = {
                "id": str(uuid.uuid4()),
                "timestamp": actual_timestamp,
                "event_id": event_id,
                "location": payload.get("location"),
                "inferred_mood": None,  # Not available retroactively
                "active_domain": active_domain,
                "energy_level": None,  # Filled by backfill_episode_energy_levels.py
                "interaction_type": interaction_type,
                "content_summary": content_summary,
                "content_full": content_full,
                "contacts_involved": contacts_involved,
                "topics": topics,
                "entities": payload.get("entities", []),
                "outcome": None,
                "user_satisfaction": None,
                "embedding_id": None,
            }

            pending_episodes.append(episode)

            # Commit in batches for performance
            if len(pending_episodes) >= batch_size:
                stats["episodes_created"] += len(pending_episodes)
                if not dry_run:
                    batch_verified = _write_episode_batch(db, pending_episodes)
                    stats["episodes_verified"] += batch_verified
                logger.info(
                    "Progress: %d/%d events processed, %d episodes created, %d verified",
                    i + 1, len(events), stats["episodes_created"], stats["episodes_verified"],
                )
                pending_episodes = []

        except Exception as e:
            logger.error("Event %s: error creating episode: %s", event_id, e)
            stats["errors"] += 1

    # Write any remaining episodes
    if pending_episodes:
        stats["episodes_created"] += len(pending_episodes)
        if not dry_run:
            batch_verified = _write_episode_batch(db, pending_episodes)
            stats["episodes_verified"] += batch_verified

    # Log a summary warning if the verified count doesn't match what was written.
    # This catches silent data loss (e.g. INSERT OR IGNORE dropping all rows due
    # to an unexpected constraint conflict or WAL frame corruption).
    if not dry_run and stats["episodes_verified"] != stats["episodes_created"]:
        logger.critical(
            "Episode backfill verification mismatch: %d episodes written but only "
            "%d confirmed present. Some episode data may have been silently lost. "
            "Run again or inspect user_model.db manually.",
            stats["episodes_created"],
            stats["episodes_verified"],
        )

    return stats


def _write_episode_batch(db: DatabaseManager, episodes: list[dict]) -> int:
    """Write a batch of episodes to user_model.db in a single transaction.

    Uses executemany for efficiency when inserting multiple rows. After the
    commit, issues a PASSIVE WAL checkpoint to flush written frames into the
    main database file — without this, a process crash after commit could lose
    data that is still only in the WAL. Finally, performs a post-write
    verification count to detect silent INSERT OR IGNORE drops.

    INSERT OR IGNORE succeeds without raising an error even if all rows are
    dropped (e.g. because of a duplicate ``id`` or ``event_id`` constraint
    violation). The verification query catches this case and logs a CRITICAL
    warning so operators know data was lost.

    Args:
        db: DatabaseManager instance
        episodes: List of episode dicts to insert

    Returns:
        The number of episode IDs from this batch confirmed present in the
        database after the commit. A value less than ``len(episodes)`` means
        some rows were silently dropped by INSERT OR IGNORE.
    """
    episode_ids = [ep["id"] for ep in episodes]

    with db.get_connection("user_model") as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO episodes
               (id, timestamp, event_id, location, inferred_mood, active_domain,
                energy_level, interaction_type, content_summary, content_full,
                contacts_involved, topics, entities, outcome, user_satisfaction, embedding_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ep["id"],
                    ep["timestamp"],
                    ep["event_id"],
                    ep.get("location"),
                    json.dumps(ep.get("inferred_mood") or {}),
                    ep.get("active_domain"),
                    ep.get("energy_level"),
                    ep.get("interaction_type", "unknown"),
                    ep.get("content_summary", ""),
                    ep.get("content_full"),
                    json.dumps(ep.get("contacts_involved") or []),
                    json.dumps(ep.get("topics") or []),
                    json.dumps(ep.get("entities") or []),
                    ep.get("outcome"),
                    ep.get("user_satisfaction"),
                    ep.get("embedding_id"),
                )
                for ep in episodes
            ],
        )
        conn.commit()

        # Flush WAL frames into the main DB file. A PASSIVE checkpoint does not
        # block concurrent readers or writers but ensures committed data is
        # durable on disk. Without this, a crash between commit and the OS
        # flushing the WAL to disk can silently lose the entire batch.
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        # Post-write verification: count how many of the batch's episode IDs
        # are actually present after the commit. INSERT OR IGNORE is silent on
        # constraint conflicts — it returns success even if 0 rows landed.
        placeholders = ", ".join("?" for _ in episode_ids)
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM episodes WHERE id IN ({placeholders})",
            episode_ids,
        )
        verified_count: int = cursor.fetchone()[0]

    if verified_count != len(episodes):
        logger.critical(
            "Post-write verification FAILED for batch of %d episodes: only %d "
            "are present in user_model.db after commit and WAL checkpoint. "
            "Possible causes: duplicate IDs, constraint violations, or WAL "
            "corruption. Inspect user_model.db and re-run the backfill.",
            len(episodes),
            verified_count,
        )

    return verified_count


def _dict_factory(cursor, row):
    """SQLite row factory that returns dicts keyed by column name."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def main():
    """CLI entry point for the episode backfill script."""
    parser = argparse.ArgumentParser(
        description="Backfill episodes from events.db into user_model.db"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without modifying the database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N eligible events (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Commit after every N episodes (default: 500)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the SQLite databases (default: data/)",
    )
    args = parser.parse_args()

    # Initialize DatabaseManager
    db = DatabaseManager(data_dir=args.data_dir)

    logger.info("Starting episode backfill from events.db...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")

    start_time = datetime.now(timezone.utc)

    # Run backfill
    stats = backfill_episodes(
        db,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Print summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total events scanned:       {stats['total_events_scanned']}")
    logger.info(f"Episodes created:           {stats['episodes_created']}")
    logger.info(f"Episodes verified:          {stats['episodes_verified']}")
    if not args.dry_run and stats["episodes_verified"] != stats["episodes_created"]:
        logger.critical(
            "VERIFICATION MISMATCH: %d written vs %d verified — check user_model.db",
            stats["episodes_created"],
            stats["episodes_verified"],
        )
    logger.info(f"Skipped (already exist):    {stats['episodes_skipped_existing']}")
    logger.info(f"Skipped (non-episodic):     {stats['events_skipped_non_episodic']}")
    logger.info(f"Errors:                     {stats['errors']}")
    logger.info(f"Elapsed time:               {elapsed:.2f}s")
    logger.info("")
    logger.info("Interaction type distribution:")
    for interaction_type, count in sorted(
        stats["type_distribution"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        logger.info(f"  {interaction_type:25s} {count:6d}")

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN: No changes were made. Run without --dry-run to apply changes.")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
