#!/usr/bin/env python3
"""
Backfill Task Extraction Script

Processes all historical emails, messages, and calendar events to extract
action items using the AI engine. This script is needed because the task
extraction feature was added after 69K+ emails were already in the database.

The script:
1. Queries events.db for all email.received, message.received, and
   calendar.event.created events
2. For each event, calls TaskManager.process_event() to extract tasks
3. Tracks progress and provides statistics
4. Handles errors gracefully (one failure doesn't stop the entire backfill)
5. Can be run multiple times safely (idempotent - won't duplicate tasks)

Usage:
    python scripts/backfill_task_extraction.py [--limit N] [--dry-run]

Options:
    --limit N    Process only the first N events (for testing)
    --dry-run    Show what would be processed without making changes
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path so we can import Life OS modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.manager import DatabaseManager
from services.task_manager.manager import TaskManager
from services.ai_engine.engine import AIEngine
from storage.user_model_store import UserModelStore
from storage.vector_store import VectorStore
import yaml


async def backfill_tasks(db: DatabaseManager, task_manager: TaskManager,
                         limit: int = None, dry_run: bool = False):
    """
    Backfill task extraction for all historical events.

    Args:
        db: Database manager for event queries
        task_manager: Task manager with AI engine wired for extraction
        limit: Optional limit on number of events to process (for testing)
        dry_run: If True, show what would be processed without making changes

    Returns:
        dict with statistics: total_events, tasks_extracted, errors
    """
    # --- Query all actionable events from the database ---
    # These are the event types that can contain action items.
    # We order by timestamp DESC so the most recent events (most likely
    # to contain still-relevant tasks) are processed first.
    with db.get_connection("events") as conn:
        query = """
            SELECT id, type, source, timestamp, priority, payload, metadata
            FROM events
            WHERE type IN ('email.received', 'message.received', 'calendar.event.created')
            ORDER BY timestamp DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query).fetchall()

    print(f"\n{'DRY RUN: ' if dry_run else ''}Found {len(rows)} events to process")
    print(f"Event types breakdown:")

    # Count events by type for progress reporting
    type_counts = {}
    for row in rows:
        event_type = row["type"]
        type_counts[event_type] = type_counts.get(event_type, 0) + 1

    for event_type, count in sorted(type_counts.items()):
        print(f"  - {event_type}: {count}")

    if dry_run:
        print("\nDry run complete. No tasks were extracted.")
        return {"total_events": len(rows), "tasks_extracted": 0, "errors": 0}

    # --- Process each event through the task extraction pipeline ---
    tasks_extracted = 0
    errors = 0
    tasks_before = _count_tasks(db)

    print(f"\nStarting backfill... (tasks before: {tasks_before})")
    print("This may take several minutes depending on event count.\n")

    for i, row in enumerate(rows, 1):
        # Reconstruct the event dict from the database row.
        # payload and metadata are stored as JSON strings and need to be deserialized.
        import json as json_module

        payload = row["payload"]
        if isinstance(payload, str):
            payload = json_module.loads(payload)

        metadata = row["metadata"]
        if metadata and isinstance(metadata, str):
            metadata = json_module.loads(metadata)
        elif not metadata:
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

        try:
            # Call the task manager's extraction logic. This is the same
            # code path that runs for new events in the live pipeline.
            await task_manager.process_event(event)

            # Progress reporting every 100 events to show we're alive
            if i % 100 == 0:
                tasks_now = _count_tasks(db)
                tasks_extracted = tasks_now - tasks_before
                print(f"Progress: {i}/{len(rows)} events processed, "
                      f"{tasks_extracted} tasks extracted so far")

        except Exception as e:
            # Fail-open: log the error but continue processing remaining events.
            # Task extraction is nice-to-have, not mission-critical.
            errors += 1
            if errors <= 5:  # Only print first 5 errors to avoid spam
                print(f"Error processing event {event['id']}: {e}")

    # --- Final statistics ---
    tasks_after = _count_tasks(db)
    tasks_extracted = tasks_after - tasks_before

    print(f"\nBackfill complete!")
    print(f"  Events processed: {len(rows)}")
    print(f"  Tasks extracted: {tasks_extracted}")
    print(f"  Errors: {errors}")
    print(f"  Tasks before: {tasks_before}")
    print(f"  Tasks after: {tasks_after}")

    return {
        "total_events": len(rows),
        "tasks_extracted": tasks_extracted,
        "errors": errors,
    }


def _count_tasks(db: DatabaseManager) -> int:
    """
    Count total tasks in the database.

    Args:
        db: Database manager

    Returns:
        Number of tasks in the tasks table
    """
    with db.get_connection("state") as conn:
        result = conn.execute("SELECT COUNT(*) as count FROM tasks").fetchone()
        return result["count"]


async def main():
    """
    Main entry point for the backfill script.

    Parses command-line arguments, initializes the database and services,
    runs the backfill, and reports statistics.
    """
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Backfill task extraction for historical events"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N events (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    args = parser.parse_args()

    # --- Load config from settings.yaml ---
    # We need the AI config (Ollama URL, model name) to initialize the AI engine.
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_dir = Path(__file__).parent.parent / "data"

    # --- Initialize core services ---
    # We only need the minimal set of dependencies required for task extraction:
    # DatabaseManager, UserModelStore (for AI context), VectorStore (for semantic
    # search in AI engine), AIEngine, and TaskManager.
    print("Initializing services...")
    db = DatabaseManager(data_dir)
    user_model_store = UserModelStore(db)

    # Initialize vector store (required by AI engine for semantic search)
    vector_store = VectorStore(
        db,
        data_dir,
        config.get("vector_store", {})
    )

    # Initialize AI engine with the config (Ollama URL, model name)
    ai_engine = AIEngine(
        db,
        user_model_store,
        config.get("ai", {}),
        vector_store=vector_store
    )

    # Initialize task manager with AI engine (no event bus needed for backfill)
    task_manager = TaskManager(db, event_bus=None, ai_engine=ai_engine)

    # --- Run the backfill ---
    start_time = datetime.now(timezone.utc)
    stats = await backfill_tasks(db, task_manager, args.limit, args.dry_run)
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print(f"\nTotal runtime: {duration:.1f}s")
    if stats["total_events"] > 0 and not args.dry_run:
        print(f"Average: {duration / stats['total_events']:.2f}s per event")


if __name__ == "__main__":
    asyncio.run(main())
