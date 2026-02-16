#!/usr/bin/env python3
"""
Backfill calendar conflict detection for historical calendar events.

The CalDAVConnector._detect_conflicts() method only runs during active sync
cycles. This script retroactively detects conflicts across all historical
calendar events in the database, publishing calendar.conflict.detected events
for any overlapping meetings that were missed.

Usage:
    python scripts/backfill_calendar_conflicts.py

Expected impact:
    - Analyzes 2,500+ historical calendar.event.created events
    - Publishes calendar.conflict.detected events for any overlaps
    - Enables the "High priority: calendar conflict" default rule retroactively
    - No user data is modified; only new events are created
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.database import DatabaseManager
from storage.event_store import EventStore
from services.event_bus.bus import EventBus


async def detect_conflicts_for_all_events(
    event_store: EventStore,
    event_bus: EventBus,
    dry_run: bool = False
) -> dict[str, int]:
    """
    Detect conflicts across all historical calendar events.

    Implements the same sweep-line algorithm as CalDAVConnector._detect_conflicts(),
    but operates on the full historical event dataset rather than just the last
    24 hours.

    Algorithm:
        1. Fetch ALL calendar.event.created events (no time filter)
        2. Parse start_time and end_time from each payload
        3. Sort by start_time
        4. Use sweep-line to detect overlaps: for each event, check if it
           overlaps with any subsequent event
        5. Two events overlap if: start1 < end2 AND start2 < end1
        6. Publish calendar.conflict.detected for each overlap

    Args:
        event_store: Database handle for reading calendar events
        event_bus: Event bus for publishing conflict events
        dry_run: If True, report conflicts without publishing events

    Returns:
        Statistics dict with:
            - events_analyzed: Total calendar events processed
            - conflicts_detected: Number of overlapping event pairs found
            - events_published: Number of conflict events published
    """
    stats = {
        "events_analyzed": 0,
        "conflicts_detected": 0,
        "events_published": 0,
        "parse_errors": 0,
    }

    # Fetch all calendar.event.created events (no time limit)
    print("Fetching historical calendar events...")
    calendar_events = event_store.get_events(
        event_type="calendar.event.created",
        limit=10000  # Generous upper bound
    )

    stats["events_analyzed"] = len(calendar_events)
    print(f"Found {stats['events_analyzed']} calendar events")

    if len(calendar_events) < 2:
        print("Not enough events to detect conflicts")
        return stats

    # Parse event times and build sortable list
    # Each entry: (start_dt, end_dt, original_event_dict, parsed_payload)
    parsed_events = []

    for evt in calendar_events:
        try:
            # Payload is stored as JSON string in the database
            raw_payload = evt["payload"]
            payload = json.loads(raw_payload)

            # Handle double-encoding (payload stored as JSON string of JSON string)
            if isinstance(payload, str):
                payload = json.loads(payload)

            # Extract ISO timestamps from payload
            start_str = payload.get("start_time")
            end_str = payload.get("end_time")

            if not start_str or not end_str:
                continue  # Skip events without time bounds

            # Parse ISO timestamps
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)

            # Skip all-day events — they don't cause traditional scheduling conflicts
            # (multiple all-day markers can coexist without issue)
            if payload.get("is_all_day"):
                continue

            parsed_events.append((start_dt, end_dt, evt, payload))

        except Exception as e:
            # Log but continue — don't let one malformed event block the rest
            print(f"Parse error for event {evt.get('id', 'unknown')}: {e}")
            stats["parse_errors"] += 1
            continue

    print(f"Successfully parsed {len(parsed_events)} timed events (skipped all-day and malformed)")

    if len(parsed_events) < 2:
        print("Not enough timed events to detect conflicts")
        return stats

    # Sort by start time (earliest first)
    parsed_events.sort(key=lambda x: x[0])

    # Sweep-line conflict detection: compare each event with subsequent events
    conflicts_detected = set()  # Track (id1, id2) pairs to avoid duplicates

    print("Running sweep-line conflict detection...")

    for i in range(len(parsed_events)):
        start1, end1, evt1, payload1 = parsed_events[i]

        # Check all subsequent events that could overlap
        for j in range(i + 1, len(parsed_events)):
            start2, end2, evt2, payload2 = parsed_events[j]

            # If the second event starts at or after the first one ends,
            # no overlap is possible (list is sorted, so no need to check further)
            if start2 >= end1:
                break

            # Overlap condition: start1 < end2 AND start2 < end1
            # We already know start2 < end1 (from the break condition above),
            # so we just verify start1 < end2
            if start1 < end2:
                # Conflict detected!
                event_pair = tuple(sorted([evt1["id"], evt2["id"]]))

                if event_pair not in conflicts_detected:
                    conflicts_detected.add(event_pair)
                    stats["conflicts_detected"] += 1

                    # Build conflict event payload matching CalDAVConnector format
                    conflict_payload = {
                        "event1": {
                            "id": payload1.get("event_id"),
                            "title": payload1.get("title"),
                            "start_time": start1.isoformat(),
                            "end_time": end1.isoformat(),
                            "calendar_id": payload1.get("calendar_id"),
                            "location": payload1.get("location"),
                        },
                        "event2": {
                            "id": payload2.get("event_id"),
                            "title": payload2.get("title"),
                            "start_time": start2.isoformat(),
                            "end_time": end2.isoformat(),
                            "calendar_id": payload2.get("calendar_id"),
                            "location": payload2.get("location"),
                        },
                        "overlap_start": max(start1, start2).isoformat(),
                        "overlap_end": min(end1, end2).isoformat(),
                    }

                    if dry_run:
                        print(f"[DRY RUN] Conflict: '{payload1.get('title')}' overlaps '{payload2.get('title')}'")
                        print(f"  Event 1: {start1.isoformat()} - {end1.isoformat()}")
                        print(f"  Event 2: {start2.isoformat()} - {end2.isoformat()}")
                        print(f"  Overlap: {conflict_payload['overlap_start']} - {conflict_payload['overlap_end']}")
                    else:
                        # Publish conflict event to the event bus
                        await event_bus.publish(
                            "calendar.conflict.detected",
                            conflict_payload,
                            source="backfill_calendar_conflicts",
                            priority=7,  # High priority — matches default rule expectations
                        )
                        stats["events_published"] += 1

    return stats


async def main():
    """
    Main entry point for backfill script.

    Initializes database and event bus, runs conflict detection across all
    historical calendar events, and reports results.
    """
    print("=" * 80)
    print("Calendar Conflict Detection Backfill")
    print("=" * 80)
    print()

    # Initialize database and event bus
    db = DatabaseManager()
    event_store = EventStore(db)
    event_bus = EventBus()

    try:
        # Connect to NATS event bus
        await event_bus.connect()
        print("Connected to event bus")
        print()

        # Run conflict detection
        stats = await detect_conflicts_for_all_events(event_store, event_bus, dry_run=False)

        # Report results
        print()
        print("=" * 80)
        print("Backfill Complete")
        print("=" * 80)
        print(f"Calendar events analyzed:  {stats['events_analyzed']}")
        print(f"Parse errors encountered:  {stats['parse_errors']}")
        print(f"Conflicts detected:        {stats['conflicts_detected']}")
        print(f"Conflict events published: {stats['events_published']}")
        print()

        if stats['conflicts_detected'] > 0:
            print(f"✓ Published {stats['events_published']} calendar.conflict.detected events")
            print("  These will now trigger the 'High priority: calendar conflict' rule")
        else:
            print("✓ No conflicts found in historical calendar data")

    finally:
        # Clean up connections
        await event_bus.disconnect()
        print()
        print("Disconnected from event bus")


if __name__ == "__main__":
    asyncio.run(main())
