"""
Backfill episode timestamps from actual event payload dates.

**Problem this fixes:**

When episodes were first created, _create_episode() looked for payload["date"]
to get the actual email timestamp, but the Google connector stores it as
payload["email_date"] (and iMessage/Signal use "sent_at" / "received_at").
The fallback was event["timestamp"] — the connector *sync* time — so all 55K+
email episodes ended up sharing the same 2-3 sync dates (e.g. 2026-02-20 to
2026-02-22), regardless of when the emails were actually sent.

This completely broke the routine detector (which requires activities on 3+
*distinct* calendar days to call something a routine) and temporal analysis.

**What this script does:**

For every episode whose current timestamp is within 24 hours of its source
event's sync timestamp (a strong signal that the wrong timestamp was used),
it looks up the original event from events.db, extracts the correct actual
timestamp using the same priority chain as the fixed _create_episode(), and
updates the episode row if a better timestamp is found.

Episodes whose timestamps already differ significantly from the event sync time
are left untouched — they were created correctly.

**Usage:**

    source .venv/bin/activate
    python scripts/backfill_episode_timestamps.py [--dry-run] [--limit N]

Options:
    --dry-run   Print what would change without writing to the database.
    --limit N   Only process the first N episodes (default: all).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_actual_timestamp(payload: dict, event_timestamp: str) -> str | None:
    """Return the best available actual-event timestamp from a payload dict.

    Uses the same priority chain as the fixed _create_episode() so that the
    backfill and the live code agree on which field to prefer.

    Args:
        payload: The deserialized event payload dict.
        event_timestamp: The event's sync timestamp (used only to detect stale
            episodes — never returned as the "better" timestamp).

    Returns:
        The best actual timestamp string, or None if no improvement is found.
    """
    return (
        payload.get("email_date")   # Google/Proton — actual Date header
        or payload.get("sent_at")   # iMessage, Signal — message send time
        or payload.get("received_at")  # arrival time for some connectors
        or payload.get("date")      # generic date field
        or payload.get("start_time")  # CalDAV / Google Calendar meeting start
    )


def _parse_ts(ts_str: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime.

    Returns None on any parse error or if the input is None/empty.
    """
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def backfill(data_dir: str = "data", dry_run: bool = False, limit: int | None = None) -> int:
    """Backfill episode timestamps from source event payloads.

    Connects to both user_model.db (episodes) and events.db (source events),
    finds episodes whose stored timestamp is suspiciously close to the event's
    sync timestamp (within 24 hours), looks up the actual event payload, and
    writes the corrected timestamp back.

    Args:
        data_dir: Directory containing the SQLite database files.
        dry_run: If True, report changes but do not write them.
        limit: Maximum number of episodes to process (None = all).

    Returns:
        Number of episodes updated (or that would be updated in dry-run mode).
    """
    data_path = Path(data_dir)
    um_db_path = str(data_path / "user_model.db")
    ev_db_path = str(data_path / "events.db")

    um_conn = sqlite3.connect(um_db_path)
    ev_conn = sqlite3.connect(ev_db_path)
    um_conn.row_factory = sqlite3.Row
    ev_conn.row_factory = sqlite3.Row

    # Attach events.db to the user_model connection so we can JOIN across DBs.
    # Using ATTACH lets us do the cross-db join in a single statement.
    um_conn.execute(f"ATTACH DATABASE ? AS eventsdb", (ev_db_path,))

    try:
        # Fetch episodes that have a source event ID to look up.
        # We join inline and filter to rows where the episode timestamp is
        # within 24 hours of the event's stored timestamp — the telltale sign
        # that the sync timestamp was used instead of the actual event date.
        query = """
            SELECT
                ep.id          AS episode_id,
                ep.timestamp   AS episode_ts,
                ep.event_id    AS event_id,
                ev.timestamp   AS event_sync_ts,
                ev.payload     AS event_payload
            FROM episodes ep
            JOIN eventsdb.events ev ON ev.id = ep.event_id
            WHERE ep.event_id IS NOT NULL
              AND ABS(
                julianday(ep.timestamp) - julianday(ev.timestamp)
              ) < 1.0   -- within 24 hours = likely using sync timestamp
            ORDER BY ep.timestamp ASC
        """
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        rows = um_conn.execute(query).fetchall()
        logger.info("Found %d episodes with potentially stale timestamps", len(rows))

        updated = 0
        skipped_no_payload = 0
        skipped_no_better_ts = 0
        skipped_ts_parse_fail = 0

        for row in rows:
            episode_id = row["episode_id"]
            event_sync_ts = row["event_sync_ts"]
            current_episode_ts = row["episode_ts"]

            # Deserialize the event payload
            try:
                raw = row["event_payload"]
                payload = json.loads(raw) if raw else {}
                # Handle double-encoded payloads (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                skipped_no_payload += 1
                continue

            # Extract the best available actual timestamp from the payload
            better_ts_str = _extract_actual_timestamp(payload, event_sync_ts)
            if not better_ts_str:
                skipped_no_better_ts += 1
                continue

            # Validate the better timestamp parses correctly
            better_dt = _parse_ts(better_ts_str)
            if not better_dt:
                skipped_ts_parse_fail += 1
                logger.debug("  Episode %s: could not parse better_ts=%r", episode_id, better_ts_str)
                continue

            # Only update if the new timestamp is meaningfully different
            # (more than 1 hour away from the current episode timestamp).
            # This avoids unnecessary writes for tiny clock skews.
            current_dt = _parse_ts(current_episode_ts)
            if current_dt and abs((better_dt - current_dt).total_seconds()) < 3600:
                skipped_no_better_ts += 1
                continue

            if dry_run:
                logger.info(
                    "  [DRY RUN] Would update episode %s: %s → %s",
                    episode_id, current_episode_ts, better_ts_str,
                )
            else:
                um_conn.execute(
                    "UPDATE episodes SET timestamp = ? WHERE id = ?",
                    (better_ts_str, episode_id),
                )

            updated += 1

        if not dry_run and updated > 0:
            um_conn.commit()

        logger.info(
            "Done. updated=%d, skipped_no_payload=%d, skipped_no_better_ts=%d, "
            "skipped_parse_fail=%d",
            updated, skipped_no_payload, skipped_no_better_ts, skipped_ts_parse_fail,
        )
        return updated

    finally:
        um_conn.execute("DETACH DATABASE eventsdb")
        um_conn.close()
        ev_conn.close()


def main() -> None:
    """CLI entry point for the episode timestamp backfill script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing to the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N episodes (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the SQLite databases (default: data/).",
    )
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=== Episode timestamp backfill (%s) ===", mode)

    count = backfill(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    if args.dry_run:
        logger.info("%d episodes would be updated. Run without --dry-run to apply.", count)
    else:
        logger.info("%d episodes updated.", count)


if __name__ == "__main__":
    main()
