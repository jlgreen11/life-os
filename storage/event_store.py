"""
Life OS — Event Store

High-level operations on the immutable event log.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from storage.manager import DatabaseManager


class EventStore:
    """High-level operations on the event log.

    The event log follows an append-only pattern: events are inserted but never
    updated or deleted.  This guarantees a complete, tamper-evident history of
    every interaction the system has observed.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def store_event(self, event: dict) -> str:
        """Store an event and return its ID.

        Appends a single event to the immutable log.  The caller is
        responsible for generating a unique ``id`` (typically a UUID).
        ``payload`` and ``metadata`` are serialized to JSON strings so
        that each event type can carry arbitrary structured data without
        requiring schema changes.
        """
        with self.db.get_connection("events") as conn:
            # Parameterized query (? placeholders) prevents SQL injection by
            # ensuring user-supplied values are never interpolated into the SQL
            # string — SQLite handles escaping internally.
            conn.execute(
                """INSERT INTO events (id, type, source, timestamp, priority, payload, metadata, embedding_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["id"],
                    event["type"],
                    event["source"],
                    event["timestamp"],
                    event.get("priority", "normal"),
                    json.dumps(event.get("payload", {})),
                    json.dumps(event.get("metadata", {})),
                    event.get("embedding_id"),
                ),
            )
        return event["id"]

    def get_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query events with optional filters.

        Uses a query-builder pattern: starts with a base ``WHERE 1=1`` clause
        (a no-op predicate) so that each optional filter can unconditionally
        append ``AND <condition>`` without worrying about whether it is the
        first predicate.  All filter values are passed as parameterized ``?``
        placeholders to prevent SQL injection.
        """
        # Base query — ``WHERE 1=1`` is a common SQL idiom that simplifies
        # dynamic query construction by guaranteeing a WHERE clause always exists.
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        # Each optional filter appends to both the SQL string and the params list.
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        if source:
            query += " AND source = ?"
            params.append(source)
        if since:
            query += " AND timestamp > ?"
            params.append(since)

        # Always order newest-first and cap the result set for safety.
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.db.get_connection("events") as conn:
            rows = conn.execute(query, params).fetchall()
            # Convert sqlite3.Row objects to plain dicts for JSON serialization.
            return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        """Return the total number of events in the log (used by health checks)."""
        with self.db.get_connection("events") as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
            return row["cnt"]

    # -------------------------------------------------------------------
    # Event tagging — stored in event_tags (separate from the immutable
    # event log) so that annotations like rule-applied tags and suppress
    # flags don't violate the append-only invariant on the events table.
    # -------------------------------------------------------------------

    def add_tag(self, event_id: str, tag: str,
                rule_id: Optional[str] = None) -> None:
        """Attach a tag to an event.

        Uses INSERT OR IGNORE so re-tagging the same event with the same
        tag is a safe no-op (idempotent).
        """
        with self.db.get_connection("events") as conn:
            conn.execute(
                """INSERT OR IGNORE INTO event_tags (event_id, tag, rule_id)
                   VALUES (?, ?, ?)""",
                (event_id, tag, rule_id),
            )

    def get_tags(self, event_id: str) -> list[str]:
        """Return all tags for an event."""
        with self.db.get_connection("events") as conn:
            rows = conn.execute(
                "SELECT tag FROM event_tags WHERE event_id = ?",
                (event_id,),
            ).fetchall()
            return [row["tag"] for row in rows]

    def has_tag(self, event_id: str, tag: str) -> bool:
        """Check whether an event has a specific tag."""
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                "SELECT 1 FROM event_tags WHERE event_id = ? AND tag = ?",
                (event_id, tag),
            ).fetchone()
            return row is not None

    def is_suppressed(self, event_id: str) -> bool:
        """Check whether an event has been suppressed by a rule action."""
        return self.has_tag(event_id, "system:suppressed")

    def get_timestamp_by_message_id(self, message_id: str) -> Optional[str]:
        """Look up an event's timestamp by its payload.message_id.

        Used by the cadence extractor to calculate response times.  The
        query uses json_extract() against an expression index on
        payload.message_id for O(log n) lookups.
        """
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                """SELECT timestamp FROM events
                   WHERE json_extract(payload, '$.message_id') = ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (message_id,),
            ).fetchone()
            return row["timestamp"] if row else None
