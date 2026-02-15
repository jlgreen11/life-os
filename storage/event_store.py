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
