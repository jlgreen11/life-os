"""
Life OS — Event Store

High-level operations on the immutable event log.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from storage.manager import DatabaseManager


class EventStore:
    """High-level operations on the event log."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def store_event(self, event: dict) -> str:
        """Store an event and return its ID."""
        with self.db.get_connection("events") as conn:
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
        """Query events with optional filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        if source:
            query += " AND source = ?"
            params.append(source)
        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.db.get_connection("events") as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_event_count(self) -> int:
        with self.db.get_connection("events") as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
            return row["cnt"]
