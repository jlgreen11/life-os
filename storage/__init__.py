"""
Life OS — Storage Package

Persistence layer built on SQLite (relational data) and LanceDB (vector embeddings).

Modules:
    manager.py          — Database connection management and schema migrations
    event_store.py      — High-level operations on the immutable event log
    user_model_store.py — User model persistence (episodes, facts, signals, mood)
    vector_store.py     — Semantic search via LanceDB with NumPy fallback
    database.py         — Backward-compatibility shim (re-exports from above modules)
"""

from storage.manager import DatabaseManager
from storage.event_store import EventStore
from storage.user_model_store import UserModelStore
from storage.vector_store import VectorStore

__all__ = [
    "DatabaseManager",
    "EventStore",
    "UserModelStore",
    "VectorStore",
]
