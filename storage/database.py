"""
Backward-compatibility shim.
Import from storage.manager, storage.event_store, or storage.user_model_store directly.
"""
from storage.manager import DatabaseManager
from storage.event_store import EventStore
from storage.user_model_store import UserModelStore

__all__ = ["DatabaseManager", "EventStore", "UserModelStore"]
