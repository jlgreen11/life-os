"""
Backward-compatibility shim.
Import from storage.manager, storage.event_store, or storage.user_model_store directly.

NOTE: This module exists solely for backward compatibility with code that imports
from ``storage.database``.  New code should import directly from the specific
sub-modules (storage.manager, storage.event_store, storage.user_model_store)
so that dependencies are explicit and circular-import risks are minimized.
"""

# Re-export the primary classes so that ``from storage.database import X`` continues to work.
from storage.manager import DatabaseManager
from storage.event_store import EventStore
from storage.user_model_store import UserModelStore

__all__ = ["DatabaseManager", "EventStore", "UserModelStore"]
