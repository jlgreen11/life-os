"""
Life OS — Base Signal Extractor

Abstract base class for all signal extractors in the passive learning pipeline.
"""

from __future__ import annotations

from typing import Any

from storage.database import DatabaseManager, UserModelStore


class BaseExtractor:
    """Base class for all signal extractors."""

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        self.db = db
        self.ums = user_model_store

    def can_process(self, event: dict) -> bool:
        """Does this extractor care about this event?"""
        raise NotImplementedError

    def extract(self, event: dict) -> list[dict]:
        """Extract signals from the event. Returns a list of signal dicts."""
        raise NotImplementedError
