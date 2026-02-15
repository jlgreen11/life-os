"""
Life OS — Base Signal Extractor

Abstract base class for all signal extractors in the passive learning pipeline.
"""

from __future__ import annotations

from typing import Any

from storage.database import DatabaseManager, UserModelStore


class BaseExtractor:
    """Base class for all signal extractors.

    Every concrete extractor (Linguistic, Cadence, Mood, Relationship, Topic)
    inherits from this class and must implement two methods:

      1. can_process(event)  -- gate: return True if this event type is relevant.
      2. extract(event)      -- do the work: analyse the event, emit signal dicts,
                                and persist updated profiles as a side-effect.

    The pipeline iterates over all registered extractors, calls can_process to
    filter, then calls extract on the ones that opt in.  This pattern keeps
    extractors loosely coupled: adding a new behavioral dimension only requires
    a new subclass and a one-line registration in the pipeline.
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        # db: general-purpose database handle (event history, raw storage).
        self.db = db
        # ums: the User Model Store, where signal profiles and semantic facts
        # are read from and written to.  Every extractor shares this store so
        # the user model is built collaboratively across dimensions.
        self.ums = user_model_store

    def can_process(self, event: dict) -> bool:
        """Return True if this extractor is interested in the given event.

        Subclasses check the event's ``type`` field against the set of
        EventType values they handle.  This acts as a lightweight routing
        filter so the pipeline can skip irrelevant extractors without
        incurring extraction cost.
        """
        raise NotImplementedError

    def extract(self, event: dict) -> list[dict]:
        """Extract signals from the event. Returns a list of signal dicts.

        Each returned dict represents one discrete observation (e.g., a
        response-time measurement, a linguistic metric snapshot).  As a
        side-effect, implementations also call their own ``_update_profile``
        method to persist running aggregates into the User Model Store.
        """
        raise NotImplementedError
