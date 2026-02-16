"""
Life OS — Signal Extractor Pipeline

The main pipeline that routes events through all extractors.
Subscribes to the NATS event bus and processes every event.
"""

from __future__ import annotations

from models.user_model import MoodState
from storage.database import DatabaseManager, UserModelStore

# Each extractor is responsible for one behavioral dimension.  The pipeline
# instantiates all of them and fans every incoming event out to each one that
# declares interest via its `can_process` method.
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor
from services.signal_extractor.temporal import TemporalExtractor
from services.signal_extractor.spatial import SpatialExtractor
from services.signal_extractor.base import BaseExtractor


class SignalExtractorPipeline:
    """
    The main pipeline that routes events through all extractors.
    Subscribes to the NATS event bus and processes every event.
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        self.db = db
        self.ums = user_model_store

        # Register every extractor that participates in the pipeline.  Each one
        # receives the shared database handle and user-model store so it can
        # both read historical context and persist the signals it produces.
        self.extractors: list[BaseExtractor] = [
            LinguisticExtractor(db, user_model_store),
            CadenceExtractor(db, user_model_store),
            MoodInferenceEngine(db, user_model_store),
            RelationshipExtractor(db, user_model_store),
            TopicExtractor(db, user_model_store),
            TemporalExtractor(db, user_model_store),
            SpatialExtractor(db, user_model_store),
        ]

        # A dedicated mood engine instance is kept separately so we can call
        # compute_current_mood() on demand without iterating the extractor list.
        self.mood_engine = MoodInferenceEngine(db, user_model_store)

    async def process_event(self, event: dict) -> list[dict]:
        """
        Process an event through all applicable extractors.
        Returns all extracted signals.

        This is the main entry point called by the NATS event-bus subscriber.
        Each incoming event is offered to every registered extractor; the
        extractor's `can_process` gate decides whether it should run.  Signals
        produced by all interested extractors are collected and returned as a
        flat list so callers can forward them downstream (e.g., to the user
        model aggregator or an analytics sink).
        """
        all_signals = []

        for extractor in self.extractors:
            # Route: let each extractor decide if this event type is relevant.
            if extractor.can_process(event):
                try:
                    # extract() both returns signals AND persists them into the
                    # extractor's own profile store as a side-effect.
                    signals = extractor.extract(event)
                    all_signals.extend(signals)
                except Exception as e:
                    # Fail-open: signal extraction must never block or crash the
                    # main event processing loop.  Log and continue.
                    print(f"Extractor {type(extractor).__name__} error: {e}")

        return all_signals

    def get_current_mood(self) -> MoodState:
        """Get the current mood estimate.

        Delegates to the dedicated MoodInferenceEngine which aggregates recent
        mood signals (sleep, language sentiment, calendar density, etc.) into a
        multi-dimensional MoodState.  Called on-demand by the orchestrator or
        periodically (every ~15 minutes) to keep the mood snapshot fresh.
        """
        return self.mood_engine.compute_current_mood()

    def get_user_summary(self) -> dict:
        """Get a summary of what we know about the user.

        Collects metadata from every signal profile (linguistic, cadence, mood,
        relationships, topics) and merges it with high-confidence semantic facts
        from the user-model store.  The result is a lightweight snapshot
        suitable for the orchestrator's system prompt or for a debugging UI.
        """
        # Gather per-dimension profile summaries (sample counts and freshness).
        profiles = {}
        for profile_type in ["linguistic", "cadence", "mood_signals", "relationships", "topics", "temporal", "spatial"]:
            profile = self.ums.get_signal_profile(profile_type)
            if profile:
                profiles[profile_type] = {
                    "samples_count": profile["samples_count"],
                    "last_updated": profile["updated_at"],
                }

        # Pull semantic facts that have accumulated enough confidence (>= 0.3)
        # to be worth mentioning, then surface only the high-confidence ones
        # (>= 0.7) in the summary payload so the orchestrator can rely on them.
        facts = self.ums.get_semantic_facts(min_confidence=0.3)

        return {
            "profiles": profiles,
            "semantic_facts_count": len(facts),
            "high_confidence_facts": [
                {"key": f["key"], "value": f["value"], "confidence": f["confidence"]}
                for f in facts
                if f["confidence"] >= 0.7
            ],
        }
