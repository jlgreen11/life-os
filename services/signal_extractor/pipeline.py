"""
Life OS — Signal Extractor Pipeline

The main pipeline that routes events through all extractors.
Subscribes to the NATS event bus and processes every event.
"""

from __future__ import annotations

from models.user_model import MoodState
from storage.database import DatabaseManager, UserModelStore

from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor
from services.signal_extractor.base import BaseExtractor


class SignalExtractorPipeline:
    """
    The main pipeline that routes events through all extractors.
    Subscribes to the NATS event bus and processes every event.
    """

    def __init__(self, db: DatabaseManager, user_model_store: UserModelStore):
        self.db = db
        self.ums = user_model_store

        # Initialize all extractors
        self.extractors: list[BaseExtractor] = [
            LinguisticExtractor(db, user_model_store),
            CadenceExtractor(db, user_model_store),
            MoodInferenceEngine(db, user_model_store),
            RelationshipExtractor(db, user_model_store),
            TopicExtractor(db, user_model_store),
        ]

        self.mood_engine = MoodInferenceEngine(db, user_model_store)

    async def process_event(self, event: dict) -> list[dict]:
        """
        Process an event through all applicable extractors.
        Returns all extracted signals.
        """
        all_signals = []

        for extractor in self.extractors:
            if extractor.can_process(event):
                try:
                    signals = extractor.extract(event)
                    all_signals.extend(signals)
                except Exception as e:
                    # Log but don't fail — signal extraction is best-effort
                    print(f"Extractor {type(extractor).__name__} error: {e}")

        return all_signals

    def get_current_mood(self) -> MoodState:
        """Get the current mood estimate."""
        return self.mood_engine.compute_current_mood()

    def get_user_summary(self) -> dict:
        """Get a summary of what we know about the user."""
        profiles = {}
        for profile_type in ["linguistic", "cadence", "mood_signals", "relationships", "topics"]:
            profile = self.ums.get_signal_profile(profile_type)
            if profile:
                profiles[profile_type] = {
                    "samples_count": profile["samples_count"],
                    "last_updated": profile["updated_at"],
                }

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
