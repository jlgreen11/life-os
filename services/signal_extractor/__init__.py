"""
Life OS — Signal Extractor Package

The passive learning engine. Watches every event and extracts
behavioral signals that build the User Model over time.
"""

from services.signal_extractor.base import BaseExtractor
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor
from services.signal_extractor.pipeline import SignalExtractorPipeline

__all__ = [
    "BaseExtractor",
    "LinguisticExtractor",
    "CadenceExtractor",
    "MoodInferenceEngine",
    "RelationshipExtractor",
    "TopicExtractor",
    "SignalExtractorPipeline",
]
