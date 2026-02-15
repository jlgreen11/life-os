"""Signal extractor pipeline — passive behavioral analysis from user interactions."""

# Import the abstract base class that all extractors inherit from.
from services.signal_extractor.base import BaseExtractor

# Import each concrete extractor — one per behavioral dimension.
from services.signal_extractor.linguistic import LinguisticExtractor
from services.signal_extractor.cadence import CadenceExtractor
from services.signal_extractor.mood import MoodInferenceEngine
from services.signal_extractor.relationship import RelationshipExtractor
from services.signal_extractor.topic import TopicExtractor

# The pipeline wires all extractors together and serves as the single entry point
# for event processing from the NATS bus.
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
