"""
Life OS — Semantic Fact Inference Service

Derives high-level semantic facts from signal profiles to populate Layer 2
(Semantic Memory) of the user model.

The signal extractors (linguistic, cadence, relationship, topic, mood) collect
raw data points and compute running statistics in signal_profiles. This service
analyzes those statistics to infer semantic facts about the user's:
  - Implicit preferences (communication style, contact priorities)
  - Expertise (topics they discuss authoritatively)
  - Values (what they prioritize based on behavior patterns)
  - Anti-preferences (patterns of avoidance or dismissal)

Semantic facts are confidence-weighted and grow stronger with repeated
observations. They inform AI decision-making throughout the system.
"""

from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer

__all__ = ["SemanticFactInferrer"]
