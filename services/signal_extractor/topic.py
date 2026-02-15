"""
Life OS — Topic Signal Extractor

Extracts topics and interests from messages, searches, and browsing.
Over time, builds a map of what the user cares about.
"""

from __future__ import annotations

import re
from collections import Counter

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


class TopicExtractor(BaseExtractor):
    """
    Extracts topics and interests from messages, searches, and browsing.
    Over time, builds a map of what the user cares about.
    """

    def can_process(self, event: dict) -> bool:
        return event.get("type") in [
            EventType.EMAIL_RECEIVED.value,
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.MESSAGE_SENT.value,
            "system.user.command",
        ]

    def extract(self, event: dict) -> list[dict]:
        payload = event.get("payload", {})
        text = payload.get("body", "") or payload.get("body_plain", "") or ""
        subject = payload.get("subject", "")
        combined = f"{subject} {text}"

        if len(combined.strip()) < 20:
            return []

        # Simple keyword extraction (in production, use LLM or NER)
        topics = self._extract_keywords(combined)

        if topics:
            signal = {
                "type": "topic",
                "timestamp": event.get("timestamp"),
                "topics": topics,
                "context": event.get("type"),
                "source": event.get("source"),
            }
            self._update_topic_map(signal)
            return [signal]

        return []

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Simple keyword extraction. In production, this would use:
        1. An LLM for entity/topic extraction
        2. A local NER model
        3. TF-IDF against the user's corpus
        """
        # Remove common stop words, extract longer meaningful words
        stop_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "i", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just",
            "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "was", "are", "been", "has", "had", "did",
            "am", "were", "does", "done", "being", "having",
        }

        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        counter = Counter(filtered)
        return [word for word, count in counter.most_common(10) if count >= 1]

    def _update_topic_map(self, signal: dict):
        existing = self.ums.get_signal_profile("topics")
        data = existing["data"] if existing else {"topic_counts": {}, "recent_topics": []}

        for topic in signal["topics"]:
            data["topic_counts"][topic] = data["topic_counts"].get(topic, 0) + 1

        data["recent_topics"].append({
            "topics": signal["topics"],
            "timestamp": signal["timestamp"],
            "context": signal["context"],
        })

        # Keep last 500 topic observations
        if len(data["recent_topics"]) > 500:
            data["recent_topics"] = data["recent_topics"][-500:]

        self.ums.update_signal_profile("topics", data)
