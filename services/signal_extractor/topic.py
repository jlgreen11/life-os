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
        # Topics are extracted from both inbound and outbound communication
        # (unlike the linguistic extractor which only analyses the user's own
        # writing).  Inbound messages reveal what *others* bring to the user's
        # attention, while outbound messages reveal what the user actively
        # engages with.  Voice/system commands are also included.
        return event.get("type") in [
            EventType.EMAIL_RECEIVED.value,
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.MESSAGE_SENT.value,
            "system.user.command",
        ]

    def extract(self, event: dict) -> list[dict]:
        """Extract topic keywords from the message body and subject line.

        The subject and body are concatenated before keyword extraction so
        that subject-only emails (e.g., calendar forwards) still yield
        topics.  Messages shorter than 20 characters are skipped because
        they rarely contain meaningful topical content.
        """
        payload = event.get("payload", {})
        text = payload.get("body", "") or payload.get("body_plain", "") or ""
        subject = payload.get("subject", "")
        # Combine subject and body so topics present only in the subject line
        # are still captured.
        combined = f"{subject} {text}"

        # Skip very short content — not enough text for reliable extraction.
        if len(combined.strip()) < 20:
            return []

        # Run keyword extraction (currently rule-based; see _extract_keywords
        # for notes on upgrading to LLM/NER in production).
        topics = self._extract_keywords(combined)

        if topics:
            signal = {
                "type": "topic",
                "timestamp": event.get("timestamp"),
                "topics": topics,
                # Preserve the originating event type and source so we can
                # later distinguish topics from emails vs. chat vs. commands.
                "context": event.get("type"),
                "source": event.get("source"),
            }
            # Merge into the running topic-frequency map.
            self._update_topic_map(signal)
            return [signal]

        return []

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Simple keyword extraction. In production, this would use:
        1. An LLM for entity/topic extraction
        2. A local NER model
        3. TF-IDF against the user's corpus

        Current approach:
          - Tokenise to words of 4+ alphabetic characters (filters out short
            function words and numeric tokens).
          - Remove a curated English stop-word set.
          - Count frequencies and return the top-10 most frequent words.

        This is deliberately simple — the accuracy is "good enough" to build
        a coarse interest map, and the intent is to replace it with an LLM
        call or local NER model once the pipeline matures.
        """
        # Comprehensive English stop-word list covering common verbs, pronouns,
        # prepositions, and conjunctions that carry no topical signal.
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

        # Extract only alphabetic tokens of 4+ characters to exclude short
        # noise words that slip past the stop-word set.
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        # Return the top-10 most frequent content words.  The count >= 1 guard
        # is effectively a no-op here but makes the intent explicit: every
        # surviving word is worth reporting.
        counter = Counter(filtered)
        return [word for word, count in counter.most_common(10) if count >= 1]

    def _update_topic_map(self, signal: dict):
        """Merge newly extracted topics into the persisted topic profile.

        The profile maintains two structures:
          - topic_counts:   an all-time frequency dict mapping each keyword to
                            the number of messages it has appeared in.  This
                            acts as a coarse "interest strength" metric.
          - recent_topics:  a ring buffer (capped at 500) of timestamped topic
                            lists.  This provides temporal context so we can
                            detect trending topics and seasonal interests.
        """
        existing = self.ums.get_signal_profile("topics")
        data = existing["data"] if existing else {"topic_counts": {}, "recent_topics": []}

        # Increment the all-time frequency counter for each extracted keyword.
        for topic in signal["topics"]:
            data["topic_counts"][topic] = data["topic_counts"].get(topic, 0) + 1

        # Append a timestamped snapshot so we can later slice by time window
        # to find "what has the user been talking about this week?"
        data["recent_topics"].append({
            "topics": signal["topics"],
            "timestamp": signal["timestamp"],
            "context": signal["context"],
        })

        # Cap the recent-topics buffer to prevent unbounded growth while
        # retaining enough observations for meaningful trend detection.
        if len(data["recent_topics"]) > 500:
            data["recent_topics"] = data["recent_topics"][-500:]

        self.ums.update_signal_profile("topics", data)
