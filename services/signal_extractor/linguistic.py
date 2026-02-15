"""
Life OS — Linguistic Signal Extractor

Builds the LinguisticProfile from outbound messages and voice commands.

Watches for: messages sent, emails sent, voice commands, draft edits.
Extracts: vocabulary complexity, formality, common patterns, per-contact style.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


class LinguisticExtractor(BaseExtractor):
    """
    Builds the LinguisticProfile from outbound messages and voice commands.
    """

    # Common hedge words/phrases
    HEDGE_PATTERNS = [
        r'\bmaybe\b', r'\bperhaps\b', r'\bI think\b', r'\bsort of\b',
        r'\bkind of\b', r'\bpossibly\b', r'\bmight\b', r'\bcould be\b',
        r'\bI guess\b', r'\bnot sure\b', r'\bprobably\b',
    ]

    ASSERTION_PATTERNS = [
        r'\bwe need to\b', r'\bwe must\b', r'\bthis has to\b',
        r'\bdefinitely\b', r'\bclearly\b', r'\bobviously\b',
        r'\bwithout a doubt\b', r'\babsolutely\b',
    ]

    PROFANITY_PATTERNS = [
        r'\bdamn\b', r'\bhell\b', r'\bshit\b', r'\bfuck\b',
        r'\bass\b', r'\bbullshit\b', r'\bcrap\b',
    ]

    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    def can_process(self, event: dict) -> bool:
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            "system.user.command",
        ]

    def extract(self, event: dict) -> list[dict]:
        payload = event.get("payload", {})
        text = payload.get("body", "") or payload.get("body_plain", "") or ""
        if not text or len(text) < 10:
            return []

        signals = []
        contact_id = payload.get("to_addresses", [None])[0] if payload.get("to_addresses") else None
        channel = payload.get("channel", event.get("source", "unknown"))

        # --- Sentence analysis ---
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        word_count = len(words)

        if word_count < 3:
            return []

        avg_sentence_length = word_count / max(len(sentences), 1)
        unique_word_ratio = len(set(w.lower() for w in words)) / word_count

        # --- Formality estimation ---
        informal_markers = sum(1 for w in words if w.lower() in [
            "hey", "yeah", "nah", "gonna", "wanna", "lol", "haha",
            "ok", "yep", "nope", "btw", "idk", "imo", "tbh",
        ])
        formal_markers = sum(1 for w in words if w.lower() in [
            "regarding", "furthermore", "therefore", "accordingly",
            "sincerely", "respectfully", "per", "pursuant",
        ])
        formality = 0.5
        if informal_markers + formal_markers > 0:
            formality = formal_markers / (informal_markers + formal_markers)

        # --- Pattern counting ---
        hedge_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.HEDGE_PATTERNS
        )
        assertion_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.ASSERTION_PATTERNS
        )
        profanity_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.PROFANITY_PATTERNS
        )
        exclamation_count = text.count("!")
        question_count = text.count("?")
        ellipsis_count = text.count("...")

        # --- Emoji extraction ---
        emojis = self.EMOJI_PATTERN.findall(text)
        emoji_count = len(emojis)

        # --- Greeting/closing detection ---
        first_line = sentences[0].lower().strip() if sentences else ""
        last_line = sentences[-1].lower().strip() if sentences else ""
        greeting = self._detect_greeting(first_line)
        closing = self._detect_closing(last_line)

        # --- Build the signal ---
        signal = {
            "type": "linguistic",
            "timestamp": event.get("timestamp"),
            "contact_id": contact_id,
            "channel": channel,
            "metrics": {
                "word_count": word_count,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "unique_word_ratio": round(unique_word_ratio, 3),
                "formality": round(formality, 2),
                "hedge_rate": round(hedge_count / max(len(sentences), 1), 3),
                "assertion_rate": round(assertion_count / max(len(sentences), 1), 3),
                "exclamation_rate": round(exclamation_count / max(len(sentences), 1), 3),
                "question_rate": round(question_count / max(len(sentences), 1), 3),
                "ellipsis_rate": round(ellipsis_count / max(len(sentences), 1), 3),
                "emoji_count": emoji_count,
                "emojis_used": emojis,
                "profanity_count": profanity_count,
                "greeting_detected": greeting,
                "closing_detected": closing,
            },
        }

        signals.append(signal)

        # Store updated profile
        self._update_profile(signal)

        return signals

    def _detect_greeting(self, text: str) -> Optional[str]:
        greetings = {
            "hey": "hey", "hi": "hi", "hello": "hello",
            "good morning": "good morning", "morning": "morning",
            "yo": "yo", "sup": "sup",
        }
        for pattern, label in greetings.items():
            if text.startswith(pattern):
                return label
        return None

    def _detect_closing(self, text: str) -> Optional[str]:
        closings = {
            "best": "best", "cheers": "cheers", "thanks": "thanks",
            "talk soon": "talk soon", "take care": "take care",
            "regards": "regards", "sincerely": "sincerely",
        }
        for pattern, label in closings.items():
            if pattern in text:
                return label
        return None

    def _update_profile(self, signal: dict):
        """Incrementally update the stored linguistic profile."""
        existing = self.ums.get_signal_profile("linguistic")
        if existing:
            data = existing["data"]
        else:
            data = {"samples": [], "per_contact": {}}

        # Keep last 500 samples for running statistics
        data["samples"].append(signal["metrics"])
        if len(data["samples"]) > 500:
            data["samples"] = data["samples"][-500:]

        # Per-contact tracking
        contact = signal.get("contact_id")
        if contact:
            if contact not in data["per_contact"]:
                data["per_contact"][contact] = []
            data["per_contact"][contact].append(signal["metrics"])
            if len(data["per_contact"][contact]) > 100:
                data["per_contact"][contact] = data["per_contact"][contact][-100:]

        # Compute running averages
        samples = data["samples"]
        data["averages"] = {
            "avg_sentence_length": statistics.mean(s["avg_sentence_length"] for s in samples),
            "formality": statistics.mean(s["formality"] for s in samples),
            "hedge_rate": statistics.mean(s["hedge_rate"] for s in samples),
            "assertion_rate": statistics.mean(s["assertion_rate"] for s in samples),
            "exclamation_rate": statistics.mean(s["exclamation_rate"] for s in samples),
            "emoji_rate": statistics.mean(s["emoji_count"] / max(s["word_count"], 1) for s in samples),
            "profanity_rate": statistics.mean(s["profanity_count"] / max(s["word_count"], 1) for s in samples),
        }

        # Detect common patterns
        all_greetings = [s["greeting_detected"] for s in samples if s["greeting_detected"]]
        all_closings = [s["closing_detected"] for s in samples if s["closing_detected"]]
        data["common_greetings"] = [g for g, _ in Counter(all_greetings).most_common(3)]
        data["common_closings"] = [c for c, _ in Counter(all_closings).most_common(3)]

        self.ums.update_signal_profile("linguistic", data)
