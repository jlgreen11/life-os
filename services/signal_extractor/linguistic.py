"""
Life OS — Linguistic Signal Extractor

Builds linguistic profiles from both outbound and inbound messages.

Outbound analysis (user's own writing) feeds the user's linguistic fingerprint.
Inbound analysis (contacts' writing) builds per-contact incoming style profiles,
enabling tone-shift detection and formality-mismatch awareness.

Watches for: messages sent/received, emails sent/received, voice commands.
Extracts: vocabulary complexity, formality, common patterns, per-contact style.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


class LinguisticExtractor(BaseExtractor):
    """
    Builds linguistic profiles from both outbound and inbound messages.

    Outbound messages update the user's own linguistic fingerprint (``linguistic``
    profile).  Inbound messages update per-contact incoming style profiles
    (``linguistic_inbound`` profile) so the system can detect tone shifts,
    formality mismatches, and unusual communication patterns from contacts.
    """

    # --- Regex pattern banks for linguistic feature detection ---
    # Each bank is a list of word-boundary-delimited patterns so we can count
    # occurrences per message and derive per-sentence rates.

    # Hedge words signal tentativeness / low confidence in the user's writing.
    HEDGE_PATTERNS = [
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bI think\b",
        r"\bsort of\b",
        r"\bkind of\b",
        r"\bpossibly\b",
        r"\bmight\b",
        r"\bcould be\b",
        r"\bI guess\b",
        r"\bnot sure\b",
        r"\bprobably\b",
    ]

    # Assertion patterns signal directness / high confidence.
    ASSERTION_PATTERNS = [
        r"\bwe need to\b",
        r"\bwe must\b",
        r"\bthis has to\b",
        r"\bdefinitely\b",
        r"\bclearly\b",
        r"\bobviously\b",
        r"\bwithout a doubt\b",
        r"\babsolutely\b",
    ]

    # Profanity patterns — tracked not to censor, but to gauge emotional
    # intensity and how the user's register shifts across contacts/channels.
    PROFANITY_PATTERNS = [
        r"\bdamn\b",
        r"\bhell\b",
        r"\bshit\b",
        r"\bfuck\b",
        r"\bass\b",
        r"\bbullshit\b",
        r"\bcrap\b",
    ]

    # Broad Unicode emoji regex covering emoticons, symbols, pictographs,
    # transport icons, and flag sequences.  Used to count emoji density, which
    # is a signal of informality and emotional expressiveness.
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    # Humor markers signal levity, playfulness, and informal register.
    # Tracking these helps calibrate the AI's tone when drafting replies —
    # a user who frequently says "lol" wants casual drafts, not formal ones.
    HUMOR_PATTERNS = [
        r"\bhaha\b",
        r"\bhehe\b",
        r"\blol\b",
        r"\blmao\b",
        r"\blmfao\b",
        r"\bjk\b",
        r"\bjust kidding\b",
        r"\brofl\b",
        r"\bxd\b",
    ]

    # Affirmative response patterns reveal how the user says "yes".  Matching
    # these builds a vocabulary the AI can mirror when accepting requests or
    # confirming plans.
    AFFIRMATIVE_PATTERNS = [
        r"\byes\b",
        r"\byeah\b",
        r"\byep\b",
        r"\bsure\b",
        r"\babsolutely\b",
        r"\bsounds good\b",
        r"\bof course\b",
        r"\bdefinitely\b",
        r"\bwill do\b",
        r"\bperfect\b",
        r"\bgreat\b",
        r"\bon it\b",
    ]

    # Negative/declination patterns reveal how the user says "no" or declines.
    # Useful for drafting polite refusals that match the user's natural voice.
    NEGATIVE_PATTERNS = [
        r"\bcan't make it\b",
        r"\bunable to\b",
        r"\bnot possible\b",
        r"\bunfortunately\b",
        r"\bsorry,?\s+i\b",
        r"\bapologies\b",
        r"\bI(?:'m| am) not\b",
        r"\bI (?:don't|can't|won't|couldn't)\b",
    ]

    # Gratitude patterns reveal how the user expresses appreciation.
    # The AI can mirror these when closing drafted messages.
    GRATITUDE_PATTERNS = [
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bappreciate\b",
        r"\bgrateful\b",
        r"\bmuch appreciated\b",
        r"\bthank\b",
    ]

    # Oxford comma detection regexes — mutually exclusive by design:
    #   OXFORD_COMMA_RE:  "eggs, bacon, and coffee"  (comma before conjunction)
    #   NO_OXFORD_RE:     "eggs, bacon and coffee"   (no comma before conjunction)
    # Both require a preceding comma in the list, so they only fire on actual
    # list constructions (not stray "and" conjunctions).
    OXFORD_COMMA_RE = re.compile(r"\b\w+,\s+\w+,\s+(?:and|or)\s+\w+", re.IGNORECASE)
    NO_OXFORD_RE = re.compile(r"\b\w+,\s+\w+\s+(?:and|or)\s+\w+", re.IGNORECASE)

    def _detect_first(self, text: str, patterns: list[str]) -> Optional[str]:
        """Return the actual matched text of the first pattern that fires, or None.

        Scans the pattern bank in order and returns the lowercased matched
        string on the first hit.  The returned value is the word or phrase as
        written by the user (e.g. ``"lol"``, ``"thank you"``), not the regex
        pattern string.  This lets the caller build frequency distributions of
        the user's actual vocabulary rather than of abstract pattern categories.

        Example::

            self._detect_first("haha that's hilarious", self.HUMOR_PATTERNS)
            # → "haha"

            self._detect_first("Of course, I'll get that done", self.AFFIRMATIVE_PATTERNS)
            # → "of course"
        """
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(0).lower()
        return None

    def can_process(self, event: dict) -> bool:
        # Analyse both directions.  Outbound messages feed the user's own
        # linguistic fingerprint; inbound messages build per-contact incoming
        # style profiles for tone-shift and formality-mismatch detection.
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
            "system.user.command",
        ]

    def extract(self, event: dict) -> list[dict]:
        """Analyse a message and return a linguistic signal.

        Works for both outbound (user's writing) and inbound (contact's writing).
        The analysis pipeline follows a fixed sequence:
          1. Tokenise into sentences and words.
          2. Compute structural metrics (sentence length, vocabulary richness).
          3. Estimate formality from marker-word ratios.
          4. Count stylistic patterns (hedges, assertions, profanity, punctuation).
          5. Extract emoji usage.
          6. Detect greeting / closing conventions.
          7. Package everything into a single signal dict and persist it.

        Outbound signals are stored in the ``linguistic`` profile (user fingerprint).
        Inbound signals are stored in the ``linguistic_inbound`` profile (per-contact
        incoming style) so the two never conflate.
        """
        payload = event.get("payload", {})
        event_type = event.get("type", "")
        # Prefer the rich "body" field; fall back to plain-text if absent.
        text = payload.get("body", "") or payload.get("body_plain", "") or ""
        # Skip very short texts — they lack enough signal to be useful and
        # would skew running averages.
        if not text or len(text) < 10:
            return []

        signals = []

        # Determine message direction.  Inbound messages come from a contact;
        # outbound messages are authored by the user.
        is_inbound = event_type in [
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
        ]

        # Resolve the relevant contact:
        #   outbound → first recipient (to_addresses)
        #   inbound  → sender (from_address)
        if is_inbound:
            contact_id = payload.get("from_address")
        else:
            contact_id = payload.get("to_addresses", [None])[0] if payload.get("to_addresses") else None
        channel = payload.get("channel", event.get("source", "unknown"))

        # --- Sentence analysis ---
        # Split on sentence-ending punctuation to approximate sentence boundaries.
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        word_count = len(words)

        # Reject trivially short messages that survived the len(text) check
        # above but contain fewer than 3 actual words.
        if word_count < 3:
            return []

        # Average sentence length proxies for syntactic complexity.
        avg_sentence_length = word_count / max(len(sentences), 1)
        # Unique-word ratio (type-token ratio) measures vocabulary richness:
        # higher values mean more diverse word choice.
        unique_word_ratio = len(set(w.lower() for w in words)) / word_count

        # --- Formality estimation ---
        # Count informal markers (slang, abbreviations) and formal markers
        # (business/academic register words).  The ratio gives a 0-1 formality
        # score; 0.5 is the neutral default when no markers are found.
        informal_markers = sum(
            1
            for w in words
            if w.lower()
            in [
                "hey",
                "yeah",
                "nah",
                "gonna",
                "wanna",
                "lol",
                "haha",
                "ok",
                "yep",
                "nope",
                "btw",
                "idk",
                "imo",
                "tbh",
            ]
        )
        formal_markers = sum(
            1
            for w in words
            if w.lower()
            in [
                "regarding",
                "furthermore",
                "therefore",
                "accordingly",
                "sincerely",
                "respectfully",
                "per",
                "pursuant",
            ]
        )
        formality = 0.5
        if informal_markers + formal_markers > 0:
            formality = formal_markers / (informal_markers + formal_markers)

        # --- Pattern counting ---
        # Aggregate regex hits across each pattern bank and normalise to
        # per-sentence rates so short and long messages are comparable.
        hedge_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.HEDGE_PATTERNS)
        assertion_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.ASSERTION_PATTERNS)
        profanity_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.PROFANITY_PATTERNS)
        # Raw punctuation counts — also normalised to per-sentence rates below.
        exclamation_count = text.count("!")
        question_count = text.count("?")
        ellipsis_count = text.count("...")

        # --- Emoji extraction ---
        emojis = self.EMOJI_PATTERN.findall(text)
        emoji_count = len(emojis)

        # --- Greeting/closing detection ---
        # Check the first and last sentences for conventional greetings and
        # sign-offs.  Tracking these reveals the user's preferred conventions
        # per contact and channel.
        first_line = sentences[0].lower().strip() if sentences else ""
        last_line = sentences[-1].lower().strip() if sentences else ""
        greeting = self._detect_greeting(first_line)
        closing = self._detect_closing(last_line)

        # --- Extended pattern detection ---
        # These feed into the higher-level LinguisticProfile fields that were
        # previously left at their zero/None defaults.

        # Humor markers reveal levity and playfulness.  Track the first
        # matching keyword per message so we can build a Top-5 list later.
        humor_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.HUMOR_PATTERNS)
        humor_type = self._detect_first(text, self.HUMOR_PATTERNS)

        # Affirmative patterns capture how the user says "yes".
        affirmative_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.AFFIRMATIVE_PATTERNS)
        affirmative_word = self._detect_first(text, self.AFFIRMATIVE_PATTERNS)

        # Negative patterns capture how the user declines or says "no".
        negative_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.NEGATIVE_PATTERNS)
        negative_word = self._detect_first(text, self.NEGATIVE_PATTERNS)

        # Gratitude patterns capture how the user expresses thanks.
        gratitude_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in self.GRATITUDE_PATTERNS)
        gratitude_word = self._detect_first(text, self.GRATITUDE_PATTERNS)

        # Oxford comma usage — counts list constructions with and without the
        # serial comma.  Needs ≥5 list instances before we can form a confident
        # preference signal.
        oxford_comma_count = len(self.OXFORD_COMMA_RE.findall(text))
        no_oxford_count = len(self.NO_OXFORD_RE.findall(text))

        # Capitalization analysis — counts sentence starts (upper vs. lower)
        # and ALL-CAPS emphasis words.  These feed into capitalization_style.
        cap_starts = sum(1 for s in sentences if s and s[0].isupper())
        lower_starts = sum(1 for s in sentences if s and s[0].islower())
        # ALL-CAPS words signal emphatic style (e.g., "I NEED this ASAP").
        # Exclude single-letter words and non-alpha tokens to avoid noise.
        all_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper() and w.isalpha())

        # Average word length is a proxy for vocabulary sophistication:
        # longer average word length → more complex/technical vocabulary.
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        # --- Build the signal ---
        # All rates are normalised per sentence so the profile stays comparable
        # regardless of message length.
        direction = "inbound" if is_inbound else "outbound"
        signal = {
            "type": "linguistic",
            "timestamp": event.get("timestamp"),
            "contact_id": contact_id,
            "channel": channel,
            "direction": direction,
            "metrics": {
                "word_count": word_count,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "unique_word_ratio": round(unique_word_ratio, 3),
                "avg_word_length": round(avg_word_length, 2),
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
                # Extended pattern fields — feed into the higher-level
                # LinguisticProfile fields (humor_markers, affirmative_patterns, etc.)
                "humor_count": humor_count,
                "humor_type": humor_type,
                "affirmative_count": affirmative_count,
                "affirmative_word": affirmative_word,
                "negative_count": negative_count,
                "negative_word": negative_word,
                "gratitude_count": gratitude_count,
                "gratitude_word": gratitude_word,
                "oxford_comma_count": oxford_comma_count,
                "no_oxford_count": no_oxford_count,
                "cap_starts": cap_starts,
                "lower_starts": lower_starts,
                "all_caps_words": all_caps_words,
                # Channel is included so profanity_contexts can identify
                # which channels the user swears in (e.g. "signal" vs "email").
                "channel": channel,
            },
        }

        signals.append(signal)

        # Persist into the appropriate profile based on direction.
        # Outbound → "linguistic" (user's own fingerprint, unchanged).
        # Inbound  → "linguistic_inbound" (per-contact incoming style).
        if is_inbound:
            self._update_inbound_profile(signal)
        else:
            self._update_profile(signal)

        return signals

    def _detect_greeting(self, text: str) -> Optional[str]:
        """Match the first sentence against known greeting patterns.

        Returns the canonical greeting label (e.g., "hey", "hello") or None.
        Uses startswith rather than substring matching to avoid false positives
        from greetings embedded mid-sentence.
        """
        greetings = {
            "hey": "hey",
            "hi": "hi",
            "hello": "hello",
            "good morning": "good morning",
            "morning": "morning",
            "yo": "yo",
            "sup": "sup",
        }
        for pattern, label in greetings.items():
            if text.startswith(pattern):
                return label
        return None

    def _detect_closing(self, text: str) -> Optional[str]:
        """Match the last sentence against known closing/sign-off patterns.

        Returns the canonical closing label or None.  Uses substring matching
        (``in``) because closings often appear alongside other words
        (e.g., "Best regards, Alex").
        """
        closings = {
            "best": "best",
            "cheers": "cheers",
            "thanks": "thanks",
            "talk soon": "talk soon",
            "take care": "take care",
            "regards": "regards",
            "sincerely": "sincerely",
        }
        for pattern, label in closings.items():
            if pattern in text:
                return label
        return None

    # Minimum samples required before per-contact averages are computed.
    # Below this threshold, a single atypical message would dominate the
    # average, producing misleading per-contact style guidance.
    _MIN_PER_CONTACT_SAMPLES = 3

    # Higher threshold for communication templates — templates feed AI-drafted
    # replies, so we need higher statistical confidence than for style averages.
    _MIN_TEMPLATE_SAMPLES = 5

    # Maximum unique contacts tracked in the linguistic_inbound profile.
    # With 13K+ qualifying events from potentially thousands of unique senders,
    # the JSON-serialized per_contact dict can grow very large.  When this
    # limit is reached, the least-recently-updated contacts are pruned so the
    # profile remains serializable within SQLite's comfortable range.
    _MAX_INBOUND_CONTACTS = 500

    def _update_profile(self, signal: dict):
        """Incrementally update the stored linguistic profile.

        The profile grows in four ways with every new sample:
          1. The raw ``samples`` ring buffer (capped at 500) stores per-message
             metric snapshots for computing running statistics.
          2. The ``per_contact`` dict maintains a separate ring buffer (capped
             at 100) per recipient so style-shift analysis is possible.
          3. Derived ``averages`` and ``common_greetings`` / ``common_closings``
             are recomputed from the full sample window on every update, keeping
             the profile always up-to-date for downstream consumers.
          4. ``per_contact_averages`` stores derived per-contact style summaries
             (mirroring what ``_update_inbound_profile`` does for inbound) so
             that ``ContextAssembler.assemble_draft_context()`` can tell the LLM
             "you write to Alice at formality 0.8" rather than "your global
             average formality is 0.4".  Only contacts with at least
             ``_MIN_PER_CONTACT_SAMPLES`` samples are included to avoid
             single-message noise.

        Example of per_contact_averages entry::

            {
                "alice@example.com": {
                    "formality": 0.78,
                    "avg_sentence_length": 14.2,
                    "hedge_rate": 0.05,
                    "assertion_rate": 0.12,
                    "exclamation_rate": 0.03,
                    "question_rate": 0.18,
                    "ellipsis_rate": 0.01,
                    "unique_word_ratio": 0.62,
                    "emoji_rate": 0.0,
                    "samples_count": 23,
                }
            }
        """
        existing = self.ums.get_signal_profile("linguistic")
        if existing:
            data = existing["data"]
        else:
            # First-time bootstrap: empty sample list and contact map.
            data = {"samples": [], "per_contact": {}}

        # Append the latest metrics snapshot to the global sample ring buffer.
        # Capping at 500 ensures bounded memory while retaining enough history
        # for statistically meaningful averages.
        data["samples"].append(signal["metrics"])
        if len(data["samples"]) > 500:
            data["samples"] = data["samples"][-500:]

        # Per-contact tracking — allows the system to detect that the user
        # writes formally to their manager but casually to close friends.
        contact = signal.get("contact_id")
        if contact:
            if contact not in data["per_contact"]:
                data["per_contact"][contact] = []
            data["per_contact"][contact].append(signal["metrics"])
            # Tighter cap per contact (100) since per-contact data is more
            # granular and there could be many contacts.
            if len(data["per_contact"][contact]) > 100:
                data["per_contact"][contact] = data["per_contact"][contact][-100:]

        # Recompute running averages across the entire sample window.  These
        # serve as the user's baseline linguistic fingerprint.
        samples = data["samples"]
        data["averages"] = {
            "avg_sentence_length": statistics.mean(s["avg_sentence_length"] for s in samples),
            "formality": statistics.mean(s["formality"] for s in samples),
            "hedge_rate": statistics.mean(s["hedge_rate"] for s in samples),
            "assertion_rate": statistics.mean(s["assertion_rate"] for s in samples),
            "exclamation_rate": statistics.mean(s["exclamation_rate"] for s in samples),
            # Sentence-ending question marks per sentence — distinguishes inquisitive
            # communicators ("Do you think we should X?") from declarative ones.
            "question_rate": statistics.mean(s["question_rate"] for s in samples),
            # Ellipsis usage ("I was thinking...") signals trailing thoughts,
            # informality, or hesitation; notable when consistently high.
            "ellipsis_rate": statistics.mean(s["ellipsis_rate"] for s in samples),
            # Type-token ratio across all messages reveals vocabulary diversity;
            # higher values indicate richer word choice.
            "unique_word_ratio": statistics.mean(s["unique_word_ratio"] for s in samples),
            # Emoji and profanity rates are normalised by word count so longer
            # messages don't inflate the averages.
            "emoji_rate": statistics.mean(s["emoji_count"] / max(s["word_count"], 1) for s in samples),
            "profanity_rate": statistics.mean(s["profanity_count"] / max(s["word_count"], 1) for s in samples),
            # New extended rates — use .get() with a 0 default so existing
            # samples (before this feature was added) are treated as zero-count
            # rather than raising a KeyError.
            "humor_rate": statistics.mean(s.get("humor_count", 0) / max(s["word_count"], 1) for s in samples),
            "affirmative_rate": statistics.mean(
                s.get("affirmative_count", 0) / max(s["word_count"], 1) for s in samples
            ),
            "negative_rate": statistics.mean(s.get("negative_count", 0) / max(s["word_count"], 1) for s in samples),
            "gratitude_rate": statistics.mean(s.get("gratitude_count", 0) / max(s["word_count"], 1) for s in samples),
        }

        # Surface the user's most-used greetings and closings (top 3 each).
        # The orchestrator can mirror these conventions when drafting replies
        # on the user's behalf.
        all_greetings = [s["greeting_detected"] for s in samples if s["greeting_detected"]]
        all_closings = [s["closing_detected"] for s in samples if s["closing_detected"]]
        data["common_greetings"] = [g for g, _ in Counter(all_greetings).most_common(3)]
        data["common_closings"] = [c for c, _ in Counter(all_closings).most_common(3)]

        # --- Derived higher-level profile fields ---
        # These correspond directly to LinguisticProfile model fields that
        # were previously left at their hardcoded defaults.

        # vocabulary_complexity: blends lexical diversity (unique_word_ratio)
        # with average word length (normalized to 0–1 on a 2–8 char scale).
        # Scores closer to 1.0 indicate richer, more complex language.
        avg_word_len = statistics.mean(s.get("avg_word_length", 4.0) for s in samples)
        avg_unique = statistics.mean(s["unique_word_ratio"] for s in samples)
        word_len_norm = min(1.0, max(0.0, (avg_word_len - 2.0) / 6.0))
        data["vocabulary_complexity"] = round(avg_unique * 0.6 + word_len_norm * 0.4, 3)

        # capitalization_style: inferred from the balance of sentence-starting
        # caps vs. lowercase starts and the frequency of ALL-CAPS emphasis words.
        total_starts = sum(s.get("cap_starts", 0) + s.get("lower_starts", 0) for s in samples)
        total_lower_starts = sum(s.get("lower_starts", 0) for s in samples)
        total_caps_words = sum(s.get("all_caps_words", 0) for s in samples)
        total_words_all = sum(s["word_count"] for s in samples)
        if total_starts > 0:
            lower_ratio = total_lower_starts / total_starts
            caps_emphasis_rate = total_caps_words / max(total_words_all, 1)
            if lower_ratio > 0.7:
                data["capitalization_style"] = "all_lower"
            elif caps_emphasis_rate > 0.05:
                data["capitalization_style"] = "all_caps_emphasis"
            else:
                data["capitalization_style"] = "standard"
        else:
            data["capitalization_style"] = "standard"

        # uses_oxford_comma: aggregate list-construction counts across all samples.
        # Require at least 5 list instances before committing to a preference so
        # a single message doesn't bias the result.
        total_oxford = sum(s.get("oxford_comma_count", 0) for s in samples)
        total_no_oxford = sum(s.get("no_oxford_count", 0) for s in samples)
        total_lists = total_oxford + total_no_oxford
        if total_lists >= 5:
            data["uses_oxford_comma"] = total_oxford >= total_no_oxford
        else:
            data["uses_oxford_comma"] = None  # insufficient data

        # Top-N characteristic vocabulary for each response category.
        # Mirrors how common_greetings / common_closings are built.
        all_humor = [s.get("humor_type") for s in samples if s.get("humor_type")]
        all_affirmative = [s.get("affirmative_word") for s in samples if s.get("affirmative_word")]
        all_negative = [s.get("negative_word") for s in samples if s.get("negative_word")]
        all_gratitude = [s.get("gratitude_word") for s in samples if s.get("gratitude_word")]
        data["top_humor_markers"] = [w for w, _ in Counter(all_humor).most_common(5)]
        data["top_affirmative_patterns"] = [w for w, _ in Counter(all_affirmative).most_common(5)]
        data["top_negative_patterns"] = [w for w, _ in Counter(all_negative).most_common(5)]
        data["top_gratitude_patterns"] = [w for w, _ in Counter(all_gratitude).most_common(5)]

        # Alias top_* keys to the canonical LinguisticProfile field names so
        # downstream consumers that deserialise into LinguisticProfile see real
        # data instead of empty defaults.  Keep the top_* keys for backward
        # compatibility with any code that reads them directly.
        data["humor_markers"] = data["top_humor_markers"]
        data["affirmative_patterns"] = data["top_affirmative_patterns"]
        data["negative_patterns"] = data["top_negative_patterns"]
        data["gratitude_patterns"] = data["top_gratitude_patterns"]

        # emoji_usage: per-emoji relative frequency dict (top 20).
        # Accumulates all emoji characters across the sample window and
        # normalises by total count so values sum to ~1.0.
        all_emojis = []
        for s in samples:
            all_emojis.extend(s.get("emojis_used", []))
        if all_emojis:
            emoji_counts = Counter(all_emojis)
            total = sum(emoji_counts.values())
            data["emoji_usage"] = {emoji: round(count / total, 4) for emoji, count in emoji_counts.most_common(20)}
        else:
            data["emoji_usage"] = {}

        # profanity_comfort: 0.0 (never swears) to 1.0 (frequently swears).
        # Derived from the average profanity_rate (profanity_count / word_count).
        # A rate of 0.02 (1 profanity per 50 words) is treated as full comfort.
        # This cap is a reasonable heuristic — typical casual speech with
        # frequent swearing rarely exceeds ~2% profanity density.
        profanity_rate = data["averages"].get("profanity_rate", 0.0)
        data["profanity_comfort"] = round(min(1.0, profanity_rate / 0.02), 4)

        # profanity_contexts: collect the channel/interaction_type from samples
        # where profanity was used, so the profile reveals *where* the user
        # swears (e.g. "signal" but not "email").  Capped at 10 to stay bounded.
        profanity_samples = [s for s in samples if s.get("profanity_count", 0) > 0]
        contexts = list(set(s.get("channel", "unknown") for s in profanity_samples))
        data["profanity_contexts"] = contexts[:10]

        # Map internal rate keys to LinguisticProfile canonical field names.
        # The averages dict uses short names like "hedge_rate" but the typed
        # LinguisticProfile model (models/user_model.py) expects "hedge_frequency"
        # etc.  We copy (not rename) so both the _rate keys and the _frequency
        # keys are available — existing consumers read _rate directly, while new
        # code that deserialises into LinguisticProfile gets non-default values.
        data["hedge_frequency"] = data["averages"].get("hedge_rate", 0.0)
        data["assertion_frequency"] = data["averages"].get("assertion_rate", 0.0)
        data["question_frequency"] = data["averages"].get("question_rate", 0.0)
        data["ellipsis_frequency"] = data["averages"].get("ellipsis_rate", 0.0)
        data["exclamation_rate"] = data["averages"].get("exclamation_rate", 0.0)
        data["avg_sentence_length"] = data["averages"].get("avg_sentence_length", 12.0)
        data["greeting_patterns"] = data.get("common_greetings", [])
        data["closing_patterns"] = data.get("common_closings", [])
        data["formality_spectrum"] = data["averages"].get("formality", 0.5)

        # Compute per-contact style averages from the per-contact ring buffers.
        # Only contacts with enough samples (_MIN_PER_CONTACT_SAMPLES) get an
        # entry — thin contacts fall back to the global averages in the draft
        # context rather than producing noisy single-message "averages".
        #
        # All metric keys mirror _update_inbound_profile() and the signal dict
        # so that ContextAssembler.assemble_draft_context() can read them with
        # the same field names regardless of direction.
        per_contact_avgs: dict = {}
        for cid, csamples in data["per_contact"].items():
            if len(csamples) < self._MIN_PER_CONTACT_SAMPLES:
                # Insufficient data — skip this contact entirely.
                continue
            per_contact_avgs[cid] = {
                "avg_sentence_length": statistics.mean(s["avg_sentence_length"] for s in csamples),
                "formality": statistics.mean(s["formality"] for s in csamples),
                "hedge_rate": statistics.mean(s["hedge_rate"] for s in csamples),
                "assertion_rate": statistics.mean(s["assertion_rate"] for s in csamples),
                "exclamation_rate": statistics.mean(s["exclamation_rate"] for s in csamples),
                "question_rate": statistics.mean(s["question_rate"] for s in csamples),
                "ellipsis_rate": statistics.mean(s["ellipsis_rate"] for s in csamples),
                "unique_word_ratio": statistics.mean(s["unique_word_ratio"] for s in csamples),
                # Normalise emoji by word count so shorter messages don't skew
                # the rate upward.
                "emoji_rate": statistics.mean(s["emoji_count"] / max(s["word_count"], 1) for s in csamples),
                # sample count lets callers gauge confidence (more = more reliable).
                "samples_count": len(csamples),
            }
        data["per_contact_averages"] = per_contact_avgs

        # Wire computed per-contact averages to the LinguisticProfile canonical
        # field name so the AI engine can look up per-contact style when drafting.
        data["style_by_contact"] = per_contact_avgs

        # Store communication templates for contacts with sufficient data so
        # the AI engine can draft messages matching per-contact writing style.
        self._store_per_contact_templates(data["per_contact"], per_contact_avgs)

        # Track how many samples are currently in the ring buffer so
        # LinguisticProfile.samples_analyzed reflects reality.
        data["samples_analyzed"] = len(data.get("samples", []))

        # Timestamp the profile update so LinguisticProfile.last_updated is
        # meaningful rather than always reflecting Python object creation time.
        data["last_updated"] = datetime.now(timezone.utc).isoformat()

        self.ums.update_signal_profile("linguistic", data)

    def _update_inbound_profile(self, signal: dict):
        """Incrementally update the inbound linguistic profile.

        Stores per-contact incoming style data in a separate ``linguistic_inbound``
        profile so it never pollutes the user's own linguistic fingerprint.

        The profile tracks:
          - ``per_contact``: per-sender ring buffer (capped at 100) of metric
            snapshots for style-shift detection ("Bob's tone changed this week").
          - ``per_contact_averages``: running averages per contact for quick
            comparison ("Alice writes formally, Carol writes casually").
          - ``per_contact_updated_at``: ISO timestamp of the most recent sample
            per contact, used for recency-based pruning when the contact count
            exceeds ``_MAX_INBOUND_CONTACTS``.

        Robustness measures:
          1. The entire method body is wrapped in a try/except so exceptions
             never propagate to the pipeline's catch-all, making root causes
             visible in the logs at the extractor level.
          2. A contact count cap (``_MAX_INBOUND_CONTACTS``) prunes the least-
             recently-updated contacts when too many unique senders accumulate,
             keeping the JSON payload within SQLite's comfortable range.
          3. Data size (bytes) and contact count are logged at DEBUG level
             before every write so operators can track profile growth over time.
          4. Post-write verification reads back the profile immediately after
             the write and logs CRITICAL if it is missing, surfacing silent
             write failures caused by WAL corruption, DB lock contention, or
             JSON serialization errors caught inside ``update_signal_profile``.
        """
        try:
            existing = self.ums.get_signal_profile("linguistic_inbound")
            if existing:
                data = existing["data"]
            else:
                data = {"per_contact": {}, "per_contact_averages": {}}

            # Ensure the recency-tracking dict exists for profiles written by
            # older code that did not include it.
            if "per_contact_updated_at" not in data:
                data["per_contact_updated_at"] = {}

            contact = signal.get("contact_id")
            if not contact:
                return

            # Append the latest metrics snapshot to this contact's ring buffer.
            if contact not in data["per_contact"]:
                data["per_contact"][contact] = []
            data["per_contact"][contact].append(signal["metrics"])
            if len(data["per_contact"][contact]) > 100:
                data["per_contact"][contact] = data["per_contact"][contact][-100:]

            # Record when this contact was last updated so we can rank contacts
            # by recency when the profile needs to be compacted.
            data["per_contact_updated_at"][contact] = datetime.now(timezone.utc).isoformat()

            # ── Contact count cap ─────────────────────────────────────────────
            # With 13K+ qualifying email events from potentially thousands of
            # unique senders, the JSON-serialized per_contact dict can exceed
            # SQLite's comfortable handling size.  When the contact count goes
            # over the limit, prune to the most-recently-updated contacts so
            # style data for active senders is always preserved.
            if len(data["per_contact"]) > self._MAX_INBOUND_CONTACTS:
                updated_at = data["per_contact_updated_at"]
                # Sort all tracked contacts by their last-updated timestamp,
                # most recent first.  Contacts missing from updated_at (legacy
                # entries) sort to the end and are pruned first.
                sorted_contacts = sorted(
                    data["per_contact"].keys(),
                    key=lambda c: updated_at.get(c, ""),
                    reverse=True,
                )
                keep = set(sorted_contacts[: self._MAX_INBOUND_CONTACTS])
                data["per_contact"] = {c: v for c, v in data["per_contact"].items() if c in keep}
                data["per_contact_averages"] = {
                    c: v for c, v in data["per_contact_averages"].items() if c in keep
                }
                data["per_contact_updated_at"] = {
                    c: v for c, v in updated_at.items() if c in keep
                }
                logger.debug(
                    "LinguisticExtractor: compacted linguistic_inbound from %d contacts "
                    "down to %d (cap=%d)",
                    len(sorted_contacts),
                    len(data["per_contact"]),
                    self._MAX_INBOUND_CONTACTS,
                )

            # Recompute running averages for this contact so downstream consumers
            # can quickly characterise a contact's communication style without
            # iterating the raw samples.
            #
            # IMPORTANT: All metrics must mirror what the signal dict stores (see
            # the ``extract()`` method above) so that downstream consumers like
            # ContextAssembler.assemble_draft_context() can reliably read them.
            # Previously ``question_rate``, ``ellipsis_rate``, and
            # ``unique_word_ratio`` were computed per-signal but never included
            # here, causing the draft context to always read 0.0 for every contact
            # despite 100K+ inbound samples.
            samples = data["per_contact"][contact]
            avgs: dict = {
                "avg_sentence_length": statistics.mean(s["avg_sentence_length"] for s in samples),
                "formality": statistics.mean(s["formality"] for s in samples),
                "hedge_rate": statistics.mean(s["hedge_rate"] for s in samples),
                "assertion_rate": statistics.mean(s["assertion_rate"] for s in samples),
                "exclamation_rate": statistics.mean(s["exclamation_rate"] for s in samples),
                # question_rate was computed per-signal but missing from averages —
                # the draft-context assembler reads this to detect inquisitive contacts.
                "question_rate": statistics.mean(s["question_rate"] for s in samples),
                # ellipsis_rate signals trailing thoughts and informality; helps
                # the draft context detect casual contacts who trail off with "...".
                "ellipsis_rate": statistics.mean(s["ellipsis_rate"] for s in samples),
                # unique_word_ratio (type-token ratio) measures vocabulary richness;
                # lets the system detect contacts with elaborate vs. simple word choice.
                "unique_word_ratio": statistics.mean(s["unique_word_ratio"] for s in samples),
                "emoji_rate": statistics.mean(s["emoji_count"] / max(s["word_count"], 1) for s in samples),
                "samples_count": len(samples),
            }

            # --- Derived higher-level fields (parity with outbound profile) ---
            # These give the AI context assembler the same richness for contacts'
            # writing style as it has for the user's own style.

            # vocabulary_complexity: blends lexical diversity (unique_word_ratio)
            # with average word length (normalized to 0–1 on a 2–8 char scale).
            # Uses .get() defaults so older samples without avg_word_length don't crash.
            avg_word_len = statistics.mean(s.get("avg_word_length", 4.0) for s in samples)
            avg_unique = statistics.mean(s["unique_word_ratio"] for s in samples)
            word_len_norm = min(1.0, max(0.0, (avg_word_len - 2.0) / 6.0))
            avgs["vocabulary_complexity"] = round(avg_unique * 0.6 + word_len_norm * 0.4, 3)

            # capitalization_style: inferred from the balance of sentence-starting
            # caps vs. lowercase starts and the frequency of ALL-CAPS emphasis words.
            total_starts = sum(s.get("cap_starts", 0) + s.get("lower_starts", 0) for s in samples)
            total_lower_starts = sum(s.get("lower_starts", 0) for s in samples)
            total_caps_words = sum(s.get("all_caps_words", 0) for s in samples)
            total_words_all = sum(s["word_count"] for s in samples)
            if total_starts > 0:
                lower_ratio = total_lower_starts / total_starts
                caps_emphasis_rate = total_caps_words / max(total_words_all, 1)
                if lower_ratio > 0.7:
                    avgs["capitalization_style"] = "all_lower"
                elif caps_emphasis_rate > 0.05:
                    avgs["capitalization_style"] = "all_caps_emphasis"
                else:
                    avgs["capitalization_style"] = "standard"
            else:
                avgs["capitalization_style"] = "standard"

            # Top-5 characteristic vocabulary per response category.  Mirrors
            # the outbound profile's humor_markers, affirmative_patterns, etc.
            all_humor = [s.get("humor_type") for s in samples if s.get("humor_type")]
            all_affirmative = [s.get("affirmative_word") for s in samples if s.get("affirmative_word")]
            all_negative = [s.get("negative_word") for s in samples if s.get("negative_word")]
            all_gratitude = [s.get("gratitude_word") for s in samples if s.get("gratitude_word")]
            avgs["humor_markers"] = [w for w, _ in Counter(all_humor).most_common(5)]
            avgs["affirmative_patterns"] = [w for w, _ in Counter(all_affirmative).most_common(5)]
            avgs["negative_patterns"] = [w for w, _ in Counter(all_negative).most_common(5)]
            avgs["gratitude_patterns"] = [w for w, _ in Counter(all_gratitude).most_common(5)]

            data["per_contact_averages"][contact] = avgs

            # ── Data size diagnostics ─────────────────────────────────────────
            # Serialize to measure size before writing so large payloads are
            # visible in logs before a potential silent write failure.
            data_json = json.dumps(data)
            data_size = len(data_json)
            contact_count = len(data["per_contact"])
            logger.debug(
                "LinguisticExtractor: writing linguistic_inbound profile "
                "(size=%d bytes, contacts=%d)",
                data_size,
                contact_count,
            )

            self.ums.update_signal_profile("linguistic_inbound", data)

            # ── Post-write verification ───────────────────────────────────────
            # Immediately read back to confirm the profile was persisted.  A
            # missing read-back indicates a silent write failure: WAL corruption,
            # database lock contention, or a JSON serialization error silently
            # caught inside ``update_signal_profile``'s try/except.  Logging
            # CRITICAL here surfaces the failure so operators can investigate
            # rather than discovering the problem indirectly from missing style
            # data in the draft-context assembler output.
            verify = self.ums.get_signal_profile("linguistic_inbound")
            if not verify:
                logger.critical(
                    "LinguisticExtractor: linguistic_inbound profile FAILED to persist "
                    "after write (size=%d bytes, contacts=%d, data_keys=%s)",
                    data_size,
                    contact_count,
                    list(data.keys()),
                )

        except Exception as exc:
            # Catch-all so a bug in this method never silently swallows the
            # real exception inside the pipeline's generic handler.  The contact
            # field helps identify whether the failure is contact-specific.
            logger.error(
                "LinguisticExtractor._update_inbound_profile raised unexpectedly "
                "(contact=%r): %s",
                signal.get("contact_id"),
                exc,
                exc_info=True,
            )

    def _store_per_contact_templates(self, per_contact_samples: dict, per_contact_avgs: dict):
        """Store communication templates derived from per-contact linguistic data.

        Iterates over contacts with sufficient outbound message samples and
        creates CommunicationTemplate entries capturing per-contact writing style
        (greeting, closing, formality, message length, emoji usage, tone notes).

        These templates enable the AI engine to draft messages that match the
        user's established writing style for each specific contact rather than
        using a one-size-fits-all global style.

        Template IDs are deterministic (hash of ``linguistic:{contact}:outbound``)
        so repeated calls for the same contact produce idempotent upserts.

        Args:
            per_contact_samples: Raw per-contact metric ring buffers from the
                linguistic profile (keyed by contact_id).
            per_contact_avgs: Pre-computed per-contact averages (keyed by
                contact_id) with formality, emoji_rate, etc.
        """
        for contact_id, avgs in per_contact_avgs.items():
            samples_count = avgs.get("samples_count", 0)
            if samples_count < self._MIN_TEMPLATE_SAMPLES:
                continue

            csamples = per_contact_samples.get(contact_id, [])
            if not csamples:
                continue

            # Extract per-contact greeting/closing patterns (most common)
            greetings = [s["greeting_detected"] for s in csamples if s.get("greeting_detected")]
            closings = [s["closing_detected"] for s in csamples if s.get("closing_detected")]
            top_greeting = Counter(greetings).most_common(1)[0][0] if greetings else None
            top_closing = Counter(closings).most_common(1)[0][0] if closings else None

            # Average word count as typical message length (in words)
            avg_word_count = statistics.mean(s["word_count"] for s in csamples)

            # Determine the most common channel for this contact
            channels = [s.get("channel", "email") for s in csamples if s.get("channel")]
            channel = Counter(channels).most_common(1)[0][0] if channels else "email"

            # Deterministic template ID — distinct from RelationshipExtractor
            # templates (which use "{address}:{channel}:{direction}" hashes)
            template_id = hashlib.sha256(
                f"linguistic:{contact_id}:outbound".encode()
            ).hexdigest()[:16]

            # Derive tone notes from linguistic feature averages
            tone_notes = []
            if avgs.get("hedge_rate", 0) > 0.1:
                tone_notes.append("tends to hedge")
            if avgs.get("assertion_rate", 0) > 0.1:
                tone_notes.append("tends to be assertive")
            if avgs.get("question_rate", 0) > 0.2:
                tone_notes.append("asks many questions")
            if avgs.get("exclamation_rate", 0) > 0.15:
                tone_notes.append("uses exclamations frequently")

            # Preserve phrase data that the relationship extractor may have
            # previously computed for this contact.  The linguistic extractor
            # does not analyse phrases itself, so without this merge the
            # INSERT OR REPLACE would overwrite common_phrases / avoids_phrases
            # with empty arrays, destroying the relationship extractor's work.
            existing_phrases = self._get_existing_phrase_data(contact_id, channel)

            template = {
                "id": template_id,
                "context": "linguistic_outbound",
                "contact_id": contact_id,
                "channel": channel,
                "greeting": top_greeting,
                "closing": top_closing,
                "formality": avgs.get("formality", 0.5),
                "typical_length": round(avg_word_count),
                "uses_emoji": avgs.get("emoji_rate", 0) > 0.01,
                "common_phrases": existing_phrases.get("common_phrases", []),
                "avoids_phrases": existing_phrases.get("avoids_phrases", []),
                "tone_notes": tone_notes,
                "example_message_ids": [],
                "samples_analyzed": samples_count,
            }

            try:
                self.ums.store_communication_template(template)
            except Exception as e:
                logger.warning("Failed to store linguistic template for %s: %s", contact_id, e)

    def _get_existing_phrase_data(self, contact_id: str, channel: str) -> dict:
        """Retrieve phrase data from existing templates for this contact.

        The relationship extractor computes common_phrases and avoids_phrases
        using exponential moving averages across messages.  This method looks
        up those values so the linguistic extractor can carry them forward
        rather than overwriting them with empty arrays.

        Checks both the linguistic extractor's own template (in case it was
        previously populated) and any relationship extractor templates for
        the same contact.

        Returns:
            Dict with ``common_phrases`` and ``avoids_phrases`` lists.
            Empty lists if no prior phrase data exists.
        """
        result: dict = {"common_phrases": [], "avoids_phrases": []}
        try:
            templates = self.ums.get_communication_templates(
                contact_id=contact_id, limit=10
            )
            for tmpl in templates:
                # Prefer the first template with non-empty phrase data
                if tmpl.get("common_phrases") and not result["common_phrases"]:
                    result["common_phrases"] = tmpl["common_phrases"]
                if tmpl.get("avoids_phrases") and not result["avoids_phrases"]:
                    result["avoids_phrases"] = tmpl["avoids_phrases"]
                if result["common_phrases"] and result["avoids_phrases"]:
                    break
        except Exception:
            logger.debug("Could not retrieve existing phrase data for %s", contact_id)
        return result
