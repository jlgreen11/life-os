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

import re
import statistics
from collections import Counter
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


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
        r'\bmaybe\b', r'\bperhaps\b', r'\bI think\b', r'\bsort of\b',
        r'\bkind of\b', r'\bpossibly\b', r'\bmight\b', r'\bcould be\b',
        r'\bI guess\b', r'\bnot sure\b', r'\bprobably\b',
    ]

    # Assertion patterns signal directness / high confidence.
    ASSERTION_PATTERNS = [
        r'\bwe need to\b', r'\bwe must\b', r'\bthis has to\b',
        r'\bdefinitely\b', r'\bclearly\b', r'\bobviously\b',
        r'\bwithout a doubt\b', r'\babsolutely\b',
    ]

    # Profanity patterns — tracked not to censor, but to gauge emotional
    # intensity and how the user's register shifts across contacts/channels.
    PROFANITY_PATTERNS = [
        r'\bdamn\b', r'\bhell\b', r'\bshit\b', r'\bfuck\b',
        r'\bass\b', r'\bbullshit\b', r'\bcrap\b',
    ]

    # Broad Unicode emoji regex covering emoticons, symbols, pictographs,
    # transport icons, and flag sequences.  Used to count emoji density, which
    # is a signal of informality and emotional expressiveness.
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
        sentences = re.split(r'[.!?]+', text)
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
        # Aggregate regex hits across each pattern bank and normalise to
        # per-sentence rates so short and long messages are comparable.
        hedge_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.HEDGE_PATTERNS
        )
        assertion_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.ASSERTION_PATTERNS
        )
        profanity_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in self.PROFANITY_PATTERNS
        )
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
            "hey": "hey", "hi": "hi", "hello": "hello",
            "good morning": "good morning", "morning": "morning",
            "yo": "yo", "sup": "sup",
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
            "best": "best", "cheers": "cheers", "thanks": "thanks",
            "talk soon": "talk soon", "take care": "take care",
            "regards": "regards", "sincerely": "sincerely",
        }
        for pattern, label in closings.items():
            if pattern in text:
                return label
        return None

    # Minimum samples required before per-contact averages are computed.
    # Below this threshold, a single atypical message would dominate the
    # average, producing misleading per-contact style guidance.
    _MIN_PER_CONTACT_SAMPLES = 3

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
        }

        # Surface the user's most-used greetings and closings (top 3 each).
        # The orchestrator can mirror these conventions when drafting replies
        # on the user's behalf.
        all_greetings = [s["greeting_detected"] for s in samples if s["greeting_detected"]]
        all_closings = [s["closing_detected"] for s in samples if s["closing_detected"]]
        data["common_greetings"] = [g for g, _ in Counter(all_greetings).most_common(3)]
        data["common_closings"] = [c for c, _ in Counter(all_closings).most_common(3)]

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
                "avg_sentence_length": statistics.mean(
                    s["avg_sentence_length"] for s in csamples
                ),
                "formality": statistics.mean(s["formality"] for s in csamples),
                "hedge_rate": statistics.mean(s["hedge_rate"] for s in csamples),
                "assertion_rate": statistics.mean(
                    s["assertion_rate"] for s in csamples
                ),
                "exclamation_rate": statistics.mean(
                    s["exclamation_rate"] for s in csamples
                ),
                "question_rate": statistics.mean(
                    s["question_rate"] for s in csamples
                ),
                "ellipsis_rate": statistics.mean(
                    s["ellipsis_rate"] for s in csamples
                ),
                "unique_word_ratio": statistics.mean(
                    s["unique_word_ratio"] for s in csamples
                ),
                # Normalise emoji by word count so shorter messages don't skew
                # the rate upward.
                "emoji_rate": statistics.mean(
                    s["emoji_count"] / max(s["word_count"], 1) for s in csamples
                ),
                # sample count lets callers gauge confidence (more = more reliable).
                "samples_count": len(csamples),
            }
        data["per_contact_averages"] = per_contact_avgs

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
        """
        existing = self.ums.get_signal_profile("linguistic_inbound")
        if existing:
            data = existing["data"]
        else:
            data = {"per_contact": {}, "per_contact_averages": {}}

        contact = signal.get("contact_id")
        if not contact:
            return

        # Append the latest metrics snapshot to this contact's ring buffer.
        if contact not in data["per_contact"]:
            data["per_contact"][contact] = []
        data["per_contact"][contact].append(signal["metrics"])
        if len(data["per_contact"][contact]) > 100:
            data["per_contact"][contact] = data["per_contact"][contact][-100:]

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
        data["per_contact_averages"][contact] = {
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

        self.ums.update_signal_profile("linguistic_inbound", data)
