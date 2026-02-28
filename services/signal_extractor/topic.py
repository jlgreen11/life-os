"""
Life OS — Topic Signal Extractor

Extracts topics and interests from messages, searches, and browsing.
Over time, builds a map of what the user cares about.

Marketing email filtering:
    Only genuine human communication (non-marketing email, direct messages, and
    user commands) is processed for topic extraction.  Marketing and automated
    sender emails are explicitly skipped because they flood the topic profile with
    promotional vocabulary (``offer``, ``shop``, ``holiday``, ``deal``, etc.) that
    has zero signal about the user's actual interests.  This prevents the semantic
    fact inferrer from producing garbage facts like ``expertise_email`` or
    ``interest_shop``.

    The same ``is_marketing_or_noreply`` predicate used by the relationship
    extractor and prediction engine is applied here.  Outbound email (``email.sent``)
    is always processed since users only write genuine emails, not automated ones.
    Direct messages (Signal, iMessage) are always processed since they are always
    human-to-human.
"""

from __future__ import annotations

import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser

from models.core import EventType
from services.signal_extractor.base import BaseExtractor
from services.signal_extractor.marketing_filter import is_marketing_or_noreply


class HTMLStripper(HTMLParser):
    """
    Strips HTML tags and extracts plain text from HTML content.

    This parser is used to clean email bodies that contain HTML markup,
    preventing HTML/CSS tokens (like 'nbsp', 'padding', 'tbody') from being
    extracted as topics and polluting the semantic memory layer.

    Example:
        >>> stripper = HTMLStripper()
        >>> stripper.feed('<p>Hello <b>world</b>!</p>')
        >>> stripper.get_text()
        'Hello world!'
    """

    # Block-level HTML elements that should have space inserted after them
    # to prevent word concatenation (e.g., </p><p> should insert space between)
    BLOCK_ELEMENTS = {
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr', 'td', 'th',
        'br', 'hr', 'blockquote', 'pre', 'section', 'article', 'header', 'footer'
    }

    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts = []
        self._in_style = False  # Track if we're inside a <style> tag
        self._in_script = False  # Track if we're inside a <script> tag

    def handle_starttag(self, tag, attrs):
        """Track when entering <style> or <script> tags to skip their content."""
        if tag == 'style':
            self._in_style = True
        elif tag == 'script':
            self._in_script = True

    def handle_endtag(self, tag):
        """Track when exiting <style> or <script> tags, add spacing for block elements."""
        if tag == 'style':
            self._in_style = False
        elif tag == 'script':
            self._in_script = False
        # Add space after block-level elements to preserve word boundaries
        elif tag in self.BLOCK_ELEMENTS:
            self.text_parts.append(' ')

    def handle_data(self, data: str):
        """Collect visible text content from HTML elements, excluding style/script."""
        # Skip CSS rules inside <style> tags and JavaScript inside <script> tags
        if not self._in_style and not self._in_script:
            self.text_parts.append(data)

    def get_text(self) -> str:
        """
        Return the extracted plain text.

        Joins all text parts, then normalizes whitespace to collapse
        multiple spaces/newlines to single space.
        """
        # Join with empty string to preserve natural word boundaries
        text = ''.join(self.text_parts)
        # Normalize whitespace: collapse runs of spaces/tabs/newlines to single space
        return re.sub(r'\s+', ' ', text).strip()


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

        HTML content is automatically stripped to prevent HTML/CSS tokens
        (like 'nbsp', 'padding', 'tbody', 'border') from being extracted as
        topics and polluting the semantic memory layer with garbage expertise.

        Marketing email filtering:
            Inbound emails (``email.received``) from marketing/automated senders
            are skipped entirely.  Without this guard, promotional vocabulary
            (``offer``, ``shop``, ``holiday``, ``deal``, ``rewards``, etc.) floods
            the topic profile and the semantic fact inferrer produces nonsensical
            facts like ``expertise_email`` or ``interest_shop``.

            Only genuine human email (where ``is_marketing_or_noreply()`` returns
            False) is processed.  Outbound email and direct messages are always
            processed — users only write genuine communication, never automated mail.

        Args:
            event: Normalised Life OS event dict with ``type``, ``payload``, etc.

        Returns:
            List containing a single topic signal dict, or empty list if the
            event carries no meaningful topical content.

        Example::

            extractor = TopicExtractor(ums)
            signals = extractor.extract({
                "type": "email.received",
                "payload": {"subject": "Meeting agenda", "body": "...", "from": "alice@example.com"},
                "timestamp": "2026-02-28T10:00:00Z",
            })
            # Returns [{"type": "topic", "topics": ["meeting", "agenda", ...], ...}]
        """
        payload = event.get("payload", {})

        # Skip marketing / automated inbound emails before doing any text work.
        # This is the primary fix for topic profile pollution: promotional emails
        # dominate inboxes and their vocabulary (shop, offer, holiday, rewards, etc.)
        # is completely uninformative about the user's real interests.
        #
        # We only apply this check to email.received — outbound email is always
        # genuine, and direct messages (Signal/iMessage) are always human-to-human.
        if event.get("type") == EventType.EMAIL_RECEIVED.value:
            from_addr = (
                payload.get("from")
                or payload.get("sender")
                or payload.get("from_address")
                or ""
            )
            if from_addr and is_marketing_or_noreply(from_addr, payload):
                # Marketing/automated sender — skip topic extraction entirely.
                # Returning an empty list is the correct no-op in the extractor
                # contract (BaseExtractor.process() checks for empty returns).
                return []

        text = payload.get("body", "") or payload.get("body_plain", "") or ""
        subject = payload.get("subject", "")

        # Strip HTML from email bodies to extract only meaningful text content.
        # Many emails contain HTML markup, and without stripping, HTML/CSS tokens
        # get extracted as "topics" and flow into semantic facts as fake "expertise".
        # This caused 18/18 expertise facts to be HTML garbage (nbsp, padding, etc.)
        # with 0.95 confidence, completely breaking Layer 2 semantic memory.
        if self._is_html(text):
            text = self._strip_html(text)

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

    def _is_html(self, text: str) -> bool:
        """
        Detect if text contains HTML markup.

        Uses a simple heuristic: if the text contains HTML tags (angle brackets
        with tag names), it's considered HTML. This catches the vast majority of
        HTML emails without requiring full HTML parsing.

        Excludes email addresses like <user@example.com> which use angle brackets
        but aren't HTML tags.

        Args:
            text: The text to check

        Returns:
            True if the text appears to contain HTML markup
        """
        # Match opening tags like <p>, <table>, <div>, etc.
        # This regex looks for < followed by a letter, then either > or space/attribute
        # This avoids matching email addresses like <user@example.com>
        return bool(re.search(r'<[a-zA-Z]+[\s>]', text))

    def _strip_html(self, html_text: str) -> str:
        """
        Strip HTML tags and extract plain text content.

        Processes HTML email bodies to remove all markup, leaving only the
        visible text content. This prevents HTML/CSS tokens from polluting
        the topic extraction and semantic memory layers.

        Args:
            html_text: HTML content to strip

        Returns:
            Plain text with HTML removed and entities decoded

        Example:
            >>> self._strip_html('<p>Order total: <b>$50</b>&nbsp;USD</p>')
            'Order total: $50 USD'
        """
        # Use HTMLStripper to extract text content
        stripper = HTMLStripper()
        try:
            stripper.feed(html_text)
            text = stripper.get_text()
        except Exception:
            # If HTML parsing fails (malformed HTML), fall back to regex strip
            # This is a safety net for edge cases where HTMLParser chokes
            text = re.sub(r'<[^>]+>', ' ', html_text)
            text = re.sub(r'\s+', ' ', text).strip()

        # Decode HTML entities (e.g., &nbsp; → space, &amp; → &)
        text = unescape(text)

        return text

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
