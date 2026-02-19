"""
Life OS — Relationship Signal Extractor

Builds the relationship graph — who matters, how they're connected,
and the dynamics of each relationship.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from datetime import datetime, timezone

from models.core import EventType
from services.signal_extractor.base import BaseExtractor
from services.signal_extractor.marketing_filter import is_marketing_or_noreply

logger = logging.getLogger(__name__)


def _compute_frequency_days(timestamps: list[str]) -> float | None:
    """Compute the average number of days between consecutive interactions.

    Used to populate ``contacts.contact_frequency_days`` — a convenience
    metric that tells the system (and the LLM) how often the user typically
    interacts with a given contact, without requiring a traversal of the
    signal profile ring buffer.

    Args:
        timestamps: List of ISO 8601 timestamp strings in insertion order
            (the ring buffer stored in the relationship signal profile).

    Returns:
        Average gap in days between consecutive interactions, or ``None``
        if fewer than two timestamps are present (can't compute a gap from
        a single data point).

    Example::

        >>> _compute_frequency_days([
        ...     "2026-02-01T10:00:00Z",
        ...     "2026-02-08T10:00:00Z",   # +7 days
        ...     "2026-02-15T10:00:00Z",   # +7 days
        ... ])
        7.0
    """
    if len(timestamps) < 2:
        return None

    try:
        # Parse and sort the ring-buffer entries; in practice they arrive
        # in chronological order, but sorting guards against edge cases.
        parsed = sorted(
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
            for ts in timestamps
            if ts
        )
        if len(parsed) < 2:
            return None

        # Compute gaps between every pair of consecutive timestamps and
        # filter out negative/zero gaps (clock skew or duplicate entries).
        gaps_seconds = [
            (parsed[i] - parsed[i - 1]).total_seconds()
            for i in range(1, len(parsed))
            if (parsed[i] - parsed[i - 1]).total_seconds() > 0
        ]
        if not gaps_seconds:
            return None

        # Convert from seconds to days.
        return sum(gaps_seconds) / len(gaps_seconds) / 86_400.0

    except Exception:
        return None


class RelationshipExtractor(BaseExtractor):
    """
    Builds the relationship graph — who matters, how they're connected,
    and the dynamics of each relationship.
    """

    def can_process(self, event: dict) -> bool:
        # Relationship mapping requires both directions: inbound messages tell
        # us who is reaching out, outbound messages tell us who the user
        # chooses to engage with.  Both are needed to compute reciprocity and
        # directional interaction ratios.
        return event.get("type") in [
            EventType.EMAIL_RECEIVED.value,
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.MESSAGE_SENT.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        """Extract per-contact interaction signals from a communication event.

        For outbound events the contacts are the recipients (to_addresses);
        for inbound events the contact is the sender (from_address).  A
        separate signal is emitted for each contact so multi-recipient emails
        correctly update every contact's profile.
        """
        payload = event.get("payload", {})
        event_type = event.get("type", "")

        # CRITICAL: Use the actual message timestamp (from email Date header or
        # message sent_at field) rather than the event sync timestamp. This is
        # essential for accurate relationship frequency analysis.
        #
        # Without this fix, all interactions appear to happen at database sync
        # time, causing gaps between interactions to collapse to ~0 days. This
        # breaks relationship maintenance predictions which depend on measuring
        # the real time between communications (e.g., "you usually email Alice
        # every 14 days, but it's been 30 days").
        #
        # Priority order:
        # 1. payload.email_date (Gmail connector, actual Date header)
        # 2. payload.sent_at / received_at (other connectors)
        # 3. event.timestamp (fallback to sync time if no actual date available)
        actual_timestamp = (
            payload.get("email_date") or
            payload.get("sent_at") or
            payload.get("received_at") or
            event.get("timestamp", "")
        )

        signals = []

        # Resolve the contact address(es) based on message direction.
        # Outbound: the user wrote to these people.
        # Inbound: someone wrote to the user.
        if "sent" in event_type.lower():
            addresses = payload.get("to_addresses", [])
        else:
            addresses = [payload.get("from_address")] if payload.get("from_address") else []

        for address in addresses:
            if not address:
                continue

            # Filter out marketing emails, no-reply addresses, and automated senders.
            # These pollute the relationship graph with non-human "relationships" that
            # should never generate relationship maintenance predictions. Without this
            # filter, the system tracks marketing senders like callofduty@comms.activision.com
            # as legitimate contacts, wasting storage and breaking relationship predictions.
            if is_marketing_or_noreply(address, payload):
                continue

            # Each signal captures a single interaction data point.  Downstream
            # aggregation in _update_contact_profiles turns these into running
            # statistics per contact.
            signal = {
                "type": "relationship_interaction",
                "timestamp": actual_timestamp,
                "contact_address": address,
                "direction": "outbound" if "sent" in event_type.lower() else "inbound",
                "channel": payload.get("channel", event.get("source", "unknown")),
                # Message length as a proxy for investment in the conversation.
                "message_length": len(payload.get("body", "") or ""),
                # Whether the message contains action items — indicates a
                # task-oriented (potentially professional) relationship.
                "has_action_items": bool(payload.get("action_items")),
                "sentiment": payload.get("sentiment"),
                "is_reply": payload.get("is_reply", False),
            }
            signals.append(signal)

        # Merge these interaction signals into persisted per-contact profiles.
        self._update_contact_profiles(signals)

        # Extract communication templates from ALL messages (both directions).
        # Outbound templates capture the user's writing style per contact/channel.
        # Inbound templates capture how each contact writes to the user.
        is_outbound = "sent" in event_type.lower()
        self._extract_communication_templates(event, addresses, is_outbound)

        return signals

    def _update_contact_profiles(self, signals: list[dict]):
        """Update contact frequency and interaction patterns.

        Each contact gets a profile dict that tracks:
          - interaction_count / inbound_count / outbound_count:
                Total and directional tallies.  The inbound/outbound ratio
                reveals reciprocity — e.g., a contact who always initiates
                but rarely receives replies may be deprioritised by the user.
          - channels_used:
                Which communication channels are active with this contact
                (email, Slack, SMS, etc.).  Multi-channel contacts tend to be
                closer relationships.
          - avg_message_length:
                Running average updated incrementally.  Longer messages
                correlate with deeper engagement.
          - interaction_timestamps:
                A ring buffer (last 100) used to compute interaction frequency,
                detect dormant relationships, and spot contact-frequency changes.
          - response_times_seconds:
                A ring buffer (last 50) tracking how long the user took to reply
                to this contact's messages. Used to compute avg_response_time_seconds.
          - avg_response_time_seconds:
                Running average of user's response time to this contact. Used by
                the semantic fact inferrer to identify high-priority relationships
                (fast responses = high priority).
        """
        existing = self.ums.get_signal_profile("relationships")
        data = existing["data"] if existing else {"contacts": {}}

        for signal in signals:
            addr = signal["contact_address"]
            # Bootstrap a new profile for contacts we have not seen before.
            if addr not in data["contacts"]:
                data["contacts"][addr] = {
                    "interaction_count": 0,
                    "inbound_count": 0,
                    "outbound_count": 0,
                    "channels_used": [],
                    "avg_message_length": 0,
                    "last_interaction": None,
                    "last_inbound_timestamp": None,
                    "interaction_timestamps": [],
                    "response_times_seconds": [],
                    "avg_response_time_seconds": None,
                }

            profile = data["contacts"][addr]
            profile["interaction_count"] += 1
            # Track direction so we can compute reciprocity ratios later.
            if signal["direction"] == "inbound":
                profile["inbound_count"] += 1
                # Store the timestamp of the last inbound message so we can
                # calculate response time when the user replies
                profile["last_inbound_timestamp"] = signal["timestamp"]
            else:
                profile["outbound_count"] += 1

                # Calculate response time if this is a reply to a recent inbound
                if profile.get("last_inbound_timestamp") and signal.get("is_reply"):
                    try:
                        from datetime import datetime
                        inbound_time = datetime.fromisoformat(profile["last_inbound_timestamp"].replace('Z', '+00:00'))
                        outbound_time = datetime.fromisoformat(signal["timestamp"].replace('Z', '+00:00'))
                        response_seconds = (outbound_time - inbound_time).total_seconds()

                        # Only track positive response times (sanity check)
                        if response_seconds > 0:
                            if "response_times_seconds" not in profile:
                                profile["response_times_seconds"] = []
                            profile["response_times_seconds"].append(response_seconds)

                            # Cap response time history at 50 entries
                            if len(profile["response_times_seconds"]) > 50:
                                profile["response_times_seconds"] = profile["response_times_seconds"][-50:]

                            # Recompute average response time from the ring buffer
                            if profile["response_times_seconds"]:
                                profile["avg_response_time_seconds"] = (
                                    sum(profile["response_times_seconds"]) / len(profile["response_times_seconds"])
                                )
                    except Exception:
                        pass  # Gracefully skip if timestamp parsing fails

            # Record new channels — deduplicated to keep the list compact.
            if signal["channel"] not in profile["channels_used"]:
                profile["channels_used"].append(signal["channel"])

            profile["last_interaction"] = signal["timestamp"]
            profile["interaction_timestamps"].append(signal["timestamp"])

            # Cap timestamp history at 100 entries to bound storage while
            # retaining enough data points for meaningful frequency analysis.
            if len(profile["interaction_timestamps"]) > 100:
                profile["interaction_timestamps"] = profile["interaction_timestamps"][-100:]

            # Incremental running average for message length.  Uses the
            # formula: new_avg = ((old_avg * (n-1)) + new_value) / n
            # to avoid storing all historical message lengths.
            n = profile["interaction_count"]
            profile["avg_message_length"] = (
                (profile["avg_message_length"] * (n - 1) + signal["message_length"]) / n
            )

        self.ums.update_signal_profile("relationships", data)

        # Denormalize key metrics back to the contacts table so that contact
        # records are self-contained and fast to query without joining signal
        # profiles.  See _sync_contact_metrics() for details.
        self._sync_contact_metrics(data["contacts"])

    def _sync_contact_metrics(self, contacts_data: dict) -> None:
        """Denormalize relationship metrics into the contacts table.

        The relationship signal profile (stored in ``user_model.db``) is the
        authoritative source for per-contact interaction data.  However, the
        ``contacts`` table in ``entities.db`` exposes three columns that mirror
        a subset of this data for convenience:

          - ``typical_response_time`` — user's average reply latency in seconds
          - ``last_contact`` — ISO timestamp of the most recent interaction
          - ``contact_frequency_days`` — average days between interactions

        These columns were defined in the schema but were never populated,
        making contact records incomplete.  This method closes that gap by
        writing the computed values back after every signal-profile update.

        The lookup from email address → ``contact_id`` is performed in a single
        batch query via ``contact_identifiers`` to avoid N+1 database
        round-trips.  Addresses with no matching contact record are silently
        skipped (they are typically unrecognised senders or marketing addresses
        that slipped past the filter before reaching this layer).

        Fail-open: any exception here is logged and swallowed.  The signal
        profile remains the authoritative source; the contacts-table columns
        are a best-effort cache only.

        Args:
            contacts_data: Mapping of email address → profile dict, as held
                in ``relationships_profile["data"]["contacts"]``.
        """
        if not contacts_data:
            return

        now = datetime.now(timezone.utc).isoformat()

        try:
            with self.db.get_connection("entities") as conn:
                # Fetch all matching contact IDs in one round-trip.
                placeholders = ",".join("?" * len(contacts_data))
                rows = conn.execute(
                    f"""SELECT lower(identifier), contact_id
                          FROM contact_identifiers
                         WHERE identifier_type = 'email'
                           AND lower(identifier) IN ({placeholders})""",
                    [addr.lower() for addr in contacts_data.keys()],
                ).fetchall()

                email_to_contact_id = {row[0]: row[1] for row in rows}

                for addr, profile in contacts_data.items():
                    contact_id = email_to_contact_id.get(addr.lower())
                    if not contact_id:
                        continue  # No matching entity — skip silently.

                    freq_days = _compute_frequency_days(
                        profile.get("interaction_timestamps", [])
                    )

                    conn.execute(
                        """UPDATE contacts
                              SET typical_response_time  = ?,
                                  last_contact           = ?,
                                  contact_frequency_days = ?,
                                  updated_at             = ?
                            WHERE id = ?""",
                        (
                            profile.get("avg_response_time_seconds"),
                            profile.get("last_interaction"),
                            freq_days,
                            now,
                            contact_id,
                        ),
                    )

        except Exception:
            # Metric sync is best-effort; never crash the extraction pipeline.
            logger.exception(
                "Failed to sync relationship metrics to contacts table; "
                "signal profile remains authoritative"
            )

    def _extract_communication_templates(self, event: dict, addresses: list[str], is_outbound: bool):
        """Extract communication style templates from messages (both directions).

        Analyzes messages to learn writing patterns per contact/channel:
        - Greeting and closing phrases (e.g., "Hey" vs "Dear" vs none)
        - Formality level (0.0 = casual, 1.0 = formal)
        - Typical message length
        - Emoji usage patterns
        - Common phrases and words
        - Tone indicators

        Templates are stored separately for each direction:
        - Outbound (user_to_contact): How the user writes TO contacts
        - Inbound (contact_to_user): How contacts write TO the user

        This enables:
        1. Style-matching when drafting replies (mirror the contact's style)
        2. Detecting formality mismatches (contact writes casually, user responds formally)
        3. Learning relationship-specific communication patterns
        4. Better prediction of incoming message characteristics

        Templates are updated incrementally as more samples accumulate. Each
        contact-channel-direction tuple gets its own template ID.

        Args:
            event: The message event (sent or received)
            addresses: List of contact addresses (recipients for outbound, sender for inbound)
            is_outbound: True if user sent this message, False if user received it
        """
        payload = event.get("payload", {})
        channel = payload.get("channel", event.get("source", "email"))
        body = payload.get("body_plain") or payload.get("body", "")

        if not body or len(body.strip()) < 10:
            # Skip very short messages — not enough data for style analysis
            return

        for address in addresses:
            if not address:
                continue

            # Filter out marketing emails — don't learn communication templates
            # from bulk senders. Templates are only useful for human contacts.
            if is_marketing_or_noreply(address, payload):
                continue

            # Generate deterministic template ID from contact + channel + direction
            # Direction is part of the ID so we store separate templates for
            # how the user writes TO a contact vs how the contact writes TO the user
            direction_suffix = "out" if is_outbound else "in"
            template_id = hashlib.sha256(
                f"{address}:{channel}:{direction_suffix}".encode()
            ).hexdigest()[:16]

            # Load existing template or bootstrap a new one
            existing = self._get_existing_template(template_id)
            samples_count = existing.get("samples_analyzed", 0)

            # Extract style features from this message
            greeting = self._extract_greeting(body)
            closing = self._extract_closing(body)
            formality = self._calculate_formality(body)
            message_length = len(body)
            # Broad emoji detection covering emoticons, symbols, pictographs, etc.
            uses_emoji = bool(re.search(
                r'[\U0001F600-\U0001F64F'  # Emoticons
                r'\U0001F300-\U0001F5FF'    # Symbols & pictographs
                r'\U0001F680-\U0001F6FF'    # Transport & map symbols
                r'\U0001F1E0-\U0001F1FF'    # Flags
                r'\U00002702-\U000027B0'    # Dingbats
                r'\U000024C2-\U0001F251]',  # Enclosed characters
                body
            ))
            words = self._extract_words(body)

            # Incremental update: blend new sample with existing template
            # using exponential moving average (weight recent samples more)
            alpha = 0.3  # Learning rate — higher = adapt faster to style changes

            # Set context based on direction for downstream filtering
            context = "user_to_contact" if is_outbound else "contact_to_user"

            template = {
                "id": template_id,
                "context": context,
                "contact_id": address,
                "channel": channel,
                "greeting": greeting if greeting else existing.get("greeting"),
                "closing": closing if closing else existing.get("closing"),
                # Blend formality scores using exponential moving average
                "formality": (
                    formality * alpha + existing.get("formality", 0.5) * (1 - alpha)
                    if samples_count > 0 else formality
                ),
                # Blend message length using exponential moving average
                "typical_length": (
                    message_length * alpha + existing.get("typical_length", 50.0) * (1 - alpha)
                    if samples_count > 0 else float(message_length)
                ),
                # Emoji usage: true if used in any recent sample (sticky flag)
                "uses_emoji": uses_emoji or existing.get("uses_emoji", False),
                # Top 10 most frequent words (excluding stop words)
                "common_phrases": self._merge_phrases(
                    existing.get("common_phrases", []),
                    words,
                    max_phrases=10
                ),
                "avoids_phrases": existing.get("avoids_phrases", []),  # Not auto-detected yet
                "tone_notes": existing.get("tone_notes", []),  # Could add sentiment here
                "example_message_ids": (
                    existing.get("example_message_ids", []) + [event.get("id")]
                )[-10:],  # Keep last 10 example IDs
                "samples_analyzed": samples_count + 1,
            }

            # Persist the updated template
            # (Telemetry is published by the UserModelStore itself)
            self.ums.store_communication_template(template)

    def _get_existing_template(self, template_id: str) -> dict:
        """Retrieve existing template from database or return empty dict."""
        try:
            with self.db.get_connection("user_model") as conn:
                row = conn.execute(
                    """SELECT * FROM communication_templates WHERE id = ?""",
                    (template_id,)
                ).fetchone()

                if row:
                    import json
                    return {
                        "id": row["id"],
                        "context": row["context"],
                        "contact_id": row["contact_id"],
                        "channel": row["channel"],
                        "greeting": row["greeting"],
                        "closing": row["closing"],
                        "formality": row["formality"],
                        "typical_length": row["typical_length"],
                        "uses_emoji": bool(row["uses_emoji"]),
                        "common_phrases": json.loads(row["common_phrases"]),
                        "avoids_phrases": json.loads(row["avoids_phrases"]),
                        "tone_notes": json.loads(row["tone_notes"]),
                        "example_message_ids": json.loads(row["example_message_ids"]),
                        "samples_analyzed": row["samples_analyzed"],
                    }
        except Exception:
            pass  # Return empty dict on any error

        return {}

    def _extract_greeting(self, body: str) -> str | None:
        """Extract greeting phrase from message opening.

        Looks for common greeting patterns in the first 100 characters:
        - "Hi/Hey/Hello [Name]"
        - "Dear [Name]"
        - "[Name]," (direct address)

        Returns:
            Greeting phrase or None if no clear greeting detected
        """
        # Check first 100 chars for greeting patterns
        opening = body[:100].strip()
        lines = opening.split('\n')
        first_line = lines[0] if lines else ""

        # Common greeting patterns (case-insensitive)
        greeting_patterns = [
            r'^(Hi|Hey|Hello|Dear|Greetings|Good\s+(?:morning|afternoon|evening))\b',
            r'^([A-Z][a-z]+),',  # Name with comma
        ]

        for pattern in greeting_patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                # Return the matched greeting (up to 30 chars)
                return match.group(0)[:30]

        return None

    def _extract_closing(self, body: str) -> str | None:
        """Extract closing phrase from message ending.

        Looks for common closing patterns in the last 150 characters:
        - "Thanks/Thank you"
        - "Best/Best regards/Regards"
        - "Sincerely/Cheers/Talk soon"

        Returns:
            Closing phrase or None if no clear closing detected
        """
        # Check last 150 chars for closing patterns
        ending = body[-150:].strip()
        lines = ending.split('\n')

        # Common closing patterns (look in last 3 lines, prioritize formal closings)
        # Check patterns from most formal to least to prefer "Best regards" over "Thanks"
        closing_patterns = [
            r'\b(Best\s+regards|Kind\s+regards|Warm\s+regards)',
            r'\b(Sincerely|Respectfully)',
            r'\b(Best|Regards)',
            r'\b(Cheers|Talk\s+soon|Take\s+care|See\s+you)',
            r'\b(Thank\s+you(?:\s+(?:so\s+much|very\s+much|again))?|Thanks(?:\s+(?:so\s+much|again))?)',
        ]

        # Look at last 3 lines in reverse order (most recent first)
        for line in reversed(lines[-3:]):
            for pattern in closing_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0)[:30]

        return None

    def _calculate_formality(self, body: str) -> float:
        """Calculate formality score based on linguistic features.

        Formality indicators:
        - High formality: long sentences, no contractions, professional vocabulary
        - Low formality: contractions, short sentences, casual language

        Returns:
            Float 0.0-1.0 where 0.0 = very casual, 1.0 = very formal
        """
        score = 0.5  # Neutral baseline

        # Check for contractions (casual indicator)
        contractions = len(re.findall(r"\b\w+'\w+\b", body))
        if contractions > 0:
            score -= 0.2

        # Check for formal greetings
        if re.search(r'\bDear\b', body, re.IGNORECASE):
            score += 0.2

        # Check for casual greetings
        if re.search(r'\b(Hey|Yo)\b', body, re.IGNORECASE):
            score -= 0.2

        # Check for professional closings
        if re.search(r'\b(Sincerely|Regards|Respectfully)\b', body, re.IGNORECASE):
            score += 0.15

        # Check for casual closings
        if re.search(r'\b(Cheers|Later|xo)\b', body, re.IGNORECASE):
            score -= 0.15

        # Sentence length (longer = more formal)
        sentences = re.split(r'[.!?]+', body)
        avg_sentence_len = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        if avg_sentence_len > 20:
            score += 0.1
        elif avg_sentence_len < 10:
            score -= 0.1

        # Clamp to 0.0-1.0 range
        return max(0.0, min(1.0, score))

    def _extract_words(self, body: str) -> list[str]:
        """Extract meaningful words from message body.

        Filters out common stop words and returns words that characterize
        the user's vocabulary and style.

        Returns:
            List of significant words (lowercase, alphabetic only)
        """
        # Common English stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them'
        }

        # Extract words (alphabetic only, 3+ chars)
        words = re.findall(r'\b[a-z]{3,}\b', body.lower())

        # Filter out stop words
        return [w for w in words if w not in stop_words]

    def _merge_phrases(self, existing: list[str], new_words: list[str],
                       max_phrases: int = 10) -> list[str]:
        """Merge new words into existing common phrases list.

        Uses word frequency to maintain the top N most common phrases.

        Args:
            existing: Current list of common phrases
            new_words: New words from the latest message
            max_phrases: Maximum number of phrases to keep

        Returns:
            Updated list of top common phrases
        """
        # Count word frequencies
        word_counts = Counter(existing)
        word_counts.update(new_words)

        # Return top N most common
        return [word for word, _ in word_counts.most_common(max_phrases)]

    # Extractor-specific extra patterns applied in addition to the shared baseline.
    # These were already present in the relationship extractor before this refactor
    # but are intentionally NOT in the prediction engine (where they could cause
    # false positives on legitimate human addresses like team@small-startup.io or
    # ens@usgs.gov for users who actually work there).
    _EXTRA_LOCALPARTS: tuple[str, ...] = (
        "emails@",       # Extractor had "emails@" (engine only has "email@")
        "acerewards@",   # Ace Hardware loyalty — extractor-specific
        "sales@", "sale@", "shop@", "store@", "merchant@",
        "concierge@",
        "flyers@", "flyer@",
        "partners@", "partner@",
        "team@",         # Generic team addresses (e.g. team@kickstarter)
        "ens@",          # Emergency Notification System (e.g. ens@usgs.gov)
        "ouch@",
        "events@", "event@",
        "uber@", "lyft@", "doordash@", "grubhub@",
        "spices@",
    )
    _EXTRA_DOMAIN_PATTERNS: tuple[str, ...] = (
        # @mail. was removed from the prediction engine in iteration 160 because
        # it incorrectly blocked Gmail/Hotmail/Protonmail users.  The relationship
        # extractor keeps it because those providers never appear as @mail.*
        # subdomains — only bulk-sender infrastructure does (mail.fidelity.com,
        # mail.schwab.com, mail.instagram.com).
        "@mail.",
        # Extractor-specific e-commerce and connection subdomains
        "@ecomm.", "@shop.", "@store.",
        "@iemail.",
        "@webstaurant",
        "@e1.",
        "@connect.",
    )

    @staticmethod
    def _is_marketing_or_noreply(from_addr: str, payload: dict) -> bool:
        """Marketing/automated-sender check for the relationship extractor.

        Applies the shared baseline from
        services.signal_extractor.marketing_filter.is_marketing_or_noreply()
        PLUS additional extractor-specific patterns that are stricter than
        what the prediction engine needs.

        The stricter patterns (team@, ens@, @mail., etc.) were deliberately
        kept out of the shared module to preserve existing prediction-engine
        behavior; they are applied here via extra_localparts /
        extra_domain_patterns so the extractor continues to enforce them while
        the shared module remains the single source of truth for all patterns
        that both components share.
        """
        return is_marketing_or_noreply(
            from_addr,
            payload,
            extra_localparts=RelationshipExtractor._EXTRA_LOCALPARTS,
            extra_domain_patterns=RelationshipExtractor._EXTRA_DOMAIN_PATTERNS,
        )
