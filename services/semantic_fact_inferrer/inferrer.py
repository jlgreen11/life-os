"""
Life OS — Semantic Fact Inference Engine

Analyzes signal profiles to derive high-level semantic facts about the user.
"""

from __future__ import annotations

import logging
from typing import Optional

from storage.user_model_store import UserModelStore

logger = logging.getLogger(__name__)


class SemanticFactInferrer:
    """
    Derives semantic facts from signal profile statistics.

    This service bridges the gap between raw signal extraction (Layer 1/0) and
    semantic memory (Layer 2). It runs periodically to analyze accumulated
    signal profiles and extract stable, high-confidence facts about the user's
    preferences, expertise, and values.

    Examples of inference:
      - Linguistic profile shows avg formality 0.2 → implicit preference for casual communication
      - Relationship profile shows consistent <1hr response to specific contact → high priority relationship
      - Topic profile shows frequent Python discussions with technical depth → expertise in Python
      - Cadence profile shows work emails only during business hours → values work-life boundaries

    Each inferred fact includes:
      - Confidence score based on sample size and consistency
      - Source episodes for provenance tracking
      - Category for semantic organization
    """

    def __init__(self, user_model_store: UserModelStore):
        """
        Initialize the semantic fact inferrer.

        Args:
            user_model_store: Storage interface for reading signal profiles
                and writing semantic facts
        """
        self.ums = user_model_store

    def _get_recent_episodes(self, interaction_type: Optional[str] = None,
                            contact: Optional[str] = None,
                            limit: int = 10) -> list[str]:
        """
        Query recent episode IDs to link as evidence for inferred facts.

        While the inferrer works with aggregate statistics, linking facts to
        their source episodes provides an audit trail and enables the confidence
        growth loop (facts re-confirmed by new episodes get +0.05 confidence).

        Args:
            interaction_type: Filter by interaction type (e.g., "communication")
            contact: Filter by contact email/address
            limit: Maximum number of episode IDs to return (default 10)

        Returns:
            List of episode IDs (most recent first)
        """
        query = "SELECT id FROM episodes WHERE 1=1"
        params = []

        if interaction_type:
            query += " AND interaction_type = ?"
            params.append(interaction_type)

        if contact:
            # Search JSON array for contact
            query += " AND json_extract(contacts_involved, '$') LIKE ?"
            params.append(f'%"{contact}"%')

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.ums.db.get_connection("user_model") as conn:
            rows = conn.execute(query, params).fetchall()
            return [row["id"] for row in rows]

    def infer_from_linguistic_profile(self):
        """
        Derive semantic facts from linguistic signal profile.

        Analyzes the user's writing style to infer:
          - Communication style preferences (formal vs. casual)
          - Per-contact style variations (indicates relationship dynamics)
          - Emotional expressiveness patterns

        Confidence threshold: Require 1+ sample before inferring facts.
        This minimal threshold enables semantic fact generation even with very
        limited outbound communication history. Initial facts will have lower
        confidence that grows as more samples accumulate.
        """
        profile = self.ums.get_signal_profile("linguistic")
        if not profile or profile.get("samples_count", 0) < 1:
            logger.debug("Linguistic profile has insufficient samples (<1), skipping inference")
            return

        data = profile["data"]
        averages = data.get("averages", {})

        # Get recent communication episodes to link as source evidence for
        # linguistic facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Implicit preference: communication formality ---
        # Formality ranges from 0 (very casual) to 1 (very formal).
        # We infer a preference if the user's average is significantly
        # skewed from the neutral 0.5 midpoint.
        formality = averages.get("formality")
        if formality is not None:
            if formality < 0.3:
                # Very casual writer — prefers informal tone
                self.ums.update_semantic_fact(
                    key="communication_style_formality",
                    category="implicit_preference",
                    value="casual",
                    confidence=min(0.95, 0.5 + (0.3 - formality)),  # Higher confidence the more casual
                    episode_id=episode_id,
                )
            elif formality > 0.7:
                # Very formal writer — prefers professional tone
                self.ums.update_semantic_fact(
                    key="communication_style_formality",
                    category="implicit_preference",
                    value="formal",
                    confidence=min(0.95, 0.5 + (formality - 0.7)),  # Higher confidence the more formal
                    episode_id=episode_id,
                )

        # --- Implicit preference: emoji usage ---
        # High emoji rate indicates expressive, informal communication style
        emoji_rate = averages.get("emoji_rate")
        if emoji_rate is not None and emoji_rate > 0.05:  # >5% of words are emojis
            self.ums.update_semantic_fact(
                key="communication_style_emoji",
                category="implicit_preference",
                value="expressive_with_emojis",
                confidence=min(0.9, 0.4 + emoji_rate * 5),  # Confidence grows with emoji density
                episode_id=episode_id,
            )

        # --- Implicit preference: exclamation usage ---
        # High exclamation rate indicates enthusiastic communication style
        exclamation_rate = averages.get("exclamation_rate")
        if exclamation_rate is not None and exclamation_rate > 0.3:  # >0.3 per sentence
            self.ums.update_semantic_fact(
                key="communication_style_enthusiasm",
                category="implicit_preference",
                value="enthusiastic",
                confidence=min(0.85, 0.5 + exclamation_rate),
                episode_id=episode_id,
            )

        # --- Implicit preference: hedge words (tentativeness) ---
        # High hedge rate indicates preference for tentative, cautious language
        hedge_rate = averages.get("hedge_rate")
        if hedge_rate is not None and hedge_rate > 0.2:  # >0.2 per sentence
            self.ums.update_semantic_fact(
                key="communication_style_directness",
                category="implicit_preference",
                value="tentative",
                confidence=min(0.8, 0.4 + hedge_rate * 2),
                episode_id=episode_id,
            )
        elif hedge_rate is not None and hedge_rate < 0.05:
            # Very low hedge rate indicates direct, assertive communication
            self.ums.update_semantic_fact(
                key="communication_style_directness",
                category="implicit_preference",
                value="direct",
                confidence=min(0.8, 0.5 + (0.05 - hedge_rate) * 5),
                episode_id=episode_id,
            )

        logger.info(f"Inferred semantic facts from linguistic profile (samples={profile.get('samples_count')})")

    def infer_from_relationship_profile(self):
        """
        Derive semantic facts from relationship signal profile.

        Analyzes communication patterns with specific contacts to infer:
          - High-priority relationships (high interaction frequency)
          - Active vs. one-sided relationships (inbound/outbound balance)
          - Multi-channel relationships (communication across platforms)

        Confidence threshold: Require 10+ samples for a specific contact
        before inferring relationship priority.

        Filtering:
          - Skip contacts with zero outbound messages (one-way relationships)
            to avoid treating marketing emails as "high priority" relationships
          - Semantic facts should reflect the user's actual communication
            patterns, not the volume of spam they receive
        """
        profile = self.ums.get_signal_profile("relationships")
        if not profile or profile.get("samples_count", 0) < 10:
            logger.debug("Relationship profile has insufficient samples (<10), skipping inference")
            return

        data = profile["data"]
        contacts = data.get("contacts", {})

        # --- Infer high-priority contacts based on interaction frequency ---
        # A contact is high-priority if they have significantly more interactions
        # than average (top 20% of contacts by interaction count)
        if not contacts:
            return

        # Filter out one-way relationships (zero outbound) to avoid polluting
        # semantic facts with marketing email senders. Only consider contacts
        # the user actually communicates with bidirectionally.
        bidirectional_contacts = {
            contact_id: contact_data
            for contact_id, contact_data in contacts.items()
            if contact_data.get("outbound_count", 0) > 0
        }

        if not bidirectional_contacts:
            logger.debug("No bidirectional contacts found, skipping relationship inference")
            return

        # Calculate average interaction count across bidirectional contacts only
        interaction_counts = [c.get("interaction_count", 0) for c in bidirectional_contacts.values()]
        if not interaction_counts:
            return

        avg_interactions = sum(interaction_counts) / len(interaction_counts)
        high_priority_threshold = avg_interactions * 2  # 2x average = high priority

        for contact_id, contact_data in bidirectional_contacts.items():
            interaction_count = contact_data.get("interaction_count", 0)
            if interaction_count < 5:
                continue  # Not enough data

            # Link to recent episodes with this contact as source evidence
            contact_episodes = self._get_recent_episodes(contact=contact_id, limit=3)
            episode_id = contact_episodes[0] if contact_episodes else None

            # --- High-priority relationship (frequent communication) ---
            if interaction_count >= high_priority_threshold:
                self.ums.update_semantic_fact(
                    key=f"relationship_priority_{contact_id}",
                    category="implicit_preference",
                    value="high_priority",
                    confidence=min(0.9, 0.6 + min(0.3, (interaction_count / avg_interactions - 2) * 0.1)),
                    episode_id=episode_id,
                )

            # --- Multi-channel relationship (communication versatility) ---
            channels_used = contact_data.get("channels_used", [])
            if len(channels_used) >= 2:
                self.ums.update_semantic_fact(
                    key=f"relationship_multichannel_{contact_id}",
                    category="implicit_preference",
                    value="multi_channel",
                    confidence=min(0.85, 0.5 + len(channels_used) * 0.15),
                    episode_id=episode_id,
                )

            # --- Relationship balance (mutual vs. one-sided) ---
            inbound_count = contact_data.get("inbound_count", 0)
            outbound_count = contact_data.get("outbound_count", 0)
            total_count = inbound_count + outbound_count

            if total_count >= 10:  # Need enough data to assess balance
                balance_ratio = min(inbound_count, outbound_count) / total_count

                if balance_ratio > 0.3:  # Both directions active (30%+ in each direction)
                    self.ums.update_semantic_fact(
                        key=f"relationship_balance_{contact_id}",
                        category="implicit_preference",
                        value="mutual",
                        confidence=min(0.85, 0.5 + balance_ratio),
                        episode_id=episode_id,
                    )
                elif outbound_count > inbound_count * 3:  # User initiates 3x more
                    self.ums.update_semantic_fact(
                        key=f"relationship_balance_{contact_id}",
                        category="implicit_preference",
                        value="user_initiated",
                        confidence=min(0.8, 0.5 + (outbound_count / total_count - 0.5)),
                        episode_id=episode_id,
                    )

        logger.info(f"Inferred semantic facts from relationship profile (samples={profile.get('samples_count')})")

    def infer_from_topic_profile(self):
        """
        Derive semantic facts from topic signal profile.

        Analyzes topic distribution and discussion patterns to infer:
          - Expertise areas (frequently discussed topics with depth)
          - Interest areas (topics mentioned often)
          - Professional vs. personal topic categories

        Confidence threshold: Require 30+ samples before inferring expertise
        to avoid false positives from temporary interests.
        """
        profile = self.ums.get_signal_profile("topics")
        if not profile or profile.get("samples_count", 0) < 30:
            logger.debug("Topic profile has insufficient samples (<30), skipping inference")
            return

        data = profile["data"]
        # The topic extractor stores data as "topic_counts", not "topic_frequencies"
        topic_counts = data.get("topic_counts", {})

        # Get recent communication episodes to link as source evidence for
        # topic-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer expertise from topic frequency ---
        # A topic becomes an expertise area if it appears in >10% of messages
        # and has been mentioned at least 10 times.
        total_samples = profile.get("samples_count", 1)

        for topic, count in topic_counts.items():
            frequency_ratio = count / total_samples

            if count >= 10 and frequency_ratio > 0.1:
                # Frequently discussed topic — likely an expertise area
                self.ums.update_semantic_fact(
                    key=f"expertise_{topic}",
                    category="expertise",
                    value=topic,
                    confidence=min(0.95, 0.5 + frequency_ratio * 2),  # Higher freq = higher confidence
                    episode_id=episode_id,
                )
            elif count >= 5 and frequency_ratio > 0.05:
                # Moderately discussed topic — area of interest
                self.ums.update_semantic_fact(
                    key=f"interest_{topic}",
                    category="implicit_preference",
                    value=topic,
                    confidence=min(0.8, 0.4 + frequency_ratio * 3),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from topic profile (samples={profile.get('samples_count')})")

    def infer_from_cadence_profile(self):
        """
        Derive semantic facts from cadence signal profile.

        Analyzes temporal communication patterns to infer:
          - Work-life boundaries (office hours preferences)
          - Peak productivity times
          - Communication rhythm preferences

        Confidence threshold: Require 50+ samples to establish reliable
        cadence patterns across different times of day.
        """
        profile = self.ums.get_signal_profile("cadence")
        if not profile or profile.get("samples_count", 0) < 50:
            logger.debug("Cadence profile has insufficient samples (<50), skipping inference")
            return

        data = profile["data"]
        # The cadence extractor stores data as "hourly_activity", not "hourly_distribution"
        hourly_activity = data.get("hourly_activity", {})

        # Get recent communication episodes to link as source evidence for
        # cadence-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer work-life boundaries from hourly activity ---
        # If 90%+ of messages are sent during business hours (9-17),
        # infer a preference for work-life separation.
        total_messages = sum(hourly_activity.values())
        if total_messages == 0:
            return

        business_hours_count = sum(
            count for hour, count in hourly_activity.items()
            if 9 <= int(hour) <= 17
        )
        business_hours_ratio = business_hours_count / total_messages

        if business_hours_ratio > 0.9:
            # Strong work-life boundaries — rarely messages outside business hours
            self.ums.update_semantic_fact(
                key="work_life_boundaries",
                category="values",
                value="strict_boundaries",
                confidence=min(0.95, 0.5 + (business_hours_ratio - 0.9) * 5),
                episode_id=episode_id,
            )
        elif business_hours_ratio < 0.3:
            # Messages at all hours — flexible schedule or always-on work style
            self.ums.update_semantic_fact(
                key="work_life_boundaries",
                category="values",
                value="flexible_boundaries",
                confidence=min(0.85, 0.5 + (0.3 - business_hours_ratio) * 2),
                episode_id=episode_id,
            )

        # --- Infer peak communication hours ---
        # Find the hour with the highest message count — this is the user's
        # most active communication time.
        if hourly_activity:
            peak_hour = max(hourly_activity, key=hourly_activity.get)
            peak_count = hourly_activity[peak_hour]
            peak_ratio = peak_count / total_messages

            if peak_ratio > 0.2:  # Peak hour accounts for >20% of all messages
                self.ums.update_semantic_fact(
                    key="peak_communication_hour",
                    category="implicit_preference",
                    value=int(peak_hour),
                    confidence=min(0.9, 0.5 + peak_ratio * 2),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from cadence profile (samples={profile.get('samples_count')})")

    def infer_from_mood_profile(self):
        """
        Derive semantic facts from mood signal profile.

        Analyzes sentiment patterns to infer:
          - Baseline stress levels
          - Pressure patterns from incoming communications
          - Language negativity indicators

        NOTE: Mood data is highly sensitive. Inferred facts are stored
        for internal prediction calibration ONLY and must never be
        shared externally or used in user-facing features without
        explicit consent.

        Confidence threshold: Require 5+ samples to enable early inference.
        This low threshold allows the system to start building mood-based
        semantic facts even with limited history. Initial facts will have
        lower confidence that grows as more data accumulates.
        """
        profile = self.ums.get_signal_profile("mood_signals")
        if not profile or profile.get("samples_count", 0) < 5:
            logger.debug("Mood profile has insufficient samples (<5), skipping inference")
            return

        data = profile["data"]
        recent_signals = data.get("recent_signals", [])

        if not recent_signals:
            logger.debug("Mood profile has no recent signals, skipping inference")
            return

        # Get recent communication episodes to link as source evidence for
        # mood-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Analyze signal patterns ---
        # Compute averages for different signal types to identify patterns
        stress_signals = [s for s in recent_signals if s.get("signal_type") in ["negative_language", "incoming_negative_language"]]
        pressure_signals = [s for s in recent_signals if s.get("signal_type") == "incoming_pressure"]

        # --- Infer baseline stress from negative language frequency ---
        if len(recent_signals) > 0:
            stress_ratio = len(stress_signals) / len(recent_signals)

            if stress_ratio > 0.3:  # >30% of signals show negative language
                # High stress baseline — frequent negative language patterns
                self.ums.update_semantic_fact(
                    key="stress_baseline",
                    category="implicit_preference",
                    value="high_stress",
                    confidence=min(0.75, 0.5 + stress_ratio),
                    episode_id=episode_id,
                )
            elif stress_ratio < 0.1:  # <10% of signals show negative language
                # Low stress baseline — positive communication patterns
                self.ums.update_semantic_fact(
                    key="stress_baseline",
                    category="implicit_preference",
                    value="low_stress",
                    confidence=min(0.8, 0.5 + (0.1 - stress_ratio) * 3),
                    episode_id=episode_id,
                )

        # --- Infer incoming pressure sensitivity ---
        # Users who consistently receive high-pressure communications may need
        # notification filtering or quiet hours protection
        if len(recent_signals) > 0:
            pressure_ratio = len(pressure_signals) / len(recent_signals)

            if pressure_ratio > 0.2:  # >20% of signals are incoming pressure
                self.ums.update_semantic_fact(
                    key="incoming_pressure_exposure",
                    category="implicit_preference",
                    value="high_pressure_environment",
                    confidence=min(0.8, 0.5 + pressure_ratio * 2),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from mood profile (samples={profile.get('samples_count')})")

    def run_all_inference(self):
        """
        Run inference across all signal profiles.

        This is the main entry point for semantic fact extraction. It
        analyzes all available signal profiles and derives semantic facts
        from each one.

        Should be called:
          - Periodically (e.g., every 6 hours via background task)
          - After significant data ingestion (e.g., after syncing 1000+ new events)
          - On-demand via admin endpoint for testing
        """
        logger.info("Starting semantic fact inference across all profiles")

        self.infer_from_linguistic_profile()
        self.infer_from_relationship_profile()
        self.infer_from_topic_profile()
        self.infer_from_cadence_profile()
        self.infer_from_mood_profile()

        logger.info("Completed semantic fact inference")
