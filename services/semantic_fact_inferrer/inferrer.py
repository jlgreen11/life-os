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

        Confidence threshold: Require 20+ samples before inferring facts to
        ensure statistical significance.
        """
        profile = self.ums.get_signal_profile("linguistic")
        if not profile or profile.get("samples_count", 0) < 20:
            logger.debug("Linguistic profile has insufficient samples (<20), skipping inference")
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
          - High-priority relationships (fast response times, high frequency)
          - Low-priority relationships (slow/no responses, low frequency)
          - Professional vs. personal relationship classifications

        Confidence threshold: Require 10+ samples for a specific contact
        before inferring relationship priority.
        """
        profile = self.ums.get_signal_profile("relationships")
        if not profile or profile.get("samples_count", 0) < 10:
            logger.debug("Relationship profile has insufficient samples (<10), skipping inference")
            return

        data = profile["data"]
        contacts = data.get("contacts", {})

        # --- Infer high-priority contacts ---
        # A contact is high-priority if:
        #   - Average response time < 1 hour (3600 seconds)
        #   - Interaction count > 5 (enough data to be confident)
        for contact_id, contact_data in contacts.items():
            interaction_count = contact_data.get("interaction_count", 0)
            if interaction_count < 5:
                continue  # Not enough data

            avg_response_time = contact_data.get("avg_response_time_seconds")
            if avg_response_time is not None and avg_response_time < 3600:  # <1 hour
                # Fast responder — high priority relationship
                # Link to recent episodes with this contact as source evidence
                contact_episodes = self._get_recent_episodes(contact=contact_id, limit=3)
                self.ums.update_semantic_fact(
                    key=f"relationship_priority_{contact_id}",
                    category="implicit_preference",
                    value="high_priority",
                    confidence=min(0.9, 0.6 + (3600 - avg_response_time) / 7200),  # Faster = higher confidence
                    episode_id=contact_episodes[0] if contact_episodes else None,
                )
            elif avg_response_time is not None and avg_response_time > 86400:  # >24 hours
                # Slow responder — low priority relationship
                contact_episodes = self._get_recent_episodes(contact=contact_id, limit=3)
                self.ums.update_semantic_fact(
                    key=f"relationship_priority_{contact_id}",
                    category="implicit_preference",
                    value="low_priority",
                    confidence=min(0.8, 0.5 + (avg_response_time - 86400) / 172800),  # Slower = higher confidence
                    episode_id=contact_episodes[0] if contact_episodes else None,
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
        topic_frequencies = data.get("topic_frequencies", {})

        # Get recent communication episodes to link as source evidence for
        # topic-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer expertise from topic frequency ---
        # A topic becomes an expertise area if it appears in >10% of messages
        # and has been mentioned at least 10 times.
        total_samples = profile.get("samples_count", 1)

        for topic, count in topic_frequencies.items():
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
        hourly_distribution = data.get("hourly_distribution", {})

        # Get recent communication episodes to link as source evidence for
        # cadence-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer work-life boundaries from hourly distribution ---
        # If 90%+ of messages are sent during business hours (9-17),
        # infer a preference for work-life separation.
        total_messages = sum(hourly_distribution.values())
        if total_messages == 0:
            return

        business_hours_count = sum(
            count for hour, count in hourly_distribution.items()
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
        if hourly_distribution:
            peak_hour = max(hourly_distribution, key=hourly_distribution.get)
            peak_count = hourly_distribution[peak_hour]
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
          - Baseline emotional tone
          - Stress triggers and patterns
          - Emotional resilience indicators

        NOTE: Mood data is highly sensitive. Inferred facts are stored
        for internal prediction calibration ONLY and must never be
        shared externally or used in user-facing features without
        explicit consent.

        Confidence threshold: Require 100+ samples to establish reliable
        mood baselines and avoid overreacting to temporary fluctuations.
        """
        profile = self.ums.get_signal_profile("mood_signals")
        if not profile or profile.get("samples_count", 0) < 100:
            logger.debug("Mood profile has insufficient samples (<100), skipping inference")
            return

        data = profile["data"]
        avg_sentiment = data.get("avg_sentiment")

        # Get recent communication episodes to link as source evidence for
        # mood-based facts. This creates an audit trail from facts back to
        # raw observations and enables the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="communication", limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer baseline emotional tone ---
        # Sentiment ranges from -1 (very negative) to +1 (very positive).
        # We infer baseline disposition from long-term average.
        if avg_sentiment is not None:
            if avg_sentiment > 0.3:
                # Consistently positive tone — optimistic baseline
                self.ums.update_semantic_fact(
                    key="emotional_baseline",
                    category="implicit_preference",
                    value="optimistic",
                    confidence=min(0.8, 0.5 + (avg_sentiment - 0.3) * 2),
                    episode_id=episode_id,
                )
            elif avg_sentiment < -0.2:
                # Consistently negative tone — may indicate chronic stress
                # Store this for internal stress detection, NOT for sharing
                self.ums.update_semantic_fact(
                    key="emotional_baseline",
                    category="implicit_preference",
                    value="stressed",
                    confidence=min(0.75, 0.5 + abs(avg_sentiment + 0.2) * 2),
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
