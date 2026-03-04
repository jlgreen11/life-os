"""
Life OS — Semantic Fact Inference Engine

Analyzes signal profiles to derive high-level semantic facts about the user.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from services.signal_extractor.marketing_filter import is_marketing_or_noreply
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

    def _early_inference_confidence(self, samples: int, old_threshold: int, base_confidence: float = 0.5) -> float:
        """Return reduced confidence for inferences below the original sample threshold.

        When sample count is between the new (lowered) threshold and the old threshold,
        scale confidence proportionally. This ensures early inferences are surfaced
        but with appropriate uncertainty.

        Args:
            samples: Current sample count for the profile.
            old_threshold: The original (pre-lowering) minimum sample threshold.
            base_confidence: The default confidence used when samples >= old_threshold.

        Returns:
            Scaled confidence between 0.3 and base_confidence.
        """
        try:
            if samples >= old_threshold:
                return base_confidence
            # Linear interpolation: at 0 samples → 0.3, at old_threshold → base_confidence
            ratio = samples / old_threshold
            return round(0.3 + (base_confidence - 0.3) * ratio, 2)
        except Exception:
            # Fail-open: if confidence calculation fails, use default
            return base_confidence

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
            logger.info("Linguistic profile has insufficient samples (<1), skipping inference")
            return {"type": "linguistic", "processed": False, "reason": "insufficient samples (<1)"}

        data = profile["data"]
        averages = data.get("averages", {})

        # Get recent outbound-email episodes to link as source evidence for
        # linguistic facts.  The linguistic profile is built from sent messages
        # (email_sent), so we look for those episodes specifically.  Passing
        # "communication" here was a pre-migration label that no longer exists
        # in the episodes table (the backfill in LifeOS._backfill_episode_classification_if_needed
        # reclassified all "communication" rows to granular types like "email_sent").
        # Without the correct type, this query always returns [] and every
        # linguistic fact is stored with episode_id=None, breaking the audit
        # trail and the confidence growth loop.
        recent_episodes = self._get_recent_episodes(interaction_type="email_sent", limit=5)
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
        return {"type": "linguistic", "processed": True, "reason": None}

    def infer_from_inbound_linguistic_profile(self):
        """
        Derive semantic facts from the inbound linguistic signal profile.

        Analyzes communication patterns from messages *received* by the user to
        infer facts about their communication environment — e.g., whether they
        operate in a formal professional setting, whether they are a go-to expert
        (high inbound question rate), or whether their contacts communicate
        cautiously (high hedge rate).

        This complements infer_from_linguistic_profile (which reads the user's
        own outbound writing) by mining the much larger pool of inbound samples
        (typically 10x–100x more data) to characterise the user's surroundings.

        Confidence threshold: Require 10+ samples. The threshold is lower than
        outbound (which uses 1) because inbound data is aggregate across many
        senders, so even small sample counts carry meaningful signal about the
        user's environment.
        """
        profile = self.ums.get_signal_profile("linguistic_inbound")
        if not profile or profile.get("samples_count", 0) < 10:
            logger.info("Inbound linguistic profile has insufficient samples (<10), skipping inference")
            return {"type": "inbound_linguistic", "processed": False, "reason": "insufficient samples (<10)"}

        data = profile["data"]
        samples = profile.get("samples_count", 0)

        # Link to recent episodes for provenance (no interaction_type filter
        # since inbound messages span email_received, message_received, etc.)
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # Scale confidence for early inferences — full confidence at 50+ samples
        base_confidence = self._early_inference_confidence(samples, old_threshold=50)

        # --- Compute aggregate inbound averages across all contacts ---
        # The inbound profile stores per_contact_averages but no global averages,
        # so we derive them by averaging across all contacts' averages.
        per_contact_avgs = data.get("per_contact_averages", {})
        if not per_contact_avgs:
            logger.info("Inbound linguistic profile has no per-contact averages, skipping inference")
            return {"type": "inbound_linguistic", "processed": False, "reason": "no per-contact data"}

        # Compute weighted averages across contacts (weighted by sample count)
        total_weight = 0
        weighted_formality = 0.0
        weighted_question_rate = 0.0
        weighted_hedge_rate = 0.0
        for _contact, avgs in per_contact_avgs.items():
            weight = avgs.get("samples_count", 1)
            total_weight += weight
            weighted_formality += avgs.get("formality", 0.5) * weight
            weighted_question_rate += avgs.get("question_rate", 0.0) * weight
            weighted_hedge_rate += avgs.get("hedge_rate", 0.0) * weight

        if total_weight == 0:
            return {"type": "inbound_linguistic", "processed": False, "reason": "no weighted data"}

        avg_formality = weighted_formality / total_weight
        avg_question_rate = weighted_question_rate / total_weight
        avg_hedge_rate = weighted_hedge_rate / total_weight

        # --- Inbound communication environment (formality) ---
        if avg_formality > 0.7:
            self.ums.update_semantic_fact(
                key="inbound_communication_environment",
                category="implicit_preference",
                value="formal_professional_environment",
                confidence=min(0.95, base_confidence + (avg_formality - 0.7)),
                episode_id=episode_id,
            )
        elif avg_formality < 0.3:
            self.ums.update_semantic_fact(
                key="inbound_communication_environment",
                category="implicit_preference",
                value="casual_informal_environment",
                confidence=min(0.95, base_confidence + (0.3 - avg_formality)),
                episode_id=episode_id,
            )

        # --- Inbound question intensity ---
        # High inbound question rate indicates the user is a go-to expert or
        # resource person — people frequently ask them questions.
        if avg_question_rate > 0.5:
            self.ums.update_semantic_fact(
                key="inbound_question_intensity",
                category="implicit_preference",
                value="frequently_asked_questions",
                confidence=min(0.9, base_confidence + avg_question_rate * 0.3),
                episode_id=episode_id,
            )

        # --- Inbound communication style (hedging) ---
        # High inbound hedge rate indicates the user's contacts tend to
        # communicate cautiously or tentatively.
        if avg_hedge_rate > 0.2:
            self.ums.update_semantic_fact(
                key="inbound_communication_style",
                category="implicit_preference",
                value="cautious_senders",
                confidence=min(0.85, base_confidence + avg_hedge_rate),
                episode_id=episode_id,
            )

        # --- Per-contact formality distribution analysis ---
        # Count how many contacts skew formal vs. casual to reinforce the
        # environment fact with stronger confidence.
        formal_contacts = sum(
            1 for avgs in per_contact_avgs.values()
            if avgs.get("formality", 0.5) > 0.7
        )
        casual_contacts = sum(
            1 for avgs in per_contact_avgs.values()
            if avgs.get("formality", 0.5) < 0.3
        )
        total_contacts = len(per_contact_avgs)

        if total_contacts > 0 and formal_contacts / total_contacts > 0.7:
            # Overwhelming majority of contacts are formal — reinforce the fact
            self.ums.update_semantic_fact(
                key="inbound_communication_environment",
                category="implicit_preference",
                value="formal_professional_environment",
                confidence=min(0.95, base_confidence + 0.15),
                episode_id=episode_id,
            )

        logger.info(
            "Inferred semantic facts from inbound linguistic profile "
            "(samples=%s, contacts=%s, avg_formality=%.2f)",
            samples, total_contacts, avg_formality,
        )
        return {"type": "inbound_linguistic", "processed": True, "reason": None}

    def infer_from_relationship_profile(self):
        """
        Derive semantic facts from relationship signal profile.

        Analyzes communication patterns with specific contacts to infer:
          - High-priority relationships (high interaction frequency)
          - Active vs. one-sided relationships (inbound/outbound balance)
          - Multi-channel relationships (communication across platforms)

        Confidence threshold: Require 5+ samples for a specific contact
        before inferring relationship priority. Early inferences (5-10 samples)
        use reduced confidence via _early_inference_confidence scaling.

        Filtering:
          - Skip contacts with zero outbound messages (one-way relationships)
            to avoid treating marketing emails as "high priority" relationships
          - Apply the shared marketing filter to exclude automated/newsletter
            senders even if the user has replied to them once.  A user may
            occasionally reply to a Royal Caribbean promotion or an Amazon
            store notification, but those senders should never appear as
            "high_priority" contacts in semantic memory.
          - Semantic facts should reflect the user's actual human communication
            patterns, not the volume of commercial email they receive.
        """
        # --- Purge stale marketing relationship facts ---
        # Marketing-sender relationship facts created by prior inference runs
        # (before this filter was applied) must be removed so they do not
        # persist in the semantic memory between inference cycles.
        self._purge_marketing_relationship_facts()

        profile = self.ums.get_signal_profile("relationships")
        if not profile or profile.get("samples_count", 0) < 5:
            logger.info("Relationship profile has insufficient samples (<5), skipping inference")
            return {"type": "relationship", "processed": False, "reason": "insufficient samples (<5)"}

        # Scale confidence down for early inferences (between new threshold 5 and old threshold 10)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=10)

        data = profile["data"]
        contacts = data.get("contacts", {})

        # --- Infer high-priority contacts based on interaction frequency ---
        # A contact is high-priority if they have significantly more interactions
        # than average (top 20% of contacts by interaction count)
        if not contacts:
            return {"type": "relationship", "processed": True, "reason": None}

        # Filter 1: Skip contacts with zero outbound messages.
        # These are pure inbound senders (newsletters, automated receipts, etc.)
        # that the user has never replied to.
        # Also defensively skip non-dict entries (e.g., None from corrupted data).
        bidirectional_contacts = {
            contact_id: contact_data
            for contact_id, contact_data in contacts.items()
            if isinstance(contact_data, dict) and contact_data.get("outbound_count", 0) > 0
        }

        # Filter 2: Apply the shared marketing filter to remove automated/commercial
        # senders that the user may have occasionally replied to.  Examples:
        #   - no-reply@accounts.google.com  (automated Google notifications)
        #   - store-news@amazon.com          (Amazon marketing)
        #   - SouthwestAirlines@iluv.southwest.com (airline promotions)
        # Without this filter, senders that appear frequently in the inbox push
        # up the average interaction count, causing them to cross the
        # high_priority_threshold and pollute semantic memory.
        human_contacts = {
            contact_id: contact_data
            for contact_id, contact_data in bidirectional_contacts.items()
            if not is_marketing_or_noreply(contact_id)
        }

        if not human_contacts:
            logger.info("No human bidirectional contacts found after marketing filter, skipping relationship inference")
            return {"type": "relationship", "processed": True, "reason": None}

        # Calculate average interaction count across human contacts only.
        # Previously this was computed over all bidirectional contacts, which
        # inflated the average with high-volume marketing senders and raised
        # the high_priority_threshold so high that real contacts rarely qualified.
        # Defensively handle non-dict contact data (e.g., None values from corrupted data)
        interaction_counts = []
        for c in human_contacts.values():
            if isinstance(c, dict):
                interaction_counts.append(c.get("interaction_count", 0))
        if not interaction_counts:
            return {"type": "relationship", "processed": True, "reason": None}

        avg_interactions = sum(interaction_counts) / len(interaction_counts)
        high_priority_threshold = avg_interactions * 2  # 2x average = high priority

        for contact_id, contact_data in human_contacts.items():
            interaction_count = contact_data.get("interaction_count", 0)
            if interaction_count < 3:
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
                    confidence=min(0.9, base_confidence + 0.1 + min(0.3, (interaction_count / avg_interactions - 2) * 0.1)),
                    episode_id=episode_id,
                )

            # --- Multi-channel relationship (communication versatility) ---
            channels_used = contact_data.get("channels_used", [])
            if len(channels_used) >= 2:
                self.ums.update_semantic_fact(
                    key=f"relationship_multichannel_{contact_id}",
                    category="implicit_preference",
                    value="multi_channel",
                    confidence=min(0.85, base_confidence + len(channels_used) * 0.15),
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
                        confidence=min(0.85, base_confidence + balance_ratio),
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
        return {"type": "relationship", "processed": True, "reason": None}

    def _purge_noise_topic_facts(self, noise_blocklist: set) -> int:
        """
        Remove previously-stored expertise/interest facts whose topic word is noise.

        When the noise blocklist is expanded (e.g., to add generic English stopwords),
        stale facts from prior inference runs may still exist in the database under
        keys like "expertise_more" or "interest_please". This method deletes those
        stale facts so the semantic memory reflects the updated, cleaner inference.

        Safety: Only facts with ``is_user_corrected = 0`` are removed. User-corrected
        facts are never touched, even if their key appears in the blocklist.

        Args:
            noise_blocklist: Set of lowercase topic words that should never appear
                as expertise or interest facts.

        Returns:
            Number of stale noise facts deleted.

        Example::

            purged = self._purge_noise_topic_facts({'more', 'view', 'please'})
            # Removes "expertise_more", "expertise_view", "interest_please" etc.
            # from semantic_facts if they exist and are not user-corrected.
        """
        if not noise_blocklist:
            return 0

        deleted = 0
        with self.ums.db.get_connection("user_model") as conn:
            # Fetch all expertise_* and interest_* facts that are not user-corrected.
            # We'll filter in Python to avoid constructing a large SQL IN clause.
            rows = conn.execute(
                "SELECT key FROM semantic_facts "
                "WHERE (key LIKE 'expertise_%' OR key LIKE 'interest_%') "
                "AND is_user_corrected = 0"
            ).fetchall()

            for row in rows:
                key = row["key"]
                # Extract the topic word from the key (everything after the first "_")
                prefix, _, topic_word = key.partition("_")
                if topic_word.lower() in noise_blocklist:
                    conn.execute(
                        "DELETE FROM semantic_facts WHERE key = ? AND is_user_corrected = 0",
                        (key,)
                    )
                    deleted += 1

        if deleted > 0:
            logger.info(
                f"Purged {deleted} stale noise topic facts (expertise_*/interest_* "
                f"whose topic word is in the updated blocklist)"
            )
        return deleted

    def _purge_marketing_relationship_facts(self) -> int:
        """
        Remove previously-stored relationship facts for marketing/automated senders.

        When the marketing filter is first applied (or improved), stale facts from
        prior inference runs may already exist in the database for senders like
        ``no-reply@accounts.google.com`` or ``store-news@amazon.com``.  This method
        deletes those relationship facts so semantic memory reflects only genuine
        human contacts.

        Safety: Only facts with ``is_user_corrected = 0`` are removed.  User-corrected
        facts are never touched.

        Returns:
            Number of stale marketing relationship facts deleted.

        Example::

            purged = self._purge_marketing_relationship_facts()
            # Removes "relationship_priority_no-reply@accounts.google.com",
            # "relationship_balance_store-news@amazon.com", etc.
        """
        deleted = 0
        # Fetch all relationship_priority_* and relationship_balance_* and
        # relationship_multichannel_* facts that are not user-corrected.
        # The contact_id is everything after the first underscore-separated prefix.
        with self.ums.db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT key FROM semantic_facts "
                "WHERE (key LIKE 'relationship_priority_%' "
                "   OR key LIKE 'relationship_balance_%' "
                "   OR key LIKE 'relationship_multichannel_%') "
                "AND is_user_corrected = 0"
            ).fetchall()

            for row in rows:
                key = row["key"]
                # Extract contact_id: everything after the prefix and its underscore.
                # e.g. "relationship_priority_no-reply@accounts.google.com"
                #        → prefix="relationship_priority", contact_id="no-reply@accounts.google.com"
                # We split on the second underscore (after "relationship_<type>_").
                parts = key.split("_", 2)  # ["relationship", "priority", "no-reply@..."]
                if len(parts) < 3:
                    continue
                contact_id = parts[2]

                if is_marketing_or_noreply(contact_id):
                    conn.execute(
                        "DELETE FROM semantic_facts WHERE key = ? AND is_user_corrected = 0",
                        (key,)
                    )
                    deleted += 1

        if deleted > 0:
            logger.info(
                "Purged %d stale marketing relationship facts "
                "(relationship_priority/balance/multichannel for automated senders)",
                deleted,
            )
        return deleted

    def infer_from_topic_profile(self):
        """
        Derive semantic facts from topic signal profile.

        Analyzes topic distribution and discussion patterns to infer:
          - Expertise areas (frequently discussed topics with depth)
          - Interest areas (topics mentioned often)
          - Professional vs. personal topic categories

        Confidence threshold: Require 15+ samples before inferring expertise
        to avoid false positives from temporary interests. Early inferences
        (15-30 samples) use reduced confidence via _early_inference_confidence scaling.
        """
        profile = self.ums.get_signal_profile("topics")
        if not profile or profile.get("samples_count", 0) < 15:
            logger.info("Topic profile has insufficient samples (<15), skipping inference")
            return {"type": "topic", "processed": False, "reason": "insufficient samples (<15)"}

        # Scale confidence down for early inferences (between new threshold 15 and old threshold 30)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=30)

        data = profile["data"]
        # The topic extractor stores data as "topic_counts", not "topic_frequencies"
        topic_counts = data.get("topic_counts", {})

        # Get recent episodes to link as source evidence for topic-based facts.
        # Topics are extracted from all inbound and outbound email, so we do not
        # filter by a specific interaction type.  The old "communication" type no
        # longer exists after the episode backfill migration; fetching without a
        # type filter ensures we always get a valid episode_id when episodes exist.
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Noise token blocklist ---
        # Topic profiles accumulate noise from three sources:
        #   1. HTML/CSS tokens from marketing emails processed before HTML stripping
        #      (e.g., 'nbsp', 'padding', 'tbody', 'border')
        #   2. Generic English stopwords that appear in nearly every email
        #      (e.g., 'more', 'here', 'please', 'free', 'valid', 'view')
        #   3. Generic email/marketing vocabulary that saturates the topic profile
        #      but carries no signal about the user's expertise or interests
        #      (e.g., 'email', 'message', 'update', 'offer', 'shop', 'account')
        #
        # Without filtering these, marketing emails dominate the topic profile and
        # produce expertise/interest facts for noise words instead of real topics.
        # A topic needs >10% frequency to reach the "expertise" threshold — but
        # with 95K samples mostly from marketing emails, generic words far exceed
        # that threshold while real expertise words (like "python" or "machine_learning")
        # get crowded out.
        TOPIC_NOISE_BLOCKLIST = {
            # HTML entities
            'nbsp', 'zwnj', 'zwj', 'lrm', 'rlm', 'mdash', 'ndash', 'hellip',
            'quot', 'apos', 'amp', 'lt', 'gt', 'copy', 'reg', 'trade',
            # Common CSS properties
            'padding', 'margin', 'border', 'width', 'height', 'color', 'background',
            'font', 'size', 'weight', 'style', 'family', 'align', 'text', 'display',
            'position', 'top', 'left', 'right', 'bottom', 'flex', 'grid', 'float',
            # HTML tags
            'div', 'span', 'table', 'tbody', 'thead', 'tfoot', 'tr', 'td', 'th',
            'img', 'href', 'src', 'alt', 'class', 'id', 'name', 'value', 'type',
            'meta', 'link', 'script', 'noscript', 'iframe', 'embed', 'object',
            # CSS/HTML keywords
            'important', 'inherit', 'auto', 'none', 'block', 'inline', 'hidden',
            'visible', 'absolute', 'relative', 'fixed', 'sticky', 'center',
            # Protocol/URL fragments
            'http', 'https', 'www', 'com', 'html', 'css', 'js', 'png', 'jpg', 'gif',
            # Common email template artifacts
            'unsubscribe', 'pixel', 'tracker', 'analytics', 'campaign', 'utm',
            # Generic English stopwords that appear in nearly every email
            'more', 'here', 'please', 'free', 'valid', 'view', 'just', 'also',
            'like', 'get', 'this', 'that', 'have', 'with', 'from', 'your', 'our',
            'all', 'new', 'now', 'one', 'not', 'use', 'can', 'will', 'has', 'are',
            'was', 'its', 'for', 'but', 'you', 'the', 'and', 'any', 'are', 'be',
            'by', 'do', 'he', 'in', 'it', 'me', 'my', 'no', 'of', 'on', 'or',
            'so', 'to', 'up', 'us', 'we',
            # Generic email/communication vocabulary (not expertise signals)
            'email', 'message', 'update', 'offer', 'shop', 'account', 'click',
            'order', 'store', 'today', 'week', 'month', 'year', 'time', 'day',
            'news', 'info', 'read', 'send', 'reply', 'forward', 'contact', 'team',
            'support', 'help', 'check', 'confirm', 'verify', 'access', 'open',
            'save', 'sale', 'deal', 'discount', 'reward', 'point', 'earn',
            'subscribe', 'newsletter', 'notification', 'alert', 'reminder',
            'manage', 'setting', 'preference', 'privacy', 'policy', 'term',
            'service', 'product', 'item', 'price', 'cost', 'amount', 'total',
            'customer', 'member', 'user', 'account', 'profile', 'password',
            'review', 'rating', 'comment', 'share', 'follow', 'like', 'post',
            # Additional CSS/font/whitespace artifacts seen in email templates
            'lspace', 'rspace', 'sans', 'serif', 'arial', 'verdana', 'helvetica',
            'line', 'letter', 'spacing', 'indent', 'overflow', 'wrap', 'break',
            'normal', 'bold', 'italic', 'underline', 'decoration', 'transform',
            'uppercase', 'lowercase', 'capitalize', 'shadow', 'opacity', 'radius',
            'cursor', 'pointer', 'hover', 'focus', 'active', 'disabled', 'checked',
            # Promotional/marketing vocabulary that floods inboxes despite HTML stripping.
            # These words appear in plain-text portions of promotional emails (subject
            # lines, call-to-action buttons, footer copy) and do NOT signal user interest.
            # This expanded set was identified by examining the top-20 topic_counts
            # from a real inbox dominated by marketing emails (2026-02-28 audit).
            'offers', 'holiday', 'rewards', 'gift', 'deals', 'information',
            'limited', 'exclusive', 'special', 'extra', 'plus', 'best', 'great',
            'amazing', 'incredible', 'fantastic', 'love', 'back', 'last', 'next',
            'only', 'come', 'going', 'want', 'need', 'make', 'find', 'take',
            'know', 'think', 'look', 'show', 'feel', 'tell', 'keep', 'turn',
            'cart', 'checkout', 'purchase', 'bought', 'spend', 'earn', 'spend',
            'shipping', 'delivery', 'returns', 'exchange', 'eligible', 'redeem',
            'activate', 'claim', 'apply', 'enter', 'register', 'sign', 'login',
            'welcome', 'thank', 'thanks', 'hello', 'dear', 'regards', 'sincerely',
            'best', 'warm', 'kind', 'invite', 'join', 'stay', 'visit', 'learn',
            'discover', 'explore', 'enjoy', 'start', 'stop', 'continue', 'complete',
            # Common proper nouns that appear in marketing greetings (e.g. "Hi Jeremy!")
            # but are not meaningful user interests. We match lowercase so "jeremy" is
            # caught whether the extractor normalises to lowercase or not.
            #
            # NOTE: The topic extractor calls .lower() on all tokens, so first names
            # in greeting lines ("Hi Jeremy,") are stored as "jeremy" in topic_counts.
            # Including common first-name patterns here prevents them from being
            # misclassified as interests. We avoid listing specific proper nouns
            # and instead catch the pattern via the marketing email filter at source —
            # but for residual facts already in the database, we add a few observed ones.
            'jeremy', 'hello', 'dear', 'friend', 'team', 'staff',
        }

        # --- Purge previously-stored garbage facts ---
        # If this blocklist has been expanded since last run, stale noise facts
        # (e.g., "expertise_more", "expertise_view", "interest_please") may still
        # exist in the database from earlier inference cycles. Remove any
        # expertise_*/interest_* facts whose topic word is now in the blocklist,
        # provided the fact has never been user-corrected (we never overwrite user edits).
        self._purge_noise_topic_facts(TOPIC_NOISE_BLOCKLIST)

        # --- Infer expertise from topic frequency ---
        # A topic becomes an expertise area if it appears in >10% of messages
        # and has been mentioned at least 10 times.
        total_samples = profile.get("samples_count", 1)
        filtered_count = 0

        for topic, count in topic_counts.items():
            # Skip noise tokens — HTML/CSS garbage and generic English words
            if topic.lower() in TOPIC_NOISE_BLOCKLIST:
                filtered_count += 1
                continue

            frequency_ratio = count / total_samples

            if count >= 5 and frequency_ratio > 0.08:
                # Frequently discussed topic — likely an expertise area
                self.ums.update_semantic_fact(
                    key=f"expertise_{topic}",
                    category="expertise",
                    value=topic,
                    confidence=min(0.95, base_confidence + frequency_ratio * 2),  # Higher freq = higher confidence
                    episode_id=episode_id,
                )
            elif count >= 3 and frequency_ratio > 0.03:
                # Moderately discussed topic — area of interest
                self.ums.update_semantic_fact(
                    key=f"interest_{topic}",
                    category="implicit_preference",
                    value=topic,
                    confidence=min(0.8, (base_confidence - 0.1) + frequency_ratio * 3),
                    episode_id=episode_id,
                )

        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} noise tokens from topic-based fact inference")
        logger.info(f"Inferred semantic facts from topic profile (samples={profile.get('samples_count')})")
        return {"type": "topic", "processed": True, "reason": None}

    def infer_from_cadence_profile(self):
        """
        Derive semantic facts from cadence signal profile.

        Analyzes temporal communication patterns to infer:
          - Work-life boundaries (office hours preferences)
          - Peak productivity times
          - Communication rhythm preferences

        Confidence threshold: Require 25+ samples to establish reliable
        cadence patterns across different times of day. Early inferences
        (25-50 samples) use reduced confidence via _early_inference_confidence scaling.
        """
        profile = self.ums.get_signal_profile("cadence")
        if not profile or profile.get("samples_count", 0) < 25:
            logger.info("Cadence profile has insufficient samples (<25), skipping inference")
            return {"type": "cadence", "processed": False, "reason": "insufficient samples (<25)"}

        # Scale confidence down for early inferences (between new threshold 25 and old threshold 50)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=50)

        data = profile["data"]
        # The cadence extractor stores data as "hourly_activity", not "hourly_distribution"
        hourly_activity = data.get("hourly_activity", {})

        # Get recent episodes to link as source evidence for cadence-based facts.
        # Cadence patterns are derived from all communication activity (email
        # received and sent, messages), so we do not restrict by interaction type.
        # The "communication" type label was retired by the episode backfill migration;
        # using no filter guarantees a valid episode_id whenever the episode log exists.
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer work-life boundaries from hourly activity ---
        # If 90%+ of messages are sent during business hours (9-17),
        # infer a preference for work-life separation.  Guard on total_messages
        # rather than returning early so the pre-computed derived-metric
        # inferences below (peak_hours, quiet_hours, domain response times) can
        # still run even when hourly_activity is empty.
        total_messages = sum(hourly_activity.values())
        if total_messages > 0:
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
                    confidence=min(0.95, base_confidence + (business_hours_ratio - 0.9) * 5),
                    episode_id=episode_id,
                )
            elif business_hours_ratio < 0.3:
                # Messages at all hours — flexible schedule or always-on work style
                self.ums.update_semantic_fact(
                    key="work_life_boundaries",
                    category="values",
                    value="flexible_boundaries",
                    confidence=min(0.85, base_confidence + (0.3 - business_hours_ratio) * 2),
                    episode_id=episode_id,
                )

            # --- Infer peak communication hours (single-hour legacy) ---
            # Find the hour with the highest message count — this is the user's
            # most active communication time.  The peak_hours multi-hour fact
            # below supersedes this for cases where multiple hours are active,
            # but keeping it ensures backward compatibility with downstream
            # code that may read peak_communication_hour directly.
            peak_hour = max(hourly_activity, key=hourly_activity.get)
            peak_count = hourly_activity[peak_hour]
            peak_ratio = peak_count / total_messages

            if peak_ratio > 0.2:  # Peak hour accounts for >20% of all messages
                self.ums.update_semantic_fact(
                    key="peak_communication_hour",
                    category="implicit_preference",
                    value=int(peak_hour),
                    confidence=min(0.9, base_confidence + peak_ratio * 2),
                    episode_id=episode_id,
                )

        # --- Infer peak communication hours from pre-computed peak_hours list ---
        # PR #276 added _compute_peak_hours() to the CadenceExtractor, which
        # produces a sorted list of hours (UTC) that exceed mean + 0.5σ activity.
        # This is more reliable than the single-hour approach above because it
        # captures a broad active window (e.g., [9,10,11,12,13]) rather than
        # just the single busiest hour, giving the LLM a richer picture.
        peak_hours = data.get("peak_hours", [])
        if len(peak_hours) >= 2:
            # Require at least 2 peak hours before storing — single-hour lists
            # mean the threshold was barely crossed and are too noisy to trust.
            self.ums.update_semantic_fact(
                key="peak_communication_hours",
                category="implicit_preference",
                value=peak_hours,  # Stored as a JSON-serialisable list, e.g. [9, 10, 11]
                confidence=min(0.9, base_confidence + len(peak_hours) * 0.04),
                episode_id=episode_id,
            )

        # --- Infer sleep / offline window from pre-computed quiet_hours_observed ---
        # PR #276 added _compute_quiet_hours() which finds contiguous spans of
        # near-zero activity (≤10% of peak hour).  The primary window (longest
        # or first detected span) is the most reliable proxy for the user's
        # typical sleep or offline period.  Knowing this helps the prediction
        # engine and briefing generator respect actual offline times rather than
        # relying solely on the configured quiet_hours setting.
        quiet_windows = data.get("quiet_hours_observed", [])
        if quiet_windows:
            # Take the first window — _compute_quiet_hours anchors on the first
            # non-quiet hour, so the first span is typically the largest one
            # (sleep window that crosses midnight, e.g., (22, 6)).
            start_h, end_h = quiet_windows[0]
            # Compute the span length (hours), accounting for midnight crossing.
            span_len = (end_h - start_h) % 24 if end_h != start_h else 24
            if span_len >= 3:
                # Require ≥3-hour span before treating it as a meaningful
                # offline window (< 3 hours is a gap in data, not sleep).
                self.ums.update_semantic_fact(
                    key="observed_quiet_window",
                    category="implicit_preference",
                    value=f"{start_h:02d}:00-{end_h:02d}:00 UTC",
                    confidence=min(0.85, base_confidence + min(span_len, 10) * 0.035),
                    episode_id=episode_id,
                )

        # --- Infer domain-level response priorities from avg_response_time_by_domain ---
        # PR #276 added _compute_domain_response_times() which groups per-contact
        # response times by email domain.  Domains the user consistently replies
        # to within 1 hour indicate high-priority communication relationships
        # (e.g., primary work domain, close family domain).  We store the top 3
        # fastest-reply domains so the prediction engine can weight them higher
        # when deciding whether to surface a follow-up prediction.
        domain_response_times = data.get("avg_response_time_by_domain", {})
        if len(domain_response_times) >= 2:
            # Sort ascending by average response time (seconds) — fastest first.
            sorted_domains = sorted(domain_response_times.items(), key=lambda x: x[1])
            fast_domains_stored = 0
            for domain, avg_seconds in sorted_domains:
                if fast_domains_stored >= 3:
                    # Cap at 3 domain facts to avoid polluting semantic memory
                    # with low-signal domains that happen to have fast responses.
                    break
                if avg_seconds > 3600:
                    # >1 hour average response = not a priority domain; stop
                    # adding facts since the list is sorted fastest-first.
                    break
                # Scale confidence: 0.7 at exactly 1 h, up to 0.9 for <6 min.
                confidence = min(0.9, 0.7 + max(0.0, (3600 - avg_seconds) / 3600) * 0.2)
                self.ums.update_semantic_fact(
                    key=f"high_priority_domain_{domain}",
                    category="implicit_preference",
                    value=f"avg_reply={int(avg_seconds // 60)}min",
                    confidence=confidence,
                    episode_id=episode_id,
                )
                fast_domains_stored += 1

        logger.info(f"Inferred semantic facts from cadence profile (samples={profile.get('samples_count')})")
        return {"type": "cadence", "processed": True, "reason": None}

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

        Confidence threshold: Require 3+ samples to enable early inference.
        This low threshold allows the system to start building mood-based
        semantic facts even with limited history. Early inferences (3-5 samples)
        use reduced confidence via _early_inference_confidence scaling.
        """
        profile = self.ums.get_signal_profile("mood_signals")
        if not profile or profile.get("samples_count", 0) < 3:
            logger.info("Mood profile has insufficient samples (<3), skipping inference")
            return {"type": "mood", "processed": False, "reason": "insufficient samples (<3)"}

        # Scale confidence down for early inferences (between new threshold 3 and old threshold 5)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=5)

        data = profile["data"]
        recent_signals = data.get("recent_signals", [])

        if not recent_signals:
            logger.info("Mood profile has no recent signals, skipping inference")
            return {"type": "mood", "processed": True, "reason": None}

        # Get recent episodes to link as source evidence for mood-based facts.
        # Mood signals are extracted from all interaction types, so we do not
        # restrict by interaction type.  The "communication" label was retired
        # by the episode backfill migration; using no filter ensures the audit
        # trail is populated whenever any episodic data exists.
        recent_episodes = self._get_recent_episodes(limit=5)
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
                    confidence=min(0.75, base_confidence + stress_ratio),
                    episode_id=episode_id,
                )
            elif stress_ratio < 0.1:  # <10% of signals show negative language
                # Low stress baseline — positive communication patterns
                self.ums.update_semantic_fact(
                    key="stress_baseline",
                    category="implicit_preference",
                    value="low_stress",
                    confidence=min(0.8, base_confidence + (0.1 - stress_ratio) * 3),
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
                    confidence=min(0.8, base_confidence + pressure_ratio * 2),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from mood profile (samples={profile.get('samples_count')})")
        return {"type": "mood", "processed": True, "reason": None}

    def infer_from_temporal_profile(self):
        """
        Derive semantic facts from temporal signal profile.

        Analyzes time-based behavior patterns to infer:
          - Chronotype (morning person vs. night owl)
          - Peak productivity windows
          - Work-life boundary preferences (strict vs. flexible)
          - Weekly rhythm patterns (productive days vs. social/recharge days)

        Confidence threshold: Require 25+ samples to establish reliable
        temporal patterns across different times of day and days of week.
        Early inferences (25-50 samples) use reduced confidence via
        _early_inference_confidence scaling.

        Semantic facts derived:
          - chronotype: "morning_person" or "night_owl" based on activity peaks
          - peak_productivity_hours: Time windows with highest activity
          - temporal_work_boundaries: Work-only-during-business-hours indicator
          - productive_day_preference: Which days show most work activity
        """
        profile = self.ums.get_signal_profile("temporal")
        if not profile or profile.get("samples_count", 0) < 25:
            logger.info("Temporal profile has insufficient samples (<25), skipping inference")
            return {"type": "temporal", "processed": False, "reason": "insufficient samples (<25)"}

        # Scale confidence down for early inferences (between new threshold 25 and old threshold 50)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=50)

        data = profile["data"]
        # Keys match what TemporalExtractor._update_profile() actually stores:
        # "activity_by_hour" and "activity_by_day" (not the old "hourly_activity" /
        # "weekly_activity" names that were never used by the extractor).
        hourly_activity = data.get("activity_by_hour", {})
        weekly_activity = data.get("activity_by_day", {})

        # Get recent episodes to link as source evidence for temporal facts
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer chronotype from hourly activity distribution ---
        # Morning person: >30% of activity before 11am
        # Night owl: >30% of activity after 8pm
        if hourly_activity:
            total_activity = sum(hourly_activity.values())
            if total_activity == 0:
                return {"type": "temporal", "processed": True, "reason": None}

            morning_activity = sum(
                count for hour, count in hourly_activity.items()
                if 6 <= int(hour) <= 10
            )
            evening_activity = sum(
                count for hour, count in hourly_activity.items()
                if 20 <= int(hour) <= 23
            )

            morning_ratio = morning_activity / total_activity
            evening_ratio = evening_activity / total_activity

            if morning_ratio > 0.3 and morning_ratio > evening_ratio * 1.5:
                # Strong morning person — most active 6-10am
                self.ums.update_semantic_fact(
                    key="chronotype",
                    category="implicit_preference",
                    value="morning_person",
                    confidence=min(0.9, base_confidence + 0.1 + morning_ratio),
                    episode_id=episode_id,
                )
            elif evening_ratio > 0.3 and evening_ratio > morning_ratio * 1.5:
                # Strong night owl — most active 8pm-11pm
                self.ums.update_semantic_fact(
                    key="chronotype",
                    category="implicit_preference",
                    value="night_owl",
                    confidence=min(0.9, base_confidence + 0.1 + evening_ratio),
                    episode_id=episode_id,
                )

            # --- Infer peak productivity hours ---
            # Find the 3-hour window with highest activity
            if len(hourly_activity) >= 3:
                peak_hour = max(hourly_activity, key=hourly_activity.get)
                peak_count = hourly_activity[peak_hour]
                peak_ratio = peak_count / total_activity

                if peak_ratio > 0.15:  # Peak hour accounts for >15% of activity
                    self.ums.update_semantic_fact(
                        key="peak_productivity_hour",
                        category="implicit_preference",
                        value=int(peak_hour),
                        confidence=min(0.85, base_confidence + peak_ratio * 2),
                        episode_id=episode_id,
                    )

        # --- Infer weekly rhythm patterns ---
        # Productive days: weekdays with high activity
        # Social days: days with social events
        # Recharge days: low-activity days (typically weekends)
        if weekly_activity:
            total_weekly = sum(weekly_activity.values())
            if total_weekly == 0:
                return {"type": "temporal", "processed": True, "reason": None}

            # Identify most productive day
            most_productive_day = max(weekly_activity, key=weekly_activity.get)
            productive_ratio = weekly_activity[most_productive_day] / total_weekly

            if productive_ratio > 0.25:  # One day dominates (>25% of weekly activity)
                self.ums.update_semantic_fact(
                    key="most_productive_day",
                    category="implicit_preference",
                    value=most_productive_day,
                    confidence=min(0.8, base_confidence + productive_ratio),
                    episode_id=episode_id,
                )

            # Detect weekend vs. weekday preferences
            weekend_activity = weekly_activity.get("saturday", 0) + weekly_activity.get("sunday", 0)
            weekday_activity = total_weekly - weekend_activity
            weekend_ratio = weekend_activity / total_weekly if total_weekly > 0 else 0

            if weekend_ratio < 0.1:  # <10% weekend activity
                # Strong weekday-only pattern — values work-life separation
                self.ums.update_semantic_fact(
                    key="temporal_work_boundaries",
                    category="values",
                    value="weekday_only_work",
                    confidence=min(0.9, base_confidence + 0.1 + (0.1 - weekend_ratio) * 5),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from temporal profile (samples={profile.get('samples_count')})")
        return {"type": "temporal", "processed": True, "reason": None}

    def infer_from_spatial_profile(self):
        """
        Derive semantic facts from spatial signal profile.

        Analyzes location-based behavior patterns to infer:
          - Primary work location (most frequent work-domain place)
          - Home office preference vs. external workspace
          - Travel frequency patterns
          - Location-based domain switching (work/personal by place)

        Confidence threshold: Require 5+ samples to avoid false positives
        from one-time event locations. Early inferences (5-10 samples) use
        reduced confidence via _early_inference_confidence scaling.

        Semantic facts derived:
          - primary_work_location: Most frequent work-domain place
          - work_location_type: "home_office" vs. "external_office" vs. "mobile_worker"
          - frequent_location_{place}: High-visit-count places
          - location_domain_{place}: Dominant domain (work/personal) per place
        """
        profile = self.ums.get_signal_profile("spatial")
        if not profile or profile.get("samples_count", 0) < 5:
            logger.info("Spatial profile has insufficient samples (<5), skipping inference")
            return {"type": "spatial", "processed": False, "reason": "insufficient samples (<5)"}

        # Scale confidence down for early inferences (between new threshold 5 and old threshold 10)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=10)

        data = profile["data"]
        # The spatial extractor stores data as JSON-encoded "place_behaviors"
        place_behaviors_json = data.get("place_behaviors", "{}")

        # Parse JSON if it's a string, otherwise use directly
        if isinstance(place_behaviors_json, str):
            try:
                place_behaviors = json.loads(place_behaviors_json)
            except Exception as e:
                logger.warning(f"Failed to parse place_behaviors JSON: {e}")
                return {"type": "spatial", "processed": True, "reason": None}
        else:
            place_behaviors = place_behaviors_json

        if not place_behaviors:
            logger.info("Spatial profile has no place behaviors, skipping inference")
            return {"type": "spatial", "processed": True, "reason": None}

        # Get recent episodes to link as source evidence for spatial facts
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Identify primary work location ---
        # Find the place with highest work-domain visit count
        work_places = {
            place_id: data
            for place_id, data in place_behaviors.items()
            if data.get("dominant_domain") == "work"
        }

        if work_places:
            primary_work_place = max(
                work_places.items(),
                key=lambda x: x[1].get("visit_count", 0)
            )
            place_id, place_data = primary_work_place
            visit_count = place_data.get("visit_count", 0)

            if visit_count >= 5:  # Require 5+ visits to establish as primary
                self.ums.update_semantic_fact(
                    key="primary_work_location",
                    category="implicit_preference",
                    value=place_id,
                    confidence=min(0.9, base_confidence + min(0.4, visit_count / 100)),
                    episode_id=episode_id,
                )

                # Infer work location type from place name patterns
                place_name_lower = place_id.lower()
                if any(keyword in place_name_lower for keyword in ["home", "residence", "apartment", "house"]):
                    # Work location is home-based
                    self.ums.update_semantic_fact(
                        key="work_location_type",
                        category="implicit_preference",
                        value="home_office",
                        confidence=min(0.85, base_confidence + 0.1 + min(0.25, visit_count / 200)),
                        episode_id=episode_id,
                    )
                elif any(keyword in place_name_lower for keyword in ["office", "building", "campus", "headquarters"]):
                    # Work location is external office
                    self.ums.update_semantic_fact(
                        key="work_location_type",
                        category="implicit_preference",
                        value="external_office",
                        confidence=min(0.85, base_confidence + 0.1 + min(0.25, visit_count / 200)),
                        episode_id=episode_id,
                    )

        # --- Identify frequent locations (any domain) ---
        # Places visited 10+ times are "frequent" and worth remembering
        total_visits = sum(p.get("visit_count", 0) for p in place_behaviors.values())

        for place_id, place_data in place_behaviors.items():
            visit_count = place_data.get("visit_count", 0)

            if visit_count >= 10:
                visit_ratio = visit_count / total_visits if total_visits > 0 else 0

                self.ums.update_semantic_fact(
                    key=f"frequent_location_{place_id}",
                    category="implicit_preference",
                    value=place_id,
                    confidence=min(0.9, base_confidence + visit_ratio * 3),
                    episode_id=episode_id,
                )

                # Also record dominant domain for this location
                dominant_domain = place_data.get("dominant_domain", "personal")
                if dominant_domain:
                    self.ums.update_semantic_fact(
                        key=f"location_domain_{place_id}",
                        category="implicit_preference",
                        value=dominant_domain,
                        confidence=min(0.85, base_confidence + 0.1 + visit_ratio * 2),
                        episode_id=episode_id,
                    )

        logger.info(f"Inferred semantic facts from spatial profile (samples={profile.get('samples_count')})")
        return {"type": "spatial", "processed": True, "reason": None}

    def infer_from_decision_profile(self):
        """
        Derive semantic facts from decision signal profile.

        Analyzes decision-making patterns to infer:
          - Risk tolerance by domain (conservative vs. aggressive)
          - Decision speed preferences (deliberate vs. quick)
          - Research depth patterns (gut feel vs. exhaustive research)
          - Delegation preferences (who the user defers to)

        Confidence threshold: Require 10+ samples to establish reliable
        decision patterns across different contexts. Early inferences
        (10-20 samples) use reduced confidence via _early_inference_confidence scaling.

        Semantic facts derived:
          - decision_speed_{domain}: How quickly user decides in different areas
          - risk_tolerance_{domain}: Conservative vs. aggressive by domain
          - research_preference: Gut-feel vs. data-driven decision maker
          - delegation_preference_{person}: Who user defers to for decisions
        """
        profile = self.ums.get_signal_profile("decision")
        if not profile or profile.get("samples_count", 0) < 10:
            logger.info("Decision profile has insufficient samples (<10), skipping inference")
            return {"type": "decision", "processed": False, "reason": "insufficient samples (<10)"}

        # Scale confidence down for early inferences (between new threshold 10 and old threshold 20)
        samples = profile.get("samples_count", 0)
        base_confidence = self._early_inference_confidence(samples, old_threshold=20)

        data = profile["data"]
        decision_speeds = data.get("decision_speed_by_domain", {})
        research_depths = data.get("research_depth_by_domain", {})

        # Get recent episodes to link as source evidence for decision facts
        recent_episodes = self._get_recent_episodes(limit=5)
        episode_id = recent_episodes[0] if recent_episodes else None

        # --- Infer decision speed preference by domain ---
        # Fast decisions: <60 seconds
        # Deliberate decisions: >1 day (86400 seconds)
        for domain, avg_seconds in decision_speeds.items():
            if avg_seconds < 60:
                # Very fast decision maker in this domain
                self.ums.update_semantic_fact(
                    key=f"decision_speed_{domain}",
                    category="implicit_preference",
                    value="quick_decision",
                    confidence=min(0.85, base_confidence + 0.1 + (60 - avg_seconds) / 100),
                    episode_id=episode_id,
                )
            elif avg_seconds > 86400:  # >1 day
                # Deliberate, slow decision maker in this domain
                self.ums.update_semantic_fact(
                    key=f"decision_speed_{domain}",
                    category="implicit_preference",
                    value="deliberate_decision",
                    confidence=min(0.85, base_confidence + min(0.35, avg_seconds / 604800)),  # Cap at 1 week
                    episode_id=episode_id,
                )

        # --- Infer research depth preference ---
        # Research depth: 0=gut feel, 1=exhaustive research
        for domain, depth_score in research_depths.items():
            if depth_score > 0.7:
                # Data-driven, exhaustive researcher in this domain
                self.ums.update_semantic_fact(
                    key=f"research_preference_{domain}",
                    category="implicit_preference",
                    value="data_driven",
                    confidence=min(0.9, base_confidence + depth_score / 2),
                    episode_id=episode_id,
                )
            elif depth_score < 0.3:
                # Gut-feel decision maker in this domain
                self.ums.update_semantic_fact(
                    key=f"research_preference_{domain}",
                    category="implicit_preference",
                    value="gut_feel",
                    confidence=min(0.85, base_confidence + (0.3 - depth_score)),
                    episode_id=episode_id,
                )

        # --- Infer overall risk tolerance ---
        # If user makes quick decisions with low research across multiple domains,
        # infer high risk tolerance
        if decision_speeds and research_depths:
            avg_decision_speed = sum(decision_speeds.values()) / len(decision_speeds)
            avg_research_depth = sum(research_depths.values()) / len(research_depths)

            # High risk: fast decisions + low research
            if avg_decision_speed < 300 and avg_research_depth < 0.3:  # <5 min + low research
                self.ums.update_semantic_fact(
                    key="risk_tolerance",
                    category="values",
                    value="high_risk_tolerance",
                    confidence=min(0.8, base_confidence + (0.3 - avg_research_depth) + (300 - avg_decision_speed) / 600),
                    episode_id=episode_id,
                )
            # Low risk: slow decisions + high research
            elif avg_decision_speed > 3600 and avg_research_depth > 0.7:  # >1 hour + high research
                self.ums.update_semantic_fact(
                    key="risk_tolerance",
                    category="values",
                    value="risk_averse",
                    confidence=min(0.8, base_confidence + avg_research_depth / 2),
                    episode_id=episode_id,
                )

        logger.info(f"Inferred semantic facts from decision profile (samples={profile.get('samples_count')})")
        return {"type": "decision", "processed": True, "reason": None}

    def _log_inference_summary(self, results: list[dict]) -> None:
        """
        Log a summary of which profile types were processed vs skipped.

        Called by run_all_inference() after all 9 inference methods complete.
        Provides at-a-glance observability at the default INFO log level so
        operators can tell whether the cognitive pipeline is actually running
        or still waiting for sufficient data.

        Args:
            results: List of status dicts from each infer_from_* method, each
                containing 'type', 'processed' (bool), and 'reason' (str or None).
        """
        processed = [r["type"] for r in results if r.get("processed")]
        skipped = [(r["type"], r.get("reason", "unknown")) for r in results if not r.get("processed")]

        processed_str = ", ".join(processed) if processed else "none"
        if skipped:
            skipped_parts = [f"{t} ({reason})" for t, reason in skipped]
            skipped_str = ", ".join(skipped_parts)
        else:
            skipped_str = "none"

        logger.info(
            "SemanticFactInferrer: inference cycle complete — processed: %s; skipped: %s",
            processed_str,
            skipped_str,
        )

    def run_all_inference(self):
        """
        Run inference across all signal profiles.

        This is the main entry point for semantic fact extraction. It
        analyzes all available signal profiles and derives semantic facts
        from each one. Each profile is processed independently so that a
        failure in one profile does not block inference for the others.

        Should be called:
          - Periodically (e.g., every 6 hours via background task)
          - After significant data ingestion (e.g., after syncing 1000+ new events)
          - On-demand via admin endpoint for testing
        """
        logger.info("Starting semantic fact inference across all profiles")

        methods = [
            ("linguistic", self.infer_from_linguistic_profile),
            ("inbound_linguistic", self.infer_from_inbound_linguistic_profile),
            ("relationship", self.infer_from_relationship_profile),
            ("topic", self.infer_from_topic_profile),
            ("cadence", self.infer_from_cadence_profile),
            ("mood", self.infer_from_mood_profile),
            ("temporal", self.infer_from_temporal_profile),
            ("spatial", self.infer_from_spatial_profile),
            ("decision", self.infer_from_decision_profile),
        ]
        results = []
        for name, method in methods:
            try:
                results.append(method())
            except Exception:
                logger.exception(
                    "SemanticFactInferrer: infer_from_%s_profile failed, continuing with remaining profiles", name
                )
                results.append({"type": name, "processed": False, "reason": "error"})

        self._log_inference_summary(results)
        logger.info("Completed semantic fact inference")
