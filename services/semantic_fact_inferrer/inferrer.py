"""
Life OS — Semantic Fact Inference Engine

Analyzes signal profiles to derive high-level semantic facts about the user.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
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
        self._last_inference_results: list[dict] = []
        self._last_inference_time: str | None = None
        self._total_facts_written_last_cycle: int = 0

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
        samples = profile.get("samples_count", 0)

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
            elif 0.3 <= formality <= 0.7 and samples >= 30:
                # Mid-range formality with sufficient data — user adapts formality
                # to context rather than having a fixed style. This IS a meaningful
                # communication style fact: context-adaptive communicators.
                self.ums.update_semantic_fact(
                    key="communication_style_formality",
                    category="implicit_preference",
                    value="balanced",
                    confidence=min(0.8, 0.3 + min(samples, 100) / 200),  # Grows with sample count
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
        elif 0.3 <= avg_formality <= 0.7 and samples >= 30:
            # Mixed formality environment — the user interacts with both
            # formal and casual contacts, indicating a diverse communication
            # environment (e.g., mix of work and personal contacts).
            self.ums.update_semantic_fact(
                key="inbound_communication_environment",
                category="implicit_preference",
                value="mixed_formality_environment",
                confidence=min(0.8, base_confidence * 0.8),
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
        logger.info("Relationship profile: %d total contacts", len(contacts))

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
        logger.info(
            "Bidirectional contacts (outbound > 0): %d out of %d",
            len(bidirectional_contacts), len(contacts),
        )

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
        logger.info(
            "Human contacts after marketing filter: %d out of %d bidirectional",
            len(human_contacts), len(bidirectional_contacts),
        )

        # Log the complete filter funnel for zero-fact diagnosis
        logger.info(
            "Relationship inference funnel: total_contacts=%d, "
            "bidirectional=%d, human_bidirectional=%d, "
            "inbound_only_total=%d",
            len(contacts),
            len(bidirectional_contacts),
            len(human_contacts),
            sum(1 for c in contacts.values()
                if isinstance(c, dict) and c.get("outbound_count", 0) == 0),
        )

        if not human_contacts:
            logger.info("No human bidirectional contacts found after marketing filter — running inbound-only inference")
            inbound_result = self._infer_from_inbound_only_contacts(contacts, base_confidence)
            # Even with no bidirectional contacts, domain breadth is a meaningful aggregate
            # fact: it measures whether the user's contact network spans many organisations
            # (indicating broad professional/social reach). Call aggregate with empty
            # human_contacts — it will only write domain breadth facts (which use all_contacts
            # and the shared marketing filter) and skip activity/network-size facts that
            # require bidirectional data.
            agg_count = self._infer_aggregate_relationship_facts(
                human_contacts={},
                all_contacts=contacts,
                base_confidence=base_confidence,
            )
            inbound_result["facts_written"] = inbound_result.get("facts_written", 0) + agg_count
            return inbound_result

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
        high_priority_count = sum(1 for c in human_contacts.values()
                                 if isinstance(c, dict) and c.get("interaction_count", 0) >= high_priority_threshold)
        logger.info(
            "Relationship threshold: avg=%.1f, high_priority_threshold=%.1f, contacts_exceeding=%d",
            avg_interactions, high_priority_threshold, high_priority_count,
        )

        facts_written = 0
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
                facts_written += 1

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
                facts_written += 1

            # --- Relationship balance (mutual vs. one-sided) ---
            inbound_count = contact_data.get("inbound_count", 0)
            outbound_count = contact_data.get("outbound_count", 0)
            total_count = inbound_count + outbound_count

            # Threshold lowered from 10 to 5: with many contacts having modest
            # individual counts, waiting for 10 interactions misses real patterns.
            if total_count >= 5:  # Need enough data to assess balance
                balance_ratio = min(inbound_count, outbound_count) / total_count

                if balance_ratio > 0.3:  # Both directions active (30%+ in each direction)
                    self.ums.update_semantic_fact(
                        key=f"relationship_balance_{contact_id}",
                        category="implicit_preference",
                        value="mutual",
                        confidence=min(0.85, base_confidence + balance_ratio),
                        episode_id=episode_id,
                    )
                    facts_written += 1
                elif outbound_count > inbound_count * 3:  # User initiates 3x more
                    self.ums.update_semantic_fact(
                        key=f"relationship_balance_{contact_id}",
                        category="implicit_preference",
                        value="user_initiated",
                        confidence=min(0.8, 0.5 + (outbound_count / total_count - 0.5)),
                        episode_id=episode_id,
                    )
                    facts_written += 1

        # Always generate aggregate-level relationship facts from the full human
        # contact network.  Per-contact facts may be sparse when 55K+ samples are
        # spread across hundreds of contacts whose individual interaction counts
        # don't exceed the 2x-average high-priority threshold.  Aggregate facts
        # (network size, activity level, domain breadth) fill this gap and are
        # reliable at any contact count >= 2.
        aggregate_count = self._infer_aggregate_relationship_facts(
            human_contacts=human_contacts,
            all_contacts=contacts,
            base_confidence=base_confidence,
        )
        facts_written += aggregate_count

        # Supplementary fallback: if the main bidirectional path produced 0 facts
        # (e.g., contacts have outbound_count=1 but don't meet interaction thresholds),
        # also run inbound-only inference to ensure at least basic relationship facts
        # are generated from the much larger pool of inbound-only contacts.
        if facts_written == 0:
            inbound_only_contacts = {
                cid: cdata for cid, cdata in contacts.items()
                if isinstance(cdata, dict) and cdata.get("outbound_count", 0) == 0
            }
            if len(inbound_only_contacts) >= 2:
                logger.info(
                    "Main relationship path produced 0 facts — running supplementary "
                    "inbound-only fallback (%d inbound-only contacts available)",
                    len(inbound_only_contacts),
                )
                fallback_result = self._infer_from_inbound_only_contacts(contacts, base_confidence)
                facts_written += fallback_result.get("facts_written", 0)

        logger.info(
            "Inferred %d semantic facts from relationship profile (samples=%s)",
            facts_written, profile.get("samples_count"),
        )
        return {"type": "relationship", "processed": True, "reason": None, "facts_written": facts_written}

    def _infer_from_inbound_only_contacts(self, contacts: dict, base_confidence: float) -> dict:
        """
        Fallback inference for when no bidirectional contacts exist.

        When the user has very few outbound messages (e.g., mostly receives email),
        the main relationship inference path produces zero facts because it requires
        outbound_count > 0.  This fallback derives communication volume and frequent
        sender facts from inbound-only contacts, after filtering out marketing/automated
        senders.

        Args:
            contacts: Full contacts dict from the relationship profile.
            base_confidence: Pre-computed confidence from _early_inference_confidence.

        Returns:
            Result dict with type, processed status, and fact count.
        """
        # Filter to non-marketing inbound-only contacts (outbound_count == 0)
        inbound_only = {
            contact_id: contact_data
            for contact_id, contact_data in contacts.items()
            if isinstance(contact_data, dict)
            and contact_data.get("outbound_count", 0) == 0
            and not is_marketing_or_noreply(contact_id)
        }

        # Log the filter funnel for diagnostics
        logger.info(
            "Inbound-only inference: total_contacts=%d, inbound_only_raw=%d, "
            "after_marketing_filter=%d, threshold=2",
            len(contacts),
            sum(1 for c in contacts.values()
                if isinstance(c, dict) and c.get("outbound_count", 0) == 0),
            len(inbound_only),
        )

        if len(inbound_only) < 2:
            logger.info(
                "Only %d inbound-only human contacts (need 2+), skipping inbound-only inference",
                len(inbound_only),
            )
            return {"type": "relationship", "processed": True, "reason": "too_few_inbound_only"}

        facts_written = 0

        # --- Communication volume category ---
        count = len(inbound_only)
        if count > 50:
            volume_value = "high_volume_email"
        elif count >= 10:
            volume_value = "moderate_volume_email"
        else:
            volume_value = "low_volume_email"

        self.ums.update_semantic_fact(
            key="communication_volume_category",
            category="implicit_preference",
            value=volume_value,
            confidence=min(0.9, base_confidence + 0.2),
            episode_id=None,
        )
        facts_written += 1

        # --- Top frequent personal senders ---
        # Even without outbound messages, contacts who send frequently
        # represent important relationships the user should be aware of.
        sorted_contacts = sorted(
            inbound_only.items(),
            key=lambda x: x[1].get("inbound_count", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        for contact_id, contact_data in sorted_contacts[:5]:
            if not isinstance(contact_data, dict):
                continue
            inbound_count = contact_data.get("inbound_count", 0)
            if inbound_count < 3:
                continue
            # Link to recent episodes with this contact for provenance
            contact_episodes = self._get_recent_episodes(contact=contact_id, limit=1)
            episode_id = contact_episodes[0] if contact_episodes else None

            self.ums.update_semantic_fact(
                key=f"frequent_sender_{contact_id}",
                category="implicit_preference",
                value="frequent_personal_sender",
                confidence=min(0.8, base_confidence + min(0.3, inbound_count / 50)),
                episode_id=episode_id,
            )
            facts_written += 1

        logger.info(
            "Inferred %d facts from %d inbound-only human contacts (fallback path)",
            facts_written, len(inbound_only),
        )
        return {
            "type": "relationship",
            "processed": True,
            "reason": None,
            "facts_written": facts_written,
            "funnel": {
                "total_contacts": len(contacts),
                "inbound_only_after_filter": len(inbound_only),
            },
        }

    def _infer_aggregate_relationship_facts(
        self,
        human_contacts: dict,
        all_contacts: dict,
        base_confidence: float,
    ) -> int:
        """
        Derive aggregate-level semantic facts from the relationship profile.

        Per-contact facts (relationship_priority_*, relationship_balance_*) are
        sparse when 55K+ samples are spread across hundreds of contacts whose
        individual interaction counts don't exceed the 2x-average threshold.
        This method fills that gap by generating facts about the user's
        *overall* communication network regardless of per-contact thresholds:

          - relationship_network_size: Categorizes the user's active human
            contact count (extensive / moderate / small).
          - regular_contact_count: Numeric count fact when 10+ contacts have
            5+ interactions, confirming a large active network.
          - communication_activity_level: Overall communication intensity
            derived from average interactions-per-human-contact.
          - contact_network_breadth: Domain diversity of the full contact list
            (distinct email domains after marketing filter).

        Args:
            human_contacts: Marketing-filtered bidirectional contacts dict
                (from ``infer_from_relationship_profile``).
            all_contacts: Complete, unfiltered contacts dict used for domain-
                diversity analysis (includes inbound-only contacts).
            base_confidence: Pre-computed confidence from
                ``_early_inference_confidence``.

        Returns:
            Number of aggregate facts written to semantic memory.
        """
        facts_written = 0

        # --- Regular contact count (human contacts with 5+ interactions) ---
        # Counts how many distinct people the user regularly communicates with.
        # A "regular contact" threshold of 5 interactions is deliberate: it
        # filters one-off exchanges while remaining reachable with modest data.
        regular_contacts = {
            cid: cdata
            for cid, cdata in human_contacts.items()
            if isinstance(cdata, dict) and cdata.get("interaction_count", 0) >= 5
        }
        regular_count = len(regular_contacts)

        if regular_count >= 10:
            # Large active network — user is highly connected across many people
            self.ums.update_semantic_fact(
                key="relationship_network_size",
                category="implicit_preference",
                value="extensive_network",
                confidence=min(0.9, base_confidence + 0.1 + min(0.3, regular_count / 100)),
                episode_id=None,
            )
            facts_written += 1
            # Precise count fact for downstream use in briefings and predictions
            self.ums.update_semantic_fact(
                key="regular_contact_count",
                category="implicit_preference",
                value=regular_count,
                confidence=min(0.85, base_confidence + 0.1),
                episode_id=None,
            )
            facts_written += 1
        elif regular_count >= 5:
            # Moderate active network
            self.ums.update_semantic_fact(
                key="relationship_network_size",
                category="implicit_preference",
                value="moderate_network",
                confidence=min(0.8, base_confidence + min(0.2, regular_count / 50)),
                episode_id=None,
            )
            facts_written += 1
        elif regular_count >= 2:
            # Small but real network (meaningful even at low counts)
            self.ums.update_semantic_fact(
                key="relationship_network_size",
                category="implicit_preference",
                value="small_network",
                confidence=min(0.7, base_confidence),
                episode_id=None,
            )
            facts_written += 1

        # --- Communication activity level ---
        # Derived from average interactions-per-human-contact, indicating
        # whether the user is a high-volume communicator or more selective.
        if human_contacts:
            total_interactions = sum(
                cdata.get("interaction_count", 0)
                for cdata in human_contacts.values()
                if isinstance(cdata, dict)
            )
            avg_per_contact = total_interactions / len(human_contacts)

            if avg_per_contact >= 20:
                activity_value = "highly_active_communicator"
                activity_confidence = min(0.85, base_confidence + 0.15)
            elif avg_per_contact >= 10:
                activity_value = "moderately_active_communicator"
                activity_confidence = min(0.8, base_confidence + 0.1)
            elif avg_per_contact >= 5:
                activity_value = "regular_communicator"
                activity_confidence = min(0.75, base_confidence + 0.05)
            else:
                activity_value = None

            if activity_value is not None:
                self.ums.update_semantic_fact(
                    key="communication_activity_level",
                    category="implicit_preference",
                    value=activity_value,
                    confidence=activity_confidence,
                    episode_id=None,
                )
                facts_written += 1

        # --- Contact network breadth (email domain diversity) ---
        # Counts distinct email domains in the full contact list (after marketing
        # filter) to determine whether the user's network spans many organisations
        # (broad) or is concentrated in one or two domains (narrow).
        domains: set[str] = set()
        for cid in all_contacts:
            if "@" in cid and not is_marketing_or_noreply(cid):
                domain = cid.split("@", 1)[1].lower()
                domains.add(domain)

        domain_count = len(domains)
        if domain_count >= 10:
            self.ums.update_semantic_fact(
                key="contact_network_breadth",
                category="implicit_preference",
                value="diverse_multi_domain_network",
                confidence=min(0.85, base_confidence + 0.1 + min(0.25, domain_count / 100)),
                episode_id=None,
            )
            facts_written += 1
        elif domain_count >= 4:
            self.ums.update_semantic_fact(
                key="contact_network_breadth",
                category="implicit_preference",
                value="moderate_domain_diversity",
                confidence=min(0.75, base_confidence + min(0.15, domain_count / 40)),
                episode_id=None,
            )
            facts_written += 1

        logger.info(
            "Aggregate relationship facts: regular_count=%d, domains=%d, "
            "facts_written=%d",
            regular_count, domain_count, facts_written,
        )
        return facts_written

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
        # Fetch relationship_priority_*, relationship_balance_*,
        # relationship_multichannel_*, and frequent_sender_* facts that are not
        # user-corrected.  frequent_sender_* facts are created by
        # _infer_from_inbound_only_contacts() and must also be purged when the
        # marketing filter improves and catches previously-missed senders.
        with self.ums.db.get_connection("user_model") as conn:
            rows = conn.execute(
                "SELECT key FROM semantic_facts "
                "WHERE (key LIKE 'relationship_priority_%' "
                "   OR key LIKE 'relationship_balance_%' "
                "   OR key LIKE 'relationship_multichannel_%' "
                "   OR key LIKE 'frequent_sender_%') "
                "AND is_user_corrected = 0"
            ).fetchall()

            for row in rows:
                key = row["key"]
                # Extract contact_id: everything after the prefix and its underscore(s).
                # relationship facts: "relationship_priority_addr@domain.com"
                #   → split("_", 2) gives ["relationship", "priority", "addr@domain.com"]
                # frequent_sender facts: "frequent_sender_addr@domain.com"
                #   → split("_", 1) gives ["frequent", "sender_addr@domain.com"]
                #   We need the part after "frequent_sender_", i.e. split("_", 2) gives
                #   ["frequent", "sender", "addr@domain.com"]
                parts = key.split("_", 2)
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
                "(relationship_priority/balance/multichannel/frequent_sender for automated senders)",
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
        logger.info(
            "Topic profile loaded: samples_count=%d, topic_counts_size=%d",
            samples, len(topic_counts),
        )

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
        facts_created = 0
        facts_failed = 0

        # Diagnostic logging: show threshold requirements at current sample size
        non_noise_count = sum(1 for t in topic_counts if t.lower() not in TOPIC_NOISE_BLOCKLIST)
        min_expertise_count = max(5, int(total_samples * 0.08))
        min_interest_count = max(3, int(total_samples * 0.03))
        logger.info(
            "Topic threshold diagnostics: total_topics=%d, non_noise_topics=%d, "
            "total_samples=%d, min_count_for_expertise=%d (8%%), min_count_for_interest=%d (3%%)",
            len(topic_counts), non_noise_count, total_samples,
            min_expertise_count, min_interest_count,
        )

        for topic, count in topic_counts.items():
            # Skip noise tokens — HTML/CSS garbage and generic English words
            if topic.lower() in TOPIC_NOISE_BLOCKLIST:
                filtered_count += 1
                continue

            frequency_ratio = count / total_samples

            if count >= 5 and frequency_ratio > 0.08:
                # Frequently discussed topic — likely an expertise area
                logger.debug(
                    "Creating expertise fact: topic=%s, count=%d, frequency=%.3f",
                    topic, count, frequency_ratio,
                )
                try:
                    self.ums.update_semantic_fact(
                        key=f"expertise_{topic}",
                        category="expertise",
                        value=topic,
                        confidence=min(0.95, base_confidence + frequency_ratio * 2),  # Higher freq = higher confidence
                        episode_id=episode_id,
                    )
                    facts_created += 1
                except Exception as e:
                    logger.error("Failed to store expertise fact for topic %s: %s", topic, e)
                    facts_failed += 1
            elif count >= 3 and frequency_ratio > 0.03:
                # Moderately discussed topic — area of interest
                logger.debug(
                    "Creating interest fact: topic=%s, count=%d, frequency=%.3f",
                    topic, count, frequency_ratio,
                )
                try:
                    self.ums.update_semantic_fact(
                        key=f"interest_{topic}",
                        category="implicit_preference",
                        value=topic,
                        confidence=min(0.8, (base_confidence - 0.1) + frequency_ratio * 3),
                        episode_id=episode_id,
                    )
                    facts_created += 1
                except Exception as e:
                    logger.error("Failed to store interest fact for topic %s: %s", topic, e)
                    facts_failed += 1

        # --- Top-N relative fallback for sparse data ---
        # When standard thresholds produce 0 facts (e.g., marketing email dilution
        # pushes real topics below absolute frequency thresholds), fall back to a
        # relative approach: take the top non-noise topics by count with reduced
        # confidence to still capture user interests from sparse data.
        if facts_created == 0:
            fallback_candidates = sorted(
                (
                    (topic, count)
                    for topic, count in topic_counts.items()
                    if topic.lower() not in TOPIC_NOISE_BLOCKLIST and count >= 2
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            if fallback_candidates:
                logger.info(
                    "Standard thresholds produced 0 facts; using top-N relative fallback "
                    "with %d candidate topics",
                    len(fallback_candidates),
                )
                for topic, count in fallback_candidates:
                    fallback_confidence = min(0.6, base_confidence * 0.6 + count / total_samples)
                    try:
                        self.ums.update_semantic_fact(
                            key=f"interest_{topic}",
                            category="implicit_preference",
                            value=topic,
                            confidence=fallback_confidence,
                            episode_id=episode_id,
                        )
                        facts_created += 1
                    except Exception as e:
                        logger.error("Failed to store fallback interest fact for topic %s: %s", topic, e)
                        facts_failed += 1

        # --- Infer diverse interests when no single topic dominates ---
        # If no topic exceeds the expertise threshold (8%) but there are many
        # non-noise topics with meaningful counts, the user is a generalist or
        # polymath with broad interests. This captures users who discuss many
        # topics without deep specialization in any single area.
        non_noise_topics = {
            topic: count
            for topic, count in topic_counts.items()
            if topic.lower() not in TOPIC_NOISE_BLOCKLIST and count >= 3
        }
        has_dominant_topic = any(
            count / total_samples > 0.08 for count in non_noise_topics.values()
        ) if total_samples > 0 else False

        if not has_dominant_topic and len(non_noise_topics) >= 5:
            try:
                self.ums.update_semantic_fact(
                    key="topic_breadth",
                    category="implicit_preference",
                    value="diverse_interests",
                    confidence=min(0.8, base_confidence * 0.8 + len(non_noise_topics) * 0.02),
                    episode_id=episode_id,
                )
                facts_created += 1
            except Exception as e:
                logger.error("Failed to store diverse_interests fact: %s", e)
                facts_failed += 1

        surviving = len(topic_counts) - filtered_count
        logger.info(
            "Topics after noise filter: %d survived out of %d total (%d noise tokens filtered)",
            surviving, len(topic_counts), filtered_count,
        )
        logger.info(
            "Topic inference complete: %d facts created, %d failed (samples=%d)",
            facts_created, facts_failed, total_samples,
        )
        return {"type": "topic", "processed": True, "reason": None, "facts_written": facts_created}

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
            elif 0.3 <= business_hours_ratio <= 0.9:
                # Moderate work-life boundaries — mostly business hours but with
                # some evening/weekend activity. Typical of professionals who
                # occasionally handle personal comms during work or check work
                # messages in the evening.
                self.ums.update_semantic_fact(
                    key="work_life_boundaries",
                    category="values",
                    value="moderate_boundaries",
                    confidence=min(0.8, base_confidence * 0.9),
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

            if peak_ratio > 0.12:  # Peak hour accounts for >12% of all messages (~3x uniform baseline)
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

    def _get_episode_count(self) -> int:
        """Return the number of rows in the episodes table.

        Used to detect the cold-start condition where the episode pipeline is
        broken or the system has not yet generated any episodes from events.

        Returns:
            Episode count, or 0 on any database error (fail-open).
        """
        try:
            with self.ums.db.get_connection("user_model") as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except Exception:
            return 0

    def _get_event_count(self) -> int:
        """Return the total number of rows in the events table.

        Used to confirm that there is enough raw data to make event-based
        fallback inference worthwhile even when episodes are absent.

        Returns:
            Event count, or 0 on any database error (fail-open).
        """
        try:
            with self.ums.db.get_connection("events") as conn:
                return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        except Exception:
            return 0

    def infer_facts_from_events(self, event_limit: int = 5000) -> dict:
        """Infer basic semantic facts directly from the events table.

        This is a cold-start fallback for when the episodes table is empty.
        It queries events.db directly — bypassing the signal-profile pipeline —
        to extract relationship, temporal, and topic facts.  All derived facts
        are stored with ``confidence=0.3`` to signal that they come from
        raw event data rather than the richer episodic signal profiles.

        Facts extracted:
          - ``event_fallback_contact_*``: Top-10 most-frequent email senders
            (email.received), filtered for marketing/no-reply addresses and
            classified as "work" or "personal" by domain.
          - ``event_fallback_active_hours``: Hours of day (0-23) with
            above-average email/calendar activity — proxy for peak productivity.
          - ``event_fallback_most_active_day``: Day of week with most events.
          - ``event_fallback_topic_*``: Top-10 words from email subject lines
            after removing stop words and standard reply/forward prefixes.

        This method supplements (never replaces) the normal signal-profile
        inference path.  When episodes exist, ``run_all_inference`` skips this
        method entirely.

        Args:
            event_limit: Maximum number of email/calendar events to scan.
                Defaults to 5000 to keep execution time below ~1 s on large
                event logs.

        Returns:
            Dict with keys: ``type``, ``processed``, ``reason``, ``facts_written``.
        """
        # Low confidence constant — event-only facts are preliminary estimates,
        # not the richer signal-profile inferences that run when episodes exist.
        COLD_START_CONFIDENCE = 0.3

        # Days of week by weekday() index (Monday=0)
        DAY_NAMES = [
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday",
        ]

        # Stop words that should never appear as topic facts from subject lines.
        # Covers common English words, email prefixes, and generic marketing terms.
        SUBJECT_STOP_WORDS = {
            "re", "fwd", "fw", "the", "a", "an", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "is", "it", "be",
            "as", "by", "with", "from", "your", "our", "we", "you",
            "i", "my", "are", "was", "has", "have", "this", "that",
            "not", "no", "can", "will", "just", "up", "out", "if",
            "about", "all", "please", "here", "more", "new", "get",
            "hi", "hello", "dear", "thank", "thanks", "update",
            "email", "message", "notification", "reminder", "info",
            "http", "https", "www", "com", "org", "net",
        }

        # Common personal email provider domains — everything else is treated
        # as a potential work domain (the "company.com → work" heuristic from
        # the task description).
        PERSONAL_DOMAINS = {
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
            "live.com", "icloud.com", "me.com", "aol.com",
            "protonmail.com", "proton.me", "hey.com", "fastmail.com",
            "ymail.com", "mail.com", "gmx.com",
        }

        # --- Step 1: Query events.db ---
        try:
            with self.ums.db.get_connection("events") as conn:
                rows = conn.execute(
                    """SELECT type, timestamp, payload
                       FROM events
                       WHERE type IN (
                           'email.received', 'email.sent',
                           'calendar.event.created'
                       )
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (event_limit,),
                ).fetchall()
        except Exception as exc:
            logger.warning("infer_facts_from_events: events query failed: %s", exc)
            return {
                "type": "event_fallback",
                "processed": False,
                "reason": f"events query failed: {exc}",
                "facts_written": 0,
            }

        if not rows:
            logger.info("infer_facts_from_events: no email/calendar events found in events.db")
            return {
                "type": "event_fallback",
                "processed": False,
                "reason": "no_events",
                "facts_written": 0,
            }

        # --- Step 2: Aggregate data from events ---
        # contact_counts: email address → inbound message count
        contact_counts: dict[str, int] = {}
        # hour_counts: 0-23 → event count
        hour_counts: dict[int, int] = {}
        # day_counts: "monday" etc. → event count
        day_counts: dict[str, int] = {}
        # subject_word_counts: lowercase word → occurrence count
        subject_word_counts: dict[str, int] = {}

        for row in rows:
            event_type = row["type"]
            payload: dict = {}
            try:
                payload = json.loads(row["payload"])
            except Exception:
                pass  # Treat malformed payload as empty — fail-open

            # -- Relationship: collect email senders --
            if event_type == "email.received":
                from_addr = payload.get("from_address", "")
                if isinstance(from_addr, str) and "@" in from_addr:
                    addr = from_addr.lower().strip()
                    if not is_marketing_or_noreply(addr):
                        contact_counts[addr] = contact_counts.get(addr, 0) + 1

            # -- Temporal: bucket event by hour and day of week --
            try:
                ts = row["timestamp"]
                if ts:
                    # Normalize 'Z' suffix so fromisoformat handles it on all
                    # Python 3.12 builds (fromisoformat('...Z') was added in 3.11
                    # but is not guaranteed across all patch versions).
                    ts_norm = ts.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts_norm)
                    hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
                    day_name = DAY_NAMES[dt.weekday()]
                    day_counts[day_name] = day_counts.get(day_name, 0) + 1
            except Exception:
                pass  # Skip unparseable timestamps — fail-open

            # -- Topic: extract keywords from email subject lines --
            if event_type in ("email.received", "email.sent"):
                subject = payload.get("subject", "")
                if isinstance(subject, str) and subject.strip():
                    cleaned = subject.strip()
                    # Strip common reply/forward prefixes (may be nested, e.g. "Re: Fwd:")
                    for prefix in ("Re: ", "RE: ", "Fwd: ", "FWD: ", "FW: ",
                                   "re: ", "fwd: ", "fw: "):
                        while cleaned.lower().startswith(prefix.lower()):
                            cleaned = cleaned[len(prefix):].strip()

                    for raw_word in cleaned.lower().split():
                        # Remove surrounding punctuation
                        word = raw_word.strip(".,!?;:\"'()[]{}-_/\\|@#$%^&*+=<>~`")
                        if (
                            len(word) >= 4
                            and word not in SUBJECT_STOP_WORDS
                            and word.isalpha()
                        ):
                            subject_word_counts[word] = (
                                subject_word_counts.get(word, 0) + 1
                            )

        # --- Step 3: Store facts ---
        facts_written = 0

        # Relationship facts: top-10 most-frequent non-marketing email senders
        if contact_counts:
            sorted_contacts = sorted(
                contact_counts.items(), key=lambda x: x[1], reverse=True
            )
            for contact_addr, count in sorted_contacts[:10]:
                if count < 2:
                    # Not enough volume to be a meaningful contact
                    break

                # Classify as work or personal by email domain
                domain = contact_addr.rsplit("@", 1)[-1] if "@" in contact_addr else ""
                relationship_type = (
                    "personal" if domain in PERSONAL_DOMAINS else "work"
                )

                # Build a key-safe representation of the email address
                # (keys cannot contain '@' or '.' as fact key separators)
                safe_addr = (
                    contact_addr.replace("@", "_at_").replace(".", "_dot_")
                )

                try:
                    self.ums.update_semantic_fact(
                        key=f"event_fallback_contact_{safe_addr}",
                        category="implicit_preference",
                        value={
                            "email": contact_addr,
                            "email_count": count,
                            "relationship_type": relationship_type,
                        },
                        confidence=COLD_START_CONFIDENCE,
                        episode_id=None,
                    )
                    facts_written += 1
                except Exception as exc:
                    logger.warning(
                        "infer_facts_from_events: failed to store contact fact "
                        "for %s: %s", contact_addr, exc
                    )

        # Temporal facts: active hours and most active day
        if hour_counts:
            total_hour_events = sum(hour_counts.values())
            if total_hour_events > 0:
                # Active hours: those that exceed 1.5x the per-hour average.
                # This threshold separates genuine peak hours from background noise
                # while remaining reachable even with sparse data (24 hours, so
                # average is total/24; 1.5x means the hour has at least 6.25% of
                # all events when they were perfectly uniformly distributed).
                avg_per_hour = total_hour_events / 24
                active_hours = sorted(
                    [h for h, c in hour_counts.items() if c > avg_per_hour * 1.5],
                    key=lambda h: hour_counts[h],
                    reverse=True,
                )
                if active_hours:
                    try:
                        self.ums.update_semantic_fact(
                            key="event_fallback_active_hours",
                            category="implicit_preference",
                            value=active_hours[:6],  # Cap at 6 hours
                            confidence=COLD_START_CONFIDENCE,
                            episode_id=None,
                        )
                        facts_written += 1
                    except Exception as exc:
                        logger.warning(
                            "infer_facts_from_events: failed to store "
                            "active_hours fact: %s", exc
                        )

        if day_counts:
            most_active_day = max(day_counts, key=day_counts.get)
            try:
                self.ums.update_semantic_fact(
                    key="event_fallback_most_active_day",
                    category="implicit_preference",
                    value=most_active_day,
                    confidence=COLD_START_CONFIDENCE,
                    episode_id=None,
                )
                facts_written += 1
            except Exception as exc:
                logger.warning(
                    "infer_facts_from_events: failed to store "
                    "most_active_day fact: %s", exc
                )

        # Topic facts: top-10 most-common subject keywords (min 2 occurrences)
        if subject_word_counts:
            sorted_words = sorted(
                subject_word_counts.items(), key=lambda x: x[1], reverse=True
            )
            stored_topics = 0
            for word, count in sorted_words:
                if stored_topics >= 10:
                    break
                if count < 2:
                    break  # List is sorted descending; no subsequent entry can qualify
                try:
                    self.ums.update_semantic_fact(
                        key=f"event_fallback_topic_{word}",
                        category="implicit_preference",
                        value={"topic": word, "count": count},
                        confidence=COLD_START_CONFIDENCE,
                        episode_id=None,
                    )
                    facts_written += 1
                    stored_topics += 1
                except Exception as exc:
                    logger.warning(
                        "infer_facts_from_events: failed to store "
                        "topic fact for %s: %s", word, exc
                    )

        logger.info(
            "infer_facts_from_events: scanned %d events, wrote %d facts "
            "(unique_senders=%d, hour_buckets=%d, subject_words=%d)",
            len(rows), facts_written,
            len(contact_counts), len(hour_counts), len(subject_word_counts),
        )

        return {
            "type": "event_fallback",
            "processed": True,
            "reason": None,
            "facts_written": facts_written,
        }

    def _log_inference_summary(self, results: list[dict]) -> None:
        """
        Log a summary of which profile types were processed vs skipped.

        Called by run_all_inference() after all 9 inference methods complete.
        Provides at-a-glance observability at the default INFO log level so
        operators can tell whether the cognitive pipeline is actually running
        or still waiting for sufficient data.

        The per-method breakdown in the log line is the key diagnostic for
        "why are there 0 facts?": it shows which methods ran, which were
        skipped, and how many new facts each method produced this cycle.

        Args:
            results: List of status dicts from each infer_from_* method, each
                containing 'type', 'processed' (bool), 'reason' (str or None),
                and 'facts_written' (int) counting new fact insertions.
        """
        processed = [r["type"] for r in results if r.get("processed")]
        skipped = [(r["type"], r.get("reason", "unknown")) for r in results if not r.get("processed")]

        # Sum facts_written across all results
        total_facts = sum(r.get("facts_written", 0) for r in results)

        # Per-method breakdown: "linguistic=0, relationship=3, topic=1, ..."
        # This is the first place to look when total_facts=0 despite data existing:
        # a method with processed=True but facts_written=0 indicates a threshold
        # or filter issue rather than missing data.
        per_method_parts = [
            f"{r['type']}={'skipped' if not r.get('processed') else r.get('facts_written', '?')}"
            for r in results
        ]
        per_method_str = ", ".join(per_method_parts)

        processed_str = ", ".join(processed) if processed else "none"
        if skipped:
            skipped_parts = [f"{t} ({reason})" for t, reason in skipped]
            skipped_str = ", ".join(skipped_parts)
        else:
            skipped_str = "none"

        logger.info(
            "SemanticFactInferrer: inference cycle complete — "
            "processed: %s; skipped: %s; total_facts_written: %d; "
            "per_method: [%s]",
            processed_str,
            skipped_str,
            total_facts,
            per_method_str,
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

        Cold-start diagnostics:
          Before running inference, logs each profile's sample count so operators
          can immediately see which profiles have data and which are empty. When
          facts = 0 despite having data, check this log line first — it will show
          whether the issue is missing data (all counts = 0) or a threshold/filter
          bug (counts > 0 but facts still not written).
        """
        logger.info("Starting semantic fact inference across all profiles")

        # --- Cold-start profile availability diagnostics ---
        # Log sample counts for all profiles BEFORE running inference.  This
        # single log line is the first place to look when semantic facts remain
        # at 0 despite the system having processed many events.
        # Format: "linguistic=0, relationship=220351, topic=95000, ..."
        _PROFILE_TYPE_MAP = [
            ("linguistic", "linguistic"),
            ("inbound_linguistic", "linguistic_inbound"),
            ("relationship", "relationships"),
            ("topic", "topics"),
            ("cadence", "cadence"),
            ("mood", "mood_signals"),
            ("temporal", "temporal"),
            ("spatial", "spatial"),
            ("decision", "decision"),
        ]
        profile_samples: dict[str, int] = {}
        for method_name, profile_type in _PROFILE_TYPE_MAP:
            try:
                p = self.ums.get_signal_profile(profile_type)
                profile_samples[method_name] = p.get("samples_count", 0) if p else 0
            except Exception:
                # Fail-open: log -1 to signal a read error rather than "no data"
                profile_samples[method_name] = -1

        sample_summary = ", ".join(f"{k}={v}" for k, v in profile_samples.items())
        logger.info(
            "SemanticFactInferrer cold-start diagnostics — profile sample counts: %s",
            sample_summary,
        )

        # Snapshot current fact count for per-cycle delta tracking
        try:
            with self.ums.db.get_connection("user_model") as conn:
                _total_facts_before = conn.execute(
                    "SELECT COUNT(*) FROM semantic_facts"
                ).fetchone()[0]
        except Exception:
            _total_facts_before = 0

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
            # Snapshot per-method fact count so we can compute an accurate delta
            # even for methods that don't return a facts_written key themselves.
            try:
                with self.ums.db.get_connection("user_model") as conn:
                    _facts_before_method = conn.execute(
                        "SELECT COUNT(*) FROM semantic_facts"
                    ).fetchone()[0]
            except Exception:
                _facts_before_method = 0

            try:
                result = method()
                if result is None:
                    result = {"type": name, "processed": False, "reason": "no result returned"}
            except Exception:
                logger.exception(
                    "SemanticFactInferrer: infer_from_%s_profile failed, continuing with remaining profiles", name
                )
                result = {"type": name, "processed": False, "reason": "error", "facts_written": 0}

            # Compute per-method facts_written as a DB delta when the method
            # didn't report it directly.  This captures new fact insertions
            # (existing facts only get confidence-bumped, so COUNT(*) won't
            # rise for them — but even 0 delta is informative for cold-start
            # debugging: it shows the method ran without errors or skips).
            if "facts_written" not in result:
                try:
                    with self.ums.db.get_connection("user_model") as conn:
                        _facts_after_method = conn.execute(
                            "SELECT COUNT(*) FROM semantic_facts"
                        ).fetchone()[0]
                    result["facts_written"] = max(0, _facts_after_method - _facts_before_method)
                except Exception:
                    result["facts_written"] = 0

            results.append(result)

        # --- Event-based fallback: supplement when episodes are absent ---
        # When the episode pipeline is broken or the system is very new, all
        # profile-based inference methods above skip because signal profiles are
        # populated from episodes.  The fallback below queries events.db directly
        # to extract basic facts (top contacts, active hours, subject keywords)
        # so semantic memory is not completely empty, unblocking the prediction
        # engine and morning briefing.
        #
        # Gate: run only when episodes=0 AND events>100 so we don't pollute
        # semantic memory with noisy event-only facts when the richer
        # episode-backed signal profiles are already available.
        try:
            _episode_count_for_fallback = self._get_episode_count()
            _event_count_for_fallback = self._get_event_count()
            if _episode_count_for_fallback == 0 and _event_count_for_fallback > 100:
                logger.info(
                    "SemanticFactInferrer: 0 episodes with %d events — "
                    "running event-based fallback inference",
                    _event_count_for_fallback,
                )
                fallback_result = self.infer_facts_from_events()
                results.append(fallback_result)
            elif _episode_count_for_fallback > 0:
                logger.debug(
                    "SemanticFactInferrer: %d episodes present — "
                    "skipping event-based fallback (normal path active)",
                    _episode_count_for_fallback,
                )
        except Exception:
            logger.exception(
                "SemanticFactInferrer: event fallback check failed, continuing"
            )

        self._log_inference_summary(results)

        # Log overall cycle delta (new facts inserted this cycle vs. total in DB)
        try:
            with self.ums.db.get_connection("user_model") as conn:
                _total_facts_after = conn.execute(
                    "SELECT COUNT(*) FROM semantic_facts"
                ).fetchone()[0]
            logger.info(
                "SemanticFactInferrer cycle complete — new_facts_inserted=%d, "
                "total_facts_in_db=%d",
                max(0, _total_facts_after - _total_facts_before),
                _total_facts_after,
            )
        except Exception:
            pass  # Fail-open: don't crash the inference loop on a diagnostic query

        # Cache results for diagnostics endpoint (/health, /admin)
        self._last_inference_results = results
        self._last_inference_time = datetime.now(timezone.utc).isoformat()
        self._total_facts_written_last_cycle = sum(r.get("facts_written", 0) for r in results)

        logger.info("Completed semantic fact inference")

    def get_diagnostics(self) -> dict:
        """Return inference engine diagnostic information.

        Follows the same pattern as PredictionEngine.get_diagnostics()
        and RoutineDetector.get_diagnostics(). Each query is wrapped in
        try/except so a single DB failure doesn't prevent other diagnostics
        from returning.

        Returns:
            Dict with keys: last_inference_time, total_facts_written_last_cycle,
            profile_availability, last_cycle, total_facts, health.
        """
        result: dict = {
            "last_inference_time": self._last_inference_time,
            "total_facts_written_last_cycle": self._total_facts_written_last_cycle,
        }

        # Profile availability: check which profiles have enough samples
        profile_availability: dict = {}
        for ptype in [
            "linguistic",
            "linguistic_inbound",
            "relationships",
            "topics",
            "cadence",
            "mood_signals",
            "temporal",
            "spatial",
            "decision",
        ]:
            try:
                profile = self.ums.get_signal_profile(ptype)
                if profile:
                    profile_availability[ptype] = {
                        "available": True,
                        "samples": profile.get("samples_count", 0),
                    }
                else:
                    profile_availability[ptype] = {"available": False, "samples": 0}
            except Exception:
                profile_availability[ptype] = {"available": False, "samples": 0, "error": True}
        result["profile_availability"] = profile_availability

        # Last cycle results (processed vs skipped)
        if self._last_inference_results:
            result["last_cycle"] = {
                "processed": [r["type"] for r in self._last_inference_results if r.get("processed")],
                "skipped": [
                    {"type": r["type"], "reason": r.get("reason", "unknown")}
                    for r in self._last_inference_results
                    if not r.get("processed")
                ],
            }
        else:
            result["last_cycle"] = None

        # Current fact count from database
        try:
            with self.ums.db.get_connection("user_model") as conn:
                row = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()
                result["total_facts"] = row[0] if row else 0
        except Exception:
            result["total_facts"] = -1

        # Health assessment
        available_count = sum(1 for v in profile_availability.values() if v.get("available"))
        if available_count == 0:
            result["health"] = "no_data"
        elif result.get("total_facts", 0) == 0 and available_count > 0:
            result["health"] = "degraded"
        else:
            result["health"] = "ok"

        return result
