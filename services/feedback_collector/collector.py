"""
Life OS — Feedback Collector

The learning loop. Watches how the user responds to every AI action
and uses that to improve the user model over time.

Most feedback is IMPLICIT — the user doesn't rate anything. We observe:
    - Did they engage with the notification?
    - How quickly did they act?
    - Did they edit the AI's draft?
    - Did they dismiss the suggestion?
    - Did they override the AI's decision?

This implicit signal is more honest than explicit ratings because
people behave differently than they self-report.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from models.core import FeedbackType, Priority
from storage.database import DatabaseManager, UserModelStore

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects and processes feedback signals to improve the user model.

    Subscribes to:
        - notification.acted_on / notification.dismissed
        - system.ai.action_taken (with user response)
        - Draft edits (diff between AI draft and what user actually sent)
        - Explicit feedback ("that was helpful" / "don't do that")
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore, event_bus: Any = None):
        self.db = db   # Database access for feedback_log and notification tables
        self.ums = ums  # User-model store for updating semantic facts from feedback
        self.bus = event_bus

    async def _publish_telemetry(self, event_type: str, payload: dict):
        """Publish a telemetry event if the event bus is available."""
        if self.bus and self.bus.is_connected:
            await self.bus.publish(event_type, payload, source="feedback_collector")

    async def process_notification_response(self, notification_id: str,
                                             response_type: str,
                                             response_time_seconds: float,
                                             context: Optional[dict] = None):
        """
        Process how the user responded to a notification.

        response_type: "acted_on", "dismissed", "ignored", "delayed"
        """
        logger.info("feedback: notification %s response_type=%s", notification_id, response_type)
        # Retrieve the original notification so we can correlate feedback
        # with the notification's domain, priority, and other metadata.
        with self.db.get_connection("state") as conn:
            notif = conn.execute(
                "SELECT * FROM notifications WHERE id = ?",
                (notification_id,),
            ).fetchone()

        if not notif:
            return

        # Build a feedback record enriched with notification metadata.
        # hour_of_day is captured so the system can learn time-based patterns
        # (e.g., user always dismisses low-priority alerts before 9 AM).
        feedback = {
            "action_id": notification_id,
            "action_type": "notification",
            "feedback_type": response_type,
            "response_latency_seconds": response_time_seconds,
            "context": {
                "priority": notif["priority"],
                "domain": notif["domain"],
                "hour_of_day": datetime.now(timezone.utc).hour,
                **(context or {}),
            },
        }

        await self._store_feedback(feedback)

        # --- Implicit feedback inference ---
        # The three response types map to distinct learning signals:
        #   DISMISSED  -> user actively swiped away / closed the notification
        #   ENGAGED    -> user tapped/opened and took action on it
        #   IGNORED    -> notification was never interacted with at all
        # Each path updates the user model differently (see _learn_from_* methods).
        if response_type == FeedbackType.DISMISSED.value:
            self._learn_from_dismissal(notif, response_time_seconds)
        elif response_type == FeedbackType.ENGAGED.value:
            self._learn_from_engagement(notif, response_time_seconds)
        elif response_type == FeedbackType.IGNORED.value:
            self._learn_from_ignore(notif)

    async def process_draft_edit(self, original_draft: str, final_message: str,
                                  contact_id: Optional[str] = None,
                                  channel: Optional[str] = None):
        """
        Process the diff between an AI-generated draft and what the user
        actually sent. Every edit is a learning signal.
        """
        accepted_as_is = original_draft == final_message
        logger.info(
            "feedback: draft edit contact=%s accepted_as_is=%s",
            contact_id,
            accepted_as_is,
        )
        if original_draft == final_message:
            # User accepted the draft as-is — strong positive signal.
            # This means the AI's tone, length, and content were all on target.
            await self._store_feedback({
                "action_id": f"draft-{datetime.now(timezone.utc).isoformat()}",
                "action_type": "draft",
                "feedback_type": FeedbackType.ENGAGED.value,
                "response_latency_seconds": 0,
                "context": {"contact_id": contact_id, "channel": channel, "accepted_as_is": True},
            })
            return

        # --- Diff analysis: learn HOW the user edited the AI's draft ---
        # We compare word-level sets (bag-of-words) rather than exact diffs
        # because we care about semantic shifts, not character positions.
        original_words = set(original_draft.lower().split())
        final_words = set(final_message.lower().split())

        added_words = final_words - original_words   # Words the user injected
        removed_words = original_words - final_words  # Words the user deleted

        # Length change as a percentage tells us if the user tends to make
        # the AI's drafts longer (more detail) or shorter (more concise).
        length_change = len(final_message) - len(original_draft)
        length_change_pct = length_change / max(len(original_draft), 1)

        # Formality shift detection: check whether the user made the draft
        # more casual or more formal. Over time, this nudges the AI's
        # default tone for this contact/channel closer to the user's style.
        informal_added = sum(1 for w in added_words if w in [
            "hey", "yeah", "lol", "haha", "gonna", "wanna",
        ])
        formal_added = sum(1 for w in added_words if w in [
            "regarding", "sincerely", "furthermore", "please",
        ])

        feedback = {
            "action_id": f"draft-{datetime.now(timezone.utc).isoformat()}",
            "action_type": "draft",
            "feedback_type": FeedbackType.OVERRIDDEN.value,
            "response_latency_seconds": 0,
            "context": {
                "contact_id": contact_id,
                "channel": channel,
                "length_change_pct": round(length_change_pct, 2),
                "words_added": len(added_words),
                "words_removed": len(removed_words),
                "formality_shift": "more_informal" if informal_added > formal_added else (
                    "more_formal" if formal_added > informal_added else "neutral"
                ),
            },
        }

        await self._store_feedback(feedback)

        # If we know the contact or channel, feed the edit signal back into
        # the communication template so future drafts are closer to the
        # user's preferred style. This is the core of the feedback loop:
        #   AI drafts -> user edits -> template adjusts -> better drafts.
        if contact_id or channel:
            self._update_template_from_edit(
                contact_id, channel, original_draft, final_message, feedback["context"]
            )

    async def process_suggestion_response(self, suggestion_id: str,
                                           accepted: bool,
                                           user_alternative: Optional[str] = None):
        """
        Process whether the user accepted or rejected a proactive suggestion.

        Explicit feedback path: the user either taps "yes" (accepted=True) or
        provides their own alternative. Both signals are stored so the
        prediction engine can measure its accuracy over time.
        """
        logger.info("feedback: suggestion %s accepted=%s", suggestion_id, accepted)
        feedback = {
            "action_id": suggestion_id,
            "action_type": "suggestion",
            "feedback_type": FeedbackType.ENGAGED.value if accepted else FeedbackType.OVERRIDDEN.value,
            "response_latency_seconds": 0,
            "context": {
                "accepted": accepted,
                "user_alternative": user_alternative,
            },
        }

        await self._store_feedback(feedback)

        # Close the loop: mark the prediction record with the user's actual
        # response so the prediction engine can compute its hit rate and
        # recalibrate confidence thresholds over time.
        try:
            with self.db.get_connection("user_model") as conn:
                conn.execute(
                    """UPDATE predictions SET
                       user_response = ?, was_accurate = ?, resolved_at = ?
                       WHERE id = ?""",
                    (
                        "accepted" if accepted else "rejected",
                        1 if accepted else 0,
                        datetime.now(timezone.utc).isoformat(),
                        suggestion_id,
                    ),
                )
        except Exception:
            logger.warning(
                "feedback: could not update prediction %s in user_model.db (non-fatal)",
                suggestion_id,
            )

    async def process_explicit_feedback(self, message: str):
        """
        Process explicit verbal feedback like "that was helpful"
        or "don't do that again" or "I prefer X over Y".
        """
        # Classify the raw message into positive / negative / neutral using
        # simple keyword matching (see _classify_explicit_feedback below).
        feedback_type = self._classify_explicit_feedback(message)

        feedback = {
            "action_id": f"explicit-{datetime.now(timezone.utc).isoformat()}",
            "action_type": "explicit",
            "feedback_type": feedback_type,
            "response_latency_seconds": 0,
            "context": {"raw_message": message},
            "notes": message,
        }

        await self._store_feedback(feedback)

        # If the message contains preference-indicating language ("prefer",
        # "like", "don't"), promote it to a semantic fact in the user model.
        # Semantic facts persist long-term and influence future AI decisions.
        # Confidence is set to 0.95 because the user stated it explicitly —
        # this is the strongest signal we can get.
        if "prefer" in message.lower() or "like" in message.lower() or "don't" in message.lower():
            self.ums.update_semantic_fact(
                key=f"user_stated_{datetime.now(timezone.utc).isoformat()[:10]}",
                category="explicit_preference",
                value=message,
                confidence=0.95,  # High confidence — user said it directly
            )

    # -------------------------------------------------------------------
    # Learning methods
    # -------------------------------------------------------------------

    def _learn_from_dismissal(self, notification: dict, response_time: float):
        """
        User dismissed a notification — reduce confidence in similar actions.

        Response time is a proxy for how much attention the user gave:
            < 2 sec  -> "irrelevant, didn't even read it"  (strong negative)
            > 10 sec -> "I read it but chose not to act"    (mild negative)
        We store this as a semantic fact so future notification decisions
        can check whether the user tends to dismiss this domain/priority.
        """
        domain = notification["domain"]
        priority = notification["priority"]

        # Quick dismissal (<2 sec) = "this was irrelevant"
        # Slow dismissal (>10 sec) = "I read it but don't need to act"
        if response_time < 2:
            self.ums.update_semantic_fact(
                key=f"notification_irrelevant_{domain}",
                category="notification_preference",
                value=f"User quickly dismisses {priority} notifications about {domain}",
                confidence=0.4,
            )

    def _learn_from_engagement(self, notification: dict, response_time: float):
        """
        User engaged with a notification — reinforce similar actions.

        Fast engagement (< 30 sec) is especially valuable: it means the
        notification was timely and relevant enough to act on immediately.
        """
        domain = notification["domain"]
        priority = notification["priority"]

        # Fast engagement means high relevance — store as positive signal
        if response_time < 30:
            self.ums.update_semantic_fact(
                key=f"notification_relevant_{domain}",
                category="notification_preference",
                value=f"User quickly acts on {priority} notifications about {domain}",
                confidence=0.6,
            )

    def _learn_from_ignore(self, notification: dict):
        """
        User ignored a notification completely — strongest negative signal.

        "Ignored" means the notification was delivered but never interacted
        with at all. This is more informative than a dismissal because the
        user didn't even bother to swipe it away.
        """
        domain = notification["domain"]
        self.ums.update_semantic_fact(
            key=f"notification_unwanted_{domain}",
            category="notification_preference",
            value=f"User ignores notifications about {domain}",
            confidence=0.5,
        )

    def _update_template_from_edit(self, contact_id: Optional[str],
                                   channel: Optional[str],
                                   original: str, final: str,
                                   edit_context: dict):
        """
        Update communication template based on how user edited a draft.

        This is the feedback loop that improves AI behavior over time:
        each edit nudges the template's formality score by +/- 0.05 toward
        the user's actual style. Over many iterations the edits should
        shrink as the AI learns the user's voice for each contact/channel.
        """
        formality_shift = edit_context.get("formality_shift", "neutral")

        if formality_shift != "neutral":
            # Look up the matching communication template for this
            # contact/channel pair, then nudge its formality score.
            try:
                with self.db.get_connection("user_model") as conn:
                    template_query = "SELECT * FROM communication_templates WHERE 1=1"
                    params = []
                    if contact_id:
                        template_query += " AND contact_id = ?"
                        params.append(contact_id)
                    if channel:
                        template_query += " AND channel = ?"
                        params.append(channel)

                    template = conn.execute(template_query, params).fetchone()
                    if template:
                        current_formality = template["formality"]
                        # Nudge formality down (toward 0.0 = casual) or up
                        # (toward 1.0 = formal) by a small step. The clamp
                        # ensures we stay within [0.0, 1.0].
                        if formality_shift == "more_informal":
                            new_formality = max(0.0, current_formality - 0.05)
                        else:
                            new_formality = min(1.0, current_formality + 0.05)

                        conn.execute(
                            "UPDATE communication_templates SET formality = ? WHERE id = ?",
                            (new_formality, template["id"]),
                        )
            except Exception:
                logger.warning(
                    "feedback: could not update communication template for %s/%s (non-fatal)",
                    contact_id,
                    channel,
                )

    def _classify_explicit_feedback(self, message: str) -> str:
        """
        Simple keyword-based classification of explicit feedback.

        Uses bag-of-words sentiment: whichever list (positive vs. negative)
        has more keyword hits wins. Falls back to ENGAGED (neutral) on a tie.
        In production this would be replaced by an LLM classifier, but the
        keyword approach is fast, deterministic, and avoids API latency.
        """
        positive_words = ["good", "great", "helpful", "perfect", "thanks", "love", "nice", "awesome"]
        negative_words = ["bad", "wrong", "stop", "don't", "annoying", "hate", "terrible", "useless"]

        msg_lower = message.lower()
        pos_count = sum(1 for w in positive_words if w in msg_lower)
        neg_count = sum(1 for w in negative_words if w in msg_lower)

        if pos_count > neg_count:
            return FeedbackType.EXPLICIT_POSITIVE.value
        elif neg_count > pos_count:
            return FeedbackType.EXPLICIT_NEGATIVE.value
        # Tie or no matches — treat as neutral engagement
        return FeedbackType.ENGAGED.value

    # -------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------

    async def _store_feedback(self, feedback: dict):
        """
        Persist a feedback entry to the feedback_log table and publish telemetry.

        Every feedback record — implicit or explicit — lands here. The log
        is the single source of truth for evaluating how well the AI's
        actions match user expectations. Analytics queries aggregate this
        table to surface trends (e.g., rising dismissal rate for a domain).
        """
        feedback_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT INTO feedback_log
                   (id, timestamp, action_id, action_type, feedback_type,
                    response_latency_seconds, context, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    now,
                    feedback["action_id"],
                    feedback["action_type"],
                    feedback["feedback_type"],
                    feedback.get("response_latency_seconds"),
                    json.dumps(feedback.get("context", {})),
                    feedback.get("notes"),
                ),
            )

        logger.info("feedback: stored %s action=%s type=%s", feedback_id, feedback["action_id"], feedback["feedback_type"])

        await self._publish_telemetry("system.feedback.recorded", {
            "feedback_id": feedback_id,
            "action_id": feedback["action_id"],
            "action_type": feedback["action_type"],
            "feedback_type": feedback["feedback_type"],
            "response_latency_seconds": feedback.get("response_latency_seconds"),
            "recorded_at": now,
        })

    def get_feedback_summary(self) -> dict:
        """
        Get a summary of feedback patterns for model evaluation.

        Returns a dict keyed by "action_type:feedback_type" (e.g.,
        "notification:dismissed") with counts. This powers the admin
        dashboard and is also consumed by the prediction engine to
        recalibrate confidence thresholds.
        """
        with self.db.get_connection("preferences") as conn:
            rows = conn.execute(
                """SELECT feedback_type, action_type, COUNT(*) as cnt
                   FROM feedback_log
                   GROUP BY feedback_type, action_type
                   ORDER BY cnt DESC"""
            ).fetchall()

            summary = {}
            for row in rows:
                key = f"{row['action_type']}:{row['feedback_type']}"
                summary[key] = row["cnt"]

            return summary
