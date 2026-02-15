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
from datetime import datetime, timezone
from typing import Any, Optional

from models.core import FeedbackType, Priority
from storage.database import DatabaseManager, UserModelStore


class FeedbackCollector:
    """
    Collects and processes feedback signals to improve the user model.
    
    Subscribes to:
        - notification.acted_on / notification.dismissed
        - system.ai.action_taken (with user response)
        - Draft edits (diff between AI draft and what user actually sent)
        - Explicit feedback ("that was helpful" / "don't do that")
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        self.db = db
        self.ums = ums

    async def process_notification_response(self, notification_id: str,
                                             response_type: str,
                                             response_time_seconds: float,
                                             context: Optional[dict] = None):
        """
        Process how the user responded to a notification.
        
        response_type: "acted_on", "dismissed", "ignored", "delayed"
        """
        # Retrieve the original notification
        with self.db.get_connection("state") as conn:
            notif = conn.execute(
                "SELECT * FROM notifications WHERE id = ?",
                (notification_id,),
            ).fetchone()

        if not notif:
            return

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

        self._store_feedback(feedback)

        # Update models based on feedback
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
        if original_draft == final_message:
            # User accepted the draft as-is — strong positive signal
            self._store_feedback({
                "action_id": f"draft-{datetime.now(timezone.utc).isoformat()}",
                "action_type": "draft",
                "feedback_type": FeedbackType.ENGAGED.value,
                "response_latency_seconds": 0,
                "context": {"contact_id": contact_id, "channel": channel, "accepted_as_is": True},
            })
            return

        # Analyze the diff
        original_words = set(original_draft.lower().split())
        final_words = set(final_message.lower().split())

        added_words = final_words - original_words
        removed_words = original_words - final_words

        # Length change
        length_change = len(final_message) - len(original_draft)
        length_change_pct = length_change / max(len(original_draft), 1)

        # Formality shift detection
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

        self._store_feedback(feedback)

        # Update the communication template for this contact
        if contact_id or channel:
            self._update_template_from_edit(
                contact_id, channel, original_draft, final_message, feedback["context"]
            )

    async def process_suggestion_response(self, suggestion_id: str,
                                           accepted: bool,
                                           user_alternative: Optional[str] = None):
        """Process whether the user accepted or rejected a proactive suggestion."""
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

        self._store_feedback(feedback)

        # Update prediction accuracy
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

    async def process_explicit_feedback(self, message: str):
        """
        Process explicit verbal feedback like "that was helpful" 
        or "don't do that again" or "I prefer X over Y".
        """
        # Classify the feedback
        feedback_type = self._classify_explicit_feedback(message)

        feedback = {
            "action_id": f"explicit-{datetime.now(timezone.utc).isoformat()}",
            "action_type": "explicit",
            "feedback_type": feedback_type,
            "response_latency_seconds": 0,
            "context": {"raw_message": message},
            "notes": message,
        }

        self._store_feedback(feedback)

        # If it contains a preference, store as semantic fact
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
        """User dismissed a notification — reduce confidence in similar actions."""
        domain = notification.get("domain")
        priority = notification.get("priority")

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
        """User engaged with a notification — reinforce similar actions."""
        domain = notification.get("domain")
        priority = notification.get("priority")

        # Fast engagement = high relevance
        if response_time < 30:
            self.ums.update_semantic_fact(
                key=f"notification_relevant_{domain}",
                category="notification_preference",
                value=f"User quickly acts on {priority} notifications about {domain}",
                confidence=0.6,
            )

    def _learn_from_ignore(self, notification: dict):
        """User ignored a notification completely — strongest negative signal."""
        domain = notification.get("domain")
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
        """Update communication template based on how user edited a draft."""
        # This is where the AI learns to write more like the user
        # over time, the edits get smaller as the template improves
        formality_shift = edit_context.get("formality_shift", "neutral")

        if formality_shift != "neutral":
            # Adjust the template's formality based on the edit direction
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
                    if formality_shift == "more_informal":
                        new_formality = max(0.0, current_formality - 0.05)
                    else:
                        new_formality = min(1.0, current_formality + 0.05)

                    conn.execute(
                        "UPDATE communication_templates SET formality = ? WHERE id = ?",
                        (new_formality, template["id"]),
                    )

    def _classify_explicit_feedback(self, message: str) -> str:
        """Simple classification of explicit feedback."""
        positive_words = ["good", "great", "helpful", "perfect", "thanks", "love", "nice", "awesome"]
        negative_words = ["bad", "wrong", "stop", "don't", "annoying", "hate", "terrible", "useless"]

        msg_lower = message.lower()
        pos_count = sum(1 for w in positive_words if w in msg_lower)
        neg_count = sum(1 for w in negative_words if w in msg_lower)

        if pos_count > neg_count:
            return FeedbackType.EXPLICIT_POSITIVE.value
        elif neg_count > pos_count:
            return FeedbackType.EXPLICIT_NEGATIVE.value
        return FeedbackType.ENGAGED.value

    # -------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------

    def _store_feedback(self, feedback: dict):
        """Store a feedback entry in the database."""
        import uuid
        with self.db.get_connection("preferences") as conn:
            conn.execute(
                """INSERT INTO feedback_log 
                   (id, timestamp, action_id, action_type, feedback_type,
                    response_latency_seconds, context, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    datetime.now(timezone.utc).isoformat(),
                    feedback["action_id"],
                    feedback["action_type"],
                    feedback["feedback_type"],
                    feedback.get("response_latency_seconds"),
                    json.dumps(feedback.get("context", {})),
                    feedback.get("notes"),
                ),
            )

    def get_feedback_summary(self) -> dict:
        """Get a summary of feedback patterns for model evaluation."""
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
