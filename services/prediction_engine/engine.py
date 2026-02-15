"""
Life OS — Prediction Engine

Forward-looking intelligence. Continuously evaluates the current state
against learned patterns to predict what the user will need before they
know they need it.

This is what creates the "blown away" moments — the AI doing something
helpful that the user never asked for, at exactly the right time.

Prediction Types:
    NEED        — "You'll probably need X soon"
    CONFLICT    — "These two things overlap / contradict"
    OPPORTUNITY — "Good time to do X based on your patterns"
    RISK        — "Something might go wrong if you don't..."
    REMINDER    — "You haven't done X and it's been a while"

Confidence Gates:
    < 0.3  OBSERVE    — Watch silently, keep learning
    0.3-0.6 SUGGEST   — "Would you like me to..."
    0.6-0.8 DEFAULT   — Do it, but make it easy to undo
    > 0.8  AUTONOMOUS — Just handle it
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from models.core import ConfidenceGate, Priority
from models.user_model import MoodState, Prediction, ReactionPrediction
from storage.database import DatabaseManager, UserModelStore


class PredictionEngine:
    """
    Generates predictions about user needs by combining:
    - Current context (time, location, calendar, mood)
    - Signal profiles (behavioral patterns)
    - Semantic memory (known facts & preferences)
    - Episodic memory (past similar situations)
    """

    def __init__(self, db: DatabaseManager, ums: UserModelStore):
        self.db = db
        self.ums = ums

    async def generate_predictions(self, current_context: dict) -> list[Prediction]:
        """
        Main prediction loop. Called periodically (every 15 min)
        and on significant context changes (location, calendar event start, etc.)
        """
        predictions = []

        predictions.extend(await self._check_calendar_conflicts(current_context))
        predictions.extend(await self._check_follow_up_needs(current_context))
        predictions.extend(await self._check_routine_deviations(current_context))
        predictions.extend(await self._check_relationship_maintenance(current_context))
        predictions.extend(await self._check_preparation_needs(current_context))
        predictions.extend(await self._check_spending_patterns(current_context))

        # Filter through the reaction predictor — will the user find this helpful
        # or annoying right now?
        filtered = []
        for pred in predictions:
            reaction = await self.predict_reaction(pred, current_context)
            if reaction.predicted_reaction in ("helpful", "neutral"):
                filtered.append(pred)

        # Store all predictions for accuracy tracking
        for pred in predictions:
            self.ums.store_prediction(pred.model_dump())

        return filtered

    # -------------------------------------------------------------------
    # Prediction Generators
    # -------------------------------------------------------------------

    async def _check_calendar_conflicts(self, ctx: dict) -> list[Prediction]:
        """Detect scheduling conflicts and tight transitions."""
        predictions = []

        with self.db.get_connection("events") as conn:
            # Get upcoming calendar events in the next 48 hours
            now = datetime.now(timezone.utc)
            tomorrow = now + timedelta(hours=48)

            events = conn.execute(
                """SELECT * FROM events 
                   WHERE type = 'calendar.event.created' 
                   AND timestamp > ? AND timestamp < ?
                   ORDER BY timestamp""",
                (now.isoformat(), tomorrow.isoformat()),
            ).fetchall()

        if len(events) < 2:
            return predictions

        # Check for overlaps and tight transitions
        for i in range(len(events) - 1):
            curr_payload = json.loads(events[i]["payload"])
            next_payload = json.loads(events[i + 1]["payload"])

            curr_end = curr_payload.get("end_time", "")
            next_start = next_payload.get("start_time", "")

            if curr_end and next_start:
                try:
                    end_dt = datetime.fromisoformat(curr_end.replace("Z", "+00:00"))
                    start_dt = datetime.fromisoformat(next_start.replace("Z", "+00:00"))
                    gap_minutes = (start_dt - end_dt).total_seconds() / 60

                    if gap_minutes < 0:
                        # Overlap!
                        predictions.append(Prediction(
                            prediction_type="conflict",
                            description=(
                                f"Calendar overlap: '{curr_payload.get('title', 'Event')}' "
                                f"and '{next_payload.get('title', 'Event')}' overlap by "
                                f"{abs(int(gap_minutes))} minutes"
                            ),
                            confidence=0.95,
                            confidence_gate=ConfidenceGate.DEFAULT,
                            time_horizon="24_hours",
                            suggested_action="Reschedule one of the conflicting events",
                        ))
                    elif gap_minutes < 15:
                        # Very tight transition
                        predictions.append(Prediction(
                            prediction_type="risk",
                            description=(
                                f"Only {int(gap_minutes)} minutes between "
                                f"'{curr_payload.get('title', 'Event')}' and "
                                f"'{next_payload.get('title', 'Event')}'"
                            ),
                            confidence=0.7,
                            confidence_gate=ConfidenceGate.SUGGEST,
                            time_horizon="24_hours",
                            suggested_action="Consider adding buffer time",
                        ))
                except (ValueError, TypeError):
                    pass

        return predictions

    async def _check_follow_up_needs(self, ctx: dict) -> list[Prediction]:
        """
        Detect messages that need a reply — things the user read but
        hasn't responded to, especially from priority contacts.
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Inbound messages from the last 48 hours
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

            inbound = conn.execute(
                """SELECT id, payload, metadata, timestamp FROM events
                   WHERE type IN ('email.received', 'message.received')
                   AND timestamp > ?
                   ORDER BY timestamp DESC""",
                (cutoff,),
            ).fetchall()

            # Outbound messages in the same window
            outbound = conn.execute(
                """SELECT payload FROM events
                   WHERE type IN ('email.sent', 'message.sent')
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        # Build set of addresses we've replied to
        replied_to_threads = set()
        for msg in outbound:
            payload = json.loads(msg["payload"])
            if payload.get("in_reply_to"):
                replied_to_threads.add(payload["in_reply_to"])

        # Find unreplied inbound messages
        for msg in inbound:
            payload = json.loads(msg["payload"])
            metadata = json.loads(msg["metadata"])
            message_id = payload.get("message_id", "")

            # Skip if already replied
            if message_id in replied_to_threads:
                continue

            # Skip marketing/automated
            if payload.get("snippet", "").lower().count("unsubscribe") > 0:
                continue

            # Check if from a priority contact
            from_addr = payload.get("from_address", "")
            is_priority = any(
                from_addr in contacts
                for contacts in [metadata.get("related_contacts", [])]
            )

            # Calculate how long it's been
            try:
                msg_time = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                hours_ago = (datetime.now(timezone.utc) - msg_time).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_ago = 24

            # Only flag if it's been more than a few hours
            if hours_ago < 3:
                continue

            confidence = 0.4
            if is_priority:
                confidence = 0.7
            if hours_ago > 24:
                confidence = min(confidence + 0.2, 0.9)
            if payload.get("requires_response"):
                confidence = min(confidence + 0.2, 0.9)

            subject = payload.get("subject", "No subject")
            predictions.append(Prediction(
                prediction_type="reminder",
                description=(
                    f"Unreplied message from {from_addr}: \"{subject}\" "
                    f"({int(hours_ago)} hours ago)"
                ),
                confidence=confidence,
                confidence_gate=self._gate_from_confidence(confidence),
                time_horizon="2_hours",
                suggested_action=f"Reply to {from_addr}",
                relevant_contacts=[from_addr],
            ))

        return predictions

    async def _check_routine_deviations(self, ctx: dict) -> list[Prediction]:
        """
        Detect when the user deviates from their usual patterns.
        E.g., usually exercises Mon/Wed/Fri but hasn't today.
        """
        predictions = []

        # Check procedural memory for routines
        with self.db.get_connection("user_model") as conn:
            routines = conn.execute(
                "SELECT * FROM routines WHERE consistency_score > 0.6"
            ).fetchall()

        if not routines:
            return predictions

        now = datetime.now(timezone.utc)
        day_name = now.strftime("%A").lower()

        for routine in routines:
            trigger = routine["trigger_condition"]
            # Simple day-based check
            if day_name in trigger.lower():
                # Check if the routine has been executed today
                # (Look for related events in the last 12 hours)
                steps = json.loads(routine["steps"])
                if steps:
                    first_action = steps[0].get("action", "") if isinstance(steps[0], dict) else str(steps[0])

                    # This is simplified — in production you'd check against actual events
                    predictions.append(Prediction(
                        prediction_type="reminder",
                        description=f"You usually {routine['name']} on {day_name.title()}s",
                        confidence=routine["consistency_score"] * 0.6,
                        confidence_gate=ConfidenceGate.SUGGEST,
                        time_horizon="24_hours",
                    ))

        return predictions

    async def _check_relationship_maintenance(self, ctx: dict) -> list[Prediction]:
        """
        Detect contacts the user hasn't been in touch with for longer
        than their typical frequency.
        """
        predictions = []

        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return predictions

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            last = data.get("last_interaction")
            count = data.get("interaction_count", 0)

            if not last or count < 5:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            # Estimate their typical contact frequency from timestamps
            timestamps = data.get("interaction_timestamps", [])
            if len(timestamps) >= 3:
                try:
                    dts = sorted([
                        datetime.fromisoformat(t.replace("Z", "+00:00"))
                        for t in timestamps[-10:]
                    ])
                    gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
                    avg_gap = sum(gaps) / len(gaps) if gaps else 30
                except (ValueError, TypeError):
                    avg_gap = 30

                if days_since > avg_gap * 1.5 and days_since > 7:
                    confidence = min(0.6, 0.3 + (days_since / avg_gap - 1.5) * 0.2)
                    predictions.append(Prediction(
                        prediction_type="opportunity",
                        description=(
                            f"It's been {days_since} days since you last "
                            f"contacted {addr} (you usually connect every ~{int(avg_gap)} days)"
                        ),
                        confidence=confidence,
                        confidence_gate=self._gate_from_confidence(confidence),
                        time_horizon="this_week",
                        suggested_action=f"Reach out to {addr}",
                        relevant_contacts=[addr],
                    ))

        return predictions

    async def _check_preparation_needs(self, ctx: dict) -> list[Prediction]:
        """
        Detect upcoming events that require preparation based on
        learned patterns. E.g., "You have a flight tomorrow — pack tonight."
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Events in the next 24-48 hours
            now = datetime.now(timezone.utc)
            window_start = now + timedelta(hours=12)
            window_end = now + timedelta(hours=48)

            upcoming = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp BETWEEN ? AND ?""",
                (window_start.isoformat(), window_end.isoformat()),
            ).fetchall()

        for event in upcoming:
            payload = json.loads(event["payload"])
            title = payload.get("title", "").lower()
            location = payload.get("location", "")

            # Travel detection
            travel_keywords = ["flight", "airport", "hotel", "travel", "trip"]
            if any(kw in title for kw in travel_keywords):
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Upcoming travel: '{payload.get('title')}'. Time to prepare.",
                    confidence=0.75,
                    confidence_gate=ConfidenceGate.DEFAULT,
                    time_horizon="24_hours",
                    suggested_action="Check packing list and confirm reservations",
                ))

            # Meeting with external people
            attendees = payload.get("attendees", [])
            if len(attendees) > 3:
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Large meeting: '{payload.get('title')}' with {len(attendees)} attendees",
                    confidence=0.5,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="24_hours",
                    suggested_action="Review agenda and prepare talking points",
                ))

        return predictions

    async def _check_spending_patterns(self, ctx: dict) -> list[Prediction]:
        """Detect spending anomalies and subscription waste."""
        predictions = []

        with self.db.get_connection("events") as conn:
            # Transactions in the last 30 days
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            transactions = conn.execute(
                """SELECT payload FROM events
                   WHERE type = 'finance.transaction.new'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if len(transactions) < 5:
            return predictions

        # Calculate spending by category
        by_category: dict[str, float] = {}
        for txn in transactions:
            payload = json.loads(txn["payload"])
            cat = payload.get("category", "uncategorized")
            amount = abs(payload.get("amount", 0))
            by_category[cat] = by_category.get(cat, 0) + amount

        total = sum(by_category.values())
        if total == 0:
            return predictions

        # Flag categories that are >25% of total spending
        for cat, amount in by_category.items():
            pct = amount / total
            if pct > 0.25 and amount > 200:
                predictions.append(Prediction(
                    prediction_type="risk",
                    description=(
                        f"Spending alert: ${amount:.0f} on '{cat}' this month "
                        f"({pct * 100:.0f}% of total)"
                    ),
                    confidence=0.5,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="this_week",
                    suggested_action=f"Review {cat} spending",
                ))

        return predictions

    # -------------------------------------------------------------------
    # Reaction Prediction — Should we surface this?
    # -------------------------------------------------------------------

    async def predict_reaction(self, prediction: Prediction,
                                context: dict) -> ReactionPrediction:
        """
        Before surfacing a prediction, estimate whether the user will
        find it helpful, annoying, or intrusive right now.
        """
        # Get current mood
        mood_profile = self.ums.get_signal_profile("mood_signals")
        mood_data = mood_profile["data"] if mood_profile else {}

        # Count recent dismissals (from feedback log)
        with self.db.get_connection("preferences") as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM feedback_log
                   WHERE feedback_type = 'dismissed' AND timestamp > ?""",
                (cutoff,),
            ).fetchone()
            recent_dismissals = row["cnt"] if row else 0

        # Scoring logic
        score = 0.5  # Start neutral

        # High stress = lower threshold for surfacing
        stress_signals = mood_data.get("recent_signals", [])
        stress_count = sum(1 for s in stress_signals[-5:]
                          if s.get("signal_type") in ["negative_language", "calendar_density"])
        if stress_count > 2:
            score -= 0.2  # Don't pile on when they're stressed

        # Many recent dismissals = they don't want interruptions
        if recent_dismissals > 3:
            score -= 0.3

        # High confidence predictions are more likely helpful
        if prediction.confidence > 0.7:
            score += 0.2

        # Conflicts and risks are more urgent than opportunities
        if prediction.prediction_type in ("conflict", "risk"):
            score += 0.2
        elif prediction.prediction_type == "opportunity":
            score -= 0.1

        # Time of day matters
        hour = datetime.now(timezone.utc).hour
        # Don't surface low-priority stuff in early morning or late night
        if hour < 7 or hour > 22:
            if prediction.prediction_type not in ("conflict", "risk"):
                score -= 0.3

        predicted = "helpful" if score > 0.4 else ("neutral" if score > 0.1 else "annoying")

        return ReactionPrediction(
            proposed_action=prediction.description,
            predicted_reaction=predicted,
            confidence=min(1.0, abs(score)),
            reasoning=f"score={score:.2f}, dismissals={recent_dismissals}, stress_signals={stress_count}",
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _gate_from_confidence(confidence: float) -> ConfidenceGate:
        if confidence < 0.3:
            return ConfidenceGate.OBSERVE
        elif confidence < 0.6:
            return ConfidenceGate.SUGGEST
        elif confidence < 0.8:
            return ConfidenceGate.DEFAULT
        else:
            return ConfidenceGate.AUTONOMOUS
