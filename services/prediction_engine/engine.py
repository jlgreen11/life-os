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
        self.db = db   # Database access for events, user_model, and preferences tables
        self.ums = ums  # User-model store for signal profiles and semantic memory
        self._last_event_cursor: int = 0  # rowid of last processed event

    def _has_new_events(self) -> bool:
        """Check if any new events have arrived since last prediction run."""
        with self.db.get_connection("events") as conn:
            row = conn.execute(
                "SELECT MAX(rowid) as max_id FROM events"
            ).fetchone()
            current_max = row["max_id"] if row and row["max_id"] else 0

        if current_max <= self._last_event_cursor:
            return False

        self._last_event_cursor = current_max
        return True

    async def generate_predictions(self, current_context: dict) -> list[Prediction]:
        """
        Main prediction loop. Called periodically (every 15 min)
        and on significant context changes (location, calendar event start, etc.)
        """
        # Skip if no new events since last run
        if not self._has_new_events():
            return []

        predictions = []

        # --- Prediction generation pipeline ---
        # Each _check_* method is a specialized detector that looks for one
        # category of predictable user need. They run independently and
        # return zero or more Prediction objects. The full set of prediction
        # types covers the most common "blown away" moments:
        predictions.extend(await self._check_calendar_conflicts(current_context))    # Scheduling overlaps & tight transitions
        predictions.extend(await self._check_follow_up_needs(current_context))       # Unreplied messages from important people
        predictions.extend(await self._check_routine_deviations(current_context))    # Missed habits or routines
        predictions.extend(await self._check_relationship_maintenance(current_context))  # Contacts going cold
        predictions.extend(await self._check_preparation_needs(current_context))     # Upcoming events needing prep
        predictions.extend(await self._check_spending_patterns(current_context))     # Spending anomalies

        # --- Accuracy-based confidence decay/boost ---
        # Adjust confidence based on historical accuracy for each prediction type.
        # This closes the feedback loop: predictions that keep getting dismissed
        # have their confidence reduced, eventually suppressing them entirely.
        for pred in predictions:
            multiplier = self._get_accuracy_multiplier(pred.prediction_type)
            pred.confidence *= multiplier
            pred.confidence_gate = self._gate_from_confidence(pred.confidence)

        # --- Reaction prediction gatekeeper ---
        # Before surfacing any prediction, ask: "Will the user find this
        # helpful or annoying right now?" This prevents piling on during
        # stressful moments or when the user has been dismissing alerts.
        filtered = []
        for pred in predictions:
            reaction = await self.predict_reaction(pred, current_context)
            if reaction.predicted_reaction in ("helpful", "neutral"):
                filtered.append(pred)

        # Confidence floor — don't surface anything below SUGGEST threshold (0.3).
        # This enables relationship maintenance, preparation, and other valuable
        # predictions that should be shown as suggestions ("Would you like...?")
        # even if confidence isn't high enough for autonomous action.
        filtered = [p for p in filtered if p.confidence >= 0.3]

        # Cap at 5 surfaced predictions per cycle, prioritized by confidence
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        filtered = filtered[:5]

        # Store ALL predictions (including filtered-out ones) for accuracy
        # tracking. Mark which ones were actually surfaced so the feedback
        # loop can distinguish them via was_surfaced=1 in queries.
        #
        # CRITICAL: Predictions that don't pass reaction prediction or
        # confidence gates are immediately resolved as 'filtered'. This
        # prevents database bloat from hundreds of thousands of unsurfaced
        # predictions that will never be shown to the user.
        surfaced_ids = {p.id for p in filtered}
        now = datetime.now(timezone.utc).isoformat()

        for pred in predictions:
            pred.was_surfaced = pred.id in surfaced_ids

            # If this prediction was filtered out, mark it as resolved
            # immediately with user_response='filtered'. This closes the
            # lifecycle for predictions that never surface, preventing them
            # from accumulating in the database indefinitely.
            if not pred.was_surfaced:
                pred.resolved_at = now
                pred.user_response = 'filtered'

            self.ums.store_prediction(pred.model_dump())

        return filtered

    # -------------------------------------------------------------------
    # Prediction Generators
    # -------------------------------------------------------------------

    async def _check_calendar_conflicts(self, ctx: dict) -> list[Prediction]:
        """
        Detect scheduling conflicts and tight transitions.

        Scans a 48-hour lookahead window and compares consecutive events
        pairwise. Flags two scenarios:
            - Overlap (gap < 0 min)  -> CONFLICT at 0.95 confidence
            - Tight transition (<15 min gap) -> RISK at 0.70 confidence

        CRITICAL FIX (iteration 107):
            The original implementation queried by event.timestamp (when the
            event was synced to the database) instead of the actual event
            start_time in the payload. This caused ALL calendar conflict
            predictions to be missed because synced events are timestamped
            in the past, even if the actual event is in the future.

            Now we:
            - Fetch all recent calendar events (last 30 days of syncs)
            - Parse start_time from each event's payload
            - Filter to events starting in the next 48 hours
            - Sort by actual start_time for accurate conflict detection
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Fetch calendar events synced in the last 30 days.
            # This captures all events the CalDAV connector has loaded,
            # including future events that were synced recently.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            events = conn.execute(
                """SELECT * FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if len(events) < 2:
            return predictions  # Need at least two events to find conflicts

        # Parse event payloads and extract actual start/end times.
        # Filter to events that START in the next 48 hours.
        now = datetime.now(timezone.utc)
        lookahead = now + timedelta(hours=48)

        parsed_events = []
        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Handle double-encoded JSON (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)

                start_str = payload.get("start_time", "")
                end_str = payload.get("end_time", "")

                if not start_str or not end_str:
                    continue  # Skip events without time bounds

                # Parse ISO timestamps. Handle both 'Z' suffix and '+00:00' format.
                # Some calendar events use date-only format (all-day events).
                try:
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                except ValueError:
                    # All-day events may be YYYY-MM-DD format without time
                    # Convert to datetime at midnight UTC for comparison
                    from datetime import date
                    try:
                        date_obj = date.fromisoformat(start_str)
                        start_dt = datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc)
                    except ValueError:
                        continue  # Skip unparseable dates

                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except ValueError:
                    # All-day event end date
                    try:
                        date_obj = date.fromisoformat(end_str)
                        end_dt = datetime.combine(date_obj, datetime.min.time(), tzinfo=timezone.utc)
                    except ValueError:
                        continue

                # Skip all-day events — they don't cause scheduling conflicts
                # in the traditional sense (you can have multiple all-day markers).
                if payload.get("is_all_day"):
                    continue

                # Only include events starting in the next 48 hours
                if start_dt >= now and start_dt <= lookahead:
                    parsed_events.append({
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "payload": payload,
                        "event_id": event["id"],
                    })

            except Exception as e:
                # Fail-open: skip individual parse errors without breaking
                # conflict detection for other events.
                continue

        if len(parsed_events) < 2:
            return predictions

        # Sort by actual event start time (not sync timestamp)
        parsed_events.sort(key=lambda e: e["start_dt"])

        # Compare each consecutive pair of events for overlap or tight gaps
        for i in range(len(parsed_events) - 1):
            curr = parsed_events[i]
            next_evt = parsed_events[i + 1]

            gap_minutes = (next_evt["start_dt"] - curr["end_dt"]).total_seconds() / 60

            if gap_minutes < 0:
                # Negative gap = events overlap in time
                predictions.append(Prediction(
                    prediction_type="conflict",
                    description=(
                        f"Calendar overlap: '{curr['payload'].get('title', 'Event')}' "
                        f"and '{next_evt['payload'].get('title', 'Event')}' overlap by "
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
                        f"'{curr['payload'].get('title', 'Event')}' and "
                        f"'{next_evt['payload'].get('title', 'Event')}'"
                    ),
                    confidence=0.7,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="24_hours",
                    suggested_action="Consider adding buffer time",
                ))

        return predictions

    async def _check_follow_up_needs(self, ctx: dict) -> list[Prediction]:
        """
        Detect messages that need a reply -- things the user read but
        hasn't responded to, especially from priority contacts.

        Strategy:
            1. Fetch all inbound messages from the last 24 hours that are:
               a) Unreplied (not in replied_to_threads set)
               b) Not marketing/automated (filtered by _is_marketing_or_noreply)
               c) Older than 3 hours (grace period for user to respond)
            2. Check if we've already created a prediction for this message
            3. Create new predictions only for messages we haven't alerted about yet

        CRITICAL FIX (iteration 62):
            Previously, this method processed ALL emails from the last 48 hours on
            EVERY prediction cycle (every 15 min). This caused:
            - 9,086 duplicate predictions created in a single batch
            - Overwhelming the database with redundant reminders
            - Breaking the accuracy feedback loop with noise

            Now we:
            - Track which message IDs we've already created predictions for
            - Only create ONE prediction per unreplied message, ever
            - Prevent reprocessing the same emails repeatedly

        PERFORMANCE FIX (iteration 81):
            With 70K+ emails in the database, scanning 48 hours of emails every
            15 minutes caused massive overhead (37K predictions/hour, 73K email
            scans every cycle). This caused the prediction engine to consume
            100% CPU continuously.

            Optimizations:
            - Reduced lookback window from 48h → 24h (cuts scan volume in half)
            - Early exit after scanning existing predictions (avoid redundant work)
            - Only fetch message IDs first, then details for new predictions
        """
        predictions = []

        # First, quickly check what messages we've already created predictions for
        # in the last 48 hours (wider than scan window to catch stragglers)
        prediction_cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        with self.db.get_connection("user_model") as conn:
            existing_predictions = conn.execute(
                """SELECT supporting_signals FROM predictions
                   WHERE prediction_type = 'reminder'
                   AND created_at > ?""",
                (prediction_cutoff,),
            ).fetchall()

        # Build a set of message IDs we've already created predictions for
        already_predicted_messages = set()
        for pred in existing_predictions:
            try:
                signals = json.loads(pred["supporting_signals"]) if pred["supporting_signals"] else {}
                msg_id = signals.get("message_id")
                if msg_id:
                    already_predicted_messages.add(msg_id)
            except (json.JSONDecodeError, TypeError):
                pass

        # Now fetch emails from a tighter window (24h instead of 48h)
        # This reduces the scan volume by 50% while still catching all actionable messages
        with self.db.get_connection("events") as conn:
            # Inbound messages from the last 24 hours
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

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

        # Build a set of thread/message IDs we've already replied to,
        # so we can exclude them from the "needs follow-up" list.
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

            # Skip if we've already created a prediction for this message
            # This is the critical fix to prevent duplicate predictions
            if message_id in already_predicted_messages:
                continue

            # Check if from a priority contact
            from_addr = payload.get("from_address", "")

            # Skip marketing/automated emails — no-reply senders, bulk
            # sender patterns, and messages containing "unsubscribe" in
            # snippet, body_plain, or body.
            if self._is_marketing_or_noreply(from_addr, payload):
                continue

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

            # Don't nag about very recent messages — give the user time
            if hours_ago < 3:
                continue

            # --- Confidence scoring for follow-up predictions ---
            # Base confidence is low (0.4) to avoid false positives.
            # Boosted by: priority contact (+0.3), age > 24h (+0.2),
            # explicit "requires_response" flag (+0.2). Capped at 0.9.
            confidence = 0.4
            if is_priority:
                confidence = 0.7   # Priority contacts start higher
            if hours_ago > 24:
                confidence = min(confidence + 0.2, 0.9)  # Aging messages get more urgent
            if payload.get("requires_response"):
                confidence = min(confidence + 0.2, 0.9)  # Explicit response request

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
                supporting_signals={
                    "contact_email": from_addr,
                    "contact_name": from_addr.split("@")[0],  # Simple heuristic
                    "message_id": message_id,
                    "hours_since_received": hours_ago,
                    "is_priority_contact": is_priority,
                    "requires_response": payload.get("requires_response", False),
                },
            ))

        return predictions

    async def _check_routine_deviations(self, ctx: dict) -> list[Prediction]:
        """
        Detect when the user deviates from their usual patterns.
        E.g., usually exercises Mon/Wed/Fri but hasn't today.

        Only considers routines with consistency_score > 0.6 — these are
        habits the user follows at least 60% of the time, so deviations
        are meaningful rather than noisy.
        """
        predictions = []

        # Load established routines from procedural memory
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
            # Simple day-based check: if today's day name appears in the
            # routine's trigger condition, the routine should have fired today.
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

        Uses the "relationships" signal profile, which tracks per-contact
        interaction history. For each contact with enough data (5+
        interactions), we compute the average gap between interactions
        and flag when the current gap exceeds 1.5x the average.
        """
        predictions = []

        # Load the relationships signal profile from the user model
        rel_profile = self.ums.get_signal_profile("relationships")
        if not rel_profile:
            return predictions

        contacts = rel_profile["data"].get("contacts", {})
        now = datetime.now(timezone.utc)

        for addr, data in contacts.items():
            last = data.get("last_interaction")
            count = data.get("interaction_count", 0)

            # Skip contacts with too little history — we need at least 5
            # interactions to establish a reliable frequency baseline.
            if not last or count < 5:
                continue

            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
            except (ValueError, TypeError):
                continue

            # Estimate typical contact frequency from the last 10 timestamps.
            # We compute the average gap in days between consecutive interactions.
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

                # Flag if the current gap exceeds 1.5x the average AND
                # it's been at least 7 days (avoid nagging about daily contacts).
                # Confidence scales linearly with how far past the threshold.
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
        learned patterns. E.g., "You have a flight tomorrow -- pack tonight."

        Looks at events 12-48 hours out (the "preparation window") and
        checks for keywords that signal preparation needs:
            - Travel keywords -> packing & reservation checks
            - Large meetings (>3 attendees) -> agenda review

        CRITICAL FIX (iteration 122):
            The original implementation queried by event.timestamp (when the
            event was synced to the database) instead of the actual event
            start_time in the payload. This caused ALL preparation need
            predictions to be missed because synced events are timestamped
            in the past, even if the actual event is in the future.

            Now we:
            - Fetch all recent calendar events (last 30 days of syncs)
            - Parse start_time from each event's payload
            - Filter to events starting in the 12-48 hour preparation window
            - Generate predictions based on actual event timing

            This is the same bug that was fixed for calendar conflicts in
            iteration 117 (PR #131).
        """
        predictions = []

        with self.db.get_connection("events") as conn:
            # Fetch calendar events synced in the last 30 days.
            # This captures all events the CalDAV connector has loaded,
            # including future events that were synced recently.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

            events = conn.execute(
                """SELECT * FROM events
                   WHERE type = 'calendar.event.created'
                   AND timestamp > ?""",
                (cutoff,),
            ).fetchall()

        if not events:
            return predictions

        # Parse event payloads and extract actual start times.
        # Filter to events that START in the 12-48 hour preparation window.
        now = datetime.now(timezone.utc)
        window_start = now + timedelta(hours=12)
        window_end = now + timedelta(hours=48)

        parsed_events = []
        for event in events:
            try:
                payload = json.loads(event["payload"])
                # Handle double-encoded JSON (rare but possible)
                if isinstance(payload, str):
                    payload = json.loads(payload)

                start_time_str = payload.get("start_time")
                if not start_time_str:
                    continue

                # Parse the start time and check if it's in our preparation window
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))

                if window_start <= start_time <= window_end:
                    parsed_events.append({
                        "start_time": start_time,
                        "payload": payload
                    })
            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                # Skip events with malformed payloads or missing start_time
                continue

        # Generate predictions for events in the preparation window
        for event in parsed_events:
            payload = event["payload"]
            title = payload.get("title", "").lower()
            location = payload.get("location", "")
            start_time = event["start_time"]

            # Calculate hours until event for more precise messaging
            hours_until = (start_time - now).total_seconds() / 3600

            # --- Travel detection ---
            # Keyword-based check for travel-related events.
            travel_keywords = ["flight", "airport", "hotel", "travel", "trip"]
            if any(kw in title for kw in travel_keywords):
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Upcoming travel in {int(hours_until)}h: '{payload.get('title')}'. Time to prepare.",
                    confidence=0.75,
                    confidence_gate=ConfidenceGate.DEFAULT,
                    time_horizon="24_hours",
                    suggested_action="Check packing list and confirm reservations",
                ))

            # --- Large meeting detection ---
            # Meetings with many attendees often need an agenda and talking points.
            attendees = payload.get("attendees", [])
            if len(attendees) > 3:
                predictions.append(Prediction(
                    prediction_type="need",
                    description=f"Large meeting in {int(hours_until)}h: '{payload.get('title')}' with {len(attendees)} attendees",
                    confidence=0.5,
                    confidence_gate=ConfidenceGate.SUGGEST,
                    time_horizon="24_hours",
                    suggested_action="Review agenda and prepare talking points",
                ))

        return predictions

    async def _check_spending_patterns(self, ctx: dict) -> list[Prediction]:
        """
        Detect spending anomalies and subscription waste.

        Aggregates the last 30 days of transactions by category and
        flags any single category that consumes >25% of total spend
        AND exceeds $200 absolute. The dual threshold avoids false
        positives for low overall spending or evenly-split budgets.
        """
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
            return predictions  # Not enough data for meaningful patterns

        # Aggregate spending by category from transaction payloads
        by_category: dict[str, float] = {}
        for txn in transactions:
            payload = json.loads(txn["payload"])
            cat = payload.get("category", "uncategorized")
            amount = abs(payload.get("amount", 0))
            by_category[cat] = by_category.get(cat, 0) + amount

        total = sum(by_category.values())
        if total == 0:
            return predictions

        # Flag categories that dominate spending (>25% share AND >$200 absolute)
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

        This is the reaction prediction gatekeeper. It scores each
        prediction on a -1.0 to +1.0 scale using multiple signals,
        then classifies the result:
            score > 0.3  -> "helpful"  (surface it)
            score > -0.1 -> "neutral"  (surface it, but lower priority)
            score <= -0.1 -> "annoying" (suppress it)

        CALIBRATION NOTE: The original thresholds (0.4 and 0.1) were too
        conservative, suppressing 99.95% of predictions and completely
        breaking the feedback loop. Recalibrated to allow more predictions
        through while still filtering truly annoying interruptions.
        """
        # --- Gather context signals ---
        # Current mood from the mood_signals profile
        mood_profile = self.ums.get_signal_profile("mood_signals")
        mood_data = mood_profile["data"] if mood_profile else {}

        # Count how many notifications the user dismissed in the last 2 hours.
        # A high count means they're in "leave me alone" mode.
        with self.db.get_connection("preferences") as conn:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM feedback_log
                   WHERE feedback_type = 'dismissed' AND timestamp > ?""",
                (cutoff,),
            ).fetchone()
            recent_dismissals = row["cnt"] if row else 0

        # --- Scoring logic ---
        # Start at 0.3 (slightly positive) to bias toward surfacing by default.
        # The original 0.5 start was too high given the penalty magnitudes.
        score = 0.3

        # Stress detection: check the last 5 mood signals for negative
        # language or calendar overload. If stressed, reduce score to
        # avoid piling more onto an overwhelmed user.
        # REDUCED from −0.2 to −0.1 to avoid over-penalizing.
        stress_signals = mood_data.get("recent_signals", [])
        stress_count = sum(1 for s in stress_signals[-5:]
                          if s.get("signal_type") in ["negative_language", "calendar_density"])
        if stress_count > 2:
            score -= 0.1

        # Dismissal fatigue: >3 dismissals in 2 hours = strong "go away" signal.
        # REDUCED from −0.3 to −0.2 and increased threshold from >3 to >5
        # to avoid being too reactive to a few dismissals.
        if recent_dismissals > 5:
            score -= 0.2

        # High-confidence predictions are more likely to be genuinely helpful
        if prediction.confidence > 0.7:
            score += 0.2

        # Urgency weighting: conflicts and risks warrant interruption more
        # than opportunities (which are nice-to-have, not need-to-know).
        if prediction.prediction_type in ("conflict", "risk"):
            score += 0.2
        elif prediction.prediction_type == "opportunity":
            score -= 0.05  # REDUCED from −0.1 to allow opportunities through

        # Time-of-day penalty: suppress non-urgent predictions during
        # early morning (before 7) and late night (after 22).
        # REDUCED from −0.3 to −0.2 to be less aggressive.
        hour = datetime.now(timezone.utc).hour
        if hour < 7 or hour > 22:
            if prediction.prediction_type not in ("conflict", "risk"):
                score -= 0.2

        # --- Classify the final score into a reaction label ---
        # RECALIBRATED: helpful >= 0.2, neutral > −0.1, else annoying.
        # The >= 0.2 threshold allows most "default" gate predictions (0.6-0.8
        # confidence) to surface unless they accumulate multiple penalties.
        # Round to 2 decimal places to avoid floating point precision issues
        # (e.g., 0.3 - 0.1 = 0.19999999... in Python).
        score = round(score, 2)
        predicted = "helpful" if score >= 0.2 else ("neutral" if score > -0.1 else "annoying")

        return ReactionPrediction(
            proposed_action=prediction.description,
            predicted_reaction=predicted,
            confidence=min(1.0, abs(score)),
            reasoning=f"score={score:.2f}, dismissals={recent_dismissals}, stress_signals={stress_count}",
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_accuracy_multiplier(self, prediction_type: str) -> float:
        """Compute confidence multiplier based on historical accuracy for this prediction type.

        Returns:
            0.0 — auto-suppress (<20% accuracy after 10+ resolved)
            0.5-1.1 — scaled by accuracy rate (50% accuracy = 1.0x baseline)
            1.0 — insufficient data (<5 resolved predictions)
        """
        with self.db.get_connection("user_model") as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_accurate = 1 THEN 1 ELSE 0 END) as accurate
                   FROM predictions
                   WHERE prediction_type = ?
                     AND was_surfaced = 1
                     AND resolved_at IS NOT NULL""",
                (prediction_type,),
            ).fetchone()

        total = row["total"] if row else 0
        accurate = row["accurate"] if row else 0

        if total < 5:
            return 1.0  # Not enough data to adjust

        accuracy_rate = accurate / total

        # Auto-suppress types with <20% accuracy after sufficient samples
        if accuracy_rate < 0.2 and total >= 10:
            return 0.0

        # Scale: 50% accuracy = 1.0x, 0% = 0.5x, 100% = 1.1x
        return 0.5 + (accuracy_rate * 0.6)

    @staticmethod
    def _is_marketing_or_noreply(from_addr: str, payload: dict) -> bool:
        """Check if an email is marketing/automated and shouldn't generate follow-up predictions.

        Filters out:
        - No-reply and automated system senders (mailer-daemon, postmaster, etc.)
        - Bulk/marketing email patterns (newsletter@, reply@, email@, service@, etc.)
        - Transactional/automated senders (orders@, auto-confirm@, shipment-tracking@, etc.)
        - Organizational bulk senders (communications@, development@, fundraising@)
        - Loyalty/rewards programs (rewards@, loyalty@, etc.)
        - Marketing domain patterns (news-*.com, email.*.com, etc.)
        - Marketing service providers (@*.e2ma.net, @*.sendgrid.net, etc.)
        - Embedded notification patterns (HOA-Notifications@, engage.ticketmaster.com)
        - Emails containing unsubscribe links

        This prevents low-quality prediction spam and protects the accuracy
        feedback loop from being polluted by marketing emails that should
        never have generated follow-up reminders.
        """
        addr_lower = from_addr.lower()

        # No-reply and automated system senders
        # Includes variations like noreply@, no-reply@, mailer-daemon@, postmaster@
        noreply_patterns = (
            "no-reply@", "noreply@", "do-not-reply@", "donotreply@",
            "mailer-daemon@", "postmaster@", "daemon@", "auto-reply@",
            "autoreply@", "automated@",
        )
        if any(pattern in addr_lower for pattern in noreply_patterns):
            return True

        # Common bulk sender local-parts (the part before @)
        # These patterns must match at the start of the email address to avoid
        # false positives like john.email@company.com or sarah.reply@startup.io
        bulk_localpart_patterns = (
            "newsletter@", "notifications@", "updates@", "digest@",
            "mailer@", "bulk@", "promo@", "marketing@",
            "reply@", "email@", "news@", "offers@", "deals@",
            "hello@", "info@", "support@", "help@",
            "service@",     # PayPal, Stripe, etc. - mostly transactional but repetitive
            "discover@",    # Common marketing pattern (Airbnb, etc.)
            "alert@", "alerts@", "notification@",
            # Transactional/automated senders
            "orders@", "order@", "receipts@", "receipt@",
            "auto-confirm@", "autoconfirm@", "confirmation@",
            "shipment-tracking@", "shipping@", "delivery@",
            "accountservice@", "account-service@",
            # Organizational bulk senders
            "communications@", "development@", "fundraising@",
            # Loyalty/rewards programs (always automated)
            "rewards@", "loyalty@",
        )
        if any(addr_lower.startswith(pattern) for pattern in bulk_localpart_patterns):
            return True

        # Embedded notification patterns (middle of local-part)
        # Catches: HOA-Notifications@, user-notifications@, system-alerts@
        embedded_notification_patterns = (
            "-notification", "-notifications", "-alert", "-alerts",
            "-update", "-updates", "-digest",
        )
        # Extract local-part (everything before @)
        local_part = addr_lower.split("@")[0] if "@" in addr_lower else addr_lower
        if any(pattern in local_part for pattern in embedded_notification_patterns):
            return True

        # Marketing domain patterns (the part after @)
        # These catch domains like news-us.hugoboss.com, email.d23.com, reply.*.com
        marketing_domain_patterns = (
            "@news-", "@email.", "@reply.", "@mailing.",
            "@newsletters.", "@promo.", "@marketing.",
            "@em.", "@mg.", "@mail.",  # Common email service provider patterns
            "@engage.", "@iluv.", "@e.", "@e2.",  # Engagement platforms (e.g., engage.ticketmaster.com)
        )
        if any(pattern in addr_lower for pattern in marketing_domain_patterns):
            return True

        # Marketing service provider subdomains
        # These are third-party email marketing platforms (e.g., @*.e2ma.net, @*.sendgrid.net)
        # Check if domain ends with these patterns
        domain = addr_lower.split("@")[1] if "@" in addr_lower else ""
        marketing_service_patterns = (
            ".e2ma.net",      # Emma email marketing
            ".sendgrid.net",  # SendGrid
            ".mailchimp.com", # Mailchimp
            ".constantcontact.com",  # Constant Contact
            ".hubspot.com",   # HubSpot
            ".marketo.com",   # Marketo
            ".pardot.com",    # Salesforce Pardot
            ".eloqua.com",    # Oracle Eloqua
        )
        if any(domain.endswith(pattern) for pattern in marketing_service_patterns):
            return True

        # Check body and snippet for unsubscribe indicators
        # Marketing emails are legally required to include unsubscribe links
        text = " ".join(filter(None, [
            payload.get("body_plain", ""),
            payload.get("snippet", ""),
            payload.get("body", ""),
        ])).lower()
        if "unsubscribe" in text:
            return True

        return False

    @staticmethod
    def _gate_from_confidence(confidence: float) -> ConfidenceGate:
        """
        Map a numeric confidence score to a ConfidenceGate enum.

        Confidence gate thresholds:
            < 0.3  -> OBSERVE    (watch silently, keep learning)
            0.3-0.6 -> SUGGEST   (ask "would you like me to...")
            0.6-0.8 -> DEFAULT   (do it, but make it easy to undo)
            > 0.8  -> AUTONOMOUS (just handle it without asking)
        """
        if confidence < 0.3:
            return ConfidenceGate.OBSERVE
        elif confidence < 0.6:
            return ConfidenceGate.SUGGEST
        elif confidence < 0.8:
            return ConfidenceGate.DEFAULT
        else:
            return ConfidenceGate.AUTONOMOUS
