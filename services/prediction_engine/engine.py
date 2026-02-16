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
        self._last_time_based_run: Optional[datetime] = None  # Last time-based prediction run

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

    def _should_run_time_based_predictions(self) -> bool:
        """
        Check if time-based predictions should run.

        Time-based predictions check temporal conditions (time passing, approaching events,
        missed routines) rather than reacting to new events. They should run periodically
        even when no new events have arrived.

        Examples of time-based predictions:
        - Relationship maintenance (days since last contact increasing)
        - Routine deviations (expected routine didn't occur)
        - Preparation needs (event approaching in time)
        - Calendar conflicts (future events coming into 48h window)

        These run every 15 minutes to detect changes in temporal state.
        """
        now = datetime.now(timezone.utc)

        # First run always executes
        if self._last_time_based_run is None:
            self._last_time_based_run = now
            return True

        # Run if 15+ minutes have passed since last time-based check
        time_since_last = (now - self._last_time_based_run).total_seconds() / 60
        if time_since_last >= 15:
            self._last_time_based_run = now
            return True

        return False

    async def generate_predictions(self, current_context: dict) -> list[Prediction]:
        """
        Main prediction loop. Called periodically (every 15 min)
        and on significant context changes (location, calendar event start, etc.)

        Runs prediction generation when EITHER:
        1. New events have arrived (event-based predictions like follow-up needs)
        2. 15+ minutes have passed (time-based predictions like relationship maintenance)

        This dual-trigger approach ensures both reactive predictions (responding to
        new events) and proactive predictions (detecting temporal conditions) work
        correctly.
        """
        # Determine which trigger conditions are met
        has_new_events = self._has_new_events()
        time_based_due = self._should_run_time_based_predictions()

        # Skip entirely if neither trigger is active
        if not has_new_events and not time_based_due:
            return []

        predictions = []

        # --- Prediction generation pipeline ---
        # Each _check_* method is a specialized detector that looks for one
        # category of predictable user need. They run independently and
        # return zero or more Prediction objects. The full set of prediction
        # types covers the most common "blown away" moments:

        # Track prediction generation for observability
        generation_stats = {}

        # TIME-BASED predictions: Run when time passes (even without new events)
        # These check temporal conditions like approaching events, missed routines,
        # and relationship gaps growing wider.
        if time_based_due:
            calendar_preds = await self._check_calendar_conflicts(current_context)
            generation_stats['calendar_conflicts'] = len(calendar_preds)
            predictions.extend(calendar_preds)

            routine_preds = await self._check_routine_deviations(current_context)
            generation_stats['routine_deviations'] = len(routine_preds)
            predictions.extend(routine_preds)

            relationship_preds = await self._check_relationship_maintenance(current_context)
            generation_stats['relationship_maintenance'] = len(relationship_preds)
            predictions.extend(relationship_preds)

            prep_preds = await self._check_preparation_needs(current_context)
            generation_stats['preparation_needs'] = len(prep_preds)
            predictions.extend(prep_preds)
        else:
            # Skip time-based predictions, mark as not run
            generation_stats['calendar_conflicts'] = '(skipped: no time trigger)'
            generation_stats['routine_deviations'] = '(skipped: no time trigger)'
            generation_stats['relationship_maintenance'] = '(skipped: no time trigger)'
            generation_stats['preparation_needs'] = '(skipped: no time trigger)'

        # EVENT-BASED predictions: Run when new events arrive
        # These react to specific events like incoming emails or spending activity.
        if has_new_events:
            followup_preds = await self._check_follow_up_needs(current_context)
            generation_stats['follow_up_needs'] = len(followup_preds)
            predictions.extend(followup_preds)

            spending_preds = await self._check_spending_patterns(current_context)
            generation_stats['spending_patterns'] = len(spending_preds)
            predictions.extend(spending_preds)
        else:
            # Skip event-based predictions, mark as not run
            generation_stats['follow_up_needs'] = '(skipped: no new events)'
            generation_stats['spending_patterns'] = '(skipped: no new events)'

        print(f"[prediction_engine] Generated predictions by type: {generation_stats} (total={len(predictions)}) [triggers: events={has_new_events}, time={time_based_due}]")

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
        #
        # Track reaction predictions for each prediction so we can log filter reasons
        reaction_map = {}
        filtered = []
        for pred in predictions:
            reaction = await self.predict_reaction(pred, current_context)
            reaction_map[pred.id] = reaction
            if reaction.predicted_reaction in ("helpful", "neutral"):
                filtered.append(pred)
            else:
                # Mark why this prediction was filtered (annoying reaction)
                pred.filter_reason = f"reaction:{reaction.predicted_reaction} ({reaction.reasoning})"

        # Confidence floor — don't surface anything below SUGGEST threshold (0.3).
        # This enables relationship maintenance, preparation, and other valuable
        # predictions that should be shown as suggestions ("Would you like...?")
        # even if confidence isn't high enough for autonomous action.
        filtered_after_confidence = []
        for p in filtered:
            if p.confidence >= 0.3:
                filtered_after_confidence.append(p)
            else:
                # Mark why this prediction was filtered (low confidence)
                p.filter_reason = f"confidence:{p.confidence:.3f} (threshold:0.3)"
        filtered = filtered_after_confidence

        # Cap at 5 surfaced predictions per cycle, prioritized by confidence
        filtered.sort(key=lambda p: p.confidence, reverse=True)
        if len(filtered) > 5:
            # Mark predictions filtered due to ranking
            for p in filtered[5:]:
                p.filter_reason = f"ranking:position_{filtered.index(p)+1} (top_5_cutoff, confidence:{p.confidence:.3f})"
        filtered = filtered[:5]

        # Log filtering results for observability
        filtered_by_reaction = len([p for p in predictions if p.filter_reason and p.filter_reason.startswith("reaction:")])
        filtered_by_confidence = len([p for p in predictions if p.filter_reason and p.filter_reason.startswith("confidence:")])
        filtered_by_ranking = len([p for p in predictions if p.filter_reason and p.filter_reason.startswith("ranking:")])
        print(f"[prediction_engine] Filtering: {len(predictions)} raw → {len(filtered)} surfaced "
              f"(filtered: {filtered_by_reaction} by reaction, {filtered_by_confidence} by confidence, {filtered_by_ranking} by ranking)")

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
                # Ensure filter_reason is set (if not already set by filtering logic above)
                if not pred.filter_reason:
                    pred.filter_reason = "unknown (should not happen)"

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
            - Overlap (gap < 0 min)  -> CONFLICT at 0.95 confidence (0.8 for all-day)
            - Tight transition (<15 min gap) -> RISK at 0.70 confidence (timed only)

        All-day event handling:
            - All-day vs all-day: No conflict (multiple all-day markers are fine)
            - All-day vs timed: Conflict detected (different locations/contexts)
            - Timed vs timed: Full conflict detection with gap analysis

        CRITICAL FIX (iteration 132):
            The code was filtering out ALL all-day events (line 261-262), which
            meant 99.9% of calendar events (2,571 of 2,573 in production) were
            ignored. This completely broke calendar conflict detection, causing
            0 predictions despite having thousands of events in the database.

            Now we:
            - Include all-day events in the conflict detection pipeline
            - Skip all-day vs all-day comparisons (multiple markers are fine)
            - Detect all-day vs timed conflicts (e.g., meeting during travel day)
            - Enable downstream features that depend on all-day events

        CRITICAL FIX (iteration 117):
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

        # Diagnostic logging for observability
        if len(events) < 2:
            print(f"[prediction_engine.calendar_conflicts] No conflicts possible: {len(events)} calendar events found (need ≥2)")
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
                # CRITICAL FIX (iteration 128): fromisoformat() parses date-only
                # strings like "2026-02-14" successfully but creates timezone-NAIVE
                # datetimes. This caused all calendar predictions to fail when
                # comparing naive vs aware datetimes. Now we explicitly check and
                # add UTC timezone to any naive datetime.
                try:
                    start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    # If timezone-naive (all-day events), make it UTC-aware
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    # Completely unparseable — skip this event
                    continue

                try:
                    end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    # If timezone-naive (all-day events), make it UTC-aware
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    # Completely unparseable — skip this event
                    continue

                # Check if event falls within the 48-hour lookahead window.
                # CRITICAL FIX (iteration 143): All-day events have date-only timestamps
                # like "2026-02-16" which parse as midnight UTC. If it's currently 18:52 UTC,
                # today's all-day events appear to have started 18+ hours ago and fail the
                # `start_dt >= now` check. This caused 99.9% of calendar events to be excluded
                # from conflict/preparation predictions (2,571 all-day vs 2 timed events).
                #
                # Solution: For all-day events, check if their DATE falls within the window
                # (today through 2 days from now) rather than checking their midnight timestamp.
                # For timed events, include events that are still ongoing OR will start within 48h.
                is_all_day = payload.get("is_all_day", False)

                in_window = False
                if is_all_day:
                    # For all-day events: check if date falls within [today, today+2 days]
                    # This captures today's events (even if midnight has passed) and upcoming ones.
                    event_date = start_dt.date()
                    today = now.date()
                    lookahead_date = (now + timedelta(days=2)).date()
                    in_window = today <= event_date <= lookahead_date
                else:
                    # For timed events: include if EITHER:
                    # 1. Event hasn't ended yet (ongoing), OR
                    # 2. Event will start within the 48h window (upcoming)
                    # This ensures we catch conflicts with events happening right now.
                    event_ended = end_dt < now
                    event_starts_soon = start_dt <= lookahead
                    in_window = not event_ended and event_starts_soon

                if in_window:
                    parsed_events.append({
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "payload": payload,
                        "event_id": event["id"],
                        "is_all_day": is_all_day,
                    })

            except Exception as e:
                # Fail-open: skip individual parse errors without breaking
                # conflict detection for other events.
                continue

        # Count event types for diagnostics
        all_day_count = sum(1 for e in parsed_events if e.get("is_all_day"))
        timed_count = len(parsed_events) - all_day_count

        if len(parsed_events) < 2:
            print(f"[prediction_engine.calendar_conflicts] No conflicts possible: {len(parsed_events)} events in 48h window (all_day={all_day_count}, timed={timed_count}, total_synced={len(events)})")
            return predictions

        # Sort by actual event start time (not sync timestamp)
        parsed_events.sort(key=lambda e: e["start_dt"])

        # Compare each consecutive pair of events for overlap or tight gaps.
        # Skip all-day event pairs (multiple all-day markers are fine), but DO
        # compare timed events with all-day events (e.g., a timed meeting during
        # an all-day conference in a different location IS a conflict).
        comparisons_made = 0
        skipped_all_day_pairs = 0
        for i in range(len(parsed_events) - 1):
            curr = parsed_events[i]
            next_evt = parsed_events[i + 1]

            # Skip if both events are all-day (no conflict between all-day markers)
            if curr.get("is_all_day") and next_evt.get("is_all_day"):
                skipped_all_day_pairs += 1
                continue

            comparisons_made += 1

            gap_minutes = (next_evt["start_dt"] - curr["end_dt"]).total_seconds() / 60

            if gap_minutes < 0:
                # Negative gap = events overlap in time.
                # For all-day events, only flag if one is timed (location conflict).
                # For timed events, always flag.
                is_all_day_conflict = curr.get("is_all_day") or next_evt.get("is_all_day")

                predictions.append(Prediction(
                    prediction_type="conflict",
                    description=(
                        f"Calendar overlap: '{curr['payload'].get('title', 'Event')}' "
                        f"and '{next_evt['payload'].get('title', 'Event')}' overlap"
                        + ("" if is_all_day_conflict else f" by {abs(int(gap_minutes))} minutes")
                    ),
                    confidence=0.8 if is_all_day_conflict else 0.95,
                    confidence_gate=ConfidenceGate.DEFAULT,
                    time_horizon="24_hours",
                    suggested_action="Reschedule one of the conflicting events",
                ))
            elif gap_minutes < 15 and not (curr.get("is_all_day") or next_evt.get("is_all_day")):
                # Very tight transition (only for timed events — all-day events
                # don't have tight transitions by definition).
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

        # Diagnostic summary
        print(f"[prediction_engine.calendar_conflicts] Analyzed {len(events)} synced events "
              f"→ {len(parsed_events)} in 48h window (all_day={all_day_count}, timed={timed_count}) "
              f"→ {comparisons_made} comparisons (skipped {skipped_all_day_pairs} all-day pairs) "
              f"→ {len(predictions)} predictions")

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

            # Skip messages with missing or empty from_address — these are
            # malformed events that shouldn't generate predictions. Without
            # this check, empty addresses bypass the marketing filter entirely
            # and create broken predictions with blank sender fields.
            if not from_addr or not from_addr.strip():
                continue

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

        CRITICAL FIX (iteration 114):
            Previously, this method had a stubbed implementation with two fatal bugs:
            1. Day-name matching logic was inverted: checked if day name IN trigger
               (e.g., "monday" in "morning") instead of checking the trigger's day pattern
            2. Never checked if routine was actually completed — created duplicate
               predictions every 15 minutes regardless of user behavior

            Now we:
            - First check if we've already created a prediction for this routine today
            - Parse routine steps to identify expected event types
            - Query events table to see if those event types occurred today
            - Only create prediction if routine hasn't been completed
            - Track routine_name in supporting_signals for deduplication
        """
        predictions = []

        # Load established routines from procedural memory
        with self.db.get_connection("user_model") as conn:
            routines = conn.execute(
                "SELECT * FROM routines WHERE consistency_score > 0.6"
            ).fetchall()

        if not routines:
            print(f"[prediction_engine.routine_deviations] No predictions: 0 routines with consistency_score > 0.6")
            return predictions

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Check what routine deviation predictions we've already created today
        # to avoid duplicate reminders every 15 minutes
        with self.db.get_connection("user_model") as conn:
            existing_predictions = conn.execute(
                """SELECT supporting_signals FROM predictions
                   WHERE prediction_type = 'opportunity'
                   AND created_at > ?""",
                (today_start.isoformat(),),
            ).fetchall()

        # Build set of routines we've already created predictions for today
        already_predicted_routines = set()
        for pred in existing_predictions:
            try:
                signals = json.loads(pred["supporting_signals"]) if pred["supporting_signals"] else {}
                routine_name = signals.get("routine_name")
                if routine_name:
                    already_predicted_routines.add(routine_name)
            except (json.JSONDecodeError, TypeError):
                pass

        # For each routine, check if it should have been completed by now
        for routine in routines:
            routine_name = routine["name"]

            # Skip if we've already created a prediction for this routine today
            if routine_name in already_predicted_routines:
                continue

            # Parse the routine steps to identify what event types to look for
            try:
                steps = json.loads(routine["steps"])
                if not steps:
                    continue

                # Extract expected event types from the workflow steps
                # Steps are dicts with "action" keys (e.g., "email_received", "task_created")
                expected_actions = []
                for step in steps[:3]:  # Only check first 3 steps for performance
                    if isinstance(step, dict):
                        action = step.get("action", "")
                        if action:
                            expected_actions.append(action)

                if not expected_actions:
                    continue

                # Map routine action types to event types
                # Routine actions use underscore format (email_received) while
                # event types use dot format (email.received)
                event_type_mapping = {
                    "email_received": "email.received",
                    "email_sent": "email.sent",
                    "message_received": "message.received",
                    "message_sent": "message.sent",
                    "task_created": "task.created",
                    "task_completed": "task.completed",
                    "calendar_event_created": "calendar.event.created",
                }

                expected_event_types = []
                for action in expected_actions:
                    event_type = event_type_mapping.get(action, action.replace("_", "."))
                    expected_event_types.append(event_type)

                # Check if any of the expected event types occurred today
                with self.db.get_connection("events") as conn:
                    result = conn.execute(
                        f"""SELECT COUNT(*) as count FROM events
                           WHERE type IN ({','.join('?' * len(expected_event_types))})
                           AND timestamp > ?""",
                        (*expected_event_types, today_start.isoformat()),
                    ).fetchone()

                if result and result["count"] > 0:
                    # Routine was completed today — no deviation
                    continue

                # Routine hasn't been completed today — create a prediction
                # Confidence is based on consistency score but capped lower since
                # routines can have legitimate skip days
                confidence = min(routine["consistency_score"] * 0.5, 0.65)

                # Only surface if confidence meets SUGGEST threshold (0.3+)
                if confidence < 0.3:
                    continue

                predictions.append(Prediction(
                    prediction_type="opportunity",
                    description=f"You usually do your '{routine_name}' routine by now",
                    confidence=confidence,
                    confidence_gate=self._gate_from_confidence(confidence),
                    time_horizon="today",
                    suggested_action=f"Start {routine_name}",
                    supporting_signals={
                        "routine_name": routine_name,
                        "consistency_score": routine["consistency_score"],
                        "expected_actions": expected_actions,
                    },
                ))

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Fail-open: skip routines with malformed data
                continue

        # Diagnostic summary
        print(f"[prediction_engine.routine_deviations] Analyzed {len(routines)} routines "
              f"(already_predicted_today={len(already_predicted_routines)}) "
              f"→ {len(predictions)} predictions")

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
            print(f"[prediction_engine.relationship_maintenance] No predictions: relationships profile not found")
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

            # Skip marketing/automated senders — relationship maintenance is
            # for human connections, not bulk email subscriptions. Without
            # this filter, the system would generate "reach out" suggestions
            # for addresses like callofduty@comms.activision.com which are
            # marketing automations, not relationships to maintain.
            if self._is_marketing_or_noreply(addr, {}):
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

        # Diagnostic summary
        total_contacts = len(contacts)
        eligible = sum(1 for data in contacts.values() if data.get("interaction_count", 0) >= 5)
        marketing_filtered = sum(1 for addr in contacts.keys() if self._is_marketing_or_noreply(addr, {}))
        print(f"[prediction_engine.relationship_maintenance] Analyzed {total_contacts} contacts "
              f"(eligible={eligible}, marketing_filtered={marketing_filtered}) "
              f"→ {len(predictions)} predictions")

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
            print(f"[prediction_engine.preparation_needs] No predictions: 0 calendar events found")
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
                # CRITICAL FIX (iteration 128): Same timezone-naive bug as calendar
                # conflicts — date-only strings parse but create naive datetimes.
                # CRITICAL FIX (iteration 143): All-day events with date-only timestamps
                # like "2026-02-16" parse as midnight UTC and fail time window checks if
                # it's already past midnight. Apply the same date-based window logic used
                # in calendar conflict detection.
                start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)

                is_all_day = payload.get("is_all_day", False)
                in_window = False

                if is_all_day:
                    # For all-day events: check if date falls within preparation window
                    # (tomorrow through 2 days from now for 12-48h window)
                    event_date = start_time.date()
                    window_start_date = window_start.date()
                    window_end_date = window_end.date()
                    in_window = window_start_date <= event_date <= window_end_date
                else:
                    # For timed events: check if start time falls in 12-48h window
                    in_window = window_start <= start_time <= window_end

                if in_window:
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

        # Diagnostic summary
        travel_events = sum(1 for e in parsed_events if any(kw in e["payload"].get("title", "").lower() for kw in ["flight", "airport", "hotel", "travel", "trip"]))
        large_meetings = sum(1 for e in parsed_events if len(e["payload"].get("attendees", [])) > 3)
        print(f"[prediction_engine.preparation_needs] Analyzed {len(events)} synced events "
              f"→ {len(parsed_events)} in 12-48h window "
              f"(travel={travel_events}, large_meetings={large_meetings}) "
              f"→ {len(predictions)} predictions")

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
            print(f"[prediction_engine.spending_patterns] No predictions: {len(transactions)} transactions found (need ≥5)")
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

        # Diagnostic summary
        high_spend_categories = sum(1 for cat, amt in by_category.items() if amt / total > 0.25 and amt > 200)
        print(f"[prediction_engine.spending_patterns] Analyzed {len(transactions)} transactions "
              f"(total=${total:.0f}, categories={len(by_category)}, high_spend={high_spend_categories}) "
              f"→ {len(predictions)} predictions")

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
        # CRITICAL: This list must be comprehensive as it's the last line of defense
        # against marketing emails polluting relationship tracking and predictions.
        # Production data showed addresses like callofduty@comms.activision.com and
        # similar patterns slipping through, creating 820 "contacts" when only ~20
        # are actual human relationships.
        #
        # IMPORTANT: Patterns must not match legitimate personal email providers.
        # The @mail. pattern was removed in iteration 160 because it incorrectly
        # blocked @gmail.com, @hotmail.com, @protonmail.com, etc., completely
        # breaking relationship maintenance predictions for all Gmail users.
        marketing_domain_patterns = (
            "@news-", "@email.", "@reply.", "@mailing.",
            "@newsletters.", "@promo.", "@marketing.",
            "@em.", "@mg.",  # Common email service provider patterns (mail. removed - too broad)
            "@engage.", "@iluv.", "@e.", "@e2.",  # Engagement platforms (e.g., engage.ticketmaster.com)
            "@comms.", "@communications.",  # Corporate communications (e.g., comms.activision.com)
            "@attn.",  # Attention/notification platforms (e.g., attn.us.lg.com)
            "@txn.", "@transactional.",  # Transaction notifications
            "@deals.", "@offers.", "@promo-",  # Promotional campaigns
            "@campaigns.", "@campaign.",  # Campaign management platforms
            "@blast.", "@bulk.",  # Bulk sender platforms
            "@lists.", "@list.",  # Mailing list managers
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
            ".sailthru.com",  # Sailthru email platform
            ".responsys.net", # Oracle Responsys
            ".exacttarget.com",  # Salesforce Marketing Cloud
            ".smtp2go.com",   # SMTP2GO transactional email
            ".postmarkapp.com",  # Postmark transactional
            ".mandrillapp.com",  # Mandrill by Mailchimp
            ".amazonses.com",    # Amazon SES
            ".sparkpostmail.com", # SparkPost
            ".sendinblue.com",   # Sendinblue/Brevo
            ".intercom-mail.com", # Intercom notifications
            ".customer.io",       # Customer.io
            ".iterable.com",      # Iterable
            ".klaviyo.com",       # Klaviyo e-commerce marketing
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

    async def get_diagnostics(self) -> dict:
        """
        Comprehensive prediction engine diagnostics.

        Returns a detailed analysis of why each prediction type is or isn't
        generating predictions, including data availability, configuration gaps,
        and actionable recommendations.

        This is the single source of truth for understanding prediction engine
        behavior and debugging issues.

        Returns:
            Dictionary with structure:
            {
                "prediction_types": {
                    "reminder": {
                        "status": "active" | "limited" | "blocked",
                        "generated_last_7d": int,
                        "data_available": {
                            "unreplied_emails": int,
                            "recent_messages": int,
                            ...
                        },
                        "blockers": ["list", "of", "issues"],
                        "recommendations": ["actionable", "steps"]
                    },
                    ...
                },
                "overall": {
                    "total_predictions_7d": int,
                    "active_types": int,
                    "blocked_types": int,
                    "health": "healthy" | "degraded" | "broken"
                }
            }
        """
        diagnostics = {"prediction_types": {}, "overall": {}}
        now = datetime.now(timezone.utc)
        week_ago = (now - timedelta(days=7)).isoformat()

        # Get prediction counts for last 7 days
        with self.db.get_connection("user_model") as conn:
            pred_counts = conn.execute(
                """SELECT prediction_type, COUNT(*) as count
                   FROM predictions
                   WHERE created_at > ?
                   GROUP BY prediction_type""",
                (week_ago,),
            ).fetchall()

        prediction_type_counts = {row["prediction_type"]: row["count"] for row in pred_counts}

        # --- Follow-up needs (reminder) ---
        with self.db.get_connection("events") as conn:
            unreplied_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'email.received'
                   AND timestamp > ?""",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()["count"]

            replied_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'email.sent'
                   AND json_extract(payload, '$.in_reply_to') IS NOT NULL
                   AND timestamp > ?""",
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()["count"]

        actual_unreplied = unreplied_count - replied_count
        reminder_status = "active" if prediction_type_counts.get("reminder", 0) > 0 else "limited"
        reminder_blockers = []
        reminder_recommendations = []

        if actual_unreplied == 0:
            reminder_status = "blocked"
            reminder_blockers.append("No unreplied emails in last 24h")
            reminder_recommendations.append("Check if email connector is syncing correctly")
        elif prediction_type_counts.get("reminder", 0) == 0:
            reminder_blockers.append("Unreplied emails exist but 0 predictions generated")
            reminder_recommendations.append("Check marketing filter - may be filtering all emails")

        diagnostics["prediction_types"]["reminder"] = {
            "status": reminder_status,
            "generated_last_7d": prediction_type_counts.get("reminder", 0),
            "data_available": {
                "unreplied_emails_24h": actual_unreplied,
                "total_received_24h": unreplied_count,
                "replies_sent_24h": replied_count,
            },
            "blockers": reminder_blockers,
            "recommendations": reminder_recommendations,
        }

        # --- Calendar conflicts ---
        with self.db.get_connection("events") as conn:
            calendar_events = conn.execute(
                "SELECT COUNT(*) as count FROM events WHERE type = 'calendar.event.created'"
            ).fetchone()["count"]

            # Count all-day vs timed events
            all_events = conn.execute(
                "SELECT payload FROM events WHERE type = 'calendar.event.created' LIMIT 1000"
            ).fetchall()

            all_day_count = 0
            timed_count = 0
            for row in all_events:
                try:
                    payload = json.loads(row["payload"])
                    if payload.get("is_all_day"):
                        all_day_count += 1
                    else:
                        timed_count += 1
                except:
                    pass

        conflict_status = "active" if prediction_type_counts.get("conflict", 0) > 0 else "blocked"
        conflict_blockers = []
        conflict_recommendations = []

        if calendar_events == 0:
            conflict_blockers.append("No calendar events in database")
            conflict_recommendations.append("Enable and configure CalDAV or Google Calendar connector")
        elif timed_count == 0:
            conflict_blockers.append("All calendar events are all-day (0 timed events for conflict detection)")
            conflict_recommendations.append("Verify calendar connector is correctly parsing timed events")
            conflict_recommendations.append("Check if calendar contains any actual appointments (not just birthdays/holidays)")

        diagnostics["prediction_types"]["conflict"] = {
            "status": conflict_status,
            "generated_last_7d": prediction_type_counts.get("conflict", 0),
            "data_available": {
                "total_calendar_events": calendar_events,
                "all_day_events": all_day_count,
                "timed_events": timed_count,
            },
            "blockers": conflict_blockers,
            "recommendations": conflict_recommendations,
        }

        # --- Relationship maintenance (opportunity) ---
        rel_profile = self.ums.get_signal_profile("relationships")
        if rel_profile:
            contacts = rel_profile["data"].get("contacts", {})
            eligible_contacts = sum(1 for data in contacts.values()
                                   if data.get("interaction_count", 0) >= 5)
            total_contacts = len(contacts)
        else:
            contacts = {}
            eligible_contacts = 0
            total_contacts = 0

        opportunity_status = "active" if prediction_type_counts.get("opportunity", 0) > 0 else "blocked"
        opportunity_blockers = []
        opportunity_recommendations = []

        if total_contacts == 0:
            opportunity_blockers.append("No contacts tracked in relationships profile")
            opportunity_recommendations.append("Ensure email connector is running and processing messages")
        elif eligible_contacts == 0:
            opportunity_blockers.append("No contacts with 5+ interactions (need history to detect maintenance needs)")
            opportunity_recommendations.append("Wait for more email history to accumulate (need 5+ interactions per contact)")
        else:
            # Check if all contacts are marketing
            marketing_count = sum(1 for addr in contacts.keys()
                                 if self._is_marketing_or_noreply(addr, {}))
            if marketing_count / total_contacts > 0.9:
                opportunity_blockers.append(f"{marketing_count}/{total_contacts} contacts are marketing/automated (no human relationships)")
                opportunity_recommendations.append("This inbox appears to contain primarily marketing emails")
                opportunity_recommendations.append("Consider filtering marketing emails before they reach Life OS")

        diagnostics["prediction_types"]["opportunity"] = {
            "status": opportunity_status,
            "generated_last_7d": prediction_type_counts.get("opportunity", 0),
            "data_available": {
                "total_contacts": total_contacts,
                "eligible_contacts": eligible_contacts,
                "marketing_filtered": sum(1 for addr in contacts.keys()
                                         if self._is_marketing_or_noreply(addr, {}))
                                     if contacts else 0,
            },
            "blockers": opportunity_blockers,
            "recommendations": opportunity_recommendations,
        }

        # --- Preparation needs (need) ---
        with self.db.get_connection("events") as conn:
            upcoming_events = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'calendar.event.created'"""
            ).fetchone()["count"]

        need_status = "active" if prediction_type_counts.get("need", 0) > 0 else "blocked"
        need_blockers = []
        need_recommendations = []

        if upcoming_events == 0:
            need_blockers.append("No calendar events available")
            need_recommendations.append("Enable calendar connector to track upcoming events")
        elif timed_count == 0:
            need_blockers.append("All events are all-day (preparation needs require timed events)")
            need_recommendations.append("Verify calendar contains actual appointments, not just reminders")

        diagnostics["prediction_types"]["need"] = {
            "status": need_status,
            "generated_last_7d": prediction_type_counts.get("need", 0),
            "data_available": {
                "total_events": upcoming_events,
                "timed_events": timed_count,
            },
            "blockers": need_blockers,
            "recommendations": need_recommendations,
        }

        # --- Spending patterns (risk) ---
        with self.db.get_connection("events") as conn:
            transaction_count = conn.execute(
                """SELECT COUNT(*) as count FROM events
                   WHERE type = 'finance.transaction.new'
                   AND timestamp > ?""",
                ((now - timedelta(days=30)).isoformat(),),
            ).fetchone()["count"]

        risk_status = "active" if prediction_type_counts.get("risk", 0) > 0 else "blocked"
        risk_blockers = []
        risk_recommendations = []

        if transaction_count == 0:
            risk_blockers.append("No finance transactions in database")
            risk_recommendations.append("Enable a finance connector (Plaid, Mint, or similar)")
        elif transaction_count < 5:
            risk_blockers.append(f"Only {transaction_count} transactions (need ≥5 for pattern detection)")
            risk_recommendations.append("Wait for more transaction history to accumulate")

        diagnostics["prediction_types"]["risk"] = {
            "status": risk_status,
            "generated_last_7d": prediction_type_counts.get("risk", 0),
            "data_available": {
                "transactions_30d": transaction_count,
            },
            "blockers": risk_blockers,
            "recommendations": risk_recommendations,
        }

        # --- Routine deviations ---
        with self.db.get_connection("user_model") as conn:
            routine_count = conn.execute(
                "SELECT COUNT(*) as count FROM routines WHERE consistency_score > 0.6"
            ).fetchone()["count"]

        routine_status = "active" if prediction_type_counts.get("routine_deviation", 0) > 0 else "blocked"
        routine_blockers = []
        routine_recommendations = []

        if routine_count == 0:
            routine_blockers.append("No routines with consistency_score > 0.6")
            routine_recommendations.append("Routine detection requires consistent behavioral patterns over time")
            routine_recommendations.append("Wait for routine detection loop to identify patterns (runs hourly)")

        diagnostics["prediction_types"]["routine_deviation"] = {
            "status": routine_status,
            "generated_last_7d": prediction_type_counts.get("routine_deviation", 0),
            "data_available": {
                "established_routines": routine_count,
            },
            "blockers": routine_blockers,
            "recommendations": routine_recommendations,
        }

        # --- Overall health ---
        total_predictions_7d = sum(prediction_type_counts.values())
        active_types = sum(1 for v in diagnostics["prediction_types"].values()
                          if v["status"] == "active")
        blocked_types = sum(1 for v in diagnostics["prediction_types"].values()
                           if v["status"] == "blocked")

        if active_types >= 3:
            health = "healthy"
        elif active_types >= 1:
            health = "degraded"
        else:
            health = "broken"

        diagnostics["overall"] = {
            "total_predictions_7d": total_predictions_7d,
            "active_types": active_types,
            "blocked_types": blocked_types,
            "total_types": len(diagnostics["prediction_types"]),
            "health": health,
        }

        return diagnostics

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
