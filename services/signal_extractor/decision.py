"""
Life OS — Decision Signal Extractor

Tracks how the user makes decisions — speed, research depth, risk tolerance,
delegation patterns, and decision fatigue indicators.

This extractor builds a DecisionProfile by analyzing:
- Decision speed: Time from question/choice to action
- Research depth: Number of search queries or messages before deciding
- Delegation patterns: When they ask for opinions vs. decide alone
- Decision fatigue: Patterns of "whatever you decide" or increased delegation

The profile enables decision-aware predictions like:
- "You typically research tech purchases for 2-3 days before buying"
- "Decision fatigue detected (15+ decisions today), delegating this choice"
- "You always defer restaurant choices to your partner"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from models.core import EventType
from models.user_model import DecisionProfile
from services.signal_extractor.base import BaseExtractor


class DecisionExtractor(BaseExtractor):
    """
    Tracks how the user makes decisions — speed, research depth, risk tolerance,
    delegation patterns, and decision fatigue indicators.

    Decision-making patterns are inferred from:
    - Task creation → completion (decision speed for action items)
    - Question patterns in messages → action taken (delegation vs. self-decision)
    - Search/research activity before purchases or commitments
    - Time-of-day patterns in decisive language vs. deferring language
    """

    def can_process(self, event: dict) -> bool:
        """
        Process events that indicate decision-making activity.

        We track:
        - Task completion (decision execution)
        - Outbound messages with decision-related patterns
        - Calendar event creation (commitment decisions)
        - Email responses (decision communication)
        """
        return event.get("type") in [
            # Task events show decision execution
            EventType.TASK_COMPLETED.value,
            EventType.TASK_CREATED.value,
            # Outbound communication shows decision patterns
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            # Calendar commitments are decisions
            EventType.CALENDAR_EVENT_CREATED.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        """
        Extract decision-making signals from the event.

        Returns signal dicts and updates the DecisionProfile as a side-effect.
        """
        event_type = event.get("type", "")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", "")

        signals = []

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Track task completion speed (decision → execution time)
            if event_type == EventType.TASK_COMPLETED.value:
                decision_speed_signal = self._track_task_decision_speed(payload, dt)
                if decision_speed_signal:
                    signals.append(decision_speed_signal)

            # Detect delegation patterns in outbound messages
            if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
                delegation_signal = self._detect_delegation_patterns(payload, dt)
                if delegation_signal:
                    signals.append(delegation_signal)

            # Track calendar commitment speed (how far in advance they schedule)
            if event_type == EventType.CALENDAR_EVENT_CREATED.value:
                commitment_signal = self._track_commitment_patterns(payload, dt)
                if commitment_signal:
                    signals.append(commitment_signal)

            # Update the profile with aggregated signals
            self._update_profile(signals, dt)

        except Exception as e:
            # Fail-open: decision extraction should never block the pipeline
            logger.error("DecisionExtractor error: %s", e, exc_info=True)

        return signals

    def _track_task_decision_speed(self, payload: dict, completed_at: datetime) -> Optional[dict]:
        """
        Track how long from task creation to completion (decision → execution).

        This reveals decision speed by domain:
        - Quick tasks (<1 hour): impulsive/urgent
        - Same-day tasks (1-8 hours): normal pace
        - Multi-day tasks (>24 hours): deliberative
        """
        task_id = payload.get("task_id")
        if not task_id:
            return None

        # Query the database for task creation event
        query = """
            SELECT timestamp, json_extract(payload, '$.title') as title,
                   json_extract(payload, '$.domain') as domain
            FROM events
            WHERE type = ? AND json_extract(payload, '$.task_id') = ?
            ORDER BY timestamp ASC
            LIMIT 1
        """

        with self.db.get_connection("events") as conn:
            row = conn.execute(query, (EventType.TASK_CREATED.value, task_id)).fetchone()

        if not row:
            return None

        created_at = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        title = row[1] or ""
        domain = row[2] or "general"

        # Calculate decision speed (seconds from creation to completion)
        decision_time_seconds = (completed_at - created_at).total_seconds()

        # Classify decision speed
        if decision_time_seconds < 3600:
            speed_category = "immediate"
        elif decision_time_seconds < 28800:  # 8 hours
            speed_category = "same_day"
        elif decision_time_seconds < 86400:  # 24 hours
            speed_category = "next_day"
        else:
            speed_category = "multi_day"

        return {
            "type": "decision_speed",
            "domain": domain,
            "decision_time_seconds": decision_time_seconds,
            "speed_category": speed_category,
            "task_title": title,
            "timestamp": completed_at.isoformat(),
        }

    def _detect_delegation_patterns(self, payload: dict, sent_at: datetime) -> Optional[dict]:
        """
        Detect delegation patterns in outbound messages.

        Looks for:
        - Asking for opinions: "what do you think?", "should I...?"
        - Deferring decisions: "you decide", "whatever you prefer"
        - Seeking consensus: "does that work for you?"
        """
        content = payload.get("content", "") or payload.get("body", "")
        if not content:
            return None

        content_lower = content.lower()

        # Delegation indicators
        delegation_patterns = [
            "you decide", "you choose", "you pick", "up to you",
            "whatever you prefer", "whatever you want", "your call",
            "i don't care", "doesn't matter to me", "either way",
        ]

        # Opinion-seeking patterns (soft delegation)
        opinion_patterns = [
            "what do you think", "what's your opinion", "should i",
            "would you", "do you think i should", "any thoughts",
            "what would you do", "advice", "suggestions",
        ]

        is_delegating = any(pattern in content_lower for pattern in delegation_patterns)
        is_seeking_opinion = any(pattern in content_lower for pattern in opinion_patterns)

        if not (is_delegating or is_seeking_opinion):
            return None

        # Extract recipient for per-contact delegation tracking
        recipient = payload.get("to", "") or payload.get("recipient", "")

        delegation_type = "full" if is_delegating else "seeking_input"

        return {
            "type": "delegation_pattern",
            "delegation_type": delegation_type,
            "recipient": recipient,
            "hour": sent_at.hour,
            "timestamp": sent_at.isoformat(),
        }

    def _track_commitment_patterns(self, payload: dict, created_at: datetime) -> Optional[dict]:
        """
        Track calendar commitment patterns (planning horizon and spontaneity).

        Reveals risk tolerance and planning depth:
        - Same-day scheduling: spontaneous/reactive
        - Week-ahead scheduling: normal planning
        - Month+ scheduling: deliberative/cautious
        """
        start_time_str = payload.get("start_time", "")
        if not start_time_str:
            return None

        try:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

        # Calculate planning horizon (how far in advance they scheduled)
        planning_horizon_seconds = (start_time - created_at).total_seconds()

        # Ignore past events (likely imported or synced)
        if planning_horizon_seconds < 0:
            return None

        # Classify planning horizon
        if planning_horizon_seconds < 3600:  # < 1 hour
            horizon_category = "immediate"
        elif planning_horizon_seconds < 86400:  # < 1 day
            horizon_category = "same_day"
        elif planning_horizon_seconds < 604800:  # < 1 week
            horizon_category = "week_ahead"
        else:
            horizon_category = "long_term"

        summary = payload.get("summary", "") or payload.get("title", "")
        domain = self._classify_event_domain(summary)

        return {
            "type": "commitment_pattern",
            "domain": domain,
            "planning_horizon_seconds": planning_horizon_seconds,
            "horizon_category": horizon_category,
            "event_summary": summary,
            "timestamp": created_at.isoformat(),
        }

    def _classify_event_domain(self, summary: str) -> str:
        """
        Classify calendar event into decision domain.

        Domains help track risk tolerance by area:
        - work: professional commitments
        - social: personal relationships
        - health: medical, fitness
        - finance: money-related
        """
        summary_lower = summary.lower()

        # Simple keyword-based classification
        # Check specific domains first (finance, health) before generic work terms
        if any(word in summary_lower for word in ["bank", "payment", "tax", "budget"]):
            return "finance"
        elif any(word in summary_lower for word in ["workout", "gym", "doctor", "dentist", "therapy"]):
            return "health"
        elif any(word in summary_lower for word in ["dinner", "lunch", "coffee", "hangout", "party"]):
            return "social"
        elif any(word in summary_lower for word in ["meeting", "standup", "call", "sync", "review"]):
            return "work"
        else:
            return "general"

    def _update_profile(self, signals: list[dict], timestamp: datetime):
        """
        Update the DecisionProfile with aggregated signals.

        Incrementally builds the profile by:
        1. Loading existing profile from the database
        2. Merging new signals into domain-specific metrics
        3. Persisting updated profile back to signal_profiles table
        """
        if not signals:
            return

        # Load existing profile
        profile_data = self.ums.get_signal_profile("decision")
        if profile_data and profile_data.get("data"):
            profile_dict = profile_data["data"]
        else:
            # Initialize empty profile
            profile_dict = DecisionProfile().model_dump()

        # Aggregate decision speed signals
        for signal in signals:
            if signal["type"] == "decision_speed":
                domain = signal["domain"]
                decision_time = signal["decision_time_seconds"]

                # Update decision_speed_by_domain with exponential moving average
                current_speeds = profile_dict.get("decision_speed_by_domain", {})
                if domain in current_speeds:
                    # EMA: 0.3 weight on new observation
                    current_speeds[domain] = 0.7 * current_speeds[domain] + 0.3 * decision_time
                else:
                    current_speeds[domain] = decision_time
                profile_dict["decision_speed_by_domain"] = current_speeds

            elif signal["type"] == "delegation_pattern":
                recipient = signal["recipient"]
                delegation_type = signal["delegation_type"]
                hour = signal["hour"]

                # Track delegation comfort (ratio of delegating vs. deciding alone)
                # For now, just increment a counter; full ratio calculation needs
                # total message volume which we'd compute in a separate aggregation pass

                # Track decision fatigue by hour (late hours = more delegation)
                if hour >= 20:  # After 8pm
                    # Detect fatigue patterns
                    current_fatigue_hour = profile_dict.get("fatigue_time_of_day")
                    if current_fatigue_hour is None or hour < current_fatigue_hour:
                        profile_dict["fatigue_time_of_day"] = hour

            elif signal["type"] == "commitment_pattern":
                domain = signal["domain"]
                planning_horizon = signal["planning_horizon_seconds"]

                # Short planning horizons = higher risk tolerance (spontaneity)
                # Long planning horizons = lower risk tolerance (cautious)
                # Map planning horizon to risk tolerance: shorter = higher risk
                risk_score = 1.0 - min(planning_horizon / 2592000, 1.0)  # 30 days = 0.0 risk

                current_risk = profile_dict.get("risk_tolerance_by_domain", {})
                if domain in current_risk:
                    # EMA: 0.3 weight on new observation
                    current_risk[domain] = 0.7 * current_risk[domain] + 0.3 * risk_score
                else:
                    current_risk[domain] = risk_score
                profile_dict["risk_tolerance_by_domain"] = current_risk

        # Update timestamp
        profile_dict["last_updated"] = timestamp.isoformat()

        # Persist updated profile
        self.ums.update_signal_profile("decision", profile_dict)
