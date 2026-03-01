"""
Life OS — Temporal Signal Extractor

Tracks the user's relationship with time — energy rhythms, productive hours,
weekly patterns, and how behavior changes throughout the day and week.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor

logger = logging.getLogger(__name__)


class TemporalExtractor(BaseExtractor):
    """
    Tracks the user's relationship with time — energy rhythms, productive hours,
    weekly patterns, and how behavior changes throughout the day and week.

    This extractor builds a TemporalProfile by analyzing:
    - Activity patterns by hour to detect energy peaks and troughs
    - Weekly rhythms to identify productive days vs. social days vs. recharge days
    - Event timing patterns (scheduling preferences, deadline behavior)

    The profile enables time-aware predictions like:
    - "Best time to schedule this meeting"
    - "You're most productive 2-4pm, blocking focus time"
    - "Energy dip detected, suggesting a break"
    """

    def can_process(self, event: dict) -> bool:
        """
        Process all user-initiated events that have timestamps.

        We track:
        - Communication events (email, messages) — shows active engagement
        - Calendar events — shows scheduled commitments and planning horizon
        - Task events — shows when work gets done
        - System commands — shows direct user interaction
        """
        return event.get("type") in [
            # User-initiated communication (outbound = active engagement)
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            # Calendar events (shows planning and commitments)
            EventType.CALENDAR_EVENT_CREATED.value,
            EventType.CALENDAR_EVENT_UPDATED.value,
            # Task activity (shows when work gets done)
            EventType.TASK_CREATED.value,
            EventType.TASK_COMPLETED.value,
            EventType.TASK_UPDATED.value,
            # Direct user commands (shows active engagement)
            EventType.USER_COMMAND.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        """
        Extract temporal signals from the event.

        Returns signal dicts that capture activity patterns, and updates the
        temporal profile as a side-effect.
        """
        timestamp = event.get("timestamp", "")
        event_type = event.get("type", "")

        signals = []

        try:
            # Normalize timestamp format (handle trailing Z)
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Determine activity type from event type
            activity_type = self._classify_activity(event_type, event.get("payload", {}))

            # Record the temporal activity signal
            signals.append({
                "type": "temporal_activity",
                "timestamp": timestamp,
                "hour": dt.hour,
                "day_of_week": dt.strftime("%A").lower(),
                "activity_type": activity_type,
                "event_type": event_type,
            })

            # For calendar events, also track the event's actual start time
            # to understand scheduling preferences (not just when the event was created)
            if event_type in [EventType.CALENDAR_EVENT_CREATED.value, EventType.CALENDAR_EVENT_UPDATED.value]:
                payload = event.get("payload", {})
                if payload.get("start_time"):
                    try:
                        event_dt = datetime.fromisoformat(payload["start_time"].replace("Z", "+00:00"))
                        signals.append({
                            "type": "temporal_scheduled_event",
                            "timestamp": timestamp,
                            "scheduled_hour": event_dt.hour,
                            "scheduled_day": event_dt.strftime("%A").lower(),
                            "advance_planning_days": (event_dt.date() - dt.date()).days if event_dt >= dt else 0,
                        })
                    except (ValueError, AttributeError):
                        pass  # Malformed start_time, skip

        except (ValueError, AttributeError) as e:
            # Malformed timestamp, skip temporal extraction for this event
            pass

        # Update the temporal profile with these signals
        if signals:
            self._update_profile(signals)

        return signals

    def _classify_activity(self, event_type: str, payload: dict) -> str:
        """
        Classify the activity type for better temporal pattern detection.

        Returns:
            "communication" — active engagement with others
            "planning" — calendar/task management
            "work" — task completion
            "command" — direct system interaction
        """
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
            return "communication"
        elif event_type in [EventType.CALENDAR_EVENT_CREATED.value, EventType.CALENDAR_EVENT_UPDATED.value,
                           EventType.TASK_CREATED.value]:
            return "planning"
        elif event_type in [EventType.TASK_COMPLETED.value, EventType.TASK_UPDATED.value]:
            return "work"
        elif event_type == EventType.USER_COMMAND.value:
            return "command"
        else:
            return "other"

    def _update_profile(self, signals: list[dict]) -> None:
        """
        Update the temporal profile with new signals.

        Aggregates activity patterns by hour and day of week to build:
        - energy_by_hour: Activity density per hour (proxy for engagement/energy)
        - productive_days / social_days / recharge_days: Weekly rhythm classification
        """
        # Load existing profile or initialize with empty structures
        existing = self.ums.get_signal_profile("temporal")
        data = existing["data"] if existing else {
            "activity_by_hour": {},  # {hour: count}
            "activity_by_day": {},   # {day: count}
            "activity_by_type": {},  # {type: count}
            "activity_by_day_and_type": {},  # {"monday:communication": count} — for day classification
            "scheduled_hours": {},   # {hour: count} — when events are scheduled
            "advance_planning_days": [],  # list of planning horizons
        }

        # Ensure activity_by_day_and_type exists for profiles created before this field was added
        if "activity_by_day_and_type" not in data:
            data["activity_by_day_and_type"] = {}

        # Process each signal
        for signal in signals:
            if signal["type"] == "temporal_activity":
                hour = str(signal["hour"])
                day = signal["day_of_week"]
                activity_type = signal["activity_type"]

                # Update hourly activity count (string keys for JSON serialization)
                if hour not in data["activity_by_hour"]:
                    data["activity_by_hour"][hour] = 0
                data["activity_by_hour"][hour] += 1

                # Update daily activity count
                if day not in data["activity_by_day"]:
                    data["activity_by_day"][day] = 0
                data["activity_by_day"][day] += 1

                # Update activity type count
                if activity_type not in data["activity_by_type"]:
                    data["activity_by_type"][activity_type] = 0
                data["activity_by_type"][activity_type] += 1

                # Update per-day-per-type count (for day classification)
                day_type_key = f"{day}:{activity_type}"
                if day_type_key not in data["activity_by_day_and_type"]:
                    data["activity_by_day_and_type"][day_type_key] = 0
                data["activity_by_day_and_type"][day_type_key] += 1

            elif signal["type"] == "temporal_scheduled_event":
                scheduled_hour = str(signal["scheduled_hour"])
                advance_days = signal["advance_planning_days"]

                # Track when events are scheduled (not when they're created)
                if scheduled_hour not in data["scheduled_hours"]:
                    data["scheduled_hours"][scheduled_hour] = 0
                data["scheduled_hours"][scheduled_hour] += 1

                # Track planning horizon (cap at last 1000 to prevent unbounded growth)
                # Only track events scheduled in the future (advance_days > 0)
                # Skip same-day events (advance_days = 0) and past events (negative)
                if advance_days > 0:
                    data["advance_planning_days"].append(advance_days)
                    if len(data["advance_planning_days"]) > 1000:
                        data["advance_planning_days"] = data["advance_planning_days"][-1000:]

        # Derive higher-level behavioral fields from the raw counters
        data = self._derive_behavioral_fields(data)

        # Persist the updated profile
        # Note: update_signal_profile expects just the data dict, and automatically
        # increments samples_count by 1 per call and updates the timestamp
        self.ums.update_signal_profile("temporal", data)

    def _derive_behavioral_fields(self, data: dict) -> dict:
        """
        Derive higher-level behavioral fields from raw activity counters.

        Computes chronotype, peak_hours, wake/sleep times, day classifications,
        and planning horizon from the raw counters already stored in the profile.
        All derivations are guarded by minimum sample thresholds to avoid noisy
        conclusions from sparse data.

        Args:
            data: The profile data dict containing raw counters.

        Returns:
            The mutated data dict with derived fields added.
        """
        activity_by_hour = data.get("activity_by_hour", {})
        total_activity = sum(activity_by_hour.values())

        # --- Chronotype (requires 50+ total activities) ---
        if total_activity >= 50:
            morning_activity = sum(
                count for hour, count in activity_by_hour.items()
                if 6 <= int(hour) <= 10
            )
            evening_activity = sum(
                count for hour, count in activity_by_hour.items()
                if 20 <= int(hour) <= 23
            )
            morning_ratio = morning_activity / total_activity
            evening_ratio = evening_activity / total_activity

            if morning_ratio > 0.3:
                data["chronotype"] = "early_bird"
            elif evening_ratio > 0.3:
                data["chronotype"] = "night_owl"
            else:
                data["chronotype"] = "variable"

        # --- Peak hours (requires 20+ total activities) ---
        if total_activity >= 20:
            # Sort hours by activity count descending, keep those with >= 5% of total
            threshold = total_activity * 0.05
            sorted_hours = sorted(
                activity_by_hour.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            peak_hours = [
                int(hour) for hour, count in sorted_hours
                if count >= threshold
            ][:3]
            if peak_hours:
                data["peak_hours"] = peak_hours

        # --- Typical wake / sleep hours (requires 20+ total activities) ---
        if total_activity >= 20:
            threshold_2pct = total_activity * 0.02
            active_hours = sorted(
                int(hour) for hour, count in activity_by_hour.items()
                if count > threshold_2pct
            )
            if active_hours:
                data["typical_wake_hour"] = active_hours[0]
                data["typical_sleep_hour"] = active_hours[-1]

        # --- Day classification (requires 20+ total activities) ---
        if total_activity >= 20:
            activity_by_day = data.get("activity_by_day", {})
            activity_by_day_and_type = data.get("activity_by_day_and_type", {})
            productive_days = []
            social_days = []
            recharge_days = []

            for day, day_total in activity_by_day.items():
                if day_total == 0:
                    recharge_days.append(day)
                    continue
                # Count work-oriented activity types for this day
                work_count = sum(
                    activity_by_day_and_type.get(f"{day}:{t}", 0)
                    for t in ("communication", "planning", "work")
                )
                work_ratio = work_count / day_total

                if work_ratio > 0.6:
                    productive_days.append(day)
                elif work_ratio < 0.3:
                    # Low work ratio with communication present => social
                    comm_count = activity_by_day_and_type.get(f"{day}:communication", 0)
                    if comm_count > 0:
                        social_days.append(day)
                    else:
                        recharge_days.append(day)
                else:
                    recharge_days.append(day)

            data["productive_days"] = productive_days
            data["social_days"] = social_days
            data["recharge_days"] = recharge_days

        # --- Median planning horizon (requires at least 3 data points) ---
        planning_days = data.get("advance_planning_days", [])
        if len(planning_days) >= 3:
            data["median_planning_horizon_days"] = float(statistics.median(planning_days))

        return data
