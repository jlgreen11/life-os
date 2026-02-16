"""
Life OS — Temporal Signal Extractor

Tracks the user's relationship with time — energy rhythms, productive hours,
weekly patterns, and how behavior changes throughout the day and week.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


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
            "scheduled_hours": {},   # {hour: count} — when events are scheduled
            "advance_planning_days": [],  # list of planning horizons
        }

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

        # Persist the updated profile
        # Note: update_signal_profile expects just the data dict, and automatically
        # increments samples_count by 1 per call and updates the timestamp
        self.ums.update_signal_profile("temporal", data)
