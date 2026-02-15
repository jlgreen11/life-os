"""
Life OS — Cadence Signal Extractor

Tracks when and how quickly the user communicates.
Reveals priorities, avoidance patterns, and natural rhythms.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


class CadenceExtractor(BaseExtractor):
    """
    Tracks when and how quickly the user communicates.
    Reveals priorities, avoidance patterns, and natural rhythms.
    """

    def can_process(self, event: dict) -> bool:
        # Cadence analysis applies to all communication events (both directions)
        # because we need inbound events to anchor response-time calculations
        # and outbound events to measure the user's actual reply latency.
        return event.get("type") in [
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_SENT.value,
            EventType.EMAIL_RECEIVED.value,
            EventType.MESSAGE_RECEIVED.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", "")
        event_type = event.get("type", "")
        source = event.get("source", "")

        signals = []

        # ----- Response-time tracking -----
        # Only outbound (user-authored) replies produce a response-time signal.
        # We look up the original inbound message by ID to compute the delta.
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
            if payload.get("is_reply") and payload.get("in_reply_to"):
                response_time = self._calculate_response_time(
                    payload["in_reply_to"], timestamp
                )
                if response_time is not None:
                    # Grab the first recipient as the contact identifier for
                    # per-contact response-time breakdowns.
                    contact = (
                        payload.get("to_addresses", [None])[0]
                        if payload.get("to_addresses")
                        else None
                    )
                    signals.append({
                        "type": "cadence_response_time",
                        "timestamp": timestamp,
                        "contact_id": contact,
                        "channel": source,
                        "response_time_seconds": response_time,
                    })

        # ----- Activity-window detection -----
        # Record the hour-of-day and day-of-week for every communication event
        # (both inbound and outbound).  Over time this builds a heatmap of the
        # user's natural activity windows — e.g., "most active 9-11am on weekdays".
        try:
            # Normalise the trailing "Z" to a proper UTC offset so fromisoformat
            # can parse it consistently across Python versions.
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            signals.append({
                "type": "cadence_activity",
                "timestamp": timestamp,
                "hour": dt.hour,
                "day_of_week": dt.strftime("%A").lower(),
                "direction": "outbound" if "sent" in event_type.lower() else "inbound",
                "channel": source,
            })
        except (ValueError, AttributeError):
            # If the timestamp is missing or malformed we silently skip the
            # activity signal rather than failing the whole extraction.
            pass

        # Persist the signals into the running cadence profile.
        self._update_profile(signals)
        return signals

    def _calculate_response_time(self, original_message_id: str,
                                  response_timestamp: str) -> Optional[float]:
        """Look up the original message and calculate response time.

        In a full implementation this would query the event store for the
        inbound message matching ``original_message_id``, extract its
        timestamp, and return the delta in seconds.  The result reveals how
        quickly the user replies to specific contacts or channels — a strong
        signal of priority and engagement.
        """
        # Placeholder — requires an event-store index keyed by payload.message_id.
        return None

    def _update_profile(self, signals: list[dict]):
        """Incrementally merge new signals into the persisted cadence profile.

        The profile stores four running aggregates:
          - response_times:             global list (capped at 1000 entries)
          - per_contact_response_times: response times bucketed by contact
          - per_channel_response_times: response times bucketed by channel
          - hourly_activity / daily_activity: histogram counters for the
            activity-window heatmap
        """
        # Load the existing profile or bootstrap with empty structures.
        existing = self.ums.get_signal_profile("cadence")
        data = existing["data"] if existing else {
            "response_times": [],
            "hourly_activity": defaultdict(int),
            "daily_activity": defaultdict(int),
            "per_contact_response_times": defaultdict(list),
            "per_channel_response_times": defaultdict(list),
        }

        for signal in signals:
            if signal["type"] == "cadence_response_time":
                rt = signal["response_time_seconds"]
                # Append to the global response-time list for overall statistics.
                data["response_times"].append(rt)

                # Also bucket by contact so we can compare how fast the user
                # replies to different people (priority signal).
                contact = signal.get("contact_id")
                if contact:
                    if contact not in data["per_contact_response_times"]:
                        data["per_contact_response_times"][contact] = []
                    data["per_contact_response_times"][contact].append(rt)

                # Bucket by channel (email vs. Slack vs. SMS) to detect
                # channel-specific habits — e.g., user replies to Slack in
                # minutes but to email in hours.
                channel = signal.get("channel")
                if channel:
                    if channel not in data["per_channel_response_times"]:
                        data["per_channel_response_times"][channel] = []
                    data["per_channel_response_times"][channel].append(rt)

            elif signal["type"] == "cadence_activity":
                # Increment histogram counters. These are stored as string keys
                # ("0"-"23" for hours, "monday"-"sunday" for days) to stay
                # JSON-serialisable.
                hour = str(signal["hour"])
                day = signal["day_of_week"]
                if hour not in data["hourly_activity"]:
                    data["hourly_activity"][hour] = 0
                data["hourly_activity"][hour] += 1
                if day not in data["daily_activity"]:
                    data["daily_activity"][day] = 0
                data["daily_activity"][day] += 1

        # Cap the global response-time list to prevent unbounded growth.
        # Keeping the most recent 1000 entries provides enough data for
        # statistical baselines while bounding storage.
        if len(data.get("response_times", [])) > 1000:
            data["response_times"] = data["response_times"][-1000:]

        self.ums.update_signal_profile("cadence", data)
