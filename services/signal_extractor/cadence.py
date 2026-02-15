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

        # For outbound messages, calculate response time if it's a reply
        if event_type in [EventType.EMAIL_SENT.value, EventType.MESSAGE_SENT.value]:
            if payload.get("is_reply") and payload.get("in_reply_to"):
                response_time = self._calculate_response_time(
                    payload["in_reply_to"], timestamp
                )
                if response_time is not None:
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

        # Track activity by hour and day
        try:
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
            pass

        self._update_profile(signals)
        return signals

    def _calculate_response_time(self, original_message_id: str,
                                  response_timestamp: str) -> Optional[float]:
        """Look up the original message and calculate response time."""
        # Search for the original message event
        # This is a simplified version; in production you'd index by message_id
        return None  # Placeholder — requires event lookup by payload.message_id

    def _update_profile(self, signals: list[dict]):
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
                data["response_times"].append(rt)

                contact = signal.get("contact_id")
                if contact:
                    if contact not in data["per_contact_response_times"]:
                        data["per_contact_response_times"][contact] = []
                    data["per_contact_response_times"][contact].append(rt)

                channel = signal.get("channel")
                if channel:
                    if channel not in data["per_channel_response_times"]:
                        data["per_channel_response_times"][channel] = []
                    data["per_channel_response_times"][channel].append(rt)

            elif signal["type"] == "cadence_activity":
                hour = str(signal["hour"])
                day = signal["day_of_week"]
                if hour not in data["hourly_activity"]:
                    data["hourly_activity"][hour] = 0
                data["hourly_activity"][hour] += 1
                if day not in data["daily_activity"]:
                    data["daily_activity"][day] = 0
                data["daily_activity"][day] += 1

        # Trim to last 1000 response times
        if len(data.get("response_times", [])) > 1000:
            data["response_times"] = data["response_times"][-1000:]

        self.ums.update_signal_profile("cadence", data)
