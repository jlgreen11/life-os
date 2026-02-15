"""
Life OS — Relationship Signal Extractor

Builds the relationship graph — who matters, how they're connected,
and the dynamics of each relationship.
"""

from __future__ import annotations

from models.core import EventType
from services.signal_extractor.base import BaseExtractor


class RelationshipExtractor(BaseExtractor):
    """
    Builds the relationship graph — who matters, how they're connected,
    and the dynamics of each relationship.
    """

    def can_process(self, event: dict) -> bool:
        return event.get("type") in [
            EventType.EMAIL_RECEIVED.value,
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.MESSAGE_SENT.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        payload = event.get("payload", {})
        event_type = event.get("type", "")
        timestamp = event.get("timestamp", "")

        signals = []

        # Determine contact
        if "sent" in event_type.lower():
            addresses = payload.get("to_addresses", [])
        else:
            addresses = [payload.get("from_address")] if payload.get("from_address") else []

        for address in addresses:
            if not address:
                continue

            signal = {
                "type": "relationship_interaction",
                "timestamp": timestamp,
                "contact_address": address,
                "direction": "outbound" if "sent" in event_type.lower() else "inbound",
                "channel": payload.get("channel", event.get("source", "unknown")),
                "message_length": len(payload.get("body", "") or ""),
                "has_action_items": bool(payload.get("action_items")),
                "sentiment": payload.get("sentiment"),
                "is_reply": payload.get("is_reply", False),
            }
            signals.append(signal)

        self._update_contact_profiles(signals)
        return signals

    def _update_contact_profiles(self, signals: list[dict]):
        """Update contact frequency and interaction patterns."""
        existing = self.ums.get_signal_profile("relationships")
        data = existing["data"] if existing else {"contacts": {}}

        for signal in signals:
            addr = signal["contact_address"]
            if addr not in data["contacts"]:
                data["contacts"][addr] = {
                    "interaction_count": 0,
                    "inbound_count": 0,
                    "outbound_count": 0,
                    "channels_used": [],
                    "avg_message_length": 0,
                    "last_interaction": None,
                    "interaction_timestamps": [],
                }

            profile = data["contacts"][addr]
            profile["interaction_count"] += 1
            if signal["direction"] == "inbound":
                profile["inbound_count"] += 1
            else:
                profile["outbound_count"] += 1

            if signal["channel"] not in profile["channels_used"]:
                profile["channels_used"].append(signal["channel"])

            profile["last_interaction"] = signal["timestamp"]
            profile["interaction_timestamps"].append(signal["timestamp"])

            # Keep last 100 timestamps for frequency analysis
            if len(profile["interaction_timestamps"]) > 100:
                profile["interaction_timestamps"] = profile["interaction_timestamps"][-100:]

            # Running average message length
            n = profile["interaction_count"]
            profile["avg_message_length"] = (
                (profile["avg_message_length"] * (n - 1) + signal["message_length"]) / n
            )

        self.ums.update_signal_profile("relationships", data)
