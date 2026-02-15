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
        # Relationship mapping requires both directions: inbound messages tell
        # us who is reaching out, outbound messages tell us who the user
        # chooses to engage with.  Both are needed to compute reciprocity and
        # directional interaction ratios.
        return event.get("type") in [
            EventType.EMAIL_RECEIVED.value,
            EventType.EMAIL_SENT.value,
            EventType.MESSAGE_RECEIVED.value,
            EventType.MESSAGE_SENT.value,
        ]

    def extract(self, event: dict) -> list[dict]:
        """Extract per-contact interaction signals from a communication event.

        For outbound events the contacts are the recipients (to_addresses);
        for inbound events the contact is the sender (from_address).  A
        separate signal is emitted for each contact so multi-recipient emails
        correctly update every contact's profile.
        """
        payload = event.get("payload", {})
        event_type = event.get("type", "")
        timestamp = event.get("timestamp", "")

        signals = []

        # Resolve the contact address(es) based on message direction.
        # Outbound: the user wrote to these people.
        # Inbound: someone wrote to the user.
        if "sent" in event_type.lower():
            addresses = payload.get("to_addresses", [])
        else:
            addresses = [payload.get("from_address")] if payload.get("from_address") else []

        for address in addresses:
            if not address:
                continue

            # Each signal captures a single interaction data point.  Downstream
            # aggregation in _update_contact_profiles turns these into running
            # statistics per contact.
            signal = {
                "type": "relationship_interaction",
                "timestamp": timestamp,
                "contact_address": address,
                "direction": "outbound" if "sent" in event_type.lower() else "inbound",
                "channel": payload.get("channel", event.get("source", "unknown")),
                # Message length as a proxy for investment in the conversation.
                "message_length": len(payload.get("body", "") or ""),
                # Whether the message contains action items — indicates a
                # task-oriented (potentially professional) relationship.
                "has_action_items": bool(payload.get("action_items")),
                "sentiment": payload.get("sentiment"),
                "is_reply": payload.get("is_reply", False),
            }
            signals.append(signal)

        # Merge these interaction signals into persisted per-contact profiles.
        self._update_contact_profiles(signals)
        return signals

    def _update_contact_profiles(self, signals: list[dict]):
        """Update contact frequency and interaction patterns.

        Each contact gets a profile dict that tracks:
          - interaction_count / inbound_count / outbound_count:
                Total and directional tallies.  The inbound/outbound ratio
                reveals reciprocity — e.g., a contact who always initiates
                but rarely receives replies may be deprioritised by the user.
          - channels_used:
                Which communication channels are active with this contact
                (email, Slack, SMS, etc.).  Multi-channel contacts tend to be
                closer relationships.
          - avg_message_length:
                Running average updated incrementally.  Longer messages
                correlate with deeper engagement.
          - interaction_timestamps:
                A ring buffer (last 100) used to compute interaction frequency,
                detect dormant relationships, and spot contact-frequency changes.
        """
        existing = self.ums.get_signal_profile("relationships")
        data = existing["data"] if existing else {"contacts": {}}

        for signal in signals:
            addr = signal["contact_address"]
            # Bootstrap a new profile for contacts we have not seen before.
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
            # Track direction so we can compute reciprocity ratios later.
            if signal["direction"] == "inbound":
                profile["inbound_count"] += 1
            else:
                profile["outbound_count"] += 1

            # Record new channels — deduplicated to keep the list compact.
            if signal["channel"] not in profile["channels_used"]:
                profile["channels_used"].append(signal["channel"])

            profile["last_interaction"] = signal["timestamp"]
            profile["interaction_timestamps"].append(signal["timestamp"])

            # Cap timestamp history at 100 entries to bound storage while
            # retaining enough data points for meaningful frequency analysis.
            if len(profile["interaction_timestamps"]) > 100:
                profile["interaction_timestamps"] = profile["interaction_timestamps"][-100:]

            # Incremental running average for message length.  Uses the
            # formula: new_avg = ((old_avg * (n-1)) + new_value) / n
            # to avoid storing all historical message lengths.
            n = profile["interaction_count"]
            profile["avg_message_length"] = (
                (profile["avg_message_length"] * (n - 1) + signal["message_length"]) / n
            )

        self.ums.update_signal_profile("relationships", data)
