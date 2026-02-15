"""
Life OS — Onboarding System

Voice-first preference capture. Apple-inspired: one idea per screen,
no wrong answers, the product adapts to you in real time.

The onboarding is a guided conversation (8-12 minutes) that captures
the user's communication style, autonomy level, life structure,
privacy boundaries, and attention preferences.

All responses are distilled into the UserPreferences model via AI parsing.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, time, timezone
from typing import Any, Optional

from storage.database import DatabaseManager


# ---------------------------------------------------------------------------
# Onboarding Flow Definition
# ---------------------------------------------------------------------------

ONBOARDING_PHASES = [
    {
        "id": "welcome",
        "title": "Welcome",
        "prompt": (
            "Welcome to Life OS. I'm going to ask you a few questions so I can "
            "work the way you want me to. There are no wrong answers — you can "
            "change any of these later. Let's start."
        ),
        "type": "info",
    },

    # Phase 1: Communication Style
    {
        "id": "morning_style",
        "title": "Morning Personality",
        "prompt": (
            "When you wake up and check your phone, do you want just the "
            "essentials — like, three bullet points and go — or do you want "
            "the full picture of what's ahead?"
        ),
        "type": "choice",
        "options": [
            {"label": "Just the essentials", "value": "minimal"},
            {"label": "Give me the full picture", "value": "detailed"},
            {"label": "Somewhere in between", "value": "balanced"},
        ],
        "maps_to": "verbosity",
    },
    {
        "id": "tone",
        "title": "Tone",
        "prompt": (
            "How do you want me to talk to you? Some people like warm and "
            "encouraging. Others prefer straight to the point. And some want "
            "it casual, like talking to a friend."
        ),
        "type": "choice",
        "options": [
            {"label": "Warm and supportive", "value": "warm"},
            {"label": "Professional and direct", "value": "professional"},
            {"label": "Casual, like a friend", "value": "casual"},
        ],
        "maps_to": "tone",
    },
    {
        "id": "proactivity",
        "title": "Proactivity",
        "prompt": (
            "Should I bring things up on my own — like reminding you about "
            "someone's birthday, or suggesting you might want to follow up "
            "with someone? Or do you prefer I only speak when spoken to?"
        ),
        "type": "choice",
        "options": [
            {"label": "Yes, be proactive", "value": "high"},
            {"label": "Sometimes, but don't overdo it", "value": "moderate"},
            {"label": "Only when I ask", "value": "low"},
        ],
        "maps_to": "proactivity",
    },

    # Phase 2: Autonomy
    {
        "id": "autonomy",
        "title": "The Leash",
        "prompt": (
            "When things come in — emails, messages, calendar conflicts — "
            "how much should I handle on my own? I can ask you about everything, "
            "or I can take care of the obvious stuff and only bother you for "
            "judgment calls."
        ),
        "type": "choice",
        "options": [
            {"label": "Ask me about everything", "value": "supervised"},
            {"label": "Handle the obvious stuff", "value": "moderate"},
            {"label": "Handle as much as possible", "value": "high"},
        ],
        "maps_to": "autonomy_level",
    },
    {
        "id": "drafting",
        "title": "Message Drafting",
        "prompt": (
            "When someone messages you, should I draft a reply for you to "
            "review and send? Or do you prefer to write your own responses?"
        ),
        "type": "choice",
        "options": [
            {"label": "Draft replies for me", "value": True},
            {"label": "I'll write my own", "value": False},
        ],
        "maps_to": "draft_replies",
    },

    # Phase 3: Life Structure
    {
        "id": "domains",
        "title": "Life Domains",
        "prompt": (
            "Tell me about the big areas of your life. What are the buckets "
            "everything falls into? For example: work, family, a hobby, "
            "health, a side project. Just name them naturally."
        ),
        "type": "free_text",
        "maps_to": "life_domains",
    },
    {
        "id": "priority_people",
        "title": "Priority People",
        "prompt": (
            "Who are the 3-5 people whose messages should always get through, "
            "no matter what? Give me names and how they're connected to you."
        ),
        "type": "free_text",
        "maps_to": "priority_contacts",
    },

    # Phase 4: Privacy & Boundaries
    {
        "id": "work_life_boundary",
        "title": "Work/Life Wall",
        "prompt": (
            "How do you feel about work and personal mixing? Some people want "
            "a hard wall — work stuff never shows up during personal time. "
            "Others prefer everything blended."
        ),
        "type": "choice",
        "options": [
            {"label": "Hard wall — keep them separate", "value": "strict_separation"},
            {"label": "Soft wall — mostly separate but flexible", "value": "soft_separation"},
            {"label": "Blended — I don't separate them", "value": "unified"},
        ],
        "maps_to": "boundary_mode",
    },
    {
        "id": "vault",
        "title": "The Vault",
        "prompt": (
            "One more thing about privacy. You can create a private "
            "compartment — I call it 'The Vault.' Anything in the Vault is "
            "behind its own passcode. It won't show up in search, briefings, "
            "or the unified inbox. You never have to explain why something "
            "is in there. Want to set one up?"
        ),
        "type": "choice",
        "options": [
            {"label": "Yes, set up a Vault", "value": True},
            {"label": "Not right now", "value": False},
        ],
        "maps_to": "vault_enabled",
    },

    # Phase 5: Attention
    {
        "id": "notifications",
        "title": "Notification Philosophy",
        "prompt": (
            "Notifications. Some people want to protect their focus and only "
            "see what's truly urgent. Others want to stay in the loop on "
            "everything. What's your style?"
        ),
        "type": "choice",
        "options": [
            {"label": "Protect my focus — urgent only", "value": "minimal"},
            {"label": "Batch things up — digest 2-3 times a day", "value": "batched"},
            {"label": "Keep me in the loop — real-time", "value": "frequent"},
        ],
        "maps_to": "notification_mode",
    },
    {
        "id": "quiet_hours",
        "title": "Quiet Hours",
        "prompt": (
            "Is there a time of day when you want total silence? Only "
            "emergencies and your priority people would get through."
        ),
        "type": "free_text",
        "maps_to": "quiet_hours",
        "hint": "Example: '10pm to 7am' or 'I don't need quiet hours'",
    },

    # Phase 6: Close
    {
        "id": "close",
        "title": "You're All Set",
        "prompt": (
            "That's everything I need. Here's what I want you to know: "
            "your data lives on your hardware, not mine. You can export "
            "everything or delete it at any time. I work for you — literally. "
            "Your subscription is my only business model. Let's get started."
        ),
        "type": "info",
    },
]


# ---------------------------------------------------------------------------
# Onboarding Manager
# ---------------------------------------------------------------------------

class OnboardingManager:
    """Manages the onboarding flow and persists preferences."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._session: dict[str, Any] = {}

    def get_flow(self) -> list[dict]:
        """Return the full onboarding flow definition."""
        return ONBOARDING_PHASES

    def get_current_step(self) -> Optional[dict]:
        """Get the next unanswered step."""
        answered = set(self._session.keys())
        for phase in ONBOARDING_PHASES:
            if phase["type"] != "info" and phase["id"] not in answered:
                return phase
        return None

    def submit_answer(self, step_id: str, value: Any):
        """Store an answer for an onboarding step."""
        self._session[step_id] = value

    def is_complete(self) -> bool:
        """Check if all required questions have been answered."""
        required = [p["id"] for p in ONBOARDING_PHASES if p["type"] != "info"]
        return all(q in self._session for q in required)

    def finalize(self) -> dict[str, Any]:
        """
        Convert all answers into structured preferences and save.
        Returns the final preferences dict.
        """
        preferences = {}

        for phase in ONBOARDING_PHASES:
            if phase["type"] == "info":
                continue

            step_id = phase["id"]
            maps_to = phase.get("maps_to")
            value = self._session.get(step_id)

            if maps_to and value is not None:
                preferences[maps_to] = value

        # Process free-text fields
        if "life_domains" in preferences:
            preferences["life_domains"] = self._parse_domains(
                preferences["life_domains"]
            )

        if "priority_contacts" in preferences:
            preferences["priority_contacts"] = self._parse_contacts(
                preferences["priority_contacts"]
            )

        if "quiet_hours" in preferences:
            preferences["quiet_hours"] = self._parse_quiet_hours(
                preferences["quiet_hours"]
            )

        # Handle vault
        if preferences.get("vault_enabled"):
            preferences["vaults"] = [{"name": "Vault", "auth_method": "pin"}]
        preferences.pop("vault_enabled", None)

        # Save to database
        with self.db.get_connection("preferences") as conn:
            for key, value in preferences.items():
                serialized = json.dumps(value) if not isinstance(value, str) else value
                conn.execute(
                    """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                       VALUES (?, ?, 'onboarding', ?)""",
                    (key, serialized, datetime.now(timezone.utc).isoformat()),
                )

            # Mark onboarding as complete
            conn.execute(
                """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                   VALUES ('onboarding_completed', 'true', 'system', ?)""",
                (datetime.now(timezone.utc).isoformat(),),
            )

        return preferences

    def _parse_domains(self, text: str) -> list[dict]:
        """Parse free-text domain description into structured domains."""
        # Simple parsing — in production, use LLM
        domains = []
        for part in text.replace(",", "\n").split("\n"):
            part = part.strip().strip("-•*").strip()
            if part:
                domains.append({"name": part.lower(), "boundary": "soft_separation"})
        return domains if domains else [{"name": "personal"}, {"name": "work"}]

    def _parse_contacts(self, text: str) -> list[dict]:
        """Parse free-text contact list into structured contacts."""
        contacts = []
        for line in text.replace(",", "\n").split("\n"):
            line = line.strip().strip("-•*").strip()
            if line:
                # Try to extract name and relationship
                parts = line.split("-", 1) if "-" in line else line.split("(", 1)
                name = parts[0].strip()
                relationship = parts[1].strip().strip(")") if len(parts) > 1 else None
                contacts.append({"name": name, "relationship": relationship})
        return contacts

    def _parse_quiet_hours(self, text: str) -> list[dict]:
        """Parse free-text quiet hours into structured time ranges."""
        text = text.lower().strip()
        if "no" in text or "don't" in text or "none" in text:
            return []

        # Try to extract time range
        import re
        time_pattern = r'(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-)\s*(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)?'
        match = re.search(time_pattern, text)

        if match:
            start_h = int(match.group(1))
            start_m = int(match.group(2) or 0)
            start_ampm = match.group(3)
            end_h = int(match.group(4))
            end_m = int(match.group(5) or 0)
            end_ampm = match.group(6)

            if start_ampm == "pm" and start_h < 12:
                start_h += 12
            if end_ampm == "pm" and end_h < 12:
                end_h += 12
            if start_ampm == "am" and start_h == 12:
                start_h = 0
            if end_ampm == "am" and end_h == 12:
                end_h = 0

            return [{
                "start": f"{start_h:02d}:{start_m:02d}",
                "end": f"{end_h:02d}:{end_m:02d}",
                "days": ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"],
            }]

        # Default fallback
        return [{"start": "22:00", "end": "07:00",
                 "days": ["monday", "tuesday", "wednesday", "thursday",
                          "friday", "saturday", "sunday"]}]
