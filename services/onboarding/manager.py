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

# Each phase is a single screen in the onboarding flow, inspired by Apple's
# one-idea-per-screen pattern. Phases are ordered to build trust progressively:
#   1. Welcome        (info)  — set expectations, no input needed
#   2. Communication  (choice) — morning style, tone, proactivity
#   3. Autonomy       (choice) — how much the AI should do on its own
#   4. Life Structure  (free_text) — domains and priority people
#   5. Privacy         (choice) — work/life boundary, Vault setup
#   6. Attention       (choice + free_text) — notification mode, quiet hours
#   7. Close           (info)  — confirmation, privacy reassurance
#
# "type" determines the UI component:
#   "info"      -> read-only screen, no user input
#   "choice"    -> single-select buttons
#   "free_text" -> open text input (parsed by _parse_* helpers)
#
# "maps_to" links each answer to a key in the UserPreferences model.
ONBOARDING_PHASES = [
    # --- Phase 0: Welcome (info-only, no input captured) ---
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

    # --- Phase 1: Communication Style ---
    # These three screens capture how the AI should talk to the user.
    # The values ("minimal"/"detailed", "warm"/"professional"/"casual", etc.)
    # directly map to UserPreferences fields used by the AI engine.
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

    # --- Phase 2: Autonomy ---
    # Controls the "leash length" — how much the AI can do without asking.
    # "supervised" = ask about everything, "high" = handle as much as possible.
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

    # --- Phase 3: Life Structure ---
    # Free-text inputs that are parsed into structured data by the
    # _parse_domains and _parse_contacts helpers below.
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

    # --- Phase 4: Privacy & Boundaries ---
    # The boundary_mode controls whether work events bleed into personal
    # time and vice versa. The Vault is an optional encrypted compartment
    # for sensitive data that won't appear in search or briefings.
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

    # --- Phase 5: Attention ---
    # Notification mode + quiet hours together define when and how
    # the system is allowed to interrupt the user.
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

    # --- Phase 6: Close (info-only, no input captured) ---
    # Reassures the user about data ownership and privacy before starting.
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
        # In-memory session holding step_id -> answer pairs.
        # This is ephemeral — it only lives for the duration of the
        # onboarding conversation. finalize() flushes it to the DB.
        self._session: dict[str, Any] = {}

    def get_flow(self) -> list[dict]:
        """Return the full onboarding flow definition."""
        return ONBOARDING_PHASES

    def get_current_step(self) -> Optional[dict]:
        """
        Get the next unanswered step.

        Iterates through the phases in order and returns the first one
        that (a) requires user input (type != "info") and (b) hasn't
        been answered yet. Returns None when all steps are complete.
        """
        answered = set(self._session.keys())
        for phase in ONBOARDING_PHASES:
            if phase["type"] != "info" and phase["id"] not in answered:
                return phase
        return None

    def submit_answer(self, step_id: str, value: Any):
        """Store an answer for an onboarding step."""
        self._session[step_id] = value

    def is_complete(self) -> bool:
        """
        Check if all required questions have been answered.

        Only non-info phases require answers; info screens are display-only.
        """
        required = [p["id"] for p in ONBOARDING_PHASES if p["type"] != "info"]
        return all(q in self._session for q in required)

    def finalize(self) -> dict[str, Any]:
        """
        Convert all answers into structured preferences and save.
        Returns the final preferences dict.
        """
        preferences = {}

        # --- Step 1: Map raw answers to preference keys ---
        # Each phase's "maps_to" field names the UserPreferences key.
        # Choice answers are already structured (e.g., "minimal", True);
        # free-text answers need further parsing (see Step 2).
        for phase in ONBOARDING_PHASES:
            if phase["type"] == "info":
                continue

            step_id = phase["id"]
            maps_to = phase.get("maps_to")
            value = self._session.get(step_id)

            if maps_to and value is not None:
                preferences[maps_to] = value

        # --- Step 2: Parse free-text fields into structured data ---
        # Free-text responses are natural language; the _parse_* helpers
        # convert them into lists/dicts the rest of the system can use.
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

        # --- Step 3: Handle vault setup ---
        # Convert the boolean "vault_enabled" answer into a vaults config
        # object, then remove the transient key so it's not persisted as-is.
        if preferences.get("vault_enabled"):
            preferences["vaults"] = [{"name": "Vault", "auth_method": "pin"}]
        preferences.pop("vault_enabled", None)

        # --- Step 4: Persist all preferences to the database ---
        # Each preference is stored as a key-value pair. Non-string values
        # are JSON-serialized. "set_by" is tagged as "onboarding" so the
        # system can distinguish onboarding defaults from later user edits.
        with self.db.get_connection("preferences") as conn:
            for key, value in preferences.items():
                serialized = json.dumps(value) if not isinstance(value, str) else value
                conn.execute(
                    """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                       VALUES (?, ?, 'onboarding', ?)""",
                    (key, serialized, datetime.now(timezone.utc).isoformat()),
                )

            # Mark onboarding as complete so the app knows to skip the
            # onboarding flow on subsequent launches.
            conn.execute(
                """INSERT OR REPLACE INTO user_preferences (key, value, set_by, updated_at)
                   VALUES ('onboarding_completed', 'true', 'system', ?)""",
                (datetime.now(timezone.utc).isoformat(),),
            )

        return preferences

    def _parse_domains(self, text: str) -> list[dict]:
        """
        Parse free-text domain description into structured domains.

        Accepts comma-separated, newline-separated, or bullet-pointed
        lists (e.g., "work, family, health" or "- work\n- family").
        Each domain gets a default "soft_separation" boundary mode.
        Falls back to ["personal", "work"] if parsing yields nothing.
        """
        # Simple parsing — in production, use LLM for more robust extraction
        domains = []
        # Normalize commas to newlines, then split and strip list markers
        for part in text.replace(",", "\n").split("\n"):
            part = part.strip().strip("-\u2022*").strip()
            if part:
                domains.append({"name": part.lower(), "boundary": "soft_separation"})
        return domains if domains else [{"name": "personal"}, {"name": "work"}]

    def _parse_contacts(self, text: str) -> list[dict]:
        """
        Parse free-text contact list into structured contacts.

        Supports formats like:
            "Sarah - wife, Tom - coworker, Mom"
            "Sarah (wife)\nTom (coworker)\nMom"
        The relationship is optional; if absent, it's stored as None.
        """
        contacts = []
        for line in text.replace(",", "\n").split("\n"):
            line = line.strip().strip("-\u2022*").strip()
            if line:
                # Try to extract name and relationship from "Name - role"
                # or "Name (role)" format
                parts = line.split("-", 1) if "-" in line else line.split("(", 1)
                name = parts[0].strip()
                relationship = parts[1].strip().strip(")") if len(parts) > 1 else None
                contacts.append({"name": name, "relationship": relationship})
        return contacts

    def _parse_quiet_hours(self, text: str) -> list[dict]:
        """
        Parse free-text quiet hours into structured time ranges.

        Handles inputs like "10pm to 7am", "22:00 - 07:00", or
        "I don't need quiet hours". Returns a list of time-range dicts
        compatible with the notification manager's quiet hours format.
        """
        text = text.lower().strip()

        # If the user declines quiet hours, return an empty list
        if "no" in text or "don't" in text or "none" in text:
            return []

        # --- Time regex pattern ---
        # Captures a time range in many natural formats:
        #   Group 1: start hour   (required, 1-2 digits)
        #   Group 2: start minutes (optional, after colon)
        #   Group 3: start am/pm   (optional)
        #   Group 4: end hour     (required, 1-2 digits)
        #   Group 5: end minutes  (optional, after colon)
        #   Group 6: end am/pm    (optional)
        # The separator between start and end can be "to" or "-".
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

            # Convert 12-hour to 24-hour format when am/pm is present
            if start_ampm == "pm" and start_h < 12:
                start_h += 12
            if end_ampm == "pm" and end_h < 12:
                end_h += 12
            if start_ampm == "am" and start_h == 12:
                start_h = 0   # 12am = midnight = 00:00
            if end_ampm == "am" and end_h == 12:
                end_h = 0     # 12am = midnight = 00:00

            # Default to every day of the week. The user can customize
            # per-day ranges later through the settings UI.
            return [{
                "start": f"{start_h:02d}:{start_m:02d}",
                "end": f"{end_h:02d}:{end_m:02d}",
                "days": ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"],
            }]

        # Default fallback: if the user said something like "yes" or
        # "evening" without specific times, use a sensible default.
        return [{"start": "22:00", "end": "07:00",
                 "days": ["monday", "tuesday", "wednesday", "thursday",
                          "friday", "saturday", "sunday"]}]
