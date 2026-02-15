"""
Life OS — Contact Type Classifier

Classifies contacts as "person" or "business" based on available signals.
Used to prioritize human contacts over businesses/automated senders in
predictions, notifications, and relationship maintenance.

Classification Signals:
    Strong "person" signals:
        - Has a phone number or iMessage channel
        - Has a personal relationship (spouse, friend, family, etc.)
        - Was seeded as a priority contact during onboarding
        - Has two-way communication (both inbound and outbound)
        - Name looks like a human name (first + last)

    Strong "business" signals:
        - Email-only contact with no phone/messaging channels
        - Generic local-part (info@, support@, noreply@, etc.)
        - Marketing/ESP subdomain patterns
        - One-way communication (only inbound, user never replies)
        - Name matches a company pattern ("Inc", "LLC", etc.)
"""

from __future__ import annotations

import json
import re
from typing import Optional


# Relationships that clearly indicate a person
PERSONAL_RELATIONSHIPS = {
    "spouse", "partner", "wife", "husband",
    "parent", "mother", "father", "mom", "dad",
    "sibling", "brother", "sister",
    "child", "son", "daughter",
    "family", "relative",
    "friend", "best friend", "close friend",
    "boyfriend", "girlfriend",
    "roommate", "housemate",
    "neighbor",
}

# Relationships that could be person or business
PROFESSIONAL_RELATIONSHIPS = {
    "boss", "manager", "supervisor",
    "coworker", "colleague", "teammate",
    "mentor", "mentee",
    "client", "customer",
    "doctor", "dentist", "therapist", "lawyer", "accountant",
}

# Generic local-parts that indicate a business/automated sender
BUSINESS_LOCALPARTS = (
    "info@", "support@", "help@", "hello@",
    "contact@", "admin@", "billing@", "sales@",
    "team@", "service@", "services@",
    "accounts@", "feedback@", "press@",
    "hr@", "careers@", "jobs@",
    "legal@", "compliance@", "privacy@",
    "office@", "reception@", "general@",
)

# Company name suffixes
COMPANY_SUFFIXES = re.compile(
    r'\b(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|GmbH|SA|SAS|PLC|LP|LLP|Group|'
    r'Holdings|Foundation|Association|Institute|University|Hospital|'
    r'Bank|Insurance|Airlines?|Airways?|Motors?|Pharma|Labs?|Studios?|'
    r'Technologies|Solutions|Consulting|Services|Systems|Software)\b',
    re.IGNORECASE,
)


def classify_contact_type(
    email_address: Optional[str] = None,
    name: Optional[str] = None,
    relationship: Optional[str] = None,
    channels: Optional[dict] = None,
    phones: Optional[list] = None,
    is_priority: bool = False,
    interaction_profile: Optional[dict] = None,
) -> str:
    """Classify a contact as "person" or "business".

    Uses a scoring approach: accumulate person/business signals and return
    the type with the stronger signal. Returns "person" when uncertain
    (fail-safe: it's better to over-prioritize than to ignore a human).

    Args:
        email_address: Primary email address of the contact.
        name: Contact display name.
        relationship: Relationship label (spouse, boss, friend, etc.).
        channels: Dict of communication channels (imessage, signal, slack, etc.).
        phones: List of phone numbers.
        is_priority: Whether the contact was marked priority.
        interaction_profile: Dict with interaction_count, inbound_count,
            outbound_count from the relationship signal profile.

    Returns:
        "person" or "business"
    """
    person_score = 0.0
    business_score = 0.0

    # --- Priority contacts are always people ---
    # These are explicitly set by the user during onboarding
    if is_priority:
        return "person"

    # --- Relationship signals ---
    if relationship:
        rel_lower = relationship.lower().strip()
        if rel_lower in PERSONAL_RELATIONSHIPS:
            return "person"  # Definitive: personal relationships are always people
        if rel_lower in PROFESSIONAL_RELATIONSHIPS:
            person_score += 0.6  # Strong person signal, but not definitive

    # --- Channel signals ---
    channels = channels or {}
    phones = phones or []

    # Phone numbers and messaging apps are strong person signals.
    # Businesses don't typically share personal phone numbers in
    # messaging contacts — this is one of the strongest person indicators.
    if phones:
        person_score += 0.9
    if any(ch in channels for ch in ("imessage", "signal", "whatsapp")):
        person_score += 0.8
    if "slack" in channels:
        person_score += 0.3  # Slack could be either, slight person lean

    # --- Email address signals ---
    if email_address:
        addr_lower = email_address.lower()

        # Generic local-parts are almost always businesses
        if any(addr_lower.startswith(p) for p in BUSINESS_LOCALPARTS):
            business_score += 0.8

        # Marketing/ESP subdomain patterns (info@mail.company.com)
        marketing_domains = (
            "@email.", "@mail.", "@reply.", "@bounce.",
            "@send.", "@mg.", "@em.", "@e.",
            "@news-", "@newsletters.", "@marketing.",
        )
        if any(p in addr_lower for p in marketing_domains):
            business_score += 0.7

        # No-reply patterns are definitively business/automated
        noreply = ("noreply@", "no-reply@", "donotreply@", "do-not-reply@")
        if any(p in addr_lower for p in noreply):
            return "business"

    # --- Name signals ---
    if name:
        # Company name patterns
        if COMPANY_SUFFIXES.search(name):
            business_score += 0.7

        # "Unknown (address)" pattern from auto-created contacts
        if name.startswith("Unknown ("):
            business_score += 0.2  # Slight lean — could be either

        # Human name heuristic: two+ capitalized words without company suffixes
        # e.g., "John Smith" or "Alice B. Cooper"
        name_parts = name.strip().split()
        if (
            2 <= len(name_parts) <= 4
            and all(p[0].isupper() for p in name_parts if p)
            and not COMPANY_SUFFIXES.search(name)
        ):
            person_score += 0.4

    # --- Interaction pattern signals ---
    if interaction_profile:
        outbound = interaction_profile.get("outbound_count", 0)
        inbound = interaction_profile.get("inbound_count", 0)
        total = interaction_profile.get("interaction_count", 0)

        # Two-way communication is a strong person signal
        if outbound > 0 and inbound > 0:
            person_score += 0.5

        # Purely one-way inbound with no replies → likely business
        if inbound > 3 and outbound == 0:
            business_score += 0.5

        # Multi-channel usage → likely person
        channels_used = interaction_profile.get("channels_used", [])
        if len(channels_used) >= 2:
            person_score += 0.3

    # --- Decision ---
    # Default to "person" when uncertain (fail-safe for prioritization)
    if business_score > person_score and business_score >= 0.5:
        return "business"
    return "person"


def classify_email_address(email_address: str) -> str:
    """Quick classification based on email address alone.

    Used when creating contacts from email events where we only have
    the sender address. Returns "person" or "business".
    """
    return classify_contact_type(email_address=email_address)
