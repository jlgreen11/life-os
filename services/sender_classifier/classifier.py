"""
Life OS — Sender Classifier

Classifies communication senders into three categories that drive
notification routing:

  HUMAN                  — Real person-to-person interaction → real-time
  COMPANY_TRANSACTIONAL  — Bills, receipts, account alerts  → weekly digest
  COMPANY_MARKETING      — Newsletters, promos, marketing   → monthly digest

The classifier builds on the existing marketing_filter module and adds
a transactional detection layer for business-to-consumer operational
emails (bills, shipping, account alerts, payment confirmations).

Classification is cached per-address in the contacts table so repeat
lookups are O(1) after the first encounter.
"""

from __future__ import annotations

import logging
import re

from models.core import SenderType
from services.signal_extractor.marketing_filter import is_marketing_or_noreply
from storage.manager import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transactional (B2C operational) detection patterns
#
# These identify business senders that aren't marketing but also aren't human.
# Bills, payment confirmations, shipping updates, account security alerts, etc.
# The user wants to see these ~weekly to stay on top of obligations.
# ---------------------------------------------------------------------------

# Local-part prefixes that indicate transactional/operational senders
_TRANSACTIONAL_LOCALPARTS = (
    # Billing & payments
    "billing@",
    "invoice@",
    "invoices@",
    "payment@",
    "payments@",
    "pay@",
    "statement@",
    "statements@",
    "collections@",
    "collection@",
    # Account management
    "security@",
    "verify@",
    "verification@",
    "password@",
    "passwordreset@",
    "password-reset@",
    "login@",
    "signin@",
    "sign-in@",
    "registration@",
    "register@",
    "activate@",
    "activation@",
    # Shipping & delivery (not already caught by marketing filter)
    "ship@",
    "shipments@",
    "dispatch@",
    # Insurance & financial services
    "claims@",
    "claim@",
    "policy@",
    "policies@",
    "premium@",
    "underwriting@",
    # Healthcare
    "appointments@",
    "appointment@",
    "scheduling@",
    "schedule@",
    "pharmacy@",
    "prescriptions@",
    "lab@",
    "labs@",
    "results@",
    # Utilities
    "utility@",
    "utilities@",
    "power@",
    "electric@",
    "electricity@",
    "water@",
    "gas@",
    "internet@",
    "broadband@",
    # Government / tax
    "tax@",
    "taxes@",
    "permit@",
    "permits@",
    "license@",
    "renewal@",
    "renewals@",
    # Legal
    "legal@",
    "compliance@",
    "documents@",
    "document@",
)

# Subject-line patterns that indicate transactional content
_TRANSACTIONAL_SUBJECT_PATTERNS = (
    # Bills & payments
    r"\b(?:your\s+)?(?:bill|invoice|statement|payment)\s+(?:is\s+)?(?:due|ready|available|past\s+due)\b",
    r"\b(?:payment\s+)?(?:confirm|receipt|received|processed|successful)\b",
    r"\bauto[- ]?pay\b",
    r"\bpay(?:ment)?\s+reminder\b",
    # Account / security
    r"\b(?:password|account)\s+(?:reset|verification|confirm|security|alert)\b",
    r"\btwo[- ]?factor\b",
    r"\bverify\s+your\s+(?:email|account|identity)\b",
    r"\bunusual\s+(?:activity|sign[- ]?in|login)\b",
    # Shipping
    r"\b(?:your\s+)?(?:order|package|shipment)\s+(?:has\s+)?(?:shipped|delivered|out\s+for\s+delivery)\b",
    r"\btracking\s+(?:number|update|info)\b",
    # Appointments
    r"\b(?:appointment|booking)\s+(?:confirmation|reminder|scheduled)\b",
    # Renewals
    r"\b(?:subscription|membership|policy|license)\s+(?:renewal|expir|cancel)\b",
)

# Compiled subject patterns for performance
_TRANSACTIONAL_SUBJECT_RE = [re.compile(p, re.IGNORECASE) for p in _TRANSACTIONAL_SUBJECT_PATTERNS]

# Channels that are always human-to-human (never company senders)
_ALWAYS_HUMAN_CHANNELS = frozenset(
    {
        "signal",
        "whatsapp",
        "imessage",
        "sms",
        "discord",
        "slack",
    }
)

# Event sources that correspond to always-human channels
_ALWAYS_HUMAN_SOURCES = frozenset(
    {
        "signal",
        "whatsapp",
        "imessage",
        "sms",
        "discord",
        "slack",
    }
)


class SenderClassifier:
    """Classifies communication senders as human, transactional, or marketing.

    Uses a layered detection strategy:
      1. Channel shortcut: Signal/iMessage/WhatsApp/SMS → always HUMAN
      2. Cache lookup: check if we've classified this address before
      3. Marketing filter: existing is_marketing_or_noreply() → COMPANY_MARKETING
      4. Transactional detection: pattern matching on address + subject → COMPANY_TRANSACTIONAL
      5. Default: HUMAN (fail-open — when unsure, treat as personal)

    Classifications are persisted to the contacts table for fast repeat lookups.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        # In-memory LRU cache for hot-path performance.
        # Maps normalized email address → SenderType.
        self._cache: dict[str, SenderType] = {}
        self._cache_max = 5000

    def classify_event(self, event: dict) -> SenderType:
        """Classify the sender of a communication event.

        Args:
            event: A Life OS event dict with type, source, payload.

        Returns:
            SenderType for the sender of this event.
        """
        event_type = event.get("type", "")
        source = event.get("source", "")
        payload = event.get("payload", {})

        # Only classify communication events
        if event_type not in (
            "email.received",
            "email.sent",
            "message.received",
            "message.sent",
            "call.received",
            "call.missed",
        ):
            return SenderType.UNKNOWN

        # Step 1: Channel shortcut — messaging apps are always human
        channel = payload.get("channel", source)
        if channel in _ALWAYS_HUMAN_CHANNELS or source in _ALWAYS_HUMAN_SOURCES:
            return SenderType.HUMAN

        # For outbound messages, the user is the sender — classify the recipients
        # as whatever they are, but the notification routing is about inbound.
        # Outbound messages don't generate notifications, so return HUMAN.
        if "sent" in event_type:
            return SenderType.HUMAN

        # Step 2: Get the sender address
        from_address = payload.get("from_address", "")
        if not from_address:
            return SenderType.UNKNOWN

        return self.classify_address(from_address, payload)

    def classify_address(
        self,
        address: str,
        payload: dict | None = None,
    ) -> SenderType:
        """Classify a single sender address.

        Args:
            address: Email address or identifier.
            payload: Optional event payload for content-based classification.

        Returns:
            SenderType classification.
        """
        if not address:
            return SenderType.UNKNOWN

        addr_lower = address.lower().strip()

        # Step 2: Check in-memory cache
        if addr_lower in self._cache:
            return self._cache[addr_lower]

        # Step 3: Check persisted classification in contacts table
        persisted = self._get_persisted_type(addr_lower)
        if persisted is not None:
            self._cache_put(addr_lower, persisted)
            return persisted

        # Step 4: Run classification logic
        sender_type = self._classify(addr_lower, payload)

        # Step 5: Persist and cache
        self._persist_type(addr_lower, sender_type)
        self._cache_put(addr_lower, sender_type)

        return sender_type

    def _classify(self, addr_lower: str, payload: dict | None) -> SenderType:
        """Run the actual classification logic."""

        # Check marketing filter first (most comprehensive)
        if is_marketing_or_noreply(addr_lower, payload):
            # Distinguish between marketing and transactional within the
            # "automated sender" category. The marketing filter catches both,
            # but we want to separate bills/alerts from promos/newsletters.
            if self._is_transactional_address(addr_lower, payload):
                return SenderType.COMPANY_TRANSACTIONAL
            return SenderType.COMPANY_MARKETING

        # Check transactional patterns not caught by marketing filter
        if self._is_transactional_address(addr_lower, payload):
            return SenderType.COMPANY_TRANSACTIONAL

        # Default: human
        return SenderType.HUMAN

    def _is_transactional_address(
        self,
        addr_lower: str,
        payload: dict | None,
    ) -> bool:
        """Detect transactional/operational business senders."""

        # Check local-part patterns
        if any(addr_lower.startswith(p) for p in _TRANSACTIONAL_LOCALPARTS):
            return True

        # Check subject line for transactional patterns
        if payload:
            subject = (payload.get("subject") or "").strip()
            if subject:
                for pattern in _TRANSACTIONAL_SUBJECT_RE:
                    if pattern.search(subject):
                        return True

            # Financial transaction events are always transactional
            body = payload.get("body_plain", "") or payload.get("body", "") or ""
            snippet = payload.get("snippet", "") or ""
            text = f"{subject} {snippet} {body[:500]}".lower()

            # Bill-pay and statement indicators
            bill_indicators = (
                "amount due",
                "minimum payment",
                "due date",
                "autopay",
                "auto-pay",
                "auto pay",
                "billing period",
                "account balance",
                "past due",
                "overdue",
                "pay now",
                "pay your bill",
                "make a payment",
                "your statement",
                "monthly statement",
                "payment received",
                "payment posted",
                "payment confirmation",
                "order confirmation",
            )
            if sum(1 for ind in bill_indicators if ind in text) >= 2:
                return True

        return False

    def _get_persisted_type(self, addr_lower: str) -> SenderType | None:
        """Look up a previously classified sender type from the DB."""
        try:
            with self.db.get_connection("entities") as conn:
                row = conn.execute(
                    """SELECT c.sender_type FROM contacts c
                       JOIN contact_identifiers ci ON c.id = ci.contact_id
                       WHERE ci.identifier_type = 'email'
                         AND lower(ci.identifier) = ?
                         AND c.sender_type IS NOT NULL""",
                    (addr_lower,),
                ).fetchone()
                if row and row["sender_type"]:
                    try:
                        return SenderType(row["sender_type"])
                    except ValueError:
                        return None
        except Exception:
            pass
        return None

    def _persist_type(self, addr_lower: str, sender_type: SenderType) -> None:
        """Persist sender classification to the contacts table."""
        try:
            with self.db.get_connection("entities") as conn:
                # Update any existing contact with this email
                conn.execute(
                    """UPDATE contacts SET sender_type = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                       WHERE id IN (
                           SELECT contact_id FROM contact_identifiers
                           WHERE identifier_type = 'email' AND lower(identifier) = ?
                       )""",
                    (sender_type.value, addr_lower),
                )
        except Exception:
            logger.debug("Could not persist sender_type for %s", addr_lower)

    def _cache_put(self, key: str, value: SenderType) -> None:
        """Add to cache with simple eviction when full."""
        if len(self._cache) >= self._cache_max:
            # Evict oldest 20% to avoid constant eviction
            evict_count = self._cache_max // 5
            keys = list(self._cache.keys())[:evict_count]
            for k in keys:
                del self._cache[k]
        self._cache[key] = value

    def reclassify_address(self, address: str, sender_type: SenderType) -> None:
        """Manually reclassify a sender address (user override).

        Args:
            address: The email address to reclassify.
            sender_type: The new classification.
        """
        addr_lower = address.lower().strip()
        self._persist_type(addr_lower, sender_type)
        self._cache_put(addr_lower, sender_type)

    def get_classification_stats(self) -> dict[str, int]:
        """Get counts of each sender type from classified contacts."""
        try:
            with self.db.get_connection("entities") as conn:
                rows = conn.execute(
                    """SELECT sender_type, COUNT(*) as cnt
                       FROM contacts
                       WHERE sender_type IS NOT NULL
                       GROUP BY sender_type"""
                ).fetchall()
                return {row["sender_type"]: row["cnt"] for row in rows}
        except Exception:
            return {}
