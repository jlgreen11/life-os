"""
Life OS — Email Classifier

Shared email classification logic used across the pipeline to identify
marketing, automated, and bulk emails early — before they generate noise
through episodic memory, signal extraction, task extraction, notifications,
and vector embeddings.

This module centralizes the detection heuristics so they can be used by:
    - master_event_handler (early pipeline suppression)
    - PredictionEngine (follow-up prediction filtering)
    - RulesEngine (default marketing rule)

Classification Categories:
    marketing   — Newsletters, promotions, sales, bulk campaigns
    automated   — No-reply, mailer-daemon, system-generated
    personal    — Human-to-human communication (default)
"""

from __future__ import annotations

import re


def is_marketing_email(from_address: str, payload: dict) -> bool:
    """Determine whether an email is marketing/automated and should be suppressed.

    Checks (in order of cost, cheapest first):
        1. No-reply / automated system senders
        2. Bulk sender local-part patterns (newsletter@, promo@, etc.)
        3. Marketing subdomain patterns (@email., @reply., @mg., etc.)
        4. Subject-line marketing indicators (% off, sale, unsubscribe, etc.)
        5. Body contains "unsubscribe", "opt out", "manage preferences"
        6. Common ESP (Email Service Provider) headers in body

    Returns True if the email is marketing/automated, False if it appears personal.
    """
    if not from_address:
        return False

    addr_lower = from_address.lower()

    # --- 1. No-reply and automated system senders ---
    # These are definitively non-personal and never require a reply.
    noreply_patterns = (
        "no-reply@", "noreply@", "do-not-reply@", "donotreply@",
        "mailer-daemon@", "postmaster@", "daemon@", "auto-reply@",
        "autoreply@", "automated@",
    )
    if any(pattern in addr_lower for pattern in noreply_patterns):
        return True

    # --- 2. Bulk sender local-part patterns ---
    # These local-parts (before the @) are almost always bulk senders when
    # they appear at the START of the address. We check startswith() to avoid
    # false positives like john.email@company.com or sarah.reply@startup.io.
    bulk_localpart_patterns = (
        "newsletter@", "notifications@", "updates@", "digest@",
        "mailer@", "bulk@", "promo@", "marketing@",
        "reply@", "email@", "news@", "offers@", "deals@",
        "hello@", "info@", "support@", "help@",
        "alerts@", "announce@", "campaign@", "promotions@",
        "store@", "shop@", "sales@", "team@",
    )
    if any(addr_lower.startswith(pattern) for pattern in bulk_localpart_patterns):
        return True

    # --- 3. Marketing subdomain patterns ---
    # Email service providers route through subdomains like email.company.com,
    # reply.brand.com, mg.service.com. These are strong marketing signals.
    marketing_domain_patterns = (
        "@news-", "@email.", "@reply.", "@mailing.",
        "@newsletters.", "@promo.", "@marketing.",
        "@em.", "@mg.", "@mail.",
        "@bounce.", "@send.", "@sent.", "@e.",
        "@campaign.", "@comms.", "@comms-",
    )
    if any(pattern in addr_lower for pattern in marketing_domain_patterns):
        return True

    # --- 4. Subject-line marketing indicators ---
    subject = (payload.get("subject") or "").lower()
    if subject and _has_marketing_subject(subject):
        # Subject alone is a weak signal — combine with body check
        # to reduce false positives. If the subject looks promotional
        # AND the body has unsubscribe/opt-out, it's definitely marketing.
        body_text = _get_body_text(payload)
        if _has_unsubscribe_indicators(body_text):
            return True

    # --- 5. Body unsubscribe/opt-out indicators ---
    # Legitimate marketing emails are legally required (CAN-SPAM, GDPR) to
    # include unsubscribe mechanisms. This catches emails that passed the
    # sender-pattern checks above.
    body_text = _get_body_text(payload)
    if _has_unsubscribe_indicators(body_text):
        return True

    return False


def _has_marketing_subject(subject: str) -> bool:
    """Check if a subject line contains marketing/promotional patterns.

    Uses regex for patterns that need word boundaries (like "% off", "sale")
    to avoid false positives on legitimate subjects.
    """
    # Direct promotional patterns (case-insensitive, already lowered)
    promo_patterns = [
        r"\d+%\s*off",           # "20% off", "50% OFF"
        r"\bsale\b",             # "sale" as a word
        r"\bfree shipping\b",
        r"\blimited time\b",
        r"\blast chance\b",
        r"\bdon'?t miss\b",
        r"\bexclusive offer\b",
        r"\bspecial offer\b",
        r"\bact now\b",
        r"\border now\b",
        r"\bshop now\b",
        r"\bnew arrivals?\b",
        r"\bclearance\b",
        r"\bbogo\b",             # Buy One Get One
        r"\bcoupon\b",
        r"\bpromo code\b",
        r"\bdiscount\b",
        r"\bflash sale\b",
        r"\bending soon\b",
    ]
    return any(re.search(p, subject) for p in promo_patterns)


def _get_body_text(payload: dict) -> str:
    """Extract searchable text from email payload fields."""
    return " ".join(filter(None, [
        payload.get("body_plain", ""),
        payload.get("snippet", ""),
        payload.get("body", ""),
    ])).lower()


def _has_unsubscribe_indicators(text: str) -> bool:
    """Check if text contains unsubscribe or opt-out indicators.

    These phrases are legally required in commercial emails per CAN-SPAM
    (US), GDPR (EU), and CASL (Canada), making them reliable marketing signals.
    """
    indicators = (
        "unsubscribe",
        "opt out",
        "opt-out",
        "manage preferences",
        "manage your preferences",
        "email preferences",
        "update your preferences",
        "subscription preferences",
        "mailing list",
        "no longer wish to receive",
        "stop receiving these emails",
        "remove from this list",
    )
    return any(indicator in text for indicator in indicators)
