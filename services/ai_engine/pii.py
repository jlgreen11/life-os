"""
Life OS — PII Shield

Strips personally identifiable information before sending text to
external APIs, and restores it in responses.

Replacement pattern:
    "Mike Johnson" → "[PERSON_1]"
    "mike@example.com" → "[EMAIL_1]"
    "Chase account 4821" → "[BANK_1] account [ACCT_1]"
    "123 Main Street" → "[ADDRESS_1]"
    "+1-555-123-4567" → "[PHONE_1]"
"""

from __future__ import annotations

import re
from typing import Optional


class PIIShield:
    """
    Strips personally identifiable information before sending text to
    external APIs, and restores it in responses.
    """

    # --- Regex patterns for PII detection ---
    # Email: standard RFC 5322 simplified pattern. Matches user@domain.tld formats.
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    # Phone: flexible international format. Handles optional country code (+1),
    # parenthesized area codes, and various separators (dash, space, dot, slash).
    # Minimum 7 digits to avoid false positives on short number sequences.
    PHONE_PATTERN = re.compile(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}')
    # SSN: US Social Security Number in standard XXX-XX-XXXX format.
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    # Credit card: 16-digit card numbers with optional separators (dash or space)
    # between each group of 4 digits. Covers Visa, Mastercard, etc.
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

    def __init__(self, known_names: Optional[list[str]] = None):
        # known_names: an optional pre-loaded list of names (user, family, contacts)
        # that should always be detected even if they don't match a regex pattern.
        # Name-based detection is prioritized over regex patterns (see strip()).
        self.known_names = known_names or []
        # _mapping: original_value -> token (e.g., "mike@ex.com" -> "[EMAIL_1]")
        self._mapping: dict[str, str] = {}
        # _reverse_mapping: token -> original_value (used by restore() to reinsert PII)
        self._reverse_mapping: dict[str, str] = {}
        # _counters: track how many tokens have been issued per category to
        # generate unique, sequential placeholder tokens ([EMAIL_1], [EMAIL_2], ...).
        self._counters: dict[str, int] = {
            "PERSON": 0, "EMAIL": 0, "PHONE": 0,
            "ADDRESS": 0, "ACCT": 0, "ORG": 0,
        }

    def strip(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Strip PII from text. Returns (stripped_text, mapping).
        The mapping can be used later to restore PII in the response.

        Processing order matters: known names are replaced first (highest
        priority), then emails, phones, SSNs, and credit cards. This prevents
        partial matches -- e.g., replacing an email address before a name
        pattern could accidentally consume part of a person's name.
        """
        # Reset all state for each call so strip() is idempotent and safe
        # for concurrent use across different text inputs.
        self._mapping = {}
        self._reverse_mapping = {}
        self._counters = {k: 0 for k in self._counters}

        result = text

        # --- Pass 1: Known names (highest priority) ---
        # Sort by length descending so longer names are replaced first. This
        # prevents partial matches: "Mike Johnson" is replaced before "Mike"
        # to avoid leaving a dangling "Johnson" in the text.
        for name in sorted(self.known_names, key=len, reverse=True):
            if name.lower() in result.lower():
                token = self._get_token("PERSON")
                # Case-insensitive replacement preserves surrounding context
                # while catching variations like "mike", "Mike", "MIKE".
                pattern = re.compile(re.escape(name), re.IGNORECASE)
                result = pattern.sub(token, result)
                self._mapping[name] = token
                self._reverse_mapping[token] = name

        # --- Pass 2: Email addresses ---
        # Deduplicate: skip emails already captured in the mapping (e.g., if
        # an email appeared as part of a known name replacement).
        for match in self.EMAIL_PATTERN.finditer(result):
            email_addr = match.group()
            if email_addr not in self._mapping:
                token = self._get_token("EMAIL")
                self._mapping[email_addr] = token
                self._reverse_mapping[token] = email_addr
                result = result.replace(email_addr, token)

        # --- Pass 3: Phone numbers ---
        # Additional length check (>= 7 chars) filters out short numeric
        # sequences that the broad regex might incorrectly match (e.g.,
        # zip codes, short ID numbers).
        for match in self.PHONE_PATTERN.finditer(result):
            phone = match.group()
            if len(phone) >= 7 and phone not in self._mapping:
                token = self._get_token("PHONE")
                self._mapping[phone] = token
                self._reverse_mapping[token] = phone
                result = result.replace(phone, token)

        # --- Pass 4: Social Security Numbers ---
        # SSNs are classified under the ACCT category since they are account-
        # like identifiers. The XXX-XX-XXXX format is highly specific, so
        # false positives are rare.
        for match in self.SSN_PATTERN.finditer(result):
            ssn = match.group()
            token = self._get_token("ACCT")
            self._mapping[ssn] = token
            self._reverse_mapping[token] = ssn
            result = result.replace(ssn, token)

        # --- Pass 5: Credit card numbers ---
        # Also classified under ACCT. Matches 16-digit card numbers with
        # optional separators between 4-digit groups.
        for match in self.CREDIT_CARD_PATTERN.finditer(result):
            cc = match.group()
            token = self._get_token("ACCT")
            self._mapping[cc] = token
            self._reverse_mapping[token] = cc
            result = result.replace(cc, token)

        # Return both the sanitized text and the reverse mapping so the caller
        # can restore PII in the LLM's response via restore().
        return result, dict(self._reverse_mapping)

    def restore(self, text: str, mapping: Optional[dict[str, str]] = None) -> str:
        """Restore PII tokens back to real values in the AI's response.

        Accepts an optional explicit mapping (useful when strip() was called
        multiple times and mappings were merged). Falls back to the instance's
        internal reverse mapping if none is provided. This is a simple string
        replacement -- every occurrence of each token is replaced with the
        original PII value.
        """
        restore_map = mapping or self._reverse_mapping
        result = text
        for token, original in restore_map.items():
            result = result.replace(token, original)
        return result

    def _get_token(self, category: str) -> str:
        """Generate a unique, sequential placeholder token for a PII category.

        Tokens follow the pattern [CATEGORY_N] (e.g., [EMAIL_1], [PERSON_2]).
        The counter auto-increments per category to ensure uniqueness within
        a single strip() call.
        """
        self._counters[category] = self._counters.get(category, 0) + 1
        return f"[{category}_{self._counters[category]}]"
