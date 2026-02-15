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

    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')

    def __init__(self, known_names: Optional[list[str]] = None):
        self.known_names = known_names or []
        self._mapping: dict[str, str] = {}
        self._reverse_mapping: dict[str, str] = {}
        self._counters: dict[str, int] = {
            "PERSON": 0, "EMAIL": 0, "PHONE": 0,
            "ADDRESS": 0, "ACCT": 0, "ORG": 0,
        }

    def strip(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Strip PII from text. Returns (stripped_text, mapping).
        The mapping can be used later to restore PII in the response.
        """
        self._mapping = {}
        self._reverse_mapping = {}
        self._counters = {k: 0 for k in self._counters}

        result = text

        # Strip known names (highest priority — these are specific to this user)
        for name in sorted(self.known_names, key=len, reverse=True):
            if name.lower() in result.lower():
                token = self._get_token("PERSON")
                # Case-insensitive replacement
                pattern = re.compile(re.escape(name), re.IGNORECASE)
                result = pattern.sub(token, result)
                self._mapping[name] = token
                self._reverse_mapping[token] = name

        # Strip emails
        for match in self.EMAIL_PATTERN.finditer(result):
            email_addr = match.group()
            if email_addr not in self._mapping:
                token = self._get_token("EMAIL")
                self._mapping[email_addr] = token
                self._reverse_mapping[token] = email_addr
                result = result.replace(email_addr, token)

        # Strip phone numbers
        for match in self.PHONE_PATTERN.finditer(result):
            phone = match.group()
            if len(phone) >= 7 and phone not in self._mapping:
                token = self._get_token("PHONE")
                self._mapping[phone] = token
                self._reverse_mapping[token] = phone
                result = result.replace(phone, token)

        # Strip SSNs
        for match in self.SSN_PATTERN.finditer(result):
            ssn = match.group()
            token = self._get_token("ACCT")
            self._mapping[ssn] = token
            self._reverse_mapping[token] = ssn
            result = result.replace(ssn, token)

        # Strip credit cards
        for match in self.CREDIT_CARD_PATTERN.finditer(result):
            cc = match.group()
            token = self._get_token("ACCT")
            self._mapping[cc] = token
            self._reverse_mapping[token] = cc
            result = result.replace(cc, token)

        return result, dict(self._reverse_mapping)

    def restore(self, text: str, mapping: Optional[dict[str, str]] = None) -> str:
        """Restore PII tokens back to real values in the AI's response."""
        restore_map = mapping or self._reverse_mapping
        result = text
        for token, original in restore_map.items():
            result = result.replace(token, original)
        return result

    def _get_token(self, category: str) -> str:
        self._counters[category] = self._counters.get(category, 0) + 1
        return f"[{category}_{self._counters[category]}]"
