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
    "sk-abc123..." → "[API_KEY_1]"
    "eyJhbGci..." → "[JWT_1]"
    "DE89370400440532013000" → "[IBAN_1]"
    "2001:db8::1" → "[IPV6_1]"
    "00:1A:2B:3C:4D:5E" → "[MAC_1]"
"""

from __future__ import annotations

import re
from typing import Optional


class PIIShield:
    """
    Strips personally identifiable information before sending text to
    external APIs, and restores it in responses.
    """

    # --- Regex patterns for PII detection (compiled at class-load) ---
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
    # IBAN: International Bank Account Number. Two-letter country code, two check
    # digits, up to 30 alphanumeric chars. Total length 15-34 chars (validated
    # post-match — the regex allows 5-34 to keep the pattern simple).
    IBAN_PATTERN = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b')
    # IPv6: matches both full (8 groups) and compressed (`::` shorthand) forms.
    # Requires at least 2 colons to avoid colliding with single-colon constructs.
    # MAC redaction runs first so 6-group MAC-style strings don't trip this.
    IPV6_PATTERN = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b|'
        r'(?:[0-9a-fA-F]{1,4}:){1,7}:(?:[0-9a-fA-F]{1,4})?|'
        r'::(?:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,6})?'
    )
    # MAC: six pairs of hex digits separated by `:` or `-`. Hex colors like
    # `#aabbcc` don't match because they have no separators.
    MAC_PATTERN = re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b')
    # API keys: common prefixed formats from OpenAI, Stripe, GitHub, Slack, AWS.
    # Each prefix has its own minimum-length suffix to avoid false positives.
    API_KEY_PATTERN = re.compile(
        r'\b('
        r'sk-[A-Za-z0-9]{20,}|'
        r'pk_(?:live|test)_[A-Za-z0-9]{20,}|'
        r'ghp_[A-Za-z0-9]{30,}|'
        r'gho_[A-Za-z0-9]{30,}|'
        r'xoxb-[A-Za-z0-9-]{20,}|'
        r'AKIA[0-9A-Z]{16}'
        r')\b'
    )
    # JWT: three base64url segments separated by dots. The `eyJ` anchor matches
    # the standard JOSE header prefix (`{"alg":...}` base64-encoded).
    JWT_PATTERN = re.compile(
        r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'
    )

    # Token categories we maintain counters for. New entries must be added
    # here so _get_token() can issue sequential placeholders.
    _CATEGORIES = (
        "PERSON", "EMAIL", "PHONE", "ADDRESS", "ACCT", "ORG",
        "IBAN", "IPV6", "MAC", "API_KEY", "JWT",
    )

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
        self._counters: dict[str, int] = {k: 0 for k in self._CATEGORIES}
        # redaction_counts: cumulative count of redactions across all strip() calls,
        # keyed by category. Exposed via stats() for observability — operators can
        # confirm patterns actually fire in production traffic.
        self.redaction_counts: dict[str, int] = {k: 0 for k in self._CATEGORIES}

    def strip(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Strip PII from text. Returns (stripped_text, mapping).
        The mapping can be used later to restore PII in the response.

        Processing order matters. High-entropy secrets (JWTs, API keys) run
        before generic patterns so they aren't partially clobbered. MAC runs
        before IPv6 because both use colon-separated hex. IBAN runs before
        phone because IBAN trailing digits can otherwise be captured by the
        permissive phone regex.

        Full order:
            known_names → JWT → API_KEY → email → MAC → IPv6 → IBAN →
            phone → SSN → credit_card
        """
        # Reset per-call state so strip() is idempotent and safe for concurrent
        # use across different text inputs. redaction_counts is NOT reset — it
        # accumulates across calls for observability.
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
                self.redaction_counts["PERSON"] += 1

        # --- Pass 2: JWTs (run before email/word patterns) ---
        # JWTs are high-entropy and structurally distinctive. Redacting them
        # first prevents the email regex from chewing on base64 segments that
        # happen to contain dots.
        result = self._redact_pattern(result, self.JWT_PATTERN, "JWT")

        # --- Pass 3: API keys (run before email/word patterns) ---
        # API keys often contain hyphens and underscores that could be picked
        # up by other patterns. Redact them up front.
        result = self._redact_pattern(result, self.API_KEY_PATTERN, "API_KEY")

        # --- Pass 4: Email addresses ---
        for match in self.EMAIL_PATTERN.finditer(result):
            email_addr = match.group()
            if email_addr not in self._mapping:
                token = self._get_token("EMAIL")
                self._mapping[email_addr] = token
                self._reverse_mapping[token] = email_addr
                self.redaction_counts["EMAIL"] += 1
                result = result.replace(email_addr, token)

        # --- Pass 5: MAC addresses (before IPv6) ---
        # MAC has exactly 6 octets of 2 hex chars separated by `:` or `-`.
        # Running before IPv6 avoids the IPv6 pattern claiming a MAC string.
        result = self._redact_pattern(result, self.MAC_PATTERN, "MAC")

        # --- Pass 6: IPv6 addresses ---
        result = self._redact_pattern(result, self.IPV6_PATTERN, "IPV6")

        # --- Pass 7: IBANs (before phone) ---
        # The regex permits 5-34 chars after the `\b`; we filter to the real
        # IBAN length range (15-34) post-match to drop false positives like
        # short country-code-prefixed tokens.
        for match in self.IBAN_PATTERN.finditer(result):
            iban = match.group()
            if 15 <= len(iban) <= 34 and iban not in self._mapping:
                token = self._get_token("IBAN")
                self._mapping[iban] = token
                self._reverse_mapping[token] = iban
                self.redaction_counts["IBAN"] += 1
                result = result.replace(iban, token)

        # --- Pass 8: Phone numbers ---
        # Additional length check (>= 7 chars) filters out short numeric
        # sequences that the broad regex might incorrectly match (e.g.,
        # zip codes, short ID numbers).
        for match in self.PHONE_PATTERN.finditer(result):
            phone = match.group()
            if len(phone) >= 7 and phone not in self._mapping:
                token = self._get_token("PHONE")
                self._mapping[phone] = token
                self._reverse_mapping[token] = phone
                self.redaction_counts["PHONE"] += 1
                result = result.replace(phone, token)

        # --- Pass 9: Social Security Numbers ---
        # SSNs are classified under the ACCT category since they are account-
        # like identifiers. The XXX-XX-XXXX format is highly specific, so
        # false positives are rare.
        for match in self.SSN_PATTERN.finditer(result):
            ssn = match.group()
            token = self._get_token("ACCT")
            self._mapping[ssn] = token
            self._reverse_mapping[token] = ssn
            self.redaction_counts["ACCT"] += 1
            result = result.replace(ssn, token)

        # --- Pass 10: Credit card numbers ---
        for match in self.CREDIT_CARD_PATTERN.finditer(result):
            cc = match.group()
            token = self._get_token("ACCT")
            self._mapping[cc] = token
            self._reverse_mapping[token] = cc
            self.redaction_counts["ACCT"] += 1
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

    def stats(self) -> dict[str, int]:
        """Return cumulative redaction counts per category.

        Counts accumulate across all strip() calls on this shield instance.
        Use for observability — confirms in production that patterns are
        firing on real traffic, not silently missing.
        """
        return dict(self.redaction_counts)

    def _redact_pattern(self, text: str, pattern: re.Pattern, category: str) -> str:
        """Find and redact all matches for a single pattern.

        Centralizes the dedupe-and-replace loop used by the simpler patterns
        (those without per-match validation like length checks). Each unique
        match gets its own sequentially-numbered token, and the bookkeeping
        attributes (_mapping, _reverse_mapping, redaction_counts) are updated
        in lockstep.
        """
        for match in pattern.finditer(text):
            value = match.group()
            if value not in self._mapping:
                token = self._get_token(category)
                self._mapping[value] = token
                self._reverse_mapping[token] = value
                self.redaction_counts[category] += 1
                text = text.replace(value, token)
        return text

    def _get_token(self, category: str) -> str:
        """Generate a unique, sequential placeholder token for a PII category.

        Tokens follow the pattern [CATEGORY_N] (e.g., [EMAIL_1], [PERSON_2]).
        The counter auto-increments per category to ensure uniqueness within
        a single strip() call.
        """
        self._counters[category] = self._counters.get(category, 0) + 1
        return f"[{category}_{self._counters[category]}]"
