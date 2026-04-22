"""Life OS v2 — PII shield.

Fresh-module rewrite of v1's ``services/ai_engine/pii.py``. Strips
personally identifiable information from text before it leaves the
local machine for the Anthropic Claude API, and reinserts it in the
response.

Contract (CEO plan §"PII redaction" + Week 7 NEXT_TASKS):

- :class:`PIIShield` has two public methods:
  :meth:`PIIShield.redact` (text → (redacted, mapping)) and
  :meth:`PIIShield.restore` (text + mapping → text).
- Four PII categories are redacted: ``EMAIL``, ``PHONE``, ``ADDRESS``,
  ``PERSON`` (the last from a caller-supplied ``known_names`` list,
  typically loaded from ``entities WHERE kind='contact'``).
- Mappings are *one-time*. The reverse mapping returned from
  :meth:`redact` is also captured on the instance; calling
  :meth:`restore` consumes (clears) the instance-captured mapping so
  it cannot be replayed. Callers who need restore on a cold instance
  pass the mapping back explicitly — restoring via an explicit
  mapping does NOT mutate the instance.

What this module deliberately does NOT own
------------------------------------------
- **Loading known names.** The caller runs the
  ``SELECT name FROM entities WHERE kind='contact'`` query (or
  equivalent) and hands the list in at construction time. Keeping the
  shield sqlite-free matches :mod:`ai.engine`'s "wire-level" framing:
  no I/O, no DB handle, just string transforms.
- **SSN / credit-card.** v1 redacted these under the generic ``ACCT``
  bucket. v2 drops them from the contract entirely — they should
  never reach an AI prompt in the first place. The caller is expected
  to strip them upstream (or not put them in events). The regex
  patterns were removed rather than left dormant so the shield is
  auditable: everything it matches is in the contract.
- **Restoring the same mapping twice.** Explicit one-time discipline
  per the task spec. Once consumed, the mapping is gone.

Pattern ordering
----------------
The :meth:`redact` passes are ordered so longer / higher-priority
matches consume text before shorter ones can nibble at fragments:

1. **Known names** (sorted longest-first so ``"Mike Johnson"`` wins
   over ``"Mike"``).
2. **Email addresses** (consumed before phone so that a dotted-digit
   domain fragment cannot masquerade as a phone number).
3. **Street addresses** (consumed before phone so the numeric
   house-number prefix cannot be chewed off by the phone regex).
4. **Phone numbers** (last, because the phone regex is the broadest
   digit-run pattern and would otherwise hoover up fragments of
   earlier categories).

Within each pass, identical values collapse to the same token (one
entry in the mapping → replace everywhere).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Email: standard simplified RFC-5322. Local part allows the common
# symbol set; domain allows multi-level subdomains and 2+ char TLDs.
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone: flexible international + domestic. Optional leading ``+``,
# optional country code in parens, then 7-15 chars of digits and
# separators (space / dash / dot / slash). A post-match digit count
# of ≥7 filters out short numeric sequences (zip codes, 4-digit PIN
# codes, years) that a regex with a low minimum would over-match.
_PHONE_PATTERN = re.compile(r"\+?\(?\d{1,4}\)?[-\s./\d]{6,20}")

# Address: US-style street line: ``<house number> <street name>
# <suffix>``. Suffix set covers the most common abbreviations plus
# the spelled-out forms. Case-insensitive; boundary-anchored. Two
# known false-negatives intentional:
#
# - Addresses without a trailing suffix ("123 Main") are NOT matched,
#   because the unanchored form would catch too many arbitrary "number
#   + noun" sequences in body text.
# - International / PO Box formats are NOT matched. The shield errs
#   toward false negatives (leave PII in) rather than false positives
#   (corrupt legitimate text) on formats the regex can't verify.
_ADDRESS_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|"
    r"Way|Court|Ct|Place|Pl|Terrace|Ter|Highway|Hwy|Parkway|Pkwy|Circle|Cir)"
)
_ADDRESS_PATTERN = re.compile(
    # House number: ``\d{1,6}`` plus an optional trailing letter so
    # "221B" / "500A" (common apartment-suffix forms) still match.
    # Middle words: ``{1,5}?`` non-greedy so the regex prefers the
    # shortest suffix-anchored match — "500 Oak Drive" wins over
    # "500 Oak Drive to 900 Elm Lane" when both are in the same text.
    r"\b\d{1,6}[A-Za-z]?\s+(?:[A-Za-z0-9][A-Za-z0-9.'-]*\s+){1,5}?" + _ADDRESS_SUFFIX + r"\b\.?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shield
# ---------------------------------------------------------------------------


class PIIShield:
    """One-shot PII redactor with explicit restore.

    Parameters
    ----------
    known_names:
        Iterable of contact names that should be replaced as
        ``[PERSON_N]`` tokens regardless of whether they match any
        regex pattern. Typically loaded by the caller from
        ``SELECT name FROM entities WHERE kind='contact'``. ``None``
        is accepted and treated as "no pre-loaded names".

    Lifecycle
    ---------
    ``redact`` is safe to call many times — each call resets internal
    state. ``restore`` is one-time against the instance-captured
    mapping: after the first call the captured mapping is cleared so
    a replay returns the input unchanged (no accidental double-scan
    of already-restored text). Callers who must restore on a separate
    instance pass the mapping back explicitly.
    """

    # Category labels surfaced in tokens. Order mirrors the v1 counter
    # dict so existing callers that inspect token categories keep the
    # same vocabulary.
    _CATEGORIES = ("PERSON", "EMAIL", "ADDRESS", "PHONE")

    def __init__(self, known_names: Iterable[str] | None = None) -> None:
        self._known_names: list[str] = list(known_names) if known_names else []
        # ``_reverse_mapping`` is the instance-captured one-time
        # mapping returned by the most recent ``redact`` call.
        # ``restore()`` clears it after use.
        self._reverse_mapping: dict[str, str] = {}
        self._forward_mapping: dict[str, str] = {}
        self._counters: dict[str, int] = dict.fromkeys(self._CATEGORIES, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace PII in ``text`` with placeholder tokens.

        Returns ``(redacted_text, mapping)`` where ``mapping`` is a
        fresh ``{token: original}`` dict. The mapping is ALSO captured
        on the instance so a subsequent :meth:`restore` call with no
        explicit mapping works — but is cleared after that one call.
        """
        # Reset per-call so every invocation is self-contained and
        # counters restart at 1. An abandoned prior call (no restore)
        # doesn't leak into this call's namespace.
        self._forward_mapping = {}
        self._reverse_mapping = {}
        self._counters = dict.fromkeys(self._CATEGORIES, 0)

        if not text:
            return text, {}

        result = text

        # Pass 1 — known names, longest-first. Longer names win so
        # "Mike Johnson" is replaced before a standalone "Mike" has a
        # chance to consume the first word and strand "Johnson".
        for name in sorted(self._known_names, key=len, reverse=True):
            if not name:
                continue
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            if pattern.search(result):
                token = self._issue_token("PERSON")
                self._forward_mapping[name] = token
                self._reverse_mapping[token] = name
                result = pattern.sub(token, result)

        # Pass 2 — emails. Run before address/phone because a dotted
        # domain ("mail.internal.co.uk") could otherwise be nibbled by
        # the phone pattern's dot-separator form.
        result = self._redact_pattern(result, _EMAIL_PATTERN, "EMAIL")

        # Pass 3 — street addresses. Run before phone because the
        # leading house number would otherwise satisfy the phone
        # regex's 7+ digit threshold via subsequent street digits.
        result = self._redact_pattern(result, _ADDRESS_PATTERN, "ADDRESS")

        # Pass 4 — phones. Broadest pattern, so runs last. The
        # post-match digit check (≥7 digits) filters out short numeric
        # runs like zip codes and years that the regex's 7-char minimum
        # could match through non-digit separators.
        result = self._redact_phones(result)

        return result, dict(self._reverse_mapping)

    def restore(self, text: str, mapping: dict[str, str] | None = None) -> str:
        """Re-insert PII into ``text``.

        Accepts an optional explicit ``mapping`` (``{token: original}``).
        When ``mapping`` is ``None``, the instance-captured mapping
        from the most recent :meth:`redact` call is consumed and then
        cleared. Explicit-mapping calls leave the instance state
        alone so the captured mapping can still be consumed later.
        """
        if mapping is None:
            restore_map = self._reverse_mapping
            # One-time: consume the instance capture. A second call
            # with no mapping becomes a no-op rather than a re-scan
            # of text that no longer contains tokens.
            self._reverse_mapping = {}
            self._forward_mapping = {}
        else:
            restore_map = mapping

        if not restore_map or not text:
            return text

        result = text
        # Longer tokens first is not strictly necessary (tokens are
        # unique per category) but it is the robust ordering if a
        # caller ever extends with overlapping category prefixes.
        for token in sorted(restore_map, key=len, reverse=True):
            result = result.replace(token, restore_map[token])
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _redact_pattern(self, text: str, pattern: re.Pattern[str], category: str) -> str:
        """Walk ``pattern`` matches left-to-right, issuing tokens.

        Dedupes identical hits within a single call — the second
        occurrence of ``alice@example.com`` reuses ``[EMAIL_1]`` so
        restore maps both back to the same value. Uses
        ``finditer`` + ``str.replace`` rather than ``sub`` so the
        forward-mapping check reflects already-tokenised text (avoids
        accidentally issuing a fresh token for a value that was
        tokenised on a prior pass).
        """
        # Snapshot matches up-front so mutations to ``text`` via
        # replace don't shift the finditer positions.
        seen: list[str] = []
        for match in pattern.finditer(text):
            value = match.group()
            if value and value not in seen and value not in self._forward_mapping:
                seen.append(value)

        result = text
        for value in seen:
            token = self._issue_token(category)
            self._forward_mapping[value] = token
            self._reverse_mapping[token] = value
            result = result.replace(value, token)
        return result

    def _redact_phones(self, text: str) -> str:
        """Phone pass with a post-match digit-count gate.

        The regex tolerates 7-15 chars of mixed digits/separators.
        A standalone count of digits (``≥7``) rejects short runs that
        match the shape but are semantically something else (year
        ranges, zip codes, PIN codes). Any value already mapped by an
        earlier pass (e.g. an email's domain digits) is skipped so the
        phone bucket doesn't double-issue.
        """
        seen: list[str] = []
        for match in _PHONE_PATTERN.finditer(text):
            value = match.group()
            digit_count = sum(1 for ch in value if ch.isdigit())
            if (
                digit_count >= 7
                and value not in seen
                and value not in self._forward_mapping
                # Any token embedded in the match is a sign the phone
                # regex strayed into already-redacted territory; skip.
                and "[" not in value
            ):
                seen.append(value)

        result = text
        for value in seen:
            token = self._issue_token("PHONE")
            self._forward_mapping[value] = token
            self._reverse_mapping[token] = value
            result = result.replace(value, token)
        return result

    def _issue_token(self, category: str) -> str:
        """Hand out the next ``[CATEGORY_N]`` token for ``category``.

        Counters reset per :meth:`redact` call, so numbering always
        starts at 1 within a single shield output.
        """
        self._counters[category] = self._counters.get(category, 0) + 1
        return f"[{category}_{self._counters[category]}]"


__all__ = ["PIIShield"]
