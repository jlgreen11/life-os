"""Tests for :mod:`ai.pii`.

Strategy
--------
Exercises the four category contract (EMAIL / PHONE / ADDRESS /
PERSON) end-to-end via :meth:`PIIShield.redact` → :meth:`restore`
round-trip. Regex behaviour is tested through its observable effect
(what gets tokenised), not by poking the compiled patterns directly,
so the module's internal pattern tweaks stay free to evolve.

Key assertions beyond the v1-style coverage:

- **One-time mapping discipline.** After a no-argument restore the
  instance-captured mapping is cleared; a replay becomes a no-op.
- **ADDRESS pattern.** Street-suffix variants match; bare "123 Main"
  without a suffix does NOT (documented false-negative — shield errs
  toward leaving PII in rather than corrupting non-address text).
- **SSN / credit-card dropped.** v2 intentionally does NOT redact
  these (per module docstring). Tests document that explicitly so a
  future reviewer can't reintroduce them without updating the
  contract.
- **No PII leaks.** A full-fixture text has all four categories and
  the redacted output is asserted absent of each original literal.
"""

from __future__ import annotations

import pytest

from ai.pii import PIIShield

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_redact_single_email():
    shield = PIIShield()
    text = "Contact me at john.doe@example.com for details."

    redacted, mapping = shield.redact(text)

    assert "john.doe@example.com" not in redacted
    assert redacted == "Contact me at [EMAIL_1] for details."
    assert mapping == {"[EMAIL_1]": "john.doe@example.com"}


def test_redact_multiple_emails_get_unique_tokens():
    shield = PIIShield()
    text = "Email alice@work.com or bob@personal.org"

    redacted, mapping = shield.redact(text)

    assert "alice@work.com" not in redacted
    assert "bob@personal.org" not in redacted
    assert mapping["[EMAIL_1]"] == "alice@work.com"
    assert mapping["[EMAIL_2]"] == "bob@personal.org"


def test_redact_email_plus_addressing():
    shield = PIIShield()
    redacted, mapping = shield.redact("Send to user+newsletter@example.com")
    assert mapping["[EMAIL_1]"] == "user+newsletter@example.com"
    assert "user+newsletter@example.com" not in redacted


def test_redact_email_multi_level_subdomain():
    shield = PIIShield()
    redacted, mapping = shield.redact("Reach admin@mail.internal.company.co.uk")
    assert mapping["[EMAIL_1]"] == "admin@mail.internal.company.co.uk"
    assert "admin@mail.internal.company.co.uk" not in redacted


def test_repeated_email_collapses_to_one_token():
    shield = PIIShield()
    text = "Email alice@example.com or try alice@example.com again"

    redacted, mapping = shield.redact(text)

    assert len(mapping) == 1
    assert mapping["[EMAIL_1]"] == "alice@example.com"
    assert redacted.count("[EMAIL_1]") == 2


def test_at_symbol_alone_not_matched():
    shield = PIIShield()
    redacted, mapping = shield.redact("The @ symbol is used for mentions")
    assert redacted == "The @ symbol is used for mentions"
    assert mapping == {}


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [
        "+1-555-123-4567",
        "(555) 123-4567",
        "555.123.4567",
        "555-123-4567",
        "+44 20 7946 0958",
    ],
)
def test_redact_various_phone_formats(number):
    shield = PIIShield()
    text = f"Call {number} anytime."

    redacted, mapping = shield.redact(text)

    # Number (modulo trailing whitespace) is gone; a PHONE token
    # replaces it somewhere in the output.
    assert number not in redacted
    assert any(token.startswith("[PHONE_") for token in mapping)


def test_short_numbers_below_seven_digits_are_not_phones():
    shield = PIIShield()
    text = "The code is 1234 and the year is 2024"

    redacted, mapping = shield.redact(text)

    assert "[PHONE_" not in redacted
    assert mapping == {}
    assert "1234" in redacted
    assert "2024" in redacted


# ---------------------------------------------------------------------------
# Address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "street",
    [
        "123 Main Street",
        "1600 Pennsylvania Avenue",
        "742 Evergreen Terrace",
        "221 Baker Road",
        "350 Oak Drive",
        "9 Rue Morgue Lane",
        "500 Olive Branch Way",
        "12 Tremont Pl",
        "7 Harbor Ct",
        "99 West Ridge Blvd",
    ],
)
def test_redact_common_street_address_formats(street):
    shield = PIIShield()
    text = f"Please send mail to {street}, Apt 4B."

    redacted, mapping = shield.redact(text)

    assert street not in redacted
    assert "[ADDRESS_1]" in redacted
    assert mapping["[ADDRESS_1]"].startswith(street)


def test_redact_address_case_insensitive_suffix():
    shield = PIIShield()
    text = "Meet me at 42 Willow ST tomorrow."

    redacted, mapping = shield.redact(text)

    assert "42 Willow ST" not in redacted
    assert any(token.startswith("[ADDRESS_") for token in mapping)


def test_address_without_suffix_is_not_matched():
    """Documents a deliberate false-negative: '123 Main' alone does
    not match because the unanchored form would hit too many number+
    noun pairs in ordinary text."""

    shield = PIIShield()
    redacted, mapping = shield.redact("I moved to 123 Main last summer")

    assert redacted == "I moved to 123 Main last summer"
    assert mapping == {}


def test_multiple_addresses_get_unique_tokens():
    shield = PIIShield()
    text = "From 500 Oak Drive to 900 Elm Lane."

    redacted, _mapping = shield.redact(text)

    assert "500 Oak Drive" not in redacted
    assert "900 Elm Lane" not in redacted
    assert "[ADDRESS_1]" in redacted
    assert "[ADDRESS_2]" in redacted


# ---------------------------------------------------------------------------
# Known names (PERSON)
# ---------------------------------------------------------------------------


def test_redact_known_name_single():
    shield = PIIShield(known_names=["Mike Johnson"])
    text = "I met with Mike Johnson yesterday."

    redacted, mapping = shield.redact(text)

    assert "Mike Johnson" not in redacted
    assert redacted == "I met with [PERSON_1] yesterday."
    assert mapping["[PERSON_1]"] == "Mike Johnson"


def test_known_name_is_case_insensitive():
    shield = PIIShield(known_names=["Sarah Connor"])
    redacted, mapping = shield.redact("Email from sarah connor received.")
    assert "sarah connor" not in redacted
    assert mapping["[PERSON_1]"] == "Sarah Connor"


def test_known_name_multiple_occurrences_share_token():
    shield = PIIShield(known_names=["Alice"])
    redacted, _mapping = shield.redact("Alice called. Tell Alice I'll call back.")
    assert redacted.count("[PERSON_1]") == 2
    assert "Alice" not in redacted


def test_longer_known_name_wins_over_shorter():
    shield = PIIShield(known_names=["Mike Johnson", "Mike"])
    redacted, mapping = shield.redact("Mike Johnson and Mike attended.")

    # "Mike Johnson" is replaced as PERSON_1; the leftover standalone
    # "Mike" is replaced as PERSON_2. Without length-priority ordering,
    # "Mike Johnson" would end up as "[PERSON_2] Johnson".
    assert mapping == {"[PERSON_1]": "Mike Johnson", "[PERSON_2]": "Mike"}
    assert "[PERSON_1]" in redacted
    assert "[PERSON_2]" in redacted


def test_multiple_distinct_known_names():
    shield = PIIShield(known_names=["Alice Brown", "Bob Smith"])
    redacted, mapping = shield.redact("Meeting with Alice Brown and Bob Smith")
    assert "Alice Brown" not in redacted
    assert "Bob Smith" not in redacted
    assert set(mapping.values()) == {"Alice Brown", "Bob Smith"}


def test_known_names_with_punctuation():
    shield = PIIShield(known_names=["O'Brien", "Mary-Jane"])
    redacted, mapping = shield.redact("O'Brien met Mary-Jane at the cafe")
    assert "O'Brien" not in redacted
    assert "Mary-Jane" not in redacted
    assert set(mapping.values()) == {"O'Brien", "Mary-Jane"}


def test_known_names_empty_list_still_redacts_regex_patterns():
    shield = PIIShield(known_names=[])
    _redacted, mapping = shield.redact("Contact john@example.com")
    assert mapping["[EMAIL_1]"] == "john@example.com"


def test_known_names_none_accepted():
    shield = PIIShield(known_names=None)
    _redacted, mapping = shield.redact("Contact john@example.com")
    assert mapping["[EMAIL_1]"] == "john@example.com"


def test_known_name_blank_string_skipped():
    # A blank entry in the known-names list should be silently
    # ignored rather than match the empty string and tokenise nothing.
    shield = PIIShield(known_names=["", "Alice"])
    _redacted, mapping = shield.redact("Alice is here")
    assert mapping == {"[PERSON_1]": "Alice"}


# ---------------------------------------------------------------------------
# Pattern ordering and cross-category interactions
# ---------------------------------------------------------------------------


def test_known_name_priority_over_email():
    # If a caller registers an email literal as a "name", the name
    # pass consumes it first and the email pass is a no-op for that
    # string. The point is determinism: one literal → one token.
    shield = PIIShield(known_names=["mike@example.com"])
    redacted, mapping = shield.redact("Contact mike@example.com for info")
    assert "[PERSON_1]" in redacted
    assert "[EMAIL_" not in redacted
    assert mapping["[PERSON_1]"] == "mike@example.com"


def test_address_redacted_before_phone_does_not_steal_digits():
    # A street address begins with digits; without the address pass
    # running first, the phone regex could chew off the house number
    # and the trailing street digits as a single phone match.
    shield = PIIShield()
    text = "Meet at 1234 Hollywood Boulevard call 555-1234567."

    _redacted, mapping = shield.redact(text)

    # Address redacted intact (with the specific suffix variant).
    address_tokens = [t for t in mapping if t.startswith("[ADDRESS_")]
    assert len(address_tokens) == 1
    assert "Hollywood Boulevard" in mapping[address_tokens[0]]
    # Phone redacted separately.
    phone_tokens = [t for t in mapping if t.startswith("[PHONE_")]
    assert len(phone_tokens) == 1
    assert "555-1234567" in mapping[phone_tokens[0]]


def test_email_digits_not_rematched_as_phone():
    shield = PIIShield()
    _redacted, mapping = shield.redact("Email 123-bob@example.com now")
    # Email wins; phone does not issue a second token from the domain.
    email_tokens = [t for t in mapping if t.startswith("[EMAIL_")]
    assert len(email_tokens) == 1
    phone_tokens = [t for t in mapping if t.startswith("[PHONE_")]
    assert phone_tokens == []


# ---------------------------------------------------------------------------
# Round-trip restore
# ---------------------------------------------------------------------------


def test_restore_round_trip_single_email():
    shield = PIIShield()
    text = "Email me at john@example.com"

    redacted, mapping = shield.redact(text)
    restored = shield.restore(redacted, mapping)

    assert restored == text


def test_restore_round_trip_preserves_context():
    shield = PIIShield()
    text = "Before john@example.com middle +1-555-123-4567 after."

    redacted, mapping = shield.redact(text)
    restored = shield.restore(redacted, mapping)

    assert restored == text


def test_restore_round_trip_full_fixture():
    shield = PIIShield(known_names=["Jane Doe"])
    text = "Jane Doe at jane@acme.com, 500 Oak Drive, +1 (555) 987-6543."

    redacted, mapping = shield.redact(text)

    # No original PII leaks into the redacted text.
    for original in ("Jane Doe", "jane@acme.com", "500 Oak Drive", "+1 (555) 987-6543"):
        assert original not in redacted

    restored = shield.restore(redacted, mapping)
    # Restore reinstates each original verbatim (case preserved for
    # regex-matched values; name case is whatever we registered).
    assert "Jane Doe" in restored
    assert "jane@acme.com" in restored
    assert "500 Oak Drive" in restored
    assert "987-6543" in restored


def test_restore_with_explicit_mapping_leaves_instance_state():
    shield = PIIShield()
    redacted, mapping = shield.redact("Call +1-555-123-4567")

    # Call with an explicit mapping — instance capture must stay
    # intact so the one-time no-arg restore is still available.
    other = PIIShield()
    assert "+1-555-123-4567" in other.restore(redacted, mapping)

    # Instance's own mapping still consumable (one-time).
    assert "+1-555-123-4567" in shield.restore(redacted)


def test_restore_without_mapping_consumes_instance_capture():
    shield = PIIShield()
    redacted, _ = shield.redact("Call +1-555-123-4567")

    first = shield.restore(redacted)
    assert "+1-555-123-4567" in first

    # Second no-arg restore is a no-op (mapping already consumed).
    second = shield.restore(redacted)
    assert second == redacted
    assert "+1-555-123-4567" not in second


def test_restore_multiple_same_token():
    shield = PIIShield(known_names=["Bob"])
    text = "Bob called Bob's office"

    redacted, mapping = shield.redact(text)
    restored = shield.restore(redacted, mapping)

    assert restored == text


def test_restore_empty_text_is_noop():
    shield = PIIShield()
    assert shield.restore("", {"[EMAIL_1]": "x@y.com"}) == ""


def test_restore_with_empty_mapping_is_noop():
    shield = PIIShield()
    assert shield.restore("Some text", {}) == "Some text"


# ---------------------------------------------------------------------------
# Redact edge cases and idempotency
# ---------------------------------------------------------------------------


def test_redact_empty_text():
    shield = PIIShield()
    redacted, mapping = shield.redact("")
    assert redacted == ""
    assert mapping == {}


def test_redact_no_pii_text_unchanged():
    shield = PIIShield()
    text = "The quick brown fox jumps over the lazy dog."
    redacted, mapping = shield.redact(text)
    assert redacted == text
    assert mapping == {}


def test_redact_resets_counters_between_calls():
    shield = PIIShield()
    _, first = shield.redact("Email alice@example.com")
    _, second = shield.redact("Email bob@example.com")
    # Each call's counters restart at 1 — both map [EMAIL_1] to their
    # respective inputs.
    assert first["[EMAIL_1]"] == "alice@example.com"
    assert second["[EMAIL_1]"] == "bob@example.com"
    assert len(first) == 1
    assert len(second) == 1


def test_redact_preserves_whitespace():
    shield = PIIShield()
    text = "Before   john@example.com   after"
    redacted, mapping = shield.redact(text)
    assert "   " in redacted
    restored = shield.restore(redacted, mapping)
    assert restored == text


def test_redact_email_at_sentence_end_keeps_terminator():
    shield = PIIShield()
    redacted, mapping = shield.redact("Contact john@example.com.")
    assert redacted.endswith(".")
    assert mapping["[EMAIL_1]"] == "john@example.com"


def test_issue_token_numbering_per_category():
    # Counters are per-category and reset inside redact(), but we can
    # observe sequential numbering by running through a single call
    # that surfaces multiple tokens per category. Names are picked so
    # they do NOT appear as substrings of the emails — otherwise the
    # case-insensitive known-name pass would consume the email's
    # local-part first and suppress the email token.
    shield = PIIShield(known_names=["Nora", "Elias"])
    text = "Nora wrote to one@x.com and Elias wrote to two@y.com"

    _redacted, mapping = shield.redact(text)

    person_tokens = sorted(t for t in mapping if t.startswith("[PERSON_"))
    email_tokens = sorted(t for t in mapping if t.startswith("[EMAIL_"))
    assert person_tokens == ["[PERSON_1]", "[PERSON_2]"]
    assert email_tokens == ["[EMAIL_1]", "[EMAIL_2]"]


# ---------------------------------------------------------------------------
# No-leak guarantees
# ---------------------------------------------------------------------------


def test_full_fixture_no_literal_leaks():
    shield = PIIShield(known_names=["Dr. Eleanor Vance"])
    text = (
        "Dr. Eleanor Vance lives at 221B Baker Street. "
        "Her email is eleanor.vance@example.org and her phone is +1-617-555-0199. "
        "Secondary contact: support@example.org, 100 Beacon Avenue."
    )

    redacted, mapping = shield.redact(text)

    # Every original PII literal must be absent from the redacted output.
    for original in (
        "Dr. Eleanor Vance",
        "221B Baker Street",
        "eleanor.vance@example.org",
        "+1-617-555-0199",
        "support@example.org",
        "100 Beacon Avenue",
    ):
        assert original not in redacted, f"leaked: {original!r}"

    # And the round-trip works: every literal reappears in the
    # restored text (fidelity for names depends on registration).
    restored = shield.restore(redacted, mapping)
    for original in (
        "Dr. Eleanor Vance",
        "Baker Street",
        "eleanor.vance@example.org",
        "+1-617-555-0199",
        "support@example.org",
        "Beacon Avenue",
    ):
        assert original in restored, f"lost on restore: {original!r}"


# ---------------------------------------------------------------------------
# Contract boundary: v2 does not redact SSN / credit card
# ---------------------------------------------------------------------------


def test_ssn_is_not_redacted_in_v2():
    """v2 drops SSN from the shield contract (see module docstring).

    Callers must strip account-like identifiers upstream. This test
    pins the contract so a future reviewer does not quietly re-add
    the v1 ACCT bucket without updating the contract first.
    """

    shield = PIIShield()
    redacted, mapping = shield.redact("SSN: 123-45-6789 on file.")
    # A PHONE token may match the 7+ digit shape; ACCT must NOT appear.
    assert "[ACCT_" not in redacted
    assert not any(t.startswith("[ACCT_") for t in mapping)


def test_credit_card_is_not_redacted_in_v2():
    shield = PIIShield()
    redacted, mapping = shield.redact("Card on file: 4532 1234 5678 9010")
    assert "[ACCT_" not in redacted
    assert not any(t.startswith("[ACCT_") for t in mapping)
