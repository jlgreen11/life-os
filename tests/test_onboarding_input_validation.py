"""
Onboarding input validation: hardening tests.

Covers the robustness guards added to ``OnboardingManager``'s free-text
parsers so that pathological user input (5,000-word paragraphs, single
relationship words, multi-range quiet hours) cannot corrupt
``UserPreferences``.

The three parsers under test are deterministic — no LLM involvement —
so every assertion below is a strict equality / boundary check.
"""

from services.onboarding.manager import OnboardingManager

# ---------------------------------------------------------------------------
# _parse_domains hardening
# ---------------------------------------------------------------------------


def test_parse_domains_caps_at_twelve_entries(db):
    """A long list should be truncated to _MAX_DOMAINS (12) entries."""
    manager = OnboardingManager(db)
    # 20 unique domain names — should be capped to 12
    text = ", ".join(f"area{i}" for i in range(20))

    result = manager._parse_domains(text)

    assert len(result) == 12
    # Preserves first-seen order
    assert result[0]["name"] == "area0"
    assert result[-1]["name"] == "area11"


def test_parse_domains_truncates_over_long_names(db):
    """A single domain name longer than _MAX_DOMAIN_NAME_LEN is truncated."""
    manager = OnboardingManager(db)
    long_name = "a" * 100

    result = manager._parse_domains(long_name)

    assert len(result) == 1
    # Truncated to 40 chars
    assert result[0]["name"] == "a" * 40
    assert len(result[0]["name"]) == 40


def test_parse_domains_dedupes_preserving_order(db):
    """Repeated domain names are deduped while preserving first-seen order."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("work, family, work, health, family, work")

    names = [d["name"] for d in result]
    assert names == ["work", "family", "health"]


def test_parse_domains_dedupes_case_insensitively(db):
    """Domains are lowercased before dedup so 'Work' and 'WORK' collapse."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("Work, WORK, work")

    assert len(result) == 1
    assert result[0]["name"] == "work"


def test_parse_domains_rejects_junk_tokens(db):
    """Refusal tokens like 'na' / 'none' / 'n/a' aren't stored as domains."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("work, na, none, n/a, family, nothing")

    names = [d["name"] for d in result]
    assert names == ["work", "family"]


def test_parse_domains_skips_empty_after_strip(db):
    """Whitespace-only or bullet-only entries are skipped."""
    manager = OnboardingManager(db)
    # Trailing/leading commas + bullet-only entries
    result = manager._parse_domains("work, , -, *, family")

    names = [d["name"] for d in result]
    assert names == ["work", "family"]


def test_parse_domains_paragraph_does_not_explode(db):
    """A 5,000-char single-line paragraph becomes at most one truncated domain."""
    manager = OnboardingManager(db)
    paragraph = "x" * 5000

    result = manager._parse_domains(paragraph)

    # Single line, single domain, truncated to the cap.
    assert len(result) == 1
    assert len(result[0]["name"]) == 40


# ---------------------------------------------------------------------------
# _parse_contacts hardening
# ---------------------------------------------------------------------------


def test_contact_with_only_relationship_phrase_is_dropped(db):
    """A line that is JUST a relationship phrase yields no contact entry.

    Previously, ``_parse_contacts("wife")`` returned a contact literally
    named "wife" — a bug that polluted UserPreferences with non-people.
    The fix: ``_extract_name_and_relationship`` returns (None, relationship)
    when the phrase exhausts the input, and ``_parse_contacts`` drops it.
    """
    manager = OnboardingManager(db)

    for label in ("wife", "Mom", "mom", "husband", "boss", "best friend"):
        result = manager._parse_contacts(label)
        assert result == [], f"expected empty for {label!r}, got {result!r}"


def test_contact_relationship_only_within_list_is_dropped(db):
    """Mixed input: real names are kept, bare relationship phrases dropped."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("Sarah - wife, wife, Tom - coworker, Mom")

    # "wife" and "Mom" entries are dropped; Sarah and Tom remain.
    names = [c["name"] for c in result]
    assert names == ["Sarah", "Tom"]


def test_contact_rejects_over_long_name(db):
    """Contact names exceeding _MAX_CONTACT_NAME_LEN (80) are rejected."""
    manager = OnboardingManager(db)
    long_name = "x" * 100  # Far over the 80-char cap

    result = manager._parse_contacts(f"Sarah - wife, {long_name} - friend, Tom - coworker")

    names = [c["name"] for c in result]
    assert names == ["Sarah", "Tom"]


def test_contact_list_capped_at_fifty(db):
    """A pathological 100-entry list is capped at _MAX_CONTACTS (50).

    We use explicit "Name - role" form so each entry has a separator and
    sidesteps the natural-language relationship-phrase fallback (which
    would otherwise mangle words like "Person" because it contains "son").
    """
    manager = OnboardingManager(db)
    text = ", ".join(f"Alice{i:03d} - friend" for i in range(100))

    result = manager._parse_contacts(text)

    assert len(result) == 50
    # Cap kicks in after the first 50 names — preserves order
    assert result[0]["name"] == "Alice000"
    assert result[-1]["name"] == "Alice049"


def test_contact_with_natural_language_still_works(db):
    """Hardening doesn't regress the existing 'my wife Sarah' parser."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("Nate my brother in law, my wife Sarah")

    assert len(result) == 2
    assert result[0] == {"name": "Nate", "relationship": "brother-in-law"}
    assert result[1] == {"name": "Sarah", "relationship": "wife"}


# ---------------------------------------------------------------------------
# _parse_quiet_hours multi-range + day qualifiers
# ---------------------------------------------------------------------------


def test_quiet_hours_two_ranges_no_qualifier(db):
    """Two time ranges separated by a comma produce two ranges, all days."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10pm to 7am, 11pm to 9am")

    assert len(result) == 2
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"
    assert result[1]["start"] == "23:00"
    assert result[1]["end"] == "09:00"
    # Neither range had a day qualifier — both default to every day
    for r in result:
        assert len(r["days"]) == 7


def test_quiet_hours_weekday_qualifier_attaches_days(db):
    """'weekdays 10pm to 7am' produces a range scoped to Mon-Fri.

    The notification manager's ``_is_quiet_hours`` compares against
    ``strftime("%A").lower()`` (full day names), so we emit full names
    rather than the abbreviations in the original task description.
    """
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("weekdays 10pm to 7am")

    assert len(result) == 1
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"
    assert result[0]["days"] == [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    ]


def test_quiet_hours_weekday_and_weekend_split(db):
    """Two ranges each get their nearest day qualifier."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("weekdays 10pm to 7am, weekends 11pm to 9am")

    assert len(result) == 2
    # First range: weekdays
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"
    assert result[0]["days"] == [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    ]
    # Second range: weekends
    assert result[1]["start"] == "23:00"
    assert result[1]["end"] == "09:00"
    assert result[1]["days"] == ["saturday", "sunday"]


def test_quiet_hours_trailing_qualifier(db):
    """Qualifier after the time range also binds to it."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10pm to 7am weekends")

    assert len(result) == 1
    assert result[0]["days"] == ["saturday", "sunday"]


def test_quiet_hours_specific_day_abbreviation(db):
    """Day abbreviations like 'sat' bind to the specific day."""
    manager = OnboardingManager(db)
    # Word-boundary regex — 'sat' shouldn't match inside 'saturday' twice.
    result = manager._parse_quiet_hours("sat 10pm to 7am")

    assert len(result) == 1
    assert result[0]["days"] == ["saturday"]


def test_quiet_hours_qualifier_outside_window_is_ignored(db):
    """A day word far away from the time range doesn't bind to it."""
    manager = OnboardingManager(db)
    # 'weekends' is buried far before the time — outside the ±30 window
    padding = "x" * 60
    text = f"weekends {padding} 10pm to 7am"

    result = manager._parse_quiet_hours(text)

    assert len(result) == 1
    # Falls back to all days because the qualifier was out of range
    assert len(result[0]["days"]) == 7


def test_quiet_hours_single_range_back_compat(db):
    """The simple '10pm to 7am' case still produces a 7-day range."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10pm to 7am")

    assert len(result) == 1
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"
    assert len(result[0]["days"]) == 7
