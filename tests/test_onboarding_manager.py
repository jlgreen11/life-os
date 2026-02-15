"""
Comprehensive test coverage for OnboardingManager service.

Tests the voice-first onboarding flow including:
- Flow definition and navigation
- Answer collection and validation
- Free-text parsing (domains, contacts, quiet hours)
- Time parsing with am/pm conversion
- Preference finalization and persistence
- Edge cases and error handling
"""

from datetime import datetime, timezone

import pytest

from services.onboarding.manager import ONBOARDING_PHASES, OnboardingManager


# ---------------------------------------------------------------------------
# Flow Navigation Tests
# ---------------------------------------------------------------------------


def test_get_flow_returns_all_phases(db):
    """get_flow() should return the complete onboarding phase list."""
    manager = OnboardingManager(db)
    flow = manager.get_flow()

    assert isinstance(flow, list)
    assert len(flow) == len(ONBOARDING_PHASES)
    assert flow[0]["id"] == "welcome"
    assert flow[-1]["id"] == "close"


def test_get_current_step_starts_with_first_non_info_step(db):
    """get_current_step() should skip info-only phases and return first question."""
    manager = OnboardingManager(db)
    current = manager.get_current_step()

    # Welcome is info-only, so first step should be morning_style
    assert current is not None
    assert current["id"] == "morning_style"
    assert current["type"] == "choice"


def test_get_current_step_advances_after_answer(db):
    """get_current_step() should advance to next unanswered step after submission."""
    manager = OnboardingManager(db)

    # Answer first step
    manager.submit_answer("morning_style", "minimal")
    current = manager.get_current_step()

    # Should now be on the second question
    assert current["id"] == "tone"


def test_get_current_step_returns_none_when_complete(db):
    """get_current_step() should return None when all required steps are answered."""
    manager = OnboardingManager(db)

    # Answer all required (non-info) steps
    required_steps = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required_steps:
        manager.submit_answer(step["id"], "test_value")

    assert manager.get_current_step() is None


def test_is_complete_false_when_unanswered(db):
    """is_complete() should return False when questions remain unanswered."""
    manager = OnboardingManager(db)
    assert manager.is_complete() is False


def test_is_complete_true_when_all_answered(db):
    """is_complete() should return True when all required questions are answered."""
    manager = OnboardingManager(db)

    required_steps = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required_steps:
        manager.submit_answer(step["id"], "test_value")

    assert manager.is_complete() is True


def test_get_answers_returns_empty_initially(db):
    """get_answers() should return empty dict before any answers are submitted."""
    manager = OnboardingManager(db)
    assert manager.get_answers() == {}


def test_get_answers_returns_submitted_values(db):
    """get_answers() should return all submitted answers."""
    manager = OnboardingManager(db)

    manager.submit_answer("morning_style", "minimal")
    manager.submit_answer("tone", "professional")

    answers = manager.get_answers()
    assert answers["morning_style"] == "minimal"
    assert answers["tone"] == "professional"


# ---------------------------------------------------------------------------
# Domain Parsing Tests
# ---------------------------------------------------------------------------


def test_parse_domains_comma_separated(db):
    """_parse_domains should handle comma-separated lists."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("work, family, health")

    assert len(result) == 3
    assert result[0]["name"] == "work"
    assert result[1]["name"] == "family"
    assert result[2]["name"] == "health"
    assert all(d.get("boundary") == "soft_separation" for d in result)


def test_parse_domains_newline_separated(db):
    """_parse_domains should handle newline-separated lists."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("work\nfamily\nhealth")

    assert len(result) == 3
    assert result[0]["name"] == "work"


def test_parse_domains_bullet_list(db):
    """_parse_domains should strip bullet markers (-, •, *)."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("- work\n• family\n* health")

    assert len(result) == 3
    assert result[0]["name"] == "work"
    assert result[1]["name"] == "family"


def test_parse_domains_mixed_whitespace(db):
    """_parse_domains should handle extra whitespace gracefully."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("  work  ,  family  ,  health  ")

    assert len(result) == 3
    assert result[0]["name"] == "work"


def test_parse_domains_empty_fallback(db):
    """_parse_domains should return default domains when input is empty."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("")

    assert len(result) == 2
    assert result[0]["name"] == "personal"
    assert result[1]["name"] == "work"


def test_parse_domains_whitespace_only_fallback(db):
    """_parse_domains should return defaults when input is only whitespace."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("   \n   ")

    assert len(result) == 2
    assert result[0]["name"] == "personal"


# ---------------------------------------------------------------------------
# Contact Parsing Tests
# ---------------------------------------------------------------------------


def test_parse_contacts_with_relationships(db):
    """_parse_contacts should parse 'Name - relationship' format."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("Sarah - wife, Tom - coworker")

    assert len(result) == 2
    assert result[0]["name"] == "Sarah"
    assert result[0]["relationship"] == "wife"
    assert result[1]["name"] == "Tom"
    assert result[1]["relationship"] == "coworker"


def test_parse_contacts_with_parentheses(db):
    """_parse_contacts should parse 'Name (relationship)' format."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("Sarah (wife)\nTom (coworker)")

    assert len(result) == 2
    assert result[0]["name"] == "Sarah"
    assert result[0]["relationship"] == "wife"


def test_parse_contacts_names_only(db):
    """_parse_contacts should handle names without relationships."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("Sarah, Tom, Mom")

    assert len(result) == 3
    assert result[0]["name"] == "Sarah"
    assert result[0]["relationship"] is None
    assert result[2]["name"] == "Mom"


def test_parse_contacts_bullet_list(db):
    """_parse_contacts should strip bullet markers."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("- Sarah - wife\n• Tom - coworker")

    assert len(result) == 2
    assert result[0]["name"] == "Sarah"


def test_parse_contacts_empty(db):
    """_parse_contacts should return empty list for empty input."""
    manager = OnboardingManager(db)
    result = manager._parse_contacts("")

    assert result == []


# ---------------------------------------------------------------------------
# Quiet Hours Parsing Tests
# ---------------------------------------------------------------------------


def test_parse_quiet_hours_simple_format(db):
    """_parse_quiet_hours should parse '10pm to 7am' format."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10pm to 7am")

    assert len(result) == 1
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"
    assert len(result[0]["days"]) == 7


def test_parse_quiet_hours_with_minutes(db):
    """_parse_quiet_hours should parse times with minutes."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10:30pm to 6:45am")

    assert len(result) == 1
    assert result[0]["start"] == "22:30"
    assert result[0]["end"] == "06:45"


def test_parse_quiet_hours_24hour_format(db):
    """_parse_quiet_hours should parse 24-hour format without am/pm."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("22:00 - 07:00")

    assert len(result) == 1
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"


def test_parse_quiet_hours_12am_midnight_conversion(db):
    """_parse_quiet_hours should correctly convert 12am to 00:00."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("12am to 8am")

    assert result[0]["start"] == "00:00"
    assert result[0]["end"] == "08:00"


def test_parse_quiet_hours_12pm_noon_conversion(db):
    """_parse_quiet_hours should correctly handle 12pm (noon)."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("12pm to 1pm")

    # 12pm should stay as 12, not convert to 24
    assert result[0]["start"] == "12:00"
    assert result[0]["end"] == "13:00"


def test_parse_quiet_hours_decline(db):
    """_parse_quiet_hours should return empty list when user declines."""
    manager = OnboardingManager(db)

    for decline_text in ["no", "don't need quiet hours", "none", "I don't need them"]:
        result = manager._parse_quiet_hours(decline_text)
        assert result == [], f"Failed for: {decline_text}"


def test_parse_quiet_hours_vague_input_gets_default(db):
    """_parse_quiet_hours should provide sensible default for vague input."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("yes")

    # Should get default 10pm-7am
    assert len(result) == 1
    assert result[0]["start"] == "22:00"
    assert result[0]["end"] == "07:00"


def test_parse_quiet_hours_afternoon_example(db):
    """_parse_quiet_hours should handle afternoon times correctly."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("2pm to 5pm")

    assert result[0]["start"] == "14:00"
    assert result[0]["end"] == "17:00"


# ---------------------------------------------------------------------------
# Preference Finalization Tests
# ---------------------------------------------------------------------------


def test_finalize_maps_choice_answers(db):
    """finalize() should map choice answers to preference keys."""
    manager = OnboardingManager(db)

    manager.submit_answer("morning_style", "minimal")
    manager.submit_answer("tone", "professional")
    manager.submit_answer("proactivity", "high")
    manager.submit_answer("autonomy", "moderate")
    manager.submit_answer("drafting", True)
    manager.submit_answer("work_life_boundary", "strict_separation")
    manager.submit_answer("vault", False)
    manager.submit_answer("notifications", "batched")
    manager.submit_answer("domains", "work, family")
    manager.submit_answer("priority_people", "Sarah")
    manager.submit_answer("quiet_hours", "10pm to 7am")

    prefs = manager.finalize()

    assert prefs["verbosity"] == "minimal"
    assert prefs["tone"] == "professional"
    assert prefs["proactivity"] == "high"
    assert prefs["autonomy_level"] == "moderate"
    assert prefs["draft_replies"] is True
    assert prefs["boundary_mode"] == "strict_separation"
    assert prefs["notification_mode"] == "batched"


def test_finalize_parses_domains(db):
    """finalize() should parse free-text domains into structured list."""
    manager = OnboardingManager(db)

    # Provide minimal answers for all required fields
    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "domains":
            manager.submit_answer("domains", "work, family, health")
        elif step["id"] in ["priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    prefs = manager.finalize()

    assert isinstance(prefs["life_domains"], list)
    assert len(prefs["life_domains"]) == 3
    assert prefs["life_domains"][0]["name"] == "work"


def test_finalize_parses_contacts(db):
    """finalize() should parse free-text contacts into structured list."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "priority_people":
            manager.submit_answer("priority_people", "Sarah - wife, Tom - friend")
        elif step["id"] in ["domains", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    prefs = manager.finalize()

    assert isinstance(prefs["priority_contacts"], list)
    assert len(prefs["priority_contacts"]) == 2
    assert prefs["priority_contacts"][0]["name"] == "Sarah"
    assert prefs["priority_contacts"][0]["relationship"] == "wife"


def test_finalize_parses_quiet_hours(db):
    """finalize() should parse free-text quiet hours into time ranges."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "quiet_hours":
            manager.submit_answer("quiet_hours", "10pm to 7am")
        elif step["id"] in ["domains", "priority_people"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    prefs = manager.finalize()

    assert isinstance(prefs["quiet_hours"], list)
    assert len(prefs["quiet_hours"]) == 1
    assert prefs["quiet_hours"][0]["start"] == "22:00"
    assert prefs["quiet_hours"][0]["end"] == "07:00"


def test_finalize_enables_vault_when_requested(db):
    """finalize() should create vaults config when vault_enabled is True."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "vault":
            manager.submit_answer("vault", True)
        elif step["id"] in ["domains", "priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    prefs = manager.finalize()

    # vault_enabled should be converted to vaults list
    assert "vault_enabled" not in prefs
    assert "vaults" in prefs
    assert isinstance(prefs["vaults"], list)
    assert prefs["vaults"][0]["name"] == "Vault"
    assert prefs["vaults"][0]["auth_method"] == "pin"


def test_finalize_skips_vault_when_declined(db):
    """finalize() should not create vaults when vault_enabled is False."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "vault":
            manager.submit_answer("vault", False)
        elif step["id"] in ["domains", "priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    prefs = manager.finalize()

    assert "vault_enabled" not in prefs
    assert "vaults" not in prefs


def test_finalize_persists_to_database(db):
    """finalize() should write all preferences to the database."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "morning_style":
            manager.submit_answer("morning_style", "minimal")
        elif step["id"] in ["domains", "priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    manager.finalize()

    # Verify preferences were written to DB
    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT value FROM user_preferences WHERE key = 'verbosity'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "minimal"


def test_finalize_marks_onboarding_complete(db):
    """finalize() should set onboarding_completed flag in database."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] in ["domains", "priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    manager.finalize()

    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT value FROM user_preferences WHERE key = 'onboarding_completed'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "true"


def test_finalize_json_serializes_complex_values(db):
    """finalize() should JSON-serialize non-string preference values."""
    manager = OnboardingManager(db)

    required = [p for p in ONBOARDING_PHASES if p["type"] != "info"]
    for step in required:
        if step["id"] == "drafting":
            manager.submit_answer("drafting", True)  # Boolean value
        elif step["id"] in ["domains", "priority_people", "quiet_hours"]:
            manager.submit_answer(step["id"], "none")
        else:
            manager.submit_answer(step["id"], "test")

    manager.finalize()

    # Verify boolean was JSON-serialized
    with db.get_connection("preferences") as conn:
        row = conn.execute(
            "SELECT value FROM user_preferences WHERE key = 'draft_replies'"
        ).fetchone()
        assert row is not None
        # Should be JSON serialized
        assert row["value"] in ["true", "True", "1"]  # JSON true variants


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


def test_submit_answer_overwrites_previous_value(db):
    """submit_answer() should allow changing an answer before finalization."""
    manager = OnboardingManager(db)

    manager.submit_answer("morning_style", "minimal")
    manager.submit_answer("morning_style", "detailed")

    assert manager.get_answers()["morning_style"] == "detailed"


def test_finalize_handles_empty_session(db):
    """finalize() should handle empty session gracefully (though shouldn't happen in practice)."""
    manager = OnboardingManager(db)

    # Finalize without answering any questions
    prefs = manager.finalize()

    # Should return empty dict (or defaults), not crash
    assert isinstance(prefs, dict)


def test_parse_domains_case_normalization(db):
    """_parse_domains should normalize domain names to lowercase."""
    manager = OnboardingManager(db)
    result = manager._parse_domains("Work, FAMILY, Health")

    assert result[0]["name"] == "work"
    assert result[1]["name"] == "family"
    assert result[2]["name"] == "health"


def test_quiet_hours_includes_all_weekdays(db):
    """Parsed quiet hours should include all 7 days by default."""
    manager = OnboardingManager(db)
    result = manager._parse_quiet_hours("10pm to 7am")

    assert len(result[0]["days"]) == 7
    assert "monday" in result[0]["days"]
    assert "sunday" in result[0]["days"]
