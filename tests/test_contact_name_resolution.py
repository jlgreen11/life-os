"""
Tests for contact name resolution in InsightEngine.

Verifies that insight summaries use human-readable contact names (resolved from
the entities database) instead of raw email addresses, while preserving email
addresses as the dedup-stable entity field.

Covers:
- _load_contact_name_map(): DB join, empty table, DB error handling
- _display_name(): hit, miss, case-insensitive lookup
- _contact_gap_insights(): name in summary, email preserved as entity
- _inbound_style_insights(): name in summary, email preserved as entity
- _cadence_response_insights(): name in summary, email preserved as entity
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.insight_engine.engine import InsightEngine


# ---------------------------------------------------------------------------
# Helper: insert a contact + identifier into the test DB
# ---------------------------------------------------------------------------

def _insert_contact(db, email: str, name: str) -> str:
    """Insert a contact and its email identifier; return the contact id."""
    contact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO contacts (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (contact_id, name, now, now),
        )
        conn.execute(
            """INSERT INTO contact_identifiers (identifier, identifier_type, contact_id)
               VALUES (?, 'email', ?)""",
            (email, contact_id),
        )
    return contact_id


# ---------------------------------------------------------------------------
# _load_contact_name_map
# ---------------------------------------------------------------------------


def test_load_contact_name_map_empty_table(db, user_model_store):
    """Returns empty dict when no contacts are in the entities DB."""
    engine = InsightEngine(db, user_model_store)
    result = engine._load_contact_name_map()
    assert result == {}


def test_load_contact_name_map_single_contact(db, user_model_store):
    """Returns mapping for a single contact."""
    _insert_contact(db, "alice@example.com", "Alice Smith")
    engine = InsightEngine(db, user_model_store)
    result = engine._load_contact_name_map()
    assert result == {"alice@example.com": "Alice Smith"}


def test_load_contact_name_map_multiple_contacts(db, user_model_store):
    """Returns mappings for all registered contacts."""
    _insert_contact(db, "alice@example.com", "Alice Smith")
    _insert_contact(db, "bob@example.com", "Bob Jones")
    engine = InsightEngine(db, user_model_store)
    result = engine._load_contact_name_map()
    assert result["alice@example.com"] == "Alice Smith"
    assert result["bob@example.com"] == "Bob Jones"


def test_load_contact_name_map_keys_are_lowercased(db, user_model_store):
    """Email keys in the returned map are normalised to lowercase."""
    _insert_contact(db, "Alice@Example.COM", "Alice Smith")
    engine = InsightEngine(db, user_model_store)
    result = engine._load_contact_name_map()
    assert "alice@example.com" in result
    assert result["alice@example.com"] == "Alice Smith"


# ---------------------------------------------------------------------------
# _display_name
# ---------------------------------------------------------------------------


def test_display_name_hit(db, user_model_store):
    """Returns the contact name when a mapping exists."""
    engine = InsightEngine(db, user_model_store)
    name_map = {"alice@example.com": "Alice Smith"}
    assert engine._display_name("alice@example.com", name_map) == "Alice Smith"


def test_display_name_miss(db, user_model_store):
    """Falls back to the raw email address when no mapping exists."""
    engine = InsightEngine(db, user_model_store)
    assert engine._display_name("unknown@example.com", {}) == "unknown@example.com"


def test_display_name_case_insensitive(db, user_model_store):
    """Lookup is case-insensitive (map keys are lowercase)."""
    engine = InsightEngine(db, user_model_store)
    name_map = {"alice@example.com": "Alice Smith"}
    # Profile data may store the address in mixed case
    assert engine._display_name("ALICE@EXAMPLE.COM", name_map) == "Alice Smith"
    assert engine._display_name("Alice@Example.Com", name_map) == "Alice Smith"


def test_display_name_empty_map(db, user_model_store):
    """Falls back gracefully when the map is empty."""
    engine = InsightEngine(db, user_model_store)
    email = "test@example.com"
    assert engine._display_name(email, {}) == email


# ---------------------------------------------------------------------------
# _contact_gap_insights: name resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contact_gap_shows_name_when_contact_exists(db, user_model_store):
    """Summary uses display name when a contact record exists for the email."""
    _insert_contact(db, "charlie@example.com", "Charlie Brown")
    engine = InsightEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    avg_gap_days = 10
    overdue_days = 25  # well past 1.5× threshold

    profile_data = {
        "contacts": {
            "charlie@example.com": {
                "interaction_count": 10,
                "outbound_count": 5,
                "last_interaction": (now - timedelta(days=overdue_days)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=overdue_days + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    # Display name should appear in the summary
    assert "Charlie Brown" in insights[0].summary
    # Raw email should NOT appear in the summary (replaced by name)
    assert "charlie@example.com" not in insights[0].summary
    # Entity must remain as email for stable dedup key
    assert insights[0].entity == "charlie@example.com"


@pytest.mark.asyncio
async def test_contact_gap_falls_back_to_email_when_no_contact(db, user_model_store):
    """Summary falls back to email address when no contact record exists."""
    engine = InsightEngine(db, user_model_store)

    now = datetime.now(timezone.utc)
    avg_gap_days = 10
    overdue_days = 25

    profile_data = {
        "contacts": {
            "nobody@example.com": {
                "interaction_count": 10,
                "outbound_count": 5,
                "last_interaction": (now - timedelta(days=overdue_days)).isoformat(),
                "interaction_timestamps": [
                    (now - timedelta(days=overdue_days + i * avg_gap_days)).isoformat()
                    for i in range(10)
                ],
            }
        }
    }
    user_model_store.update_signal_profile("relationships", profile_data)

    insights = engine._contact_gap_insights()
    assert len(insights) == 1
    assert "nobody@example.com" in insights[0].summary
    assert insights[0].entity == "nobody@example.com"


# ---------------------------------------------------------------------------
# _inbound_style_insights: name resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_style_shows_name_when_contact_exists(db, user_model_store):
    """Style mismatch summary uses display name when a contact record exists."""
    _insert_contact(db, "formal@example.com", "Dr. Jane Formal")
    engine = InsightEngine(db, user_model_store)

    # Outbound baseline: casual user (formality 0.2)
    user_model_store.update_signal_profile("linguistic", {"averages": {"formality": 0.2}})
    # Inbound profile: formal@example.com writes very formally (0.9) — gap = 0.7 > 0.3
    inbound_data = {
        "per_contact_averages": {
            "formal@example.com": {
                "formality": 0.9,
                "samples_count": 10,
            }
        }
    }
    user_model_store.update_signal_profile("linguistic_inbound", inbound_data)

    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert "Dr. Jane Formal" in insights[0].summary
    assert "formal@example.com" not in insights[0].summary
    # Entity preserved as email for stable dedup
    assert insights[0].entity == "formal@example.com"


@pytest.mark.asyncio
async def test_inbound_style_falls_back_to_email_when_no_contact(db, user_model_store):
    """Style mismatch summary falls back to email when no contact record exists."""
    engine = InsightEngine(db, user_model_store)

    user_model_store.update_signal_profile("linguistic", {"averages": {"formality": 0.2}})
    inbound_data = {
        "per_contact_averages": {
            "unknown@example.com": {
                "formality": 0.9,
                "samples_count": 10,
            }
        }
    }
    user_model_store.update_signal_profile("linguistic_inbound", inbound_data)

    insights = engine._inbound_style_insights()
    assert len(insights) == 1
    assert "unknown@example.com" in insights[0].summary
    assert insights[0].entity == "unknown@example.com"


# ---------------------------------------------------------------------------
# _cadence_response_insights: name resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cadence_fastest_contact_shows_name(db, user_model_store):
    """Fast-reply insight summary uses display name when a contact record exists."""
    _insert_contact(db, "fast@example.com", "Speedy Gonzalez")
    engine = InsightEngine(db, user_model_store)

    # Global avg: 3 h (10 800 s).  fast@example.com avg: 600 s (10 min) — well under 50%.
    global_avg = 10800.0
    fast_avg = 600.0

    cadence_data = {
        "response_times": [global_avg] * 15,
        "per_contact_response_times": {
            "fast@example.com": [fast_avg] * 5,
        },
        "per_channel_response_times": {},
        "hourly_activity": {},
    }
    user_model_store.update_signal_profile("cadence", cadence_data)

    insights = engine._cadence_response_insights()

    # Find the fastest_contacts sub-insight
    fast_insights = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast_insights) == 1
    assert "Speedy Gonzalez" in fast_insights[0].summary
    assert "fast@example.com" not in fast_insights[0].summary
    # Entity preserved as email for stable dedup
    assert fast_insights[0].entity == "fast@example.com"


@pytest.mark.asyncio
async def test_cadence_fastest_contact_falls_back_to_email(db, user_model_store):
    """Fast-reply insight summary falls back to email when no contact record exists."""
    engine = InsightEngine(db, user_model_store)

    global_avg = 10800.0
    fast_avg = 600.0

    cadence_data = {
        "response_times": [global_avg] * 15,
        "per_contact_response_times": {
            "anon@example.com": [fast_avg] * 5,
        },
        "per_channel_response_times": {},
        "hourly_activity": {},
    }
    user_model_store.update_signal_profile("cadence", cadence_data)

    insights = engine._cadence_response_insights()
    fast_insights = [i for i in insights if i.category == "fastest_contacts"]
    assert len(fast_insights) == 1
    assert "anon@example.com" in fast_insights[0].summary
    assert fast_insights[0].entity == "anon@example.com"
