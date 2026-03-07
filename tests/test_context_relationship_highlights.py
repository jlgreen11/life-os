"""Tests for ContextAssembler._get_relationship_highlights_context().

Verifies that the briefing context includes relationship maintenance data
from the contacts table in entities.db, correctly identifying overdue and
on-track contacts, respecting priority ordering, and capping results.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from services.ai_engine.context import ContextAssembler
from storage.user_model_store import UserModelStore


@pytest.fixture()
def context_assembler(db, event_bus):
    """A ContextAssembler wired to the temporary DatabaseManager."""
    ums = UserModelStore(db, event_bus=event_bus)
    return ContextAssembler(db, ums)


def _insert_contact(db, name, last_contact_days_ago, frequency_days, is_priority=0):
    """Helper to insert a contact with relationship metrics into entities.db."""
    contact_id = str(uuid.uuid4())
    last_contact = (datetime.now(timezone.utc) - timedelta(days=last_contact_days_ago)).isoformat()
    with db.get_connection("entities") as conn:
        conn.execute(
            """INSERT INTO contacts (id, name, last_contact, contact_frequency_days, is_priority)
               VALUES (?, ?, ?, ?, ?)""",
            (contact_id, name, last_contact, frequency_days, is_priority),
        )
    return contact_id


def test_relationship_highlights_overdue_contact(db, context_assembler):
    """A contact whose days-since-contact exceeds 1.5x frequency appears as overdue."""
    _insert_contact(db, "Alice Smith", last_contact_days_ago=15, frequency_days=5)

    result = context_assembler._get_relationship_highlights_context()

    assert "Relationship highlights:" in result
    assert "Alice Smith" in result
    assert "15 days ago" in result
    assert "every 5 days" in result
    assert "overdue" in result


def test_relationship_highlights_on_track_contact(db, context_assembler):
    """A contact with recent last_contact relative to frequency appears as on track."""
    _insert_contact(db, "Bob Jones", last_contact_days_ago=3, frequency_days=7)

    result = context_assembler._get_relationship_highlights_context()

    assert "Bob Jones" in result
    assert "on track" in result


def test_relationship_highlights_empty_when_no_metrics(db, context_assembler):
    """Returns empty string when no contacts have frequency/last_contact metrics."""
    # Insert a contact without metrics
    with db.get_connection("entities") as conn:
        conn.execute(
            "INSERT INTO contacts (id, name) VALUES (?, ?)",
            (str(uuid.uuid4()), "No Metrics Person"),
        )

    result = context_assembler._get_relationship_highlights_context()

    assert result == ""


def test_relationship_highlights_priority_ordering(db, context_assembler):
    """Priority contacts appear before non-priority contacts."""
    _insert_contact(db, "Regular Contact", last_contact_days_ago=20, frequency_days=5, is_priority=0)
    _insert_contact(db, "Priority Contact", last_contact_days_ago=3, frequency_days=7, is_priority=1)

    result = context_assembler._get_relationship_highlights_context()

    # Priority Contact should appear before Regular Contact in the output
    priority_pos = result.index("Priority Contact")
    regular_pos = result.index("Regular Contact")
    assert priority_pos < regular_pos


def test_relationship_highlights_cap_at_8(db, context_assembler):
    """Only the top 8 contacts are included even when more have metrics."""
    for i in range(12):
        _insert_contact(db, f"Contact {i:02d}", last_contact_days_ago=i + 1, frequency_days=3)

    result = context_assembler._get_relationship_highlights_context()

    # Count the number of bullet points (lines starting with "- ")
    contact_lines = [line for line in result.split("\n") if line.startswith("- ")]
    assert len(contact_lines) == 8


def test_relationship_highlights_in_briefing_context(db, context_assembler):
    """Relationship highlights section appears in the full briefing context."""
    _insert_contact(db, "Alice Smith", last_contact_days_ago=15, frequency_days=5)

    briefing = context_assembler.assemble_briefing_context()

    assert "Relationship highlights:" in briefing
    assert "Alice Smith" in briefing
