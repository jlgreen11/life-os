"""
Tests for communication template extraction in RelationshipExtractor.

Verifies that the system correctly learns user writing style per contact/channel
from outbound messages, including greeting/closing patterns, formality levels,
and common vocabulary.
"""

import json
import pytest

from models.core import EventType
from services.signal_extractor.relationship import RelationshipExtractor


def test_template_extraction_basic(db, user_model_store):
    """Test that outbound messages create communication templates."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    event = {
        "id": "evt-001",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": ["alice@example.com"],
            "channel": "email",
            "body": "Hi Alice,\n\nThanks for your email. I'll get back to you soon.\n\nBest regards",
        },
    }

    extractor.extract(event)

    # Verify template was created
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ?",
            ("alice@example.com",)
        ).fetchall()

    assert len(templates) == 1
    template = templates[0]
    assert template["contact_id"] == "alice@example.com"
    assert template["channel"] == "email"
    assert template["greeting"] == "Hi"
    assert template["closing"] == "Best regards"
    assert template["samples_analyzed"] == 1
    assert 0.0 <= template["formality"] <= 1.0


def test_template_formality_detection(db, user_model_store):
    """Test formality score calculation for different writing styles."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    # Formal message
    formal_event = {
        "id": "evt-formal",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": ["prof@university.edu"],
            "channel": "email",
            "body": (
                "Dear Professor Smith,\n\n"
                "I am writing to inquire about the research position. "
                "I have reviewed the requirements carefully and believe my background "
                "in computational linguistics would be a strong fit.\n\n"
                "Sincerely"
            ),
        },
    }

    extractor.extract(formal_event)

    with db.get_connection("user_model") as conn:
        formal_template = conn.execute(
            "SELECT formality FROM communication_templates WHERE contact_id = ?",
            ("prof@university.edu",)
        ).fetchone()

    # Formal message should have high formality score
    assert formal_template["formality"] > 0.6

    # Casual message
    casual_event = {
        "id": "evt-casual",
        "type": EventType.MESSAGE_SENT.value,
        "source": "signal_msg",
        "timestamp": "2026-02-15T10:05:00Z",
        "payload": {
            "to_addresses": ["buddy@example.com"],
            "channel": "signal_msg",
            "body": "Hey! Let's grab coffee later. I'll call you.",
        },
    }

    extractor.extract(casual_event)

    with db.get_connection("user_model") as conn:
        casual_template = conn.execute(
            "SELECT formality FROM communication_templates WHERE contact_id = ?",
            ("buddy@example.com",)
        ).fetchone()

    # Casual message should have low formality score
    assert casual_template["formality"] < 0.5


def test_template_greeting_extraction(db, user_model_store):
    """Test greeting pattern detection."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    test_cases = [
        ("Hi John,\n\nJust wanted to follow up on our conversation.", "Hi"),
        ("Hey there,\nHow are you? I hope all is well.", "Hey"),
        ("Dear Dr. Smith,\nI hope this finds you well and in good spirits.", "Dear"),
        ("Hello!\nGreat to hear from you. Let's catch up soon.", "Hello"),
        ("Good morning,\nThanks for your note. I appreciate it.", "Good morning"),
    ]

    for i, (body, expected_greeting) in enumerate(test_cases):
        event = {
            "id": f"evt-greeting-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": [f"contact{i}@example.com"],
                "channel": "email",
                "body": body,
            },
        }

        extractor.extract(event)

        with db.get_connection("user_model") as conn:
            template = conn.execute(
                "SELECT greeting FROM communication_templates WHERE contact_id = ?",
                (f"contact{i}@example.com",)
            ).fetchone()

        assert template["greeting"] == expected_greeting


def test_template_closing_extraction(db, user_model_store):
    """Test closing pattern detection."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    test_cases = [
        ("Message text here.\n\nThanks", "Thanks"),
        ("Message text here.\n\nThank you so much", "Thank you so much"),
        ("Message text here.\n\nBest regards", "Best regards"),
        ("Message text here.\n\nCheers", "Cheers"),
        ("Message text here.\n\nSincerely", "Sincerely"),
    ]

    for i, (body, expected_closing) in enumerate(test_cases):
        event = {
            "id": f"evt-closing-{i}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": [f"contact{i}@example.com"],
                "channel": "email",
                "body": body,
            },
        }

        extractor.extract(event)

        with db.get_connection("user_model") as conn:
            template = conn.execute(
                "SELECT closing FROM communication_templates WHERE contact_id = ?",
                (f"contact{i}@example.com",)
            ).fetchone()

        assert template["closing"] == expected_closing


def test_template_emoji_detection(db, user_model_store):
    """Test emoji usage tracking."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    # Message with emoji
    event_with_emoji = {
        "id": "evt-emoji",
        "type": EventType.MESSAGE_SENT.value,
        "source": "signal_msg",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": ["friend@example.com"],
            "channel": "signal_msg",
            "body": "Hey! That sounds great 😊 Let's do it!",
        },
    }

    extractor.extract(event_with_emoji)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT uses_emoji FROM communication_templates WHERE contact_id = ?",
            ("friend@example.com",)
        ).fetchone()

    assert template["uses_emoji"] == 1  # True stored as 1

    # Message without emoji
    event_no_emoji = {
        "id": "evt-no-emoji",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": ["work@example.com"],
            "channel": "email",
            "body": "Hello, this is a professional email.",
        },
    }

    extractor.extract(event_no_emoji)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT uses_emoji FROM communication_templates WHERE contact_id = ?",
            ("work@example.com",)
        ).fetchone()

    assert template["uses_emoji"] == 0  # False stored as 0


def test_template_incremental_update(db, user_model_store):
    """Test that templates update incrementally with new samples."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    contact = "alice@example.com"

    # First message
    event1 = {
        "id": "evt-001",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": "Hi Alice,\n\nShort message.\n\nBest",
        },
    }

    extractor.extract(event1)

    with db.get_connection("user_model") as conn:
        template1 = conn.execute(
            "SELECT samples_analyzed, typical_length FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

    assert template1["samples_analyzed"] == 1
    first_length = template1["typical_length"]

    # Second message (much longer)
    event2 = {
        "id": "evt-002",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T11:00:00Z",
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": (
                "Hi Alice,\n\n"
                "This is a much longer message with lots of content. "
                "I wanted to follow up on our discussion about the project timeline. "
                "There are several points I think we should address before moving forward. "
                "First, we need to finalize the technical specifications. "
                "Second, we should align on the delivery schedule. "
                "Third, let's make sure everyone on the team is on the same page.\n\n"
                "Best regards"
            ),
        },
    }

    extractor.extract(event2)

    with db.get_connection("user_model") as conn:
        template2 = conn.execute(
            "SELECT samples_analyzed, typical_length FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

    assert template2["samples_analyzed"] == 2
    # Typical length should have increased (exponential moving average)
    assert template2["typical_length"] > first_length


def test_template_common_phrases(db, user_model_store):
    """Test common phrase extraction and merging."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    contact = "alice@example.com"

    # Send multiple messages with overlapping vocabulary
    messages = [
        "I think the project timeline needs adjustment. The project is ambitious.",
        "The project scope is clear. I think we should proceed with the project plan.",
        "Project milestones look good. The timeline seems reasonable for this project.",
    ]

    for i, body in enumerate(messages):
        event = {
            "id": f"evt-{i:03d}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": f"2026-02-15T{10+i:02d}:00:00Z",
            "payload": {
                "to_addresses": [contact],
                "channel": "email",
                "body": body,
            },
        }
        extractor.extract(event)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT common_phrases FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

    common_phrases = json.loads(template["common_phrases"])
    # "project" should be the most common word
    assert "project" in common_phrases
    assert "timeline" in common_phrases


def test_template_per_channel(db, user_model_store):
    """Test that templates are tracked separately per channel."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    contact = "alice@example.com"

    # Formal email
    email_event = {
        "id": "evt-email",
        "type": EventType.EMAIL_SENT.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": "Dear Alice,\n\nI hope this message finds you well.\n\nSincerely",
        },
    }

    extractor.extract(email_event)

    # Casual message
    msg_event = {
        "id": "evt-msg",
        "type": EventType.MESSAGE_SENT.value,
        "source": "signal_msg",
        "timestamp": "2026-02-15T10:05:00Z",
        "payload": {
            "to_addresses": [contact],
            "channel": "signal_msg",
            "body": "Hey Alice! Quick question...",
        },
    }

    extractor.extract(msg_event)

    # Should have two separate templates
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT channel, formality, greeting FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchall()

    assert len(templates) == 2

    # Find email and message templates
    email_template = next(t for t in templates if t["channel"] == "email")
    msg_template = next(t for t in templates if t["channel"] == "signal_msg")

    # Email should be more formal
    assert email_template["formality"] > msg_template["formality"]
    assert email_template["greeting"] == "Dear"
    assert msg_template["greeting"] == "Hey"


def test_template_skips_short_messages(db, user_model_store):
    """Test that very short messages don't create templates."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    # Very short message
    event = {
        "id": "evt-short",
        "type": EventType.MESSAGE_SENT.value,
        "source": "signal_msg",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "to_addresses": ["alice@example.com"],
            "channel": "signal_msg",
            "body": "ok",  # Too short
        },
    }

    extractor.extract(event)

    # Should not have created a template
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ?",
            ("alice@example.com",)
        ).fetchall()

    assert len(templates) == 0


def test_template_no_extraction_for_inbound(db, user_model_store):
    """Test that inbound messages don't create templates (only outbound do)."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    # Inbound message
    event = {
        "id": "evt-inbound",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "email",
        "timestamp": "2026-02-15T10:00:00Z",
        "payload": {
            "from_address": "sender@example.com",
            "channel": "email",
            "body": "Hi there,\n\nThis is an inbound message.\n\nBest",
        },
    }

    extractor.extract(event)

    # Should not have created a template (only outbound messages create templates)
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ?",
            ("sender@example.com",)
        ).fetchall()

    assert len(templates) == 0


def test_template_example_message_ids(db, user_model_store):
    """Test that example message IDs are tracked and capped at 10."""
    extractor = RelationshipExtractor(db=db, user_model_store=user_model_store)

    contact = "alice@example.com"

    # Send 15 messages
    for i in range(15):
        event = {
            "id": f"evt-{i:03d}",
            "type": EventType.EMAIL_SENT.value,
            "source": "email",
            "timestamp": f"2026-02-15T{10+(i//10):02d}:{(i*4)%60:02d}:00Z",
            "payload": {
                "to_addresses": [contact],
                "channel": "email",
                "body": f"Message number {i}. This is a test message.",
            },
        }
        extractor.extract(event)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT example_message_ids FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

    example_ids = json.loads(template["example_message_ids"])
    # Should cap at 10 most recent
    assert len(example_ids) == 10
    # Should contain the last 10 IDs
    assert "evt-014" in example_ids
    assert "evt-005" in example_ids
    assert "evt-004" not in example_ids  # Too old, should be dropped
