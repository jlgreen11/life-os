"""
Tests for bidirectional communication template extraction.

Validates that the RelationshipExtractor correctly extracts templates from
both outbound messages (user to contact) and inbound messages (contact to user).
"""

import pytest
from datetime import datetime, timezone

from services.signal_extractor.relationship import RelationshipExtractor


@pytest.fixture
def relationship_extractor(db, user_model_store):
    """Create a RelationshipExtractor with test database and user model store."""
    return RelationshipExtractor(db, user_model_store)


def test_extract_outbound_template(db, user_model_store, relationship_extractor):
    """
    Outbound messages should create user_to_contact templates.

    When the user sends a message, the extractor should learn the user's
    writing style for that contact-channel pair. This enables style-matching
    when drafting future messages.
    """
    event = {
        "id": "msg-001",
        "type": "email.sent",
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": ["alice@example.com"],
            "channel": "email",
            "body": "Hey Alice,\n\nHope you're doing well! Quick question about the project.\n\nThanks,\nJeremy",
        },
    }

    # Process the outbound message
    signals = relationship_extractor.extract(event)

    # Verify template was created with correct direction
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ? AND context = ?",
            ("alice@example.com", "user_to_contact")
        ).fetchall()

        assert len(templates) == 1
        template = dict(templates[0])

        # Verify extracted features
        assert template["context"] == "user_to_contact"
        assert template["channel"] == "email"
        assert template["greeting"] == "Hey"  # Detected greeting
        assert template["closing"] == "Thanks"  # Detected closing
        assert 0.0 <= template["formality"] <= 1.0  # Formality score computed
        assert template["typical_length"] > 0  # Message length tracked
        assert template["samples_analyzed"] == 1  # First sample


def test_extract_inbound_template(db, user_model_store, relationship_extractor):
    """
    Inbound messages should create contact_to_user templates.

    When the user receives a message, the extractor should learn how that
    contact writes TO the user. This enables style-mismatch detection and
    better understanding of relationship dynamics.
    """
    event = {
        "id": "msg-002",
        "type": "email.received",
        "source": "gmail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": "bob@company.com",
            "channel": "email",
            "body": "Dear Jeremy,\n\nI trust this email finds you well. Per our previous discussion, I wanted to follow up regarding the quarterly report.\n\nBest regards,\nRobert",
        },
    }

    # Process the inbound message
    signals = relationship_extractor.extract(event)

    # Verify template was created with correct direction
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ? AND context = ?",
            ("bob@company.com", "contact_to_user")
        ).fetchall()

        assert len(templates) == 1
        template = dict(templates[0])

        # Verify extracted features
        assert template["context"] == "contact_to_user"
        assert template["channel"] == "email"
        assert template["greeting"] == "Dear"  # Formal greeting
        assert template["closing"] == "Best regards"  # Formal closing
        assert template["formality"] > 0.5  # High formality (formal language)
        assert template["typical_length"] > 0
        assert template["samples_analyzed"] == 1


def test_bidirectional_templates_separate_storage(db, user_model_store, relationship_extractor):
    """
    Outbound and inbound templates should be stored separately.

    The same contact-channel pair should have TWO templates:
    - user_to_contact: how user writes to them
    - contact_to_user: how they write to user

    This enables style comparison and mismatch detection.
    """
    contact = "charlie@example.com"

    # Send a casual message
    outbound_event = {
        "id": "out-001",
        "type": "message.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "slack",
            "body": "hey! yeah that works for me 👍",
        },
    }

    # Receive a formal message from the same contact
    inbound_event = {
        "id": "in-001",
        "type": "message.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": contact,
            "channel": "slack",
            "body": "Hello Jeremy,\n\nThank you for your response. I appreciate your flexibility.\n\nBest regards,\nCharlie",
        },
    }

    # Process both messages
    relationship_extractor.extract(outbound_event)
    relationship_extractor.extract(inbound_event)

    # Verify TWO separate templates exist
    with db.get_connection("user_model") as conn:
        all_templates = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ? ORDER BY context",
            (contact,)
        ).fetchall()

        assert len(all_templates) == 2

        # User's style: casual
        user_template = dict(all_templates[1])  # contact_to_user is second
        assert user_template["context"] == "user_to_contact"
        assert user_template["formality"] < 0.5  # Casual
        assert user_template["uses_emoji"] == 1  # Uses emoji

        # Contact's style: formal
        contact_template = dict(all_templates[0])  # contact_to_user is first alphabetically
        assert contact_template["context"] == "contact_to_user"
        assert contact_template["formality"] > 0.5  # Formal
        assert contact_template["uses_emoji"] == 0  # No emoji


def test_incremental_template_updates(db, user_model_store, relationship_extractor):
    """
    Templates should update incrementally as more samples arrive.

    Multiple messages should blend their features using exponential moving
    average rather than replacing the template each time.
    """
    contact = "diana@example.com"

    # First message: casual
    event1 = {
        "id": "msg-1",
        "type": "email.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": "hey diana, quick question about the project timeline. let me know when you're free to chat!",
        },
    }

    # Second message: slightly more formal
    event2 = {
        "id": "msg-2",
        "type": "email.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": "Hi Diana,\n\nThank you for the update. I have reviewed the timeline and it looks good.\n\nBest,\nJeremy",
        },
    }

    # Process first message
    relationship_extractor.extract(event1)

    with db.get_connection("user_model") as conn:
        template_v1 = conn.execute(
            "SELECT formality, samples_analyzed FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        assert template_v1["samples_analyzed"] == 1
        formality_v1 = template_v1["formality"]

    # Process second message
    relationship_extractor.extract(event2)

    with db.get_connection("user_model") as conn:
        template_v2 = conn.execute(
            "SELECT formality, samples_analyzed FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        assert template_v2["samples_analyzed"] == 2  # Incremented
        formality_v2 = template_v2["formality"]

        # Formality should have increased (blended toward the more formal second message)
        assert formality_v2 > formality_v1


def test_template_extraction_skips_short_messages(db, user_model_store, relationship_extractor):
    """
    Very short messages should not trigger template extraction.

    Messages under 10 characters lack enough signal for style analysis and
    should be ignored to avoid polluting templates with noise.
    """
    event = {
        "id": "short-msg",
        "type": "message.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": ["eve@example.com"],
            "channel": "slack",
            "body": "ok",  # Only 2 characters
        },
    }

    relationship_extractor.extract(event)

    # No template should be created
    with db.get_connection("user_model") as conn:
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM communication_templates WHERE contact_id = ?",
            ("eve@example.com",)
        ).fetchone()["cnt"]

        assert count == 0


def test_template_extraction_emoji_detection(db, user_model_store, relationship_extractor):
    """
    Templates should track emoji usage accurately.

    If a user uses emoji with a contact, the template should mark uses_emoji=1
    and keep it sticky (once true, stays true even if later messages don't have emoji).
    """
    contact = "frank@example.com"

    # First message with emoji
    event1 = {
        "id": "emoji-1",
        "type": "message.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "slack",
            "body": "sounds good! 😊",
        },
    }

    relationship_extractor.extract(event1)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT uses_emoji FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        assert template["uses_emoji"] == 1  # Emoji detected

    # Second message without emoji
    event2 = {
        "id": "emoji-2",
        "type": "message.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "slack",
            "body": "thanks for the update",
        },
    }

    relationship_extractor.extract(event2)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT uses_emoji FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        # Emoji flag should still be true (sticky)
        assert template["uses_emoji"] == 1


def test_template_extraction_common_phrases(db, user_model_store, relationship_extractor):
    """
    Templates should track frequently used words/phrases.

    The common_phrases list should accumulate the most frequent significant
    words across all messages to that contact (top 10).
    """
    contact = "grace@example.com"

    # Send multiple messages with repeated words
    for i in range(3):
        event = {
            "id": f"phrase-{i}",
            "type": "email.sent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "to_addresses": [contact],
                "channel": "email",
                "body": f"Hi Grace, quick update on the project timeline and budget discussions. Project progressing well.",
            },
        }
        relationship_extractor.extract(event)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT common_phrases FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        import json
        phrases = json.loads(template["common_phrases"])

        # "project" should be in the top phrases (appeared 6 times)
        assert "project" in phrases
        # Stop words like "the", "on", "and" should be excluded
        assert "the" not in phrases


def test_template_extraction_multiple_recipients(db, user_model_store, relationship_extractor):
    """
    Multi-recipient messages should create separate templates per recipient.

    When a user sends one email to multiple people, each recipient should
    get their own template update (since relationships are independent).
    """
    event = {
        "id": "multi-recv",
        "type": "email.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": ["henry@example.com", "iris@example.com"],
            "channel": "email",
            "body": "Hi team,\n\nJust a quick update on the project status.\n\nBest,\nJeremy",
        },
    }

    relationship_extractor.extract(event)

    # Both recipients should have templates
    with db.get_connection("user_model") as conn:
        henry_template = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ?",
            ("henry@example.com",)
        ).fetchone()

        iris_template = conn.execute(
            "SELECT * FROM communication_templates WHERE contact_id = ?",
            ("iris@example.com",)
        ).fetchone()

        assert henry_template is not None
        assert iris_template is not None

        # Both should have the same style features (same message)
        assert henry_template["formality"] == iris_template["formality"]
        assert henry_template["greeting"] == iris_template["greeting"]


def test_template_extraction_preserves_example_ids(db, user_model_store, relationship_extractor):
    """
    Templates should maintain a rolling window of example message IDs.

    The last 10 message IDs should be stored for provenance and context
    retrieval (so the user can see what messages informed the template).
    """
    contact = "jack@example.com"

    # Send 15 messages (more than the 10-message cap)
    for i in range(15):
        event = {
            "id": f"example-{i:02d}",
            "type": "email.sent",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "to_addresses": [contact],
                "channel": "email",
                "body": f"Hi Jack, this is message number {i} with enough content for extraction.",
            },
        }
        relationship_extractor.extract(event)

    with db.get_connection("user_model") as conn:
        template = conn.execute(
            "SELECT example_message_ids FROM communication_templates WHERE contact_id = ?",
            (contact,)
        ).fetchone()

        import json
        example_ids = json.loads(template["example_message_ids"])

        # Should have exactly 10 IDs (the last 10)
        assert len(example_ids) == 10
        # Should contain the most recent messages
        assert "example-14" in example_ids
        assert "example-05" in example_ids
        # Should NOT contain the oldest messages
        assert "example-00" not in example_ids
        assert "example-01" not in example_ids


def test_template_extraction_channel_separation(db, user_model_store, relationship_extractor):
    """
    Templates should be channel-specific.

    The same contact may have different communication styles across channels
    (formal on email, casual on Slack). Each channel should have its own template.
    """
    contact = "karen@example.com"

    # Formal email
    email_event = {
        "id": "email-001",
        "type": "email.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "email",
            "body": "Dear Karen,\n\nI wanted to follow up on our previous discussion regarding the project timeline.\n\nBest regards,\nJeremy",
        },
    }

    # Casual Slack message
    slack_event = {
        "id": "slack-001",
        "type": "message.sent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "to_addresses": [contact],
            "channel": "slack",
            "body": "hey karen! quick q about the timeline, got a sec?",
        },
    }

    relationship_extractor.extract(email_event)
    relationship_extractor.extract(slack_event)

    # Should have two separate templates
    with db.get_connection("user_model") as conn:
        templates = conn.execute(
            "SELECT channel, formality, greeting FROM communication_templates WHERE contact_id = ? ORDER BY channel",
            (contact,)
        ).fetchall()

        assert len(templates) == 2

        # Email template: formal
        email_template = dict(templates[0])
        assert email_template["channel"] == "email"
        assert email_template["formality"] > 0.5
        assert email_template["greeting"] == "Dear"

        # Slack template: casual
        slack_template = dict(templates[1])
        assert slack_template["channel"] == "slack"
        assert slack_template["formality"] < 0.5
        assert slack_template["greeting"] == "hey"


def test_inbound_template_formality_detection(db, user_model_store, relationship_extractor):
    """
    Inbound templates should correctly assess formality from received messages.

    This enables the system to understand how each contact communicates and
    potentially match their style in responses.
    """
    # Very formal inbound message
    formal_event = {
        "id": "formal-in",
        "type": "email.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": "lawyer@lawfirm.com",
            "channel": "email",
            "body": "Dear Mr. Greenwood,\n\nPursuant to our previous correspondence, I am writing to provide you with an update regarding the aforementioned matter.\n\nSincerely,\nAttorney Smith",
        },
    }

    # Very casual inbound message
    casual_event = {
        "id": "casual-in",
        "type": "message.received",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "from_address": "buddy@example.com",
            "channel": "slack",
            "body": "yo! wanna grab lunch later? haha let me know",
        },
    }

    relationship_extractor.extract(formal_event)
    relationship_extractor.extract(casual_event)

    with db.get_connection("user_model") as conn:
        formal_template = conn.execute(
            "SELECT formality FROM communication_templates WHERE contact_id = ?",
            ("lawyer@lawfirm.com",)
        ).fetchone()

        casual_template = conn.execute(
            "SELECT formality FROM communication_templates WHERE contact_id = ?",
            ("buddy@example.com",)
        ).fetchone()

        # Formal message should have high formality score
        assert formal_template["formality"] > 0.7

        # Casual message should have low formality score
        assert casual_template["formality"] < 0.3
