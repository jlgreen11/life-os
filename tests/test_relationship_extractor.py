"""
Comprehensive test coverage for RelationshipExtractor.

The RelationshipExtractor builds the relationship graph by tracking interaction
patterns, computing reciprocity ratios, and extracting communication style
templates. Tests cover:

1. Event filtering (can_process)
2. Basic signal extraction (inbound/outbound messages)
3. Contact profile updates (incremental stats, ring buffers)
4. Communication template extraction (greeting/closing detection, formality)
5. Edge cases (empty messages, multi-recipient, missing fields)
"""

import hashlib

import pytest

from models.core import EventType
from services.signal_extractor.relationship import RelationshipExtractor


class TestRelationshipExtractor:
    """Test suite for RelationshipExtractor."""

    @pytest.fixture
    def extractor(self, db, user_model_store):
        """Create a RelationshipExtractor instance with test dependencies."""
        return RelationshipExtractor(db=db, user_model_store=user_model_store)

    # -------------------------------------------------------------------------
    # Event Filtering Tests
    # -------------------------------------------------------------------------

    def test_can_process_accepts_email_received(self, extractor):
        """Verify extractor accepts email.received events."""
        event = {"type": EventType.EMAIL_RECEIVED.value}
        assert extractor.can_process(event) is True

    def test_can_process_accepts_email_sent(self, extractor):
        """Verify extractor accepts email.sent events."""
        event = {"type": EventType.EMAIL_SENT.value}
        assert extractor.can_process(event) is True

    def test_can_process_accepts_message_received(self, extractor):
        """Verify extractor accepts message.received events."""
        event = {"type": EventType.MESSAGE_RECEIVED.value}
        assert extractor.can_process(event) is True

    def test_can_process_accepts_message_sent(self, extractor):
        """Verify extractor accepts message.sent events."""
        event = {"type": EventType.MESSAGE_SENT.value}
        assert extractor.can_process(event) is True

    def test_can_process_rejects_calendar_events(self, extractor):
        """Verify extractor rejects non-communication events."""
        event = {"type": EventType.CALENDAR_EVENT_CREATED.value}
        assert extractor.can_process(event) is False

    def test_can_process_rejects_system_events(self, extractor):
        """Verify extractor rejects system events."""
        event = {"type": EventType.CONNECTOR_SYNC_COMPLETE.value}
        assert extractor.can_process(event) is False

    # -------------------------------------------------------------------------
    # Signal Extraction Tests
    # -------------------------------------------------------------------------

    def test_extract_inbound_email_signal(self, extractor):
        """Verify signal extraction from inbound email."""
        event = {
            "id": "evt-001",
            "type": EventType.EMAIL_RECEIVED.value,
            "source": "gmail",
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "alice@example.com",
                "subject": "Project Update",
                "body": "Hi, here's the latest on the project...",
                "channel": "email",
                "is_reply": False,
                "action_items": ["Review slides"],
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 1
        signal = signals[0]
        assert signal["type"] == "relationship_interaction"
        assert signal["contact_address"] == "alice@example.com"
        assert signal["direction"] == "inbound"
        assert signal["channel"] == "email"
        assert signal["message_length"] > 0  # Message has content
        assert signal["has_action_items"] is True
        assert signal["is_reply"] is False

    def test_extract_outbound_email_signal(self, extractor):
        """Verify signal extraction from outbound email."""
        event = {
            "id": "evt-002",
            "type": EventType.EMAIL_SENT.value,
            "source": "gmail",
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "to_addresses": ["bob@example.com"],
                "subject": "Re: Project Update",
                "body": "Thanks for the update! Looks great.",
                "channel": "email",
                "is_reply": True,
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 1
        signal = signals[0]
        assert signal["contact_address"] == "bob@example.com"
        assert signal["direction"] == "outbound"
        assert signal["message_length"] == 35
        assert signal["is_reply"] is True

    def test_extract_multi_recipient_email(self, extractor):
        """Verify extractor creates separate signals for each recipient."""
        event = {
            "id": "evt-003",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T12:00:00Z",
            "payload": {
                "to_addresses": ["alice@example.com", "bob@example.com", "charlie@example.com"],
                "body": "Team meeting at 3pm",
                "channel": "email",
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 3
        addresses = [s["contact_address"] for s in signals]
        assert "alice@example.com" in addresses
        assert "bob@example.com" in addresses
        assert "charlie@example.com" in addresses
        # All signals should have identical properties except contact_address
        assert all(s["direction"] == "outbound" for s in signals)
        assert all(s["message_length"] > 0 for s in signals)

    def test_extract_handles_missing_body(self, extractor):
        """Verify extractor handles events with missing message body."""
        event = {
            "id": "evt-004",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T13:00:00Z",
            "payload": {
                "from_address": "dave@example.com",
                "subject": "Empty message",
                # No body field
                "channel": "email",
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 1
        assert signals[0]["message_length"] == 0

    def test_extract_handles_missing_addresses(self, extractor):
        """Verify extractor gracefully handles events without sender/recipients."""
        event = {
            "id": "evt-005",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T14:00:00Z",
            "payload": {
                # Missing to_addresses
                "body": "Test message",
                "channel": "email",
            },
        }

        signals = extractor.extract(event)

        # Should return empty list when no valid addresses
        assert len(signals) == 0

    # -------------------------------------------------------------------------
    # Contact Profile Update Tests
    # -------------------------------------------------------------------------

    def test_contact_profile_creation(self, extractor):
        """Verify contact profile is created on first interaction."""
        event = {
            "id": "evt-006",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "newcontact@example.com",
                "body": "Hello!",
                "channel": "email",
            },
        }

        extractor.extract(event)

        # Verify profile was created
        profile = extractor.ums.get_signal_profile("relationships")
        assert profile is not None
        assert "newcontact@example.com" in profile["data"]["contacts"]

        contact = profile["data"]["contacts"]["newcontact@example.com"]
        assert contact["interaction_count"] == 1
        assert contact["inbound_count"] == 1
        assert contact["outbound_count"] == 0
        assert contact["channels_used"] == ["email"]
        assert contact["last_interaction"] == "2026-02-15T10:00:00Z"

    def test_contact_profile_incremental_update(self, extractor):
        """Verify contact profile updates incrementally with new interactions."""
        # First interaction
        event1 = {
            "id": "evt-007",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "alice@example.com",
                "body": "First message",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        # Second interaction (outbound)
        event2 = {
            "id": "evt-008",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "to_addresses": ["alice@example.com"],
                "body": "Reply message",
                "channel": "email",
            },
        }
        extractor.extract(event2)

        # Verify profile was updated
        profile = extractor.ums.get_signal_profile("relationships")
        contact = profile["data"]["contacts"]["alice@example.com"]
        assert contact["interaction_count"] == 2
        assert contact["inbound_count"] == 1
        assert contact["outbound_count"] == 1
        assert contact["last_interaction"] == "2026-02-15T11:00:00Z"

    def test_contact_profile_multi_channel(self, extractor):
        """Verify multi-channel tracking for the same contact."""
        # Email interaction
        event1 = {
            "id": "evt-009",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "bob@example.com",
                "body": "Email message",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        # Slack interaction with same contact
        event2 = {
            "id": "evt-010",
            "type": EventType.MESSAGE_RECEIVED.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "from_address": "bob@example.com",
                "body": "Slack message",
                "channel": "slack",
            },
        }
        extractor.extract(event2)

        # Verify channels are tracked
        profile = extractor.ums.get_signal_profile("relationships")
        contact = profile["data"]["contacts"]["bob@example.com"]
        assert contact["interaction_count"] == 2
        assert "email" in contact["channels_used"]
        assert "slack" in contact["channels_used"]
        assert len(contact["channels_used"]) == 2

    def test_contact_profile_avg_message_length(self, extractor):
        """Verify incremental running average for message length."""
        # First message: 100 chars
        event1 = {
            "id": "evt-011",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "charlie@example.com",
                "body": "x" * 100,
                "channel": "email",
            },
        }
        extractor.extract(event1)

        profile = extractor.ums.get_signal_profile("relationships")
        contact = profile["data"]["contacts"]["charlie@example.com"]
        assert contact["avg_message_length"] == 100.0

        # Second message: 200 chars (average should be 150)
        event2 = {
            "id": "evt-012",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "from_address": "charlie@example.com",
                "body": "y" * 200,
                "channel": "email",
            },
        }
        extractor.extract(event2)

        profile = extractor.ums.get_signal_profile("relationships")
        contact = profile["data"]["contacts"]["charlie@example.com"]
        assert contact["avg_message_length"] == 150.0

    def test_contact_profile_timestamp_ring_buffer(self, extractor):
        """Verify interaction timestamps are capped at 100 entries."""
        # Create 120 interactions
        for i in range(120):
            event = {
                "id": f"evt-{i:03d}",
                "type": EventType.EMAIL_RECEIVED.value,
                "timestamp": f"2026-02-15T{i % 24:02d}:00:00Z",
                "payload": {
                    "from_address": "spammer@example.com",
                    "body": f"Message {i}",
                    "channel": "email",
                },
            }
            extractor.extract(event)

        # Verify only last 100 timestamps are kept
        profile = extractor.ums.get_signal_profile("relationships")
        contact = profile["data"]["contacts"]["spammer@example.com"]
        assert len(contact["interaction_timestamps"]) == 100
        assert contact["interaction_count"] == 120  # Count is still correct

    # -------------------------------------------------------------------------
    # Communication Template Extraction Tests
    # -------------------------------------------------------------------------

    def test_template_extraction_greeting_hi(self, extractor):
        """Verify greeting extraction for 'Hi' pattern."""
        event = {
            "id": "evt-013",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["alice@example.com"],
                "body_plain": "Hi Alice,\n\nJust following up on our meeting.\n\nBest,\nUser",
                "channel": "email",
            },
        }

        extractor.extract(event)

        # Verify template was created
        template_id = hashlib.sha256(b"alice@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT greeting FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        assert row is not None
        # Extractor captures the greeting pattern, which may vary slightly
        assert row["greeting"].startswith("Hi")

    def test_template_extraction_greeting_dear(self, extractor):
        """Verify greeting extraction for 'Dear' pattern."""
        event = {
            "id": "evt-014",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["bob@example.com"],
                "body_plain": "Dear Bob,\n\nThank you for your inquiry.\n\nSincerely,\nUser",
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"bob@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT greeting FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        # Extractor captures the greeting pattern, which may vary slightly
        assert row["greeting"].startswith("Dear")

    def test_template_extraction_closing_best_regards(self, extractor):
        """Verify closing extraction for 'Best regards' pattern."""
        event = {
            "id": "evt-015",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["charlie@example.com"],
                "body_plain": "Hi Charlie,\n\nPlease find attached.\n\nBest regards,\nUser",
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"charlie@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT closing FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        assert row["closing"] == "Best regards"

    def test_template_extraction_closing_thanks(self, extractor):
        """Verify closing extraction for 'Thanks' pattern."""
        event = {
            "id": "evt-016",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["dave@example.com"],
                "body_plain": "Hey Dave,\n\nCan you send me the file?\n\nThanks!",
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"dave@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT closing FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        # Extractor captures the closing pattern
        assert row["closing"].startswith("Thanks")

    def test_template_extraction_formality_formal(self, extractor):
        """Verify formality calculation for formal messages."""
        event = {
            "id": "evt-017",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["formal@example.com"],
                "body_plain": (
                    "Dear Sir or Madam,\n\n"
                    "I am writing to formally request information regarding the proposed "
                    "collaboration between our organizations. I would appreciate the "
                    "opportunity to discuss this matter further at your earliest convenience.\n\n"
                    "Sincerely,\nUser"
                ),
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"formal@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT formality FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        # Formal message should have high formality score (> 0.7)
        assert row["formality"] > 0.7

    def test_template_extraction_formality_casual(self, extractor):
        """Verify formality calculation for casual messages."""
        event = {
            "id": "evt-018",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["casual@example.com"],
                "body_plain": (
                    "Hey!\n\n"
                    "What's up? I'm thinking we should grab coffee sometime. "
                    "Let me know if you're free!\n\n"
                    "Cheers"
                ),
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"casual@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT formality FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        # Casual message should have low formality score (< 0.4)
        assert row["formality"] < 0.4

    def test_template_extraction_emoji_detection(self, extractor):
        """Verify emoji usage detection in templates."""
        event = {
            "id": "evt-019",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["emoji@example.com"],
                "body_plain": "Hey! That sounds great 😊 Let's do it!",
                "channel": "email",
            },
        }

        extractor.extract(event)

        template_id = hashlib.sha256(b"emoji@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT uses_emoji FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        assert row["uses_emoji"] == 1  # SQLite stores boolean as int

    def test_template_extraction_incremental_update(self, extractor):
        """Verify template updates incrementally with exponential moving average."""
        # First message: formal (long, no contractions)
        event1 = {
            "id": "evt-020",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["evolving@example.com"],
                "body_plain": (
                    "Dear Colleague,\n\n"
                    "I am writing to inform you of the upcoming changes to our "
                    "organizational structure.\n\n"
                    "Best regards"
                ),
                "channel": "email",
            },
        }
        extractor.extract(event1)

        template_id = hashlib.sha256(b"evolving@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row1 = conn.execute(
                "SELECT formality, samples_analyzed FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        initial_formality = row1["formality"]
        assert row1["samples_analyzed"] == 1

        # Second message: casual (contractions, short)
        event2 = {
            "id": "evt-021",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "to_addresses": ["evolving@example.com"],
                "body_plain": "Hey! What's up? Let's chat later.",
                "channel": "email",
            },
        }
        extractor.extract(event2)

        with extractor.db.get_connection("user_model") as conn:
            row2 = conn.execute(
                "SELECT formality, samples_analyzed FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        # Formality should have decreased (blended with casual message)
        assert row2["formality"] < initial_formality
        assert row2["samples_analyzed"] == 2

    def test_template_extraction_common_phrases(self, extractor):
        """Verify common phrase tracking across messages."""
        # First message with specific vocabulary
        event1 = {
            "id": "evt-022",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["phrases@example.com"],
                "body_plain": "Thanks for the update on the project timeline and deliverables.",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        # Second message with overlapping vocabulary
        event2 = {
            "id": "evt-023",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "to_addresses": ["phrases@example.com"],
                "body_plain": "The project is progressing well. Thanks for checking.",
                "channel": "email",
            },
        }
        extractor.extract(event2)

        template_id = hashlib.sha256(b"phrases@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT common_phrases FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        import json
        phrases = json.loads(row["common_phrases"])

        # "thanks" and "project" should appear in top phrases (appeared twice)
        assert "thanks" in phrases
        assert "project" in phrases

    def test_template_extraction_skips_short_messages(self, extractor):
        """Verify template extraction is skipped for very short messages."""
        event = {
            "id": "evt-024",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["short@example.com"],
                "body_plain": "Ok",  # Too short for style analysis
                "channel": "email",
            },
        }

        extractor.extract(event)

        # Verify no template was created
        template_id = hashlib.sha256(b"short@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row = conn.execute(
                "SELECT id FROM communication_templates WHERE id = ?",
                (template_id,)
            ).fetchone()

        assert row is None

    def test_template_extraction_uses_body_plain_fallback(self, extractor):
        """Verify template extraction uses body_plain or falls back to body."""
        # Event with body_plain
        event1 = {
            "id": "evt-025",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["fallback1@example.com"],
                "body_plain": "Hi,\n\nThis is the plain text version.\n\nBest",
                "body": "<html>HTML version</html>",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        template_id1 = hashlib.sha256(b"fallback1@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row1 = conn.execute(
                "SELECT greeting FROM communication_templates WHERE id = ?",
                (template_id1,)
            ).fetchone()

        # Should extract from body_plain (not HTML)
        assert row1["greeting"].startswith("Hi")

        # Event with only body (no body_plain)
        event2 = {
            "id": "evt-026",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["fallback2@example.com"],
                "body": "Hey there,\n\nThis is the only version.\n\nCheers",
                "channel": "email",
            },
        }
        extractor.extract(event2)

        template_id2 = hashlib.sha256(b"fallback2@example.com:email").hexdigest()[:16]
        with extractor.db.get_connection("user_model") as conn:
            row2 = conn.execute(
                "SELECT greeting FROM communication_templates WHERE id = ?",
                (template_id2,)
            ).fetchone()

        # Should fall back to body field
        assert row2["greeting"].startswith("Hey")

    def test_template_extraction_per_contact_channel(self, extractor):
        """Verify separate templates are created for different contact-channel pairs."""
        # Same contact, different channels
        event1 = {
            "id": "evt-027",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["multi@example.com"],
                "body_plain": "Hi,\n\nFormal email message.\n\nBest regards",
                "channel": "email",
            },
        }
        extractor.extract(event1)

        event2 = {
            "id": "evt-028",
            "type": EventType.MESSAGE_SENT.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "to_addresses": ["multi@example.com"],
                "body": "hey! what's up?",
                "channel": "slack",
            },
        }
        extractor.extract(event2)

        # Verify two separate templates were created
        template_id_email = hashlib.sha256(b"multi@example.com:email").hexdigest()[:16]
        template_id_slack = hashlib.sha256(b"multi@example.com:slack").hexdigest()[:16]

        with extractor.db.get_connection("user_model") as conn:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM communication_templates WHERE id IN (?, ?)",
                (template_id_email, template_id_slack)
            ).fetchone()

        assert count["cnt"] == 2

    # -------------------------------------------------------------------------
    # Edge Case Tests
    # -------------------------------------------------------------------------

    def test_extract_handles_none_sentiment(self, extractor):
        """Verify extractor handles missing sentiment field gracefully."""
        event = {
            "id": "evt-029",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "from_address": "nosent@example.com",
                "body": "Message without sentiment analysis",
                "channel": "email",
                # sentiment field is missing
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 1
        assert signals[0]["sentiment"] is None

    def test_extract_handles_empty_recipient_list(self, extractor):
        """Verify extractor handles empty recipient list gracefully."""
        event = {
            "id": "evt-030",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": [],  # Empty list
                "body": "Message to nobody",
                "channel": "email",
            },
        }

        signals = extractor.extract(event)

        assert len(signals) == 0

    def test_extract_filters_none_addresses(self, extractor):
        """Verify extractor filters out None values in address lists."""
        event = {
            "id": "evt-031",
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "to_addresses": ["alice@example.com", None, "bob@example.com", None],
                "body": "Message with null addresses",
                "channel": "email",
            },
        }

        signals = extractor.extract(event)

        # Should only create signals for non-None addresses
        assert len(signals) == 2
        addresses = [s["contact_address"] for s in signals]
        assert "alice@example.com" in addresses
        assert "bob@example.com" in addresses
        assert None not in addresses
