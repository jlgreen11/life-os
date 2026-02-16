"""
Tests for refined interaction type classification in episode creation.

Verifies that events are mapped to granular interaction types (15+ distinct types)
instead of coarse categories, enabling the routine detector to identify behavioral
patterns like "email_received at 9am every day" vs "email_sent in the evening".
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from main import LifeOS


@pytest.fixture
def lifeos(tmp_path):
    """Create a minimal LifeOS instance for testing classification logic."""
    # Create a temporary config file
    config_data = {
        "data_dir": str(tmp_path / "data"),
        "nats_url": "nats://localhost:4222",
        "web_port": 8080,
        "embedding_model": "all-MiniLM-L6-v2",
        "ai": {
            "ollama_url": "http://localhost:11434",
            "ollama_model": "mistral",
            "use_cloud": False,
        },
        "connectors": {},
    }

    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml.dump(config_data))

    # Create data directory
    (tmp_path / "data").mkdir(exist_ok=True)

    # Create LifeOS instance - the classification method is stateless
    return LifeOS(str(config_file))


class TestEmailClassification:
    """Test email event classification into granular types."""

    def test_email_received_classified_as_email_received(self, lifeos):
        """Inbound emails should map to 'email_received' for inbox-checking routine detection."""
        interaction_type = lifeos._classify_interaction_type(
            "email.received",
            {"from_address": "sender@example.com", "subject": "Test"}
        )
        assert interaction_type == "email_received"

    def test_email_sent_classified_as_email_sent(self, lifeos):
        """Outbound emails should map to 'email_sent' for correspondence routine detection."""
        interaction_type = lifeos._classify_interaction_type(
            "email.sent",
            {"to_addresses": ["recipient@example.com"], "subject": "Reply"}
        )
        assert interaction_type == "email_sent"

    def test_email_types_are_distinct(self, lifeos):
        """Received and sent emails must have different types for pattern detection."""
        received = lifeos._classify_interaction_type("email.received", {})
        sent = lifeos._classify_interaction_type("email.sent", {})
        assert received != sent


class TestMessagingClassification:
    """Test messaging event classification (IM, chat, SMS)."""

    def test_message_received_classified_as_message_received(self, lifeos):
        """Inbound messages should map to 'message_received'."""
        interaction_type = lifeos._classify_interaction_type(
            "message.received",
            {"from_address": "+15551234567", "body_plain": "Hello"}
        )
        assert interaction_type == "message_received"

    def test_message_sent_classified_as_message_sent(self, lifeos):
        """Outbound messages should map to 'message_sent'."""
        interaction_type = lifeos._classify_interaction_type(
            "message.sent",
            {"to_addresses": ["+15551234567"], "body_plain": "Hi"}
        )
        assert interaction_type == "message_sent"

    def test_messages_distinct_from_emails(self, lifeos):
        """Messages and emails should have different types for channel-specific routines."""
        email = lifeos._classify_interaction_type("email.received", {})
        message = lifeos._classify_interaction_type("message.received", {})
        assert email != message


class TestCallClassification:
    """Test phone call event classification."""

    def test_call_received_classified_as_call_answered(self, lifeos):
        """Received/answered calls should map to 'call_answered'."""
        interaction_type = lifeos._classify_interaction_type(
            "call.received",
            {"from_address": "+15551234567", "duration": 120}
        )
        assert interaction_type == "call_answered"

    def test_call_missed_classified_as_call_missed(self, lifeos):
        """Missed calls should map to 'call_missed'."""
        interaction_type = lifeos._classify_interaction_type(
            "call.missed",
            {"from_address": "+15551234567"}
        )
        assert interaction_type == "call_missed"

    def test_answered_and_missed_calls_are_distinct(self, lifeos):
        """Answered and missed calls should have different types for follow-up patterns."""
        answered = lifeos._classify_interaction_type("call.received", {})
        missed = lifeos._classify_interaction_type("call.missed", {})
        assert answered != missed


class TestCalendarClassification:
    """Test calendar event classification."""

    def test_calendar_event_with_participants_classified_as_meeting_scheduled(self, lifeos):
        """Calendar events with participants should map to 'meeting_scheduled'."""
        interaction_type = lifeos._classify_interaction_type(
            "calendar.event.created",
            {
                "title": "Team Standup",
                "participants": ["alice@example.com", "bob@example.com"]
            }
        )
        assert interaction_type == "meeting_scheduled"

    def test_calendar_event_with_attendees_classified_as_meeting_scheduled(self, lifeos):
        """Calendar events with attendees field should also map to 'meeting_scheduled'."""
        interaction_type = lifeos._classify_interaction_type(
            "calendar.event.created",
            {
                "title": "Client Call",
                "attendees": ["client@external.com"]
            }
        )
        assert interaction_type == "meeting_scheduled"

    def test_calendar_event_without_participants_classified_as_calendar_blocked(self, lifeos):
        """Solo calendar events should map to 'calendar_blocked' for time-blocking routines."""
        interaction_type = lifeos._classify_interaction_type(
            "calendar.event.created",
            {"title": "Deep Work", "participants": None}
        )
        assert interaction_type == "calendar_blocked"

    def test_calendar_updated_classified_as_calendar_reviewed(self, lifeos):
        """Calendar updates should map to 'calendar_reviewed' for planning routines."""
        interaction_type = lifeos._classify_interaction_type(
            "calendar.event.updated",
            {"title": "Updated Meeting"}
        )
        assert interaction_type == "calendar_reviewed"

    def test_calendar_types_are_granular(self, lifeos):
        """Calendar events should have 3+ distinct types for routine detection."""
        meeting = lifeos._classify_interaction_type(
            "calendar.event.created",
            {"participants": ["someone@example.com"]}
        )
        block = lifeos._classify_interaction_type(
            "calendar.event.created",
            {"participants": None}
        )
        review = lifeos._classify_interaction_type("calendar.event.updated", {})

        assert len({meeting, block, review}) == 3


class TestFinancialClassification:
    """Test financial transaction classification."""

    def test_negative_transaction_classified_as_spending(self, lifeos):
        """Negative transactions (debits) should map to 'spending'."""
        interaction_type = lifeos._classify_interaction_type(
            "finance.transaction.new",
            {"amount": -45.23, "merchant": "Whole Foods"}
        )
        assert interaction_type == "spending"

    def test_positive_transaction_classified_as_income(self, lifeos):
        """Positive transactions (credits) should map to 'income'."""
        interaction_type = lifeos._classify_interaction_type(
            "finance.transaction.new",
            {"amount": 2500.00, "merchant": "Employer"}
        )
        assert interaction_type == "income"

    def test_spending_and_income_are_distinct(self, lifeos):
        """Spending and income should have different types for financial habit detection."""
        spending = lifeos._classify_interaction_type(
            "finance.transaction.new",
            {"amount": -100}
        )
        income = lifeos._classify_interaction_type(
            "finance.transaction.new",
            {"amount": 100}
        )
        assert spending != income


class TestTaskClassification:
    """Test task event classification."""

    def test_task_created_classified_as_task_created(self, lifeos):
        """Task creation should map to 'task_created' for work planning routine detection."""
        interaction_type = lifeos._classify_interaction_type(
            "task.created",
            {"title": "Implement feature X"}
        )
        assert interaction_type == "task_created"

    def test_task_completed_classified_as_task_completed(self, lifeos):
        """Task completion should map to 'task_completed' for execution pattern detection."""
        interaction_type = lifeos._classify_interaction_type(
            "task.completed",
            {"title": "Implement feature X"}
        )
        assert interaction_type == "task_completed"

    def test_task_creation_and_completion_are_distinct(self, lifeos):
        """Task creation and completion should have different types for workflow analysis."""
        created = lifeos._classify_interaction_type("task.created", {})
        completed = lifeos._classify_interaction_type("task.completed", {})
        assert created != completed


class TestLocationClassification:
    """Test location event classification."""

    def test_location_arrived_classified_as_location_arrived(self, lifeos):
        """Location arrivals should map to 'location_arrived' for context entry routines."""
        interaction_type = lifeos._classify_interaction_type(
            "location.arrived",
            {"location": "Home"}
        )
        assert interaction_type == "location_arrived"

    def test_location_departed_classified_as_location_departed(self, lifeos):
        """Location departures should map to 'location_departed' for context exit routines."""
        interaction_type = lifeos._classify_interaction_type(
            "location.departed",
            {"location": "Office"}
        )
        assert interaction_type == "location_departed"

    def test_location_changed_classified_as_location_changed(self, lifeos):
        """Generic location changes should map to 'location_changed'."""
        interaction_type = lifeos._classify_interaction_type(
            "location.changed",
            {"location": "Gym"}
        )
        assert interaction_type == "location_changed"

    def test_location_types_are_granular(self, lifeos):
        """Location events should have 3+ distinct types for location-based routines."""
        arrived = lifeos._classify_interaction_type("location.arrived", {})
        departed = lifeos._classify_interaction_type("location.departed", {})
        changed = lifeos._classify_interaction_type("location.changed", {})

        assert len({arrived, departed, changed}) == 3


class TestContextClassification:
    """Test context event classification (device/activity state)."""

    def test_context_location_classified_as_context_location(self, lifeos):
        """Context location events should map to 'context_location'."""
        interaction_type = lifeos._classify_interaction_type(
            "context.location",
            {"location": "37.7749,-122.4194"}
        )
        assert interaction_type == "context_location"

    def test_context_activity_classified_as_context_activity(self, lifeos):
        """Context activity events should map to 'context_activity'."""
        interaction_type = lifeos._classify_interaction_type(
            "context.activity",
            {"activity": "walking"}
        )
        assert interaction_type == "context_activity"


class TestUserCommandClassification:
    """Test user command classification."""

    def test_user_command_classified_as_user_command(self, lifeos):
        """User commands should map to 'user_command' for explicit interaction patterns."""
        interaction_type = lifeos._classify_interaction_type(
            "system.user.command",
            {"command": "show tasks"}
        )
        assert interaction_type == "user_command"


class TestFallbackClassification:
    """Test fallback behavior for unmapped event types."""

    def test_unknown_event_type_extracts_suffix(self, lifeos):
        """Unknown event types should extract the suffix after the last dot."""
        interaction_type = lifeos._classify_interaction_type(
            "system.rule.triggered",
            {}
        )
        assert interaction_type == "triggered"

    def test_unknown_event_without_dot_returns_other(self, lifeos):
        """Unknown event types without dots should return 'other'."""
        interaction_type = lifeos._classify_interaction_type(
            "unknowneventtype",
            {}
        )
        assert interaction_type == "other"


class TestGranularityRequirement:
    """Test that the classification provides sufficient granularity for routine detection."""

    def test_at_least_15_distinct_types(self, lifeos):
        """The classifier should support at least 15 distinct interaction types."""
        event_samples = [
            ("email.received", {}),
            ("email.sent", {}),
            ("message.received", {}),
            ("message.sent", {}),
            ("call.received", {}),
            ("call.missed", {}),
            ("calendar.event.created", {"participants": ["alice@example.com"]}),
            ("calendar.event.created", {}),
            ("calendar.event.updated", {}),
            ("finance.transaction.new", {"amount": -50}),
            ("finance.transaction.new", {"amount": 50}),
            ("task.created", {}),
            ("task.completed", {}),
            ("location.arrived", {}),
            ("location.departed", {}),
            ("location.changed", {}),
            ("context.location", {}),
            ("context.activity", {}),
            ("system.user.command", {}),
        ]

        types = {lifeos._classify_interaction_type(et, p) for et, p in event_samples}
        assert len(types) >= 15, f"Only {len(types)} distinct types, need 15+"

    def test_no_coarse_buckets(self, lifeos):
        """The classifier should NOT use coarse buckets like 'communication'."""
        # Test multiple communication-related events
        email_type = lifeos._classify_interaction_type("email.received", {})
        message_type = lifeos._classify_interaction_type("message.received", {})
        call_type = lifeos._classify_interaction_type("call.received", {})

        # None should be the old coarse "communication" type
        assert email_type != "communication"
        assert message_type != "communication"
        assert call_type != "communication"

        # They should all be distinct
        assert email_type != message_type
        assert message_type != call_type
        assert email_type != call_type


class TestRoutineDetectorDataQuality:
    """Integration tests verifying that classified episodes enable routine detection."""

    def test_email_inbox_check_routine_detectable(self, lifeos):
        """Morning inbox checking should be detectable as a routine with granular types."""
        # Simulate 5 days of morning email checking
        morning_emails = [
            lifeos._classify_interaction_type("email.received", {})
            for _ in range(5)
        ]

        # All should be the same type (email_received)
        assert len(set(morning_emails)) == 1
        assert morning_emails[0] == "email_received"

        # NOT the old coarse "communication" type
        assert morning_emails[0] != "communication"

    def test_evening_correspondence_routine_detectable(self, lifeos):
        """Evening email sending should be detectable as separate from inbox checking."""
        inbox_type = lifeos._classify_interaction_type("email.received", {})
        send_type = lifeos._classify_interaction_type("email.sent", {})

        # Must be different types for the routine detector to distinguish these patterns
        assert inbox_type != send_type

    def test_meeting_vs_calendar_blocking_distinguishable(self, lifeos):
        """Meeting attendance vs time-blocking should be distinguishable patterns."""
        meeting_type = lifeos._classify_interaction_type(
            "calendar.event.created",
            {"participants": ["team@example.com"]}
        )
        blocking_type = lifeos._classify_interaction_type(
            "calendar.event.created",
            {}
        )

        # Must be different types to detect different calendar usage patterns
        assert meeting_type != blocking_type
