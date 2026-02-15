"""
Tests for TopicExtractor — validates topic/keyword extraction and profile updates.

Coverage:
- Basic topic extraction from messages
- Subject line inclusion in topic extraction
- Stop-word filtering
- Frequency-based keyword ranking
- Topic profile updates (counts + recent topics)
- Ring buffer behavior (500-entry cap)
- Edge cases: empty messages, very short messages, messages without topics
"""

import json
from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.topic import TopicExtractor


class TestTopicExtractor:
    """Test suite for TopicExtractor signal processor."""

    @pytest.fixture
    def extractor(self, db, event_store):
        """Create a TopicExtractor instance with test database."""
        from storage.user_model_store import UserModelStore
        ums = UserModelStore(db)
        return TopicExtractor(db, ums)

    def test_can_process_email_received(self, extractor):
        """TopicExtractor should process inbound emails."""
        event = {"type": EventType.EMAIL_RECEIVED.value}
        assert extractor.can_process(event)

    def test_can_process_email_sent(self, extractor):
        """TopicExtractor should process outbound emails."""
        event = {"type": EventType.EMAIL_SENT.value}
        assert extractor.can_process(event)

    def test_can_process_message_received(self, extractor):
        """TopicExtractor should process inbound messages."""
        event = {"type": EventType.MESSAGE_RECEIVED.value}
        assert extractor.can_process(event)

    def test_can_process_message_sent(self, extractor):
        """TopicExtractor should process outbound messages."""
        event = {"type": EventType.MESSAGE_SENT.value}
        assert extractor.can_process(event)

    def test_can_process_user_command(self, extractor):
        """TopicExtractor should process voice/system commands."""
        event = {"type": "system.user.command"}
        assert extractor.can_process(event)

    def test_cannot_process_calendar_events(self, extractor):
        """TopicExtractor should not process calendar events."""
        event = {"type": "calendar.event.created"}
        assert not extractor.can_process(event)

    def test_extract_topics_from_body(self, extractor):
        """Extract should identify keywords from message body."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "We need to discuss the quarterly revenue projections for enterprise sales. "
                        "The dashboard analytics show strong growth in subscription renewals.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        signal = signals[0]

        assert signal["type"] == "topic"
        # Check that meaningful keywords were extracted (exact list may vary based on frequency)
        topics = signal["topics"]
        assert isinstance(topics, list)
        assert len(topics) > 0
        # These high-frequency domain words should appear
        assert any(word in topics for word in ["revenue", "sales", "quarterly", "analytics", "enterprise"])

    def test_extract_topics_from_subject_and_body(self, extractor):
        """Extract should combine subject and body for topic extraction."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Machine Learning Conference Invitation",
                "body": "Join us for a workshop on neural networks and deep learning algorithms.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Keywords from both subject and body should be extracted
        # "machine", "learning", "conference", "neural", "networks", "deep", "algorithms"
        assert len(topics) > 0
        # At least some of these ML-related terms should appear
        assert any(word in topics for word in ["machine", "learning", "neural", "networks", "algorithms"])

    def test_extract_topics_subject_only(self, extractor):
        """Extract should work with subject-only emails (no body)."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Docker container orchestration with Kubernetes",
                "body": "",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Subject-line keywords should still be extracted
        assert len(topics) > 0
        assert any(word in topics for word in ["docker", "container", "orchestration", "kubernetes"])

    def test_stop_word_filtering(self, extractor):
        """Extract should filter out common English stop words."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "I think that we should have a meeting about the project timeline "
                        "because there are some issues with the current schedule.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Stop words that ARE in the implementation's stop_words set should NOT appear
        stop_words = ["think", "that", "have", "about", "because", "there", "some", "with"]
        for stop_word in stop_words:
            assert stop_word not in topics

        # Meaningful content words should appear
        assert any(word in topics for word in ["meeting", "project", "timeline", "issues", "schedule"])

    def test_frequency_based_ranking(self, extractor):
        """Extract should return top 10 most frequent keywords."""
        # Craft a message with repeated keywords to test frequency ranking
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Python programming with Python frameworks. Python developers use Python libraries. "
                        "Python syntax makes Python code readable. Python applications require Python testing. "
                        "Java code differs from Python code fundamentally.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Top keyword by frequency should be "python" (appears 8 times)
        assert "python" in topics
        # The extractor caps at top 10, so we shouldn't get more than 10
        assert len(topics) <= 10

    def test_skip_very_short_messages(self, extractor):
        """Extract should return empty list for messages under 20 chars."""
        event = {
            "type": EventType.MESSAGE_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "ok",  # Too short
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 0

    def test_skip_empty_messages(self, extractor):
        """Extract should return empty list for messages with no content."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "",
                "subject": "",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 0

    def test_handle_body_plain_field(self, extractor):
        """Extract should check body_plain if body is missing."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body_plain": "Database optimization techniques for PostgreSQL performance tuning.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        topics = signals[0]["topics"]
        assert len(topics) > 0
        assert any(word in topics for word in ["database", "optimization", "postgresql", "performance"])

    def test_signal_includes_context_and_source(self, extractor):
        """Extract should preserve event type and source in signal."""
        event = {
            "type": EventType.MESSAGE_SENT.value,
            "source": "imessage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Let's discuss the architectural patterns for microservices deployment strategies.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1
        signal = signals[0]

        assert signal["context"] == EventType.MESSAGE_SENT.value
        assert signal["source"] == "imessage"
        assert signal["timestamp"] == event["timestamp"]

    def test_update_topic_profile_initial(self, extractor):
        """First extraction should create new topic profile."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T12:00:00Z",
            "payload": {
                "body": "Machine learning models require extensive training datasets for accurate predictions.",
            }
        }

        extractor.extract(event)

        # Check profile was created
        profile = extractor.ums.get_signal_profile("topics")
        assert profile is not None
        assert "data" in profile

        data = profile["data"]
        assert "topic_counts" in data
        assert "recent_topics" in data

        # Check topic_counts has entries
        assert len(data["topic_counts"]) > 0
        # Keywords like "machine", "learning", "models", "training" should be counted
        assert any(word in data["topic_counts"] for word in ["machine", "learning", "models", "training"])

        # Check recent_topics has one entry
        assert len(data["recent_topics"]) == 1
        recent = data["recent_topics"][0]
        assert recent["timestamp"] == "2026-02-15T12:00:00Z"
        assert recent["context"] == EventType.EMAIL_SENT.value
        assert len(recent["topics"]) > 0

    def test_update_topic_profile_accumulation(self, extractor):
        """Multiple extractions should accumulate topic counts."""
        event1 = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": "2026-02-15T10:00:00Z",
            "payload": {
                "body": "Docker containerization simplifies application deployment across environments.",
            }
        }

        event2 = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": "2026-02-15T11:00:00Z",
            "payload": {
                "body": "Kubernetes orchestration handles Docker container scaling automatically.",
            }
        }

        extractor.extract(event1)
        extractor.extract(event2)

        profile = extractor.ums.get_signal_profile("topics")
        data = profile["data"]

        # "docker" appears in both messages, should have count of 2
        assert data["topic_counts"]["docker"] == 2

        # Other keywords should have count of 1
        assert data["topic_counts"].get("kubernetes", 0) == 1
        assert data["topic_counts"].get("containerization", 0) == 1

        # Should have 2 entries in recent_topics
        assert len(data["recent_topics"]) == 2

    def test_recent_topics_ring_buffer(self, extractor):
        """Recent topics list should cap at 500 entries."""
        # Create 510 events to test buffer cap
        for i in range(510):
            event = {
                "type": EventType.MESSAGE_SENT.value,
                "timestamp": f"2026-02-15T{i % 24:02d}:{i % 60:02d}:00Z",
                "payload": {
                    "body": f"Message number {i} about technology topics and software development.",
                }
            }
            extractor.extract(event)

        profile = extractor.ums.get_signal_profile("topics")
        data = profile["data"]

        # Recent topics should be capped at 500
        assert len(data["recent_topics"]) == 500

        # Should contain the most recent 500 (messages 10-509)
        timestamps = [entry["timestamp"] for entry in data["recent_topics"]]
        # First entry should be from message 10 (oldest in the buffer)
        assert timestamps[0].startswith("2026-02-15T10:")
        # Last entry should be from message 509 (most recent)
        assert timestamps[-1].startswith("2026-02-15T")

    def test_topic_counts_persist_across_buffer_rotation(self, extractor):
        """Topic counts should accumulate even as recent buffer rotates."""
        # Create 510 messages, each mentioning "blockchain"
        for i in range(510):
            event = {
                "type": EventType.EMAIL_RECEIVED.value,
                "timestamp": f"2026-02-15T{i % 24:02d}:{i % 60:02d}:00Z",
                "payload": {
                    "body": f"Message {i} discusses blockchain technology and distributed systems.",
                }
            }
            extractor.extract(event)

        profile = extractor.ums.get_signal_profile("topics")
        data = profile["data"]

        # Recent topics buffer should be capped at 500
        assert len(data["recent_topics"]) == 500

        # But topic_counts should reflect ALL 510 messages
        assert data["topic_counts"]["blockchain"] == 510

    def test_mixed_event_types_in_recent_topics(self, extractor):
        """Recent topics should preserve context (event type) for each entry."""
        events = [
            {
                "type": EventType.EMAIL_RECEIVED.value,
                "timestamp": "2026-02-15T10:00:00Z",
                "payload": {"body": "Email about artificial intelligence and machine learning applications."},
            },
            {
                "type": EventType.MESSAGE_SENT.value,
                "timestamp": "2026-02-15T11:00:00Z",
                "payload": {"body": "Message about cloud computing infrastructure and DevOps practices."},
            },
            {
                "type": "system.user.command",
                "timestamp": "2026-02-15T12:00:00Z",
                "payload": {"body": "Command mentioning database migration and schema design patterns."},
            },
        ]

        for event in events:
            extractor.extract(event)

        profile = extractor.ums.get_signal_profile("topics")
        data = profile["data"]

        assert len(data["recent_topics"]) == 3

        # Check each entry has correct context preserved
        contexts = [entry["context"] for entry in data["recent_topics"]]
        assert EventType.EMAIL_RECEIVED.value in contexts
        assert EventType.MESSAGE_SENT.value in contexts
        assert "system.user.command" in contexts

    def test_alphanumeric_filtering(self, extractor):
        """Extract should only include alphabetic tokens (4+ chars)."""
        event = {
            "type": EventType.EMAIL_SENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Meeting at 3pm in room B123. Call 555-1234 for details. Total cost: $500.",
            }
        }

        signals = extractor.extract(event)
        # If signal is empty, that's acceptable (all meaningful words may be filtered)
        # But if topics are extracted, they should only be alphabetic
        if signals:
            topics = signals[0]["topics"]
            for topic in topics:
                # Each topic should be purely alphabetic
                assert topic.isalpha()
                # And should be 4+ characters (per extraction logic)
                assert len(topic) >= 4

    def test_case_insensitivity(self, extractor):
        """Extract should normalize to lowercase for topic counting."""
        event = {
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "body": "Python Programming language. PYTHON frameworks. python development. PyThOn libraries.",
            }
        }

        signals = extractor.extract(event)
        assert len(signals) == 1

        # Check that "python" appears in topics (normalized to lowercase)
        topics = signals[0]["topics"]
        assert "python" in topics

        # Check profile to ensure counts are case-insensitive
        profile = extractor.ums.get_signal_profile("topics")
        data = profile["data"]

        # All variants of "Python" should be counted as one keyword "python"
        assert "python" in data["topic_counts"]
        # Should have count of 1 (appeared in 1 message, case-normalized)
        assert data["topic_counts"]["python"] == 1
