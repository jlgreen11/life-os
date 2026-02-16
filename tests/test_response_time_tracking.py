"""
Tests for response time tracking in RelationshipExtractor.

Tests the fix for the broken semantic fact inference pipeline: the
RelationshipExtractor now computes avg_response_time_seconds so the
SemanticFactInferrer can identify high-priority relationships.
"""

import pytest
from datetime import datetime, timezone, timedelta

from services.signal_extractor.relationship import RelationshipExtractor
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


@pytest.fixture
def relationship_extractor(db, user_model_store):
    """Create a RelationshipExtractor instance."""
    return RelationshipExtractor(db=db, user_model_store=user_model_store)


@pytest.fixture
def semantic_inferrer(user_model_store):
    """Create a SemanticFactInferrer instance."""
    return SemanticFactInferrer(user_model_store)


class TestResponseTimeCalculation:
    """Test response time tracking in relationship profiles."""

    def test_response_time_calculated_for_reply(self, relationship_extractor, user_model_store):
        """Response time is calculated when user replies to an inbound message."""
        contact = "alice@example.com"
        base_time = datetime.now(timezone.utc)

        # Step 1: Inbound message from Alice
        inbound_event = {
            "type": "email.received",
            "timestamp": base_time.isoformat(),
            "source": "email",
            "payload": {
                "from_address": contact,
                "subject": "Question for you",
                "body": "Hey, can you help me with something?",
                "channel": "email",
            },
        }
        relationship_extractor.extract(inbound_event)

        # Step 2: User replies 30 minutes later
        reply_time = base_time + timedelta(minutes=30)
        outbound_event = {
            "type": "email.sent",
            "timestamp": reply_time.isoformat(),
            "source": "email",
            "payload": {
                "to_addresses": [contact],
                "subject": "Re: Question for you",
                "body": "Sure, happy to help!",
                "channel": "email",
                "is_reply": True,
            },
        }
        relationship_extractor.extract(outbound_event)

        # Verify: Response time is tracked
        profile = user_model_store.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"][contact]

        assert "response_times_seconds" in contact_data
        assert len(contact_data["response_times_seconds"]) == 1
        # 30 minutes = 1800 seconds
        assert abs(contact_data["response_times_seconds"][0] - 1800) < 5  # Allow 5s tolerance

        assert "avg_response_time_seconds" in contact_data
        assert abs(contact_data["avg_response_time_seconds"] - 1800) < 5

    def test_response_time_not_calculated_for_non_reply(self, relationship_extractor, user_model_store):
        """Response time is NOT calculated for outbound messages that aren't replies."""
        contact = "bob@example.com"
        base_time = datetime.now(timezone.utc)

        # Step 1: Inbound message from Bob
        inbound_event = {
            "type": "email.received",
            "timestamp": base_time.isoformat(),
            "source": "email",
            "payload": {
                "from_address": contact,
                "body": "Hello!",
                "channel": "email",
            },
        }
        relationship_extractor.extract(inbound_event)

        # Step 2: User sends a NEW message to Bob (not a reply)
        new_message_time = base_time + timedelta(hours=2)
        outbound_event = {
            "type": "email.sent",
            "timestamp": new_message_time.isoformat(),
            "source": "email",
            "payload": {
                "to_addresses": [contact],
                "body": "Hey Bob, wanted to follow up on something else",
                "channel": "email",
                "is_reply": False,  # Explicitly NOT a reply
            },
        }
        relationship_extractor.extract(outbound_event)

        # Verify: No response time tracked (because is_reply=False)
        profile = user_model_store.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"][contact]

        assert contact_data.get("response_times_seconds", []) == []
        assert contact_data.get("avg_response_time_seconds") is None

    def test_multiple_response_times_averaged(self, relationship_extractor, user_model_store):
        """Multiple response times are averaged correctly."""
        contact = "carol@example.com"
        base_time = datetime.now(timezone.utc)

        # Simulate 3 inbound→reply cycles with different response times
        response_times = [15 * 60, 45 * 60, 90 * 60]  # 15min, 45min, 90min in seconds

        for i, response_seconds in enumerate(response_times):
            # Inbound message
            inbound_time = base_time + timedelta(hours=i * 3)
            inbound_event = {
                "type": "message.received",
                "timestamp": inbound_time.isoformat(),
                "source": "signal",
                "payload": {
                    "from_address": contact,
                    "body": f"Message {i+1}",
                    "channel": "signal",
                },
            }
            relationship_extractor.extract(inbound_event)

            # User's reply
            reply_time = inbound_time + timedelta(seconds=response_seconds)
            outbound_event = {
                "type": "message.sent",
                "timestamp": reply_time.isoformat(),
                "source": "signal",
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Reply {i+1}",
                    "channel": "signal",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Verify: All response times tracked and averaged
        profile = user_model_store.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"][contact]

        assert len(contact_data["response_times_seconds"]) == 3
        expected_avg = sum(response_times) / 3  # (15 + 45 + 90) / 3 = 50 minutes
        assert abs(contact_data["avg_response_time_seconds"] - expected_avg) < 10

    def test_response_times_capped_at_50(self, relationship_extractor, user_model_store):
        """Response time ring buffer is capped at 50 entries."""
        contact = "dave@example.com"
        base_time = datetime.now(timezone.utc)

        # Simulate 60 inbound→reply cycles (exceeds 50 cap)
        for i in range(60):
            inbound_time = base_time + timedelta(hours=i)
            inbound_event = {
                "type": "email.received",
                "timestamp": inbound_time.isoformat(),
                "payload": {
                    "from_address": contact,
                    "body": f"Msg {i}",
                    "channel": "email",
                },
            }
            relationship_extractor.extract(inbound_event)

            reply_time = inbound_time + timedelta(minutes=10)
            outbound_event = {
                "type": "email.sent",
                "timestamp": reply_time.isoformat(),
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Reply {i}",
                    "channel": "email",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Verify: Only last 50 response times are kept
        profile = user_model_store.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"][contact]

        assert len(contact_data["response_times_seconds"]) == 50

    def test_negative_response_time_ignored(self, relationship_extractor, user_model_store):
        """Negative response times (clock skew) are ignored."""
        contact = "eve@example.com"
        base_time = datetime.now(timezone.utc)

        # Inbound message
        inbound_event = {
            "type": "email.received",
            "timestamp": base_time.isoformat(),
            "payload": {
                "from_address": contact,
                "body": "Hello",
                "channel": "email",
            },
        }
        relationship_extractor.extract(inbound_event)

        # Reply with EARLIER timestamp (clock skew scenario)
        reply_time = base_time - timedelta(minutes=10)  # Before inbound!
        outbound_event = {
            "type": "email.sent",
            "timestamp": reply_time.isoformat(),
            "payload": {
                "to_addresses": [contact],
                "body": "Hi",
                "channel": "email",
                "is_reply": True,
            },
        }
        relationship_extractor.extract(outbound_event)

        # Verify: Negative response time is NOT tracked
        profile = user_model_store.get_signal_profile("relationships")
        contact_data = profile["data"]["contacts"][contact]

        assert contact_data.get("response_times_seconds", []) == []
        assert contact_data.get("avg_response_time_seconds") is None

    def test_invalid_timestamp_handled_gracefully(self, relationship_extractor, user_model_store):
        """Invalid timestamps don't crash the extractor."""
        contact = "frank@example.com"

        # Inbound with malformed timestamp
        inbound_event = {
            "type": "email.received",
            "timestamp": "not-a-valid-timestamp",
            "payload": {
                "from_address": contact,
                "body": "Hello",
                "channel": "email",
            },
        }
        relationship_extractor.extract(inbound_event)  # Should not crash

        # Reply also with malformed timestamp
        outbound_event = {
            "type": "email.sent",
            "timestamp": "also-invalid",
            "payload": {
                "to_addresses": [contact],
                "body": "Hi",
                "channel": "email",
                "is_reply": True,
            },
        }
        relationship_extractor.extract(outbound_event)  # Should not crash

        # Verify: Profile exists but no response time calculated
        profile = user_model_store.get_signal_profile("relationships")
        assert contact in profile["data"]["contacts"]
        contact_data = profile["data"]["contacts"][contact]
        assert contact_data.get("response_times_seconds", []) == []


class TestSemanticFactInferenceIntegration:
    """Test that semantic fact inference now works with response time data."""

    def test_high_priority_contact_inferred(self, relationship_extractor, semantic_inferrer, user_model_store):
        """Fast responder (< 1 hour avg) is inferred as high priority."""
        contact = "alice@example.com"
        base_time = datetime.now(timezone.utc)

        # Simulate 10 fast responses (avg 20 minutes)
        for i in range(10):
            inbound_time = base_time + timedelta(hours=i * 2)
            inbound_event = {
                "type": "email.received",
                "timestamp": inbound_time.isoformat(),
                "payload": {
                    "from_address": contact,
                    "body": f"Question {i}",
                    "channel": "email",
                },
            }
            relationship_extractor.extract(inbound_event)

            reply_time = inbound_time + timedelta(minutes=20)  # Fast reply
            outbound_event = {
                "type": "email.sent",
                "timestamp": reply_time.isoformat(),
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Answer {i}",
                    "channel": "email",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Run semantic fact inference
        semantic_inferrer.infer_from_relationship_profile()

        # Verify: High priority fact was inferred
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        high_priority_facts = [
            f for f in facts
            if f["key"] == f"relationship_priority_{contact}" and f["value"] == "high_priority"
        ]

        assert len(high_priority_facts) == 1
        fact = high_priority_facts[0]
        assert fact["confidence"] > 0.6  # Should have high confidence

    def test_low_priority_contact_inferred(self, relationship_extractor, semantic_inferrer, user_model_store):
        """Slow responder (> 24 hours avg) is inferred as low priority."""
        contact = "bob@example.com"
        base_time = datetime.now(timezone.utc)

        # Simulate 10 slow responses (avg 36 hours)
        for i in range(10):
            inbound_time = base_time + timedelta(days=i * 3)
            inbound_event = {
                "type": "email.received",
                "timestamp": inbound_time.isoformat(),
                "payload": {
                    "from_address": contact,
                    "body": f"Hey {i}",
                    "channel": "email",
                },
            }
            relationship_extractor.extract(inbound_event)

            reply_time = inbound_time + timedelta(hours=36)  # Slow reply
            outbound_event = {
                "type": "email.sent",
                "timestamp": reply_time.isoformat(),
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Sorry for delay {i}",
                    "channel": "email",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Run semantic fact inference
        semantic_inferrer.infer_from_relationship_profile()

        # Verify: Low priority fact was inferred
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        low_priority_facts = [
            f for f in facts
            if f["key"] == f"relationship_priority_{contact}" and f["value"] == "low_priority"
        ]

        assert len(low_priority_facts) == 1
        fact = low_priority_facts[0]
        assert fact["confidence"] > 0.5

    def test_neutral_priority_no_fact_inferred(self, relationship_extractor, semantic_inferrer, user_model_store):
        """Contacts with neutral response time (1-24 hours) don't get priority facts."""
        contact = "carol@example.com"
        base_time = datetime.now(timezone.utc)

        # Simulate 10 medium responses (avg 5 hours - in neutral range)
        for i in range(10):
            inbound_time = base_time + timedelta(hours=i * 8)
            inbound_event = {
                "type": "email.received",
                "timestamp": inbound_time.isoformat(),
                "payload": {
                    "from_address": contact,
                    "body": f"Message {i}",
                    "channel": "email",
                },
            }
            relationship_extractor.extract(inbound_event)

            reply_time = inbound_time + timedelta(hours=5)  # Neutral
            outbound_event = {
                "type": "email.sent",
                "timestamp": reply_time.isoformat(),
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Response {i}",
                    "channel": "email",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Run semantic fact inference
        semantic_inferrer.infer_from_relationship_profile()

        # Verify: No priority fact for this contact
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_facts = [
            f for f in facts
            if f["key"] == f"relationship_priority_{contact}"
        ]

        assert len(priority_facts) == 0

    def test_insufficient_data_no_inference(self, relationship_extractor, semantic_inferrer, user_model_store):
        """Contacts with < 5 interactions don't trigger semantic facts."""
        contact = "dave@example.com"
        base_time = datetime.now(timezone.utc)

        # Only 3 responses (below threshold of 5)
        for i in range(3):
            inbound_time = base_time + timedelta(hours=i)
            inbound_event = {
                "type": "email.received",
                "timestamp": inbound_time.isoformat(),
                "payload": {
                    "from_address": contact,
                    "body": f"Hi {i}",
                    "channel": "email",
                },
            }
            relationship_extractor.extract(inbound_event)

            reply_time = inbound_time + timedelta(minutes=10)  # Fast
            outbound_event = {
                "type": "email.sent",
                "timestamp": reply_time.isoformat(),
                "payload": {
                    "to_addresses": [contact],
                    "body": f"Hello {i}",
                    "channel": "email",
                    "is_reply": True,
                },
            }
            relationship_extractor.extract(outbound_event)

        # Run semantic fact inference
        semantic_inferrer.infer_from_relationship_profile()

        # Verify: No fact inferred (insufficient data)
        facts = user_model_store.get_semantic_facts(category="implicit_preference")
        priority_facts = [
            f for f in facts
            if f["key"] == f"relationship_priority_{contact}"
        ]

        assert len(priority_facts) == 0
